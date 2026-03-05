#
# Copyright 2025 Splunk Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import dataclasses
from typing import List, Tuple, Union, Optional, Dict
from typing_extensions import override

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from timesfm import pytorch_patched_decoder as ppd

from .constants import Constants
from .rope import StackedDecoderRoPE


@dataclasses.dataclass
class CiscoTsmMRConfig(ppd.TimesFMConfig):
  """Config extension to toggle multi-resolution behaviors.

  - use_resolution_embeddings: add scale embeddings (coarse/fine) to the token stream.
  - use_special_token: insert a learned special token between streams.
  - agg_factor: aggregation factor from fine-res to coarse-res (e.g., 60 for min->hr).
  """

  use_resolution_embeddings: bool = True
  use_special_token: bool = True
  agg_factor: int = 60


class PatchedTSMultiResolutionDecoder(ppd.PatchedTimeSeriesDecoder):
  """Extension of upstream decoder with multi-resolution support.

  This class keeps the upstream API intact, while enabling two optional
  behaviors:
    - scale embedding per token for coarse/fine streams,
    - an optional learned special token between streams.
  """

  @override
  def __init__(self, config: CiscoTsmMRConfig):
    super().__init__(config)
    self.config: CiscoTsmMRConfig = config

    self.stacked_transformer = StackedDecoderRoPE(
      hidden_size=self.config.hidden_size,
      intermediate_size=self.config.intermediate_size,
      num_heads=self.config.num_heads,
      num_kv_heads=self.config.num_kv_heads,
      head_dim=self.config.head_dim,
      num_layers=self.config.num_layers,
      rms_norm_eps=self.config.rms_norm_eps,
    )

    # Multi-resolution Embedding Layer
    if self.config.use_resolution_embeddings:
      self.multi_resolution = nn.Embedding(num_embeddings=3,
                                            embedding_dim=self.config.hidden_size)

    # Special Token between streams
    if self.config.use_special_token:
      self.special_token = nn.Parameter(torch.zeros(1, 1, self.config.hidden_size))
      nn.init.normal_(self.special_token, mean=0.0, std=0.02)

  
  def _reverse_transform_segments(
    self,
    outputs: torch.Tensor,
    stats_list: List[Tuple[torch.Tensor, torch.Tensor]],
    indices_list: List[Tuple[int, int]],
  ) -> torch.Tensor:
    """Reverse-transform with per-timeseries stats.

    Args:
      outputs: [B, N, P, Q]
      stats_list: list of (mu, sigma) each shaped [B]
      indices_list: matching list of (start_N, end_N) segment ranges over N
    """
    B, N, _, _ = outputs.shape
    device = outputs.device
    dtype = outputs.dtype

    if len(indices_list) == 0:
      return outputs

    # Build [S] tensors of segment starts/ends (S = number of streams)
    starts = torch.tensor([s for (s, _) in indices_list], device=device)
    ends = torch.tensor([e for (_, e) in indices_list], device=device)

    # Per-batch stats stacked as [B, S]
    mus_BS = torch.stack([mu.to(dtype) for (mu, _) in stats_list], dim=1)       # [B, S]
    sigmas_BS = torch.stack([sigma.to(dtype) for (_, sigma) in stats_list], dim=1)  # [B, S]

    # Build boolean mask per segment over N: [S, N]
    pos_N = torch.arange(N, device=device)
    seg_mask_SN = ((pos_N.unsqueeze(0) >= starts.unsqueeze(1)) &
                   (pos_N.unsqueeze(0) < ends.unsqueeze(1)))  # [S, N]

    # Expand to broadcast shapes:
    # seg_mask: [1, S, N, 1, 1], mus_BS/sigmas_BS: [B, S, 1, 1, 1]
    seg_mask = seg_mask_SN.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(dtype)  # [1, S, N, 1, 1]
    mus_BS111 = mus_BS.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)                      # [B, S, 1, 1, 1]
    sigmas_BS111 = sigmas_BS.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)                # [B, S, 1, 1, 1]

    # Aggregate per-position parameters
    mu_map = (mus_BS111 * seg_mask).sum(dim=1)                                     # [B, N, 1, 1]
    sigma_map = (sigmas_BS111 * seg_mask).sum(dim=1)                                # [B, N, 1, 1]

    # For positions not covered by any segment, keep outputs unchanged: sigma=1, mu=0
    covered = (seg_mask.sum(dim=1) > 0).to(dtype)                               # [1, N, 1, 1]
    sigma_map = sigma_map + (1.0 - covered).expand(B, -1, -1, -1)                # add 1 where uncovered

    return outputs * sigma_map + mu_map


  @override
  def _postprocess_output(
    self,
    model_output: torch.Tensor,
    horizon_len: int,
    head: nn.Module,
    num_outputs: int,
    stats_list: List[Tuple[torch.Tensor, torch.Tensor]],
    indices_list: List[Tuple[int, int]],
  ) -> torch.Tensor:
    """Postprocess output of stacked transformer."""

    # B x N x (H.Q)
    output_BNHO = head(model_output)

    # Reshape using view
    b, n, _ = output_BNHO.shape
    output_BNHO = output_BNHO.view(b, n, horizon_len, num_outputs)
    return self._reverse_transform_segments(output_BNHO, stats_list, indices_list)


  @staticmethod
  def _left_pad_to_patch_boundary(
    ts: torch.Tensor, pad: torch.Tensor, patch_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Left-pads time series and padding mask so the length is a multiple of patch_len.
    Args:
      ts: [B, T] time series values.
      pad: [B, T] padding mask (1.0 = padded, 0.0 = real).
      patch_len: patch length to align to.
    Returns:
      Tuple of (ts, pad) left-padded to the next multiple of patch_len.
    """
    rem = ts.shape[1] % patch_len

    if rem:
      pad_len = patch_len - rem
      ts_pad = torch.zeros((ts.shape[0], pad_len), device=ts.device, dtype=ts.dtype)
      pad_pad = torch.ones((pad.shape[0], pad_len), device=pad.device, dtype=pad.dtype)
      ts = torch.cat([ts_pad, ts], dim=1)
      pad = torch.cat([pad_pad, pad], dim=1)

    return ts, pad
  

  @override
  def forward(
    self,
    input_ts: Union[List[torch.Tensor], torch.Tensor],
    input_padding: Union[List[torch.Tensor], torch.Tensor],
    freq: torch.Tensor,
  ) -> torch.Tensor:
    """Multi-resolution forward pass.
    Args:
      input_ts: list of batched tensors for coarse/fine resolution streams.
      input_padding: list of batched paddings for coarse/fine resolution streams.
      freq: batched tensor of frequency indices.
    """
    num_outputs = len(self.config.quantiles) + 1

    if isinstance(input_ts, torch.Tensor):
      raise ValueError("PatchedTSMultiResolutionDecoder expects multi-resolution inputs as a list of tensors.")

    # Multi-resolution processing
    ts_coarse, ts_fine = input_ts
    pad_coarse, pad_fine = input_padding

    # left-pad to the next multiple of `patch_len` so the most recent timesteps stay aligned in the last patch
    if ts_coarse.ndim == 3 and ts_coarse.shape[-1] == 1:
      ts_coarse = ts_coarse.squeeze(-1)
    if ts_fine.ndim == 3 and ts_fine.shape[-1] == 1:
      ts_fine = ts_fine.squeeze(-1)
    if pad_coarse.ndim == 3 and pad_coarse.shape[-1] == 1:
      pad_coarse = pad_coarse.squeeze(-1)
    if pad_fine.ndim == 3 and pad_fine.shape[-1] == 1:
      pad_fine = pad_fine.squeeze(-1)

    patch_len = self.config.patch_len
    ts_coarse, pad_coarse = self._left_pad_to_patch_boundary(ts_coarse, pad_coarse, patch_len)
    ts_fine, pad_fine = self._left_pad_to_patch_boundary(ts_fine, pad_fine, patch_len)

    model_input_coarse, pad_coarse, stats_coarse, _ = self._preprocess_input(
      input_ts=ts_coarse,
      input_padding=pad_coarse,
    )
    model_input_fine, pad_fine, stats_fine, _ = self._preprocess_input(
      input_ts=ts_fine,
      input_padding=pad_fine,
    )

    B = model_input_coarse.shape[0]
    Ncoarse = model_input_coarse.shape[1]
    Nfine = model_input_fine.shape[1]
    D = model_input_coarse.shape[2]
    
    device = model_input_coarse.device

    # Special Token between streams
    if self.config.use_special_token:
      spec_tok = self.special_token.to(device).expand(B, 1, D)
      spec_pad = torch.zeros(B, 1, device=device, dtype=pad_coarse.dtype)

      model_input = torch.cat([model_input_coarse, spec_tok, model_input_fine], dim=1)     # [B, N1+1+N2, D]
      patched_padding = torch.cat([pad_coarse, spec_pad, pad_fine], dim=1)
      
      # Keep mask to drop the special token position after decoding
      keep_mask = torch.ones(Ncoarse + 1 + Nfine, device=device, dtype=torch.bool)
      keep_mask[Ncoarse] = False  # special token index
      spec_len = 1
    else:
      model_input = torch.cat([model_input_coarse, model_input_fine], dim=1)               # [B, N1+N2, D]
      patched_padding = torch.cat([pad_coarse, pad_fine], dim=1)
      keep_mask = None
      spec_len = 0

    # Multi-resolution Embedding
    if self.config.use_resolution_embeddings:
      mr_coarse = torch.zeros(Ncoarse, dtype=torch.long, device=device)
      mr_spec = torch.full((spec_len,), 1, dtype=torch.long, device=device)              # 1 for special token
      mr_fine = torch.full((Nfine,), 2, dtype=torch.long, device=device)

      mr_idx = torch.cat([mr_coarse, mr_spec, mr_fine], dim=0)                      # [N_total]
      mr_idx = mr_idx.unsqueeze(0).expand(B, -1)                                    # [B, N_total]
      
      mr_emb = self.multi_resolution(mr_idx)                                        # [B, N_total, D]
      model_input += mr_emb
  
    if freq.device != device:
      freq = freq.to(device)
    
    f_emb = self.freq_emb(freq)  # [B, 1, D]
    model_input += f_emb

    padding_mask = ppd.convert_paddings_to_mask(patched_padding, model_input.dtype)
    atten_mask = ppd.causal_mask(model_input)
    atten_mask[..., :Ncoarse, :Ncoarse] = 0.0  # allow bidirectional attention for coarse-res patches
    mask = ppd.merge_masks(padding_mask, atten_mask)
    model_output = self.stacked_transformer(model_input, patched_padding, mask=mask)

    # Project and apply per-segment reverse-transform

    indices_list = [
        (0, Ncoarse),
        (Ncoarse + spec_len, Ncoarse + spec_len + Nfine),
    ]
    stats_list = [stats_coarse, stats_fine]

    output_min_all = self._postprocess_output(
        model_output=model_output,
        horizon_len=self.config.horizon_len,
        head=self.horizon_ff_layer,
        num_outputs=num_outputs,
        stats_list=stats_list,
        indices_list=indices_list,
    )

    if keep_mask is not None:
      output_min_all = output_min_all[:, keep_mask, :, :]

    return output_min_all


  def _trim_context(self, ts: torch.Tensor, pad: torch.Tensor, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Trim context to be aligned to patch boundaries and within max length.
    Args:
      ts: [B, T, C] input time series tensor.
      pad: [B, T, 1] input padding tensor.
      max_len: maximum allowed length.
    Returns:
      Trimmed (ts, pad) tensors.
    """
    target_len = max(self.config.patch_len, (max_len // self.config.patch_len) * self.config.patch_len)
    
    if ts.shape[1] > target_len:
      ts = ts[:, -target_len:, :]
      pad = pad[:, -target_len:, :]
    
    rem = ts.shape[1] % self.config.patch_len
    
    if rem:
      ts = ts[:, rem:, :]
      pad = pad[:, rem:, :]
    
    return ts, pad
  

  @staticmethod
  def _normalize_with_pad(
      context: torch.Tensor,
      pad_mask: Optional[torch.Tensor] = None,
      clamp_range: Tuple[float, float] = (-1000.0, 1000.0),
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Context-level normalization.

    Args:
      context: [B, T] or [B, T, 1] raw values.
      pad_mask: [B, T] or [B, T, 1] padding mask (1.0 padded, 0.0 real).
      clamp_range: clamp range for normalized values.
    Returns:
      ctx_normalized: normalized context with same rank as `context`.
      context_mean: [B, 1] mean used for normalization.
      context_std: [B, 1] std used for normalization (before EPS).
    """
    add_channel = False
    
    if context.ndim == 3 and context.shape[-1] == 1:
      add_channel = True
      context_2d = context.squeeze(-1)
    else:
      context_2d = context

    if pad_mask is None:
      pad_2d = torch.zeros_like(context_2d)
    elif pad_mask.ndim == 3 and pad_mask.shape[-1] == 1:
      pad_2d = pad_mask.squeeze(-1)
    else:
      pad_2d = pad_mask

    if context_2d.ndim != 2 or pad_2d.ndim != 2:
      raise ValueError(f"_normalize_with_pad expects [B, T] inputs, got context={context_2d.shape} pad={pad_2d.shape}")
    if context_2d.shape != pad_2d.shape:
      raise ValueError(f"context and pad must have same shape, got context={context_2d.shape} pad={pad_2d.shape}")

    valid = 1.0 - pad_2d
    count = valid.sum(dim=1, keepdim=True).clamp_min(1.0)

    context_mean = (context_2d * valid).sum(dim=1, keepdim=True) / count

    seq_len = context_2d.shape[1]
    seq_len_t = count.new_tensor(float(seq_len))
    pad_bool = pad_2d.to(dtype=torch.bool)

    filled_context = torch.where(pad_bool, context_mean, context_2d)
    context_std = filled_context.std(dim=1, keepdim=True, unbiased=False) * torch.sqrt(seq_len_t / count)

    if (not torch.isfinite(context_mean).all()) or (not torch.isfinite(context_std).all()):
      context64 = context_2d.to(dtype=torch.float64)
      valid64 = valid.to(dtype=torch.float64)
      count64 = valid64.sum(dim=1, keepdim=True).clamp_min(1.0)
      mean64 = (context64 * valid64).sum(dim=1, keepdim=True) / count64

      seq_len64 = count64.new_tensor(float(seq_len))
      filled64 = torch.where(pad_bool, mean64, context64)
      std64 = filled64.std(dim=1, keepdim=True, unbiased=False) * torch.sqrt(seq_len64 / count64)

      context_mean = mean64.to(dtype=context_2d.dtype)
      context_std = std64.to(dtype=context_2d.dtype)

    context_std = context_std.clamp_min(1e-2)

    ctx_normalized = (context_2d - context_mean) / (context_std + Constants.EPS.value)
    ctx_normalized = ctx_normalized * valid
    ctx_normalized = torch.clamp(ctx_normalized, *clamp_range)

    if add_channel:
      ctx_normalized = ctx_normalized.unsqueeze(-1)

    return ctx_normalized, context_mean, context_std
  

  @override
  def decode(
      self,
      input_ts: Union[List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
      paddings: Union[List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
      freq: torch.Tensor,
      horizon_len: int,
      offsets_fine: Optional[Union[List[float], torch.Tensor]] = None,
      scales_fine: Optional[Union[List[float], torch.Tensor]] = None,
      output_patch_len: int = 128,
      offsets_coarse: Optional[Union[List[float], torch.Tensor]] = None,
      scales_coarse: Optional[Union[List[float], torch.Tensor]] = None,
  ) -> List[Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]]:
    """Autoregressive Multiresolution Decoding.

    Args:
      input_ts: list of [B, T1, C], [B, T2, C] tensors for coarse/fine resolution streams.
      paddings: list of [B, T1, 1], [B, T2, 1] paddings for coarse/fine resolution streams.
      freq: [B] tensor of frequency indices.
      horizon_len: total forecast horizon length (in fine-res steps).
      offsets_fine: list of length B of denormalization offsets for fine-res stream.
      scales_fine: list of length B of denormalization scales for fine-res stream.
      output_patch_len: number of fine-res steps to decode per iteration.
      offsets_coarse: list of length B of denormalization offsets for coarse-res stream.
      scales_coarse: list of length B of denormalization scales for coarse-res stream.

    Returns:
      A list of length B of dictionaries:
        - "mean": mean forecast array of length `horizon_len`,
        - "quantiles": dict of quantile forecasts, each an array of length `horizon_len`.
    """
    if self.config.agg_factor <= 0:
      raise ValueError("agg_factor must be positive for autoregressive decoding.")

    q_levels = self.config.quantiles
    expected_q_plus_mean = len(q_levels) + 1
    expected_q_only = len(q_levels)

    if not isinstance(input_ts, (List, Tuple)) or len(input_ts) != 2:
      raise ValueError("Multi-resolution autoregressive decoding expects [coarse_res, fine_res] inputs.")

    coarse_ts, fine_ts = input_ts
    coarse_pad, fine_pad = paddings

    # Ensure 3D shapes
    if coarse_ts.ndim == 2:
      coarse_ts = coarse_ts.unsqueeze(-1)
    if fine_ts.ndim == 2:
      fine_ts = fine_ts.unsqueeze(-1)
    if coarse_pad.ndim == 2:
      coarse_pad = coarse_pad.unsqueeze(-1)
    if fine_pad.ndim == 2:
      fine_pad = fine_pad.unsqueeze(-1)

    device = fine_ts.device
    B = fine_ts.shape[0]
    patch_len = self.config.patch_len
    output_patch_len = output_patch_len or self.config.horizon_len

    # Offsets/scales used to invert the initial stream-normalization and to denormalize predictions.
    offsets_fine = [0.0] * B if offsets_fine is None else offsets_fine
    scales_fine = [1.0] * B if scales_fine is None else scales_fine
    batch_offsets_fine = torch.as_tensor(offsets_fine, dtype=torch.float32, device=device).view(B, 1)
    batch_scales_fine = torch.as_tensor(scales_fine, dtype=torch.float32, device=device).view(B, 1)

    if (offsets_coarse is None) ^ (scales_coarse is None):
      raise ValueError("offsets_coarse and scales_coarse must be provided together (or both omitted).")

    batch_offsets_coarse = None
    batch_scales_coarse = None
    if offsets_coarse is not None and scales_coarse is not None:
      batch_offsets_coarse = torch.as_tensor(offsets_coarse, dtype=torch.float32, device=device).view(B, 1)
      batch_scales_coarse = torch.as_tensor(scales_coarse, dtype=torch.float32, device=device).view(B, 1)

    # Keep working windows aligned and trimming to patch boundaries for performing decoding step.
    max_ctx_len_coarse = max(patch_len, (coarse_ts.shape[1] // patch_len) * patch_len)
    coarse_ts, coarse_pad = self._trim_context(coarse_ts, coarse_pad, max_ctx_len_coarse)
    max_ctx_len_fine = max(patch_len, (fine_ts.shape[1] // patch_len) * patch_len)
    fine_ts, fine_pad = self._trim_context(fine_ts, fine_pad, max_ctx_len_fine)

    # Reconstruct raw/original-scale contexts so stream stats can be recomputed each AR step.
    fine_raw = fine_ts.to(torch.float32) * (batch_scales_fine.view(B, 1, 1) + Constants.EPS.value) + batch_offsets_fine.view(B, 1, 1)
    if batch_offsets_coarse is not None and batch_scales_coarse is not None:
      coarse_raw = (
        coarse_ts.to(torch.float32) * (batch_scales_coarse.view(B, 1, 1) + Constants.EPS.value)
        + batch_offsets_coarse.view(B, 1, 1)
      )
    else:
      coarse_raw = coarse_ts.to(torch.float32)

    remaining = horizon_len
    mean_chunks = []
    quant_chunks = []
    coarse_source_buffer = torch.empty((B, 0), device=device, dtype=torch.float32)
    
    # Number of decode steps to perform
    num_decode_patches = math.ceil(horizon_len / output_patch_len)

    for _ in range(num_decode_patches):
      # Recompute stream-level stats on the current raw contexts (per AR step).
      coarse_norm, _, _ = self._normalize_with_pad(coarse_raw, coarse_pad)
      fine_norm, offset_f, scale_f = self._normalize_with_pad(fine_raw, fine_pad)

      preds = self([coarse_norm, fine_norm], [coarse_pad.float(), fine_pad.float()], freq)
      if preds.ndim != 4:
        raise ValueError(f"Unexpected prediction rank: {preds.shape}")

      num_coarse_patches = coarse_norm.shape[1] // patch_len
      num_fine_patches = fine_norm.shape[1] // patch_len
      fine_patch_idx = num_coarse_patches + num_fine_patches - 1
      
      if fine_patch_idx >= preds.shape[1]:
        raise ValueError(f"Fine patch index {fine_patch_idx} out of range for preds shape {preds.shape}")

      fine_patch = preds[:, fine_patch_idx, :, :]
      
      if fine_patch.shape[1] < output_patch_len:
        raise ValueError(f"Model horizon - {fine_patch.shape[1]} < requested output_patch_len {output_patch_len}")
      
      fine_patch = fine_patch[:, :output_patch_len, :]

      C = fine_patch.shape[2]
      if C == expected_q_plus_mean:
        mean_channel = fine_patch[..., 0]      # [B, L]
        quant_block = fine_patch[..., 1:]      # [B, L, Q]
      elif C == expected_q_only:
        mean_channel = None
        quant_block = fine_patch
      else:
        raise ValueError(f"Channel count {C} != {expected_q_plus_mean} or {expected_q_only}")

      if mean_channel is None:
        mean_channel = quant_block.median(dim=-1).values  # [B, L]

      step_taken = min(remaining, output_patch_len)
      mean_denorm = mean_channel * (scale_f + Constants.EPS.value) + offset_f
      quant_denorm = quant_block * (scale_f.unsqueeze(-1) + Constants.EPS.value) + offset_f.unsqueeze(-1)
      mean_denorm = torch.nan_to_num(mean_denorm, nan=0.0, posinf=0.0, neginf=-0.0)
      quant_denorm = torch.nan_to_num(quant_denorm, nan=0.0, posinf=0.0, neginf=-0.0)

      mean_chunks.append(mean_denorm[:, :step_taken])
      quant_chunks.append(quant_denorm[:, :step_taken, :])
      remaining -= step_taken

      # Append raw/original-scale minute predictions for the next step.
      fine_append = mean_denorm[:, :output_patch_len].unsqueeze(-1)
      fine_raw = torch.cat([fine_raw, fine_append.to(fine_raw.dtype)], dim=1)
      fine_pad = torch.cat(
        [fine_pad, torch.zeros((B, output_patch_len, 1), device=device, dtype=fine_pad.dtype)],
        dim=1)

      # Aggregate fine predictions into the coarse stream using a streaming buffer so no remainder is dropped.
      coarse_source_buffer = torch.cat(
        [coarse_source_buffer, mean_denorm[:, :output_patch_len].to(torch.float32)],
        dim=1,
      )

      full_blocks = coarse_source_buffer.shape[1] // self.config.agg_factor
      
      if full_blocks > 0:
        take_len = full_blocks * self.config.agg_factor
        blocks = coarse_source_buffer[:, :take_len].view(B, full_blocks, self.config.agg_factor)
        coarse_agg = blocks.mean(dim=2)  # [B, full_blocks]
        coarse_source_buffer = coarse_source_buffer[:, take_len:]
        coarse_raw = torch.cat([coarse_raw, coarse_agg.unsqueeze(-1).to(coarse_raw.dtype)], dim=1)
        coarse_pad = torch.cat(
          [coarse_pad, torch.zeros((B, full_blocks, 1), device=device, dtype=coarse_pad.dtype)],
          dim=1,
        )

      # Keep contexts aligned and bounded.
      fine_raw, fine_pad = self._trim_context(fine_raw, fine_pad, max_ctx_len_fine)
      coarse_raw, coarse_pad = self._trim_context(coarse_raw, coarse_pad, max_ctx_len_coarse)

      if remaining <= 0:
        break

    mean_full = torch.cat(mean_chunks, dim=1)[:, :horizon_len]
    quant_full = torch.cat(quant_chunks, dim=1)[:, :horizon_len, :]

    mean_np = mean_full.cpu().numpy()
    quant_np = quant_full.cpu().numpy()

    final_predictions = []
    for i in range(B):
      q_arr = np.transpose(quant_np[i], (1, 0))  # [Q, H]
      final_predictions.append(
          {
            "mean": mean_np[i],
            "quantiles": {str(q_levels[q_i]): q_arr[q_i] for q_i in range(q_arr.shape[0])}
          }
      )
    
    return final_predictions
