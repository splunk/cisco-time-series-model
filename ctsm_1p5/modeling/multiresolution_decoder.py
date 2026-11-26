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

"""Fixed MultiResolution decoder for CTSM 1.5."""

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from .constants import *
from .layers import (
    AttentionProjections,
    RMSNorm,
    ResidualBlock,
    TransformerMLP,
)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding (RoPE) helper for attention.

    This is parameter-free (buffers only) and rotates query/key vectors based on position.
    """

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()

        if dim % 2 != 0:
            raise ValueError(f'RoPE requires an even head_dim, got dim={dim}.')

        inv_freq = 1.0 / base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        self.register_buffer('inv_freq', inv_freq, persistent=False)


    def _cos_sin(
        self,
        position_ids: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)

        pos = position_ids.to(torch.float32)
        freqs = torch.einsum('bt,d->btd', pos, self.inv_freq)

        return (freqs.cos().to(dtype), freqs.sin().to(dtype))


    def apply(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Rotate query or key coordinates in pairs."""

        real, imaginary = x.chunk(2, dim=-1)

        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        rotated_real = real * cos - imaginary * sin
        rotated_imaginary = imaginary * cos + real * sin

        return torch.cat((rotated_real, rotated_imaginary), dim=-1)


class CTSMAttentionRoPE(AttentionProjections):
    """Cisco TSM attention with rotary positional embeddings applied to Q/K."""

    def __init__(self) -> None:
        super().__init__(
            hidden_size=HIDDEN_SIZE,
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
        )
        self.rope = RotaryPositionalEmbedding(HEAD_DIM)


    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary attention with the supplied additive mask."""

        batch_size, sequence_length, _ = hidden_states.shape
        head_shape = (self.num_heads, self.head_dim)

        query, key, value = self.qkv_proj(hidden_states).chunk(3, dim=-1)
        query = self._scale_queries(query.unflatten(-1, head_shape)).transpose(1, 2)

        key = key.unflatten(-1, head_shape).transpose(1, 2)
        value = value.unflatten(-1, head_shape).transpose(1, 2)

        if position_ids is None:
            position_ids = torch.arange(sequence_length, device=hidden_states.device)

        if position_ids.size(-1) != sequence_length:
            raise ValueError("position_ids must contain one position per token")

        positions = position_ids.expand(batch_size, sequence_length)
        cos, sin = self.rope._cos_sin(positions, dtype=query.dtype)
        query = self.rope.apply(query, cos, sin)
        key = self.rope.apply(key, cos, sin)

        scores = torch.matmul(query, key.transpose(-1, -2)) + mask
        weights = scores.float().softmax(dim=-1).to(query.dtype)
        combined = torch.matmul(weights, value).transpose(1, 2).contiguous().flatten(2)

        return weights, self.o_proj(combined)


class CTSMDecoderLayerRoPE(nn.Module):
    """Cisco TSM decoder layer with RoPE attention."""

    def __init__(self) -> None:
        super().__init__()
        self.self_attn = CTSMAttentionRoPE()
        self.mlp = TransformerMLP(hidden_size=HIDDEN_SIZE, intermediate_size=INTERMEDIATE_SIZE)
        self.input_layernorm = RMSNorm(HIDDEN_SIZE, eps=RMS_NORM_EPS)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
        paddings: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply attention and the feed-forward residual block."""

        attention_input = self.input_layernorm(hidden_states)
        weights, attention_output = self.self_attn(attention_input, mask, position_ids)
        residual = hidden_states + attention_output

        return weights, self.mlp(residual, paddings)


class StackedDecoderRoPE(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([CTSMDecoderLayerRoPE() for _ in range(NUM_LAYERS)])


class MultiResolutionDecoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self._patch_len_coarse = self._patch_len_fine = PATCH_LENGTH
        self._num_mask_slots = OUTPUT_PATCH_LENGTH // PATCH_LENGTH

        self.input_ff_layer = ResidualBlock(
            input_dims=2 * PATCH_LENGTH, output_dims=HIDDEN_SIZE, hidden_dims=HIDDEN_SIZE,
        )
        self.horizon_ff_layer = ResidualBlock(
            input_dims=HIDDEN_SIZE, output_dims=PATCH_LENGTH * len(QUANTILES), hidden_dims=HIDDEN_SIZE,
        )

        self.stacked_transformer = StackedDecoderRoPE()
        self.multi_resolution = nn.Embedding(num_embeddings=3, embedding_dim=HIDDEN_SIZE)
        self.special_token = nn.Parameter(torch.zeros(1, 1, HIDDEN_SIZE))
        self.mask_token = nn.Parameter(torch.zeros(1, self._num_mask_slots, HIDDEN_SIZE))

        self.fine_tod_ff = ResidualBlock(
            input_dims=2 * PATCH_LENGTH, output_dims=HIDDEN_SIZE, hidden_dims=HIDDEN_SIZE,
        )
        self.fine_dow_ff = ResidualBlock(
            input_dims=2 * PATCH_LENGTH, output_dims=HIDDEN_SIZE, hidden_dims=HIDDEN_SIZE,
        )


    @staticmethod
    def update_running_stats(
        n: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pool the current patch into the previous population statistics."""
        observed = torch.logical_not(mask)
        patch_count = observed.sum(dim=-1, dtype=x.dtype)
        empty_patch = patch_count == 0
        patch_divisor = torch.where(empty_patch, 1, patch_count)

        patch_mean = (x * observed).sum(dim=-1) / patch_divisor
        patch_mean = torch.where(empty_patch, 0, patch_mean)
        patch_offsets = x - patch_mean[:, None]
        patch_variance = (patch_offsets.square() * observed).sum(dim=-1) / patch_divisor
        patch_variance = torch.where(empty_patch, 0, patch_variance)

        count = n + patch_count
        empty = count == 0
        divisor = torch.where(empty, 1, count)
        mean = (n * mu + patch_mean * patch_count) / divisor
        mean = torch.where(empty, 0, mean)

        previous_mass = n * sigma.square()
        patch_mass = patch_count * patch_variance
        previous_shift = n * (mu - mean).square()
        patch_shift = patch_count * (patch_mean - mean).square()
        variance = (previous_mass + patch_mass + previous_shift + patch_shift) / divisor
        variance = torch.where(empty, 0, variance)

        variance = variance.clamp_min(0)
        positive = variance > 0
        safe_variance = torch.where(positive, variance, 1)
        deviation = torch.where(positive, safe_variance.sqrt(), 0)

        return count, mean, deviation

    def _forward_transform(
        self,
        inputs: torch.Tensor,
        patched_pads: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Normalize each patch with causal running statistics."""

        count = torch.zeros(inputs.size(0), device=inputs.device)
        mean = torch.zeros_like(count)
        sigma = torch.zeros_like(count)

        patch_means = []
        patch_sigmas = []

        for patch_index in range(inputs.size(1)):
            count, mean, sigma = self.update_running_stats(
                count,
                mean,
                sigma,
                inputs[:, patch_index],
                patched_pads[:, patch_index],
            )
            patch_means.append(mean)
            patch_sigmas.append(sigma)

        means = torch.stack(patch_means, dim=1).unsqueeze(-1)
        divisors = torch.stack(patch_sigmas, dim=1).unsqueeze(-1)
        divisors = torch.where(divisors < TOLERANCE, 1.0, divisors)

        normalized = (inputs - means) / divisors
        is_legacy_pad = (inputs - PAD_VALUE).abs() < TOLERANCE

        return torch.where(is_legacy_pad, PAD_VALUE, normalized), (mean, sigma)


    def _preprocess_input(
        self,
        input_ts: torch.Tensor,
        input_padding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Pack normalized values and padding for the shared input projection."""

        values = input_ts.unflatten(-1, (-1, PATCH_LENGTH))
        padding = input_padding.unflatten(-1, (-1, PATCH_LENGTH))
        already_padded = (padding - 1.0).abs() < TOLERANCE
        is_legacy_pad = ~already_padded & ((values - PAD_VALUE).abs() < TOLERANCE)
        values = torch.where(already_padded, 0.0, values)
        padding = torch.where(is_legacy_pad, torch.ones_like(padding), padding)

        normalized, stats = self._forward_transform(values, padding)
        masked_normalized = normalized * (1.0 - padding)
        value_and_padding = torch.cat((masked_normalized, padding), dim=-1)

        return value_and_padding, padding.min(dim=-1).values, stats, masked_normalized


    def _preprocess_time_channel_with_ff(
        self,
        sin_ts: torch.Tensor,
        cos_ts: torch.Tensor,
        input_padding: torch.Tensor,
        ff_layer: nn.Module,
    ) -> torch.Tensor:
        """Tokenize a bounded sin/cos fine time channel (NO z-norm) via its own FF layer.

        sin_ts/cos_ts: [B, L] already left-padded to a patch_len multiple and aligned to
        the fine value grid. input_padding: [B, L] fine step-level padding (1 == pad).
        The two bounded features are patched and concatenated (2*patch_len per patch),
        padded positions zeroed, then projected to hidden_size. Returns [B, N, hidden].
        """

        bsize = sin_ts.shape[0]
        sin_p = sin_ts.view(bsize, -1, self._patch_len_fine)
        cos_p = cos_ts.view(bsize, -1, self._patch_len_fine)

        pads = input_padding.view(bsize, -1, self._patch_len_fine)
        keep = 1.0 - pads

        sin_p = sin_p * keep
        cos_p = cos_p * keep
        concat_inputs = torch.cat([sin_p, cos_p], dim=-1)

        return ff_layer(concat_inputs)


    def _forward_axial(self, h: torch.Tensor, gpad: torch.Tensor, Ncoarse: int) -> torch.Tensor:
        """Apply 24 time-wise layers followed by one space-wise layer.

        Args:
          h: [B, V, N, D]. Row 0 is the target, rows 1.. are the calendar channels.
          gpad: [B, V, N], 1 = padded, denied as an attention key.
          Ncoarse: coarse slot count, for the bidirectional-coarse block.

        Returns:
          [B, V, N, D].
        """

        B, V, N, D = h.shape
        layers = self.stacked_transformer.layers

        dtype, device = (h.dtype, h.device)
        limits = torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)
        mask_value = torch.tensor(-0.7 * limits.max, dtype=dtype, device="cpu")

        pad_bv = gpad.reshape(B * V, N)
        time_pad_mask = pad_bv.detach()[:, None, None, :] * mask_value

        row_attn = torch.zeros(V, 1, N, N, dtype=dtype, device=device)
        causal = h.new_ones((N, N)).triu(diagonal=1) * mask_value
        causal[:Ncoarse, :Ncoarse] = 0.0
        row_attn[0, 0] = causal

        time_first = time_pad_mask
        time_second = row_attn.unsqueeze(0).expand(B, V, 1, N, N).reshape(B * V, 1, N, N)
        time_first_queries, time_second_queries = time_first.size(-2), time_second.size(-2)

        if time_first_queries == 1 and time_second_queries > 1:
            time_first = torch.minimum(time_first, time_first.transpose(-1, -2))

        time_mask = torch.minimum(time_first, time_second)
        time_pos = torch.arange(N, device=device, dtype=torch.long)

        pad_bn = gpad.permute(0, 2, 1).reshape(B * N, V)
        space_mask = pad_bn.detach()[:, None, None, :] * mask_value

        row_dir = torch.zeros(V, V, dtype=dtype, device=device)
        row_dir[1:, 0] = mask_value.to(device)

        space_first = space_mask
        space_second = row_dir.reshape(1, 1, V, V)
        space_first_queries, space_second_queries = space_first.size(-2), space_second.size(-2)

        if space_first_queries == 1 and space_second_queries > 1:
            space_first = torch.minimum(space_first, space_first.transpose(-1, -2))

        space_mask = torch.minimum(space_first, space_second)
        space_pos = torch.zeros(V, device=device, dtype=torch.long)

        last = len(layers) - 1

        for i, layer in enumerate(layers):
            if i != last:
                _, out = layer(
                    hidden_states=h.reshape(B * V, N, D),
                    mask=time_mask,
                    paddings=pad_bv,
                    position_ids=time_pos,
                )
                h = out.reshape(B, V, N, D)
            else:
                _, out = layer(
                    hidden_states=h.permute(0, 2, 1, 3).reshape(B * N, V, D),
                    mask=space_mask,
                    paddings=pad_bn,
                    position_ids=space_pos,
                )
                h = out.reshape(B, N, V, D).permute(0, 2, 1, 3)

        return h


    def forward(
        self,
        input_ts: Sequence[torch.Tensor],
        input_padding: Sequence[torch.Tensor] | torch.Tensor,
        calendar_row_padding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Multi-resolution forward pass.
        Args:
          input_ts: list of batched tensors for coarse/fine resolution streams.
          input_padding: list of batched paddings for coarse/fine resolution streams.
          calendar_row_padding: optional [B, 2] mask marking whole calendar rows
            absent, ordered TOD then DOW. A 1 denies that row everywhere,
            including its query slots, which the per-position pads cannot express. None
            reproduces the training path exactly, as does an all-zero mask.
        """

        num_outputs = len(QUANTILES)

        if isinstance(input_ts, torch.Tensor):
            raise ValueError('MultiResolutionDecoder expects multi-resolution inputs as a list of tensors.')

        expected_len = 10
        if len(input_ts) != expected_len:
            raise ValueError(f'forward expects coarse/fine values and eight calendar streams, got {len(input_ts)} streams.')

        ts_coarse, ts_fine = input_ts[:2]
        tod_sin = tod_cos = dow_sin = dow_cos = None

        time_streams = input_ts[2:]
        _i = 0
        tod_sin, tod_cos = (time_streams[_i], time_streams[_i + 1])
        _i += 2
        dow_sin, dow_cos = (time_streams[_i], time_streams[_i + 1])
        _i += 2
        fut_tod_sin = fut_tod_cos = fut_dow_sin = fut_dow_cos = None
        fut_tod_sin, fut_tod_cos = (time_streams[_i], time_streams[_i + 1])
        _i += 2
        fut_dow_sin, fut_dow_cos = (time_streams[_i], time_streams[_i + 1])
        _i += 2

        pad_coarse, pad_fine = input_padding

        if ts_coarse.ndim == 3 and ts_coarse.shape[-1] == 1:
            ts_coarse = ts_coarse.squeeze(-1)

        if ts_fine.ndim == 3 and ts_fine.shape[-1] == 1:
            ts_fine = ts_fine.squeeze(-1)

        if pad_coarse.ndim == 3 and pad_coarse.shape[-1] == 1:
            pad_coarse = pad_coarse.squeeze(-1)

        if pad_fine.ndim == 3 and pad_fine.shape[-1] == 1:
            pad_fine = pad_fine.squeeze(-1)

        def _squeeze_time(t: torch.Tensor | None) -> torch.Tensor | None:
            if t is not None and t.ndim == 3 and (t.shape[-1] == 1):
                return t.squeeze(-1)

            return t

        tod_sin = _squeeze_time(tod_sin)
        tod_cos = _squeeze_time(tod_cos)

        dow_sin = _squeeze_time(dow_sin)
        dow_cos = _squeeze_time(dow_cos)

        fut_tod_sin = _squeeze_time(fut_tod_sin)
        fut_tod_cos = _squeeze_time(fut_tod_cos)

        fut_dow_sin = _squeeze_time(fut_dow_sin)
        fut_dow_cos = _squeeze_time(fut_dow_cos)

        pad_coarse_steps = pad_coarse
        pad_fine_steps = pad_fine

        patch_len_coarse = self._patch_len_coarse
        patch_len_fine = self._patch_len_fine

        rem_coarse = ts_coarse.shape[1] % patch_len_coarse

        if rem_coarse:
            pad_len = patch_len_coarse - rem_coarse
            ts_pad = torch.zeros((ts_coarse.shape[0], pad_len), device=ts_coarse.device, dtype=ts_coarse.dtype)
            pad_pad = torch.ones((pad_coarse.shape[0], pad_len), device=pad_coarse.device, dtype=pad_coarse.dtype)
            ts_coarse = torch.cat([ts_pad, ts_coarse], dim=1)
            pad_coarse = torch.cat([pad_pad, pad_coarse], dim=1)
            pad_coarse_steps = pad_coarse

        rem_fine = ts_fine.shape[1] % patch_len_fine
        if rem_fine:
            pad_len = patch_len_fine - rem_fine
            ts_pad = torch.zeros((ts_fine.shape[0], pad_len), device=ts_fine.device, dtype=ts_fine.dtype)
            pad_pad = torch.ones((pad_fine.shape[0], pad_len), device=pad_fine.device, dtype=pad_fine.dtype)
            ts_fine = torch.cat([ts_pad, ts_fine], dim=1)
            pad_fine = torch.cat([pad_pad, pad_fine], dim=1)
            pad_fine_steps = pad_fine

            if tod_sin is not None:
                tod_sin = torch.cat([ts_pad, tod_sin], dim=1)
                tod_cos = torch.cat([ts_pad, tod_cos], dim=1)

            if dow_sin is not None:
                dow_sin = torch.cat([ts_pad, dow_sin], dim=1)
                dow_cos = torch.cat([ts_pad, dow_cos], dim=1)

        coarse_ff_inputs, pad_coarse, stats_coarse, _ = self._preprocess_input(
            input_ts=ts_coarse, input_padding=pad_coarse,
        )
        fine_ff_inputs, pad_fine, stats_fine, _ = self._preprocess_input(input_ts=ts_fine, input_padding=pad_fine)

        n_coarse_tokens = coarse_ff_inputs.shape[1]

        packed_tokens = self.input_ff_layer(torch.cat([coarse_ff_inputs, fine_ff_inputs], dim=1))

        model_input_coarse, model_input_fine = torch.split(
            packed_tokens, [n_coarse_tokens, fine_ff_inputs.shape[1]], dim=1,
        )

        cal_fine_rows = []

        for sin_ts, cos_ts, fut_sin, fut_cos, ff_layer in (
            (tod_sin, tod_cos, fut_tod_sin, fut_tod_cos, self.fine_tod_ff),
            (dow_sin, dow_cos, fut_dow_sin, fut_dow_cos, self.fine_dow_ff),
        ):
            query_sin, query_cos = (fut_sin, fut_cos)
            n_context_tokens = sin_ts.shape[1] // patch_len_fine
            packed_tokens = self._preprocess_time_channel_with_ff(
                torch.cat([sin_ts, query_sin], dim=1),
                torch.cat([cos_ts, query_cos], dim=1),
                torch.cat([pad_fine_steps, torch.zeros_like(query_sin)], dim=1),
                ff_layer,
            )
            ctx_tokens, query_tokens = torch.split(packed_tokens, [n_context_tokens, self._num_mask_slots], dim=1)
            cal_fine_rows.append(torch.cat([ctx_tokens, query_tokens], dim=1))

        msk_tok = self.mask_token.to(model_input_fine.device).expand(model_input_fine.shape[0], -1, -1)
        model_input_fine = torch.cat([model_input_fine, msk_tok], dim=1)
        pad_fine = torch.cat(
            [pad_fine, torch.zeros(pad_fine.shape[0], msk_tok.shape[1], device=pad_fine.device, dtype=pad_fine.dtype)],
            dim=1,
        )
        B = model_input_coarse.shape[0]

        Ncoarse = model_input_coarse.shape[1]
        Nfine = model_input_fine.shape[1]

        D = model_input_coarse.shape[2]
        device = model_input_coarse.device

        spec_tok = self.special_token.to(device).expand(B, 1, D)
        spec_pad = torch.zeros(B, 1, device=device, dtype=pad_coarse.dtype)

        model_input = torch.cat([model_input_coarse, spec_tok, model_input_fine], dim=1)
        patched_padding = torch.cat([pad_coarse, spec_pad, pad_fine], dim=1)

        spec_len = 1

        lead = Ncoarse + spec_len
        lead_pad = torch.ones(B, lead, device=device, dtype=patched_padding.dtype)

        rows = [model_input]
        rows_pad = [patched_padding]

        if calendar_row_padding is not None:
            if calendar_row_padding.shape != (B, len(cal_fine_rows)):
                raise ValueError(f'calendar_row_padding must be [{B}, {len(cal_fine_rows)}], got {tuple(calendar_row_padding.shape)}.')

            calendar_row_padding = calendar_row_padding.to(device=device, dtype=patched_padding.dtype)

        for row_idx, row_fine in enumerate(cal_fine_rows):
            lead_content = torch.zeros(B, lead, D, device=device, dtype=model_input.dtype)
            rows.append(torch.cat([lead_content, row_fine], dim=1))
            row_pad = torch.cat([lead_pad, pad_fine], dim=1)

            if calendar_row_padding is not None:
                row_pad = torch.maximum(row_pad, calendar_row_padding[:, row_idx:row_idx + 1])

            rows_pad.append(row_pad)

        model_input = torch.stack(rows, dim=1)
        gpad = torch.stack(rows_pad, dim=1)

        mr_coarse = torch.zeros(Ncoarse, dtype=torch.long, device=device)
        mr_spec = torch.full((spec_len,), 1, dtype=torch.long, device=device)
        mr_fine = torch.full((Nfine,), 2, dtype=torch.long, device=device)

        mr_idx = torch.cat([mr_coarse, mr_spec, mr_fine], dim=0)
        mr_idx = mr_idx.unsqueeze(0).expand(B, -1)
        mr_emb = self.multi_resolution(mr_idx)
        model_input += mr_emb.unsqueeze(1)

        h = self._forward_axial(model_input, gpad, Ncoarse)

        M = self._num_mask_slots
        assert M > 0
        assert M * self._patch_len_fine == OUTPUT_PATCH_LENGTH

        queries = h[:, 0, -M:, :]
        out = self.horizon_ff_layer(queries)
        assert out.shape[-1] == self._patch_len_fine * num_outputs

        out = out.view(B, M, self._patch_len_fine, num_outputs)
        mu_f, sigma_f = stats_fine
        out = out * sigma_f.to(out.dtype)[:, None, None, None] + mu_f.to(out.dtype)[:, None, None, None]
        out = out.reshape(B, OUTPUT_PATCH_LENGTH, num_outputs)

        return out.unsqueeze(1)


    @staticmethod
    def _normalize_with_pad(
        context: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        clamp_range: tuple[float, float] = (-1000.0, 1000.0),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize a raw context while excluding left-padding from its statistics.

        ``decode`` appends forecasts in raw units. Recomputing these statistics on
        every autoregressive step keeps the next model input in the same normalized
        space as a freshly constructed context.
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
            raise ValueError(f'_normalize_with_pad expects [B, T] inputs, got context={context_2d.shape} pad={pad_2d.shape}')

        if context_2d.shape != pad_2d.shape:
            raise ValueError(f'context and pad must have same shape, got context={context_2d.shape} pad={pad_2d.shape}')

        valid = 1.0 - pad_2d
        count = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        context_mean = (context_2d * valid).sum(dim=1, keepdim=True) / count

        seq_len = context_2d.shape[1]
        seq_len_t = count.new_tensor(float(seq_len))

        pad_bool = pad_2d.to(dtype=torch.bool)
        filled_context = torch.where(pad_bool, context_mean, context_2d)

        context_std = filled_context.std(dim=1, keepdim=True, unbiased=False) * torch.sqrt(seq_len_t / count)
        context_std = context_std.clamp_min(0.01)
        normalized_unclamped = (context_2d - context_mean) / (context_std + EPS) * valid
        ctx_normalized = torch.clamp(normalized_unclamped, *clamp_range)

        retry_rows = (
            ~torch.isfinite(context_mean).all(dim=1, keepdim=True)
            | ~torch.isfinite(context_std).all(dim=1, keepdim=True)
            | ~torch.isfinite(normalized_unclamped).all(dim=1, keepdim=True)
        )

        if retry_rows.any():
            observed = valid != 0
            if not torch.isfinite(context_2d.masked_select(observed)).all():
                raise ValueError('decoder normalization requires finite valid observations')

            context64 = context_2d.to(dtype=torch.float64)
            valid64 = valid.to(dtype=torch.float64)
            count64 = valid64.sum(dim=1, keepdim=True).clamp_min(1.0)
            clean64 = torch.where(observed, context64, torch.zeros_like(context64))
            mean64 = (clean64 * valid64).sum(dim=1, keepdim=True) / count64
            seq_len64 = count64.new_tensor(float(seq_len))
            filled64 = torch.where(pad_bool, mean64, context64)
            std64 = (
                filled64.std(dim=1, keepdim=True, unbiased=False) * torch.sqrt(seq_len64 / count64)
            ).clamp_min(0.01)
            centered64 = torch.where(observed, context64 - mean64, torch.zeros_like(context64))
            normalized64 = torch.clamp(centered64 / (std64 + EPS) * valid64, *clamp_range)

            context_mean = torch.where(retry_rows, mean64.to(context_2d.dtype), context_mean)
            context_std = torch.where(retry_rows, std64.to(context_2d.dtype), context_std)
            ctx_normalized = torch.where(retry_rows, normalized64.to(context_2d.dtype), ctx_normalized)

            if (
                not torch.isfinite(context_mean).all()
                or not torch.isfinite(context_std).all()
                or (not torch.isfinite(ctx_normalized).all())
            ):
                raise FloatingPointError('decoder normalization retry produced nonfinite or unrepresentable statistics')

        if add_channel:
            ctx_normalized = ctx_normalized.unsqueeze(-1)

        return (ctx_normalized, context_mean, context_std)


    def _trim_context(
        self,
        ts: torch.Tensor,
        pad: torch.Tensor,
        max_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Trim context to be aligned to patch boundaries and within max length.
        Args:
          ts: [B, T, C] input time series tensor.
          pad: [B, T, 1] input padding tensor.
          max_len: maximum allowed length.
        Returns:
          Trimmed (ts, pad) tensors.
        """

        patch_len = PATCH_LENGTH
        target_len = max(patch_len, max_len // patch_len * patch_len)

        if ts.shape[1] > target_len:
            ts = ts[:, -target_len:, :]
            pad = pad[:, -target_len:, :]

        rem = ts.shape[1] % patch_len
        if rem:
            ts = ts[:, rem:, :]
            pad = pad[:, rem:, :]

        return (ts, pad)


    @staticmethod
    def _extrapolate_future_clock(
        tod: torch.Tensor,
        dow: torch.Tensor | None,
        n_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Extrapolate the horizon clock from the last two context steps.

        Mirrors trainer.py so inference sees the same future-clock construction the
        weights were trained on. tod: [B, T] raw seconds-since-midnight. dow: [B, T]
        raw weekday 0..6, or None. Returns (fut_tod [B, n_steps] in seconds,
        fut_wpos [B, n_steps] continuous week position in [0, 7), or None).

        Left-padding keeps the right edge real, so the last two steps give the true
        cadence. Each mod keeps a wrap (midnight, Sunday->Monday) positive and keeps
        the accumulated phase exact instead of drifting in float32.
        """

        if tod.shape[1] < 2:
            raise ValueError(f'future-clock extrapolation needs >= 2 context steps, got {tod.shape[1]}.')

        last_tod = tod[:, -1:].float()
        prev_tod = tod[:, -2:-1].float()

        steps = torch.arange(1, n_steps + 1, device=tod.device, dtype=torch.float32)
        step_seconds = (last_tod - prev_tod) % 86400.0

        fut_tod = (last_tod + step_seconds * steps) % 86400.0
        fut_wpos = None

        if dow is not None:
            w_last = dow[:, -1:].float() + last_tod / 86400.0
            w_prev = dow[:, -2:-1].float() + prev_tod / 86400.0
            step_days = (w_last - w_prev) % 7.0
            fut_wpos = (w_last + step_days * steps) % 7.0

        return (fut_tod, fut_wpos)


    def _clock_streams(
        self,
        tod: torch.Tensor | None,
        dow: torch.Tensor | None,
    ) -> tuple[list[torch.Tensor], torch.Tensor | None, torch.Tensor | None]:
        """Build the sin/cos streams `forward` unpacks, from the raw fine-grid clock.

        tod/dow: [B, T] or [B, T, 1] raw, aligned to the fine value grid. Emits the
        streams in the exact order forward's `time_streams` unpack expects. Also returns the RAW
        extrapolated horizon clock for the autoregressive loop to append.
        """

        two_pi = 2.0 * math.pi
        streams: list[torch.Tensor] = []

        if tod is None:
            return (streams, None, None)

        if tod.ndim == 3:
            tod = tod.squeeze(-1)

        if dow is not None and dow.ndim == 3:
            dow = dow.squeeze(-1)

        tod_frac = tod.float() / 86400.0
        ang = two_pi * tod_frac

        streams += [torch.sin(ang), torch.cos(ang)]
        ang = two_pi * (dow.float() + tod_frac) / 7.0
        streams += [torch.sin(ang), torch.cos(ang)]

        fut_tod = fut_wpos = None
        fut_tod, fut_wpos = self._extrapolate_future_clock(tod, dow, OUTPUT_PATCH_LENGTH)

        ang = two_pi * fut_tod / 86400.0
        streams += [torch.sin(ang), torch.cos(ang)]
        ang = two_pi * fut_wpos / 7.0
        streams += [torch.sin(ang), torch.cos(ang)]

        return (streams, fut_tod, fut_wpos)


    def decode(
        self,
        input_ts: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor],
        paddings: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor],
        *,
        horizon_len: int,
        offsets: Sequence[float] | np.ndarray | torch.Tensor | None = None,
        scales: Sequence[float] | np.ndarray | torch.Tensor | None = None,
        coarse_offsets: Sequence[float] | np.ndarray | torch.Tensor | None = None,
        coarse_scales: Sequence[float] | np.ndarray | torch.Tensor | None = None,
        fine_tod: torch.Tensor | None = None,
        fine_dow: torch.Tensor | None = None,
        calendar_row_padding: torch.Tensor | None = None,
        coarse_source_history: torch.Tensor | None = None,
        coarse_source_padding: torch.Tensor | None = None,
        feedback_quantile_index: int | None = None,
        _first_block_cache: list[torch.Tensor] | None = None,
    ) -> list[dict[str, np.ndarray | dict[str, np.ndarray]]]:
        """Autoregressive Multiresolution Decoding.

        Args:
          input_ts: list of [B, T1, C], [B, T2, C] tensors for coarse/fine resolution streams.
          paddings: list of [B, T1, 1], [B, T2, 1] paddings for coarse/fine resolution streams.
          horizon_len: total forecast horizon length (in fine-res steps).
          offsets: list of length B of denormalization offsets for fine-res stream.
          scales: list of length B of denormalization scales for fine-res stream.
          coarse_offsets: list of length B of normalization offsets for the coarse-res
            stream (the mean used to normalize coarse_ts). Needed to re-normalize
            autoregressively-appended coarse points into the coarse stream's own space.
            Defaults to `offsets` (the fine stats).
          coarse_scales: list of length B of normalization scales for the coarse-res
            stream (the std used to normalize coarse_ts). Defaults to `scales`.
          fine_tod: [B, T2] or [B, T2, 1] RAW seconds-since-midnight (0..86399),
            aligned step-for-step with the fine value stream. Required when any
            calendar row is active (masked UV rows still carry placeholder clocks).
          fine_dow: [B, T2] or [B, T2, 1] RAW weekday (0..6), same alignment.
            The HORIZON clock is not passed in -- it is extrapolated from the last
            two context steps, the same way trainer.py does it.
          calendar_row_padding: optional [B, n_fine_time] whole-row absence mask, forwarded
            unchanged to every autoregressive iteration -- it is indexed per (series, row),
            not per window, so unlike the clock streams it does not grow with the horizon.
          coarse_source_history: raw observed-history tail, right-aligned in
            [B, COARSE_CONTEXT_LENGTH * AGG_FACTOR] (or with a singleton channel). Required to
            rebuild coarse means at each new origin during multi-block decoding.
          coarse_source_padding: matching left-padding mask; 1 for absent, 0 for real.
          feedback_quantile_index: quantile to append to the fine and coarse history
            between blocks. The default is native P50. Each block still returns
            its P50 as the point forecast and all native quantile levels.
          _first_block_cache: private cache scoped to paths sharing one source batch.

        Returns:
          A list of length B of dictionaries, each containing:
            - "mean": a NumPy array of length `horizon_len` containing the median,
            - "quantiles": a dict mapping each quantile string to its NumPy array.
        """

        output_patch_len = OUTPUT_PATCH_LENGTH

        if fine_tod is None:
            raise ValueError('decode requires fine_tod (raw seconds since midnight, on the fine grid).')

        if fine_dow is None:
            raise ValueError('decode requires fine_dow (raw weekday 0-6).')

        q_levels = QUANTILES

        if feedback_quantile_index is not None and (
            not isinstance(feedback_quantile_index, int)
            or isinstance(feedback_quantile_index, bool)
            or not 0 <= feedback_quantile_index < len(q_levels)
        ):
            raise ValueError(f"feedback_quantile_index must be an integer from 0 to {len(q_levels) - 1}")

        if not isinstance(input_ts, (list, tuple)) or len(input_ts) != 2:
            raise ValueError('Multi-resolution autoregressive decoding expects [coarse_res, fine_res] inputs.')

        coarse_ts, fine_ts = input_ts
        coarse_pad, fine_pad = paddings

        if coarse_ts.ndim == 2:
            coarse_ts = coarse_ts.unsqueeze(-1)

        if fine_ts.ndim == 2:
            fine_ts = fine_ts.unsqueeze(-1)

        if coarse_pad.ndim == 2:
            coarse_pad = coarse_pad.unsqueeze(-1)

        if fine_pad.ndim == 2:
            fine_pad = fine_pad.unsqueeze(-1)

        if fine_tod is not None and fine_tod.ndim == 2:
            fine_tod = fine_tod.unsqueeze(-1)

        if fine_dow is not None and fine_dow.ndim == 2:
            fine_dow = fine_dow.unsqueeze(-1)

        if fine_tod is not None:
            fine_tod = fine_tod.float()

        if fine_dow is not None:
            fine_dow = fine_dow.float()

        for _name, _clk in (('fine_tod', fine_tod), ('fine_dow', fine_dow)):
            if _clk is not None and _clk.shape[1] != fine_ts.shape[1]:
                raise ValueError(
                    f'{_name} has {_clk.shape[1]} steps but the fine stream has {fine_ts.shape[1]}; '
                    'the clock must be aligned step-for-step.'
                )

        device = fine_ts.device
        batch_size = fine_ts.shape[0]

        patch_len_coarse = self._patch_len_coarse
        patch_len_fine = self._patch_len_fine

        offsets = [0.0] * batch_size if offsets is None else offsets
        scales = [1.0] * batch_size if scales is None else scales

        batch_offsets = torch.as_tensor(offsets, dtype=torch.float32, device=device).view(batch_size, 1)
        batch_scales = torch.as_tensor(scales, dtype=torch.float32, device=device).view(batch_size, 1)

        coarse_offsets = offsets if coarse_offsets is None else coarse_offsets
        coarse_scales = scales if coarse_scales is None else coarse_scales

        batch_coarse_offsets = torch.as_tensor(
            coarse_offsets, dtype=torch.float32, device=device,
        ).view(batch_size, 1)
        batch_coarse_scales = torch.as_tensor(
            coarse_scales, dtype=torch.float32, device=device,
        ).view(batch_size, 1)

        max_ctx_len_coarse = max(patch_len_coarse, coarse_ts.shape[1] // patch_len_coarse * patch_len_coarse)
        coarse_ts, coarse_pad = self._trim_context(coarse_ts, coarse_pad, max_ctx_len_coarse)
        max_ctx_len_fine = max(patch_len_fine, fine_ts.shape[1] // patch_len_fine * patch_len_fine)

        if fine_tod is not None:
            fine_tod, _ = self._trim_context(fine_tod, fine_pad, max_ctx_len_fine)

        if fine_dow is not None:
            fine_dow, _ = self._trim_context(fine_dow, fine_pad, max_ctx_len_fine)

        fine_ts, fine_pad = self._trim_context(fine_ts, fine_pad, max_ctx_len_fine)
        source_raw = source_pad = None
        provided_source = coarse_source_history is not None or coarse_source_padding is not None

        if provided_source or horizon_len > output_patch_len:
            if not isinstance(coarse_source_history, torch.Tensor) or not isinstance(coarse_source_padding, torch.Tensor):
                raise ValueError(
                    'origin_aligned multi-block decoding requires raw coarse_source_history and coarse_source_padding tensors'
                )

            source_raw, source_pad = (coarse_source_history, coarse_source_padding)

            if source_raw.ndim == 3 and source_raw.shape[-1] == 1:
                source_raw = source_raw.squeeze(-1)

            if source_pad.ndim == 3 and source_pad.shape[-1] == 1:
                source_pad = source_pad.squeeze(-1)

            expected_shape = (batch_size, COARSE_CONTEXT_LENGTH * AGG_FACTOR)
            if source_raw.shape != expected_shape or source_pad.shape != expected_shape:
                raise ValueError(f'raw coarse source and padding must have shape {expected_shape}')

            if source_raw.dtype != torch.float32 or source_pad.dtype not in (torch.float32, torch.bool):
                raise ValueError('raw coarse source must be float32 and padding float32 or bool')

            if source_raw.device != device or source_pad.device != device:
                raise ValueError('raw coarse source and padding must be on the fine input device')

            if not torch.all((source_pad == 0) | (source_pad == 1)):
                raise ValueError('raw coarse source padding must contain only 0 or 1')

            source_pad = source_pad.to(dtype=torch.float32)
            if torch.any(source_pad[:, 1:] > source_pad[:, :-1]):
                raise ValueError('raw coarse source padding must be a contiguous left prefix')

            if not torch.isfinite(source_raw.masked_select(source_pad == 0)).all():
                raise ValueError('raw coarse source has nonfinite real observations')

            source_raw = torch.where(source_pad.bool(), torch.zeros_like(source_raw), source_raw)

        fine_raw = (
            fine_ts.to(torch.float32) * (batch_scales.view(batch_size, 1, 1) + EPS)
            + batch_offsets.view(batch_size, 1, 1)
        )
        coarse_raw = (
            coarse_ts.to(torch.float32) * (batch_coarse_scales.view(batch_size, 1, 1) + EPS)
            + batch_coarse_offsets.view(batch_size, 1, 1)
        )

        remaining = horizon_len
        mean_chunks = []
        quant_chunks = []

        num_decode_patches = math.ceil(horizon_len / output_patch_len)

        for block_index in range(num_decode_patches):
            coarse_norm, _, _ = self._normalize_with_pad(coarse_raw, coarse_pad)
            fine_norm, offset_f, scale_f = self._normalize_with_pad(fine_raw, fine_pad)
            clock_streams, fut_tod, fut_wpos = self._clock_streams(fine_tod, fine_dow)
            share_first = block_index == 0 and _first_block_cache is not None

            if share_first and _first_block_cache:
                preds = _first_block_cache[0]
            else:
                preds = self(
                    [coarse_norm, fine_norm] + clock_streams,
                    [coarse_pad.float(), fine_pad.float()],
                    calendar_row_padding=calendar_row_padding,
                )
                if share_first:
                    _first_block_cache.append(preds)

            if not torch.isfinite(preds).all():
                raise FloatingPointError('Nonfinite raw decoder forecast')

            if preds.ndim != 4:
                raise ValueError(f'Unexpected prediction rank: {preds.shape}')

            fine_patch = preds[:, -1, :, :]
            if fine_patch.shape[1] < output_patch_len:
                raise ValueError(f'Model horizon - {fine_patch.shape[1]} < requested output_patch_len {output_patch_len}')

            fine_patch = fine_patch[:, :output_patch_len, :]
            C = fine_patch.shape[2]
            if C != len(q_levels):
                raise ValueError(f'Expected {len(q_levels)} quantile channels, got {C}')

            quant_block = fine_patch
            mean_channel = quant_block[..., MEDIAN_QUANTILE_INDEX]
            step_taken = min(remaining, output_patch_len)
            mean_denorm = mean_channel * (scale_f + EPS) + offset_f
            quant_denorm = quant_block * (scale_f.unsqueeze(-1) + EPS) + offset_f.unsqueeze(-1)

            if not torch.isfinite(mean_denorm).all() or not torch.isfinite(quant_denorm).all():
                raise FloatingPointError('Nonfinite denormalized forecast before decoder nan_to_num')

            mean_denorm = torch.nan_to_num(mean_denorm, nan=0.0, posinf=0.0, neginf=-0.0)
            quant_denorm = torch.nan_to_num(quant_denorm, nan=0.0, posinf=0.0, neginf=-0.0)
            mean_chunks.append(mean_denorm[:, :step_taken])
            quant_chunks.append(quant_denorm[:, :step_taken, :])
            remaining -= step_taken

            if remaining <= 0:
                break

            feedback = (
                mean_denorm if feedback_quantile_index is None else quant_denorm[..., feedback_quantile_index]
            )
            feedback = feedback[:, :output_patch_len]
            fine_append = feedback.unsqueeze(-1)
            fine_raw = torch.cat([fine_raw, fine_append.to(fine_raw.dtype)], dim=1)
            fine_pad = torch.cat(
                [fine_pad, torch.zeros((batch_size, output_patch_len, 1), device=device, dtype=fine_pad.dtype)],
                dim=1,
            )

            if fine_tod is not None and fut_tod is None:
                fut_tod, fut_wpos = self._extrapolate_future_clock(
                    fine_tod.squeeze(-1),
                    None if fine_dow is None else fine_dow.squeeze(-1),
                    OUTPUT_PATCH_LENGTH,
                )

            if fine_tod is not None:
                fine_tod = torch.cat([fine_tod, fut_tod[:, :output_patch_len].unsqueeze(-1)], dim=1)

            if fine_dow is not None:
                fine_dow = torch.cat(
                    [
                        fine_dow,
                        torch.remainder(
                            torch.round(fut_wpos[:, :output_patch_len] - fut_tod[:, :output_patch_len] / 86400.0),
                            7.0,
                        ).unsqueeze(-1),
                    ],
                    dim=1,
                )

            source_len = source_raw.shape[1]
            source_raw = torch.cat([source_raw, feedback.to(torch.float32)], dim=1)[:, -source_len:]
            source_pad = torch.cat(
                [source_pad, torch.zeros((batch_size, output_patch_len), device=device, dtype=source_pad.dtype)],
                dim=1,
            )[:, -source_len:]

            blocks = source_raw.reshape(batch_size, COARSE_CONTEXT_LENGTH, AGG_FACTOR)
            block_pads = source_pad.reshape(batch_size, COARSE_CONTEXT_LENGTH, AGG_FACTOR)
            complete = torch.all(block_pads == 0, dim=-1)
            block_means = blocks.mean(dim=-1)
            retry_bins = complete & ~torch.isfinite(block_means)

            if retry_bins.any():
                block_means = block_means.clone()
                block_means[retry_bins] = blocks[retry_bins].to(torch.float64).mean(dim=-1).to(blocks.dtype)
                if not torch.isfinite(block_means.masked_select(complete)).all():
                    raise FloatingPointError('origin-aligned coarse aggregation retry produced nonfinite valid means')

            means = torch.where(complete, block_means, torch.zeros_like(blocks[:, :, 0]))
            coarse_raw = means.unsqueeze(-1)
            coarse_pad = (~complete).to(dtype=coarse_pad.dtype).unsqueeze(-1)

            if fine_tod is not None:
                fine_tod, _ = self._trim_context(fine_tod, fine_pad, max_ctx_len_fine)

            if fine_dow is not None:
                fine_dow, _ = self._trim_context(fine_dow, fine_pad, max_ctx_len_fine)

            fine_raw, fine_pad = self._trim_context(fine_raw, fine_pad, max_ctx_len_fine)
            coarse_raw, coarse_pad = self._trim_context(coarse_raw, coarse_pad, max_ctx_len_coarse)

        mean_full = torch.cat(mean_chunks, dim=1)[:, :horizon_len]
        quant_full = torch.cat(quant_chunks, dim=1)[:, :horizon_len, :]
        mean_np = mean_full.cpu().numpy()
        quant_np = quant_full.cpu().numpy()

        final_predictions = []

        for i in range(batch_size):
            q_arr = np.transpose(quant_np[i], (1, 0))
            final_predictions.append({
                'mean': mean_np[i],
                'quantiles': {str(q_levels[q_i]): q_arr[q_i] for q_i in range(q_arr.shape[0])},
            })

        return final_predictions
