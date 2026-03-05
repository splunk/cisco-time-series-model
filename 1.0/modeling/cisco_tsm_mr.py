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
import logging
from os import path
from typing import Any, List, Sequence, Union, Tuple, Dict, Optional
from typing_extensions import override

import numpy as np

import torch

from huggingface_hub import snapshot_download

from timesfm import TimesFmHparams, TimesFmCheckpoint
from timesfm.timesfm_torch import TimesFmTorch
from timesfm.timesfm_base import strip_leading_nans, linear_interpolation

from .patched_decoder_multi_resolution import CiscoTsmMRConfig, PatchedTSMultiResolutionDecoder
from .constants import Constants


class CiscoTsmMR(TimesFmTorch):
  """Cisco Time Series Model Multi-resolution Forecast API."""

  @override
  def __init__(
    self,
    hparams: TimesFmHparams,
    checkpoint: TimesFmCheckpoint,
    *,
    agg_factor: int = Constants.AGG_FACTOR.value,
  ) -> None:
    """Initializes the CiscoTsmMR forecasting model with the given hyperparameters and checkpoint.
    Args:
      hparams: TimesFmHparams object containing model hyperparameters.
      checkpoint: TimesFmCheckpoint object containing checkpoint info (local or HF repo).
      agg_factor: Aggregation factor to derive coarse context from fine context.
    Returns:
      None
    """
    
    if agg_factor <= 0:
      raise ValueError("agg_factor must be positive")

    self.agg_factor = agg_factor

    if self.agg_factor != Constants.AGG_FACTOR.value:
      logging.warning(
        "agg_factor=%d differs from the value used during training (%d). The model is calibrated for %d-to-1 aggregation; other values are unsupported and may yield unreliable forecasts.",
        self.agg_factor,
        Constants.AGG_FACTOR.value,
        Constants.AGG_FACTOR.value,
      )
    
    super().__init__(hparams, checkpoint)


  @override
  def __post_init__(self):
    # Building MR config
    self._model_config = CiscoTsmMRConfig(
      num_layers=self.num_layers,
      num_heads=self.num_heads,
      hidden_size=self.model_dims,
      intermediate_size=self.model_dims,
      patch_len=self.input_patch_len,
      horizon_len=self.output_patch_len,
      head_dim=self.model_dims // self.num_heads,
      quantiles=self.quantiles,
      use_positional_embedding=self.use_pos_emb,
      use_resolution_embeddings=True,
      use_special_token=True,
      agg_factor=self.agg_factor,
    )
    self._model = None
    self.num_cores = 1
    self.global_batch_size = self.per_core_batch_size
    self._device = torch.device("cpu")
    if (torch.cuda.is_available() and self.backend == "gpu"):
      self._device = torch.device("cuda:0")


  @override
  def load_from_checkpoint(
    self,
    checkpoint: TimesFmCheckpoint,
  ) -> None:
    """Loads a Multiresolution Model checkpoint from path and prepares the MR decoder for inference.
    Args:
      checkpoint: TimesFmCheckpoint object containing checkpoint info (local or HF repo).
    Returns:
      None
    """

    checkpoint_path = checkpoint.path
    repo_id = checkpoint.huggingface_repo_id
    
    if checkpoint_path is None:
      checkpoint_path = path.join(
        snapshot_download(repo_id, local_dir=checkpoint.local_dir),
        Constants.CHECKPOINT_NAME.value
      )
    
    loaded_checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")

    self._model = PatchedTSMultiResolutionDecoder(self._model_config)

    logging.info("Loading checkpoint from %s", checkpoint_path)
    incompatible = self._model.load_state_dict(loaded_checkpoint, strict=True)

    if getattr(incompatible, "missing_keys", None) or getattr(incompatible, "unexpected_keys", None):
      logging.info(
        "MR decoder state load differences. missing=%s unexpected=%s",
        getattr(incompatible, "missing_keys", []),
        getattr(incompatible, "unexpected_keys", []),
      )

    logging.info("Loaded model from checkpoint: %s", checkpoint_path)
    logging.info("Sending checkpoint to device %s", self._device)

    self._model.to(self._device)
    self._model.eval()
  

  @staticmethod
  def slice_fine_context(
    series: Union[np.ndarray, Sequence[float]], 
    fine_len: int = Constants.CONTEXT_LEN_FINE.value
  ) -> np.ndarray:
    """Return the rightmost fine_len points (or entire series if shorter).

    Args:
      series: fine-resolution (e.g minute-level) time series data.
      fine_len: desired length of fine-level context to extract.

    Returns:
      1D array representing the fine-level context of length <= fine_len.
    """
    series = np.asarray(series, dtype=np.float32).reshape(-1)
    return series[-fine_len:]
  
  
  @staticmethod
  def build_coarse_context(
    series: np.ndarray, 
    agg_factor: int = Constants.AGG_FACTOR.value, 
    max_coarse_ctx: int = Constants.CONTEXT_LEN_COARSE.value
  ) -> np.ndarray:
    """Construct coarse context by:
    1. Taking up to rightmost (max_coarse_ctx * block) raw fine samples.
    2. Partitioning into consecutive non-overlapping blocks of 'block' size from left to right (chronological order preserved).
    3. Computing the mean of each block.
    
    Args:
      series: array of fine-resolution (e.g minute-level) time series data.
      agg_factor: aggregation factor to form coarse blocks.
      max_coarse_ctx: maximum number of coarse points to return.
    
    Returns:
      1D array representing coarse context with length <= max_coarse_ctx.
    """
    series = np.asarray(series, dtype=np.float32).reshape(-1)

    needed_raw = max_coarse_ctx * agg_factor
    raw_slice = series[-needed_raw:]

    # Ensure we only form full blocks; drop partial leading block if length not multiple.
    remainder = raw_slice.shape[0] % agg_factor
    if remainder:
      raw_slice = raw_slice[remainder:]  # align to block boundary at the right edge

    if raw_slice.shape[0] == 0:
      return np.empty((0,), dtype=np.float32)

    coarse = raw_slice.reshape(-1, agg_factor).mean(axis=1, dtype=np.float32)
    if coarse.shape[0] > max_coarse_ctx:
      coarse = coarse[-max_coarse_ctx:]
    return coarse.astype(np.float32, copy=False)


  @staticmethod
  def build_multi_resolution(
    series: np.ndarray, 
    agg_factor: int = Constants.AGG_FACTOR.value
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Builds multi-resolution contexts from a fine-resolution time series.
    Args:
      series: array of fine-resolution (e.g minute-level) time series data.
      agg_factor: aggregation factor to form coarse blocks.
    Returns:
      Tuple of:
      - coarse_ctx: 1D array representing the coarse context.
      - fine_ctx: 1D array representing the fine context.
    """

    coarse_ctx = CiscoTsmMR.build_coarse_context(series, agg_factor=agg_factor, max_coarse_ctx=Constants.CONTEXT_LEN_COARSE.value)
    fine_ctx = CiscoTsmMR.slice_fine_context(series, fine_len=Constants.CONTEXT_LEN_FINE.value)
    return coarse_ctx, fine_ctx
  

  @staticmethod
  def _normalize_inputs(inputs: Sequence[Any]) -> List[Dict[str, Any]]:
    """Normalizes input series into a list of (possibly multi-resolution) series.

    Args:
      inputs: single series or list/tuple of series. Each series can be:
        - single-resolution: list-like or array-like of fine-resolution data.
        - multi-resolution: tuple/list of two elements (coarse_series, fine_series)

    Returns:
      List of dictionaries, each with keys:
        - "multi_res": bool indicating if multi-resolution.
        - "coarse": 1D float32 numpy array (if multi_res is True).
        - "fine": 1D float32 numpy array.
    """

    def _to_numpy_1d(x: Any) -> np.ndarray:
      if isinstance(x, np.ndarray):
        return np.asarray(x, dtype=np.float32).reshape(-1)
      if isinstance(x, torch.Tensor):
        return x.detach().cpu().to(torch.float32).reshape(-1).numpy()
      if isinstance(x, (List, Tuple)):
        return np.asarray(x, dtype=np.float32).reshape(-1)
      return np.asarray([x], dtype=np.float32)

    if isinstance(inputs, (np.ndarray, torch.Tensor)):
      return [{"multi_res": False, "fine": _to_numpy_1d(inputs)}]

    if not isinstance(inputs, (List, Tuple)):
      return [{"multi_res": False, "fine": _to_numpy_1d(inputs)}]

    if inputs and all(not isinstance(x, (List, Tuple, np.ndarray, torch.Tensor)) for x in inputs):
      return [{"multi_res": False, "fine": _to_numpy_1d(inputs)}]

    normalized_series = []
    for elem in inputs:
      if (
        isinstance(elem, (List, Tuple))
        and len(elem) == 2
        and all(isinstance(sub, (List, Tuple, np.ndarray, torch.Tensor)) for sub in elem)
      ):
        normalized_series.append(
          {"multi_res": True, "coarse": _to_numpy_1d(elem[0]), "fine": _to_numpy_1d(elem[1])}
        )
      else:
        normalized_series.append({"multi_res": False, "fine": _to_numpy_1d(elem)})

    return normalized_series
  

  @staticmethod
  def _replace_non_finite(arr: np.ndarray) -> np.ndarray:
    """Replaces non-finite values (inf, -inf) in the array with NaN.
    Args:
      arr: input numpy array.
    Returns:
      numpy array with non-finite values replaced by NaN.
    """
    if arr.size == 0:
      return arr
    if np.isfinite(arr).all():
      return arr
    arr = arr.copy()
    arr[~np.isfinite(arr)] = np.nan
    
    return arr

  
  @override
  def forecast(self, 
               inputs: Sequence[Any],
               horizon_len: Optional[int] = None, 
               batch_size: int = 8) -> List[Dict[str, Any]]:
    """Forecasts from a single fine-resolution stream.

    Derives the coarse-resolution stream by aggregating the fine-resolution
    context in blocks of `self.agg_factor` (e.g., 60 minutes -> hourly) and then runs multi-resolution decoding.

    Args:
      inputs: single series or list/tuple of multiple series. Each series can be:
        - single-resolution: list-like or array-like of fine-resolution data.
        - multi-resolution: tuple/list of two elements (coarse_series, fine_series)
          - coarse_series: list-like or array-like of coarse-resolution data.
          - fine_series: list-like or array-like of fine-resolution data.
          - If single-resolution, the coarse series will be derived by aggregating
            the fine series in blocks of `self.agg_factor`.
          - If multi-resolution, the provided coarse series will be used directly.
      horizon_len: forecast horizon length. 
        - if None, uses the model's configured horizon length (output_patch_len).
        - if > output_patch_len, the model will predict autoregressively until horizon_len is reached.
      batch_size: batch size for forecasting.

    Returns:
      List of dicts, each with:
        - "mean": np.ndarray of shape (horizon_len,) — point forecast.
        - "quantiles": Dict[str, np.ndarray] — keyed by quantile level string.
    """
    if self._model is None:
      raise ValueError("Checkpoint is not properly loaded.")
    
    if horizon_len is None:
      horizon_len = self.output_patch_len

    if horizon_len <= 0:
      raise ValueError("horizon_len must be positive")

    if batch_size <= 0:
      raise ValueError("batch_size must be positive")

    inputs = self._normalize_inputs(inputs)

    N = len(inputs)
    if N == 0:
      return []

    CONTEXT_LEN_COARSE = Constants.CONTEXT_LEN_COARSE.value
    CONTEXT_LEN_FINE = Constants.CONTEXT_LEN_FINE.value

    coarse_ctx_padded = np.zeros((N, CONTEXT_LEN_COARSE), dtype=np.float32)
    coarse_pad_mask = np.zeros((N, CONTEXT_LEN_COARSE), dtype=np.float32)
    fine_ctx_padded = np.zeros((N, CONTEXT_LEN_FINE), dtype=np.float32)
    fine_pad_mask = np.zeros((N, CONTEXT_LEN_FINE), dtype=np.float32)

    for i, item in enumerate(inputs):
      if item["multi_res"]:
        coarse_series = np.array(item["coarse"], dtype=np.float32).reshape(-1)
        fine_series = np.array(item["fine"], dtype=np.float32).reshape(-1)

        coarse_series = self._replace_non_finite(coarse_series)
        fine_series = self._replace_non_finite(fine_series)

        coarse_series = strip_leading_nans(coarse_series)
        fine_series = strip_leading_nans(fine_series)
        coarse_series = linear_interpolation(coarse_series)
        fine_series = linear_interpolation(fine_series)

        coarse_ctx_1d = coarse_series
        fine_ctx_1d = fine_series
      else:
        series = np.array(item["fine"], dtype=np.float32).reshape(-1)
        series = self._replace_non_finite(series)
        series = strip_leading_nans(series)
        series = linear_interpolation(series)

        coarse_ctx_1d, fine_ctx_1d = self.build_multi_resolution(series, self.agg_factor)

      coarse_len_raw = coarse_ctx_1d.shape[0]
      if coarse_len_raw >= CONTEXT_LEN_COARSE:
        coarse_ctx_padded[i, :] = coarse_ctx_1d[-CONTEXT_LEN_COARSE:]
      else:
        pad_len = CONTEXT_LEN_COARSE - coarse_len_raw
        if coarse_len_raw:
          coarse_ctx_padded[i, pad_len:] = coarse_ctx_1d
        coarse_pad_mask[i, :pad_len] = 1.0

      fine_len_raw = fine_ctx_1d.shape[0]
      if fine_len_raw >= CONTEXT_LEN_FINE:
        fine_ctx_padded[i, :] = fine_ctx_1d[-CONTEXT_LEN_FINE:]
      else:
        pad_len = CONTEXT_LEN_FINE - fine_len_raw
        if fine_len_raw:
          fine_ctx_padded[i, pad_len:] = fine_ctx_1d
        fine_pad_mask[i, :pad_len] = 1.0

    # Batch normalize on CPU (vectorized torch ops) to avoid per-series overhead.
    coarse_ctx_t = torch.from_numpy(coarse_ctx_padded)
    coarse_pad_t = torch.from_numpy(coarse_pad_mask)
    fine_ctx_t = torch.from_numpy(fine_ctx_padded)
    fine_pad_t = torch.from_numpy(fine_pad_mask)

    norm_coarse, offset_coarse, scale_coarse = PatchedTSMultiResolutionDecoder._normalize_with_pad(coarse_ctx_t, pad_mask=coarse_pad_t)
    norm_fine, offset_fine, scale_fine = PatchedTSMultiResolutionDecoder._normalize_with_pad(fine_ctx_t, pad_mask=fine_pad_t)

    offsets_fine = offset_fine.squeeze(1)
    scales_fine = scale_fine.squeeze(1)
    offsets_coarse = offset_coarse.squeeze(1)
    scales_coarse = scale_coarse.squeeze(1)

    final_predictions = []

    for start in range(0, N, batch_size):
      end = min(start + batch_size, N)

      batch_coarse = norm_coarse[start:end].unsqueeze(-1).to(self._device)
      batch_coarse_pad = coarse_pad_t[start:end].unsqueeze(-1).to(self._device)
      batch_fine = norm_fine[start:end].unsqueeze(-1).to(self._device)
      batch_fine_pad = fine_pad_t[start:end].unsqueeze(-1).to(self._device)

      freq_tensor = torch.zeros((end - start, 1), dtype=torch.long, device=self._device)

      with torch.no_grad():
        preds = self._model.decode([batch_coarse, batch_fine],
                                   [batch_coarse_pad.float(), batch_fine_pad.float()],
                                   freq_tensor, 
                                   horizon_len=horizon_len, 
                                   offsets_fine=offsets_fine[start:end],
                                   scales_fine=scales_fine[start:end],
                                   offsets_coarse=offsets_coarse[start:end],
                                   scales_coarse=scales_coarse[start:end],
                                   output_patch_len=self.output_patch_len)

      final_predictions.extend(preds)

    return final_predictions
