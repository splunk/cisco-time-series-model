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

"""Checkpoint loading and native multi-resolution forecasting for CTSM 1.5."""

from pathlib import Path
from typing import Any
from collections.abc import Mapping, Sequence

import torch
import numpy as np
from safetensors.torch import load_file

from .preprocessing import interpolate_missing, trim_leading_missing
from .multiresolution_decoder import MultiResolutionDecoder
from .constants import AGG_FACTOR, COARSE_CONTEXT_LENGTH, EPS, FINE_CONTEXT_LENGTH, OUTPUT_PATCH_LENGTH


class CiscoTsmMR:
    """Load the fixed CTSM 1.5 decoder and run autoregressive forecast paths."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        self._device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self._ctx_len_fine = FINE_CONTEXT_LENGTH
        self._ctx_len_coarse = COARSE_CONTEXT_LENGTH
        self.agg_factor = AGG_FACTOR

        path = Path(checkpoint_path)
        if path.suffix.lower() == ".safetensors":
            state = load_file(path, device="cpu")
        else:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(checkpoint, Mapping):
                raise ValueError("Expected a CTSM 1.5 weight mapping or training checkpoint")
            state = checkpoint["model"] if "model" in checkpoint else checkpoint

        self._model = MultiResolutionDecoder()
        self._model.load_state_dict(state, strict=True)
        self._model.to(self._device).eval()


    @staticmethod
    def normalize_with_pad(
        context: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        clamp_range: tuple[float, float] = (-1000, 1000),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize context with padding mask.
        Args:
          context: tensor of shape [B, T] or [B, T, 1].
          pad_mask: tensor of shape [B, T], with 1.0 for padded positions and 0.0 for actual data.
          clamp_range: tuple of (min, max) to clamp normalized values. Default (-1000, 1000).
        Returns:
          ctx_normalized: normalized context tensor with same shape as input.
          offset: mean used for normalization, shape [B, 1].
          scale: stddev used for normalization, shape [B, 1].
          eps: small epsilon value used for numerical stability.
        """

        if context.ndim == 3:
            context = context.squeeze(-1)

        if pad_mask is None:
            pad_mask = torch.zeros_like(context)

        valid = (1.0 - pad_mask)
        count = valid.sum(dim=1, keepdim=True).clamp_min(1.0)

        context_mean = (context * valid).sum(dim=1, keepdim=True) / count
        context_var = (((context - context_mean) * valid) ** 2).sum(dim=1, keepdim=True) / count
        context_std = context_var.sqrt()

        min_scale = 1e-2
        context_std = context_std.clamp_min(min_scale)

        ctx_normalized = (context - context_mean) / (context_std + EPS)
        stats = (context_mean, context_std)
        ctx_normalized = ctx_normalized * valid
        normalized_unclamped = ctx_normalized
        ctx_normalized = torch.clamp(ctx_normalized, *clamp_range)

        offset, scale = stats

        retry_rows = (
            ~torch.isfinite(offset).all(dim=1, keepdim=True)
            | ~torch.isfinite(scale).all(dim=1, keepdim=True)
            | ~torch.isfinite(normalized_unclamped).all(dim=1, keepdim=True)
        )

        if retry_rows.any():
            observed = valid != 0

            if not torch.isfinite(context.masked_select(observed)).all():
                raise ValueError("normalization requires finite valid observations")

            context64 = context.to(torch.float64)
            valid64 = valid.to(torch.float64)

            count64 = valid64.sum(dim=1, keepdim=True).clamp_min(1.0)
            clean64 = torch.where(observed, context64, torch.zeros_like(context64))

            mean64 = (clean64 * valid64).sum(dim=1, keepdim=True) / count64
            centered64 = torch.where(observed, context64 - mean64, torch.zeros_like(context64))
            std64 = (
                ((centered64 * valid64).square().sum(dim=1, keepdim=True) / count64)
                .sqrt()
                .clamp_min(min_scale)
            )

            normalized64 = torch.clamp(centered64 / (std64 + EPS) * valid64, *clamp_range)

            offset = torch.where(retry_rows, mean64.to(context.dtype), offset)
            scale = torch.where(retry_rows, std64.to(context.dtype), scale)
            ctx_normalized = torch.where(retry_rows, normalized64.to(context.dtype), ctx_normalized)

            if (
                not torch.isfinite(offset).all()
                or not torch.isfinite(scale).all()
                or not torch.isfinite(ctx_normalized).all()
            ):
                raise FloatingPointError("normalization retry produced nonfinite or unrepresentable statistics")

        return ctx_normalized, offset, scale


    @staticmethod
    def build_coarse_context(series: np.ndarray) -> np.ndarray:
        """Average the complete, right-aligned groups of 60 observations."""
        series = np.asarray(series, dtype=np.float32).reshape(-1)

        needed_raw = COARSE_CONTEXT_LENGTH * AGG_FACTOR
        raw_slice = series[-needed_raw:]

        remainder = raw_slice.shape[0] % AGG_FACTOR

        if remainder:
            raw_slice = raw_slice[remainder:]

        if raw_slice.shape[0] == 0:
            return np.empty((0,), dtype=np.float32)

        blocks = raw_slice.reshape(-1, AGG_FACTOR)

        with np.errstate(over="ignore", invalid="ignore"):
            coarse = blocks.mean(axis=1, dtype=np.float32)

        retry_bins = ~np.isfinite(coarse)

        if retry_bins.any():
            if not np.isfinite(blocks).all():
                raise ValueError("coarse aggregation requires finite valid observations")

            with np.errstate(over="ignore", invalid="ignore"):
                coarse[retry_bins] = blocks[retry_bins].mean(axis=1, dtype=np.float64).astype(np.float32)

            if not np.isfinite(coarse).all():
                raise FloatingPointError("coarse aggregation retry produced nonfinite means")

        return coarse.astype(np.float32, copy=False)


    @staticmethod
    def _clock_from_timestamps(ts: np.ndarray | Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
        """Raw (seconds-since-midnight, weekday) from a series' timestamps.

        Weekday is Monday=0 -- the convention of the context_1min_dow column the
        model trained on, confirmed against the weekday/weekend split in the training
        sample dump. 1970-01-01 was a Thursday, hence the +3. Timestamps must be
        tz-naive: the clock feature carries local-time semantics.
        """

        t = np.asarray(ts, dtype="datetime64[s]")
        days = t.astype("datetime64[D]")

        tod = (t - days).astype("timedelta64[s]").astype(np.float32)
        dow = ((days.astype("int64") + 3) % 7).astype(np.float32)

        return tod, dow


    def forecast(
        self,
        inputs: np.ndarray | Sequence[np.ndarray | Sequence[float]],
        horizon_len: int | None = OUTPUT_PATCH_LENGTH,
        batch_size: int = 8,
        timestamps: np.ndarray | Sequence[Any] | None = None,
        allow_missing_calendar: bool = False,
        *,
        feedback_quantile_index: int | None = None,
    ) -> list[dict[str, Any]]:
        """Forecast raw histories, optionally overriding the median used for feedback."""

        (predictions,) = self.forecast_paths(
            inputs,
            horizon_len,
            batch_size,
            timestamps,
            allow_missing_calendar,
            feedback_quantile_indices=(feedback_quantile_index,),
        )

        return predictions


    def forecast_paths(
        self,
        inputs: np.ndarray | Sequence[np.ndarray | Sequence[float]],
        horizon_len: int | None = OUTPUT_PATCH_LENGTH,
        batch_size: int = 8,
        timestamps: np.ndarray | Sequence[Any] | None = None,
        allow_missing_calendar: bool = False,
        *,
        feedback_quantile_indices: Sequence[int | None],
    ) -> tuple[list[dict[str, Any]], ...]:
        """Forecast independent feedback paths from a shared initial model forward.

        Paths are returned in the supplied index order. A path configured with
        ``None`` uses native P50 feedback. Later model calls retain the original
        batch geometry for every path.
        """

        if horizon_len is None:
            horizon_len = OUTPUT_PATCH_LENGTH

        if horizon_len <= 0 or batch_size <= 0:
            raise ValueError("horizon_len and batch_size must be positive")

        indices = tuple(feedback_quantile_indices)

        if not indices:
            raise ValueError("At least one feedback path is required")

        predictions: tuple[list[dict[str, Any]], ...] = tuple([] for _ in indices)
        count = len(inputs)

        if count == 0:
            return predictions

        if timestamps is None and not allow_missing_calendar:
            raise ValueError("Supply past timestamps or enable value-only forecasting")

        if (
            timestamps is not None
            and count == 1
            and len(timestamps)
            and not isinstance(timestamps[0], (list, tuple, np.ndarray))
        ):
            timestamps = [timestamps]

        if timestamps is not None and len(timestamps) != count:
            raise ValueError(f"Expected {count} timestamp arrays, got {len(timestamps)}")

        retain_coarse_source = horizon_len > OUTPUT_PATCH_LENGTH

        fine_values = np.zeros((count, FINE_CONTEXT_LENGTH), dtype=np.float32)
        coarse_values = np.zeros((count, COARSE_CONTEXT_LENGTH), dtype=np.float32)

        fine_padding = np.ones_like(fine_values)
        coarse_padding = np.ones_like(coarse_values)

        time_of_day = np.zeros_like(fine_values)
        day_of_week = np.zeros_like(fine_values)

        calendar_padding = None if timestamps is not None else torch.ones((count, 2), dtype=torch.float32)

        if retain_coarse_source:
            raw_length = COARSE_CONTEXT_LENGTH * AGG_FACTOR
            source_values = np.zeros((count, raw_length), dtype=np.float32)
            source_padding = np.ones_like(source_values)

        for row, history in enumerate(inputs):
            values = np.array(history, dtype=np.float32, copy=True).reshape(-1)
            values[~np.isfinite(values)] = np.nan

            values = interpolate_missing(trim_leading_missing(values))

            fine = values[-FINE_CONTEXT_LENGTH:]
            coarse = self.build_coarse_context(values)

            if len(fine):
                fine_values[row, -len(fine):] = fine
                fine_padding[row, -len(fine):] = 0.0

            if len(coarse):
                coarse_values[row, -len(coarse):] = coarse
                coarse_padding[row, -len(coarse):] = 0.0

            if retain_coarse_source:
                raw_tail = values[-raw_length:]

                if len(raw_tail):
                    source_values[row, -len(raw_tail):] = raw_tail
                    source_padding[row, -len(raw_tail):] = 0.0

            if timestamps is not None:
                tod, dow = self._clock_from_timestamps(timestamps[row])

                if len(tod) < len(fine):
                    raise ValueError(f"Series {row} has {len(tod)} timestamps for {len(fine)} fine steps")

                if len(fine) < 2:
                    raise ValueError(f"Series {row} needs at least two fine steps to infer the cadence")

                time_of_day[row, -len(fine):] = tod[-len(fine):]
                day_of_week[row, -len(fine):] = dow[-len(fine):]

        coarse_pad = torch.from_numpy(coarse_padding)
        fine_pad = torch.from_numpy(fine_padding)

        coarse, coarse_offset, coarse_scale = self.normalize_with_pad(
            torch.from_numpy(coarse_values), pad_mask=coarse_pad,
        )
        fine, fine_offset, fine_scale = self.normalize_with_pad(
            torch.from_numpy(fine_values), pad_mask=fine_pad,
        )

        tod = torch.from_numpy(time_of_day)
        dow = torch.from_numpy(day_of_week)

        with torch.no_grad():
            for start in range(0, count, batch_size):
                rows = slice(start, start + batch_size)
                coarse_kwargs = {}

                if retain_coarse_source:
                    coarse_kwargs = {
                        "coarse_source_history": torch.from_numpy(source_values[rows]).to(self._device),
                        "coarse_source_padding": torch.from_numpy(source_padding[rows]).to(self._device),
                    }

                input_ts = [
                    coarse[rows].unsqueeze(-1).to(self._device),
                    fine[rows].unsqueeze(-1).to(self._device),
                ]
                paddings = [
                    coarse_pad[rows].unsqueeze(-1).to(self._device).float(),
                    fine_pad[rows].unsqueeze(-1).to(self._device).float(),
                ]

                clock_tod = tod[rows].unsqueeze(-1).to(self._device)
                clock_dow = dow[rows].unsqueeze(-1).to(self._device)
                clock_padding = None if calendar_padding is None else calendar_padding[rows].to(self._device)

                first_block: list[torch.Tensor] | None = [] if len(indices) > 1 else None

                for index, path in zip(indices, predictions, strict=True):
                    path.extend(
                        self._model.decode(
                            input_ts,
                            paddings,
                            horizon_len=horizon_len,
                            offsets=fine_offset[rows, 0],
                            scales=fine_scale[rows, 0],
                            coarse_offsets=coarse_offset[rows, 0],
                            coarse_scales=coarse_scale[rows, 0],
                            fine_tod=clock_tod,
                            fine_dow=clock_dow,
                            calendar_row_padding=clock_padding,
                            feedback_quantile_index=index,
                            _first_block_cache=first_block,
                            **coarse_kwargs,
                        )
                    )

        return predictions
