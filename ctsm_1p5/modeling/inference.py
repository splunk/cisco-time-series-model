"""CTSM 1.5 forecasting policies over the native multi-resolution decoder.

The native model owns history preparation and autoregression. This module only
selects inference policies, constructs optional clocks, and transforms complete
forecasts. It performs no training or benchmark-specific input routing.
"""

from __future__ import annotations

import threading
from pathlib import Path
from datetime import date
from typing import Any, Sequence
from dataclasses import asdict, dataclass, replace

import torch
import numpy as np

from . import postprocessing
from .cisco_tsm_mr import CiscoTsmMR
from .calendar import _timestamps_from_start
from .constants import OUTPUT_PATCH_LENGTH, QUANTILES, TOLERANCE


_HF_WEIGHTS_FILENAME = "model.safetensors"


@dataclass(frozen=True)
class InferenceOptions:
    """Calendar and flip equivariance settings.

    Set ``use_timestamps=None`` to infer calendar mode from the input.
    """

    use_timestamps: bool | None = None
    flip_equivariance: bool = True

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name == "use_timestamps" and value is None:
                continue

            if type(value) is not bool:
                raise TypeError(f"{name} must be a bool")


class CTSMForecaster:
    """One loaded model with reusable defaults and isolated per-call overrides.

    Pass raw univariate histories. Each result contains all 15 ``quantiles``
    as arrays over the forecast horizon. The ``mean`` field holds the
    postprocessed median (0.5 quantile).
    """

    def __init__(
        self,
        native_model: CiscoTsmMR,
        options: InferenceOptions | None = None,
    ) -> None:
        self._native = native_model
        self._lock = threading.RLock()
        self._options = options if options is not None else InferenceOptions()

        if not isinstance(self._options, InferenceOptions):
            raise TypeError("options must be InferenceOptions")

        self._flip_grid = None


    @property
    def options(self) -> InferenceOptions:
        return self._options


    def configure(
        self,
        *,
        use_timestamps: bool | None = None,
        flip_equivariance: bool | None = None,
    ) -> CTSMForecaster:
        """Update defaults without reloading weights; retain omitted options."""

        with self._lock:
            changes = {
                k: v
                for k, v in (("use_timestamps", use_timestamps), ("flip_equivariance", flip_equivariance))
                if v is not None
            }
            self._options = replace(self._options, **changes)

        return self


    def _forecast_blocks(
        self,
        histories: Sequence[np.ndarray | Sequence[float]],
        *,
        horizon_len: int,
        batch_size: int,
        timestamps: np.ndarray | Sequence[Any] | None,
    ) -> tuple[list[dict[str, Any]], ...]:

        kwargs = dict(
            inputs=histories,
            horizon_len=horizon_len,
            batch_size=batch_size,
            timestamps=timestamps,
            allow_missing_calendar=timestamps is None,
        )

        if horizon_len <= OUTPUT_PATCH_LENGTH:
            median = self._native.forecast(**kwargs)

            if len(median) != len(histories):
                raise ValueError("Native forecast batch size does not match the input histories")

            return (median,)

        median, lower, upper = self._native.forecast_paths(
            **kwargs,
            feedback_quantile_indices=(None, QUANTILES.index(0.25), QUANTILES.index(0.75)),
        )

        if len(median) != len(histories):
            raise ValueError("Native forecast batch size does not match the input histories")

        return postprocessing.pool_three_paths(lower, median, upper, OUTPUT_PATCH_LENGTH)


    def _predict(
        self,
        histories: Sequence[np.ndarray | Sequence[float]],
        *,
        horizon_len: int,
        batch_size: int,
        timestamps: np.ndarray | Sequence[Any] | None,
        options: InferenceOptions,
    ) -> list[dict[str, Any]]:

        if options.flip_equivariance and self._flip_grid is None:
            self._flip_grid = postprocessing._complement_grid(QUANTILES)

        decisions = [postprocessing.gates(x) for x in histories]
        kwargs = dict(horizon_len=horizon_len, batch_size=batch_size, timestamps=timestamps)

        blocks = self._forecast_blocks(histories, **kwargs)

        if options.flip_equivariance:
            reflected = self._forecast_blocks(postprocessing._reflected_inputs(histories), **kwargs)
            blocks = tuple(
                postprocessing._merge_symmetric(block, negative, *self._flip_grid)
                for block, negative in zip(blocks, reflected, strict=True)
            )

        predictions = blocks[0] if len(blocks) == 1 else postprocessing.join_forecast_blocks(*blocks)
        predictions = postprocessing.sort_quantiles(predictions)

        return [
            postprocessing.transform_forecast(p, *decision)
            for p, decision in zip(predictions, decisions, strict=True)
        ]


    def forecast(
        self,
        inputs: np.ndarray | Sequence[float] | Sequence[np.ndarray | Sequence[float]],
        horizon_len: int = OUTPUT_PATCH_LENGTH,
        batch_size: int = 8,
        timestamps: np.ndarray | Sequence[Any] | None = None,
        *,
        start_time: str | date | np.datetime64 | np.ndarray | Sequence[str | date | np.datetime64] | None = None,
        freq_seconds: float | Sequence[float] | None = None,
        freq: str | float | Sequence[str | float] | None = None,
        use_timestamps: bool | None = None,
        flip_equivariance: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Forecast raw histories in their original units.

        By default, calendar features are enabled when context ``timestamps``
        or ``start_time`` with a cadence is supplied. Supply cadence as numeric
        ``freq_seconds`` or ``freq`` (frequency labels such as ``'5T'``, ``'W'``,
        and ``'M'``, or a number of seconds).
        Scalar starts and cadences broadcast over the batch. ``start_time`` labels
        the first supplied value, before any native history trimming.

        ``use_timestamps=True`` requires a clock. With ``use_timestamps=False``,
        calendar features are masked even if a clock is supplied. Option overrides
        apply only to this call. Beyond 128 steps, each sign orientation pools
        independent P25, P50, and P75 feedback paths. When enabled, flip
        equivariance is applied before quantile sorting. Clip/snap is applied
        afterward where the observed values qualify.

        The ``mean`` field holds the postprocessed median (0.5 quantile)
        and equals ``quantiles["0.5"]``.
        """

        if isinstance(inputs, np.ndarray) and inputs.ndim == 1:
            histories = [inputs]
        else:
            histories = list(inputs)

            if histories and np.isscalar(histories[0]):
                histories = [np.asarray(histories)]

        if not histories or any(np.asarray(x).ndim != 1 for x in histories):
            raise ValueError("inputs must contain one or more univariate raw histories")

        if freq is not None and freq_seconds is not None:
            raise ValueError("Supply either numeric freq_seconds or freq, not both")

        cadence_supplied = freq is not None or freq_seconds is not None
        metadata_supplied = start_time is not None or cadence_supplied

        if timestamps is not None and metadata_supplied:
            raise ValueError("Supply either timestamps or start_time with a frequency, not both")

        if metadata_supplied and (start_time is None or not cadence_supplied):
            raise ValueError("Supply start_time together with freq_seconds or freq")

        with self._lock:
            changes = {
                k: v
                for k, v in (("use_timestamps", use_timestamps), ("flip_equivariance", flip_equivariance))
                if v is not None
            }
            effective = replace(self._options, **changes)

            if effective.use_timestamps is False:
                timestamps = None
            elif metadata_supplied:
                timestamps = _timestamps_from_start(histories, start_time, freq_seconds, freq=freq)
            elif effective.use_timestamps is True and timestamps is None:
                raise ValueError("TS mode requires context timestamps or start_time with freq_seconds or freq")

            return self._predict(
                histories,
                horizon_len=horizon_len,
                batch_size=batch_size,
                timestamps=timestamps,
                options=effective,
            )

    def describe(self) -> dict[str, Any]:
        """Return model geometry, active defaults, and inference policies."""

        options = self._options

        return dict(
            parameters=sum(p.numel() for p in self._native._model.parameters()),
            device=str(self._native._device),
            fine_context=self._native._ctx_len_fine,
            coarse_context=self._native._ctx_len_coarse,
            aggregation=self._native.agg_factor,
            native_horizon=OUTPUT_PATCH_LENGTH,
            quantiles=list(QUANTILES),
            defaults=asdict(options),
            tolerance=TOLERANCE,
            postprocessing_order=(
                "pool long-horizon paths -> optional flip equivariance -> quantile sort -> clip/snap"
            ),
            missing_value_policy="legacy",
            coarse_update_policy="origin_aligned",
            normfix=True,
            clip_snap=True,
        )


def load_model(
    checkpoint_path: str | Path | None = None,
    *,
    huggingface_repo_id: str | None = None,
    device: str | torch.device | None = None,
    use_timestamps: bool | None = None,
    flip_equivariance: bool | None = None,
) -> CTSMForecaster:
    """Load CTSM 1.5 weights from a local path or a Hugging Face repository.

    Supply exactly one of ``checkpoint_path`` and ``huggingface_repo_id``.
    Hugging Face weights are downloaded into the Hub cache.
    Omitted options use :class:`InferenceOptions`.
    """
    if checkpoint_path is None:
        if huggingface_repo_id is None:
            raise ValueError("Supply either checkpoint_path or huggingface_repo_id")

        from huggingface_hub import hf_hub_download

        checkpoint_path = hf_hub_download(
            repo_id=huggingface_repo_id,
            filename=_HF_WEIGHTS_FILENAME,
        )
    elif huggingface_repo_id is not None:
        raise ValueError("Supply only one of checkpoint_path and huggingface_repo_id")

    checkpoint = Path(checkpoint_path).expanduser()

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    changes = {
        k: v
        for k, v in (("use_timestamps", use_timestamps), ("flip_equivariance", flip_equivariance))
        if v is not None
    }
    options = InferenceOptions(**changes)
    native = CiscoTsmMR(str(checkpoint), device=device)
    return CTSMForecaster(native, options)
