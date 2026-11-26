"""Raw-history timestamp construction, independent of model execution."""

import re
import warnings
from numbers import Real
from typing import List, Union, Any

import numpy as np
import pandas as pd


def parallel(value: Any, name: str, history_count: int) -> list[Any]:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value[()]

    if pd.api.types.is_scalar(value):
        return [value] * history_count

    try:
        values = list(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a scalar or one value per input history") from error

    if len(values) != history_count:
        raise ValueError(f"{name} must contain one value per input history")

    return values


def _timestamps_from_start(histories: list[np.ndarray],
                           start_time: pd.Timestamp,
                           freq_seconds: int or None = None,
                           *,
                           freq: str or None = None) -> list[np.ndarray]:
    """Build complete raw-history clocks from seconds or GIFT-style cadences."""

    if (freq_seconds is None) == (freq is None):
        raise ValueError("Supply exactly one of freq_seconds or freq")

    name = "freq" if freq is not None else "freq_seconds"
    starts = parallel(start_time, "start_time", len(histories))
    frequencies = parallel(freq if freq is not None else freq_seconds, name, len(histories))

    clocks = []

    for i, (history, start, cadence) in enumerate(zip(histories, starts, frequencies)):
        if start is None or isinstance(start, (Real, np.bool_)):
            raise ValueError("start_time must be an ISO datetime or datetime object, not a numeric epoch")

        try:
            first = pd.Timestamp(start)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"Invalid start_time for history {i}: {start!r}") from error

        if pd.isna(first):
            raise ValueError("start_time cannot be NaT or missing")

        if first.tzinfo is not None:
            raise ValueError("start_time must be timezone-naive in the clock timezone you want the model to see")

        if not len(history):
            raise ValueError("Each input history must contain at least one past value")

        calendar_offset = None

        if name == "freq" and isinstance(cadence, str):
            value = str(cadence).strip()

            # Follows the GIFT-style for cadence.
            legacy = re.fullmatch(r"(\d+)?(W|M|Q|A|Y)(?:-([A-Z]{3}))?", value)

            if legacy:
                count, unit, anchor = legacy.groups()
                count = int(count or 1)
                valid_anchors = (set("MON TUE WED THU FRI SAT SUN".split()) if unit == "W"
                                 else set("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split()))

                if count <= 0 or anchor and (unit == "M" or anchor not in valid_anchors):
                    raise ValueError(f"Invalid GIFT frequency: {cadence!r}")

                if unit == "W":
                    offset = pd.Timedelta(days=7 * count)
                else:
                    months = count * ({"Q": 3, "A": 12, "Y": 12}.get(unit, 1))
                    calendar_offset = (pd.offsets.MonthEnd(months) if first.is_month_end
                                       else pd.DateOffset(months=months))

            else:
                try:
                    fixed_alias = re.fullmatch(r"(\d+)?(T|H|S)", value)

                    if fixed_alias:
                        count, unit = fixed_alias.groups()
                        value = (count or "1") + {"T": "min", "H": "h", "S": "s"}[unit]

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", FutureWarning)
                        offset = pd.tseries.frequencies.to_offset(value)

                    if offset.n <= 0:
                        raise ValueError("Frequency must be positive")

                    try:
                        nanos = offset.nanos
                    except ValueError:
                        nanos = None

                    if nanos is None:
                        if not offset.is_on_offset(first):
                            raise ValueError("start_time does not match the frequency anchor")

                    elif nanos < 1_000_000_000 or nanos % 1_000_000_000:
                        raise ValueError("Frequency must resolve to positive whole seconds")

                except (TypeError, ValueError, OverflowError) as error:
                    raise ValueError(f"Invalid freq {cadence!r}: {error}") from error

        else:
            seconds = cadence

            try:
                valid_seconds = (not isinstance(seconds, (bool, np.bool_)) and isinstance(seconds, Real)
                                 and np.isfinite(float(seconds)) and seconds > 0 and seconds == int(seconds))
            except (TypeError, ValueError, OverflowError):
                valid_seconds = False

            if not valid_seconds:
                raise ValueError("freq_seconds must contain positive whole seconds, e.g. 300 for five minutes")

            try:
                offset = pd.Timedelta(seconds=int(seconds))
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("Frequency exceeds the supported timedelta range") from error

        try:
            if calendar_offset is not None:
                # Reference the original date each time: Jan31 -> Feb28 -> Mar31,
                # rather than allowing February's clipping to drift to Mar28.
                clock = pd.DatetimeIndex([first + j * calendar_offset for j in range(len(history))]).to_numpy(dtype="datetime64[ns]")

            else:
                clock = pd.date_range(start=first, periods=len(history), freq=offset).to_numpy(dtype="datetime64[ns]")
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"Clock for history {i} exceeds the supported datetime range") from error

        clocks.append(clock)

    return clocks
