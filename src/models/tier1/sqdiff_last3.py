"""Squared difference from the trailing four-point mean."""

import numpy
import pandas


SQDIFF_WINDOW = 4


def score_sqdiff_last3(values):
    """Return the channel-wise project extension of the official univariate score."""
    values = numpy.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("values must be a nonempty 2D time-by-channel array")
    if len(values) < SQDIFF_WINDOW:
        raise ValueError(f"SQDIFF_LAST3 needs at least {SQDIFF_WINDOW} timesteps")
    if not numpy.isfinite(values).all():
        raise ValueError("values must be finite")

    frame = pandas.DataFrame(values)
    trailing_mean = frame.rolling(SQDIFF_WINDOW).mean().bfill().ffill()
    scores = ((frame - trailing_mean) * 4 / 3) ** 2
    return {
        "scores": scores.to_numpy(),
        "source_start": 0,
        "source_end_exclusive": len(values),
        "alignment": "same_timestep",
        "primitive": "squared_difference",
        "calibration_mode": "none",
        "native_source_start": 3,
        "native_source_end_exclusive": len(values),
        "boundary_policy": "backfill_first_complete_rolling_mean",
        "evaluation_mode": "offline_noncausal",
        "lookahead": 0,
        "maximum_effective_lookahead": 3,
        "normalization_scope": "none",
    }
