"""Official Last-1, Last-3, and Centered-5 squared-difference scores."""

import numpy
import pandas


def _score_squared_difference(values, *, window, centered):
    values = numpy.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("values must be a nonempty 2D time-by-channel array")
    if len(values) < window:
        raise ValueError(f"squared difference needs at least {window} timesteps")
    if not numpy.isfinite(values).all():
        raise ValueError("values must be finite")

    frame = pandas.DataFrame(values)
    mean = frame.rolling(window, center=centered).mean().bfill().ffill()
    scores = ((frame - mean) * window / (window - 1.0)) ** 2
    return {
        "scores": scores.to_numpy(),
        "source_start": 0,
        "source_end_exclusive": len(values),
        "alignment": "same_timestep",
        "primitive": "squared_difference",
        "calibration_mode": "none",
        "native_source_start": window // 2 if centered else window - 1,
        "native_source_end_exclusive": len(values) - (window - 1) // 2 if centered else len(values),
        "boundary_policy": (
            "backfill_and_forward_fill_complete_rolling_mean" if centered
            else "backfill_first_complete_rolling_mean"
        ),
        "evaluation_mode": "offline_noncausal",
        "lookahead": (window - 1) // 2 if centered else 0,
        "maximum_effective_lookahead": window - 1,
        "normalization_scope": "none",
    }


def score_sqdiff_last1(values):
    """Apply the official last-one score independently to each channel."""
    return _score_squared_difference(values, window=2, centered=False)


def score_sqdiff_last3(values):
    """Apply the official last-three score independently to each channel."""
    return _score_squared_difference(values, window=4, centered=False)


def score_sqdiff_centered5(values):
    """Apply the official centered-five score independently to each channel."""
    return _score_squared_difference(values, window=5, centered=True)
