"""Centered moving-window variance score."""

import numpy
import pandas


MWVAR_WINDOW = 96


def score_mwvar(values):
    """Return the channel-wise project extension of the official univariate score."""
    values = numpy.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("values must be a nonempty 2D time-by-channel array")
    if len(values) < MWVAR_WINDOW:
        raise ValueError(f"MWVAR needs at least {MWVAR_WINDOW} timesteps")
    if not numpy.isfinite(values).all():
        raise ValueError("values must be finite")

    scores = (
        pandas.DataFrame(values)
        .rolling(MWVAR_WINDOW, center=True)
        .var(ddof=1)
        .bfill()
        .ffill()
        .to_numpy()
    )
    return {
        "scores": scores,
        "source_start": 0,
        "source_end_exclusive": len(values),
        "alignment": "same_timestep",
        "primitive": "sample_variance",
        "calibration_mode": "none",
        "native_source_start": 48,
        "native_source_end_exclusive": len(values) - 47,
        "boundary_repeat": {"left": 48, "right": 47},
        "evaluation_mode": "offline_noncausal",
        "lookahead": 47,
        "maximum_effective_lookahead": 95,
        "normalization_scope": "none",
    }
