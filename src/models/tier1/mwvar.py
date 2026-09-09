"""Centered moving-window variance score."""

import numpy
import pandas


MWVAR_WINDOW = 96


def score_mwvar(values, *, window: int = MWVAR_WINDOW):
    """Return the channel-wise project extension of the official univariate score."""
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError("MWVAR window must be an integer of at least two")
    values = numpy.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("values must be a nonempty 2D time-by-channel array")
    if len(values) < window:
        raise ValueError(f"MWVAR needs at least {window} timesteps")
    if not numpy.isfinite(values).all():
        raise ValueError("values must be finite")

    scores = (
        pandas.DataFrame(values)
        .rolling(window, center=True)
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
        "native_source_start": window // 2,
        "native_source_end_exclusive": len(values) - (window - 1) // 2,
        "boundary_repeat": {"left": window // 2, "right": (window - 1) // 2},
        "evaluation_mode": "offline_noncausal",
        "lookahead": (window - 1) // 2,
        "maximum_effective_lookahead": window - 1,
        "normalization_scope": "none",
    }
