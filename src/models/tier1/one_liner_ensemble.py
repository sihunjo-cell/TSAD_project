"""Official Var-96 ensembles, applied independently to each channel."""

import numpy

from .mwvar import score_mwvar
from .sqdiff import score_sqdiff_centered5, score_sqdiff_last3


def _score_mwvar96_ensemble(values, *, difference_scorer, difference_name):
    variance = score_mwvar(values, window=96)
    difference = difference_scorer(values)
    normalized = []
    ranges = {}
    for name, output in (("MWVAR96", variance), (difference_name, difference)):
        scores = output["scores"]
        if not numpy.isfinite(scores).all():
            raise ValueError(f"{name} scores must be finite before official minmax")
        minimum = scores.min(axis=0)
        maximum = scores.max(axis=0)
        span = maximum - minimum
        if numpy.any(span == 0):
            channels = numpy.flatnonzero(span == 0).tolist()
            raise ValueError(f"{name} has zero score range in channels {channels}; official minmax is undefined")
        normalized.append((scores - minimum) / span)
        ranges[name] = {"minimum": minimum.tolist(), "maximum": maximum.tolist()}

    length = len(variance["scores"])
    return {
        "scores": numpy.maximum(*normalized),
        "source_start": 0,
        "source_end_exclusive": length,
        "alignment": "same_timestep",
        "primitive": "official_minmax_component_max",
        "calibration_mode": "none",
        "native_source_start": variance["native_source_start"],
        "native_source_end_exclusive": variance["native_source_end_exclusive"],
        "boundary_policy": "component_rolling_fill_before_global_score_minmax",
        "evaluation_mode": "offline_noncausal",
        "lookahead": length - 1,
        "maximum_effective_lookahead": length - 1,
        "normalization_scope": "full_evaluation_per_channel_score_minmax",
        "component_score_ranges": ranges,
        "score_normalization_source_start": 0,
        "score_normalization_source_end_exclusive": length,
    }


def score_mwvar96_sqdiff_last3(values):
    """Normalize Var-96 and Last-3 over the evaluation input, then take their max."""
    return _score_mwvar96_ensemble(
        values, difference_scorer=score_sqdiff_last3, difference_name="SQDIFF_LAST3",
    )


def score_mwvar96_sqdiff_centered5(values):
    """Normalize Var-96 and Centered-5 over the evaluation input, then take their max."""
    return _score_mwvar96_ensemble(
        values, difference_scorer=score_sqdiff_centered5, difference_name="SQDIFF_CENTERED5",
    )
