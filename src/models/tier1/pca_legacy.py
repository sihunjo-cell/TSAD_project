"""Fit-only TSB-AD PCA component-distance score."""

import math
import warnings
import numpy
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial.distance import cdist
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PCA_COMPONENTS = (0.25, 0.5, 0.75, None)
PCA_WINDOW = 100


def _as_matrix(values, *, minimum_rows=1):
    values = numpy.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("values must be a nonempty 2D time-by-channel array")
    if len(values) < minimum_rows:
        raise ValueError(f"values need at least {minimum_rows} timesteps")
    if not numpy.isfinite(values).all():
        raise ValueError("values must be finite")
    return values


def _make_windows(values):
    return sliding_window_view(
        values, window_shape=PCA_WINDOW, axis=0,
    ).reshape(len(values) - PCA_WINDOW + 1, -1)


def _normalize_windows(windows):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        normalized = zscore(windows, axis=1, ddof=1)
    return numpy.nan_to_num(normalized)


class PcaLegacy:
    """Fit the official PCA core without target-wide test statistics."""

    def __init__(self, *, n_components):
        if n_components not in PCA_COMPONENTS:
            raise ValueError(f"n_components must be one of {PCA_COMPONENTS}")
        self.n_components = n_components
        self.window = PCA_WINDOW
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=0)
        self._is_fitted = False

    def fit(self, fit_values):
        fit_values = _as_matrix(fit_values, minimum_rows=PCA_WINDOW + 1)
        windows = _normalize_windows(_make_windows(fit_values))
        scaled = self.scaler.fit_transform(windows)
        self.pca.fit(scaled)
        self.selected_components_ = self.pca.components_
        self.selected_w_components_ = self.pca.explained_variance_ratio_
        self._is_fitted = True
        return self

    def score(self, values):
        if not self._is_fitted:
            raise RuntimeError("fit must be called before score")
        values = _as_matrix(values, minimum_rows=PCA_WINDOW)
        windows = _normalize_windows(_make_windows(values))
        scaled = self.scaler.transform(windows)
        window_scores = numpy.sum(
            cdist(scaled, self.selected_components_)
            / self.selected_w_components_,
            axis=1,
        ).ravel()
        left = math.ceil((PCA_WINDOW - 1) / 2)
        right = (PCA_WINDOW - 1) // 2
        scores = numpy.concatenate((
            numpy.repeat(window_scores[0], left),
            window_scores,
            numpy.repeat(window_scores[-1], right),
        ))
        return {
            "scores": scores,
            "source_start": 0,
            "source_end_exclusive": len(values),
            "alignment": "centered_window_edge_repeat",
            "primitive": "weighted_component_distance",
            "calibration_mode": "validation_median_iqr",
            "native_source_start": left,
            "native_source_end_exclusive": len(values) - right,
            "boundary_repeat": {"left": left, "right": right},
            "evaluation_mode": "offline_noncausal",
            "lookahead": right,
            "maximum_effective_lookahead": PCA_WINDOW - 1,
            "normalization_scope": "current_prefix_validation",
        }
