"""TSB-AD PCA official evaluation fit and the historical prefix-fit adapter."""

import math
import time
import warnings
from copy import deepcopy

import numpy
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial.distance import cdist
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_info, threadpool_limits


PCA_COMPONENTS = (0.25, 0.5, 0.75, None)
PCA_WINDOW = 100
PCA_DISTANCE_CHUNK_ROWS = 1024
PCA_FIT_BLAS_THREADS = 8


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

    def checkpoint(self):
        if not self._is_fitted:
            raise RuntimeError("fit must be called before checkpoint")
        return {
            "model_config": {"n_components": self.n_components, "window": self.window},
            "scaler": deepcopy(self.scaler),
            "pca": deepcopy(self.pca),
        }

    @classmethod
    def from_checkpoint(cls, checkpoint):
        """Restore a trusted local checkpoint without fitting either estimator."""
        config = checkpoint["model_config"]
        if config.get("official_procedure"):
            raise ValueError("official evaluation-fit checkpoints cannot restore the prefix-fit adapter")
        if config["window"] != PCA_WINDOW:
            raise ValueError("checkpoint window does not match the PCA recipe")
        adapter = cls(n_components=config["n_components"])
        adapter.scaler = deepcopy(checkpoint["scaler"])
        adapter.pca = deepcopy(checkpoint["pca"])
        check_is_fitted(adapter.scaler)
        check_is_fitted(adapter.pca)
        if (adapter.pca.n_components != adapter.n_components
                or adapter.scaler.n_features_in_ != adapter.pca.n_features_in_):
            raise ValueError("checkpoint fitted state does not match the PCA recipe")
        adapter.selected_components_ = adapter.pca.components_
        adapter.selected_w_components_ = adapter.pca.explained_variance_ratio_
        adapter._is_fitted = True
        return adapter

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


def score_pca_official(values, *, n_components=None, zero_pruning=True):
    """Fit and score the same complete evaluation input as the official run_PCA."""
    preprocessing_started = time.perf_counter()
    if n_components not in PCA_COMPONENTS:
        raise ValueError(f"n_components must be one of {PCA_COMPONENTS}")
    if zero_pruning is not True:
        raise ValueError("official PCA requires zero_pruning=True")
    values = _as_matrix(values, minimum_rows=PCA_WINDOW)
    windows = _make_windows(values)
    axis, degrees = (0, 0) if values.shape[1] == 1 else (1, 1)
    with numpy.errstate(divide="ignore", invalid="ignore"):
        normalized = numpy.nan_to_num(
            (windows - windows.mean(axis=axis, keepdims=True))
            / windows.std(axis=axis, ddof=degrees, keepdims=True),
        )
    del windows
    scaler = StandardScaler().fit(normalized)
    standardized = scaler.transform(normalized)
    del normalized
    nonzero = numpy.any(standardized != 0, axis=0)
    if not numpy.any(nonzero):
        raise ValueError("official PCA has no features after zero-pruned window columns")
    fitted = standardized[:, nonzero]
    del standardized
    fit_started = time.perf_counter()
    with threadpool_limits(limits=PCA_FIT_BLAS_THREADS, user_api="blas"):
        parallelism = {
            "scope": "pca_fit_only",
            "fit_blas_threads_requested": PCA_FIT_BLAS_THREADS,
            "blas_libraries": [
                {key: library.get(key) for key in ("internal_api", "version", "num_threads")}
                for library in threadpool_info() if library["user_api"] == "blas"
            ],
        }
        pca = PCA(n_components=n_components, random_state=0).fit(fitted)
        weights = pca.explained_variance_ratio_
        initial_solver = pca._fit_svd_solver
        initial_zero_weight_count = int(numpy.count_nonzero(weights == 0))
        if (initial_solver == "covariance_eigh" and initial_zero_weight_count
                and numpy.isfinite(weights).all()):
            # Covariance eigendecomposition can round a small variance to zero.
            pca.set_params(svd_solver="full").fit(fitted)
            weights = pca.explained_variance_ratio_
    fit_finished = time.perf_counter()
    if not numpy.isfinite(weights).all() or numpy.any(weights == 0):
        raise ValueError(
            "official PCA component weights must be finite and nonzero "
            f"(solver={pca._fit_svd_solver}, shape={fitted.shape})"
        )
    solver_record = {
        "requested": "auto", "initial": initial_solver, "actual": pca._fit_svd_solver,
        "initial_zero_weight_count": initial_zero_weight_count,
    }
    distance_started = time.perf_counter()
    window_scores = numpy.empty(len(fitted))
    for start in range(0, len(fitted), PCA_DISTANCE_CHUNK_ROWS):
        stop = start + PCA_DISTANCE_CHUNK_ROWS
        window_scores[start:stop] = numpy.sum(
            cdist(fitted[start:stop], pca.components_) / weights, axis=1,
        )
    parallelism["phase_wall_seconds"] = {
        "preprocessing": fit_started - preprocessing_started,
        "fit": fit_finished - fit_started,
        "component_distances": time.perf_counter() - distance_started,
    }
    if not numpy.isfinite(window_scores).all():
        raise ValueError("official PCA weighted component distances are nonfinite")
    left, right = math.ceil((PCA_WINDOW - 1) / 2), (PCA_WINDOW - 1) // 2
    length = len(values)
    return {
        "scores": numpy.pad(window_scores, (left, right), mode="edge"),
        "source_start": 0,
        "source_end_exclusive": length,
        "alignment": "centered_window_edge_repeat",
        "primitive": "weighted_component_distance",
        "calibration_mode": "none",
        "native_source_start": left,
        "native_source_end_exclusive": length - right,
        "boundary_repeat": {"left": left, "right": right},
        "evaluation_mode": "offline_noncausal",
        "lookahead": length - 1,
        "maximum_effective_lookahead": length - 1,
        "normalization_scope": "full_evaluation_window_standardscaler",
        "official_procedure": True,
        "fit_source": "full_evaluation",
        "actual_fit_row_count": length,
        "fit_source_range": [0, length],
        "window_normalization": "column_zscore_ddof0" if axis == 0 else "row_zscore_ddof1",
        "zero_pruning": True,
        "zero_pruned_window_feature_count": int((~nonzero).sum()),
        "retained_window_feature_count": int(nonzero.sum()),
        "pca_solver": solver_record,
        "pca_parallelism": parallelism,
        "checkpoint": {
            "model_config": {
                "n_components": n_components, "window": PCA_WINDOW,
                "official_procedure": True, "zero_pruning": True,
            },
            "scaler": scaler,
            "pca": pca,
            "pca_solver": solver_record.copy(),
            "pca_parallelism": deepcopy(parallelism),
            "nonzero_window_features": nonzero,
            "fit_source": "full_evaluation",
            "fit_source_range": [0, length],
        },
    }
