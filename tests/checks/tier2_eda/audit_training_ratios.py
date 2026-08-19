"""고정한 모델 입력 길이로 비율별 window 수를 계산한다."""

import math

from src.data_split.take_training_prefix import compute_kept_length
from src.common.experiment_config import load_validation_fraction


SOURCE_COMMITS = {
    "d-ailin/GDN": "9853899da860682669a134e4af315d036aab4eca",
    "GraGOD": "ec8cd452a410ba903a31beb097a010ba0448c095",
    "manigalati/usad": "e25af45c8e1c32783aed94fc7e5ab85effef2b1a",
    "TheDatumOrg/TSB-AD": "e0975a5f7d3e65ab77e9fab24d1b5b51acda8f48",
}
VALIDATION_FRACTION = load_validation_fraction()

MODEL_CANDIDATES = {
    "GHL": (
        {"model": "CI-AE", "model_input_window_size": 100, "train_stride": 1, "evaluation_stride": 1,
         "batch_size": 128, "target_horizon": 0, "drop_last": True, "max_epochs": 50,
         "training_unit": "one_model_per_channel", "early_stopping_rule": "none",
         "optimizer_steps_per_batch": 1},
        {"model": "LSTM-AD", "model_input_window_size": 100, "train_stride": 1, "evaluation_stride": 1,
         "batch_size": 128, "target_horizon": 1, "drop_last": False, "max_epochs": 50,
         "training_unit": "one_multivariate_model", "early_stopping_rule": "patience=3",
         "optimizer_steps_per_batch": 1},
        {"model": "USAD", "model_input_window_size": 10, "train_stride": 1, "evaluation_stride": 1,
         "batch_size": 128, "target_horizon": 0, "drop_last": False, "max_epochs": 70,
         "training_unit": "one_multivariate_model", "early_stopping_rule": "none",
         "optimizer_steps_per_batch": 2},
        {"model": "GDN", "model_input_window_size": 5, "train_stride": 1, "evaluation_stride": 1,
         "batch_size": 32, "target_horizon": 1, "drop_last": False, "max_epochs": 50,
         "training_unit": "one_multivariate_model", "early_stopping_rule": "patience=10",
         "optimizer_steps_per_batch": 1},
    ),
    "HAI": (
        {"model": "GDN", "model_input_window_size": 5, "train_stride": 1, "evaluation_stride": 1,
         "batch_size": 32, "target_horizon": 1, "drop_last": False, "max_epochs": 50,
         "training_unit": "one_model_over_all_train_sessions", "early_stopping_rule": "patience=10",
         "optimizer_steps_per_batch": 1},
    ),
}


def compute_split_lengths(total_length: int, ratio_percent: int) -> tuple[int, int, int]:
    validation_length = math.ceil(total_length * VALIDATION_FRACTION)
    fit_pool_length = total_length - validation_length
    fit_subset_length = compute_kept_length(fit_pool_length, ratio_percent / 100)
    return fit_pool_length, fit_subset_length, validation_length


def compute_window_count(length: int, window_size: int, stride: int, target_horizon: int) -> int:
    available_starts = length - window_size - target_horizon + 1
    return max((available_starts - 1) // stride + 1, 0)


def compute_window_counts_by_segment(
    lengths: list[int] | tuple[int, ...],
    window_size: int,
    stride: int,
    target_horizon: int,
) -> list[int]:
    return [compute_window_count(length, window_size, stride, target_horizon) for length in lengths]


def count_batches(window_count: int, batch_size: int, drop_last: bool) -> int:
    return window_count // batch_size if drop_last else math.ceil(window_count / batch_size)


def build_ratio_rows(
    dataset: str,
    source: str,
    total_length: int,
    ratio_percent: int,
    feature_count: int,
) -> list[dict]:
    fit_pool_length, fit_subset_length, validation_length = compute_split_lengths(
        total_length, ratio_percent
    )
    rows = []
    for candidate in MODEL_CANDIDATES[dataset]:
        window_size = candidate["model_input_window_size"]
        train_windows = compute_window_count(
            fit_subset_length, window_size, candidate["train_stride"], candidate["target_horizon"]
        )
        validation_windows = compute_window_count(
            validation_length, window_size, candidate["evaluation_stride"], candidate["target_horizon"]
        )
        batch_size = candidate["batch_size"]
        full_batches, remainder = divmod(train_windows, batch_size)
        drop_last = candidate["drop_last"]
        batches = count_batches(train_windows, batch_size, drop_last)
        updates = batches * candidate["optimizer_steps_per_batch"]
        validation_batches = count_batches(validation_windows, batch_size, False)
        last_batch_size = batch_size if full_batches else 0
        if not drop_last and train_windows:
            last_batch_size = remainder or batch_size
        reconstruction_model = candidate["model"] in {"CI-AE", "USAD"}
        first_score_index = math.ceil((window_size - 1) / 2) if reconstruction_model else window_size
        score_length = (
            "test_length-model_input_window_size+1"
            if reconstruction_model
            else "test_length-model_input_window_size"
        )
        rows.append({
            "dataset": dataset,
            "file_or_session": source,
            "ratio": ratio_percent,
            "stage": "normal",
            "model": candidate["model"],
            "original_train_length": total_length,
            "fit_pool_length": fit_pool_length,
            "fit_subset_length": fit_subset_length,
            "unused_fit_pool_length": fit_pool_length - fit_subset_length,
            "validation_length": validation_length,
            **candidate,
            "train_window_count": train_windows,
            "validation_window_count": validation_windows,
            "full_train_batches": full_batches,
            "train_batch_count_per_epoch": batches,
            "last_train_batch_size": last_batch_size,
            "dropped_train_windows": remainder if drop_last else 0,
            "optimizer_updates_per_epoch": updates,
            "validation_batch_count": validation_batches,
            "maximum_optimizer_updates": updates * candidate["max_epochs"],
            "model_instance_count": feature_count if candidate["model"] == "CI-AE" else 1,
            "first_valid_score_index": first_score_index,
            "expected_saved_score_length": score_length,
            "train_feasible": train_windows > 0 and updates > 0,
            "validation_feasible": validation_windows > 0,
            "execution_feasible": train_windows > 0 and updates > 0 and validation_windows > 0,
            "fit_validation_boundary_crossing_window_count": 0,
            "windows_excluded_to_preserve_fit_validation_boundary": 0,
            "file_or_session_boundary_crossing_window_count": 0,
            "workload_status": (
                "fixed_project_transfer_rule"
                if candidate["model"] == "USAD"
                else "fixed_for_pretraining_audit"
            ),
            "candidate_status": (
                "window and batch_size=128 fixed by project transfer rule"
                if candidate["model"] == "USAD"
                else (
                    "window and workload fixed; project topk does not change these counts"
                    if candidate["model"] == "GDN"
                    else "window and workload settings fixed"
                )
            ),
        })
    return rows


def build_training_workload_summary_rows(
    ratio_rows: list[dict], dataset: str, feature_count: int
) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in ratio_rows:
        source = row["file_or_session"] if dataset == "GHL" else "all_train_sessions"
        grouped.setdefault((source, row["ratio"], row["model"]), []).append(row)

    summaries = []
    for (source, ratio, model), rows in sorted(grouped.items()):
        first = rows[0]
        train_windows_per_model = sum(row["train_window_count"] for row in rows)
        validation_windows_per_model = sum(row["validation_window_count"] for row in rows)
        model_instances = feature_count if model == "CI-AE" else 1
        train_batches_per_model = count_batches(
            train_windows_per_model, first["batch_size"], first["drop_last"]
        )
        validation_batches_per_model = count_batches(
            validation_windows_per_model, first["batch_size"], False
        )
        updates_per_model = train_batches_per_model * first["optimizer_steps_per_batch"]
        updates_all_models = updates_per_model * model_instances
        hypothetical_session_crossings = 0
        if dataset == "HAI":
            hypothetical_session_crossings = max(
                compute_window_count(
                    sum(row["fit_subset_length"] for row in rows),
                    first["model_input_window_size"],
                    first["train_stride"],
                    first["target_horizon"],
                ) - train_windows_per_model,
                0,
            )
        summaries.append({
            "dataset": dataset,
            "training_scope": source,
            "ratio": ratio,
            "model": model,
            "training_unit": first["training_unit"],
            "source_session_count": len(rows),
            "model_instance_count": model_instances,
            "train_window_count_per_model": train_windows_per_model,
            "train_window_count_all_models": train_windows_per_model * model_instances,
            "validation_window_count_per_model": validation_windows_per_model,
            "validation_window_count_all_models": validation_windows_per_model * model_instances,
            "batch_size": first["batch_size"],
            "drop_last": first["drop_last"],
            "train_batches_per_epoch_per_model": train_batches_per_model,
            "train_batches_per_epoch_all_models": train_batches_per_model * model_instances,
            "optimizer_steps_per_batch": first["optimizer_steps_per_batch"],
            "optimizer_updates_per_epoch_per_model": updates_per_model,
            "optimizer_updates_per_epoch_all_models": updates_all_models,
            "validation_batches_per_model": validation_batches_per_model,
            "validation_batches_all_models": validation_batches_per_model * model_instances,
            "max_epochs": first["max_epochs"],
            "maximum_optimizer_updates": updates_all_models * first["max_epochs"],
            "early_stopping_rule": first["early_stopping_rule"],
            "workload_status": first["workload_status"],
            "actual_optimizer_updates_status": (
                "batch_size fixed by project transfer rule; execution log required"
                if model == "USAD" else "execution log required"
            ),
            "fit_validation_boundary_crossing_window_count": 0,
            "session_boundary_crossing_window_count": 0,
            "windows_excluded_to_preserve_session_boundaries": hypothetical_session_crossings,
        })
    return summaries


def build_model_window_source_rows(dataset: str) -> list[dict]:
    rows = [
        ("GDN", "논문", "GDN AAAI 2021", "5", "SWaT·WADI", "해당 없음", "paper", True,
         "논문 실험값을 우선한다."),
        ("GDN", "공식 코드", "d-ailin/GDN main.py 기본값", "15", "명령행 기본값", "기본값 사용 시 15", SOURCE_COMMITS["d-ailin/GDN"], False,
         "논문 실험값 5와 다르므로 채택하지 않는다."),
        ("GDN", "현재 config", "configs/gdn_hyperparams.yaml", "5", "GHL·HAI", "명시값 5", "project", True,
         "논문값과 일치하며 모든 비율과 두 데이터셋에 고정한다."),
    ]
    if dataset == "GHL":
        rows += [
            ("CI-AE", "TSB-AD wrapper", "run_AutoEncoder", "100", "채널별 univariate 입력", "slidingWindow=window_size", SOURCE_COMMITS["TheDatumOrg/TSB-AD"], True,
             "채택 구현의 실제 기본값이며 채널별 입력에서 window 변환이 적용된다."),
            ("CI-AE", "TSB-AD 구현 동작", "AutoEncoder.fit", "100", "univariate에서만 window 변환", "다변량은 변환 생략", SOURCE_COMMITS["TheDatumOrg/TSB-AD"], True,
             "CI-AE는 채널별 입력으로 univariate 경로를 강제한다."),
            ("LSTM-AD", "원 논문", "EncDec-AD ICML Workshop 2016", "dataset-specific", "공정·주기별 30~500", "해당 없음", "paper", False,
             "보편적인 입력 길이를 제시하지 않는다."),
            ("LSTM-AD", "TSB-AD wrapper", "run_LSTMAD", "100", "TSB-AD 이식 기준", "window_size=100", SOURCE_COMMITS["TheDatumOrg/TSB-AD"], True,
             "현재 채택 구현의 실제 기본값을 따른다."),
            ("USAD", "논문 민감도", "USAD KDD 2020", "10", "window 5·10·20·50·100 비교", "해당 없음", "paper", True,
             "비교값 중 10이 가장 높았으며 GHL 결과 전 transfer 기준으로 고정한다."),
            ("USAD", "논문 부록", "USAD supplementary", "SWaT 12; WADI 10; SMD·SMAP·MSL 5", "데이터셋별", "해당 없음", "paper", False,
             "데이터셋마다 값이 달라 10을 보편적 기본값이라고 부르지 않는다."),
            ("USAD", "공식 코드", "manigalati/usad SWaT notebook", "12", "SWaT", "window_size=12", SOURCE_COMMITS["manigalati/usad"], False,
             "SWaT 재현값이며 GHL transfer 기준과 구분한다."),
            ("USAD", "TSB-AD wrapper", "run_USAD", "5", "TSB-AD 기본값", "win_size=5", SOURCE_COMMITS["TheDatumOrg/TSB-AD"], False,
             "논문 민감도와 WADI 설정을 우선해 명시적으로 10으로 덮어쓴다."),
        ]
    return [
        {
            "dataset": dataset,
            "model": model,
            "source_type": source_type,
            "source": source,
            "window_value": value,
            "applies_to": applies_to,
            "actual_override": override,
            "source_commit": commit,
            "project_selected": selected,
            "reason": reason,
        }
        for model, source_type, source, value, applies_to, override, commit, selected, reason in rows
    ]
