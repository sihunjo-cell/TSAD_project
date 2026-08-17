"""GHL −TOPK·TopK 민감도 팔의 조합과 완전성을 검사한다."""

import itertools
from pathlib import Path

import yaml

from experiments.exp02_gdn_ghl.check_completeness import (
    is_run_directory_complete,
    run_directory,
)
from src.common.experiment_config import load_planned_ratios_and_seeds


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def load_arm_config() -> dict:
    with (REPOSITORY_ROOT / "configs" / "gdn_hyperparams.yaml").open(encoding="utf-8") as file:
        return yaml.safe_load(file)["analysis_arms"]["GHL"]


def build_notopk_specs():
    arm = load_arm_config()
    _, seeds = load_planned_ratios_and_seeds("GHL")
    return list(itertools.product(range(1, 26), arm["ratios"], seeds))


def build_topk_sensitivity_specs():
    arm = load_arm_config()
    _, seeds = load_planned_ratios_and_seeds("GHL")
    return list(itertools.product(range(1, 26), arm["ratios"], arm["topk_values"], seeds))


def notopk_run_directory(experiment_dir, spec) -> Path:
    series, ratio, seed = spec
    return (
        Path(experiment_dir) / "control_runs" / "notopk"
        / f"series_{series:02d}" / f"r{ratio:03d}" / f"s{seed}"
    )


def sensitivity_run_directory(experiment_dir, spec) -> Path:
    series, ratio, topk, seed = spec
    if topk == 5:
        return run_directory(experiment_dir, (series, ratio, seed))
    return (
        Path(experiment_dir) / "control_runs" / f"topk_k{topk:03d}"
        / f"series_{series:02d}" / f"r{ratio:03d}" / f"s{seed}"
    )


def is_notopk_complete(experiment_dir, spec, expected_git_hashes=None) -> bool:
    model = load_arm_config()["notopk_model"]
    return is_run_directory_complete(
        notopk_run_directory(experiment_dir, spec), spec, model, expected_git_hashes,
    )


def is_topk_sensitivity_complete(experiment_dir, spec, expected_git_hashes=None) -> bool:
    series, ratio, topk, seed = spec
    model = "GDN" if topk == 5 else f"GDN_K{topk}"
    return is_run_directory_complete(
        sensitivity_run_directory(experiment_dir, spec), (series, ratio, seed), model,
        expected_git_hashes,
    )


def find_missing_notopk_runs(experiment_dir, specs=None):
    specs = build_notopk_specs() if specs is None else specs
    return [spec for spec in specs if not is_notopk_complete(experiment_dir, spec)]


def find_missing_topk_sensitivity_runs(experiment_dir, specs=None):
    specs = build_topk_sensitivity_specs() if specs is None else specs
    return [spec for spec in specs if not is_topk_sensitivity_complete(experiment_dir, spec)]
