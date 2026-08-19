"""학습 없이 로컬 계층 2 모델의 입력·출력 형상을 확인한다."""

import ast
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import torch
from torch import nn

from src.models.tier2.utils.utility import get_activation_by_name


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MODEL_SOURCES = {
    "CI-AE": REPOSITORY_ROOT / "src/models/tier2/AE/AE_raw(TSB).py",
    "LSTM-AD": REPOSITORY_ROOT / "src/models/tier2/LSTMAD/LSTMAD_raw.py",
    "USAD": REPOSITORY_ROOT / "src/models/tier2/USAD/USAD_raw.py",
}


def expected_forward_contracts(dataset: str, feature_count: int) -> dict[str, dict]:
    contracts = {
        "CI-AE": {"input_shape": (2, 100), "output_shapes": ((2, 100),)},
        "LSTM-AD": {
            "input_shape": (2, 100, feature_count),
            "output_shapes": ((1, 2, feature_count),),
        },
        "USAD": {
            "input_shape": (2, 10, feature_count),
            "output_shapes": ((2, 10 * feature_count),) * 3,
        },
        "GDN": {
            "input_shape": (2, feature_count, 5),
            "output_shapes": ((2, feature_count),),
        },
    }
    return contracts if dataset == "GHL" else {"GDN": contracts["GDN"]}


def _load_class_from_source(model: str, class_name: str):
    path = MODEL_SOURCES[model]
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    namespace = {"torch": torch, "nn": nn, "get_activation_by_name": get_activation_by_name}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[class_node], type_ignores=[])), str(path), "exec"), namespace)
    return namespace[class_name]


def _check_full_module_import(model: str) -> tuple[bool, str]:
    path = MODEL_SOURCES[model]
    module_name = f"src.models.tier2.{path.parent.name}._tier2_eda_import_check"
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError("모듈 명세를 만들지 못했습니다.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return True, ""
    except Exception as error:  # 환경 의존성도 실행 전 계약의 일부다.
        return False, f"{type(error).__name__}: {error}"
    finally:
        sys.modules.pop(module_name, None)


def _check_gdn_entrypoint_import(dataset: str) -> tuple[bool, str]:
    module_name = (
        "tests.ghl_main.run_tier2" if dataset == "GHL" else "tests.hai_extension.run_gdn"
    )
    try:
        importlib.import_module(module_name)
        return True, ""
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"
    finally:
        sys.modules.pop(module_name, None)


def _forward(model: str, feature_count: int, topk: int) -> tuple[tuple[int, ...], ...]:
    if model == "CI-AE":
        model_class = _load_class_from_source(model, "InnerAutoencoder")
        network = model_class(100, hidden_neurons=(64, 32))
        input_tensor = torch.zeros(2, 100)
    elif model == "LSTM-AD":
        model_class = _load_class_from_source(model, "LSTMModel")
        network = model_class(100, feature_count, 20, 1, 2, 2, torch.device("cpu"))
        input_tensor = torch.zeros(2, 100, feature_count)
    elif model == "USAD":
        model_class = _load_class_from_source(model, "USADModel")
        network = model_class(feature_count, n_window=10)
        input_tensor = torch.zeros(2, 10, feature_count)
    else:
        from src.models.tier2.GDN.model import GDN

        nodes = torch.arange(feature_count)
        edge_index = torch.cartesian_prod(nodes, nodes).T.contiguous()
        network = GDN(
            edge_index=[edge_index],
            n_features=feature_count,
            embed_dim=64,
            out_layer_inter_dim=128,
            window_size=5,
            out_layer_num=1,
            topk=topk,
            heads=1,
            dropout=0.2,
        )
        input_tensor = torch.zeros(2, feature_count, 5)

    network.eval()
    with torch.no_grad():
        output = network(input_tensor)
    outputs = output if isinstance(output, tuple) else (output,)
    return tuple(tuple(value.shape) for value in outputs)


def run_forward_smoke(dataset: str, feature_count: int, topk: int) -> list[dict]:
    sys.dont_write_bytecode = True
    torch.manual_seed(0)
    contracts = expected_forward_contracts(dataset, feature_count)
    rows = []
    for model, contract in contracts.items():
        try:
            actual_shapes = _forward(model, feature_count, topk)
            error = ""
        except Exception as forward_error:
            actual_shapes = ()
            error = f"{type(forward_error).__name__}: {forward_error}"

        if model == "GDN":
            full_import_ready, import_error = _check_gdn_entrypoint_import(dataset)
            source_class = "src.models.tier2.GDN.model.GDN"
        else:
            full_import_ready, import_error = _check_full_module_import(model)
            class_name = {"CI-AE": "InnerAutoencoder", "LSTM-AD": "LSTMModel", "USAD": "USADModel"}[model]
            source_class = f"{MODEL_SOURCES[model].name}:{class_name}"

        reference_source_matches = model != "USAD"

        rows.append({
            "dataset": dataset,
            "model": model,
            "source_class": source_class,
            "synthetic_batch_size": 2,
            "input_shape": json.dumps(contract["input_shape"]),
            "expected_output_shapes": json.dumps(contract["output_shapes"]),
            "actual_output_shapes": json.dumps(actual_shapes),
            "forward_passed": actual_shapes == contract["output_shapes"],
            "reference_source_matches": reference_source_matches,
            "reference_forward_passed": actual_shapes == contract["output_shapes"] and reference_source_matches,
            "forward_error": error,
            "full_module_import_ready": full_import_ready,
            "full_module_import_error": import_error,
            "model_training_executed": False,
            "scope": (
                "local TSB port interface only; official USAD source is not yet present"
                if model == "USAD"
                else "model core forward only; wrapper fit and predict were not called"
            ),
        })
    return rows
