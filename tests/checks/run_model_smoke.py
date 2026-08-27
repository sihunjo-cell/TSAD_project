"""실데이터·checkpoint 없이 활성 모델의 점수 정렬 계약을 점검한다."""

import json
import sys
from pathlib import Path

import numpy
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.models.tier1 import PcaLegacy, score_mwvar, score_sqdiff_last3
from src.models.tier2.alora.adapter import stitch_alora_scores
from src.models.tier2.gdn_official.adapter import build_forecast_arrays, topk_from_rho
from src.models.tier2.paano.adapter import stitch_patch_scores
from src.models.tier3.time_rcd import score_time_rcd
from src.models.tier3.tspulse import score_tspulse


OUTPUT_PATH = (
    REPOSITORY_ROOT / "experiments" / "checks" / "reference_code"
    / "active_models" / "source_only_smoke.json"
)


class _ConstantTimeRCD(torch.nn.Module):
    def forward(self, *, time_series, mask):
        del mask
        logits = torch.zeros((*time_series.shape, 2), device=time_series.device)
        logits[..., 1] = 1.0
        return logits


def _shape(output):
    return list(numpy.asarray(output["scores"]).shape)


def run_source_only_smoke() -> dict:
    """CATCH를 제외한 모델의 순수 점수·stitching 경로만 합성 입력으로 실행한다."""
    random = numpy.random.default_rng(25)
    matrix = random.normal(size=(140, 4)).astype(numpy.float32)
    models = {}

    def check(name, operation):
        try:
            details = operation()
        except Exception as error:  # pragma: no cover - 보고서에 실패 원인을 보존한다.
            models[name] = {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
        else:
            models[name] = {"status": "passed", **details}

    check("MWVAR", lambda: {"score_shape": _shape(score_mwvar(matrix))})
    check("SQDIFF_LAST3", lambda: {"score_shape": _shape(score_sqdiff_last3(matrix))})

    def check_pca():
        model = PcaLegacy(n_components=0.5).fit(matrix)
        return {"score_shape": _shape(model.score(matrix[-120:]))}

    check("PCA_LEGACY", check_pca)
    check("PaAno", lambda: {
        "score_shape": list(stitch_patch_scores(numpy.arange(7), 4, 10).shape),
    })
    check("ALoRa", lambda: {
        "score_shape": list(stitch_alora_scores(
            numpy.ones((3, 3)), numpy.ones(3),
        ).shape),
    })

    def check_gdn():
        arrays = build_forecast_arrays((matrix[:9],), window_size=5)
        return {
            "window_shape": list(arrays[0][0].shape),
            "score_shape": list(arrays[0][1].shape),
            "topk": topk_from_rho(4, 0.3),
        }

    check("GDN", check_gdn)
    check("TimeRCD", lambda: {
        "score_shape": _shape(score_time_rcd(
            _ConstantTimeRCD(), matrix[:9], context_length=5,
        )),
    })

    def check_tspulse():
        values = matrix[:12]

        def raw_heads(_values, *, aggregation_window):
            del aggregation_window
            raw = numpy.arange(4, dtype=numpy.float64)
            return {head: raw for head in ("time", "fft", "pred")}

        outputs = score_tspulse(
            values,
            raw_head_function=raw_heads,
            aggregation_window=4,
            context_length=8,
        )
        return {"score_shapes": {name: _shape(output) for name, output in outputs.items()}}

    check("TSPulse", check_tspulse)
    return {
        "input_kind": "synthetic_only",
        "uses_labels": False,
        "downloads_checkpoints": False,
        "models": models,
    }


def main() -> None:
    report = run_source_only_smoke()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    if any(result["status"] != "passed" for result in report["models"].values()):
        raise RuntimeError(f"source-only smoke 실패: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
