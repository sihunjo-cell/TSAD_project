"""PaAno의 fit-only memory와 세션별 score 정렬을 소유한다."""

import numpy
import torch

from .official import (
    PatchEncoder,
    choose_training_batch_size,
    encode_patches,
    score_embeddings,
    select_memory_bank,
    train_encoder,
)


def make_patch_tensor(values, patch_size: int):
    values = numpy.asarray(values, dtype=numpy.float32)
    if values.ndim != 2 or len(values) < patch_size:
        raise ValueError("PaAno 입력은 patch 이상 길이의 2차원 배열이어야 한다")
    if not numpy.isfinite(values).all():
        raise ValueError("PaAno 입력에는 유한값만 허용한다")
    return torch.from_numpy(numpy.ascontiguousarray(values)).unfold(0, patch_size, 1)


def stitch_patch_scores(patch_scores, patch_size: int, length: int):
    patch_scores = numpy.asarray(patch_scores, dtype=numpy.float32)
    if patch_scores.ndim != 1 or len(patch_scores) != length - patch_size + 1:
        raise ValueError("PaAno patch 점수와 원시 길이가 맞지 않는다")
    sums = numpy.zeros(length, dtype=numpy.float32)
    coverage = numpy.zeros(length, dtype=numpy.int64)
    for start, score in enumerate(patch_scores):
        sums[start:start + patch_size] += score
        coverage[start:start + patch_size] += 1
    return sums / coverage


class PaAnoAdapter:
    def __init__(
        self,
        *,
        patch_size: int,
        learning_rate: float,
        device: str,
        iterations: int = 100,
        batch_size: int = 512,
        weight_decay: float = 1e-4,
        memory_fraction: float = 0.1,
        neighbors: int = 3,
        use_revin: bool = True,
        memory_seed: int = 42,
        encoder_factory=None,
        trainer=None,
    ):
        self.patch_size = patch_size
        self.learning_rate = learning_rate
        self.device = device
        self.iterations = iterations
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.memory_fraction = memory_fraction
        self.top_k = neighbors
        self.use_revin = use_revin
        self.memory_seed = memory_seed
        self.encoder_factory = encoder_factory or (
            lambda channel_count: PatchEncoder(
                in_channels=channel_count, use_revin=use_revin,
            )
        )
        self.trainer = trainer or train_encoder
        self.model = None
        self.memory_bank = None
        self.channel_count = None

    def fit(self, fit_values):
        fit_patches = make_patch_tensor(fit_values, self.patch_size)
        if len(fit_patches) <= self.patch_size:
            raise ValueError("PaAno fit patch가 pretext 간격보다 길어야 한다")
        self.channel_count = fit_patches.shape[1]
        self.model = self.encoder_factory(fit_patches.shape[1]).to(self.device)
        training_log = self.trainer(
            self.model,
            fit_patches,
            iterations=self.iterations,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            device=self.device,
        )
        training_log = dict(training_log)
        update_count = int(training_log["optimizer_updates"])
        effective_batch_size = choose_training_batch_size(
            len(fit_patches), self.batch_size,
        )
        batches_per_epoch = (len(fit_patches) + effective_batch_size - 1) // effective_batch_size
        complete_epochs, remaining_batches = divmod(update_count, batches_per_epoch)
        training_log["training_examples_seen"] = complete_epochs * len(fit_patches) + sum(
            min(effective_batch_size, len(fit_patches) - batch * effective_batch_size)
            for batch in range(remaining_batches)
        )
        fit_embeddings = encode_patches(self.model, fit_patches, self.device)
        self.memory_bank = select_memory_bank(
            fit_embeddings,
            fraction=self.memory_fraction,
            random_seed=self.memory_seed,
        )
        return training_log

    def checkpoint(self):
        if self.model is None or self.memory_bank is None:
            raise RuntimeError("PaAno fit을 먼저 실행해야 한다")
        return {
            "model_state_dict": {
                name: value.detach().cpu()
                for name, value in self.model.state_dict().items()
            },
            "memory_bank": self.memory_bank.detach().cpu(),
            "model_config": {
                "channel_count": self.channel_count,
                "patch_size": self.patch_size,
                "learning_rate": self.learning_rate,
                "iterations": self.iterations,
                "batch_size": self.batch_size,
                "weight_decay": self.weight_decay,
                "memory_fraction": self.memory_fraction,
                "neighbors": self.top_k,
                "use_revin": self.use_revin,
                "memory_seed": self.memory_seed,
            },
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint,
        *,
        device: str,
        encoder_factory=None,
        trainer=None,
    ):
        config = dict(checkpoint["model_config"])
        channel_count = config.pop("channel_count")
        adapter = cls(
            device=device,
            encoder_factory=encoder_factory,
            trainer=trainer,
            **config,
        )
        adapter.channel_count = channel_count
        adapter.model = adapter.encoder_factory(channel_count).to(device)
        adapter.model.load_state_dict(checkpoint["model_state_dict"])
        adapter.memory_bank = torch.as_tensor(
            checkpoint["memory_bank"], dtype=torch.float32,
        ).detach().cpu()
        return adapter

    def score(self, values):
        if self.model is None or self.memory_bank is None:
            raise RuntimeError("PaAno fit을 먼저 실행해야 한다")
        values = numpy.asarray(values, dtype=numpy.float32)
        patches = make_patch_tensor(values, self.patch_size)
        embeddings = encode_patches(self.model, patches, self.device)
        patch_scores = score_embeddings(embeddings, self.memory_bank, self.top_k)
        scores = stitch_patch_scores(patch_scores.numpy(), self.patch_size, len(values))
        return {
            "scores": scores,
            "source_start": 0,
            "source_end_exclusive": len(values),
            "alignment": "overlap_mean",
            "primitive": "top3_cosine_distance",
            "calibration_mode": "validation_median_iqr",
            "evaluation_mode": "offline_noncausal",
            "lookahead": self.patch_size - 1,
            "maximum_effective_lookahead": self.patch_size - 1,
            "normalization_scope": "current_prefix_validation",
        }
