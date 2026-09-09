"""PaAno의 fit-only memory와 세션별 score 정렬을 소유한다."""

import numpy
import torch
from torch.utils.data import DataLoader, TensorDataset

from .official import (
    PatchEncoder,
    choose_training_batch_size,
    encode_patches,
    score_embeddings,
    score_patches,
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
    patch_scores = numpy.nan_to_num(patch_scores, nan=0.0, posinf=0.0, neginf=0.0)
    kernel = numpy.ones(patch_size, dtype=numpy.float32)
    sums = numpy.convolve(patch_scores, kernel, mode="full")[:length]
    coverage = numpy.convolve(numpy.ones_like(patch_scores), kernel, mode="full")[:length]
    scores = numpy.divide(sums, coverage, out=numpy.zeros(length, dtype=numpy.float32), where=coverage != 0)
    return numpy.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


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
        memory_policy: str | None = None,
        full_prefix: bool = False,
        official_procedure: bool = False,
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
        if memory_policy is None:
            memory_policy = "official_minimum" if official_procedure else "legacy_fraction"
        if memory_policy not in {"legacy_fraction", "official_minimum", "paper_fraction"}:
            raise ValueError(f"알 수 없는 PaAno memory_policy: {memory_policy}")
        if official_procedure and memory_policy == "legacy_fraction":
            raise ValueError("PaAno official procedure requires rounded memory selection")
        self.memory_policy = memory_policy
        self.full_prefix = full_prefix
        self.official_procedure = official_procedure
        self.training_context = None
        self.encoder_factory = encoder_factory or (
            lambda channel_count: PatchEncoder(
                in_channels=channel_count, use_revin=use_revin,
            )
        )
        self.trainer = trainer or train_encoder
        self.model = None
        self.memory_bank = None
        self.channel_count = None

    def prepare_model(self, channel_count: int) -> None:
        if type(channel_count) is not int or channel_count < 1:
            raise ValueError("PaAno channel_count는 양의 정수여야 한다")
        if self.model is not None:
            if self.channel_count != channel_count:
                raise ValueError("PaAno 준비 모델과 fit 채널 수가 다르다")
            return
        self.channel_count = channel_count
        self.model = self.encoder_factory(channel_count).to(self.device)

    def fit(self, fit_values):
        fit_patches = make_patch_tensor(fit_values, self.patch_size)
        if len(fit_patches) <= self.patch_size:
            raise ValueError("PaAno fit patch가 pretext 간격보다 길어야 한다")
        if self.official_procedure:
            if self.model is not None:
                raise ValueError("PaAno official fit must initialize its encoder after the source loader warmup")
            if self.batch_size < 2 or (
                len(fit_patches) % self.batch_size == 1
                and self.iterations >= (len(fit_patches) + self.batch_size - 1) // self.batch_size
            ):
                raise ValueError("PaAno official batches cannot contain a singleton; the requested batch size is preserved")
            # The source inspects a shuffled batch before constructing the encoder.
            next(iter(DataLoader(
                TensorDataset(fit_patches, torch.arange(len(fit_patches)).unsqueeze(1)),
                batch_size=self.batch_size, shuffle=True,
            )))
        self.prepare_model(fit_patches.shape[1])
        if self.official_procedure:
            self.training_context = numpy.asarray(fit_values, dtype=numpy.float32)[
                -(self.patch_size - 1):
            ].copy() if self.patch_size > 1 else numpy.empty((0, self.channel_count), dtype=numpy.float32)
        training_log = self.trainer(
            self.model,
            fit_patches,
            iterations=self.iterations,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            device=self.device,
            **({"official_procedure": True} if self.official_procedure else {}),
        )
        training_log = dict(training_log)
        update_count = int(training_log["optimizer_updates"])
        effective_batch_size = self.batch_size if self.official_procedure else choose_training_batch_size(
            len(fit_patches), self.batch_size,
        )
        training_log["effective_batch_size"] = effective_batch_size
        training_log["training_patch_count"] = len(fit_patches)
        batches_per_epoch = (len(fit_patches) + effective_batch_size - 1) // effective_batch_size
        complete_epochs, remaining_batches = divmod(update_count, batches_per_epoch)
        training_log["training_examples_seen"] = complete_epochs * len(fit_patches) + sum(
            min(effective_batch_size, len(fit_patches) - batch * effective_batch_size)
            for batch in range(remaining_batches)
        )
        fit_embeddings = encode_patches(
            self.model, fit_patches, self.device,
            **({"batch_size": self.batch_size, "shuffle": True} if self.official_procedure else {}),
        )
        self.memory_bank = select_memory_bank(
            fit_embeddings,
            fraction=self.memory_fraction,
            random_seed=self.memory_seed,
            memory_policy=self.memory_policy,
        )
        training_log["memory_policy"] = self.memory_policy
        training_log["memory_count"] = len(self.memory_bank)
        if self.official_procedure:
            training_log.update(
                official_protocol="paper_tuning_v4",
                checkpoint_selection="minimum_training_iteration_loss",
                training_scope="current_prefix",
                model_internal_validation=None,
                memory_embedding_order="shuffled_training_loader",
                initialization_order="shuffled_loader_warmup_then_encoder",
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
            **({"training_context": self.training_context.copy()} if self.official_procedure else {}),
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
                "memory_policy": self.memory_policy,
                "full_prefix": self.full_prefix,
                "official_procedure": self.official_procedure,
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
        adapter.prepare_model(channel_count)
        adapter.model.load_state_dict(checkpoint["model_state_dict"])
        adapter.memory_bank = torch.as_tensor(
            checkpoint["memory_bank"], dtype=torch.float32,
        ).detach().cpu()
        if adapter.official_procedure:
            context = numpy.asarray(checkpoint["training_context"], dtype=numpy.float32)
            if context.shape != (adapter.patch_size - 1, channel_count) or not numpy.isfinite(context).all():
                raise ValueError("PaAno official checkpoint training context is invalid")
            adapter.training_context = context.copy()
        return adapter

    def score(self, values):
        if self.model is None or self.memory_bank is None:
            raise RuntimeError("PaAno fit을 먼저 실행해야 한다")
        values = numpy.asarray(values, dtype=numpy.float32)
        source_length = len(values)
        context_length = 0
        if self.official_procedure:
            if self.training_context is None:
                raise RuntimeError("PaAno official score needs its fitted training context")
            context_length = len(self.training_context)
            values = numpy.concatenate((self.training_context, values), axis=0)
        patches = make_patch_tensor(values, self.patch_size)
        if self.official_procedure:
            patch_scores = score_patches(
                self.model, patches, self.memory_bank, device=self.device,
                top_k=self.top_k, batch_size=self.batch_size,
            )
        else:
            embeddings = encode_patches(self.model, patches, self.device)
            patch_scores = score_embeddings(embeddings, self.memory_bank, self.top_k)
        scores = stitch_patch_scores(patch_scores.numpy(), self.patch_size, len(values))
        return {
            "scores": scores[context_length:],
            "source_start": 0,
            "source_end_exclusive": source_length,
            "alignment": "overlap_mean",
            "primitive": "top3_cosine_distance",
            "calibration_mode": "none" if self.full_prefix else "validation_median_iqr",
            "evaluation_mode": "offline_noncausal",
            "lookahead": self.patch_size - 1,
            "maximum_effective_lookahead": self.patch_size - 1,
            "normalization_scope": "none" if self.full_prefix else "current_prefix_validation",
            **({
                "calibration_mode": "none",
                "normalization_scope": "none",
                "native_postprocessing": True,
                "official_protocol": "paper_tuning_v4",
                "native_postprocessing_recipe": "top3_cosine_overlap_mean",
                "training_context_rows": context_length,
            } if self.official_procedure else {}),
        }
