"""Tier 2 모델 adapter의 합성 학습·정렬 계약."""

import unittest

import numpy
import torch

from src.models.tier2.AE.adapter import CI_AE_CONFIG, score_ci_ae, train_ci_ae
from src.models.tier2.GDN.adapter import GDN_CONFIG, score_gdn, train_gdn
from src.models.tier2.LSTMAD.adapter import (
    LSTM_AD_CONFIG,
    score_lstm_ad,
    train_lstm_ad,
)
from src.models.tier2.USAD.adapter import USAD_CONFIG, score_usad, train_usad
from src.models.tier2.USAD.official import UsadModel


class TestFixedConfiguration(unittest.TestCase):
    def test_fixed_windows_and_batches(self):
        self.assertEqual((CI_AE_CONFIG["window_size"], CI_AE_CONFIG["batch_size"]), (100, 128))
        self.assertEqual((LSTM_AD_CONFIG["window_size"], LSTM_AD_CONFIG["batch_size"]), (100, 128))
        self.assertEqual((USAD_CONFIG["window_size"], USAD_CONFIG["batch_size"]), (10, 128))
        self.assertEqual((GDN_CONFIG["window_size"], GDN_CONFIG["batch_size"]), (5, 32))

    def test_official_usad_forward_shape(self):
        model = UsadModel(window_size=190, latent_size=100)
        batch = torch.rand(4, 190)
        reconstruction1, reconstruction2, reconstruction3 = model(batch)
        self.assertEqual(reconstruction1.shape, batch.shape)
        self.assertEqual(reconstruction2.shape, batch.shape)
        self.assertEqual(reconstruction3.shape, batch.shape)
        self.assertEqual(model.encoder.linear1.out_features, 95)
        self.assertEqual(model.encoder.linear2.out_features, 47)
        self.assertEqual(model.encoder.linear3.out_features, 100)


class TestSyntheticOptimizerSteps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        random_generator = numpy.random.default_rng(0)
        cls.train = random_generator.random((240, 19), dtype=numpy.float32)
        cls.validation = random_generator.random((120, 19), dtype=numpy.float32)
        cls.test = random_generator.random((130, 19), dtype=numpy.float32)

    def test_ci_ae_update_and_alignment(self):
        models, log = train_ci_ae((self.train,), device="cpu", epochs=1)
        scores = score_ci_ae(models, (self.validation, self.test), device="cpu")
        self.assertEqual(len(models), 19)
        self.assertGreater(log["optimizer_updates"], 0)
        self.assertEqual([score.shape for score in scores], [(21, 19), (31, 19)])

    def test_lstm_update_and_alignment(self):
        model, log = train_lstm_ad(
            (self.train,), (self.validation,), device="cpu", epochs=1,
        )
        scores = score_lstm_ad(model, (self.validation, self.test), device="cpu")
        self.assertGreater(log["optimizer_updates"], 0)
        self.assertEqual([score.shape for score in scores], [(20, 19), (30, 19)])

    def test_usad_two_updates_per_batch_and_alignment(self):
        model, log = train_usad((self.train,), device="cpu", epochs=1)
        scores = score_usad(model, (self.validation, self.test), device="cpu")
        self.assertEqual(log["optimizer_updates"], 2 * log["batch_iterations"])
        self.assertEqual([score.shape for score in scores], [(111, 19), (121, 19)])

    def test_gdn_update_contract_and_alignment(self):
        model, log = train_gdn(
            (self.train,), (self.validation,), topk=5, device="cpu", epochs=1,
        )
        scores = score_gdn(model, (self.validation, self.test), device="cpu")
        self.assertGreater(log["optimizer_updates"], 0)
        self.assertEqual([score.shape for score in scores], [(115, 19), (125, 19)])
        self.assertEqual(model.gnn_layers[0].gnn.dropout, 0)
        self.assertEqual(model.dp.p, 0.2)
        self.assertEqual(log["scheduler"], "none")
        self.assertEqual(log["gradient_clip"], "none")


if __name__ == "__main__":
    unittest.main()
