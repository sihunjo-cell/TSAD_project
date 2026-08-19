"""대문자 로컬 GDN adapter의 학습 결과 계약."""

import unittest

import numpy

from src.models.tier2.GDN.adapter import run_gdn_sessions


class TestGdnSessions(unittest.TestCase):
    def test_reference_sessions_stay_separate_and_test_is_unpadded(self):
        random_generator = numpy.random.default_rng(1)
        train_sessions = (
            random_generator.random((12, 6), dtype=numpy.float32),
            random_generator.random((13, 6), dtype=numpy.float32),
        )
        validation_sessions = (random_generator.random((11, 6), dtype=numpy.float32),)
        test_sessions = (
            random_generator.random((10, 6), dtype=numpy.float32),
            random_generator.random((14, 6), dtype=numpy.float32),
        )
        result = run_gdn_sessions(
            train_sessions, validation_sessions, test_sessions,
            topk=2, device="cpu", epochs=1,
        )
        self.assertEqual(
            [errors.shape for errors in result["reference_errors"]],
            [(7, 6), (8, 6), (6, 6)],
        )
        self.assertEqual(
            [errors.shape for errors in result["test_errors"]],
            [(5, 6), (9, 6)],
        )
        self.assertEqual(result["training_log"]["scheduler"], "none")
        self.assertEqual(result["training_log"]["gradient_clip"], "none")


if __name__ == "__main__":
    unittest.main()
