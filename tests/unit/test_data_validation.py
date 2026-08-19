"""src/data_split/validation_split.py 단위 테스트."""

import unittest

import numpy

from src.data_split.take_training_prefix import take_training_prefix
from src.data_split.validation_split import InsufficientTrainLengthError, validation_split


class TestValidationSplit(unittest.TestCase):
    def test_tail_becomes_validation_no_shuffle(self):
        # 20행 정상 구간에서 뒤 2행을 먼저 고정 validation으로 둔다.
        time_index, channel_index = numpy.meshgrid(range(20), range(3), indexing="ij")
        normal_training = time_index * 10 + channel_index
        fit_pool, val_part = validation_split(
            normal_training, val_fraction=0.1, min_train_length=1
        )
        self.assertEqual(fit_pool.shape, (18, 3))
        self.assertEqual(val_part.shape, (2, 3))
        numpy.testing.assert_array_equal(val_part[:, 0], [180, 190])
        # 셔플 없음: train+val 이어붙이면 원 배열 그대로 (시간 순서 보존).
        numpy.testing.assert_array_equal(
            numpy.concatenate([fit_pool, val_part]), normal_training
        )

    def test_ratios_are_applied_after_fixed_validation_split(self):
        normal_training = numpy.arange(100).reshape(100, 1)
        fit_pool, validation = validation_split(
            normal_training, val_fraction=0.1, min_train_length=1,
        )
        five_percent, _ = take_training_prefix(fit_pool, 0.05)
        full_fit, _ = take_training_prefix(fit_pool, 1.0)
        self.assertEqual((len(fit_pool), len(five_percent), len(full_fit)), (90, 5, 90))
        numpy.testing.assert_array_equal(validation[:, 0], numpy.arange(90, 100))

    def test_val_always_nonempty_by_ceil(self):
        train_part, val_part = validation_split(numpy.zeros((5, 3)), val_fraction=0.1, min_train_length=1)
        self.assertEqual(val_part.shape[0], 1)  # ceil(0.5) = 1
        self.assertEqual(train_part.shape[0], 4)

    def test_raises_with_full_context_when_below_minimum(self):
        with self.assertRaises(InsufficientTrainLengthError) as raised:
            validation_split(
                numpy.zeros((5, 3)),
                val_fraction=0.1,
                min_train_length=10,
                split_info={"T": 5, "ratio": "before_ratio"},
            )
        message = str(raised.exception)
        for expected_fragment in ("T=5", "ratio=before_ratio", "val_fraction=0.1",
                                  "정상 구간 길이=5", "fit pool 길이=4", "하한=10"):
            self.assertIn(expected_fragment, message)


if __name__ == "__main__":
    unittest.main()
