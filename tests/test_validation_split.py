"""src/data_split/validation_split.py 단위 테스트."""

import unittest

import numpy

from src.data_split.front_trim_split import front_trim_split
from src.data_split.validation_split import InsufficientTrainLengthError, validation_split


class TestValidationSplit(unittest.TestCase):
    def test_tail_becomes_validation_no_shuffle(self):
        # 20행 축소 구간, val_fraction=0.1 → val=ceil(2)=2, train=18. 뒤쪽이 validation.
        time_index, channel_index = numpy.meshgrid(range(20), range(3), indexing="ij")
        reduced = time_index * 10 + channel_index
        train_part, val_part = validation_split(reduced, val_fraction=0.1, min_train_length=1)
        self.assertEqual(train_part.shape, (18, 3))
        self.assertEqual(val_part.shape, (2, 3))
        numpy.testing.assert_array_equal(val_part[:, 0], [180, 190])
        # 셔플 없음: train+val 이어붙이면 원 배열 그대로 (시간 순서 보존).
        numpy.testing.assert_array_equal(numpy.concatenate([train_part, val_part]), reduced)

    def test_val_always_nonempty_by_ceil(self):
        train_part, val_part = validation_split(numpy.zeros((5, 3)), val_fraction=0.1, min_train_length=1)
        self.assertEqual(val_part.shape[0], 1)  # ceil(0.5) = 1
        self.assertEqual(train_part.shape[0], 4)

    def test_raises_with_full_context_when_below_minimum(self):
        reduced, split_info = front_trim_split(numpy.zeros((100, 3)), 0.05)
        with self.assertRaises(InsufficientTrainLengthError) as raised:
            validation_split(reduced, val_fraction=0.1, min_train_length=10, split_info=split_info)
        message = str(raised.exception)
        for expected_fragment in ("T=100", "ratio=0.05", "val_fraction=0.1",
                                  "축소 구간 길이=5", "train 길이=4", "하한=10"):
            self.assertIn(expected_fragment, message)


if __name__ == "__main__":
    unittest.main()
