"""src/data_split/back_trim_split.py 단위 테스트."""

import unittest

import numpy

from src.data_split.back_trim_split import back_trim_split
from src.data_split.front_trim_split import ALLOWED_RATIO_FRACTIONS, front_trim_split


class TestBackTrimSplit(unittest.TestCase):
    def test_kept_length_always_equals_front(self):
        # 통제 비교 성립 조건: 같은 T·p에서 두 방향의 kept_length 가 항상 같다 (계획서 7-1).
        for total_length in (1, 7, 19, 20, 100, 101):
            train_array = numpy.zeros((total_length, 3))
            for ratio in ALLOWED_RATIO_FRACTIONS:
                _, front_info = front_trim_split(train_array, ratio)
                _, back_info = back_trim_split(train_array, ratio)
                self.assertEqual(front_info["kept_length"], back_info["kept_length"],
                                 msg=f"T={total_length}, ratio={ratio}")

    def test_takes_from_the_end(self):
        time_index, channel_index = numpy.meshgrid(range(20), range(3), indexing="ij")
        train_array = time_index * 10 + channel_index
        kept, split_info = back_trim_split(train_array, 0.50)
        self.assertEqual(split_info["kept_index_range"], (10, 20))
        numpy.testing.assert_array_equal(kept[0], [100, 101, 102])
        numpy.testing.assert_array_equal(kept[-1], [190, 191, 192])

    def test_rejects_disallowed_ratios(self):
        with self.assertRaises(ValueError):
            back_trim_split(numpy.zeros((20, 3)), 0.3)


if __name__ == "__main__":
    unittest.main()
