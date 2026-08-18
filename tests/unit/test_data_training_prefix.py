"""src/data_split/take_training_prefix.py 단위 테스트."""

import unittest

import numpy

from src.data_split.take_training_prefix import take_training_prefix


class TestTakeTrainingPrefix(unittest.TestCase):
    def test_kept_lengths_for_T20_all_ratios(self):
        # 손검산: p=0.05→ceil(1)=1, 0.10→2, 0.20→4, 0.50→10, 1.00→20.
        train_array = numpy.zeros((20, 3))
        expected_by_ratio = {0.05: 1, 0.10: 2, 0.20: 4, 0.50: 10, 1.00: 20}
        for ratio, expected_length in expected_by_ratio.items():
            kept, split_info = take_training_prefix(train_array, ratio)
            self.assertEqual(kept.shape[0], expected_length)
            self.assertEqual(split_info["kept_length"], expected_length)
            self.assertEqual(split_info["kept_index_range"], (0, expected_length))
            self.assertEqual(split_info["T"], 20)

    def test_ceil_boundary_T19(self):
        # ceil(0.05*19) = ceil(0.95) = 1.
        kept, split_info = take_training_prefix(numpy.zeros((19, 3)), 0.05)
        self.assertEqual(split_info["kept_length"], 1)

    def test_2d_time_axis_only_channels_preserved(self):
        # 값 = 시점*10 + 채널 — 시간 축만 줄고 채널 축·값은 보존한다.
        time_index, channel_index = numpy.meshgrid(range(20), range(3), indexing="ij")
        train_array = time_index * 10 + channel_index
        kept, _ = take_training_prefix(train_array, 0.20)
        self.assertEqual(kept.shape, (4, 3))
        numpy.testing.assert_array_equal(kept[0], [0, 1, 2])
        numpy.testing.assert_array_equal(kept[-1], [30, 31, 32])

    def test_rejects_disallowed_ratios(self):
        for bad_ratio in (0.3, 0.15, 0.0, 1.5):
            with self.assertRaises(ValueError):
                take_training_prefix(numpy.zeros((20, 3)), bad_ratio)


if __name__ == "__main__":
    unittest.main()
