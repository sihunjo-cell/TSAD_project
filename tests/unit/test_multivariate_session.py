"""ALoRa와 GDN이 공유하는 다변량 세션 입력 계약."""

import unittest

import numpy


class TestMultivariateSession(unittest.TestCase):
    def test_converts_valid_values_and_rejects_shape_or_nonfinite_values(self):
        try:
            from src.common.validate_multivariate_session import (
                as_finite_multivariate_session,
            )
        except ModuleNotFoundError:
            self.fail("shared multivariate session validator is missing")

        converted = as_finite_multivariate_session([[1, 2], [3, 4]], "session")
        self.assertEqual(converted.dtype, numpy.float32)
        numpy.testing.assert_array_equal(converted, [[1, 2], [3, 4]])

        invalid_values = (
            ([1, 2], "shape"),
            ([[1], [2]], "shape"),
            ([[1, numpy.inf]], "non-finite"),
        )
        for values, message in invalid_values:
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, message):
                    as_finite_multivariate_session(values, "session")


if __name__ == "__main__":
    unittest.main()
