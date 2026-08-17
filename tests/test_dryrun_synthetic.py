"""3b-4 드라이런이 검증 실패를 정상 종료로 숨기지 않는지 확인한다."""

import contextlib
import io
import unittest

from experiments.exp00_gragod_recon.dryrun_synthetic import check


class TestDryrunCheck(unittest.TestCase):
    def test_failed_check_stops_the_dryrun(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "shape 검사"):
                check("shape 검사", False)


if __name__ == "__main__":
    unittest.main()
