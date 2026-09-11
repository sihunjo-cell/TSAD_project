"""Lightning에서 새 프로세스의 BLAS 초기화와 PCA 스레드 복원을 확인한다."""

import os
import subprocess
import sys
import unittest
from pathlib import Path


class TestBlasStartup(unittest.TestCase):
    def test_fresh_runner_can_expand_blas_and_restore_serial_execution(self):
        probe = r'''
import resource
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
from tests.ghl_main import run_ratio_tuning
from concurrent.futures import ProcessPoolExecutor
import numpy
from sklearn.decomposition import PCA
from src.models.tier1.pca_legacy import PCA_FIT_BLAS_THREADS
from threadpoolctl import threadpool_info, threadpool_limits

def require_serial():
    libraries = [row for row in threadpool_info() if row["user_api"] == "blas"]
    assert libraries and all(row["num_threads"] == 1 for row in libraries), libraries

require_serial()
matrix = numpy.random.default_rng(0).normal(size=(2048, 1024))
reference = PCA(n_components=0.75, svd_solver="full").fit(matrix)
with threadpool_limits(limits=PCA_FIT_BLAS_THREADS, user_api="blas"):
    actual = PCA(n_components=0.75, svd_solver="full").fit(matrix)
    numpy.testing.assert_allclose(
        actual.explained_variance_ratio_, reference.explained_variance_ratio_,
        rtol=1e-10, atol=1e-14,
    )
    assert actual.n_components_ == reference.n_components_
require_serial()
with ProcessPoolExecutor(max_workers=2) as executor:
    pending = [executor.submit(threadpool_info) for _ in range(2)]
    for future in pending:
        libraries = [row for row in future.result() if row["user_api"] == "blas"]
        assert libraries and all(row["num_threads"] == 1 for row in libraries), libraries
print(f"BLAS 시작·1→{PCA_FIT_BLAS_THREADS}→1 복원·채점 worker·PCA 대조 PASS", flush=True)
'''
        environment = os.environ.copy()
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[name] = "1"
        result = subprocess.run(
            [sys.executable, "-X", "faulthandler", "-u", "-c", probe],
            cwd=Path(__file__).resolve().parents[2], env=environment,
            capture_output=True, text=True, timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr[-6000:])
        self.assertIn("PCA 대조 PASS", result.stdout)
        print(result.stdout.strip())


if __name__ == "__main__":
    unittest.main()
