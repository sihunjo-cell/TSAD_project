"""Lightning Dev18 진입점의 가속기 차단 계약을 검증한다."""

import unittest


class TestLightningDev18(unittest.TestCase):
    def _require_lightning_cuda(self):
        try:
            from tests.checks.run_lightning_dev18 import require_lightning_cuda
        except ImportError as error:
            self.fail(f"Lightning 진입점이 없다: {error}")
        return require_lightning_cuda

    def test_rejects_non_linux_runtime(self):
        require_lightning_cuda = self._require_lightning_cuda()
        with self.assertRaisesRegex(RuntimeError, "Linux"):
            require_lightning_cuda(system_name="Windows", cuda_available=True)

    def test_rejects_cpu_studio_before_runtime_is_sealed(self):
        require_lightning_cuda = self._require_lightning_cuda()
        with self.assertRaisesRegex(RuntimeError, "GPU"):
            require_lightning_cuda(system_name="Linux", cuda_available=False)

    def test_accepts_linux_cuda_runtime(self):
        require_lightning_cuda = self._require_lightning_cuda()
        self.assertIsNone(
            require_lightning_cuda(system_name="Linux", cuda_available=True),
        )

    def test_determinism_is_set_before_runtime_identity_is_sealed(self):
        try:
            from tests.checks.run_lightning_dev18 import seal_lightning_runtime
        except ImportError as error:
            self.fail(f"Lightning runtime 봉인 순서가 없다: {error}")
        events = []

        def set_seed(seed):
            events.append(("seed", seed))

        def collect_identity():
            events.append(("identity", None))
            return {"torch": {"deterministic_algorithms": True}}

        def ensure_snapshot(*, environment):
            events.append(("snapshot", environment))
            return {"sha256": "a" * 64}

        evidence = seal_lightning_runtime(
            set_reproducible_seed=set_seed,
            collect_environment_identity=collect_identity,
            ensure_runtime_snapshot=ensure_snapshot,
        )

        self.assertEqual([event[0] for event in events], ["seed", "identity", "snapshot"])
        self.assertEqual(events[0][1], 0)
        self.assertEqual(evidence, {"sha256": "a" * 64})


if __name__ == "__main__":
    unittest.main()
