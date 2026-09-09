"""실행 구간 RSS 표본과 수집 종료·실패 기록을 확인한다."""

import unittest
from threading import Event, current_thread, main_thread
from unittest.mock import patch

from src.common.measure_process_memory import start_process_memory_sampling


class TestProcessMemorySampling(unittest.TestCase):
    def test_samples_between_boundaries_and_preserves_the_stopped_report(self):
        sampled = Event()
        boundary_values = iter((10, 20))

        def read_memory():
            if current_thread() is main_thread():
                value = next(boundary_values)
            else:
                value = 90
                sampled.set()
            return {"rss_bytes": value, "process_lifetime_peak_bytes": 900, "reason": None}

        sampler = start_process_memory_sampling(read_memory, interval_seconds=.001)
        try:
            self.assertTrue(sampled.wait(2))
        finally:
            report = sampler.stop()
        self.assertEqual(report["cpu_rss_sampled_peak_bytes"], 90)
        self.assertGreaterEqual(report["cpu_rss_sample_count"], 3)
        self.assertEqual(report["cpu_rss_sample_count"], report["cpu_rss_sample_attempt_count"])
        self.assertEqual(report["cpu_rss_sample_interval_seconds"], .001)
        self.assertEqual(report["cpu_rss_sampling_status"], "sampled")
        self.assertEqual(report["cpu_rss_sampling_scope"], "current_process_rss_excluding_children")
        self.assertEqual(report["cpu_rss_peak_kind"], "sampled_lower_bound")
        self.assertTrue(report["cpu_rss_sampling_thread_stopped"])
        self.assertEqual(sampler.stop(), report)

    def test_reader_failures_do_not_replace_the_model_error(self):
        def read_memory():
            raise OSError("RSS unavailable")

        sampler = start_process_memory_sampling(read_memory)
        with self.assertRaisesRegex(RuntimeError, "model failed"):
            try:
                raise RuntimeError("model failed")
            finally:
                report = sampler.stop()
        self.assertIsNone(report["cpu_rss_sampled_peak_bytes"])
        self.assertEqual(report["cpu_rss_sample_count"], 0)
        self.assertGreaterEqual(report["cpu_rss_sample_attempt_count"], 2)
        self.assertEqual(report["cpu_rss_sampling_status"], "unavailable")
        self.assertEqual(report["cpu_rss_sampling_reason"], "no_valid_rss_samples")
        self.assertIn("rss_reader_error:OSError: RSS unavailable", report["cpu_rss_sampling_errors"])

    def test_blocked_reader_cannot_change_report_after_bounded_stop(self):
        entered, release = Event(), Event()

        def read_memory():
            if current_thread() is not main_thread():
                entered.set()
                release.wait(2)
                return {"rss_bytes": 900, "reason": None}
            return {"rss_bytes": 10, "reason": None}

        sampler = start_process_memory_sampling(read_memory, interval_seconds=.001)
        try:
            self.assertTrue(entered.wait(2))
            report = sampler.stop()
            self.assertEqual(report["cpu_rss_sampled_peak_bytes"], 10)
            self.assertFalse(report["cpu_rss_sampling_thread_stopped"])
            self.assertEqual(report["cpu_rss_sampling_status"], "partial")
            self.assertIn("sampling_thread_stop_timeout", report["cpu_rss_sampling_errors"])
        finally:
            release.set()
            sampler._thread.join(timeout=2)
            sampler.stop()
        self.assertEqual(report["cpu_rss_sampled_peak_bytes"], 10)
        self.assertEqual(report["cpu_rss_sample_count"], 2)

    def test_validates_interval_and_keeps_start_errors_and_interrupts_visible(self):
        def read_memory():
            return {"rss_bytes": 10, "reason": None}

        for interval in (0, -1, float("nan"), float("inf"), 1e100, True, "0.05"):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                start_process_memory_sampling(read_memory, interval_seconds=interval)
        with patch("src.common.measure_process_memory.Thread.start", side_effect=RuntimeError("no thread")):
            report = start_process_memory_sampling(read_memory).stop()
        self.assertEqual(report["cpu_rss_sampling_status"], "partial")
        self.assertIn("sampling_thread_start_error:RuntimeError: no thread", report["cpu_rss_sampling_errors"])
        for interruption in (KeyboardInterrupt, SystemExit):
            def interrupt_reader():
                raise interruption()

            with self.subTest(interruption=interruption), self.assertRaises(interruption):
                start_process_memory_sampling(interrupt_reader)


if __name__ == "__main__":
    unittest.main()
