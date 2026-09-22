"""현재 프로세스의 실행 구간 RSS를 표본으로 측정한다."""

from threading import TIMEOUT_MAX, Event, Lock, Thread


def start_process_memory_sampling(read_memory, *, interval_seconds=0.05):
    """RSS reader를 즉시 호출하고 주기 수집을 시작한다. 종료 시 stop()을 호출한다."""
    if (type(interval_seconds) not in (int, float)
            or not 0 < interval_seconds <= TIMEOUT_MAX):
        raise ValueError("RSS 수집 간격은 thread 대기 한도 이내의 유한한 양수여야 한다")
    if not callable(read_memory):
        raise ValueError("RSS reader는 호출 가능해야 한다")
    return _ProcessMemorySampler(read_memory, float(interval_seconds))


class _ProcessMemorySampler:
    def __init__(self, read_memory, interval_seconds):
        self._read_memory = read_memory
        self._interval_seconds = interval_seconds
        self._stop = Event()
        self._lock = Lock()
        self._closed = False
        self._started = False
        self._report = None
        self._peak_bytes = None
        self._sample_count = 0
        self._attempt_count = 0
        self._errors = {}
        self._thread = Thread(target=self._collect, name="process-rss-sampler", daemon=True)
        self._sample()
        try:
            self._thread.start()
            self._started = True
        except Exception as error:
            self._record_error(f"sampling_thread_start_error:{type(error).__name__}: {error}")
        except BaseException:
            self._stop.set()
            raise

    def _record_error(self, reason):
        self._errors[reason] = self._errors.get(reason, 0) + 1

    def _sample(self, *, final=False):
        with self._lock:
            if self._closed and not final:
                return
            self._attempt_count += 1
        try:
            reading = self._read_memory()
            value = reading["rss_bytes"]
            reason = reading.get("reason")
            if reason is None and (type(value) is not int or value < 0):
                reason = "invalid_rss_bytes"
        except Exception as error:
            value = None
            reason = f"rss_reader_error:{type(error).__name__}: {error}"
        with self._lock:
            if self._closed and not final:
                return
            if reason is not None:
                self._record_error(str(reason))
            else:
                self._sample_count += 1
                self._peak_bytes = value if self._peak_bytes is None else max(self._peak_bytes, value)

    def _collect(self):
        while not self._stop.wait(self._interval_seconds):
            self._sample()

    def stop(self):
        """수집을 닫고 고정된 결과를 반환한다. 일반 계측 오류는 결과에 기록한다."""
        if self._report is not None:
            return self._report
        self._stop.set()
        try:
            if self._started:
                self._thread.join(timeout=0.2)
                if self._thread.is_alive():
                    with self._lock:
                        self._record_error("sampling_thread_stop_timeout")
        except Exception as error:
            with self._lock:
                self._record_error(f"sampling_thread_stop_error:{type(error).__name__}: {error}")
        finally:
            with self._lock:
                self._closed = True
        self._sample(final=True)
        with self._lock:
            status = "unavailable" if not self._sample_count else "partial" if self._errors else "sampled"
            self._report = {
                "cpu_rss_sampled_peak_bytes": self._peak_bytes,
                "cpu_rss_sample_interval_seconds": self._interval_seconds,
                "cpu_rss_sample_count": self._sample_count,
                "cpu_rss_sample_attempt_count": self._attempt_count,
                "cpu_rss_sampling_status": status,
                "cpu_rss_sampling_reason": ("no_valid_rss_samples" if not self._sample_count
                                            else "sampling_incomplete" if self._errors else None),
                "cpu_rss_sampling_errors": dict(self._errors),
                "cpu_rss_sampling_scope": "current_process_rss_excluding_children",
                "cpu_rss_peak_kind": "sampled_lower_bound",
                "cpu_rss_sampling_thread_stopped": not self._thread.is_alive(),
            }
        return self._report
