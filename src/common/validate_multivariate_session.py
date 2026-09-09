"""다변량 세션 입력 검증."""

import numpy


def as_finite_multivariate_session(values, name: str) -> numpy.ndarray:
    session = numpy.asarray(values, dtype=numpy.float32)
    if session.ndim != 2 or session.shape[1] < 2:
        raise ValueError(f"{name} must have shape (time, at least two channels)")
    if not numpy.isfinite(session).all():
        raise ValueError(f"{name} contains non-finite values")
    return session
