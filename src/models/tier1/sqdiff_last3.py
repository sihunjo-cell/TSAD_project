"""Existing Last-3 entrypoint and its internal rolling window."""

from .sqdiff import score_sqdiff_last3


SQDIFF_WINDOW = 4


__all__ = ["SQDIFF_WINDOW", "score_sqdiff_last3"]
