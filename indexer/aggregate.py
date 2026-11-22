"""Per-second aggregation, following the grader's rules (section 5 of the brief)."""
from __future__ import annotations

import numpy as np

from .infer import CLASS_NAMES


def median_floor(values: np.ndarray) -> int:
    v = np.sort(values)
    n = len(v)
    if n == 0:
        return 0
    if n % 2:
        return int(v[n // 2])
    return int((int(v[n // 2 - 1]) + int(v[n // 2])) // 2)


def video_lines(video: str, secs: np.ndarray, counts: np.ndarray, last_sec: int):
    """secs: (F,) second of each analysed frame; counts: (F, 5) per-frame class counts.

    Yields one dict per second 0..last_sec.
    """
    if last_sec < 0:
        return
    order = np.argsort(secs, kind="stable")
    secs, counts = secs[order], counts[order]
    bounds = np.searchsorted(secs, np.arange(last_sec + 2))
    for s in range(last_sec + 1):
        a, b = bounds[s], bounds[s + 1]
        n = b - a
        if n == 0:
            yield {"video": video, "second": s, "objects": [], "people": 0}
            continue
        c = counts[a:b]
        present = (c > 0).sum(0) * 2 >= n
        objs = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES)) if present[i]]
        yield {"video": video, "second": s, "objects": objs, "people": median_floor(c[:, 0])}
