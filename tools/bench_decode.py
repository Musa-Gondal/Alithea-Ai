"""Decode-speed check for one video: raw decode vs. our sampler."""
import glob
import io
import sys
import time

import av
import pyarrow.parquet as pq

sys.path.insert(0, ".")
from indexer.decode import iter_sampled_frames  # noqa: E402

f = sorted(glob.glob("/data/finevideo/data/*.parquet"))[10]
b = pq.read_table(f, columns=["mp4"]).column(0)[26].as_py()
for tt in ("AUTO", "SLICE", "FRAME"):
    for tc in (1, 2, 4):
        c = av.open(io.BytesIO(b))
        s = c.streams.video[0]
        s.thread_type = tt
        s.thread_count = tc
        t = time.time()
        n = 0
        for fr in c.decode(s):
            n += 1
            if n >= 3000:
                break
        print(tt, tc, f"{n / (time.time() - t):.0f} fps raw decode", flush=True)
for kw in ({}, {"fast_resize": True}, {"skip_nonref": True}):
    t = time.time()
    n = 0
    g = iter_sampled_frames(b, 3, 2, **kw)
    for s_, img in g:
        n += 1
        if n >= 300:
            break
    print(kw, f"{100 / (time.time() - t):.1f} s-of-video/s", flush=True)
