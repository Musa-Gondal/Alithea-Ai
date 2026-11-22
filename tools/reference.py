#!/usr/bin/env python
"""Reference results the way the graders describe them: plain Ultralytics YOLO11l on EVERY frame.

    python tools/reference.py --input DIR --output ref.jsonl [--max-videos-per-file N] [--limit-files N]

Uses the stock `YOLO.predict` (FP32, imgsz=640, conf=0.4, default NMS) so it shares no inference
code with run.py; only decoding/rotation (PyAV) is shared.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indexer.aggregate import video_lines  # noqa: E402
from indexer.decode import DecodeError, iter_sampled_frames  # noqa: E402

COCO = {0: 0, 2: 1, 16: 2, 63: 3, 67: 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--weights", default=str(Path(__file__).resolve().parents[1] / "weights" / "yolo11l.pt"))
    ap.add_argument("--max-videos-per-file", type=int, default=0)
    ap.add_argument("--skip-files", type=int, default=0)
    ap.add_argument("--limit-files", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from ultralytics import YOLO

    model = YOLO(args.weights)
    files = sorted(Path(args.input).rglob("*.parquet"))[args.skip_files:]
    if args.limit_files:
        files = files[: args.limit_files]
    out = open(args.output, "w")
    t0 = time.time()
    nfr = 0
    for path in files:
        try:
            pf = pq.ParquetFile(str(path))
        except Exception:
            continue
        row = 0
        for batch in pf.iter_batches(batch_size=1, columns=["mp4"]):
            if args.max_videos_per_file and row >= args.max_videos_per_file:
                break
            vid = f"{path.name}:{row}"
            row += 1
            data = batch.column(0)[0].as_py()
            if not data:
                continue
            secs, counts, buf, bsecs = [], [], [], []
            error = None
            last = -1

            def flush():
                nonlocal buf, bsecs
                if not buf:
                    return
                res = model.predict(buf, imgsz=640, conf=0.4, verbose=False)
                for r, s in zip(res, bsecs):
                    c = np.zeros(5, np.int32)
                    for k in r.boxes.cls.int().tolist():
                        if k in COCO:
                            c[COCO[k]] += 1
                    counts.append(c)
                    secs.append(s)
                buf, bsecs = [], []

            try:
                gen = iter_sampled_frames(data, 0, raw=True)
                while True:
                    try:
                        s, img = next(gen)
                    except StopIteration as st:
                        last = st.value
                        break
                    buf.append(img)
                    bsecs.append(s)
                    nfr += 1
                    if len(buf) >= args.batch:
                        flush()
                flush()
            except DecodeError as e:
                error = str(e)
            except Exception as e:
                flush()
                error = f"decode failed: {type(e).__name__}"
                last = (max(secs) - 1) if secs else -1
            if counts:
                for d in video_lines(vid, np.asarray(secs, np.int32), np.stack(counts), last):
                    out.write(json.dumps(d) + "\n")
            if error:
                out.write(json.dumps({"video": vid, "error": error}) + "\n")
            out.flush()
            print(f"[ref] {vid} last_sec={last} frames={nfr} {nfr / (time.time() - t0):.0f} fps",
                  file=sys.stderr, flush=True)
    out.close()


if __name__ == "__main__":
    main()
