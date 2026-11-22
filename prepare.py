#!/usr/bin/env python
"""Preparation step (run once, with internet, before the timed run): fetch yolo11l.pt into ./weights."""
from __future__ import annotations

import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt"
DEST = Path(__file__).resolve().parent / "weights" / "yolo11l.pt"


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if not DEST.exists() or DEST.stat().st_size < 1_000_000:
        print(f"downloading {URL}")
        tmp = DEST.with_suffix(".part")
        with urllib.request.urlopen(URL) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.rename(DEST)
    print(f"weights: {DEST} ({DEST.stat().st_size / 1e6:.1f} MB, "
          f"sha256 {hashlib.sha256(DEST.read_bytes()).hexdigest()[:16]})")
    # warm import so the timed run doesn't pay first-import costs (pyc compilation)
    import torch  # noqa: F401
    import ultralytics  # noqa: F401
    from ultralytics import YOLO

    YOLO(str(DEST))
    print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())

    # TensorRT FP16 engine for this GPU (optional: run.py falls back to PyTorch without it)
    if torch.cuda.is_available() and "--no-trt" not in sys.argv:
        sys.path.insert(0, str(DEST.parent.parent))
        from indexer.trt import build_engine

        engine = DEST.parent / "yolo11l_fp16.engine"
        try:
            if not engine.exists():
                print("building TensorRT engine (a few minutes)...", flush=True)
                build_engine(str(DEST), str(engine))
            print(f"engine: {engine} ({engine.stat().st_size / 1e6:.1f} MB)")
        except Exception as e:
            print(f"TensorRT engine build failed ({type(e).__name__}: {e}); run.py will use PyTorch")
            if engine.exists():
                engine.unlink()


if __name__ == "__main__":
    sys.exit(main())
