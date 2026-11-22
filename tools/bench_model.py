"""Micro-benchmark of YOLO11l forward throughput on the GPU for a few settings."""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from indexer.infer import load_model  # noqa: E402

w = os.environ.get("YOLO_WEIGHTS", "weights/yolo11l.pt")
dev = torch.device("cuda")
torch.backends.cudnn.benchmark = True


def bench(model, bs, cl, iters=30):
    x = torch.rand(bs, 3, 384, 640, device=dev, dtype=torch.float16)
    if cl:
        x = x.contiguous(memory_format=torch.channels_last)
    with torch.inference_mode():
        for _ in range(5):
            model(x)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(iters):
            model(x)
        torch.cuda.synchronize()
    return bs * iters / (time.time() - t)


m = load_model(w, dev, True)
for bs in (32, 64, 128):
    print("nchw", bs, f"{bench(m, bs, False):.0f} img/s", flush=True)
m = m.to(memory_format=torch.channels_last)
for bs in (32, 64, 128):
    print("channels_last", bs, f"{bench(m, bs, True):.0f} img/s", flush=True)
try:
    mc = torch.compile(m, mode="max-autotune-no-cudagraphs", dynamic=False)
    print("compiled cl 64", f"{bench(mc, 64, True):.0f} img/s", flush=True)
except Exception as e:
    print("compile failed", e)
