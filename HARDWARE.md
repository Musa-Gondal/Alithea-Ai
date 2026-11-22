# Hardware used for the full 50-file run

**Where:** a rented cloud GPU on [Modal](https://modal.com). It was a serverless container started with
`modal_app.py` (`gpu="L40S", cpu=16, memory=65536`). The timed run had networking blocked, like the
graders' run.

| | |
|---|---|
| GPU | NVIDIA L40S, 46 GB (46068 MiB), driver 620.70 |
| CPU | 16 vCPUs reserved. The CPU model is not exposed inside Modal's sandbox (`lscpu` shows "unknown"). |
| RAM | 64 GB limit for the container |
| Disk | Modal Volume (network-attached, SSD-backed) holding the 50 Parquet files (about 25 GB) |
| OS | Debian 12 (bookworm) container image, Python 3.11, PyTorch 2.7.1 + CUDA 12.6, TensorRT 10.13 |

## Full-run time (start to finish)

The final version took **906 s (15 min 6 s)** for all 50 files:
- 1,650 videos;
- 461,757 output lines;
- 1,344,118 frames analysed;
- 0 errors.

It was run with:

```bash
python run.py --input /data/finevideo/data --output results.jsonl
```

The TensorRT engine was built beforehand by `prepare.py`, which is untimed.

Other full-run times on the same machine:

| version | full 50 files |
|---|---|
| PyTorch FP16 (no TensorRT), shared-memory ring buffer | 1,574 s |
| TensorRT FP16, one inference thread | 960 s |
| **final: TensorRT FP16, 2 inference threads, pinned staging buffers** | **906 s** |
