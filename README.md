# Video indexing at speed

For every second of every FineVideo video in a folder of Parquet files, `run.py` reports which of
**person, car, dog, laptop, cell phone** appear (YOLO11l, imgsz 640, conf ≥ 0.4) and how many people
are visible. The output is one JSON line per second.

## Install (Ubuntu 22.04, Python 3.11, NVIDIA driver with CUDA 12)

```bash
sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0
```

```bash
python3.11 -m venv .venv && source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

`torch==2.7.1` from PyPI ships CUDA 12.6 runtime libraries, so only the driver is needed.

## Preparation command (run once, needs internet, not timed)

```bash
python prepare.py
```

This downloads the official `yolo11l.pt` into `weights/` and warms up the imports.

## Run (offline)

```bash
python run.py --input <folder of .parquet files> --output results.jsonl
```

Progress is printed to stderr every 30 s. Useful options (defaults are the submitted configuration):

| option | meaning |
|---|---|
| `--fps-samples K` | frames analysed per second of video (`0` = every frame) |
| `--workers N` | decoder processes (default: CPU count − 2) |
| `--batch B` | GPU batch size (default 64) |
| `--fp32` | disable FP16 |

## How it works

1. **Reader thread.** Streams rows out of each Parquet file one at a time with `pyarrow` (`mp4`
   column only). The video id is `<file name>:<row>`.
2. **Decoder processes** (one video each, a pool sized to the CPU). Each video is decoded with
   PyAV/FFmpeg. Timestamps are taken from `pts` relative to the first decoded frame, and the
   display-matrix rotation tag is applied. Only **K evenly spaced frames per second** are
   converted to BGR, resized and padded exactly like Ultralytics' `LetterBox` (rect mode, e.g.
   384×640 for 16:9).
3. **Shared-memory hand-off.** Letterboxed uint8 frames go to the main process as torch
   shared-memory tensors.
4. **GPU.** Frames from different videos with the same shape are batched. YOLO11l runs in FP16.
   Post-processing stays on the GPU and mirrors Ultralytics: best class per box, conf > 0.4,
   class-aware NMS at IoU 0.7, max 300 detections. Only per-frame counts of the five classes
   come back to the CPU.
5. **Aggregation** uses the grader's rules. An object is present if it is found in at least
   half of the analysed frames of that second. People is the median count, rounded down at .5.
   One line is written for every second up to the one holding the last frame.
6. **Damaged input** never stops the run. Unreadable Parquet files, missing or garbage bytes,
   videos that fail part-way and decoder crashes each produce an `error` line, and seconds
   already finished are kept.

## Checking correctness

- `tools/reference.py` is plain `ultralytics` `YOLO.predict` on **every** frame (the graders'
  method).
- `tools/evaluate.py --ref ref.jsonl --pred results.jsonl` reports object F1, exact people
  accuracy and within-1 people accuracy.

## Files

- `run.py`, `indexer/`: the pipeline.
- `prepare.py`: the weights download.
- `tools/`: the reference and the evaluator.
- `modal_app.py`, `modal_download.py`: how I ran it on Modal (not needed by graders).
- `results.jsonl.gz`: results for the 50 files.
- `HARDWARE.md`, `WRITEUP.md`, `ai_chats/`.
