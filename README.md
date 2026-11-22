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

`torch==2.7.1` from PyPI ships CUDA 12.6 runtime libraries and `tensorrt-cu12` ships TensorRT 10.13,
so only the NVIDIA driver is needed from the system.

## Preparation command (run once, needs internet, not timed)

```bash
python prepare.py
```

This does three things:
- downloads the official `yolo11l.pt` into `weights/`;
- exports it to ONNX;
- builds a **TensorRT FP16 engine for the local GPU** (`weights/yolo11l_fp16.engine`, about
  5 minutes). The engine has a dynamic batch size of 1–32 and dynamic image sizes up to 640×640.

If the engine build fails for any reason, `run.py` automatically uses the PyTorch FP16 model
instead. That is slower but gives the same results. `run.py` never needs the internet.

## Run (offline)

```bash
python run.py --input <folder of .parquet files> --output results.jsonl
```

Progress is printed to stderr every 30 s. Useful options (defaults are the submitted configuration):

| option | meaning |
|---|---|
| `--fps-samples K` | frames analysed per second of video (`0` = every frame) |
| `--workers N` | decoder processes (default: CPU count − 2) |
| `--batch B` | frames collected before a GPU call (default 32) |
| `--no-trt` | ignore the TensorRT engine and use PyTorch FP16 |
| `--infer-threads T` | GPU inference threads, each with its own CUDA stream (default 2) |
| `--fp32` | PyTorch FP32 (slow; for checks) |

## How it works

1. **Reader thread.** Reads each Parquet row group with `pyarrow` (`mp4` column only). The video
   id is `<file name>:<row>`.
2. **Decoder processes** (one video each, a pool sized to the CPU). Each video is decoded with
   PyAV/FFmpeg. Non-reference frames are skipped by the decoder (`skip_frame=NONREF`), but
   packet timestamps still give the true last second. Timestamps come from `pts` relative to the
   first decoded frame, and the display-matrix rotation tag is applied. Only **3 evenly spaced
   frames per second** are converted. swscale does the YUV→BGR conversion and the bilinear
   resize in one pass, and the frame is then padded like Ultralytics' `LetterBox` (rect mode,
   e.g. 384×640 for 16:9).
3. **Shared-memory hand-off.** Decoders write letterboxed uint8 frames into a ring of
   fixed-size slots in one `multiprocessing.shared_memory` block. Queues only carry slot numbers.
   The Parquet reader hands out the biggest videos of each file first, so long videos don't
   finish last.
4. **GPU.** Frames from different videos with the same shape are batched. Batches are split or
   padded to fixed sizes of 4, 8, 16 or 32, so cuDNN/TensorRT never retune for odd sizes. Two
   inference threads, each with its own CUDA stream, run YOLO11l (TensorRT FP16, or PyTorch FP16
   channels-last as the fallback).
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
- `tools/bench_model.py` and `tools/bench_decode.py` are micro-benchmarks for the GPU and the
  decoder.

## Files

- `run.py`, `indexer/`: the pipeline.
- `prepare.py`: the weights download.
- `tools/`: the reference and the evaluator.
- `modal_app.py`, `modal_download.py`: how I ran it on Modal (not needed by graders).
- `results.jsonl.gz`: results for the 50 files.
- `HARDWARE.md`, `WRITEUP.md`, `ai_chats/`.
