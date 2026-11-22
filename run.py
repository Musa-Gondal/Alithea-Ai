#!/usr/bin/env python
"""Index FineVideo parquet files: per second, which of five objects appear and how many people.

    python run.py --input <folder of .parquet files> --output results.jsonl

Pipeline: a reader thread streams videos (mp4 bytes) out of the parquet files -> a pool of
decoder processes decode each video with PyAV, keep K evenly spaced frames per second, rotate
and letterbox them -> frames travel to the main process through shared memory -> the main
process batches them by shape and runs YOLO11l (FP16) on the GPU -> per-second aggregation ->
JSON lines.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DEFAULT_WEIGHTS = HERE / "weights" / "yolo11l.pt"


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def available_cpus() -> int:
    n = os.cpu_count() or 1
    try:
        n = min(n, len(os.sched_getaffinity(0)))
    except Exception:
        pass
    try:  # cgroup v2 quota (containers)
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()
        if q != "max":
            n = min(n, max(1, int(int(q) / int(p))))
    except Exception:
        pass
    return max(1, n)


def list_parquets(folder: Path):
    files = sorted(p for p in folder.rglob("*.parquet") if p.is_file())
    return files


# --------------------------------------------------------------------------------------------
# decoder worker process
# --------------------------------------------------------------------------------------------
def decode_worker(wid, task_q, out_q, k, chunk, decode_threads, shutdown):
    import torch

    torch.set_num_threads(1)
    torch.multiprocessing.set_sharing_strategy("file_system")
    from indexer.decode import DecodeError, iter_sampled_frames

    while True:
        task = task_q.get()
        if task is None:
            break
        vid, data = task
        out_q.put(("start", wid, vid))
        secs, frames = [], []
        n_sent = 0
        error = None
        last_sec = -1

        def flush():
            nonlocal secs, frames, n_sent
            if frames:
                t = torch.from_numpy(np.stack(frames))
                out_q.put(("frames", vid, np.asarray(secs, dtype=np.int32), t))
                n_sent += len(frames)
                secs, frames = [], []

        try:
            gen = iter_sampled_frames(data, k, decode_threads)
            shape = None
            while True:
                try:
                    s, img = next(gen)
                except StopIteration as stop:
                    last_sec = stop.value if stop.value is not None else -1
                    break
                if shape is not None and img.shape != shape:
                    flush()  # keep every chunk single-shaped
                shape = img.shape
                secs.append(s)
                frames.append(img)
                last_sec = max(last_sec, s)
                if len(frames) >= chunk:
                    flush()
            flush()
            if last_sec < 0:
                error = "no frames could be decoded"
        except DecodeError as e:
            flush()
            error = str(e)
        except Exception as e:  # damaged stream part-way, OOM, ...
            try:
                flush()
            except Exception:
                pass
            error = f"decode failed: {type(e).__name__}"
            # keep only seconds that were fully decoded
            last_sec = last_sec - 1 if last_sec >= 0 else -1
        del data
        out_q.put(("end", vid, int(last_sec), n_sent, error))
    # shared-memory tensors die with their producer: stay alive until the main process is done
    out_q.put(("bye", wid))
    shutdown.wait()


# --------------------------------------------------------------------------------------------
# reader thread: parquet -> (video id, mp4 bytes)
# --------------------------------------------------------------------------------------------
def reader_thread(files, task_q, emit_error, n_workers, stop_evt, stats, max_per_file=0):
    import pyarrow.parquet as pq

    for path in files:
        name = path.name
        row = 0
        try:
            pf = pq.ParquetFile(str(path))
            for batch in pf.iter_batches(batch_size=1, columns=["mp4"]):
                if max_per_file and row >= max_per_file:
                    break
                col = batch.column(0)
                for i in range(len(col)):
                    if max_per_file and row >= max_per_file:
                        break
                    vid = f"{name}:{row}"
                    row += 1
                    v = col[i]
                    if not v.is_valid:
                        emit_error(vid, "missing mp4 bytes")
                        continue
                    data = v.as_py()
                    if not data:
                        emit_error(vid, "empty mp4 bytes")
                        continue
                    stats["videos"] += 1
                    task_q.put((vid, data))
                    if stop_evt.is_set():
                        return
        except Exception as e:
            emit_error(f"{name}:{row}", f"could not read parquet: {type(e).__name__}")
    for _ in range(n_workers):
        task_q.put(None)


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--weights", default=os.environ.get("YOLO_WEIGHTS", str(DEFAULT_WEIGHTS)))
    ap.add_argument("--fps-samples", type=int, default=int(os.environ.get("SAMPLES_PER_SEC", 3)),
                    help="frames analysed per second of video (<=0: every frame)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--decode-threads", type=int, default=2)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-files", type=int, default=0)
    ap.add_argument("--skip-files", type=int, default=0)
    ap.add_argument("--max-videos-per-file", type=int, default=0)
    args = ap.parse_args()

    t_start = time.time()
    import torch
    import torch.multiprocessing as mp

    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    except Exception:
        pass
    mp.set_sharing_strategy("file_system")

    files = list_parquets(Path(args.input))[args.skip_files:]
    if args.limit_files:
        files = files[: args.limit_files]
    n_workers = args.workers or max(1, available_cpus() - 2)
    print(f"[run] {len(files)} parquet files, {n_workers} decoder workers, "
          f"{args.fps_samples} frames/s", file=sys.stderr, flush=True)

    out_f = open(args.output, "w", buffering=1 << 20)
    out_lock = threading.Lock()

    def write(obj):
        with out_lock:
            out_f.write(json.dumps(obj) + "\n")

    def emit_error(vid, msg):
        write({"video": vid, "error": msg})

    ctx = mp.get_context("spawn")
    shutdown = ctx.Event()
    task_q = ctx.Queue(maxsize=max(2, n_workers))
    out_q = ctx.Queue(maxsize=max(64, 4 * n_workers))

    def spawn(wid):
        p = ctx.Process(target=decode_worker,
                        args=(wid, task_q, out_q, args.fps_samples, args.chunk, args.decode_threads,
                              shutdown),
                        daemon=True)
        p.start()
        return p

    procs = {wid: spawn(wid) for wid in range(n_workers)}

    # model loads while workers start up
    from indexer.aggregate import video_lines
    from indexer.infer import Detector

    det = Detector(args.weights, device=args.device, half=not args.fp32)

    stats = defaultdict(int)
    stop_evt = threading.Event()
    rt = threading.Thread(target=reader_thread,
                          args=(files, task_q, emit_error, n_workers, stop_evt, stats,
                                args.max_videos_per_file), daemon=True)
    rt.start()

    # per-video state
    vsecs = defaultdict(list)        # vid -> list of np arrays (seconds)
    vcounts = defaultdict(list)      # vid -> list of np arrays (counts)
    vdone_frames = defaultdict(int)  # frames inferred so far
    vend = {}                        # vid -> (last_sec, n_sent, error)
    worker_vid = {}                  # wid -> vid currently decoding
    pending = defaultdict(list)      # shape -> list of (vid, secs, tensor)
    pending_n = defaultdict(int)
    byes = set()
    t_gpu = 0.0
    n_frames = 0

    def finalize(vid):
        last_sec, n_sent, error = vend.pop(vid)
        if vsecs.get(vid):
            secs = np.concatenate(vsecs.pop(vid))
            counts = np.concatenate(vcounts.pop(vid))
        else:
            vsecs.pop(vid, None)
            vcounts.pop(vid, None)
            secs = np.zeros(0, np.int32)
            counts = np.zeros((0, 5), np.int32)
        vdone_frames.pop(vid, None)
        lines = [json.dumps(d) for d in video_lines(vid, secs, counts, last_sec)]
        if error:
            lines.append(json.dumps({"video": vid, "error": error}))
        if lines:
            with out_lock:
                out_f.write("\n".join(lines) + "\n")
        stats["done"] += 1

    def run_batch(shape):
        nonlocal t_gpu, n_frames
        items = pending.pop(shape)
        pending_n.pop(shape, None)
        tensors = [t for _, _, t in items]
        x = tensors[0] if len(tensors) == 1 else torch.cat(tensors)
        t0 = time.time()
        counts = det(x).numpy()
        t_gpu += time.time() - t0
        n_frames += len(x)
        off = 0
        touched = set()
        for vid, secs, t in items:
            n = len(t)
            vsecs[vid].append(secs)
            vcounts[vid].append(counts[off:off + n])
            vdone_frames[vid] += n
            off += n
            touched.add(vid)
        for vid in touched:
            if vid in vend and vdone_frames[vid] >= vend[vid][1]:
                finalize(vid)

    def flush_vid(vid):
        for shape in [s for s, items in pending.items() if any(v == vid for v, _, _ in items)]:
            run_batch(shape)

    def handle(msg):
        kind = msg[0]
        if kind == "frames":
            _, vid, secs, t = msg
            shape = tuple(t.shape[1:])
            pending[shape].append((vid, secs, t))
            pending_n[shape] += len(t)
            if pending_n[shape] >= args.batch:
                run_batch(shape)
        elif kind == "start":
            worker_vid[msg[1]] = msg[2]
        elif kind == "end":
            _, vid, last_sec, n_sent, error = msg
            for wid, v in list(worker_vid.items()):
                if v == vid:
                    del worker_vid[wid]
            vend[vid] = (last_sec, n_sent, error)
            if vdone_frames[vid] >= n_sent:
                finalize(vid)
            else:
                flush_vid(vid)
        elif kind == "bye":
            byes.add(msg[1])

    last_report = time.time()
    while len(byes) < n_workers:
        try:
            msg = out_q.get(timeout=0.02)
        except queue.Empty:
            if pending:
                run_batch(max(pending_n, key=pending_n.get))
                continue
            # liveness check: replace crashed decoders
            for wid, p in list(procs.items()):
                if wid not in byes and not p.is_alive():
                    vid = worker_vid.pop(wid, None)
                    if vid is not None and vid not in vend:
                        flush_vid(vid)
                        secs = vsecs.get(vid)
                        last = int(max(a.max() for a in secs)) - 1 if secs else -1
                        vend[vid] = (last, vdone_frames[vid], "decoder crashed")
                        finalize(vid)
                    procs[wid] = spawn(wid)
            continue
        except Exception as e:  # a tensor from a crashed producer, etc.
            print(f"[run] dropped message: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            continue
        handle(msg)
        if time.time() - last_report > 30:
            last_report = time.time()
            el = time.time() - t_start
            print(f"[run] {el:7.0f}s videos {stats['done']}/{stats['videos']} frames {n_frames} "
                  f"({n_frames / max(el, 1e-6):.0f}/s, gpu busy {t_gpu / max(el, 1e-6):.0%})",
                  file=sys.stderr, flush=True)

    for shape in list(pending):
        run_batch(shape)
    for vid in list(vend):
        finalize(vid)
    shutdown.set()
    for p in procs.values():
        p.join(timeout=5)

    rt.join(timeout=5)
    out_f.close()
    el = time.time() - t_start
    print(f"[run] done: {stats['done']} videos, {n_frames} frames in {el:.1f}s "
          f"(gpu {t_gpu:.1f}s)", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
