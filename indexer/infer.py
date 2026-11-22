"""Batched YOLO11l inference on the GPU with a minimal, Ultralytics-equivalent post-processing.

Only per-frame counts of the five target classes are produced; boxes never leave the GPU.
"""
from __future__ import annotations

import os
import sys

import torch
import torchvision

# COCO ids used by yolo11l.pt
CLASS_IDS = (0, 2, 16, 63, 67)
CLASS_NAMES = ("person", "car", "dog", "laptop", "cell phone")
CONF = 0.4
IOU = 0.7
MAX_DET = 300
MAX_WH = 7680


def load_model(weights: str, device: torch.device, half: bool = True):
    from ultralytics import YOLO

    model = YOLO(weights).model
    model = model.fuse(verbose=False) if hasattr(model, "fuse") else model
    model = model.to(device).eval()
    if half:
        model = model.half()
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class Detector:
    def __init__(self, weights: str, device: str = "cuda", half: bool = True, engine: str | None = None):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.half = half and self.device.type == "cuda"
        self.dtype = torch.float16 if self.half else torch.float32
        self.weights = weights
        self.torch_model = None
        self.backend = "torch"
        self.model = None
        if engine and self.device.type == "cuda" and os.path.exists(engine):
            try:
                from .trt import TRTModel

                self.model = TRTModel(engine, self.device)
                self.backend = "tensorrt"
            except Exception as e:  # fall back to PyTorch if TensorRT is unusable here
                print(f"[infer] TensorRT engine unusable ({type(e).__name__}: {e}); using PyTorch",
                      file=sys.stderr, flush=True)
        if self.model is None:
            self.model = self.torch_model = load_model(weights, self.device, self.half)
        # map coco id -> local index 0..4, everything else -> -1
        lut = torch.full((128,), -1, dtype=torch.long)
        for i, c in enumerate(CLASS_IDS):
            lut[c] = i
        self.lut = lut.to(self.device)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

    MAX_BS = 32
    BUCKETS = (4, 8, 16, 32)

    @torch.inference_mode()
    def __call__(self, batch_u8: torch.Tensor) -> torch.Tensor:
        """batch_u8: (B, H, W, 3) uint8 BGR letterboxed frames (CPU).

        Returns a (B, 5) int32 CPU tensor: per frame, number of detections of each target class.
        Batches are split/padded to a few fixed sizes so cuDNN autotuning happens only once per
        (bucket, shape) instead of for every distinct batch size.
        """
        outs = []
        for i in range(0, len(batch_u8), self.MAX_BS):
            part = batch_u8[i:i + self.MAX_BS]
            n = len(part)
            bucket = next(b for b in self.BUCKETS if b >= n)
            if bucket > n:
                pad = torch.zeros((bucket - n,) + tuple(part.shape[1:]), dtype=part.dtype)
                part = torch.cat([part, pad])
            outs.append(self._run(part)[:n])
        return torch.cat(outs) if len(outs) > 1 else outs[0]

    def _run(self, batch_u8: torch.Tensor) -> torch.Tensor:
        x = batch_u8.to(self.device, non_blocking=True)
        x = x.flip(-1).permute(0, 3, 1, 2)  # BGR->RGB, BHWC->BCHW (NHWC memory = channels_last)
        pred = None
        if self.backend == "tensorrt" and self.model.supports(tuple(x.shape)):
            try:
                pred = self.model(x.to(torch.float16).div_(255.0).contiguous())
            except Exception as e:
                print(f"[infer] TensorRT call failed ({e}); using PyTorch for this batch",
                      file=sys.stderr, flush=True)
                pred = None
        if pred is None:
            if self.torch_model is None:  # e.g. an aspect ratio outside the engine's profile
                self.torch_model = load_model(self.weights, self.device, self.half)
            x = x.to(self.dtype).div_(255.0).contiguous(memory_format=torch.channels_last)
            pred = self.torch_model(x)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        return self.postprocess(pred.float())

    def postprocess(self, pred: torch.Tensor) -> torch.Tensor:
        # pred: (B, 4 + nc, N) as in ultralytics; best class per box (multi_label=False)
        B = pred.shape[0]
        scores, cls = pred[:, 4:, :].max(1)                    # (B, N)
        local = self.lut[cls]                                  # (B, N)
        keep = (scores > CONF) & (local >= 0)
        counts = torch.zeros((B, len(CLASS_IDS)), dtype=torch.int32, device=pred.device)
        if not bool(keep.any()):
            return counts.cpu()
        bi, ni = keep.nonzero(as_tuple=True)
        boxes = pred[bi, :4, ni]                               # (M, 4) xywh
        xyxy = torch.empty_like(boxes)
        xyxy[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2
        xyxy[:, 2:] = boxes[:, :2] + boxes[:, 2:] / 2
        sc = scores[bi, ni]
        lc = local[bi, ni]
        # class-aware NMS per image in one call: shift boxes per (image, class)
        group = bi * len(CLASS_IDS) + lc
        off = (group.to(xyxy.dtype) * MAX_WH).unsqueeze(1)
        kept = torchvision.ops.nms(xyxy + off, sc, IOU)
        kb, kc = bi[kept], lc[kept]
        if kept.numel() > MAX_DET:  # per-image max_det cap (practically never hit)
            order = torch.argsort(kb * 0 - sc[kept], stable=True)
            kb, kc, ks = kb[order], kc[order], sc[kept][order]
            rank = torch.zeros_like(kb)
            for b in kb.unique():
                m = kb == b
                rank[m] = torch.arange(int(m.sum()), device=kb.device)
            sel = rank < MAX_DET
            kb, kc = kb[sel], kc[sel]
        flat = (kb * len(CLASS_IDS) + kc)
        counts.view(-1).index_add_(0, flat, torch.ones_like(flat, dtype=torch.int32))
        return counts.cpu()
