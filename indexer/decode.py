"""Video decoding, rotation, per-second frame sampling and YOLO letterboxing (CPU side)."""
from __future__ import annotations

import io
import math

import av
import cv2
import numpy as np

IMGSZ = 640
STRIDE = 32
PAD_VALUE = 114

av.logging.set_level(av.logging.PANIC)  # damaged videos spam the log otherwise
cv2.setNumThreads(1)  # we parallelise across processes, not inside cv2


def letterbox_shape(h: int, w: int, imgsz: int = IMGSZ, stride: int = STRIDE):
    """Same arithmetic as ultralytics LetterBox(auto=True): returns (new_w, new_h, top, bottom, left, right)."""
    r = min(imgsz / h, imgsz / w)
    new_w, new_h = round(w * r), round(h * r)
    dw, dh = (imgsz - new_w) % stride, (imgsz - new_h) % stride
    dw, dh = dw / 2, dh / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    return new_w, new_h, top, bottom, left, right


def letterbox(img: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Ultralytics-equivalent letterbox (bilinear resize, grey 114 padding, centred, rect/auto mode)."""
    h, w = img.shape[:2]
    new_w, new_h, top, bottom, left, right = letterbox_shape(h, w)
    if (w, h) != (new_w, new_h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    oh, ow = new_h + top + bottom, new_w + left + right
    if out is None:
        out = np.empty((oh, ow, 3), dtype=np.uint8)
    if top or bottom or left or right:
        out[:top] = PAD_VALUE
        out[top + new_h:] = PAD_VALUE
        out[:, :left] = PAD_VALUE
        out[:, left + new_w:] = PAD_VALUE
    out[top:top + new_h, left:left + new_w] = img
    return out


def _display_rotation(stream, first_frame) -> int:
    """Clockwise rotation (0/90/180/270) needed to show the frame upright, like ffmpeg's autorotate."""
    deg = None
    try:
        deg = first_frame.rotation  # PyAV >= 13: counter-clockwise angle from the display matrix
    except Exception:
        pass
    if not deg:
        try:  # old-style "rotate" tag (clockwise degrees)
            tag = stream.metadata.get("rotate")
            if tag is not None:
                return int(round(float(tag))) % 360
        except Exception:
            pass
        return 0
    return int(round(-float(deg))) % 360


def _rotate(img: np.ndarray, rot_cw: int) -> np.ndarray:
    if rot_cw == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if rot_cw == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if rot_cw == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


class DecodeError(Exception):
    pass


def iter_sampled_frames(data: bytes, k: int, decode_threads: int = 0, raw: bool = False):
    """Decode a video and yield (second, letterboxed_bgr_uint8) for up to k frames per second.

    k <= 0 means "every frame" (used to build the reference).
    Frames are chosen as the first frame at/after the targets s + (j + 0.5) / k; a second that has
    frames but hit no target gets its last frame.  The generator *returns* the index of the last
    second that holds a decoded frame (-1 if none) via StopIteration.value.
    """
    try:
        container = av.open(io.BytesIO(data), mode="r", metadata_errors="ignore")
    except Exception as e:  # not a container we can parse
        raise DecodeError(f"could not open video: {type(e).__name__}") from None
    try:
        if not container.streams.video:
            raise DecodeError("no video stream")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if decode_threads:
            stream.thread_count = decode_threads
        tb = stream.time_base
        fps = float(stream.average_rate or stream.guessed_rate or 30)

        t0 = None
        rot = 0
        last_sec = -1
        cur_sec = -1
        taken = 0          # samples taken in cur_sec
        next_j = 0         # next target index in cur_sec
        pending = None     # last unsampled frame of cur_sec (fallback sample)
        idx = 0

        def convert(fr):
            img = fr.to_ndarray(format="bgr24")
            if rot:
                img = _rotate(img, rot)
            return img if raw else letterbox(img)

        for frame in container.decode(stream):
            if frame.pts is not None and tb is not None:
                t = float(frame.pts * tb)
            else:
                t = idx / fps
            idx += 1
            if t0 is None:
                t0 = t
                rot = _display_rotation(stream, frame)
            t -= t0
            if t < 0:
                continue
            s = int(math.floor(t))
            if s < cur_sec:  # non-monotonic timestamps: count it in the current second
                s = cur_sec
            if s > cur_sec:
                if pending is not None and taken == 0:
                    yield cur_sec, convert(pending)
                cur_sec, taken, next_j, pending = s, 0, 0, None
            last_sec = s
            if k <= 0:
                yield s, convert(frame)
                taken += 1
                continue
            frac = t - s
            if next_j < k and frac >= (next_j + 0.5) / k:
                yield s, convert(frame)
                taken += 1
                while next_j < k and frac >= (next_j + 0.5) / k:
                    next_j += 1
                pending = None
            else:
                pending = frame
        if pending is not None and taken == 0:
            yield cur_sec, convert(pending)
        return last_sec
    finally:
        try:
            container.close()
        except Exception:
            pass
