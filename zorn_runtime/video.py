from __future__ import annotations

import cv2
from PIL import Image

from .preprocessing import resize_image


def get_video_metadata(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {"fps": fps, "frame_count": frame_count, "duration": duration, "width": width, "height": height}


def _similarity_score(a, b):
    # Small grayscale thumbnail; 0 = very different, 1 = identical-ish.
    import numpy as np
    aa = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    bb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    aa = cv2.resize(aa, (32, 32)).astype(np.float32)
    bb = cv2.resize(bb, (32, 32)).astype(np.float32)
    diff = np.mean(np.abs(aa - bb)) / 255.0
    return 1.0 - float(diff)


def sample_video_frames(path: str, max_frames: int = 8, frame_max_size: int = 512, similarity_threshold: float = 0.985):
    if max_frames < 1:
        return []
    meta = get_video_metadata(path)
    count = meta["frame_count"]
    if count <= 0:
        return []

    # Candidate positions cover the full duration. We then drop near-duplicates.
    candidates = __import__('numpy').linspace(0, max(0, count - 1), num=max_frames, dtype=int)
    cap = cv2.VideoCapture(path)
    frames = []
    previous = None
    try:
        for idx in candidates.tolist():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if previous is not None and _similarity_score(previous, rgb) >= similarity_threshold:
                continue
            image = Image.fromarray(rgb).convert("RGB")
            image = resize_image(image, frame_max_size)
            timestamp = float(idx / meta["fps"]) if meta["fps"] > 0 else 0.0
            frames.append({"image": image, "index": int(idx), "timestamp": timestamp})
            previous = rgb
    finally:
        cap.release()
    return frames
