from __future__ import annotations

import os
import time

from .memory import get_process_memory_mb


def _cpu_percent():
    try:
        import psutil
        return float(psutil.cpu_percent(interval=0.15))
    except Exception:
        return None


def benchmark_text(backend, max_new_tokens: int = 64):
    messages = [{"role": "user", "content": [{"type": "text", "text": "Reply with one short sentence: what is ZORN SIGHT?"}]}]
    before = get_process_memory_mb()
    t0 = time.perf_counter()
    result = backend.generate(messages, max_new_tokens=max_new_tokens)
    latency = time.perf_counter() - t0
    after = get_process_memory_mb()
    token_count = result.get("generated_tokens")
    tps = (token_count / latency) if token_count and latency > 0 else None
    return {
        "backend": backend.name,
        "generation_latency": latency,
        "total_latency": latency,
        "tokens_per_second": tps,
        "memory_before_mb": before,
        "memory_after_mb": after,
        "cpu_percent": _cpu_percent(),
        "text": result.get("text", ""),
    }


def benchmark_image(backend, image):
    before = get_process_memory_mb()
    t0 = time.perf_counter()
    result = backend.analyze_image(image, "Describe this image in one short sentence.", max_new_tokens=64)
    latency = time.perf_counter() - t0
    return {
        "backend": backend.name,
        "generation_latency": latency,
        "total_latency": latency,
        "memory_before_mb": before,
        "memory_after_mb": get_process_memory_mb(),
        "cpu_percent": _cpu_percent(),
        "text": result.get("text", ""),
    }


def benchmark_video(backend, frames):
    before = get_process_memory_mb()
    t0 = time.perf_counter()
    result = backend.analyze_video(frames, "Describe the main event in this video in one short sentence.", max_new_tokens=64)
    latency = time.perf_counter() - t0
    return {
        "backend": backend.name,
        "generation_latency": latency,
        "total_latency": latency,
        "num_frames": len(frames),
        "memory_before_mb": before,
        "memory_after_mb": get_process_memory_mb(),
        "cpu_percent": _cpu_percent(),
        "text": result.get("text", ""),
    }
