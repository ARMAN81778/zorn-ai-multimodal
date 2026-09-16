from __future__ import annotations

import importlib.util
import os
import platform


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def is_intel_cpu() -> bool:
    text = (platform.processor() + " " + platform.machine()).lower()
    if "intel" in text:
        return True
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
            return "genuineintel" in f.read().lower()
    except OSError:
        return False


def system_info() -> dict:
    cpu = platform.processor() or platform.machine() or "Unknown CPU"
    return {
        "cpu": cpu,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "intel_cpu": is_intel_cpu(),
        "openvino_available": _has("openvino") and _has("optimum.intel"),
        "qnn_available": _qnn_available(),
        "torch_available": _has("torch"),
        "cores": os.cpu_count() or 1,
    }


def _qnn_available() -> bool:
    if not _has("onnxruntime"):
        return False
    try:
        import onnxruntime as ort
        return "QNNExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def select_backend(preferred: str | None = None) -> str:
    info = system_info()
    if preferred:
        return preferred
    if info["intel_cpu"] and info["openvino_available"]:
        return "openvino"
    if info["qnn_available"]:
        return "qnn"
    return "pytorch"


def print_system_info(selected_backend: str | None = None):
    info = system_info()
    print("\nZORN SIGHT SYSTEM INFORMATION")
    print("----------------------------")
    print(f"CPU: {info['cpu']}")
    print(f"Platform: {info['platform']}")
    print(f"Python: {info['python']}")
    print(f"CPU cores: {info['cores']}")
    print(f"Intel CPU: {info['intel_cpu']}")
    print(f"OpenVINO: {info['openvino_available']}")
    print(f"QNN: {info['qnn_available']}")
    print(f"Selected backend: {selected_backend or select_backend()}")
    return info
