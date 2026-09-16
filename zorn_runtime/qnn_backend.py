from __future__ import annotations

import importlib.util

from .backend import InferenceBackend


class QNNBackend(InferenceBackend):
    """QNN capability detector and integration point.

    QNN execution requires an ONNX/QNN-compiled graph and compatible
    Qualcomm runtime libraries. This repository does not pretend to execute
    QNN when those artifacts are absent.
    """

    name = "qnn"
    device = "Qualcomm NPU (QNN)"

    def __init__(self, model_id: str, verbose: bool = False):
        super().__init__(model_id, verbose)
        self.load_model()

    def load_model(self):
        if not importlib.util.find_spec("onnxruntime"):
            raise RuntimeError("onnxruntime is not installed; QNN cannot be used.")
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "QNNExecutionProvider" not in providers:
            raise RuntimeError(
                "QNNExecutionProvider is unavailable. Install the Qualcomm/QNN ONNX Runtime package on supported hardware."
            )
        raise RuntimeError(
            "QNN provider detected, but no compiled SmolVLM2 QNN graph is bundled. "
            "Use the documented ONNX → Qualcomm AI Hub/QNN deployment path."
        )

    def generate(self, *args, **kwargs):
        raise RuntimeError("QNN execution is not configured in this repository.")

    def analyze_image(self, *args, **kwargs):
        raise RuntimeError("QNN execution is not configured in this repository.")

    def analyze_video(self, *args, **kwargs):
        raise RuntimeError("QNN execution is not configured in this repository.")
