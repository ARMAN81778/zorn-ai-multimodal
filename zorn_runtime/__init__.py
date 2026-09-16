from __future__ import annotations

from .backend import InferenceBackend
from .device import select_backend

# SmolVLM2's official 500M video-capable checkpoint. It supports text, images,
# multiple images and video through the Transformers multimodal chat API.
MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"


def get_backend(preferred: str | None = None, model_id: str = MODEL_ID, verbose: bool = False):
    selected = select_backend(preferred)

    if selected == "openvino":
        from .openvino_backend import OpenVINOBackend
        try:
            return OpenVINOBackend(model_id=model_id, verbose=verbose)
        except Exception as exc:
            if verbose:
                print(f"OpenVINO backend unavailable: {exc}")
            if preferred == "openvino":
                raise
            selected = "pytorch"

    if selected == "qnn":
        from .qnn_backend import QNNBackend
        try:
            return QNNBackend(model_id=model_id, verbose=verbose)
        except Exception as exc:
            if verbose:
                print(f"QNN backend unavailable: {exc}")
            if preferred == "qnn":
                raise
            selected = "pytorch"

    from .pytorch_backend import PyTorchBackend
    return PyTorchBackend(model_id=model_id, verbose=verbose)


__all__ = ["MODEL_ID", "InferenceBackend", "get_backend"]
