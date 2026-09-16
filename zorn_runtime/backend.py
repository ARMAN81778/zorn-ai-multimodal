from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class InferenceBackend(ABC):
    """Common interface used by the CLI, Gradio demo and benchmark."""

    name = "unknown"
    device = "CPU"

    def __init__(self, model_id: str, verbose: bool = False):
        self.model_id = model_id
        self.verbose = verbose
        self.model = None
        self.processor = None

    @abstractmethod
    def load_model(self):
        raise NotImplementedError

    @abstractmethod
    def generate(self, messages: List[Dict[str, Any]], images=None, max_new_tokens: int = 200):
        raise NotImplementedError

    @abstractmethod
    def analyze_image(self, image, question: str, history=None, max_new_tokens: int = 200):
        raise NotImplementedError

    @abstractmethod
    def analyze_video(self, frames, question: str, history=None, max_new_tokens: int = 200):
        raise NotImplementedError

    def info(self) -> Dict[str, Any]:
        return {"backend": self.name, "device": self.device, "model_id": self.model_id}
