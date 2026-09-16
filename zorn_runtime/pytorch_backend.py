from __future__ import annotations

from typing import Any, Dict, List, Optional

from .backend import InferenceBackend


class PyTorchBackend(InferenceBackend):
    name = "pytorch"
    device = "CPU"

    def __init__(self, model_id: str, verbose: bool = False):
        super().__init__(model_id, verbose)
        self._torch = None
        self.load_model()

    def load_model(self):
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch backend requires torch and transformers. Install requirements.txt."
            ) from exc

        self._torch = torch
        if self.verbose:
            print(f"Loading {self.model_id} with Transformers/PyTorch...")

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        # CPU-friendly loading. dtype is left to the model/config so this also
        # works across machines where bfloat16 CPU support differs.
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_id)
        self.model.to("cpu")
        self.model.eval()
        self.device = "CPU"
        return self

    @staticmethod
    def _content_image(image):
        return {"type": "image", "image": image}

    def _prepare(self, messages):
        # SmolVLM's current processor accepts local PIL images directly in the
        # multimodal chat template.
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs.to(self.model.device)

    def _decode(self, generated, input_len: int) -> str:
        ids = generated[:, input_len:]
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
        return text.strip()

    def _generate_messages(self, messages, max_new_tokens=200):
        inputs = self._prepare(messages)
        input_len = int(inputs["input_ids"].shape[-1])
        with self._torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )
        return self._decode(output_ids, input_len), int(output_ids.shape[-1] - input_len)

    def generate(self, messages: List[Dict[str, Any]], images=None, max_new_tokens: int = 200):
        # Existing CLI messages already use the Transformers multimodal format.
        text, generated_tokens = self._generate_messages(messages, max_new_tokens)
        return {"text": text, "generated_tokens": generated_tokens}

    def analyze_image(self, image, question: str, history=None, max_new_tokens: int = 200):
        messages = list(history or [])
        messages.append({
            "role": "user",
            "content": [self._content_image(image), {"type": "text", "text": question}],
        })
        text, generated_tokens = self._generate_messages(messages, max_new_tokens)
        return {"text": text, "generated_tokens": generated_tokens}

    def analyze_video(self, frames, question: str, history=None, max_new_tokens: int = 200):
        messages = list(history or [])
        content = [self._content_image(frame["image"]) for frame in frames]
        content.append({"type": "text", "text": question})
        messages.append({"role": "user", "content": content})
        text, generated_tokens = self._generate_messages(messages, max_new_tokens)
        return {"text": text, "generated_tokens": generated_tokens}
