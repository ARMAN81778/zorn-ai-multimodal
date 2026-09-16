from __future__ import annotations

from .backend import InferenceBackend


class OpenVINOBackend(InferenceBackend):
    name = "openvino"
    device = "CPU"

    def __init__(self, model_id: str, verbose: bool = False):
        super().__init__(model_id, verbose)
        self._torch = None
        self.load_model()

    def load_model(self):
        try:
            import torch
            from transformers import AutoProcessor
            from optimum.intel import OVModelForVisualCausalLM
        except ImportError as exc:
            raise RuntimeError(
                "OpenVINO backend requires optimum-intel, openvino and transformers."
            ) from exc

        self._torch = torch
        if self.verbose:
            print(f"Loading {self.model_id} with OpenVINO...")

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = OVModelForVisualCausalLM.from_pretrained(
            self.model_id,
            export=True,
            device="CPU",
            ov_config={
                "PERFORMANCE_HINT": "LATENCY",
                "INFERENCE_NUM_THREADS": str(max(1, __import__('os').cpu_count() or 1)),
            },
        )
        self.device = "CPU / OpenVINO"
        return self

    def _generate(self, messages, max_new_tokens):
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_len = int(inputs["input_ids"].shape[-1])
        with self._torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )
        text = self.processor.batch_decode(
            output_ids[:, input_len:], skip_special_tokens=True
        )[0]
        return text.strip(), int(output_ids.shape[-1] - input_len)

    def generate(self, messages, images=None, max_new_tokens=200):
        text, generated_tokens = self._generate(messages, max_new_tokens)
        return {"text": text, "generated_tokens": generated_tokens}

    def analyze_image(self, image, question, history=None, max_new_tokens=200):
        messages = list(history or []) + [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }]
        text, generated_tokens = self._generate(messages, max_new_tokens)
        return {"text": text, "generated_tokens": generated_tokens}

    def analyze_video(self, frames, question, history=None, max_new_tokens=200):
        content = [{"type": "image", "image": f["image"]} for f in frames]
        content.append({"type": "text", "text": question})
        messages = list(history or []) + [{"role": "user", "content": content}]
        text, generated_tokens = self._generate(messages, max_new_tokens)
        return {"text": text, "generated_tokens": generated_tokens}
