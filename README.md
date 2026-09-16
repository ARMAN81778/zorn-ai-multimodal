# ZORN SIGHT

**Fast Multimodal Edge AI for Intel and Qualcomm**

ZORN SIGHT is a local, offline-first multimodal assistant built around
[HuggingFaceTB/SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct).
It is not a new foundation model — the base model is unchanged. The
contribution of this project is the **optimized inference/runtime
architecture** wrapped around that model: smart input compression, a
pluggable hardware backend layer (PyTorch, Intel OpenVINO, Qualcomm QNN),
evidence-grounded video understanding, and measured on-device benchmarking.

## 1. Problem

Small multimodal models are becoming capable enough to run on-device, but
most demo code treats every backend the same way, processes video frame by
frame with no sampling strategy, and provides no real measurement of
performance across hardware targets. That makes it hard to know whether an
"edge AI" claim is actually true.

## 2. Solution

ZORN SIGHT wraps SmolVLM2-500M-Instruct in a runtime that:

- compresses inputs before they ever reach the model (image resizing, smart
  video frame sampling, redundant-frame removal)
- selects the best available inference backend for the current hardware
- reports real, measured latency and memory numbers instead of marketing
  numbers
- keeps every video answer grounded in the actual frames that were sampled

## 3. Why ZORN SIGHT

- **Modular runtime** — swap PyTorch, OpenVINO, or QNN without touching
  application code.
- **Evidence-first** — video answers are always accompanied by the sampled
  frame indices/timestamps that were actually shown to the model.
- **Offline-first** — after the model is downloaded once, no internet
  connection or cloud AI API is required.
- **Honest benchmarking** — every number shown is measured on the machine
  running the demo, never invented.

## 4. Architecture

```
User
 ↓
ZORN SIGHT Interface (CLI / Gradio demo)
 ↓
Input Router (Text / Image / Video)
 ↓
Multimodal Preprocessor
 ↓
Smart Input Compression
 ├── image resizing
 ├── video frame sampling
 └── redundant-frame removal
 ↓
SmolVLM2-500M-Instruct
 ↓
Optimized Inference Runtime
 ├── Intel OpenVINO backend
 ├── PyTorch/Transformers backend (fallback)
 ├── Qualcomm QNN backend (deployment path)
 ↓
Response (Text + TTS Voice)
 ↓
Evidence / Performance Metrics
```

## 5. Multimodal Capabilities

- Multi-turn text chat with bounded conversation memory
- Image understanding (`analyze_image`)
- Short video understanding with smart frame sampling (`analyze_video`)
- Spoken responses via TTS

## 6. Intel Optimization

`zorn_runtime/openvino_backend.py` uses `optimum-intel`'s
`OVModelForVisualCausalLM` to export SmolVLM2 to OpenVINO IR on first run,
cache the converted model locally, and run inference with:

- CPU device targeting
- `PERFORMANCE_HINT=LATENCY`
- configurable inference streams and thread count
- an IR cache so conversion only happens once

If OpenVINO or the conversion path is unavailable in your environment, ZORN
SIGHT automatically and visibly falls back to the PyTorch backend — it never
pretends a conversion succeeded when it did not.

## 7. Qualcomm Optimization

`zorn_runtime/qnn_backend.py` checks for the ONNX Runtime QNN Execution
Provider. If it is not installed, the app prints:

```
Qualcomm QNN backend unavailable — using fallback backend.
```

and continues on PyTorch/OpenVINO. The documented deployment path is:

```
PyTorch / Hugging Face model
 → ONNX export
 → Qualcomm AI Hub / QNN compilation
 → Snapdragon NPU execution
```

This path has **not** been executed on real Snapdragon hardware as part of
this repository — it is a documented, ready-to-run path, not a tested claim.

## 8. Smart Video Processing

`zorn_runtime/video.py` reads video metadata, evenly samples up to
`--max-frames` candidate frames across the full duration, drops frames that
are near-duplicates of the previous sampled frame, resizes the remaining
frames, and returns each with its frame index and timestamp.

## 9. Evidence System

Every video (and image) answer returns the sampled frame indices and
timestamps that were actually fed to the model:

```json
{
  "answer": "...",
  "evidence": [
    {"timestamp": 3.2, "frame_index": 48}
  ]
}
```

No timestamp is ever fabricated — it always corresponds to a frame that was
sampled and shown to the model.

## 10. Voice Interface

`zorn_runtime/tts.py` uses gTTS by default and saves audio to
`outputs/response.mp3`. If gTTS cannot reach the internet, the app prints:

```
TTS unavailable offline. Text response remains available.
```

and continues without crashing. The module is structured so an offline TTS
engine (e.g. Piper, Coqui) can be swapped in later behind the same
`speak()` / `save_audio()` interface.

## 11. Offline-First Architecture

Hugging Face is used **only** to download the model weights the first time
the app runs. After that, everything — chat, image analysis, video
analysis, and benchmarking — runs locally with no cloud AI API calls
(no OpenAI, no Gemini, no remote inference provider).

## 12. Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Optional: Intel OpenVINO acceleration
pip install "optimum[openvino]" nncf openvino

# Optional: Qualcomm QNN execution provider
pip install onnxruntime-qnn
```

The SmolVLM2-500M-Instruct model is **not** included in this repository. It
downloads automatically from Hugging Face the first time you run the app,
and is cached locally afterward by the standard Hugging Face cache.

## 13. Usage

```bash
python zorn_sight_voice.py
```

```bash
python zorn_sight_voice.py \
    --image examples/test.jpg \
    --question "Describe this image"
```

```bash
python zorn_sight_voice.py \
    --video examples/test.mp4 \
    --question "What happens in this video?" \
    --max-frames 8
```

Add `--voice` to any command to also generate a spoken response.

## 14. Benchmarking

```bash
python benchmark.py
```

```bash
python benchmark.py --image examples/test.jpg --video examples/test.mp4
```

```bash
python benchmark.py --compare
```

`--compare` runs the same prompts on both the PyTorch and OpenVINO backends
(when OpenVINO is available) and prints a side-by-side latency comparison.
Every number shown is measured live — nothing is hardcoded.

## 15. Project Structure

```
zorn-sight/
├── zorn_sight_voice.py
├── demo.py
├── benchmark.py
├── requirements.txt
├── README.md
├── DEMO.md
├── LICENSE
├── .gitignore
├── config.yaml
├── zorn_runtime/
│   ├── __init__.py
│   ├── backend.py
│   ├── pytorch_backend.py
│   ├── openvino_backend.py
│   ├── qnn_backend.py
│   ├── device.py
│   ├── benchmark.py
│   ├── preprocessing.py
│   ├── video.py
│   ├── memory.py
│   └── tts.py
├── examples/
│   └── README.md
└── outputs/
    └── .gitkeep
```

## 16. Technical Architecture

- **`zorn_runtime/backend.py`** defines a common `InferenceBackend`
  interface (`load_model`, `generate`, `analyze_image`, `analyze_video`).
- **`zorn_runtime/pytorch_backend.py`** loads SmolVLM2-500M-Instruct with
  `AutoProcessor` / `AutoModelForImageTextToText` and runs on CPU.
- **`zorn_runtime/openvino_backend.py`** converts/loads the model through
  `optimum-intel`, caching the IR locally.
- **`zorn_runtime/qnn_backend.py`** detects the QNN execution provider and
  documents the export path without claiming untested execution.
- **`zorn_runtime/__init__.py`** exposes `get_backend()`, which auto-selects
  a backend based on detected hardware and gracefully falls back.

## 17. Limitations

- OpenVINO conversion for VLM architectures depends on `optimum-intel`
  version support; if conversion fails in your environment, ZORN SIGHT
  automatically falls back to PyTorch and tells you so.
- The Qualcomm QNN backend is a documented deployment path, not a tested
  execution path, since this repository was built and demoed on Intel/x86
  hardware.
- SmolVLM2-500M is a small model; answers are good for concise
  description/QA tasks, not long-form reasoning.

## 18. Future Work

- Swap gTTS for a fully offline TTS engine (e.g. Piper) behind the existing
  `zorn_runtime/tts.py` interface.
- Complete and test the ONNX → Qualcomm AI Hub → QNN export path on real
  Snapdragon hardware.
- Add INT8 post-training quantization via NNCF for the OpenVINO backend.
- Streaming token generation in the CLI and Gradio demo.

## 19. Hackathon Demo Instructions

See [`DEMO.md`](DEMO.md) for the full 60–90 second demo script.

### Quick Demo

```bash
pip install -r requirements.txt
python demo.py
```

1. Open the local Gradio URL printed in the terminal.
2. Check the Hardware/Runtime/Status panel.
3. Ask a question in the Chat tab.
4. Upload an image and ask a question in Image Understanding.
5. Upload a short video and ask a question in Video Understanding — review
   the Evidence panel for sampled frames/timestamps.
6. Click "🔊 Speak Response" to generate real audio.
7. Run the Benchmark tab and read the measured latency/memory values.

Judges can run the entire project without any model file in GitHub: the
model downloads automatically from Hugging Face on first run and is cached
locally by `transformers`/`optimum` afterward.

---

**One compact multimodal model. One runtime architecture. Multiple hardware
targets.**
