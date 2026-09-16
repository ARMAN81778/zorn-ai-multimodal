import os
import time

from zorn_runtime import device, get_backend, MODEL_ID
from zorn_runtime.memory import ConversationMemory, get_process_memory_mb
from zorn_runtime.preprocessing import load_image, resize_image
from zorn_runtime.video import sample_video_frames, get_video_metadata
from zorn_runtime.tts import save_audio
from PIL import Image

BACKEND = None
MEMORY = ConversationMemory(max_history=10)
SYS_INFO = None


def init_backend():
    global BACKEND, SYS_INFO
    SYS_INFO = device.system_info()
    BACKEND = get_backend(model_id=MODEL_ID, verbose=True)
    return BACKEND


def status_text():
    ov = "AVAILABLE" if SYS_INFO["openvino_available"] else "NOT AVAILABLE"
    qnn = "AVAILABLE" if SYS_INFO["qnn_available"] else "NOT AVAILABLE"
    msg = (
        f"CPU: {SYS_INFO['cpu']}\n"
        f"Backend selected: {BACKEND.name}\n"
        f"OpenVINO: {ov}\n"
        f"QNN: {qnn}\n"
        f"Model: {MODEL_ID}\n"
        f"Status: Offline / Model Cached\n"
    )
    if BACKEND.name != "openvino" and SYS_INFO["openvino_available"] is False:
        msg += "OpenVINO unavailable — using PyTorch fallback\n"
    if not SYS_INFO["qnn_available"]:
        msg += "QNN unavailable — Qualcomm deployment path ready\n"
    return msg


def chat_fn(message, history):
    if BACKEND is None:
        init_backend()

    messages = MEMORY.as_messages() + [
        {"role": "user", "content": [{"type": "text", "text": message}]}
    ]
    t0 = time.time()
    try:
        result = BACKEND.generate(messages, images=None, max_new_tokens=200)
    except Exception as e:
        return f"Generation failed: {e}", ""
    latency = time.time() - t0

    answer = result["text"]
    MEMORY.add("user", message)
    MEMORY.add("assistant", answer)

    perf = f"Backend: {BACKEND.name} | Device: CPU | Latency: {latency:.2f}s | Memory: {get_process_memory_mb()} MB"
    return answer, perf


def image_fn(image, question):
    if BACKEND is None:
        init_backend()
    if image is None:
        return "Please upload an image.", "", ""
    if not question:
        question = "Describe this image and identify the most important objects."

    image = resize_image(Image.fromarray(image).convert("RGB"))

    t0 = time.time()
    try:
        result = BACKEND.analyze_image(image, question, max_new_tokens=200)
    except Exception as e:
        return f"Image analysis failed: {e}", "", ""
    latency = time.time() - t0

    answer = result["text"]
    perf = f"Backend: {BACKEND.name} | Device: CPU | Latency: {latency:.2f}s | Memory: {get_process_memory_mb()} MB"
    return answer, "No video evidence for image input.", perf


def video_fn(video_path, question, max_frames):
    if BACKEND is None:
        init_backend()
    if not video_path:
        return "Please upload a video.", "", ""
    if not question:
        question = "What is happening in this video?"

    try:
        meta = get_video_metadata(video_path)
        frames = sample_video_frames(video_path, max_frames=int(max_frames))
    except Exception as e:
        return f"Could not process video: {e}", "", ""
    if not frames:
        return "Could not extract frames from this video.", "", ""

    t0 = time.time()
    try:
        result = BACKEND.analyze_video(frames, question, max_new_tokens=200)
    except Exception as e:
        return f"Video analysis failed: {e}", "", ""
    latency = time.time() - t0

    answer = result["text"]
    evidence_lines = [
        f"Frame {f['index']} at {f['timestamp']}s" for f in frames
    ]
    evidence = (
        f"Duration: {meta['duration']:.2f}s | Frames sampled: {len(frames)}\n"
        + "\n".join(evidence_lines)
    )
    perf = f"Backend: {BACKEND.name} | Device: CPU | Latency: {latency:.2f}s | Memory: {get_process_memory_mb()} MB"
    return answer, evidence, perf


def speak_fn(text):
    if not text:
        return None
    path = save_audio(text)
    return path


def benchmark_fn():
    if BACKEND is None:
        init_backend()
    from zorn_runtime.benchmark import benchmark_text
    res = benchmark_text(BACKEND)
    lines = [
        f"Backend: {res['backend']}",
        f"Device: CPU",
        f"Generation latency: {res['generation_latency']:.3f} sec" if res["generation_latency"] else "Generation latency: n/a",
        f"Tokens/sec: {res['tokens_per_second']:.2f}" if res["tokens_per_second"] else "Tokens/sec: n/a",
        f"Memory: {res['memory_after_mb']} MB",
    ]
    return "\n".join(lines)


def build_ui():
    import gradio as gr

    init_backend()

    with gr.Blocks(title="ZORN SIGHT") as demo:
        gr.Markdown("# ZORN SIGHT\n### Fast Multimodal Edge AI")
        status_box = gr.Textbox(value=status_text(), label="Hardware / Runtime / Status", lines=6, interactive=False)

        with gr.Tab("Chat"):
            chat_input = gr.Textbox(label="Message")
            chat_btn = gr.Button("Ask")
            chat_output = gr.Textbox(label="Result", lines=4)
            chat_perf = gr.Textbox(label="Performance")
            chat_btn.click(chat_fn, inputs=[chat_input, gr.State([])], outputs=[chat_output, chat_perf])

        with gr.Tab("Image Understanding"):
            image_input = gr.Image(label="Upload Image")
            image_question = gr.Textbox(label="Question", value="Describe this image and identify the most important objects.")
            image_btn = gr.Button("Analyze Image")
            image_output = gr.Textbox(label="Result", lines=4)
            image_evidence = gr.Textbox(label="Evidence")
            image_perf = gr.Textbox(label="Performance")
            image_btn.click(
                image_fn,
                inputs=[image_input, image_question],
                outputs=[image_output, image_evidence, image_perf],
            )

        with gr.Tab("Video Understanding"):
            video_input = gr.Video(label="Upload Video")
            video_question = gr.Textbox(label="Question", value="What is happening in this video?")
            max_frames_slider = gr.Slider(2, 16, value=8, step=1, label="Max Frames")
            video_btn = gr.Button("Analyze Video")
            video_output = gr.Textbox(label="Result", lines=4)
            video_evidence = gr.Textbox(label="Evidence", lines=6)
            video_perf = gr.Textbox(label="Performance")
            video_btn.click(
                video_fn,
                inputs=[video_input, video_question, max_frames_slider],
                outputs=[video_output, video_evidence, video_perf],
            )

        with gr.Tab("Voice"):
            voice_text = gr.Textbox(label="Text to speak")
            voice_btn = gr.Button("🔊 Speak Response")
            voice_audio = gr.Audio(label="Voice Output")
            voice_btn.click(speak_fn, inputs=[voice_text], outputs=[voice_audio])

        with gr.Tab("Benchmark"):
            bench_btn = gr.Button("Run Benchmark")
            bench_output = gr.Textbox(label="Measured Performance", lines=6)
            bench_btn.click(benchmark_fn, inputs=[], outputs=[bench_output])

        gr.Markdown(
            "### WHY ZORN SIGHT?\n"
            "One compact multimodal model. One optimized runtime. Multiple hardware targets.\n\n"
            "✓ Text  ✓ Image  ✓ Video  ✓ Voice  ✓ Evidence  \n"
            "✓ Intel CPU optimization  ✓ Qualcomm deployment path  \n"
            "✓ Offline inference  ✓ Real benchmarks"
        )

    return demo


def main():
    try:
        demo = build_ui()
        demo.launch()
    except Exception as e:
        print(f"Gradio demo unavailable ({e}). Falling back to terminal demo.")
        os.system("python zorn_sight_voice.py")


if __name__ == "__main__":
    main()
