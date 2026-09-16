import argparse
import os
import sys
import time

from zorn_runtime import device, get_backend, MODEL_ID
from zorn_runtime.memory import ConversationMemory, get_process_memory_mb
from zorn_runtime.preprocessing import load_image, prepare_image
from zorn_runtime.video import sample_video_frames, get_video_metadata
from zorn_runtime.tts import speak, save_audio


def print_header():
    print("=" * 50)
    print("ZORN SIGHT — Fast Multimodal Edge AI")
    print("=" * 50)


def print_menu():
    print("\nZORN SIGHT")
    print("--------------------------------")
    print("1. Chat")
    print("2. Analyze Image")
    print("3. Analyze Video")
    print("4. Benchmark")
    print("5. System Information")
    print("6. Exit")


def print_result_panel(backend_name, device_name, input_type, latency, memory_mb):
    print("\n[ZORN SIGHT]")
    print(f"Backend: {backend_name}")
    print(f"Device: {device_name}")
    print(f"Input: {input_type}")
    if latency is not None:
        print(f"Latency: {latency:.3f} sec")
    if memory_mb is not None:
        print(f"Memory: {memory_mb} MB")


def safe_load_backend(preferred, model_id):
    print("\nLoading ZORN SIGHT runtime...")
    print(f"Model: {model_id}")
    try:
        backend = get_backend(preferred=preferred, model_id=model_id)
        print(f"Backend ready: {backend.name}")
        return backend
    except Exception as e:
        print(f"Fatal error while loading model: {e}")
        print("Please check your internet connection and installed dependencies.")
        sys.exit(1)


def do_chat(backend, memory, speak_reply=False):
    print("\nType 'exit' to return to menu.")
    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue
        if user_text.lower() in ("exit", "quit", "back"):
            break

        messages = memory.as_messages()
        messages = messages + [
            {"role": "user", "content": [{"type": "text", "text": user_text}]}
        ]

        t0 = time.time()
        try:
            result = backend.generate(messages, images=None, max_new_tokens=200)
        except Exception as e:
            print(f"Generation failed: {e}")
            continue
        latency = time.time() - t0

        answer = result["text"]
        print(f"\nAI: {answer}")
        print_result_panel(backend.name, "CPU", "text", latency, get_process_memory_mb())

        memory.add("user", user_text)
        memory.add("assistant", answer)

        if speak_reply:
            path = speak(answer)
            if path:
                print(f"Voice saved to: {path}")


def analyze_image_flow(backend, image_path, question, speak_reply=False, memory=None):
    if not os.path.isfile(image_path):
        print(f"Image not found: {image_path}")
        return None

    try:
        image = prepare_image(load_image(image_path))
    except Exception as e:
        print(f"Could not load image: {e}")
        return None

    history = memory.as_messages() if memory else []

    t0 = time.time()
    try:
        result = backend.analyze_image(image, question, history=history, max_new_tokens=200)
    except Exception as e:
        print(f"Image analysis failed: {e}")
        return None
    latency = time.time() - t0

    answer = result["text"]
    print(f"\nAI: {answer}")
    print_result_panel(backend.name, "CPU", "image", latency, get_process_memory_mb())

    if memory is not None:
        memory.add("user", question)
        memory.add("assistant", answer)

    if speak_reply:
        path = speak(answer)
        if path:
            print(f"Voice saved to: {path}")

    return {"answer": answer, "latency": latency}


def analyze_video_flow(backend, video_path, question, max_frames=8, speak_reply=False, memory=None):
    if not os.path.isfile(video_path):
        print(f"Video not found: {video_path}")
        return None

    try:
        meta = get_video_metadata(video_path)
        print(f"Video duration: {meta['duration']:.2f}s, fps: {meta['fps']:.2f}")
        frames = sample_video_frames(video_path, max_frames=max_frames)
    except Exception as e:
        print(f"Could not process video: {e}")
        return None

    if not frames:
        print("No frames could be extracted from this video.")
        return None

    print(f"Sampled {len(frames)} frames for analysis.")

    history = memory.as_messages() if memory else []

    t0 = time.time()
    try:
        result = backend.analyze_video(frames, question, history=history, max_new_tokens=200)
    except Exception as e:
        print(f"Video analysis failed: {e}")
        return None
    latency = time.time() - t0

    answer = result["text"]
    evidence = [{"timestamp": f["timestamp"], "frame_index": f["index"]} for f in frames]

    print(f"\nAI: {answer}")
    print("\nEvidence (sampled frames):")
    for e in evidence:
        print(f"  - frame {e['frame_index']} at {e['timestamp']}s")

    print_result_panel(backend.name, "CPU", "video", latency, get_process_memory_mb())

    if memory is not None:
        memory.add("user", question)
        memory.add("assistant", answer)

    if speak_reply:
        path = speak(answer)
        if path:
            print(f"Voice saved to: {path}")

    return {"answer": answer, "evidence": evidence, "latency": latency}


def run_benchmark(backend, image_path=None, video_path=None, max_frames=8):
    from zorn_runtime.benchmark import benchmark_text, benchmark_image, benchmark_video

    print("\nZORN SIGHT BENCHMARK")
    print("----------------------------")

    text_res = benchmark_text(backend)
    print(f"Backend: {text_res['backend']}")
    print(f"Text generation latency: {text_res['generation_latency']:.3f} sec")
    if text_res["tokens_per_second"]:
        print(f"Tokens/sec: {text_res['tokens_per_second']:.2f}")
    print(f"Memory: {text_res['memory_after_mb']} MB")

    if image_path and os.path.isfile(image_path):
        image = prepare_image(load_image(image_path))
        img_res = benchmark_image(backend, image)
        print(f"\nImage latency: {img_res['total_latency']:.3f} sec")
        print(f"Memory: {img_res['memory_after_mb']} MB")

    if video_path and os.path.isfile(video_path):
        frames = sample_video_frames(video_path, max_frames=max_frames)
        if frames:
            vid_res = benchmark_video(backend, frames)
            print(f"\nVideo processing latency: {vid_res['total_latency']:.3f} sec")
            print(f"Frames processed: {vid_res['num_frames']}")
            print(f"Memory: {vid_res['memory_after_mb']} MB")


def interactive_menu(backend, args):
    memory = ConversationMemory(max_history=args.max_history)

    while True:
        print_menu()
        try:
            choice = input("\nSelect an option: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting ZORN SIGHT.")
            break

        if choice == "1":
            do_chat(backend, memory, speak_reply=args.voice)
        elif choice == "2":
            path = input("Image path: ").strip()
            question = input("Question: ").strip() or "Describe this image."
            analyze_image_flow(backend, path, question, speak_reply=args.voice, memory=memory)
        elif choice == "3":
            path = input("Video path: ").strip()
            question = input("Question: ").strip() or "What happens in this video?"
            analyze_video_flow(
                backend, path, question,
                max_frames=args.max_frames, speak_reply=args.voice, memory=memory,
            )
        elif choice == "4":
            image_path = input("Optional image path for benchmark (enter to skip): ").strip() or None
            video_path = input("Optional video path for benchmark (enter to skip): ").strip() or None
            run_benchmark(backend, image_path=image_path, video_path=video_path, max_frames=args.max_frames)
        elif choice == "5":
            device.print_system_info(selected_backend=backend.name)
        elif choice == "6":
            print("Exiting ZORN SIGHT.")
            break
        else:
            print("Invalid option. Please choose 1-6.")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="ZORN SIGHT — Fast Multimodal Edge AI")
    parser.add_argument("--image", type=str, default=None, help="Path to an image file")
    parser.add_argument("--video", type=str, default=None, help="Path to a video file")
    parser.add_argument("--question", type=str, default=None, help="Question to ask about the image/video")
    parser.add_argument("--max-frames", type=int, default=8, help="Maximum frames sampled from a video")
    parser.add_argument("--max-history", type=int, default=10, help="Maximum conversation turns retained")
    parser.add_argument("--backend", type=str, default=None, choices=["pytorch", "openvino", "qnn"],
                         help="Force a specific inference backend")
    parser.add_argument("--voice", action="store_true", help="Speak the response using TTS")
    parser.add_argument("--model-id", type=str, default=MODEL_ID, help="Hugging Face model identifier")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    print_header()
    sys_info = device.print_system_info(selected_backend=args.backend)

    backend = safe_load_backend(args.backend, args.model_id)

    if args.image:
        question = args.question or "What is happening in this image?"
        analyze_image_flow(backend, args.image, question, speak_reply=args.voice)
        return

    if args.video:
        question = args.question or "What happens in this video?"
        analyze_video_flow(backend, args.video, question, max_frames=args.max_frames, speak_reply=args.voice)
        return

    interactive_menu(backend, args)


if __name__ == "__main__":
    main()
