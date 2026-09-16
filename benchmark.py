import argparse
import os

from zorn_runtime import device, get_backend, MODEL_ID
from zorn_runtime.benchmark import benchmark_text, benchmark_image, benchmark_video
from zorn_runtime.preprocessing import load_image, prepare_image
from zorn_runtime.video import sample_video_frames


def print_block(title, res):
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Backend: {res['backend']}")
    if res.get("generation_latency") is not None:
        print(f"Generation latency: {res['generation_latency']:.3f} sec")
    if res.get("tokens_per_second"):
        print(f"Tokens/sec: {res['tokens_per_second']:.2f}")
    print(f"Total latency: {res['total_latency']:.3f} sec")
    print(f"Memory: {res['memory_after_mb']} MB")
    if res.get("cpu_percent") is not None:
        print(f"CPU utilization: {res['cpu_percent']}%")


def main():
    parser = argparse.ArgumentParser(description="ZORN SIGHT benchmark")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
    parser.add_argument("--compare", action="store_true",
                         help="Compare PyTorch and OpenVINO backends if both are available")
    args = parser.parse_args()

    print("ZORN SIGHT BENCHMARK")
    print("----------------------------")
    sys_info = device.system_info()
    print(f"CPU: {sys_info['cpu']}")
    print(f"Intel CPU: {sys_info['intel_cpu']}")
    print(f"OpenVINO available: {sys_info['openvino_available']}")

    backends_to_run = []
    if args.compare and sys_info["openvino_available"]:
        backends_to_run = ["pytorch", "openvino"]
    else:
        backends_to_run = [None]

    results = {}

    for preferred in backends_to_run:
        label = preferred or device.select_backend()
        print(f"\nLoading backend: {label} ...")
        try:
            backend = get_backend(preferred=preferred, model_id=args.model_id, verbose=True)
        except Exception as e:
            print(f"Skipping backend {label}: {e}")
            continue

        text_res = benchmark_text(backend)
        print_block(f"TEXT — {backend.name}", text_res)
        results[backend.name] = {"text": text_res}

        if args.image and os.path.isfile(args.image):
            image = prepare_image(load_image(args.image))
            img_res = benchmark_image(backend, image)
            print_block(f"IMAGE — {backend.name}", img_res)
            results[backend.name]["image"] = img_res

        if args.video and os.path.isfile(args.video):
            frames = sample_video_frames(args.video, max_frames=args.max_frames)
            if frames:
                vid_res = benchmark_video(backend, frames)
                print_block(f"VIDEO — {backend.name}", vid_res)
                results[backend.name]["video"] = vid_res

    if len(results) > 1:
        print("\nCOMPARISON: PyTorch vs OpenVINO")
        print("--------------------------------")
        for name, r in results.items():
            gen = r["text"].get("generation_latency")
            print(f"{name}: text generation latency = {gen:.3f} sec" if gen else f"{name}: n/a")


if __name__ == "__main__":
    main()
