import argparse
import torch
import os
import imageio
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

TIME_DIVISION_FACTOR = 4
TIME_DIVISION_REMAINDER = 1


def align_num_frames(frame_count: int) -> int:
    """Round down to the largest count compatible with Wan VACE (4n + 1)."""
    if frame_count < 1:
        raise ValueError("input video has no frames")
    if frame_count % TIME_DIVISION_FACTOR == TIME_DIVISION_REMAINDER:
        return frame_count
    aligned = ((frame_count - TIME_DIVISION_REMAINDER) // TIME_DIVISION_FACTOR) * TIME_DIVISION_FACTOR + TIME_DIVISION_REMAINDER
    return max(TIME_DIVISION_REMAINDER, aligned)


def probe_input_video(video_path: str) -> tuple[int, int, int, int]:
    reader = imageio.get_reader(video_path)
    try:
        meta = reader.get_meta_data()
        fps = meta.get("fps", 16)
        if fps is None or fps <= 0:
            fps = 16
        fps = max(1, int(round(fps)))

        frame_count = reader.count_frames()
        num_frames = align_num_frames(frame_count)
        if num_frames != frame_count:
            print(
                f"Adjusted num_frames from {frame_count} to {num_frames} "
                f"(must satisfy {TIME_DIVISION_FACTOR}n + {TIME_DIVISION_REMAINDER})"
            )

        first_frame = reader.get_data(0)
        height, width = first_frame.shape[:2]
        return height, width, num_frames, fps
    finally:
        reader.close()


def log_device_status(pipe, device: str, stage: str) -> None:
    gpu_id = int(device.split(":")[-1]) if ":" in device else 0
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(gpu_id) / (1024 ** 3)
        print(f"[device] {stage}: cuda:{gpu_id} allocated={alloc:.2f}GB reserved={reserved:.2f}GB")

    for name in ("vace", "dit", "vae", "text_encoder"):
        model = getattr(pipe, name, None)
        if model is None:
            continue
        try:
            param = next(model.parameters())
            print(f"[device] {stage}: {name}.parameters -> device={param.device}, dtype={param.dtype}")
        except StopIteration:
            pass


def ensure_cuda_device(device_id: int) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Ditto inference requires a GPU.")

    device_count = torch.cuda.device_count()
    if device_id < 0 or device_id >= device_count:
        raise RuntimeError(
            f"Invalid --device_id {device_id}. "
            f"This machine exposes cuda devices 0-{device_count - 1} only."
        )

    props = torch.cuda.get_device_properties(device_id)
    device = f"cuda:{device_id}"
    print(
        f"[device] target GPU: {device} ({props.name}, "
        f"{props.total_memory / (1024 ** 3):.1f} GB)"
    )
    torch.zeros(1, device=device)
    print(f"[device] cuda sanity check passed on {device}")
    return device


def main(args):

    device = ensure_cuda_device(args.device_id)

    model_config_kwargs = {}
    if args.local_model_path:
        model_config_kwargs["local_model_path"] = args.local_model_path
    if args.skip_model_download:
        model_config_kwargs["skip_download"] = True
    if not args.load_on_gpu:
        model_config_kwargs["offload_device"] = "cpu"

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors", **model_config_kwargs),
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **model_config_kwargs),
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="Wan2.1_VAE.pth", **model_config_kwargs),
        ],
        redirect_common_files=False,
    )
    print(f"[device] initial load target: {'GPU ' + device if args.load_on_gpu else 'CPU (offload_device=cpu)'}")
    log_device_status(pipe, device, "after model load")
    if args.lora_path:
        print(f"Loading Ditto LoRA model: {args.lora_path} (alpha={args.lora_alpha})")
        if not os.path.exists(args.lora_path):
            print(f"Error: LoRA file not found at {args.lora_path}")
            return
        pipe.load_lora(pipe.vace, args.lora_path, alpha=args.lora_alpha)

    pipe.enable_vram_management()
    print("[device] vram management enabled: weights mostly on CPU, computation on GPU during inference")
    log_device_status(pipe, device, "after vram management")

    print(f"Loading input video: {args.input_video}")
    if not os.path.exists(args.input_video):
        print(f"Error: Input video file not found at {args.input_video}")
        return

    if args.match_input_video:
        height, width, num_frames, fps = probe_input_video(args.input_video)
        print(f"Matched input video: {width}x{height}, {num_frames} frames, {fps} fps")
    else:
        height, width, num_frames, fps = args.height, args.width, args.num_frames, args.fps

    video = VideoData(args.input_video, height=height, width=width)

    available_frames = len(video)
    if num_frames > available_frames:
        num_frames = align_num_frames(available_frames)
        print(
            f"Warning: requested frame count exceeds video length ({available_frames}). "
            f"Using {num_frames} frames instead."
        )

    video = [video[i] for i in range(num_frames)]
    
    reference_image = None

    log_device_status(pipe, device, "before inference")
    video = pipe(
        prompt=args.prompt,
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        vace_video=video,
        vace_reference_image=reference_image,
        height=height,
        width=width,
        num_frames=num_frames,
        seed=args.seed,
        tiled=True,
    )
    log_device_status(pipe, device, "after inference")

    output_dir = os.path.dirname(args.output_video)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    save_video(video, args.output_video, fps=fps, quality=args.quality)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InstructV2V Pipeline.")

    parser.add_argument("--input_video", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--output_video", type=str, required=True, help="Path to save the output video file.")
    parser.add_argument("--lora_path", type=str, default=None, help="Optional path to a LoRA model file (.safetensors).")
    parser.add_argument("--local_model_path", type=str, default=None, help="Root directory for pretrained base models (expects Wan-AI/Wan2.1-VACE-14B underneath).")
    parser.add_argument("--skip_model_download", action="store_true", help="Use local pretrained models only; do not download from Hugging Face.")
    parser.add_argument("--load_on_gpu", action="store_true", help="Load base model weights directly onto GPU instead of CPU.")
    parser.add_argument("--device_id", type=int, default=0, help="The ID of the CUDA device to use (e.g., 0, 1, 2).")
    parser.add_argument("--prompt", type=str, required=True, help="The positive prompt describing the target style.")
    parser.add_argument("--height", type=int, default=480, help="The height to use for video processing.")
    parser.add_argument("--width", type=int, default=832, help="The width to use for video processing.")
    parser.add_argument("--num_frames", type=int, default=73, help="The number of video frames to process.")
    parser.add_argument("--match_input_video", action="store_true", help="Use each input video's native resolution, frame count, and fps.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducible results.")

    parser.add_argument("--lora_alpha", type=float, default=1.0, help="The alpha (weight) value for the LoRA model.")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second (FPS) for the output video.")
    parser.add_argument("--quality", type=int, default=5, help="Quality of the output video (CRF value, lower is better).")

    args = parser.parse_args()
    main(args)