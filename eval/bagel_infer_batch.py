# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
Text-to-Image (T2I) inference script.
Generates images from text prompts. Supports single-prompt and batch modes
with optional multi-GPU parallelism.
"""

import argparse
import json
import os
import re
import sys
import random
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_loader import load_model_and_inferencer


def set_seed(seed):
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def text_to_image(
    inferencer,
    prompt: str,
    image_shapes: tuple = (1024, 1024),
    think: bool = False,
    cfg_text_scale: float = 4.0,
    cfg_interval: tuple = (0.4, 1.0),
    timestep_shift: float = 3.0,
    num_timesteps: int = 50,
    cfg_renorm_min: float = 0.0,
    cfg_renorm_type: str = "global",
    max_think_token_n: int = 1024,
    do_sample: bool = False,
    text_temperature: float = 0.3,
    seed: int = 0,
):
    """
    Text-to-image generation.

    Args:
        inferencer: InterleaveInferencer instance.
        prompt: Input text prompt.
        image_shapes: Target image size as (height, width).
        think: Whether to enable thinking (chain-of-thought) mode.
        cfg_text_scale: Text classifier-free guidance scale.
        cfg_interval: Interval over which CFG is applied.
        timestep_shift: Timestep shift for the noise schedule.
        num_timesteps: Number of denoising steps.
        cfg_renorm_min: Minimum value for CFG renormalization.
        cfg_renorm_type: CFG renormalization type.
        max_think_token_n: Maximum number of thinking tokens.
        do_sample: Whether to use sampling for text generation.
        text_temperature: Temperature for text sampling.
        seed: Random seed (0 to disable).

    Returns:
        tuple: (generated image, thinking text or None)
    """
    set_seed(seed)

    inference_hyper = dict(
        max_think_token_n=max_think_token_n if think else 1024,
        do_sample=do_sample if think else False,
        text_temperature=text_temperature if think else 0.3,
        cfg_text_scale=cfg_text_scale,
        cfg_interval=list(cfg_interval),
        timestep_shift=timestep_shift,
        num_timesteps=num_timesteps,
        cfg_renorm_min=cfg_renorm_min,
        cfg_renorm_type=cfg_renorm_type,
        image_shapes=image_shapes,
    )

    result = inferencer(text=prompt, think=think, **inference_hyper)

    return result["image"], result.get("text", None)


def sanitize_filename(name: str, max_len: int = 200) -> str:
    """Convert a string into a filesystem-safe filename."""
    safe = re.sub(r'[<>:"/\\|?*]', '_', name)
    safe = re.sub(r'\s+', '_', safe)
    safe = safe.strip('._')[:max_len] or "unnamed"
    return safe


def _resolve_output_paths(item, key, output_dir):
    """Determine the image and trajectory output paths for a given item."""
    index = item.get("index", key)
    safe_name = sanitize_filename(str(index))
    source_category = item.get("source_category", "").strip()
    if source_category:
        cat_dir = os.path.join(output_dir, sanitize_filename(source_category))
    else:
        cat_dir = output_dir
    os.makedirs(cat_dir, exist_ok=True)
    img_path = os.path.join(cat_dir, f"{safe_name}.png")
    traj_path = os.path.join(cat_dir, f"{safe_name}_traj.json")
    return img_path, traj_path


def _save_trajectory(traj_path, item, key, img_path, status="success", error=None):
    """Save a trajectory JSON file recording the generation result."""
    traj = {
        "index": item.get("index", key),
        "ip_name": item.get("ip_name", ""),
        "image_prompt": item.get("image_prompt", ""),
        "source_category": item.get("source_category", ""),
        "country": item.get("country", ""),
        "language": item.get("language", ""),
        "generated_image": os.path.basename(img_path),
        "status": status,
    }
    if error:
        traj["error"] = str(error)
    with open(traj_path, 'w', encoding='utf-8') as f:
        json.dump(traj, f, ensure_ascii=False, indent=2)


def run_batch_inference(
    inferencer,
    prompt_json_path: str,
    output_dir: str,
    **text_to_image_kwargs,
):
    """
    Batch inference: read prompts from a JSON file, generate images, and save results.
    """
    with open(prompt_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    items = list(data.items()) if isinstance(data, dict) else []
    if not items:
        print("No valid data found in JSON file")
        return

    print(f"Total prompts to generate: {len(items)}")

    for key, item in tqdm(items, desc="Generating"):
        if not isinstance(item, dict):
            continue
        image_prompt = item.get("image_prompt")
        if not image_prompt:
            tqdm.write(f"Skipping {key}: no image_prompt field")
            continue

        img_path, traj_path = _resolve_output_paths(item, key, output_dir)

        if os.path.exists(img_path):
            tqdm.write(f"Already exists, skipping: {img_path}")
            continue

        try:
            image, _ = text_to_image(
                inferencer,
                prompt=image_prompt,
                **text_to_image_kwargs,
            )
            image.save(img_path)
            _save_trajectory(traj_path, item, key, img_path)
        except Exception as e:
            tqdm.write(f"Generation failed for {key}: {e}")
            _save_trajectory(traj_path, item, key, img_path, status="error", error=e)


def run_worker_chunk(chunk_file: str):
    """Worker mode: load a serialized chunk file and process its items."""
    import pickle
    with open(chunk_file, 'rb') as f:
        chunk_data = pickle.load(f)
    items = chunk_data["items"]
    model_path = chunk_data["model_path"]
    output_dir = chunk_data["output_dir"]
    mode = chunk_data["mode"]
    t2i_kwargs = chunk_data["t2i_kwargs"]
    inferencer = load_model_and_inferencer(model_path, mode=mode)
    for key, item in tqdm(items, desc="Generating"):
        if not isinstance(item, dict) or not item.get("image_prompt"):
            continue
        img_path, traj_path = _resolve_output_paths(item, key, output_dir)
        if os.path.exists(img_path):
            continue
        try:
            image, _ = text_to_image(inferencer, prompt=item["image_prompt"], **t2i_kwargs)
            image.save(img_path)
            _save_trajectory(traj_path, item, key, img_path)
        except Exception as e:
            tqdm.write(f"Generation failed for {key}: {e}")
            _save_trajectory(traj_path, item, key, img_path, status="error", error=e)


def run_parallel_batch_inference(prompt_json_path: str, output_dir: str, num_gpus: int = 8, **kwargs):
    """Multi-GPU parallel inference: distribute items across num_gpus worker processes."""
    import pickle
    import shutil
    import subprocess
    import tempfile
    with open(prompt_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    items = [(k, v) for k, v in (data.items() if isinstance(data, dict) else [])
             if isinstance(v, dict) and v.get("image_prompt")]
    if not items:
        print("No valid image_prompt entries in JSON")
        return
    model_path = kwargs.pop("model_path")
    mode = kwargs.pop("mode", 1)
    t2i_kwargs = kwargs
    chunks = [items[i::num_gpus] for i in range(num_gpus)]
    temp_dir = tempfile.mkdtemp(prefix="t2i_chunks_")
    script_path = os.path.abspath(__file__)
    procs = []
    for i in range(num_gpus):
        if not chunks[i]:
            continue
        cf = os.path.join(temp_dir, f"chunk_{i}.pkl")
        with open(cf, 'wb') as f:
            pickle.dump({"items": chunks[i], "model_path": model_path, "output_dir": output_dir,
                        "mode": mode, "t2i_kwargs": t2i_kwargs}, f)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)
        p = subprocess.Popen([sys.executable, script_path, "--worker_mode", "--chunk_file", cf],
                            env=env)
        procs.append((i, p))
    for i, p in procs:
        p.wait()
        if p.returncode != 0:
            print(f"GPU {i} worker exited with code: {p.returncode}")
    try:
        shutil.rmtree(temp_dir)
    except OSError:
        pass


def load_prompt_data_from_directory(
    prompt_json_dir: str,
    samples_per_file: int = 0,
    total_target_samples: int = 0,
    seed: int = 0,
):
    """
    Load and merge multiple JSON files from a directory into a single dict.

    Each JSON file is expected to be a dict mapping keys to items with an
    ``image_prompt`` field.  A ``source_category`` field is added based on
    the filename (without extension) so that outputs can be organized into
    per-category subdirectories.
    """
    if not os.path.isdir(prompt_json_dir):
        raise ValueError(f"Directory not found: {prompt_json_dir}")

    rng = random.Random(seed if seed > 0 else 0)
    merged = {}
    json_files = sorted([f for f in os.listdir(prompt_json_dir) if f.endswith(".json")])
    if not json_files:
        raise ValueError(f"No JSON files found in directory: {prompt_json_dir}")

    for jf in json_files:
        category = os.path.splitext(jf)[0]
        jpath = os.path.join(prompt_json_dir, jf)
        try:
            with open(jpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skipping {jpath}: failed to read ({e})")
            continue

        if not isinstance(data, dict):
            print(f"Skipping {jpath}: top-level JSON is not a dict")
            continue

        items = [(k, v) for k, v in data.items() if isinstance(v, dict) and v.get("image_prompt")]
        if samples_per_file > 0 and len(items) > samples_per_file:
            items = rng.sample(items, samples_per_file)

        for k, v in items:
            item = dict(v)
            item_index = str(item.get("index", k))
            if not item.get("source_category"):
                item["source_category"] = category

            merged_key = f"{category}_{item_index}"
            if merged_key in merged:
                suffix = 1
                while f"{merged_key}_{suffix}" in merged:
                    suffix += 1
                merged_key = f"{merged_key}_{suffix}"
            merged[merged_key] = item

    if total_target_samples > 0 and len(merged) > total_target_samples:
        keep_keys = rng.sample(list(merged.keys()), total_target_samples)
        merged = {k: merged[k] for k in keep_keys}

    if not merged:
        raise ValueError(f"No valid image_prompt data in directory: {prompt_json_dir}")

    return merged


def main():
    parser = argparse.ArgumentParser(description="BAGEL Text-to-Image Inference")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the model checkpoint")
    parser.add_argument("--mode", type=int, default=1, choices=[1, 2, 3],
                       help="Loading mode: 1=full precision, 2=NF4, 3=INT8")
    parser.add_argument("--prompt", type=str, help="Input text prompt (single mode)")
    parser.add_argument("--output", type=str, help="Output image path (single mode)")
    parser.add_argument("--prompt_json", type=str, default=None,
                       help="JSON file containing image_prompt entries (batch mode)")
    parser.add_argument("--prompt_json_dir", type=str, default=None,
                       help="Directory of JSON files (batch mode). Takes precedence over --prompt_json")
    parser.add_argument("--samples_per_file", type=int, default=0,
                       help="Max samples per JSON file in directory mode (0 = all)")
    parser.add_argument("--total_target_samples", type=int, default=0,
                       help="Total sample cap in directory mode (0 = no cap)")
    parser.add_argument("--output_dir", type=str, default="bagel_t2i_outputs",
                       help="Output directory for batch mode")
    parser.add_argument("--num_gpus", type=int, default=8,
                       help="Number of GPUs for parallel batch inference")
    parser.add_argument("--worker_mode", action="store_true",
                       help=argparse.SUPPRESS)
    parser.add_argument("--chunk_file", type=str, default=None,
                       help=argparse.SUPPRESS)
    parser.add_argument("--image_height", type=int, default=1024, help="Image height")
    parser.add_argument("--image_width", type=int, default=1024, help="Image width")
    parser.add_argument("--think", action="store_true", help="Enable thinking (chain-of-thought) mode")
    parser.add_argument("--cfg_text_scale", type=float, default=4.0, help="Text CFG scale")
    parser.add_argument("--cfg_interval_start", type=float, default=0.4, help="CFG interval start")
    parser.add_argument("--cfg_interval_end", type=float, default=1.0, help="CFG interval end")
    parser.add_argument("--timestep_shift", type=float, default=3.0, help="Timestep shift")
    parser.add_argument("--num_timesteps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--cfg_renorm_min", type=float, default=0.0, help="CFG renormalization minimum")
    parser.add_argument("--cfg_renorm_type", type=str, default="global",
                       choices=["global", "channel", "text_channel"],
                       help="CFG renormalization type")
    parser.add_argument("--max_think_token_n", type=int, default=1024, help="Max thinking tokens")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling for text generation")
    parser.add_argument("--text_temperature", type=float, default=0.3, help="Text sampling temperature")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (0 to disable)")
    parser.add_argument("--save_thinking", type=str, default=None, help="Save thinking text to file (optional)")

    args = parser.parse_args()

    if args.worker_mode and args.chunk_file:
        run_worker_chunk(args.chunk_file)
        return

    batch_mode = args.prompt is None or args.output is None

    if batch_mode:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if args.prompt_json_dir:
            try:
                merged_data = load_prompt_data_from_directory(
                    prompt_json_dir=args.prompt_json_dir,
                    samples_per_file=args.samples_per_file,
                    total_target_samples=args.total_target_samples,
                    seed=args.seed,
                )
            except Exception as e:
                print(f"Error: failed to load from directory: {e}")
                sys.exit(1)
            merged_prompt_json = os.path.join(output_dir, "_merged_prompt_input.json")
            with open(merged_prompt_json, "w", encoding="utf-8") as f:
                json.dump(merged_data, f, ensure_ascii=False, indent=2)
            prompt_json_path = merged_prompt_json
            print(f"Directory mode: merged {len(merged_data)} samples -> {prompt_json_path}")
        else:
            prompt_json_path = args.prompt_json
            if not prompt_json_path or not os.path.exists(prompt_json_path):
                print(f"Error: JSON file not found: {prompt_json_path}")
                sys.exit(1)
    else:
        output_dir = None
        prompt_json_path = None

    use_parallel = batch_mode and args.num_gpus > 1
    if not use_parallel:
        print("Loading model...")
        inferencer = load_model_and_inferencer(args.model_path, mode=args.mode)
        print("Model loaded.")

    t2i_kwargs = dict(
        image_shapes=(args.image_height, args.image_width),
        think=args.think,
        cfg_text_scale=args.cfg_text_scale,
        cfg_interval=(args.cfg_interval_start, args.cfg_interval_end),
        timestep_shift=args.timestep_shift,
        num_timesteps=args.num_timesteps,
        cfg_renorm_min=args.cfg_renorm_min,
        cfg_renorm_type=args.cfg_renorm_type,
        max_think_token_n=args.max_think_token_n,
        do_sample=args.do_sample,
        text_temperature=args.text_temperature,
        seed=args.seed,
    )

    if batch_mode:
        print(f"\nBatch mode: {prompt_json_path}")
        print(f"Output directory: {output_dir}")
        print(f"Image size: {args.image_height}x{args.image_width}")
        if use_parallel:
            print(f"Parallel GPUs: {args.num_gpus}")
            print("Launching GPU workers...")
            run_parallel_batch_inference(
                prompt_json_path=prompt_json_path,
                output_dir=output_dir,
                num_gpus=args.num_gpus,
                model_path=args.model_path,
                mode=args.mode,
                **t2i_kwargs,
            )
        else:
            run_batch_inference(
                inferencer,
                prompt_json_path=prompt_json_path,
                output_dir=output_dir,
                **t2i_kwargs,
            )
        print(f"\nBatch generation complete. Images saved to: {output_dir}")
    else:
        print(f"\nPrompt: {args.prompt}")
        print(f"Image size: {args.image_height}x{args.image_width}")
        print("Generating image...")

        image, thinking = text_to_image(
            inferencer,
            prompt=args.prompt,
            **t2i_kwargs,
        )

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        image.save(args.output)
        print(f"\nImage saved to: {args.output}")

        if thinking and args.save_thinking:
            with open(args.save_thinking, 'w', encoding='utf-8') as f:
                f.write(thinking)
            print(f"Thinking text saved to: {args.save_thinking}")
        elif thinking:
            print(f"\nThinking:\n{thinking}")


if __name__ == "__main__":
    main()
