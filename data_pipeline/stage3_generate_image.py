# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""
Stage 3: Generate images from Stage 2 recaption trajectories.

Reads trajectory JSON files from the input directory, extracts the <recaption>
content as the generation prompt, optionally loads reference images from the
intermediate directory, then calls the OpenAI Images API to produce output images.

Environment variables:
    OPENAI_API_KEY   – required
    OPENAI_BASE_URL  – optional, override the default API base URL
    IMAGE_GEN_MODEL  – optional, defaults to "gpt-image-1"
"""
import argparse
import base64
import json
import logging
import os
import re
import time
import traceback
from io import BytesIO
from pathlib import Path

from openai import OpenAI
from PIL import Image
from tqdm import tqdm

log = logging.getLogger("stage3")


# ---------------------------------------------------------------------------
# Image utility helpers
# ---------------------------------------------------------------------------

def get_image_from_path(image_path: str):
    """Open an image from a local file path.

    Returns a PIL Image or None on failure.
    """
    try:
        return Image.open(image_path)
    except Exception as e:
        print(f"Error opening local image: {e}")
        return None


def encode_image_from_path(image_path: str, resize_action=None) -> str:
    """Read a local image, optionally resize it, and return a base64-encoded PNG string."""
    image = get_image_from_path(image_path)
    if image is None:
        raise ValueError(f"Failed to open image: {image_path}")

    if resize_action is None:
        target_pixels = 1024 * 1024
        aspect_ratio = image.size[0] / image.size[1]
        new_height = int((target_pixels / aspect_ratio) ** 0.5)
        new_width = int(new_height * aspect_ratio)
        resize_action = (new_width, new_height)

    image = image.resize(resize_action)

    buf = BytesIO()
    try:
        image.save(buf, format=image.format or "PNG")
    except Exception:
        image.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# OpenAI-based image generation client
# ---------------------------------------------------------------------------

class ImageGenApi:
    """Wrapper around the OpenAI Images API for image generation.

    Uses ``client.images.generate()`` to create images from text prompts.

    Note:
        The standard OpenAI image generation endpoint does not accept reference
        images.  If you need reference-image-conditioned generation, consider
        using a multimodal chat completion model (e.g. GPT-4o) that can accept
        images in its messages and output image content, or use an image-edit
        endpoint.  This class focuses on text-to-image generation to keep the
        open-source pipeline simple.
    """

    def __init__(self, model: str | None = None, size: str = "1024x1024",
                 quality: str = "auto", timeout: int = 600):
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set")

        client_kwargs = {"api_key": api_key, "timeout": timeout}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)

        self.model = model or os.environ.get("IMAGE_GEN_MODEL", "gpt-image-1")
        self.size = size
        self.quality = quality

    def call_data_eval(self, prompt: str, ref_image_urls: list[str] | None = None,
                       seed: int | None = None, resize_action=None):
        """Generate an image and return raw PNG bytes.

        The ``ref_image_urls`` parameter is accepted for interface compatibility
        with the rest of the pipeline but is **not** sent to the API (the
        standard images endpoint does not support reference images).  A log
        message is emitted when reference images are provided but ignored.

        Args:
            prompt: The text prompt for image generation.
            ref_image_urls: Reference image paths (logged but not used).
            seed: Not used; kept for interface compatibility.
            resize_action: Not used; kept for interface compatibility.

        Returns:
            Raw PNG image bytes, or raises on failure.
        """
        if ref_image_urls:
            print(f"  [info] {len(ref_image_urls)} reference image(s) provided but "
                  "the text-to-image endpoint does not support them; they will be ignored.")

        result = self.client.images.generate(
            model=self.model,
            prompt=prompt,
            n=1,
            size=self.size,
            quality=self.quality,
        )

        image_data = result.data[0]

        if hasattr(image_data, 'b64_json') and image_data.b64_json:
            return base64.b64decode(image_data.b64_json)

        if hasattr(image_data, 'url') and image_data.url:
            import requests
            resp = requests.get(image_data.url, timeout=120)
            resp.raise_for_status()
            return resp.content

        raise RuntimeError("API response contained neither b64_json nor url")


# ---------------------------------------------------------------------------
# Recaption / reference-image helpers
# ---------------------------------------------------------------------------

def extract_instruction_from_recaption(recaption_text: str) -> str:
    """Extract the content inside ``<recaption>`` tags, excluding ``<think>`` blocks.

    Falls back to ``<Instruction>`` tags, then ``<result><Instruction>`` nesting,
    and finally returns the text with all XML-like tags stripped.
    """
    if not recaption_text:
        return ""

    # Primary format: <recaption>...</recaption>
    match = re.search(r"<recaption>(.*?)</recaption>", recaption_text, re.DOTALL)
    if match:
        content = re.sub(r"<[^>]+>", "", match.group(1)).strip()
        if content:
            return content

    # Legacy: <Instruction>...</Instruction>
    match = re.search(r"<Instruction>(.*?)</Instruction>", recaption_text, re.DOTALL)
    if match:
        content = re.sub(r"<[^>]+>", "", match.group(1)).strip()
        if content:
            return content

    # Legacy: <result><Instruction>...</Instruction></result>
    match = re.search(
        r"<result>.*?<Instruction>(.*?)</Instruction>.*?</result>",
        recaption_text, re.DOTALL,
    )
    if match:
        content = re.sub(r"<[^>]+>", "", match.group(1)).strip()
        if content:
            return content

    # Last resort: strip <think> blocks then all remaining tags
    text = re.sub(r"<think>.*?</think>", "", recaption_text, flags=re.DOTALL)
    cleaned = re.sub(r"<[^>]+>", "", text).strip()
    return cleaned


def get_reference_images(ip_index, intermediate_dir: str) -> list[str]:
    """Return paths to reference images (image_1, image_2) for a given IP index.

    Args:
        ip_index: The IP index (used as a subdirectory name).
        intermediate_dir: Root intermediate directory from Stage 2.

    Returns:
        A list of existing image file paths (at most 2).
    """
    ref_images = []
    ip_dir = os.path.join(intermediate_dir, str(ip_index))

    if not os.path.exists(ip_dir):
        return ref_images

    for idx in [1, 2]:
        for ext in [".jpg", ".jpeg", ".png", ".gif"]:
            img_path = os.path.join(ip_dir, f"image_{idx}{ext}")
            if os.path.exists(img_path):
                ref_images.append(img_path)
                break

    return ref_images


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_trajectory(trajectory_file: str, api: ImageGenApi,
                       output_dir: str, intermediate_dir: str,
                       n_generation: int = 1) -> dict:
    """Process a single trajectory JSON file and generate image(s).

    Args:
        trajectory_file: Path to the trajectory JSON.
        api: An ImageGenApi instance.
        output_dir: Directory to write generated images.
        intermediate_dir: Stage 2 intermediate directory (for reference images).
        n_generation: Number of images to generate per IP.

    Returns:
        A result dict with keys: ip_index, ip_name, status, prompt, generated_images.
    """
    try:
        with open(trajectory_file, "r", encoding="utf-8") as f:
            trajectory = json.load(f)

        ip_index = trajectory.get("ip_index", "")
        ip_name = trajectory.get("ip_name", "")

        recaption_text = trajectory.get("recaption", "")
        if not recaption_text:
            turns = trajectory.get("turns", [])
            if turns:
                recaption_text = turns[-1].get("recaption", "")

        if not recaption_text:
            print(f"  [warn] No recaption found in {trajectory_file}")
            return {"ip_index": ip_index, "ip_name": ip_name,
                    "status": "no_recaption", "generated_images": []}

        prompt = extract_instruction_from_recaption(recaption_text)
        if not prompt:
            print(f"  [warn] Failed to extract instruction from recaption")
            return {"ip_index": ip_index, "ip_name": ip_name,
                    "status": "no_instruction", "generated_images": []}

        print(f"\n  Processing IP: {ip_name} (index: {ip_index})")
        print(f"  Prompt (first 200 chars): {prompt[:200]}...")

        ref_images = get_reference_images(ip_index, intermediate_dir)
        print(f"  Reference images: {len(ref_images)}")

        generated_images: list[str | None] = []

        for gen_idx in range(n_generation):
            output_file = os.path.join(output_dir, f"{ip_index}_{gen_idx}.png")

            if os.path.exists(output_file):
                print(f"  Image {gen_idx} already exists: {output_file}")
                generated_images.append(output_file)
                continue

            max_retries = 5
            success = False

            for attempt in range(1, max_retries + 1):
                try:
                    img_bytes = api.call_data_eval(
                        prompt=prompt,
                        ref_image_urls=ref_images,
                    )
                    if img_bytes is None:
                        raise RuntimeError("API returned None")

                    img = Image.open(BytesIO(img_bytes)).convert("RGB")
                    img.save(output_file)

                    print(f"  [ok] Generated image {gen_idx}: {output_file}")
                    generated_images.append(output_file)
                    success = True
                    break

                except Exception as e:
                    print(f"  [warn] Error generating image {gen_idx} "
                          f"(attempt {attempt}/{max_retries}): {e}")
                    if attempt < max_retries:
                        time.sleep(2 * attempt)

            if not success:
                print(f"  [error] Failed to generate image {gen_idx} "
                      f"after {max_retries} attempts")
                generated_images.append(None)

        return {
            "ip_index": ip_index,
            "ip_name": ip_name,
            "status": "success",
            "prompt": prompt,
            "generated_images": generated_images,
        }

    except Exception as e:
        print(f"  [error] Error processing trajectory {trajectory_file}: {e}")
        traceback.print_exc()
        idx = trajectory.get("ip_index", "") if "trajectory" in dir() else ""
        name = trajectory.get("ip_name", "") if "trajectory" in dir() else ""
        return {
            "ip_index": idx,
            "ip_name": name,
            "status": "error",
            "error": str(e),
            "generated_images": [],
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(name)s] %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Stage 3: Generate images from Stage 2 recaption trajectories.",
    )
    parser.add_argument(
        "--input_dir", type=str, default=None,
        help="Directory containing *_trajectory.json files from Stage 2.",
    )
    parser.add_argument(
        "--intermediate_dir", type=str, default=None,
        help="Stage 2 intermediate directory (default: <input_dir>/intermediate).",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to write generated images.",
    )
    parser.add_argument(
        "--n_generation", type=int, default=1,
        help="Number of images to generate per IP (default: 1).",
    )
    parser.add_argument(
        "--trajectory_file", type=str, default=None,
        help="Process a single trajectory file instead of the whole input_dir.",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Image generation model name (default: env IMAGE_GEN_MODEL or 'gpt-image-1').",
    )
    parser.add_argument(
        "--size", type=str, default="1024x1024",
        help="Generated image size, e.g. '1024x1024' (default: 1024x1024).",
    )
    args = parser.parse_args()

    if not args.input_dir and not args.trajectory_file:
        parser.error("--input_dir is required when --trajectory_file is not specified")

    if args.intermediate_dir:
        intermediate_dir = args.intermediate_dir
    elif args.input_dir:
        intermediate_dir = os.path.join(args.input_dir, "intermediate")
    else:
        intermediate_dir = os.path.join(os.path.dirname(args.trajectory_file), "intermediate")

    os.makedirs(args.output_dir, exist_ok=True)

    api = ImageGenApi(model=args.model, size=args.size)

    print(f"Input directory:        {args.input_dir or '(single file mode)'}")
    print(f"Intermediate directory: {intermediate_dir}")
    print(f"Output directory:       {args.output_dir}")
    print(f"Model:                  {api.model}")

    if args.trajectory_file:
        trajectory_files = [Path(args.trajectory_file)]
    else:
        trajectory_files = sorted(Path(args.input_dir).glob("*_trajectory.json"))

    if not trajectory_files:
        print(f"[warn] No trajectory files found in {args.input_dir}")
        return

    print(f"Found {len(trajectory_files)} trajectory file(s)")
    print(f"Processing mode: one JSON -> {args.n_generation} image(s)\n")

    summary_file = os.path.join(args.output_dir, "generation_summary.json")
    results: list[dict] = []
    if os.path.exists(summary_file):
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"Loaded {len(results)} existing results from summary file")
        except Exception as e:
            print(f"[warn] Failed to load existing summary: {e}")
            results = []

    processed_indices = {
        r.get("ip_index")
        for r in results
        if r.get("status") == "success" and r.get("generated_images")
    }

    for idx, traj_file in enumerate(trajectory_files, 1):
        traj_path = str(traj_file)

        try:
            with open(traj_path, "r", encoding="utf-8") as f:
                preview = json.load(f)
            ip_index = preview.get("ip_index", "")
            ip_name = preview.get("ip_name", "")
        except Exception as e:
            print(f"\n[{idx}/{len(trajectory_files)}] [warn] "
                  f"Failed to read {traj_file.name}: {e}")
            continue

        output_file = os.path.join(args.output_dir, f"{ip_index}_0.png")
        if ip_index in processed_indices and os.path.exists(output_file):
            print(f"\n[{idx}/{len(trajectory_files)}] Skipping {ip_name} "
                  f"(index: {ip_index}) - already processed")
            continue

        print(f"\n{'=' * 60}")
        print(f"[{idx}/{len(trajectory_files)}] Processing: {ip_name} (index: {ip_index})")
        print(f"File: {traj_file.name}")
        print(f"{'=' * 60}")

        result = process_trajectory(
            traj_path, api, args.output_dir, intermediate_dir,
            n_generation=args.n_generation,
        )

        existing_idx = next(
            (i for i, r in enumerate(results) if r.get("ip_index") == ip_index),
            None,
        )
        if existing_idx is not None:
            results[existing_idx] = result
        else:
            results.append(result)

        try:
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"  [warn] Failed to save summary: {e}")

        if result.get("status") == "success":
            generated_count = sum(
                1 for img in result.get("generated_images", []) if img is not None
            )
            print(f"  [ok] Successfully generated {generated_count} image(s)")
        else:
            print(f"  [error] Failed: {result.get('status')} - "
                  f"{result.get('error', 'Unknown error')}")

    success_count = sum(1 for r in results if r.get("status") == "success")
    total_images = sum(len(r.get("generated_images", [])) for r in results)
    successful_images = sum(
        1 for r in results for img in r.get("generated_images", []) if img is not None
    )

    print(f"\n{'=' * 60}")
    print("Final Generation Summary:")
    print(f"  Total trajectories processed: {len(results)}")
    print(f"  Successful: {success_count}")
    print(f"  Total images generated: {successful_images}/{total_images}")
    print(f"  Summary saved to: {summary_file}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
