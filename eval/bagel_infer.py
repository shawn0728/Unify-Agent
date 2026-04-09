# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
Text & Image to Image (TI2I) inference script.
Generates edited images given a reference image and an editing instruction.
"""

import argparse
import os
import sys
import random
import numpy as np
import torch
from PIL import Image

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


def image_text_to_image(
    inferencer,
    image_path: str,
    prompt: str,
    think: bool = False,
    cfg_text_scale: float = 4.0,
    cfg_img_scale: float = 2.0,
    cfg_interval: tuple = (0.0, 1.0),
    timestep_shift: float = 3.0,
    num_timesteps: int = 50,
    cfg_renorm_min: float = 0.0,
    cfg_renorm_type: str = "text_channel",
    max_think_token_n: int = 1024,
    do_sample: bool = False,
    text_temperature: float = 0.3,
    seed: int = 0,
):
    """
    Image-and-text to image generation (image editing).

    Args:
        inferencer: InterleaveInferencer instance.
        image_path: Path to the input image.
        prompt: Editing instruction text.
        think: Whether to enable the thinking (chain-of-thought) mode.
        cfg_text_scale: Text classifier-free guidance scale.
        cfg_img_scale: Image classifier-free guidance scale.
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

    image = Image.open(image_path).convert('RGB')

    inference_hyper = dict(
        max_think_token_n=max_think_token_n if think else 1024,
        do_sample=do_sample if think else False,
        text_temperature=text_temperature if think else 0.3,
        cfg_text_scale=cfg_text_scale,
        cfg_img_scale=cfg_img_scale,
        cfg_interval=list(cfg_interval),
        timestep_shift=timestep_shift,
        num_timesteps=num_timesteps,
        cfg_renorm_min=cfg_renorm_min,
        cfg_renorm_type=cfg_renorm_type,
    )

    result = inferencer(image=image, text=prompt, think=think, **inference_hyper)

    return result["image"], result.get("text", None)


def main():
    parser = argparse.ArgumentParser(description="BAGEL Text & Image to Image Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--mode", type=int, default=1, choices=[1, 2, 3],
                       help="Loading mode: 1=full precision, 2=NF4, 3=INT8")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--prompt", type=str, required=True, help="Editing instruction text")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output image")
    parser.add_argument("--think", action="store_true", help="Enable thinking (chain-of-thought) mode")
    parser.add_argument("--cfg_text_scale", type=float, default=4.0, help="Text CFG scale")
    parser.add_argument("--cfg_img_scale", type=float, default=2.0, help="Image CFG scale")
    parser.add_argument("--cfg_interval_start", type=float, default=0.0, help="CFG interval start")
    parser.add_argument("--cfg_interval_end", type=float, default=1.0, help="CFG interval end")
    parser.add_argument("--timestep_shift", type=float, default=3.0, help="Timestep shift")
    parser.add_argument("--num_timesteps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--cfg_renorm_min", type=float, default=0.0, help="CFG renormalization minimum")
    parser.add_argument("--cfg_renorm_type", type=str, default="text_channel",
                       choices=["global", "channel", "text_channel"],
                       help="CFG renormalization type")
    parser.add_argument("--max_think_token_n", type=int, default=1024, help="Max thinking tokens")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling for text generation")
    parser.add_argument("--text_temperature", type=float, default=0.3, help="Text sampling temperature")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (0 to disable)")
    parser.add_argument("--save_thinking", type=str, default=None, help="Save thinking text to file (optional)")

    args = parser.parse_args()

    print("Loading model...")
    inferencer = load_model_and_inferencer(args.model_path, mode=args.mode)
    print("Model loaded.")

    print(f"\nInput image: {args.image}")
    print(f"Editing instruction: {args.prompt}")
    print("Generating edited image...")

    image, thinking = image_text_to_image(
        inferencer,
        image_path=args.image,
        prompt=args.prompt,
        think=args.think,
        cfg_text_scale=args.cfg_text_scale,
        cfg_img_scale=args.cfg_img_scale,
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

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    image.save(args.output)
    print(f"\nEdited image saved to: {args.output}")

    if thinking and args.save_thinking:
        with open(args.save_thinking, 'w', encoding='utf-8') as f:
            f.write(thinking)
        print(f"Thinking text saved to: {args.save_thinking}")
    elif thinking:
        print(f"\nThinking:\n{thinking}")


if __name__ == "__main__":
    main()
