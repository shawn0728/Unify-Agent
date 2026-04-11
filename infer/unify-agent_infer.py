# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
Multi-turn inference script for Unify-Agent (single GPU).
Integrates recaption generation and image synthesis using the BAGEL model
with multi-turn dialogue and tool-augmented search.
"""

import os
import sys
import json
import argparse
from copy import deepcopy
import torch
from PIL import Image
from typing import Dict, List, Optional, Tuple, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_loader import load_model_and_inferencer
from data.data_utils import pil_img2rgb
from infer_utils import (
    set_seed,
    _append_debug_log,
    call_gemini3_flash_api,
    call_gemini3_flash_with_image,
    detect_image_format_from_bytes,
    judge_image_quality,
    extract_tool_call,
    extract_recaption_content,
    has_recaption_tag,
    normalize_recaption_text,
    load_prompt_template,
    download_image_to_bytes,
    execute_text_search,
    execute_search_image,
    download_and_judge_search_images,
    get_tools_definition,
    build_initial_prompt,
    build_stage3_prompt,
)


class MultiTurnInferencer:
    """Multi-turn inferencer for dialogue and image generation."""
    
    def __init__(self, inferencer, output_dir: str):
        self.inferencer = inferencer
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "intermediate"), exist_ok=True)

    def _update_context_text_safe(self, text: str, gen_context: Dict[str, Any]) -> Dict[str, Any]:
        """Update text context under bf16 autocast for dtype consistency."""
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            return self.inferencer.update_context_text(text, gen_context)

    def _update_context_image_safe(
        self,
        image: Image.Image,
        gen_context: Dict[str, Any],
        vae: bool,
        vit: bool,
    ) -> Dict[str, Any]:
        """Update image context under bf16 autocast for dtype consistency."""
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            return self.inferencer.update_context_image(image, gen_context, vae=vae, vit=vit)
    
    def text_to_text(
        self,
        prompt: str,
        gen_context: Optional[Dict[str, Any]] = None,
        max_length: int = 512,
        do_sample: bool = True,
        temperature: float = 0.7,
    ) -> Tuple[str, Dict[str, Any]]:
        """Text-to-text generation (supports reusing gen_context)."""
        if gen_context is None:
            gen_context = self.inferencer.init_gen_context()
        gen_context = self._update_context_text_safe(prompt, gen_context)
        
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            output_text = self.inferencer.gen_text(
                gen_context,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
            )
        gen_context = self._update_context_text_safe(output_text, gen_context)
        return output_text, gen_context
    
    def image_text_to_text(
        self,
        images: List[Image.Image],
        prompt: str,
        gen_context: Optional[Dict[str, Any]] = None,
        max_length: int = 1024,
        do_sample: bool = False,
        temperature: float = 0.3,
    ) -> Tuple[str, Dict[str, Any]]:
        """Image+text to text generation (for recaption, supports reusing gen_context)."""
        if gen_context is None:
            gen_context = self.inferencer.init_gen_context()
        
        for image in images:
            gen_context = self._update_context_image_safe(
                image, gen_context, vae=True, vit=True
            )
        
        gen_context = self._update_context_text_safe(prompt, gen_context)
        
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            output_text = self.inferencer.gen_text(
                gen_context,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
            )
        gen_context = self._update_context_text_safe(output_text, gen_context)
        return output_text, gen_context
    
    def image_text_to_image(
        self,
        images: List[Image.Image],
        prompt: str,
        gen_context: Optional[Dict[str, Any]] = None,
        add_ref_images_to_context: bool = True,
        add_prompt_to_context: bool = True,
        image_shapes: Tuple[int, int] = (1024, 1024),
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 2.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        timestep_shift: float = 3.0,
        num_timesteps: int = 50,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "text_channel",
        seed: int = 0,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """Image+text to image generation (supports shared gen_context)."""
        set_seed(seed)
        
        if not images:
            raise ValueError("No reference images provided")

        if gen_context is None:
            gen_context = self.inferencer.init_gen_context()

        # Build CFG contexts from existing history
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        # ref_images may already be in context from Stage 3; skip if not needed
        if add_ref_images_to_context:
            for img in images:
                gen_context = self._update_context_image_safe(
                    pil_img2rgb(img), gen_context, vae=True, vit=True
                )
                cfg_text_context = deepcopy(gen_context)

        # prompt may already be in context from Stage 3; skip if not needed
        if add_prompt_to_context and prompt:
            gen_context = self._update_context_text_safe(prompt, gen_context)
            cfg_img_context = self._update_context_text_safe(prompt, cfg_img_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            generated_image = self.inferencer.gen_image(
                image_shapes,
                gen_context,
                cfg_text_precontext=cfg_text_context,
                cfg_img_precontext=cfg_img_context,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=list(cfg_interval),
                timestep_shift=timestep_shift,
                num_timesteps=num_timesteps,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
            )

        # Write generated image into context
        gen_context = self._update_context_image_safe(
            pil_img2rgb(generated_image), gen_context, vae=True, vit=False
        )
        return generated_image, gen_context

    def tmi2i_image_text_to_image(
        self,
        image_paths: List[str],
        prompt: str,
        fallback_single: bool = False,
        think: bool = False,
        image_shapes: Tuple[int, int] = (1024, 1024),
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 2.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        timestep_shift: float = 3.0,
        num_timesteps: int = 50,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "text_channel",
        max_think_token_n: int = 1024,
        do_sample: bool = False,
        text_temperature: float = 0.3,
        seed: int = 0,
    ) -> Image.Image:
        """Stage 4 image generation using tmi2i_infer.py logic.

        Unlike image_text_to_image():
        - Does not reuse multi-turn gen_context
        - Only injects ref images (image_1, image_2) + prompt (recaption)
        - Directly calls inferencer.interleave_inference
        """
        set_seed(seed)

        # Load up to 2 images; resizing is handled by inferencer.interleave_inference.
        images: List[Image.Image] = []
        for i, p in enumerate((image_paths or [])[:2]):
            if p and os.path.exists(p):
                img = Image.open(p).convert("RGB")
                images.append(img)
            else:
                print(f"  Warning: Image {i+1} path invalid or not found: {p}")

        if not images:
            raise ValueError("At least one valid image is required")

        if fallback_single and len(images) > 1:
            images = [images[0]]
            print("  Using fallback_single mode, only image_1 as reference")

        # Input list: ref images (image_1/image_2) + prompt
        input_list: List[Any] = []
        input_list.extend(images)
        input_list.append(prompt)

        inference_hyper = dict(
            think=think,
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

        output_list = self.inferencer.interleave_inference(
            input_lists=input_list,
            **inference_hyper,
        )

        for out in output_list:
            if isinstance(out, Image.Image):
                return out

        raise ValueError("No image generated")
    
    def process_ip(
        self,
        ip_data: Dict,
        ip_index: str,
        reference_images: Optional[List[str]] = None,
        execute_tools: bool = False,
        seed: int = 42,
        stage1_do_sample: bool = True,
        stage1_temperature: float = 0.7,
        stage2_do_sample: bool = True,
        stage2_temperature: float = 0.7,
        stage1_max_length: int = 1024,
        stage2_max_length: int = 768,
        ip_output_dir: Optional[str] = None,
        max_search_turns: int = 6,
    ) -> Dict:
        """Process a single IP entry through multi-turn inference and image generation.
        
        Args:
            ip_output_dir: Optional per-IP output directory; defaults to self.output_dir
            max_search_turns: Max turns for the search phase
        """
        out_dir = ip_output_dir if ip_output_dir is not None else self.output_dir
        ip_name = ip_data.get('ip_name', '')
        image_prompt = ip_data.get('image_prompt', '')
        language = ip_data.get('language', 'zh')
        country = ip_data.get('country', '')
        
        print(f"\n{'='*60}")
        print(f"Processing: {ip_name or '(no IP name)'} (index: {ip_index})")
        print(f"Image Prompt: {image_prompt}")
        print(f"{'='*60}\n")
        
        ip_intermediate_dir = os.path.join(out_dir, "intermediate", str(ip_index))
        os.makedirs(ip_intermediate_dir, exist_ok=True)
        
        trajectory = {
            'ip_index': ip_index,
            'ip_name': ip_name,
            'image_prompt': image_prompt,
            'language': language,
            'country': country,
            'turns': [],
            'full_response': []
        }

        def persist_trajectory():
            """Persist trajectory to disk regardless of success/failure."""
            full_response = []
            for turn in trajectory.get('turns', []):
                full_response.append({
                    'turn': turn.get('turn', 0),
                    'input': turn.get('input', ''),
                    'response_text': turn.get('response_text', ''),
                    'tool_output': turn.get('tool_output', None),
                    'tool_output_full': turn.get('tool_output_full', None),
                })
            trajectory['full_response'] = full_response
            output_file = os.path.join(out_dir, f"{ip_index}_trajectory.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(trajectory, f, ensure_ascii=False, indent=2)
            print(f"\n✅ Trajectory saved: {output_file}")
        
        shared_gen_context = self.inferencer.init_gen_context()
        
        # ====== Dynamic Search Phase (replaces fixed Stage 1 + Stage 2) ======
        MAX_SEARCH_TURNS = max_search_turns
        turn_counter = 0
        text_search_result = ""
        text_search_full_output = None
        image_search_result = ""
        image_search_full_output = None
        downloaded_images = []
        image_judge_results = []
        search_phase_complete = False

        print(f"Search Phase: Dynamic multi-turn search (max {MAX_SEARCH_TURNS} turns)")
        initial_prompt = build_initial_prompt(image_prompt, ip_name, country)
        current_prompt = initial_prompt

        while turn_counter < MAX_SEARCH_TURNS and not search_phase_complete:
            print(f"\n--- Search Turn {turn_counter} ---")

            if turn_counter == 0:
                cur_max_length = stage1_max_length
                cur_do_sample = stage1_do_sample
                cur_temperature = stage1_temperature
            else:
                cur_max_length = stage2_max_length
                cur_do_sample = stage2_do_sample
                cur_temperature = stage2_temperature

            turn_response, shared_gen_context = self.text_to_text(
                prompt=current_prompt,
                gen_context=shared_gen_context,
                max_length=cur_max_length,
                do_sample=cur_do_sample,
                temperature=cur_temperature,
            )

            print(f"  Response (first 200 chars): {turn_response[:200]}...")

            tool_call = extract_tool_call(turn_response)

            if not tool_call:
                lower_resp = turn_response.lower()
                if ("text_search" in lower_resp) or ("<instruction" in lower_resp):
                    guessed_q = (ip_name.split("//")[0].strip() if ip_name else "") or image_prompt.strip()[:128]
                    tool_call = {
                        "name": "text_search",
                        "parameters": {"q": guessed_q, "hl": language, "top_k": 5},
                    }
                    _append_debug_log(
                        run_id="post-fix",
                        hypothesis_id="H6",
                        location="multi_turn_infer.py:process_ip:search_fallback",
                        message="fallback synthesized text_search tool call",
                        data={
                            "turn": turn_counter,
                            "guessed_q": guessed_q,
                            "language": language,
                            "response_preview": turn_response[:300],
                        },
                    )
                elif ("search_image" in lower_resp) or ("image query" in lower_resp):
                    guessed_q = (ip_name.split("//")[0].strip() if ip_name else "") or image_prompt.strip()[:128]
                    tool_call = {
                        "name": "search_image",
                        "parameters": {"q": guessed_q, "hl": language, "num": 5},
                    }
                    _append_debug_log(
                        run_id="post-fix",
                        hypothesis_id="H6",
                        location="multi_turn_infer.py:process_ip:search_fallback",
                        message="fallback synthesized search_image tool call",
                        data={
                            "turn": turn_counter,
                            "guessed_q": guessed_q,
                            "language": language,
                            "response_preview": turn_response[:300],
                        },
                    )

            if not tool_call:
                print(f"  ⚠️ No valid tool call detected at turn {turn_counter}. Ending search phase.")
                trajectory['turns'].append({
                    'turn': turn_counter,
                    'stage': 1 if turn_counter == 0 else 2,
                    'input': current_prompt,
                    'response_text': turn_response,
                    'tool_output': None,
                    'tool_output_full': None,
                })
                turn_counter += 1
                break

            # ---- Handle text_search ----
            if tool_call.get('name') == 'text_search':
                print(f"  Tool call: text_search (query: {tool_call['parameters'].get('q', '')[:60]})")
                tool_output = ""
                tool_output_full = None

                if execute_tools:
                    tool_output, tool_output_full = execute_text_search(
                        tool_call['parameters'], use_summary=True
                    )
                    text_search_result = tool_output
                    text_search_full_output = tool_output_full
                    print(f"  ✅ Text search completed")
                else:
                    tool_output = f"[Tool execution skipped] Query: {tool_call['parameters'].get('q', '')}"
                    text_search_result = tool_output

                trajectory['turns'].append({
                    'turn': turn_counter,
                    'stage': 1,
                    'input': current_prompt,
                    'response_text': turn_response,
                    'tool_output': tool_output if tool_output else None,
                    'tool_output_full': tool_output_full,
                })

                observation = f"<observation>\n{tool_output}\n</observation>\n\n" if tool_output else ""
                current_prompt = f"""{observation}Great, now you have background knowledge about this IP. Based on what you learned, please search for reference images that capture the visual characteristics of this IP/character. Use the information from the text search to craft a more precise image query.

Call `search_image` to find reference visuals:
<tool_call>
{{"name": "search_image", "arguments": {{"q": "your refined image query based on what you learned", "hl": "{language}", "num": 5}}}}
</tool_call>"""

                turn_counter += 1
                continue

            # ---- Handle search_image ----
            elif tool_call.get('name') == 'search_image':
                print(f"  Tool call: search_image (query: {tool_call['parameters'].get('q', '')[:60]})")
                tool_output = ""
                tool_output_full = None
                current_downloaded = []
                current_judge_results = []

                if execute_tools:
                    tool_output, search_result = execute_search_image(tool_call['parameters'])
                    tool_output_full = search_result
                    image_search_result = tool_output
                    image_search_full_output = search_result

                    if search_result and 'images' in search_result:
                        current_downloaded, current_judge_results = download_and_judge_search_images(
                            search_result, ip_name, ip_intermediate_dir, image_prompt=image_prompt
                        )
                else:
                    tool_output = f"[Tool execution skipped] Query: {tool_call['parameters'].get('q', '')}"

                turn_data = {
                    'turn': turn_counter,
                    'stage': 2,
                    'input': current_prompt,
                    'response_text': turn_response,
                    'tool_output': tool_output if tool_output else None,
                    'tool_output_full': tool_output_full,
                }
                if current_judge_results:
                    turn_data['image_judge_results'] = current_judge_results
                trajectory['turns'].append(turn_data)

                if current_downloaded:
                    downloaded_images = current_downloaded
                    image_judge_results = current_judge_results
                    search_phase_complete = True
                elif not execute_tools:
                    search_phase_complete = True
                else:
                    observation = f"<observation>\n{tool_output}\n</observation>\n\n" if tool_output else ""
                    current_prompt = f"""{observation}The image search returned results but no high-quality reference images could be downloaded. Please try a different, more specific search query to find better reference images for this IP/character.

You can also call `text_search` to gather more specific information first, then search for images again.

<tool_call>
{{"name": "search_image", "arguments": {{"q": "your refined and more specific image query", "hl": "{language}", "num": 5}}}}
</tool_call>"""

                turn_counter += 1
                continue

            # ---- Unknown tool ----
            else:
                print(f"  ⚠️ Unknown tool call: {tool_call.get('name')}. Ending search phase.")
                trajectory['turns'].append({
                    'turn': turn_counter,
                    'stage': 2,
                    'input': current_prompt,
                    'response_text': turn_response,
                    'tool_output': None,
                    'tool_output_full': None,
                })
                turn_counter += 1
                break

        if reference_images:
            downloaded_images = reference_images[:2]
            print(f"  Using provided reference images: {downloaded_images}")

        print(f"\n  Search phase ended after {turn_counter} turn(s). "
              f"Downloaded images: {len(downloaded_images)}")
        
        # Stage 3: Generate recaption
        print("\nStage 3: Generate Recaption")
        
        if not downloaded_images:
            print("  ❌ No reference images available. Cannot generate recaption.")
            trajectory['recaption'] = ""
            persist_trajectory()
            return trajectory
        
        ref_images_pil = []
        for img_path in downloaded_images:
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                ref_images_pil.append(img)
                print(f"  ✅ Loaded reference image: {img_path}")
            else:
                print(f"  ⚠️ Image not found: {img_path}")
        
        if not ref_images_pil:
            print("  ❌ No valid reference images loaded. Cannot generate recaption.")
            trajectory['recaption'] = ""
            persist_trajectory()
            return trajectory
        
        images_to_process = ref_images_pil[:2]
        stage3_observation = f"<observation>\n{image_search_result}\n</observation>\n\n" if image_search_result else ""
        recaption_system_prompt = load_prompt_template(country=country)
        stage3_prompt = build_stage3_prompt(
            image_prompt,
            text_search_result,
            language,
            num_images=len(images_to_process),
        )
        stage3_prompt = f"{recaption_system_prompt}\n\n{stage3_observation}{stage3_prompt}"
        
        print(f"  Processing {len(images_to_process)} reference image(s) simultaneously...")
        
        turn_2_response, shared_gen_context = self.image_text_to_text(
            images=images_to_process,
            prompt=stage3_prompt,
            gen_context=shared_gen_context,
            max_length=1024,
            do_sample=False,
            temperature=0.3,
        )
        
        # Extract and normalize recaption
        normalized_response = normalize_recaption_text(turn_2_response, language=language)
        recaption = extract_recaption_content(normalized_response)
        
        print(f"Response (first 200 chars): {turn_2_response[:200]}...")
        trajectory['recaption'] = recaption
        
        trajectory['turns'].append({
            'turn': 2,
            'stage': 3,
            'input': stage3_prompt,
            'response_text': turn_2_response,
            'tool_output': None
        })
        
        # Stage 4: Generate image
        print("\nStage 4: Generate Image")
        
        if not recaption:
            print("  ❌ No recaption available. Cannot generate image.")
            trajectory['generated_image'] = None
            persist_trajectory()
            return trajectory
        
        try:
            image_paths_for_generation = downloaded_images[:2]
            print(
                f"  Using {len(image_paths_for_generation)} reference image(s) for generation (no shared context)..."
            )

            generated_image = self.tmi2i_image_text_to_image(
                image_paths=image_paths_for_generation,
                prompt=recaption,
                fallback_single=False,
                think=False,
                image_shapes=(1024, 1024),
                cfg_text_scale=4.0,
                cfg_img_scale=2.0,
                cfg_interval=(0.0, 1.0),
                timestep_shift=3.0,
                num_timesteps=50,
                cfg_renorm_min=0.0,
                cfg_renorm_type="text_channel",
                max_think_token_n=1024,
                do_sample=False,
                text_temperature=0.3,
                seed=seed,
            )
            
            output_image_path = os.path.join(out_dir, f"{ip_index}_generated.png")
            generated_image.save(output_image_path)
            print(f"  ✅ Generated image saved: {output_image_path}")
            
            trajectory['generated_image'] = output_image_path
            
        except Exception as e:
            print(f"  ❌ Error generating image: {e}")
            import traceback
            traceback.print_exc()
            trajectory['generated_image'] = None
        
        persist_trajectory()
        
        return trajectory


def main():
    parser = argparse.ArgumentParser(description="BAGEL Multi-turn Inference")
    parser.add_argument("--model_path", type=str, default="csfufu/Unify-Agent", help="Path or HuggingFace repo id of the model (default: csfufu/Unify-Agent)")
    parser.add_argument("--base_model_path", type=str, default=None, help="Base model directory (config/tokenizer/ae loaded from here)")
    parser.add_argument("--ema_path", type=str, default=None, help="EMA weights path (overrides model_path/ema.safetensors)")
    parser.add_argument("--cast_ema_to_bfloat16", action="store_true", help="Cast EMA weights to bfloat16 before loading (with cache)")
    parser.add_argument("--ema_bf16_cache_path", type=str, default=None, help="EMA bfloat16 cache file path (optional)")
    parser.add_argument("--mode", type=int, default=1, choices=[1, 2, 3], 
                       help="Loading mode: 1=full precision, 2=NF4, 3=INT8")
    parser.add_argument("--ip_data", type=str, required=True, help="IP data JSON file path or JSON string")
    parser.add_argument("--ip_index", type=str, default=None, help="IP index to process (if not specified, processes all entries)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (if used with --model_name, becomes output_base/model_name)")
    parser.add_argument("--output_base", type=str, default="./output",
                       help="Base output directory (used with --model_name)")
    parser.add_argument("--model_name", type=str, default=None,
                       help="Model name (used to construct output_dir = output_base/model_name)")
    parser.add_argument("--reference_images", type=str, nargs='+', default=None, 
                       help="Reference image paths (optional)")
    parser.add_argument("--execute_tools", action="store_true", 
                       help="Execute search tools (requires API keys)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--stage1_do_sample", dest="stage1_do_sample", action="store_true", default=True, help="Enable sampling for Stage 1 (default: True)")
    parser.add_argument("--stage1_no_sample", dest="stage1_do_sample", action="store_false", help="Disable sampling for Stage 1 (deterministic)")
    parser.add_argument("--stage1_temperature", type=float, default=0.7, help="Stage 1 temperature (default: 0.7)")
    parser.add_argument("--stage1_max_length", type=int, default=1024, help="Stage 1 max generation tokens (default: 1024)")
    parser.add_argument("--stage2_do_sample", dest="stage2_do_sample", action="store_true", default=True, help="Enable sampling for Stage 2 (default: True)")
    parser.add_argument("--stage2_no_sample", dest="stage2_do_sample", action="store_false", help="Disable sampling for Stage 2 (deterministic)")
    parser.add_argument("--stage2_temperature", type=float, default=0.7, help="Stage 2 temperature (default: 0.7)")
    parser.add_argument("--stage2_max_length", type=int, default=768, help="Stage 2 max generation tokens (default: 768)")
    parser.add_argument(
        "--allow_batch_ip",
        action="store_true",
        help="Allow batch processing of all samples when --ip_index is not specified",
    )
    parser.add_argument(
        "--per_ip_subdir",
        action="store_true",
        help="Use per-IP subdirectory output_dir/{ip_key}/ for results",
    )
    parser.add_argument(
        "--max_search_turns",
        type=int,
        default=6,
        help="Max search turns (default: 6)",
    )
    
    args = parser.parse_args()
    
    if args.model_name:
        args.output_dir = os.path.join(args.output_base, args.model_name)
        print(f"📂 Output dir (from --model_name): {args.output_dir}")
    if not args.output_dir:
        raise ValueError("Must specify --output_dir or --model_name")
    
    set_seed(args.seed)
    
    print("Loading model...")
    inferencer = load_model_and_inferencer(
        args.model_path,
        mode=args.mode,
        base_model_path=args.base_model_path,
        ema_path=args.ema_path,
        cast_ema_to_bfloat16=args.cast_ema_to_bfloat16,
        ema_bf16_cache_path=args.ema_bf16_cache_path,
    )
    print("Model loaded.")
    
    multi_turn_inferencer = MultiTurnInferencer(inferencer, args.output_dir)
    
    try:
        if os.path.exists(args.ip_data):
            print(f"📖 Loading IP data from: {args.ip_data}")
            with open(args.ip_data, 'r', encoding='utf-8') as f:
                all_ip_data = json.load(f)
            
            if isinstance(all_ip_data, dict):
                if args.ip_index:
                    if args.ip_index in all_ip_data:
                        ip_entry = all_ip_data[args.ip_index]
                        ip_list = [(args.ip_index, ip_entry)]
                    else:
                        print(f"❌ IP index '{args.ip_index}' not found in file")
                        return
                else:
                    if args.allow_batch_ip:
                        ip_list = list(all_ip_data.items())
                        print(f"📋 Found {len(ip_list)} IP entries in file")
                    else:
                        raise ValueError(
                            "For stable debugging, please provide --ip_index to run a single sample. "
                            "Use --allow_batch_ip if you really want batch mode."
                        )
            else:
                raise ValueError("ip_data file format invalid: expected dict format")
        else:
            ip_data = json.loads(args.ip_data)
            ip_index = args.ip_index or "0"
            ip_list = [(ip_index, ip_data)]
    except Exception as e:
        print(f"❌ Error loading IP data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    results = []
    for ip_index, ip_entry in ip_list:
        try:
            ip_data = {
                'ip_name': ip_entry.get('ip_name', ip_entry.get('ip_name_en', ip_entry.get('ip_name_zh', ip_entry.get('p_en_name', ip_entry.get('p_cn_name', ''))))),
                'image_prompt': ip_entry.get('image_prompt', ''),
                'language': ip_entry.get('language', 'zh'),
                'country': ip_entry.get('country', '')
            }
            
            if not ip_data['image_prompt']:
                print(f"⚠️ Skipping {ip_index}: missing image_prompt")
                continue
            
            if not ip_data['ip_name']:
                ip_data['ip_name'] = ip_entry.get('ip_name_zh') or ip_entry.get('ip_name_en') or ip_entry.get('p_cn_name') or ip_entry.get('p_en_name') or ''
            
            display_name = ip_data['ip_name'] or '(prompt-only mode)'
            print(f"\n{'='*60}")
            print(f"Processing IP {ip_index}: {display_name}")
            print(f"{'='*60}")
            
            ip_output_dir = os.path.join(args.output_dir, str(ip_index)) if args.per_ip_subdir else None
            if ip_output_dir:
                os.makedirs(ip_output_dir, exist_ok=True)
            
            reference_images = args.reference_images
            if not reference_images:
                lookup_dir = ip_output_dir or args.output_dir
                intermediate_dir = os.path.join(lookup_dir, "intermediate", str(ip_index))
                if os.path.exists(intermediate_dir):
                    ref_images = []
                    for idx in [1, 2]:
                        for ext in ['.jpg', '.jpeg', '.png']:
                            img_path = os.path.join(intermediate_dir, f"image_{idx}{ext}")
                            if os.path.exists(img_path):
                                ref_images.append(img_path)
                                break
                    if ref_images:
                        reference_images = ref_images
                        print(f"  📁 Found reference images in intermediate directory: {reference_images}")
            
            trajectory = multi_turn_inferencer.process_ip(
                ip_data=ip_data,
                ip_index=ip_index,
                reference_images=reference_images,
                execute_tools=args.execute_tools,
                seed=args.seed,
                stage1_do_sample=args.stage1_do_sample,
                stage1_temperature=args.stage1_temperature,
                stage2_do_sample=args.stage2_do_sample,
                stage2_temperature=args.stage2_temperature,
                stage1_max_length=args.stage1_max_length,
                stage2_max_length=args.stage2_max_length,
                ip_output_dir=ip_output_dir,
                max_search_turns=args.max_search_turns,
            )
            
            results.append({
                'ip_index': ip_index,
                'ip_name': ip_data['ip_name'],
                'status': 'success' if trajectory.get('recaption') and trajectory.get('generated_image') else 'partial',
                'recaption': trajectory.get('recaption', ''),
                'generated_image': trajectory.get('generated_image', '')
            })
            
            print(f"\n✅ Completed IP {ip_index}")
            print(f"Recaption: {trajectory.get('recaption', '')[:200]}...")
            print(f"Generated image: {trajectory.get('generated_image', 'N/A')}")
            
        except Exception as e:
            print(f"\n❌ Error processing IP {ip_index}: {e}")
            import traceback
            traceback.print_exc()
            try:
                error_trajectory = {
                    "ip_index": ip_index,
                    "ip_name": ip_entry.get('ip_name', ip_entry.get('ip_name_en', ip_entry.get('ip_name_zh', 'Unknown'))),
                    "image_prompt": ip_entry.get('image_prompt', ''),
                    "language": ip_entry.get('language', 'zh'),
                    "country": ip_entry.get('country', ''),
                    "turns": [],
                    "full_response": [],
                    "recaption": "",
                    "generated_image": None,
                    "error": str(e),
                }
                error_dir = os.path.join(args.output_dir, str(ip_index)) if args.per_ip_subdir else args.output_dir
                os.makedirs(error_dir, exist_ok=True)
                error_output_file = os.path.join(error_dir, f"{ip_index}_trajectory.json")
                with open(error_output_file, 'w', encoding='utf-8') as f:
                    json.dump(error_trajectory, f, ensure_ascii=False, indent=2)
                print(f"📝 Error trajectory saved: {error_output_file}")
            except Exception as save_err:
                print(f"⚠️ Failed to save error trajectory for {ip_index}: {save_err}")
            results.append({
                'ip_index': ip_index,
                'ip_name': ip_entry.get('ip_name', 'Unknown'),
                'status': 'error',
                'error': str(e)
            })
            continue
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total processed: {len(results)}")
    print(f"  Successful: {sum(1 for r in results if r.get('status') == 'success')}")
    print(f"  Partial: {sum(1 for r in results if r.get('status') == 'partial')}")
    print(f"  Errors: {sum(1 for r in results if r.get('status') == 'error')}")
    print(f"{'='*60}")
    
    summary_file = os.path.join(args.output_dir, "processing_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n📄 Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()