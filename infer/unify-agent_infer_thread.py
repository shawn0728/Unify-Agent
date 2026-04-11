# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
Multi-turn threaded inference script for Unify-Agent (multi-GPU).
Integrates recaption generation and image synthesis using the BAGEL model
with multi-turn dialogue, tool-augmented search, and multi-GPU parallelism.
"""

import os
import sys
import json
import re
import argparse
import random
from copy import deepcopy
import torch
import subprocess as _subprocess
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
    """Multi-turn inferencer using the BAGEL model for dialogue and image generation."""
    
    def __init__(self, inferencer, output_dir: str):
        self.inferencer = inferencer
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "intermediate"), exist_ok=True)

    def _update_context_text_safe(self, text: str, gen_context: Dict[str, Any]) -> Dict[str, Any]:
        """Update text context under bf16 autocast to avoid dtype mismatch."""
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            return self.inferencer.update_context_text(text, gen_context)

    def _update_context_image_safe(
        self,
        image: Image.Image,
        gen_context: Dict[str, Any],
        vae: bool,
        vit: bool,
    ) -> Dict[str, Any]:
        """Update image context under bf16 autocast to avoid dtype mismatch."""
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
        """Text-to-text generation (supports reusing a shared gen_context)."""
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
        # Write model output back to context for subsequent stages
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
        """Image+text to text generation (for recaption, supports shared gen_context)."""
        if gen_context is None:
            gen_context = self.inferencer.init_gen_context()
        
        # Update image context first, then text context
        # Order aligned with training: ref_images -> user(prompt) -> assistant(response)
        for image in images:
            gen_context = self._update_context_image_safe(
                image, gen_context, vae=True, vit=True
            )
        
        # Update text context
        gen_context = self._update_context_text_safe(prompt, gen_context)
        
        # Generate text
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            output_text = self.inferencer.gen_text(
                gen_context,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
            )
        # Write model output back to context for subsequent stages
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
        """Image+text to image generation (for final image, supports shared gen_context)."""
        set_seed(seed)
        
        if not images:
            raise ValueError("No reference images provided")

        if gen_context is None:
            gen_context = self.inferencer.init_gen_context()

        # Build CFG context using interleave logic, reusing existing history
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        # Inject ref_images into context (optional)
        # In multi-turn shared context flow, ref_images are usually already in context from Stage3
        if add_ref_images_to_context:
            for img in images:
                gen_context = self._update_context_image_safe(
                    pil_img2rgb(img), gen_context, vae=False, vit=True
                )
                cfg_text_context = deepcopy(gen_context)

        # Inject prompt into context (optional)
        # In multi-turn shared context flow, recaption is usually already written as assistant output in Stage3
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

        # Output into context: gen_image
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
    ) -> Tuple[Image.Image, Optional[str]]:
        """Stage 4 image generation using tmi2i_infer.py logic.

        Unlike image_text_to_image():
        - Does not reuse/depend on multi-turn gen_context
        - Only injects two ref images (image_1, image_2) + prompt (recaption)
        - Directly calls inferencer.interleave_inference

        Returns:
            (generated_image, thinking_text)  thinking_text is non-empty only when think=True
        """
        set_seed(seed)

        # Load images (up to 2), keeping original aspect ratio.
        # Actual resize is handled by inferencer.interleave_inference internally
        # via self.vae_transform.resize_transform (consistent with training).
        images: List[Image.Image] = []
        for i, p in enumerate((image_paths or [])[:2]):
            if p and os.path.exists(p):
                img = Image.open(p).convert("RGB")
                images.append(img)
            else:
                print(f"  Warning: image {i+1} path invalid or not found: {p}")

        if not images:
            raise ValueError("At least one valid image is required")

        if fallback_single and len(images) > 1:
            images = [images[0]]
            print("  Using fallback_single mode, only image_1 as reference")

        # Input list: ref images (in order: image_1/image_2) + prompt
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

        generated_image = None
        thinking_text = None
        for out in output_list:
            if isinstance(out, Image.Image):
                generated_image = out
            elif isinstance(out, str):
                thinking_text = out

        if generated_image is None:
            raise ValueError("No image was generated")

        return generated_image, thinking_text
    
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
        think: bool = False,
    ) -> Dict:
        """Process a single IP with multi-turn inference and image generation.
        
        Args:
            ip_output_dir: Optional per-IP output directory; defaults to self.output_dir.
            max_search_turns: Max turns in the search phase.
        """
        out_dir = ip_output_dir if ip_output_dir is not None else self.output_dir
        ip_name = ip_data.get('ip_name', '')
        image_prompt = ip_data.get('image_prompt', '')
        language = ip_data.get('language', 'zh')
        country = ip_data.get('country', '')
        
        print(f"\n{'='*60}")
        print(f"Processing IP: {ip_name} (index: {ip_index})")
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
            """Persist trajectory to disk regardless of success/partial/error."""
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
        
        # Reuse the same gen_context across stages to avoid train/infer mismatch
        shared_gen_context = self.inferencer.init_gen_context()
        
        # ====== Dynamic Search Phase (replaces fixed Stage 1 + Stage 2) ======
        # Support dynamic tool calls (text_search / search_image) in the search phase,
        # allowing multiple text_search turns before image_search, or image_search retries.
        MAX_IMAGE_QUALITY_RETRIES = 10
        image_quality_retries = 0
        MAX_SEARCH_TURNS = max_search_turns + MAX_IMAGE_QUALITY_RETRIES
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
                    guessed_q = ip_name.split("//")[0].strip() if ip_name else ""
                    if not guessed_q:
                        guessed_q = image_prompt.strip()[:128]
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
                    guessed_q = ip_name.split("//")[0].strip() if ip_name else ""
                    if not guessed_q:
                        guessed_q = image_prompt.strip()[:128]
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
                            search_result, ip_name, ip_intermediate_dir
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
                    all_scores = [r.get("score", -1) for r in current_judge_results]
                    both_below_threshold = len(all_scores) >= 2 and all(s < 5 for s in all_scores[:2])
                    any_above_threshold = any(s >= 5 for s in all_scores)

                    if both_below_threshold and not any_above_threshold and image_quality_retries < MAX_IMAGE_QUALITY_RETRIES:
                        image_quality_retries += 1
                        score_summary = ", ".join([f"{s}/10" for s in all_scores[:2]])
                        print(f"  ⚠️ Both image scores below 5 ({score_summary}). "
                              f"Image quality retry {image_quality_retries}/{MAX_IMAGE_QUALITY_RETRIES}.")
                        observation = f"<observation>\n{tool_output}\n</observation>\n\n" if tool_output else ""
                        current_prompt = f"""{observation}The image search returned results, but the quality of the downloaded reference images is too low (scores: {score_summary}). Both images scored below 5/10, which is not sufficient for generating a high-quality output.

Please change your search query and try again with different keywords to find better, higher-quality reference images. Consider:
- Using more specific or descriptive keywords
- Trying alternative names or descriptions
- Adding style or quality-related terms to the query

<tool_call>
{{"name": "search_image", "arguments": {{"q": "your completely different and more specific image query", "hl": "{language}", "num": 10}}}}
</tool_call>"""
                    else:
                        if both_below_threshold and image_quality_retries >= MAX_IMAGE_QUALITY_RETRIES:
                            print(f"  ⚠️ Image quality retries exhausted ({MAX_IMAGE_QUALITY_RETRIES}). "
                                  f"Proceeding with best available images.")
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

        # If reference image paths are provided, use them
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
        
        # Load reference images
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
        
        # Build prompt, requiring simultaneous analysis of all images (up to 2)
        images_to_process = ref_images_pil[:2]
        stage3_observation = f"<observation>\n{image_search_result}\n</observation>\n\n" if image_search_result else ""
        # Load system prompt aligned with stage2_prompt_recaption.py (by country)
        recaption_system_prompt = load_prompt_template(country=country)
        stage3_prompt = build_stage3_prompt(
            image_prompt,
            text_search_result,
            language,
            num_images=len(images_to_process),
        )
        stage3_prompt = f"{recaption_system_prompt}\n\n{stage3_observation}{stage3_prompt}"
        
        # Pass all reference images at once for unified recaption generation
        # BAGEL's image_text_to_text supports multiple images via sequential update_context_image calls
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
            # Stage4 uses tmi2i logic: independent of multi-turn context.
            # Only injects Stage3 recaption + two ref images (image_1/image_2).

            # Use reference image paths selected in Stage2/Stage3 (up to 2)
            image_paths_for_generation = downloaded_images[:2]
            print(
                f"  Using {len(image_paths_for_generation)} reference image(s) for generation (no shared context)..."
            )

            generated_image, thinking_text = self.tmi2i_image_text_to_image(
                image_paths=image_paths_for_generation,
                prompt=recaption,
                fallback_single=False,
                think=think,
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
            
            # Save generated image
            output_image_path = os.path.join(out_dir, f"{ip_index}_generated.png")
            generated_image.save(output_image_path)
            print(f"  ✅ Generated image saved: {output_image_path}")
            
            trajectory['generated_image'] = output_image_path

            if thinking_text:
                trajectory['thinking'] = thinking_text
                thinking_path = os.path.join(out_dir, f"{ip_index}_thinking.txt")
                with open(thinking_path, 'w', encoding='utf-8') as f:
                    f.write(thinking_text)
                print(f"  ✅ Thinking content saved: {thinking_path}")
                print(f"  Thinking (first 200 chars): {thinking_text[:200]}...")
            
        except Exception as e:
            print(f"  ❌ Error generating image: {e}")
            import traceback
            traceback.print_exc()
            trajectory['generated_image'] = None
        
        persist_trajectory()
        
        return trajectory


def _prepare_ip_data_from_entry(ip_index, ip_entry):
    """Extract a standardized ip_data dict from raw ip_entry; returns None if invalid."""
    ip_data = {
        'ip_name': ip_entry.get('ip_name', ip_entry.get('ip_name_en', ip_entry.get('ip_name_zh', ip_entry.get('p_en_name', ip_entry.get('p_cn_name', ''))))),
        'image_prompt': ip_entry.get('image_prompt', ''),
        'language': ip_entry.get('language', 'zh'),
        'country': ip_entry.get('country', '')
    }
    if not ip_data['image_prompt']:
        return None
    if not ip_data['ip_name']:
        ip_data['ip_name'] = (
            ip_entry.get('ip_name_zh') or ip_entry.get('ip_name_en')
            or ip_entry.get('p_cn_name') or ip_entry.get('p_en_name')
            or str(ip_index)
        )
    if not ip_data['ip_name']:
        return None
    return ip_data


def _find_reference_images(ip_index, output_dir, per_ip_subdir, cli_reference_images=None):
    """Find existing reference image paths."""
    if cli_reference_images:
        return cli_reference_images
    ip_output_dir = os.path.join(output_dir, str(ip_index)) if per_ip_subdir else None
    lookup_dir = ip_output_dir or output_dir
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
            return ref_images
    return None


def _claim_next_task(pending_dir):
    """Atomically claim the next task file from pending dir (via os.rename).
    Returns (task_dict, claimed_path) on success, (None, None) if no tasks.
    """
    try:
        fnames = sorted(os.listdir(pending_dir))
    except OSError:
        return None, None
    for fname in fnames:
        if not fname.endswith('.json'):
            continue
        src = os.path.join(pending_dir, fname)
        dst = src + '.claimed'
        try:
            os.rename(src, dst)
        except (OSError, FileNotFoundError):
            continue
        try:
            with open(dst, 'r', encoding='utf-8') as f:
                return json.load(f), dst
        except Exception:
            return None, None
    return None, None


def _run_worker_loop(args):
    """Worker subprocess entry: loads model, loops claiming and processing IPs from file queue.
    CUDA_VISIBLE_DEVICES is set by the parent process in the environment.
    """
    gpu_id = args._worker_gpu_id
    task_dir = args._task_dir
    pending_dir = os.path.join(task_dir, 'pending')
    done_dir = os.path.join(task_dir, 'done')

    print(f"[GPU {gpu_id}] Worker starting, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")

    set_seed(args.seed)

    print(f"[GPU {gpu_id}] Loading model...")
    inferencer = load_model_and_inferencer(
        args.model_path,
        mode=args.mode,
        base_model_path=getattr(args, 'base_model_path', None),
        ema_path=getattr(args, 'ema_path', None),
        cast_ema_to_bfloat16=getattr(args, 'cast_ema_to_bfloat16', False),
        ema_bf16_cache_path=getattr(args, 'ema_bf16_cache_path', None),
    )
    print(f"[GPU {gpu_id}] Model loaded successfully!")

    multi_turn = MultiTurnInferencer(inferencer, args.output_dir)

    processed_count = 0
    while True:
        task, claimed_path = _claim_next_task(pending_dir)
        if task is None:
            break

        ip_index = task['ip_index']
        ip_entry = task['ip_entry']

        category = task.get('category', 'unknown')
        reference_images = task.get('reference_images')

        ip_data = _prepare_ip_data_from_entry(ip_index, ip_entry)
        if ip_data is None:
            print(f"[GPU {gpu_id}] Skipping {ip_index}: missing required fields")
            continue

        ip_output_dir = (
            os.path.join(args.output_dir, category, str(ip_index))
            if args.per_ip_subdir else None
        )
        if ip_output_dir:
            os.makedirs(ip_output_dir, exist_ok=True)

        print(f"\n[GPU {gpu_id}] {'='*50}")
        print(f"[GPU {gpu_id}] Processing IP {ip_index}: {ip_data['ip_name']}")
        print(f"[GPU {gpu_id}] {'='*50}")

        result = {'ip_index': ip_index, 'ip_name': ip_data['ip_name'], 'gpu_id': gpu_id}
        try:
            trajectory = multi_turn.process_ip(
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
            status = 'success' if trajectory.get('recaption') and trajectory.get('generated_image') else 'partial'
            result.update({
                'status': status,
                'recaption': trajectory.get('recaption', ''),
                'generated_image': trajectory.get('generated_image', ''),
            })
            processed_count += 1
            print(f"[GPU {gpu_id}] Completed IP {ip_index} ({status}), total: {processed_count}")

        except Exception as e:
            print(f"[GPU {gpu_id}] Error processing IP {ip_index}: {e}")
            import traceback
            traceback.print_exc()
            result.update({'status': 'error', 'error': str(e)})
            try:
                err_dir = (
                    os.path.join(args.output_dir, str(ip_index))
                    if args.per_ip_subdir else args.output_dir
                )
                os.makedirs(err_dir, exist_ok=True)
                err_traj = {
                    "ip_index": ip_index,
                    "ip_name": ip_data.get('ip_name', 'Unknown'),
                    "image_prompt": ip_data.get('image_prompt', ''),
                    "error": str(e), "gpu_id": gpu_id,
                    "turns": [], "full_response": [],
                    "recaption": "", "generated_image": None,
                }
                with open(os.path.join(err_dir, f"{ip_index}_trajectory.json"), 'w', encoding='utf-8') as f:
                    json.dump(err_traj, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        done_file = os.path.join(done_dir, f"{ip_index}.json")
        try:
            with open(done_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        if claimed_path and os.path.exists(claimed_path):
            try:
                os.remove(claimed_path)
            except Exception:
                pass

    print(f"[GPU {gpu_id}] Worker finished. Total processed: {processed_count}")

def load_ip_data_from_directory(
    dir_path: str, 
    samples_per_file: int = 50, 
    total_target: int = 400,
    seed: int = None
) -> List[Tuple[str, Dict, str]]:
    """
    Load all IP data JSON files from a directory, with random sampling.
    Args:
        dir_path: Path to the JSON directory
        samples_per_file: Number of samples per JSON file (default 50)
        total_target: Target total samples (default 400)
        seed: Random seed for reproducibility
    Returns: [(ip_index, ip_entry, ip_category), ...] list
    """
    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)
    
    ip_list = []
    if not os.path.isdir(dir_path):
        raise ValueError(f"Not a directory: {dir_path}")
    
    json_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.json')])
    print(f"📁 Found {len(json_files)} JSON files in {dir_path}")
    
    for json_file in json_files:
        file_path = os.path.join(dir_path, json_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                ip_dict = json.load(f)  # { "hash1": { entry1 }, ... }
            
            if not isinstance(ip_dict, dict):
                print(f"  ⚠️ Skip {json_file}: not a dict format")
                continue
            
            # Filter valid ip_entry items
            all_entries = [
                (k, v) for k, v in ip_dict.items() 
                if isinstance(v, dict)
            ]
            
            if not all_entries:
                print(f"  ⚠️ Skip {json_file}: no valid entries")
                continue
            
            # Random sampling: take all if <= samples_per_file, else sample
            if len(all_entries) <= samples_per_file:
                sampled = all_entries
                print(f"  📄 {json_file}: taking all {len(all_entries)} entries")
            else:
                sampled = random.sample(all_entries, samples_per_file)
                print(f"  🎲 {json_file}: sampled {samples_per_file}/{len(all_entries)} entries")
            
            for ip_index, ip_entry in sampled:
                # Extract category: prefer ip_category field, else use filename
                category = ip_entry.get('ip_category', '').strip()
                if not category:
                    category = os.path.splitext(json_file)[0]
                
                # Clean category for use as directory name
                category = re.sub(r'[<>:"/\\|？*]', '_', category)
                category = category.strip() or "unknown"
                
                # Ensure ip_entry has index field
                if 'index' not in ip_entry:
                    ip_entry['index'] = str(ip_index)
                
                ip_list.append((str(ip_index), ip_entry, category))
                
        except Exception as e:
            print(f"  ✗ Failed to load {json_file}: {e}")
            continue
    
    # If total exceeds target, truncate randomly
    if total_target > 0 and len(ip_list) > total_target:
        print(f"⚠️ Total {len(ip_list)} entries exceed target {total_target}, truncating...")
        ip_list = random.sample(ip_list, total_target)
    
    print(f"✅ Final: {len(ip_list)} IP entries loaded (target: {total_target})")
    return ip_list

def load_benchmark_json(json_path: str) -> List[Tuple[str, Dict, str]]:
    """Load IP data from benchmark JSON file (test_mini.json format).

    Expected format:
      { "hash": { "index", "country", "ip_name", "image_prompt",
                   "source_category", "seed1_url", "seed2_url", "language" }, ... }

    Returns: [(ip_index, ip_entry, source_category), ...]
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Benchmark JSON must be a dict, got {type(data)}")

    ip_list = []
    for ip_index, ip_entry in data.items():
        if not isinstance(ip_entry, dict):
            continue
        if not ip_entry.get('image_prompt'):
            continue

        source_category = ip_entry.get('source_category', 'unknown').strip()
        source_category = re.sub(r'[<>:"/\\|？*]', '_', source_category) or 'unknown'

        if 'index' not in ip_entry:
            ip_entry['index'] = str(ip_index)

        ip_list.append((str(ip_index), ip_entry, source_category))

    print(f"✅ Loaded {len(ip_list)} entries from benchmark JSON: {json_path}")
    return ip_list


def main():
    # ---------- Worker mode fast path ----------
    # When called as subprocess worker, parse hidden args + restore full args from file
    if '--_worker_mode' in sys.argv:
        _wp = argparse.ArgumentParser()
        _wp.add_argument('--_worker_mode', action='store_true')
        _wp.add_argument('--_task_dir', type=str, required=True)
        _wp.add_argument('--_worker_gpu_id', type=int, required=True)
        _wp.add_argument('--_args_file', type=str, required=True)
        _wargs = _wp.parse_args()
        with open(_wargs._args_file, 'r', encoding='utf-8') as _f:
            _full = json.load(_f)
        _full['_worker_mode'] = True
        _full['_task_dir'] = _wargs._task_dir
        _full['_worker_gpu_id'] = _wargs._worker_gpu_id
        _full['num_gpus'] = 1
        _run_worker_loop(argparse.Namespace(**_full))
        return

    parser = argparse.ArgumentParser(description="BAGEL Multi-turn Inference")
    parser.add_argument("--model_path", type=str, default="csfufu/Unify-Agent", help="Path or HuggingFace repo id of the model (default: csfufu/Unify-Agent)")
    parser.add_argument("--base_model_path", type=str, default=None, help="Base model directory (config/tokenizer/ae loaded from here)")
    parser.add_argument("--ema_path", type=str, default=None, help="EMA weight path (overrides model_path/ema.safetensors)")
    parser.add_argument("--cast_ema_to_bfloat16", action="store_true", help="Cast EMA weights to bfloat16 before loading (with cache)")
    parser.add_argument("--ema_bf16_cache_path", type=str, default=None, help="EMA bfloat16 cache file path (optional)")
    parser.add_argument("--mode", type=int, default=1, choices=[1, 2, 3], 
                       help="Loading mode: 1=full precision, 2=NF4, 3=INT8")
    parser.add_argument("--ip_data", type=str, default=None, 
                    help="Single IP data JSON file path (deprecated, use --ip_data_dir)")
    parser.add_argument("--ip_data_dir", type=str, default=None, 
                    help="IP data directory containing multiple JSON files")
    parser.add_argument("--benchmark_json", type=str, default=None,
                    help="Benchmark JSON file path (test_mini.json format); uses seed images as references, skipping search")
    parser.add_argument("--ip_index", type=str, default=None, help="IP index (if specified, only process this entry; otherwise process all)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory; when used with --model_name, output_dir = output_base/model_name")
    parser.add_argument("--output_base", type=str, default="./output",
                       help="Output root directory (used with --model_name)")
    parser.add_argument("--model_name", type=str, default=None,
                       help="Model name, used to construct output_dir = output_base/model_name")
    parser.add_argument("--reference_images", type=str, nargs='+', default=None, 
                       help="Reference image paths (optional)")
    parser.add_argument("--execute_tools", action="store_true", 
                       help="Execute search tools (requires search API configuration)")
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
        help="Allow batch processing of all samples when --ip_index is not specified (default: off)",
    )
    parser.add_argument(
        "--per_ip_subdir",
        action="store_true",
        help="Use a separate subdirectory output_dir/{ip_key}/ for each IP",
    )
    parser.add_argument(
        "--max_search_turns",
        type=int,
        default=6,
        help="Max turns for the search phase (default: 6)",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs; >1 enables multi-GPU parallel inference (default: 1)",
    )
    parser.add_argument(
        "--enable_think",
        type=bool,
        default=False,
        help="Enable thinking mode (default: False)",
    )
    parser.add_argument(
        "--samples_per_file",
        type=int,
        default=50,
        help="Number of IP samples per JSON file (default: 50)",
    )
    parser.add_argument(
        "--total_target_samples",
        type=int,
        default=400,
        help="Target total samples; truncated randomly if exceeded (default: 400, 0=no limit)"
    )
    
    args = parser.parse_args()
    
    # Resolve output_dir: auto-construct output_base/model_name when --model_name is set
    if args.model_name:
        args.output_dir = os.path.join(args.output_base, args.model_name)
        print(f"📂 Output dir (from --model_name): {args.output_dir}")
    if not args.output_dir:
        raise ValueError("Must specify --output_dir or --model_name")
    
    set_seed(args.seed)
    
    # Load IP data
    # ====== Load IP data (supports benchmark_json / directory / single file) ======
    ip_list = []  # [(ip_index, ip_entry, category), ...]
    is_benchmark = False

    try:
        if args.benchmark_json:
            ip_list = load_benchmark_json(args.benchmark_json)
            args.per_ip_subdir = True
            is_benchmark = True
        else:
            ip_list = load_ip_data_from_directory(args.ip_data_dir, args.samples_per_file, args.total_target_samples, args.seed)
            args.per_ip_subdir = True

    except Exception as e:
        print(f"❌ Error loading IP data: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\n📋 Total IPs to process: {len(ip_list)}")
    
    # ====== Multi-GPU parallel mode (subprocess + file queue) ======
    if args.num_gpus > 1:
        task_dir = os.path.join(args.output_dir, '_task_queue')
        pending_dir = os.path.join(task_dir, 'pending')
        done_dir = os.path.join(task_dir, 'done')
        for _d in [pending_dir, done_dir]:
            os.makedirs(_d, exist_ok=True)
        # Clean up leftover tasks
        for _d in [pending_dir]:
            for _f in os.listdir(_d):
                _fp = os.path.join(_d, _f)
                if os.path.isfile(_fp):
                    os.remove(_fp)

        # Write task files
        task_count = 0
        for ip_index, ip_entry, category in ip_list:
            ref_imgs = _find_reference_images(
                ip_index, args.output_dir, args.per_ip_subdir, args.reference_images
            )
            task_payload = {
                'ip_index': str(ip_index),
                'ip_entry': ip_entry,
                'reference_images': ref_imgs,
                'category': category,
            }
            with open(os.path.join(pending_dir, f'{ip_index}.json'), 'w', encoding='utf-8') as _f:
                json.dump(task_payload, _f, ensure_ascii=False)
            task_count += 1

        print(f"📦 Queued {task_count} tasks for {args.num_gpus} GPU workers")

        # Save full args for workers to read
        args_dict = {k: v for k, v in vars(args).items() if not k.startswith('_')}
        args_file = os.path.join(task_dir, 'worker_args.json')
        with open(args_file, 'w', encoding='utf-8') as _f:
            json.dump(args_dict, _f, ensure_ascii=False)

        # Launch subprocess workers (CUDA_VISIBLE_DEVICES set in env before process start)
        script_path = os.path.abspath(__file__)
        procs = []
        for gpu_id in range(args.num_gpus):
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            cmd = [
                sys.executable, '-u', script_path,
                '--_worker_mode',
                '--_task_dir', task_dir,
                '--_worker_gpu_id', str(gpu_id),
                '--_args_file', args_file,
            ]
            p = _subprocess.Popen(cmd, env=env)
            procs.append((gpu_id, p))
            print(f"🚀 Launched worker GPU {gpu_id} (pid={p.pid}, CUDA_VISIBLE_DEVICES={gpu_id})")

        # Wait for all workers to exit
        for gpu_id, p in procs:
            p.wait()
            if p.returncode != 0:
                print(f"⚠️ Worker GPU {gpu_id} exited with code {p.returncode}")
            else:
                print(f"✅ Worker GPU {gpu_id} finished successfully")

        # Collect results from done directory
        results = []
        for fname in os.listdir(done_dir):
            if fname.endswith('.json'):
                try:
                    with open(os.path.join(done_dir, fname), 'r', encoding='utf-8') as _f:
                        results.append(json.load(_f))
                except Exception:
                    pass
    
    # ====== Single GPU mode (original flow) ======
    else:
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
        
        results = []
        for ip_index, ip_entry, category in ip_list:
            try:
                ip_data = _prepare_ip_data_from_entry(ip_index, ip_entry)
                if ip_data is None:
                    print(f"⚠️ Skipping {ip_index}: missing required fields")
                    continue
                
                print(f"\n{'='*60}")
                print(f"Processing IP {ip_index}: {ip_data['ip_name']}")
                print(f"{'='*60}")
                
                ip_output_dir = os.path.join(args.output_dir, category, str(ip_index)) if args.per_ip_subdir else None
                if ip_output_dir:
                    os.makedirs(ip_output_dir, exist_ok=True)
                
                reference_images = _find_reference_images(
                    ip_index, args.output_dir, args.per_ip_subdir, args.reference_images
                )
                if reference_images:
                    print(f"  📁 Found reference images: {reference_images}")
                
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
                    think=args.enable_think,
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
                        "turns": [], "full_response": [],
                        "recaption": "", "generated_image": None,
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
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total processed: {len(results)}")
    print(f"  Successful: {sum(1 for r in results if r.get('status') == 'success')}")
    print(f"  Partial: {sum(1 for r in results if r.get('status') == 'partial')}")
    print(f"  Errors: {sum(1 for r in results if r.get('status') == 'error')}")
    print(f"{'='*60}")
    
    # Save results summary
    summary_file = os.path.join(args.output_dir, "processing_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n📄 Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()

