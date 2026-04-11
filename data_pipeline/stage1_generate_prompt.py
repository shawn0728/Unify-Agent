# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""
Stage 1: Generate image-generation prompts from an IP CSV.

Reads IP names from a CSV file and uses an OpenAI-compatible LLM to produce
image-generation prompts of the form "who is doing what in which scene".
"""
import argparse
import json
import logging
import os
import re
import time
import traceback

import pandas as pd
from tqdm import tqdm

log = logging.getLogger("stage1")

# ---------------------------------------------------------------------------
# API configuration (set via environment variables)
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", None)
STAGE1_MODEL = os.environ.get("STAGE1_MODEL", "gpt-4o")

MAX_RETRIES = 5
RETRY_DELAY = 30


def _get_openai_client():
    """Return an OpenAI client, optionally pointed at a custom base URL."""
    import openai
    kwargs = {"api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    return openai.OpenAI(**kwargs)


def infer_text(prompt, system_prompt=None):
    """
    Call an OpenAI-compatible chat-completions endpoint (with retries).

    Args:
        prompt: User message.
        system_prompt: Optional system message.

    Returns:
        The assistant's text reply, or None on failure.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    client = _get_openai_client()
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=STAGE1_MODEL,
                messages=messages,
                max_tokens=4096,
                temperature=0.7,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (attempt + 1) if "429" in str(e) else RETRY_DELAY
                print(f"  Warning: API error (attempt {attempt + 1}/{MAX_RETRIES}): "
                      f"{str(e)[:200]}  — retrying in {wait}s ...")
                time.sleep(wait)
            else:
                print(f"  Error: API failed after {MAX_RETRIES} attempts: {last_error}")
                return None

    return None


# ---------------------------------------------------------------------------
# System prompt (Image-Generation Prompt Expert)
# ---------------------------------------------------------------------------

system_prompt = """
You are a specialized Image Generation Prompt Expert. Your mission is to generate a single, high-quality image generation prompt based on a specific IP (person/character). You must describe "Who is doing What in Which scene."

## 1. Core Logic & Language Rules (CRITICAL)

**Step 1: Determine the IP's Nationality**
* **Case A: If the IP is Chinese** (Mainland, Hong Kong, Taiwan, Macau):
    * **Prompt Language**: Must be **Chinese**.
    * **Tag Language**: Must be **Chinese**.
    * **Language Code**: `zh`
* **Case B: If the IP is NOT Chinese** (USA, UK, Japan, Korea, Europe, etc.):
    * **Prompt Language**: Must be **English**.
    * **Tag Language**: Must be **English**.
    * **Language Code**: `en`

## 2. Prompt Construction Requirements

1.  **Content Elements**:
    * **Subject**: The IP's name (and brief identity if needed for context).
    * **Scene**: A specific, realistic location fitting the IP's profession (e.g., Office, Studio, Stadium, Stage, Cafe, Street).
    * **Action**: A dynamic verb describing what they are doing (e.g., Interviewing, Singing, coding, running, drinking coffee).
2.  **Realism Constraint**:
    * The scene and action must align with the IP's public persona or profession.
    * Do not hallucinate impossible scenarios unless the IP is a fictional fantasy character.
3.  **Syntactic Diversity** (Avoid Repetition):
    * Do not always use the structure "Name is doing X in Y".
    * **Vary your sentence structures**:
        * *Scene-first*: "In the [Scene], [Name] is [Action]..."
        * *Action-focused*: "[Name] is [Action] while located in [Scene]..."
        * *Descriptive*: "Surrounded by [Context], [Name] is [Action]..."

## 3. Output Format

You must output **ONLY** the XML tags below. Do not output markdown code blocks (like ```xml), explanations, or conversational filler.

<Image_Prompt>The full descriptive sentence</Image_Prompt>
<Tag_Name>Category of the action (e.g., Interview, Speech, Daily Life)</Tag_Name>
<Language>zh OR en</Language>

*Note on <Tag_Name>*: If Language is `zh`, the tag must be Chinese (e.g., "主持", "街拍"). If Language is `en`, the tag must be English (e.g., "Hosting", "Street Snap").

**REMEMBER: Output ONLY the three XML tags above, nothing else.**

## 4. Examples

**Input:** Lei Jun (Chinese Tech CEO)
**Output:**
<Image_Prompt>雷军站在小米新品发布会的舞台中央，正在充满激情地介绍最新的智能手机</Image_Prompt>
<Tag_Name>演讲</Tag_Name>
<Language>zh</Language>

**Input:** Robert Downey Jr. (American Actor)
**Output:**
<Image_Prompt>Sitting in a relaxed pose on a Hollywood talk show set, Robert Downey Jr. is laughing while telling a story</Image_Prompt>
<Tag_Name>Interview</Tag_Name>
<Language>en</Language>

**Input:** Liu Xiang (Chinese Athlete)
**Output:**
<Image_Prompt>在阳光明媚的田径场跑道上，刘翔正在进行跨栏训练准备</Image_Prompt>
<Tag_Name>运动</Tag_Name>
<Language>zh</Language>

**Input:** Gordon Ramsay (British Chef)
**Output:**
<Image_Prompt>Gordon Ramsay is carefully plating a gourmet dish in a busy high-end restaurant kitchen</Image_Prompt>
<Tag_Name>Cooking</Tag_Name>
<Language>en</Language>
"""


def parse_response(response):
    """Extract Image_Prompt, Tag_Name, and Language from LLM response."""
    if response is None:
        return "", "", ""

    def extract_tag(tag_name, text):
        match = re.search(rf'<{tag_name}>(.*?)</{tag_name}>', text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def extract_json_from_text(text):
        """Try to extract a JSON object from the text."""
        json_match = re.search(r'```json\s*\n(.*?)\n```', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                return None

        try:
            return json.loads(json_str)
        except Exception:
            return None

    # Try JSON format first
    json_data = extract_json_from_text(response)
    if json_data:
        image_prompt = json_data.get('prompt', '')
        tags = json_data.get('tags', [])

        tag_name = ""
        if isinstance(tags, list) and len(tags) > 0:
            meaningful_tags = [t for t in tags if t and isinstance(t, str) and len(t) > 1]
            if meaningful_tags:
                for tag in meaningful_tags:
                    if any(kw in tag for kw in [
                        '主播', '主持', '记者', '演员', 'Host', 'Interview',
                        'Actor', 'Speech', 'Sport', 'Cooking',
                        '主持人', '采访', '表演', '演讲', '运动', '烹饪',
                    ]):
                        tag_name = tag
                        break
                if not tag_name:
                    tag_name = meaningful_tags[0]

        if image_prompt and re.search(r'[\u4e00-\u9fff]', image_prompt):
            language = "zh"
        elif image_prompt:
            language = "en"
        else:
            language = ""

        return image_prompt, tag_name, language

    # Fall back to XML tag parsing
    image_prompt = extract_tag("Image_Prompt", response)
    tag_name = extract_tag("Tag_Name", response)
    language = extract_tag("Language", response)

    return image_prompt, tag_name, language


def generate_prompt_for_ip(ip_name, country, ip_info=None):
    """
    Generate an image-generation prompt for a single IP.

    Args:
        ip_name: Display name (name_zh for Chinese IPs, name_en otherwise).
        country: Nationality string (used to choose language).
        ip_info: Optional dict / Series with 'category' and 'remark' fields.

    Returns:
        (image_prompt, tag_name, language) or (None, None, None) on failure.
    """
    user_prompt = "Please generate an image generation prompt for the following IP:\n\n"
    user_prompt += f"IP Name: {ip_name}\n"
    user_prompt += f"IP Country: {country}\n"

    if ip_info is not None:
        if pd.notna(ip_info.get('category')):
            user_prompt += f"IP Category: {ip_info['category']}\n"
        if pd.notna(ip_info.get('remark')):
            user_prompt += f"IP Description: {ip_info['remark']}\n"

    user_prompt += (
        f'\nBased on the above information, generate an image generation prompt '
        f'describing "{ip_name} doing what in which scene". '
        f'Note: The IP\'s country is {country}. '
    )
    if country == "中国":
        user_prompt += "Since this is a Chinese IP, the prompt and tag must be in Chinese (Language: zh)."
    else:
        user_prompt += "Since this is a non-Chinese IP, the prompt and tag must be in English (Language: en)."

    try:
        response_text = infer_text(user_prompt, system_prompt=system_prompt)

        if response_text is None:
            print(f"  [FAIL] API returned None for '{ip_name}'")
            return None, None, None

        image_prompt, tag_name, language = parse_response(response_text)

        if image_prompt and language:
            return image_prompt, tag_name, language
        else:
            print(f"  [WARN] Failed to parse response for '{ip_name}': "
                  f"{response_text[:200]}...")
            return None, None, None

    except Exception as e:
        print(f"  [FAIL] Prompt generation failed for '{ip_name}': {e}")
        traceback.print_exc()
        return None, None, None


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(name)s] %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Stage 1: Generate image-generation prompts from an IP CSV."
    )
    parser.add_argument(
        "--ip_csv", required=True,
        help="Path to the input IP CSV file.",
    )
    parser.add_argument(
        "--output_json", required=True,
        help="Path to write the output JSON file.",
    )
    parser.add_argument(
        "--existing_json", default=None,
        help="Path to an existing JSON file whose IPs should be skipped.",
    )
    args = parser.parse_args()

    ip_csv_path = args.ip_csv
    output_json_path = args.output_json
    existing_json_path = args.existing_json

    output_dir = os.path.dirname(output_json_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Stage 1: Generating image-generation prompts for IPs")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Read IP CSV
    # ------------------------------------------------------------------
    print(f"  Reading IP data: {ip_csv_path}")
    try:
        df_ip = pd.read_csv(ip_csv_path)
        print(f"  Loaded {len(df_ip)} IP records")
    except Exception as e:
        print(f"  [FAIL] Could not read IP CSV: {e}")
        return

    # ------------------------------------------------------------------
    # Load already-processed IPs (from existing JSON)
    # ------------------------------------------------------------------
    processed_indices = set()
    if existing_json_path and os.path.exists(existing_json_path):
        print(f"  Loading existing results: {existing_json_path}")
        try:
            with open(existing_json_path, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            processed_indices = set(existing_results.keys())
            print(f"  Found {len(processed_indices)} already-generated IPs")
        except Exception as e:
            print(f"  [WARN] Could not read existing file: {e} — starting fresh")

    # ------------------------------------------------------------------
    # Load output file for incremental saving
    # ------------------------------------------------------------------
    all_results = {}
    if os.path.exists(output_json_path):
        print(f"  Loading output file: {output_json_path}")
        try:
            with open(output_json_path, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
            print(f"  Output file already contains {len(all_results)} records")
        except Exception as e:
            print(f"  [WARN] Could not read output file: {e} — starting fresh")

    # ------------------------------------------------------------------
    # Determine which IPs still need processing
    # ------------------------------------------------------------------
    ip_to_process = []
    for idx, row in df_ip.iterrows():
        ip_index = row.get('index', '')
        country = row.get('country', '')
        ip_name = row.get('name_zh', '') if country == "中国" else row.get('name_en', '')

        if ip_index and ip_index not in processed_indices and ip_index not in all_results:
            if pd.notna(ip_name) and ip_name != '':
                ip_to_process.append((idx, row))

    print(f"  IPs to process: {len(ip_to_process)}")

    # ------------------------------------------------------------------
    # Generate prompts
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Generating prompts")
    print("=" * 60)

    new_count = 0
    for _idx, row in tqdm(ip_to_process, desc="Generating prompts", unit="IP"):
        ip_index = row.get('index', '')
        country = row.get('country', '')
        ip_name = row.get('name_zh', '') if country == "中国" else row.get('name_en', '')

        if pd.isna(ip_name) or ip_name == '':
            continue
        if ip_index in processed_indices or ip_index in all_results:
            tqdm.write(f"  [SKIP] Already processed: {ip_name} (index: {ip_index})")
            continue

        tqdm.write(f"\n  Processing: {ip_name} (country: {country}, index: {ip_index})")

        image_prompt, tag_name, language = generate_prompt_for_ip(ip_name, country, row)

        if image_prompt and language:
            result = {
                'index': ip_index,
                'country': country,
                'ip_name': ip_name,
                'ip_name_zh': row.get('name_zh', ''),
                'ip_name_en': row.get('name_en', ''),
                'ip_category': row.get('category', ''),
                'image_prompt': image_prompt,
                'tag_name': tag_name,
                'language': language,
            }
            all_results[ip_index] = result
            new_count += 1
            tqdm.write(f"  [OK] {image_prompt[:50]}... (Tag: {tag_name}, Lang: {language})")

            if new_count % 10 == 0:
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)
                tqdm.write(f"  [SAVE] Checkpoint — {new_count} new records so far")
        else:
            tqdm.write(f"  [FAIL] {ip_name}")

        time.sleep(0.5)

    # ------------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------------
    if new_count > 0:
        print("\n" + "=" * 60)
        print("Saving results")
        print("=" * 60)

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        print(f"  Results saved to: {output_json_path}")
        print(f"  New records: {new_count}  |  Total records: {len(all_results)}")
    else:
        print("\n  No new prompts were generated.")


if __name__ == "__main__":
    main()
