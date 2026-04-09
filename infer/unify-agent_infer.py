# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
Multi-turn inference script for Unify-Agent.
Integrates recaption generation and image synthesis using the BAGEL model
with multi-turn dialogue and tool-augmented search.
"""

import os
import sys
import json
import re
import argparse
import random
from copy import deepcopy
import numpy as np
import torch
import time
import base64
import uuid
import requests
import shutil
from PIL import Image
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_loader import load_model_and_inferencer
from data.data_utils import pil_img2rgb

# API configuration (set via environment variables)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", None)
REASONING_MODEL = os.environ.get("AGENT_REASONING_MODEL", "gpt-4o")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")

MAX_RETRIES = 3
RETRY_DELAY = 2
DEBUG_LOG_PATH = os.environ.get("SFT_DEBUG_LOG", "")


def _append_debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: Dict[str, Any]):
    if not DEBUG_LOG_PATH:
        return
    payload = {
        "id": f"dbg_{int(time.time() * 1000)}_{hypothesis_id}",
        "timestamp": int(time.time() * 1000),
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
    }
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def set_seed(seed):
    """Set random seed for reproducibility."""
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _get_openai_client():
    """Get an OpenAI client instance."""
    kwargs = {}
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    return __import__("openai").OpenAI(**kwargs)


def call_gemini3_flash_api(prompt, system_prompt=None, timeout=60):
    """
    Call an OpenAI-compatible API for text reasoning (with retries).
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    client = _get_openai_client()
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=REASONING_MODEL,
                messages=messages,
                max_tokens=4096,
                temperature=0.0,
            )
            text_response = response.choices[0].message.content or ""
            return {"success": True, "text": text_response, "raw": None}
        except Exception as e:
            last_error = str(e)
            if attempt < MAX_RETRIES:
                print(f"  Warning: API failed (attempt {attempt + 1}/{MAX_RETRIES + 1}): {str(e)[:200]}")
                time.sleep(RETRY_DELAY)
            else:
                return {"success": False, "error": str(e), "raw": None}

    return {"success": False, "error": last_error, "raw": None}


def detect_image_format_from_bytes(img_bytes):
    """Detect image format from raw bytes."""
    try:
        # JPEG: FF D8 FF
        if len(img_bytes) >= 3 and img_bytes[:3] == b'\xff\xd8\xff':
            return 'image/jpeg'
        # PNG: 89 50 4E 47 0D 0A 1A 0A
        elif len(img_bytes) >= 8 and img_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            return 'image/png'
        # GIF not supported
        elif len(img_bytes) >= 6:
            if img_bytes[:6] == b'GIF87a' or img_bytes[:6] == b'GIF89a':
                return None
        # WebP not supported
        if len(img_bytes) >= 12:
            if img_bytes[:4] == b'RIFF' and img_bytes[8:12] == b'WEBP':
                return None
        
        try:
            img = Image.open(BytesIO(img_bytes))
            format_lower = img.format.lower() if img.format else 'jpeg'
            if format_lower in ['gif', 'webp']:
                return None
            format_map = {
                'jpeg': 'image/jpeg',
                'jpg': 'image/jpeg',
                'png': 'image/png',
            }
            return format_map.get(format_lower, 'image/jpeg')
        except:
            pass
        
        return 'image/jpeg'
    except Exception as e:
        print(f"  ⚠️ Error detecting image format: {e}, defaulting to image/jpeg")
        return 'image/jpeg'


def call_gemini3_flash_with_image(prompt, image_path, system_prompt=None, timeout=60):
    """
    Call an OpenAI-compatible multimodal API for image+text reasoning (with retries).
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if not image_path or not os.path.exists(image_path):
        return {"success": False, "error": f"Image not found: {image_path}", "raw": None}

    try:
        with open(image_path, 'rb') as f:
            img_bytes = f.read()
        mime_type = detect_image_format_from_bytes(img_bytes)
        if mime_type is None:
            return {"success": False, "error": "Unsupported image format (gif/webp)", "raw": None}
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        image_url = f"data:{mime_type};base64,{img_base64}"
    except Exception as e:
        return {"success": False, "error": f"Failed to load image: {str(e)}", "raw": None}

    user_content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]
    messages.append({"role": "user", "content": user_content})

    client = _get_openai_client()
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=REASONING_MODEL,
                messages=messages,
                max_tokens=4096,
                temperature=0.0,
            )
            text_response = response.choices[0].message.content or ""
            return {"success": True, "text": text_response, "raw": None}
        except Exception as e:
            last_error = str(e)
            if attempt < MAX_RETRIES:
                print(f"  Warning: API failed (attempt {attempt + 1}/{MAX_RETRIES + 1}): {str(e)[:200]}")
                time.sleep(RETRY_DELAY)
            else:
                return {"success": False, "error": str(e), "raw": None}

    return {"success": False, "error": last_error, "raw": None}


def judge_image_quality(image_path, ip_name, image_prompt=""):
    """
    Score image quality using an LLM judge.
    
    Args:
        image_path: Path to image file
        ip_name: IP/character name for evaluation context (may be empty)
        image_prompt: Scene description used as context when ip_name is empty
    
    Returns:
        dict with score, reason, is_text_heavy, has_watermark.
              Returns {"score": -1, ...} on failure.
    """
    sys_prompt = """You are an expert Image Quality Assessor for an AI training pipeline. 
Your task is to evaluate whether a downloaded image is a high-quality visual reference for a specific Intellectual Property (IP).

You must rate the image on a scale of 0 to 10 based on the following strict rubrics:

### 1. IP Consistency (Critical)
- **Pass:** The image clearly depicts the specific character/object requested.
- **Fail (Score 0):** The image shows a completely different character, a landscape, a real person (cosplay) if the IP is anime, or an unrelated object.

### 2. Layout & Composition
- **Subject-Centric:** The IP subject must be the main focus.
- **Face Visibility:** For characters, the face must be clearly visible (preferably front-facing or 3/4 view). REJECT if the face is tiny, distant, or back-facing.
- **No Text-Heavy:** REJECT images that are primarily movie posters with large text overlays, book covers, or infographics where the subject is obscured by text.
- **No Collages:** REJECT split-screens, manga panels, or multiple images stitched together. Single-scene images only.

### 3. Visual Quality
- **Clarity:** The image must be sharp. REJECT if it is severely blurry, pixelated, or has heavy jpeg compression artifacts.

### 4. Watermarks & Obstructions
- **Reject:** Large, obstructive watermarks covering the face or main body (e.g., full-screen stock photo watermarks).
- **Accept:** Small, unobtrusive logos in the corner are acceptable but lower the score slightly.

### Scoring Guide:
- **0:** Wrong IP or completely unusable.
- **1-5 (Reject):** Correct IP but violates a major rule (Blurry, Huge Watermark, Text-Heavy, Collage, Tiny Face).
- **6-7 (Borderline):** Usable but not ideal (Side profile, small corner watermark, medium resolution).
- **8-10 (Excellent):** Perfect reference (High-res, clear front-facing, no text, clean).
"""

    eval_subject = f'the IP: "{ip_name}"' if ip_name else f'the scene: "{image_prompt[:200]}"'
    question = f"""Please evaluate the provided image for {eval_subject}.

Assess the image based on the system rubrics.
Return your response in JSON format ONLY with the following structure:
{{
    "score": <int, 0-10>,
    "reason": "<string, a concise explanation of the score, mentioning any specific flaws like 'text-heavy', 'watermark', or 'blurry'>",
    "is_text_heavy": <bool>,
    "has_watermark": <bool>
}}
"""
    
    display_subject = ip_name if ip_name else image_prompt[:60]
    print(f"  🔍 Judging image quality: {os.path.basename(image_path)} for '{display_subject}'")
    
    result = call_gemini3_flash_with_image(question, image_path, system_prompt=sys_prompt, timeout=60)
    
    if result.get("success") and result.get("text"):
        try:
            response_text = result["text"].strip()
            
            try:
                judge_result = json.loads(response_text)
            except json.JSONDecodeError:
                json_match = re.search(r'\{[^{}]*"score"[^{}]*\}', response_text, re.DOTALL)
                if json_match:
                    judge_result = json.loads(json_match.group(0))
                else:
                    json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
                    if json_match:
                        judge_result = json.loads(json_match.group(0))
                    else:
                        print(f"  ⚠️ Failed to extract JSON from response: {response_text[:200]}")
                        return {"score": -1, "reason": "Failed to parse JSON response", "is_text_heavy": False, "has_watermark": False}
            
            score = judge_result.get("score", -1)
            reason = judge_result.get("reason", "No reason provided")
            is_text_heavy = judge_result.get("is_text_heavy", False)
            has_watermark = judge_result.get("has_watermark", False)
            
            print(f"  ✅ Score: {score}/10 - {reason[:80]}...")
            
            return {
                "score": score,
                "reason": reason,
                "is_text_heavy": is_text_heavy,
                "has_watermark": has_watermark
            }
            
        except Exception as e:
            print(f"  ❌ Error parsing judge result: {e}")
            return {"score": -1, "reason": f"Parse error: {str(e)}", "is_text_heavy": False, "has_watermark": False}
    else:
        error_msg = result.get("error", "Unknown error")
        print(f"  ❌ Judge API failed: {error_msg}")
        return {"score": -1, "reason": f"API error: {error_msg}", "is_text_heavy": False, "has_watermark": False}


def extract_tool_call(text: str) -> Optional[Dict]:
    """Extract tool call from text."""
    tool_call_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    tool_matches = list(re.finditer(tool_call_pattern, text, re.DOTALL))
    # region agent log
    _append_debug_log(
        run_id="pre-fix",
        hypothesis_id="H2",
        location="multi_turn_infer.py:extract_tool_call:entry",
        message="tool call extraction input stats",
        data={
            "text_len": len(text) if text is not None else 0,
            "contains_tool_call_tag": "<tool_call>" in (text or ""),
            "contains_json_fence": "```json" in (text or "") or "```" in (text or ""),
            "regex_match_count": len(tool_matches),
            "text_preview": (text or "")[:400],
        },
    )
    # endregion
    
    if tool_matches:
        tool_match = tool_matches[0]
        tool_json = tool_match.group(1).strip()
        try:
            tool_dict = json.loads(tool_json)
            params = tool_dict.get('arguments', tool_dict.get('parameters', {}))
            tool_name = tool_dict.get('name', '')
            # region agent log
            _append_debug_log(
                run_id="pre-fix",
                hypothesis_id="H3",
                location="multi_turn_infer.py:extract_tool_call:json_ok",
                message="tool call parsed successfully",
                data={
                    "tool_name": tool_name,
                    "param_keys": sorted(list(params.keys())) if isinstance(params, dict) else [],
                },
            )
            # endregion
            return {
                "name": tool_name,
                "parameters": params
            }
        except Exception as e:
            # region agent log
            _append_debug_log(
                run_id="pre-fix",
                hypothesis_id="H3",
                location="multi_turn_infer.py:extract_tool_call:json_fail",
                message="tool call json parse failed",
                data={
                    "error": str(e),
                    "tool_json_preview": tool_json[:400],
                },
            )
            # endregion
            print(f"  ⚠️ Warning: Failed to parse tool_call JSON: {e}")
            return None
    
    # region agent log
    _append_debug_log(
        run_id="pre-fix",
        hypothesis_id="H2",
        location="multi_turn_infer.py:extract_tool_call:no_match",
        message="tool call regex not matched",
        data={
            "contains_instruction_tag": "<Instruction" in (text or ""),
            "contains_braces": "{" in (text or "") and "}" in (text or ""),
        },
    )
    # endregion
    return None


def extract_recaption_content(text: str) -> str:
    """Extract content from <recaption> tags."""
    if not text:
        return text
    
    match = re.search(r'<recaption>(.*?)</recaption>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    match = re.search(r'<recaption>(.*)', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    return text


def has_recaption_tag(text: str) -> bool:
    """Check if text contains recaption tags."""
    return '<recaption>' in text.lower() or 'recaption' in text.lower()


def normalize_recaption_text(text: str, language: str = 'zh') -> str:
    """Normalize recaption text by standardizing XML tag casing."""
    if not text:
        return text
    normalized = text
    normalized = re.sub(r'<Think>', '<think>', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'</Think>', '</think>', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'<Instruction>', '<recaption>', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'</Instruction>', '</recaption>', normalized, flags=re.IGNORECASE)
    return normalized


def get_english_recaption_prompt() -> str:
    return '''Role Setting
You are a professional visual language reasoning assistant. Your task is to generate a reasoning process and a final detailed image description based on **two reference images (image_1, image_2)**, the original instruction, and text search results.

Input Information
1. **Reference Images**: Explicitly labeled Reference Image 1 (refer to as image_1) and Reference Image 2 (refer to as image_2).
2. **Original Instruction**: The user's short request.
3. **Background Info**: Text information from previous search steps.

Output Format
Strictly follow XML format:
<think>
[Deep reasoning here:
 1. Analyze visual features of image_1 and image_2.
 2. Combine with background info to plan the fusion.
 3. Explicitly state what comes from image_1 and what comes from image_2.]
</think>
<recaption>
[The final detailed image description, including "Scene Description" and "Preservation Statement"]
</recaption>

Core Rules & Constraints
1. **Reference Principle**: You MUST strictly use "image_1" and "image_2" to refer to the images. DO NOT use vague terms like "the first image" or "reference picture".

2. **Descriptive Style**: The content of <recaption> must be a **description of the final result** (Descriptive), NOT an editing command (Imperative).
   - BAD: "Please put the man from image_1 on the left..."
   - GOOD: "In this realistic outdoor portrait, the man from image_1 stands on the left..."

3. **Preservation Statement**: At the end of the description, you MUST explicitly state what specifically is preserved from image_1 and image_2.
   - Must include phrases like: "The final image completely preserves [features] from image_1..." and "The final image fully retains [features] from image_2...".
   - **Facial Features Preservation (CRITICAL)**: If image_1 and/or image_2 contain identifiable persons, you MUST preserve their exact facial features as shown in the reference images. Include detailed descriptions such as: face shape, eyebrow shape, eye characteristics, facial expression, skin tone/complexion, hairstyle, and any distinctive facial features. Use the format: "Preserve the exact facial features of [person name/description] as shown in image_1 and image_2: [detailed facial feature description]. Maintain [their/his/her] [appearance/clothing/style] as referenced in both images."

4. **Language Consistency (MANDATORY)**: The content of <recaption> MUST be written entirely in the SAME language as the original instruction. If the original instruction is in Chinese, ALL descriptions in <recaption> must be in Chinese - NO English allowed. If the original instruction is in English, use English throughout. NEVER mix languages within the same description.

Start the task. Output <think> and <recaption>.'''


def get_chinese_recaption_prompt() -> str:
    return '''角色设定
你是一个专业的视觉语言推理助手。你的任务是基于给定的**两张参考图（image_1, image_2）**、原始指令和文本搜索结果，生成推理过程和最终的画面描述。

输入信息
1. **参考图**：系统已提供明确标注的 Reference Image 1 (指代为 image_1) 和 Reference Image 2 (指代为 image_2)。
2. **原始指令**：用户的简短需求。
3. **背景知识**：前序步骤搜索到的文本信息。

输出格式
必须严格遵循XML格式：
<think>
[在此处进行深度推理：
 1. 分析 image_1 和 image_2 的视觉特征（人物、衣着、动作、光影）。
 2. 结合背景知识和原始指令，规划如何融合这两张图。
 3. 明确哪些元素来自 image_1，哪些来自 image_2，哪些是新生成的。]
</think>
<recaption>
[这里填写最终的画面详细描述，包含"场景描述"和"保留特征声明"]
</recaption>

核心规则与约束 (必须严格遵守)
1. **指代原则**：必须严格使用 "image_1" 和 "image_2" 来指代两张参考图。禁止使用 "图一"、"第一张图"、"参考图" 等模糊表述。

2. **描述性风格**：<recaption> 的内容必须是**一段完成后的画面描述**（Descriptive），而不是编辑指令（Imperative）。
   - 错误写法："请把 image_1 的人放在左边..."
   - 正确写法："这张写实风格的户外合影中，来自 image_1 的男士在左..."

3. **保留特征声明**：在描述的最后，必须明确写出从 image_1 和 image_2 中分别保留了什么。
   - 必须包含类似句式："最终图像完整保留了 image_1 中..." 和 "最终图像完整保留了 image_2 中..."。
   - **面部特征保留（关键要求）**：如果 image_1 和/或 image_2 中包含可识别的人物，必须保留他们在参考图中展现的确切面部特征。需要详细描述包括：脸型、眉毛形状、眼睛特征、面部表情、肤色、发型以及任何独特的面部特征。使用格式："保留 image_1 和 image_2 中[人物姓名/描述]的确切面部特征：[详细的面部特征描述]。保持[他/她]在两张图片中展现的[外观/衣着/风格]。"

4. **语言统一（强制要求）**：<recaption> 中的内容必须**全部使用与原始指令相同的语言**。如果原始指令是中文，则 <recaption> 中的所有描述都必须是中文，禁止使用任何英文。如果原始指令是英文，则全部使用英文。绝对禁止在同一段描述中混用中英文。

示例输出风格参考：
"这张[风格]的图片中，来自 image_1 的[主体]位于[位置]... 背景是[环境]... 光影呈现[效果]... 保留 image_1 和 image_2 中[人物姓名]的确切面部特征：[详细描述如：椭圆形脸型、温和的弯眉、明亮有神的眼睛、温暖的笑容、白皙的肤色等]。保持[他/她]在两张图片中展现的[专业外观/深色外套配彩色内搭等]。最终图像完整保留了 image_1 中[主体]的面部特征、发型和衣着细节。最终图像完整保留了 image_2 中[主体]的动作和配饰。"

开始任务，输出 <think> 和 <recaption>。'''


def load_prompt_template(country: str = None) -> str:
    """Load recaption system prompt based on country."""
    if country == "中国":
        return get_chinese_recaption_prompt()
    else:
        return get_english_recaption_prompt()


def download_image_to_bytes(image_url):
    """Download image and return raw bytes (with retries)."""
    proxies = {
        'http': os.environ.get('http_proxy', ''),
        'https': os.environ.get('https_proxy', ''),
    }
    if not proxies['http']:
        proxies = None
    
    try:
        parsed_url = urlparse(image_url)
        referer = f"{parsed_url.scheme}://{parsed_url.netloc}/"
    except:
        referer = ''
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': referer,
    }
    
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.get(image_url, headers=headers, proxies=proxies, timeout=30)
            response.raise_for_status()
            return True, response.content, None
        except Exception as e:
            last_error = str(e)
            if attempt < MAX_RETRIES:
                print(f"  Warning: Image download failed (attempt {attempt + 1}/{MAX_RETRIES + 1}): {str(e)[:100]}")
                print(f"  Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
    
    return False, None, last_error


def execute_text_search(parameters, use_summary=True):
    """Execute text_search tool with optional LLM summarization."""
    query = parameters.get('q', '')
    lang = parameters.get('hl', 'en')
    top_k = parameters.get('top_k', 5)
    
    if not query:
        return "Error: 'q' parameter is required for text_search", {}
    
    try:
        text_search_debug = {
            "request": {
                "q": query,
                "hl": lang,
                "top_k": top_k,
            },
            "serper_payload": None,
            "serper_raw_response": None,
            "summary_items": [],
        }
        serper_headers = {
            'X-API-KEY': SERPER_API_KEY,
            'Content-Type': 'application/json'
        }
        serper_payload = {
            "q": query,
            "location": "United States",
            "hl": lang,
            "num": min(top_k, 20)
        }
        text_search_debug["serper_payload"] = serper_payload
        
        print(f"  Searching: q='{query}', hl='{lang}'")
        
        serper_data = None
        last_serper_error = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                serper_response = requests.post(
                    "https://google.serper.dev/search",
                    headers=serper_headers,
                    json=serper_payload,
                    timeout=60
                )
                
                if serper_response.status_code != 200:
                    last_serper_error = f"Serper API error: {serper_response.status_code} - {serper_response.text[:200]}"
                    if attempt < MAX_RETRIES:
                        print(f"  Warning: Serper API failed (attempt {attempt + 1}/{MAX_RETRIES + 1}): {last_serper_error[:100]}")
                        print(f"  Retrying in {RETRY_DELAY}s...")
                        time.sleep(RETRY_DELAY)
                        continue
                    else:
                        print(f"  ❌ Error: {last_serper_error}")
                        return f"Tool execution error:\n{last_serper_error}", {}
                
                serper_data = serper_response.json()
                text_search_debug["serper_raw_response"] = serper_data
                
                if serper_data.get("code") and serper_data.get("code") != 0:
                    last_serper_error = f"Serper API returned error: {serper_data.get('msg', 'Unknown error')}"
                    if attempt < MAX_RETRIES:
                        print(f"  Warning: Serper API returned error (attempt {attempt + 1}/{MAX_RETRIES + 1}): {last_serper_error[:100]}")
                        print(f"  Retrying in {RETRY_DELAY}s...")
                        time.sleep(RETRY_DELAY)
                        continue
                    else:
                        print(f"  ❌ Error: {last_serper_error}")
                        return f"Tool execution error:\n{last_serper_error}", {}
                
                break
                
            except requests.exceptions.RequestException as e:
                last_serper_error = f"Serper API request exception: {str(e)}"
                if attempt < MAX_RETRIES:
                    print(f"  Warning: Serper API exception (attempt {attempt + 1}/{MAX_RETRIES + 1}): {str(e)[:100]}")
                    print(f"  Retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    print(f"  ❌ Error: {last_serper_error}")
                    return f"Tool execution error:\n{last_serper_error}", {}
        
        if serper_data is None:
            return f"Tool execution error:\n{last_serper_error}", {}
        
        results_summary = []
        
        answer_box = serper_data.get("answerBox")
        if answer_box:
            title = answer_box.get("title", "")
            snippet = answer_box.get("snippet", "")
            link = answer_box.get("link", "")
            if snippet:
                results_summary.append(f"[Answer Box] {title}\n{snippet}")
                print(f"  ✅ Found answerBox with snippet")
        
        organic_results = serper_data.get("organic", [])
        
        if not organic_results and not results_summary:
            return "Tool execution result:\nNo relevant web pages found for the query.", text_search_debug
        
        jina_api_key = os.environ.get("JINA_API_KEY", "")
        jina_headers = {
            'Authorization': f'Bearer {jina_api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        }
        
        for idx, result in enumerate(organic_results[:top_k]):
            url = result.get('link', '')
            title = result.get('title', '')
            snippet = result.get('snippet', '')
            
            if not url:
                continue
            
            jina_success = False
            content = ""
            
            for jina_attempt in range(MAX_RETRIES + 1):
                try:
                    jina_payload = {"url": url}
                    jina_response = requests.post(
                        "https://r.jina.ai/",
                        headers=jina_headers,
                        json=jina_payload,
                        timeout=60
                    )
                    
                    if jina_response.status_code == 200:
                        jina_data = jina_response.json()
                        
                        if isinstance(jina_data, dict):
                            if jina_data.get("code") == 200 or jina_data.get("status") == 20000:
                                content = jina_data.get("data", {}).get("content", "")
                            elif "content" in jina_data:
                                content = jina_data.get("content", "")
                            else:
                                content = jina_data.get("text", "") or jina_data.get("markdown", "")
                        
                        if content:
                            jina_success = True
                            print(f"  ✅ Jina read success for {title[:50]}... (content length: {len(content)})")
                            break
                except Exception as e:
                    if jina_attempt < MAX_RETRIES:
                        print(f"  Warning: Jina API exception (attempt {jina_attempt + 1}/{MAX_RETRIES + 1}): {str(e)[:100]}")
                        time.sleep(RETRY_DELAY)
            
            if jina_success and content and use_summary:
                summary_prompt = f"""Based on the following webpage content, provide a concise summary that is relevant to the query: "{query}"

Webpage Title: {title}
Content:
{content[:2000]}

Please provide a focused summary (2-3 sentences maximum) that directly addresses the query. Focus on the most relevant information.

You MUST format your response using the following structure:
<think>
[Your thinking process about what information is most relevant to the query]
</think>
<response>
[Your concise summary here - 2-3 sentences maximum]
</response>"""

                gemini_result = call_gemini3_flash_api(summary_prompt, timeout=60)
                
                if gemini_result.get("success") and gemini_result.get("text"):
                    full_response = gemini_result["text"]
                    if full_response:
                        response_match = re.search(r'<response>(.*?)</response>', full_response, re.DOTALL)
                        if response_match:
                            summary = response_match.group(1).strip()
                            if summary:
                                results_summary.append(f"[{len(results_summary)+1}] {title}\n{summary}")
                                text_search_debug["summary_items"].append({
                                    "title": title,
                                    "url": url,
                                    "summary": summary,
                                    "source": "gemini_summary",
                                })
                                continue
                        else:
                            summary = full_response.strip()
                            if summary:
                                results_summary.append(f"[{len(results_summary)+1}] {title}\n{summary}")
                                text_search_debug["summary_items"].append({
                                    "title": title,
                                    "url": url,
                                    "summary": summary,
                                    "source": "gemini_full_response",
                                })
                                continue
                
                results_summary.append(f"[{len(results_summary)+1}] {title}\n{content[:500]}...")
                text_search_debug["summary_items"].append({
                    "title": title,
                    "url": url,
                    "summary": f"{content[:500]}...",
                    "source": "jina_fallback_content",
                })
                continue
            
            if not jina_success and snippet:
                results_summary.append(f"[{len(results_summary)+1}] {title}\n{snippet}")
                print(f"  📄 Using snippet for {title} (Jina read failed)")
                text_search_debug["summary_items"].append({
                    "title": title,
                    "url": url,
                    "summary": snippet,
                    "source": "serper_snippet_fallback",
                })
        
        if results_summary:
            return "Tool execution result:\n" + "\n\n".join(results_summary), text_search_debug
        else:
            return "Tool execution result:\nNo content could be extracted from the search results.", text_search_debug
            
    except Exception as e:
        error_msg = f"Error executing text_search: {str(e)}"
        print(f"  ❌ Error: {error_msg}")
        import traceback
        traceback.print_exc()
        return f"Tool execution error:\n{error_msg}", {
            "request": {
                "q": query,
                "hl": lang,
                "top_k": top_k,
            },
            "error": error_msg,
        }


def execute_search_image(parameters):
    """Execute image search."""
    query = parameters.get('q', '')
    location = parameters.get('location', 'United States')
    hl = parameters.get('hl', 'en')
    num = parameters.get('num', 5)
    
    if not query:
        return "Error: 'q' parameter is required for search_image", {}
    
    url = "https://google.serper.dev/images"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    data = {
        "q": query,
        "location": location,
        "hl": hl,
        "num": num,
        "tbs": "qdr:h",
    }
    
    print(f"  🖼️  Calling search_image API: q='{query}'")
    
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            
            if response.status_code != 200:
                last_error = f"Search image API error: {response.status_code} - {response.text[:200]}"
                if attempt < MAX_RETRIES:
                    print(f"  Warning: search_image API status error (attempt {attempt + 1}/{MAX_RETRIES + 1}): {response.status_code}")
                    print(f"  Retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    print(f"  ❌ Error: {last_error}")
                    return f"Tool execution error:\n{last_error}", {}
            
            result = response.json()
            
            if result.get("code") and result.get("code") != 0:
                last_error = f"Search image API returned error: {result.get('msg', 'Unknown error')}"
                if attempt < MAX_RETRIES:
                    print(f"  Warning: search_image API returned error (attempt {attempt + 1}/{MAX_RETRIES + 1}): {last_error[:100]}")
                    print(f"  Retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    print(f"  ❌ Error: {last_error}")
                    return f"Tool execution error:\n{last_error}", {}
            
            images = result.get("images", [])
            if not images:
                return "Tool execution result:\nNo images found for the query.", {}
            
            images_info = []
            for idx, img in enumerate(images[:num]):
                title = img.get("title", "")
                images_info.append(f"[{idx+1}] {title}")
            
            return "Tool execution result:\n" + "\n\n".join(images_info), result
            
        except Exception as e:
            last_error = f"Error executing search_image: {str(e)}"
            if attempt < MAX_RETRIES:
                print(f"  Warning: search_image API exception (attempt {attempt + 1}/{MAX_RETRIES + 1}): {str(e)[:100]}")
                print(f"  Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
                continue
            else:
                print(f"  ❌ Error: {last_error}")
                return f"Tool execution error:\n{last_error}", {}
    
    return f"Tool execution error:\n{last_error}", {}


def download_and_judge_search_images(search_result, ip_name, ip_intermediate_dir, image_prompt=""):
    """Download searched images and score quality; return best images.

    Args:
        search_result: Raw API response from execute_search_image
        ip_name: IP name for quality evaluation (may be empty)
        ip_intermediate_dir: Directory for intermediate files
        image_prompt: Scene description as evaluation context when ip_name is empty

    Returns:
        (downloaded_images, judge_results): Best image paths and their scores
    """
    downloaded_images = []
    judge_results = []

    if not (search_result and 'images' in search_result):
        return downloaded_images, judge_results

    images = search_result.get('images', [])

    tmp_dir = os.path.join(ip_intermediate_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    print(f"\n📁 Created tmp directory: {tmp_dir}")

    tmp_images = []
    download_count = 0

    for idx, img in enumerate(images):
        image_url = img.get('imageUrl', '')
        if not image_url:
            continue

        print(f"\n  📥 Processing image {idx+1}/{len(images)}: {image_url[:80]}...")

        success, img_bytes, error_msg = download_image_to_bytes(image_url)
        if not success:
            print(f"  ❌ Download failed: {error_msg}")
            continue

        mime_type = detect_image_format_from_bytes(img_bytes)
        if mime_type is None:
            print(f"  ⏭️ Skipping unsupported format")
            continue

        ext = '.jpg' if mime_type == 'image/jpeg' else '.png'

        tmp_path = os.path.join(tmp_dir, f"candidate_{download_count+1}{ext}")
        try:
            with open(tmp_path, 'wb') as f:
                f.write(img_bytes)
            tmp_images.append((tmp_path, image_url))
            download_count += 1
            print(f"  ✅ Saved to tmp: {tmp_path} (format: {mime_type})")
        except Exception as e:
            print(f"  ❌ Failed to save: {e}")

    print(f"\n📊 Downloaded {len(tmp_images)} valid images to tmp directory")

    image_scores = []

    if len(tmp_images) > 0:
        print(f"\n🔍 Starting image quality assessment...")
        for tmp_path, image_url in tmp_images:
            judge_result = judge_image_quality(tmp_path, ip_name, image_prompt=image_prompt)
            score = judge_result.get("score", -1)
            image_scores.append((tmp_path, score, judge_result))
            print(f"    {os.path.basename(tmp_path)}: Score = {score}")

    valid_scores = [(path, score, result) for path, score, result in image_scores if score >= 0]
    valid_scores.sort(key=lambda x: x[1], reverse=True)

    print(f"\n📈 Score ranking (top candidates):")
    for i, (path, score, result) in enumerate(valid_scores[:5]):
        print(f"    {i+1}. {os.path.basename(path)}: {score}/10 - {result.get('reason', '')[:60]}...")

    selected_count = 0
    for tmp_path, score, judge_result in valid_scores[:2]:
        ext = os.path.splitext(tmp_path)[1]
        final_path = os.path.join(ip_intermediate_dir, f"image_{selected_count+1}{ext}")

        try:
            shutil.copy2(tmp_path, final_path)
            downloaded_images.append(final_path)
            judge_results.append({
                "path": os.path.basename(final_path),
                "score": score,
                "reason": judge_result.get("reason", ""),
                "is_text_heavy": judge_result.get("is_text_heavy", False),
                "has_watermark": judge_result.get("has_watermark", False)
            })
            selected_count += 1
            print(f"  ✅ Selected image_{selected_count}: {final_path} (score: {score})")
        except Exception as e:
            print(f"  ❌ Failed to copy: {e}")

    print(f"\n✅ Selected {len(downloaded_images)} best images based on quality scores")
    return downloaded_images, judge_results


def get_tools_definition() -> str:
    """Get tool definitions."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "text_search",
                "description": "Search the web for text information about a query. Use this to get background information about IPs, people, places, or topics.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "q": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "hl": {
                            "type": "string",
                            "description": "Language code (e.g., 'en', 'zh')",
                            "default": "en"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return",
                            "default": 5
                        }
                    },
                    "required": ["q"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_image",
                "description": "Search for images on the web. Use this to find relevant images about the IP or topic.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "q": {
                            "type": "string",
                            "description": "Search query for images"
                        },
                        "location": {
                            "type": "string",
                            "description": "Location for search",
                            "default": "United States"
                        },
                        "hl": {
                            "type": "string",
                            "description": "Language code",
                            "default": "en"
                        },
                        "num": {
                            "type": "integer",
                            "description": "Number of images to return",
                            "default": 5
                        }
                    },
                    "required": ["q"]
                }
            }
        }
    ]
    return json.dumps(tools, indent=2, ensure_ascii=False)


def build_initial_prompt(image_prompt: str, ip_name: str, country: str) -> str:
    """Build initial prompt (supports prompt-only mode without ip_name)."""
    tools_text = f"<tools>\n{get_tools_definition()}\n</tools>"
    
    if ip_name:
        goal_text = f'Create a detailed recaption for "{image_prompt}" about the IP "{ip_name}" ({country}).'
        workflow_step1 = "1. First, search for background information about this IP/character to understand who they are, their characteristics, style, and context. This knowledge will help you craft more accurate image search queries."
        workflow_step2 = "2. Then, search for reference images of this IP/character. Good reference visuals are essential for the final detailed description."
        start_instruction = f'Now, let\'s start by gathering background information about "{ip_name}". Please call `text_search` to learn more about this IP.'
    else:
        country_hint = f" ({country})" if country else ""
        goal_text = f'Create a detailed recaption for the following scene description{country_hint}: "{image_prompt}".'
        workflow_step1 = "1. First, analyze the scene description and search for background information about the key subjects, characters, or elements mentioned. This knowledge will help you craft more accurate image search queries."
        workflow_step2 = "2. Then, search for reference images of the key subjects or characters described. Good reference visuals are essential for the final detailed description."
        start_instruction = f'Now, let\'s start by analyzing the scene description and searching for background information about the key subjects mentioned. Please call `text_search` with an appropriate query derived from the scene description.'

    initial_prompt = f"""{tools_text}

You are helping to build a high-quality visual generation dataset. Your task is to gather information and reference images for creating detailed image descriptions.

**Your Goal**: {goal_text}

**Natural Workflow** (think step by step):
{workflow_step1}
{workflow_step2}
3. Finally, I will provide you with the downloaded reference images, and you will generate a detailed <recaption> that references "image_1" and "image_2" specifically.

**Tool Call Format** (IMPORTANT - use ONLY this format):
<tool_call>
{{"name": "tool_name", "arguments": {{"param1": "value1"}}}}
</tool_call>

Examples:
- Text search: <tool_call>{{"name": "text_search", "arguments": {{"q": "search query", "hl": "zh", "top_k": 2}}}}</tool_call>
- Image search: <tool_call>{{"name": "search_image", "arguments": {{"q": "image query", "hl": "zh", "num": 5}}}}</tool_call>

{start_instruction}"""
    
    return initial_prompt


def build_stage3_prompt(
    image_prompt: str,
    text_search_result: str,
    language: str = "zh",
    num_images: int = 1
) -> str:
    """Build Stage 3 prompt (recaption generation).
    
    Aligned with stage2_prompt_recaption.py:
    - Includes FACIAL FEATURES PRESERVATION requirement
    - Includes preservation statement requirement
    
    Args:
        image_prompt: Image prompt text
        text_search_result: Text search result
        language: Language code (zh/en)
        num_images: Number of reference images (1 or 2)
    """
    if num_images == 2:
        image_reference = '"image_1" and "image_2"'
        analysis_instruction = "[Analysis of image_1...]\n[Analysis of image_2...]\n[Integration strategy...]"
        requirement_text = 'You MUST specifically analyze and explicitly refer to visual details from "image_1" and "image_2" in your <think> process.'
        facial_ref = "image_1 and image_2"
    else:
        image_reference = '"image_1"'
        analysis_instruction = "[Analysis of image_1...]"
        requirement_text = 'You MUST specifically analyze and explicitly refer to visual details from "image_1" in your <think> process.'
        facial_ref = "image_1"

    facial_preservation = (
        f"CRITICAL FACIAL FEATURES PRESERVATION: If {facial_ref} contain identifiable persons, "
        f"you MUST preserve their exact facial features in the <recaption>. Include detailed descriptions of: "
        f"face shape, eyebrow shape, eye characteristics, facial expression, skin tone/complexion, hairstyle, "
        f'and any distinctive facial features. Use format: "Preserve the exact facial features of '
        f'[person name/description] as shown in {facial_ref}: [detailed facial feature description]. '
        f'Maintain [their/his/her] [appearance/clothing/style] as referenced in both images."'
    )

    if language == "zh":
        lang_requirement = (
            f'CRITICAL LANGUAGE REQUIREMENT: The <recaption> content MUST be written entirely in the SAME language '
            f'as the original instruction ("{image_prompt}"). If the instruction is in Chinese, use ONLY Chinese '
            f'in <recaption>. If the instruction is in English, use ONLY English. NEVER mix languages.'
        )
    else:
        lang_requirement = (
            f'CRITICAL LANGUAGE REQUIREMENT: The <recaption> content MUST be written entirely in the SAME language '
            f'as the original instruction ("{image_prompt}"). If the instruction is in English, use ONLY English '
            f'in <recaption>. NEVER mix languages.'
        )

    prompt = f"""[SYSTEM]: Entering Stage 3 (Final Synthesis).
Text Context: {text_search_result[:500] if text_search_result else 'No text context available'}...

Here are the Reference Images you found. 
IMPORTANT: strictly refer to them as {image_reference} in your reasoning.

Task: Generate a detailed recaption for "{image_prompt}".
Requirement: {requirement_text}

{facial_preservation}

{lang_requirement}

Output Format:
<think>
{analysis_instruction}
</think>
<recaption>
[Detailed scene description with preservation statements including facial features...]
</recaption>

IMPORTANT: Output exactly ONE <recaption> block when finished. Do NOT repeat or output multiple recaptions."""

    return prompt


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
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
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