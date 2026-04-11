# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
Stage 2: Given IP image_prompts, use text_search and search_image tools to
generate detailed recaptions through a multi-turn agent trajectory.
"""

import argparse
import base64
import json
import logging
import os
import re
import shutil
import time
import traceback
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

log = logging.getLogger("stage2")

# ---------------------------------------------------------------------------
# API configuration (set via environment variables)
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", None)
REASONING_MODEL = os.environ.get("AGENT_REASONING_MODEL", "gpt-4o")
MULTITURN_MODEL = os.environ.get("AGENT_MULTITURN_MODEL", "gpt-4o")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")
JINA_API_KEY = os.environ.get("JINA_API_KEY", "")

MAX_RETRIES = 3
RETRY_DELAY = 2


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

def _get_openai_client():
    """Get an OpenAI client instance."""
    import openai
    kwargs = {"api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    return openai.OpenAI(**kwargs)


# ---------------------------------------------------------------------------
# LLM API calls (text, multimodal, multi-turn)
# ---------------------------------------------------------------------------

def call_reasoning_api(prompt, system_prompt=None, timeout=60):
    """
    Call an OpenAI-compatible API for text reasoning (with retries).

    Args:
        prompt: User prompt text.
        system_prompt: Optional system prompt.
        timeout: Timeout in seconds.

    Returns:
        dict: {"success": True/False, "text": "...", "raw": None}
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


def call_reasoning_api_with_image(prompt, image_path, system_prompt=None, timeout=60):
    """
    Call an OpenAI-compatible multimodal API for image+text reasoning (with retries).

    Args:
        prompt: User prompt text.
        image_path: Path to an image file.
        system_prompt: Optional system prompt.
        timeout: Timeout in seconds.

    Returns:
        dict: {"success": True/False, "text": "...", "raw": None}
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


def call_multiturn_api(messages, system_instruction=None, max_tokens=32768):
    """
    Call an OpenAI-compatible API for multi-turn dialogue (with retries).

    Args:
        messages: List of messages in standard OpenAI format
                  [{"role": "user"/"assistant", "content": ...}, ...]
        system_instruction: Optional system prompt string.
        max_tokens: Maximum tokens in response.

    Returns:
        str: The assistant's response text.

    Raises:
        Exception: If all retry attempts fail.
    """
    api_messages = []
    if system_instruction:
        api_messages.append({"role": "system", "content": system_instruction})
    api_messages.extend(messages)

    client = _get_openai_client()
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MULTITURN_MODEL,
                messages=api_messages,
                max_tokens=max_tokens,
                temperature=0.7,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            last_error = str(e)
            if attempt < MAX_RETRIES:
                print(f"  Warning: Multi-turn API failed (attempt {attempt + 1}/{MAX_RETRIES + 1}): {str(e)[:200]}")
                time.sleep(RETRY_DELAY)
            else:
                raise Exception(f"Multi-turn API failed after {MAX_RETRIES + 1} attempts: {last_error}")

    raise Exception(f"Multi-turn API failed after {MAX_RETRIES + 1} attempts: {last_error}")


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

def get_tools_definition():
    """Get tool definitions (text_search and search_image only)."""
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
                            "default": 8
                        }
                    },
                    "required": ["q"]
                }
            }
        }
    ]
    return json.dumps(tools, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool call extraction
# ---------------------------------------------------------------------------

def extract_tool_call(text):
    """Extract the first tool call from text. Returns a JSON string or None."""

    # Reject incorrect formats
    if re.search(r'<function_calls>', text, re.IGNORECASE):
        print(f"  ERROR: Detected <function_calls> format. Must use <tool_call> format instead.")
        return None
    if re.search(r'<invoke\s+name=', text, re.IGNORECASE):
        print(f"  ERROR: Detected <invoke> format. Must use <tool_call> format instead.")
        return None

    tool_call_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    tool_matches = list(re.finditer(tool_call_pattern, text, re.DOTALL))

    if len(tool_matches) > 1:
        print(f"  Warning: Found {len(tool_matches)} tool calls in response, only processing the first one")

    if tool_matches:
        tool_match = tool_matches[0]
        tool_json = tool_match.group(1).strip()
        try:
            tool_dict = json.loads(tool_json)
            params = tool_dict.get('arguments', tool_dict.get('parameters', {}))
            tool_name = tool_dict.get('name', '')

            if tool_name in ['text_search', 'local_search', 'web_search']:
                if 'query' in params and 'q' not in params:
                    params['q'] = params.pop('query')
                if 'lang' in params and 'hl' not in params:
                    params['hl'] = params.pop('lang')

            return json.dumps({
                "name": tool_dict['name'],
                "parameters": params
            }, ensure_ascii=False)
        except Exception as e:
            print(f"  Warning: Failed to parse tool_call JSON: {e}")
            pass

    # Fallback: <text_search> or <local_search> tags
    text_search_pattern = r'<(?:text_search|local_search)>\s*(\{.*?\})\s*</(?:text_search|local_search)>'
    text_match = re.search(text_search_pattern, text, re.DOTALL)
    if text_match:
        tool_json = text_match.group(1).strip()
        try:
            tool_dict = json.loads(tool_json)
            if 'query' in tool_dict and 'q' not in tool_dict:
                tool_dict['q'] = tool_dict.pop('query')
            if 'lang' in tool_dict and 'hl' not in tool_dict:
                tool_dict['hl'] = tool_dict.pop('lang')
            return json.dumps({
                "name": "text_search",
                "parameters": tool_dict
            }, ensure_ascii=False)
        except Exception:
            return json.dumps({
                "name": "text_search",
                "parameters": {"q": tool_json}
            }, ensure_ascii=False)

    # Fallback: <search_image> tag
    search_image_pattern = r'<search_image>\s*(\{.*?\})\s*</search_image>'
    search_image_match = re.search(search_image_pattern, text, re.DOTALL)
    if search_image_match:
        tool_json = search_image_match.group(1).strip()
        try:
            tool_dict = json.loads(tool_json)
            return json.dumps({
                "name": "search_image",
                "parameters": tool_dict
            }, ensure_ascii=False)
        except Exception:
            return json.dumps({
                "name": "search_image",
                "parameters": {"q": tool_json}
            }, ensure_ascii=False)

    return None


# ---------------------------------------------------------------------------
# Web search execution (Serper + Jina)
# ---------------------------------------------------------------------------

def execute_text_search(parameters):
    """Execute text_search tool via Serper API + Jina reader."""
    query = parameters.get('q', '')
    lang = parameters.get('hl', 'en')
    top_k = parameters.get('top_k', 5)

    if not query:
        return "Error: 'q' parameter is required for text_search", {}

    try:
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
                        time.sleep(RETRY_DELAY)
                        continue
                    else:
                        print(f"  Error: {last_serper_error}")
                        return f"Tool execution error:\n{last_serper_error}", {}

                serper_data = serper_response.json()

                if serper_data.get("code") and serper_data.get("code") != 0:
                    last_serper_error = f"Serper API returned error: {serper_data.get('msg', 'Unknown error')}"
                    if attempt < MAX_RETRIES:
                        print(f"  Warning: Serper API returned error (attempt {attempt + 1}/{MAX_RETRIES + 1}): {last_serper_error[:100]}")
                        time.sleep(RETRY_DELAY)
                        continue
                    else:
                        print(f"  Error: {last_serper_error}")
                        return f"Tool execution error:\n{last_serper_error}", {}

                break

            except requests.exceptions.RequestException as e:
                last_serper_error = f"Serper API request exception: {str(e)}"
                if attempt < MAX_RETRIES:
                    print(f"  Warning: Serper API exception (attempt {attempt + 1}/{MAX_RETRIES + 1}): {str(e)[:100]}")
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    print(f"  Error: {last_serper_error}")
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
                results_summary.append(f"[Answer Box] {title}\n{snippet}\nSource: {link}")
                print(f"  Found answerBox with snippet")

        organic_results = serper_data.get("organic", [])

        if not organic_results and not results_summary:
            return "Tool execution result:\nNo relevant web pages found for the query.", {}

        jina_headers = {
            'Authorization': f'Bearer {JINA_API_KEY}',
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

            if JINA_API_KEY:
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
                                print(f"  Jina read success for {title[:50]}... (content length: {len(content)})")
                                break
                            else:
                                if jina_attempt < MAX_RETRIES:
                                    print(f"  Warning: Jina returned no content (attempt {jina_attempt + 1}/{MAX_RETRIES + 1}) for {title[:50]}...")
                                    time.sleep(RETRY_DELAY)
                                else:
                                    print(f"  Warning: Jina response for {title[:50]}... has no content field")
                        else:
                            if jina_attempt < MAX_RETRIES:
                                print(f"  Warning: Jina API status {jina_response.status_code} (attempt {jina_attempt + 1}/{MAX_RETRIES + 1}) for {url[:50]}...")
                                time.sleep(RETRY_DELAY)
                            else:
                                print(f"  Warning: Jina API returned status {jina_response.status_code} for {url[:50]}...")

                    except Exception as e:
                        if jina_attempt < MAX_RETRIES:
                            print(f"  Warning: Jina API exception (attempt {jina_attempt + 1}/{MAX_RETRIES + 1}) for {url[:50]}...: {str(e)[:100]}")
                            time.sleep(RETRY_DELAY)
                        else:
                            print(f"  Warning: Error processing URL {url[:50]}... with Jina: {str(e)[:100]}")

            if jina_success and content:
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

                gemini_result = call_reasoning_api(summary_prompt, timeout=60)

                if gemini_result.get("success") and gemini_result.get("text"):
                    full_response = gemini_result["text"]
                    if full_response:
                        response_match = re.search(r'<response>(.*?)</response>', full_response, re.DOTALL)
                        if response_match:
                            summary = response_match.group(1).strip()
                            if summary:
                                results_summary.append(f"[{len(results_summary)+1}] {title}\n{summary}\nSource: {url}")
                                continue
                        else:
                            print(f"  Warning: No <response> tag found in LLM response, using full response")
                            summary = full_response.strip()
                            if summary:
                                results_summary.append(f"[{len(results_summary)+1}] {title}\n{summary}\nSource: {url}")
                                continue
                else:
                    error_msg = gemini_result.get("error", "Unknown error")
                    print(f"  Warning: LLM summarization failed: {error_msg}")

                results_summary.append(f"[{len(results_summary)+1}] {title}\n{content[:500]}...\nSource: {url}")
                continue

            if not jina_success and snippet:
                results_summary.append(f"[{len(results_summary)+1}] {title}\n{snippet}\nSource: {url}")
                print(f"  Using snippet for {title} (Jina read failed)")

        if results_summary:
            return "Tool execution result:\n" + "\n\n".join(results_summary), {}
        else:
            return "Tool execution result:\nNo content could be extracted from the search results.", {}

    except Exception as e:
        error_msg = f"Error executing text_search: {str(e)}"
        print(f"  Error: {error_msg}")
        traceback.print_exc()
        return f"Tool execution error:\n{error_msg}", {}


def execute_search_image(parameters):
    """Execute search_image tool via Serper API (with retries)."""
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

    print(f"  Calling search_image API: q='{query}'")

    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)

            if response.status_code != 200:
                last_error = f"Search image API error: {response.status_code} - {response.text[:200]}"
                if attempt < MAX_RETRIES:
                    print(f"  Warning: search_image API status error (attempt {attempt + 1}/{MAX_RETRIES + 1}): {response.status_code}")
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    print(f"  Error: {last_error}")
                    return f"Tool execution error:\n{last_error}", {}

            result = response.json()

            if result.get("code") and result.get("code") != 0:
                last_error = f"Search image API returned error: {result.get('msg', 'Unknown error')}"
                if attempt < MAX_RETRIES:
                    print(f"  Warning: search_image API returned error (attempt {attempt + 1}/{MAX_RETRIES + 1}): {last_error[:100]}")
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    print(f"  Error: {last_error}")
                    return f"Tool execution error:\n{last_error}", {}

            images = result.get("images", [])
            if not images:
                return "Tool execution result:\nNo images found for the query.", {}

            images_info = []
            for idx, img in enumerate(images[:num]):
                image_url = img.get("imageUrl", "")
                title = img.get("title", "")
                source = img.get("source", "")
                link = img.get("link", "")
                images_info.append(f"[{idx+1}] {title}\nImage URL: {image_url}\nSource: {source}\nLink: {link}")

            return "Tool execution result:\n" + "\n\n".join(images_info), result

        except Exception as e:
            last_error = f"Error executing search_image: {str(e)}"
            if attempt < MAX_RETRIES:
                print(f"  Warning: search_image API exception (attempt {attempt + 1}/{MAX_RETRIES + 1}): {str(e)[:100]}")
                time.sleep(RETRY_DELAY)
                continue
            else:
                print(f"  Error: {last_error}")
                return f"Tool execution error:\n{last_error}", {}

    return f"Tool execution error:\n{last_error}", {}


# ---------------------------------------------------------------------------
# Image download utilities
# ---------------------------------------------------------------------------

def download_image(image_url, save_path):
    """Download an image to a file (with retries)."""
    proxies = {
        'http': os.environ.get('http_proxy', ''),
        'https': os.environ.get('https_proxy', ''),
    }
    if not proxies['http']:
        proxies = None

    try:
        from urllib.parse import urlparse
        parsed_url = urlparse(image_url)
        referer = f"{parsed_url.scheme}://{parsed_url.netloc}/"
    except Exception:
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
            with open(save_path, "wb") as f:
                f.write(response.content)
            print(f"  Image downloaded: {save_path}")
            return True
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                print(f"  Warning: Image download failed (attempt {attempt + 1}/{MAX_RETRIES + 1}): {str(e)[:100]}")
                time.sleep(RETRY_DELAY)
            else:
                print(f"  Failed to download image after {MAX_RETRIES + 1} attempts: {last_error}")
                return False

    return False


def download_image_to_bytes(image_url):
    """Download image and return raw bytes (with retries).

    Returns:
        tuple: (success: bool, img_bytes: bytes or None, error_msg: str or None)
    """
    proxies = {
        'http': os.environ.get('http_proxy', ''),
        'https': os.environ.get('https_proxy', ''),
    }
    if not proxies['http']:
        proxies = None

    try:
        from urllib.parse import urlparse
        parsed_url = urlparse(image_url)
        referer = f"{parsed_url.scheme}://{parsed_url.netloc}/"
    except Exception:
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
                time.sleep(RETRY_DELAY)

    return False, None, last_error


# ---------------------------------------------------------------------------
# Image format detection & compression
# ---------------------------------------------------------------------------

def detect_image_format_from_bytes(img_bytes):
    """Detect image MIME type from raw bytes.

    Returns:
        str: MIME type string (e.g. 'image/jpeg', 'image/png'), or
             None if the format is unsupported (gif/webp).
    """
    try:
        if len(img_bytes) >= 3 and img_bytes[:3] == b'\xff\xd8\xff':
            return 'image/jpeg'
        elif len(img_bytes) >= 8 and img_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            return 'image/png'
        elif len(img_bytes) >= 6:
            if img_bytes[:6] == b'GIF87a' or img_bytes[:6] == b'GIF89a':
                print(f"  Warning: GIF format detected, skipping (not supported)")
                return None
        if len(img_bytes) >= 12:
            if img_bytes[:4] == b'RIFF' and img_bytes[8:12] == b'WEBP':
                print(f"  Warning: WebP format detected, skipping (not supported)")
                return None

        try:
            img = Image.open(BytesIO(img_bytes))
            format_lower = img.format.lower() if img.format else 'jpeg'
            if format_lower in ['gif', 'webp']:
                print(f"  Warning: {format_lower.upper()} format detected via PIL, skipping (not supported)")
                return None
            format_map = {
                'jpeg': 'image/jpeg',
                'jpg': 'image/jpeg',
                'png': 'image/png',
            }
            return format_map.get(format_lower, 'image/jpeg')
        except Exception:
            pass

        return 'image/jpeg'
    except Exception as e:
        print(f"  Warning: Error detecting image format: {e}, defaulting to image/jpeg")
        return 'image/jpeg'


MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB


def compress_image_if_needed(image_path, max_size_bytes=MAX_IMAGE_SIZE_BYTES):
    """Compress image if it exceeds the size limit.

    Args:
        image_path: Path to image file.
        max_size_bytes: Maximum file size in bytes (default 5 MB).

    Returns:
        True if already within limit or compressed successfully,
        a new file path string if format changed (e.g. PNG -> JPG),
        or False on failure.
    """
    try:
        file_size = os.path.getsize(image_path)
        if file_size <= max_size_bytes:
            return True

        print(f"  Warning: Image size {file_size / 1024 / 1024:.2f}MB exceeds {max_size_bytes / 1024 / 1024:.2f}MB, compressing...")

        img = Image.open(image_path)

        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')

        quality = 95
        min_quality = 20

        while quality >= min_quality:
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=quality, optimize=True)
            compressed_size = buffer.tell()

            if compressed_size <= max_size_bytes:
                new_path = image_path
                if image_path.lower().endswith('.png'):
                    new_path = image_path[:-4] + '.jpg'
                    if new_path != image_path and os.path.exists(image_path):
                        os.remove(image_path)

                img.save(new_path, format='JPEG', quality=quality, optimize=True)
                new_size = os.path.getsize(new_path)
                print(f"  Compressed to {new_size / 1024 / 1024:.2f}MB (quality={quality})")

                if new_path != image_path:
                    return new_path
                return True

            quality -= 5

        print(f"  Warning: Quality reduction not enough, resizing image...")

        scale = 0.9
        while scale >= 0.3:
            new_width = int(img.width * scale)
            new_height = int(img.height * scale)
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)

            buffer = BytesIO()
            resized_img.save(buffer, format='JPEG', quality=85, optimize=True)
            compressed_size = buffer.tell()

            if compressed_size <= max_size_bytes:
                new_path = image_path
                if image_path.lower().endswith('.png'):
                    new_path = image_path[:-4] + '.jpg'
                    if new_path != image_path and os.path.exists(image_path):
                        os.remove(image_path)

                resized_img.save(new_path, format='JPEG', quality=85, optimize=True)
                new_size = os.path.getsize(new_path)
                print(f"  Resized to {new_width}x{new_height}, {new_size / 1024 / 1024:.2f}MB")

                if new_path != image_path:
                    return new_path
                return True

            scale -= 0.1

        print(f"  Failed to compress image below {max_size_bytes / 1024 / 1024:.2f}MB")
        return False

    except Exception as e:
        print(f"  Error compressing image: {e}")
        return False


# ---------------------------------------------------------------------------
# Image quality judging
# ---------------------------------------------------------------------------

def judge_image_quality(image_path, ip_name):
    """Score image quality using an LLM judge.

    Args:
        image_path: Path to image file.
        ip_name: IP name for evaluation context.

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

    question = f"""Please evaluate the provided image for the IP: "{ip_name}".

Assess the image based on the system rubrics.
Return your response in JSON format ONLY with the following structure:
{{
    "score": <int, 0-10>,
    "reason": "<string, a concise explanation of the score, mentioning any specific flaws like 'text-heavy', 'watermark', or 'blurry'>",
    "is_text_heavy": <bool>,
    "has_watermark": <bool>
}}
"""

    print(f"  Judging image quality: {os.path.basename(image_path)} for IP '{ip_name}'")

    result = call_reasoning_api_with_image(question, image_path, system_prompt=sys_prompt, timeout=60)

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
                        print(f"  Warning: Failed to extract JSON from response: {response_text[:200]}")
                        return {"score": -1, "reason": "Failed to parse JSON response", "is_text_heavy": False, "has_watermark": False}

            score = judge_result.get("score", -1)
            reason = judge_result.get("reason", "No reason provided")
            is_text_heavy = judge_result.get("is_text_heavy", False)
            has_watermark = judge_result.get("has_watermark", False)

            print(f"  Score: {score}/10 - {reason[:80]}...")

            return {
                "score": score,
                "reason": reason,
                "is_text_heavy": is_text_heavy,
                "has_watermark": has_watermark
            }

        except Exception as e:
            print(f"  Error parsing judge result: {e}")
            return {"score": -1, "reason": f"Parse error: {str(e)}", "is_text_heavy": False, "has_watermark": False}
    else:
        error_msg = result.get("error", "Unknown error")
        print(f"  Judge API failed: {error_msg}")
        return {"score": -1, "reason": f"API error: {error_msg}", "is_text_heavy": False, "has_watermark": False}


# ---------------------------------------------------------------------------
# Recaption text helpers
# ---------------------------------------------------------------------------

def has_recaption_tag(text):
    """Check if text contains recaption tags (signals conversation end)."""
    return '<recaption>' in text.lower() or 'recaption' in text.lower()


def extract_recaption_content(text):
    """Extract content from <recaption> tags.

    Returns:
        str: Content inside tags, or original text if tags not found.
    """
    if not text:
        return text

    match = re.search(r'<recaption>(.*?)</recaption>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    match = re.search(r'<recaption>(.*)', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return text


def normalize_recaption_text(text, language='zh'):
    """Normalize recaption text by standardizing XML tag casing."""
    if not text:
        return text

    normalized = text
    normalized = re.sub(r'<Think>', '<think>', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'</Think>', '</think>', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'<Instruction>', '<recaption>', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'</Instruction>', '</recaption>', normalized, flags=re.IGNORECASE)

    return normalized


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

def load_prompt_template(country=None):
    """Load recaption system prompt based on country.

    Args:
        country: If "中国", use Chinese template; otherwise English.
    """
    if country == "中国":
        print(f"  Using Chinese template")
        return get_chinese_recaption_prompt()
    else:
        print(f"  Using English template")
        return get_english_recaption_prompt()


def get_english_recaption_prompt():
    """English recaption system prompt template."""
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

2. **Descriptive Style**: The content of <Instruction> must be a **description of the final result** (Descriptive), NOT an editing command (Imperative).
   - BAD: "Please put the man from image_1 on the left..."
   - GOOD: "In this realistic outdoor portrait, the man from image_1 stands on the left..."

3. **Preservation Statement**: At the end of the description, you MUST explicitly state what specifically is preserved from image_1 and image_2.
   - Must include phrases like: "The final image completely preserves [features] from image_1..." and "The final image fully retains [features] from image_2...".
   - **Facial Features Preservation (CRITICAL)**: If image_1 and/or image_2 contain identifiable persons, you MUST preserve their exact facial features as shown in the reference images. Include detailed descriptions such as: face shape, eyebrow shape, eye characteristics, facial expression, skin tone/complexion, hairstyle, and any distinctive facial features. Use the format: "Preserve the exact facial features of [person name/description] as shown in image_1 and image_2: [detailed facial feature description]. Maintain [their/his/her] [appearance/clothing/style] as referenced in both images."

4. **Language Consistency (MANDATORY)**: The content of <recaption> MUST be written entirely in the SAME language as the original instruction. If the original instruction is in Chinese, ALL descriptions in <recaption> must be in Chinese - NO English allowed. If the original instruction is in English, use English throughout. NEVER mix languages within the same description.

Start the task. Output <think> and <recaption>.'''


def get_chinese_recaption_prompt():
    """Chinese recaption system prompt template."""
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


# ---------------------------------------------------------------------------
# Stage 3 image injection prompt builder
# ---------------------------------------------------------------------------

def _build_stage3_content(image_prompt, text_search_result, downloaded_images, tool_output=None):
    """Build Stage 3 multimodal content list with image injection.

    Args:
        image_prompt: The image prompt string.
        text_search_result: Cached text search result.
        downloaded_images: List of local image file paths.
        tool_output: Optional tool output to prepend (for immediate injection).

    Returns:
        list: Content list in standard OpenAI multimodal format.
    """
    content_list = []

    observation_prefix = ""
    if tool_output:
        observation_prefix = f"""<observation>
{tool_output}
</observation>

"""

    prompt_intro = f"""{observation_prefix}[SYSTEM]: Entering Stage 3 (Final Synthesis).
Text Context: {text_search_result[:500] if text_search_result else 'No text context available'}...

Here are the Reference Images you found. 
IMPORTANT: strictly refer to them as "image_1" and "image_2" in your reasoning.

Task: Generate a detailed recaption for "{image_prompt}".
Requirement: You MUST specifically analyze and explicitly refer to visual details from "image_1" and "image_2" in your <think> process.

CRITICAL FACIAL FEATURES PRESERVATION: If image_1 and/or image_2 contain identifiable persons, you MUST preserve their exact facial features in the <recaption>. Include detailed descriptions of: face shape, eyebrow shape, eye characteristics, facial expression, skin tone/complexion, hairstyle, and any distinctive facial features. Use format: "Preserve the exact facial features of [person name/description] as shown in image_1 and image_2: [detailed facial feature description]. Maintain [their/his/her] [appearance/clothing/style] as referenced in both images."

CRITICAL LANGUAGE REQUIREMENT: The <recaption> content MUST be written entirely in the SAME language as the original instruction ("{image_prompt}"). If the instruction is in Chinese, use ONLY Chinese in <recaption>. If the instruction is in English, use ONLY English. NEVER mix languages.

Output Format:
<think>
[Analysis of image_1...]
[Analysis of image_2...]
[Integration strategy...]
</think>
<recaption>
[Detailed scene description with preservation statements including facial features...]
</recaption>

Output <recaption> when finished."""

    content_list.append({"type": "text", "text": prompt_intro})

    for idx, img_path in enumerate(downloaded_images[:2]):
        if os.path.exists(img_path):
            ref_name = f"image_{idx+1}"

            with open(img_path, 'rb') as f:
                img_bytes = f.read()
                mime_type = detect_image_format_from_bytes(img_bytes)
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')

                content_list.append({
                    "type": "text",
                    "text": f"\n[Reference Image {idx+1} (ID: {ref_name})]:\n"
                })

                content_list.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{img_base64}"
                    }
                })
            print(f"  Adding Reference Image {idx+1} (ID: {ref_name}): {img_path} (format: {mime_type})")

    content_list.append({"type": "text", "text": "\nBased on the above, generate <think> and <recaption> now."})

    return content_list


# ---------------------------------------------------------------------------
# Main IP processing
# ---------------------------------------------------------------------------

def process_ip(ip_data, ip_index, output_dir):
    """Process a single IP through the 3-stage trajectory.

    Args:
        ip_data: Dict with ip_name, image_prompt, language, country.
        ip_index: Unique index/key for this IP.
        output_dir: Root output directory for trajectories.
    """
    ip_name = ip_data.get('ip_name', '')
    image_prompt = ip_data.get('image_prompt', '')
    language = ip_data.get('language', 'zh')
    country = ip_data.get('country', '')

    intermediate_dir = os.path.join(output_dir, "intermediate")
    os.makedirs(intermediate_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing IP: {ip_name} (index: {ip_index})")
    print(f"Goal: Force 3-stage workflow (Text -> Image -> Recaption)")
    print(f"Country: {country}")
    print(f"Image Prompt: {image_prompt}")
    print(f"{'='*60}\n")

    ip_intermediate_dir = os.path.join(intermediate_dir, str(ip_index))
    os.makedirs(ip_intermediate_dir, exist_ok=True)

    trajectory = {
        'ip_index': ip_index,
        'ip_name': ip_name,
        'image_prompt': image_prompt,
        'language': language,
        'country': country,
        'turns': []
    }

    tools_text = f"<tools>\n{get_tools_definition()}\n</tools>"

    initial_prompt = f"""You are helping to build a high-quality visual generation dataset. Your task is to gather information and reference images for creating detailed image descriptions.

**Your Goal**: Create a detailed recaption for "{image_prompt}" about the IP "{ip_name}" ({country}).

**Natural Workflow** (think step by step):
1. First, search for background information about this IP/character to understand who they are, their characteristics, style, and context. This knowledge will help you craft more accurate image search queries.
2. Then, search for reference images of this IP/character. Good reference visuals are essential for the final detailed description.
3. Finally, I will provide you with the downloaded reference images, and you will generate a detailed <recaption> that references "image_1" and "image_2" specifically.

**Tool Call Format** (IMPORTANT - use ONLY this format):
<tool_call>
{{"name": "tool_name", "arguments": {{"param1": "value1"}}}}
</tool_call>

Examples:
- Text search: <tool_call>{{"name": "text_search", "arguments": {{"q": "search query", "hl": "zh", "top_k": 5}}}}</tool_call>
- Image search: <tool_call>{{"name": "search_image", "arguments": {{"q": "image query", "hl": "zh", "num": 8}}}}</tool_call>

Now, let's start by gathering background information about "{ip_name}". Please call `text_search` to learn more about this IP."""

    # Standard OpenAI message format
    chat_messages = [{
        "role": "user",
        "content": tools_text + "\n\n" + initial_prompt
    }]

    MAX_TURNS = 20
    downloaded_images = []
    text_search_result = ""

    # State machine: 1=Text, 2=Image, 3=Recaption
    stage = 1
    recaption_started = False

    for turn_num in range(MAX_TURNS):
        print(f"\n{'─'*60}")
        print(f"Turn {turn_num + 1} (Current Stage: {stage})")
        print(f"{'─'*60}\n")

        try:
            # Extract current input (last user message text) for logging
            current_input = ""
            if chat_messages:
                for msg in reversed(chat_messages):
                    if msg.get('role') == 'user':
                        content = msg.get('content', '')
                        if isinstance(content, str):
                            current_input = content
                        elif isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and item.get('type') == 'text':
                                    current_input += item.get('text', '')
                        break

            print(f"Calling multi-turn API...")
            response_text = call_multiturn_api(
                messages=chat_messages,
                system_instruction=None,
                max_tokens=32768
            )

            print(f"Model response (first 200 chars):")
            print(response_text[:200] + "...")

            current_turn_data = {
                'turn': turn_num,
                'stage': stage,
                'input': current_input,
                'response_text': response_text,
                'tool_output': None
            }
            trajectory['turns'].append(current_turn_data)

            # Append assistant response to conversation
            chat_messages.append({
                "role": "assistant",
                "content": response_text
            })

            # Check if conversation is complete
            if stage == 3 and has_recaption_tag(response_text):
                print(f"\nRecaption tag detected, conversation completed")
                if '<result>' in response_text or '<think>' in response_text.lower() or '<recaption>' in response_text.lower():
                    normalized_recaption = normalize_recaption_text(response_text, language=language)
                    trajectory['recaption'] = normalized_recaption
                break

            # -----------------------------------------------------------
            # State machine: Stage 3 image injection (when no tool call)
            # -----------------------------------------------------------
            print(f"  Debug: stage={stage}, recaption_started={recaption_started}, downloaded_images count={len(downloaded_images)}")
            if stage == 3 and not recaption_started:
                tool_call_json = extract_tool_call(response_text)
                if not tool_call_json:
                    # Try to find images from intermediate directory if none downloaded
                    if len(downloaded_images) == 0:
                        print(f"  downloaded_images is empty, checking intermediate directory: {ip_intermediate_dir}")
                        for img_num in [1, 2]:
                            img_found = False
                            for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                                img_path = os.path.join(ip_intermediate_dir, f"image_{img_num}{ext}")
                                if os.path.exists(img_path):
                                    downloaded_images.append(img_path)
                                    print(f"  Found existing image: {img_path}")
                                    img_found = True
                                    break
                            if not img_found:
                                print(f"  Warning: image_{img_num} not found in {ip_intermediate_dir}")

                    if len(downloaded_images) == 0:
                        print(f"  ERROR: No images found in intermediate directory. Cannot proceed with Stage 3.")
                        try:
                            files = os.listdir(ip_intermediate_dir)
                            print(f"  Files in directory: {files}")
                        except Exception as e:
                            print(f"  Cannot list directory: {e}")

                    if len(downloaded_images) >= 1:
                        recaption_started = True
                        print(f"\nPreparing Stage 3 Prompt with Images...")
                        print(f"  Using {len(downloaded_images)} images from downloaded_images list")

                        final_system_prompt = load_prompt_template(country=country)

                        content_list = _build_stage3_content(
                            image_prompt, text_search_result, downloaded_images
                        )

                        recaption_messages = chat_messages + [{
                            "role": "user",
                            "content": content_list
                        }]

                        recaption_text = call_multiturn_api(
                            messages=recaption_messages,
                            system_instruction=final_system_prompt,
                            max_tokens=32768
                        )

                        if recaption_text:
                            normalized_recaption = normalize_recaption_text(recaption_text, language=language)
                            trajectory['recaption'] = normalized_recaption
                            print(f"Recaption generation completed")
                            print(f"Recaption content (first 200 chars): {normalized_recaption[:200]}...")

                            if trajectory['turns']:
                                trajectory['turns'][-1]['recaption'] = normalized_recaption
                        else:
                            print(f"Warning: Recaption generation failed, no content retrieved")

                        break
                    else:
                        print(f"  ERROR: No images found for Stage 3. Cannot generate recaption.")

            # -----------------------------------------------------------
            # Tool call handling
            # -----------------------------------------------------------
            tool_call_json = extract_tool_call(response_text)

            if tool_call_json:
                try:
                    tool_dict = json.loads(tool_call_json) if isinstance(tool_call_json, str) else tool_call_json
                    tool_name = tool_dict.get('name', '')
                    parameters = tool_dict.get('parameters', {})

                    print(f"\nTool call detected: {tool_name}")
                    print(f"  Parameters: {json.dumps(parameters, ensure_ascii=False)}")

                    tool_output = ""
                    next_instruction = ""
                    should_inject_images_immediately = False

                    if tool_name == 'text_search':
                        tool_result, _ = execute_text_search(parameters)
                        tool_output = tool_result
                        text_search_result = tool_result
                        stage = 2
                        next_instruction = """\n\nGreat, now you have background knowledge about this IP. Based on what you learned, please search for reference images that capture the visual characteristics of this IP/character. Use the information from the text search to craft a more precise image query.

Call `search_image` to find reference visuals:
<tool_call>
{"name": "search_image", "arguments": {"q": "your refined image query based on what you learned", "hl": "zh", "num": 8}}
</tool_call>"""

                    elif tool_name == 'search_image':
                        parameters['num'] = 5
                        tool_result, search_result = execute_search_image(parameters)
                        tool_output = tool_result

                        if search_result and 'images' in search_result:
                            images = search_result.get('images', [])

                            tmp_dir = os.path.join(ip_intermediate_dir, "tmp")
                            os.makedirs(tmp_dir, exist_ok=True)
                            print(f"\nCreated tmp directory: {tmp_dir}")

                            tmp_images = []
                            download_count = 0

                            for idx, img in enumerate(images):
                                image_url = img.get('imageUrl', '')
                                if not image_url:
                                    continue

                                print(f"\n  Processing image {idx+1}/{len(images)}: {image_url[:80]}...")

                                success, img_bytes, error_msg = download_image_to_bytes(image_url)
                                if not success:
                                    print(f"  Download failed: {error_msg}")
                                    continue

                                mime_type = detect_image_format_from_bytes(img_bytes)
                                if mime_type is None:
                                    print(f"  Skipping unsupported format")
                                    continue

                                ext = '.jpg' if mime_type == 'image/jpeg' else '.png'

                                tmp_path = os.path.join(tmp_dir, f"candidate_{download_count+1}{ext}")
                                try:
                                    with open(tmp_path, 'wb') as f:
                                        f.write(img_bytes)
                                    tmp_images.append((tmp_path, image_url))
                                    download_count += 1
                                    print(f"  Saved to tmp: {tmp_path} (format: {mime_type})")
                                except Exception as e:
                                    print(f"  Failed to save: {e}")

                            print(f"\nDownloaded {len(tmp_images)} valid images to tmp directory")

                            image_scores = []

                            if len(tmp_images) > 0:
                                print(f"\nStarting image quality assessment...")
                                for tmp_path, image_url in tmp_images:
                                    judge_result = judge_image_quality(tmp_path, ip_name)
                                    score = judge_result.get("score", -1)
                                    image_scores.append((tmp_path, score, judge_result))
                                    print(f"    {os.path.basename(tmp_path)}: Score = {score}")

                            valid_scores = [(path, score, result) for path, score, result in image_scores if score >= 0]
                            valid_scores.sort(key=lambda x: x[1], reverse=True)

                            print(f"\nScore ranking (top candidates):")
                            for i, (path, score, result) in enumerate(valid_scores[:5]):
                                print(f"    {i+1}. {os.path.basename(path)}: {score}/10 - {result.get('reason', '')[:60]}...")

                            selected_count = 0
                            for tmp_path, score, judge_result in valid_scores[:2]:
                                ext = os.path.splitext(tmp_path)[1]
                                final_path = os.path.join(ip_intermediate_dir, f"image_{selected_count+1}{ext}")

                                try:
                                    shutil.copy2(tmp_path, final_path)

                                    compress_result = compress_image_if_needed(final_path)
                                    if isinstance(compress_result, str):
                                        final_path = compress_result
                                    elif not compress_result:
                                        print(f"  Warning: Could not compress {final_path} below 5MB")

                                    downloaded_images.append(final_path)
                                    selected_count += 1
                                    print(f"  Selected image_{selected_count}: {final_path} (score: {score})")
                                except Exception as e:
                                    print(f"  Failed to copy: {e}")

                            print(f"\nSelected {len(downloaded_images)} best images based on quality scores")

                            current_turn_data['image_judge_results'] = [
                                {
                                    "path": os.path.basename(path),
                                    "score": score,
                                    "reason": result.get("reason", ""),
                                    "is_text_heavy": result.get("is_text_heavy", False),
                                    "has_watermark": result.get("has_watermark", False)
                                }
                                for path, score, result in image_scores
                            ]

                        if len(downloaded_images) >= 1:
                            stage = 3
                            should_inject_images_immediately = True
                            print(f"  Stage 2 Complete. {len(downloaded_images)} images downloaded. Will inject images immediately.")
                        else:
                            next_instruction = """\n\nIt seems no images were successfully downloaded. Please try searching again with a different or more specific query. Consider using the IP's name along with visual descriptors.

<tool_call>
{"name": "search_image", "arguments": {"q": "try a different query", "hl": "zh", "num": 5}}
</tool_call>"""
                    else:
                        tool_output = f"Unknown tool: {tool_name}"

                    print(f"  Tool result: {tool_output[:100]}...")
                    current_turn_data['tool_output'] = tool_output

                    # If Stage 2 complete with images, immediately inject for Stage 3
                    if should_inject_images_immediately and len(downloaded_images) >= 1:
                        print(f"\nStage 2 completed with images, immediately injecting images for Stage 3...")

                        final_system_prompt = load_prompt_template(country=country)

                        content_list = _build_stage3_content(
                            image_prompt, text_search_result, downloaded_images,
                            tool_output=tool_output
                        )

                        chat_messages.append({
                            "role": "user",
                            "content": content_list
                        })

                        recaption_started = True
                        continue
                    else:
                        chat_messages.append({
                            "role": "user",
                            "content": f"<observation>\n{tool_output}\n</observation>{next_instruction}"
                        })

                except Exception as e:
                    print(f"  Error executing tool: {e}")
                    error_msg = f"Error executing tool: {str(e)}"
                    current_turn_data['tool_error'] = error_msg
                    chat_messages.append({
                        "role": "user",
                        "content": f"<observation>\n{error_msg}\n</observation>"
                    })
            else:
                # No tool call detected; guide the model based on current stage
                print(f"  No tool call detected")

                wrong_format_detected = False
                format_error_msg = ""
                if re.search(r'<function_calls>', response_text, re.IGNORECASE):
                    wrong_format_detected = True
                    format_error_msg = "You used <function_calls> format, which is WRONG."
                elif re.search(r'<invoke\s+name=', response_text, re.IGNORECASE):
                    wrong_format_detected = True
                    format_error_msg = "You used <invoke> format, which is WRONG."

                if stage == 3 and not has_recaption_tag(response_text):
                    pass
                elif stage < 3:
                    example_params = '{"q": "your query", "hl": "zh", "top_k": 5}' if stage == 1 else '{"q": "your query", "hl": "zh", "num": 5}'

                    if wrong_format_detected:
                        error_prefix = f"I noticed you used an incorrect format ({format_error_msg}). "
                    else:
                        error_prefix = ""

                    if stage == 1:
                        guidance = f"""{error_prefix}To proceed, we first need background information about this IP. Please call `text_search` to gather relevant knowledge.

Use this format:
<tool_call>
{{"name": "text_search", "arguments": {example_params}}}
</tool_call>"""
                    else:
                        guidance = f"""{error_prefix}Now that we have background information, let's find reference images. Please call `search_image` to get visual references.

Use this format:
<tool_call>
{{"name": "search_image", "arguments": {example_params}}}
</tool_call>"""

                    chat_messages.append({
                        "role": "user",
                        "content": guidance
                    })

        except Exception as e:
            print(f"  Error in turn {turn_num + 1}: {e}")
            traceback.print_exc()
            current_turn_data['error'] = str(e)
            break

    # Build full_response summary
    full_response = []
    for turn in trajectory.get('turns', []):
        turn_data = {
            'turn': turn.get('turn', 0),
            'input': turn.get('input', ''),
            'response_text': turn.get('response_text', ''),
            'tool_output': turn.get('tool_output', None)
        }
        full_response.append(turn_data)
    trajectory['full_response'] = full_response

    if 'recaption' in trajectory and trajectory['recaption']:
        trajectory['recaption'] = extract_recaption_content(trajectory['recaption'])

    output_file = os.path.join(output_dir, f"{ip_index}_trajectory.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(trajectory, f, ensure_ascii=False, indent=2)
    print(f"\nTrajectory saved: {output_file}")

    return trajectory


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(name)s] %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Stage 2: Generate multi-turn agent trajectories with search tools and recaption."
    )
    parser.add_argument(
        "--ip_data", type=str, required=True,
        help="Path to the IP prompts JSON file."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for trajectory files."
    )
    parser.add_argument(
        "--ip_index", type=str, default=None,
        help="Specific IP index to process. If not specified, processes all IPs."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Reading IP prompts: {args.ip_data}")
    with open(args.ip_data, 'r', encoding='utf-8') as f:
        ip_prompts = json.load(f)

    if args.ip_index is not None:
        if args.ip_index in ip_prompts:
            ip_list = [(args.ip_index, ip_prompts[args.ip_index])]
        else:
            print(f"Error: IP index '{args.ip_index}' not found in data.")
            return
    else:
        ip_list = sorted(ip_prompts.items(), key=lambda x: x[0])

    print(f"Total IPs to process: {len(ip_list)}")

    for ip_index, ip_data in ip_list:
        try:
            process_ip(ip_data, ip_index, args.output_dir)
        except Exception as e:
            print(f"Error processing IP {ip_index}: {e}")
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
