#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
Score FactIP benchmark images using an OpenAI-compatible multimodal API.

For each item under {base_dir}/{category}/{ip_index}/:
  - GT1: intermediate/{ip_index}/image_1.{jpg,png}
  - GT2: intermediate/{ip_index}/image_2.{jpg,png}
  - AS:  {ip_index}_generated.png
  - Prompt: image_prompt from {ip_index}_trajectory.json

Scores are written to {ip_index}_score.json in the same directory.

Usage:
  # Dry run (list items only)
  python score_factip.py --base-dir ./results/MyModel --dry-run

  # Score a specific category
  python score_factip.py --base-dir ./results/MyModel --category Animation --limit 10

  # Full scoring with parallel workers
  python score_factip.py --base-dir ./results/MyModel --workers 8

Environment variables:
  OPENAI_API_KEY      - API key (required)
  OPENAI_BASE_URL     - Custom base URL for OpenAI-compatible endpoints (optional)
  FACTIP_EVAL_MODEL   - Model name to use for evaluation (default: gpt-4o)
  FACTIP_GT_DIR       - Fallback directory for ground-truth images (optional)
"""

import argparse
import base64
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai

logging.basicConfig(
    format='[%(asctime)s][%(levelname)5s] %(message)s',
    level=logging.INFO)
logger = logging.getLogger(__name__)

EVAL_MODEL = os.environ.get("FACTIP_EVAL_MODEL", "gpt-4o")

SYSTEM_PROMPT_EVAL = """You are an image evaluation assistant. Compare AS (assistant-generated image) against GT1 and GT2 (ground-truth images) given the Prompt.

Your task is to return exactly 6 fields: 5 integer scores (0-10) and 1 rationale string.

Evaluate these 5 dimensions independently:

1. "clarity": image sharpness, absence of blur/artifacts/noise, and richness of visible details.
2. "content_quality": faithfulness to the Prompt, subject completeness, and semantic coherence.
3. "aesthetics": visual appeal, composition, lighting/color harmony, and style consistency.
4. "text_relevance_ip": IP identity consistency. This means whether AS preserves the same character/object/IP identity as GT1/GT2 based on distinctive traits (e.g. face, hairstyle, costume, colors, species/object-defining features). Do NOT require exact matching of pose, background, camera angle, or composition.
5. "overall_score": holistic judgment of the final image quality. This should not be a simple average of the above scores.

Important instructions:
- Use GT1 and GT2 jointly to infer the stable identity and attributes of the IP.
- Do not penalize AS for differences that also vary between GT1 and GT2.
- Score each dimension independently before assigning the overall score.
- All five score fields must be integers from 0 to 10.
- "rationale" must be a single short string with at least two concrete evidence points, separated by semicolons.

Overall score rubric:
- 0: total failure
- 1-3: severe issues
- 4-6: usable but clearly worse than GT
- 7-9: good and comparable to GT
- 10: nearly perfect

You MUST respond with ONLY this exact JSON structure, with all 6 keys present and no extra keys:

{"clarity": 7, "content_quality": 8, "aesthetics": 7, "text_relevance_ip": 8, "overall_score": 7, "rationale": "Evidence 1; Evidence 2"}

Do not use markdown fences.
Do not output any text before or after the JSON.
If uncertain, still output the best-effort JSON with all required keys.
"""


def encode_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_mime(image_path):
    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    return {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "gif": "image/gif", "webp": "image/webp"}.get(ext, "image/png")


def image_to_data_uri(image_path):
    mime = get_image_mime(image_path)
    b64 = encode_image_base64(image_path)
    return f"data:{mime};base64,{b64}"


GT_FALLBACK_DIR = os.environ.get("FACTIP_GT_DIR", "")


def find_gt_images(item_dir, ip_index, category=None):
    gt1, gt2 = None, None

    intermediate_dir = os.path.join(item_dir, "intermediate", ip_index)
    for ext in ("jpg", "jpeg", "png", "webp"):
        p = os.path.join(intermediate_dir, f"image_1.{ext}")
        if os.path.exists(p):
            gt1 = p
            break
    for ext in ("jpg", "jpeg", "png", "webp"):
        p = os.path.join(intermediate_dir, f"image_2.{ext}")
        if os.path.exists(p):
            gt2 = p
            break

    if gt1 and gt2:
        return gt1, gt2

    if category and GT_FALLBACK_DIR:
        fallback_dir = os.path.join(GT_FALLBACK_DIR, category, ip_index)
        if os.path.isdir(fallback_dir):
            for ext in ("png", "jpg", "jpeg", "webp"):
                p = os.path.join(fallback_dir, f"seed1.{ext}")
                if os.path.exists(p):
                    gt1 = gt1 or p
                    break
            for ext in ("png", "jpg", "jpeg", "webp"):
                p = os.path.join(fallback_dir, f"seed2.{ext}")
                if os.path.exists(p):
                    gt2 = gt2 or p
                    break

    return gt1, gt2


def _detect_layout(base_dir):
    """Auto-detect whether base_dir uses subdirectory or flat layout.

    Subdirectory layout: {cat}/{ip_index}/{ip_index}_generated.png
    Flat layout:         {cat}/{ip_index}.png + {ip_index}_traj.json
    """
    for cat in os.listdir(base_dir):
        cat_dir = os.path.join(base_dir, cat)
        if not os.path.isdir(cat_dir):
            continue
        entries = os.listdir(cat_dir)
        if any(e.endswith(".png") and not os.path.isdir(os.path.join(cat_dir, e)) for e in entries):
            return "flat"
        if any(os.path.isdir(os.path.join(cat_dir, e)) and not e.startswith(".") for e in entries):
            return "subdir"
    return "subdir"


def _collect_subdir(base_dir, categories):
    """Collect items from subdirectory layout: {cat}/{id}/{id}_generated.png"""
    items = []
    for cat in sorted(categories):
        cat_dir = os.path.join(base_dir, cat)
        if not os.path.isdir(cat_dir):
            continue

        for ip_index in sorted(os.listdir(cat_dir)):
            item_dir = os.path.join(cat_dir, ip_index)
            if not os.path.isdir(item_dir):
                continue

            traj_path = os.path.join(item_dir, ip_index + "_trajectory.json")
            gen_path = os.path.join(item_dir, ip_index + "_generated.png")
            score_path = os.path.join(item_dir, ip_index + "_score.json")

            if not os.path.exists(traj_path) or not os.path.exists(gen_path):
                continue

            gt1, gt2 = find_gt_images(item_dir, ip_index, category=cat)
            if not gt1 or not gt2:
                logger.warning("GT images missing for %s/%s", cat, ip_index)
                continue

            items.append({
                "category": cat,
                "ip_index": ip_index,
                "item_dir": item_dir,
                "traj_path": traj_path,
                "gen_path": gen_path,
                "gt1_path": gt1,
                "gt2_path": gt2,
                "score_path": score_path,
            })
    return items


def _collect_flat(base_dir, categories):
    """Collect items from flat layout: {cat}/{id}.png + {id}_traj.json"""
    items = []
    for cat in sorted(categories):
        cat_dir = os.path.join(base_dir, cat)
        if not os.path.isdir(cat_dir):
            continue

        seen = set()
        for fname in sorted(os.listdir(cat_dir)):
            if not fname.endswith(".png"):
                continue
            ip_index = fname[:-4]
            if ip_index in seen:
                continue
            seen.add(ip_index)

            gen_path = os.path.join(cat_dir, fname)
            traj_path = os.path.join(cat_dir, ip_index + "_traj.json")
            score_path = os.path.join(cat_dir, ip_index + "_score.json")

            if not os.path.exists(traj_path):
                continue

            gt1, gt2 = find_gt_images(cat_dir, ip_index, category=cat)
            if not gt1 or not gt2:
                logger.warning("GT images missing for %s/%s", cat, ip_index)
                continue

            items.append({
                "category": cat,
                "ip_index": ip_index,
                "item_dir": cat_dir,
                "traj_path": traj_path,
                "gen_path": gen_path,
                "gt1_path": gt1,
                "gt2_path": gt2,
                "score_path": score_path,
            })
    return items


def collect_items(base_dir, category=None):
    if category:
        categories = [category]
    else:
        categories = [
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
            and d not in ("intermediate", "_task_queue")
            and not d.startswith(".")
            and not d.endswith(".json")
        ]

    layout = _detect_layout(base_dir)
    logger.info("Detected layout: %s", layout)

    if layout == "flat":
        return _collect_flat(base_dir, categories)
    else:
        return _collect_subdir(base_dir, categories)


def call_eval_api(prompt, gt1_path, gt2_path, as_path, max_retries=3):
    """Call an OpenAI-compatible multimodal API to evaluate the generated image."""
    client = openai.OpenAI()

    gt1_uri = image_to_data_uri(gt1_path)
    gt2_uri = image_to_data_uri(gt2_path)
    as_uri = image_to_data_uri(as_path)

    user_content = [
        {"type": "text", "text": f"Prompt: {prompt}"},
        {"type": "text", "text": "GT1 (Ground Truth Image #1):"},
        {"type": "image_url", "image_url": {"url": gt1_uri}},
        {"type": "text", "text": "GT2 (Ground Truth Image #2):"},
        {"type": "image_url", "image_url": {"url": gt2_uri}},
        {"type": "text", "text": "AS (Assistant-generated Image):"},
        {"type": "image_url", "image_url": {"url": as_uri}},
        {"type": "text", "text": 'Evaluate AS vs GT1/GT2. Reply with ONLY a JSON containing ALL 6 keys: '
                                  '"clarity", "content_quality", "aesthetics", "text_relevance_ip", '
                                  '"overall_score", "rationale". All scores are integers 0-10.'},
    ]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_EVAL},
        {"role": "user", "content": user_content},
    ]

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=EVAL_MODEL,
                messages=messages,
                max_tokens=1024,
                temperature=0.0,
            )
            content = response.choices[0].message.content
            return {"content": content}
        except Exception as e:
            logger.warning("Attempt %d/%d failed: %s", attempt, max_retries, e)
            if attempt < max_retries:
                sleep_time = 2 ** attempt
                logger.info("Retrying in %ds...", sleep_time)
                time.sleep(sleep_time)

    return {"error": "max retries exceeded"}


DIMENSION_ALIASES = {
    "clarity": ["clarity", "clarity_score", "sharpness", "image_clarity"],
    "content_quality": ["content_quality", "content_quality_score", "content", "quality",
                        "prompt_adherence", "prompt_adherence_score"],
    "aesthetics": ["aesthetics", "aesthetics_score", "aesthetic", "aesthetic_score",
                   "beauty", "artistic_quality"],
    "text_relevance_ip": ["text_relevance_ip", "text_relevance_ip_score", "ip_consistency",
                          "ip_consistency_score", "character_consistency",
                          "character_consistency_score", "ip_score", "ip_similarity"],
    "overall_score": ["overall_score", "score", "overall", "total_score", "final_score",
                      "overall_quality_score"],
}


def _normalize_keys(parsed):
    """Map model output keys to canonical dimension names."""
    result = {}
    used_keys = set()
    for canonical, aliases in DIMENSION_ALIASES.items():
        for alias in aliases:
            if alias in parsed and alias not in used_keys:
                val = parsed[alias]
                if isinstance(val, (int, float)):
                    result[canonical] = int(round(val)) if isinstance(val, float) else val
                else:
                    result[canonical] = val
                used_keys.add(alias)
                break

    rationale = ""
    for key in ("rationale", "reason", "explanation", "summary"):
        if key in parsed and isinstance(parsed[key], str):
            rationale = parsed[key]
            used_keys.add(key)
            break

    if not rationale:
        parts = []
        for key in ("key_strengths", "strengths", "pros"):
            if key in parsed and isinstance(parsed[key], list):
                parts.extend(parsed[key])
                used_keys.add(key)
        for key in ("minor_weaknesses", "weaknesses", "cons", "issues"):
            if key in parsed and isinstance(parsed[key], list):
                parts.extend(["[Weakness] " + w for w in parsed[key]])
                used_keys.add(key)
        if parts:
            rationale = " | ".join(parts)

    result["rationale"] = rationale

    for k, v in parsed.items():
        if k not in used_keys and k not in result:
            result[k] = v
    return result


def extract_score_json(api_response):
    """Extract and parse score JSON from the API response."""
    try:
        if "error" in api_response and isinstance(api_response["error"], str):
            return {"error": api_response["error"]}

        content = api_response.get("content", "")
        if not content:
            return {"error": "could not extract content from API response"}

        if isinstance(content, str):
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            parsed = json.loads(content)
            parsed = _normalize_keys(parsed)
            return parsed

        return {"error": "unexpected content type"}

    except json.JSONDecodeError as e:
        return {"raw_content": str(content)[:1000], "error": "JSON parse failed: " + str(e)}
    except Exception as e:
        return {"error": str(e)}


def score_one_item(item, dry_run=False):
    cat = item["category"]
    ip_index = item["ip_index"]
    score_path = item["score_path"]

    if os.path.exists(score_path):
        logger.info("[SKIP] Already scored: %s/%s", cat, ip_index)
        return {"status": "skipped", "ip_index": ip_index}

    with open(item["traj_path"], "r") as f:
        traj = json.load(f)
    prompt = traj.get("image_prompt", traj.get("recaption", ""))

    if dry_run:
        logger.info("[DRY] %s/%s | prompt=%s...", cat, ip_index, prompt[:80])
        return {"status": "dry_run", "ip_index": ip_index}

    logger.info("[SCORING] %s/%s", cat, ip_index)

    api_response = call_eval_api(
        prompt, item["gt1_path"], item["gt2_path"], item["gen_path"]
    )
    score_result = extract_score_json(api_response)

    output = {
        "ip_index": ip_index,
        "category": cat,
        "prompt": prompt,
        "gt1": item["gt1_path"],
        "gt2": item["gt2_path"],
        "generated": item["gen_path"],
        "clarity": score_result.get("clarity"),
        "content_quality": score_result.get("content_quality"),
        "aesthetics": score_result.get("aesthetics"),
        "text_relevance_ip": score_result.get("text_relevance_ip"),
        "overall_score": score_result.get("overall_score"),
        "rationale": score_result.get("rationale", ""),
        "eval_result_raw": score_result,
    }

    if "error" in score_result:
        output["error"] = score_result["error"]

    with open(score_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    score_val = score_result.get("overall_score", "N/A")
    dims = "/".join(str(score_result.get(d, "?")) for d in
                    ["clarity", "content_quality", "aesthetics", "text_relevance_ip"])
    logger.info("[DONE] %s/%s -> overall=%s dims=[%s]", cat, ip_index, score_val, dims)

    return {"status": "scored", "ip_index": ip_index, "score": score_val}


def main():
    parser = argparse.ArgumentParser(description="Score FactIP images via OpenAI-compatible API")
    parser.add_argument("--base-dir", type=str, required=True,
                        help="Root directory of the benchmark results")
    parser.add_argument("--dry-run", action="store_true", help="Only list items, don't call API")
    parser.add_argument("--limit", type=int, default=None, help="Max items to process")
    parser.add_argument("--category", type=str, default=None, help="Only score this category")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent API workers")
    parser.add_argument("--model", type=str, default=None,
                        help="Override the evaluation model name (default: env FACTIP_EVAL_MODEL or gpt-4o)")
    args = parser.parse_args()

    if args.model:
        global EVAL_MODEL
        EVAL_MODEL = args.model

    base_dir = args.base_dir
    logger.info("Collecting items from %s ...", base_dir)
    items = collect_items(base_dir, category=args.category)
    logger.info("Found %d scorable items", len(items))

    if args.limit:
        items = items[:args.limit]
        logger.info("Limited to %d items", len(items))

    if args.dry_run:
        for item in items:
            score_one_item(item, dry_run=True)
        logger.info("Dry run complete.")
        return

    results = {"scored": 0, "skipped": 0, "error": 0}

    if args.workers <= 1:
        for item in items:
            r = score_one_item(item)
            results[r["status"]] = results.get(r["status"], 0) + 1
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(score_one_item, item): item for item in items}
            for future in as_completed(futures):
                try:
                    r = future.result()
                    results[r["status"]] = results.get(r["status"], 0) + 1
                except Exception as e:
                    logger.error("Worker error: %s", e)
                    results["error"] += 1

    logger.info("Done! Results: %s", results)


if __name__ == "__main__":
    main()
