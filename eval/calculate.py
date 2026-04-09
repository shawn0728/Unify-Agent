# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
FactIP Benchmark score calculator.

Reads per-item score JSONs produced by score_factip.py,
aggregates per-subtask and overall averages, and writes a report.

All scores are reported on a 0-100 scale.
"""

import json
import os
import argparse
import unicodedata
from collections import defaultdict

SKIP_DIRS = {"intermediate", "_task_queue", "traj", "eval"}

METRICS = ["clarity", "content_quality", "aesthetics", "text_relevance_ip"]

SCORE_SCALE = 10


def compute_overall_score(clarity, content_quality, aesthetics, text_relevance_ip):
    return (
        0.05 * clarity +
        0.10 * content_quality +
        0.10 * aesthetics +
        0.75 * text_relevance_ip
    )


def display_width(s: str) -> int:
    """Compute the terminal display width of a string (CJK chars count as 2)."""
    w = 0
    for ch in s:
        if unicodedata.east_asian_width(ch) in ('F', 'W'):
            w += 2
        else:
            w += 1
    return w


def pad_to_width(s: str, width: int) -> str:
    """Pad a string to a given display width with trailing spaces."""
    dw = display_width(s)
    return s + ' ' * max(0, width - dw)


def extract_scores_from_data(data: dict):
    """
    Extract the four metric scores from a score JSON.

    Checks top-level fields first; falls back to nested eval_result_raw.
    Returns a dict with METRICS keys if valid, else None.
    """
    scores = {}
    for m in METRICS:
        val = data.get(m)
        if isinstance(val, (int, float)):
            scores[m] = float(val)

    if len(scores) == len(METRICS):
        return scores

    raw = data.get("eval_result_raw")
    if isinstance(raw, dict):
        fallback = raw
        raw_content = raw.get("raw_content")
        if isinstance(raw_content, str):
            try:
                fallback = json.loads(raw_content)
            except json.JSONDecodeError:
                pass
        for m in METRICS:
            if m not in scores:
                val = fallback.get(m)
                if isinstance(val, (int, float)):
                    scores[m] = float(val)

    if len(scores) == len(METRICS):
        return scores
    return None


def collect_scores_for_model(model_dir: str):
    """
    Walk model_dir to collect scores across all subtasks.

    Supports two directory layouts:
      - Nested: {subtask}/{item_id}/{item_id}_score.json
      - Flat:   {subtask}/{item_id}_score.json
    """
    subtask_scores = defaultdict(list)

    for subtask_name in sorted(os.listdir(model_dir)):
        subtask_path = os.path.join(model_dir, subtask_name)
        if not os.path.isdir(subtask_path) or subtask_name in SKIP_DIRS:
            continue

        found_nested = False
        found_flat = False

        for item_name in sorted(os.listdir(subtask_path)):
            item_full = os.path.join(subtask_path, item_name)

            if os.path.isdir(item_full) and item_name not in SKIP_DIRS:
                score_file = os.path.join(item_full, f"{item_name}_score.json")
                if os.path.exists(score_file):
                    found_nested = True
                    _load_score_file(score_file, item_name, subtask_name, subtask_scores)

            elif item_name.endswith("_score.json") and os.path.isfile(item_full):
                found_flat = True
                item_id = item_name[: -len("_score.json")]
                _load_score_file(item_full, item_id, subtask_name, subtask_scores)

        if not found_nested and not found_flat:
            pass

    return subtask_scores


def _load_score_file(score_file: str, item_id: str, subtask_name: str, subtask_scores: dict):
    """Read a single score JSON and append to subtask_scores."""
    try:
        with open(score_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, Exception) as e:
        print(f"  Warning: Failed to parse {score_file}: {e}")
        return

    scores = extract_scores_from_data(data)
    if scores is None:
        print(f"  Warning: {score_file} incomplete scores, skipping.")
        return

    entry = {"item_id": item_id}
    entry.update(scores)
    entry["overall"] = compute_overall_score(
        entry["clarity"], entry["content_quality"],
        entry["aesthetics"], entry["text_relevance_ip"]
    )
    subtask_scores[subtask_name].append(entry)


def compute_and_write(model_name: str, model_dir: str):
    """Compute and display scores for a single model, then write to file."""
    subtask_scores = collect_scores_for_model(model_dir)

    if not subtask_scores:
        print(f"  No valid scores found for model '{model_name}'. Skipping.")
        return

    output_path = os.path.join(model_dir, "overall_score.txt")
    lines = []

    NAME_COL_W = 20

    def add(text=""):
        lines.append(text)
        print(text)

    def fmt_row(name, c, cq, ae, tr, ov, cnt):
        padded = pad_to_width(name, NAME_COL_W)
        return f"{padded}  {c:>8s}  {cq:>9s}  {ae:>10s}  {tr:>8s}  {ov:>8s}  {cnt:>6s}"

    add(f"{'=' * 78}")
    add(f"  FactIP Evaluation Report: {model_name}")
    add(f"  (scores on 0-100 scale)")
    add(f"{'=' * 78}")
    add()

    add(fmt_row("Subtask", "clarity", "content_q", "aesthetics", "text_rel", "overall", "count"))
    add("-" * 78)

    all_entries = []

    for subtask_name in sorted(subtask_scores.keys()):
        entries = subtask_scores[subtask_name]
        n = len(entries)
        all_entries.extend(entries)

        avg = {}
        for m in METRICS + ["overall"]:
            avg[m] = sum(e[m] for e in entries) / n * SCORE_SCALE

        add(fmt_row(
            subtask_name,
            f"{avg['clarity']:.1f}",
            f"{avg['content_quality']:.1f}",
            f"{avg['aesthetics']:.1f}",
            f"{avg['text_relevance_ip']:.1f}",
            f"{avg['overall']:.1f}",
            f"{n}",
        ))

    add("-" * 78)

    total_n = len(all_entries)
    total_avg = {}
    for m in METRICS + ["overall"]:
        total_avg[m] = sum(e[m] for e in all_entries) / total_n * SCORE_SCALE

    add(fmt_row(
        "TOTAL",
        f"{total_avg['clarity']:.1f}",
        f"{total_avg['content_quality']:.1f}",
        f"{total_avg['aesthetics']:.1f}",
        f"{total_avg['text_relevance_ip']:.1f}",
        f"{total_avg['overall']:.1f}",
        f"{total_n}",
    ))
    add()
    add("Overall = 0.05*clarity + 0.10*content_quality + 0.10*aesthetics + 0.75*text_relevance_ip")
    add()

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n=> Saved to {output_path}\n")


def main():
    parser = argparse.ArgumentParser(description="FactIP Benchmark Score Calculator")
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help="A single model directory name to evaluate. "
             "If omitted, all model directories under --base_dir are processed."
    )
    parser.add_argument(
        '--base_dir',
        type=str,
        required=True,
        help="Base directory containing model result folders."
    )
    args = parser.parse_args()

    base = args.base_dir

    if args.model:
        model_dir = os.path.join(base, args.model)
        if not os.path.isdir(model_dir):
            print(f"Error: Model directory '{model_dir}' not found.")
            return
        compute_and_write(args.model, model_dir)
    else:
        models = sorted([
            d for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d)) and d not in SKIP_DIRS
        ])
        if not models:
            print(f"No model directories found under {base}")
            return
        print(f"Found {len(models)} model(s): {models}\n")
        for model_name in models:
            model_dir = os.path.join(base, model_name)
            compute_and_write(model_name, model_dir)


if __name__ == "__main__":
    main()
