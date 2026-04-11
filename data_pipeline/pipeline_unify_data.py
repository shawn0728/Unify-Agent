# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""
Unified three-stage pipeline:
1. Stage 1: Load IP prompt data
2. Stage 2: Generate recaption and trajectory from prompts
3. Stage 3: Generate images from trajectory

Output structure:
{output_dir}/{source_IP}/
  - images/        (Stage 3 generated images)
  - intermediate/  (Stage 2 intermediate images)
  - traj/          (Stage 2 trajectory JSON files)
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import traceback
from pathlib import Path

from tqdm import tqdm

log = logging.getLogger("pipeline")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def ensure_dir(path):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)
    return path


def run_stage1(ip_data_path, source_ip=None):
    """
    Stage 1: Load IP prompt data from a JSON file.

    Args:
        ip_data_path: Path to the IP data JSON file.
        source_ip: Source IP name for filtering (optional).

    Returns:
        dict: IP prompts dict keyed by ip_index.
    """
    print("=" * 60)
    print("Stage 1: Load IP Prompts")
    print("=" * 60)

    if not os.path.exists(ip_data_path):
        print(f"[ERROR] IP data file not found: {ip_data_path}")
        return {}

    try:
        with open(ip_data_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)
        print(f"[OK] Loaded prompts for {len(prompts)} IPs from {ip_data_path}")

        if source_ip:
            filtered = {}
            for ip_index, ip_data in prompts.items():
                if ip_data.get("source", "") == source_ip:
                    filtered[ip_index] = ip_data
            if filtered:
                print(f"[OK] Filtered to {len(filtered)} IPs matching source '{source_ip}'")
                return filtered
            print(f"[WARN] No IPs matched source '{source_ip}', returning all {len(prompts)} IPs")

        return prompts
    except Exception as e:
        print(f"[ERROR] Failed to read IP data file: {e}")
        return {}


def run_stage2_for_ip(ip_data, ip_index, output_base_dir, ip_data_path):
    """
    Stage 2: Generate recaption and trajectory for a single IP via subprocess.

    Args:
        ip_data: IP data dict.
        ip_index: IP index string.
        output_base_dir: Base output directory for this source IP.
        ip_data_path: Path to the IP data JSON file.

    Returns:
        tuple: (trajectory_file_path, intermediate_dir) or (None, None) on failure.
    """
    print(f"\n{'=' * 60}")
    print(f"Stage 2: Generate Trajectory (IP: {ip_data.get('ip_name', 'Unknown')}, Index: {ip_index})")
    print(f"{'=' * 60}")

    traj_dir = ensure_dir(os.path.join(output_base_dir, "traj"))
    intermediate_dir = ensure_dir(os.path.join(output_base_dir, "intermediate"))

    stage2_script = os.path.join(SCRIPT_DIR, "stage2_generate_trajectory.py")
    if not os.path.exists(stage2_script):
        print(f"[ERROR] Stage 2 script not found: {stage2_script}")
        return None, None

    try:
        cmd = [
            sys.executable, stage2_script,
            "--ip_data", ip_data_path,
            "--output_dir", output_base_dir,
            "--ip_index", str(ip_index),
        ]
        print(f"[RUN] {' '.join(cmd)}")
        result = subprocess.run(cmd, timeout=3600)

        if result.returncode != 0:
            print(f"[ERROR] Stage 2 exited with code {result.returncode}")
            return None, None

        trajectory_files = list(Path(traj_dir).glob(f"{ip_index}_trajectory.json"))
        if not trajectory_files:
            trajectory_files = list(Path(traj_dir).glob("*_trajectory.json"))

        if trajectory_files:
            trajectory_file = str(trajectory_files[0])
            print(f"[OK] Trajectory file: {trajectory_file}")
            return trajectory_file, intermediate_dir

        print("[ERROR] No trajectory file found after Stage 2")
        return None, None

    except subprocess.TimeoutExpired:
        print("[ERROR] Stage 2 timed out")
        return None, None
    except Exception as e:
        print(f"[ERROR] Stage 2 failed: {e}")
        traceback.print_exc()
        return None, None


def run_stage3_for_ip(trajectory_file, intermediate_dir, output_base_dir, ip_index):
    """
    Stage 3: Generate images for a single IP via subprocess.

    Args:
        trajectory_file: Path to the trajectory JSON file.
        intermediate_dir: Path to the intermediate images directory.
        output_base_dir: Base output directory for this source IP.
        ip_index: IP index string.

    Returns:
        list: Paths of generated images, or empty list on failure.
    """
    print(f"\n{'=' * 60}")
    print(f"Stage 3: Generate Images (Index: {ip_index})")
    print(f"{'=' * 60}")

    images_dir = ensure_dir(os.path.join(output_base_dir, "images"))

    stage3_script = os.path.join(SCRIPT_DIR, "stage3_generate_image.py")
    if not os.path.exists(stage3_script):
        print(f"[ERROR] Stage 3 script not found: {stage3_script}")
        return []

    try:
        traj_dir = os.path.dirname(str(trajectory_file))
        cmd = [
            sys.executable, stage3_script,
            "--trajectory_file", str(trajectory_file),
            "--input_dir", traj_dir,
            "--intermediate_dir", str(intermediate_dir),
            "--output_dir", str(images_dir),
        ]
        print(f"[RUN] {' '.join(cmd)}")
        result = subprocess.run(cmd, timeout=7200)

        if result.returncode != 0:
            print(f"[ERROR] Stage 3 exited with code {result.returncode}")
            return []

        generated = sorted(Path(images_dir).glob(f"{ip_index}_*.png"))
        if not generated:
            generated = sorted(Path(images_dir).glob(f"{ip_index}_*.jpg"))
        if not generated:
            generated = sorted(
                p for p in Path(images_dir).iterdir()
                if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
            )

        image_paths = [str(p) for p in generated]
        print(f"[OK] Generated {len(image_paths)} image(s)")
        return image_paths

    except subprocess.TimeoutExpired:
        print("[ERROR] Stage 3 timed out")
        return []
    except Exception as e:
        print(f"[ERROR] Stage 3 failed: {e}")
        traceback.print_exc()
        return []


def process_single_ip(
    ip_data,
    ip_index,
    source_ip_name,
    output_base_dir,
    ip_data_path,
    skip_stage2=False,
    skip_stage3=False,
    existing_result=None,
):
    """
    Run the full pipeline for a single IP.

    Output directory layout:
        {output_base_dir}/
          images/                        (Stage 3 output)
          intermediate/{ip_index}/       (Stage 2 intermediate images)
          traj/{ip_index}_trajectory.json (Stage 2 trajectory)

    Args:
        ip_data: IP data dict.
        ip_index: IP index string.
        source_ip_name: Source IP name.
        output_base_dir: Root output directory for this source IP.
        ip_data_path: Path to IP data JSON file.
        skip_stage2: Skip Stage 2 (reuse existing trajectory).
        skip_stage3: Skip Stage 3.
        existing_result: Previous processing result (used when skipping stages).

    Returns:
        dict: Processing result.
    """
    print(f"\n{'=' * 80}")
    print(f"Processing IP: {ip_data.get('ip_name', 'Unknown')} (Index: {ip_index})")
    print(f"{'=' * 80}")

    ensure_dir(output_base_dir)

    result = {
        "ip_index": ip_index,
        "ip_name": ip_data.get("ip_name", ""),
        "status": "pending",
        "stage1": {"status": "skipped"},
        "stage2": {"status": "pending"},
        "stage3": {"status": "pending"},
        "output_dir": output_base_dir,
    }

    # --- Stage 2 --------------------------------------------------------
    if skip_stage2 and existing_result:
        print("[SKIP] Stage 2 — reusing existing result")
        result["stage1"] = existing_result.get("stage1", {"status": "success"})
        result["stage2"] = existing_result.get("stage2", {})

        trajectory_file = result["stage2"].get("trajectory_file")
        intermediate_dir = result["stage2"].get("intermediate_dir")

        if not trajectory_file or not os.path.exists(trajectory_file):
            print(f"[WARN] Trajectory file missing: {trajectory_file}")
            result["stage2"]["status"] = "failed"
            result["status"] = "failed"
            result["error"] = "Trajectory file not found"
            return result

        if not intermediate_dir or not os.path.exists(intermediate_dir):
            print(f"[WARN] Intermediate directory missing: {intermediate_dir}")
            result["stage2"]["status"] = "failed"
            result["status"] = "failed"
            result["error"] = "Intermediate directory not found"
            return result
    else:
        if not ip_data.get("image_prompt"):
            result["stage1"]["status"] = "failed"
            result["status"] = "failed"
            result["error"] = "Missing image_prompt in IP data"
            return result

        result["stage1"]["status"] = "success"

        trajectory_file, intermediate_dir = run_stage2_for_ip(
            ip_data, ip_index, output_base_dir, ip_data_path
        )

        if trajectory_file and os.path.exists(trajectory_file):
            result["stage2"]["status"] = "success"
            result["stage2"]["trajectory_file"] = trajectory_file
            result["stage2"]["intermediate_dir"] = intermediate_dir
        else:
            result["stage2"]["status"] = "failed"
            result["status"] = "failed"
            result["error"] = "Stage 2 failed to produce trajectory"
            return result

    # --- Stage 3 --------------------------------------------------------
    if skip_stage3:
        print("[SKIP] Stage 3")
        result["stage3"]["status"] = "skipped"
        result["status"] = "success"
        return result

    generated_images = run_stage3_for_ip(
        trajectory_file, intermediate_dir, output_base_dir, ip_index
    )

    if generated_images:
        result["stage3"]["status"] = "success"
        result["stage3"]["generated_images"] = generated_images
        result["status"] = "success"
    else:
        result["stage3"]["status"] = "failed"
        result["status"] = "partial"

    return result


def save_summary(summary_file, results, existing_results):
    """Merge current results with existing ones and write to summary file."""
    processed_indices = {r.get("ip_index") for r in results}
    all_results = [
        er for idx, er in existing_results.items() if idx not in processed_indices
    ]
    all_results.extend(results)
    all_results.sort(key=lambda x: x.get("ip_index", ""))

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)


def is_fully_successful(result):
    """Check whether all three stages succeeded."""
    if result.get("status") != "success":
        return False
    return all(
        result.get(stage, {}).get("status") == "success"
        for stage in ("stage1", "stage2", "stage3")
    )


def can_skip_stage2(result):
    """Check whether Stage 1 and Stage 2 both succeeded (so only Stage 3 needs rerun)."""
    return (
        result.get("stage1", {}).get("status") == "success"
        and result.get("stage2", {}).get("status") == "success"
    )


def _setup_logging():
    """Configure pipeline-wide logging format."""
    fmt = "[%(asctime)s] [%(name)s] %(levelname)s  %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S")


def main():
    _setup_logging()
    parser = argparse.ArgumentParser(
        description="Unified three-stage data pipeline (prompt -> trajectory -> image)"
    )
    parser.add_argument(
        "--source_ip", type=str, required=True,
        help="Source IP name (used as output subdirectory, e.g. 'douban_celebrity')",
    )
    parser.add_argument(
        "--ip_data", type=str, required=True,
        help="Path to IP data JSON file (prompt definitions)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Base output directory",
    )
    parser.add_argument(
        "--ip_index", type=str, default=None,
        help="Process only the specified IP index (optional, processes all if omitted)",
    )
    parser.add_argument("--skip_stage1", action="store_true", help="Skip Stage 1 (prompt loading)")
    parser.add_argument("--skip_stage2", action="store_true", help="Skip Stage 2 (trajectory generation)")
    parser.add_argument("--skip_stage3", action="store_true", help="Skip Stage 3 (image generation)")

    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, args.source_ip)
    ensure_dir(output_dir)

    print("=" * 80)
    print("Unified Three-Stage Pipeline")
    print("=" * 80)
    print(f"Source IP   : {args.source_ip}")
    print(f"IP data file: {args.ip_data}")
    print(f"Output dir  : {output_dir}")
    print("=" * 80)

    # Stage 1: load prompts
    ip_prompts = run_stage1(args.ip_data, args.source_ip)

    if not ip_prompts:
        print("[ERROR] No IP prompts available — exiting")
        return

    print(f"\n[OK] {len(ip_prompts)} IP(s) ready for processing")

    # Load existing results for incremental processing
    summary_file = os.path.join(output_dir, "processing_summary.json")
    existing_results = {}
    if os.path.exists(summary_file):
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                for item in json.load(f):
                    ip_idx = item.get("ip_index")
                    if ip_idx:
                        existing_results[ip_idx] = item
            print(f"[OK] Loaded {len(existing_results)} existing result(s)")
        except Exception as e:
            print(f"[WARN] Failed to read existing results: {e}")

    # Filter IPs if --ip_index is specified
    ip_list = list(ip_prompts.items())
    if args.ip_index:
        ip_list = [(idx, data) for idx, data in ip_list if idx == args.ip_index]
        if not ip_list:
            print(f"[ERROR] No IP found with index '{args.ip_index}'")
            return

    skipped_count = 0
    retry_count = 0
    new_count = 0
    results = []

    for ip_index, ip_data in tqdm(ip_list, desc="Processing IPs"):
        existing = existing_results.get(ip_index)
        skip_s2 = False

        if existing:
            if is_fully_successful(existing):
                print(f"\n[SKIP] {ip_data.get('ip_name', 'Unknown')} (index: {ip_index}) — already fully successful")
                results.append(existing)
                skipped_count += 1
                save_summary(summary_file, results, existing_results)
                continue

            if can_skip_stage2(existing):
                print(f"\n[RETRY] Stage 3 only for {ip_data.get('ip_name', 'Unknown')} (index: {ip_index})")
                retry_count += 1
                skip_s2 = True
            else:
                print(f"\n[RETRY] Full reprocess for {ip_data.get('ip_name', 'Unknown')} (index: {ip_index})")
                retry_count += 1
        else:
            new_count += 1

        try:
            result = process_single_ip(
                ip_data,
                ip_index,
                args.source_ip,
                output_dir,
                ip_data_path=args.ip_data,
                skip_stage2=skip_s2 or args.skip_stage2,
                skip_stage3=args.skip_stage3,
                existing_result=existing if skip_s2 else None,
            )
            results.append(result)
        except Exception as e:
            print(f"[ERROR] Failed to process IP {ip_index}: {e}")
            traceback.print_exc()
            results.append({
                "ip_index": ip_index,
                "ip_name": ip_data.get("ip_name", ""),
                "status": "error",
                "error": str(e),
            })

        save_summary(summary_file, results, existing_results)

    # Final statistics
    final_results = []
    if os.path.exists(summary_file):
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                final_results = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to read final results: {e}")
            final_results = results

    success_count = sum(1 for r in final_results if is_fully_successful(r))
    partial_count = sum(1 for r in final_results if r.get("status") == "partial")
    failed_count = sum(1 for r in final_results if r.get("status") == "failed")

    print("\n" + "=" * 80)
    print("Pipeline Complete — Summary")
    print("=" * 80)
    print(f"Total IPs        : {len(final_results)}")
    print(f"Fully successful : {success_count}")
    print(f"Partial success  : {partial_count}")
    print(f"Failed           : {failed_count}")
    print(f"\nThis run:")
    print(f"  Skipped (already done) : {skipped_count}")
    print(f"  Retried                : {retry_count}")
    print(f"  New                    : {new_count}")
    print(f"Results saved to: {summary_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
