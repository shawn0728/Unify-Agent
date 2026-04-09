import argparse
import csv
import math
import os
import sys
import time
from itertools import islice
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import torch
from torch.utils.data import DataLoader

try:
    from data.umm_sft_dataset import SftAgenticIterableDataset
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from data.umm_sft_dataset import SftAgenticIterableDataset


DEFAULT_OUTPUT_ROOT = os.environ.get("UMM_SFT_OUTPUT_ROOT", "./data/umm_sft_output")


class DummyTokenizer:
    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        return [abs(hash(tok)) % 10007 + 1 for tok in text.split()]


class DummyImageTransform:
    def __init__(self, image_size: int = 256, stride: int = 16):
        self.image_size = image_size
        self.stride = stride

    def __call__(self, _image):
        return torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)


def discover_triplets(output_root: str) -> List[Tuple[str, str, str]]:
    triplets: List[Tuple[str, str, str]] = []
    root = Path(output_root)
    if not root.exists():
        raise FileNotFoundError(f"output_root 不存在: {output_root}")

    for category_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        traj_dir = category_dir / "traj"
        ref_dir = category_dir / "intermediate"
        gen_dir = category_dir / "images"
        if not (traj_dir.is_dir() and ref_dir.is_dir() and gen_dir.is_dir()):
            continue
        for traj_json in sorted(traj_dir.glob("*_trajectory.json")):
            triplets.append((str(traj_json), str(ref_dir), str(gen_dir)))
    return triplets


def make_dataset(
    triplets: Sequence[Tuple[str, str, str]],
    local_rank: int = 0,
    world_size: int = 1,
    num_workers: int = 1,
) -> SftAgenticIterableDataset:
    json_path_list = [t[0] for t in triplets]
    reference_list = [t[1] for t in triplets]
    generation_list = [t[2] for t in triplets]
    num_used_data = [1 for _ in triplets]

    transform = DummyImageTransform(image_size=256, stride=16)
    vit_transform = DummyImageTransform(image_size=224, stride=14)
    tokenizer = DummyTokenizer()

    return SftAgenticIterableDataset(
        dataset_name="umm_sft_test",
        transform=transform,
        vit_transform=vit_transform,
        tokenizer=tokenizer,
        json_path_list=json_path_list,
        reference_list=reference_list,
        generation_list=generation_list,
        num_used_data=num_used_data,
        local_rank=local_rank,
        world_size=world_size,
        num_workers=num_workers,
        data_status=None,
        shuffle_lines=True,
        shuffle_seed=0,
    )


def sample_row_indices(dataset: SftAgenticIterableDataset, n: int, epoch_seed: int) -> List[int]:
    dataset.set_epoch(seed=epoch_seed)
    iterator = iter(dataset)
    rows: List[int] = []
    for sample in islice(iterator, n):
        rows.append(sample["data_indexes"]["data_indexes"])
    return rows


def test_traversal(dataset: SftAgenticIterableDataset, num_workers: int, max_samples: int) -> Dict[str, float]:
    loader = DataLoader(dataset, batch_size=None, num_workers=num_workers)
    start = time.perf_counter()
    got = 0
    total_tokens = 0
    for sample in islice(loader, max_samples):
        got += 1
        total_tokens += int(sample["num_tokens"])
        assert "image_tensor_list" in sample
        assert "text_ids_list" in sample
        assert "sequence_plan" in sample
        assert "data_indexes" in sample
    elapsed = time.perf_counter() - start
    if got == 0:
        raise RuntimeError("遍历失败：0 个样本。请检查数据目录或过滤条件。")
    return {
        "samples": got,
        "elapsed_sec": elapsed,
        "samples_per_sec": got / elapsed if elapsed > 0 else float("inf"),
        "tokens_per_sec": total_tokens / elapsed if elapsed > 0 else float("inf"),
    }


def test_randomness(dataset: SftAgenticIterableDataset, check_samples: int) -> Dict[str, object]:
    rows_seed_42_a = sample_row_indices(dataset, check_samples, epoch_seed=42)
    rows_seed_42_b = sample_row_indices(dataset, check_samples, epoch_seed=42)
    rows_seed_43 = sample_row_indices(dataset, check_samples, epoch_seed=43)
    return {
        "same_seed_consistent": rows_seed_42_a == rows_seed_42_b,
        "different_seed_changed": rows_seed_42_a != rows_seed_43,
        "seed_42_preview": rows_seed_42_a[:10],
        "seed_43_preview": rows_seed_43[:10],
    }


def test_speed(dataset: SftAgenticIterableDataset, num_workers: int, num_samples: int) -> Dict[str, float]:
    loader = DataLoader(dataset, batch_size=None, num_workers=num_workers)
    start = time.perf_counter()
    got = 0
    total_tokens = 0
    for sample in islice(loader, num_samples):
        got += 1
        total_tokens += int(sample["num_tokens"])
    elapsed = time.perf_counter() - start
    if got == 0:
        raise RuntimeError("速度测试失败：0 个样本。")
    return {
        "samples": got,
        "elapsed_sec": elapsed,
        "samples_per_sec": got / elapsed if elapsed > 0 else float("inf"),
        "ms_per_sample": (elapsed * 1000.0) / got if got > 0 else float("inf"),
        "tokens_per_sec": total_tokens / elapsed if elapsed > 0 else float("inf"),
    }


def parse_worker_counts(worker_counts: str) -> List[int]:
    values: List[int] = []
    for item in worker_counts.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 1:
            raise ValueError(f"worker 数必须 >= 1，但收到: {value}")
        values.append(value)
    if not values:
        raise ValueError("未解析出任何 worker 数，请检查 --benchmark-workers")
    return sorted(set(values))


def benchmark_workers(
    triplets: Sequence[Tuple[str, str, str]],
    worker_counts: Sequence[int],
    speed_samples: int,
    repeats: int,
) -> List[Dict[str, float]]:
    all_rows: List[Dict[str, float]] = []
    for workers in worker_counts:
        for run_idx in range(repeats):
            dataset = make_dataset(triplets, num_workers=workers)
            metrics = test_speed(
                dataset=dataset,
                num_workers=workers,
                num_samples=speed_samples,
            )
            row = {
                "workers": workers,
                "run_idx": run_idx + 1,
                "samples": metrics["samples"],
                "elapsed_sec": metrics["elapsed_sec"],
                "samples_per_sec": metrics["samples_per_sec"],
                "ms_per_sample": metrics["ms_per_sample"],
                "tokens_per_sec": metrics["tokens_per_sec"],
            }
            all_rows.append(row)
            print(
                f"[BENCH] workers={workers:<2d} run={run_idx + 1}/{repeats} "
                f"samples/s={row['samples_per_sec']:.2f} ms/sample={row['ms_per_sample']:.2f}"
            )
    return all_rows


def aggregate_benchmark_rows(rows: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    by_worker: Dict[int, List[Dict[str, float]]] = {}
    for row in rows:
        by_worker.setdefault(int(row["workers"]), []).append(row)

    summary: List[Dict[str, float]] = []
    for workers, items in sorted(by_worker.items()):
        samples_per_sec = [x["samples_per_sec"] for x in items]
        ms_per_sample = [x["ms_per_sample"] for x in items]
        tokens_per_sec = [x["tokens_per_sec"] for x in items]
        summary.append(
            {
                "workers": workers,
                "runs": len(items),
                "samples_per_sec_mean": sum(samples_per_sec) / len(samples_per_sec),
                "samples_per_sec_max": max(samples_per_sec),
                "ms_per_sample_mean": sum(ms_per_sample) / len(ms_per_sample),
                "tokens_per_sec_mean": sum(tokens_per_sec) / len(tokens_per_sec),
            }
        )
    summary.sort(key=lambda x: x["samples_per_sec_mean"], reverse=True)
    return summary


def _sample_std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(var)


def recommend_best_worker(
    detail_rows: Sequence[Dict[str, float]],
    summary_rows: Sequence[Dict[str, float]],
) -> Dict[str, object]:
    by_worker: Dict[int, List[float]] = {}
    for row in detail_rows:
        by_worker.setdefault(int(row["workers"]), []).append(float(row["samples_per_sec"]))

    candidates: List[Dict[str, float]] = []
    for summary in summary_rows:
        workers = int(summary["workers"])
        runs = by_worker.get(workers, [])
        mean_sps = float(summary["samples_per_sec_mean"])
        std_sps = _sample_std(runs)
        cv = (std_sps / mean_sps) if mean_sps > 1e-9 else float("inf")
        # 综合打分：吞吐优先，波动惩罚；cv>=50% 时惩罚封顶。
        score = mean_sps * (1.0 - min(cv, 0.5))
        candidates.append(
            {
                "workers": workers,
                "mean_sps": mean_sps,
                "std_sps": std_sps,
                "cv": cv,
                "score": score,
            }
        )

    candidates.sort(key=lambda x: x["score"], reverse=True)
    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None
    throughput_gap = None
    if second is not None and second["mean_sps"] > 0:
        throughput_gap = (best["mean_sps"] - second["mean_sps"]) / second["mean_sps"]
    return {
        "best": best,
        "second": second,
        "throughput_gap_vs_second": throughput_gap,
        "all_candidates": candidates,
    }


def save_benchmark_csv(
    output_dir: Path,
    detail_rows: Sequence[Dict[str, float]],
    summary_rows: Sequence[Dict[str, float]],
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    detail_path = output_dir / f"worker_benchmark_detail_{ts}.csv"
    summary_path = output_dir / f"worker_benchmark_summary_{ts}.csv"

    with detail_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "workers",
                "run_idx",
                "samples",
                "elapsed_sec",
                "samples_per_sec",
                "ms_per_sample",
                "tokens_per_sec",
            ],
        )
        writer.writeheader()
        for row in detail_rows:
            writer.writerow(row)

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "workers",
                "runs",
                "samples_per_sec_mean",
                "samples_per_sec_max",
                "ms_per_sample_mean",
                "tokens_per_sec_mean",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    return detail_path, summary_path


def try_save_benchmark_plot(output_dir: Path, summary_rows: Sequence[Dict[str, float]]) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    plot_path = output_dir / f"worker_benchmark_plot_{ts}.png"

    workers = [int(r["workers"]) for r in sorted(summary_rows, key=lambda x: int(x["workers"]))]
    samples_per_sec = [r["samples_per_sec_mean"] for r in sorted(summary_rows, key=lambda x: int(x["workers"]))]
    ms_per_sample = [r["ms_per_sample_mean"] for r in sorted(summary_rows, key=lambda x: int(x["workers"]))]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(workers, samples_per_sec, marker="o", color="tab:blue", label="samples/sec (mean)")
    ax1.set_xlabel("num_workers")
    ax1.set_ylabel("samples/sec", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, linestyle="--", alpha=0.35)

    ax2 = ax1.twinx()
    ax2.plot(workers, ms_per_sample, marker="s", color="tab:red", label="ms/sample (mean)")
    ax2.set_ylabel("ms/sample", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path


def main():
    parser = argparse.ArgumentParser(description="SftAgenticIterableDataset 测试脚本")
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-items", type=int, default=3000, help="最多加载多少条 trajectory 作为样本池")
    parser.add_argument("--traverse-samples", type=int, default=200, help="遍历测试采样数")
    parser.add_argument("--random-check-samples", type=int, default=100, help="随机性检测采样数")
    parser.add_argument("--speed-samples", type=int, default=500, help="速度测试采样数")
    parser.add_argument("--traverse-workers", type=int, default=2, help="遍历测试 dataloader worker 数")
    parser.add_argument("--speed-workers", type=int, default=4, help="速度测试 dataloader worker 数")
    parser.add_argument("--benchmark-workers", type=str, default="", help="worker 扫描列表，如 1,2,4,8")
    parser.add_argument("--benchmark-repeats", type=int, default=2, help="每个 worker 重复次数")
    parser.add_argument("--benchmark-output-dir", type=str, default="data/benchmark_outputs", help="benchmark 输出目录")
    args = parser.parse_args()

    triplets = discover_triplets(args.output_root)
    if not triplets:
        raise RuntimeError(f"未发现可用数据，请检查目录: {args.output_root}")
    triplets = triplets[: args.max_items]
    print(f"[INFO] 发现可用样本池: {len(triplets)}")
    print(f"[INFO] output_root: {args.output_root}")

    dataset_for_traversal = make_dataset(triplets, num_workers=args.traverse_workers)
    dataset_for_random = make_dataset(triplets, num_workers=1)
    dataset_for_speed = make_dataset(triplets, num_workers=args.speed_workers)

    print("\n========== 1) 遍历测试 ==========")
    traversal_metrics = test_traversal(
        dataset_for_traversal,
        num_workers=args.traverse_workers,
        max_samples=args.traverse_samples,
    )
    print(traversal_metrics)

    print("\n========== 2) 随机性测试 ==========")
    random_metrics = test_randomness(
        dataset_for_random,
        check_samples=args.random_check_samples,
    )
    print(random_metrics)
    if not random_metrics["same_seed_consistent"]:
        print("[WARN] 同 seed 下顺序不一致，请检查 worker / seed 管理逻辑。")
    if not random_metrics["different_seed_changed"]:
        print("[WARN] 不同 seed 下顺序未变化，可能样本数量太小或已被固定。")

    print("\n========== 3) 读取速度测试 ==========")
    speed_metrics = test_speed(
        dataset_for_speed,
        num_workers=args.speed_workers,
        num_samples=args.speed_samples,
    )
    print(speed_metrics)

    if args.benchmark_workers.strip():
        print("\n========== 4) Worker 数目对比 Benchmark ==========")
        worker_counts = parse_worker_counts(args.benchmark_workers)
        detail_rows = benchmark_workers(
            triplets=triplets,
            worker_counts=worker_counts,
            speed_samples=args.speed_samples,
            repeats=args.benchmark_repeats,
        )
        summary_rows = aggregate_benchmark_rows(detail_rows)
        print("\n[BENCH] 按平均 samples/s 排序:")
        for row in summary_rows:
            print(
                f"  workers={int(row['workers']):<2d} runs={int(row['runs'])} "
                f"mean={row['samples_per_sec_mean']:.2f} "
                f"max={row['samples_per_sec_max']:.2f} "
                f"mean_ms={row['ms_per_sample_mean']:.2f}"
            )

        rec = recommend_best_worker(detail_rows, summary_rows)
        best = rec["best"]
        second = rec["second"]
        gap = rec["throughput_gap_vs_second"]
        print("\n[BENCH] 自动推荐:")
        print(
            f"  推荐 workers={int(best['workers'])}, "
            f"mean={best['mean_sps']:.2f} samples/s, "
            f"std={best['std_sps']:.2f}, cv={best['cv']*100:.2f}%, "
            f"score={best['score']:.2f}"
        )
        if second is not None and gap is not None:
            print(
                f"  次优 workers={int(second['workers'])}, "
                f"mean={second['mean_sps']:.2f} samples/s, "
                f"std={second['std_sps']:.2f}, cv={second['cv']*100:.2f}%"
            )
            if gap < 0.03:
                print("  建议：最优与次优吞吐差 < 3%，优先选择更稳定（cv 更低）或更省资源的 worker。")
            else:
                print("  建议：当前最优与次优差距明显，可优先使用推荐 worker。")
        if best["cv"] > 0.1:
            print("  提示：推荐 worker 的波动较高（cv > 10%），建议增加 repeats 再确认。")

        detail_csv, summary_csv = save_benchmark_csv(
            output_dir=Path(args.benchmark_output_dir),
            detail_rows=detail_rows,
            summary_rows=summary_rows,
        )
        print(f"[BENCH] 明细CSV: {detail_csv}")
        print(f"[BENCH] 汇总CSV: {summary_csv}")

        plot_path = try_save_benchmark_plot(Path(args.benchmark_output_dir), summary_rows)
        if plot_path is None:
            print("[BENCH] 未生成图像：当前环境缺少 matplotlib（CSV 已保存，可自行画图）。")
        else:
            print(f"[BENCH] 可视化图: {plot_path}")

    print("\n[DONE] 测试完成。")


if __name__ == "__main__":
    main()
