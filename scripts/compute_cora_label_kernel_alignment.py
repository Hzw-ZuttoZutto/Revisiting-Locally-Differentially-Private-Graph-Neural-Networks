#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator
import numpy as np
import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor, matmul

from datasets import load_dataset


DEFAULT_FEATURE_DIMS = [
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
]
DEFAULT_X_STEPS = [0, 2, 4, 8, 16, 32, 64]
DEFAULT_SEEDS = [12345, 12346, 12347, 12348, 12349]
DEFAULT_OUTPUT_DIR = Path(
    "paper_experiments/label_kernel_alignment/cora_random_normal_hoa_cosine"
)
DEFAULT_CHUNK_DIM = 512
DEFAULT_ETA = 1e-12
DATASET_NAME = "cora"
STEP_COLORS = {
    0: "#3B4252",
    2: "#234A6F",
    4: "#A7583B",
    8: "#5E7D4D",
    16: "#6E5A8A",
    32: "#8C6D31",
    64: "#3F6B6E",
}


def parse_int_list(raw: str) -> list[int]:
    items = [item.strip() for item in raw.split(",")]
    values: list[int] = []
    for item in items:
        if item == "":
            continue
        values.append(int(item))
    if len(values) == 0:
        raise ValueError("expected at least one integer")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute cora random-feature HOA cosine label-kernel alignment curves."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--feature-dims",
        type=parse_int_list,
        default=list(DEFAULT_FEATURE_DIMS),
        help="Comma-separated feature dimensions.",
    )
    parser.add_argument(
        "--x-steps",
        type=parse_int_list,
        default=list(DEFAULT_X_STEPS),
        help="Comma-separated HOA step counts.",
    )
    parser.add_argument(
        "--seeds",
        type=parse_int_list,
        default=list(DEFAULT_SEEDS),
        help="Comma-separated random seeds.",
    )
    parser.add_argument(
        "--chunk-dim",
        type=int,
        default=DEFAULT_CHUNK_DIM,
        help="Feature chunk width used during generation and aggregation.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=DEFAULT_ETA,
        help="Stability constant for cosine row normalization.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for the heavy linear algebra.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def move_sparse_tensor(adj_t: SparseTensor, device: torch.device) -> SparseTensor:
    row, col, value = adj_t.coo()
    if value is None:
        value = torch.ones(row.numel(), dtype=torch.float32, device=row.device)
    if row.device == device and col.device == device and value.device == device:
        return adj_t
    return SparseTensor(
        row=row.to(device),
        col=col.to(device),
        value=value.to(device=device, dtype=torch.float32),
        sparse_sizes=adj_t.sparse_sizes(),
    ).coalesce()


def center_kernel(kernel: torch.Tensor) -> torch.Tensor:
    row_mean = kernel.mean(dim=1, keepdim=True)
    col_mean = kernel.mean(dim=0, keepdim=True)
    total_mean = kernel.mean()
    return kernel - row_mean - col_mean + total_mean


def build_centered_label_kernel_factor(
    labels: torch.Tensor,
    *,
    num_classes: int,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    num_nodes = int(labels.numel())
    counts = torch.bincount(labels, minlength=num_classes).to(torch.float64)
    scaled = torch.zeros((num_nodes, num_classes), dtype=torch.float64)
    scaled[torch.arange(num_nodes), labels.cpu()] = counts[labels.cpu()].rsqrt()
    scaled = scaled.to(device)
    centered_factor = scaled - scaled.mean(dim=0, keepdim=True)
    gram_small = centered_factor.T @ centered_factor
    label_norm = float(torch.linalg.matrix_norm(gram_small, ord="fro").item())
    if not math.isfinite(label_norm) or label_norm <= 0.0:
        raise RuntimeError("Centered label kernel has zero or invalid Frobenius norm.")
    return centered_factor.T.contiguous(), label_norm


def generate_random_chunk(
    *,
    num_nodes: int,
    width: int,
    feature_dim: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    chunk = torch.normal(
        mean=0.0,
        std=1.0,
        size=(num_nodes, width),
        generator=generator,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    if device.type != "cpu":
        chunk = chunk.to(device)
    return chunk


def iter_chunk_widths(total_dim: int, chunk_dim: int):
    remaining = total_dim
    while remaining > 0:
        width = min(chunk_dim, remaining)
        yield width
        remaining -= width


def compute_step_feature_norms(
    *,
    num_nodes: int,
    feature_dim: int,
    tracked_steps: list[int],
    chunk_dim: int,
    seed: int,
    adj_t: SparseTensor,
    device: torch.device,
    eta: float,
) -> dict[int, torch.Tensor]:
    tracked_positive_steps = [step for step in tracked_steps if step > 0]
    tracked_step_set = set(tracked_positive_steps)
    max_step = max(tracked_positive_steps, default=0)

    norm_sq = {
        step: torch.zeros(num_nodes, dtype=torch.float64, device=device)
        for step in tracked_steps
    }
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    for width in iter_chunk_widths(feature_dim, chunk_dim):
        chunk = generate_random_chunk(
            num_nodes=num_nodes,
            width=width,
            feature_dim=feature_dim,
            generator=generator,
            device=device,
        )
        if 0 in norm_sq:
            norm_sq[0] += torch.sum(chunk * chunk, dim=1, dtype=torch.float64)

        if max_step <= 0:
            continue

        current = chunk
        accumulated = torch.zeros_like(chunk)
        for step in range(1, max_step + 1):
            current = matmul(adj_t, current, reduce="add")
            accumulated.add_(current)
            if step in tracked_step_set:
                features = accumulated / float(step)
                norm_sq[step] += torch.sum(features * features, dim=1, dtype=torch.float64)

    norms = {
        step: torch.sqrt(step_norm_sq).add(float(eta))
        for step, step_norm_sq in norm_sq.items()
    }
    return norms


def compute_alignments_for_seed(
    *,
    num_nodes: int,
    feature_dim: int,
    tracked_steps: list[int],
    chunk_dim: int,
    seed: int,
    adj_t: SparseTensor,
    label_factor_t: torch.Tensor,
    label_norm: float,
    device: torch.device,
    eta: float,
) -> dict[int, dict[str, float]]:
    norms = compute_step_feature_norms(
        num_nodes=num_nodes,
        feature_dim=feature_dim,
        tracked_steps=tracked_steps,
        chunk_dim=chunk_dim,
        seed=seed,
        adj_t=adj_t,
        device=device,
        eta=eta,
    )

    tracked_positive_steps = [step for step in tracked_steps if step > 0]
    tracked_step_set = set(tracked_positive_steps)
    max_step = max(tracked_positive_steps, default=0)

    kernel_acc = {
        step: torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
        for step in tracked_steps
    }
    numerator_acc = {step: 0.0 for step in tracked_steps}

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    for width in iter_chunk_widths(feature_dim, chunk_dim):
        chunk = generate_random_chunk(
            num_nodes=num_nodes,
            width=width,
            feature_dim=feature_dim,
            generator=generator,
            device=device,
        )

        if 0 in kernel_acc:
            normalized = chunk / norms[0].to(dtype=chunk.dtype).unsqueeze(1)
            projection = label_factor_t @ normalized.to(torch.float64)
            numerator_acc[0] += float(torch.sum(projection.square()).item())
            kernel_acc[0].addmm_(normalized, normalized.T, beta=1.0, alpha=1.0)

        if max_step <= 0:
            continue

        current = chunk
        accumulated = torch.zeros_like(chunk)
        for step in range(1, max_step + 1):
            current = matmul(adj_t, current, reduce="add")
            accumulated.add_(current)
            if step in tracked_step_set:
                features = accumulated / float(step)
                normalized = features / norms[step].to(dtype=features.dtype).unsqueeze(1)
                projection = label_factor_t @ normalized.to(torch.float64)
                numerator_acc[step] += float(torch.sum(projection.square()).item())
                kernel_acc[step].addmm_(normalized, normalized.T, beta=1.0, alpha=1.0)

    results: dict[int, dict[str, float]] = {}
    for step in tracked_steps:
        centered_kernel = center_kernel(kernel_acc[step].to(dtype=torch.float64))
        kernel_norm = float(torch.linalg.matrix_norm(centered_kernel, ord="fro").item())
        if not math.isfinite(kernel_norm) or kernel_norm <= 0.0:
            raise RuntimeError(
                f"Centered representation kernel has zero or invalid Frobenius norm "
                f"for feature_dim={feature_dim}, x_steps={step}, seed={seed}."
            )

        alignment = numerator_acc[step] / (kernel_norm * label_norm)
        if not math.isfinite(alignment):
            raise RuntimeError(
                f"Alignment became non-finite for feature_dim={feature_dim}, "
                f"x_steps={step}, seed={seed}."
            )
        if alignment < -1.000001 or alignment > 1.000001:
            raise RuntimeError(
                f"Alignment={alignment} is outside the expected range [-1, 1] "
                f"for feature_dim={feature_dim}, x_steps={step}, seed={seed}."
            )

        results[step] = {
            "alignment": alignment,
            "numerator": numerator_acc[step],
            "representation_kernel_fro": kernel_norm,
            "label_kernel_fro": label_norm,
        }

    return results


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(
    *,
    summary_rows: list[dict[str, object]],
    output_pdf: Path,
    output_png: Path,
    feature_dims: list[int],
    x_steps: list[int],
) -> None:
    rows_by_step: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        rows_by_step[int(row["x_steps"])].append(row)

    fig, ax = plt.subplots(figsize=(7.2, 4.3))

    y_values_all: list[float] = []
    for step in x_steps:
        rows = sorted(rows_by_step[step], key=lambda item: int(item["feature_dim"]))
        x_vals = [int(row["feature_dim"]) for row in rows]
        y_vals = [float(row["alignment_mean"]) for row in rows]
        y_values_all.extend(y_vals)
        ax.plot(
            x_vals,
            y_vals,
            color=STEP_COLORS.get(step, None),
            linewidth=1.8,
            label=f"x_steps={step}",
        )

    y_min = min(y_values_all)
    y_max = max(y_values_all)
    y_span = max(y_max - y_min, 1e-6)
    margin = y_span * 0.08

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Feature Dimension")
    ax.set_ylabel("Label-Kernel Alignment")
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.xaxis.set_major_locator(FixedLocator(feature_dims))
    ax.xaxis.set_major_formatter(FixedFormatter([str(dim) for dim in feature_dims]))
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=False, fontsize=8, handlelength=2.3)
    ax.set_title("Cora Random-Normal HOA Cosine Label-Kernel Alignment", fontsize=11)
    fig.tight_layout()

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, bbox_inches="tight", dpi=220)
    plt.close(fig)


def validate_grid(
    *,
    by_seed_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    feature_dims: list[int],
    x_steps: list[int],
    seeds: list[int],
) -> None:
    expected_by_seed = len(feature_dims) * len(x_steps) * len(seeds)
    if len(by_seed_rows) != expected_by_seed:
        raise RuntimeError(
            f"Expected {expected_by_seed} per-seed rows, found {len(by_seed_rows)}."
        )

    expected_summary = len(feature_dims) * len(x_steps)
    if len(summary_rows) != expected_summary:
        raise RuntimeError(
            f"Expected {expected_summary} summary rows, found {len(summary_rows)}."
        )

    for row in by_seed_rows:
        alignment = float(row["alignment"])
        if not math.isfinite(alignment):
            raise RuntimeError(f"Found non-finite alignment row: {row}")
        if alignment < -1.000001 or alignment > 1.000001:
            raise RuntimeError(f"Found out-of-range alignment row: {row}")


def summarize_rows(
    by_seed_rows: list[dict[str, object]],
    *,
    feature_dims: list[int],
    x_steps: list[int],
    seeds: list[int],
) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in by_seed_rows:
        grouped[(int(row["x_steps"]), int(row["feature_dim"]))].append(float(row["alignment"]))

    summary_rows: list[dict[str, object]] = []
    for step in x_steps:
        for feature_dim in feature_dims:
            values = grouped[(step, feature_dim)]
            if len(values) != len(seeds):
                raise RuntimeError(
                    f"Expected {len(seeds)} values for x_steps={step}, feature_dim={feature_dim}, "
                    f"found {len(values)}."
                )
            summary_rows.append(
                {
                    "dataset": DATASET_NAME,
                    "kernel": "cosine",
                    "x_steps": step,
                    "feature_dim": feature_dim,
                    "alignment_mean": f"{float(np.mean(values)):.12f}",
                    "alignment_std": f"{float(np.std(values, ddof=0)):.12f}",
                    "seed_count": len(seeds),
                }
            )
    return summary_rows


def main() -> int:
    args = parse_args()
    feature_dims = sorted({int(value) for value in args.feature_dims})
    x_steps = sorted({int(value) for value in args.x_steps})
    seeds = [int(value) for value in args.seeds]

    if any(value <= 0 for value in feature_dims):
        raise ValueError("--feature-dims must all be > 0")
    if any(value < 0 for value in x_steps):
        raise ValueError("--x-steps must all be >= 0")
    if len(seeds) == 0:
        raise ValueError("--seeds must contain at least one value")
    if args.chunk_dim <= 0:
        raise ValueError("--chunk-dim must be > 0")
    if not math.isfinite(args.eta) or args.eta <= 0.0:
        raise ValueError("--eta must be finite and > 0")

    device = resolve_device(args.device)
    print(f"[Info] device={device}")
    print(f"[Info] feature_dims={feature_dims}")
    print(f"[Info] x_steps={x_steps}")
    print(f"[Info] seeds={seeds}")

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    data = load_dataset(
        dataset=DATASET_NAME,
        data_dir="./datasets",
        data_range=(0.0, 1.0),
        val_ratio=0.25,
        test_ratio=0.25,
    )
    labels = data.y if data.y.dim() == 1 else data.y.argmax(dim=1)
    labels = labels.to(torch.long).cpu()
    num_nodes = int(data.num_nodes)
    num_classes = int(labels.max().item()) + 1
    adj_t = gcn_norm(data.adj_t.cpu(), add_self_loops=False).coalesce()
    adj_t = move_sparse_tensor(adj_t, device)
    label_factor_t, label_norm = build_centered_label_kernel_factor(
        labels,
        num_classes=num_classes,
        device=device,
    )

    by_seed_rows: list[dict[str, object]] = []
    started_all = time.perf_counter()

    for feature_dim in feature_dims:
        print(f"[Info] feature_dim={feature_dim}")
        for seed in seeds:
            started_seed = time.perf_counter()
            seed_results = compute_alignments_for_seed(
                num_nodes=num_nodes,
                feature_dim=feature_dim,
                tracked_steps=x_steps,
                chunk_dim=int(args.chunk_dim),
                seed=seed,
                adj_t=adj_t,
                label_factor_t=label_factor_t,
                label_norm=label_norm,
                device=device,
                eta=float(args.eta),
            )
            elapsed_seed = time.perf_counter() - started_seed
            for step in x_steps:
                metrics = seed_results[step]
                by_seed_rows.append(
                    {
                        "dataset": DATASET_NAME,
                        "kernel": "cosine",
                        "seed": seed,
                        "x_steps": step,
                        "feature_dim": feature_dim,
                        "alignment": f"{metrics['alignment']:.12f}",
                        "numerator": f"{metrics['numerator']:.12f}",
                        "representation_kernel_fro": f"{metrics['representation_kernel_fro']:.12f}",
                        "label_kernel_fro": f"{metrics['label_kernel_fro']:.12f}",
                        "elapsed_sec_for_seed_dim": f"{elapsed_seed:.6f}",
                        "device": str(device),
                        "eta": f"{float(args.eta):.12g}",
                        "chunk_dim": int(args.chunk_dim),
                    }
                )
            print(
                f"  [Done] seed={seed} elapsed={elapsed_seed:.2f}s "
                f"align(x_steps=0)={seed_results[x_steps[0]]['alignment']:.6f}"
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary_rows = summarize_rows(
        by_seed_rows,
        feature_dims=feature_dims,
        x_steps=x_steps,
        seeds=seeds,
    )
    validate_grid(
        by_seed_rows=by_seed_rows,
        summary_rows=summary_rows,
        feature_dims=feature_dims,
        x_steps=x_steps,
        seeds=seeds,
    )

    output_dir = args.output_dir.resolve()
    by_seed_path = output_dir / "alignment_by_seed.csv"
    summary_path = output_dir / "alignment_summary.csv"
    output_pdf = output_dir / "cora_label_kernel_alignment_cosine.pdf"
    output_png = output_dir / "cora_label_kernel_alignment_cosine.png"

    write_csv(
        by_seed_path,
        by_seed_rows,
        fieldnames=[
            "dataset",
            "kernel",
            "seed",
            "x_steps",
            "feature_dim",
            "alignment",
            "numerator",
            "representation_kernel_fro",
            "label_kernel_fro",
            "elapsed_sec_for_seed_dim",
            "device",
            "eta",
            "chunk_dim",
        ],
    )
    write_csv(
        summary_path,
        summary_rows,
        fieldnames=[
            "dataset",
            "kernel",
            "x_steps",
            "feature_dim",
            "alignment_mean",
            "alignment_std",
            "seed_count",
        ],
    )
    plot_summary(
        summary_rows=summary_rows,
        output_pdf=output_pdf,
        output_png=output_png,
        feature_dims=feature_dims,
        x_steps=x_steps,
    )

    elapsed_all = time.perf_counter() - started_all
    print(f"[Done] wrote {by_seed_path}")
    print(f"[Done] wrote {summary_path}")
    print(f"[Done] wrote {output_pdf}")
    print(f"[Done] wrote {output_png}")
    print(f"[Done] total_elapsed={elapsed_all:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
