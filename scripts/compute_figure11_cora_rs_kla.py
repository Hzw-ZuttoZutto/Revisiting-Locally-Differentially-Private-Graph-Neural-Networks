#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator
import numpy as np
import torch
import yaml
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor, matmul

from datasets import load_dataset
from transforms import FeaturePerturbation


DEFAULT_CONFIG_DIR = Path("configs_final/figure11")
DEFAULT_OUTPUT_DIR = Path("paper_experiments/figure11_kernel_alignment/cora_input_feature_cosine")
DEFAULT_SEEDS = [12345, 12346, 12347, 12348, 12349]
DEFAULT_ETA = 1e-12
STEP_COLORS = {
    0: "#111111",
    2: "#234A6F",
    4: "#A7583B",
    8: "#5E7D4D",
    16: "#6E5A8A",
    32: "#8C6D31",
    64: "#3F6B6E",
}
EXPECTED_DATASET = "cora"
EXPECTED_FEATURE = "raw"
EXPECTED_MECHANISM = "mbm"
EXPECTED_SMOOTHER = "hoa"


def parse_int_list(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",")]
    parsed = [int(item) for item in values if item != ""]
    if len(parsed) == 0:
        raise ValueError("expected at least one integer")
    return parsed


def parse_float_list(raw: str) -> list[float]:
    values = [item.strip() for item in raw.split(",")]
    parsed = [float(item) for item in values if item != ""]
    if len(parsed) == 0:
        raise ValueError("expected at least one float")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Figure 11 style RS / KLA curves on cora input features."
    )
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--seeds",
        type=parse_int_list,
        default=list(DEFAULT_SEEDS),
        help="Comma-separated random seeds.",
    )
    parser.add_argument(
        "--x-steps",
        type=parse_int_list,
        default=None,
        help="Optional comma-separated x_steps subset. Defaults to all steps parsed from config-dir.",
    )
    parser.add_argument(
        "--x-eps",
        type=parse_float_list,
        default=None,
        help="Optional comma-separated x_eps subset. Defaults to the shared x_eps grid parsed from config-dir.",
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
        help="Device used for dense kernel computations.",
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


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def step_sort_key(path: Path) -> int:
    name = path.stem
    try:
        return int(name.split("=", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Unexpected figure11 config filename: {path.name}") from exc


def format_x_eps_tick(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def center_kernel(kernel: torch.Tensor) -> torch.Tensor:
    row_mean = kernel.mean(dim=1, keepdim=True)
    col_mean = kernel.mean(dim=0, keepdim=True)
    total_mean = kernel.mean()
    return kernel - row_mean - col_mean + total_mean


def cosine_normalize_rows(matrix: torch.Tensor, eta: float) -> torch.Tensor:
    norms = torch.linalg.vector_norm(matrix, ord=2, dim=1, keepdim=True)
    return matrix / (norms + float(eta))


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


def load_figure11_grid(config_dir: Path) -> tuple[list[int], list[float]]:
    config_paths = sorted(config_dir.glob("x_steps=*.yaml"), key=step_sort_key)
    if len(config_paths) != 7:
        raise ValueError(f"Expected 7 figure11 yaml files under {config_dir}, found {len(config_paths)}")

    shared_x_eps: list[float] | None = None
    parsed_steps: list[int] = []

    for config_path in config_paths:
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)

        try:
            dataset_values = payload["search_space"]["dataset"]["datasets"]
            feature_values = payload["search_space"]["feature_transformation"]["features"]
            mechanism_values = payload["search_space"]["feature_perturbation"]["mechanisms"]
            smoother_values = payload["search_space"]["calibrator"]["smoother"]
            x_steps_values = payload["search_space"]["calibrator"]["x_steps"]
            x_eps_values = payload["search_space"]["feature_perturbation"]["x_eps"]
        except Exception as exc:
            raise ValueError(f"{config_path} is missing expected figure11 fields") from exc

        if dataset_values != [EXPECTED_DATASET]:
            raise ValueError(f"{config_path}: expected datasets=[{EXPECTED_DATASET!r}], found {dataset_values!r}")
        if feature_values != [EXPECTED_FEATURE]:
            raise ValueError(f"{config_path}: expected features=[{EXPECTED_FEATURE!r}], found {feature_values!r}")
        if mechanism_values != [EXPECTED_MECHANISM]:
            raise ValueError(f"{config_path}: expected mechanisms=[{EXPECTED_MECHANISM!r}], found {mechanism_values!r}")
        if smoother_values != [EXPECTED_SMOOTHER]:
            raise ValueError(f"{config_path}: expected smoother=[{EXPECTED_SMOOTHER!r}], found {smoother_values!r}")
        if len(x_steps_values) != 1:
            raise ValueError(f"{config_path}: expected exactly one x_steps value, found {x_steps_values!r}")

        parsed_step = int(x_steps_values[0])
        expected_step = step_sort_key(config_path)
        if parsed_step != expected_step:
            raise ValueError(
                f"{config_path}: filename implies x_steps={expected_step}, but yaml contains {parsed_step}"
            )
        parsed_steps.append(parsed_step)

        current_x_eps = [float(value) for value in x_eps_values]
        if shared_x_eps is None:
            shared_x_eps = current_x_eps
        elif current_x_eps != shared_x_eps:
            raise ValueError(f"{config_path}: x_eps grid does not match the other figure11 configs")

    if shared_x_eps is None:
        raise RuntimeError(f"Failed to parse x_eps grid from {config_dir}")
    return parsed_steps, shared_x_eps


def build_centered_label_kernel(labels: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, float]:
    num_nodes = int(labels.numel())
    num_classes = int(labels.max().item()) + 1
    counts = torch.bincount(labels, minlength=num_classes).to(torch.float64)
    scaled = torch.zeros((num_nodes, num_classes), dtype=torch.float64)
    scaled[torch.arange(num_nodes), labels.cpu()] = counts[labels.cpu()].rsqrt()
    label_kernel = (scaled @ scaled.T).to(device)
    centered = center_kernel(label_kernel)
    norm = float(torch.linalg.matrix_norm(centered, ord="fro").item())
    if not math.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("Centered label kernel has zero or invalid Frobenius norm.")
    return centered, norm


def build_anchor_kernels(
    *,
    adj_t: SparseTensor,
    num_nodes: int,
    x_steps: list[int],
    device: torch.device,
    eta: float,
) -> dict[int, tuple[torch.Tensor, float]]:
    results: dict[int, tuple[torch.Tensor, float]] = {}
    tracked_steps = sorted(x_steps)
    max_step = max(tracked_steps)
    step_set = set(tracked_steps)

    identity = torch.eye(num_nodes, dtype=torch.float32, device=device)

    def finalize(step: int, operator_matrix: torch.Tensor) -> None:
        normalized = cosine_normalize_rows(operator_matrix, eta=float(eta))
        kernel = (normalized @ normalized.T).to(torch.float64)
        centered = center_kernel(kernel)
        norm = float(torch.linalg.matrix_norm(centered, ord="fro").item())
        if not math.isfinite(norm) or norm <= 0.0:
            raise RuntimeError(f"Anchor kernel has zero or invalid Frobenius norm for x_steps={step}")
        results[step] = (centered, norm)

    if 0 in step_set:
        finalize(0, identity)

    if max_step <= 0:
        return results

    current = identity
    accumulated = torch.zeros_like(identity)
    for step in range(1, max_step + 1):
        current = matmul(adj_t, current, reduce="add")
        accumulated = accumulated + current
        if step in step_set:
            finalize(step, accumulated / float(step))

    return results


def compute_representations_for_steps(
    *,
    private_x: torch.Tensor,
    adj_t: SparseTensor,
    x_steps: list[int],
) -> dict[int, torch.Tensor]:
    results: dict[int, torch.Tensor] = {}
    tracked_steps = sorted(x_steps)
    step_set = set(tracked_steps)
    max_step = max(tracked_steps)

    if 0 in step_set:
        results[0] = private_x

    if max_step <= 0:
        return results

    current = private_x
    accumulated = torch.zeros_like(private_x)
    for step in range(1, max_step + 1):
        current = matmul(adj_t, current, reduce="add")
        accumulated = accumulated + current
        if step in step_set:
            results[step] = accumulated / float(step)

    return results


def compute_centered_cosine_kernel(features: torch.Tensor, eta: float) -> tuple[torch.Tensor, float]:
    normalized = cosine_normalize_rows(features, eta=float(eta))
    kernel = (normalized @ normalized.T).to(torch.float64)
    centered = center_kernel(kernel)
    norm = float(torch.linalg.matrix_norm(centered, ord="fro").item())
    if not math.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("Centered representation kernel has zero or invalid Frobenius norm.")
    return centered, norm


def compute_alignment(
    centered_left: torch.Tensor,
    left_norm: float,
    centered_right: torch.Tensor,
    right_norm: float,
) -> float:
    value = float(torch.sum(centered_left * centered_right).item() / (left_norm * right_norm))
    if not math.isfinite(value):
        raise RuntimeError("Alignment became non-finite.")
    if value < -1.000001 or value > 1.000001:
        raise RuntimeError(f"Alignment out of expected range [-1,1]: {value}")
    return value


def build_private_features(
    *,
    raw_x: torch.Tensor,
    x_eps: float,
    seed: int,
) -> tuple[torch.Tensor, int | None]:
    seed_everything(seed)
    container = SimpleNamespace(x=raw_x.clone())
    perturbation = FeaturePerturbation(
        mechanism="mbm",
        x_eps=float(x_eps),
        m="best",
        norm=False,
        norm_scale="none",
        feature="raw",
        data_range=(0.0, 1.0),
    )
    container = perturbation(container)
    resolved_m = getattr(container, "feature_mechanism_resolved_m", None)
    return container.x, resolved_m


def summarize_rows(
    rows: list[dict[str, object]],
    *,
    x_steps: list[int],
    x_eps_values: list[float],
    seeds: list[int],
) -> list[dict[str, object]]:
    grouped: dict[tuple[int, float], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["x_steps"]), float(row["x_eps"]))].append(row)

    summary_rows: list[dict[str, object]] = []
    for step in x_steps:
        for x_eps in x_eps_values:
            group = grouped[(step, x_eps)]
            if len(group) != len(seeds):
                raise RuntimeError(
                    f"Expected {len(seeds)} rows for x_steps={step}, x_eps={x_eps}, found {len(group)}"
                )
            rs_values = [float(row["rs"]) for row in group]
            kla_values = [float(row["kla"]) for row in group]
            resolved_m_values = [int(row["resolved_m"]) for row in group if row["resolved_m"] not in ("", None)]
            unique_m = sorted(set(resolved_m_values))
            summary_rows.append(
                {
                    "dataset": EXPECTED_DATASET,
                    "representation_kernel": "cosine",
                    "x_steps": step,
                    "x_eps": f"{x_eps:.12g}",
                    "rs_mean": f"{float(np.mean(rs_values)):.12f}",
                    "rs_std": f"{float(np.std(rs_values, ddof=0)):.12f}",
                    "kla_mean": f"{float(np.mean(kla_values)):.12f}",
                    "kla_std": f"{float(np.std(kla_values, ddof=0)):.12f}",
                    "seed_count": len(seeds),
                    "resolved_m_values": ",".join(str(value) for value in unique_m),
                }
            )
    return summary_rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(
    *,
    summary_rows: list[dict[str, object]],
    metric_field: str,
    ylabel: str,
    output_pdf: Path,
    output_png: Path,
    x_steps: list[int],
    x_eps_values: list[float],
) -> None:
    rows_by_step: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        rows_by_step[int(row["x_steps"])].append(row)

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    y_values_all: list[float] = []

    for step in sorted(x_steps):
        rows = sorted(rows_by_step[step], key=lambda item: float(item["x_eps"]))
        x_vals = [float(row["x_eps"]) for row in rows]
        y_vals = [float(row[metric_field]) for row in rows]
        y_values_all.extend(y_vals)
        ax.plot(
            x_vals,
            y_vals,
            color=STEP_COLORS[step],
            linewidth=1.8,
            label=f"x_steps={step}",
        )

    y_min = min(y_values_all)
    y_max = max(y_values_all)
    y_span = max(y_max - y_min, 1e-6)
    margin = y_span * 0.08

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Privacy Budget ε")
    ax.set_ylabel(ylabel)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.xaxis.set_major_locator(FixedLocator(x_eps_values))
    ax.xaxis.set_major_formatter(FixedFormatter([format_x_eps_tick(value) for value in x_eps_values]))
    ax.tick_params(axis="x", labelrotation=50, labelsize=7)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=False, fontsize=8, handlelength=2.3)
    fig.tight_layout()

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, bbox_inches="tight", dpi=220)
    plt.close(fig)


def validate_rows(
    *,
    by_seed_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    x_steps: list[int],
    x_eps_values: list[float],
    seeds: list[int],
) -> None:
    expected_seed_rows = len(x_steps) * len(x_eps_values) * len(seeds)
    if len(by_seed_rows) != expected_seed_rows:
        raise RuntimeError(f"Expected {expected_seed_rows} per-seed rows, found {len(by_seed_rows)}")
    expected_summary_rows = len(x_steps) * len(x_eps_values)
    if len(summary_rows) != expected_summary_rows:
        raise RuntimeError(f"Expected {expected_summary_rows} summary rows, found {len(summary_rows)}")

    for row in by_seed_rows:
        for field_name in ("rs", "kla"):
            value = float(row[field_name])
            if not math.isfinite(value):
                raise RuntimeError(f"Found non-finite {field_name} row: {row}")
            if value < -1.000001 or value > 1.000001:
                raise RuntimeError(f"Found out-of-range {field_name} row: {row}")


def main() -> int:
    args = parse_args()
    if not math.isfinite(args.eta) or args.eta <= 0.0:
        raise ValueError("--eta must be finite and > 0")
    if len(args.seeds) == 0:
        raise ValueError("--seeds must contain at least one seed")

    device = resolve_device(args.device)
    config_dir = args.config_dir.resolve()
    parsed_steps, parsed_x_eps = load_figure11_grid(config_dir)

    selected_steps = parsed_steps if args.x_steps is None else sorted({int(value) for value in args.x_steps})
    selected_x_eps = parsed_x_eps if args.x_eps is None else sorted({float(value) for value in args.x_eps})

    if any(step not in parsed_steps for step in selected_steps):
        raise ValueError(f"Requested x_steps must be a subset of {parsed_steps}, got {selected_steps}")
    if any(x_eps not in parsed_x_eps for x_eps in selected_x_eps):
        raise ValueError(f"Requested x_eps must be a subset of {parsed_x_eps}, got {selected_x_eps}")

    print(f"[Info] device={device}")
    print(f"[Info] config_dir={config_dir}")
    print(f"[Info] x_steps={selected_steps}")
    print(f"[Info] x_eps={selected_x_eps}")
    print(f"[Info] seeds={args.seeds}")

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    data = load_dataset(
        dataset=EXPECTED_DATASET,
        data_dir="./datasets",
        data_range=(0.0, 1.0),
        val_ratio=0.25,
        test_ratio=0.25,
    )
    labels = data.y if data.y.dim() == 1 else data.y.argmax(dim=1)
    labels = labels.to(torch.long).cpu()
    num_nodes = int(data.num_nodes)
    raw_x = data.x.to(device=device, dtype=torch.float32)
    adj_t = gcn_norm(data.adj_t.cpu(), add_self_loops=False).coalesce()
    adj_t = move_sparse_tensor(adj_t, device)

    centered_label_kernel, label_kernel_norm = build_centered_label_kernel(labels, device)
    anchor_kernels = build_anchor_kernels(
        adj_t=adj_t,
        num_nodes=num_nodes,
        x_steps=selected_steps,
        device=device,
        eta=float(args.eta),
    )

    by_seed_rows: list[dict[str, object]] = []
    started_all = time.perf_counter()

    for x_eps in selected_x_eps:
        print(f"[Info] x_eps={x_eps}")
        for seed in args.seeds:
            started_seed = time.perf_counter()
            private_x, resolved_m = build_private_features(
                raw_x=raw_x,
                x_eps=float(x_eps),
                seed=int(seed),
            )
            representations = compute_representations_for_steps(
                private_x=private_x,
                adj_t=adj_t,
                x_steps=selected_steps,
            )
            for step in selected_steps:
                centered_rep_kernel, rep_kernel_norm = compute_centered_cosine_kernel(
                    representations[step],
                    eta=float(args.eta),
                )
                centered_anchor_kernel, anchor_kernel_norm = anchor_kernels[step]
                rs = compute_alignment(
                    centered_rep_kernel,
                    rep_kernel_norm,
                    centered_anchor_kernel,
                    anchor_kernel_norm,
                )
                kla = compute_alignment(
                    centered_rep_kernel,
                    rep_kernel_norm,
                    centered_label_kernel,
                    label_kernel_norm,
                )
                by_seed_rows.append(
                    {
                        "dataset": EXPECTED_DATASET,
                        "representation_kernel": "cosine",
                        "seed": int(seed),
                        "x_steps": int(step),
                        "x_eps": f"{float(x_eps):.12g}",
                        "rs": f"{rs:.12f}",
                        "kla": f"{kla:.12f}",
                        "resolved_m": "" if resolved_m is None else int(resolved_m),
                        "label_kernel_fro": f"{label_kernel_norm:.12f}",
                        "anchor_kernel_fro": f"{anchor_kernel_norm:.12f}",
                        "representation_kernel_fro": f"{rep_kernel_norm:.12f}",
                    }
                )
            elapsed_seed = time.perf_counter() - started_seed
            print(
                f"  [Done] seed={seed} resolved_m={resolved_m} "
                f"elapsed={elapsed_seed:.2f}s rs(x_steps={selected_steps[0]})="
                f"{float(by_seed_rows[-len(selected_steps)]['rs']):.6f}"
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary_rows = summarize_rows(
        by_seed_rows,
        x_steps=selected_steps,
        x_eps_values=selected_x_eps,
        seeds=args.seeds,
    )
    validate_rows(
        by_seed_rows=by_seed_rows,
        summary_rows=summary_rows,
        x_steps=selected_steps,
        x_eps_values=selected_x_eps,
        seeds=args.seeds,
    )

    output_dir = args.output_dir.resolve()
    metrics_by_seed_path = output_dir / "metrics_by_seed.csv"
    metrics_summary_path = output_dir / "metrics_summary.csv"
    rs_pdf = output_dir / "cora_rs_vs_xeps_all_steps.pdf"
    rs_png = output_dir / "cora_rs_vs_xeps_all_steps.png"
    kla_pdf = output_dir / "cora_kla_vs_xeps_all_steps.pdf"
    kla_png = output_dir / "cora_kla_vs_xeps_all_steps.png"

    write_csv(
        metrics_by_seed_path,
        [
            "dataset",
            "representation_kernel",
            "seed",
            "x_steps",
            "x_eps",
            "rs",
            "kla",
            "resolved_m",
            "label_kernel_fro",
            "anchor_kernel_fro",
            "representation_kernel_fro",
        ],
        by_seed_rows,
    )
    write_csv(
        metrics_summary_path,
        [
            "dataset",
            "representation_kernel",
            "x_steps",
            "x_eps",
            "rs_mean",
            "rs_std",
            "kla_mean",
            "kla_std",
            "seed_count",
            "resolved_m_values",
        ],
        summary_rows,
    )

    plot_metric(
        summary_rows=summary_rows,
        metric_field="rs_mean",
        ylabel="RS",
        output_pdf=rs_pdf,
        output_png=rs_png,
        x_steps=selected_steps,
        x_eps_values=selected_x_eps,
    )
    plot_metric(
        summary_rows=summary_rows,
        metric_field="kla_mean",
        ylabel="KLA",
        output_pdf=kla_pdf,
        output_png=kla_png,
        x_steps=selected_steps,
        x_eps_values=selected_x_eps,
    )

    elapsed_all = time.perf_counter() - started_all
    print(f"[Done] wrote {metrics_by_seed_path}")
    print(f"[Done] wrote {metrics_summary_path}")
    print(f"[Done] wrote {rs_pdf}")
    print(f"[Done] wrote {rs_png}")
    print(f"[Done] wrote {kla_pdf}")
    print(f"[Done] wrote {kla_png}")
    print(f"[Done] total_elapsed={elapsed_all:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
