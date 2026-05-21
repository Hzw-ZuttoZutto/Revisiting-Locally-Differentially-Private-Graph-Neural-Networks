#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import matmul

from datasets import load_dataset

DATASET_SPECS = [
    ('cora', 64),
    ('lastfm', 64),
    ('citeseer', 64),
    ('ogbn-arxiv', 16),
    ('Books-History', 16),
    ('Books-Children', 8),
]
DEFAULT_FEATURE_DIM = 3200
DEFAULT_SEED = 12345
DEFAULT_CHUNK_DIM = 320
DEFAULT_OUTPUT_DIR = Path('paper_experiments/class_similarity_heatmaps/random3200_hoa')


def sanitize_name(name: str) -> str:
    return name.lower().replace(' ', '-').replace('_', '-').replace('/', '-')


def format_value(value: float) -> str:
    if not np.isfinite(value):
        return 'N/A'
    abs_value = abs(value)
    if abs_value == 0:
        return '0'
    if abs_value >= 10:
        return f'{value:.2f}'
    if abs_value >= 1:
        return f'{value:.3f}'
    if abs_value >= 1e-2:
        return f'{value:.4f}'
    return f'{value:.2e}'


def compute_class_similarity_matrix(
    *,
    dataset_name: str,
    hops: int,
    feature_dim: int,
    seed: int,
    chunk_dim: int,
) -> tuple[np.ndarray, list[int], int, int, float, float]:
    data = load_dataset(
        dataset=dataset_name,
        data_dir='./datasets',
        data_range=(0.0, 1.0),
        val_ratio=0.25,
        test_ratio=0.25,
    )
    y = data.y if data.y.dim() == 1 else data.y.argmax(dim=1)
    y = y.to(torch.long).cpu()
    num_nodes = int(data.num_nodes)
    num_classes = int(y.max().item()) + 1
    class_counts_tensor = torch.bincount(y, minlength=num_classes).to(torch.float64)
    class_counts = [int(v) for v in class_counts_tensor.tolist()]

    adj_t = data.adj_t.cpu()
    adj_t = gcn_norm(adj_t, add_self_loops=False)

    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)
    similarity = torch.zeros((num_classes, num_classes), dtype=torch.float64)
    class_sum_sq_norm = torch.zeros(num_classes, dtype=torch.float64)
    class_row_sq_norm = torch.zeros(num_classes, dtype=torch.float64)
    count_column = class_counts_tensor.unsqueeze(1)
    feature_scale = math.sqrt(feature_dim)

    for start in range(0, feature_dim, chunk_dim):
        width = min(chunk_dim, feature_dim - start)
        x = torch.empty((num_nodes, width), dtype=torch.float32)
        x.normal_(mean=0.0, std=1.0, generator=generator)
        x.div_(feature_scale)

        if hops > 0:
            x_i = x
            h = torch.zeros_like(x)
            for _ in range(hops):
                x_i = matmul(adj_t, x_i, reduce='add')
                h.add_(x_i)
            features = h.div_(float(hops))
        else:
            features = x

        features64 = features.to(torch.float64)
        class_sums = torch.zeros((num_classes, width), dtype=torch.float64)
        class_sums.index_add_(0, y, features64)
        class_means = class_sums / count_column
        similarity += class_means @ class_means.T

        class_sum_sq_norm += class_sums.square().sum(dim=1)
        row_sq = features64.square().sum(dim=1)
        class_row_sq_norm.index_add_(0, y, row_sq)

    matrix = similarity.cpu().numpy()
    diagonal = np.empty(num_classes, dtype=np.float64)
    diagonal.fill(np.nan)
    counts_np = class_counts_tensor.cpu().numpy()
    sum_sq_np = class_sum_sq_norm.cpu().numpy()
    row_sq_np = class_row_sq_norm.cpu().numpy()
    valid_diag = counts_np >= 2
    diagonal[valid_diag] = (sum_sq_np[valid_diag] - row_sq_np[valid_diag]) / (
        counts_np[valid_diag] * (counts_np[valid_diag] - 1.0)
    )
    np.fill_diagonal(matrix, diagonal)

    s_in_bal = float(np.nanmean(diagonal)) if np.any(valid_diag) else float('nan')
    offdiag_mask = ~np.eye(num_classes, dtype=bool)
    s_out_bal = float(np.nanmean(matrix[offdiag_mask])) if num_classes > 1 else float('nan')
    return matrix, class_counts, num_nodes, num_classes, s_in_bal, s_out_bal


def save_matrix_csv(matrix: np.ndarray, output_path: Path) -> None:
    with output_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        header = ['class'] + [str(idx) for idx in range(matrix.shape[1])]
        writer.writerow(header)
        for idx, row in enumerate(matrix):
            serialized = []
            for value in row:
                if np.isfinite(value):
                    serialized.append(f'{float(value):.10f}')
                else:
                    serialized.append('nan')
            writer.writerow([idx, *serialized])


def build_class_local_color_intensity(matrix: np.ndarray) -> np.ndarray:
    num_classes = matrix.shape[0]
    row_normalized = np.full(matrix.shape, np.nan, dtype=np.float64)

    for row in range(num_classes):
        values = matrix[row]
        finite_mask = np.isfinite(values)
        if not np.any(finite_mask):
            continue
        finite_values = values[finite_mask]
        row_min = float(np.min(finite_values))
        row_max = float(np.max(finite_values))
        if row_max > row_min:
            row_normalized[row, finite_mask] = (finite_values - row_min) / (row_max - row_min)
        else:
            row_normalized[row, finite_mask] = 0.5

    # Symmetric merge: a cell becomes dark if it is locally prominent for either endpoint class.
    return np.fmax(row_normalized, row_normalized.T)


def plot_heatmap(
    *,
    matrix: np.ndarray,
    dataset_name: str,
    hops: int,
    feature_dim: int,
    seed: int,
    class_counts: list[int],
    s_in_bal: float,
    s_out_bal: float,
    output_path: Path,
) -> None:
    num_classes = matrix.shape[0]
    base_size = max(8.0, min(24.0, 4.0 + num_classes * 0.42))
    fig, ax = plt.subplots(figsize=(base_size, base_size), dpi=200)

    color_intensity = build_class_local_color_intensity(matrix)
    masked = np.ma.masked_invalid(color_intensity)
    cmap = plt.get_cmap('Blues').copy()
    cmap.set_bad(color='#d9d9d9')
    image = ax.imshow(masked, cmap=cmap, aspect='equal', vmin=0.0, vmax=1.0)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label('Class-local normalized intensity', fontsize=10)

    tick_labels = [str(idx) for idx in range(num_classes)]
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(tick_labels, fontsize=max(5, 12 - num_classes * 0.15), rotation=90)
    ax.set_yticklabels(tick_labels, fontsize=max(5, 12 - num_classes * 0.15))
    ax.set_xlabel('Class ID')
    ax.set_ylabel('Class ID')

    title = (
        f'{dataset_name} | random_normal dim={feature_dim} | HOA hops={hops} | seed={seed}\n'
        f'diag: mean over i<j within class | offdiag: mean over cross-class pairs\n'
        f'color: class-local scale (symmetric max merge) | S_in^bal={format_value(s_in_bal)} | S_out^bal={format_value(s_out_bal)}'
    )
    ax.set_title(title, fontsize=max(9, 15 - num_classes * 0.1))

    text_size = max(3.0, 10.0 - num_classes * 0.12)
    for row in range(num_classes):
        for col in range(num_classes):
            value = float(matrix[row, col])
            intensity = float(color_intensity[row, col]) if np.isfinite(color_intensity[row, col]) else float('nan')
            if not np.isfinite(intensity):
                text_color = 'black'
            else:
                text_color = 'white' if intensity > 0.55 else 'black'
            ax.text(
                col,
                row,
                format_value(value),
                ha='center',
                va='center',
                color=text_color,
                fontsize=text_size,
            )

    ax.set_xticks(np.arange(-0.5, num_classes, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, num_classes, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.3)
    ax.tick_params(which='minor', bottom=False, left=False)

    summary_lines = [
        'Class counts:',
        ', '.join(f'{idx}:{count}' for idx, count in enumerate(class_counts)),
    ]
    fig.text(0.5, 0.01, '\n'.join(summary_lines), ha='center', va='bottom', fontsize=max(7, 12 - num_classes * 0.08))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

def main() -> int:
    parser = argparse.ArgumentParser(description='Compute random-feature HOA class-similarity heatmaps.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--feature-dim', type=int, default=DEFAULT_FEATURE_DIM)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--chunk-dim', type=int, default=DEFAULT_CHUNK_DIM)
    args = parser.parse_args()

    if args.feature_dim <= 0:
        raise ValueError('--feature-dim must be > 0')
    if args.chunk_dim <= 0:
        raise ValueError('--chunk-dim must be > 0')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []

    for dataset_name, hops in DATASET_SPECS:
        start_time = time.perf_counter()
        print(f'[start] dataset={dataset_name} hops={hops}')
        matrix, class_counts, num_nodes, num_classes, s_in_bal, s_out_bal = compute_class_similarity_matrix(
            dataset_name=dataset_name,
            hops=hops,
            feature_dim=args.feature_dim,
            seed=args.seed,
            chunk_dim=args.chunk_dim,
        )
        stem = sanitize_name(dataset_name)
        png_path = args.output_dir / f'{stem}_hoa_similarity.png'
        csv_path = args.output_dir / f'{stem}_hoa_similarity.csv'
        plot_heatmap(
            matrix=matrix,
            dataset_name=dataset_name,
            hops=hops,
            feature_dim=args.feature_dim,
            seed=args.seed,
            class_counts=class_counts,
            s_in_bal=s_in_bal,
            s_out_bal=s_out_bal,
            output_path=png_path,
        )
        save_matrix_csv(matrix, csv_path)
        elapsed = time.perf_counter() - start_time
        print(f'[done] dataset={dataset_name} num_nodes={num_nodes} num_classes={num_classes} elapsed_sec={elapsed:.2f}')
        summary_rows.append(
            {
                'dataset': dataset_name,
                'hops': hops,
                'num_nodes': num_nodes,
                'num_classes': num_classes,
                'feature_dim': args.feature_dim,
                'seed': args.seed,
                'chunk_dim': args.chunk_dim,
                'diag_metric': 'mean_{i<j,y_i=y_j=c}<h_i,h_j>',
                "offdiag_metric": "mean_{y_i=c,y_j=c'}<h_i,h_j>",
                's_in_bal': f'{s_in_bal:.10f}' if np.isfinite(s_in_bal) else 'nan',
                's_out_bal': f'{s_out_bal:.10f}' if np.isfinite(s_out_bal) else 'nan',
                'png_path': str(png_path),
                'csv_path': str(csv_path),
                'elapsed_sec': round(elapsed, 4),
            }
        )

    summary_path = args.output_dir / 'summary.csv'
    with summary_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f'[summary] {summary_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
