#!/usr/bin/env python3
import argparse
import csv
import os
import time
import uuid
from pathlib import Path

import torch
import torch.nn.functional as F

from datasets import load_dataset
from main import parse_cli_args, seed_everything
from models import NodeClassifier
from pre_smoothing_feature_cache import prepare_pre_smoothing_input
from trainer import Trainer
from utils import from_args


BACKBONES = {
    "gcn": {"candidate_id": 119, "x_steps": 8, "learning_rate": 0.01, "weight_decay": 0.01, "dropout": 0.5},
    "sage": {"candidate_id": 223, "x_steps": 64, "learning_rate": 0.01, "weight_decay": 0.001, "dropout": 0.25},
    "gat": {"candidate_id": 224, "x_steps": 64, "learning_rate": 0.01, "weight_decay": 0.001, "dropout": 0.5},
}
FIELDNAMES = [
    "backbone", "candidate_id", "repeat_id", "seed", "best_epoch", "epochs_ran",
    "test_n", "num_classes", "majority_class", "majority_class_test_count",
    "majority_baseline_accuracy_pct", "test_accuracy_pct", "predicted_majority_count",
    "predicted_majority_share_pct", "dominant_predicted_class", "dominant_predicted_count",
    "dominant_predicted_share_pct", "dominant_matches_majority", "mean_max_probability_pct",
    "correct_predictions", "elapsed_sec",
]


def build_model_args(backbone, seed):
    hp = BACKBONES[backbone]
    argv = [
        "--dataset", "actor",
        "--feature", "random_normal",
        "--mechanism", "mbm",
        "--x_eps", "inf",
        "--m", "best",
        "--model", backbone,
        "--hidden_dim", "16",
        "--optimizer", "adam",
        "--device", "cuda",
        "--val_ratio", "0.25",
        "--test_ratio", "0.25",
        "--data_range", "0.0", "1.0",
        "--feature_dim", "932",
        "--random_normal_mean", "0",
        "--random_normal_std", "1",
        "--smoother", "hoa",
        "--norm", "false",
        "--collect_grad_stats", "false",
        "--gradient_clip", "false",
        "--gradient_clip_max_norm", "1.0",
        "--sim_epoch_refresh", "false",
        "--show_progress", "false",
        "--log_every_epoch", "false",
        "--pre_smoothing_feature_cache_root", "/data/hzw/Rethinking_DP_GNN_runtime/cache/rebuttal_heter",
        "--x_steps", str(hp["x_steps"]),
        "--learning_rate", str(hp["learning_rate"]),
        "--weight_decay", str(hp["weight_decay"]),
        "--dropout", str(hp["dropout"]),
        "--max_epochs", "500",
        "--patience", "150",
        "--seed", str(seed),
        "--repeats", "1",
    ]
    _, args = parse_cli_args(argv)
    return args


def run_one(backbone, repeat_id, base_seed):
    started = time.time()
    seed = base_seed + repeat_id
    args = build_model_args(backbone, seed)
    seed_everything(seed)

    dataset = from_args(load_dataset, args)
    data = dataset.clone().to(args.device)
    data, _ = prepare_pre_smoothing_input(data, args, rewrite_seed=seed)
    input_dim = int(getattr(data, "operator_num_features", data.num_features))
    model = from_args(NodeClassifier, args, input_dim=input_dim, num_classes=data.num_classes)
    trainer = from_args(Trainer, args, logger=None)

    epoch_stats = []

    def validation_step_with_counts(eval_data, gnn_adj_t=None, smoother_adj_t=None):
        logits = model._forward_logits(eval_data, gnn_adj_t=gnn_adj_t, smoother_adj_t=smoother_adj_t)
        target = model._target_indices(eval_data.y)
        test_logits = logits[eval_data.test_mask]
        test_target = target[eval_data.test_mask]
        pred = test_logits.argmax(dim=1)
        true_counts = torch.bincount(test_target, minlength=eval_data.num_classes)
        pred_counts = torch.bincount(pred, minlength=eval_data.num_classes)
        majority_class = int(true_counts.argmax().item())
        dominant_predicted_class = int(pred_counts.argmax().item())
        correct = int((pred == test_target).sum().item())
        epoch_stats.append({
            "test_n": int(test_target.numel()),
            "num_classes": int(eval_data.num_classes),
            "majority_class": majority_class,
            "majority_class_test_count": int(true_counts[majority_class].item()),
            "predicted_majority_count": int(pred_counts[majority_class].item()),
            "dominant_predicted_class": dominant_predicted_class,
            "dominant_predicted_count": int(pred_counts[dominant_predicted_class].item()),
            "mean_max_probability_pct": float(test_logits.softmax(dim=1).max(dim=1).values.mean().item() * 100.0),
            "correct_predictions": correct,
        })
        return {
            "val/loss": F.cross_entropy(logits[eval_data.val_mask], target[eval_data.val_mask]),
            "val/acc": model.accuracy(pred=logits[eval_data.val_mask], target=target[eval_data.val_mask]) * 100,
            "test/acc": model.accuracy(pred=test_logits, target=test_target) * 100,
        }

    model.validation_step = validation_step_with_counts
    best_metrics = trainer.fit(model, data, diagnostic_dir=None)
    best_epoch = int(best_metrics["epoch"])
    stat = epoch_stats[best_epoch - 1]
    test_n = stat["test_n"]
    test_accuracy_pct = float(best_metrics["test/acc"].detach().item())
    counted_accuracy_pct = 100.0 * stat["correct_predictions"] / test_n
    if abs(test_accuracy_pct - counted_accuracy_pct) > 1e-5:
        raise RuntimeError(f"accuracy mismatch: metric={test_accuracy_pct}, counted={counted_accuracy_pct}")

    hp = BACKBONES[backbone]
    return {
        "backbone": backbone,
        "candidate_id": hp["candidate_id"],
        "repeat_id": repeat_id,
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_ran": len(epoch_stats),
        **stat,
        "majority_baseline_accuracy_pct": 100.0 * stat["majority_class_test_count"] / test_n,
        "test_accuracy_pct": counted_accuracy_pct,
        "predicted_majority_share_pct": 100.0 * stat["predicted_majority_count"] / test_n,
        "dominant_predicted_share_pct": 100.0 * stat["dominant_predicted_count"] / test_n,
        "dominant_matches_majority": int(stat["dominant_predicted_class"] == stat["majority_class"]),
        "elapsed_sec": time.time() - started,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, default=12345)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"shard_{args.shard_id}.csv"
    tasks = [(backbone, repeat_id) for backbone in BACKBONES for repeat_id in range(30)]
    tasks = [task for task_id, task in enumerate(tasks) if task_id % args.num_shards == args.shard_id]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for backbone, repeat_id in tasks:
            row = run_one(backbone, repeat_id, args.base_seed)
            writer.writerow(row)
            handle.flush()
            print(
                f"gpu_shard={args.shard_id} backbone={backbone} seed={row['seed']} "
                f"acc={row['test_accuracy_pct']:.4f} pred_majority={row['predicted_majority_share_pct']:.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
