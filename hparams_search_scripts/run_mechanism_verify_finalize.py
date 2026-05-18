#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

try:
    from hparams_search_scripts.mechanism_stage_context import build_recommended_command_parts, resolve_job_context
    from hparams_search_scripts import mechanism_stage_utils
except ModuleNotFoundError:
    from mechanism_stage_context import build_recommended_command_parts, resolve_job_context  # type: ignore
    import mechanism_stage_utils  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize verify outputs and emit best artifacts.")
    parser.add_argument("job_dir", type=str, help="job directory containing verify outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ctx = resolve_job_context(args.job_dir)
    job_dir = ctx.job_dir

    summary_rows = mechanism_stage_utils.aggregate_verify_results(ctx.job_spec, job_dir)
    mechanism_stage_utils.write_verify_summary(job_dir, summary_rows)
    winner = summary_rows[0]
    candidate = mechanism_stage_utils.job_candidate_by_id(ctx.job_spec, int(winner["candidate_id"]))
    recommended_parts = build_recommended_command_parts(ctx, candidate)
    recommended_command = mechanism_stage_utils.shell_join(recommended_parts)
    mechanism_stage_utils.recommended_command_path(job_dir).write_text(
        recommended_command + "\n",
        encoding="utf-8",
    )

    best_config = {
        "schema_version": mechanism_stage_utils.SCHEMA_VERSION,
        "job_id": ctx.job_spec["job_id"],
        "best_rank": 1,
        "best_candidate": candidate.to_dict(),
        "fixed_params": ctx.fixed_params,
        "defaults": ctx.defaults,
        "verify_metrics": {
            "val_acc": {
                "mean": float(winner["val_acc_mean"]),
                "std": float(winner["val_acc_std"]),
                "min": float(winner["val_acc_min"]),
                "max": float(winner["val_acc_max"]),
                "n": int(winner["n"]),
            },
            "test_acc": {
                "mean": float(winner["test_acc_mean"]),
                "std": float(winner["test_acc_std"]),
                "min": float(winner["test_acc_min"]),
                "max": float(winner["test_acc_max"]),
                "n": int(winner["n"]),
            },
        },
        "artifacts": {
            "verify_summary_csv": str(mechanism_stage_utils.verify_summary_path(job_dir)),
            "recommended_command_txt": str(mechanism_stage_utils.recommended_command_path(job_dir)),
        },
        "recommended_command": recommended_command,
    }
    if winner.get("sanity_e_pg_mean") not in (None, ""):
        best_config["verify_metrics"]["sanity_e_pg"] = {
            "mean": float(winner["sanity_e_pg_mean"]),
            "std": float(winner["sanity_e_pg_std"]),
            "min": float(winner["sanity_e_pg_min"]),
            "max": float(winner["sanity_e_pg_max"]),
            "n": int(winner["sanity_e_pg_n"]),
        }
    mechanism_stage_utils.write_yaml_file(
        mechanism_stage_utils.best_config_path(job_dir),
        best_config,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
