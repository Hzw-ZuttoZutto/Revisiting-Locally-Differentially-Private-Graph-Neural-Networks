from __future__ import annotations

import csv
from pathlib import Path
import yaml
from .paths import REPO_ROOT, REFERENCE_ROOT

FIXED_ROOT = REPO_ROOT / "scripts" / "aec" / "fixed_hparams"
FORBIDDEN_DATA_FIELDS = {
    "source_job",
    "source_config",
    "source_job_dir",
    "source_job_id",
    "source_manifest",
    "reference_repeats",
    "bootstrap_key",
    "bootstrap_val_key",
    "actual_verify_rank",
    "ci_method",
}
FORBIDDEN_DATA_TERMS = (
    "historical",
    "paper_experiments",
    "rebuttal_experiments",
    "runtime://",
    "artifact://",
)

def validate_reference() -> list[str]:
    errors=[]
    required=["figure1_plot_data.csv","figure3_plot_data.csv","figure4_plot_data.csv","figure5_plot_data.csv","figure6_plot_data.csv","figure7_plot_data.csv","figure1_points.csv","figure6_points.csv","figure7_points.csv","table4_seed_rows.csv","table6_seed_rows.csv","reference_manifest.yaml"]
    for name in required:
        if not (REFERENCE_ROOT/name).is_file(): errors.append(f"missing {name}")
    for path in REFERENCE_ROOT.glob("*.csv"):
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        if not rows: errors.append(f"empty {path.name}")
        forbidden = FORBIDDEN_DATA_FIELDS.intersection(reader.fieldnames or [])
        if forbidden:
            errors.append(f"internal fields {sorted(forbidden)} in {path.name}")

    for path in [*FIXED_ROOT.glob("*.yaml"), *REFERENCE_ROOT.glob("*.yaml")]:
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        for token in FORBIDDEN_DATA_TERMS:
            if token in lowered:
                errors.append(f"internal history token {token!r} in {path.relative_to(REPO_ROOT)}")

    for path in FIXED_ROOT.glob("*.yaml"):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for point in data.get("points", []):
            point_fields = set(point)
            candidate_fields = set((point.get("candidate") or {}).keys())
            forbidden = FORBIDDEN_DATA_FIELDS.intersection(point_fields | candidate_fields)
            if forbidden:
                errors.append(f"internal fields {sorted(forbidden)} in {path.name}")

    for name in ("scripts/prepare_reference_data.py", "scripts/prepare_table_data.py"):
        if (REPO_ROOT / name).exists():
            errors.append(f"author-only export tool remains: {name}")
    forbidden=("/" + "data/hzw","/" + "home/hzw")
    for path in (REPO_ROOT/"scripts"/"aec").rglob("*.py"):
        text=path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text: errors.append(f"absolute path {token} in {path}")
    return errors

def main() -> None:
    errors=validate_reference()
    if errors:
        raise SystemExit("\n".join(errors))
    print("AEC artifact validation passed")

if __name__ == "__main__": main()
