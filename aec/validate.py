from __future__ import annotations

import csv
import os
from pathlib import Path
from .paths import REPO_ROOT, REFERENCE_ROOT

def validate_reference() -> list[str]:
    errors=[]
    required=["figure1_plot_data.csv","figure3_plot_data.csv","figure4_plot_data.csv","figure5_plot_data.csv","figure6_plot_data.csv","figure7_plot_data.csv","figure1_points.csv","figure6_points.csv","figure7_points.csv","table4_seed_rows.csv","table6_seed_rows.csv","reference_manifest.yaml"]
    for name in required:
        if not (REFERENCE_ROOT/name).is_file(): errors.append(f"missing {name}")
    for path in REFERENCE_ROOT.glob("*.csv"):
        with path.open(newline="") as handle:
            rows=list(csv.DictReader(handle))
        if not rows: errors.append(f"empty {path.name}")
    forbidden=("/" + "data/hzw","/" + "home/hzw")
    for path in (REPO_ROOT/"aec").rglob("*.py"):
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
