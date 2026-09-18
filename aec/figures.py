from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

from .paths import REFERENCE_ROOT, WORK_ROOT, output_dir

FIGURE_TABLES = {1:"figure1_plot_data.csv", 3:"figure3_plot_data.csv", 4:"figure4_plot_data.csv", 5:"figure5_plot_data.csv", 6:"figure6_plot_data.csv", 7:"figure7_plot_data.csv"}

def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))

def _import_draw(name: str):
    if str(Path(__file__).resolve().parents[1]) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    return __import__(f"draw_figure.{name}", fromlist=[name])

def _source_table(figure_id: int, mode: str) -> Path:
    if mode.startswith("fixed"):
        candidate = WORK_ROOT / "fixed_runs" / f"figure{figure_id}" / "plot_data.csv"
        if candidate.is_file(): return candidate
    if mode.startswith("search"):
        candidate = WORK_ROOT / "search" / f"figure{figure_id}" / mode.split("_", 1)[-1] / "plot_data.csv"
        if candidate.is_file(): return candidate
    return REFERENCE_ROOT / FIGURE_TABLES[figure_id]

def render_reference(figure_id: int, *, mode: str = "reference", destination: Path | None = None) -> dict[str, object]:
    destination = destination or output_dir(mode, figure_id)
    destination.mkdir(parents=True, exist_ok=True)
    if figure_id in (2, 8):
        import subprocess
        script = Path(__file__).resolve().parents[1] / "draw_figure" / f"draw_figure{figure_id}.py"
        subprocess.run([sys.executable, str(script), "--output-dir", str(destination)], check=True)
    else:
        module = _import_draw(f"draw_figure{figure_id}")
        source_table = _source_table(figure_id, mode)
        rows = _rows(source_table)
        if figure_id == 1:
            module.plot_panels(rows, destination)
        elif figure_id == 3:
            module.validate_plot_rows(rows)
            module.plot_panels(rows, destination)
        elif figure_id == 4:
            metrics = _load_metrics()
            module.plot_curves(rows, metrics["cora_featfree"], destination)
        elif figure_id == 5:
            metrics = _load_metrics()
            module.plot_curves(rows, metrics["cora_featfree"], destination)
        elif figure_id == 6:
            module.validate_plot_rows(rows)
            pdf_path, png_path = module.save_figure(rows, destination)
        elif figure_id == 7:
            metrics = _load_metrics()
            module.plot_curves(rows, metrics["flickr_featfree"], destination)
        source = _source_table(figure_id, mode)
        (destination / source.name).write_bytes(source.read_bytes())
    manifest = {"figure_id": figure_id, "mode": mode, "output_dir": str(destination), "reference_table": str(_source_table(figure_id, mode).name) if figure_id in FIGURE_TABLES else None}
    (destination / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest

def _load_metrics() -> dict[str, float]:
    import yaml
    with (REFERENCE_ROOT / "reference_metrics.yaml").open(encoding="utf-8") as handle:
        return {key: float(value) for key, value in yaml.safe_load(handle).items()}
