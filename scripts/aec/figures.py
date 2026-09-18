from __future__ import annotations

import csv, json, sys, tempfile
from pathlib import Path
from .paths import REFERENCE_ROOT, WORK_ROOT

FIGURE_TABLES={1:"figure1_plot_data.csv",3:"figure3_plot_data.csv",4:"figure4_plot_data.csv",5:"figure5_plot_data.csv",6:"figure6_plot_data.csv",7:"figure7_plot_data.csv"}

def _rows(path):
    with Path(path).open(newline="",encoding="utf-8") as h: return list(csv.DictReader(h))

def _import_draw(name):
    repo=Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path: sys.path.insert(0,str(repo))
    return __import__(f"draw_figure.{name}",fromlist=[name])

def _source_table(figure_id, mode):
    if mode.startswith("fixed"):
        p=WORK_ROOT/"fixed_runs"/f"figure{figure_id}"/"plot_data.csv"
        if p.is_file(): return p
    if mode.startswith("search"):
        p=WORK_ROOT/"search"/f"figure{figure_id}"/mode.split("_",1)[-1]/"plot_data.csv"
        if p.is_file(): return p
    return REFERENCE_ROOT/FIGURE_TABLES[figure_id]

def _show_png(path):
    data=Path(path).read_bytes()
    try:
        from IPython.display import Image, display
        display(Image(data=data, format="png"))
    except Exception:
        print(f"Generated PNG in memory: {len(data)} bytes")

def render_reference(figure_id:int, *, mode="reference", destination=None):
    with tempfile.TemporaryDirectory(prefix=f"aec-figure{figure_id}-") as tmp:
        tmp_path=Path(tmp)
        if figure_id in (2,8):
            import subprocess
            script=Path(__file__).resolve().parents[2]/"draw_figure"/f"draw_figure{figure_id}.py"
            subprocess.run([sys.executable,str(script),"--output-dir",str(tmp_path)],check=True)
            png=next(tmp_path.glob("*.png"))
        else:
            module=_import_draw(f"draw_figure{figure_id}"); rows=_rows(_source_table(figure_id,mode))
            if figure_id==1: module.plot_panels(rows,tmp_path)
            elif figure_id==3: module.validate_plot_rows(rows); module.plot_panels(rows,tmp_path)
            elif figure_id==4: module.plot_curves(rows,_load_metrics()["cora_featfree"],tmp_path)
            elif figure_id==5: module.plot_curves(rows,_load_metrics()["cora_featfree"],tmp_path)
            elif figure_id==6: module.validate_plot_rows(rows); module.save_figure(rows,tmp_path)
            elif figure_id==7: module.plot_curves(rows,_load_metrics()["flickr_featfree"],tmp_path)
            png=next(tmp_path.glob("*.png"))
        _show_png(png)
    result={"figure_id":figure_id,"mode":mode,"displayed":True,"source_table":FIGURE_TABLES.get(figure_id)}
    print(json.dumps(result,indent=2)); return result

def _load_metrics():
    import yaml
    with (REFERENCE_ROOT/"reference_metrics.yaml").open(encoding="utf-8") as h: return {k:float(v) for k,v in yaml.safe_load(h).items()}
