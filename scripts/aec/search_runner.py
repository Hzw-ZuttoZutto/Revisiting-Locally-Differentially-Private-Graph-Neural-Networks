from __future__ import annotations

import csv, json, math, subprocess, sys
from pathlib import Path
from typing import Any
import yaml
from .paths import parse_gpu_ids, max_parallel_per_gpu, WORK_ROOT

REPO_ROOT=Path(__file__).resolve().parents[2]

def _configs(figure_id: int) -> list[Path]:
    roots={1:REPO_ROOT/"configs_AEC/figure1",3:REPO_ROOT/"configs_AEC/figure3",4:REPO_ROOT/"configs_AEC/figure4",5:REPO_ROOT/"configs_AEC/figure5",6:REPO_ROOT/"configs_AEC/figure6",7:REPO_ROOT/"configs_AEC/figure7","table4":REPO_ROOT/"configs_AEC/table4","table6":REPO_ROOT/"configs_AEC/table6"}
    if figure_id == 5:
        return sorted(p for base in REPO_ROOT.glob("configs_AEC/figure5/*") for p in base.rglob("*.yaml"))
    return sorted(roots[figure_id].rglob("*.yaml")) if figure_id in roots else []

def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle: return yaml.safe_load(handle)

def scaled_config(path: Path, *, output: Path) -> Path:
    data=load_config(path)
    datasets=data.get("search_space",{}).get("dataset",{}).get("datasets",[])
    if "figure1" in str(path): data["search_space"]["dataset"]["datasets"]=[x for x in datasets if str(x).lower() in {"cora","facebook"}]
    elif "figure6" in str(path): data["search_space"]["dataset"]["datasets"]=[x for x in datasets if str(x).lower() in {"actor","flickr"}]
    data["device"]["gpu_ids"]=parse_gpu_ids(); data["device"]["max_parallel_per_gpu"]=max_parallel_per_gpu()
    data.setdefault("defaults",{}).setdefault("stage",{}).setdefault("verify",{})["repeats"]=3
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(yaml.safe_dump(data,sort_keys=False),encoding="utf-8"); return output

def run_search(config: Path, *, mode: str="scaled", output_root: Path|None=None, dry_run: bool=True) -> dict[str,object]:
    output_root=output_root or WORK_ROOT/"search"/config.stem/mode; actual=config
    if mode=="scaled": actual=scaled_config(config,output=WORK_ROOT/"scaled_configs"/config.name)
    command=[sys.executable,"-u","-m","hparams_search_scripts.run_mechanism_hparam_search","--config",str(actual),"--output_root_dir",str(output_root)]
    result={"mode":mode,"config":str(actual),"output_root":str(output_root),"command":command,"dry_run":dry_run}
    if not dry_run: subprocess.run(command,check=True,cwd=str(REPO_ROOT))
    return result

def run_search_for_figure(figure_id: int|str, *, mode: str="scaled", execute: bool=False) -> dict[str,object]:
    if figure_id in (2,8): return {"figure_id":figure_id,"mode":"analytic","configs":0}
    paths=_configs(figure_id); selected=paths
    if mode=="scaled":
        # Keep one config per plotted claim family; all x-axis values stay in that config.
        selected=[]; seen=set()
        for p in paths:
            family=p.parts[-3] if len(p.parts)>=3 else p.name
            if family not in seen: selected.append(p); seen.add(family)
    results=[]
    for p in selected:
        rel=p.relative_to(REPO_ROOT).with_suffix("").as_posix().replace("/","__")
        results.append(run_search(p,mode=mode,output_root=WORK_ROOT/"search"/f"figure{figure_id}"/mode/rel,dry_run=not execute))
    if execute: _aggregate_search_output(figure_id, mode)
    return {"figure_id":figure_id,"mode":mode,"configs_total":len(paths),"configs_selected":len(selected),"runs":results}


def _aggregate_search_output(figure_id, mode):
    from .paths import REFERENCE_ROOT
    table= {1:"figure1_plot_data.csv",3:"figure3_plot_data.csv",4:"figure4_plot_data.csv",5:"figure5_plot_data.csv",6:"figure6_plot_data.csv",7:"figure7_plot_data.csv"}.get(figure_id)
    if not table: return
    ref=REFERENCE_ROOT/table
    if not ref.is_file(): return
    with ref.open(newline="",encoding="utf-8") as handle: rows=list(csv.DictReader(handle))
    manifests=[]
    root=WORK_ROOT/"search"/f"figure{figure_id}"/mode
    for path in root.rglob("manifest.csv"):
        with path.open(newline="",encoding="utf-8") as handle: manifests.extend(csv.DictReader(handle))
    def eq(row,a,b): return str(row.get(a,"" )).lower()==str(b).lower()
    for out in rows:
        candidates=[]
        for m in manifests:
            if out.get("dataset") and not eq(m,"dataset",out["dataset"]): continue
            if out.get("backbone") and not eq(m,"backbone",out["backbone"]): continue
            if out.get("x_eps") and not eq(m,"x_eps",out["x_eps"]): continue
            if out.get("mechanism") and not eq(m,"mechanism",out["mechanism"]): continue
            if out.get("smoother") and not eq(m,"smoother",out["smoother"]): continue
            if out.get("norm") and not eq(m,"norm",out["norm"]): continue
            if out.get("norm_scale") and not eq(m,"norm_scale",out["norm_scale"]): continue
            if m.get("best_verify_test_acc_mean","")=="": continue
            candidates.append(m)
        if not candidates: continue
        m=candidates[0]
        mean=float(m["best_verify_test_acc_mean"]); std=float(m.get("best_verify_test_acc_std") or 0.0); n=int(m.get("verify_done") or 1); half=1.96*std/math.sqrt(max(n,1))
        out["test_acc_mean"]=f"{mean:.12g}"; out["test_acc_std"]=f"{std:.12g}"; out["test_acc_ci_low"]=f"{mean-half:.12g}"; out["test_acc_ci_high"]=f"{mean+half:.12g}"; out["val_acc_mean"]=m.get("best_verify_val_acc_mean",out.get("val_acc_mean","")); out["n"]=str(n)
    root.mkdir(parents=True,exist_ok=True); output=root/"plot_data.csv"
    with output.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
