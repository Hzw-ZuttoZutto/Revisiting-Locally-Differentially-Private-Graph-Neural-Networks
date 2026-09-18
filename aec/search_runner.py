from __future__ import annotations

import json, subprocess, sys
from pathlib import Path
from typing import Any
import yaml
from .paths import parse_gpu_ids, max_parallel_per_gpu, WORK_ROOT

REPO_ROOT=Path(__file__).resolve().parents[1]

def _configs(figure_id: int) -> list[Path]:
    roots={1:REPO_ROOT/"configs_AEC/figure1",3:REPO_ROOT/"configs_AEC/figure3",4:REPO_ROOT/"configs_AEC/figure4",5:REPO_ROOT/"configs_AEC/figure6",6:REPO_ROOT/"configs_AEC/figure6",7:REPO_ROOT/"configs_AEC/figure7","table4":REPO_ROOT/"configs_AEC/table4","table6":REPO_ROOT/"configs_AEC/table6"}
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
    results=[run_search(p,mode=mode,output_root=WORK_ROOT/"search"/f"figure{figure_id}"/mode,dry_run=not execute) for p in selected]
    return {"figure_id":figure_id,"mode":mode,"configs_total":len(paths),"configs_selected":len(selected),"runs":results}
