from __future__ import annotations

import json, os, subprocess, sys
from pathlib import Path
from typing import Any
import yaml
from .paths import REFERENCE_ROOT, FIXED_ROOT, WORK_ROOT, parse_gpu_ids, max_parallel_per_gpu

REPO_ROOT=Path(__file__).resolve().parents[1]

def fixed_points(figure_id: int) -> list[dict[str,Any]]:
    path=FIXED_ROOT/f"figure{figure_id}.yaml"
    if not path.is_file(): return []
    data=yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return list(data.get("points", data.get("rows", [])))

def _coverage(figure_id, points):
    out=[]
    for point in points:
        fp=point.get("fixed_params", point)
        if figure_id==1 and str(fp.get("dataset","")).lower() not in {"cora","facebook"}: continue
        if figure_id==6 and (str(fp.get("dataset","")).lower() not in {"actor","flickr"} or str(fp.get("backbone","")).lower()!="sage"): continue
        if not fp.get("mechanism") or str(fp.get("feature","")) not in {"raw","random_normal","operator"}: continue
        out.append(point)
    return out

def plan_fixed_jobs(figure_id: int, *, repeats:int=3, limit:int|None=None) -> list[dict[str,object]]:
    jobs=[]
    for point in _coverage(figure_id,fixed_points(figure_id)):
        fp=point.get("fixed_params",point); cand=point.get("candidate",{})
        for repeat in range(1,repeats+1):
            jobs.append({"figure_id":figure_id,"point_id":point.get("point_id"),"dataset":fp.get("dataset"),"backbone":fp.get("backbone"),"mechanism":fp.get("mechanism"),"x_eps":fp.get("x_eps"),"repeat":repeat,"seed":12345+repeat-1,"candidate":cand})
    return jobs if limit is None else jobs[:limit]

def _bool(value): return str(value).lower() in {"true","1","yes"}

def _command(point, seed, out):
    fp=point.get("fixed_params",point); defaults=point.get("defaults",{}) or {}; model=defaults.get("model",{}); trainer=defaults.get("trainer",{}); data=defaults.get("dataset",{}); cand=point.get("candidate",{})
    args=[sys.executable,str(REPO_ROOT/"main.py"),"--dataset",str(fp["dataset"]),"--feature",str(fp.get("feature","raw")),"--mechanism",str(fp.get("mechanism","mbm")),"--x_eps",str(fp.get("x_eps","inf")),"--m",str(fp.get("m","best")),"--model",str(fp.get("backbone","sage")),"--hidden_dim",str(model.get("hidden_dim",16)),"--optimizer",str(trainer.get("optimizer","adam")),"--device","cuda","--val_ratio",str(data.get("val_ratio",.25)),"--test_ratio",str(data.get("test_ratio",.25)),"--data_range",*map(str,data.get("data_range",[0.,1.])),"--norm",str(bool(fp.get("norm",False))).lower(),"--gradient_clip",str(bool(trainer.get("gradient_clip",False))).lower(),"--gradient_clip_max_norm",str(trainer.get("gradient_clip_max_norm",1.0)),"--sim_epoch_refresh",str(bool(trainer.get("sim_epoch_refresh",False))).lower(),"--show_progress","false","--log_every_epoch","false","--x_steps",str(cand.get("x_steps",0)),"--learning_rate",str(cand.get("learning_rate",.001)),"--weight_decay",str(cand.get("weight_decay",0.)),"--dropout",str(cand.get("dropout",.5)),"--max_epochs",str((defaults.get("stage",{}).get("verify",{}) or {}).get("max_epochs",500)),"--patience",str((defaults.get("stage",{}).get("verify",{}) or {}).get("patience",150)),"-s",str(seed),"-r","1","-o",str(out)]
    if fp.get("smoother") not in (None,"","none"): args += ["--smoother",str(fp["smoother"])]
    if _bool(fp.get("use_nfr",False)) and cand.get("tao2") not in (None,"none",""): args += ["--use_nfr","true","--tao2",str(cand["tao2"])]
    return args

def run_fixed(figure_id:int, *, repeats:int=3, limit:int|None=None, execute:bool=False)->list[dict[str,object]]:
    points=_coverage(figure_id,fixed_points(figure_id)); point_map={p.get("point_id"):p for p in points}; jobs=plan_fixed_jobs(figure_id,repeats=repeats,limit=limit); root=WORK_ROOT/"fixed_runs"/f"figure{figure_id}"; root.mkdir(parents=True,exist_ok=True)
    manifest=root/"run_manifest.jsonl"
    with manifest.open("w",encoding="utf-8") as handle:
        for index,job in enumerate(jobs):
            point=point_map[job["point_id"]]; out=root/f"job_{index:05d}"; cmd=_command(point,int(job["seed"]),out); job={**job,"command":cmd,"output":str(out)}
            handle.write(json.dumps(job)+"\n")
            if execute: subprocess.run(cmd,cwd=str(REPO_ROOT),check=True)
    print(f"planned {len(jobs)} fixed jobs for figure {figure_id}; execute={execute}")
    return jobs

def run_fixed_table(table:str, *, repeats:int=3, execute:bool=False)->dict[str,object]:
    rows=[]; path=REFERENCE_ROOT/f"{table}_seed_rows.csv"
    import csv
    with path.open(newline="",encoding="utf-8") as h: raw=list(csv.DictReader(h))
    seen=set()
    for r in raw:
        key=(r["setting"],r["dataset"],r["backbone"],r.get("feature_dim",""))
        if key not in seen: seen.add(key); rows.append({"table":table,"setting":r["setting"],"dataset":r["dataset"],"backbone":r["backbone"],"feature_dim":r.get("feature_dim",""),"repeats":repeats})
    out=WORK_ROOT/"fixed_runs"/table; out.mkdir(parents=True,exist_ok=True); (out/"run_plan.json").write_text(json.dumps({"table":table,"jobs":rows,"execute":execute},indent=2)); print(f"planned {len(rows)} {table} groups")
    return {"table":table,"jobs":rows}
