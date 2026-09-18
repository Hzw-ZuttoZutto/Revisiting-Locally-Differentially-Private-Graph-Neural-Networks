from __future__ import annotations

import json, os, subprocess, sys
from pathlib import Path
from typing import Any
import yaml
from .paths import REFERENCE_ROOT, FIXED_ROOT, WORK_ROOT, parse_gpu_ids, max_parallel_per_gpu

REPO_ROOT=Path(__file__).resolve().parents[2]

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
    if execute: _collect_generated_plot_data(figure_id, jobs)
    print(f"planned {len(jobs)} fixed jobs for figure {figure_id}; execute={execute}")
    return jobs

def run_fixed_table(table:str, *, repeats:int=3, execute:bool=False)->dict[str,object]:
    path=FIXED_ROOT/f"{table}.yaml"; data=yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {"points":[]}
    points=list(data.get("points",[])); out=WORK_ROOT/"fixed_runs"/table; out.mkdir(parents=True,exist_ok=True)
    manifest=out/"run_manifest.jsonl"; planned=[]
    with manifest.open("w",encoding="utf-8") as handle:
        for index, point in enumerate(points):
            for repeat in range(1,repeats+1):
                seed=12345+repeat-1; job={"table":table,"point_id":point.get("point_id"),"setting":point.get("setting"),"dataset":point.get("fixed_params",{}).get("dataset"),"backbone":point.get("fixed_params",{}).get("backbone"),"feature_dim":point.get("fixed_params",{}).get("feature_dim"),"repeat":repeat,"seed":seed}
                out_dir=out/f"job_{index:04d}_repeat_{repeat:02d}"; job["output"]=str(out_dir); job["command"]=_command(point,seed,out_dir); handle.write(json.dumps(job)+"\n"); planned.append(job)
                if execute: subprocess.run(job["command"],cwd=str(REPO_ROOT),check=True)
    if execute: _collect_table_outputs(table, planned)
    print(f"planned {len(planned)} {table} jobs; execute={execute}"); return {"table":table,"jobs":planned}

def _collect_table_outputs(table, planned):
    import csv
    rows=[]
    for job in planned:
        files=sorted(Path(job["output"]).rglob("*.csv"));
        if not files: continue
        with files[-1].open(newline="",encoding="utf-8") as handle: records=list(csv.DictReader(handle))
        if not records: continue
        rec=records[-1]; rows.append({"table":table,"setting":job["setting"],"dataset":job["dataset"],"backbone":job["backbone"],"feature_dim":job.get("feature_dim") or "","seed":job["seed"],"val_acc":rec.get("val/acc",""),"test_acc":rec.get("test/acc","")})
    if rows:
        path=REFERENCE_ROOT/f"{table}_fixed_seed_rows.csv"
        with path.open("w",newline="",encoding="utf-8") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def _collect_generated_plot_data(figure_id, jobs):
    import csv, math
    ref_path=REFERENCE_ROOT/f"figure{figure_id}_plot_data.csv"
    if not ref_path.is_file(): return
    with ref_path.open(newline="",encoding="utf-8") as handle: reference=list(csv.DictReader(handle))
    values={}
    for job in jobs:
        point_id=job.get("point_id")
        if not point_id: continue
        pattern=f"job_*_repeat_{int(job['repeat']):02d}/**/*.csv"
        for path in sorted((WORK_ROOT/"fixed_runs"/f"figure{figure_id}").glob(pattern)):
            with path.open(newline="",encoding="utf-8") as handle: rows=list(csv.DictReader(handle))
            if not rows: continue
            row=max(rows,key=lambda r: float(r.get("epoch",0) or 0))
            try: values.setdefault(point_id,[]).append((float(row["val/acc"]),float(row["test/acc"])))
            except (KeyError,ValueError): continue
    for row in reference:
        samples=values.get(row.get("point_id"),[])
        if len(samples)<1: continue
        vals=[x[1] for x in samples]; val=[x[0] for x in samples]; mean=sum(vals)/len(vals); std=(sum((x-mean)**2 for x in vals)/(len(vals)-1))**0.5 if len(vals)>1 else 0.0; half=1.96*std/math.sqrt(len(vals)) if vals else 0.0
        row["test_acc_mean"]=f"{mean:.12g}"; row["test_acc_std"]=f"{std:.12g}"; row["test_acc_ci_low"]=f"{mean-half:.12g}"; row["test_acc_ci_high"]=f"{mean+half:.12g}"; row["test_acc_min"]=f"{min(vals):.12g}"; row["test_acc_max"]=f"{max(vals):.12g}"; row["val_acc_mean"]=f"{sum(val)/len(val):.12g}"; row["n"]=str(len(vals))
    out=WORK_ROOT/"fixed_runs"/f"figure{figure_id}"/"plot_data.csv"
    with out.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(reference[0])); writer.writeheader(); writer.writerows(reference)
