from __future__ import annotations

import json, tempfile
from pathlib import Path
import yaml
from .paths import REFERENCE_ROOT, FIXED_ROOT, WORK_ROOT
from .search_runner import run_search, _aggregate_search_output

REPO_ROOT=Path(__file__).resolve().parents[2]

def fixed_points(figure_id:int):
    path=FIXED_ROOT/f"figure{figure_id}.yaml"
    if not path.is_file(): return []
    return list((yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("points",[]))

def _coverage(figure_id, points):
    result=[]
    for point in points:
        fp=point.get("fixed_params",point); dataset=str(fp.get("dataset","")).lower(); backbone=str(fp.get("backbone","")).lower()
        if figure_id==1 and dataset not in {"cora","facebook"}: continue
        if figure_id==6 and (dataset not in {"actor","flickr"} or backbone!="sage"): continue
        if not fp.get("mechanism"): continue
        result.append(point)
    return result

def plan_fixed_jobs(figure_id:int, *, repeats:int=3, limit:int|None=None):
    points=_coverage(figure_id,fixed_points(figure_id))
    jobs=[{"figure_id":figure_id,"point_id":p.get("point_id"),"dataset":p.get("fixed_params",{}).get("dataset"),"backbone":p.get("fixed_params",{}).get("backbone"),"mechanism":p.get("fixed_params",{}).get("mechanism"),"x_eps":p.get("fixed_params",{}).get("x_eps"),"repeats":repeats} for p in points]
    return jobs if limit is None else jobs[:limit]

def _one_candidate_config(point, repeats:int, gpu_ids=None, max_parallel=None):
    fp=point.get("fixed_params",point); cand=point.get("candidate",{}); defaults=point.get("defaults",{}) or {}
    feature=str(fp.get("feature","raw")); feature_cfg={k:[] for k in ("sim_reference_eps","feature_dim","scale","feature_preprojection","preprojection_output_dim","random_normal_mean","random_normal_std","shared_value","degree_bucket_num_buckets","degree_bucket_range_max","deepwalk_walk_length","deepwalk_number_walks","deepwalk_window_size","deepwalk_workers","deepwalk_undirected")}
    feature_cfg["features"]=[feature]; feature_cfg["scale"]=[fp.get("scale",1)]
    for k in feature_cfg:
        if k not in {"features","scale"} and fp.get(k) is not None: feature_cfg[k]=[fp[k]]
    if feature=="sim": feature_cfg["sim_reference_eps"]=[fp.get("sim_reference_eps")]
    perturb={"mechanisms":[fp.get("mechanism")],"x_eps":[fp.get("x_eps")],"m":[fp.get("m","best")]}
    norm=bool(fp.get("norm",False)); cal={"norm":[norm],"norm_scale":[fp.get("norm_scale","none")] if norm else [],"x_steps":[cand.get("x_steps",0)],"smoother":[fp.get("smoother")] if fp.get("smoother") not in (None,"none","") else ["hoa"]}
    nfr=bool(fp.get("use_nfr",False)); nfr_cfg={"use_nfr":[nfr],"tao2":[cand.get("tao2")] if nfr else []}
    device={"device":"gpu","cpu_worker_count":None,"gpu_ids":gpu_ids or [0],"max_parallel_per_gpu":max_parallel or 1,"gpu_launch_interval_sec":0.1}
    clean_defaults=dict(defaults)
    clean_stage=dict(clean_defaults.get("stage",{}))
    clean_grid=dict(clean_stage.get("grid",{})); clean_verify=dict(clean_stage.get("verify",{}))
    clean_grid.pop("max_epochs",None); clean_verify.pop("max_epochs",None)
    clean_grid["repeats"]=1; clean_verify["repeats"]=repeats
    clean_stage["grid"]=clean_grid; clean_stage["verify"]=clean_verify; clean_defaults["stage"]=clean_stage
    return {"seed":12345,"device":device,"defaults":clean_defaults,"search_space":{"dataset":{"datasets":[fp.get("dataset")]},"feature_transformation":feature_cfg,"feature_perturbation":perturb,"calibrator":cal,"model":{"backbones":[fp.get("backbone")],"dropout":[cand.get("dropout",0.5)]},"trainer":{"learning_rate":[cand.get("learning_rate",0.001)],"weight_decay":[cand.get("weight_decay",0.0)]},"nfr":nfr_cfg}}

def run_fixed(figure_id:int, *, repeats:int=3, limit:int|None=None, execute:bool=False):
    points=_coverage(figure_id,fixed_points(figure_id)); jobs=plan_fixed_jobs(figure_id,repeats=repeats,limit=limit); root=WORK_ROOT/"search"/f"figure{figure_id}"/"fixed"
    root.mkdir(parents=True,exist_ok=True)
    for idx,job in enumerate(jobs):
        point=next(p for p in points if p.get("point_id")==job["point_id"]); config=root/f"point_{idx:05d}.yaml"; config.write_text(yaml.safe_dump(_one_candidate_config(point,repeats),sort_keys=False),encoding="utf-8")
        if execute: run_search(config,mode="full",output_root=root/f"result_{idx:05d}",dry_run=False)
    if execute: _aggregate_search_output(figure_id,"fixed")
    print(f"planned {len(jobs)} fixed YAML jobs for figure {figure_id}; execute={execute}")
    return jobs

def run_fixed_table(table:str, *, repeats:int=3, execute:bool=False):
    path=FIXED_ROOT/f"{table}.yaml"; points=list((yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("points",[])) if path.is_file() else []
    root=WORK_ROOT/"search"/table/"fixed"; root.mkdir(parents=True,exist_ok=True); jobs=[]
    for idx,point in enumerate(points):
        config=root/f"point_{idx:05d}.yaml"; config.write_text(yaml.safe_dump(_one_candidate_config(point,repeats),sort_keys=False),encoding="utf-8"); jobs.append({"table":table,"point_id":point.get("point_id"),"config":str(config)})
        if execute: run_search(config,mode="full",output_root=root/f"result_{idx:05d}",dry_run=False)
    print(f"planned {len(jobs)} {table} YAML jobs; execute={execute}"); return {"table":table,"jobs":jobs}


def _collect_table_outputs(table, points, root):
    import csv, re
    point_by_id={p.get("point_id"):p for p in points}; rows=[]
    for result in sorted(root.glob("result_*")):
        manifests=sorted(result.rglob("manifest.csv"))
        if not manifests: continue
        with manifests[0].open(newline="",encoding="utf-8") as handle: manifest=next(csv.DictReader(handle),None)
        if not manifest: continue
        job=Path(manifest["job_dir"]); best_path=job/"best_config.yaml"
        if not best_path.is_file(): continue
        best=yaml.safe_load(best_path.read_text(encoding="utf-8")); cid=int(best.get("best_candidate",{}).get("candidate_id",0)); point_id=None
        for p in points:
            fp=p.get("fixed_params",{})
            if fp.get("dataset")==manifest.get("dataset") and fp.get("backbone")==manifest.get("backbone") and str(fp.get("feature_dim",""))==str(manifest.get("feature_dim","")):
                point_id=p.get("point_id"); break
        if point_id is None: continue
        point=point_by_id[point_id]; fp=point.get("fixed_params",{}); setting=point.get("setting")
        for child in sorted((job/"verify_top5").glob("rank=*__repeat=*")):
            match=re.match(r"rank=(\\d+)__repeat=(\\d+)__candidate=(\\d+)__",child.name)
            if not match or int(match.group(3)) != cid: continue
            files=sorted(child.glob("*.csv"));
            if len(files)!=1: continue
            with files[0].open(newline="",encoding="utf-8") as handle: rec=next(csv.DictReader(handle),None)
            if not rec: continue
            rows.append({"table":table,"setting":setting,"dataset":str(fp.get("dataset")).replace("flickr","flickr"),"backbone":fp.get("backbone",""),"feature_dim":fp.get("feature_dim","") or "","seed":rec.get("seed",""),"val_acc":rec.get("val/acc",""),"test_acc":rec.get("test/acc","")})
    if rows:
        out=root/f"{table}_seed_rows.csv"
        with out.open("w",newline="",encoding="utf-8") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
