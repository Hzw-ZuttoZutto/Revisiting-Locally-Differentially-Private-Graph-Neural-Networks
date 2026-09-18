#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, re, yaml
from pathlib import Path

def read_csv(path):
    with path.open(newline="", encoding="utf-8") as h: return list(csv.DictReader(h))

def extract(manifest, setting, table, points=None):
    rows=[]
    for m in read_csv(manifest):
        job=Path(m["job_dir"])
        rank=int(m.get("best_rank") or 1); cid=int(m.get("best_candidate_id") or 0)
        if points is not None:
            spec=yaml.safe_load((job/"job_spec.yaml").read_text(encoding="utf-8")); best=yaml.safe_load((job/"best_config.yaml").read_text(encoding="utf-8")); points.append({"point_id": table + "_" + m["dataset"] + "_" + m["backbone"] + "_" + str(m.get("feature_dim", "")), "table": table, "setting": setting, "fixed_params": spec.get("fixed_params",{}), "candidate": best.get("best_candidate",{}), "defaults": spec.get("defaults",{})})
        selected=[]
        for child in sorted((job/"verify_top5").glob(f"rank={rank:02d}__repeat=*")):
            match=re.match(r"rank=(\d+)__repeat=(\d+)__candidate=(\d+)__",child.name)
            if not match: continue
            files=sorted(child.glob("*.csv"))
            if len(files)==1:
                r=read_csv(files[0]);
                if len(r)==1: selected.append(r[0])
        selected.sort(key=lambda r:int(r.get("seed",0)))
        if len(selected) < 10: raise RuntimeError(f"{job}: expected 10 selected seeds, found {len(selected)}")
        for r in selected[:10]:
            rows.append({
                "table":table,"setting":setting,"dataset":m["dataset"].replace("attributedgraph-flickr","flickr"),
                "backbone":m["backbone"].lower(),"feature_dim":m.get("feature_dim",""),
                "smoother":m.get("smoother",""),"seed":r.get("seed",""),
                "val_acc":r.get("val/acc",""),"test_acc":r.get("test/acc",""),
            })
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--experiments-root",type=Path,required=True); args=ap.parse_args(); root=args.experiments_root
    out=Path(__file__).resolve().parents[1]/"scripts/aec/reference"; out.mkdir(parents=True,exist_ok=True)
    t4=[]; p4=[]
    for p in sorted((root/"table4").glob("table4_FeatFree-P/*/direct.yaml/manifest.csv")):
        t4.extend(extract(p,"FeatFree-P","table4",p4))
    t6=[]; p6=[]
    for p in sorted((root/"table6").rglob("random_projected.yaml/manifest.csv")):
        setting=next(part for part in p.parts if part.startswith("table6_FeatFree-")).replace("table6_","")
        t6.extend(extract(p,setting,"table6",p6))
    for name, rows, points in (("table4",t4,p4),("table6",t6,p6)):
        (out.parent/"fixed_hparams"/f"{name}.yaml").write_text(yaml.safe_dump({"schema_version":2,"table":name,"points":points},sort_keys=False),encoding="utf-8")
        fields=["table","setting","dataset","backbone","feature_dim","smoother","seed","val_acc","test_acc"]
        with (out/f"{name}_seed_rows.csv").open("w",newline="",encoding="utf-8") as h:
            w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(rows)
        print(name,len(rows))
if __name__=="__main__": main()
