from __future__ import annotations

import csv, math
from collections import defaultdict
from pathlib import Path
from .paths import REFERENCE_ROOT

def _read(path):
    with Path(path).open(newline="",encoding="utf-8") as h: return list(csv.DictReader(h))

def _stats(rows):
    vals=[float(r["test_acc"]) for r in rows]; mean=sum(vals)/len(vals); std=(sum((x-mean)**2 for x in vals)/(len(vals)-1))**0.5 if len(vals)>1 else 0.; return mean,std

def _display(markdown):
    try:
        from IPython.display import Markdown, display
        display(Markdown(markdown))
    except Exception: print(markdown)

def table4_summary():
    ff=_read(REFERENCE_ROOT/"table4_seed_rows.csv"); groups=defaultdict(list)
    for r in ff: groups[(r["backbone"],r["dataset"])].append(r)
    plot=_read(REFERENCE_ROOT/"figure1_plot_data.csv"); best={}
    for r in plot:
        if not r.get("pipeline","").startswith("figure3_pipeline"): continue
        if str(r.get("x_eps")) not in {"10","10.0"}: continue
        key=(r["backbone"],r["dataset"]); val=float(r.get("val_acc_mean",-1))
        if key not in best or val>best[key][0]: best[key]=(val,r)
    rows=[]; datasets=["cora","lastfm","citeseer","facebook"]
    for backbone in ["gcn","sage","gat"]:
        for setting in ["Best-LDP","gain","FeatFree-P"]:
            cells=[]
            for dataset in datasets:
                ffmean,ffstd=_stats(groups[(backbone,dataset)]); br=best[(backbone,dataset)][1]; bmean=float(br["test_acc_mean"]); bstd=float(br["test_acc_std"]);
                cells.append((bmean,bstd) if setting=="Best-LDP" else ((ffmean-bmean, math.sqrt(max(0,ffstd**2+bstd**2))) if setting=="gain" else (ffmean,ffstd)))
            rows.append((backbone.upper(),setting,cells))
    return rows

def table6_summary():
    raw=_read(REFERENCE_ROOT/"table6_seed_rows.csv"); grouped=defaultdict(list)
    for r in raw: grouped[(r["setting"],r["backbone"],r["dataset"],r["feature_dim"])].append(r)
    selected={}; datasets=["cora","lastfm","citeseer","facebook"]
    for key,items in grouped.items():
        val=sum(float(r["val_acc"]) for r in items)/len(items);
        if key[:3] not in selected or val>selected[key[:3]][0]: selected[key[:3]]=(val,key,items)
    rows=[]
    for backbone in ["gcn","sage","gat"]:
        for setting in ["FeatFree-Kprop","FeatFree-HOA"]:
            label="Kprop" if setting.endswith("Kprop") else "HOA"; cells=[]
            for dataset in datasets:
                _,key,items=selected[(setting,backbone,dataset)]; cells.append((*_stats(items),key[3]))
            rows.append((backbone.upper(),label,cells))
    return rows

def render_table(table, *, mode="reference", destination=None):
    datasets=["Cora","LastFM","CiteSeer","Facebook"]
    rows=table4_summary() if table=="table4" else table6_summary()
    lines=["| Backbone | Setting | "+" | ".join(datasets)+" |","|---|---|"+"---|"*4]
    for backbone,setting,cells in rows:
        if table=="table6": values=[f"{m:.2f} ({s:.2f}) [d={d}]" for m,s,d in cells]
        elif setting=="gain": values=[f"+{m:.2f}" if m>=0 else f"{m:.2f}" for m,_,*rest in cells]
        else: values=[f"{m:.2f} ({s:.2f})" for m,s,*rest in cells]
        lines.append(f"| {backbone} | {setting} | "+" | ".join(values)+" |")
    markdown="\n".join(lines)+"\n"; _display(markdown); return {"table":table,"mode":mode,"displayed":True,"rows":len(rows)}
