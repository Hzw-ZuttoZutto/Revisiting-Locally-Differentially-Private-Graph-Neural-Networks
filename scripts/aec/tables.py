from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from .paths import REFERENCE_ROOT, normalize_mode, result_root

def _read(path):
    with Path(path).open(newline="",encoding="utf-8") as h: return list(csv.DictReader(h))

def _table_seed_rows(table, mode):
    mode = normalize_mode(mode)
    if mode != "reference":
        candidate = result_root(table, mode) / f"{table}_seed_rows.csv"
        if candidate.is_file(): return _read(candidate)
        raise FileNotFoundError(f"No generated table data for {table} mode={mode}; run the experiment first")
    return _read(REFERENCE_ROOT / f"{table}_seed_rows.csv")

def _stats(rows):
    vals=[float(r["test_acc"]) for r in rows]; mean=sum(vals)/len(vals); std=(sum((x-mean)**2 for x in vals)/(len(vals)-1))**0.5 if len(vals)>1 else 0.; return mean,std

def _display(markdown):
    try:
        from IPython.display import Markdown, display
        display(Markdown(markdown))
    except Exception: print(markdown)

def table4_summary(mode="reference"):
    from .search_results import table_datasets

    mode = normalize_mode(mode)
    ff=_table_seed_rows("table4",mode); groups=defaultdict(list)
    for r in ff: groups[(r["setting"],r["backbone"],r["dataset"])].append(r)
    datasets = table_datasets("table4", mode)
    if mode in {"scaled", "full"}:
        return [
            (backbone.upper(), setting, [_stats(groups[(setting, backbone, dataset)]) for dataset in datasets])
            for backbone in ("gcn", "sage", "gat") for setting in ("Best-LDP", "FeatFree-P")
        ]
    plot=_read(REFERENCE_ROOT/"figure1_plot_data.csv"); best={}
    for r in plot:
        if not r.get("pipeline","").startswith("figure3_pipeline"): continue
        if str(r.get("x_eps")) not in {"10","10.0"}: continue
        key=(r["backbone"],r["dataset"]); val=float(r.get("val_acc_mean",-1))
        if key not in best or val>best[key][0]: best[key]=(val,r)
    rows=[]; datasets=["cora","lastfm","citeseer","facebook"]
    for backbone in ["gcn","sage","gat"]:
        for setting in ["Best-LDP","FeatFree-P"]:
            cells=[]
            for dataset in datasets:
                ffmean,ffstd=_stats(groups[("FeatFree-P",backbone,dataset)]); br=best[(backbone,dataset)][1]; bmean=float(br["test_acc_mean"]); bstd=float(br["test_acc_std"]);
                cells.append((bmean,bstd) if setting=="Best-LDP" else (ffmean,ffstd))
            rows.append((backbone.upper(),setting,cells))
    return rows

def table6_summary(mode="reference"):
    raw=_table_seed_rows("table6",mode); grouped=defaultdict(list)
    for r in raw: grouped[(r["setting"],r["backbone"],r["dataset"],r["feature_dim"])].append(r)
    selected={}; datasets=["cora","lastfm","citeseer","facebook"]
    for key,items in sorted(grouped.items(), key=lambda item: (*item[0][:3], int(item[0][3]))):
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
    from .search_results import table_datasets, validate_search_output

    mode = normalize_mode(mode)
    if table not in {"table4", "table6"}:
        raise ValueError(f"Unsupported table: {table}")
    if mode in {"scaled", "full"}:
        validate_search_output(table, mode)
    labels = {"cora": "Cora", "lastfm": "LastFM", "citeseer": "CiteSeer", "facebook": "Facebook"}
    datasets=[labels[name] for name in table_datasets(table, mode)]
    rows=table4_summary(mode) if table=="table4" else table6_summary(mode)
    lines=["| Backbone | Setting | "+" | ".join(datasets)+" |","|---|---|"+"---|"*len(datasets)]
    for backbone,setting,cells in rows:
        if table=="table6": values=[f"{m:.2f} ± {s:.2f}" for m,s,d in cells]
        else: values=[f"{m:.2f} ± {s:.2f}" for m,s,*rest in cells]
        lines.append(f"| {backbone} | {setting} | "+" | ".join(values)+" |")
    markdown="\n".join(lines)+"\n"; _display(markdown); return {"table":table,"mode":mode,"displayed":True,"rows":len(rows)}
