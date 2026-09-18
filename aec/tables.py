from __future__ import annotations

import csv, json, math
from collections import defaultdict
from pathlib import Path
from .paths import REFERENCE_ROOT, OUTPUT_ROOT

def _read(table, mode=None):
    candidate = REFERENCE_ROOT/f"{table}_{mode}_seed_rows.csv" if mode and mode.startswith(("fixed","search")) else REFERENCE_ROOT/f"{table}_seed_rows.csv"
    if not candidate.is_file(): candidate = REFERENCE_ROOT/f"{table}_seed_rows.csv"
    with candidate.open(newline="",encoding="utf-8") as h: return list(csv.DictReader(h))

def _num(values):
    xs=[float(v) for v in values]
    mean=sum(xs)/len(xs); std=(sum((x-mean)**2 for x in xs)/(len(xs)-1))**0.5 if len(xs)>1 else 0.0
    return mean,std

def summarize(table, mode=None):
    rows=_read(table, mode); groups=defaultdict(list)
    for r in rows:
        key=(r["setting"],r["backbone"],r["dataset"],r.get("feature_dim",""))
        groups[key].append(r["test_acc"])
    out=[]
    for (setting,backbone,dataset,feature_dim), vals in sorted(groups.items()):
        mean,std=_num(vals); out.append({"table":table,"setting":setting,"backbone":backbone,"dataset":dataset,"feature_dim":feature_dim,"n":len(vals),"test_acc_mean":mean,"test_acc_std":std})
    return out

def render_table(table, *, mode="reference", destination=None):
    destination=Path(destination or OUTPUT_ROOT)/table
    destination.mkdir(parents=True,exist_ok=True)
    summary=summarize(table, mode)
    fields=list(summary[0]) if summary else []
    with (destination/f"{table}.csv").open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(summary)
    if table=="table4":
        datasets=["cora","lastfm","citeseer","facebook"]
        lines=["| Backbone | Setting | "+" | ".join(datasets)+" |","|---|---|"+"---|"*len(datasets)]
        seen=set()
        for r in summary:
            key=(r["backbone"],r["setting"])
            if key in seen or r["feature_dim"]: continue
            seen.add(key); cells=[]
            for d in datasets:
                q=next(x for x in summary if x["backbone"]==r["backbone"] and x["setting"]==r["setting"] and x["dataset"]==d and not x["feature_dim"]); cells.append(f"{q['test_acc_mean']:.2f} ({q['test_acc_std']:.2f})")
            lines.append(f"| {r['backbone'].upper()} | {r['setting']} | "+" | ".join(cells)+" |")
    else:
        lines=["| Backbone | Setting | Feature dim | Dataset | Test accuracy |","|---|---|---:|---|---:|"]
        for r in summary: lines.append(f"| {r['backbone'].upper()} | {r['setting']} | {r['feature_dim']} | {r['dataset']} | {r['test_acc_mean']:.2f} ({r['test_acc_std']:.2f}) |")
    (destination/f"{table}.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    tex=[]
    if table=="table4":
        tex += [r"\begin{tabular}{llrrrr}", r"\toprule", "Backbone & Setting & Cora & LastFM & CiteSeer & Facebook \\", r"\midrule"]
        for r in summary:
            if r["feature_dim"]: continue
            if r["dataset"]!="cora": continue
            vals=[next(x for x in summary if x["backbone"]==r["backbone"] and x["setting"]==r["setting"] and x["dataset"]==d and not x["feature_dim"]) for d in ["cora","lastfm","citeseer","facebook"]]
            tex.append(f"{r['backbone'].upper()} & {r['setting']} & " + " & ".join(f"{x['test_acc_mean']:.2f} ({x['test_acc_std']:.2f})" for x in vals) + r" \\")
    else:
        tex += [r"\begin{tabular}{lllrr}", r"\toprule", "Backbone & Setting & Feature dim & Dataset & Test accuracy \\", r"\midrule"]
        for r in summary: tex.append(f"{r['backbone'].upper()} & {r['setting']} & {r['feature_dim']} & {r['dataset']} & {r['test_acc_mean']:.2f} ({r['test_acc_std']:.2f}) \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (destination/f"{table}.tex").write_text("\n".join(tex)+"\n",encoding="utf-8")
    payload={"table":table,"mode":mode,"rows":len(summary),"seed_rows":len(_read(table, mode))}
    (destination/"run_manifest.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2)); return payload
