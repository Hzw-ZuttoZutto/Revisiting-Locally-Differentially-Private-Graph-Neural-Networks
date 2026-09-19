from __future__ import annotations

import csv, json, math, re, subprocess, sys
from pathlib import Path
from typing import Any
import yaml
from .paths import FIXED_ROOT, REFERENCE_ROOT, parse_gpu_ids, max_parallel_per_gpu, WORK_ROOT

REPO_ROOT=Path(__file__).resolve().parents[2]

PIPELINE_AXES = {
    "figure3_pipeline1": {"mechanism": "mbm", "smoother": "kprop", "use_nfr": "false"},
    "figure3_pipeline2": {"mechanism": "hds", "smoother": "kprop", "use_nfr": "false"},
    "figure3_pipeline3": {"mechanism": "mbm", "smoother": "hoa", "use_nfr": "true"},
    "figure3_pipeline4": {"mechanism": "pm", "smoother": "hoa", "use_nfr": "true"},
}


def _fixed_point_id_by_result_index(figure_id: int) -> dict[int, str]:
    path = FIXED_ROOT / f"figure{figure_id}.yaml"
    if not path.is_file():
        return {}
    points = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("points", [])
    selected = []
    for point in points:
        fixed = point.get("fixed_params", point)
        dataset = str(fixed.get("dataset", "")).lower()
        backbone = str(fixed.get("backbone", "")).lower()
        if figure_id == 1 and dataset not in {"cora", "facebook"}:
            continue
        if figure_id == 6 and (dataset not in {"actor", "flickr"} or backbone != "sage"):
            continue
        if fixed.get("mechanism"):
            selected.append(str(point.get("point_id", "")))
    return {index: point_id for index, point_id in enumerate(selected) if point_id}


def _result_index(job_dir: str) -> int | None:
    matches = re.findall(r"(?:^|/)result_(\d+)(?:/|$)", str(job_dir).replace("\\", "/"))
    return int(matches[-1]) if matches else None

def _configs(figure_id: int) -> list[Path]:
    roots={1:REPO_ROOT/"configs_AEC/figure1",3:REPO_ROOT/"configs_AEC/figure3",4:REPO_ROOT/"configs_AEC/figure4",5:REPO_ROOT/"configs_AEC/figure5",6:REPO_ROOT/"configs_AEC/figure6",7:REPO_ROOT/"configs_AEC/figure7","table4":REPO_ROOT/"configs_AEC/table4","table6":REPO_ROOT/"configs_AEC/table6"}
    if figure_id == 5:
        return sorted(p for base in REPO_ROOT.glob("configs_AEC/figure5/*") for p in base.rglob("*.yaml"))
    return sorted(roots[figure_id].rglob("*.yaml")) if figure_id in roots else []

def _claim_paths(figure_id):
    root=REPO_ROOT
    if figure_id==1:
        paths=[]
        paths += sorted((root/"configs_AEC/figure1/FeatFree/cora").rglob("*.yaml"))
        paths += sorted((root/"configs_AEC/figure1/FeatFree/facebook").rglob("*.yaml"))
        paths += sorted((root/"configs_AEC/figure1/LDPGNN").rglob("*.yaml"))
        paths += sorted((root/"configs_AEC/figure1/Non-private").rglob("*.yaml"))
        return paths
    if figure_id==6:
        paths=[]
        paths += sorted((root/"configs_AEC/figure6/FeatFree/actor/sage").rglob("*.yaml"))
        paths += sorted((root/"configs_AEC/figure6/FeatFree/flickr/sage").rglob("*.yaml"))
        paths += sorted((root/"configs_AEC/figure6/LDPGNN/sage").rglob("*.yaml"))
        paths += sorted((root/"configs_AEC/figure6/Non-private").rglob("*.yaml"))
        return paths
    if figure_id==5: return _configs(5)
    return _configs(figure_id)

def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle: return yaml.safe_load(handle)

def scaled_config(path: Path, *, output: Path) -> Path:
    data=load_config(path)
    datasets=data.get("search_space",{}).get("dataset",{}).get("datasets",[])
    if "figure1" in str(path): data["search_space"]["dataset"]["datasets"]=[x for x in datasets if str(x).lower() in {"cora","facebook"}]
    elif "configs_AEC/figure6" in str(path): data["search_space"]["dataset"]["datasets"]=[x for x in datasets if str(x).lower() in {"actor","flickr"}]
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

def _select_paths(figure_id, paths, mode):
    if mode == "full":
        return paths
    selected=[]
    for path in paths:
        text=path.as_posix().lower()
        if figure_id == 1:
            if "ldpgnn" in text or ("featfree" in text and any(f"/{d}/" in text for d in ("cora","facebook"))) or "non-private" in text:
                selected.append(path)
        elif figure_id == 3:
            if path.name in {"LDP.yaml","SIM.yaml"}: selected.append(path)
        elif figure_id == 5:
            selected.append(path)
        elif figure_id == 6:
            if "ldpgnn/sage" in text or ("featfree" in text and "/sage/" in text) or "non-private" in text:
                selected.append(path)
        elif figure_id in {4,7,"table4","table6"}:
            selected.append(path)
    return selected or paths


def run_search_for_figure(figure_id: int|str, *, mode: str="scaled", execute: bool=False) -> dict[str,object]:
    if figure_id in (2,8): return {"figure_id":figure_id,"mode":"analytic","configs":0}
    paths=_configs(figure_id); selected=_select_paths(figure_id, paths, mode)
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
    root=WORK_ROOT/"search"/f"figure{figure_id}"/mode
    batch_manifest = root/"manifest.csv"
    manifests=[]
    if batch_manifest.is_file():
        with batch_manifest.open(newline="",encoding="utf-8") as handle:
            manifests.extend(csv.DictReader(handle))
    else:
        for path in root.rglob("manifest.csv"):
            with path.open(newline="",encoding="utf-8") as handle:
                manifests.extend(csv.DictReader(handle))
    def eq(row,a,b): return str(row.get(a,"" )).lower()==str(b).lower()
    def apply_manifest(out, manifest):
        mean=float(manifest["best_verify_test_acc_mean"])
        std=float(manifest.get("best_verify_test_acc_std") or 0.0)
        n=int(manifest.get("verify_done") or 1)
        half=1.96*std/math.sqrt(max(n,1))
        out["test_acc_mean"]=f"{mean:.12g}"
        out["test_acc_std"]=f"{std:.12g}"
        out["test_acc_ci_low"]=f"{mean-half:.12g}"
        out["test_acc_ci_high"]=f"{mean+half:.12g}"
        out["val_acc_mean"]=manifest.get("best_verify_val_acc_mean", out.get("val_acc_mean", ""))
        out["n"]=str(n)

    fixed_point_map = _fixed_point_id_by_result_index(figure_id) if mode == "fixed" else {}
    fixed_manifests = {}
    if fixed_point_map:
        for manifest in manifests:
            point_id = fixed_point_map.get(_result_index(manifest.get("job_dir", "")))
            if point_id:
                fixed_manifests[point_id] = manifest

    missing = []
    for out in rows:
        pipeline = out.get("pipeline", "")
        expected_axes = PIPELINE_AXES.get(pipeline)

        if fixed_point_map:
            # Fixed runs are bound to the point index used to create result_N,
            # never to display axes such as epsilon or norm_scale. Baseline
            # lines intentionally remain the frozen reference values.
            if figure_id in {1, 6} and expected_axes is None:
                continue
            point_id = out.get("point_id", "")
            if point_id not in fixed_point_map.values():
                continue
            manifest = fixed_manifests.get(point_id)
            if not manifest or manifest.get("best_verify_test_acc_mean", "") == "":
                missing.append({"point_id": point_id, "pipeline": pipeline})
                continue
            apply_manifest(out, manifest)
            continue

        # Figure 1 and Figure 6 contain fixed baseline lines whose x-axis is
        # privacy budget for presentation only. They must remain frozen and
        # must never be replaced by a privacy-method manifest.
        if figure_id in {1, 6} and expected_axes is None:
            continue
        candidates=[]
        for m in manifests:
            if out.get("dataset") and not eq(m,"dataset",out["dataset"]): continue
            if out.get("backbone") and not eq(m,"backbone",out["backbone"]): continue
            if out.get("x_eps") and not eq(m,"x_eps",out["x_eps"]): continue
            if out.get("mechanism") and not eq(m,"mechanism",out["mechanism"]): continue
            if out.get("smoother") and not eq(m,"smoother",out["smoother"]): continue
            if out.get("norm") and not eq(m,"norm",out["norm"]): continue
            if out.get("norm_scale") and not eq(m,"norm_scale",out["norm_scale"]): continue
            if expected_axes:
                if any(not eq(m, key, value) for key, value in expected_axes.items()): continue
            if m.get("best_verify_test_acc_mean","")=="": continue
            candidates.append(m)
        if not candidates:
            missing.append({
                "source": out.get("source", ""),
                "pipeline": out.get("pipeline", ""),
                "dataset": out.get("dataset", ""),
                "backbone": out.get("backbone", ""),
                "x_eps": out.get("x_eps", ""),
            })
            continue
        apply_manifest(out, candidates[0])
    if missing:
        sample = "; ".join(str(item) for item in missing[:5])
        raise RuntimeError(
            f"Missing completed fixed/search results for {len(missing)} plot rows; sample: {sample}"
        )
    root.mkdir(parents=True,exist_ok=True); output=root/"plot_data.csv"
    with output.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
