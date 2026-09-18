#!/usr/bin/env python3
"""Author-only export. Reads historical results; never trains or modifies them."""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    args = parser.parse_args()
    runtime = args.runtime_root.resolve()
    paper = runtime / "paper_experiments"
    rebuttal = runtime / "rebuttal_experiments"
    ref = REPO / "scripts/aec/reference"
    fixed = REPO / "scripts/aec/fixed_hparams"
    ref.mkdir(parents=True, exist_ok=True)
    fixed.mkdir(parents=True, exist_ok=True)
    from draw_figure import draw_figure7 as f7
    from hparams_search_scripts import mechanism_stage_utils as u

    from draw_figure import draw_figure6 as f6
    heter_main = f6.load_main_rows(rebuttal / "figure1_heter", bootstrap_samples=1000, bootstrap_seed=12345)
    heter_featfree = f6.load_featfree_rows(rebuttal / "featfree_heter" / "HOA", bootstrap_samples=1000, bootstrap_seed=12345)
    heter_clean = f6.load_clean_rows(rebuttal / "clean_heter", bootstrap_samples=1000, bootstrap_seed=12345)
    tables = {
        1: read_csv(paper / "main_add_10seed_backfill/plots/main_add_10seed_panel_plot_data.csv"),
        3: read_csv(paper / "figure4_final/plots/figure4_final_panel_plot_data.csv"),
        4: read_csv(paper / "figure5_final/plots/figure5_final_norm_scale_plot_data.csv"),
        5: read_csv(paper / "figure6/plots/figure6_tao2_plot_data.csv"),
        6: heter_main + heter_featfree + heter_clean,
        7: f7.build_plot_rows(f7.load_records(rebuttal / "figure5_heter/figure5.yaml"),
                            bootstrap_samples=1000, bootstrap_seed=12345),
    }
    long_index = defaultdict(list)
    for r in read_csv(paper / "main_add_10seed_backfill/test_acc_long.csv"):
        long_index[(r['dataset'], r['backbone'], r['pipeline'], float(r['x_eps']))].append(r)

    def resolve_source(value):
        value = str(value).replace("runtime://", str(runtime) + "/").replace("artifact://", str(REPO) + "/")
        return Path(value).resolve()

    def provenance(path):
        path = Path(path).resolve()
        for base, prefix in [(runtime, "historical"), (REPO, "repository")]:
            if path.is_relative_to(base):
                return f"{prefix}/{path.relative_to(base).as_posix()}"
        raise ValueError(f"Unrecognized provenance root: {path}")

    def portable(value):
        if isinstance(value, dict):
            return {k: portable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [portable(v) for v in value]
        if isinstance(value, Path) or isinstance(value, str) and value.startswith('/'):
            return provenance(value)
        if value == 'attributedgraph-flickr':
            return 'flickr'
        return value

    all_points = {}
    for fig in (1, 6, 3, 4, 5, 7):
        points = {}
        samples = {}
        plot_rows = []
        for template in tables[fig]:
            template = dict(template)
            pipeline = template.get('pipeline', '')
            if fig == 1 and pipeline.startswith('figure3_pipeline'):
                raw = long_index[(template['dataset'], template['backbone'], pipeline, float(template['x_eps']))]
                if len(raw) != 10:
                    raise ValueError(('main seeds', template, len(raw)))
                job = resolve_source(raw[0]['source_job_dir'])
                source_records = [read_csv(resolve_source(r['result_csv_path']))[0] for r in raw]
                metrics = [(int(r['seed']), float(r['val_acc']), float(r['test_acc'])) for r in raw]
            else:
                job = resolve_source(template['source_job_dir'])
                candidate_id = int(template['candidate_id'])
                matched = []
                for child in (job / 'verify_top5').iterdir():
                    m = re.match(r'rank=(\d+)__repeat=(\d+)__candidate=(\d+)__', child.name)
                    if m and int(m[3]) == candidate_id:
                        files = sorted(child.glob('*.csv'))
                        if len(files) != 1:
                            raise ValueError(('ambiguous verify CSV', child))
                        matched.append((int(m[2]), read_csv(files[0])[0]))
                matched.sort(key=lambda x: x[0])
                expected = 20 if fig == 3 else 10
                if [x[0] for x in matched] != list(range(1, expected + 1)):
                    raise ValueError(('incomplete selected candidate', job, candidate_id))
                source_records = [x[1] for x in matched]
                metrics = [(int(r['seed']), float(r['val/acc']), float(r['test/acc'])) for r in source_records]
            old_spec = yaml.safe_load((job / 'job_spec.yaml').read_text())
            params = {k: v for k, v in old_spec['fixed_params'].items() if k in u.OUTER_AXIS_NAMES}
            params['dataset'] = 'flickr' if params['dataset'] == 'attributedgraph-flickr' else params['dataset']
            defaults = copy.deepcopy(old_spec['defaults'])
            defaults.pop('diagnostics', None)
            defaults['trainer'].pop('collect_grad_stats', None)
            selected = {k: template.get(k, template.get('candidate_tao2', 'none')) for k in u.CANDIDATE_AXIS_NAMES}
            selected['candidate_id'] = int(template['candidate_id'])
            selected['x_steps'] = int(selected['x_steps'])
            for k in ('learning_rate', 'weight_decay', 'dropout', 'tao2'):
                actual = source_records[0].get(k)
                if actual not in (None, ''):
                    selected[k] = actual
                if selected[k] is None:
                    selected[k] = 'none'
            # Preserve effective seeds from the backfill export, not its multi-repeat CSV header.
            signature = json.dumps([fig, params, selected], sort_keys=True)
            point_id = f'f{fig}_' + hashlib.sha256(signature.encode()).hexdigest()[:16]
            point = {
                'point_id': point_id, 'figure_id': fig,
                'fixed_params': params, 'candidate': selected, 'defaults': defaults,
                'base_seed': min(x[0] for x in metrics),
                'reference_repeats': len(metrics),
                'source_job_id': old_spec['job_id'], 'source_job': provenance(job),
                'source_config': provenance(job / 'job_spec.yaml'),
                'plot_rows': [],
            }
            if point_id not in points:
                points[point_id] = point
                samples[point_id] = metrics
            elif samples[point_id] != metrics:
                raise ValueError(('inconsistent duplicate baseline', point_id))
            # Record exact bootstrap grouping strings: canonical dataset spelling does not change the frozen CI.
            if fig == 1:
                if pipeline.startswith('figure3_pipeline'):
                    key = ['main', template['backbone'], pipeline, template['dataset'], template['x_eps']]
                    vkey = ['main-val', *key[1:]]
                elif pipeline == 'featfree':
                    key = ['reference', template['backbone'], template['dataset'], template['smoother'], int(template['feature_dim'])]
                    vkey = ['reference-val', *key[1:]]
                else:
                    key = vkey = []
            elif fig == 6:
                if pipeline.startswith('figure3_pipeline'):
                    key = [template['source'], template['backbone'], template['dataset'], pipeline, template['x_eps']]
                elif pipeline == 'featfree':
                    key = [template['source'], template['backbone'], template['dataset'], int(template['feature_dim'])]
                else:
                    key = [template['source'], template['backbone'], template['dataset']]
                vkey, key = ['val', *key], ['test', *key]
            elif fig == 3:
                key = [template['source'], template['mechanism'], template['x_eps']]
                vkey = key
            else:
                key = [str(template['epsilon']), str(template['tao2']) if fig == 5 else str(template['scale_exponent'])]
                vkey = key
            clean = portable(template)
            clean.update(point_id=point_id, bootstrap_key='|'.join(map(str, key)),
                         bootstrap_val_key='|'.join(map(str, vkey)))
            plot_rows.append(clean)
            points[point_id]['plot_rows'].append({k: v for k, v in clean.items()
                                                 if not k.startswith(('test_acc_', 'val_acc_'))})
            for name, index in [('test_acc_mean', 2), ('val_acc_mean', 1)]:
                if template.get(name) not in ('', None):
                    mean = sum(x[index] for x in metrics) / len(metrics)
                    if abs(mean - float(template[name])) > 1e-7:
                        raise ValueError(('plot/raw mean mismatch', fig, point_id, name, mean, template[name]))
        # Scalar FeatFree references must also be independently rerun, once per figure.
        if fig in (4, 5, 7):
            base_fig, dataset = (6, 'flickr') if fig == 7 else (1, 'cora')
            baseline = next(copy.deepcopy(p) for p in all_points[base_fig].values()
                            if p['fixed_params']['dataset'] == dataset and p['fixed_params']['backbone'] == 'sage'
                            and p['fixed_params']['feature'] == 'random_normal')
            original_id = baseline['point_id']
            baseline.update(point_id=f'f{fig}_featfree_baseline', figure_id=fig, plot_rows=[], scalar_baseline=True)
            points[baseline['point_id']] = baseline
            baseline_samples = read_csv(ref / f'figure{base_fig}_points.csv')
            samples[baseline['point_id']] = [(int(r['seed']), float(r['val_acc']), float(r['test_acc']))
                                           for r in baseline_samples if r['point_id'] == original_id]
        rows = []
        for pid, point in points.items():
            for seed, val, test in samples[pid]:
                rows.append(dict(point_id=pid, figure_id=fig, dataset=point['fixed_params']['dataset'],
                                 backbone=point['fixed_params']['backbone'], seed=seed, val_acc=val, test_acc=test,
                                 source_job_id=point['source_job_id']))
        write_csv(ref / f'figure{fig}_points.csv', rows)
        write_csv(ref / f'figure{fig}_plot_data.csv', plot_rows)
        (fixed / f'figure{fig}.yaml').write_text(yaml.safe_dump({'schema_version': 2, 'figure_id': fig,
            'points': list(points.values())}, sort_keys=False), encoding='utf-8')
        all_points[fig] = points
        print(f'Figure {fig}: {len(points)} unique training points, {len(rows)} seed rows, {len(plot_rows)} plotted rows', flush=True)
    # Remove only superseded files created by the initial AEC scaffold.
    for path in ref.glob('*_seed_rows.csv'):
        path.unlink()
    manifest = {'schema_version': 2, 'figures': {}, 'files': {}}
    for fig, points in all_points.items():
        manifest['figures'][fig] = {'points': len(points), 'seed_rows': sum(p['reference_repeats'] for p in points.values()),
                                    'repeats': sorted({p['reference_repeats'] for p in points.values()})}
    for path in sorted([*ref.glob('*.csv'), *fixed.glob('*.yaml')]):
        manifest['files'][path.relative_to(REPO).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    (ref / 'reference_manifest.yaml').write_text(yaml.safe_dump(manifest, sort_keys=False))


if __name__ == '__main__':
    main()
