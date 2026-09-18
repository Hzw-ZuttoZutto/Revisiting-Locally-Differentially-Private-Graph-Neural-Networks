#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_ROOT="$SCRIPT_DIR/configs_AEC/figure1/LDPGNN"
OUTPUT_ROOT="$SCRIPT_DIR/experiments_AEC/figure1/LDPGNN"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS='[0,2,3,4,6,7]'
MAX_PARALLEL_PER_GPU=10

usage() {
  echo "Usage: $0 [--list]"
  echo "  --list  Show missing jobs/verify slots without running experiments."
}

MODE=run
case "${1:-}" in
  "") ;;
  --list) MODE=list ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

if [[ ! -d "$CONFIG_ROOT" ]]; then
  echo "Config directory not found: $CONFIG_ROOT" >&2
  exit 1
fi
if [[ ! -d "$OUTPUT_ROOT" ]]; then
  echo "AEC output directory not found: $OUTPUT_ROOT" >&2
  exit 1
fi

mapfile -t CONFIGS < <(find "$CONFIG_ROOT" -type f -name '*.yaml' | sort)
if [[ "${#CONFIGS[@]}" -ne 12 ]]; then
  echo "Expected 12 Figure 1 LDPGNN configs, found ${#CONFIGS[@]}" >&2
  exit 1
fi

manifest_pending() {
  local manifest_path="$1"
  "$PYTHON_BIN" -B - "$manifest_path" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"Manifest not found: {path}")
pending_jobs = 0
pending_slots = 0
with path.open("r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        total = int(row.get("verify_total", "0") or 0)
        done = int(row.get("verify_done", "0") or 0)
        missing = max(0, total - done)
        if missing:
            pending_jobs += 1
            pending_slots += missing
print(pending_jobs, pending_slots)
PY
}

print_inventory() {
  local total_jobs=0
  local total_slots=0
  echo "Missing AEC verify inventory:"
  for config_path in "${CONFIGS[@]}"; do
    local relative="${config_path#"$CONFIG_ROOT/"}"
    local output_dir="$OUTPUT_ROOT/$relative"
    local pending_jobs pending_slots
    read -r pending_jobs pending_slots < <(manifest_pending "$output_dir/manifest.csv")
    printf '  %-28s jobs=%3d  slots=%4d\n' "$relative" "$pending_jobs" "$pending_slots"
    total_jobs=$((total_jobs + pending_jobs))
    total_slots=$((total_slots + pending_slots))
  done
  echo "Total: jobs=$total_jobs slots=$total_slots"
}

print_inventory
if [[ "$MODE" == list ]]; then
  exit 0
fi

TMP_ROOT="$(mktemp -d -t aec-missing-verify.XXXXXX)"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_ROOT="$SCRIPT_DIR/experiments_AEC/resume_logs/$RUN_ID"
mkdir -p "$LOG_ROOT"
cleanup() {
  rm -rf -- "$TMP_ROOT"
}
trap cleanup EXIT

echo
echo "Execution settings:"
echo "  GPUs: 0,2,3,4,6,7"
echo "  max_parallel_per_gpu: $MAX_PARALLEL_PER_GPU"
echo "  log directory: $LOG_ROOT"

for config_path in "${CONFIGS[@]}"; do
  relative="${config_path#"$CONFIG_ROOT/"}"
  output_dir="$OUTPUT_ROOT/$relative"
  read -r pending_jobs pending_slots < <(manifest_pending "$output_dir/manifest.csv")
  if [[ "$pending_slots" -eq 0 ]]; then
    echo "[skip] $relative is already complete"
    continue
  fi

  temp_config="$TMP_ROOT/${relative//\//__}"
  sed -E \
    -e "s/^  gpu_ids:.*$/  gpu_ids: $GPU_IDS/" \
    -e "s/^  max_parallel_per_gpu:.*$/  max_parallel_per_gpu: $MAX_PARALLEL_PER_GPU/" \
    "$config_path" > "$temp_config"

  "$PYTHON_BIN" -B - "$temp_config" <<'PY'
import sys
from pathlib import Path
import yaml

path = Path(sys.argv[1])
data = yaml.safe_load(path.read_text(encoding="utf-8"))
assert data["device"]["gpu_ids"] == [0, 2, 3, 4, 6, 7]
assert data["device"]["max_parallel_per_gpu"] == 10
PY

  log_name="${relative//\//__}"
  log_path="$LOG_ROOT/${log_name%.yaml}.log"
  echo
  echo "[run] $relative: jobs=$pending_jobs slots=$pending_slots"
  set +e
  "$PYTHON_BIN" -u -m hparams_search_scripts.run_mechanism_hparam_search \
    --config "$temp_config" \
    --output_root_dir "$output_dir" \
    2>&1 | tee "$log_path"
  runner_status=${PIPESTATUS[0]}
  set -e

  # The runner copies the temporary device settings into input_config.yaml.
  # Restore the canonical AEC config regardless of success or failure.
  cp -- "$config_path" "$output_dir/input_config.yaml"
  if [[ "$runner_status" -ne 0 ]]; then
    echo "[failed] $relative (see $log_path)" >&2
    exit "$runner_status"
  fi

  "$PYTHON_BIN" -B - "$output_dir/manifest.csv" <<'PY'
import csv
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
with manifest.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
if not rows:
    raise SystemExit(f"Empty manifest: {manifest}")

for row in rows:
    if row["search_status"] not in {"completed", "skipped_existing_result"}:
        raise SystemExit(
            f"Job is not complete after runner success: {row['job_dir']} "
            f"status={row['search_status']}"
        )
    if row["verify_done"] != row["verify_total"]:
        raise SystemExit(
            f"Verify count mismatch: {row['job_dir']} "
            f"{row['verify_done']}/{row['verify_total']}"
        )
    job_dir = Path(row["job_dir"])
    for required in ("best_config.yaml", "verify_top5_summary.csv", "recommended_command.txt"):
        if not (job_dir / required).is_file():
            raise SystemExit(f"Missing final artifact: {job_dir / required}")

    provenance = job_dir / "aec_provenance"
    provenance.mkdir(exist_ok=True)
    archived = {
        "AEC_BEST_CANDIDATE_10SEED.csv": "best_candidate_10seed.precompletion.csv",
        "AEC_BEST_CANDIDATE_10SEED_SUMMARY.yaml": "best_candidate_10seed_summary.precompletion.yaml",
    }
    for source_name, target_name in archived.items():
        source = job_dir / source_name
        target = provenance / target_name
        if source.exists():
            if target.exists():
                source.unlink()
            else:
                source.replace(target)
    for stale_name in ("AEC_INCOMPLETE.yaml", "AEC_MISSING_VERIFY.csv"):
        (job_dir / stale_name).unlink(missing_ok=True)
PY

  read -r remaining_jobs remaining_slots < <(manifest_pending "$output_dir/manifest.csv")
  if [[ "$remaining_slots" -ne 0 ]]; then
    echo "[failed] $relative still has $remaining_slots missing slots" >&2
    exit 1
  fi
  echo "[done] $relative"
done

"$PYTHON_BIN" -B - "$SCRIPT_DIR/experiments_AEC" <<'PY'
import csv
import sys
from collections import Counter
from pathlib import Path
import yaml

root = Path(sys.argv[1]).resolve()
sources_path = root / "assembly_sources.csv"
with sources_path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    fieldnames = list(reader.fieldnames or [])
    rows = list(reader)

updated = 0
for row in rows:
    if not row["config"].startswith("figure1/LDPGNN/"):
        continue
    job_dir = Path(row["target_job_dir"])
    if not all((job_dir / name).is_file() for name in (
        "best_config.yaml", "verify_top5_summary.csv", "recommended_command.txt"
    )):
        raise SystemExit(f"Cannot finalize global inventory; incomplete job: {job_dir}")
    row["assembly_status"] = "completed_after_resume"
    row["verify_done"] = "50"
    row["verify_missing"] = "0"
    updated += 1
if updated != 480:
    raise SystemExit(f"Expected to update 480 source rows, updated {updated}")

temporary = sources_path.with_suffix(".csv.tmp")
with temporary.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
temporary.replace(sources_path)

summary_path = root / "assembly_summary.yaml"
summary = yaml.safe_load(summary_path.read_text(encoding="utf-8"))
summary["initial_complete_job_count"] = 786
summary["initial_partial_job_count"] = 480
summary["initial_partial_verify_present_per_job"] = 30
summary["initial_partial_verify_missing_per_job"] = 20
summary["complete_job_count"] = 1266
summary["partial_job_count"] = 0
summary["resumed_job_count"] = 480
summary.pop("partial_verify_present_per_job", None)
summary.pop("partial_verify_missing_per_job", None)
summary_path.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")

statuses = Counter(row["assembly_status"] for row in rows)
expected = Counter({"complete": 786, "completed_after_resume": 480})
if statuses != expected:
    raise SystemExit(f"Unexpected final source statuses: {statuses}")
print("Global AEC inventory updated:", dict(statuses))
PY

echo
print_inventory
echo "All missing AEC verify slots are complete."
echo "Logs: $LOG_ROOT"
