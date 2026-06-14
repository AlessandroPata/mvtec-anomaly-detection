#!/usr/bin/env bash
# Sprint 3: per-category memory bank tuning + Ridge fusion (C=0.5) + FPR@95 selection
#
# Key changes vs v4:
#   - memory_bank DISABLED for: bottle, cable, grid, metal_nut, transistor, wood
#   - memory_bank topk=1 (max, PatchCore standard) for: toothbrush
#   - memory_bank topk=3 topk_mean for all others with bank
#   - score_fusion_learned.C=0.5 (ridge, overrides per-cat yaml C=2-8)
#   - selection: 0.4*auroc + 0.4*auprc + 0.2*(1-FPR@95) (new)
#   - capsule/transistor: unfreeze_from=none (v3b finding)
#
# Composite baseline: 0.7866 (best-of-v3/v3b/v4)
set -euo pipefail

python -c "import hydra" 2>/dev/null || pip install hydra-core --quiet
python -c "import pytorch_msssim" 2>/dev/null || pip install pytorch-msssim --quiet
echo "[deps] OK"

cd /notebooks/storage_project_outputs_datasets/project/ocgan-modernized

LOG_DIR="logs/v5_sprint3"
STATE_DIR="logs/v5_sprint3_state"
CATEGORIES=(bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper)
SEEDS=(43 44 45 46 47)
PARALLEL_JOBS=2

mkdir -p "${LOG_DIR}" "${STATE_DIR}"

# Categories where memory bank is harmful (all formulas < v3 baseline)
NO_BANK_CATS="bottle cable grid metal_nut transistor wood"

# Categories that benefit from max aggregation (k=1) over k=3
MAX_BANK_CATS="toothbrush"

# Categories where frozen backbone is better (v3b finding + ablation)
FROZEN_CATS="capsule transistor"

run_one() {
  local category="$1"
  local seed="$2"
  local run_name="${category}_v5_s${seed}"
  local log_file="${LOG_DIR}/${run_name}.log"
  local done_file="${STATE_DIR}/${run_name}.done"

  if [[ -f "${done_file}" ]]; then
    echo "===== SKIP DONE ${run_name} ====="
    return 0
  fi

  # Memory bank config
  local mb_enabled="true"
  local mb_topk=3
  local mb_agg="topk_mean"
  for no_bank_cat in ${NO_BANK_CATS}; do
    [[ "${category}" == "${no_bank_cat}" ]] && mb_enabled="false" && break
  done
  for max_cat in ${MAX_BANK_CATS}; do
    [[ "${category}" == "${max_cat}" ]] && mb_topk=1 && break
  done

  # Backbone freeze
  local unfreeze="layer3"
  for frozen_cat in ${FROZEN_CATS}; do
    [[ "${category}" == "${frozen_cat}" ]] && unfreeze="none" && break
  done

  echo "===== RUNNING ${run_name} (bank=${mb_enabled} agg=${mb_agg} topk=${mb_topk} unfreeze=${unfreeze}) =====" | tee "${log_file}"

  python scripts/train.py \
    --config-path ../configs \
    --config-name "experiments/final_per_category/${category}" \
    project.seed="${seed}" \
    project.experiment_name="${run_name}" \
    project.deterministic=true \
    model.reconstruction.use_skip_connections=true \
    model.backbone.unfreeze_from="${unfreeze}" \
    scoring_topk=100 \
    dataset.val_mixed_ratio=0.25 \
    memory_bank.enabled="${mb_enabled}" \
    "memory_bank.feature_level=layer2+layer3" \
    memory_bank.max_train_batches=-1 \
    memory_bank.max_patches=10000 \
    memory_bank.aggregation="${mb_agg}" \
    memory_bank.topk="${mb_topk}" \
    ++memory_bank.candidate_pool_size=20000 \
    score_fusion_learned.C=0.5 \
    logging.save_debug_images=false \
    logging.save_score_histograms=false \
    analysis.save_failure_analysis=false \
    checkpoint.save_last=false \
    checkpoint.save_best=false \
    runtime.detect_nan=false \
    training.resume=false \
    2>&1 | tee -a "${log_file}"

  if grep -q "Training finished." "${log_file}"; then
    touch "${done_file}"
  fi
}

export -f run_one
export LOG_DIR STATE_DIR NO_BANK_CATS MAX_BANK_CATS FROZEN_CATS

TMP_TASKS="$(mktemp)"
for category in "${CATEGORIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "${category}|${seed}" >> "${TMP_TASKS}"
  done
done

echo "=== v5 Sprint3: $(wc -l < "${TMP_TASKS}") runs, parallel=${PARALLEL_JOBS} ==="

if command -v parallel >/dev/null 2>&1; then
  parallel --jobs "${PARALLEL_JOBS}" --colsep '\|' run_one {1} {2} :::: "${TMP_TASKS}"
else
  while IFS='|' read -r cat seed; do run_one "${cat}" "${seed}"; done < "${TMP_TASKS}"
fi

rm -f "${TMP_TASKS}"

echo "=== v5 DONE. Results: ==="
python3 - <<'PYEOF'
import glob, re, statistics, os
V3_COMP = {'bottle':0.8836,'cable':0.5901,'capsule':0.6400,'carpet':0.9186,
           'grid':0.7105,'hazelnut':0.9866,'leather':0.8804,'metal_nut':0.6533,
           'pill':0.6438,'screw':0.7968,'tile':0.9368,'toothbrush':0.4657,
           'transistor':0.6430,'wood':0.9799,'zipper':0.7704}
V4_COMP = {'carpet':0.9415,'hazelnut':0.9928,'leather':0.8937,'pill':0.7439,
           'screw':0.8211,'tile':0.9444,'toothbrush':0.5681,'zipper':0.7938}
PREV_BEST = {**V3_COMP, **V4_COMP}
results = {}
for f in glob.glob("logs/v5_sprint3/*.log"):
    base = os.path.basename(f).replace(".log","")
    m = re.match(r'^(.+)_v5_s\d+$', base)
    if not m: continue
    cat = m.group(1)
    auroc = None
    with open(f) as fh:
        for line in fh:
            m2 = re.search(r'\[Test\] AUROC=([0-9.]+)', line)
            if m2: auroc = float(m2.group(1))
    if auroc: results.setdefault(cat, []).append(auroc)
total_v5, total_prev = [], []
print(f"{'Cat':12s}  {'prev_best':9s}  {'v5_mean':7s}  {'delta':6s}  n")
print("-"*55)
for cat in sorted(PREV_BEST):
    prev = PREV_BEST[cat]
    vals = results.get(cat, [])
    if vals:
        mean = statistics.mean(vals)
        d = mean - prev
        total_v5.append(mean); total_prev.append(prev)
        print(f"{cat:12s}  {prev:.4f}     {mean:.4f}   {d:+.4f}  {len(vals)}/5")
if total_v5:
    print("-"*55)
    print(f"{'OVERALL':12s}  {statistics.mean(total_prev):.4f}     {statistics.mean(total_v5):.4f}   {statistics.mean(total_v5)-statistics.mean(total_prev):+.4f}")
PYEOF
