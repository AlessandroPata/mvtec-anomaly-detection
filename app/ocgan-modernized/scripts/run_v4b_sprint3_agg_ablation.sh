#!/usr/bin/env bash
# Sprint 3 ablation: test aggregation formula on key categories
# Compares topk_mean/topk=1 (PatchCore max), topk_mean/topk=3, topk_reweighted/topk=3
# Target: bottle/cable (hurt by v4), pill/toothbrush (won), capsule/transistor (hurt)
set -euo pipefail

python -c "import hydra" 2>/dev/null || pip install hydra-core --quiet
python -c "import pytorch_msssim" 2>/dev/null || pip install pytorch-msssim --quiet
echo "[deps] OK"

cd /notebooks/storage_project_outputs_datasets/project/ocgan-modernized

LOG_DIR="logs/v4b_agg_ablation"
STATE_DIR="logs/v4b_agg_ablation_state"
mkdir -p "${LOG_DIR}" "${STATE_DIR}"

# Categories: hurt (bottle, cable, transistor), won (pill, toothbrush), mixed (capsule)
CATEGORIES=(bottle cable capsule pill toothbrush transistor)
SEEDS=(43 44 45)  # 3 seeds sufficient for ablation
AGGS=(topk_mean_k1 topk_mean_k3 topk_reweighted_k3)

run_one() {
  local category="$1"
  local seed="$2"
  local agg_key="$3"
  local run_name="${category}_v4b_${agg_key}_s${seed}"
  local log_file="${LOG_DIR}/${run_name}.log"
  local done_file="${STATE_DIR}/${run_name}.done"

  if [[ -f "${done_file}" ]]; then
    echo "===== SKIP ${run_name} ====="
    return 0
  fi

  # Parse agg_key -> aggregation + topk params
  local agg topk
  case "${agg_key}" in
    topk_mean_k1)   agg="topk_mean"; topk=1 ;;
    topk_mean_k3)   agg="topk_mean"; topk=3 ;;
    topk_reweighted_k3) agg="topk_reweighted"; topk=3 ;;
  esac

  # Use frozen backbone for capsule/transistor (v3b finding)
  local unfreeze="layer3"
  [[ "${category}" == "capsule" || "${category}" == "transistor" ]] && unfreeze="none"

  echo "===== RUNNING ${run_name} (agg=${agg} topk=${topk} unfreeze=${unfreeze}) =====" | tee "${log_file}"

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
    memory_bank.enabled=true \
    "memory_bank.feature_level=layer2+layer3" \
    memory_bank.max_train_batches=-1 \
    memory_bank.max_patches=10000 \
    memory_bank.aggregation="${agg}" \
    memory_bank.topk="${topk}" \
    ++memory_bank.candidate_pool_size=20000 \
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
export LOG_DIR STATE_DIR

TMP_TASKS="$(mktemp)"
for cat in "${CATEGORIES[@]}"; do
  for agg in "${AGGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo "${cat}|${seed}|${agg}" >> "${TMP_TASKS}"
    done
  done
done

TOTAL=$(wc -l < "${TMP_TASKS}")
echo "=== Aggregation ablation: ${TOTAL} runs ($(wc -w <<< "${CATEGORIES[@]}") cats × 3 aggs × ${#SEEDS[@]} seeds) ==="

if command -v parallel >/dev/null 2>&1; then
  parallel --jobs 2 --colsep '\|' run_one {1} {2} {3} :::: "${TMP_TASKS}"
else
  while IFS='|' read -r cat seed agg; do run_one "${cat}" "${seed}" "${agg}"; done < "${TMP_TASKS}"
fi

rm -f "${TMP_TASKS}"

echo "=== Ablation complete. Results: ==="
python3 - <<'PYEOF'
import glob, re, statistics, os
results = {}
for f in glob.glob("logs/v4b_agg_ablation/*.log"):
    base = os.path.basename(f).replace(".log","")
    m = re.match(r'^(.+)_v4b_(topk\w+)_s\d+$', base)
    if not m: continue
    cat, agg = m.group(1), m.group(2)
    auroc = None
    with open(f) as fh:
        for line in fh:
            m2 = re.search(r'\[Test\] AUROC=([0-9.]+)', line)
            if m2: auroc = float(m2.group(1))
    if auroc: results.setdefault((cat, agg), []).append(auroc)

V3_BASE = {'bottle':0.8836,'cable':0.5901,'capsule':0.6400,'pill':0.6438,'toothbrush':0.4657,'transistor':0.6430}
cats = sorted(set(c for c,_ in results))
aggs = ['topk_mean_k1','topk_mean_k3','topk_reweighted_k3']
print(f"{'Cat':12s}  {'v3base':7s}", end="")
for a in aggs: print(f"  {a:20s}", end="")
print()
print("-"*80)
for cat in cats:
    base = V3_BASE.get(cat, 0)
    print(f"{cat:12s}  {base:.4f} ", end="")
    for agg in aggs:
        vals = results.get((cat, agg), [])
        if vals:
            mean = statistics.mean(vals)
            delta = mean - base
            sign = "+" if delta >= 0 else ""
            print(f"  {mean:.4f}({sign}{delta:.3f})      ", end="")
        else:
            print(f"  {'—':20s}", end="")
    print()
PYEOF
