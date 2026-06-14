#!/usr/bin/env bash
# P1 tuning: grid (layer1+layer2+layer3) and screw (DINOv2)
set -euo pipefail
cd "$(dirname "$0")/.."

CSV="logs/patchcore_p1.csv"
export PATCHCORE_CSV="$CSV"

BASE="model.backbone.name=wide_resnet50_2
memory_bank.aggregation=topk_reweighted
memory_bank.topk=9
memory_bank.max_patches=70000
+memory_bank.candidate_pool_size=20000"

echo "=== P1 tuning: grid + screw ==="

# --- Grid: try layer1+layer2+layer3 vs baseline layer2+layer3 ---
for SEED in 43 44 45; do
  DONE="logs/.done_p1_grid_l123_s${SEED}"
  [ -f "$DONE" ] && { echo "SKIP grid l1+l2+l3 s$SEED"; continue; }
  echo "--- grid layer1+layer2+layer3 seed=$SEED ---"
  /usr/local/bin/python scripts/patchcore_pure.py \
    --config-path ../configs \
    --config-name experiments/final_per_category/grid \
    $BASE \
    memory_bank.feature_level=layer1+layer2+layer3 \
    project.seed=$SEED
  touch "$DONE"
done

for SEED in 43 44 45; do
  DONE="logs/.done_p1_grid_l23_s${SEED}"
  [ -f "$DONE" ] && { echo "SKIP grid l2+l3 s$SEED"; continue; }
  echo "--- grid layer2+layer3 seed=$SEED (baseline) ---"
  /usr/local/bin/python scripts/patchcore_pure.py \
    --config-path ../configs \
    --config-name experiments/final_per_category/grid \
    $BASE \
    memory_bank.feature_level=layer2+layer3 \
    project.seed=$SEED
  touch "$DONE"
done

# --- Screw: try layer1+layer2+layer3 vs baseline layer2+layer3 ---
for SEED in 43 44 45; do
  DONE="logs/.done_p1_screw_l123_s${SEED}"
  [ -f "$DONE" ] && { echo "SKIP screw l1+l2+l3 s$SEED"; continue; }
  echo "--- screw layer1+layer2+layer3 seed=$SEED ---"
  /usr/local/bin/python scripts/patchcore_pure.py \
    --config-path ../configs \
    --config-name experiments/final_per_category/screw \
    $BASE \
    memory_bank.feature_level=layer1+layer2+layer3 \
    project.seed=$SEED
  touch "$DONE"
done

for SEED in 43 44 45; do
  DONE="logs/.done_p1_screw_l23_s${SEED}"
  [ -f "$DONE" ] && { echo "SKIP screw l2+l3 s$SEED"; continue; }
  echo "--- screw layer2+layer3 seed=$SEED (baseline) ---"
  /usr/local/bin/python scripts/patchcore_pure.py \
    --config-path ../configs \
    --config-name experiments/final_per_category/screw \
    $BASE \
    memory_bank.feature_level=layer2+layer3 \
    project.seed=$SEED
  touch "$DONE"
done

echo ""
echo "=== P1 Results ==="
python -c "
import pandas as pd
df = pd.read_csv('$CSV')
print(df[['category','seed','feature_level','auroc']].to_string(index=False))
print()
summary = df.groupby(['category','feature_level'])['auroc'].mean().reset_index()
print(summary.to_string(index=False))
"
