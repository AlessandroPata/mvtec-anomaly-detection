#!/usr/bin/env bash
set -euo pipefail

cd /notebooks/storage/project/ocgan-modernized
source .venv/bin/activate

CATEGORY="${1:?Usage: bash scripts/run_best_mvtec_category.sh <category>}"

case "${CATEGORY}" in
  bottle) CONFIG="experiments/bottle_t1c_m1c_lf1c_oc0_s44" ;;
  cable) CONFIG="experiments/cable_t1b_m1d_lf1d_oc0_s46" ;;
  capsule) CONFIG="experiments/capsule_t1c_m1c_lf1c_oc0_s43" ;;
  carpet) CONFIG="experiments/carpet_t1a_m1a_lf1b_oc0_s46" ;;
  grid) CONFIG="experiments/grid_t1a_m1a_lf1b_oc0_s46" ;;
  hazelnut) CONFIG="experiments/hazelnut_t1d_m1d_lf1c_oc0_s44" ;;
  leather) CONFIG="experiments/leather_t1d_m1d_lf1c_oc0_s43" ;;
  metal_nut) CONFIG="experiments/metal_nut_t1a_m1a_lf1b_oc0_s43" ;;
  pill) CONFIG="experiments/pill_t1b_m1d_lf1d_oc0_s44" ;;
  screw) CONFIG="experiments/screw_t1a_m1a_lf1b_oc0_s46" ;;
  tile) CONFIG="experiments/tile_t1a_m1a_lf1b_oc0_s46" ;;
  toothbrush) CONFIG="experiments/toothbrush_t1b_m1c_lf1b_oc0_s45" ;;
  transistor) CONFIG="experiments/transistor_t1b_m1d_lf1d_oc0_s45" ;;
  wood) CONFIG="experiments/wood_t1a_m1a_lf1b_oc0_s43" ;;
  zipper) CONFIG="experiments/zipper_t1b_m1d_lf1d_oc0_s46" ;;
  *) echo "Unknown category: ${CATEGORY}" ; exit 1 ;;
esac

echo "Running category=${CATEGORY} with config=${CONFIG}"

python scripts/train.py \
  --config-path ../configs \
  --config-name "${CONFIG}" \
  dataset.category="${CATEGORY}"