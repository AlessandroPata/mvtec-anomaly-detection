#!/usr/bin/env bash
set -euo pipefail

cd /notebooks/storage/project/ocgan-modernized
source .venv/bin/activate

LOG_DIR="logs/production_export"
mkdir -p "${LOG_DIR}"

run_one() {
  local category="$1"
  local seed="$2"
  local teacher="$3"
  local memory="$4"
  local fusion_c="$5"
  local max_patches="$6"
  local run_name="${category}_production_s${seed}"

  echo "===== RUNNING ${run_name} ====="

  python scripts/train.py \
    --config-path ../configs \
    --config-name "experiments/final_per_category_v2/${category}" \
    project.seed="${seed}" \
    project.experiment_name="${run_name}" \
    teacher_student.score_weight="${teacher}" \
    memory_bank.score_weight="${memory}" \
    memory_bank.max_patches="${max_patches}" \
    score_fusion_learned.enabled=true \
    score_fusion_learned.C="${fusion_c}" \
    one_class.score_weight="0.0" \
    logging.save_debug_images=false \
    logging.save_score_histograms=false \
    analysis.save_failure_analysis=false \
    checkpoint.save_last=false \
    checkpoint.save_best=true \
    training.resume=false \
    2>&1 | tee "${LOG_DIR}/${run_name}.log"
}

run_one bottle 43 0.4 0.02 1.0 4096
run_one cable 46 0.05 0.6 4.0 4096
run_one capsule 46 0.4 0.02 4.0 2048
run_one carpet 43 0.1 0.4 8.0 2048
run_one grid 43 0.4 0.02 8.0 2048
run_one hazelnut 43 0.05 0.1 8.0 1024
run_one leather 43 0.1 0.02 8.0 4096
run_one metal_nut 44 0.2 0.4 1.0 2048
run_one pill 43 0.2 0.4 1.0 4096
run_one screw 43 0.05 0.02 2.0 2048
run_one tile 46 0.1 0.4 1.0 1024
run_one toothbrush 44 0.05 0.6 2.0 4096
run_one transistor 44 0.05 0.4 2.0 4096
run_one wood 43 0.05 0.1 1.0 2048
run_one zipper 43 0.4 0.6 1.0 2048