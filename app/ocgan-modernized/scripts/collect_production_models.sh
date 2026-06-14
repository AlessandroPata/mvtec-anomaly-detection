#!/usr/bin/env bash
set -euo pipefail

BASE_OUT="/notebooks/storage/outputs/ocgan-modernized"
TARGET="/notebooks/storage/project/ocgan-modernized/production_models"

mkdir -p "${TARGET}"

collect_one() {
  local category="$1"
  local seed="$2"
  local run_dir
  run_dir=$(find "${BASE_OUT}" -maxdepth 1 -type d -name "${category}_production_s${seed}_seed${seed}_*" | sort | tail -n 1)

  if [[ -z "${run_dir}" ]]; then
    echo "Run dir not found for ${category}"
    return 1
  fi

  mkdir -p "${TARGET}/${category}"
  cp "${run_dir}/best_checkpoint.pt" "${TARGET}/${category}/model.pt"

  if [[ -f "${run_dir}/config.yaml" ]]; then
    cp "${run_dir}/config.yaml" "${TARGET}/${category}/config.yaml"
  fi

  cat > "${TARGET}/${category}/manifest.json" <<EOF
{
  "category": "${category}",
  "seed": ${seed},
  "run_dir": "${run_dir}",
  "checkpoint": "model.pt"
}
EOF
}

collect_one bottle 43
collect_one cable 46
collect_one capsule 46
collect_one carpet 43
collect_one grid 43
collect_one hazelnut 43
collect_one leather 43
collect_one metal_nut 44
collect_one pill 43
collect_one screw 43
collect_one tile 46
collect_one toothbrush 44
collect_one transistor 44
collect_one wood 43
collect_one zipper 43

echo "Saved production models to ${TARGET}"