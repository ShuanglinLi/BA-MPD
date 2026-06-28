#!/usr/bin/env bash
set -euo pipefail

RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${DCASE_TASK1_REPO:?Set DCASE_TASK1_REPO to the official DCASE 2024 Task 1 training repository.}"
: "${DCASE24_DATASET_DIR:?Set DCASE24_DATASET_DIR to the TAU Urban Acoustic Scenes development dataset directory.}"

PYTHON="${PYTHON:-python}"
BUDGETS="${BUDGETS:-5 10 25}"
TAU_TEACHER_LOGITS_DIR="${TAU_TEACHER_LOGITS_DIR:-${RELEASE_ROOT}/artifacts/teacher_logits/tau}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RELEASE_ROOT}/outputs/tau}"
PROJECT_NAME="${PROJECT_NAME:-BA_MPD_release_tau}"
SEED="${SEED:-1234}"

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/metrics"

install_release_splits() {
  mkdir -p "${DCASE_TASK1_REPO}/split_setup"
  cp "${RELEASE_ROOT}/splits/tau/train_5.csv" "${DCASE_TASK1_REPO}/split_setup/split5.csv"
  cp "${RELEASE_ROOT}/splits/tau/train_10.csv" "${DCASE_TASK1_REPO}/split_setup/split10.csv"
  cp "${RELEASE_ROOT}/splits/tau/train_25.csv" "${DCASE_TASK1_REPO}/split_setup/split25.csv"
  cp "${RELEASE_ROOT}/splits/tau/train_full.csv" "${DCASE_TASK1_REPO}/split_setup/split100.csv"
  cp "${RELEASE_ROOT}/splits/tau/test.csv" "${DCASE_TASK1_REPO}/split_setup/test.csv"
}

run_one() {
  local budget="$1"
  local teacher_logits="${TAU_TEACHER_LOGITS_DIR}/train_${budget}_logits.npy"
  local teacher_index="${TAU_TEACHER_LOGITS_DIR}/train_${budget}_logits.index.json"
  local tag="tau_budget${budget}_ba_mpd"
  local metrics="${OUTPUT_ROOT}/metrics/${tag}.json"
  local log_file="${OUTPUT_ROOT}/logs/${tag}.log"

  if [[ ! -s "${teacher_logits}" ]]; then
    echo "Missing teacher logits: ${teacher_logits}" >&2
    exit 2
  fi
  if [[ ! -s "${teacher_index}" ]]; then
    echo "Missing teacher-logits index: ${teacher_index}" >&2
    exit 2
  fi

  echo "[$(date '+%F %T')] start TAU budget=${budget} BA-MPD"
  (
    cd "${DCASE_TASK1_REPO}"
    export DCASE24_DATASET_DIR
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export WANDB_SILENT="${WANDB_SILENT:-true}"
    export TOKENIZERS_PARALLELISM=false
    "${PYTHON}" run_training.py \
      --project_name "${PROJECT_NAME}" \
      --experiment_name "${tag}" \
      --method kd \
      --subset "${budget}" \
      --seed "${SEED}" \
      --n_epochs 150 \
      --batch_size 256 \
      --num_workers 2 \
      --precision 32 \
      --teacher_logits_npy "${teacher_logits}" \
      --teacher_logits_index "${teacher_index}" \
      --kd_weight 0.3 \
      --kd_temperature 2.0 \
      --kd_teacher_temperature 2.0 \
      --kd_delay_epochs 10 \
      --kd_ramp_epochs 10 \
      --kd_loss_type anchor_tad \
      --tad_top_k 2 \
      --tad_tail_weight 2.0 \
      --kd_luminet_blend 0.0 \
      --metrics_out "${metrics}"
  ) 2>&1 | tee "${log_file}"
  echo "[$(date '+%F %T')] done TAU budget=${budget} BA-MPD"
}

install_release_splits

for budget in ${BUDGETS}; do
  run_one "${budget}"
done

echo "TAU BA-MPD runs finished. Outputs are under ${OUTPUT_ROOT}."
