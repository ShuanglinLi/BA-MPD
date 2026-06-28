#!/usr/bin/env bash
set -euo pipefail

RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${RELEASE_ROOT}"

: "${BEANS_CBI_DATA_DIR:?Set BEANS_CBI_DATA_DIR to the BEANS-CBI parquet directory.}"

PYTHON="${PYTHON:-python}"
OBJECTIVE="${OBJECTIVE:-ba-mpd}"
BUDGETS="${BUDGETS:-10 25 50}"
TOP_K="${TOP_K:-5}"
ALPHA="${ALPHA:-2.0}"
TEACHER_LOGITS_DIR="${TEACHER_LOGITS_DIR:-artifacts/teacher_logits/beans_cbi}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/beans_cbi}"

mkdir -p "${OUTPUT_ROOT}/logs"

run_one() {
  local budget="$1"
  local out_dir="${OUTPUT_ROOT}/${OBJECTIVE}_budget${budget}"
  local log_file="${OUTPUT_ROOT}/logs/${OBJECTIVE}_budget${budget}.log"
  local teacher_logits="${TEACHER_LOGITS_DIR}/train_${budget}_logits.npy"
  local teacher_index="${TEACHER_LOGITS_DIR}/train_${budget}_logits.index.json"

  if [[ ! -s "${teacher_logits}" ]]; then
    echo "Missing teacher logits: ${teacher_logits}" >&2
    exit 2
  fi
  if [[ ! -s "${teacher_index}" ]]; then
    echo "Missing teacher-logits index: ${teacher_index}" >&2
    exit 2
  fi

  echo "[$(date '+%F %T')] start BEANS-CBI budget=${budget} objective=${OBJECTIVE}"
  "${PYTHON}" -m ba_mpd.train_bampd_beans \
    --objective "${OBJECTIVE}" \
    --data-dir "${BEANS_CBI_DATA_DIR}" \
    --train-csv "splits/beans_cbi/train_${budget}.csv" \
    --valid-csv "splits/beans_cbi/valid.csv" \
    --test-csv "splits/beans_cbi/test.csv" \
    --label-mapping-csv "splits/beans_cbi/label_mapping.csv" \
    --teacher-logits-npy "${teacher_logits}" \
    --teacher-logits-index "${teacher_index}" \
    --output-dir "${out_dir}" \
    --budget "${budget}" \
    --top-k "${TOP_K}" \
    --alpha "${ALPHA}" \
    --temperature 2.0 \
    --kd-weight 0.3 \
    --epochs 20 \
    --batch-size 32 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --warmup-steps 200 \
    --pin-memory \
    --persistent-workers 2>&1 | tee "${log_file}"
  echo "[$(date '+%F %T')] done BEANS-CBI budget=${budget} objective=${OBJECTIVE}"
}

for budget in ${BUDGETS}; do
  run_one "${budget}"
done

echo "BEANS-CBI ${OBJECTIVE} runs finished. Outputs are under ${OUTPUT_ROOT}."
