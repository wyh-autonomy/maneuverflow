#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"

: "${NAVSIM_EXP_ROOT:?Set NAVSIM_EXP_ROOT to an output root.}"
: "${METRIC_CACHE_PATH:?Set METRIC_CACHE_PATH to the navtest metric cache.}"
: "${SUBMISSION_FILE:?Set SUBMISSION_FILE to submission.pkl.}"
: "${SCORE_OUTPUT_DIR:?Set SCORE_OUTPUT_DIR for the score CSV.}"

[[ -d "${METRIC_CACHE_PATH}" ]] || { echo "Missing metric cache: ${METRIC_CACHE_PATH}" >&2; exit 2; }
[[ -s "${SUBMISSION_FILE}" ]] || { echo "Missing submission: ${SUBMISSION_FILE}" >&2; exit 2; }
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || { echo "Python not found: ${PYTHON_BIN}" >&2; exit 2; }

mkdir -p "${SCORE_OUTPUT_DIR}"
export NAVSIM_DEVKIT_ROOT="${PROJECT_ROOT}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1

arguments=(
  "${PROJECT_ROOT}/navsim/planning/script/run_pdm_score_from_submission.py"
  "train_test_split=navtest"
  "submission_file_path=${SUBMISSION_FILE}"
  "metric_cache_path=${METRIC_CACHE_PATH}"
  "output_dir=${SCORE_OUTPUT_DIR}"
)

printf 'Running:'
printf ' %q' "${PYTHON_BIN}" "${arguments[@]}"
printf '\n'
[[ "${DRY_RUN}" == "1" ]] && exit 0

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" "${arguments[@]}"
