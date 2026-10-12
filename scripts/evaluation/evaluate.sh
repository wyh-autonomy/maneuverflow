#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
AGENT_CONFIG="${AGENT_CONFIG:-maneuverflow_agent}"
case "${AGENT_CONFIG}" in
  maneuverflow_agent) ;;
  maneuverflow_ema_agent)
    [[ -n "${CHECKPOINT:-}" ]] || { echo "Set CHECKPOINT to the exported EMA weights." >&2; exit 2; }
    ;;
  *) echo "AGENT_CONFIG must be maneuverflow_agent or maneuverflow_ema_agent." >&2; exit 2 ;;
esac
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/checkpoints/maneuverflow_tf_0.906.pt}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-maneuverflow_tf_0906_navtest_v1}"
INFERENCE_DEVICE="${INFERENCE_DEVICE:-auto}"
MAX_SCENES="${MAX_SCENES:-0}"
DRY_RUN="${DRY_RUN:-0}"

: "${OPENSCENE_DATA_ROOT:?Set OPENSCENE_DATA_ROOT to the NAVSIM data root.}"
: "${NAVSIM_EXP_ROOT:?Set NAVSIM_EXP_ROOT to an output root.}"
: "${NUPLAN_MAPS_ROOT:?Set NUPLAN_MAPS_ROOT to the nuPlan map root.}"
: "${METRIC_CACHE_PATH:?Set METRIC_CACHE_PATH to the navtest metric cache.}"

required_directories=(
  "${OPENSCENE_DATA_ROOT}/navsim_logs/test"
  "${OPENSCENE_DATA_ROOT}/sensor_blobs/test"
  "${NUPLAN_MAPS_ROOT}"
  "${METRIC_CACHE_PATH}"
)
for directory in "${required_directories[@]}"; do
  [[ -d "${directory}" ]] || { echo "Missing directory: ${directory}" >&2; exit 2; }
done
[[ -f "${CHECKPOINT}" ]] || { echo "Missing checkpoint: ${CHECKPOINT}" >&2; exit 2; }
[[ "${MAX_SCENES}" =~ ^[0-9]+$ ]] || { echo "MAX_SCENES must be a non-negative integer." >&2; exit 2; }
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || { echo "Python not found: ${PYTHON_BIN}" >&2; exit 2; }

export NAVSIM_DEVKIT_ROOT="${PROJECT_ROOT}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1

arguments=(
  "${PROJECT_ROOT}/navsim/planning/script/run_pdm_score.py"
  "experiment_name=${EXPERIMENT_NAME}"
  "train_test_split=navtest"
  "agent=${AGENT_CONFIG}"
  "agent.checkpoint_path=${CHECKPOINT}"
  "agent.config.inference_device=${INFERENCE_DEVICE}"
  "worker=sequential"
  "metric_cache_path=${METRIC_CACHE_PATH}"
)
if [[ "${MAX_SCENES}" -gt 0 ]]; then
  arguments+=("train_test_split.scene_filter.max_scenes=${MAX_SCENES}")
fi

printf 'Running:'
printf ' %q' "${PYTHON_BIN}" "${arguments[@]}"
printf '\n'
[[ "${DRY_RUN}" == "1" ]] && exit 0

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" "${arguments[@]}"
