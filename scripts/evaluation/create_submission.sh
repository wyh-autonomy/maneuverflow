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
INFERENCE_DEVICE="${INFERENCE_DEVICE:-auto}"
DRY_RUN="${DRY_RUN:-0}"
MAX_SCENES="${MAX_SCENES:-0}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-maneuverflow_submission}"

: "${OPENSCENE_DATA_ROOT:?Set OPENSCENE_DATA_ROOT to the NAVSIM data root.}"
: "${NAVSIM_EXP_ROOT:?Set NAVSIM_EXP_ROOT to an output root.}"
: "${NUPLAN_MAPS_ROOT:?Set NUPLAN_MAPS_ROOT to the nuPlan map root.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for the generated submission.pkl.}"
: "${TEAM_NAME:?Set TEAM_NAME, preferably as an HTML repository link.}"
: "${AUTHORS:?Set AUTHORS.}"
: "${EMAIL:?Set EMAIL.}"
: "${INSTITUTION:?Set INSTITUTION.}"
: "${COUNTRY:?Set COUNTRY.}"

for directory in \
  "${OPENSCENE_DATA_ROOT}/navsim_logs/test" \
  "${OPENSCENE_DATA_ROOT}/sensor_blobs/test" \
  "${NUPLAN_MAPS_ROOT}"; do
  [[ -d "${directory}" ]] || { echo "Missing directory: ${directory}" >&2; exit 2; }
done
[[ -f "${CHECKPOINT}" ]] || { echo "Missing checkpoint: ${CHECKPOINT}" >&2; exit 2; }
[[ "${MAX_SCENES}" =~ ^[0-9]+$ ]] || { echo "MAX_SCENES must be a non-negative integer." >&2; exit 2; }
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || { echo "Python not found: ${PYTHON_BIN}" >&2; exit 2; }

hydra_string() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '"%s"' "${value}"
}

mkdir -p "${OUTPUT_DIR}"
export NAVSIM_DEVKIT_ROOT="${PROJECT_ROOT}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1

arguments=(
  "${PROJECT_ROOT}/navsim/planning/script/run_create_submission_pickle.py"
  "train_test_split=navtest"
  "agent=${AGENT_CONFIG}"
  "agent.checkpoint_path=${CHECKPOINT}"
  "agent.config.inference_device=${INFERENCE_DEVICE}"
  "experiment_name=${EXPERIMENT_NAME}"
  "output_dir=${OUTPUT_DIR}"
  "team_name=$(hydra_string "${TEAM_NAME}")"
  "authors=$(hydra_string "${AUTHORS}")"
  "email=$(hydra_string "${EMAIL}")"
  "institution=$(hydra_string "${INSTITUTION}")"
  "country=$(hydra_string "${COUNTRY}")"
)
if [[ "${MAX_SCENES}" -gt 0 ]]; then
  arguments+=("train_test_split.scene_filter.max_scenes=${MAX_SCENES}")
fi

printf 'Running:'
printf ' %q' "${PYTHON_BIN}" "${arguments[@]}"
printf '\n'
[[ "${DRY_RUN}" == "1" ]] && exit 0

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" "${arguments[@]}"
test -s "${OUTPUT_DIR}/submission.pkl"
printf 'Created %s\n' "${OUTPUT_DIR}/submission.pkl"
