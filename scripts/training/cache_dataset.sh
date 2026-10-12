#!/usr/bin/env bash
set -euo pipefail

# Build the reusable current-observation/current-supervision cache consumed by
# both ManeuverFlow optimization stages. Future-BEV targets are intentionally
# excluded; Joint fine-tuning constructs them online from future observations.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

: "${OPENSCENE_DATA_ROOT:?Set OPENSCENE_DATA_ROOT to the NAVSIM data root.}"
: "${NAVSIM_EXP_ROOT:?Set NAVSIM_EXP_ROOT to an experiment-output root.}"
: "${NUPLAN_MAPS_ROOT:?Set NUPLAN_MAPS_ROOT to the nuPlan map root.}"
: "${MANEUVERFLOW_TRAIN_CACHE:?Set MANEUVERFLOW_TRAIN_CACHE to the output cache root.}"
: "${MANEUVERFLOW_MGMP_PRIOR_DIR:?Set MANEUVERFLOW_MGMP_PRIOR_DIR to the five-file M-GMP prior directory.}"

CACHE_WORKER="${CACHE_WORKER:-ray_distributed_no_torch}"
CACHE_WORKERS="${CACHE_WORKERS:-4}"
FORCE_CACHE_COMPUTATION="${FORCE_CACHE_COMPUTATION:-true}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-maneuverflow_current_supervision_cache}"
DRY_RUN="${DRY_RUN:-0}"


# The epoch value is required while Hydra constructs the Stage-1 agent, but it
# does not affect the feature/target tensors written by this cache-only command.
export MANEUVERFLOW_STAGE1_MAX_EPOCHS="${MANEUVERFLOW_STAGE1_MAX_EPOCHS:-1}"
export NAVSIM_DEVKIT_ROOT="${PROJECT_ROOT}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi

required_directories=(
  "${OPENSCENE_DATA_ROOT}/navsim_logs/trainval"
  "${OPENSCENE_DATA_ROOT}/sensor_blobs/trainval"
  "${NUPLAN_MAPS_ROOT}"
  "${MANEUVERFLOW_MGMP_PRIOR_DIR}"
)
for directory in "${required_directories[@]}"; do
  if [[ ! -d "${directory}" ]]; then
    echo "Required directory does not exist: ${directory}" >&2
    exit 2
  fi
done

required_prior_files=(
  maneuver_displacement_mean.npy
  maneuver_displacement_std.npy
  maneuver_heading_mean.npy
  maneuver_heading_std.npy
  maneuver_component_counts.npy
)
for filename in "${required_prior_files[@]}"; do
  if [[ ! -f "${MANEUVERFLOW_MGMP_PRIOR_DIR}/${filename}" ]]; then
    echo "Required M-GMP prior file does not exist: ${MANEUVERFLOW_MGMP_PRIOR_DIR}/${filename}" >&2
    exit 2
  fi
done

"${PYTHON_BIN}" \
  "${PROJECT_ROOT}/navsim/agents/maneuverflow/mgmp_prior.py" \
  --prior-dir "${MANEUVERFLOW_MGMP_PRIOR_DIR}"

if [[ ! "${CACHE_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "CACHE_WORKERS must be a positive integer." >&2
  exit 2
fi
if [[ "${FORCE_CACHE_COMPUTATION}" != "true" && "${FORCE_CACHE_COMPUTATION}" != "false" ]]; then
  echo "FORCE_CACHE_COMPUTATION must be either true or false." >&2
  exit 2
fi

arguments=(
  "${PROJECT_ROOT}/navsim/planning/script/run_dataset_caching.py"
  "experiment_name=${EXPERIMENT_NAME}"
  "train_test_split=navtrain"
  "agent=maneuverflow_generation_pretraining"
  "worker=${CACHE_WORKER}"
  "cache_path=${MANEUVERFLOW_TRAIN_CACHE}"
  "force_cache_computation=${FORCE_CACHE_COMPUTATION}"
)
if [[ "${CACHE_WORKER}" == "ray_distributed_no_torch" ]]; then
  arguments+=("worker.threads_per_node=${CACHE_WORKERS}")
fi

printf 'ManeuverFlow current-supervision cache command:'
printf ' %q' "${PYTHON_BIN}" "${arguments[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" "${arguments[@]}"
