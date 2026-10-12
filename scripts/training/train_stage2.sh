#!/usr/bin/env bash
set -euo pipefail

# Stage 2: joint fine-tuning of generation, verification, and ECFR.
# The current cache is reused, while Future-BEV targets are built online from
# raw future camera and LiDAR observations through SceneLoader.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

: "${OPENSCENE_DATA_ROOT:?Set OPENSCENE_DATA_ROOT to the NAVSIM data root.}"
: "${NAVSIM_EXP_ROOT:?Set NAVSIM_EXP_ROOT to an experiment-output root.}"
: "${NUPLAN_MAPS_ROOT:?Set NUPLAN_MAPS_ROOT to the nuPlan map root.}"
: "${MANEUVERFLOW_TRAIN_CACHE:?Set MANEUVERFLOW_TRAIN_CACHE to the completed current-supervision cache.}"
: "${MANEUVERFLOW_METRIC_CACHE:?Set MANEUVERFLOW_METRIC_CACHE to the static NAVSIM scene metric cache.}"
MANEUVERFLOW_JOINT_MAX_EPOCHS="${MANEUVERFLOW_JOINT_MAX_EPOCHS:-20}"

NUM_GPUS="${NUM_GPUS:-4}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-maneuverflow_joint_finetuning}"
RESUME_CKPT_PATH="${RESUME_CKPT_PATH:-}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "${RESUME_CKPT_PATH}" ]]; then
  : "${MANEUVERFLOW_STAGE1_CHECKPOINT:?Set MANEUVERFLOW_STAGE1_CHECKPOINT to a Stage-1 Lightning checkpoint.}"
fi

export NAVSIM_DEVKIT_ROOT="${PROJECT_ROOT}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export MANEUVERFLOW_JOINT_MAX_EPOCHS
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi

required_directories=(
  "${OPENSCENE_DATA_ROOT}/navsim_logs/trainval"
  "${OPENSCENE_DATA_ROOT}/sensor_blobs/trainval"
  "${NUPLAN_MAPS_ROOT}"
  "${MANEUVERFLOW_TRAIN_CACHE}"
  "${MANEUVERFLOW_METRIC_CACHE}"
)
for directory in "${required_directories[@]}"; do
  if [[ ! -d "${directory}" ]]; then
    echo "Required directory does not exist: ${directory}" >&2
    exit 2
  fi
done
if [[ -z "${RESUME_CKPT_PATH}" && ! -f "${MANEUVERFLOW_STAGE1_CHECKPOINT}" ]]; then
  echo "Stage-1 checkpoint does not exist: ${MANEUVERFLOW_STAGE1_CHECKPOINT}" >&2
  exit 2
fi
if [[ -n "${RESUME_CKPT_PATH}" && ! -f "${RESUME_CKPT_PATH}" ]]; then
  echo "Resume checkpoint does not exist: ${RESUME_CKPT_PATH}" >&2
  exit 2
fi

integer_values=(
  "${NUM_GPUS}"
  "${PER_DEVICE_BATCH_SIZE}"
  "${ACCUMULATE_GRAD_BATCHES}"
  "${MANEUVERFLOW_JOINT_MAX_EPOCHS}"
)
for value in "${integer_values[@]}"; do
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPU, batch, and epoch values must be positive integers." >&2
    exit 2
  fi
done

if [[ ! "${NUM_WORKERS}" =~ ^[0-9]+$ ]]; then
  echo "NUM_WORKERS must be a non-negative integer." >&2
  exit 2
fi

GLOBAL_BATCH_SIZE=$((NUM_GPUS * PER_DEVICE_BATCH_SIZE * ACCUMULATE_GRAD_BATCHES))
if [[ "${GLOBAL_BATCH_SIZE}" -ne 32 ]]; then
  echo "Stage 2 requires global batch size 32; resolved ${GLOBAL_BATCH_SIZE}." >&2
  echo "Set NUM_GPUS, PER_DEVICE_BATCH_SIZE, and ACCUMULATE_GRAD_BATCHES so their product is 32." >&2
  exit 2
fi

arguments=(
  "${PROJECT_ROOT}/navsim/planning/script/run_training.py"
  "experiment_name=${EXPERIMENT_NAME}"
  "train_test_split=navtrain"
  "agent=maneuverflow_joint_finetuning"
  "cache_path=${MANEUVERFLOW_TRAIN_CACHE}"
  "use_cache_without_dataset=false"
  "force_cache_computation=false"
  "dataloader.params.batch_size=${PER_DEVICE_BATCH_SIZE}"
  "dataloader.params.num_workers=${NUM_WORKERS}"
  "trainer.params.devices=${NUM_GPUS}"
  "trainer.params.accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES}"
  "trainer.params.max_epochs=${MANEUVERFLOW_JOINT_MAX_EPOCHS}"
  "trainer.params.precision=32-true"
  "trainer.params.gradient_clip_val=1.0"
  "trainer.params.gradient_clip_algorithm=norm"
)
if [[ -n "${RESUME_CKPT_PATH}" ]]; then
  arguments+=("resume_ckpt_path=${RESUME_CKPT_PATH}" "agent.initialize_from_checkpoint=false" "agent.checkpoint_path=null")
fi

printf 'ManeuverFlow Joint fine-tuning command:'
printf ' %q' "${PYTHON_BIN}" "${arguments[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" "${arguments[@]}"
