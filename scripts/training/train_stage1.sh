#!/usr/bin/env bash
set -euo pipefail

# Stage 1: generation pretraining with perception, M-GMP, and M3F.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

: "${OPENSCENE_DATA_ROOT:?Set OPENSCENE_DATA_ROOT to the NAVSIM data root.}"
: "${NAVSIM_EXP_ROOT:?Set NAVSIM_EXP_ROOT to an experiment-output root.}"
: "${MANEUVERFLOW_TRAIN_CACHE:?Set MANEUVERFLOW_TRAIN_CACHE to the completed current-supervision cache.}"
: "${MANEUVERFLOW_MGMP_PRIOR_DIR:?Set MANEUVERFLOW_MGMP_PRIOR_DIR to the five-file M-GMP prior directory.}"
MANEUVERFLOW_STAGE1_MAX_EPOCHS="${MANEUVERFLOW_STAGE1_MAX_EPOCHS:-100}"

NUM_GPUS="${NUM_GPUS:-4}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-maneuverflow_generation_pretraining}"
RESUME_CKPT_PATH="${RESUME_CKPT_PATH:-}"
DRY_RUN="${DRY_RUN:-0}"


export NAVSIM_DEVKIT_ROOT="${PROJECT_ROOT}"
export MANEUVERFLOW_STAGE1_MAX_EPOCHS
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi

required_directories=(
  "${MANEUVERFLOW_TRAIN_CACHE}"
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
  maneuver_mgmp_metadata.json
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

integer_values=(
  "${NUM_GPUS}"
  "${PER_DEVICE_BATCH_SIZE}"
  "${ACCUMULATE_GRAD_BATCHES}"
  "${MANEUVERFLOW_STAGE1_MAX_EPOCHS}"
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
  echo "Stage 1 requires global batch size 32; resolved ${GLOBAL_BATCH_SIZE}." >&2
  echo "Set NUM_GPUS, PER_DEVICE_BATCH_SIZE, and ACCUMULATE_GRAD_BATCHES so their product is 32." >&2
  exit 2
fi
if [[ -n "${RESUME_CKPT_PATH}" && ! -f "${RESUME_CKPT_PATH}" ]]; then
  echo "Resume checkpoint does not exist: ${RESUME_CKPT_PATH}" >&2
  exit 2
fi

arguments=(
  "${PROJECT_ROOT}/navsim/planning/script/run_training.py"
  "experiment_name=${EXPERIMENT_NAME}"
  "train_test_split=navtrain"
  "agent=maneuverflow_generation_pretraining"
  "cache_path=${MANEUVERFLOW_TRAIN_CACHE}"
  "use_cache_without_dataset=true"
  "force_cache_computation=false"
  "dataloader.params.batch_size=${PER_DEVICE_BATCH_SIZE}"
  "dataloader.params.num_workers=${NUM_WORKERS}"
  "trainer.params.devices=${NUM_GPUS}"
  "trainer.params.accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES}"
  "trainer.params.max_epochs=${MANEUVERFLOW_STAGE1_MAX_EPOCHS}"
  "trainer.params.precision=32-true"
  "trainer.params.gradient_clip_val=1.0"
  "trainer.params.gradient_clip_algorithm=norm"
)
if [[ -n "${RESUME_CKPT_PATH}" ]]; then
  arguments+=("resume_ckpt_path=${RESUME_CKPT_PATH}" "agent.initialize_from_checkpoint=false" "agent.checkpoint_path=null")
fi

printf 'ManeuverFlow Generation Pretraining command:'
printf ' %q' "${PYTHON_BIN}" "${arguments[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" "${arguments[@]}"
