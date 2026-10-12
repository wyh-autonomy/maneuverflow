# Training and inference

Complete [installation](install.md) and [data preparation](data.md), apply the
[workspace exports](data.md#workspace-and-runtime-paths), and run commands from
the repository root.

- [Inference, evaluation, and submissions](#evaluation-and-submission)
- [Two-stage training](#train-the-tf-model)
- [Training settings](#training-settings)
- [Resume training](#resume-training)
- [Export and evaluate trained weights](#export-and-evaluate-trained-weights)

## Evaluation and submission

Prepare test logs/sensors, maps, and the [released checkpoint](install.md#3-fetch-the-released-checkpoint).
The checkpoint includes its M-GMP prior. For newly trained weights, follow
[export and evaluation](#export-and-evaluate-trained-weights).

### 1. Cache navtest scene metrics

Cache preparation uses the [NAVSIM v1.0 PDM reference planner](https://github.com/autonomousvision/navsim/blob/9f3ea68ffda0c6afe37b51b402f3245ea9936e18/navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py).
Final evaluation uses NAVSIM v1.1 scoring. Use the cache paths from
[data preparation](data.md#workspace-and-runtime-paths) to create fresh caches.

```bash
python navsim/planning/script/run_metric_caching.py \
  train_test_split=navtest \
  worker=sequential \
  cache.cache_path="${METRIC_CACHE_PATH}" \
  cache.force_feature_computation=true
```

This cache contains static scene artifacts for NAVSIM scoring.

### 2. Evaluate the released checkpoint

Run inference and evaluation on navtest:

```bash
bash scripts/evaluation/evaluate.sh
```

The defaults select `maneuverflow_agent` and the bundled TF 0.906 checkpoint.
Set `INFERENCE_DEVICE=cpu` or `INFERENCE_DEVICE=cuda` to select a device.
`MAX_SCENES=1` runs one scene; the default evaluates the full split.

The score CSV is written to:

```text
${NAVSIM_EXP_ROOT}/maneuverflow_tf_0906_navtest_v1/<run-timestamp>/<score-timestamp>.csv
```

Set `EXPERIMENT_NAME` to change the output directory name.

### 3. Create and score a submission

To save predictions, set your identity metadata and output directory:

```bash
export OUTPUT_DIR="${MANEUVERFLOW_WORKSPACE}/submissions/tf_0906"
export TEAM_NAME='Your team'
export AUTHORS='Author One, Author Two'
export EMAIL='contact@example.org'
export INSTITUTION='Your institution'
export COUNTRY='Your country'
bash scripts/evaluation/create_submission.sh
```

Predictions are saved to `${OUTPUT_DIR}/submission.pkl`. Score them with the
navtest metric cache:

```bash
SUBMISSION_FILE="${OUTPUT_DIR}/submission.pkl" \
SCORE_OUTPUT_DIR="${MANEUVERFLOW_WORKSPACE}/submission_scores/tf_0906" \
  bash scripts/evaluation/score_submission.sh
```

## Train the TF model

Both stages use the **navtrain training partition (approximately 80k scenes)**.
The held-out navtrain validation partition is used for checkpoint selection;
`navtest` is used for evaluation. Prepare full trainval sensors, including future frames.

### 1. Build the M-GMP prior

Build the prior directly from raw logs using the same
[training log partition](../navsim/planning/script/config/training/default_train_val_test_log_split.yaml)
as both training stages:

```bash
python scripts/training/build_mgmp_prior.py \
  --navsim-log-path "${OPENSCENE_DATA_ROOT}/navsim_logs/trainval" \
  --output-dir "${MANEUVERFLOW_MGMP_PRIOR_DIR}"
```

Use `--overwrite` to rebuild an existing prior.

### 2. Cache current inputs and ground truth

```bash
bash scripts/training/cache_dataset.sh
```

The loaders select training and validation entries separately.
Use `FORCE_CACHE_COMPUTATION=false` to reuse existing entries.

### 3. Cache navtrain scene metrics

```bash
python navsim/planning/script/run_metric_caching.py \
  train_test_split=navtrain \
  worker=sequential \
  cache.cache_path="${MANEUVERFLOW_METRIC_CACHE}" \
  cache.force_feature_computation=true
```

Keep the cache roles separate:

| Variable | Contents | Used by |
| --- | --- | --- |
| `MANEUVERFLOW_TRAIN_CACHE` | Current sensor tensors and ground truth | Both training stages |
| `MANEUVERFLOW_METRIC_CACHE` | Static navtrain scene artifacts | Stage 2 candidate scoring |
| `METRIC_CACHE_PATH` | Static navtest scene artifacts | Evaluation and submission scoring |

Stage 2 encodes raw future sensors online with the EMA target encoder.

### 4. Train Stage 1

```bash
bash scripts/training/train_stage1.sh
```

Stage 1 trains perception and trajectory generation. Lightning saves the best
`val/loss` checkpoint and `last.ckpt` under:

```text
${NAVSIM_EXP_ROOT}/maneuverflow_generation_pretraining/<run-timestamp>/lightning_logs/version_<n>/checkpoints/
```

Locate the checkpoint from your completed run:

```bash
find "${NAVSIM_EXP_ROOT}/maneuverflow_generation_pretraining" -name last.ckpt -print
```

### 5. Train Stage 2

Set the actual Stage-1 checkpoint path, then start joint training:

```bash
export MANEUVERFLOW_STAGE1_CHECKPOINT=/absolute/path/to/your/stage1/checkpoints/last.ckpt
bash scripts/training/train_stage2.sh
```

Stage 2 initializes the online model and EMA target encoder from Stage 1.
Checkpoints are saved under:

```text
${NAVSIM_EXP_ROOT}/maneuverflow_joint_finetuning/<run-timestamp>/lightning_logs/version_<n>/checkpoints/
```

## Training settings

| Setting | Stage 1 | Stage 2 |
| --- | --- | --- |
| Epochs | 100 | 20 |
| Base learning rate | 4e-4 | 2e-4 |
| Global batch size | 32 | 32 |

In Stage 2, the future-target encoder is updated after every optimizer step
with EMA decay `0.999`. The future-feature loss weight is `0.5`.

Defaults use four GPUs with batch size 8 each. For one GPU, preserve global batch 32:

```bash
NUM_GPUS=1 PER_DEVICE_BATCH_SIZE=8 ACCUMULATE_GRAD_BATCHES=4 \
  bash scripts/training/train_stage1.sh
NUM_GPUS=1 PER_DEVICE_BATCH_SIZE=8 ACCUMULATE_GRAD_BATCHES=4 \
  bash scripts/training/train_stage2.sh
```

Keep `NUM_GPUS * PER_DEVICE_BATCH_SIZE * ACCUMULATE_GRAD_BATCHES = 32`.
Use `MANEUVERFLOW_STAGE1_MAX_EPOCHS` or `MANEUVERFLOW_JOINT_MAX_EPOCHS` to
change a stage's duration.

## Resume training

Restore full training state using a checkpoint from the same stage and configuration:

```bash
RESUME_CKPT_PATH=/absolute/path/to/your/stage1/checkpoints/last.ckpt \
  bash scripts/training/train_stage1.sh

RESUME_CKPT_PATH=/absolute/path/to/your/stage2/checkpoints/last.ckpt \
  bash scripts/training/train_stage2.sh
```

A Stage-2 resume restores the online model, EMA target encoder, and optimizer state.

## Export and evaluate trained weights

Export the chosen Stage-2 checkpoint:

```bash
export MANEUVERFLOW_STAGE2_CHECKPOINT=/absolute/path/to/your/stage2/checkpoints/last.ckpt
export EMA_INFERENCE_CHECKPOINT="${NAVSIM_EXP_ROOT}/exports/maneuverflow_tf_ema.pt"

python scripts/evaluation/export_checkpoint.py \
  --input "${MANEUVERFLOW_STAGE2_CHECKPOINT}" \
  --output "${EMA_INFERENCE_CHECKPOINT}"
```

The exporter writes the online planner with its M-GMP prior.
Use `--force` to overwrite an existing export.

After preparing test data and the [navtest metric cache](#1-cache-navtest-scene-metrics),
evaluate the exported planner with `maneuverflow_ema_agent`:

```bash
AGENT_CONFIG=maneuverflow_ema_agent CHECKPOINT="${EMA_INFERENCE_CHECKPOINT}" \
EXPERIMENT_NAME=maneuverflow_tf_ema_navtest \
  bash scripts/evaluation/evaluate.sh
```

Use `maneuverflow_agent` for TF 0.906 and `maneuverflow_ema_agent` for new exports.

To save predictions from these weights, set the
[identity fields](#3-create-and-score-a-submission), then run:

```bash
OUTPUT_DIR="${MANEUVERFLOW_WORKSPACE}/submissions/tf_ema" \
AGENT_CONFIG=maneuverflow_ema_agent CHECKPOINT="${EMA_INFERENCE_CHECKPOINT}" \
  bash scripts/evaluation/create_submission.sh
```

The output is `${MANEUVERFLOW_WORKSPACE}/submissions/tf_ema/submission.pkl`.

To score these predictions locally with the navtest metric cache:

```bash
SUBMISSION_FILE="${MANEUVERFLOW_WORKSPACE}/submissions/tf_ema/submission.pkl" \
SCORE_OUTPUT_DIR="${MANEUVERFLOW_WORKSPACE}/submission_scores/tf_ema" \
  bash scripts/evaluation/score_submission.sh
```
