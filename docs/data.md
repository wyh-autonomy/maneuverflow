# Data preparation

Install the [environment](install.md) first. ManeuverFlow uses NAVSIM v1,
[OpenScene v1.1](https://huggingface.co/datasets/OpenDriveLab/OpenScene), and nuPlan maps.

## Choose the data you need

| Workflow | Raw split | Scene filter | Sensors |
| --- | --- | --- | --- |
| Inference and evaluation | `test` | `navtest` | Current cameras and LiDAR |
| Two-stage training | `trainval` | `navtrain` with training/validation log partitions | Current and future cameras/LiDAR at 1, 2, 3, and 4 seconds |

Training uses the **navtrain training partition (approximately 80k scenes)**,
with a separate navtrain validation partition. `navtest` is used for evaluation.
Prepare full `trainval` sensors for current inputs and the EMA encoder's future inputs.

## Workspace and runtime paths

From the repository root, set your workspace to an absolute path outside the checkout.
Apply these exports in each new shell:

```bash
export NAVSIM_DEVKIT_ROOT="$(pwd)"
export MANEUVERFLOW_WORKSPACE=/absolute/path/to/maneuverflow_workspace
export OPENSCENE_DATA_ROOT="${MANEUVERFLOW_WORKSPACE}/dataset"
export NAVSIM_EXP_ROOT="${MANEUVERFLOW_WORKSPACE}/exp"
export NUPLAN_MAPS_ROOT="${OPENSCENE_DATA_ROOT}/maps"
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0

export MANEUVERFLOW_MGMP_PRIOR_DIR="${MANEUVERFLOW_WORKSPACE}/assets/mgmp_prior"
export MANEUVERFLOW_TRAIN_CACHE="${MANEUVERFLOW_WORKSPACE}/cache/current"
export MANEUVERFLOW_METRIC_CACHE="${MANEUVERFLOW_WORKSPACE}/cache/navtrain_metric_v1_0"
export METRIC_CACHE_PATH="${MANEUVERFLOW_WORKSPACE}/cache/navtest_metric_v1_0"

mkdir -p "${OPENSCENE_DATA_ROOT}/navsim_logs" "${OPENSCENE_DATA_ROOT}/sensor_blobs"
mkdir -p "${NAVSIM_EXP_ROOT}" "${MANEUVERFLOW_WORKSPACE}/downloads"
```

Arrange the raw data as follows; prepare only the splits you need:

```text
maneuverflow_workspace/
  dataset/
    maps/nuplan-maps-v1.0.json
    maps/<map-location>/...
    navsim_logs/trainval/<log-name>.pkl
    navsim_logs/test/<log-name>.pkl
    sensor_blobs/trainval/<log-name>/...
    sensor_blobs/test/<log-name>/...
  assets/mgmp_prior/
  cache/current/
  cache/navtrain_metric_v1_0/
  cache/navtest_metric_v1_0/
  exp/
  downloads/
```

Existing datasets can be symlinked into this layout.

## 1. Download the maps

```bash
mkdir -p "${MANEUVERFLOW_WORKSPACE}/downloads/maps"
(
  cd "${MANEUVERFLOW_WORKSPACE}/downloads/maps"
  bash -e "${NAVSIM_DEVKIT_ROOT}/download/download_maps.sh"
)
mv -T "${MANEUVERFLOW_WORKSPACE}/downloads/maps/maps" "${NUPLAN_MAPS_ROOT}"
```

Set `NUPLAN_MAPS_ROOT` to the directory containing `nuplan-maps-v1.0.json`.

## 2. Prepare test data for navtest

```bash
mkdir -p "${MANEUVERFLOW_WORKSPACE}/downloads/test"
(
  cd "${MANEUVERFLOW_WORKSPACE}/downloads/test"
  bash -e "${NAVSIM_DEVKIT_ROOT}/download/download_test.sh"
)
mv -T "${MANEUVERFLOW_WORKSPACE}/downloads/test/test_navsim_logs/test" \
  "${OPENSCENE_DATA_ROOT}/navsim_logs/test"
mv -T "${MANEUVERFLOW_WORKSPACE}/downloads/test/test_sensor_blobs/test" \
  "${OPENSCENE_DATA_ROOT}/sensor_blobs/test"
```

Place log pickles directly in `navsim_logs/test`.

## 3. Prepare trainval data for two-stage training

Use NAVSIM's [full trainval downloader](https://github.com/autonomousvision/navsim/blob/v1.1/download/download_trainval.sh):

```bash
mkdir -p "${MANEUVERFLOW_WORKSPACE}/downloads/trainval"
wget -O "${MANEUVERFLOW_WORKSPACE}/downloads/trainval/download_trainval.sh" \
  https://raw.githubusercontent.com/autonomousvision/navsim/v1.1/download/download_trainval.sh
(
  cd "${MANEUVERFLOW_WORKSPACE}/downloads/trainval"
  bash -e ./download_trainval.sh
)
mv -T "${MANEUVERFLOW_WORKSPACE}/downloads/trainval/trainval_navsim_logs/trainval" \
  "${OPENSCENE_DATA_ROOT}/navsim_logs/trainval"
mv -T "${MANEUVERFLOW_WORKSPACE}/downloads/trainval/trainval_sensor_blobs/trainval" \
  "${OPENSCENE_DATA_ROOT}/sensor_blobs/trainval"
```

The `navtrain` filter reads the `trainval` directories. Keep the raw sensors
after caching: Stage 2 loads future observations on every training pass.

Continue with [training](train_eval.md#train-the-tf-model) or
[inference and evaluation](train_eval.md#evaluation-and-submission).
