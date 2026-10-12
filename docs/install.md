# Installation

Use Linux with Bash, Git, Git LFS, Conda, `wget`, `tar`, and `unzip`.
Training requires a CUDA-capable GPU. Package versions are defined in
[environment.yml](../environment.yml) and [requirements.txt](../requirements.txt).

## 1. Clone the repository

```bash
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/wyh-autonomy/maneuverflow.git
cd maneuverflow
```

## 2. Create the environment

```bash
conda env create -f environment.yml
conda activate navsim
pip install -e .
```

The environment uses Python 3.9 and PyTorch 2.4.1.

## 3. Fetch the released checkpoint

For inference with TF 0.906:

```bash
git lfs pull --include="checkpoints/maneuverflow_tf_0.906.pt"
sha256sum checkpoints/maneuverflow_tf_0.906.pt
```

The expected hash is listed on the [checkpoint page](../checkpoints/README.md).

## Set runtime paths

Apply the [workspace environment block](data.md#workspace-and-runtime-paths)
in each shell used for the tutorials. Run subsequent commands from the repository root.

Continue with [data preparation](data.md), then
[inference and evaluation](train_eval.md#evaluation-and-submission) or
[training](train_eval.md#train-the-tf-model).
