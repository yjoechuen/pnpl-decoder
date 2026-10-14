# Kaggle free-GPU training: step by step

Develop and commit code in this repository, then train it in a Kaggle Notebook.
This guide trains the new **multi-scale CNN–Conformer** from `config_conformer.json`.
Load `main` to get the new model; record the resolved commit SHA for each run.
The larger model is a candidate, not a proven score improvement.

## Verification status

The new model passed CPU forward/backward, optimizer-update, checkpoint
round-trip, and baseline-split compatibility tests. The test suite had
**47 passing tests and four GPU-dependent tests skipped** in the CPU workspace.
The configured raw training subset was approximately **4.92 GiB**, excluding
download scratch space and metadata.

Single-GPU training was previously verified on Kaggle. The newer two-GPU path
still needs the hardware smoke test below; CPU tests alone do not verify CUDA.

## 1. Create a Kaggle Notebook

1. Open <https://www.kaggle.com/code> and create a notebook.
2. Select **Settings → Accelerator → GPU T4 x2**.
3. Enable **Internet**.
4. Complete account verification if requested.

Free GPU availability depends on your quota and current availability.

## 2. Clone and install the verified version

Run this cell in a fresh notebook:

```python
from pathlib import Path
import subprocess
import sys

repo = Path("/kaggle/working/pnpl-decoder")

if not repo.exists():
    subprocess.run(
        ["git", "clone",
         "https://github.com/yjoechuen/pnpl-decoder.git",
         str(repo)],
        check=True,
    )

sys.path.insert(0, str(repo))

from kaggle_runner import sync_repo, KaggleExperiment

# Load the new model. Use an exact commit SHA instead for repeated runs.
sync_repo(ref="main")
print('Code commit:', subprocess.check_output(
    ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip())
```

This installs the repo while retaining Kaggle's GPU-enabled PyTorch.
**Do not install a CPU-only PyTorch build.**

For future development, replace the commit with your newly pushed commit or
branch. `sync_repo` refuses to overwrite local edits. Use a fresh notebook
Python session when updating imported helper code.

## 3. Verify GPUs and disk space

```python
import shutil
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

assert torch.cuda.is_available(), "Enable GPU in notebook settings"
assert torch.cuda.device_count() >= 2, "Select GPU T4 x2"

for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))

free_gib = shutil.disk_usage("/kaggle/working").free / 2**30
print(f"Free disk: {free_gib:.1f} GiB")
```

## 4. Check the download plan

```python
subprocess.run(
    [
        sys.executable, str(repo / "decoder.py"), "prepare",
        "--config", str(repo / "config.json"),
        "--data", "/kaggle/working/pnpl-data",
        "--index-only",
        "--preflight",
    ],
    check=True,
)
```

This checks file sizes and disk headroom without downloading training files.

## 5. Prepare training data

```python
data_run = KaggleExperiment(
    data="/kaggle/working/pnpl-data",
)

data_run.prepare()
```

The default uses **HDF5 indexing**, avoiding duplicated epoch arrays.
Expected outputs include `samples_h5.csv`, `vocabulary.json`,
`split_manifest.json`, and raw MEG/event files.

An unauthenticated Hugging Face warning is normally harmless for these public
files. Optional authentication belongs in Kaggle Secrets, not notebook source.

Preparation intentionally uses the original `config.json`: the Conformer uses
identical data selections and splits, so that cache can be reused. If the cache
is already prepared, skip this step. Do not prepare the existing cache using
`config_conformer.json`; preparation rejects full-config changes.

## 6. Verify two-GPU forward/backward execution

```python
subprocess.run(
    [
        sys.executable, "-m", "pytest", "-q",
        "tests/test_models.py::test_conformer_cuda_training",
    ],
    cwd=repo,
    check=True,
)
```

If pytest is missing, install it with
`subprocess.run([sys.executable, "-m", "pip", "install", "pytest"], check=True)`.

The parametrized test covers one and two GPUs and should pass on T4 x2. An unsupported-determinism **warning** is
expected; a traceback or nonzero exit is not. GPU runs are not guaranteed to
be bitwise reproducible.

## 7. Start training on both T4s

```python
run = KaggleExperiment(
    config="/kaggle/working/pnpl-decoder/config_conformer.json",
    data="/kaggle/working/pnpl-data",
    output="/kaggle/working/conformer-001",
)
run.train(devices=2)
```

This runs up to **30 epochs**, with early stopping after eight epochs without
raw validation BAcc@10 improvement. Checkpoints go to
`/kaggle/working/conformer-001/`. Never resume the CNN checkpoint into this model.

The configured batch size is global across GPUs. GPU memory is **not pooled**.
DataParallel overhead or HDF5 reads may make this compact model slower on two
GPUs than on one. Compare in a separate output directory:

```python
single_gpu = KaggleExperiment(
    config="/kaggle/working/pnpl-decoder/config_conformer.json",
    data="/kaggle/working/pnpl-data",
    output="/kaggle/working/conformer-single-t4",
)
single_gpu.train(devices=1)
```

There is no need to prepare the same data again.

Inspect `conformer-001/validation_report.json` before submission. Compare against
the GPU baseline's raw BAcc@10 **27.42%** and corrected **31.24%** on the same
split. Validation also selects checkpoints/prior correction, so it is not an
independent test estimate. See [CONFORMER_EXPERIMENT.md](CONFORMER_EXPERIMENT.md)
for architecture, caveats, and next experiments.

## 8. Generate the submission after training

```python
submission = run.predict(csv="/kaggle/working/submission_conformer.csv")
print(submission)
```

This writes `/kaggle/working/submission_conformer.csv`.
Expected output: **960 rows, 50 primary scores**.

Download the CSV from notebook outputs and upload it through the competition's
**Submit Predictions** page:
<https://www.kaggle.com/competitions/pnpl-competition-2026-deep>.
Accept the rules and check that submissions are open. Manual upload requires
no API token in your notebook. Verify scoring status after upload.

**Always regenerate the CSV after training:** an old prediction file does not
update when model weights change.

## 9. Preserve the experiment and resume safely

Before ending the session, save a notebook version with outputs and preserve
the entire experiment directory, including:

- `best.pt`
- `latest.pt`
- `run_config.json`
- `history.json`
- `validation_report.json`

For an interrupted run, restore its output directory and use:

```python
run.train(devices=2, resume=True)
```

Keep the configuration and training data unchanged when resuming. Do not load
untrusted `latest.pt` files. Do not assume `/kaggle/working` survives a new
session.

To save GPU quota, prepare data in a CPU session and preserve it as a private
Kaggle Dataset. Attached `/kaggle/input` data is read-only; do not prepare into
that directory. HDF5 cache relocation requires preserving the relative raw
file layout under the data root. Legacy epoch caches store absolute paths and
need those paths restored or explicitly rebased.

## Repeatable development loop

1. Edit model/config/tests locally.
2. Run tests, commit, and push.
3. Load that commit in a fresh Kaggle session.
4. Reuse compatible prepared data and use a new output directory.
5. Train and inspect validation results.
6. Generate fresh predictions, submit, and preserve artifacts.

Freeze validation groups when expanding training data so comparisons remain
meaningful. Do not use the reserved final-test data during model development.
