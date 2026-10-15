# pnpl-decoder

**Latest Kaggle sweep:** import [kaggle_spatial_next.ipynb](kaggle_spatial_next.ipynb)
and follow [SPATIAL_NEXT_PLAN.md](SPATIAL_NEXT_PLAN.md). It stages prediction
analysis, spatial seeds/balancing, normalization/dropout, data expansion,
multi-scale and geometry candidates, then local ensembles. Submission is opt-in.

**Start here for new experiments:** [RESEARCH_PLAN.md](RESEARCH_PLAN.md).
The current emphasis is domain generalization and controlled experiments, not
larger models selected on narrative-only validation. `study.py` creates configs,
audits development data, evaluates early checkpoints and raw ensembles;
`research.py` implements label-blind preprocessing and per-domain diagnostics.

Completed runs and model-decision notes: [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md).

Develop code in this Git repository; **run training in a Kaggle Notebook with GPU T4 x2** for [PNPL Competition 2026: Deep](https://www.kaggle.com/competitions/pnpl-competition-2026-deep).
Predict 50 word scores from one `(306, 250)` MEG window. Primary metric: macro per-word recall at top 10 (chance 20%); top-1 balanced accuracy breaks ties.

## What this repo does

This repo is deliberately narrow:

- prepare a clean, reproducible Deep-track training subset,
- train one auditable baseline model,
- write a valid Kaggle submission CSV,
- make it obvious where to swap in stronger models.

It is **not** trying to be a general-purpose framework yet.

## Kaggle setup and daily workflow

For the latest model, follow [KAGGLE_QUICKSTART.md](KAGGLE_QUICKSTART.md): it prepares
expanded training data and trains `config_wide_cnn_expanded.json` in a separate output
directory. The short workflow below remains the original CNN baseline.

Create a Kaggle Notebook, select **Settings → Accelerator → GPU T4 x2**, and enable **Internet**. Account verification and available quota may be required. Keep Kaggle's preinstalled CUDA-enabled PyTorch: **do not install a CPU-only torch build**.

Import `kaggle_notebook.ipynb` into Kaggle, or start with these cells:

```python
from pathlib import Path
import subprocess
import sys

repo = Path('/kaggle/working/pnpl-decoder')
if not repo.exists():
    subprocess.run(['git', 'clone', 'https://github.com/yjoechuen/pnpl-decoder.git', str(repo)], check=True)
sys.path.insert(0, str(repo))
from kaggle_runner import sync_repo, KaggleExperiment
sync_repo(ref='main')  # Prefer an exact commit SHA for a recorded experiment.

import torch
print(torch.__version__, torch.cuda.device_count())
assert torch.cuda.is_available(), 'Enable GPU in notebook settings'
run = KaggleExperiment(output='/kaggle/working/experiment-001')
```

Run each stage in a separate cell, **in this order**:

```python
run.prepare()
```

```python
run.train(devices=2)  # Use devices=1 for comparison or if only one GPU is available.
```

```python
submission = run.predict()  # Always regenerate after training finishes.
```

Download `/kaggle/working/submission_deep.csv` from notebook outputs and use Kaggle's **Submit Predictions**. No API token is needed for manual upload. Never put tokens in notebook source; optional HF authentication belongs in Kaggle Secrets.

### Edit locally → push → train remotely

1. Edit model/config/tests locally, run tests, commit, and push your branch.
2. In a fresh Kaggle notebook session, set `sync_repo(ref='your-branch-or-commit')` to load that code.
3. Reuse the prepared data when unchanged; choose a **new output directory per experiment**.
4. Train, inspect local validation, then generate predictions from the selected checkpoint.
5. Save a Kaggle notebook version with outputs, and download `best.pt`, `latest.pt`, configuration and reports before ending the session.

`sync_repo` refuses to overwrite local edits and only performs fast-forward branch updates. Restart the notebook Python session when updating helper code; training itself runs in fresh subprocesses so model edits do not suffer from stale imports. A private repo requires authentication configured separately—never embed credentials in the clone URL.

### Using both T4s

`run.train(devices=2)` or `pnpl-decoder train --device cuda --devices 2` explicitly enables **DataParallel** across both T4s. The configured batch size is **global**, split between GPUs; GPU memory is not pooled. This small CNN may actually run faster on one GPU because replication/communication overhead dominates. Benchmark both before using two GPUs for larger models. DistributedDataParallel and mixed precision are not yet implemented.

Checkpoints use plain model keys, so two-GPU-trained weights work for single-GPU inference. Resume trusted local checkpoints with `run.train(devices=2, resume=True)` using the same configuration and training data. Save/copy the entire output directory before a session ends; `/kaggle/working` is not guaranteed to persist across sessions. Do not load untrusted `latest.pt` files.

### Data and storage

Default downloads are a small subset, not the full dataset. `run.prepare()` defaults to disk-efficient HDF5 indexing (`--index-only`): it avoids storing duplicated epoch arrays and writes `samples_h5.csv`. Training selects that index automatically if no epoch cache exists. Use `run.prepare(index_only=False)` for the faster, larger epoch cache. HDF5 reading can become the GPU bottleneck; benchmark both within your storage budget.

Prepare/cache data in a CPU session to save GPU quota, then attach a **private Kaggle Dataset** containing that cache for subsequent runs. Attached `/kaggle/input` data is read-only: don't run preparation there. HDF5 indices use relative paths for recordings under the cache root, allowing relocation. The original epoch cache stores absolute paths, so moving it requires copying it back to its original `/kaggle/working` path or explicitly rebasing its `samples.csv` paths. Keep validation groups fixed when expanding training data.

### Local development

Python 3.10+; install a PyTorch build appropriate for your local hardware, then:

```bash
python3 -m pip install -e '.[dev]'
python3 -m pytest -q
```

Local CPU tests cover pipeline behavior. CUDA smoke tests run only when GPUs are available. One/two-GPU checkpoint compatibility is covered; actual two-T4 throughput must be verified in Kaggle.

Fixed seeds and deterministic-kernel preferences are enabled. Unsupported deterministic CUDA operations (such as adaptive-pooling backward) emit warnings instead of stopping training. GPU runs are not guaranteed to be bitwise reproducible. The tests include a CUDA forward/backward/optimizer smoke test, skipped on CPU-only systems.
The final command uploads to the requested competition and requires Kaggle authentication and accepted rules. Set `KAGGLE_API_TOKEN` securely in your shell/environment; **never commit credentials**. Check the competition's current deadline and submission allowance before uploading. This workspace initially has neither GPU nor Kaggle credentials.

## Evidence-based model, not a claimed winning solution

**Current candidate:** [wider residual CNN and data expansion](WIDE_CNN_EXPERIMENT.md).
Use `config_wide_cnn.json` to compare on existing data, then
`config_wide_cnn_expanded.json` for extra Sherlock/Moth training sessions with
the original validation groups frozen. Both reuse the baseline preprocessing.

**Current objective: raw validation BAcc@10.** Checkpoints/early stopping use
raw scores. Inference now defaults to raw; use CLI `--prior-corrected` or notebook
`predict(prior_corrected=True)` only for an explicit correction experiment.
The wider CNN reached a reported interim 32.96% raw score at epoch 13 versus
27.42% for the GPU baseline. This is not a verified Kaggle improvement.

**New GPU candidate:** [multi-scale CNN–Conformer experiment](CONFORMER_EXPERIMENT.md),
configured in `config_conformer.json`. The original CNN remains the default so
old checkpoints continue to load. The candidate reuses the baseline data/split;
it is implemented and CPU-tested, not yet GPU-benchmarked or scored on Kaggle.

The encoder is a small sensor-projection CNN followed by four residual, dilated depthwise temporal convolution blocks, GroupNorm, GELU and four-bin temporal pooling. It preserves onset-relative latency and predicts individual unknown-label test trials—**no same-label trial averaging**. Training uses AdamW, cosine decay, gradient clipping, square-root inverse-frequency cross-entropy weights, sensor dropout and short time masking. Independent per-window/channel normalization is identical at training and inference and respects the very small physical units of MEG.

Convolutional temporal backbones and ensembles were effective in the [organizers' 2025 competition retrospective](https://libribrain.com/publications/2025-pnpl-competition-reflections.pdf). The [2025 MEGConformer phoneme winner](https://www.isca-archive.org/odyssey_2026/zuazo26_odyssey.pdf) supports local convolutional extraction and per-window normalization. This implementation is a lightweight **inspired baseline**, not a reproduction of either system; their tasks/averaged-trial results differ from 2026 single-trial word classification.

## Repository layout

```text
pnpl-decoder/
├── decoder.py          # CLI + data prep + training + inference
├── config.json         # baseline subset and hyperparameters
├── kaggle_runner.py    # notebook helpers: sync, prepare, train, predict
├── kaggle_notebook.ipynb # importable Kaggle entry point
├── tests/test_decoder.py
└── README.md
```

The main extension points live in `decoder.py`:

- `select_training_records(...)`: expand from the small baseline subset to more official train data.
- `build_model(...)`: replace the current temporal CNN with your next architecture.
- `augment(...)`: add or remove training-time augmentations.
- `choose_prior_alpha(...)`: adjust validation-time class-prior correction.

If you want to explore multiple models, the cleanest next step is to keep the preparation pipeline fixed and only branch inside `build_model(...)`.

## Workflow

### 1. Prepare

`pnpl-decoder prepare`

- downloads only the configured official train runs,
- filters events to the 50-word competition vocabulary,
- extracts `(306, 250)` windows,
- normalizes each window independently,
- writes a reproducible `samples.csv` and per-run cached `.npy` epoch files.

### 2. Train

`pnpl-decoder train`

- loads the cached epochs with memory mapping,
- trains the baseline model,
- tracks validation BAcc@10 and BAcc@1,
- saves `best.pt` and resumable `latest.pt`,
- writes validation predictions and a compact report.

### Disk-efficient preparation and paths

The original epoch cache remains supported and is still the default. To avoid
storing both full recordings and expanded float32 epochs, use HDF5 on-demand
windows. This retains the selected processed H5 recordings and event TSVs,
not raw acquisition files, and normalizes each requested window identically to
the epoch cache. It trades repeated disk reads/normalization for less storage;
it is not remote streaming and does not rotate or delete shards.

```bash
# Metadata only: exact processed-H5/event sizes at the configured revisions.
# No data downloads, training, or preparation-directory creation.
pnpl-decoder prepare --index-only --preflight --disk-budget-gib 20 --headroom-gib 5

# Separate index alongside existing epochs; existing samples.csv stays intact.
pnpl-decoder prepare --index-only --disk-budget-gib 20 --headroom-gib 5
pnpl-decoder train --data-mode h5 --output artifacts/h5-run
```

`--preflight` reports each pinned file's exact remote payload size, totals for
H5/events, and additional download bytes. A same-sized local file is assumed
complete for budgeting, not cryptographically verified; Hugging Face still
handles download metadata and resumption. Existing incomplete downloads are
conservatively charged the full payload, not only the remaining transfer bytes.
If remote metadata is missing/unavailable, preparation fails closed rather than
guessing. `--disk-budget-gib` caps existing plus additional preparation data in
the data/raw directories; training artifacts are outside that budget.
`--headroom-gib` reserves free filesystem space (default 5 GiB), separately on
each filesystem when data and raw paths differ. Headroom also accommodates
indexes, download metadata/scratch, and other overhead; payload totals are not
an exact peak-filesystem prediction. Other concurrent writers can still exhaust disk.

For epoch mode the payload-only preflight cannot know the cache size without
events. Actual preparation downloads the small event TSVs first, counts matching
words, then checks a conservative epoch-cache upper bound before downloading
any H5. It rechecks headroom before each H5. Cache replacement may temporarily
need both old and new files; existing caches are counted conservatively, even
if reusable. Index-only mode allocates no epoch files. The plan is saved in
`download_plan.json`; its local/download flags describe the initial preflight.

Index-only preparation writes `samples_h5.csv` and `data_report_h5.json` without
deleting or replacing existing epochs, `samples.csv`, or `data_report.json`.
Both index formats from new preparation include `h5_file` and `start_sample`.
`train --data-mode auto` prefers existing `samples.csv` epochs, otherwise the H5
index; select `h5` explicitly if both exist. Legacy epoch indexes still work.
H5 paths inside the data directory are relative so copying that directory to
Kaggle works; external `--raw` paths and legacy epoch paths are absolute and
must be kept at the same location or updated after copying. H5 handles are
process-local and limited to four open recordings; CLI loaders use zero workers.

All CLI data/output paths can be overridden. Defaults remain repository-local
with `--profile workspace`. `--profile kaggle` (before the subcommand) defaults
to `/kaggle/working/pnpl-decoder/{data,artifacts}`; installed wheels include the
baseline config. `prepare --raw PATH` separates recordings from the index/cache.
Use a new data directory for a different preparation config; do not rebuild or
delete existing artifacts just to switch storage modes.

```bash
pnpl-decoder --profile kaggle prepare --data /kaggle/working/my-data --index-only --disk-budget-gib 15
pnpl-decoder --profile kaggle train --data /kaggle/working/my-data --data-mode h5
```

### Resume and Kaggle

`pnpl-decoder train --resume` restores trusted local `latest.pt`: model, AdamW
moments, cosine scheduler, Python/NumPy/CPU torch/CUDA RNGs, shuffle generator,
history, best metric and early-stop counter. Resume uses the **same config and
total epoch budget** (including `--epochs`); changing the cosine schedule is not
an exact continuation. New checkpoints also validate a path-independent sample
index fingerprint and embed best weights. Legacy latest checkpoints remain
supported but lack that fingerprint/embedded best state, so keep their companion
`best.pt`. The old weights-only fallback is explicitly a warm start, not full
resume, and records the optimizer/RNG reset in `resume_note.json`.

Checkpoint and JSON writes use same-filesystem temporary files, fsync and atomic
replacement. `latest.pt` is authoritative if interruption occurs between epoch
artifact writes; resume repairs best/history from it. Best tensors are copied
to CPU without moving the live model/optimizer. Completed or early-stopped runs
do not take extra optimization steps on resume; validation artifacts can be
regenerated. A non-resume run refuses an output containing checkpoints.
Resume is at **completed epoch boundaries**, not mid-batch: an interrupted epoch
is replayed from the last checkpoint. Bitwise equivalence is not promised across
devices/library versions or unsupported deterministic CUDA kernels. Preserve
`latest.pt` and the same data/config outside ephemeral Kaggle sessions.

`kaggle_entry.ipynb` only installs a pinned repository commit and invokes the CLI;
it has no duplicate model/training code. After these changes are published,
fill `REPO_COMMIT` with that full 40-character SHA. It deliberately refuses an
unset pin rather than silently installing the older remote commit or a moving
branch. Dependencies other than `pnpl` remain version ranges, not an environment
lock; run versions are recorded. Enable Internet/GPU, set a disk budget, and
restore trusted checkpoints before setting `RESUME=True`. The notebook does not
touch the reserved final test or automatically predict/submit.
Set notebook `CONFIG` to a custom config path used by both preparation and
training, or leave it `None` for the installed baseline. For resume, set
`TOTAL_EPOCHS` to the original `--epochs` override if one was used, not the
remaining epochs; leave it `None` when the config's epoch budget was used.

### 3. Predict

`pnpl-decoder predict`

- downloads the official Deep holdout,
- runs inference in canonical row order,
- writes a Kaggle-ready submission CSV using `pnpl.competition.write_submission`.

### 4. Submit

`pnpl-decoder submit`

- shells out to the Kaggle CLI,
- requires local Kaggle auth and accepted competition rules.

## Data and leakage-safe validation

- Use pinned revisions of the official `pnpl/LibriBrain` and `pnpl/LibriBrain2` repositories. Only subject zero records in the **official train partition** are eligible.
- Default resource-limited subset: 10 Sherlock1 training sessions plus the first six official-train TheMoth sessions. This is **not the full dataset**. `config.json` defines tasks/session limits (`null` means all train sessions of that named task).
- Split about 20% of whole `(task, subject, session)` groups into validation, separately within each corpus, with a fixed seed/hash. All runs of a session stay together. No random trial-level split; overlapping MEG windows cannot cross splits.
- Sherlock1 official validation/test sessions 11/12 are excluded. No holdout labels, stimulus retrieval, audio encoders, external data, or pretrained weights are used.
- Word IDs are explicitly remapped into the official competition vocabulary order; apostrophe typography is normalized without merging `its` and `it's`.
- Cache normalized 1-second float32 epochs on disk with a memory-mapped reader. Cache creation records sample counts, dropped boundary windows and class coverage. Preparation fails if a training class is absent. Missing validation classes are explicitly reported; `bacc_present` excludes missing classes, while `bacc` divides by all 50 and counts missing classes as zero.
- Select best checkpoint on validation BAcc@10; then choose a small prior correction from five strengths using validation predictions. Consequently reported validation scores are model-selection estimates, **not unbiased independent test scores**.
- Session separation is necessary but does not by itself guarantee unseen text across corpora with repeated utterances. The default Sherlock/Moth narrative subset avoids MOCHA's repeated-sentence split complication; audit repeated stimulus IDs if extending to those corpora.

Changing the preparation config requires a new `--data` cache directory (or explicit deletion/rebuild). For larger runs add `Sherlock2` … `Sherlock9`, more `TheMoth`, and subsequently audited TIMIT/MOCHATIMIT data, increase width/epochs and train multiple seeds. Do not assume all >100 hours fit on a small workspace disk.

## Outputs

### Reserved final test (not development feedback)

`final_test.json` reserves the official **Sherlock1 subject 0, session 12, run 2**
test record at a pinned Hugging Face revision. It is excluded from the existing
training/validation manifest and remains excluded when official training data expands.
Prepare it separately, without evaluating a model:

```bash
python final_test.py
```

The default cache is `/workspace/protected/pnpl-final-test`, **outside this repo**.
It contains normalized `(306, 250)` windows, separate labels, vocabulary, row
provenance, and a checksummed manifest. Preparation checks disk headroom and
session disjointness; output paths must not overlap the repository or development
directory in either direction, including resolved symlinks. Reruns verify the
saved cache and restore its protection marker if missing or changed. Installed
packages find the default reservation JSON in `sys.prefix/share/pnpl-decoder`
when it is not beside the module. Training rejects final-test
markers, non-training partitions, and unexpected split labels.

This single audiobook session tests within-corpus generalization, not performance
across all corpora. Class coverage is recorded, not used to adapt the test set.
No model should be scored on it until search is finished and a candidate/config,
checkpoint, preprocessing, and prior correction have been frozen on development
data. Do not feed final-test results back into automated search.

**Protection limits:** the cache directory has owner-only permissions, but same-user
code can still read it, and the source is publicly downloadable. Future generated-code
workers must have no network and no mount of this directory, raw test data, or labels.
Use a separate user/container and a trusted external scorer for real isolation.
This setup supplies accidental-use guards, not that sandbox. No final-test evaluation
command is added yet, deliberately avoiding premature exposure of scores.

`data/split_manifest.json`, `samples.csv`, `vocabulary.json`, `data_report.json`: reproducible splits, label mapping, coverage and provenance.

`artifacts/baseline/best.pt`: model weights, config, vocabulary, train priors and selected correction.
`latest.pt`: resumable local training state.
`history.json`, `validation_report.json`, `validation_predictions.npz`, `validation_rows.csv`: training history and auditable validation predictions.

`artifacts/submission_deep.csv`: primary-only CSV (`index` + 50 words) produced through the official `pnpl.competition.write_submission`; the optional Moses block is deliberately omitted. The official holdout loader defines canonical indices. A sidecar JSON records row counts and checkpoint/submission hashes. No holdout-based model selection.

Raw data, caches, credentials, model artifacts and submissions are Git-ignored; code remains ready to share. Dataset licenses are CC-BY-NC-4.0; retain attribution and consult the [official rules](https://libribrain.com/editions/2026/rules/) and Kaggle rules before any extensions. Checkpoints/training code may be requested for verification.

## Suggested next model experiments

Once this baseline is stable, keep the data split fixed and try changes in this order:

1. wider temporal CNN,
2. deeper residual dilations,
3. CNN front-end + compact Conformer,
4. multi-seed ensembling,
5. broader official-train coverage.

That sequence keeps attribution of gains relatively clean.

Official interfaces: [dataset](https://neural-processing-lab.github.io/pnpl/libribrain100.html), [submission](https://neural-processing-lab.github.io/pnpl/competition.html).
