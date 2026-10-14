# Generalization-first experiment workflow

The wider CNN achieved 32.96% raw development recall but 21.152% publicly,
versus 24.071% for the older **corrected** baseline submission. Prediction mode,
data and architecture differ: this does not isolate overfitting. Stop selecting
ever-larger models on one narrative-only split.

## Implemented tools

- `models.TemporalSpatialCNN`: 32 shared temporal filter banks, two spatial
  filters per bank, three compact residual temporal blocks, eight latency bins.
  Temporal patterns are extracted **before** bank-specific spatial mixing.
  It has **100,306 parameters** in the supplied configuration.
- `experiments/`: generated controlled configs; `training_seed` is separate from
  split `seed`, and explicit validation groups remain fixed.
- Natural sampling + weighted CE versus inverse-frequency sampling + **ordinary
  CE**. Double balancing is rejected.
- Window/channel z-score versus **training-only channel statistics**, fitted
  from up to 512 deterministic training windows and saved in checkpoints.
- Fixed-epoch weights (`epoch_003.pt`, `epoch_005.pt`, `epoch_008.pt`,
  `epoch_013.pt`) without changing the original cosine schedule. Early stopping
  can prevent later checkpoints from being reached.
- Domain/session raw recall, class coverage, frequency-only no-MEG control,
  prediction diversity, validation loss, online training recall, and clean
  training evaluation at the selected checkpoint.
- Training-label-shuffle negative control. This tests whether the pipeline
  learns useful label-dependent signal; not a proof that all leakage is absent.
- Local single-checkpoint and mean-probability ensemble evaluation. Optional
  uniform raw ensemble submission after local selection, never holdout tuning.

## Validation suites

`experiments/legacy_*.json` reuse the expanded cache with its original three
validation sessions. Use them for inexpensive first comparisons only.

`experiments/development_data.json` uses:

- Original Sherlock1/Moth session validation.
- **All Sherlock2 sessions as held-out-book development validation.** Retrain
  models on this split: old expanded-data weights already trained on Sherlock2
  and cannot be evaluated as an unseen-book model.
- TIMIT sessions 1–3 training and session 4 sentence validation. Repeated full
  normalized sentence texts are hashed and removed from training if present in
  sentence validation, including repeats across speakers.

This suite includes only **official training-partition** records; we make our
own development split. The reserved final test stays excluded. MOCHA is not
added yet because repeated stimulus auditing should precede its inclusion.
Sentence validation is closer to one holdout regime, **not a proxy for isolated
word trials**, and no unseen-book guarantee extends beyond the named corpus.

Pinned raw payload total: approximately **13.57 GiB** (roughly 2.51 GiB more than
the 11.05 GiB expanded raw subset). Exact counts/class coverage are computed
during preparation. Inspect missing classes and `bacc_present`; fixed-50 `bacc`
counts absent classes as zero. Never silently compare domains with different
coverage. Do not score the final test until model choices are frozen.

## Kaggle cells

Load updated code in a fresh session using the quickstart setup. Generate
configs if changing the suite generator:

```python
import subprocess, sys
repo='/kaggle/working/pnpl-decoder'
subprocess.run([sys.executable,f'{repo}/study.py','configs','--output',f'{repo}/experiments'],check=True)
```

Prepare a **new** index directory, sharing raw payloads:

```python
from kaggle_runner import KaggleExperiment
development=KaggleExperiment(
    config=f'{repo}/experiments/development_data.json',
    data='/kaggle/working/pnpl-development',
    output='/kaggle/working/development-reference')
development.prepare(raw='/kaggle/working/pnpl-data/raw',preflight=True,disk_budget_gib=18)
development.prepare(raw='/kaggle/working/pnpl-data/raw',disk_budget_gib=18)
```

The preparation cap is not an assumption about free space; leave 5 GiB default
filesystem headroom, and preserve cache path layout between sessions.

Audit data without looking at holdout labels:

```python
subprocess.run([sys.executable,f'{repo}/study.py','audit',
    '--data','/kaggle/working/pnpl-development',
    '--output','/kaggle/working/development-audit.json'],check=True)
```

Optional `--holdout-data PATH` reads only whitelisted structural arrays from
already downloaded competition NPZs. It does not score or fit statistics.
Metadata checks are **not certification of preprocessing compatibility**:
channel ordering/filtering/onset details must be compared with organizer
documentation when the holdout lacks those fields. Workspace sample audit
found float32, 306 channels, 250 Hz, physical MEG scales and valid bounds.
Channel-name/type attributes exist; magnetometers/gradiometers have different
units. Do not mix raw amplitudes without per-channel scaling.

Run these first, each in a new output directory:

```python
def experiment(name):
    return KaggleExperiment(config=f'{repo}/experiments/{name}.json',
        data='/kaggle/working/pnpl-development',output=f'/kaggle/working/study-{name}')

experiment('small_reference').train(devices=1,data_mode='h5')
experiment('small_balanced').train(devices=1,data_mode='h5')
experiment('temporal_spatial').train(devices=1,data_mode='h5')
```

Next compare `reference`, `balanced`, `dropout`, `train_normalization`, and
`shuffled_control`. GPU memory/throughput for the temporal-first model must be
measured; the first convolution creates bank × sensor × time activations, so
parameter count alone does not predict memory. Reduce batch size in a copied
config if needed. Two-GPU execution is optional, not automatically faster.

Do not run every experiment blindly: record per-domain/class results and stop
unpromising branches. Raw overall recall currently selects checkpoints for
compatibility; it is dominated by larger domains. Also evaluate retained early
epochs, and compare domains individually before accepting a model.

```python
subprocess.run([sys.executable,f'{repo}/study.py','evaluate',
    '--data','/kaggle/working/pnpl-development',
    '--checkpoints','/kaggle/working/study-temporal_spatial/epoch_005.pt',
    '--output','/kaggle/working/spatial-epoch5','--device','cuda'],check=True)
```

The evaluator checks data selection/split/vocabulary compatibility; it will
reject old expanded-data checkpoints on the held-out-book suite. Use `legacy_*`
configs for current-cache retrospective analyses.

## Multi-seed confirmation and ensembles

`seed2027`/`seed2028` repeat the **reference** with different training RNG seeds
and unchanged data selection. After choosing a different architecture, copy its
config and change only `training_seed`. Keep identical preprocessing/validation
policy for comparisons, and do not modify resumed run configs.

```python
subprocess.run([sys.executable,f'{repo}/study.py','evaluate',
    '--data','/kaggle/working/pnpl-development',
    '--checkpoints','/kaggle/working/run-seed1/best.pt','/kaggle/working/run-seed2/best.pt',
    '--output','/kaggle/working/ensemble-evaluation','--device','cuda'],check=True)
```

Default ensemble weights are uniform; manually supplied `--weights` must be
chosen using development only. Scores remain optimistic after repeated local
selection. A promising uniform ensemble can produce a CSV via `study.py
predict-ensemble --checkpoints ... --data HOLDOUT_CACHE --csv OUTPUT.csv`.
Single models still use `run.predict()` with raw defaults.

## Reproducibility and limitations

The synthetic integration test exercises balancing, saved training statistics,
training, early checkpoints, domain reports and ensemble evaluation. CPU
tiny-batch tests verify optimization/checkpoint round-trips; CUDA tests skip
without GPUs. No new accuracy claims follow from these tests. No large new
payload download, competitive GPU training, or final-test scoring was performed
in the workspace. This is the infrastructure to run those experiments reliably.
