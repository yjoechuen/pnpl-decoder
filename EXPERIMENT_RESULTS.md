# Experiment results and decisions

## Wide residual CNN — expanded Sherlock/Moth training, run 01

**Source:** completed Kaggle logs supplied by the user. Results were not
independently recomputed. Exact executed Git SHA, checkpoint hash and environment
versions were not supplied; preserve the Kaggle `run_config.json`, `best.pt`,
validation predictions and row metadata to complete provenance. Do not infer
the executed commit from the current repository HEAD.

Machine-readable history and metrics:
[results/wide_cnn_expanded_run01.json](results/wide_cnn_expanded_run01.json).

### Setup

- Configuration: `config_wide_cnn_expanded.json`; 503,474-parameter residual CNN.
- Training: 37,834 windows, 31 sessions. Validation: 2,389 windows, 50 classes.
- Frozen validation: Sherlock1 subject 0 sessions 9/10; Moth subject 0 session 4.
- Requested execution: Kaggle CUDA, two devices; global batch 128.
- Maximum 24 epochs; patience 6; raw BAcc@10 selects the checkpoint.
- Output directory: `/kaggle/working/wide-cnn-expanded-001`.

### Completed results

| Measure | Result |
|---|---:|
| Best raw BAcc@10 | **32.9599959962%**, epoch 13 |
| Raw BAcc@1 at selected epoch | **5.0875013873%** |
| Highest raw BAcc@1 in run | **5.4948379320%**, epoch 9 |
| Last epoch | 19: six consecutive epochs without improvement |
| Last raw BAcc@10 | 28.5911447491% |
| Training loss, epoch 1 → 19 | 3.7682 → 1.8648 |
| Corrected BAcc@10, diagnostic only | 35.6767509863%, alpha 0.75 |
| Corrected BAcc@1, diagnostic only | 5.0020296550% |
| Kaggle score for wider-CNN submission | **21.152%**, user-reported |

Raw performance is the current objective. Correction is retained for traceability,
not used to judge this experiment. New prediction code defaults to raw even if
this checkpoint stores `prior_alpha=0.75`.

### Interpretation and next decisions

1. **Keep epoch 13, not epoch 19.** Later training reduced training loss while
   validation deteriorated. This is consistent with overfitting, not definitive
   proof: training recall and validation loss were not logged in this run.
   New code records these diagnostics for future runs.
2. **The joint intervention improved raw validation.** The earlier GPU baseline
   reached 27.4184%; this run improves by **5.5416 percentage points**. Model,
   data volume and epoch budget all changed, so the gain cannot be attributed
   solely to architecture. Compare the wider model on the original small subset.
3. **Confirm before scaling.** Repeat with another training seed and fixed
   validation groups; test dropout 0.3. Do not extend training just because
   training loss continues falling.
4. **Investigate rare-word failures.** Seven classes have zero raw top-10 recall
   at the selected epoch. Join the saved competition vocabulary and validation
   counts to inspect these classes; very small class counts make recall noisy.
5. **Do not promise leaderboard gains.** Validation is small, narrative-only,
   and repeatedly used for selection. The user reported **21.152%** for
   `submission_wide_cnn.csv`, versus 24.071% for the corrected older baseline.
   The wider submission's mode is presumed raw but not verified from its sidecar;
   its exact checkpoint hash was not supplied. This comparison changes both
   model/data and potentially prediction mode. Keep
   the reserved final-test set outside development feedback.

### Earlier reference results

| Run | Raw validation BAcc@10 | Status |
|---|---:|---|
| Small CNN, GPU | 27.4183526852% | Completed; epoch 12 selected |
| CNN–Conformer | 22.6448925434% at epoch 14 | Interim logs only |
| Wide CNN, expanded | **32.9599959962%** | Completed; epoch 13 selected |

The earlier CPU baseline submitted to Kaggle scored 24.071% using correction.
That is a different checkpoint and prediction mode, not a raw holdout reference
for this model.

## Recording future experiments

Record exact config, executed commit, package versions, data fingerprint,
checkpoint hash, history, vocabulary, validation predictions/rows and prediction
mode. Record unsuccessful runs too. Separate local model-selection scores from
independent holdout results. Machine-readable entries should explicitly mark
unavailable metadata, rather than guess it.

## Development suite on held-out-book + sentence validation

**Source:** user-supplied Kaggle logs from the generalization-first development
suite. These are local development results, not Kaggle leaderboard scores.
The executed commit was likely recent `main`, but no exact SHA, checkpoint hash,
or sidecar JSON was supplied, so they are left unknown.

Machine-readable transcription:
[results/development_suite_run01.json](results/development_suite_run01.json).

### Shared setup

- Development split: 22,063 training windows and 21,499 validation windows.
- Validation includes original Sherlock1/Moth sessions plus held-out Sherlock2
  and sentence-domain validation, making it much harder than the older three-session split.
- Execution shown in logs: `cuda × 2`, `data_mode="h5"`, batch size 128.
- Current objective: **raw** BAcc@10.

### Results recorded so far

| Run | Best epoch | Best raw BAcc@10 | Best raw BAcc@1 | Runtime per epoch | Status |
|---|---:|---:|---:|---:|---|
| `small_reference` | 7 | **28.00%** | 3.28% | 75–80 s | completed |
| `small_balanced` | 3 | **29.13%** | 4.30% | 71–80 s | completed |
| `temporal_spatial_cnn` | 7 | **28.30%** | 3.54% | 186–194 s | completed |

### Public Kaggle scores from this suite

| Submission | Development source | Prediction mode | Kaggle score |
|---|---|---|---:|
| `submission_small_balanced_raw.csv` | `small_balanced`, epoch 3 | raw | **22.905%** |
| `submission_temporal_spatial.csv` | `temporal_spatial_cnn`, epoch 7 | raw | **28.785%** |

These scores are user-reported Kaggle public results. Exact submission-sidecar
metadata, checkpoint hashes and executed commit SHA were not supplied here.

### Immediate interpretation

1. **Balanced sampling improved the small CNN.** `small_balanced` outperformed
   `small_reference` by about 1.13 raw percentage points on this harder split.
2. **The temporal-first spatial CNN transferred better to Kaggle despite weaker
   local development score.** This reverses the local ranking and is currently the
   strongest empirical result in the repository.
3. **Our current development metric is still mismatched to holdout utility.** The
   model with lower local raw BAcc@10 achieved the best Kaggle score.
4. **This suite is more realistic than the earlier narrative-only split.** Do not
   compare its absolute percentages directly to older 32.96% wide-CNN validation.
5. **Two GPUs appear underutilized for these small HDF5-fed CNNs.** Low utilization
   is consistent with CPU/input bottlenecks rather than insufficient model size.

### Next decisions informed by these logs

- Use `temporal_spatial_cnn` as the current submission candidate and starting
  point for follow-up experiments.
- Prefer **one GPU** for similar CNN runs unless throughput measurements prove otherwise.
- Do **not** deprioritize the temporal-spatial model based on local raw validation alone;
  instead, treat this as evidence that its inductive bias may match the holdout better
  than the current development suite reveals.
- Next local comparisons should focus on **temporal-spatial variants**: balanced sampling,
  dropout, training-only normalization, and additional training seeds.
- Re-examine development reporting: per-domain scores, top-10 diversity, and any
  sentence-vs-isolated-word gap may be more informative than one overall raw score.
- Keep `small_balanced` as a useful fast baseline, but not as the leading model.
