# Spatial model: evidence and staged next steps

## Evidence and objective

User-reported raw public scores: spatial CNN **28.785%**, small balanced CNN
**22.905%**. Their local raw development ranking was reversed: 28.30% versus
29.13%. Sampling/loss differs as well as architecture. This supports further
spatial-model experiments, not a conclusion that one mechanism caused the gain.
The wide CNN's 32.96% local / 21.152% public run used a different validation set.

Sources reviewed for the plan:

- Mantegna et al., *LibriBrain100* (Aug 2026), https://arxiv.org/abs/2608.25204:
  within-subject depth and stimulus diversity matter; broad data can improve
  some subject-zero domains, not uniformly.
- Mantegna et al., *2026 PNPL Competition* (Sep 2026), https://arxiv.org/abs/2609.03231:
  Deep versus Broad objectives and independent-stimulus holdout.
- Jayalath & Parker Jones, *MEG-XL* (2026), https://arxiv.org/abs/2602.02494:
  long-context masked token pretraining and sensor metadata help low-data
  transfer. Results do not establish superiority with isolated one-second inputs.
- d'Ascoli et al., *Towards decoding individual words from non-invasive brain
  recordings* (2025), https://pmc.ncbi.nlm.nih.gov/articles/PMC12658044/:
  spatial attention + convolutions + semantic contrastive learning; sentence
  context improves their task but is not automatically available here.
- Official rules: https://libribrain.com/editions/2026/rules/.

These are targeted primary-source findings, not a systematic literature review.
No published accuracy transfers directly to this vocabulary/holdout. External
pretrained features require rule/license verification. Full MEG-XL pretraining,
large Transformers, Mamba, semantic/audio targets, domain-adversarial objectives,
and top-k surrogate losses remain lower priority until simpler experiments work.

## Entry point

Import **`kaggle_spatial_next.ipynb`** into a fresh Kaggle notebook. Enable Internet
and GPU; keep CUDA PyTorch. It implements steps 1–8 as separate cells. Preparation
and CPU audits may run before enabling GPUs to conserve quota. Fresh notebooks
cannot access old checkpoints unless you restore private notebook/dataset outputs.

`NextStage` creates missing configs in `experiments/spatial_next/` without replacing
your edits. Completion requires an atomic `completed.json` marker plus all reports;
saved config and data fingerprints are checked before skipping. Interrupted runs
resume from `latest.pt`; only `best.pt` remaining uses the documented weights-only
warm start (history must also be restored). Use a new root for reruns/config
changes. No auto-upload or final-test evaluation. Default is **one GPU**.

## Step 1 — diagnose existing predictions

Restore `study-small_balanced` and `study-temporal_spatial` outputs. Run paired
domain analysis with `next_stage.compare`. It verifies identical rows and labels,
computes B-minus-A recall and grouped bootstrap intervals. Sentence groups use
text hashes, narrative groups use sessions. Few session clusters make intervals
unreliable; resamples can lose class coverage. This is descriptive development
analysis, not evidence from independent repeated subjects.

Inspect domain reports (Sherlock1, held-out Sherlock2, Moth, TIMIT), class counts,
prediction diversity and frequency control. Predefine summaries rather than
choosing domain weights based on public scores. Equal-domain recall and overall
recall answer different questions; always inspect coverage.

## Steps 2–4 — repeatable anchor and one-factor changes

- `anchor`, `seed2027`, `seed2028`: spatial CNN, unchanged split/data/loss,
  independent training seeds. Uniform probability ensemble first.
- `balanced`: spatial architecture with inverse-frequency sampling and ordinary
  CE; compare with weighted-CE natural-sampling anchor. Completes the architecture
  × balancing comparison against the existing small models.
- `dropout`: 0.2 → 0.3, otherwise anchor unchanged.
- `normalization`: training-only channel mean/std, rather than per-window
  normalization; preserves amplitude but may amplify session shift. Statistics
  are saved with weights and never fitted to validation/holdout.

Checkpoint selection still uses raw overall recall. Evaluate saved early epochs
and domains locally before accepting a candidate. No claim that epoch 5 is best.

## Step 5 — stimulus diversity, not just narrative hours

Expand TIMIT from 4 to 10 selected official-train sessions and Moth from 12 to 18.
Validation groups stay fixed, including TIMIT session 4; repeated sentence texts
in validation are removed from training. Confirm retained counts after purging.
Total pinned raw payload is **18.87 GiB**, about **5.30 GiB extra** relative to
the 13.57 GiB development suite. The preparation cap is 24 GiB plus actual
filesystem headroom checks; a small Kaggle output disk may not fit. Attach a
private raw-data dataset or use CPU preparation/storage rather than overriding
checks. Metadata/HF scratch/checkpoints require more space than raw payload.

`expanded` uses natural corpus sampling. `expanded_corpus_balanced` samples
equally from Sherlock, Moth and TIMIT groups (not each Sherlock book separately),
while retaining the same class-weighted loss. This is a domain-exposure experiment,
not class balancing. Isolated-word holdout remains a distinct regime.

## Step 6 — controlled multi-scale spatial model

`MultiScaleSpatialCNN`: 32 total temporal banks split 10/12/10 at kernels
15/31/63, followed by the existing spatial/temporal backbone. Bank count is held
constant; parameter/FLOP budgets are close but not identical. Compare against
`expanded` with identical data/sampling. Single-trial inputs only.

## Step 7 — geometry-aware independent candidate

`GeometryCNN` uses smooth coordinate-initialized spatial mixtures with learned
deviations, temporal residual blocks, and within-window attention pooling.
This is **BrainModule-inspired**, not an exact published reproduction. It is
implemented, but the experiment requires verified sensor geometry.

Export from a permitted subject-zero recording and its matching serialised H5:

```python
# Install mne only if needed; no pretrained model or stimulus retrieval.
from next_stage import export_geometry
export_geometry('matching_recording.fif','matching_serialised_meg.h5',
                '/kaggle/working/sensor_geometry.json')
```

Exporter reads FIF header metadata, checks mag/grad types, matches H5 names, and
writes nondegenerate coordinates. Training independently verifies the exact channel
order across every indexed recording; coordinate names cannot self-certify their
ordering. Geometry names and the spatial prior are saved in checkpoints. Resume
and inference do not need the original geometry JSON.
Do not invent positions. Metadata alone is not complete certification of head
position coordinate consistency; use the same reference system/preprocessing.
The geometry cell explicitly reports a pending prerequisite when absent. It
does not claim that this stage ran. Model buffers preserve positions-derived
priors so inference needs no external geometry file.

## Step 8 — local ensembles then limited confirmation

The notebook evaluates a predefined uniform three-seed ensemble. Additional
combinations require explicit local selection. Development evaluation currently
requires identical prepared data configs; use the anchor ensemble or expanded
architecture ensemble, not mixed-data checkpoints in that helper. Submission
generation defaults OFF and produces raw probabilities when explicitly enabled.
No automated public-leaderboard search. Preserve hashes/configs and record all
experiments, including regressions.

## Throughput and reproducibility

- Two loader workers, spawn-safe H5 handles, deterministic worker seeds,
  persistent workers and prefetching; pinned memory/nonblocking CUDA transfers.
- Sampler RNG is independent of worker/iterator RNG, preserving epoch-boundary
  sample order on resume. Current worker preprocessing is deterministic; random
  augmentation runs in the parent. Future stochastic worker transforms would
  require additional restart-aware seeding.
- Optional FP16 autocast and gradient scaling. Scaler state is checkpointed.
  Anchor is rerun because AMP is not guaranteed equivalent to old FP32 execution.
- Explicit full CSV parsing removes the mixed-type warning.
- History now records training/validation duration, loader-wait time and training
  throughput. Loader wait is host timing, not a CUDA profiler breakdown.
- Keep batch size unchanged for performance benchmarks: changing it also changes
  optimizer steps. Compare one/two GPUs in separate roots. DataParallel overhead
  is significant; low sampled utilization alone does not prove a bottleneck.

T4s support FP16; unscaled/nonfinite gradients still require investigation. Set
`amp=false` to debug. Strict bitwise reproducibility is not promised. Start with
the supplied CUDA smoke tests before expensive training. Current verification is
CPU/synthetic only; no new public score or full GPU sweep has been run here.

Success means repeatable improvement, understood per-domain tradeoffs, useful
sentence-domain results, and complementary errors—not one local/public maximum.
Our best reported public score remains 28.785%; no next score is promised.
