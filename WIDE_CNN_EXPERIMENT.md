# Wider residual CNN and modest data expansion

The Conformer learned slowly in the reported run. This candidate returns to
the baseline's sensor projection and residual temporal convolutions rather
than adding attention. It is implemented and CPU-tested, **not yet GPU-scored**.

## Model

- 128-feature sensor projection (baseline: 48).
- Six residual blocks, dilations `[1, 2, 4, 8, 1, 2]`.
- Expanded pointwise feature mixing and depthwise 7-sample temporal convolution.
- GroupNorm, GELU, dropout and half-strength residual updates.
- Eight onset-relative pooling bins and a linear 50-word head.
- **503,474 parameters**, versus 35,810 baseline and 1,798,579 Conformer.

No external data, audio/text inputs, pretrained weights, attention, or trial
averaging. Normalization, loss weighting, augmentation and prior correction
remain those of the baseline. Equal-bin means avoid adaptive-pooling backward.
Smaller parameter count does not guarantee faster runtime or higher accuracy.

## Two experiments

1. `config_wide_cnn.json`: existing 16 sessions; reuse the prepared baseline
   cache and measure the model/training-budget change alone.
2. `config_wide_cnn_expanded.json`: 34 sessions: Sherlock1 train sessions,
   all 12 Sherlock2 sessions, and the first 12 official-train Moth sessions.

Both explicitly reserve the exact original validation groups:
Sherlock1 subject 0 sessions 9 and 10, plus Moth subject 0 session 4.
All added sessions enter training. Official test/validation partitions stay
excluded, including Sherlock1 session 12 reserved for final evaluation.
The expanded configuration has 31 training sessions and three validation
sessions; exact window counts are computed during preparation, not guessed.
The validation set is still small and narrative-only, so it does not solve
the known holdout-distribution gap. Repeated training seeds require keeping
these fixed groups, not regenerating a random validation split.

Both configurations use 24 epochs, patience 6, global batch 128, AdamW LR 1e-3.

## Kaggle: model-only comparison first

After updating code and restarting notebook Python imports:

```python
from kaggle_runner import KaggleExperiment

small = KaggleExperiment(
    config='/kaggle/working/pnpl-decoder/config_wide_cnn.json',
    data='/kaggle/working/pnpl-data',
    output='/kaggle/working/wide-cnn-small-001',
)
small.train(devices=1)  # Benchmark against devices=2 in another output directory.
```

No re-preparation is needed: the explicit groups reproduce the legacy split.

## Expand data without redownloading the original subset

The new index lives separately; raw files are shared with the existing cache:

```python
expanded = KaggleExperiment(
    config='/kaggle/working/pnpl-decoder/config_wide_cnn_expanded.json',
    data='/kaggle/working/pnpl-expanded',
    output='/kaggle/working/wide-cnn-expanded-001',
)
raw = '/kaggle/working/pnpl-data/raw'
expanded.prepare(raw=raw, preflight=True, disk_budget_gib=16)
expanded.prepare(raw=raw, disk_budget_gib=16)
expanded.train(devices=2)
```

Pinned remote-size inspection found **11.05 GiB total raw data**, about
**6.14 GiB additional** if the original raw subset is already present.
Filesystem headroom defaults to another 5 GiB for download scratch and other
files. The 16 GiB preparation budget is a cap, not an assertion that Kaggle
has space; preparation checks actual free space. HDF5 indexing avoids duplicate
epoch storage. Shared raw paths outside the new cache are absolute: preserve
that layout when reusing caches in later sessions.

No large MEG download was initiated in the development workspace; download
and training are intentionally performed in your Kaggle notebook.

## Validation and submission

Compare raw and corrected validation BAcc@10 against baseline 27.42%/31.24%,
plus per-class recall. Use a new output directory and do not resume the
Conformer or narrow CNN into this architecture. Only submit a promising model:

```python
expanded.predict(csv='/kaggle/working/submission_wide_cnn.csv')
```

The larger model and longer epoch budget are both changes; an expanded-data
run additionally changes training volume. Keep the model-only run to understand
which intervention helps. No drastic gain is promised. Preserve reports and
weights before terminating the notebook session.
