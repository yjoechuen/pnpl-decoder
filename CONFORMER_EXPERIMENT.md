# Experiment: multi-scale CNN–Conformer

This is a stronger **candidate**, not a demonstrated improvement. No GPU or
Kaggle score has been measured for it yet. The aim is to beat the small CNN
under the same train/validation split before scaling data or ensembling.

## Architecture

`models.py` implements `MultiScaleConformer`:

1. Learned projection from 306 sensors to 128 features.
2. Parallel depthwise temporal convolutions with kernels 5, 15 and 31 samples.
3. Learned fusion and stride-4 downsampling to 63 temporal tokens.
4. Sinusoidal onset-relative position encoding.
5. Four Conformer blocks: half-step feedforward, temporal self-attention,
   GLU/depthwise convolution, half-step feedforward, and normalization.
6. Attention pooling plus four latency-preserving mean-pooling bins.
7. A 50-word classification head.

Convolutional frontends and Conformer-style encoders are motivated by successful
MEG decoding systems discussed in the README. This is not a reproduction of a
published winner. GroupNorm replaces batch-dependent convolution normalization.
No pretrained model, audio/text features, label-based averaging, or cross-trial
context is used.

The candidate has **1,798,579 parameters** versus **35,810** in the original
CNN. Extra capacity may improve representation learning or overfit this small
subset; validation must decide. Expect slower training than the baseline.

## Configuration and fair comparison

`config_conformer.json` uses the **same tasks, seed, split fraction and pinned
data revisions** as `config.json`. Existing prepared data can be reused for
training. Do not call preparation with the new model config on the old cache:
preparation enforces full-config equality, while training intentionally checks
only data-related settings.

Candidate settings: 30 epochs, patience 8, global batch size 64, AdamW learning
rate 3e-4, dropout 0.2, weight decay 0.01. The model and training-budget changes
are both part of this candidate, so this is not a pure architecture ablation.

## Kaggle notebook cells

First load the updated repo following `KAGGLE_QUICKSTART.md`. Use the commit
containing these files or the appropriate development branch, rather than the
older pinned baseline commit.

Prepare with the original data configuration if needed:

```python
from kaggle_runner import KaggleExperiment

data_run = KaggleExperiment(data='/kaggle/working/pnpl-data')
data_run.prepare()
```

Verify the new model on actual CUDA hardware:

```python
import subprocess, sys

subprocess.run(
    [sys.executable, '-m', 'pytest', '-q',
     'tests/test_models.py::test_conformer_cuda_training'],
    cwd='/kaggle/working/pnpl-decoder', check=True,
)
```

Train in a new output directory; do not resume a baseline checkpoint into this
architecture:

```python
candidate = KaggleExperiment(
    config='/kaggle/working/pnpl-decoder/config_conformer.json',
    data='/kaggle/working/pnpl-data',
    output='/kaggle/working/conformer-001',
)
candidate.train(devices=2)
```

Start with one GPU if troubleshooting memory or two-GPU overhead. If necessary,
lower `batch_size` in a copied experiment config before training. Both GPUs
share a global batch, not combined memory. Mixed precision is not implemented.

Inspect validation before submission:

```python
import json
from pathlib import Path

report = json.loads(Path('/kaggle/working/conformer-001/validation_report.json').read_text())
print('Best epoch:', report['best_epoch'])
print('Raw BAcc@10:', report['raw_top10']['bacc'])
print('Corrected BAcc@10:', report['corrected_top10']['bacc'])
print('Corrected BAcc@1:', report['corrected_top1']['bacc'])
```

Reference GPU baseline: raw BAcc@10 **27.42%**, corrected **31.24%**.
Earlier CPU baseline: corrected **32.34%**, Kaggle public **24.071%**. These are
not interchangeable evaluation sets. Prior correction and checkpoint selection
both use validation, so the corrected score is not an independent test estimate.
Examine per-class/corpus performance and repeated seeds before interpreting a
small gain as reliable. Do not evaluate the reserved final-test set during search.

If the candidate warrants a submission:

```python
candidate.predict(csv='/kaggle/working/submission_conformer.csv')
```

Download and manually submit the freshly generated CSV. Preserve checkpoint,
run config, validation predictions and reports. Avoid tuning against repeated
Kaggle feedback; improve locally first.

## If the larger model does not improve

- Check overfitting: falling train loss with flat/falling validation scores.
- Compare a smaller width/layer count and stronger dropout.
- Expand diverse official training data while freezing validation groups.
- Compare one/two-GPU speed; HDF5 disk reads can dominate runtime.
- Repeat promising configurations with independent training seeds while keeping
  the validation manifest unchanged (the current seed also controls the split).

No specific accuracy increase is promised.
