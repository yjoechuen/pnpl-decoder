# pnpl-decoder

Reproducible first baseline for [PNPL Competition 2026: Deep](https://www.kaggle.com/competitions/pnpl-competition-2026-deep).
Predict 50 word scores from one `(306, 250)` MEG window. Primary metric: macro per-word recall at top 10 (chance 20%); top-1 balanced accuracy breaks ties.

## What this repo does

This repo is deliberately narrow:

- prepare a clean, reproducible Deep-track training subset,
- train one auditable baseline model,
- write a valid Kaggle submission CSV,
- make it obvious where to swap in stronger models.

It is **not** trying to be a general-purpose framework yet.

## Setup and run

Python 3.10+; install the appropriate PyTorch build for your hardware first. CPU works for this compact baseline; GPU is recommended for scaling.

```bash
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install -e '.[dev]'
python3 -m pytest -q
pnpl-decoder prepare
pnpl-decoder train
pnpl-decoder predict
pnpl-decoder submit
```

For CUDA, install a CUDA PyTorch build instead of the CPU build; training/inference automatically selects CUDA when available. Use `--device cpu` to override.
The final command uploads to the requested competition and requires Kaggle authentication and accepted rules. Set `KAGGLE_API_TOKEN` securely in your shell/environment; **never commit credentials**. Check the competition's current deadline and submission allowance before uploading. This workspace initially has neither GPU nor Kaggle credentials.

## Evidence-based model, not a claimed winning solution

The encoder is a small sensor-projection CNN followed by four residual, dilated depthwise temporal convolution blocks, GroupNorm, GELU and four-bin temporal pooling. It preserves onset-relative latency and predicts individual unknown-label test trials—**no same-label trial averaging**. Training uses AdamW, cosine decay, gradient clipping, square-root inverse-frequency cross-entropy weights, sensor dropout and short time masking. Independent per-window/channel normalization is identical at training and inference and respects the very small physical units of MEG.

Convolutional temporal backbones and ensembles were effective in the [organizers' 2025 competition retrospective](https://libribrain.com/publications/2025-pnpl-competition-reflections.pdf). The [2025 MEGConformer phoneme winner](https://www.isca-archive.org/odyssey_2026/zuazo26_odyssey.pdf) supports local convolutional extraction and per-window normalization. This implementation is a lightweight **inspired baseline**, not a reproduction of either system; their tasks/averaged-trial results differ from 2026 single-trial word classification.

## Repository layout

```text
pnpl-decoder/
├── decoder.py          # CLI + data prep + training + inference
├── config.json         # baseline subset and hyperparameters
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
