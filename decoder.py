"""Prepare, train, predict and submit a single-trial Deep-track MEG baseline."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import time
import unicodedata

import h5py
from huggingface_hub import hf_hub_download
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
COMPETITION = "pnpl-competition-2026-deep"
PROC = "bads+headpos+sss+notch+bp+ds"


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def config(path):
    return json.loads(Path(path).read_text())


def seed_everything(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def word_key(word):
    # Standardize typography, not lexical identity: do not strip internal apostrophes.
    return unicodedata.normalize("NFKC", str(word)).strip().lower().replace("’", "'").replace("‘", "'")


def normalize_windows(x):
    """Independent channel/window z-score, valid even when MEG is in tesla."""
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite MEG values")
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return np.clip((x - mean) / np.maximum(std, np.float32(1e-30)), -10, 10).astype(np.float32)


def grouped_split(records, fraction, seed):
    """Split whole (task, subject, session) groups within each corpus."""
    if not 0 < fraction < 1:
        raise ValueError("val_fraction must be between zero and one")
    by_corpus = {}
    for r in records:
        by_corpus.setdefault(r["corpus"], set()).add((r["task"], r["subject"], r["session"]))
    val_groups = set()
    for corpus, groups in sorted(by_corpus.items()):
        if len(groups) < 2:
            raise ValueError(f"Need at least two sessions for {corpus}")
        ordered = sorted(groups, key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest())
        n = min(len(groups) - 1, max(1, round(len(groups) * fraction)))
        val_groups.update(ordered[:n])
    result = []
    for r in records:
        group = (r["task"], r["subject"], r["session"])
        result.append({**r, "split": "val" if group in val_groups else "train"})
    assert not ({(r['task'], r['subject'], r['session']) for r in result if r['split'] == 'train'} &
                {(r['task'], r['subject'], r['session']) for r in result if r['split'] == 'val'})
    return result


def select_training_records(cfg):
    from dataclasses import asdict
    from pnpl.datasets.libribrain100.manifest import select_records
    from pnpl.datasets.libribrain100.constants import CORPORA
    records = select_records(subjects={"0"}, corpus=set(CORPORA), partition="train")
    result = []
    for task, limit in cfg["tasks"].items():
        candidates = sorted((r for r in records if r.task == task), key=lambda r: (int(r.session), int(r.run)))
        if not candidates:
            raise ValueError(f"No official train records for {task}")
        selected = candidates if limit is None else candidates[:int(limit)]
        result.extend(asdict(r) for r in selected)
    from pnpl.datasets.libribrain100.constants import REPO_KEY_TO_ID
    for r in result:
        r["repo"] = REPO_KEY_TO_ID[r["repo"]]
    return grouped_split(result, cfg["val_fraction"], cfg["seed"])


def download(repo, name, cfg, directory):
    return Path(hf_hub_download(repo_id=repo, repo_type="dataset", filename=name,
                              revision=cfg["revisions"][repo], local_dir=directory))


def prepare(args):
    from pnpl.competition import load_vocabulary
    cfg = config(args.config)
    out = Path(args.data)
    out.mkdir(parents=True, exist_ok=True)
    # The preparation config is immutable unless the user removes/rebuilds the cache.
    if (out / "prepared_config.json").exists() and config(out / "prepared_config.json") != cfg:
        raise ValueError("Config differs from prepared cache; use a new --data directory")
    vocab = list(load_vocabulary("primary"))
    mapping = {word_key(w): i for i, w in enumerate(vocab)}
    assert len(mapping) == 50
    records = select_training_records(cfg)
    save_json(out / "prepared_config.json", cfg)
    save_json(out / "vocabulary.json", vocab)
    save_json(out / "split_manifest.json", records)
    all_rows, summaries = [], []
    for number, r in enumerate(records):
        stem = f"sub-{r['subject']}_ses-{r['session']}_task-{r['task']}_run-{r['run']}"
        name = f"{r['task']}/derivatives/events/{stem}_events.tsv"
        events = download(r["repo"], name, cfg, out / "raw")
        df = pd.read_csv(events, sep="\t")
        required = {"kind", "segment", "timemeg"}
        if not required.issubset(df):
            raise ValueError(f"Missing event columns: {required - set(df)}")
        words = df.loc[df.kind.eq("word")].dropna(subset=["segment", "timemeg"]).copy()
        words["label"] = words.segment.map(word_key).map(mapping)
        words = words.dropna(subset=["label"])
        h5_name = f"{r['task']}/derivatives/serialised/{stem}_proc-{PROC}_meg.h5"
        path = download(r["repo"], h5_name, cfg, out / "raw")
        epoch_path = out / "epochs" / f"{stem}.npy"
        epoch_path.parent.mkdir(exist_ok=True)
        rows, dropped = [], 0
        with h5py.File(path, "r") as f:
            sfreq = float(f.attrs["sample_frequency"])
            if sfreq != 250 or f["data"].shape[0] != 306:
                raise ValueError(f"Unexpected H5 format for {stem}")
            for event_idx, row in words.iterrows():
                onset = float(row.timemeg)
                start = int(onset * sfreq)
                if not np.isfinite(onset) or start < 0 or start + 250 > f["data"].shape[1]:
                    dropped += 1
                    continue
                rows.append({"file": str(epoch_path.resolve()), "epoch": len(rows), "label": int(row.label),
                             "word": vocab[int(row.label)], "onset": onset, "event_index": int(event_idx),
                             **r})
            expected = (len(rows), 306, 250)
            if not epoch_path.exists() or np.load(epoch_path, mmap_mode="r").shape != expected:
                tmp = epoch_path.with_suffix(".tmp.npy")
                epochs = np.lib.format.open_memmap(tmp, mode="w+", dtype="float32", shape=expected)
                for row in rows:
                    start = int(row["onset"] * sfreq)
                    epochs[row["epoch"]] = normalize_windows(f["data"][:, start:start + 250])
                epochs.flush()
                del epochs
                tmp.replace(epoch_path)
        all_rows.extend(rows)
        summaries.append({**r, "samples": len(rows), "boundary_dropped": dropped,
                          "h5_bytes": path.stat().st_size})
        print(f"Prepared {number + 1}/{len(records)} {stem}: {len(rows)} {r['split']} windows", flush=True)
    frame = pd.DataFrame(all_rows)
    frame.to_csv(out / "samples.csv", index=False)
    counts = {s: np.bincount(frame.loc[frame.split.eq(s), "label"], minlength=50).tolist() for s in ["train", "val"]}
    if any(c == 0 for c in counts["train"]):
        missing = [vocab[i] for i, c in enumerate(counts["train"]) if c == 0]
        raise ValueError(f"Training subset missing classes {missing}; expand tasks/sessions")
    save_json(out / "data_report.json", {"records": summaries, "class_counts": counts,
              "n_samples": len(frame), "missing_val_classes": [vocab[i] for i,c in enumerate(counts['val']) if not c],
              "normalization": "per-window, per-channel population z-score; clip [-10,10]",
              "window_seconds": [0, 1], "sfreq": 250, "training_partition": "official train only"})
    print(json.dumps({"samples": len(frame), "split_counts": frame.split.value_counts().to_dict()}), flush=True)


class EpochDataset(Dataset):
    def __init__(self, frame):
        self.rows = list(frame[["file", "epoch", "label"]].itertuples(index=False, name=None))
        self.arrays = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, epoch, label = self.rows[i]
        if path not in self.arrays:
            self.arrays[path] = np.load(path, mmap_mode="r")
        return torch.from_numpy(self.arrays[path][epoch].copy()), int(label)


class TemporalBlock(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(width, width, 7, padding=3*dilation, dilation=dilation, groups=width),
            nn.GroupNorm(8, width), nn.GELU(), nn.Conv1d(width, width, 1), nn.Dropout(dropout))

    def forward(self, x):
        return x + self.net(x)


class TemporalCNN(nn.Module):
    """Sensor projection + residual dilated temporal convolutions + positional pooling."""
    def __init__(self, width=48, dropout=0.2):
        super().__init__()
        if width % 8:
            raise ValueError("width must be divisible by 8")
        self.encoder = nn.Sequential(nn.Conv1d(306, width, 1), nn.GroupNorm(8, width), nn.GELU(),
                                     nn.AvgPool1d(2),
                                     *(TemporalBlock(width, d, dropout) for d in [1, 2, 4, 8]))
        # Four bins preserve onset-relative latency, unlike global pooling alone.
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(4), nn.Flatten(), nn.Dropout(dropout), nn.Linear(width*4, 50))

    def forward(self, x):
        return self.head(self.encoder(x))


def balanced_metrics(logits, labels, k=10):
    logits, labels = np.asarray(logits), np.asarray(labels, dtype=np.int64)
    if logits.ndim != 2 or logits.shape[1] != 50 or len(logits) != len(labels):
        raise ValueError("Expected N x 50 logits and N labels")
    if not np.isfinite(logits).all() or np.any((labels < 0) | (labels >= 50)) or not 1 <= k <= 50:
        raise ValueError("Invalid scores, labels or k")
    # Stable sort makes tied-score behavior reproducible.
    top = np.argsort(-logits, axis=1, kind="stable")[:, :k]
    correct = (top == labels[:, None]).any(axis=1)
    counts = np.bincount(labels, minlength=50)
    hits = np.bincount(labels, weights=correct, minlength=50)
    recall = np.divide(hits, counts, out=np.zeros(50), where=counts > 0)
    return {"bacc": float(recall.mean()), "bacc_present": float(recall[counts > 0].mean()),
            "coverage": int((counts > 0).sum()), "per_class_recall": recall.tolist()}


def augment(x):
    x = x.clone()
    keep = (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) > 0.05)
    x *= keep
    if random.random() < 0.5:
        start = random.randrange(0, x.shape[-1] - 10)
        x[:, :, start:start+10] = 0
    return x


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def train(args):
    cfg = config(args.config)
    if args.epochs is not None:
        cfg['epochs'] = args.epochs
    data, out = Path(args.data), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    prepared = config(data / "prepared_config.json")
    for key in ["tasks", "val_fraction", "seed", "revisions"]:
        if cfg[key] != prepared[key]:
            raise ValueError(f"Training config differs from preparation: {key}")
    seed_everything(cfg["seed"], cfg["threads"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    frame = pd.read_csv(data / "samples.csv")
    train_rows, val_rows = frame[frame.split.eq("train")], frame[frame.split.eq("val")]
    for rows in [train_rows, val_rows]:
        if not len(rows):
            raise ValueError("Empty train/validation split")
    train_groups = set(zip(train_rows.task, train_rows.subject, train_rows.session))
    assert not train_groups.intersection(zip(val_rows.task, val_rows.subject, val_rows.session))
    counts = np.bincount(train_rows.label, minlength=50)
    if np.any(counts == 0):
        raise ValueError("Training must cover every competition word")
    generator = torch.Generator().manual_seed(cfg["seed"])
    train_loader = DataLoader(EpochDataset(train_rows), batch_size=cfg["batch_size"], shuffle=True, generator=generator)
    val_loader = DataLoader(EpochDataset(val_rows), batch_size=cfg["batch_size"])
    model = TemporalCNN(cfg["width"], cfg["dropout"]).to(device)
    weights = (counts.sum() / counts) ** 0.5
    weights /= weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["epochs"])
    versions = {p: importlib.metadata.version(p) for p in ["torch", "numpy", "pandas", "h5py", "pnpl", "huggingface_hub"]}
    save_json(out / "run_config.json", {"config": cfg, "versions": versions, "device": str(device),
                                         "parameters": sum(p.numel() for p in model.parameters())})
    history, best, stale, start_epoch = [], -1, 0, 0
    if args.resume:
        latest = out / 'latest.pt'
        if latest.exists():
            # This file is produced locally by this program, not an untrusted download.
            state = torch.load(latest, map_location='cpu', weights_only=False)
            if state['config'] != cfg:
                raise ValueError('Resume config differs from latest checkpoint')
            model.load_state_dict(state['state_dict'])
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            torch.set_rng_state(state['torch_rng'])
            if device.type == 'cuda' and state.get('cuda_rng') is not None:
                torch.cuda.set_rng_state_all(state['cuda_rng'])
            np.random.set_state(state['numpy_rng'])
            random.setstate(state['python_rng'])
            generator.set_state(state['loader_rng'])
            history, best, stale, start_epoch = state['history'], state['best'], state['stale'], state['epoch']
        elif (out / 'best.pt').exists():
            # Compatibility with the first run before latest-state saving was added.
            state = torch.load(out / 'best.pt', map_location='cpu', weights_only=True)
            model.load_state_dict(state['state_dict'])
            start_epoch = state['epoch']
            history = config(out / 'history.json')[:start_epoch]
            best = max(r['val_bacc10'] for r in history)
            save_json(out / 'resume_note.json', {'warm_start_epoch': start_epoch,
                      'warning': 'First interrupted run saved weights only; optimizer and RNG reset for this continuation.'})
            print('Warm-starting saved weights; optimizer reset (recorded in resume_note.json)', flush=True)
        else:
            raise FileNotFoundError('No checkpoint to resume')
    for epoch in range(start_epoch, cfg["epochs"]):
        started = time.monotonic()
        model.train()
        total, n = 0.0, 0
        for x, y in train_loader:
            x, y = augment(x.to(device)), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(y)
            n += len(y)
        scheduler.step()
        logits, labels = evaluate(model, val_loader, device)
        top10, top1 = balanced_metrics(logits, labels), balanced_metrics(logits, labels, 1)
        row = {"epoch": epoch + 1, "train_loss": total/n, "val_bacc10": top10["bacc"],
               "val_bacc1": top1["bacc"], "val_coverage": top10["coverage"], "seconds": time.monotonic()-started}
        history.append(row)
        print(json.dumps(row), flush=True)
        save_json(out / "history.json", history)
        if top10["bacc"] > best:
            best, stale = top10["bacc"], 0
            torch.save({"state_dict": model.cpu().state_dict(), "config": cfg,
                        "vocabulary": config(data / "vocabulary.json"), "epoch": epoch+1,
                        "train_counts": counts.tolist()}, out / "best.pt")
            model.to(device)
        else:
            stale += 1
        torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'config': cfg, 'epoch': epoch+1,
                    'history': history, 'best': best, 'stale': stale,
                    'torch_rng': torch.get_rng_state(), 'numpy_rng': np.random.get_state(),
                    'python_rng': random.getstate(), 'loader_rng': generator.get_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None}, out / 'latest.pt')
        if stale >= cfg["patience"]:
            break
    checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    logits, labels = evaluate(model, val_loader, device)
    # Cross-entropy weighting is intentionally moderate. A small prior correction
    # is selected on validation to target macro recall rather than class frequency.
    log_prior = np.log(counts/counts.sum())
    trials = [{"alpha": a, "bacc10": balanced_metrics(logits-a*log_prior, labels)["bacc"]}
              for a in [0.0, 0.25, 0.5, 0.75, 1.0]]
    alpha = max(trials, key=lambda t: t["bacc10"])["alpha"]
    checkpoint["prior_alpha"] = alpha
    torch.save(checkpoint, out / "best.pt")
    corrected = logits - alpha*log_prior
    np.savez(out / "validation_predictions.npz", logits=logits, labels=labels)
    val_rows.to_csv(out / "validation_rows.csv", index=False)
    report = {"best_epoch": checkpoint["epoch"], "raw_top10": balanced_metrics(logits, labels),
              "corrected_top10": balanced_metrics(corrected, labels), "corrected_top1": balanced_metrics(corrected, labels, 1),
              "prior_alpha": alpha, "calibration_search": trials,
              "warning": "Validation also selects checkpoint/prior correction; not an independent test estimate."}
    save_json(out / "validation_report.json", report)
    print(json.dumps(report), flush=True)


def predict(args):
    from pnpl.competition import LibriBrainCompetitionHoldout, load_vocabulary, write_submission
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = checkpoint["config"]
    seed_everything(cfg["seed"], cfg["threads"])
    if checkpoint["vocabulary"] != list(load_vocabulary("primary")):
        raise ValueError("Competition vocabulary differs from checkpoint")
    base = Path(args.data)
    for suffix in ["holdout01_sentence", "holdout2_word"]:
        download("pnpl/LibriBrain-Competition-2026", f"COMPETITION_HOLDOUT/subj00_{suffix}.npz", cfg, base)
    holdout = LibriBrainCompetitionHoldout(track="deep", data_path=base, download=False)
    model = TemporalCNN(cfg["width"], cfg["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()
    log_prior = np.log(np.asarray(checkpoint["train_counts"])/sum(checkpoint["train_counts"]))
    probabilities = []
    with torch.inference_mode():
        for meg, _ in holdout.iter_windows(batch_size=cfg["batch_size"]):
            logits = model(torch.from_numpy(normalize_windows(meg)).to(device)).cpu().numpy()
            logits -= checkpoint.get("prior_alpha", 0.0) * log_prior
            probabilities.append(torch.from_numpy(logits).softmax(-1).numpy())
    scores = np.concatenate(probabilities)
    if scores.shape != (len(holdout.indices), 50) or not np.isfinite(scores).all():
        raise ValueError("Invalid holdout predictions")
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    write_submission(args.csv, indices=holdout.indices, primary_probs=scores)
    save_json(Path(args.csv).with_suffix(".json"), {"rows": len(scores), "counts": holdout.counts(),
              "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
              "submission_sha256": hashlib.sha256(Path(args.csv).read_bytes()).hexdigest(),
              "competition": COMPETITION, "holdout_revision": cfg["revisions"]["pnpl/LibriBrain-Competition-2026"]})
    print(f"Wrote {args.csv}: {len(scores)} rows, 50 primary scores", flush=True)


def submit(args):
    if not Path(args.csv).is_file():
        raise FileNotFoundError(args.csv)
    authenticated = bool(os.getenv("KAGGLE_API_TOKEN")) or bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")) or (Path.home()/".kaggle/kaggle.json").exists()
    if not authenticated:
        raise RuntimeError("No Kaggle credentials. Configure KAGGLE_API_TOKEN securely and accept competition rules on Kaggle.")
    subprocess.run(["kaggle", "competitions", "submit", "-c", COMPETITION, "-f", args.csv, "-m", args.message], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--config", default=str(ROOT/"config.json"))
    p.add_argument("--data", default=str(ROOT/"data"))
    p.set_defaults(fn=prepare)
    p = sub.add_parser("train")
    p.add_argument("--config", default=str(ROOT/"config.json"))
    p.add_argument("--data", default=str(ROOT/"data"))
    p.add_argument("--output", default=str(ROOT/"artifacts/baseline"))
    p.add_argument("--device", choices=["cpu", "cuda"])
    p.add_argument("--resume", action="store_true", help="Resume trusted local latest.pt, or warm-start best.pt")
    p.add_argument("--epochs", type=int, help="Override epoch budget")
    p.set_defaults(fn=train)
    p = sub.add_parser("predict")
    p.add_argument("--checkpoint", default=str(ROOT/"artifacts/baseline/best.pt"))
    p.add_argument("--data", default=str(ROOT/"data/holdout"))
    p.add_argument("--csv", default=str(ROOT/"artifacts/submission_deep.csv"))
    p.add_argument("--device", choices=["cpu", "cuda"])
    p.set_defaults(fn=predict)
    p = sub.add_parser("submit")
    p.add_argument("--csv", default=str(ROOT/"artifacts/submission_deep.csv"))
    p.add_argument("--message", default="Single-trial residual temporal CNN; session-held-out validation")
    p.set_defaults(fn=submit)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
