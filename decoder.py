"""Prepare, train, predict and submit a single-trial Deep-track MEG baseline.

The repository is intentionally centered on one strong, auditable baseline.
Future model work should mainly touch:

1. :func:`build_model` for new architectures,
2. :func:`augment` for training-time augmentation,
3. :func:`select_training_records` for larger training subsets.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any
import unicodedata

import h5py
from huggingface_hub import HfApi, hf_hub_download
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from progress import Progress, progress_iter

ROOT = Path(__file__).resolve().parent
COMPETITION = "pnpl-competition-2026-deep"
PROC = "bads+headpos+sss+notch+bp+ds"
N_CHANNELS = 306
WINDOW_SAMPLES = 250
N_CLASSES = 50
PRIOR_ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
GIB = 1024**3


def resource_path(name: str) -> Path:
    local = ROOT / name
    return local if local.is_file() else Path(sys.prefix) / "share/pnpl-decoder" / name


def default_config_path() -> Path:
    return resource_path("config.json")


@contextmanager
def atomic_output(path: str | Path):
    """Replace a file only after a complete, flushed write on the same filesystem."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp = Path(name)
    try:
        yield tmp
        with tmp.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        tmp.unlink(missing_ok=True)


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    with atomic_output(path) as tmp:
        torch.save(state, tmp)


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path

    @property
    def best_checkpoint(self) -> Path:
        return self.root / "best.pt"

    @property
    def latest_checkpoint(self) -> Path:
        return self.root / "latest.pt"

    @property
    def history(self) -> Path:
        return self.root / "history.json"

    @property
    def run_config(self) -> Path:
        return self.root / "run_config.json"

    @property
    def validation_predictions(self) -> Path:
        return self.root / "validation_predictions.npz"

    @property
    def validation_rows(self) -> Path:
        return self.root / "validation_rows.csv"

    @property
    def validation_report(self) -> Path:
        return self.root / "validation_report.json"

    @property
    def resume_note(self) -> Path:
        return self.root / "resume_note.json"


def save_json(path: str | Path, value: Any) -> None:
    with atomic_output(path) as tmp:
        tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def seed_everything(seed: int, threads: int = 4) -> None:
    """Seed RNGs; prefer deterministic kernels without blocking CUDA training.

    Some CUDA operations (including adaptive pooling backward) lack a
    deterministic implementation. Warn on those operations rather than crash;
    GPU runs are therefore not guaranteed to be bitwise reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """Independent reproducible augmentation/NumPy seeds in loader processes."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_loader(dataset, cfg, device, shuffle=False, sampler=None, generator=None):
    workers = int(cfg.get('num_workers', 0))
    if workers < 0:
        raise ValueError('num_workers must be nonnegative')
    # Sampling RNG is checkpointed separately from worker/iterator base seeds.
    # Dataset preprocessing is deterministic; augmentation happens in the parent.
    if shuffle:
        if sampler is not None:
            raise ValueError('shuffle and sampler are mutually exclusive')
        sampler = torch.utils.data.RandomSampler(dataset, generator=generator)
    worker_generator = torch.Generator().manual_seed(cfg.get('training_seed',cfg.get('seed',2026)))
    kwargs = dict(batch_size=cfg['batch_size'], shuffle=False, sampler=sampler,
                   generator=worker_generator, num_workers=workers,
                  pin_memory=device.type == 'cuda', worker_init_fn=seed_worker)
    if workers:
        kwargs.update(persistent_workers=True, prefetch_factor=cfg.get('prefetch_factor', 2),
                      multiprocessing_context='spawn')
    return DataLoader(dataset, **kwargs)


def word_key(word: str) -> str:
    # Standardize typography, not lexical identity: do not strip internal apostrophes.
    return unicodedata.normalize("NFKC", str(word)).strip().lower().replace("’", "'").replace("‘", "'")


def normalize_windows(x: np.ndarray) -> np.ndarray:
    """Independent channel/window z-score, valid even when MEG is in tesla."""
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite MEG values")
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return np.clip((x - mean) / np.maximum(std, np.float32(1e-30)), -10, 10).astype(np.float32)


def grouped_split(records: list[dict[str, Any]], fraction: float, seed: int) -> list[dict[str, Any]]:
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


def select_training_records(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Choose official subject-0 train runs and add leakage-safe split labels."""
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
    if "validation_groups" in cfg:
        requested = cfg["validation_groups"]
        if not requested or any(len(g) != 3 for g in requested):
            raise ValueError("validation_groups must contain (task, subject, session) triples")
        groups = {tuple(str(value) for value in g) for g in requested}
        available = {(r['task'], r['subject'], r['session']) for r in result}
        if not groups.issubset(available):
            raise ValueError(f"Validation groups absent from selected official train records: {groups - available}")
        if groups == available:
            raise ValueError("Fixed validation groups leave no training sessions")
        return [{**r, "split": "val" if (r['task'], r['subject'], r['session']) in groups else "train"}
                for r in result]
    return grouped_split(result, cfg["val_fraction"], cfg["seed"])


def download(repo: str, name: str, cfg: dict[str, Any], directory: str | Path) -> Path:
    return Path(hf_hub_download(repo_id=repo, repo_type="dataset", filename=name,
                              revision=cfg["revisions"][repo], local_dir=directory))


def run_stem(record: dict[str, Any]) -> str:
    return f"sub-{record['subject']}_ses-{record['session']}_task-{record['task']}_run-{record['run']}"


def recording_names(record: dict[str, Any]) -> tuple[str, str]:
    stem = run_stem(record)
    return (f"{record['task']}/derivatives/events/{stem}_events.tsv",
            f"{record['task']}/derivatives/serialised/{stem}_proc-{PROC}_meg.h5")


def reject_final_test_path(path: Path) -> None:
    if any((parent / "FINAL_TEST_ONLY").exists() for parent in [path.resolve(), *path.resolve().parents]):
        raise ValueError("Final-test cache cannot be used for training or preparation")


def download_plan(records: list[dict[str, Any]], cfg: dict[str, Any], raw: Path) -> dict[str, Any]:
    """Get exact pinned remote sizes; no payload downloads or directory creation."""
    by_repo: dict[str, set[str]] = {}
    for record in records:
        if record.get("partition") != "train" or record.get("split") not in {"train", "val"}:
            raise ValueError("Preparation requires official training records only")
        by_repo.setdefault(record["repo"], set()).update(recording_names(record))
    files = []
    api = HfApi()
    for repo, names in sorted(by_repo.items()):
        ordered = sorted(names)
        sizes = {}
        for start in range(0, len(ordered), 100):
            info = api.get_paths_info(repo, ordered[start:start + 100], repo_type="dataset",
                                     revision=cfg["revisions"][repo])
            sizes.update({item.path: item.size for item in info})
        if set(sizes) != names or any(not isinstance(size, int) or size < 0 for size in sizes.values()):
            raise ValueError(f"Missing files or exact sizes at pinned revision for {repo}")
        for name in ordered:
            local = raw / name
            reject_final_test_path(local)
            present = local.is_file() and local.stat().st_size == sizes[name]
            files.append({"repo": repo, "revision": cfg["revisions"][repo], "name": name,
                          "bytes": sizes[name], "local_complete": present,
                          "download_bytes": 0 if present else sizes[name]})
    return {"files": files, "raw_bytes": sum(f["bytes"] for f in files),
            "h5_bytes": sum(f["bytes"] for f in files if f["name"].endswith(".h5")),
            "event_bytes": sum(f["bytes"] for f in files if f["name"].endswith(".tsv")),
            "download_bytes": sum(f["download_bytes"] for f in files)}


def existing_bytes(roots: list[Path]) -> int:
    seen = set()
    total = 0
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                stat = path.stat()
                key = (stat.st_dev, stat.st_ino)
                if key not in seen:
                    seen.add(key)
                    total += stat.st_size
    return total


def check_disk_space(out: Path, raw: Path, download_bytes: int, cache_bytes: int,
                     budget_gib: float | None, headroom_gib: float) -> None:
    """Budget covers existing + additional preparation data, not training artifacts."""
    if (not np.isfinite(headroom_gib) or headroom_gib < 0
            or (budget_gib is not None and (not np.isfinite(budget_gib) or budget_gib <= 0))):
        raise ValueError("Disk budget must be positive and headroom nonnegative")
    allocations: dict[int, tuple[Path, int]] = {}
    for path, amount in [(raw, download_bytes), (out, cache_bytes)]:
        parent = path.resolve()
        while not parent.exists():
            parent = parent.parent
        device = parent.stat().st_dev
        previous = allocations.get(device, (parent, 0))
        allocations[device] = (parent, previous[1] + amount)
    for parent, amount in allocations.values():
        if shutil.disk_usage(parent).free < amount + int(headroom_gib * GIB):
            raise RuntimeError(f"Insufficient disk headroom on {parent}: need {amount} additional bytes "
                               f"plus {headroom_gib:g} GiB headroom")
    if budget_gib is not None:
        required = existing_bytes([out, raw]) + download_bytes + cache_bytes
        if required > int(budget_gib * GIB):
            raise RuntimeError(f"Preparation disk budget exceeded: need up to {required} bytes, "
                               f"budget {budget_gib:g} GiB")


def read_word_events(path: Path, mapping: dict[str, int]) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    required = {"kind", "segment", "timemeg"}
    if not required.issubset(frame):
        raise ValueError(f"Missing event columns: {required - set(frame)}")
    words = frame.loc[frame.kind.eq("word")].dropna(subset=["segment", "timemeg"]).copy()
    words["label"] = words.segment.map(word_key).map(mapping)
    return words.dropna(subset=["label"])


def competition_vocabulary() -> list[str]:
    from pnpl.competition import load_vocabulary
    return list(load_vocabulary("primary"))


def competition_vocabulary_mapping() -> tuple[list[str], dict[str, int]]:
    vocab = competition_vocabulary()
    mapping = {word_key(word): i for i, word in enumerate(vocab)}
    if len(vocab) != N_CLASSES or len(mapping) != N_CLASSES:
        raise ValueError("Unexpected competition vocabulary")
    return vocab, mapping


def verify_prepared_config(data_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    reject_final_test_path(data_dir)
    prepared = config(data_dir / "prepared_config.json")
    for key in ["tasks", "val_fraction", "seed", "revisions"]:
        if cfg[key] != prepared[key]:
            raise ValueError(f"Training config differs from preparation: {key}")
    if select_training_records(cfg) != select_training_records(prepared):
        raise ValueError("Training validation groups differ from prepared split")
    if cfg.get('purge_repeated_sentences',False) != prepared.get('purge_repeated_sentences',False):
        raise ValueError('Sentence purging policy differs from prepared cache')
    return prepared


def split_sample_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not frame["split"].isin(["train", "val"]).all():
        raise ValueError("Training cache contains a forbidden split")
    if "partition" in frame and not frame["partition"].eq("train").all():
        raise ValueError("Training cache contains non-training partition records")
    train_rows = frame[frame.split.eq("train")]
    val_rows = frame[frame.split.eq("val")]
    for rows in [train_rows, val_rows]:
        if not len(rows):
            raise ValueError("Empty train/validation split")
    train_groups = set(zip(train_rows.task, train_rows.subject, train_rows.session))
    val_groups = set(zip(val_rows.task, val_rows.subject, val_rows.session))
    assert not train_groups.intersection(val_groups)
    return train_rows, val_rows


def class_counts(labels: pd.Series | np.ndarray) -> np.ndarray:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=N_CLASSES)
    if len(counts) != N_CLASSES:
        raise ValueError("Unexpected label dimension")
    return counts


def build_model(cfg: dict[str, Any]) -> nn.Module:
    """Single extension point for swapping in future model architectures."""
    name = cfg.get("model", "temporal_cnn")  # Legacy checkpoints remain loadable.
    if name == "temporal_cnn":
        return TemporalCNN(cfg["width"], cfg["dropout"])
    if name == "multiscale_conformer":
        from models import MultiScaleConformer
        return MultiScaleConformer(cfg["width"], cfg["dropout"],
                                   layers=cfg.get("conformer_layers", 4),
                                   heads=cfg.get("attention_heads", 4),
                                   kernel_size=cfg.get("conformer_kernel", 15))
    if name == "wide_residual_cnn":
        from models import WideResidualCNN
        return WideResidualCNN(cfg["width"], cfg["dropout"],
                               dilations=cfg.get("cnn_dilations", [1, 2, 4, 8, 1, 2]),
                               pooling_bins=cfg.get("pooling_bins", 8))
    if name == "temporal_spatial_cnn":
        from models import TemporalSpatialCNN
        return TemporalSpatialCNN(banks=cfg.get('temporal_banks',32),depth=cfg.get('spatial_depth',2),
                                  dropout=cfg['dropout'],pooling_bins=cfg.get('pooling_bins',8))
    if name == 'multiscale_spatial_cnn':
        from models import MultiScaleSpatialCNN
        if cfg.get('temporal_banks',32) != 32:
            raise ValueError('Multiscale spatial CNN requires 32 temporal banks')
        return MultiScaleSpatialCNN(dropout=cfg['dropout'],depth=cfg.get('spatial_depth',2),
                                    pooling_bins=cfg.get('pooling_bins',8))
    if name == 'geometry_cnn':
        from models import GeometryCNN
        return GeometryCNN(width=cfg.get('geometry_width',64),dropout=cfg['dropout'])
    raise ValueError(f"Unknown model: {name}")


def configure_gpus(model: nn.Module, device: torch.device, devices: int) -> nn.Module:
    """Use one or two GPUs; serialize the underlying model's portable weights.

    DataParallel uses a global batch size. For this small baseline its overhead
    may exceed its benefit; benchmark against one GPU before scaling up.
    """
    if devices < 1 or (devices > 1 and device.type != "cuda"):
        raise ValueError("Multiple devices require CUDA; --devices must be positive")
    if device.type == "cuda" and devices > torch.cuda.device_count():
        raise ValueError(f"Requested {devices} GPUs, available {torch.cuda.device_count()}")
    model = model.to(device)
    if devices > 1:
        model = PortableDataParallel(model, device_ids=list(range(devices)))
    return model


class PortableDataParallel(nn.DataParallel):
    """DataParallel execution with single-model state_dict compatibility."""

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.module.load_state_dict(*args, **kwargs)


def choose_prior_alpha(logits: np.ndarray, labels: np.ndarray, log_prior: np.ndarray) -> tuple[float, list[dict[str, float]]]:
    trials = [
        {"alpha": alpha, "bacc10": balanced_metrics(logits - alpha * log_prior, labels)["bacc"]}
        for alpha in PRIOR_ALPHA_GRID
    ]
    return max(trials, key=lambda trial: trial["bacc10"])["alpha"], trials


def prepare(args):
    """Prepare epoch caches or an HDF5 window index from official train runs."""
    cfg = config(args.config)
    out = Path(args.data)
    raw = Path(getattr(args, "raw", None) or out / "raw")
    index_only = getattr(args, "index_only", False)
    headroom = getattr(args, "headroom_gib", 5.0)
    budget = getattr(args, "disk_budget_gib", None)
    reject_final_test_path(out)
    reject_final_test_path(raw)
    # Only preparation-relevant fields must agree, not architecture/optimizer settings.
    if (out / "prepared_config.json").exists():
        verify_prepared_config(out, cfg)
    vocab, mapping = competition_vocabulary_mapping()
    records = select_training_records(cfg)
    plan = download_plan(records, cfg, raw)
    plan.update({"mode": "h5" if index_only else "epochs", "data": str(out.resolve()),
                 "raw": str(raw.resolve()), "disk_budget_gib": budget, "headroom_gib": headroom,
                 "epoch_cache_bytes": None if not index_only else 0,
                 "note": "Exact remote payload sizes; epoch allocation checked after event-only download. "
                         "Indexes, HF metadata and download scratch need filesystem headroom."})
    print(f"Data: {len(records)} recordings | raw {plan['raw_bytes']/GIB:.2f} GiB | "
          f"download {plan['download_bytes']/GIB:.2f} GiB | mode {'HDF5 index' if index_only else 'epoch cache'}",flush=True)
    check_disk_space(out, raw, plan["download_bytes"], 0, budget, headroom)
    if getattr(args, "preflight", False):
        print(f"Disk check passed; reserve {headroom:g} GiB. No payloads downloaded.",flush=True)
        return
    # Event files are small. Count all vocabulary windows before any H5 download.
    words_by_run = {}
    for r in progress_iter(records,len(records),'Event metadata',key='metadata'):
        event_name, _ = recording_names(r)
        event_path = download(r["repo"], event_name, cfg, raw)
        all_events = pd.read_csv(event_path, sep='\t')
        words = read_word_events(event_path, mapping)
        if cfg.get('purge_repeated_sentences') and r['task'] in ['TIMIT','MOCHATIMIT']:
            from research import add_stimulus_keys
            words['stimulus_key'] = add_stimulus_keys(all_events).loc[words.index]
            if words['stimulus_key'].isna().any():
                raise ValueError('Missing sentence IDs; refusing sentence validation')
        words_by_run[run_stem(r)] = words
    # Remove whole repeated sentence texts from training when they occur in
    # sentence-domain validation. Uses metadata only, never model outcomes.
    reserved_sentences = set()
    for r in records:
        words = words_by_run[run_stem(r)]
        if r['split']=='val' and 'stimulus_key' in words:
            reserved_sentences.update(words.stimulus_key)
    purged = 0
    for r in records:
        words = words_by_run[run_stem(r)]
        if r['split']=='train' and 'stimulus_key' in words:
            keep = ~words.stimulus_key.isin(reserved_sentences)
            purged += int((~keep).sum())
            words_by_run[run_stem(r)] = words.loc[keep]
    cache_bytes = 0 if index_only else sum(
        len(words) * N_CHANNELS * WINDOW_SAMPLES * 4 + 256 for words in words_by_run.values())
    h5_download_bytes = sum(f["download_bytes"] for f in plan["files"] if f["name"].endswith(".h5"))
    check_disk_space(out, raw, h5_download_bytes, cache_bytes, budget, headroom)
    out.mkdir(parents=True, exist_ok=True)
    plan["epoch_cache_bytes_upper_bound"] = cache_bytes
    plan['sentence_windows_purged'] = purged
    save_json(out / "download_plan.json", plan)
    save_json(out / "prepared_config.json", cfg)
    save_json(out / "vocabulary.json", vocab)
    save_json(out / "split_manifest.json", records)
    all_rows, summaries = [], []
    for number, r in enumerate(progress_iter(records,len(records),'Prepare recordings',key='prepare')):
        stem = run_stem(r)
        words = words_by_run[stem]
        _, h5_name = recording_names(r)
        # Recheck before each large payload; other processes may consume disk.
        remaining = sum(f["bytes"] for f in plan["files"] if f["name"].endswith(".h5")
                        and (not (raw / f["name"]).is_file()
                             or (raw / f["name"]).stat().st_size != f["bytes"]))
        check_disk_space(out, raw, remaining, cache_bytes, budget, headroom)
        path = download(r["repo"], h5_name, cfg, raw)
        epoch_path = out / "epochs" / f"{stem}.npy"
        rows, dropped = [], 0
        with h5py.File(path, "r") as f:
            sfreq = float(f.attrs["sample_frequency"])
            if sfreq != 250 or f["data"].ndim != 2 or f["data"].shape[0] != N_CHANNELS:
                raise ValueError(f"Unexpected H5 format for {stem}")
            for event_idx, row in words.iterrows():
                onset = float(row.timemeg)
                if not np.isfinite(onset):
                    dropped += 1
                    continue
                start = int(onset * sfreq)
                if start < 0 or start + WINDOW_SAMPLES > f["data"].shape[1]:
                    dropped += 1
                    continue
                h5_path = path.resolve()
                if h5_path.is_relative_to(out.resolve()):
                    h5_path = h5_path.relative_to(out.resolve())
                rows.append({"file": str(epoch_path.resolve()), "epoch": len(rows), "label": int(row.label),
                              "stimulus_key": row.get('stimulus_key',''),
                             "h5_file": str(h5_path),
                             "start_sample": start,
                             "word": vocab[int(row.label)], "onset": onset, "event_index": int(event_idx),
                             **r})
            expected = (len(rows), N_CHANNELS, WINDOW_SAMPLES)
            if not index_only and (not epoch_path.exists() or np.load(epoch_path, mmap_mode="r").shape != expected):
                with atomic_output(epoch_path) as tmp:
                    epochs = np.lib.format.open_memmap(tmp, mode="w+", dtype="float32", shape=expected)
                    for row in rows:
                        start = row["start_sample"]
                        epochs[row["epoch"]] = normalize_windows(f["data"][:, start:start + WINDOW_SAMPLES])
                    epochs.flush()
                    del epochs
            if not index_only:
                cache_bytes -= len(words) * N_CHANNELS * WINDOW_SAMPLES * 4 + 256
        all_rows.extend(rows)
        summaries.append({**r, "samples": len(rows), "boundary_dropped": dropped,
                          "h5_bytes": path.stat().st_size})
    frame = pd.DataFrame(all_rows)
    if cfg.get('purge_repeated_sentences'):
        sentence = frame[frame.task.isin(['TIMIT','MOCHATIMIT'])]
        if set(sentence.loc[sentence.split.eq('train'),'stimulus_key']) & set(sentence.loc[sentence.split.eq('val'),'stimulus_key']):
            raise ValueError('Repeated sentence text crosses development splits')
    split_sample_frame(frame)
    counts = {split: class_counts(frame.loc[frame.split.eq(split), "label"]).tolist() for split in ["train", "val"]}
    if any(c == 0 for c in counts["train"]):
        missing = [vocab[i] for i, c in enumerate(counts["train"]) if c == 0]
        raise ValueError(f"Training subset missing classes {missing}; expand tasks/sessions")
    sample_path = out / ("samples_h5.csv" if index_only else "samples.csv")
    with atomic_output(sample_path) as tmp:
        frame.to_csv(tmp, index=False)
    save_json(out / ("data_report_h5.json" if index_only else "data_report.json"), {"records": summaries, "class_counts": counts,
              "mode": "h5" if index_only else "epochs",
              "n_samples": len(frame), "missing_val_classes": [vocab[i] for i,c in enumerate(counts['val']) if not c],
              "normalization": "per-window, per-channel population z-score; clip [-10,10]",
              "window_seconds": [0, 1], "sfreq": 250, "training_partition": "official train only"})
    print(f"Ready: {len(frame):,} windows | train {sum(counts['train']):,} | "
          f"val {sum(counts['val']):,} | purged {purged:,} | metadata: {out}",flush=True)


class EpochDataset(Dataset):
    """Memory-mapped cache of normalized 1-second epochs."""

    def __init__(self, frame):
        self.rows = list(frame[["file", "epoch", "label"]].itertuples(index=False, name=None))
        for path in {row[0] for row in self.rows}:
            reject_final_test_path(Path(path))
        self.arrays = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, epoch, label = self.rows[i]
        if path not in self.arrays:
            self.arrays[path] = np.load(path, mmap_mode="r")
        return torch.from_numpy(self.arrays[path][epoch].copy()), int(label)


class H5WindowDataset(Dataset):
    """Slice and normalize one window at a time; handles are process-local and bounded."""

    def __init__(self, frame, data_dir: Path, max_open_files: int = 4, normalization='window_zscore', stats=None):
        if max_open_files < 1:
            raise ValueError("max_open_files must be positive")
        self.rows = []
        checked_paths = set()
        for path, start, label in frame[["h5_file", "start_sample", "label"]].itertuples(index=False, name=None):
            path = Path(path)
            path = path if path.is_absolute() else data_dir / path
            if path not in checked_paths:
                reject_final_test_path(path)
                checked_paths.add(path)
            if not np.isfinite(start) or int(start) != start or start < 0:
                raise ValueError("Invalid HDF5 start sample")
            self.rows.append((str(path), int(start), int(label)))
        self.max_open_files = max_open_files
        self.handles = OrderedDict()
        self.pid = os.getpid()
        self.normalization, self.stats = normalization, stats

    def __len__(self):
        return len(self.rows)

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["handles"] = OrderedDict()
        return state

    def __del__(self):
        if hasattr(self, "handles"):
            self.close()

    def __getitem__(self, i):
        if self.pid != os.getpid():
            self.close()
            self.pid = os.getpid()
        path, start, label = self.rows[i]
        if path not in self.handles:
            if len(self.handles) >= self.max_open_files:
                _, old = self.handles.popitem(last=False)
                old.close()
            handle = h5py.File(path, "r")
            if (float(handle.attrs.get("sample_frequency", 0)) != 250
                    or handle["data"].ndim != 2 or handle["data"].shape[0] != N_CHANNELS):
                handle.close()
                raise ValueError(f"Unexpected H5 format for {path}")
            self.handles[path] = handle
        self.handles.move_to_end(path)
        data = self.handles[path]["data"]
        if start + WINDOW_SAMPLES > data.shape[1]:
            raise ValueError(f"HDF5 window exceeds recording boundary: {path}")
        from research import preprocess
        return torch.from_numpy(preprocess(data[:, start:start + WINDOW_SAMPLES], self.normalization, self.stats)), label


def training_frame(data: Path, mode: str) -> tuple[pd.DataFrame, str]:
    reject_final_test_path(data)
    if mode == "auto":
        mode = "epochs" if (data / "samples.csv").exists() else "h5"
    if mode == "h5" and (data / "samples_h5.csv").exists():
        path = data / "samples_h5.csv"
    else:
        path = data / "samples.csv"
    frame = pd.read_csv(path, low_memory=False)
    if mode == "h5" and not {"h5_file", "start_sample"}.issubset(frame):
        raise ValueError("HDF5 index missing; run prepare --index-only first")
    return frame, mode


def sample_fingerprint(frame: pd.DataFrame) -> str:
    # Paths can change when a cache is copied from workspace to Kaggle.
    columns = [c for c in frame if c not in {"file", "h5_file"}]
    return hashlib.sha256(frame[columns].to_csv(index=False).encode()).hexdigest()


def restore_training_state(state, model, optimizer, scheduler, generator, device):
    model.load_state_dict(state["state_dict"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["torch_rng"])
    if device.type == "cuda" and state.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    generator.set_state(state["loader_rng"])


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
        self.encoder = nn.Sequential(nn.Conv1d(N_CHANNELS, width, 1), nn.GroupNorm(8, width), nn.GELU(),
                                     nn.AvgPool1d(2),
                                     *(TemporalBlock(width, d, dropout) for d in [1, 2, 4, 8]))
        # Four bins preserve onset-relative latency, unlike global pooling alone.
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(4), nn.Flatten(), nn.Dropout(dropout), nn.Linear(width*4, N_CLASSES))

    def forward(self, x):
        return self.head(self.encoder(x))


def balanced_metrics(logits, labels, k=10):
    """Competition-style balanced recall@k over the fixed 50-word vocabulary."""
    logits, labels = np.asarray(logits), np.asarray(labels, dtype=np.int64)
    if logits.ndim != 2 or logits.shape[1] != N_CLASSES or len(logits) != len(labels):
        raise ValueError("Expected N x 50 logits and N labels")
    if not np.isfinite(logits).all() or np.any((labels < 0) | (labels >= N_CLASSES)) or not 1 <= k <= N_CLASSES:
        raise ValueError("Invalid scores, labels or k")
    # Stable sort makes tied-score behavior reproducible.
    top = np.argsort(-logits, axis=1, kind="stable")[:, :k]
    correct = (top == labels[:, None]).any(axis=1)
    counts = np.bincount(labels, minlength=N_CLASSES)
    hits = np.bincount(labels, weights=correct, minlength=N_CLASSES)
    recall = np.divide(hits, counts, out=np.zeros(N_CLASSES), where=counts > 0)
    return {"bacc": float(recall.mean()), "bacc_present": float(recall[counts > 0].mean()),
            "coverage": int((counts > 0).sum()), "per_class_recall": recall.tolist()}


def augment(x):
    """Lightweight augmentation consistent with single-trial MEG baselines."""
    x = x.clone()
    keep = (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) > 0.05)
    x *= keep
    if random.random() < 0.5:
        start = random.randrange(0, x.shape[-1] - 10)
        x[:, :, start:start+10] = 0
    return x


@torch.inference_mode()
def evaluate(model, loader, device, description='Validate'):
    model.eval()
    logits, labels = [], []
    for x, y in progress_iter(loader,len(loader),description,key='batches'):
        logits.append(model(x.to(device,non_blocking=device.type=='cuda')).float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def train(args):
    """Train the baseline model and save auditable validation artifacts."""
    cfg = config(args.config)
    if args.epochs is not None:
        cfg['epochs'] = args.epochs
    data, out = Path(args.data), Path(args.output)
    reject_final_test_path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = ArtifactPaths(out)
    if not args.resume and (paths.latest_checkpoint.exists() or paths.best_checkpoint.exists()):
        raise ValueError("Output already contains checkpoints; use --resume or a new --output directory")
    verify_prepared_config(data, cfg)
    train_seed = cfg.get('training_seed',cfg['seed'])
    seed_everything(train_seed, cfg["threads"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    frame, data_mode = training_frame(data, getattr(args, "data_mode", "auto"))
    fingerprint = sample_fingerprint(frame)
    train_rows, val_rows = split_sample_frame(frame)
    original_train_rows = train_rows.copy()
    if cfg.get('shuffle_training_labels',False):
        train_rows = train_rows.copy()
        train_rows['label'] = np.random.default_rng(train_seed).permutation(train_rows.label.to_numpy())
    counts = class_counts(train_rows.label)
    if np.any(counts == 0):
        raise ValueError("Training must cover every competition word")
    generator = torch.Generator().manual_seed(train_seed)
    norm = cfg.get('normalization','window_zscore')
    if norm != 'window_zscore' and data_mode != 'h5':
        raise ValueError('Normalization ablation requires --data-mode h5 (raw windows)')
    if norm not in ['window_zscore','train_channel_zscore']:
        raise ValueError('Unknown normalization')
    stats = None
    if norm == 'train_channel_zscore':
        from research import fit_channel_stats
        stats = fit_channel_stats(original_train_rows,data,max_windows=cfg.get('stats_windows',512))
        save_json(out/'preprocessing_stats.json',stats)

    def dataset(rows):
        return H5WindowDataset(rows, data, normalization=norm, stats=stats) if data_mode == "h5" else EpochDataset(rows)

    sampling = cfg.get('sampling','natural')
    if sampling not in ['natural','class_balanced','corpus_balanced']:
        raise ValueError('Unknown sampling policy')
    sampler = WeightedRandomSampler(torch.tensor(1/counts[train_rows.label.to_numpy()],dtype=torch.double),
                                   len(train_rows),replacement=True,generator=generator) if sampling=='class_balanced' else None
    if sampling=='corpus_balanced':
        domain=train_rows.task.map(lambda t:'Sherlock' if str(t).startswith('Sherlock') else t)
        frequencies=domain.value_counts()
        sampler=WeightedRandomSampler(torch.tensor(domain.map(lambda d:1/frequencies[d]).to_numpy(),dtype=torch.double),
                                      len(train_rows),replacement=True,generator=generator)
    train_loader = make_loader(dataset(train_rows),cfg,device,shuffle=sampler is None,
                               sampler=sampler,generator=generator)
    val_loader = make_loader(dataset(val_rows),cfg,device)
    model = configure_gpus(build_model(cfg), device, getattr(args, "devices", 1))
    geometry_names = None
    if cfg.get('model')=='geometry_cnn':
        from models import load_sensor_geometry, dataset_channel_names
        geometry_names = dataset_channel_names(frame,data)
        target = model.module if isinstance(model,nn.DataParallel) else model
        if not args.resume:
            target.initialize_geometry(load_sensor_geometry(cfg['geometry_file'],geometry_names))
    weights = (counts.sum() / counts) ** cfg.get('loss_weight_power',0.5)
    if sampling=='class_balanced' and cfg.get('loss_weight_power',0.5)!=0:
        raise ValueError('Class-balanced sampling requires loss_weight_power=0 to avoid double balancing')
    weights /= weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["epochs"])
    amp = bool(cfg.get('amp',False)) and device.type=='cuda'
    scaler = torch.amp.GradScaler('cuda',enabled=amp)
    versions = {p: importlib.metadata.version(p) for p in ["torch", "numpy", "pandas", "h5py", "pnpl", "huggingface_hub"]}
    history, best, stale, start_epoch = [], -1, 0, 0
    best_state = None
    if args.resume:
        latest = paths.latest_checkpoint
        if latest.exists():
            # This file is produced locally by this program, not an untrusted download.
            state = torch.load(latest, map_location='cpu', weights_only=False)
            if state['config'] != cfg:
                raise ValueError('Resume config differs from latest checkpoint')
            if state.get('sample_fingerprint', fingerprint) != fingerprint:
                raise ValueError('Resume sample index differs from latest checkpoint')
            if geometry_names is not None and state.get('geometry_channel_names') != geometry_names:
                raise ValueError('Resume geometry channel order differs from data')
            restore_training_state(state, model, optimizer, scheduler, generator, device)
            if amp and state.get('scaler'):
                scaler.load_state_dict(state['scaler'])
            elif amp:
                print('AMP enabled without saved scaler state; starting a fresh gradient scale.',flush=True)
            history, best, stale, start_epoch = state['history'], state['best'], state['stale'], state['epoch']
            best_state = state.get('best_checkpoint')
            if best_state is not None:
                save_checkpoint(paths.best_checkpoint, best_state)
            elif not paths.best_checkpoint.exists():
                raise FileNotFoundError('Legacy latest.pt requires its companion best.pt')
            save_json(paths.history, history)
        elif paths.best_checkpoint.exists():
            # Compatibility with the first run before latest-state saving was added.
            state = torch.load(paths.best_checkpoint, map_location='cpu', weights_only=True)
            if state['config'] != cfg:
                raise ValueError('Warm-start config differs from best checkpoint')
            if geometry_names is not None and state.get('geometry_channel_names') != geometry_names:
                raise ValueError('Warm-start geometry channel order differs from data')
            model.load_state_dict(state['state_dict'])
            best_state = state
            start_epoch = state['epoch']
            history = config(paths.history)[:start_epoch]
            best = max(r['val_bacc10'] for r in history)
            save_json(paths.resume_note, {'warm_start_epoch': start_epoch,
                      'warning': 'First interrupted run saved weights only; optimizer and RNG reset for this continuation.'})
            print('Warm-starting saved weights; optimizer reset (recorded in resume_note.json)', flush=True)
        else:
            raise FileNotFoundError('No checkpoint to resume')
    save_json(paths.run_config, {"config": cfg, "versions": versions, "device": str(device),
                                 "devices": getattr(args, "devices", 1),
                                 "amp_enabled":amp, "num_workers":cfg.get('num_workers',0),
                                "data_mode": data_mode, "sample_fingerprint": fingerprint,
                                "parameters": sum(p.numel() for p in model.parameters())})
    print(f"Train: {cfg.get('model','temporal_cnn')} | {device} × {getattr(args,'devices',1)} | "
          f"{len(train_rows):,} train / {len(val_rows):,} val | batch {cfg['batch_size']} | {data_mode}",flush=True)
    epoch_bar=Progress(cfg['epochs'],'Epochs (maximum budget)',key='epochs',initial=start_epoch,leave=True)
    for epoch in range(start_epoch, cfg["epochs"]):
        if stale >= cfg["patience"]:
            break
        started = time.monotonic()
        model.train()
        total, n = 0.0, 0
        train_counts = np.zeros(N_CLASSES, dtype=np.int64)
        train_hits = np.zeros(N_CLASSES, dtype=np.int64)
        train_started=time.monotonic()
        data_wait=0.0
        previous=time.monotonic()
        for x, y in progress_iter(train_loader,len(train_loader),f'Train epoch {epoch+1}',key='batches'):
            data_wait += time.monotonic()-previous
            x, y = augment(x.to(device,non_blocking=device.type=='cuda')), y.to(device,non_blocking=device.type=='cuda')
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
                batch_logits = model(x)
                loss = criterion(batch_logits, y)
            with torch.no_grad():
                correct = (batch_logits.topk(10, dim=1).indices == y[:, None]).any(dim=1)
                train_counts += torch.bincount(y, minlength=N_CLASSES).cpu().numpy()
                train_hits += torch.bincount(y[correct], minlength=N_CLASSES).cpu().numpy()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(y)
            n += len(y)
            previous=time.monotonic()
        training_seconds=time.monotonic()-train_started
        scheduler.step()
        validation_started=time.monotonic()
        logits, labels = evaluate(model, val_loader, device)
        validation_seconds=time.monotonic()-validation_started
        top10, top1 = balanced_metrics(logits, labels), balanced_metrics(logits, labels, 1)
        train_recall = np.divide(train_hits, train_counts, out=np.zeros(N_CLASSES), where=train_counts > 0)
        # Standard unweighted CE: interpretable independently of class weighting.
        val_loss = torch.nn.functional.cross_entropy(torch.from_numpy(logits), torch.from_numpy(labels)).item()
        row = {"epoch": epoch + 1, "train_loss": total/n, "val_bacc10": top10["bacc"],
               "train_bacc10_online": float(train_recall.mean()), "val_loss": val_loss,
               "train_seconds":training_seconds,"validation_seconds":validation_seconds,
               "loader_wait_seconds":data_wait,"training_windows_per_second":n/training_seconds,
               "val_bacc1": top1["bacc"], "val_coverage": top10["coverage"], "seconds": time.monotonic()-started}
        history.append(row)
        print(f"Epoch {epoch+1:02d}/{cfg['epochs']} | loss {total/n:.3f} | val loss {val_loss:.3f} | "
              f"raw@10 {top10['bacc']:.2%} | raw@1 {top1['bacc']:.2%} | {row['seconds']:.0f}s",flush=True)
        if top10["bacc"] > best:
            best, stale = top10["bacc"], 0
            best_state = {"state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                          "config": cfg, "vocabulary": config(data / "vocabulary.json"), "epoch": epoch+1,
                          "train_counts": counts.tolist(), 'geometry_channel_names':geometry_names}
        else:
            stale += 1
        if best_state is None:
            best_state = torch.load(paths.best_checkpoint, map_location='cpu', weights_only=True)
        best_state['preprocessing_stats'] = stats
        if epoch+1 in cfg.get('save_epochs',[]):
            save_checkpoint(out/f'epoch_{epoch+1:03d}.pt',{
                'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                'config':cfg,'vocabulary':config(data/'vocabulary.json'),'epoch':epoch+1,
                'train_counts':counts.tolist(),'preprocessing_stats':stats,'prior_alpha':0.0,
                'geometry_channel_names':geometry_names})
        # latest.pt is authoritative, including best weights if interruption occurs
        # between the separate best/history writes.
        save_checkpoint(paths.latest_checkpoint, {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                      'scaler':scaler.state_dict(), 'amp_enabled':amp,
                      'geometry_channel_names':geometry_names,
                    'scheduler': scheduler.state_dict(), 'config': cfg, 'epoch': epoch+1,
                    'history': history, 'best': best, 'stale': stale,
                    'torch_rng': torch.get_rng_state(), 'numpy_rng': np.random.get_state(),
                     'python_rng': random.getstate(), 'loader_rng': generator.get_state(),
                     'sample_fingerprint': fingerprint, 'best_checkpoint': best_state,
                     'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None})
        save_checkpoint(paths.best_checkpoint, best_state)
        save_json(paths.history, history)
        epoch_bar.update(best=f'{best:.2%}',patience=f'{stale}/{cfg["patience"]}')
        if stale >= cfg["patience"]:
            break
    epoch_bar.close()
    checkpoint = torch.load(paths.best_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    logits, labels = evaluate(model, val_loader, device)
    # Cross-entropy weighting is intentionally moderate. A small prior correction
    # is selected on validation to target macro recall rather than class frequency.
    log_prior = np.log(counts/counts.sum())
    alpha, trials = choose_prior_alpha(logits, labels, log_prior)
    checkpoint["prior_alpha"] = alpha
    save_checkpoint(paths.best_checkpoint, checkpoint)
    corrected = logits - alpha*log_prior
    np.savez(paths.validation_predictions, logits=logits, labels=labels)
    val_rows.to_csv(paths.validation_rows, index=False)
    report = {"best_epoch": checkpoint["epoch"], "raw_top10": balanced_metrics(logits, labels),
              "corrected_top10": balanced_metrics(corrected, labels), "corrected_top1": balanced_metrics(corrected, labels, 1),
              "prior_alpha": alpha, "calibration_search": trials,
              "warning": "Validation also selects checkpoint/prior correction; not an independent test estimate."}
    save_json(paths.validation_report, report)
    from research import domain_report
    save_json(out/'domain_report.json',domain_report(logits,labels,val_rows,counts))
    if cfg.get('evaluate_training',False):
        clean_loader=make_loader(dataset(original_train_rows),cfg,device)
        train_logits,train_labels=evaluate(model,clean_loader,device)
        save_json(out/'training_evaluation.json',domain_report(train_logits,train_labels,original_train_rows,counts))
    save_json(out/'completed.json',{'config':cfg,'sample_fingerprint':fingerprint,
                                  'best_epoch':checkpoint['epoch']})
    print(f"Finished: best epoch {checkpoint['epoch']} | raw@10 {report['raw_top10']['bacc']:.2%} | "
          f"reports and checkpoints: {out}",flush=True)


def prediction_logits(logits, checkpoint, alpha=0.0):
    """Raw logits by default; correction is an explicit inference opt-in."""
    if alpha == 0:
        return logits
    counts = np.asarray(checkpoint["train_counts"], dtype=np.float64)
    if counts.shape != (N_CLASSES,) or np.any(counts <= 0):
        raise ValueError("Prior correction requires positive counts for all classes")
    return logits - alpha * np.log(counts / counts.sum())


def predict(args):
    """Run the trained model on the official deep holdout and write a CSV."""
    from pnpl.competition import LibriBrainCompetitionHoldout, load_vocabulary, write_submission
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = checkpoint["config"]
    if cfg.get('shuffle_training_labels',False):
        raise ValueError('Shuffled-label negative controls cannot generate competition submissions')
    seed_everything(cfg["seed"], cfg["threads"])
    if checkpoint["vocabulary"] != list(load_vocabulary("primary")):
        raise ValueError("Competition vocabulary differs from checkpoint")
    base = Path(args.data)
    for suffix in ["holdout01_sentence", "holdout2_word"]:
        download("pnpl/LibriBrain-Competition-2026", f"COMPETITION_HOLDOUT/subj00_{suffix}.npz", cfg, base)
    holdout = LibriBrainCompetitionHoldout(track="deep", data_path=base, download=False)
    model = build_model(cfg)
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()
    alpha = checkpoint.get("prior_alpha", 0.0) if getattr(args, "prior_corrected", False) else 0.0
    probabilities = []
    with torch.inference_mode():
        batches=(len(holdout.indices)+cfg['batch_size']-1)//cfg['batch_size']
        for meg, _ in progress_iter(holdout.iter_windows(batch_size=cfg["batch_size"]),batches,'Predict',key='predict'):
            from research import preprocess
            normalized = preprocess(meg,cfg.get('normalization','window_zscore'),checkpoint.get('preprocessing_stats'))
            logits = model(torch.from_numpy(normalized).to(device)).cpu().numpy()
            logits = prediction_logits(logits, checkpoint, alpha)
            probabilities.append(torch.from_numpy(logits).softmax(-1).numpy())
    scores = np.concatenate(probabilities)
    if scores.shape != (len(holdout.indices), N_CLASSES) or not np.isfinite(scores).all():
        raise ValueError("Invalid holdout predictions")
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    write_submission(args.csv, indices=holdout.indices, primary_probs=scores)
    save_json(Path(args.csv).with_suffix(".json"), {"rows": len(scores), "counts": holdout.counts(),
              "prediction_mode": "prior_corrected" if getattr(args, "prior_corrected", False) else "raw",
              "prior_alpha": alpha,
              "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
              "submission_sha256": hashlib.sha256(Path(args.csv).read_bytes()).hexdigest(),
              "competition": COMPETITION, "holdout_revision": cfg["revisions"]["pnpl/LibriBrain-Competition-2026"]})
    print(f"Wrote {args.csv}: {len(scores)} rows, 50 primary scores", flush=True)


def submit(args):
    """Submit an already-written CSV through the Kaggle CLI."""
    if not Path(args.csv).is_file():
        raise FileNotFoundError(args.csv)
    authenticated = bool(os.getenv("KAGGLE_API_TOKEN")) or bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")) or (Path.home()/".kaggle/kaggle.json").exists()
    if not authenticated:
        raise RuntimeError("No Kaggle credentials. Configure KAGGLE_API_TOKEN securely and accept competition rules on Kaggle.")
    subprocess.run(["kaggle", "competitions", "submit", "-c", COMPETITION, "-f", args.csv, "-m", args.message], check=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["workspace", "kaggle"], default="workspace",
                        help="Default writable paths; explicit subcommand paths take precedence")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--config", default=str(default_config_path()))
    p.add_argument("--data")
    p.add_argument("--raw", help="Raw H5/event directory, default: DATA/raw")
    p.add_argument("--index-only", action="store_true", help="Write samples_h5.csv without allocating epoch files")
    p.add_argument("--preflight", action="store_true", help="Report exact pinned H5/event sizes and check disk; no downloads")
    p.add_argument("--disk-budget-gib", type=float, help="Maximum existing + additional preparation data in GiB")
    p.add_argument("--headroom-gib", type=float, default=5.0, help="Filesystem free-space reserve (default: 5 GiB)")
    p.set_defaults(fn=prepare)
    p = sub.add_parser("train")
    p.add_argument("--config", default=str(default_config_path()))
    p.add_argument("--data")
    p.add_argument("--output")
    p.add_argument("--data-mode", choices=["auto", "epochs", "h5"], default="auto",
                   help="auto prefers existing samples.csv epoch cache, otherwise samples_h5.csv")
    p.add_argument("--device", choices=["cpu", "cuda"])
    p.add_argument("--resume", action="store_true", help="Resume trusted local latest.pt, or warm-start best.pt")
    p.add_argument("--devices", type=int, default=1, help="GPU count; 2 uses both Kaggle T4s with a global batch size")
    p.add_argument("--epochs", type=int, help="Override epoch budget")
    p.set_defaults(fn=train)
    p = sub.add_parser("predict")
    p.add_argument("--checkpoint")
    p.add_argument("--prior-corrected", action="store_true", help="Opt into checkpoint prior correction; default predictions are raw")
    p.add_argument("--data")
    p.add_argument("--csv")
    p.add_argument("--device", choices=["cpu", "cuda"])
    p.set_defaults(fn=predict)
    p = sub.add_parser("submit")
    p.add_argument("--csv")
    p.add_argument("--message", default="Single-trial residual temporal CNN; session-held-out validation")
    p.set_defaults(fn=submit)
    return parser


def resolve_paths(args):
    base = Path("/kaggle/working/pnpl-decoder") if args.profile == "kaggle" else ROOT
    defaults = {"data": base / ("data/holdout" if args.command == "predict" else "data"),
                "output": base / "artifacts/baseline", "checkpoint": base / "artifacts/baseline/best.pt",
                "csv": base / "artifacts/submission_deep.csv"}
    for name, value in defaults.items():
        if hasattr(args, name) and getattr(args, name) is None:
            setattr(args, name, str(value))
    return args


def main():
    args = resolve_paths(build_parser().parse_args())
    args.fn(args)


if __name__ == "__main__":
    main()
