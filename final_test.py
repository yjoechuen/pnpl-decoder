"""Prepare the reserved official test session without evaluating a model.

This is an accidental-use guard, not a security sandbox. A future search worker
must be isolated and must not receive a mount of this cache or raw source data.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil

import h5py
from huggingface_hub import HfApi, hf_hub_download
import numpy as np
import pandas as pd

from decoder import (
    ROOT, PROC, N_CHANNELS, WINDOW_SAMPLES, competition_vocabulary_mapping,
    atomic_output, config, normalize_windows, resource_path, run_stem, save_json, word_key,
)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_disjoint(reserved, development):
    """Exclude the whole session, not just its selected run."""
    keys = ("task", "subject", "session")
    group = tuple(str(reserved[k]) for k in keys)
    if any(tuple(str(row[k]) for k in keys) == group for row in development):
        raise ValueError("Reserved final-test session overlaps development data")


def prepare(args):
    out = Path(args.output).resolve()
    for protected in [ROOT.resolve(), Path(args.development).resolve()]:
        if out.is_relative_to(protected) or protected.is_relative_to(out):
            raise ValueError("Final-test output must not overlap the repository or development directory")

    from pnpl.datasets.libribrain100.constants import REPO_KEY_TO_ID
    from pnpl.datasets.libribrain100.manifest import select_records

    spec = config(args.spec)
    assert_disjoint(spec, config(Path(args.development) / "split_manifest.json"))
    official = [asdict(r) for r in select_records(
        subjects={spec["subject"]}, corpus={"sherlock"}, partition="test"
    )]
    if not any(
        all(str(row[k]) == spec[k] for k in ("subject", "session", "task", "run", "partition"))
        and REPO_KEY_TO_ID[row["repo"]] == spec["repo"] for row in official
    ):
        raise ValueError("Reservation is not an official PNPL test record")

    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(out, 0o700)
    with atomic_output(out / "FINAL_TEST_ONLY") as marker:
        marker.write_text("Never expose this directory to search/training workers.\n")
    manifest = out / "manifest.json"
    if manifest.exists():
        existing = config(manifest)
        if existing["reservation"] != spec:
            raise ValueError("Existing final-test reservation differs")
        if existing["development_manifest_sha256"] != file_hash(Path(args.development) / "split_manifest.json"):
            raise ValueError("Development manifest changed; audit the reservation before reuse")
        for name, checksum in existing["cache_sha256"].items():
            if file_hash(out / name) != checksum:
                raise ValueError(f"Cache checksum differs: {name}")
        print("Existing final-test cache verified; no model evaluated.")
        return

    stem = run_stem(spec)
    names = [f"Sherlock1/derivatives/events/{stem}_events.tsv",
             f"Sherlock1/derivatives/serialised/{stem}_proc-{PROC}_meg.h5"]
    info = HfApi().get_paths_info(spec["repo"], names, repo_type="dataset", revision=spec["revision"])
    sizes = {item.path: item.size for item in info}
    if set(sizes) != set(names):
        raise ValueError("Reserved files are missing from pinned revision")
    # Conservatively allow a full uncompressed recording-sized epoch cache plus
    # overlapping word windows, and retain 5 GiB filesystem headroom.
    raw_bytes = sum(sizes.values())
    if shutil.disk_usage(out).free < raw_bytes * 5 + 5 * 1024**3:
        raise RuntimeError("Insufficient disk headroom for final-test preparation")
    files = [Path(hf_hub_download(
        repo_id=spec["repo"], repo_type="dataset", revision=spec["revision"],
        filename=name, local_dir=out / "raw",
    )) for name in names]
    vocab, mapping = competition_vocabulary_mapping()
    events = pd.read_csv(files[0], sep="\t")
    required = {"kind", "segment", "timemeg"}
    if not required.issubset(events):
        raise ValueError("Missing required event columns")
    words = events.loc[events.kind.eq("word")].copy()
    words["label"] = words.segment.map(word_key).map(mapping)
    words = words.dropna(subset=["label", "timemeg"])
    valid, dropped = [], 0
    with h5py.File(files[1], "r") as recording:
        if float(recording.attrs["sample_frequency"]) != 250 or recording["data"].shape[0] != N_CHANNELS:
            raise ValueError("Unexpected MEG format")
        for index, row in words.iterrows():
            onset = float(row.timemeg)
            if not np.isfinite(onset):
                dropped += 1
                continue
            start = int(onset * 250)
            if start < 0 or start + WINDOW_SAMPLES > recording["data"].shape[1]:
                dropped += 1
                continue
            valid.append((index, start, int(row.label)))
        if not valid:
            raise ValueError("No valid in-vocabulary final-test windows")
        cache_bytes = len(valid) * N_CHANNELS * WINDOW_SAMPLES * 4
        if shutil.disk_usage(out).free < cache_bytes + 5 * 1024**3:
            raise RuntimeError("Insufficient headroom for epoch cache")
        epochs = np.lib.format.open_memmap(out / "windows.tmp.npy", mode="w+", dtype="float32",
                                           shape=(len(valid), N_CHANNELS, WINDOW_SAMPLES))
        for i, (_, start, _) in enumerate(valid):
            epochs[i] = normalize_windows(recording["data"][:, start:start + WINDOW_SAMPLES])
        epochs.flush()
        del epochs
    (out / "windows.tmp.npy").replace(out / "windows.npy")
    np.save(out / "labels.npy", np.array([label for _, _, label in valid], dtype=np.int64))
    save_json(out / "rows.json", [{"row": i, "event_index": int(idx), "start_sample": start}
                                 for i, (idx, start, _) in enumerate(valid)])
    save_json(out / "vocabulary.json", vocab)
    counts = np.bincount([label for _, _, label in valid], minlength=len(vocab))
    cache_names = ["windows.npy", "labels.npy", "rows.json", "vocabulary.json"]
    save_json(manifest, {
        "reservation": spec, "split": "final_test", "n_windows": len(valid),
        "shape": [len(valid), N_CHANNELS, WINDOW_SAMPLES], "class_counts": counts.tolist(),
        "missing_classes": [vocab[i] for i, n in enumerate(counts) if n == 0],
        "boundary_or_nonfinite_onsets_dropped": dropped,
        "normalization": "per-window, per-channel population z-score; clip [-10,10]",
        "source_sha256": {name: file_hash(path) for name, path in zip(names, files)},
        "cache_sha256": {name: file_hash(out / name) for name in cache_names},
        "development_manifest_sha256": file_hash(Path(args.development) / "split_manifest.json"),
        "protection": "Owner-only directory; accidental-use guards, not isolation from same-user code",
    })
    print(json.dumps({"cache": str(out), "windows": len(valid), "class_coverage": int((counts > 0).sum()),
                      "model_evaluated": False}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default=str(resource_path("final_test.json")))
    parser.add_argument("--development", default=str(ROOT / "data"))
    parser.add_argument("--output", default="/workspace/protected/pnpl-final-test")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
