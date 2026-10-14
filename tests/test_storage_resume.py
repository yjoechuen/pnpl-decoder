from argparse import Namespace
import json
import pickle
import random
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

import decoder as d


def recording(path, samples=500):
    values = np.random.default_rng(4).normal(size=(306, samples)).astype("float32") * 1e-13
    values[0] = 0
    with h5py.File(path, "w") as handle:
        handle.attrs["sample_frequency"] = 250
        handle.create_dataset("data", data=values)
    return values


def test_h5_matches_epochs_and_bounds_handles(tmp_path):
    values = recording(tmp_path / "one.h5")
    recording(tmp_path / "two.h5")
    normalized = d.normalize_windows(values[:, 100:350])
    np.save(tmp_path / "epochs.npy", normalized[None])
    frame = pd.DataFrame({"h5_file": ["one.h5", "two.h5"], "start_sample": [100, 0],
                          "label": [7, 8], "file": [str(tmp_path / "epochs.npy")] * 2, "epoch": [0, 0]})
    dataset = d.H5WindowDataset(frame, tmp_path, max_open_files=1)
    actual, label = dataset[0]
    torch.testing.assert_close(actual, d.EpochDataset(frame)[0][0], rtol=0, atol=0)
    assert label == 7
    first_handle = next(iter(dataset.handles.values()))
    dataset[1]
    assert not first_handle.id.valid
    assert len(dataset.handles) == 1
    copied = pickle.loads(pickle.dumps(dataset))
    assert not copied.handles
    torch.testing.assert_close(copied[0][0], actual)
    dataset.close()
    copied.close()


def test_h5_boundaries_and_format(tmp_path):
    recording(tmp_path / "one.h5", samples=250)
    frame = pd.DataFrame({"h5_file": ["one.h5"], "start_sample": [1], "label": [0]})
    dataset = d.H5WindowDataset(frame, tmp_path)
    with pytest.raises(ValueError, match="boundary"):
        dataset[0]
    dataset.close()
    frame["start_sample"] = -1
    with pytest.raises(ValueError, match="start sample"):
        d.H5WindowDataset(frame, tmp_path)
    with h5py.File(tmp_path / "one.h5", "r+") as handle:
        handle.attrs["sample_frequency"] = 100
    frame["start_sample"] = 0
    with pytest.raises(ValueError, match="format"):
        d.H5WindowDataset(frame, tmp_path)[0]


def test_protected_raw_paths_rejected(tmp_path):
    (tmp_path / "FINAL_TEST_ONLY").touch()
    frame = pd.DataFrame({"h5_file": ["raw/test.h5"], "start_sample": [0], "label": [0]})
    with pytest.raises(ValueError, match="Final-test"):
        d.H5WindowDataset(frame, tmp_path)
    with pytest.raises(ValueError, match="Final-test"):
        d.EpochDataset(pd.DataFrame({"file": [str(tmp_path / "raw/test.npy")], "epoch": [0], "label": [0]}))


def test_exact_download_plan_and_missing_metadata(tmp_path, monkeypatch):
    record = dict(repo="repo", task="task", subject="0", session="1", run="1", partition="train", split="train")
    event, h5 = d.recording_names(record)
    sizes = {event: 10, h5: 100}
    calls = []

    def info(repo, names, **kwargs):
        calls.append(kwargs)
        return [SimpleNamespace(path=name, size=sizes[name]) for name in names]

    monkeypatch.setattr(d, "HfApi", lambda: SimpleNamespace(get_paths_info=info))
    local = tmp_path / event
    local.parent.mkdir(parents=True)
    local.write_bytes(b"x" * 10)
    plan = d.download_plan([record], {"revisions": {"repo": "pinned"}}, tmp_path)
    assert (plan["raw_bytes"], plan["h5_bytes"], plan["event_bytes"], plan["download_bytes"]) == (110, 100, 10, 100)
    assert calls == [{"repo_type": "dataset", "revision": "pinned"}]
    sizes[h5] = None
    with pytest.raises(ValueError, match="exact sizes"):
        d.download_plan([record], {"revisions": {"repo": "pinned"}}, tmp_path)


def test_disk_budget_headroom_and_no_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(d.shutil, "disk_usage", lambda path: SimpleNamespace(free=100))
    absent = tmp_path / "absent"
    d.check_disk_space(absent, absent / "raw", 50, 40, None, 0)
    with pytest.raises(RuntimeError, match="headroom"):
        d.check_disk_space(absent, absent / "raw", 50, 60, None, 0)
    with pytest.raises(RuntimeError, match="budget"):
        d.check_disk_space(absent, absent / "raw", 50, 40, 80 / d.GIB, 0)
    assert not absent.exists()
    with pytest.raises(ValueError):
        d.check_disk_space(absent, absent, 0, 0, None, -1)


def test_preflight_and_index_only_preserve_epochs(tmp_path, monkeypatch):
    cfg = d.config(d.ROOT / "config.json")
    config_path = tmp_path / "config.json"
    d.save_json(config_path, cfg)
    records = [dict(repo="pnpl/LibriBrain", task="Sherlock1", subject="0", session=str(i), run="1",
                    corpus="sherlock", partition="train", split=split) for i, split in [(1, "train"), (2, "val")]]
    source = tmp_path / "source"
    source.mkdir()
    files = {}
    for record in records:
        event_name, h5_name = d.recording_names(record)
        event = source / f"{record['session']}.tsv"
        pd.DataFrame({"kind": ["word"] * 53, "segment": [f"word{i}" for i in range(50)] + ["word0"] * 3,
                      "timemeg": [0] * 50 + [float("inf"), -1, 2]}).to_csv(event, sep="\t", index=False)
        h5 = source / f"{record['session']}.h5"
        recording(h5)
        files[event_name], files[h5_name] = event, h5
    monkeypatch.setattr(d, "select_training_records", lambda cfg: records)
    monkeypatch.setattr(d, "competition_vocabulary_mapping", lambda: ([f"word{i}" for i in range(50)], {f"word{i}": i for i in range(50)}))
    monkeypatch.setattr(d, "HfApi", lambda: SimpleNamespace(get_paths_info=lambda repo, names, **kw:
                        [SimpleNamespace(path=n, size=files[n].stat().st_size) for n in names]))
    downloads = []

    def download(repo, name, cfg, directory):
        downloads.append(name)
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        d.shutil.copyfile(files[name], target)
        return target

    monkeypatch.setattr(d, "download", download)
    out = tmp_path / "prepared"
    args = Namespace(config=config_path, data=out, raw=None, index_only=True, preflight=True,
                     headroom_gib=0, disk_budget_gib=1)
    d.prepare(args)
    assert not downloads and not out.exists()
    out.mkdir()
    (out / "epochs").mkdir()
    sentinel = out / "epochs/keep.npy"
    sentinel.write_bytes(b"existing epochs")
    (out / "samples.csv").write_text("existing index\n")
    args.preflight = False
    d.prepare(args)
    assert sentinel.read_bytes() == b"existing epochs"
    assert (out / "samples.csv").read_text() == "existing index\n"
    assert list((out / "epochs").iterdir()) == [sentinel]
    frame = pd.read_csv(out / "samples_h5.csv")
    assert len(frame) == 100
    assert frame.start_sample.eq(0).all()
    assert all(not d.Path(path).is_absolute() for path in frame.h5_file)
    assert d.config(out / "data_report_h5.json")["records"][0]["boundary_dropped"] == 3
    d.split_sample_frame(frame)
    args.index_only = False
    args.disk_budget_gib = .01
    downloads.clear()
    with pytest.raises(RuntimeError, match="budget"):
        d.prepare(args)
    assert downloads and all(name.endswith(".tsv") for name in downloads)
    assert (out / "samples.csv").read_text() == "existing index\n"
    args.disk_budget_gib = 1
    d.prepare(args)
    epoch_frame, mode = d.training_frame(out, "auto")
    assert mode == "epochs"
    h5_dataset = d.H5WindowDataset(frame, out)
    torch.testing.assert_close(d.EpochDataset(epoch_frame)[0][0], h5_dataset[0][0], rtol=0, atol=0)
    h5_dataset.close()
    assert sentinel.read_bytes() == b"existing epochs"


def test_atomic_checkpoint_failure_preserves_previous(tmp_path, monkeypatch):
    path = tmp_path / "latest.pt"
    d.save_checkpoint(path, {"epoch": 1})
    original = path.read_bytes()

    def interrupted(state, tmp):
        tmp.write_bytes(b"partial")
        raise OSError("interrupted")

    monkeypatch.setattr(d.torch, "save", interrupted)
    with pytest.raises(OSError):
        d.save_checkpoint(path, {"epoch": 2})
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_restore_optimizer_scheduler_and_all_cpu_rngs(tmp_path):
    d.seed_everything(42, threads=2)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 4)
    generator = torch.Generator().manual_seed(7)
    # A synthetic optimizer step populates moments; no dataset or CLI training.
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    state = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "torch_rng": torch.get_rng_state(),
             "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
             "loader_rng": generator.get_state(), "cuda_rng": None}
    d.save_checkpoint(tmp_path / "latest.pt", state)
    expected = (torch.rand(3), np.random.rand(3), random.random(), torch.randperm(10, generator=generator))
    other = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(other.parameters(), lr=.5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 4)
    loaded = torch.load(tmp_path / "latest.pt", weights_only=False)
    d.restore_training_state(loaded, other, opt, sched, generator, torch.device("cpu"))
    for actual, wanted in zip(other.parameters(), model.parameters()):
        torch.testing.assert_close(actual, wanted)
    assert sched.state_dict() == scheduler.state_dict()
    assert opt.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    for actual, wanted in zip(opt.state.values(), optimizer.state.values()):
        for key in actual:
            torch.testing.assert_close(actual[key], wanted[key])
    torch.testing.assert_close(torch.rand(3), expected[0], rtol=0, atol=0)
    np.testing.assert_array_equal(np.random.rand(3), expected[1])
    assert random.random() == expected[2]
    torch.testing.assert_close(torch.randperm(10, generator=generator), expected[3])


def test_cli_profiles_and_legacy_auto_mode(tmp_path):
    args = d.resolve_paths(d.build_parser().parse_args(["--profile", "kaggle", "train", "--data", "/custom"]))
    assert args.data == "/custom"
    assert args.output == "/kaggle/working/pnpl-decoder/artifacts/baseline"
    args = d.resolve_paths(d.build_parser().parse_args(["prepare"]))
    assert args.data == str(d.ROOT / "data")
    pd.DataFrame({"file": ["legacy.npy"], "epoch": [0], "label": [0]}).to_csv(tmp_path / "samples.csv", index=False)
    _, mode = d.training_frame(tmp_path, "auto")
    assert mode == "epochs"
    with pytest.raises(ValueError, match="index missing"):
        d.training_frame(tmp_path, "h5")


def test_installed_config_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "ROOT", tmp_path / "site-packages")
    monkeypatch.setattr(d.sys, "prefix", str(tmp_path))
    installed = tmp_path / "share/pnpl-decoder/config.json"
    d.save_json(installed, {"installed": True})
    assert d.default_config_path() == installed
    args = d.build_parser().parse_args(["train"])
    assert d.config(args.config) == {"installed": True}


def resume_fixture(tmp_path):
    cfg = d.config(d.ROOT / "config.json")
    cfg["epochs"] = 4
    data, out = tmp_path / "data", tmp_path / "output"
    data.mkdir()
    out.mkdir()
    d.save_json(tmp_path / "config.json", cfg)
    d.save_json(data / "prepared_config.json", cfg)
    d.save_json(data / "vocabulary.json", [str(i) for i in range(50)])
    frame = pd.DataFrame({"file": [str(data / "never-read.npy")] * 100, "epoch": list(range(100)),
                          "label": list(range(50)) * 2, "split": ["train"] * 50 + ["val"] * 50,
                          "task": ["task"] * 100, "subject": [0] * 100, "session": [1] * 50 + [2] * 50})
    frame.to_csv(data / "samples.csv", index=False)
    model = torch.nn.Linear(2, 50)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["epochs"])
    best = {"state_dict": model.state_dict(), "config": cfg, "epoch": 1,
            "vocabulary": [str(i) for i in range(50)], "train_counts": [1] * 50}
    state = {"state_dict": model.state_dict(), "config": cfg, "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "history": [{"epoch": 1, "val_bacc10": .2}],
             "best": .2, "stale": cfg["patience"], "epoch": 1, "best_checkpoint": best,
             "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
             "python_rng": random.getstate(), "loader_rng": torch.Generator().get_state(), "cuda_rng": None,
             "sample_fingerprint": d.sample_fingerprint(pd.read_csv(data / "samples.csv"))}
    args = Namespace(config=tmp_path / "config.json", data=data, output=out,
                     epochs=None, resume=True, device="cpu", data_mode="auto")
    return args, state


def test_resume_early_stopped_repairs_best_without_training(tmp_path, monkeypatch):
    args, state = resume_fixture(tmp_path)
    d.save_checkpoint(args.output / "latest.pt", state)
    monkeypatch.setattr(d, "build_model", lambda cfg: torch.nn.Linear(2, 50))
    monkeypatch.setattr(d, "evaluate", lambda *args: (np.eye(50), np.arange(50)))
    monkeypatch.setattr(d, "augment", lambda x: pytest.fail("must not optimize an early-stopped run"))
    d.train(args)
    restored = torch.load(args.output / "best.pt", weights_only=True)
    assert restored["epoch"] == 1 and "prior_alpha" in restored
    assert d.config(args.output / "history.json") == state["history"]
    assert torch.load(args.output / "latest.pt", weights_only=False)["epoch"] == 1


def test_resume_mismatch_preserves_run_metadata_and_no_overwrite(tmp_path, monkeypatch):
    args, state = resume_fixture(tmp_path)
    state["sample_fingerprint"] = "changed"
    d.save_checkpoint(args.output / "latest.pt", state)
    d.save_json(args.output / "run_config.json", {"original": True})
    monkeypatch.setattr(d, "build_model", lambda cfg: torch.nn.Linear(2, 50))
    with pytest.raises(ValueError, match="sample index"):
        d.train(args)
    assert d.config(args.output / "run_config.json") == {"original": True}
    args.resume = False
    with pytest.raises(ValueError, match="already contains"):
        d.train(args)


def test_kaggle_notebook_is_thin_and_requires_full_pin():
    notebook = json.loads((d.ROOT / "kaggle_entry.ipynb").read_text())
    code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "[0-9a-f]{40}" in code
    assert "pnpl-decoder" in code and "--preflight" in code and "--index-only" in code
    assert "optimizer" not in code and "loss.backward" not in code


@pytest.mark.parametrize("custom_config,total_epochs,resume", [(None, None, False), ("/custom/config.json", 6, True)])
def test_kaggle_notebook_shares_config_and_original_epoch_override(monkeypatch, custom_config, total_epochs, resume):
    notebook = json.loads((d.ROOT / "kaggle_entry.ipynb").read_text())
    code = "".join(notebook["cells"][-1]["source"])
    calls = []
    import subprocess

    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: calls.append(command.copy()))
    exec(compile(code, "kaggle_entry.ipynb", "exec"), {
        "subprocess": subprocess, "CONFIG": custom_config, "TOTAL_EPOCHS": total_epochs,
        "RESUME": resume, "DATA": d.Path("/data"), "OUTPUT": d.Path("/output"),
        "DISK_BUDGET_GIB": 15, "HEADROOM_GIB": 5,
    })
    assert len(calls) == 3
    for command in calls:
        assert command[command.index("--config") + 1] == str(custom_config or d.default_config_path())
    train = calls[-1]
    if total_epochs is None:
        assert "--epochs" not in train
    else:
        assert train[train.index("--epochs") + 1] == str(total_epochs)
    assert ("--resume" in train) == resume
    assert all("--epochs" not in command for command in calls[:2])
