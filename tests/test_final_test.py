from argparse import Namespace
from dataclasses import make_dataclass

import pandas as pd
import pytest

import decoder
import final_test
from decoder import split_sample_frame, verify_prepared_config
from final_test import assert_disjoint


def test_reserved_session_excludes_every_run():
    reserved = {"task": "Sherlock1", "subject": "0", "session": "12"}
    assert_disjoint(reserved, [{**reserved, "session": "1"}])
    with pytest.raises(ValueError, match="overlaps"):
        assert_disjoint(reserved, [{**reserved, "run": "1"}])


def test_final_test_directory_rejected_by_training(tmp_path):
    (tmp_path / "FINAL_TEST_ONLY").touch()
    with pytest.raises(ValueError, match="Final-test"):
        verify_prepared_config(tmp_path, {})


def test_final_test_rows_rejected_by_training():
    frame = pd.DataFrame({"split": ["train", "val", "final_test"]})
    with pytest.raises(ValueError, match="forbidden"):
        split_sample_frame(frame)


def test_official_test_partition_rejected_by_training():
    frame = pd.DataFrame({"split": ["train", "val"], "partition": ["train", "test"]})
    with pytest.raises(ValueError, match="non-training"):
        split_sample_frame(frame)


@pytest.mark.parametrize("protected_name", ["repository", "development"])
@pytest.mark.parametrize("relationship", ["equal", "ancestor", "descendant", "symlink"])
def test_output_overlap_rejected_before_writes(tmp_path, monkeypatch, protected_name, relationship):
    root = tmp_path / "repository/repo"
    development = tmp_path / "external/development"
    root.mkdir(parents=True)
    development.mkdir(parents=True)
    monkeypatch.setattr(final_test, "ROOT", root)
    protected = root if protected_name == "repository" else development
    output = {"equal": protected, "ancestor": protected.parent,
              "descendant": protected / "final-test", "symlink": tmp_path / "alias"}[relationship]
    if relationship == "symlink":
        output.symlink_to(protected, target_is_directory=True)
    before = {path: path.stat().st_mode for path in [tmp_path, root, root.parent, development, development.parent]}
    monkeypatch.setattr(final_test.os, "chmod", lambda *args: pytest.fail("must not chmod overlapping output"))
    monkeypatch.setattr(final_test, "config", lambda *args: pytest.fail("validate paths before reading data"))
    with pytest.raises(ValueError, match="overlap"):
        final_test.prepare(Namespace(output=output, development=development, spec=tmp_path / "unused.json"))
    assert all(path.stat().st_mode == mode for path, mode in before.items())
    assert not list(tmp_path.rglob("FINAL_TEST_ONLY"))
    assert not (protected / "final-test").exists()


@pytest.mark.parametrize("marker_contents", [None, "incorrect marker\n"])
def test_manifest_reuse_restores_marker_without_downloads(tmp_path, monkeypatch, marker_contents):
    from pnpl.datasets.libribrain100 import manifest
    from pnpl.datasets.libribrain100.constants import REPO_KEY_TO_ID

    monkeypatch.setattr(final_test, "ROOT", tmp_path / "repository")
    development, output = tmp_path / "development", tmp_path / "protected"
    development.mkdir()
    output.mkdir()
    spec = decoder.config(decoder.ROOT / "final_test.json")
    decoder.save_json(tmp_path / "spec.json", spec)
    decoder.save_json(development / "split_manifest.json", [])
    repo_key = next(key for key, value in REPO_KEY_TO_ID.items() if value == spec["repo"])
    fields = ("subject", "session", "task", "run", "partition", "repo")
    record_type = make_dataclass("Record", fields)
    record = record_type(**{key: spec[key] if key != "repo" else repo_key for key in fields})
    monkeypatch.setattr(manifest, "select_records", lambda **kwargs: [record])
    (output / "rows.json").write_text("[]\n")
    decoder.save_json(output / "manifest.json", {
        "reservation": spec,
        "development_manifest_sha256": final_test.file_hash(development / "split_manifest.json"),
        "cache_sha256": {"rows.json": final_test.file_hash(output / "rows.json")},
    })
    marker = output / "FINAL_TEST_ONLY"
    if marker_contents is not None:
        marker.write_text(marker_contents)
    monkeypatch.setattr(final_test, "HfApi", lambda: pytest.fail("reuse must not fetch metadata"))
    monkeypatch.setattr(final_test, "hf_hub_download", lambda **kwargs: pytest.fail("reuse must not download"))
    original_manifest = (output / "manifest.json").read_bytes()
    final_test.prepare(Namespace(spec=tmp_path / "spec.json", development=development, output=output))
    assert marker.read_text() == "Never expose this directory to search/training workers.\n"
    assert (output / "manifest.json").read_bytes() == original_manifest
    assert output.stat().st_mode & 0o777 == 0o700
    with pytest.raises(ValueError, match="Final-test"):
        verify_prepared_config(output, {})


def test_installed_final_test_spec_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(decoder, "ROOT", tmp_path / "site-packages")
    monkeypatch.setattr(decoder.sys, "prefix", str(tmp_path))
    installed = tmp_path / "share/pnpl-decoder/final_test.json"
    decoder.save_json(installed, {"installed": True})
    monkeypatch.setattr(final_test, "prepare", lambda args: observed.append(args))
    monkeypatch.setattr(decoder.sys, "argv", ["final_test.py"])
    observed = []
    final_test.main()
    assert observed[0].spec == str(installed)
    assert decoder.config(observed[0].spec) == {"installed": True}
    local = decoder.ROOT / "final_test.json"
    decoder.save_json(local, {"local": True})
    final_test.main()
    assert observed[1].spec == str(local)
