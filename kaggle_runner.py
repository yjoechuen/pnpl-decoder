"""Notebook helpers that launch fresh processes, avoiding stale Python imports."""
from pathlib import Path
import subprocess
import sys


def sync_repo(ref="main", destination="/kaggle/working/pnpl-decoder",
              url="https://github.com/yjoechuen/pnpl-decoder.git"):
    """Clone/update a clean checkout. A commit SHA pins a reproducible run."""
    repo = Path(destination)
    if not repo.parent.is_dir():
        raise FileNotFoundError(repo.parent)
    if not repo.exists():
        subprocess.run(["git", "clone", url, str(repo)], check=True)
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)
    if status.strip():
        raise RuntimeError("Local edits exist; preserve them before syncing")
    subprocess.run(["git", "fetch", "origin"], cwd=repo, check=True)
    subprocess.run(["git", "checkout", ref], cwd=repo, check=True)
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip()
    if branch:
        subprocess.run(["git", "pull", "--ff-only", "origin", branch], cwd=repo, check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(repo)], check=True)
    return repo


class KaggleExperiment:
    """Prepare, train on T4 x2, then predict with a portable best checkpoint."""

    def __init__(self, repo="/kaggle/working/pnpl-decoder", data="/kaggle/working/pnpl-data",
                 output="/kaggle/working/baseline", config=None):
        self.repo, self.data, self.output = map(Path, (repo, data, output))
        self.config = Path(config) if config else self.repo / "config.json"

    def _run(self, command, *args):
        subprocess.run([sys.executable, str(self.repo / "decoder.py"), command, *map(str, args)], check=True)

    def prepare(self, index_only=True, raw=None, preflight=False, disk_budget_gib=None):
        args = ["--config", self.config, "--data", self.data]
        if index_only:
            args += ["--index-only"]
        if raw is not None:
            args += ["--raw", raw]
        if preflight:
            args += ["--preflight"]
        if disk_budget_gib is not None:
            args += ["--disk-budget-gib", disk_budget_gib]
        self._run("prepare", *args)

    def train(self, devices=2, epochs=None, resume=False, data_mode='auto'):
        args = ["--config", self.config, "--data", self.data, "--output", self.output,
                "--device", "cuda", "--devices", devices]
        if epochs is not None:
            args += ["--epochs", epochs]
        if resume:
            args += ["--resume"]
        args += ['--data-mode',data_mode]
        self._run("train", *args)

    def predict(self, csv="/kaggle/working/submission_deep.csv", prior_corrected=False):
        args = ["--checkpoint", self.output / "best.pt",
                "--data", self.data / "holdout", "--csv", csv, "--device", "cuda"]
        if prior_corrected:
            args += ["--prior-corrected"]
        self._run("predict", *args)
        return Path(csv)
