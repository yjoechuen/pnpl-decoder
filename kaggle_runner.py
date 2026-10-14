"""Notebook helpers that launch fresh processes, avoiding stale Python imports."""
from pathlib import Path
import json
import os
import subprocess
import sys


def notebook_process(command):
    """Render throttled child events as two reusable notebook bars; stream errors."""
    from tqdm.auto import tqdm
    from progress import PREFIX
    env=dict(os.environ,PNPL_NOTEBOOK_PROGRESS='1',HF_HUB_DISABLE_PROGRESS_BARS='1',PYTHONUNBUFFERED='1')
    bars={}
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                             text=True,bufsize=1,env=env)
    try:
        for line in process.stdout:
            if not line.startswith(PREFIX):
                print(line.rstrip(),flush=True)
                continue
            event=json.loads(line[len(PREFIX):])
            key=event['key']
            if key not in bars:
                bars[key]=tqdm(total=event['total'],desc=event['description'],mininterval=1,
                               leave=key=='epochs')
            bar=bars[key]
            if event['n']<bar.n or event['total']!=bar.total:
                bar.reset(total=event['total'])
            bar.set_description(event['description'],refresh=False)
            bar.update(event['n']-bar.n)
            if event['metrics']:bar.set_postfix(event['metrics'],refresh=False)
            if event['closed']:
                bar.close()
                del bars[key]
        code=process.wait()
        if code:raise subprocess.CalledProcessError(code,command)
    finally:
        for bar in bars.values():bar.close()
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill();process.wait()
        process.stdout.close()


def in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None and getattr(get_ipython(),'kernel',None) is not None
    except ImportError:
        return False


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
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-e", str(repo)], check=True)
    return repo


class KaggleExperiment:
    """Prepare, train on T4 x2, then predict with a portable best checkpoint."""

    def __init__(self, repo="/kaggle/working/pnpl-decoder", data="/kaggle/working/pnpl-data",
                 output="/kaggle/working/baseline", config=None, notebook_progress=None):
        self.repo, self.data, self.output = map(Path, (repo, data, output))
        self.config = Path(config) if config else self.repo / "config.json"
        self.notebook_progress=in_notebook() if notebook_progress is None else notebook_progress

    def _run(self, command, *args):
        cmd=[sys.executable,str(self.repo/'decoder.py'),command,*map(str,args)]
        if self.notebook_progress:notebook_process(cmd)
        else:subprocess.run(cmd,check=True)

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
