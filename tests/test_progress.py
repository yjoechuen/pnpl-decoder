import json
import subprocess
import sys

import pytest

from progress import Progress,PREFIX
from kaggle_runner import notebook_process


def test_events_throttled_and_completed(monkeypatch,capsys):
    monkeypatch.setenv('PNPL_NOTEBOOK_PROGRESS','1')
    bar=Progress(3,'Test')
    for _ in range(3):bar.update()
    bar.close()
    lines=capsys.readouterr().out.splitlines()
    assert len(lines)==2
    assert json.loads(lines[-1][len(PREFIX):])['n']==3
    assert json.loads(lines[-1][len(PREFIX):])['closed']


def test_notebook_streams_messages_and_raises(capsys):
    notebook_process([sys.executable,'-c',"print('useful summary')"])
    assert 'useful summary' in capsys.readouterr().out
    with pytest.raises(subprocess.CalledProcessError):
        notebook_process([sys.executable,'-c',"raise RuntimeError('visible failure')"])
    assert 'visible failure' in capsys.readouterr().out
