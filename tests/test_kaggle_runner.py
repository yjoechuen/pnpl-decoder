from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from decoder import PortableDataParallel, TemporalCNN, configure_gpus
from kaggle_runner import KaggleExperiment


def test_portable_weights():
    model = TemporalCNN(width=32)
    wrapped = PortableDataParallel(model)
    assert not any(k.startswith('module.') for k in wrapped.state_dict())
    clone = TemporalCNN(width=32)
    clone.load_state_dict(wrapped.state_dict())
    wrapped.load_state_dict(clone.state_dict())
    assert configure_gpus(clone, torch.device('cpu'), 1) is clone
    with pytest.raises(ValueError):
        configure_gpus(clone, torch.device('cpu'), 2)


def test_runner_arguments():
    run = KaggleExperiment(repo='/tmp/repo', output='/tmp/run')
    with patch('kaggle_runner.subprocess.run') as launch:
        run.train(devices=2, epochs=12, resume=True)
        args = launch.call_args.args[0]
        assert args[args.index('--devices') + 1] == '2'
        assert '--resume' in args
        assert args[args.index('--output') + 1] == '/tmp/run'
        assert run.predict('/tmp/submission.csv') == Path('/tmp/submission.csv')


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='Two CUDA GPUs required')
def test_two_gpu_backward_and_checkpoint():
    from decoder import seed_everything
    seed_everything(2026, threads=2)
    model = configure_gpus(TemporalCNN(width=32), torch.device('cuda'), 2)
    optimizer = torch.optim.AdamW(model.parameters())
    loss = torch.nn.functional.cross_entropy(model(torch.randn(8, 306, 250, device='cuda')),
                                            torch.arange(8, device='cuda'))
    loss.backward()
    optimizer.step()
    TemporalCNN(width=32).load_state_dict(model.state_dict())
    assert torch.isfinite(loss)
