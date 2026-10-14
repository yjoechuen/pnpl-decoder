import pytest
import torch

from decoder import build_model, config, configure_gpus, seed_everything, select_training_records
from models import MultiScaleConformer


def test_conformer_shapes_and_training():
    seed_everything(2026, threads=2)
    model = MultiScaleConformer(width=32, layers=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    before = model.head[-1].weight.detach().clone()
    logits = model(torch.randn(3, 306, 250))
    assert logits.shape == (3, 50)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2]))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert not torch.equal(before, model.head[-1].weight)


def test_checkpoint_roundtrip(tmp_path):
    cfg = dict(model='multiscale_conformer', width=32, dropout=0.2, conformer_layers=1)
    model = build_model(cfg).eval()
    x = torch.randn(2, 306, 250)
    path = tmp_path / 'model.pt'
    torch.save({'config': cfg, 'state_dict': model.state_dict()}, path)
    checkpoint = torch.load(path, weights_only=True)
    restored = build_model(checkpoint['config']).eval()
    restored.load_state_dict(checkpoint['state_dict'])
    with torch.no_grad():
        torch.testing.assert_close(model(x), restored(x))


def test_conformer_keeps_baseline_split():
    baseline, candidate = config('config.json'), config('config_conformer.json')
    assert select_training_records(baseline) == select_training_records(candidate)
    assert build_model(baseline).__class__.__name__ == 'TemporalCNN'
    with pytest.raises(ValueError):
        build_model(dict(model='unknown', width=32, dropout=0.2))


@pytest.mark.parametrize('kwargs', [{'width': 33}, {'heads': 3}, {'layers': 0}, {'kernel_size': 4}])
def test_invalid_model_config(kwargs):
    with pytest.raises(ValueError):
        MultiScaleConformer(**kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('devices', [1, 2])
def test_conformer_cuda_training(devices):
    if torch.cuda.device_count() < devices:
        pytest.skip('Not enough GPUs')
    seed_everything(2026, threads=2)
    model = configure_gpus(MultiScaleConformer(width=32, layers=1), torch.device('cuda'), devices)
    optimizer = torch.optim.AdamW(model.parameters())
    logits = model(torch.randn(8, 306, 250, device='cuda'))
    loss = torch.nn.functional.cross_entropy(logits, torch.arange(8, device='cuda'))
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
