import numpy as np
import pytest
import torch

from decoder import build_model, config, seed_everything, select_training_records, verify_prepared_config
from models import WideResidualCNN


def test_wide_cnn_optimizer_and_checkpoint(tmp_path):
    seed_everything(2026, threads=2)
    cfg = dict(model='wide_residual_cnn', width=32, dropout=0.1)
    model = build_model(cfg)
    x, y = torch.randn(4, 306, 250), torch.arange(4)
    before = model.head[-1].weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert not torch.equal(before, model.head[-1].weight)
    model.eval()
    path = tmp_path / 'checkpoint.pt'
    torch.save({'config': cfg, 'state_dict': model.state_dict()}, path)
    state = torch.load(path, weights_only=True)
    restored = build_model(state['config']).eval()
    restored.load_state_dict(state['state_dict'])
    with torch.no_grad():
        torch.testing.assert_close(restored(x), model(x))


def test_tiny_batch_overfit():
    seed_everything(2026, threads=2)
    model = WideResidualCNN(width=16, dropout=0, dilations=[1, 2])
    x, y = torch.randn(4, 306, 250), torch.arange(4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    initial = torch.nn.functional.cross_entropy(model(x), y).item()
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
    assert loss.item() < initial * 0.2
    assert (model(x).argmax(-1) == y).all()


def test_fixed_validation_and_expansion():
    baseline = select_training_records(config('config.json'))
    small = select_training_records(config('config_wide_cnn.json'))
    expanded = select_training_records(config('config_wide_cnn_expanded.json'))
    assert small == baseline
    assert [r for r in expanded if r['split'] == 'val'] == [r for r in baseline if r['split'] == 'val']
    assert len(expanded) == 34
    assert len([r for r in expanded if r['split'] == 'train']) == 31
    assert all(r['partition'] == 'train' for r in expanded)
    cfg = config('config_wide_cnn.json')
    cfg['validation_groups'] = [['Sherlock1', '0', '12']]
    with pytest.raises(ValueError):
        select_training_records(cfg)


def test_existing_cache_compatible_and_changed_split_rejected(tmp_path):
    import json
    (tmp_path/'prepared_config.json').write_text(json.dumps(config('config.json')))
    verify_prepared_config(tmp_path, config('config_wide_cnn.json'))
    cfg = config('config_wide_cnn.json')
    cfg['validation_groups'] = [['Sherlock1', '0', '1']]
    with pytest.raises(ValueError, match='validation groups'):
        verify_prepared_config(tmp_path, cfg)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_wide_cnn_cuda():
    from decoder import configure_gpus
    seed_everything(2026, threads=2)
    model = configure_gpus(WideResidualCNN(width=32), torch.device('cuda'), min(2, torch.cuda.device_count()))
    loss = torch.nn.functional.cross_entropy(model(torch.randn(8, 306, 250, device='cuda')),
                                            torch.arange(8, device='cuda'))
    loss.backward()
    assert torch.isfinite(loss)
