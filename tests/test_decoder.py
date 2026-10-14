import numpy as np
import pytest
import torch

from decoder import TemporalCNN, balanced_metrics, grouped_split, normalize_windows, select_training_records, config, word_key


def test_shapes_and_gradients():
    torch.set_num_threads(2)
    model = TemporalCNN(width=32)
    y = model(torch.randn(3, 306, 250))
    assert y.shape == (3, 50)
    y.sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_tesla_normalization_and_constant_channels():
    x = np.random.default_rng(1).normal(size=(2, 306, 250)).astype('float32')*1e-13
    x[:, 0] = 0
    a = normalize_windows(x)
    assert np.isfinite(a).all()
    np.testing.assert_allclose(a[:, 1:].std(-1), 1, atol=1e-5)
    assert not a[:, 0].any()
    np.testing.assert_allclose(a, normalize_windows(x*1e6), atol=2e-6)


def test_metric_balances_classes_not_trials():
    labels = np.array([0]*9+[1])
    scores = np.zeros((10, 50))
    scores[:, 0] = 1
    m = balanced_metrics(scores, labels, 1)
    assert m['bacc'] == pytest.approx(1/50)
    assert m['bacc_present'] == pytest.approx(.5)
    assert m['coverage'] == 2
    scores[-1, 1] = 2
    assert balanced_metrics(scores, labels, 1)['bacc_present'] == 1


def test_top10_chance_and_perfect():
    labels = np.arange(50)
    assert balanced_metrics(np.zeros((50, 50)), labels)['bacc'] == pytest.approx(.2)
    assert balanced_metrics(np.eye(50), labels)['bacc'] == 1
    with pytest.raises(ValueError):
        balanced_metrics(np.full((50, 50), np.nan), labels)


def test_session_split_deterministic_and_disjoint():
    records = [dict(task='Sherlock1', subject='0', session=str(i), corpus='sherlock', run=str(j))
               for i in range(10) for j in range(2)]
    result = grouped_split(records, .2, 2026)
    assert result == grouped_split(records, .2, 2026)
    groups = {s: {r['session'] for r in result if r['split'] == s} for s in ['train', 'val']}
    assert not groups['train'] & groups['val']
    assert len(groups['val']) == 2


def test_official_training_only():
    records = select_training_records(config('config.json'))
    assert all(r['partition'] == 'train' and r['subject'] == '0' for r in records)
    assert len(records) == 16
    assert not any(r['task'] == 'Sherlock1' and r['session'] in ['11', '12'] for r in records)


def test_vocabulary_typography():
    assert word_key(' It’s ') == word_key("it's")
    assert word_key('its') != word_key("it's")
