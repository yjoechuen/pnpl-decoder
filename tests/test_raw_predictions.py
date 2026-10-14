from unittest.mock import patch

import numpy as np

from decoder import build_parser, prediction_logits
from kaggle_runner import KaggleExperiment


def test_raw_is_default_and_preserves_rankings():
    logits = np.random.default_rng(1).normal(size=(5, 50))
    checkpoint = {'prior_alpha': 0.5, 'train_counts': list(range(1, 51))}
    np.testing.assert_array_equal(prediction_logits(logits, checkpoint), logits)
    expected = logits - .5 * np.log(np.arange(1, 51) / sum(range(1, 51)))
    np.testing.assert_allclose(prediction_logits(logits, checkpoint, .5), expected)
    assert not build_parser().parse_args(['predict']).prior_corrected
    assert build_parser().parse_args(['predict', '--prior-corrected']).prior_corrected


def test_runner_correction_requires_opt_in():
    run = KaggleExperiment()
    with patch('kaggle_runner.subprocess.run') as launch:
        run.predict()
        assert '--prior-corrected' not in launch.call_args.args[0]
        run.predict(prior_corrected=True)
        assert '--prior-corrected' in launch.call_args.args[0]
