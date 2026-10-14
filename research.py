"""Development-only audits, domain reports and label-blind preprocessing."""
import hashlib
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def preprocess(x, mode='window_zscore', stats=None):
    from decoder import normalize_windows
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite input')
    if mode == 'window_zscore':
        return normalize_windows(x)
    if mode != 'train_channel_zscore' or stats is None:
        raise ValueError('Unknown preprocessing or missing training statistics')
    mean = np.asarray(stats['mean'], dtype=np.float32)[:, None]
    std = np.asarray(stats['std'], dtype=np.float32)[:, None]
    return np.clip((x - mean) / np.maximum(std, 1e-30), -10, 10).astype(np.float32)


def fit_channel_stats(frame, data_dir, max_windows=512):
    """Deterministic, evenly spaced training-only sample; no validation/holdout fitting."""
    if not frame['split'].eq('train').all() or frame.empty:
        raise ValueError('Statistics must be fitted on training rows only')
    indices = np.linspace(0, len(frame)-1, min(max_windows, len(frame)), dtype=int)
    sums, squares, count = np.zeros(306), np.zeros(306), 0
    from progress import progress_iter
    for row in progress_iter(frame.iloc[indices].itertuples(),len(indices),'Fit training normalization',key='stats'):
        path = Path(row.h5_file)
        path = path if path.is_absolute() else Path(data_dir) / path
        from decoder import reject_final_test_path
        reject_final_test_path(path)
        with h5py.File(path, 'r') as f:
            x = np.asarray(f['data'][:, row.start_sample:row.start_sample+250], dtype=np.float64)
        if x.shape != (306, 250) or not np.isfinite(x).all():
            raise ValueError('Invalid statistics window')
        sums += x.sum(-1)
        squares += (x*x).sum(-1)
        count += x.shape[-1]
    mean = sums/count
    std = np.sqrt(np.maximum(squares/count-mean*mean, 1e-60))
    return {'mean': mean.tolist(), 'std': std.tolist(), 'fit_windows': len(indices), 'source': 'training_only'}


def add_stimulus_keys(events):
    """Hash normalized sentence text to group repeats across speakers/recordings."""
    from decoder import word_key
    words = events.loc[events.kind.eq('word')].copy()
    if 'sentenceidx' not in words:
        raise ValueError('Sentence control requires sentenceidx metadata')
    texts = words.groupby('sentenceidx', dropna=False)['segment'].apply(
        lambda values: ' '.join(word_key(v) for v in values))
    keys = texts.map(lambda text: hashlib.sha256(text.encode()).hexdigest())
    return events['sentenceidx'].map(keys)


def domain_report(logits, labels, rows, train_counts):
    from decoder import balanced_metrics
    labels = np.asarray(labels)
    if len(rows) != len(labels):
        raise ValueError('Row/prediction mismatch')
    constant = np.broadcast_to(np.asarray(train_counts, dtype=float), (len(labels), 50))
    result = {'overall': balanced_metrics(logits, labels),
              'frequency_only': balanced_metrics(constant, labels), 'domains': {}, 'sessions': {}}
    for column, target in [('task', 'domains'), ('session_key', 'sessions')]:
        values = rows[column] if column in rows else rows.task.astype(str)+'/'+rows.session.astype(str)
        for group in sorted(values.unique()):
            mask = np.asarray(values.eq(group))
            result[target][str(group)] = {'n': int(mask.sum()), 'class_counts': np.bincount(labels[mask], minlength=50).tolist(),
                                        **balanced_metrics(logits[mask], labels[mask])}
    result['distinct_top1'] = int(np.unique(np.argmax(logits, axis=1)).size)
    result['mean_logit_std_across_trials'] = float(np.std(logits, axis=0).mean())
    return result


def audit_cache(data_dir, holdout_dir=None):
    """Read only development cache; report raw channel/metadata consistency, not certify it."""
    from decoder import training_frame, split_sample_frame, reject_final_test_path
    data_dir = Path(data_dir)
    reject_final_test_path(data_dir)
    frame, _ = training_frame(data_dir, 'h5')
    split_sample_frame(frame)
    report = {'files': [], 'limitations': ['Channel ordering cannot be certified if channel names are absent.',
              'Filtering and event onset provenance require comparison with official holdout metadata.']}
    from progress import progress_iter
    groups=frame.groupby('h5_file')
    for name, rows in progress_iter(groups,len(groups),'Audit recordings',key='audit'):
        path = Path(name)
        path = path if path.is_absolute() else data_dir/path
        reject_final_test_path(path)
        with h5py.File(path, 'r') as f:
            row = rows.iloc[0]
            x = np.asarray(f['data'][:, int(row.start_sample):int(row.start_sample)+250])
            attrs = {k: str(v) for k,v in f.attrs.items()}
            report['files'].append({'file': str(path), 'shape': list(f['data'].shape), 'dtype': str(f['data'].dtype),
                'attrs': attrs, 'finite_sample': bool(np.isfinite(x).all()),
                'channel_std_min': float(x.std(-1).min()), 'channel_std_median': float(np.median(x.std(-1))),
                'start_samples_valid': bool((rows.start_sample >= 0).all() and (rows.start_sample+250 <= f['data'].shape[1]).all())})
    if 'stimulus_key' in frame:
        a = set(frame.loc[frame.split.eq('train') & frame.task.isin(['TIMIT','MOCHATIMIT']), 'stimulus_key'])
        b = set(frame.loc[frame.split.eq('val') & frame.task.isin(['TIMIT','MOCHATIMIT']), 'stimulus_key'])
        report['sentence_train_val_overlap'] = len(a & b)
    names=[f['attrs'].get('channel_names') for f in report['files']]
    report['channel_order_consistent_across_training_files'] = bool(names and all(n is not None for n in names) and len(set(names))==1)
    report['metadata_frequency_consistent'] = len({f['attrs'].get('sample_frequency') for f in report['files']})==1
    if holdout_dir is not None:
        reject_final_test_path(Path(holdout_dir))
        report['holdout_metadata'] = []
        for path in sorted(Path(holdout_dir).rglob('*.npz')):
            # Label-blind structural audit only. Never inspect labels/targets.
            with np.load(path,allow_pickle=False) as archive:
                safe={}
                for key in archive.files:
                    if key.lower() in {'meg','data','x','indices','index','sensor_xyz','ch_names','channel_names','sfreq'}:
                        try:
                            array=archive[key]
                            safe[key]={'shape':list(array.shape),'dtype':str(array.dtype)}
                        except ValueError:
                            safe[key]={'unreadable_without_pickle':True}
                report['holdout_metadata'].append({'file':str(path),'safe_arrays':safe})
    return report
