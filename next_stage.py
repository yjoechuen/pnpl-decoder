"""Staged spatial-model experiments and paired group-bootstrap analysis."""
from pathlib import Path
import numpy as np
import pandas as pd

from decoder import config,save_json,balanced_metrics


def export_geometry(raw_fif,training_h5,output):
    """Use real recording metadata, align by name, require mag/grad channels.

    Optional mne dependency. Geometry is exported once on CPU; no signal data
    from FIF are loaded. Obtain a matching allowed recording through official data.
    """
    import mne
    import h5py
    with h5py.File(training_h5,'r') as f:
        names=f.attrs['channel_names']
        if isinstance(names,bytes):names=names.decode()
        expected=[n.strip() for n in names.split(',')]
    info=mne.io.read_info(raw_fif,verbose=False)
    lookup={ch['ch_name']:ch for ch in info['chs']}
    if len(expected)!=306 or any(n not in lookup for n in expected):raise ValueError('Channel mapping failed')
    for name in expected:
        if mne.channel_type(info,info['ch_names'].index(name)) not in ['mag','grad']:
            raise ValueError('Geometry reference must contain mag/grad channels')
    positions=[lookup[n]['loc'][:3].tolist() for n in expected]
    payload={'channel_names':expected,'positions':positions,
             'source_fif':str(raw_fif),'reference_h5':str(training_h5)}
    from models import load_sensor_geometry
    # Validate before publishing the file.
    import tempfile,json
    with tempfile.NamedTemporaryFile(mode='w',suffix='.json') as f:
        json.dump(payload,f);f.flush();load_sensor_geometry(f.name,expected)
    save_json(output,payload)


class NextStage:
    """Fresh-process staged Kaggle training. Completed runs are never overwritten."""
    def __init__(self,repo='/kaggle/working/pnpl-decoder',root='/kaggle/working/spatial-next',devices=1):
        self.repo,self.root=Path(repo),Path(root)
        self.devices=devices
        generate(self.repo)

    def experiment(self,name,expanded=False):
        from kaggle_runner import KaggleExperiment
        return KaggleExperiment(repo=self.repo,config=self.repo/'experiments/spatial_next'/f'{name}.json',
            data='/kaggle/working/pnpl-next-expanded' if expanded else '/kaggle/working/pnpl-development',
            output=self.root/name)

    def train(self,names):
        from decoder import config,sample_fingerprint,training_frame,verify_prepared_config
        for name in names:
            expanded=name in ['expanded','expanded_corpus_balanced','multiscale','geometry']
            run=self.experiment(name,expanded)
            cfg=config(run.config)
            if (run.output/'run_config.json').exists() and config(run.output/'run_config.json')['config']!=cfg:
                raise ValueError(f'{name}: saved run config differs; use a new output root')
            marker=run.output/'completed.json'
            required=['best.pt','history.json','validation_report.json','domain_report.json',
                      'validation_predictions.npz','validation_rows.csv']
            if cfg.get('evaluate_training',False):required.append('training_evaluation.json')
            if marker.exists() and all((run.output/p).exists() for p in required):
                done=config(marker)
                verify_prepared_config(run.data,cfg)
                frame,_=training_frame(run.data,'h5')
                if done['config']!=cfg or done['sample_fingerprint']!=sample_fingerprint(frame):
                    raise ValueError(f'{name}: completed run config/data differs')
                print(f'Skip completed: {name}');continue
            resume=any((run.output/p).exists() for p in ['latest.pt','best.pt'])
            if name=='geometry' and not resume and not Path(cfg['geometry_file']).is_file():
                raise FileNotFoundError('Export verified sensor_geometry.json before geometry run')
            run.train(devices=self.devices,data_mode='h5',resume=resume)

    def prepare_expansion(self,preflight=False):
        run=self.experiment('data_expanded',True)
        run.prepare(raw='/kaggle/working/pnpl-data/raw',preflight=preflight,disk_budget_gib=24)

    def summary(self):
        table=[]
        for path in self.root.glob('*/domain_report.json'):
            report=config(path)
            row={'run':path.parent.name,'overall_raw':report['overall']['bacc']}
            row.update({name:values['bacc_present'] for name,values in report['domains'].items()})
            table.append(row)
        return pd.DataFrame(table).sort_values('run') if table else pd.DataFrame()

    def ensemble(self,names,output=None):
        import sys
        from kaggle_runner import notebook_process
        if not names:raise ValueError('At least one checkpoint required')
        configs=[config(self.repo/'experiments/spatial_next'/f'{n}.json') for n in names]
        if any(c['tasks']!=configs[0]['tasks'] for c in configs):
            raise ValueError('Use checkpoints trained on the same data configuration for local ensemble evaluation')
        expanded=names[0] in ['expanded','multiscale','geometry','expanded_corpus_balanced']
        data='/kaggle/working/pnpl-next-expanded' if expanded else '/kaggle/working/pnpl-development'
        notebook_process([sys.executable,str(self.repo/'study.py'),'evaluate','--checkpoints',
            *[str(self.root/n/'best.pt') for n in names],'--data',data,
            '--output',str(output or self.root/'ensemble'),'--device','cuda'])


def generate(repo,overwrite=False):
    repo=Path(repo)
    base=config(repo/'experiments/temporal_spatial.json')
    base.update(num_workers=2,prefetch_factor=2,amp=True)
    expanded={**base,'tasks':{**base['tasks'],'TheMoth':18,'TIMIT':10}}
    variants={'anchor':base,'seed2027':{**base,'training_seed':2027},
        'seed2028':{**base,'training_seed':2028},
        'balanced':{**base,'sampling':'class_balanced','loss_weight_power':0},
        'dropout':{**base,'dropout':.3},
        'normalization':{**base,'normalization':'train_channel_zscore'},
        'expanded':expanded,'multiscale':{**expanded,'model':'multiscale_spatial_cnn'},
        'geometry':{**expanded,'model':'geometry_cnn','geometry_file':'/kaggle/working/sensor_geometry.json'},
        'expanded_corpus_balanced':{**expanded,'sampling':'corpus_balanced'}}
    for name,cfg in {**variants,'data_expanded':expanded}.items():
        path=repo/'experiments/spatial_next'/f'{name}.json'
        if overwrite or not path.exists():save_json(path,cfg)
        else:
            if name in variants:variants[name]=config(path)
    return variants


def compare(run_a,run_b,output,repeats=1000,seed=2026):
    """Paired cluster bootstrap, shared rows required. No holdout access."""
    from decoder import sample_fingerprint
    a,b=Path(run_a),Path(run_b)
    rows=pd.read_csv(a/'validation_rows.csv',low_memory=False)
    other=pd.read_csv(b/'validation_rows.csv',low_memory=False)
    if sample_fingerprint(rows)!=sample_fingerprint(other):raise ValueError('Validation rows differ')
    with np.load(a/'validation_predictions.npz') as f:la,ya=f['logits'],f['labels']
    with np.load(b/'validation_predictions.npz') as f:lb,yb=f['logits'],f['labels']
    if not np.array_equal(ya,yb):raise ValueError('Labels differ')
    result={'source':'development only','difference_direction':'B minus A','domains':{}}
    rng=np.random.default_rng(seed)
    for task,group in rows.groupby('task'):
        idx=group.index.to_numpy()
        groups=group.session.astype(str)
        if task in ['TIMIT','MOCHATIMIT']:
            if 'stimulus_key' not in group or group.stimulus_key.isna().any():raise ValueError('Sentence keys required')
            groups=group.stimulus_key
        clusters=[group.index[groups.eq(key)].to_numpy() for key in groups.unique()]
        observed=balanced_metrics(lb[idx],yb[idx])['bacc_present']-balanced_metrics(la[idx],ya[idx])['bacc_present']
        differences=[]
        for _ in range(repeats):
            selected=np.concatenate([clusters[j] for j in rng.integers(0,len(clusters),len(clusters))])
            differences.append(balanced_metrics(lb[selected],yb[selected])['bacc_present']-balanced_metrics(la[selected],ya[selected])['bacc_present'])
        result['domains'][task]={'difference':observed,'clusters':len(clusters),
            'coverage':int(len(np.unique(ya[idx]))),'interval95':np.quantile(differences,[.025,.975]).tolist(),
            'warning':'Few session clusters yield unreliable intervals; class support may vary in resamples.'}
    save_json(output,result)
    return result
