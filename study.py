"""Create controlled configs, audit development data and evaluate raw ensembles."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from decoder import (ROOT, config, save_json, training_frame, split_sample_frame, sample_fingerprint,
                     build_model, H5WindowDataset, EpochDataset, evaluate, seed_everything,
                     reject_final_test_path, verify_prepared_config)
from research import audit_cache, domain_report


def generate(directory):
    """Fixed data/split per suite; change one factor per named experiment."""
    base=config(ROOT/'config_wide_cnn_expanded.json')
    base.update(save_epochs=[3,5,8,13],training_seed=2026,normalization='window_zscore',
                sampling='natural',loss_weight_power=0.5,evaluate_training=True)
    domain=dict(base)
    # Whole Sherlock2 is held out from retraining, not used to evaluate weights
    # that previously trained on it. Sentence repeated-text windows are purged.
    domain['tasks']={'Sherlock1':10,'Sherlock2':12,'TheMoth':12,'TIMIT':4}
    domain['validation_groups']=base['validation_groups']+[
        ['Sherlock2','0',str(i)] for i in range(1,13)]+[['TIMIT','0','4']]
    domain['purge_repeated_sentences']=True
    variants={'reference':{},'small_reference':{'model':'temporal_cnn','width':48,'epochs':12,'patience':5},
              'small_balanced':{'model':'temporal_cnn','width':48,'epochs':12,'patience':5,'sampling':'class_balanced','loss_weight_power':0},
              'balanced':{'sampling':'class_balanced','loss_weight_power':0},
              'temporal_spatial':{'model':'temporal_spatial_cnn','temporal_banks':32,'spatial_depth':2},
              'train_normalization':{'normalization':'train_channel_zscore'},
              'dropout':{'dropout':0.3},'shuffled_control':{'shuffle_training_labels':True},
              'seed2027':{'training_seed':2027},'seed2028':{'training_seed':2028}}
    destination=Path(directory)
    destination.mkdir(parents=True,exist_ok=True)
    save_json(destination/'development_data.json',domain)
    for name,changes in variants.items():
        save_json(destination/f'{name}.json',{**domain,**changes})
    # Current-cache configs permit inexpensive first comparisons before adding
    # sentence data. They are not substitutes for cross-domain confirmation.
    for name,changes in variants.items():
        save_json(destination/f'legacy_{name}.json',{**base,**changes})
    save_json(destination/'suite.json',{'data_config':'development_data.json',
        'selection':'Inspect each domain and class coverage; no holdout or final-test tuning.',
        'reference':'Retrain the reference on this new split; old weights trained on Sherlock2 are invalid here.',
        'variants':list(variants)})


def local_evaluate(checkpoints,data,output,device='cpu',weights=None):
    """Mean probabilities, fixed supplied weights; never access competition holdout."""
    data=Path(data)
    reject_final_test_path(data)
    frame,mode=training_frame(data,'h5')
    train,val=split_sample_frame(frame)
    seed_everything(2026)
    probabilities=[]
    identities=[]
    vocab=None
    for path in checkpoints:
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        cfg=checkpoint['config']
        if cfg.get('shuffle_training_labels'):
            raise ValueError('Do not include shuffled controls in ensembles')
        verify_prepared_config(data,cfg)
        if vocab is None:
            vocab=checkpoint['vocabulary']
        if vocab != checkpoint['vocabulary']:
            raise ValueError('Vocabulary mismatch')
        dataset=H5WindowDataset(val,data,normalization=cfg.get('normalization','window_zscore'),
                               stats=checkpoint.get('preprocessing_stats'))
        model=build_model(cfg).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        logits,labels=evaluate(model,DataLoader(dataset,batch_size=cfg['batch_size']),torch.device(device))
        probabilities.append(torch.from_numpy(logits).softmax(-1).numpy())
        identities.append({'checkpoint':str(path),'epoch':checkpoint['epoch']})
        dataset.close()
    weights=np.ones(len(probabilities)) if weights is None else np.asarray(weights,dtype=float)
    if len(weights)!=len(probabilities) or not np.isfinite(weights).all() or np.any(weights<0) or weights.sum()<=0:
        raise ValueError('Invalid ensemble weights')
    probs=np.average(np.stack(probabilities),axis=0,weights=weights)
    report=domain_report(probs,labels,val,np.bincount(train.label,minlength=50))
    report.update(checkpoints=identities,weights=(weights/weights.sum()).tolist(),
                  sample_fingerprint=sample_fingerprint(val),warning='Development selection score, not independent test.')
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    save_json(output/'evaluation.json',report)
    np.savez(output/'predictions.npz',probabilities=probs,labels=labels)
    val.to_csv(output/'rows.csv',index=False)


def ensemble_submission(checkpoints,data,output,device='cuda'):
    """Uniform raw ensemble only after local selection; no holdout-based tuning."""
    from decoder import predict
    from types import SimpleNamespace
    import tempfile
    vocab=None
    with tempfile.TemporaryDirectory() as tmp:
        frames=[]
        for i,path in enumerate(checkpoints):
            state=torch.load(path,map_location='cpu',weights_only=True)
            if state['config'].get('shuffle_training_labels'):
                raise ValueError('Cannot submit shuffled-label controls')
            if vocab is None:vocab=state['vocabulary']
            if vocab!=state['vocabulary']:raise ValueError('Vocabulary mismatch')
            csv=Path(tmp)/f'model{i}.csv'
            predict(SimpleNamespace(checkpoint=path,data=data,csv=csv,device=device,prior_corrected=False))
            frames.append(pd.read_csv(csv))
        if not frames:raise ValueError('Need checkpoints')
        first=frames[0]
        for frame in frames[1:]:
            if list(frame.columns)!=list(first.columns) or not frame['index'].equals(first['index']):
                raise ValueError('Submission index/column mismatch')
        scores=np.mean([f.iloc[:,1:].to_numpy() for f in frames],axis=0)
        from pnpl.competition import write_submission
        Path(output).parent.mkdir(parents=True,exist_ok=True)
        write_submission(output,indices=first['index'].to_numpy(),primary_probs=scores)
        save_json(Path(output).with_suffix('.json'),{'prediction_mode':'raw_uniform_ensemble',
                  'checkpoints':list(map(str,checkpoints)),'n_models':len(frames),'rows':len(first)})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('configs');p.add_argument('--output',default='experiments')
    p=sub.add_parser('audit');p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--holdout-data',help='Optional label-blind array metadata inspection; no scoring')
    p=sub.add_parser('evaluate');p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--checkpoints',nargs='+',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--weights',nargs='+',type=float)
    p=sub.add_parser('predict-ensemble');p.add_argument('--checkpoints',nargs='+',required=True)
    p.add_argument('--data',required=True);p.add_argument('--csv',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args()
    if args.command=='configs':generate(args.output)
    elif args.command=='audit':save_json(args.output,audit_cache(args.data,args.holdout_data))
    elif args.command=='evaluate':local_evaluate(args.checkpoints,args.data,args.output,args.device,args.weights)
    else:ensemble_submission(args.checkpoints,args.data,args.csv,args.device)


if __name__=='__main__':main()
