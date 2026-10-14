import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from decoder import build_model, seed_everything, select_training_records, config
from research import preprocess, fit_channel_stats, add_stimulus_keys, domain_report
from study import generate


def test_temporal_spatial_overfit_and_checkpoint(tmp_path):
    seed_everything(1,threads=2)
    cfg={'model':'temporal_spatial_cnn','temporal_banks':4,'spatial_depth':2,'dropout':0.0}
    model=build_model(cfg)
    x=torch.randn(4,306,250);y=torch.arange(4)
    opt=torch.optim.AdamW(model.parameters(),lr=.003)
    initial=torch.nn.functional.cross_entropy(model(x),y).item()
    for _ in range(35):
        opt.zero_grad();loss=torch.nn.functional.cross_entropy(model(x),y);loss.backward();opt.step()
    assert loss.item()<initial*.2
    assert (model(x).argmax(-1)==y).all()
    path=tmp_path/'model.pt'
    torch.save(model.state_dict(),path)
    clone=build_model(cfg);clone.load_state_dict(torch.load(path,weights_only=True))
    model.eval();clone.eval()
    torch.testing.assert_close(model(x),clone(x))


def test_training_only_stats_and_batch_preprocessing(tmp_path):
    rng=np.random.default_rng(1)
    x=rng.normal(size=(306,750))*1e-13
    with h5py.File(tmp_path/'raw.h5','w') as f:f['data']=x
    frame=pd.DataFrame({'h5_file':['raw.h5']*2,'start_sample':[0,250],'split':['train']*2})
    stats=fit_channel_stats(frame,tmp_path)
    a=preprocess(x[:,:250],'train_channel_zscore',stats)
    b=preprocess(np.stack([x[:,:250]]*2),'train_channel_zscore',stats)
    np.testing.assert_allclose(a,b[0])
    assert np.isfinite(a).all()
    with pytest.raises(ValueError):fit_channel_stats(frame.assign(split='val'),tmp_path)


def test_sentence_keys_group_repeats_across_speakers():
    frame=pd.DataFrame({'kind':['word']*4,'sentenceidx':[0,0,1,1],
                        'segment':['Hello','world','hello','world'],'wavile':['a']*2+['b']*2})
    keys=add_stimulus_keys(frame)
    assert keys.nunique()==1


def test_domains_and_frequency_control():
    labels=np.arange(50)
    rows=pd.DataFrame({'task':['a']*25+['b']*25,'session':['1']*50})
    report=domain_report(np.eye(50),labels,rows,np.arange(1,51))
    assert report['overall']['bacc']==1
    assert report['frequency_only']['bacc']==pytest.approx(.2)
    assert report['domains']['a']['coverage']==25
    assert report['domains']['a']['bacc_present']==1


def test_config_suite_fixes_split_and_varies_training_seed(tmp_path):
    generate(tmp_path)
    ref=config(tmp_path/'reference.json');other=config(tmp_path/'seed2027.json')
    assert select_training_records(ref)==select_training_records(other)
    records=select_training_records(ref)
    assert all(r['split']=='val' for r in records if r['task']=='Sherlock2')
    assert any(r['task']=='TIMIT' and r['split']=='train' for r in records)
    assert ref['purge_repeated_sentences']


def test_sentence_key_missing_metadata_fails():
    with pytest.raises(ValueError):
        add_stimulus_keys(pd.DataFrame({'kind':['word'],'segment':['hello']}))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_temporal_spatial_cuda_backward():
    from decoder import configure_gpus
    model=configure_gpus(build_model({'model':'temporal_spatial_cnn','dropout':.3}),torch.device('cuda'),min(2,torch.cuda.device_count()))
    loss=torch.nn.functional.cross_entropy(model(torch.randn(4,306,250,device='cuda')),torch.arange(4,device='cuda'))
    loss.backward();assert torch.isfinite(loss)
