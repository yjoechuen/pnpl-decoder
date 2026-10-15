from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import torch

from decoder import config, save_json, train
from study import local_evaluate


def test_train_balanced_stats_epochs_and_ensemble(tmp_path):
    data=tmp_path/'data';data.mkdir()
    cfg=config('config.json')
    cfg.update(width=16,epochs=1,batch_size=25,threads=2,sampling='class_balanced',loss_weight_power=0,
               normalization='train_channel_zscore',save_epochs=[1],evaluate_training=True,training_seed=11)
    save_json(data/'prepared_config.json',cfg)
    save_json(data/'vocabulary.json',[str(i) for i in range(50)])
    rng=np.random.default_rng(1)
    with h5py.File(data/'raw.h5','w') as f:
        f['data']=rng.normal(size=(306,500)).astype('float32')*1e-13
        f.attrs['sample_frequency']=250
    rows=[]
    for split,session in [('train','1'),('val','2')]:
        for label in range(50):
            rows.append(dict(h5_file='raw.h5',start_sample=label,file='unused',epoch=label,label=label,
                             task='synthetic',subject='0',session=session,partition='train',split=split))
    pd.DataFrame(rows).to_csv(data/'samples_h5.csv',index=False)
    path=tmp_path/'config.json';save_json(path,cfg)
    output=tmp_path/'run'
    train(SimpleNamespace(config=path,data=data,output=output,epochs=None,resume=False,device='cpu',devices=1,data_mode='h5'))
    assert (output/'epoch_001.pt').is_file()
    state=torch.load(output/'best.pt',weights_only=True)
    assert state['preprocessing_stats']['source']=='training_only'
    assert (output/'domain_report.json').is_file()
    assert (output/'training_evaluation.json').is_file()
    assert (output/'completed.json').is_file()
    local_evaluate([output/'best.pt',output/'epoch_001.pt'],data,tmp_path/'ensemble')
    assert (tmp_path/'ensemble/evaluation.json').is_file()


def test_geometry_resume_without_external_metadata(tmp_path):
    data=tmp_path/'data';data.mkdir()
    names=[f'MEG{i}' for i in range(306)]
    geometry=tmp_path/'geometry.json'
    rng=np.random.default_rng(9)
    save_json(geometry,{'channel_names':names,'positions':rng.normal(size=(306,3)).tolist()})
    cfg=config('config.json')
    cfg.update(model='geometry_cnn',geometry_width=8,geometry_file=str(geometry),epochs=1,
               batch_size=25,threads=2,num_workers=2,amp=True)
    save_json(data/'prepared_config.json',cfg)
    save_json(data/'vocabulary.json',[str(i) for i in range(50)])
    with h5py.File(data/'raw.h5','w') as f:
        f['data']=rng.normal(size=(306,500)).astype('float32')*1e-13
        f.attrs['sample_frequency']=250
        f.attrs['channel_names']=','.join(names)
    rows=[dict(h5_file='raw.h5',start_sample=label,label=label,task='synthetic',
               subject='0',session=session,partition='train',split=split)
          for split,session in [('train','1'),('val','2')] for label in range(50)]
    pd.DataFrame(rows).to_csv(data/'samples_h5.csv',index=False)
    path=tmp_path/'config.json';save_json(path,cfg)
    output=tmp_path/'run'
    args=SimpleNamespace(config=path,data=data,output=output,epochs=None,resume=False,
                         device='cpu',devices=1,data_mode='h5')
    train(args)
    before=torch.load(output/'best.pt',weights_only=True)['state_dict']['spatial_prior'].clone()
    geometry.unlink()
    (output/'completed.json').unlink()
    args.resume=True
    train(args)
    after=torch.load(output/'best.pt',weights_only=True)['state_dict']['spatial_prior']
    torch.testing.assert_close(before,after)
    assert (output/'completed.json').exists()
