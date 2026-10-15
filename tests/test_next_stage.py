import numpy as np
import pandas as pd
import pytest
import torch

from decoder import build_model,config,make_loader,seed_everything,select_training_records
from next_stage import generate,compare


@pytest.mark.parametrize('name',['multiscale_spatial_cnn','geometry_cnn'])
def test_candidate_backward(name):
    seed_everything(1,threads=2)
    model=build_model({'model':name,'dropout':.2})
    if name=='geometry_cnn':model.initialize_geometry(np.random.default_rng(1).normal(size=(306,3)))
    logits=model(torch.randn(2,306,250))
    assert logits.shape==(2,50)
    logits.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    clone=build_model({'model':name,'dropout':.2});clone.load_state_dict(model.state_dict())
    model.eval();clone.eval()
    x=torch.randn(2,306,250)
    with torch.no_grad():torch.testing.assert_close(model(x),clone(x))


def test_generate_stage_configs(tmp_path):
    import shutil
    (tmp_path/'experiments').mkdir()
    shutil.copy('experiments/temporal_spatial.json',tmp_path/'experiments/temporal_spatial.json')
    variants=generate(tmp_path)
    assert select_training_records(variants['anchor'])==select_training_records(variants['seed2027'])
    assert [r for r in select_training_records(variants['expanded']) if r['split']=='val']==[
        r for r in select_training_records(variants['anchor']) if r['split']=='val']
    assert variants['balanced']['loss_weight_power']==0
    from decoder import save_json
    customized={**variants['anchor'],'amp':False}
    save_json(tmp_path/'experiments/spatial_next/anchor.json',customized)
    assert generate(tmp_path)['anchor']['amp'] is False


def test_paired_bootstrap(tmp_path):
    rows=pd.DataFrame({'task':['TIMIT']*50,'session':['4']*50,'label':range(50),'stimulus_key':[str(i//5) for i in range(50)]})
    for name,logits in [('a',np.zeros((50,50))),('b',np.eye(50))]:
        path=tmp_path/name;path.mkdir()
        rows.to_csv(path/'validation_rows.csv',index=False)
        np.savez(path/'validation_predictions.npz',logits=logits,labels=np.arange(50))
    report=compare(tmp_path/'a',tmp_path/'b',tmp_path/'comparison.json',repeats=20)
    assert report['domains']['TIMIT']['difference']==pytest.approx(.8)


def test_loader_worker_seed_and_options():
    from torch.utils.data import TensorDataset
    loader=make_loader(TensorDataset(torch.randn(4,2),torch.arange(4)),{'batch_size':2,'num_workers':0},torch.device('cpu'))
    assert len(list(loader))==2


@pytest.mark.parametrize('weighted',[False,True])
def test_persistent_loader_resume_sample_order(weighted):
    from torch.utils.data import TensorDataset,WeightedRandomSampler
    dataset=TensorDataset(torch.arange(24))
    cfg={'batch_size':4,'num_workers':2}
    generator=torch.Generator().manual_seed(42)
    def loader(g):
        sampler=WeightedRandomSampler(torch.ones(24),24,generator=g) if weighted else None
        return make_loader(dataset,cfg,torch.device('cpu'),shuffle=not weighted,sampler=sampler,generator=g)
    uninterrupted=loader(generator)
    list(uninterrupted)
    saved=generator.get_state()
    expected=torch.cat([x[0] for x in uninterrupted])
    restored=torch.Generator();restored.set_state(saved)
    resumed=loader(restored)
    actual=torch.cat([x[0] for x in resumed])
    assert torch.equal(actual,expected)


def test_geometry_channel_integrity_and_rigid_invariance(tmp_path):
    import h5py
    from decoder import save_json
    from models import load_sensor_geometry,dataset_channel_names,GeometryCNN
    names=[f'MEG{i}' for i in range(306)]
    xyz=np.random.default_rng(42).normal(size=(306,3)).astype('float32')
    path=tmp_path/'geometry.json'
    save_json(path,{'channel_names':names,'positions':xyz.tolist()})
    with h5py.File(tmp_path/'raw.h5','w') as f:f.attrs['channel_names']=','.join(names)
    frame=pd.DataFrame({'h5_file':['raw.h5']})
    assert dataset_channel_names(frame,tmp_path)==names
    np.testing.assert_allclose(load_sensor_geometry(path,names),xyz)
    with pytest.raises(ValueError):load_sensor_geometry(path,names[::-1])
    save_json(path,{'channel_names':names,'positions':np.zeros((306,3)).tolist()})
    with pytest.raises(ValueError,match='Degenerate'):load_sensor_geometry(path,names)
    with h5py.File(tmp_path/'other.h5','w') as f:f.attrs['channel_names']=','.join(names[::-1])
    with pytest.raises(ValueError,match='order differs'):
        dataset_channel_names(pd.DataFrame({'h5_file':['raw.h5','other.h5']}),tmp_path)
    a,b=GeometryCNN(),GeometryCNN()
    a.initialize_geometry(xyz)
    rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]],dtype='float32')
    b.initialize_geometry(xyz@rotation+np.array([2,-3,4],dtype='float32'))
    torch.testing.assert_close(a.spatial_prior,b.spatial_prior,atol=1e-4,rtol=1e-4)


def test_multiscale_config_is_honored():
    model=build_model({'model':'multiscale_spatial_cnn','dropout':.2,'spatial_depth':3,'pooling_bins':5})
    assert model.backbone.bins==5
    assert model.backbone.spatial[0].out_channels==96
    with pytest.raises(ValueError,match='32 temporal banks'):
        build_model({'model':'multiscale_spatial_cnn','dropout':.2,'temporal_banks':16})


def test_stage_partial_reports_resume_and_config_guard(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from next_stage import NextStage
    from decoder import save_json
    cfg={'amp':True}
    path=tmp_path/'cfg.json';save_json(path,cfg)
    output=tmp_path/'run';output.mkdir()
    save_json(output/'validation_report.json',{})
    (output/'best.pt').touch()
    calls=[]
    run=SimpleNamespace(output=output,config=path,train=lambda **kw:calls.append(kw))
    stage=object.__new__(NextStage);stage.devices=1
    monkeypatch.setattr(stage,'experiment',lambda *args:run)
    stage.train(['anchor'])
    assert calls[0]['resume'] is True
    save_json(output/'run_config.json',{'config':{'amp':False}})
    with pytest.raises(ValueError,match='saved run config differs'):stage.train(['anchor'])


def test_prepared_cache_allows_training_only_changes(tmp_path):
    from decoder import save_json,verify_prepared_config
    base=config('experiments/temporal_spatial.json')
    save_json(tmp_path/'prepared_config.json',base)
    verify_prepared_config(tmp_path,{**base,'amp':True,'dropout':.9,'num_workers':2})
    with pytest.raises(ValueError):verify_prepared_config(tmp_path,{**base,'purge_repeated_sentences':False})


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_amp_backward():
    model=build_model({'model':'multiscale_spatial_cnn','dropout':.2}).cuda()
    opt=torch.optim.AdamW(model.parameters())
    scaler=torch.amp.GradScaler('cuda')
    with torch.autocast('cuda',dtype=torch.float16):
        loss=torch.nn.functional.cross_entropy(model(torch.randn(2,306,250,device='cuda')),torch.tensor([0,1],device='cuda'))
    scaler.scale(loss).backward();scaler.step(opt);scaler.update()
    assert torch.isfinite(loss)
