import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path
import cv2
import numpy as np
import pytest
import torch
from scipy.io import savemat
from pcd_diffusion.prior_token_model import PriorTokenCodec, PriorTokenDiffusion
from pcd_diffusion.physics import measure
from pcd_diffusion.fixed_mask_data import (load_mask,FixedMaskVideoDataset,load_six_gray,
                                        transform_frames,frame_metrics,ASSETS)

torch.set_num_threads(2)


def test_prior_token_encoder_prior_and_decoder_shapes():
    codec=PriorTokenCodec(frames=8,dim=8,encoder_blocks=1)
    x=torch.rand(1,8,17,19)
    prior=codec.encode(x,x)
    assert prior.shape==(1,16,256)
    condition=codec.encode(x)
    assert condition.shape==prior.shape
    out,state=codec.decode(x,prior)
    assert out.shape==x.shape
    again,_=codec.decode(out,condition,state)
    assert torch.isfinite(again).all()


@pytest.mark.parametrize('checkpointing',[False,True])
def test_prior_token_full_training_gradient_and_target_free_inference(checkpointing):
    torch.manual_seed(15)
    net=PriorTokenDiffusion(frames=8,width=8,blocks=1,steps=2,reconstruction_dim=8,
                         encoder_blocks=1,motion_backend='difference',gradient_checkpointing=checkpointing)
    x=torch.rand(1,1,8,16,16); mask=(torch.rand_like(x)>0.5).float(); mask[:,:,0]=1
    y=measure(x,mask)
    out,loss=net(y,mask,target=x); loss['total'].backward()
    assert out.shape==x.shape and torch.isfinite(loss['total'])
    
    
    for module in [net.codec.gt_le,net.codec.condition_le,net.codec.reconstruction,
                   net.denoiser,net.token_correction,net.pixel_correction]:
        grads=[p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(g.abs().max()>0 for g in grads)
    net.eval()
    with torch.no_grad():
        torch.manual_seed(19); a=net(y,mask,stochastic=False)
        torch.manual_seed(19); b=net(y,mask,stochastic=False)
    torch.testing.assert_close(a,b)


def test_fixed_mask_and_opencv_luma(tmp_path):
    mask=np.zeros((16,16,8),dtype=np.uint8); mask[:,:,::2]=1
    savemat(tmp_path/'mask.mat',{'mask':mask})
    root=tmp_path/'frames'; seq=root/'one'; seq.mkdir(parents=True)
    images=[]
    for i in range(9):
        im=np.full((19,21,3),(i*10,100,220),dtype=np.uint8)
        cv2.imwrite(str(seq/f'{i:03d}.png'),im); images.append(im)
    split=tmp_path/'train.txt'; split.write_text('one\n')
    ds=FixedMaskVideoDataset(root,tmp_path/'mask.mat',split,augmentation='none')
    assert len(ds)==2
    item=ds[0]
    expected=np.stack([cv2.cvtColor(cv2.resize(im,(16,16)),cv2.COLOR_BGR2YCrCb)[:,:,0] for im in images[:8]]).astype(np.float32)/255
    np.testing.assert_array_equal(item['x'][0].numpy(),expected)
    np.testing.assert_array_equal(item['mask'][0].numpy(),mask.transpose(2,0,1))
    torch.testing.assert_close(measure(item['x'][None],item['mask'][None])[0],item['y'])
    ds.augmentation='random_scale_crop_flip'
    for _ in range(4):
        sample=ds[0]
        torch.testing.assert_close(sample['mask'],item['mask'])
        torch.testing.assert_close(measure(sample['x'][None],sample['mask'][None])[0],sample['y'])


def test_six_gray_ignores_embedded_meas_and_mask(tmp_path):
    maskdir=tmp_path/'test_datasets/mask'; maskdir.mkdir(parents=True)
    sim=tmp_path/'test_datasets/simulation'; sim.mkdir()
    mask=np.ones((16,16,8),dtype=np.float32)
    savemat(maskdir/'random_mask.mat',{'mask':mask})
    for rel in ASSETS:
        if rel.startswith('simulation/'):
            savemat(tmp_path/'test_datasets'/rel,{'orig':np.full((16,16,16),128,dtype=np.uint8),
                    'meas':np.zeros((16,16,5)),'mask':np.zeros_like(mask)})
    seqs,_=load_six_gray(sim,maskdir/'random_mask.mat')
    assert len(seqs)==6
    for seq in seqs:
        assert seq['x'].shape==(2,8,16,16)
        np.testing.assert_allclose(seq['y'],128/255*8)
        assert seq['mask'].min()==1


def test_grayscale_metrics_do_not_clip():
    truth=np.ones((16,16),dtype=np.float64)
    pred=np.full((16,16),1.2,dtype=np.float32)
    result=frame_metrics(pred,truth)
    assert 13.9<result['psnr']<14.1
    same=frame_metrics(pred,pred)
    assert same['psnr']==100 and abs(same['ssim']-1)<1e-12


def test_training_fixed_mask_manifest_and_resume_guard(tmp_path):
    import json
    from pcd_diffusion.engine import train
    root=tmp_path/'frames'; folder=root/'seq'; folder.mkdir(parents=True)
    for i in range(3): cv2.imwrite(str(folder/f'{i}.png'),np.full((16,16,3),100+i,np.uint8))
    savemat(tmp_path/'mask.mat',{'mask':np.ones((16,16,2),np.float32)})
    cfg={'model':{'width':4,'blocks':1,'steps':2},'train':{'epochs':1,'threads':2,'amp':False},
         'data':{'type':'fixed_mask_video','mask_path':str(tmp_path/'mask.mat'),'augmentation':'none'}}
    path=tmp_path/'config.json'; path.write_text(json.dumps(cfg))
    ck=train(path,root,tmp_path/'run','cpu')
    saved=torch.load(ck,weights_only=True)
    assert saved['data_manifest']['sample_count']==2
    assert (tmp_path/'run/data_manifest.json').exists()
    train(path,root,tmp_path/'run','cpu',resume=ck,epochs=2)
    savemat(tmp_path/'mask.mat',{'mask':np.full((16,16,2),0.5,np.float32)})
    with pytest.raises(ValueError,match='Resume data protocol differs'):
        train(path,root,tmp_path/'run','cpu',resume=ck,epochs=3)


@pytest.mark.skipif(not os.environ.get('RECONSTRUCTION_REFERENCE_ROOT'),reason='External upstream parity test')
def test_reference_encoder_and_reconstructor_numerical_parity():
    root=Path(os.environ['RECONSTRUCTION_REFERENCE_ROOT'])/'train_code_syn/architecture'
    package=types.ModuleType('_pcd_reconstruction_reference'); package.__path__=[str(root)]
    sys.modules[package.__name__]=package
    le=importlib.import_module('_pcd_reconstruction_reference.latent_encoder_arch')
    arch=importlib.import_module('_pcd_reconstruction_reference.LADE_DUN_arch')
    codec=PriorTokenCodec(frames=8,dim=8,encoder_blocks=1).eval()
    original=le.latent_encoder_gelu_mobile(in_chans=16,block_num=1).eval()
    original.load_state_dict(codec.gt_le.state_dict(),strict=True)
    x=torch.rand(1,8,16,16); gt=torch.rand_like(x)
    with torch.no_grad():
        prior=codec.encode(x,gt)
        torch.testing.assert_close(prior,original(x,gt),atol=0,rtol=0)
        decoder=arch.DUN_Denoiser(codec.reconstruction.opt,use_prior=True).eval()
        decoder.load_state_dict(codec.reconstruction.state_dict(),strict=True)
        p2=codec.down_1_channel(codec.down_1_patch(prior.transpose(1,2)).transpose(1,2))
        p3=codec.down_2_channel(codec.down_2_patch(p2.transpose(1,2)).transpose(1,2))
        expected=decoder(x,prior=[prior,p2,p3])[0]
        actual=codec.decode(x,prior)[0]
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)


@pytest.mark.skipif(not os.environ.get('VIDEO_REFERENCE_ROOT'),reason='External dataset and metric parity test')
def test_real_assets_and_upstream_metric_parity():
    root=Path(os.environ['VIDEO_REFERENCE_ROOT'])
    seqs,_=load_six_gray(root/'test_datasets/simulation',root/'test_datasets/mask/random_mask.mat')
    assert [len(s['x']) for s in seqs]==[4,4,1,4,1,6]
    assert sum(s['x'].shape[0]*s['x'].shape[1] for s in seqs)==160
    spec=importlib.util.spec_from_file_location('_reference_metrics',root/'cacti/utils/metrics.py')
    metrics=importlib.util.module_from_spec(spec); spec.loader.exec_module(metrics)
    rng=np.random.default_rng(19)
    for seq in seqs:
        p=seq['x'][0,0].astype(np.float32)+rng.normal(0,0.1,(256,256)).astype(np.float32)
        gt=seq['x'][0,0]
        result=frame_metrics(p,gt)
        assert abs(result['psnr']-metrics.compare_psnr(gt*255,p*255))<1e-10
        assert abs(result['ssim']-metrics.compare_ssim(gt*255,p*255))<1e-10
        
        
    package=types.ModuleType('_pcd_video_reference'); package.__path__=[str(root/'cacti/datasets')]
    sys.modules[package.__name__]=package
    registry=types.ModuleType('_pcd_video_reference.builder')
    registry.DATASETS=types.SimpleNamespace(register_module=lambda cls:cls)
    sys.modules[registry.__name__]=registry
    ds_class=importlib.import_module('_pcd_video_reference.six_gray_sim_data').SixGraySimData
    ds=ds_class(str(root/'test_datasets/simulation'),mask=seqs[0]['mask'])
    for seq in seqs:
        i=ds.data_name_list.index(Path(seq['source']).name)
        y,gt=ds[i]
        np.testing.assert_array_equal(gt,seq['x'])
        np.testing.assert_allclose(y,seq['y'],atol=1e-14,rtol=0)
