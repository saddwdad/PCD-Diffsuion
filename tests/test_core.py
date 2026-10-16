import json
import numpy as np
import pytest
import torch
from pcd_diffusion.physics import measure, adjoint, gap
from pcd_diffusion.stare import affine_scan
from pcd_diffusion.model import PCDDiffusion, MotionSchedule
from pcd_diffusion.motion import MotionPrior
from pcd_diffusion.data import synthetic, load_sample, SnapshotDataset
from pcd_diffusion.metrics import quality
from pcd_diffusion.engine import train, load_checkpoint, infer

torch.set_num_threads(2)


def test_adjoint_identity():
    torch.manual_seed(10)
    x=torch.randn(2,3,4,7,9,dtype=torch.double)
    mask=torch.rand(2,1,4,7,9,dtype=torch.double)
    y=torch.randn(2,3,7,9,dtype=torch.double)
    torch.testing.assert_close((measure(x,mask)*y).sum(),(x*adjoint(y,mask)).sum())


def test_exact_gap_and_zero_mask():
    x=torch.randn(2,1,4,7,9)
    mask=torch.rand_like(x); mask[:,:,:,:,0]=0
    y=measure(torch.randn_like(x),mask)
    projected=gap(x,y,mask)
    torch.testing.assert_close(measure(projected,mask),y,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(projected[:,:,:,:,0],x[:,:,:,:,0])


@pytest.mark.parametrize('length',[1,2,7,16])
def test_scan_matches_recurrence_values_and_gradients(length):
    a=torch.rand(2,3,length,dtype=torch.double,requires_grad=True)
    u=torch.randn(2,3,length,dtype=torch.double,requires_grad=True)
    state=torch.zeros_like(u[...,0]); expected=[]
    for i in range(length):
        state=a[...,i]*state+u[...,i]; expected.append(state)
    expected=torch.stack(expected,-1); actual=affine_scan(a,u)
    torch.testing.assert_close(actual,expected)
    grad1=torch.autograd.grad(actual.square().sum(),(a,u),retain_graph=True,allow_unused=True)
    grad2=torch.autograd.grad(expected.square().sum(),(a,u),allow_unused=True)
    for g1,g2 in zip(grad1,grad2):
        if g1 is None: g1=torch.zeros_like(a)
        if g2 is None: g2=torch.zeros_like(a)
        torch.testing.assert_close(g1,g2)


def test_schedule_products_and_bounds():
    s=MotionSchedule(16); q=torch.rand(2,1,1,5,7)
    product=torch.ones_like(q)
    for i in range(16):
        beta,ab,prev=s.at(i,q)
        assert ((beta>0)&(beta<1)).all()
        torch.testing.assert_close(prev,product,atol=1e-6,rtol=1e-5)
        product=product*(1-beta)
        torch.testing.assert_close(ab,product)
    assert ab.max()<0.02


@pytest.mark.parametrize('channels,frames,height,width',[(1,1,5,7),(1,4,8,9),(3,3,9,7)])
def test_model_train_infer_and_gradient_coverage(channels,frames,height,width):
    torch.manual_seed(4)
    net=PCDDiffusion(channels=channels,width=4,blocks=1,steps=3)
    x=torch.rand(1,channels,frames,height,width)
    mask=(torch.rand(1,1,frames,height,width)>0.5).float(); mask[:,:,0]=1
    y=measure(x,mask)
    pred,loss=net(y,mask,target=x)
    assert pred.shape==x.shape and torch.isfinite(loss['total'])
    loss['total'].backward()
    for name,p in net.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
    net.eval()
    with torch.no_grad():
        torch.manual_seed(11); a=net(y,mask,stochastic=False)
        torch.manual_seed(11); b=net(y,mask,stochastic=False)
        torch.testing.assert_close(a,b)
        c=net(y,mask,stochastic=False,final_projection=True)
        torch.testing.assert_close(measure(c,mask),y,atol=2e-5,rtol=1e-5)


def test_latent_embedding_commutes_with_mask():
    net=PCDDiffusion(width=4,blocks=1,steps=3)
    x=torch.rand(1,1,3,8,9); mask=torch.rand(1,1,3,8,9)
    left=measure(net.embed(x),mask)
    right=net.embed(measure(x,mask).unsqueeze(2)).squeeze(2)
    torch.testing.assert_close(left,right)


def test_motion_static_and_dynamic():
    m=MotionPrior()
    x=torch.zeros(1,1,3,8,9)
    torch.testing.assert_close(m(x),torch.full((1,1,1,8,9),0.5))
    x[:,:,1]=1
    assert (m(x)>0.5).all()


def test_checkpointed_gradients():
    a=PCDDiffusion(width=4,blocks=1,steps=2)
    b=PCDDiffusion(width=4,blocks=1,steps=2,gradient_checkpointing=True)
    b.load_state_dict(a.state_dict())
    x=torch.rand(1,1,2,5,7); mask=torch.ones_like(x); y=measure(x,mask)
    torch.manual_seed(55); _,la=a(y,mask,target=x); la['total'].backward()
    torch.manual_seed(55); _,lb=b(y,mask,target=x); lb['total'].backward()
    for pa,pb in zip(a.parameters(),b.parameters()): torch.testing.assert_close(pa.grad,pb.grad)


def test_data_synthesis_and_flip(tmp_path):
    synthetic(tmp_path,count=1,frames=3,size=8,channels=3)
    item=load_sample(tmp_path/'0000.npz',True)
    torch.testing.assert_close(measure(item['x'][None],item['mask'][None])[0],item['y'])
    ds=SnapshotDataset(tmp_path,augment=True)
    for _ in range(4):
        item=ds[0]
        torch.testing.assert_close(measure(item['x'][None],item['mask'][None])[0],item['y'])


def test_bad_data_rejected(tmp_path):
    np.savez(tmp_path/'bad.npz',mask=np.zeros((2,5,5)),y=np.ones((1,5,5)))
    with pytest.raises(ValueError,match='unobserved'): load_sample(tmp_path/'bad.npz')
    np.savez(tmp_path/'bad.npz',mask=np.ones((2,5,5)),x=np.ones((2,1,5,5))*255)
    with pytest.raises(ValueError,match='normalized'): load_sample(tmp_path/'bad.npz')


def test_metrics_identity():
    x=torch.rand(2,3,4,12,14)
    m=quality(x,x)
    assert abs(m['ssim']-1)<1e-6 and m['psnr']>100


def test_training_resume_matches_uninterrupted_and_measurement_only_inference(tmp_path):
    synthetic(tmp_path/'data',count=2,frames=2,size=6)
    cfg={'model':{'width':4,'blocks':1,'steps':2},'train':{'epochs':2,'amp':False,'threads':2,'seed':19}}
    config=tmp_path/'config.json'; config.write_text(json.dumps(cfg))
    full=train(config,tmp_path/'data',tmp_path/'full','cpu')
    partial=train(config,tmp_path/'data',tmp_path/'part','cpu',epochs=1)
    resumed=train(config,tmp_path/'data',tmp_path/'part','cpu',resume=partial,epochs=2)
    a,_=load_checkpoint(full,torch.device('cpu')); b,_=load_checkpoint(resumed,torch.device('cpu'))
    for name,t in a.state_dict().items(): torch.testing.assert_close(t,b.state_dict()[name],atol=0,rtol=0)
    item=load_sample(tmp_path/'data'/'0000.npz')
    np.savez(tmp_path/'measurement.npz',y=item['y'].numpy(),mask=item['mask'][0].numpy())
    report=infer(resumed,tmp_path/'measurement.npz',tmp_path/'prediction','cpu')
    assert 'psnr' not in report and (tmp_path/'prediction'/'reconstruction.gif').exists()
