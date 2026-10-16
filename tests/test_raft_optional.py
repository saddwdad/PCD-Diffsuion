



import os
import pytest
import torch
from pcd_diffusion.model import PCDDiffusion
from pcd_diffusion.physics import measure
from pcd_diffusion.engine import load_checkpoint


@pytest.mark.skipif(os.environ.get('PCD_TEST_RAFT')!='1',reason='Opt-in pretrained RAFT test')
def test_pretrained_raft_training_freeze_and_checkpoint(tmp_path):
    torch.set_num_threads(2)
    args=dict(width=4,blocks=1,steps=2,motion_backend='raft',raft_updates=2)
    model=PCDDiffusion(**args).train()
    x=torch.rand(1,1,2,16,16); mask=torch.ones_like(x); y=measure(x,mask)
    _,loss=model(y,mask,target=x)
    loss['total'].backward()
    assert torch.isfinite(loss['total'])
    assert not model.motion.flow.training
    assert all(p.grad is None for p in model.motion.flow.parameters())
    path=tmp_path/'raft.pt'
    torch.save({'model':model.state_dict(),'config':{'model':args}},path)
    restored,_=load_checkpoint(path,torch.device('cpu'))
    with torch.no_grad():
        torch.manual_seed(9); a=model.eval()(y,mask,stochastic=False)
        torch.manual_seed(9); b=restored(y,mask,stochastic=False)
    torch.testing.assert_close(a,b)


@pytest.mark.skipif(os.environ.get('PCD_TEST_RAFT')!='1',reason='Opt-in pretrained RAFT test')
def test_prior_token_pretrained_raft_and_factory_checkpoint(tmp_path):
    from pcd_diffusion.factory import build_model
    torch.set_num_threads(2)
    args=dict(type='prior_tokens',frames=2,width=8,blocks=1,steps=2,reconstruction_dim=8,
              encoder_blocks=1,motion_backend='raft',raft_updates=2,gradient_checkpointing=False)
    net=build_model(args).train()
    x=torch.rand(1,1,2,16,16); mask=torch.ones_like(x); y=measure(x,mask)
    _,loss=net(y,mask,target=x); loss['total'].backward()
    assert torch.isfinite(loss['total'])
    assert all(p.grad is None for p in net.motion.flow.parameters())
    path=tmp_path/'prior_token_raft.pt'
    torch.save({'model':net.state_dict(),'config':{'model':args}},path)
    restored,_=load_checkpoint(path,torch.device('cpu'))
    with torch.no_grad():
        torch.manual_seed(13); a=net.eval()(y,mask,stochastic=False)
        torch.manual_seed(13); b=restored(y,mask,stochastic=False)
    torch.testing.assert_close(a,b)
