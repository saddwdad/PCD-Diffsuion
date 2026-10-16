





from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .vendor.prior_reconstruction.latent_encoder_arch import latent_encoder_gelu_mobile
from .vendor.prior_reconstruction.prior_decoder import PriorGuidedReconstructor
from .motion import MotionPrior
from .model import MotionSchedule
from .stare import STARE
from .physics import validate, measure, backproject, gap


class PriorTokenCodec(nn.Module):
    def __init__(self, frames=8, dim=28, encoder_blocks=2):
        super().__init__()
        if frames < 1 or dim < 8 or dim % 2:
            raise ValueError('frames >= 1; reconstruction dim must be even and >= 8')
        self.frames = frames
        self.gt_le = latent_encoder_gelu_mobile(in_chans=2*frames, block_num=encoder_blocks)
        self.condition_le = latent_encoder_gelu_mobile(in_chans=frames, block_num=encoder_blocks)
        opt = SimpleNamespace(in_dim=frames, out_dim=frames, dim=dim, bias=False,
                              DW_Expand=1, LayerNorm_type='BiasFree',
                              ffn_name='Gated_Dconv_FeedForward', FFN_Expand=2.66,
                              act_fn_name='gelu')
        self.reconstruction = PriorGuidedReconstructor(opt, use_prior=True)
        
        self.down_1_patch = nn.Linear(16, 4)
        self.down_1_channel = nn.Linear(256, 256)
        self.down_2_patch = nn.Linear(4, 1)
        self.down_2_channel = nn.Linear(256, 256)

    @staticmethod
    def pad(x):
        
        h,w=x.shape[-2:]
        return F.pad(x,(0,(-w)%8,0,(-h)%8),mode='replicate')

    def encode(self, base, target=None):
        base=self.pad(base)
        if target is None:
            return self.condition_le(base)
        return self.gt_le(base,self.pad(target))

    def decode(self, base, tokens, state=None):
        h,w=base.shape[-2:]
        p2=self.down_1_channel(self.down_1_patch(tokens.transpose(1,2)).transpose(1,2))
        p3=self.down_2_channel(self.down_2_patch(p2.transpose(1,2)).transpose(1,2))
        args={} if state is None else dict(enc_outputs=state[0],bottleneck_out=state[1],dec_outputs=state[2])
        out,enc,bottleneck,dec=self.reconstruction(self.pad(base),prior=[tokens,p2,p3],**args)
        return out[...,:h,:w],(enc,bottleneck,dec)


class PriorTokenDiffusion(nn.Module):
    def __init__(self, channels=1, frames=8, width=64, blocks=4, steps=128,
                 reconstruction_dim=28, encoder_blocks=2, motion_backend='raft',
                 raft_weights=None, raft_updates=12, gradient_checkpointing=True,
                 lambda_diff=0.5, lambda_measure=0.5, lambda_recon=1.0,
                 lambda_autoencoder=0.1):
        super().__init__()
        if channels!=1:
            raise ValueError('Prior-token reconstruction currently supports grayscale video only')
        if width<1 or blocks<1:
            raise ValueError('width and blocks must be positive')
        self.channels,self.frames,self.steps=channels,frames,steps
        self.gradient_checkpointing=gradient_checkpointing
        self.loss_weights=(lambda_diff,lambda_measure,lambda_recon,lambda_autoencoder)
        self.codec=PriorTokenCodec(frames,reconstruction_dim,encoder_blocks)
        self.motion=MotionPrior(motion_backend,raft_weights,raft_updates)
        self.schedule=MotionSchedule(steps)
        self.token_in=nn.Conv3d(512,width,1)
        self.denoiser=STARE(width,blocks)
        self.token_out=nn.Conv3d(width,256,1)
        self.pixel_correction=nn.Conv2d(frames,frames,3,padding=1,bias=False)
        nn.init.dirac_(self.pixel_correction.weight)
        self.token_correction=nn.Linear(256,256,bias=False)
        nn.init.eye_(self.token_correction.weight)

    @staticmethod
    def grid(tokens):
        return tokens.transpose(1,2).reshape(tokens.shape[0],256,1,4,4)

    @staticmethod
    def tokens(grid):
        return grid.squeeze(2).flatten(2).transpose(1,2)

    def predict_epsilon(self,z,condition,beta,q,time):
        features=self.token_in(torch.cat([z,condition],dim=1))
        return self.token_out(self.denoiser(features,beta,q,time))

    def forward(self,y,mask,target=None,stochastic=True,final_projection=False):
        validate(y,mask,1)
        if mask.shape[2]!=self.frames or min(y.shape[-2:])<16:
            raise ValueError(f'Prior-token model requires N={self.frames} and H,W >= 16')
        if target is not None and target.shape!=(y.shape[0],1,self.frames,*y.shape[-2:]):
            raise ValueError('Target shape mismatch')
        base=backproject(y,mask).squeeze(1)
        condition=self.grid(self.codec.encode(base))
        clean=self.grid(self.codec.encode(base,target.squeeze(1))) if target is not None else None
        motion=self.motion(base.unsqueeze(1))
        q=F.adaptive_avg_pool2d(motion.squeeze(2),(4,4)).unsqueeze(2)
        if target is not None:
            _,ab,_=self.schedule.at(self.steps-1,q)
            z=ab.sqrt()*clean+(1-ab).sqrt()*torch.randn_like(clean)
        else:
            z=torch.randn_like(condition)
        x=base
        state=None
        diffusion_loss=y.new_zeros(())
        for i in reversed(range(self.steps)):
            beta,ab,prev=self.schedule.at(i,q)
            if target is not None:
                eps_target=((z-ab.sqrt()*clean)/(1-ab).clamp_min(1e-6).sqrt()).detach()
            if self.training and self.gradient_checkpointing:
                eps=checkpoint(self.predict_epsilon,z,condition,beta,q,(i+1)/self.steps,use_reentrant=False)
            else:
                eps=self.predict_epsilon(z,condition,beta,q,(i+1)/self.steps)
            if target is not None:
                diffusion_loss=diffusion_loss+F.mse_loss(eps.float(),eps_target.float())/self.steps
            z=(z-beta/(1-ab).clamp_min(1e-6).sqrt()*eps)/(1-beta).sqrt()
            if stochastic and i>0:
                variance=(beta*(1-prev)/(1-ab).clamp_min(1e-6)).clamp_min(0)
                z=z+variance.sqrt()*torch.randn_like(z)
            
            
            raw,_=self.codec.decode(x,self.tokens(z),state)
            correction=(gap(raw.unsqueeze(1),y,mask).squeeze(1)-raw)*motion.squeeze(2)
            corrected=raw+self.pixel_correction(correction)
            delta=self.codec.encode(base,corrected)-self.codec.encode(base,raw)
            z=z+self.grid(self.token_correction(delta))
            x,state=self.codec.decode(corrected,self.tokens(z),state)
            if i>0:
                motion=self.motion(x.unsqueeze(1))
                q=F.adaptive_avg_pool2d(motion.squeeze(2),(4,4)).unsqueeze(2)
        x=x.unsqueeze(1)
        if target is None:
            return gap(x,y,mask) if final_projection else x
        ae,_=self.codec.decode(base,self.tokens(clean))
        losses=dict(diffusion=diffusion_loss,
                    measure=F.mse_loss(measure(x,mask).float(),y.float()),
                    reconstruction=F.mse_loss(x.float(),target.float()),
                    autoencoder=F.mse_loss(ae.unsqueeze(1).float(),target.float()))
        losses['total']=sum(w*l for w,l in zip(self.loss_weights,losses.values()))
        return x,losses
