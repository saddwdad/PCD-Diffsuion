import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .physics import measure, adjoint, backproject, validate, gap
from .motion import MotionPrior
from .stare import STARE


class MotionSchedule(nn.Module):
    def __init__(self, steps):
        super().__init__()
        if steps < 2:
            raise ValueError('At least two diffusion steps required')
        t = torch.linspace(0, 1, steps+1)
        ab = torch.cos((t + 0.008) / 1.008 * math.pi / 2).square()
        beta = (1 - ab[1:] / ab[:-1]).clamp(1e-4, 0.94)
        self.dense = nn.Parameter(torch.logit((beta * 0.9) / 0.999))
        self.sparse = nn.Parameter(torch.logit(beta / 0.999))
        self.steps = steps

    def at(self, index, q):
        




        d = (0.999 * self.dense.sigmoid()).clamp_min(1e-5)
        s = (0.999 * self.sparse.sigmoid()).clamp_min(1e-5)
        beta = q * d[index] + (1 - q) * s[index]
        prefix = q.unsqueeze(0) * d[:index+1, None, None, None, None, None]
        prefix = prefix + (1-q).unsqueeze(0) * s[:index+1, None, None, None, None, None]
        ab = torch.exp(torch.log1p(-prefix).sum(0))
        prev = ab / (1 - beta)
        return beta, ab, prev.clamp(max=1)


class PCDDiffusion(nn.Module):
    def __init__(self, channels=1, width=16, blocks=2, steps=4,
                 motion_backend='difference', raft_weights=None, raft_updates=12,
                 gradient_checkpointing=False, lambda_diff=0.5,
                 lambda_measure=0.5, lambda_recon=1.0, lambda_autoencoder=0.1):
        super().__init__()
        if channels not in (1, 3) or width < channels or blocks < 1:
            raise ValueError('channels must be 1 or 3; width >= channels; blocks >= 1')
        self.channels, self.width, self.steps = channels, width, steps
        self.gradient_checkpointing = gradient_checkpointing
        self.loss_weights = (lambda_diff, lambda_measure, lambda_recon, lambda_autoencoder)
        
        self.embed = nn.Conv3d(channels, width, 1, bias=False)
        self.decode = nn.Conv3d(width, channels, 1, bias=False)
        with torch.no_grad():
            basis = torch.linalg.qr(torch.randn(width, channels)).Q
            self.embed.weight.copy_(basis[:, :, None, None, None])
            self.decode.weight.copy_(basis.T[:, :, None, None, None])
        self.motion = MotionPrior(motion_backend, raft_weights, raft_updates)
        self.schedule = MotionSchedule(steps)
        self.denoiser = STARE(width, blocks)
        self.gap_conv = nn.Conv3d(width, width, (1,3,3), padding=(0,1,1), bias=False)
        nn.init.dirac_(self.gap_conv.weight)

    def feature_gap(self, z, yp, mask, q):
        energy = mask.square().sum(2).clamp_min(1e-8)
        residual = adjoint((yp - measure(z, mask)) / energy, mask)
        return z + self.gap_conv(q * residual)

    def forward(self, y, mask, target=None, stochastic=True, final_projection=False):
        



        validate(y, mask, self.channels)
        yp = self.embed(y.unsqueeze(2)).squeeze(2)
        q = self.motion(backproject(y, mask))
        shape = (y.shape[0], self.width, mask.shape[2], *y.shape[-2:])
        clean = self.embed(target) if target is not None else None
        if target is not None:
            if target.shape != (y.shape[0], self.channels, mask.shape[2], *y.shape[-2:]):
                raise ValueError('Target shape mismatch')
            _, ab, _ = self.schedule.at(self.steps-1, q)
            z = ab.sqrt() * clean + (1-ab).sqrt() * torch.randn_like(clean)
        else:
            z = torch.randn(shape, device=y.device, dtype=y.dtype)
        diffusion_loss = y.new_zeros(())
        for i in reversed(range(self.steps)):
            beta, ab, prev = self.schedule.at(i, q)
            if target is not None:
                
                
                
                eps_target = ((z - ab.sqrt()*clean) / (1-ab).clamp_min(1e-6).sqrt()).detach()
            if self.training and self.gradient_checkpointing:
                eps = checkpoint(self.denoiser, z, beta, q, (i+1)/self.steps, use_reentrant=False)
            else:
                eps = self.denoiser(z, beta, q, (i+1)/self.steps)
            if target is not None:
                diffusion_loss = diffusion_loss + F.mse_loss(eps.float(), eps_target.float()) / self.steps
            z = (z - beta / (1-ab).clamp_min(1e-6).sqrt()*eps) / (1-beta).sqrt()
            if stochastic and i > 0:
                variance = (beta*(1-prev)/(1-ab).clamp_min(1e-6)).clamp_min(0)
                z = z + variance.sqrt() * torch.randn_like(z)
            z = self.feature_gap(z, yp, mask, q)
            x = self.decode(z)
            if i > 0:
                q = self.motion(x)
        if target is None:
            
            
            return gap(x, y, mask) if final_projection else x
        losses = dict(diffusion=diffusion_loss,
                      measure=F.mse_loss(measure(x, mask).float(), y.float()),
                      reconstruction=F.mse_loss(x.float(), target.float()),
                      autoencoder=F.mse_loss(self.decode(clean).float(), target.float()))
        losses['total'] = sum(w*l for w,l in zip(self.loss_weights, losses.values()))
        return x, losses
