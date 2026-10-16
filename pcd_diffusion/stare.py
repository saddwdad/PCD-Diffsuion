




import torch
from torch import nn
from torch.nn import functional as F


def affine_scan(a, u):
    




    offset = 1
    while offset < u.shape[-1]:
        u = torch.cat([u[..., :offset], u[..., offset:] + a[..., offset:] * u[..., :-offset]], -1)
        a = torch.cat([a[..., :offset], a[..., offset:] * a[..., :-offset]], -1)
        offset *= 2
    return u


class GaussianSSM(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.GroupNorm(1, width)
        self.in_proj = nn.Conv3d(width, width*4, 1)
        self.local = nn.Conv3d(width, width, 3, padding=1, groups=width)
        self.log_a = nn.Parameter(torch.zeros(1, width, 1, 1, 1))
        self.skip = nn.Parameter(torch.ones(1, width, 1, 1, 1))
        self.sigma = nn.Parameter(torch.ones(2))
        self.out = nn.Conv3d(width, width, 1)

    def forward(self, x):
        v, delta, b, c = self.in_proj(self.norm(x)).chunk(4, 1)
        v = F.silu(self.local(v))
        n, h, w = x.shape[-3:]
        
        def coords(size):
            return torch.linspace(-1, 1, size, device=x.device) if size > 1 else x.new_zeros(1)
        tn, yy, xx = coords(n), coords(h), coords(w)
        sig_s, sig_t = F.softplus(self.sigma.float()) + 0.05
        ws = torch.exp(-(yy[:, None]**2 + xx[None, :]**2) / (2*sig_s**2))[None,None,None]
        wt = torch.exp(-tn**2 / (2*sig_t**2))[None,None,:,None,None]
        
        a = torch.exp(-F.softplus(delta.float()) * self.log_a.float().exp()) * wt
        u = torch.tanh(b.float()) * ws * v.float() * ws * wt
        result = torch.zeros_like(u)
        for axis in (2, 3, 4):
            aa, uu = a.movedim(axis, -1), u.movedim(axis, -1)
            fwd = affine_scan(aa, uu)
            rev = affine_scan(aa.flip(-1), uu.flip(-1)).flip(-1)
            result = result + (fwd + rev).movedim(-1, axis) / 6
        y = torch.tanh(c.float()) * result + self.skip * v
        return x + self.out(y.to(x.dtype))


class STARE(nn.Module):
    def __init__(self, width, blocks):
        super().__init__()
        self.condition = nn.Conv3d(3, width, 1)
        self.blocks = nn.ModuleList([GaussianSSM(width) for _ in range(blocks)])
        self.head = nn.Conv3d(width, width, 3, padding=1)
        self.correction = nn.Sequential(nn.Conv3d(width+1, width, 1), nn.SiLU(), nn.Conv3d(width, width, 1))

    def forward(self, z, beta, q, time):
        beta = beta.expand(z.shape[0], 1, *z.shape[-3:])
        q = q.expand_as(beta)
        f = z + self.condition(torch.cat([beta, q, torch.full_like(beta, time)], 1))
        for block in self.blocks:
            f = block(f)
        return self.head(f) + self.correction(torch.cat([f, q], 1))
