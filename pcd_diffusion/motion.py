
import math
import torch
from torch import nn
from torch.nn import functional as F


class MotionPrior(nn.Module):
    def __init__(self, backend='difference', raft_weights=None, raft_updates=12):
        super().__init__()
        if backend not in ('difference', 'raft'):
            raise ValueError('motion_backend must be difference or raft')
        self.backend, self.raft_updates = backend, raft_updates
        self.flow = None
        if backend == 'raft':
            from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
            self.flow = raft_small(weights=None if raft_weights else Raft_Small_Weights.DEFAULT)
            if raft_weights:
                self.flow.load_state_dict(torch.load(raft_weights, map_location='cpu', weights_only=True))
            self.flow.requires_grad_(False).eval()
        self.attention = nn.Conv2d(3, 3, 1)
        self.slope = nn.Parameter(torch.tensor(1.0))
        coords = torch.arange(-2, 3).float()
        kernel = torch.exp(-(coords[:, None] ** 2 + coords[None, :] ** 2) / 2)
        self.register_buffer('kernel', (kernel / kernel.sum())[None, None])

    def train(self, mode=True):
        super().train(mode)
        if self.flow is not None:
            self.flow.eval()
        return self

    @torch.no_grad()
    def raw_motion(self, x):
        b, c, n, h, w = x.shape
        if n == 1:
            return x.new_zeros(b, 1, 1, h, w)
        x = x.detach().float().clamp(0, 1)
        if self.backend == 'difference':
            
            raw = (x[:, :, 1:] - x[:, :, :-1]).abs().mean(1, keepdim=True)
        else:
            raw_frames = []
            rh, rw = max(128, math.ceil(h / 8) * 8), max(128, math.ceil(w / 8) * 8)
            with torch.autocast(device_type=x.device.type, enabled=False):
                for i in range(n - 1):
                    a, z = x[:, :, i], x[:, :, i + 1]
                    if c == 1:
                        a, z = a.repeat(1, 3, 1, 1), z.repeat(1, 3, 1, 1)
                    a = F.interpolate(a, (rh, rw), mode='bilinear', align_corners=False)
                    z = F.interpolate(z, (rh, rw), mode='bilinear', align_corners=False)
                    flow = self.flow(a * 2 - 1, z * 2 - 1, num_flow_updates=self.raft_updates)[-1]
                    flow = F.interpolate(flow, (h, w), mode='bilinear', align_corners=False)
                    flow = flow * flow.new_tensor([w / rw, h / rh])[None, :, None, None]
                    raw_frames.append(flow.norm(dim=1, keepdim=True))
            raw = torch.stack(raw_frames, 2)
        return torch.cat([raw, raw[:, :, -1:]], dim=2)

    def forward(self, x):
        raw = self.raw_motion(x)
        b, _, n, h, w = raw.shape
        flat = raw.permute(0, 2, 1, 3, 4).reshape(b*n, 1, h, w)
        local = F.avg_pool2d(F.pad(flat, (1, 1, 1, 1), mode='replicate'), 3, 1)
        region = F.conv2d(F.pad(flat, (2, 2, 2, 2), mode='replicate'), self.kernel)
        
        vals, order = flat.flatten(2).sort(-1)
        weights = (1 + flat.flatten(2)).gather(-1, order)
        index = (weights.cumsum(-1) < weights.sum(-1, keepdim=True) / 2).sum(-1, keepdim=True)
        glob = vals.gather(-1, index.clamp_max(h*w-1))[..., None].expand(-1, -1, h, w)
        multi = torch.cat([local, region, glob], dim=1)
        agg = (multi * self.attention(multi).softmax(dim=1)).sum(1, keepdim=True)
        agg = agg.reshape(b, n, 1, h, w).transpose(1, 2)
        
        return torch.sigmoid(F.softplus(self.slope) * agg.sum(2, keepdim=True))
