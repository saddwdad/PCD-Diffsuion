
import torch


def measure(x, mask):
    return (x * mask).sum(dim=2)


def adjoint(y, mask):
    return y.unsqueeze(2) * mask


def gap(x, y, mask):
    



    energy = mask.square().sum(2)
    return x + adjoint((y - measure(x, mask)) / energy.clamp_min(1e-8), mask)


def backproject(y, mask):
    return adjoint(y / mask.square().sum(2).clamp_min(1e-8), mask)


def validate(y, mask, channels):
    if y.ndim != 4 or mask.ndim != 5 or mask.shape[1] != 1:
        raise ValueError('Expected y[B,C,H,W], mask[B,1,N,H,W]')
    if y.shape[0] != mask.shape[0] or y.shape[-2:] != mask.shape[-2:]:
        raise ValueError('Snapshot/mask batch or spatial shape mismatch')
    if y.shape[1] != channels or mask.shape[2] < 1:
        raise ValueError('Wrong number of color channels or empty video')
    if not torch.isfinite(y).all() or not torch.isfinite(mask).all():
        raise ValueError('Nonfinite input')
    if (mask < 0).any():
        raise ValueError('Mask must be nonnegative')
