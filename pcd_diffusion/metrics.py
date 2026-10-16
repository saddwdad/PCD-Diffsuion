import torch
from torch.nn import functional as F


def quality(pred, target):
    




    pred,target = pred.float().clamp(0,1), target.float()
    b,c,n,h,w = pred.shape
    p = pred.permute(0,2,1,3,4).reshape(b*n,c,h,w)
    t = target.permute(0,2,1,3,4).reshape_as(p)
    mse = (p-t).square().mean((1,2,3))
    psnr = (-10*torch.log10(mse.clamp_min(1e-12))).mean()
    k = min(11,h,w)
    k -= (k+1)%2
    coord = torch.arange(k,device=p.device).float()-(k-1)/2
    g = torch.exp(-coord.square()/(2*1.5**2)); g = g/g.sum()
    win = (g[:,None]*g[None,:])[None,None].expand(c,1,k,k)
    conv = lambda a: F.conv2d(a,win,groups=c)
    mp,mt = conv(p),conv(t)
    vp,vt = (conv(p*p)-mp*mp).clamp_min(0),(conv(t*t)-mt*mt).clamp_min(0)
    covariance = conv(p*t)-mp*mt
    ssim = ((2*mp*mt+0.01**2)*(2*covariance+0.03**2)/((mp*mp+mt*mt+0.01**2)*(vp+vt+0.03**2))).mean()
    return {'psnr':psnr.item(), 'ssim':ssim.item()}
