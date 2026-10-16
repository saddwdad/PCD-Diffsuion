from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


def load_sample(path, require_target=False):
    
    with np.load(path, allow_pickle=False) as pack:
        if 'mask' not in pack:
            raise ValueError(f'{path}: missing mask')
        mask = np.array(pack['mask'], dtype=np.float32)
        x = np.array(pack['x'], dtype=np.float32) if 'x' in pack else None
        y = np.array(pack['y'], dtype=np.float32) if 'y' in pack else None
    if mask.ndim != 3 or min(mask.shape) < 1 or not np.isfinite(mask).all() or (mask < 0).any():
        raise ValueError(f'{path}: mask must be finite nonnegative [N,H,W]')
    n,h,w = mask.shape
    if require_target and x is None:
        raise ValueError(f'{path}: training/evaluation requires x')
    if x is not None:
        if x.ndim != 4 or x.shape[0] != n or x.shape[1] not in (1,3) or x.shape[-2:] != (h,w):
            raise ValueError(f'{path}: x must be [N,C,H,W], C=1 or 3')
        if not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
            raise ValueError(f'{path}: x must be normalized to [0,1]')
        if y is None:
            y = (x * mask[:,None]).sum(0)
    if y is None or y.ndim != 3 or y.shape[0] not in (1,3) or y.shape[-2:] != (h,w):
        raise ValueError(f'{path}: provide y[C,H,W] or x[N,C,H,W]')
    if not np.isfinite(y).all() or (x is not None and y.shape[0] != x.shape[1]):
        raise ValueError(f'{path}: invalid y')
    if ((mask.sum(0) == 0)[None] & (np.abs(y) > 1e-6)).any():
        raise ValueError(f'{path}: nonzero measurement at completely unobserved pixel')
    result = {'y': torch.from_numpy(y), 'mask': torch.from_numpy(mask[None]), 'name': Path(path).stem}
    if x is not None:
        result['x'] = torch.from_numpy(x.transpose(1,0,2,3).copy())
    return result


class SnapshotDataset(Dataset):
    def __init__(self, directory, augment=False):
        self.files = sorted(Path(directory).glob('*.npz'))
        if not self.files:
            raise ValueError(f'No .npz samples in {directory}')
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        item = load_sample(self.files[index], require_target=True)
        if self.augment and torch.rand(()) < 0.5:
            for key in ('x','y','mask'):
                item[key] = item[key].flip(-1)
        return item


def random_mask(rng, frames, height, width):
    mask = (rng.random((frames,height,width)) > 0.5).astype(np.float32)
    missing = mask.sum(0) == 0
    mask[0, missing] = 1
    return mask


def synthetic(directory, count=8, frames=4, size=24, channels=1, seed=7):
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    yy,xx = np.mgrid[:size,:size]
    for sample in range(count):
        bg = rng.uniform(0.03,0.15,(channels,1,1)) + xx[None]/size*0.1
        x = np.broadcast_to(bg,(frames,channels,size,size)).copy()
        px,py = rng.integers(0,size,2)
        vx,vy = rng.choice([-2,-1,1,2],2)
        color = rng.uniform(0.5,1,(channels,1,1))
        for t in range(frames):
            disk = (xx-(px+vx*t)%size)**2 + (yy-(py+vy*t)%size)**2 < (size/6)**2
            x[t] = np.where(disk[None],color,x[t])
        mask = random_mask(rng,frames,size,size)
        np.savez_compressed(out/f'{sample:04d}.npz',x=x.astype('float32'),mask=mask)


def prepare_frames(root, output, frames=8, size=256, channels=1, seed=7, split_file=None):
    root,out = Path(root),Path(output)
    out.mkdir(parents=True, exist_ok=True)
    names = Path(split_file).read_text().splitlines() if split_file else [p.name for p in root.iterdir() if p.is_dir()]
    rng = np.random.default_rng(seed)
    total = 0
    for name in sorted(names):
        paths = sorted(p for p in (root/name).iterdir() if p.suffix.lower() in ('.jpg','.png','.jpeg'))
        for start in range(0,len(paths)-frames+1,frames):
            images = []
            for path in paths[start:start+frames]:
                with Image.open(path) as im:
                    arr = np.asarray(im.convert('L' if channels==1 else 'RGB').resize((size,size),Image.Resampling.BICUBIC), dtype=np.float32)/255
                images.append(arr[None] if channels==1 else arr.transpose(2,0,1))
            np.savez_compressed(out/f'{name}_{start:05d}.npz',x=np.stack(images),mask=random_mask(rng,frames,size,size))
            total += 1
    if total == 0:
        raise ValueError('No full video clips found; check root/split/frames')
    return total
