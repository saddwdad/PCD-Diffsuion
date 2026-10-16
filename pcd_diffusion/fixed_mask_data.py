




import hashlib
import json
import math
from pathlib import Path
import urllib.request
import cv2
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

COMMIT='59630253f779a3eaba38dd43201b3a82705bf99e'
ASSETS={
 'mask/random_mask.mat':'ad66fd694f23352d4415ab3f77e6bcce484937d2e8f3f7a881d46d5a88a47dba',
 'simulation/aerial32_cacti.mat':'1823e13a4fedde36ca4ffc07b1040495601c5f740ef4ffa25249bdff6763c468',
 'simulation/crash32_cacti.mat':'ebd4d639e20d1336f8af6352d09c2ff27eb3cebcc13949a48f10b174b03e7f50',
 'simulation/drop8_cacti.mat':'1a2974249f29e93fc648100c9c4ca9128c8285dbd8e56bfe0b0d953e6447e9c2',
 'simulation/kobe_cacti.mat':'098321e61e862461cdbf609a33abe6248fbe4dcf99a1f9386008525097e89262',
 'simulation/runner8_cacti.mat':'3c58416f7d19b97776f1b7d6471b85309a1eb05b9c51c59f8e6f507605ea10ed',
 'simulation/traffic_cacti.mat':'b43d542d3eef97d2bb5d666ff36f5dd60239d35bbcf2b17121c923b019318cf4',
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch_assets(root):
    
    root=Path(root)
    for relative,digest in ASSETS.items():
        local=Path('masks/random_mask.mat') if relative.startswith('mask/') else Path('six_gray')/Path(relative).name
        out=root/local
        if out.exists():
            if sha256(out)!=digest:
                raise ValueError(f'Existing file has unexpected hash: {out}; no overwrite performed')
            print(f'Verified {relative}',flush=True)
            continue
        out.parent.mkdir(parents=True,exist_ok=True)
        url=f'https://raw.githubusercontent.com/ucaswangls/STFormer/{COMMIT}/test_datasets/{relative}'
        temp=out.with_suffix('.download')
        with urllib.request.urlopen(url,timeout=90) as response, temp.open('wb') as stream:
            while chunk:=response.read(1024*1024): stream.write(chunk)
        if sha256(temp)!=digest:
            raise ValueError(f'Download hash mismatch: {temp}')
        temp.replace(out)
        print(f'Downloaded and verified {relative}',flush=True)


def load_mask(path):
    raw=loadmat(path)
    if 'mask' not in raw: raise ValueError(f'{path}: missing mask variable')
    mask=np.asarray(raw['mask'],dtype=np.float32)
    if mask.ndim!=3 or min(mask.shape)<1 or not np.isfinite(mask).all() or (mask<0).any():
        raise ValueError('Expected finite nonnegative MATLAB mask[H,W,N]')
    return np.ascontiguousarray(mask.transpose(2,0,1))


def draw_uniform(low,high):
    return low+(high-low)*torch.rand(()).item()


def transform_frames(images,height,width,mode):
    




    if mode not in ('none','paper','random_scale_crop_flip'):
        raise ValueError('augmentation must be none, paper or random_scale_crop_flip')
    if mode=='random_scale_crop_flip':
        if torch.rand(())<0.5:
            scale=draw_uniform(0.8,1.2)
            images=[cv2.resize(im,None,fx=scale,fy=scale,interpolation=cv2.INTER_LINEAR) for im in images]
        if torch.rand(())<0.5:
            h,w=images[0].shape[:2]
            
            
            ch=int(torch.randint(max(1,min(height//2,h-1)),h,()).item()) if h>1 else 1
            cw=int(torch.randint(max(1,min(width//2,w-1)),w,()).item()) if w>1 else 1
            top=int(torch.randint(h-ch+1,()).item()); left=int(torch.randint(w-cw+1,()).item())
            images=[im[top:top+ch,left:left+cw] for im in images]
    if mode!='none' and torch.rand(())<0.5:
        images=[cv2.flip(im,1) for im in images]
    if mode=='random_scale_crop_flip' and torch.rand(())<0.5:
        images=[cv2.flip(im,-1) for im in images]
    images=[cv2.resize(im,(width,height),interpolation=cv2.INTER_LINEAR) for im in images]
    return np.stack([cv2.cvtColor(im,cv2.COLOR_BGR2YCrCb)[:,:,0].astype(np.float32)/255 for im in images])


class FixedMaskVideoDataset(Dataset):
    def __init__(self,root,mask_path,split_file=None,augmentation='paper',sampling='sliding'):
        self.root=Path(root); self.mask_path=Path(mask_path); self.mask=load_mask(mask_path)
        self.frames,self.height,self.width=self.mask.shape
        self.augmentation,self.sampling=augmentation,sampling
        if sampling not in ('sliding','nonoverlap','suffix_nonoverlap'):
            raise ValueError('Invalid clip sampling mode')
        names=Path(split_file).read_text(encoding='utf-8').splitlines() if split_file else [p.name for p in self.root.iterdir() if p.is_dir()]
        self.names=sorted(name.strip() for name in names if name.strip())
        if len(self.names)!=len(set(self.names)): raise ValueError('Duplicate sequence names in split')
        self.clips=[]
        for name in self.names:
            folder=self.root/name
            paths=sorted(p for p in folder.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png'))
            length=len(paths)
            if sampling=='suffix_nonoverlap':
                
                
                starts=[start+offset for start in range(length-self.frames)
                        for offset in range(0,length-start-self.frames+1,self.frames)]
            else:
                starts=range(0,length-self.frames+1,1 if sampling=='sliding' else self.frames)
            for start in starts: self.clips.append((name,start,paths[start:start+self.frames]))
        if not self.clips: raise ValueError('No full DAVIS clips: check root, split and mask frame count')
        self.manifest={'type':'fixed_mask_video','root':str(self.root.resolve()),'mask_path':str(self.mask_path.resolve()),
                       'mask_sha256':sha256(mask_path),'mask_shape_NHW':list(self.mask.shape),
                       'sequences':self.names,'sample_count':len(self.clips),'augmentation':augmentation,
                       'sampling':sampling,'split_sha256':sha256(split_file) if split_file else None}

    def __len__(self): return len(self.clips)

    def __getitem__(self,index):
        name,start,paths=self.clips[index]
        images=[cv2.imread(str(p),cv2.IMREAD_COLOR) for p in paths]
        if any(im is None for im in images): raise ValueError(f'Cannot read frame in {name}/{start}')
        x=transform_frames(images,self.height,self.width,self.augmentation)
        y=(x*self.mask).sum(axis=0)
        return {'x':torch.from_numpy(x[None]),'mask':torch.from_numpy(self.mask[None].copy()),
                'y':torch.from_numpy(y[None]),'name':f'{name}_{start:05d}'}


def load_six_gray(root,mask_path=''):
    




    if not root or not mask_path:
        raise ValueError('Set the benchmark root and mask path before running')
    root=Path(root)
    mask_path=Path(mask_path)
    mask=load_mask(mask_path); n,h,w=mask.shape
    sequences=[]
    for rel in ASSETS:
        if not rel.startswith('simulation/'): continue
        path=root/Path(rel).name
        raw=loadmat(path)
        key=next((key for key in ('orig','patch_save','p1','p2','p3') if key in raw),None)
        if key is None: raise ValueError(f'{path}: no supported video key')
        video=np.asarray(raw[key],dtype=np.float64)/255
        if video.ndim!=3 or video.shape[0]<h or video.shape[1]<w or video.shape[2]%n:
            raise ValueError(f'{path}: expected H,W,frames divisible by mask frames')
        if not np.isfinite(video).all() or video.min()<0 or video.max()>1:
            raise ValueError(f'{path}: invalid video range')
        video=video[:h,:w,:].transpose(2,0,1)
        
        clips=video.reshape(-1,n,h,w)
        measurements=(clips*mask).sum(1)
        sequences.append({'name':Path(rel).stem,'x':clips,'y':measurements,'mask':mask,
                          'source':str(path),'source_sha256':sha256(path)})
    return sequences,{'mask_path':str(mask_path),'mask_sha256':sha256(mask_path),'upstream_commit':COMMIT}


def prepare_six_gray(root,output,mask_path=''):
    seqs,metadata=load_six_gray(root,mask_path)
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    metadata['sequences']=[]
    for seq in seqs:
        for i,(x,y) in enumerate(zip(seq['x'],seq['y'])):
            np.savez_compressed(out/f"{seq['name']}_{i:03d}.npz",x=x[:,None].astype(np.float32),
                                y=y[None].astype(np.float32),mask=seq['mask'])
        metadata['sequences'].append({k:seq[k] for k in ('name','source','source_sha256')})
        metadata['sequences'][-1]['snapshots']=len(seq['x'])
    (out/'manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return metadata


def frame_metrics(pred,target):
    
    p=np.asarray(pred*255,dtype=np.float32); t=np.asarray(target*255,dtype=np.float32)
    rmse=math.sqrt(float(np.mean((p-t)**2)))
    psnr=100.0 if rmse==0 else 20*math.log10(255/rmse)
    
    p=np.asarray(pred*255,dtype=np.float64); t=np.asarray(target*255,dtype=np.float64)
    if min(p.shape)<11: raise ValueError('Grayscale SSIM needs dimensions >= 11')
    g=cv2.getGaussianKernel(11,1.5); window=g@g.T
    conv=lambda a:cv2.filter2D(a,-1,window)[5:-5,5:-5]
    mp,mt=conv(p),conv(t)
    vp,vt=conv(p*p)-mp*mp,conv(t*t)-mt*mt
    covariance=conv(p*t)-mp*mt
    c1,c2=(0.01*255)**2,(0.03*255)**2
    ssim=((2*mp*mt+c1)*(2*covariance+c2)/((mp*mp+mt*mt+c1)*(vp+vt+c2))).mean()
    return {'psnr':psnr,'ssim':float(ssim)}


@torch.no_grad()
def evaluate_six_gray(model,root,device,seed=2026,mask_path='',output_dir=None):
    from .physics import measure
    seqs,metadata=load_six_gray(root,mask_path)
    old=model.training; model.eval()
    device=torch.device(device)
    devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    report={'protocol':'STFormer six-gray, raw predictions, mean frames then mean six sequences',
            'seed':seed,'data':metadata,'sequences':[]}
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for seq in seqs:
            rows=[]; recon=[]; mm=[]
            mask=torch.from_numpy(seq['mask'][None,None]).to(device)
            for i,(truth,measurement) in enumerate(zip(seq['x'],seq['y'])):
                y=torch.from_numpy(measurement[None,None].astype(np.float32)).to(device)
                pred=model(y,mask,stochastic=False)
                if not torch.isfinite(pred).all(): raise FloatingPointError('Nonfinite benchmark output')
                raw=pred[0,0].float().cpu().numpy()
                recon.append(raw)
                mm.append((measure(pred,mask)-y).square().mean().item())
                for j in range(len(truth)):
                    rows.append({'snapshot':i,'frame':j,**frame_metrics(raw[j],truth[j])})
            row={'name':seq['name'],'snapshots':len(seq['x']),'frames':len(rows),
                 'psnr':float(np.mean([r['psnr'] for r in rows])),
                 'ssim':float(np.mean([r['ssim'] for r in rows])),
                 'measurement_mse':float(np.mean(mm)),'per_frame':rows,'source_sha256':seq['source_sha256']}
            report['sequences'].append(row)
            print(json.dumps({k:row[k] for k in ('name','snapshots','psnr','ssim')}),flush=True)
            if output_dir:
                out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(out/f"{seq['name']}.npz",reconstruction=np.stack(recon))
    model.train(old)
    report['mean']={key:float(np.mean([r[key] for r in report['sequences']])) for key in ('psnr','ssim')}
    return report
