import json
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from PIL import Image
from .model import PCDDiffusion
from .factory import build_model
from .data import SnapshotDataset, load_sample
from .physics import measure
from .metrics import quality


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Use --device cpu or install a CUDA-enabled torch build.')
    return device


def load_config(path):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(config.get('model'),dict) or not isinstance(config.get('train'),dict):
        raise ValueError('Configuration requires model and train objects')
    return config


def batch_to(item, device):
    return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in item.items()}


def load_checkpoint(path, device, motion_override=None):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    cfg = ck['config']
    args = dict(cfg['model'])
    
    if args.get('motion_backend') == 'raft':
        args['motion_backend'] = 'difference'
        model = build_model(args)
        from torchvision.models.optical_flow import raft_small
        model.motion.backend = 'raft'
        model.motion.flow = raft_small(weights=None).requires_grad_(False).eval()
    else:
        model = build_model(args)
    model.load_state_dict(ck['model'], strict=True)
    if motion_override is not None:
        if motion_override != 'difference':
            raise ValueError('Only explicit difference override supported')
        model.motion.backend = motion_override
    return model.to(device).eval(), ck


@torch.no_grad()
def evaluate_model(model, directory, device, seed=2026, final_projection=False):
    
    was_training = model.training
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    results=[]
    model.eval()
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for item in DataLoader(SnapshotDataset(directory),batch_size=1):
            item=batch_to(item,device)
            pred=model(item['y'],item['mask'],stochastic=False,final_projection=final_projection)
            row=quality(pred,item['x'])
            row['measurement_mse']=(measure(pred,item['mask'])-item['y']).square().mean().item()
            row['name']=item['name'][0]
            results.append(row)
    model.train(was_training)
    mean={key:float(np.mean([r[key] for r in results])) for key in ('psnr','ssim','measurement_mse')}
    return {'mean':mean,'samples':results,'seed':seed,'final_projection':final_projection}


def train(config_path, data, output, device_name='auto', val=None, resume=None, epochs=None,
          mask_path=None, split_file=None):
    config=load_config(config_path)
    tc=config['train']
    if epochs is not None:
        tc['epochs']=epochs
    if tc['epochs'] < 1:
        raise ValueError('epochs must be positive')
    world=int(os.environ.get('WORLD_SIZE',1)); rank=int(os.environ.get('RANK',0))
    local=int(os.environ.get('LOCAL_RANK',0))
    device=device_for(device_name)
    if world>1:
        if device.type=='cuda':
            torch.cuda.set_device(local); device=torch.device('cuda',local)
        dist.init_process_group('nccl' if device.type=='cuda' and os.name!='nt' else 'gloo')
    seed_all(tc.get('seed',7)+rank)
    torch.set_num_threads(tc.get('threads',4))
    dc=config.get('data',{'type':'npz'})
    if dc.get('type')=='fixed_mask_video':
        from .fixed_mask_data import FixedMaskVideoDataset
        dc=dict(dc)
        if mask_path is not None: dc['mask_path']=str(mask_path)
        if split_file is not None: dc['split_file']=str(split_file)
        config['data']=dc
        ds=FixedMaskVideoDataset(data,dc['mask_path'],dc.get('split_file'),
                               dc.get('augmentation','paper'),dc.get('sampling','sliding'))
        if config['model'].get('frames',ds.frames)!=ds.frames:
            raise ValueError('Model frame count and fixed mask frame count disagree')
    elif dc.get('type','npz')=='npz':
        ds=SnapshotDataset(data,augment=True)
    else:
        raise ValueError(f"Unknown dataset type: {dc.get('type')}")
    sampler=DistributedSampler(ds,shuffle=True,seed=tc.get('seed',7)) if world>1 else None
    loader=DataLoader(ds,batch_size=tc.get('batch_size',1),shuffle=sampler is None,
                      sampler=sampler,num_workers=tc.get('workers',0),pin_memory=device.type=='cuda')
    ck=None
    if resume:
        net,ck=load_checkpoint(resume,device)
        if ck['config']['model']!=config['model']:
            raise ValueError('Resume requires the identical model config')
        if ck['config']['train'].get('batch_size',1)!=tc.get('batch_size',1):
            raise ValueError('Resume requires the same per-rank batch size')
        if ck.get('data_manifest') and hasattr(ds,'manifest'):
            keys=('mask_sha256','split_sha256','sequences','sampling','augmentation')
            if any(ck['data_manifest'].get(k)!=ds.manifest.get(k) for k in keys):
                raise ValueError('Resume data protocol differs: mask/split/sampling/augmentation changed')
    else:
        net=build_model(config['model']).to(device)
    
    model=DDP(net,device_ids=[local] if device.type=='cuda' else None,find_unused_parameters=True) if world>1 else net
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                               lr=tc.get('lr',1e-4),weight_decay=tc.get('weight_decay',1e-5))
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=tc.get('lr_step',100),gamma=0.5)
    use_amp=bool(tc.get('amp',True) and device.type=='cuda')
    scaler=torch.amp.GradScaler('cuda',enabled=use_amp)
    start=0; best=-float('inf')
    if ck:
        optimizer.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler'])
        scaler.load_state_dict(ck['scaler']); start=ck['epoch']+1; best=ck['best_psnr']
        
        
        rngs=ck.get('rng_by_rank',[])
        if rngs and len(rngs)!=world:
            raise ValueError('Resume requires the same world size for RNG restoration')
        if rngs:
            rng=rngs[rank]; torch.set_rng_state(rng['cpu'])
            if device.type=='cuda' and rng['cuda'] is not None:
                torch.cuda.set_rng_state(rng['cuda'],device)
    if start >= tc['epochs']:
        raise ValueError(f'Checkpoint already completed {start} epochs; --epochs must be larger')
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    if rank==0:
        (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
        if hasattr(ds,'manifest'):
            (out/'data_manifest.json').write_text(json.dumps(ds.manifest,indent=2),encoding='utf-8')
        print(json.dumps({'device':str(device),'world_size':world,'parameters':sum(p.numel() for p in net.parameters()),'samples':len(ds)}),flush=True)
    try:
        for epoch in range(start,tc['epochs']):
            model.train()
            if sampler:
                sampler.set_epoch(epoch)
            accum=torch.zeros(6,device=device)
            for batch in loader:
                batch=batch_to(batch,device); optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type,enabled=use_amp):
                    _,loss=model(batch['y'],batch['mask'],target=batch['x'])
                if not torch.isfinite(loss['total']):
                    raise FloatingPointError(f'Nonfinite loss at epoch {epoch+1}')
                scaler.scale(loss['total']).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),tc.get('clip_grad',1.0),error_if_nonfinite=True)
                scaler.step(optimizer); scaler.update()
                accum[:5]+=torch.stack([loss[k].detach() for k in ('total','diffusion','measure','reconstruction','autoencoder')])
                accum[5]+=1
            scheduler.step()
            if world>1:
                dist.all_reduce(accum)
            row={'epoch':epoch+1,'lr':optimizer.param_groups[0]['lr']}
            row.update({k:v for k,v in zip(('total','diffusion','measure','reconstruction','autoencoder'),(accum[:5]/accum[5]).tolist())})
            is_best=False
            if rank==0 and val:
                metrics=evaluate_model(net,val,device)
                row['validation']=metrics['mean']
                is_best=metrics['mean']['psnr']>best
                if is_best: best=metrics['mean']['psnr']
            rng={'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(device) if device.type=='cuda' else None}
            rngs=[None]*world
            if world>1:
                dist.all_gather_object(rngs,rng)
            else:
                rngs=[rng]
            if rank==0:
                payload={'model':net.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                         'scaler':scaler.state_dict(),'epoch':epoch,'best_psnr':best,'config':config,'rng_by_rank':rngs}
                if hasattr(ds,'manifest'): payload['data_manifest']=ds.manifest
                temp=out/'last.tmp'; torch.save(payload,temp); temp.replace(out/'last.pt')
                if is_best: torch.save(payload,out/'best.pt')
                with (out/'train.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
            if world>1: dist.barrier()
    finally:
        if world>1: dist.destroy_process_group()
    return out/'last.pt'


@torch.no_grad()
def infer(checkpoint_path, source, output, device_name='auto', seed=7, final_projection=False, stochastic=False):
    device=device_for(device_name); seed_all(seed); torch.set_num_threads(4)
    model,_=load_checkpoint(checkpoint_path,device)
    item=load_sample(source)
    y,mask=item['y'][None].to(device),item['mask'][None].to(device)
    x=model(y,mask,stochastic=stochastic,final_projection=final_projection)
    if not torch.isfinite(x).all(): raise FloatingPointError('Nonfinite reconstruction')
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    raw=x[0].permute(1,0,2,3).cpu().numpy()
    np.savez_compressed(out/'reconstruction.npz',x=raw,y=item['y'].numpy(),mask=item['mask'][0].numpy())
    frames=[]
    for i,frame in enumerate(raw):
        arr=(np.clip(frame,0,1)*255).round().astype(np.uint8)
        im=Image.fromarray(arr[0] if arr.shape[0]==1 else arr.transpose(1,2,0))
        im.save(out/f'frame_{i:03d}.png'); frames.append(im.convert('RGB'))
    frames[0].save(out/'reconstruction.gif',save_all=True,append_images=frames[1:],duration=100,loop=0)
    report={'seed':seed,'stochastic':stochastic,'final_projection':final_projection,
            'measurement_mse':(measure(x,mask)-y).square().mean().item(),
            'raw_min':float(raw.min()),'raw_max':float(raw.max())}
    if 'x' in item:
        report.update(quality(x,item['x'][None].to(device)))
    (out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)
    return report
