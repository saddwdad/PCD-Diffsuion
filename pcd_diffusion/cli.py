import argparse
import json
from pathlib import Path
from .data import synthetic, prepare_frames
from .engine import train, infer, load_checkpoint, evaluate_model, device_for


def main():
    p=argparse.ArgumentParser(description='Independent PCD-Diffusion implementation')
    sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('make-demo',help='Generate reproducible moving-object SCI samples')
    s.add_argument('--output',required=True); s.add_argument('--count',type=int,default=8)
    s.add_argument('--frames',type=int,default=4); s.add_argument('--size',type=int,default=24)
    s.add_argument('--channels',type=int,choices=[1,3],default=1); s.add_argument('--seed',type=int,default=7)
    s=sub.add_parser('prepare',help='Convert frame folders (including DAVIS) to NPZ')
    s.add_argument('--root',required=True); s.add_argument('--output',required=True)
    s.add_argument('--split-file'); s.add_argument('--frames',type=int,default=8)
    s.add_argument('--size',type=int,default=256); s.add_argument('--channels',type=int,choices=[1,3],default=1)
    s.add_argument('--seed',type=int,default=7)
    s=sub.add_parser('train')
    s.add_argument('--config',required=True); s.add_argument('--data',required=True); s.add_argument('--output',required=True)
    s.add_argument('--device',default='auto'); s.add_argument('--val'); s.add_argument('--resume'); s.add_argument('--epochs',type=int)
    s.add_argument('--mask',help='Fixed MATLAB mask for fixed_mask_video data')
    s.add_argument('--split-file',help='DAVIS sequence-name split file')
    s=sub.add_parser('infer')
    s.add_argument('--checkpoint',required=True); s.add_argument('--input',required=True); s.add_argument('--output',required=True)
    s.add_argument('--device',default='auto'); s.add_argument('--seed',type=int,default=7)
    s.add_argument('--final-projection',action='store_true'); s.add_argument('--stochastic',action='store_true')
    s=sub.add_parser('evaluate')
    s.add_argument('--checkpoint',required=True); s.add_argument('--data',required=True); s.add_argument('--output',required=True)
    s.add_argument('--device',default='auto'); s.add_argument('--seed',type=int,default=2026)
    s.add_argument('--final-projection',action='store_true')
    s=sub.add_parser('convert-mat',help='Explicit MAT axes; no guessing/scaling measurement independently')
    s.add_argument('--input',required=True); s.add_argument('--output',required=True)
    s.add_argument('--x-key',default='orig'); s.add_argument('--mask-key',default='mask'); s.add_argument('--y-key')
    s.add_argument('--x-layout',choices=['HWN','NHW','NHWC','NCHW'],default='HWN')
    s.add_argument('--mask-layout',choices=['HWN','NHW'],default='HWN'); s.add_argument('--scale',type=float,default=255)
    s=sub.add_parser('fetch-data',aliases=['fetch-benchmark-assets'],help='Fetch 7 pinned mask/benchmark assets with SHA256 checks')
    s.add_argument('--output',required=True)
    s=sub.add_parser('prepare-six-gray',aliases=['prepare-grayscale-benchmark'],help='Prepare all 6 official grayscale sequences using the fixed mask')
    s.add_argument('--root',default='',help='Local directory containing six MAT videos'); s.add_argument('--output',required=True); s.add_argument('--mask',default='')
    s=sub.add_parser('evaluate-six-gray',aliases=['evaluate-grayscale-benchmark'],help='Official-style raw-output per-frame/six-sequence evaluation')
    s.add_argument('--checkpoint',required=True); s.add_argument('--root',default='',help='Local directory containing six MAT videos')
    s.add_argument('--output',required=True); s.add_argument('--mask',default=''); s.add_argument('--save-recon')
    s.add_argument('--device',default='auto'); s.add_argument('--seed',type=int,default=2026)
    args=p.parse_args()
    if args.command=='make-demo':
        if args.count<1 or args.frames<1 or args.size<4: p.error('count/frames >= 1 and size >= 4 required')
        synthetic(args.output,args.count,args.frames,args.size,args.channels,args.seed)
        print(f'Wrote {args.count} samples to {args.output}')
    elif args.command=='prepare':
        print(f'Wrote {prepare_frames(args.root,args.output,args.frames,args.size,args.channels,args.seed,args.split_file)} samples')
    elif args.command=='train':
        train(args.config,args.data,args.output,args.device,args.val,args.resume,args.epochs,args.mask,args.split_file)
    elif args.command=='infer':
        infer(args.checkpoint,args.input,args.output,args.device,args.seed,args.final_projection,args.stochastic)
    elif args.command=='evaluate':
        import torch
        torch.set_num_threads(4)
        device=device_for(args.device); model,_=load_checkpoint(args.checkpoint,device)
        report=evaluate_model(model,args.data,device,args.seed,args.final_projection)
        path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report['mean']))
    elif args.command in ('fetch-data','fetch-benchmark-assets'):
        from .fixed_mask_data import fetch_assets
        fetch_assets(args.output)
    elif args.command in ('prepare-six-gray','prepare-grayscale-benchmark'):
        from .fixed_mask_data import prepare_six_gray
        print(json.dumps(prepare_six_gray(args.root,args.output,args.mask),indent=2))
    elif args.command in ('evaluate-six-gray','evaluate-grayscale-benchmark'):
        import torch
        from .fixed_mask_data import evaluate_six_gray
        torch.set_num_threads(4)
        device=device_for(args.device); model,_=load_checkpoint(args.checkpoint,device)
        report=evaluate_six_gray(model,args.root,device,args.seed,args.mask,args.save_recon)
        report['checkpoint']=str(Path(args.checkpoint).resolve())
        path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report['mean']))
    else:
        import numpy as np
        from scipy.io import loadmat
        from .data import load_sample
        if args.scale<=0: p.error('scale must be positive')
        mat=loadmat(args.input)
        x=np.asarray(mat[args.x_key],dtype=np.float32)/args.scale
        if args.x_layout=='HWN': x=x.transpose(2,0,1)[:,None]
        elif args.x_layout=='NHW': x=x[:,None]
        elif args.x_layout=='NHWC': x=x.transpose(0,3,1,2)
        mask=np.asarray(mat[args.mask_key],dtype=np.float32)
        if args.mask_layout=='HWN': mask=mask.transpose(2,0,1)
        values={'x':x,'mask':mask}
        if args.y_key:
            y=np.asarray(mat[args.y_key],dtype=np.float32)/args.scale
            values['y']=y[None] if y.ndim==2 else y.transpose(2,0,1)
        out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out,**values)
        load_sample(out,require_target=True)
        print(f'Converted and validated {out}')


if __name__=='__main__': main()
