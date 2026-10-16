

from collections import defaultdict

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from .reconstruction_layers import *
from .mobileNetV3_cassi import Block as MBlock



class CrossSpectralFlow(nn.Module):
    def __init__(self, 
                 opt,
                 dim, 
                 num_heads, 
                 bias=False,
                 LayerNorm_type="WithBias",
                 use_prior=False,
    ):
        super().__init__()
        self.opt = opt
        self.dim = dim
        self.use_prior = use_prior
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.norm = LayerNorm(dim, LayerNorm_type=LayerNorm_type)
       

        self.qk =  DWPWConvL(dim,dim*4,2,2,0) 

        
        self.spatial_int_down = nn.Conv2d(1,1,2,2,0) 
        self.qkv_dwconv = nn.Conv2d(dim*4, dim*4, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)

        
        self.qkv_v =  nn.Sequential(nn.Conv2d(dim,dim*2, kernel_size=1, groups=dim),
                                    nn.Conv2d(dim*2, dim*2, kernel_size=(3,5), padding=(1,2),bias=False, groups=dim))
        self.qkv_v_down =  DWPWConvL(dim*2,dim*2,2,2,0)

        self.project_out =  DWPWConvTL(dim*2, dim, 4,2,1) 
        

        

    def forward(self, x, spatial_interaction=None,prior=None):
        
        b,c,h,w = x.shape
        x = self.norm(x)
        
        qk = self.qkv_dwconv(self.qk(x))
        q,k = qk.chunk(2, dim=1)
        v = self.qkv_v(x)
        
        if spatial_interaction is not None:
            spatial_interaction_down = self.spatial_int_down(spatial_interaction)
            q = q * spatial_interaction_down
            k = k * spatial_interaction_down
            v = v * spatial_interaction
            

        v_down = self.qkv_v_down(v)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads) 
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        v_down = rearrange(v_down, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        
        

        out = (attn @ v_down)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h//2, w=w//2) 

        out = self.project_out(out) 
        return out,v

class CrossPriorFlow(nn.Module):
    def __init__(self, 
                 opt,
                 dim, 
                 num_heads, 
                 bias=False,
                 LayerNorm_type="WithBias",
                 use_prior=True,
    ):
        super().__init__()
        self.opt = opt
        self.dim = dim
        self.use_prior = use_prior
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        
        self.norm_z = LayerNorm_Without_Shape(256, LayerNorm_type=LayerNorm_type)

        self.qkv_zl = nn.Linear(256,dim*4)
        self.qkv =  DWPWConvL(dim*2,dim*2,2,2,0)
            
        self.project_out = DWPWConvTL(dim*2, dim, 4,2,1)
        

        

    def forward(self, x,q, spatial_interaction=None,prior=None):
        
        b,c,h,w = x.shape
        

        kv = self.qkv_zl(prior)
        k, v = kv.chunk(2, dim=-1) 


        q = rearrange(q, 'b (head c) w h -> b head (w h) c', head=self.num_heads)
        k = rearrange(k, 'b n (head c) -> b head n c', head=self.num_heads)
        v = rearrange(v, 'b n (head c) -> b head n c', head=self.num_heads)


        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out,  'b head (h w) c-> b (head c) h w', head=self.num_heads, h=h//2, w=w//2) 

        out = self.project_out(out) 
        return out


class MBlocks(nn.Module):
    def __init__(self, 
                 in_dim, 
                 out_dim,
                 bias=False
    ):
        super(MBlocks, self).__init__()
        self.mb = MBlock(3,out_dim,out_dim*2,out_dim,nn.GELU,True,1)

    def forward(self, x):
        out = self.mb(x)
        return out


class SpatialFlow(nn.Module):
    def __init__(self,
                 dim,  
                 DW_Expand=2, 
                 bias=False,
                 LayerNorm_type="WithBias"
    ):
        super().__init__()
        self.norm = LayerNorm(dim, LayerNorm_type = LayerNorm_type)
        self.inception = MBlocks(
            in_dim=dim, 
            out_dim=dim*DW_Expand, 
            bias=bias
        )
    
    def forward(self, x):
        x = self.norm(x)
        x = self.inception(x)
        return x

    
class TridentTransformer(nn.Module):
    def __init__(self, 
                 opt,
                 dim, 
                 num_heads, 
                 use_prior = True,
                 use_ch_split = False,
    ):
        super().__init__()
        self.opt = opt
        self.use_prior = use_prior
        self.split_ch  = 2 if use_ch_split else 1
        dw_channel = dim//self.split_ch * opt.DW_Expand
        
       
        self.spatial_branch = SpatialFlow(
            dim//self.split_ch, 
            DW_Expand=opt.DW_Expand,
            bias=opt.bias,
            LayerNorm_type=opt.LayerNorm_type
        )
        self.spatial_gelu = nn.GELU()
        self.spatial_conv = nn.Conv2d(in_channels=dw_channel, out_channels=dim//self.split_ch, kernel_size=1, padding=0, stride=1, groups=1, bias=opt.bias)

    
        self.spatial_interaction = nn.Conv2d(dw_channel, 1, kernel_size=1, bias=opt.bias)

    
        self.spectral_branch = CrossSpectralFlow(
            opt,
            dim//self.split_ch, 
            num_heads=num_heads, 
            bias=opt.bias,
            LayerNorm_type=opt.LayerNorm_type,
            use_prior=use_prior,

        )
        if use_prior:
            self.HIM = CrossPriorFlow(
                opt,
                dim//self.split_ch, 
                num_heads=num_heads, 
                bias=opt.bias,
                LayerNorm_type=opt.LayerNorm_type,
                use_prior=use_prior,

            )



        self.spectral_interaction = nn.Sequential(
            nn.Conv2d(dim//self.split_ch, dim // 8, kernel_size=1, bias=opt.bias),
            LayerNorm(dim // 8, opt.LayerNorm_type),
            nn.GELU(),
            nn.Conv2d(dim // 8, dw_channel, kernel_size=1, bias=opt.bias),
        )
        self.spec_to_prior_inter = DWPWConvL(2*dim//self.split_ch, 2*dim//self.split_ch,2,2,0)
        self.add_ss = nn.Sequential(nn.Conv2d(dim,dim,1,1,0),
                               
                                 )  
        if self.split_ch>1:
            in_dim = 3*dim//self.split_ch
        else:
            in_dim = 2*dim
        self.add_ssp = nn.Sequential(nn.Conv2d(in_dim,dim,1,1,0),
          
                                 )
        self.ffn = Residual(
            FFN_FN(
                dim=dim, 
                ffn_name=opt.ffn_name,
                ffn_expansion_factor=opt.FFN_Expand, 
                bias=opt.bias,
                LayerNorm_type=opt.LayerNorm_type
            )
        )


        
    def forward(self, x,prior=None):
        log_dict = defaultdict(list)
        b, c, h, w = x.shape
        if self.split_ch>1:
            x_spa,x_spec = x.chunk(2, dim=1)
        else:
            x_spa = x
            x_spec = x

        spatial_fea = 0
        spectral_fea = 0
    

        spatial_identity = x_spa
        spatial_fea = self.spatial_branch(x_spa)
            

        spatial_interaction = None

        spatial_interaction = self.spatial_interaction(spatial_fea)
        log_dict['block_spatial_interaction'] = spatial_interaction
        

        spatial_fea = self.spatial_gelu(spatial_fea)


           
        spectral_identity = x_spec
        spectral_fea,qkv  = self.spectral_branch(x_spec, spatial_interaction)

        spectral_interaction= self.spectral_interaction(
            F.adaptive_avg_pool2d(spectral_fea, output_size=1))
        spectral_interaction = torch.sigmoid(spectral_interaction).tile((1, 1, h, w))
        spatial_fea = spectral_interaction * spatial_fea
            
        if self.use_prior:
            qkv = self.spec_to_prior_inter(qkv)
            z_fea = self.HIM(x,q=qkv,prior=prior)

            

        spatial_fea = self.spatial_conv(spatial_fea)


        spatial_fea = spatial_identity + spatial_fea
        log_dict['block_spatial_fea'] = spatial_fea

        spectral_fea = spectral_identity + spectral_fea
        
        log_dict['block_spectral_fea'] = spectral_fea
        
        if self.split_ch>1:
            fea_ss = torch.concat([spatial_fea,spectral_fea],dim=1)
        else:
            fea_ss = spatial_fea + spectral_fea
            
        fea_ss = self.add_ss(fea_ss)
        fea = torch.concat([fea_ss,z_fea],dim=1)
        fea = self.add_ssp(fea)
        
        out = self.ffn(fea)


        return out, log_dict






class PriorGuidedReconstructor(nn.Module):
    def __init__(self, opt,use_prior=False):
        super().__init__()
        self.use_prior = use_prior
 
        self.opt = opt
        self.embedding = nn.Conv2d(opt.in_dim, opt.dim, kernel_size=1, stride=1, padding=0, bias=opt.bias)
        

                
        self.Encoder = nn.ModuleList([
            
        TridentTransformer(opt = opt, dim = opt.dim * 2 ** 0, num_heads = 2 ** 0,
                          use_prior=self.use_prior,use_ch_split=False),
        TridentTransformer(opt = opt, dim = opt.dim * 2 ** 1, num_heads = 2 ** 1,
                          use_prior=self.use_prior,use_ch_split=True),
                      
        ])

        self.BottleNeck = TridentTransformer(opt = opt, dim = opt.dim * 2 ** 2, num_heads = 2 ** 2,
                                            use_prior=self.use_prior,use_ch_split=True)
                      
        self.Decoder = nn.ModuleList([

                    TridentTransformer(opt = opt, dim = opt.dim * 2 ** 1, num_heads = 2 ** 1,
                                      use_prior=self.use_prior,use_ch_split=True),
              TridentTransformer(opt = opt, dim = opt.dim * 2 ** 0, num_heads = 2 ** 0,
                          use_prior=self.use_prior,use_ch_split=False)
                      
        
        ])
                


        self.BlockInteractions = nn.ModuleList([
            BlockInteraction(opt.dim * 7, opt.dim * 1),
            BlockInteraction(opt.dim * 7, opt.dim * 2)
        ])

        self.Downs = nn.ModuleList([
            DownSample(opt.dim * 2 ** 0, bias=opt.bias),
            DownSample(opt.dim * 2 ** 1, bias=opt.bias)
        ])

        self.Ups = nn.ModuleList([
            UpSample(opt.dim * 2 ** 2, bias=opt.bias),
            UpSample(opt.dim * 2 ** 1, bias=opt.bias)
        ])

        self.fusions = nn.ModuleList([
            nn.Conv2d(
                in_channels = opt.dim * 2 ** 2,
                out_channels = opt.dim * 2 ** 1,
                kernel_size = 3,
                stride = 1,
                padding = 1,
                bias = opt.bias
            ),
            nn.Conv2d(
                in_channels = opt.dim * 2 ** 1,
                out_channels = opt.dim * 2 ** 0,
                kernel_size = 3,
                stride = 1,
                padding = 1,
                bias = opt.bias
            )
        ])

 
        self.stage_interactions = nn.ModuleList([
            StageInteraction(dim = opt.dim * 2 ** 0, act_fn_name=opt.act_fn_name, bias=opt.bias),
            StageInteraction(dim = opt.dim * 2 ** 1, act_fn_name=opt.act_fn_name, bias=opt.bias),
            StageInteraction(dim = opt.dim * 2 ** 2, act_fn_name=opt.act_fn_name, bias=opt.bias),
            StageInteraction(dim = opt.dim * 2 ** 1, act_fn_name=opt.act_fn_name, bias=opt.bias),
            StageInteraction(dim = opt.dim * 2 ** 0, act_fn_name=opt.act_fn_name, bias=opt.bias)
        ])


        self.mapping = nn.Conv2d(opt.dim, opt.out_dim, kernel_size=1, stride=1, padding=0, bias=opt.bias)

    def forward(self, x, enc_outputs=None, bottleneck_out=None, dec_outputs=None
                ,prior=None):
        b, c, h_inp, w_inp = x.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        enc_outputs_l = []
        dec_outputs_l = []
        x1 = self.embedding(x)

        res1, log_dict1 = self.Encoder[0](x1,prior=prior[0])

        
        if (enc_outputs is not None) and (dec_outputs is not None):
            res1 = self.stage_interactions[0](res1, enc_outputs[0], dec_outputs[0])
        res12 = F.interpolate(res1, scale_factor=0.5, mode='bilinear') 

        x2 = self.Downs[0](res1)

        res2, log_dict2 = self.Encoder[1](x2,prior=prior[1])

        if (enc_outputs is not None) and (dec_outputs is not None):
            res2 = self.stage_interactions[1](res2, enc_outputs[1], dec_outputs[1])
        res21 = F.interpolate(res2, scale_factor=2, mode='bilinear') 


        x4 = self.Downs[1](res2)
 
        res4, log_dict3 = self.BottleNeck(x4,prior=prior[2])
 
        
        
        
        if bottleneck_out is not None:
            res4 = self.stage_interactions[2](res4, bottleneck_out, bottleneck_out)

        res42 = F.interpolate(res4, scale_factor=2, mode='bilinear') 
        res41 = F.interpolate(res42, scale_factor=2, mode='bilinear') 
       
   
        res1 = self.BlockInteractions[0](res1, res21, res41) 
        res2 = self.BlockInteractions[1](res12, res2, res42) 
        enc_outputs_l.append(res1)
        enc_outputs_l.append(res2)
        

        dec_res2 = self.Ups[0](res4) 
        dec_res2 = torch.cat([dec_res2, res2], dim=1) 
        dec_res2 = self.fusions[0](dec_res2) 
 
        dec_res2, log_dict4 = self.Decoder[0](dec_res2,prior=prior[1])
 
        
        
        
        if (enc_outputs is not None) and (dec_outputs is not None):
            dec_res2 = self.stage_interactions[3](dec_res2, enc_outputs[1], dec_outputs[1])
        

        dec_res1 = self.Ups[1](dec_res2) 
        dec_res1 = torch.cat([dec_res1, res1], dim=1) 
        dec_res1 = self.fusions[1](dec_res1) 
        
 
        dec_res1, log_dict5 = self.Decoder[1](dec_res1,prior=prior[0])
 
        
        if (enc_outputs is not None) and (dec_outputs is not None):
            dec_res1 = self.stage_interactions[4](dec_res1, enc_outputs[0], dec_outputs[0])

        dec_outputs_l.append(dec_res1)
        dec_outputs_l.append(dec_res2)

        out = self.mapping(dec_res1) + x

        return out[:, :, :h_inp, :w_inp], enc_outputs_l, res4, dec_outputs_l


