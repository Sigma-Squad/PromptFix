import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self,*,in_channels,default_eps,force_type_convert,out_channels=None,conv_shortcut=False,dropout,temb_channels=512,):
        super(ResNetBlock,self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut
        self.force_type_convert = force_type_convert

        self.norm1 = nn.GroupNorm(num_groups=32,num_channels=self.in_channels,affine=True) if default_eps else torch.nn.GroupNorm(num_groups=32,num_channels=self.in_channels,affine=True,eps = 1e-6)
        self.conv1 = nn.Conv2d(in_channels,out_channels,3,1,1)
        self.temb_channels = temb_channels
        self.default_eps = default_eps
        
        if temb_channels >0:
            self.temb_proj = nn.Linear(temb_channels,in_channels)
        
        self.norm2 = nn.GroupNorm(num_groups=32,num_channels=self.out_channels,affine=True) if default_eps else torch.nn.GroupNorm(num_groups=32,num_channels=self.out_channels,affine=True,eps = 1e-6)
        self.droupout_percent = dropout 
        self.droupout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(self.out_channels,self.out_channels,3,1,1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(self.in_channels,self.out_channels,3,1,1)
            else:
                self.nin_shortcut = nn.Conv2d(self.in_channels,self.out_channels,1,1,0)

    def forward(self,x,temb,skip_x=None):
        h = x
        if self.force_type_convert:
            h = self.norm1.float()(h.float())
            h = h.half()
        else:
            h = self.norm1(h)

        h = h*torch.sigmoid(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(temb*torch.sigmoid(temb))[:,:,None,None]
        
        if self.force_type_convert:
            h = self.norm2.float()(h.float())
            h = h.half()
        else:
            h = self.norm2(h)
            
        h = h*torch.sigmoid(h)
        h = self.droupout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

class Encoder(nn.Module):
    def __init__(self, *,ch,out_ch,ch_mult=(1,2,4,8),num_res_blocks,attn_resolutions,default_eps=False,force_type_convert=False,dropout=0.0,
                 resample_with_conv=True,in_channels,resolution,z_channels,double_z = True,use_linear_attn=False,attn_type="vanilla",**kwargs):
        super().__init__()

        self.ch = ch
        self.temb_ch=0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.conv_in = nn.Conv2d(in_channels,self.ch,3,1,1)

        self.input_channel_mult = (1,)+tuple(ch_mult)
        
        self.down = nn.ModuleList()
        for level in range(self.num_resolutions):
            
            block = nn.ModuleList()
            attn = nn.ModuleList()

            block_in = self.ch*self.input_channel_mult[level]
            block_out = self.ch*self.input_channel_mult[level+1]
            
            for i in range(self.num_res_blocks):
                block.append(ResNetBlock(in_channels=block_in,default_eps=default_eps,force_type_convert=force_type_convert,out_channels=block_out,dropout=dropout,temb_channels=self.temb_ch))
                block_in = block_out
                
                if resolution in attn_resolutions:
                    attn.append(nn.MultiheadAttention(block_out,1,dropout=dropout)) 
            # self.down.append(nn.Conv2d(self.ch*self.input_channel_mult[level],self.ch*self.input_channel_mult[level+1],3,2,1))
        


