import torch
import torch.nn as nn
import numpy as np

def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)

def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv3d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv3d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )
    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1, 0, 1)  # (pad_e_l, pad_e_r, pad_a_l, pad_a_r, pad_r_l, pad_r_r)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool3d(x, kernel_size=2, stride=2)
        return x

class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv3d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv3d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )
    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]
 
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, r, a, e = q.shape
        # reshape and permute q, k, v for attention computation
        q = q.view(b, c, -1).permute(0, 2, 1)  # [b, rae, c]
        k = k.view(b, c, -1)                   # [b, c, rae]
        w_ = torch.bmm(q, k)                   # [b, rae, rae]
        w_ = w_ * (int(c) ** -0.5)
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.view(b, c, -1)                    # [b, c, rae]
        h_ = torch.bmm(v, w_.permute(0, 2, 1))
        h_ = h_.view(b, c, r, a, e)             # [b, c, r, a, e]
        
        h_ = self.proj_out(h_)

        return x + h_

class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch=128,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        attn_resolutions=((8, 4, 2),),
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=2,  # intensity and doppler
        resolution=(128, 64, 32),  # (range, azimuth, elevation)
        z_channels=16,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        #downsampling
        self.conv_in = torch.nn.Conv3d(  
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                # curr_res = curr_res // 2
                curr_res = tuple([int(x / 2) for x in curr_res])
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d( 
            block_in,
            z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        
        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h) # swish
        h = self.conv_out(h)
        return h

class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch=128,
        out_ch = 2,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(),
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=2,
        resolution=(128, 64, 32),  # (range, azimuth, elevation)
        z_channels=16,
        give_pre_end=False,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        # curr_res = resolution // 2 ** (self.num_resolutions - 1)
        curr_res = tuple([int(x // 2 ** (self.num_resolutions - 1)) for x in resolution])
        self.z_shape = (1, z_channels, curr_res[0], curr_res[1], curr_res[2])  
        print(
            "Working with z of shape {} = {} dimensions.".format(
                self.z_shape, np.prod(self.z_shape)
            )
        )    

        # z to block_in    
        self.conv_in = torch.nn.Conv3d( 
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res *= 2
            self.up.insert(0, up) # prepend to get consistent order
        
        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d( 
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z):
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h
        
        h = self.norm_out(h)
        h = nonlinearity(h) # swish
        h = self.conv_out(h)
        return h

class RadarAutoencoder(nn.Module):
    def __init__(
        self,
        *,
        basic_channel=128,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        embed_dim=16,

    ):
        super().__init__()
        self.encoder = Encoder(ch=basic_channel,ch_mult=ch_mult,num_res_blocks=num_res_blocks, z_channels=embed_dim)
        self.decoder = Decoder(ch=basic_channel,ch_mult=ch_mult,num_res_blocks=num_res_blocks, z_channels=embed_dim)
        self.embed_dim = embed_dim


    def encode(self, x:torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return z
    
    def decode(self, z:torch.Tensor) -> torch.Tensor:
        dec = self.decoder(z)
        return dec
    
    def _encode(self, x:torch.Tensor) -> torch.Tensor:
        '''
        Called by the AR model
        '''
        inputs = x.permute(0, 4, 1, 2, 3)
        z = self.encode(inputs) # [B, embed_dim, R/d, A/d, E/d]
        return z.permute(0, 2, 3, 4, 1) # [B, R/d, A/d, E/d, embed_dim]

    def forward(self, inputs):
        '''
        Args:
            inputs (torch.Tensor): Radar Cube of shape (B, R, A, E, 2)
        '''
         # (B, R, A, E, 2) -> (B, 2, R, A, E)   
        inputs = inputs.permute(0, 4, 1, 2, 3)
        z = self.encode(inputs)
        recon = self.decode(z)      # [B, embed_dim, R/d, A/d, E/d], e.g. (B, 16, 8, 4, 2)
         # (B, 2, R, A, E) -> (B, R, A, E, 2)
        output = recon.permute(0, 2, 3, 4, 1)
        return {'pred': output, 'latent': z}
    
def create_autoencoder(
    basic_channel=128,
    ch_mult=(1, 1, 2, 2, 4),
    num_res_blocks=2,
    embed_dim=16,
):
    model = RadarAutoencoder(
        basic_channel=basic_channel,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        embed_dim=embed_dim
    )
    return model
        

def ae_ch128_mult5_n2_d16():
    return create_autoencoder(
        basic_channel=128,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        embed_dim=16,
    )

def ae_ch64_mult5_n2_d16():
    return create_autoencoder(
        basic_channel=64,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        embed_dim=16,
    )


def ae_ch16_mult5_n2_d16():
    return create_autoencoder(
        basic_channel=16,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        embed_dim=16,
    )