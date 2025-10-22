# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import drop_path, to_2tuple
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.models.registry import register_model

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0., attn_head_dim=None, use_rpb=False, window_size=14):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        # relative positional bias option
        self.use_rpb = use_rpb
        if use_rpb:
            self.window_size = window_size
            self.rpb_table = nn.Parameter(torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
            trunc_normal_(self.rpb_table, std=.02)

            coords_h = torch.arange(window_size)
            coords_w = torch.arange(window_size)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, h, w
            coords_flatten = torch.flatten(coords, 1)  # 2, h*w
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, h*w, h*w
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # h*w, h*w, 2
            relative_coords[:, :, 0] += window_size - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size - 1
            relative_coords[:, :, 0] *= 2 * window_size - 1
            relative_position_index = relative_coords.sum(-1)  # h*w, h*w
            self.register_buffer("relative_position_index", relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.use_rpb:
            relative_position_bias = self.rpb_table[self.relative_position_index.view(-1)].view(
                self.window_size * self.window_size, self.window_size * self.window_size, -1)  # h*w,h*w,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, h*w, h*w
            attn += relative_position_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None, use_rpb=False, window_size=14):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim,
            use_rpb=use_rpb, window_size=window_size)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class Block1(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None, use_rpb=False, window_size=14):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim,
            use_rpb=use_rpb, window_size=window_size)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.norm3 = norm_layer(dim)
        self.conv_layer = Cnn_block1(dim=dim, k=3, window_size=window_size)
        # self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.norm4 = norm_layer(dim)
        
        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_3 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            # self.gamma_4 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2, self.gamma_3 = None, None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp1(self.norm2(x)))
            x = x + self.drop_path(self.conv_layer(x))
            # x = x + self.drop_path(self.mlp2(self.norm4(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp1(self.norm2(x)))
            x = x + self.drop_path(self.gamma_3 * self.conv_layer(x))
            # x = x + self.drop_path(self.gamma_4 *self.mlp2(self.norm4(x)))
        return x

class Block2(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None, use_rpb=False, window_size=14):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim,
            use_rpb=use_rpb, window_size=window_size)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.norm3 = norm_layer(dim)
        self.conv_layer = Cnn_block(dim=dim, k=3, window_size=window_size)
        # self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.norm4 = norm_layer(dim)
        
        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_3 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            # self.gamma_4 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2, self.gamma_3 = None, None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp1(self.norm2(x)))
            x = x + self.drop_path(self.conv_layer(x))
            # x = x + self.drop_path(self.mlp2(self.norm4(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp1(self.norm2(x)))
            x = x + self.drop_path(self.gamma_3 *self.conv_layer(x))
            # x = x + self.drop_path(self.gamma_4 *self.mlp2(self.norm4(x)))
        return x

class Block3(nn.Module):

    def __init__(self, dim, num_heads, in_channels, hidden_channels, patchsize, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None, use_rpb=False, window_size=14):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim,
            use_rpb=use_rpb, window_size=window_size)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.norm3 = norm_layer(dim)
        self.conv_layer = Cnn_block2(in_channels=in_channels, hidden_channels=hidden_channels, window_size=window_size, patch_size=patchsize)
        # self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.norm4 = norm_layer(dim)
        
        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_3 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_4 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp1(self.norm2(x)))
            x = x + self.drop_path(self.conv_layer(x))
            # x = x + self.drop_path(self.conv_layer(self.norm3(x)))
            # x = x + self.drop_path(self.mlp2(self.norm4(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            x = x + self.drop_path(self.gamma_3 *self.conv_layer(self.norm3(x)))
            x = x + self.drop_path(self.gamma_4 *self.mlp2(self.norm4(x)))
        return x

class Block4(nn.Module):

    def __init__(self, dim, num_heads, in_channels, patchsize, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None, use_rpb=False, window_size=14):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim,
            use_rpb=use_rpb, window_size=window_size)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.norm3 = norm_layer(dim)
        self.conv_layer = Cnn_block3(in_channels=in_channels, window_size=window_size, patch_size=patchsize)
        # self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.norm4 = norm_layer(dim)
        
        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_3 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            # self.gamma_4 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp1(self.norm2(x)))
            x = x + self.drop_path(self.conv_layer(x))
            # x = x + self.drop_path(self.mlp2(self.norm4(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            x = x + self.drop_path(self.gamma_3 *self.conv_layer(x))
            # x = x + self.drop_path(self.gamma_4 *self.mlp2(self.norm4(x)))
        return x

class Block5(nn.Module):

    def __init__(self, dim, num_heads, in_channels, patchsize, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None, use_rpb=False, window_size=14):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim,
            use_rpb=use_rpb, window_size=window_size)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.norm3 = norm_layer(dim)
        self.conv_layer = Cnn_block3(in_channels=in_channels, window_size=window_size, patch_size=patchsize)
        self.conv_layer2 = Cnn_block3(in_channels=in_channels, window_size=window_size, patch_size=patchsize)
        # self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.norm4 = norm_layer(dim)
        
        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_3 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_4 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2, self.gamma_3, self.gamma_4 = None, None, None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.conv_layer(x))
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            x = x + self.drop_path(self.conv_layer2(x))
            # x = x + self.drop_path(self.mlp2(self.norm4(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.conv_layer(x))
            x = x + self.drop_path(self.gamma_2 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_3 * self.mlp(self.norm2(x)))
            x = x + self.drop_path(self.gamma_4 * self.conv_layer2(x))
            # x = x + self.drop_path(self.gamma_4 *self.mlp2(self.norm4(x)))
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, mask_cent=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.mask_cent = mask_cent

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        if self.mask_cent:
            x[:, -1] = x[:, -1] - 0.5
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class PatchEmbed2(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, overlap=1, mask_cent=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        kernel = patch_size + 2 * overlap
        kernel = to_2tuple(kernel)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.mask_cent = mask_cent

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel, stride=patch_size, padding=overlap)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        if self.mask_cent:
            x[:, -1] = x[:, -1] - 0.5
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31


def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''
    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

##################################### Colorization #################################


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class CnnHead(nn.Module):
    def __init__(self, embed_dim, num_classes, window_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.window_size = window_size

        self.head = nn.Conv2d(embed_dim, num_classes, kernel_size=3, stride=1, padding=1, padding_mode='reflect')

    def forward(self, x):
        x = rearrange(x, 'b (p1 p2) c -> b c p1 p2', p1=self.window_size, p2=self.window_size)
        x = self.head(x)
        x = rearrange(x, 'b c p1 p2 -> b (p1 p2) c')
        return x

class CnnHead2(nn.Module):
    def __init__(self, embed_dim, patch_size, window_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.window_size = window_size

        self.head = nn.Conv2d(embed_dim // (patch_size ** 2), 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = rearrange(x, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=self.patch_size, p2=self.patch_size)
        x = rearrange(x, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', h=self.window_size, w=self.window_size, p1=self.patch_size, p2=self.patch_size)
        x = self.head(x)
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', h=self.window_size, w=self.window_size, p1=self.patch_size, p2=self.patch_size)
        x = rearrange(x, 'b n (p1 p2) c -> b n (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        return x

class CnnHead3(nn.Module):
    def __init__(self, embed_dim, patch_size, window_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.window_size = window_size

        self.head = nn.Sequential(
                        nn.LayerNorm(window_size * patch_size, eps=1e-6, elementwise_affine=True),
                        nn.Conv2d(embed_dim // (patch_size ** 2), 2 * embed_dim // (patch_size ** 2), stride=1, padding=1),
                        nn.GELU(),
                        nn.LayerNorm(window_size * patch_size, eps=1e-6, elementwise_affine=True),
                        nn.Conv2d(2 * embed_dim // (patch_size ** 2), 2, 3, stride=1, padding=1),
                        nn.GELU()
                        )

    def forward(self, x):
        x = rearrange(x, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=self.patch_size, p2=self.patch_size)
        x = rearrange(x, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', h=self.window_size, w=self.window_size, p1=self.patch_size, p2=self.patch_size)
        x = self.head(x)
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', h=self.window_size, w=self.window_size, p1=self.patch_size, p2=self.patch_size)
        x = rearrange(x, 'b n (p1 p2) c -> b n (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        return x

class UnetHead(nn.Module):
    def __init__(self, embed_dim, patch_size, window_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.window_size = window_size

        self.head = nn.Conv2d(embed_dim // (patch_size ** 2), 2, kernel_size=3, stride=1, padding=1)
        self.head = Unet_module(embed_dim // (patch_size ** 2), 2, patch_size * window_size)

    def forward(self, x):
        x = rearrange(x, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=self.patch_size, p2=self.patch_size)
        x = rearrange(x, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', h=self.window_size, w=self.window_size, p1=self.patch_size, p2=self.patch_size)
        x = self.head(x)
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', h=self.window_size, w=self.window_size, p1=self.patch_size, p2=self.patch_size)
        x = rearrange(x, 'b n (p1 p2) c -> b n (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        return x

class ChannelAtten(nn.Module):
    def __init__(self, in_features):
        super(ChannelAtten, self).__init__()

        channel_block = [ 
                          nn.Conv2d(in_features, in_features // 2, 1),
                          nn.GELU(),
                          nn.Conv2d(in_features // 2, in_features, 1),
                          nn.Sigmoid()
                        ]
        
        self.channel_block = nn.Sequential(*channel_block)

    def forward(self, x):
        base = F.adaptive_avg_pool2d(x, (1, 1))
        base = base.view(base.size(0), base.size(1), 1, 1)
        base = self.channel_block(base).expand(base.size(0), base.size(1), x.size(2), x.size(3))
        out = torch.mul(x, base)

        return out

class Cnn_block(nn.Module):
    def __init__(self, dim, k, window_size):
        super().__init__()

        conv_block1 = [ nn.LayerNorm(window_size, eps=1e-6, elementwise_affine=True),
                        nn.Conv2d(dim, dim, 1),
                        nn.Conv2d(dim, dim, k, stride=1, padding=1, groups=dim, bias=False),
                        nn.GELU(),
                        ChannelAtten(dim),
                        nn.Conv2d(dim, dim, 1)
        ]

        conv_block2 = [ nn.LayerNorm(window_size, eps=1e-6, elementwise_affine=True),
                        nn.Conv2d(dim, dim, 1),
                        nn.GELU(),
                        nn.Conv2d(dim, dim, 1)
        ]

        self.conv_block1 = nn.Sequential(*conv_block1)
        self.conv_block2 = nn.Sequential(*conv_block2)
        self.window_size = window_size

    def forward(self, x):
        x = rearrange(x, 'b (p1 p2) c -> b c p1 p2', p1=self.window_size, p2=self.window_size)
        out1 = self.conv_block1(x) + x
        out = self.conv_block2(out1) + out1
        out = rearrange(out, 'b c p1 p2 -> b (p1 p2) c')
        return out

class Cnn_block1(nn.Module):
    def __init__(self, dim, k, window_size):
        super().__init__()

        conv_block = [ nn.LayerNorm(window_size, eps=1e-6, elementwise_affine=True),
                        nn.Conv2d(dim, dim, k, stride=1, padding=1, groups=dim, bias=False),
                        nn.GELU(),
                        # nn.LayerNorm(window_size, eps=1e-6, elementwise_affine=True),
                        # nn.Conv2d(dim, dim, k, stride=1, padding=1, groups=dim, bias=False),
                        # nn.GELU()
        ]

        self.conv_block = nn.Sequential(*conv_block)
        self.window_size = window_size

    def forward(self, x):
        x = rearrange(x, 'b (p1 p2) c -> b c p1 p2', p1=self.window_size, p2=self.window_size)
        out = self.conv_block(x)
        out = rearrange(out, 'b c p1 p2 -> b (p1 p2) c')
        return out

class Cnn_block2(nn.Module):
    def __init__(self, in_channels, hidden_channels, window_size, patch_size):
        super().__init__()

        initial_conv = [nn.Conv2d(in_channels, hidden_channels, 3, stride=1, padding=1),
                        nn.GELU()
        ]

        conv_block1 = [ nn.LayerNorm(window_size * patch_size, eps=1e-6, elementwise_affine=True),
                        nn.Conv2d(hidden_channels, hidden_channels, 1),
                        nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1, groups=hidden_channels, bias=False),
                        nn.GELU(),
                        ChannelAtten(hidden_channels),
                        nn.Conv2d(hidden_channels, hidden_channels, 1)
        ]

        conv_block2 = [ nn.LayerNorm(window_size * patch_size, eps=1e-6, elementwise_affine=True),
                        nn.Conv2d(hidden_channels, hidden_channels, 1),
                        nn.GELU(),
                        nn.Conv2d(hidden_channels, hidden_channels, 1)
        ]

        output_conv = [nn.Conv2d(hidden_channels, in_channels, 3, stride=1, padding=1),
                        nn.GELU()
        ]

        self.initial_conv = nn.Sequential(*initial_conv)
        self.conv_block1 = nn.Sequential(*conv_block1)
        self.conv_block2 = nn.Sequential(*conv_block2)
        self.output_conv = nn.Sequential(*output_conv)
        self.window_size = window_size
        self.patch_size = patch_size

    def forward(self, x):
        x = rearrange(x, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=self.patch_size, p2=self.patch_size)
        x = rearrange(x, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', h=self.window_size, w=self.window_size, p1=self.patch_size, p2=self.patch_size)
        x = self.initial_conv(x)
        x = self.conv_block1(x) + x
        x = self.conv_block2(x) + x
        x = self.output_conv(x)
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', h=self.window_size, w=self.window_size, p1=self.patch_size, p2=self.patch_size)
        x = rearrange(x, 'b n (p1 p2) c -> b n (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        return x

class Cnn_block3(nn.Module):
    def __init__(self, in_channels, window_size, patch_size):
        super().__init__()

        conv_block = [  nn.LayerNorm(window_size * patch_size, eps=1e-6, elementwise_affine=True),
                        nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1),
                        nn.GELU(),
                        nn.LayerNorm(window_size * patch_size, eps=1e-6, elementwise_affine=True),
                        nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1),
                        nn.GELU()
        ]

        self.conv_block = nn.Sequential(*conv_block)
        self.window_size = window_size
        self.patch_size = patch_size

    def forward(self, x):
        x = rearrange(x, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=self.patch_size, p2=self.patch_size)
        x = rearrange(x, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', h=self.window_size, w=self.window_size, p1=self.patch_size, p2=self.patch_size)
        x = self.conv_block(x)
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', h=self.window_size, w=self.window_size, p1=self.patch_size, p2=self.patch_size)
        x = rearrange(x, 'b n (p1 p2) c -> b n (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        return x

class Fusion_block(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channnels, input_size):
        super().__init__()

        initial_conv = [nn.Conv2d(in_channels, hidden_channels, 3, stride=1, padding=1),
                        nn.GELU()
        ]

        conv_block1 = [ nn.LayerNorm(input_size, eps=1e-6, elementwise_affine=True),
                        nn.Conv2d(hidden_channels, hidden_channels, 1),
                        nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1, groups=hidden_channels, bias=False),
                        nn.GELU(),
                        ChannelAtten(hidden_channels),
                        nn.Conv2d(hidden_channels, hidden_channels, 1)
        ]

        conv_block2 = [ nn.LayerNorm(input_size, eps=1e-6, elementwise_affine=True),
                        nn.Conv2d(hidden_channels, hidden_channels, 1),
                        nn.GELU(),
                        nn.Conv2d(hidden_channels, hidden_channels, 1)
        ]

        output_conv = [nn.Conv2d(hidden_channels, out_channnels, 3, stride=1, padding=1),
                        nn.GELU()
        ]

        self.initial_conv = nn.Sequential(*initial_conv)
        self.conv_block1 = nn.Sequential(*conv_block1)
        self.conv_block2 = nn.Sequential(*conv_block2)
        self.output_conv = nn.Sequential(*output_conv)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.conv_block1(x) + x
        x = self.conv_block2(x) + x
        x = self.output_conv(x)
        return x

class LocalAttentionHead(nn.Module):
    def __init__(
            self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None, use_rpb=False, window_size=14):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        # masking attn
        mask = torch.ones((window_size**2, window_size**2))
        kernel_size = 3
        for i in range(window_size):
            for j in range(window_size):
                cur_map = torch.ones((window_size, window_size))
                stx, sty = max(i - kernel_size // 2, 0), max(j - kernel_size // 2, 0)
                edx, edy = min(i + kernel_size // 2, window_size - 1), min(j + kernel_size // 2, window_size - 1)
                cur_map[stx:edx + 1, sty:edy + 1] = 0
                cur_map = cur_map.flatten()
                mask[i * window_size + j] = cur_map
        self.register_buffer('mask', mask)

        # relative positional bias option
        self.use_rpb = use_rpb
        if use_rpb:
            self.window_size = window_size
            self.rpb_table = nn.Parameter(torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
            trunc_normal_(self.rpb_table, std=.02)

            coords_h = torch.arange(window_size)
            coords_w = torch.arange(window_size)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, h, w
            coords_flatten = torch.flatten(coords, 1)  # 2, h*w
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, h*w, h*w
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # h*w, h*w, 2
            relative_coords[:, :, 0] += window_size - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size - 1
            relative_coords[:, :, 0] *= 2 * window_size - 1
            relative_position_index = relative_coords.sum(-1)  # h*w, h*w
            self.register_buffer("relative_position_index", relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # masking attn
        mask_value = max_neg_value(attn)
        attn.masked_fill_(self.mask.bool(), mask_value)

        if self.use_rpb:
            relative_position_bias = self.rpb_table[self.relative_position_index.view(-1)].view(
                self.window_size * self.window_size, self.window_size * self.window_size, -1)  # h*w,h*w,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, h*w, h*w
            attn += relative_position_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class IReWindColor(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, out_chans=2, embed_dim=512, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_rpb=False, avg_hint=False, head_mode='default', mask_cent=False):
        super().__init__()
        self.num_classes = out_chans * patch_size ** 2
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.avg_hint = avg_hint
        self.window_size = img_size // patch_size

        # self.mask_token = nn.Parameter(torch.zeros(2))
        # trunc_normal_(self.mask_token, std=.02)

        self.patch_embed = PatchEmbed(img_size=img_size // 2, patch_size=patch_size // 2,
                                      in_chans=in_chans, embed_dim=embed_dim // 4, mask_cent=mask_cent)
        self.patch_embed2 = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim, mask_cent=mask_cent)
        num_patches = self.patch_embed.num_patches  # 2

        self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim // 4)
        self.pos_embed2 = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth // 3)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([Block1(
            dim=embed_dim // 4, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
            init_values=init_values, use_rpb=use_rpb, window_size=img_size // patch_size)
            for i in range(depth // 3)])
        
        dpr2 = [x.item() for x in torch.linspace(0, drop_path_rate, 2 * depth // 3)]  # stochastic depth decay rule
        self.blocks2 = nn.ModuleList([Block1(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr2[i], norm_layer=norm_layer,
            init_values=init_values, use_rpb=use_rpb, window_size=img_size // patch_size)
            for i in range(2 * depth // 3)])

        self.norm = norm_layer(embed_dim // 4)
        self.norm2 = norm_layer(embed_dim)

        self.merge_patch = nn.Conv2d(embed_dim // 4, embed_dim, 2, 2)

        if head_mode == 'linear':
            self.head = nn.Linear(embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        elif head_mode == 'cnn':
            self.head = CnnHead(embed_dim, self.num_classes, window_size=img_size // patch_size)
        elif head_mode == 'locattn':
            self.head = LocalAttentionHead(embed_dim, self.num_classes, window_size=img_size // patch_size)
        else:
            raise NotImplementedError('Check head type')

        self.tanh = nn.Tanh()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()  # (B, 14*14, 768)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = rearrange(x, 'b (p1 p2) c -> b c p1 p2', p1=self.window_size, p2=self.window_size)
        return x

    def forward(self, x, mask):
        # mask is 1D of 2D if 2D
        B, _, H, W = x.shape
        if mask.dim() == 2:
            _, L = mask.shape
            # assume square inputs
            hint_size = int(math.sqrt(H * W // L))
            _device = '.cuda' if x.device.type == 'cuda' else ''

            # hint location = 0, non-hint location = 1
            mask = torch.reshape(mask, (B, H // hint_size, W // hint_size))
            _mask = mask.unsqueeze(1).type(f'torch{_device}.FloatTensor')
            _full_mask = F.interpolate(_mask, scale_factor=hint_size)  # Needs to be Float
            full_mask = _full_mask.type(f'torch{_device}.BoolTensor')

            # mask ab channels
            if self.avg_hint:
                _avg_x = F.interpolate(x, size=(H // hint_size, W // hint_size), mode='bilinear')
                _avg_x[:, 1, :, :].masked_fill_(mask.squeeze(1), 0)
                _avg_x[:, 2, :, :].masked_fill_(mask.squeeze(1), 0)
                x_ab = F.interpolate(_avg_x, scale_factor=hint_size, mode='nearest')[:, 1:, :, :]
                x = torch.cat((x[:, 0, :, :].unsqueeze(1), x_ab), dim=1)
            else:
                x[:, 1, :, :].masked_fill_(full_mask.squeeze(1), 0)
                x[:, 2, :, :].masked_fill_(full_mask.squeeze(1), 0)
        elif mask.dim() == 4:  # (B, 3, H, W) Zhang mode. channel 0,1 is ab. channel2 is mask
            # TODO ONLY FOR DEMO
            _device = '.cuda' if x.device.type == 'cuda' else ''
            _full_mask_ab = mask.type(f'torch{_device}.FloatTensor')[:, :2, :, :]
            x[:, 1:, :, :] = _full_mask_ab
            _full_mask = mask[:, 2, :, :].unsqueeze(1).type(f'torch{_device}.FloatTensor')
        else:
            raise NotImplementedError('Check the mask dimension')

        if self.in_chans == 4:
            x = torch.cat((x, 1 - _full_mask), dim=1)
        
        short_cut = self.patch_embed2(x)
        short_cut = short_cut + self.pos_embed2.type_as(short_cut).to(x.device).clone().detach()
        x_l, x_r = torch.chunk(x, 2, dim=2)
        x_1, x_2 = torch.chunk(x_l, 2, dim=3)
        x_3, x_4 = torch.chunk(x_r, 2, dim=3)
        x_1 = self.forward_features(x_1)
        x_2 = self.forward_features(x_2)
        x_3 = self.forward_features(x_3)
        x_4 = self.forward_features(x_4)
        x_l = torch.cat((x_1, x_2), dim=3)
        x_r = torch.cat((x_3, x_4), dim=3)
        x = torch.cat((x_l, x_r), dim=2)
        x = self.merge_patch(x)
        x = rearrange(x, 'b c p1 p2 -> b (p1 p2) c')
        x = x + short_cut
        for blk in self.blocks2:
            x = blk(x)
        x = self.norm2(x)
        x = self.head(x)
        x = self.tanh(x)
        # x = rearrange(x, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=self.patch_size, p2=self.patch_size)
        # x = rearrange(x, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', h=self.window_size, w=self.window_size, p1=self.patch_size, p2=self.patch_size)        
        return x

@register_model
def irewindcolor(pretrained=False, **kwargs):
    model = IReWindColor(
        out_chans=2,
        img_size=224,
        patch_size=16,
        in_chans=4,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model