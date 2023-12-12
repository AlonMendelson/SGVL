import sys
sys.path.insert(0, '/home/gamir/DER-Roei/alon/SGVL/BLIP')
'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
 * Based on timm code base
 * https://github.com/rwightman/pytorch-image-models/tree/master/timm
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath
from timm.models.helpers import named_apply, adapt_input_conv

from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper
from loralib import layers as lora_layers
import math
import functools
from operator import mul

from torch.nn.modules.utils import _pair
import collections
from itertools import repeat

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, lora = -1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        if lora == -1:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        else: 
            self.proj = lora_layers.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size, r=lora)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., lora = -1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if lora != -1:
            self.fc1 = lora_layers.Linear(in_features,hidden_features,r = lora)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        if lora != -1:
            self.fc2 = lora_layers.Linear(hidden_features,out_features,r = lora)
        else:
            self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., lora = -1, prompts_lora = -1,  objects = 0, relations = 0, prompt_attention = False, mask_attention = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        if lora != -1:
            self.qkv = lora_layers.Linear(dim, dim * 3, bias=qkv_bias, r = lora)
        else:
            self.qkv = nn.Linear(dim, dim * 3,bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        if lora != -1:
            self.proj = lora_layers.Linear(dim, dim, r = lora)
        else:
            self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.prompt_attention = prompt_attention
        self.objects = objects + relations
        if prompt_attention:
            if prompts_lora == -1:
                self.qkv_prompts = nn.Linear(dim, dim * 3,bias=qkv_bias)
                self.proj_prompts = nn.Linear(dim, dim)
            else:
                self.qkv_prompts = lora_layers.Linear(dim, dim * 3,bias=qkv_bias, r = prompts_lora)
                self.proj_prompts = lora_layers.Linear(dim, dim, r=prompts_lora)
        self.attn_gradients = None
        self.attention_map = None
        self.mask_attention = mask_attention
        if self.mask_attention:
            self.mask = []
            self.mask.append([False] + [True for i in range (self.objects)] + [False for i in range(196)])
            for i in range(objects):
                self.mask.append([False for i in range(196 + 1 + self.objects)])
            for i in range(196):
                self.mask.append([False] + [True for i in range (self.objects)] + [False for i in range(196)])
            self.mask = torch.BoolTensor(self.mask)
        
    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map
    
    def forward(self, x, register_hook=False):
        B,N,C = x.shape
        if self.prompt_attention:
            #split tokens
            patch_cls = torch.cat([x[:,0,:].unsqueeze(1),x[:,self.objects + 1 :,:]], dim = 1)
            prompts = x[: , 1: 1 + self.objects, :]
            B1, N1, C1 = patch_cls.shape
            qkv_patch_cls = self.qkv(patch_cls).reshape(B1, N1, 3, self.num_heads, C1 // self.num_heads).permute(2, 0, 3, 1, 4)
            q_patch_cls, k_patch_cls, v_patch_cls = qkv_patch_cls[0], qkv_patch_cls[1], qkv_patch_cls[2]
            q_cls, k_cls, v_cls = q_patch_cls[:,:,0,:].unsqueeze(2), k_patch_cls[:,:,0,:].unsqueeze(2), v_patch_cls[:,:,0,:].unsqueeze(2)
            q_patch, k_patch, v_patch = q_patch_cls[:,:,1:,:], k_patch_cls[:,:,1:,:], v_patch_cls[:,:,1:,:]
            B1, N1, C1 = prompts.shape
            qkv_prompts = self.qkv_prompts(prompts).reshape(B1, N1, 3, self.num_heads, C1 // self.num_heads).permute(2, 0, 3, 1, 4)
            q_prompts, k_prompts, v_prompts = qkv_prompts[0], qkv_prompts[1], qkv_prompts[2]
            q,k,v = torch.cat([q_cls,q_prompts,q_patch], dim = 2), torch.cat([k_cls,k_prompts,k_patch], dim = 2), torch.cat([v_cls,v_prompts,v_patch], dim = 2)                
        else:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.mask_attention:
            mask = self.mask.to(attn.device).unsqueeze(0).unsqueeze(0).expand(attn.shape[0],attn.shape[1],-1,-1)
            attn.masked_fill_(mask, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
                
        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.prompt_attention:
            patch_cls = torch.cat([x[:,0,:].unsqueeze(1),x[:,self.objects + 1 :,:]], dim = 1)
            prompts = x[: , 1: 1 + self.objects, :]
            patch_cls = self.proj(patch_cls)
            patch_cls = self.proj_drop(patch_cls)
            cls = patch_cls[:,0,:].unsqueeze(1)
            patch = patch_cls[:,1:,:]
            prompts = self.proj_prompts(prompts)
            prompts = self.proj_drop(prompts)
            x = torch.cat([cls,prompts,patch],dim = 1)
        else:
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_grad_checkpointing=False, lora = -1, prompts_lora = -1, objects = 0, relations=0, prompt_attention = False, prompt_attention_full = False, mask_attention=False):
        super().__init__()
        self.prompt_attention_full = prompt_attention_full
        self.norm1 = norm_layer(dim)
        if self.prompt_attention_full:
            self.norm1_prompts = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, lora = lora, prompts_lora = prompts_lora, objects = objects, relations=relations, prompt_attention = prompt_attention, mask_attention=mask_attention)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        if self.prompt_attention_full:
            self.norm2_prompts = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, lora = lora)
        if self.prompt_attention_full:
            self.mlp_prompts = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, lora = prompts_lora)
        self.objects = objects + relations


        if use_grad_checkpointing:
            self.attn = checkpoint_wrapper(self.attn)
            self.mlp = checkpoint_wrapper(self.mlp)

    def forward(self, x, register_hook=False):
        if self.prompt_attention_full:
            patch_cls = torch.cat([x[:,0,:].unsqueeze(1),x[:,self.objects + 1 :,:]], dim = 1)
            prompts = x[: , 1: 1 + self.objects, :]
            patch_cls = self.norm1(patch_cls)
            prompts = self.norm1_prompts(prompts)
            cls = patch_cls[:,0,:].unsqueeze(1)
            patch = patch_cls[:,1:,:]
            x_new = torch.cat([cls,prompts,patch],dim=1)
            x_new = self.drop_path(self.attn(x_new, register_hook=register_hook))
            x = x + x_new
            patch_cls = torch.cat([x[:,0,:].unsqueeze(1),x[:,self.objects + 1 :,:]], dim = 1)
            prompts = x[: , 1: 1 + self.objects, :]
            patch_cls = self.norm2(patch_cls)
            prompts = self.norm2_prompts(prompts)
            patch_cls = self.mlp(patch_cls)
            prompts = self.mlp_prompts(prompts)
            cls = patch_cls[:,0,:].unsqueeze(1)
            patch = patch_cls[:,1:,:]
            x_new = self.drop_path(torch.cat([cls,prompts,patch],dim=1))
            x  = x + x_new
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), register_hook=register_hook))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    
class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None, 
                 use_grad_checkpointing=False, ckpt_layer=0, lora = -1, prompts_lora = -1, objects = 0, relations=0, prompt_attention = False, prompt_attention_full = False, mask_layers=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        mask_layers_list = []
        if mask_layers != None: 
            mask_layers = mask_layers.split(",")
            for i in range(depth):
                if str(i) in mask_layers:
                    mask_layers_list.append(True)
                else:
                    mask_layers_list.append(False)
        else:
            mask_layers_list = [False] * depth
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, lora=lora)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.objects = objects
        self.relations = relations
        if self.objects > 0:
            val = math.sqrt(6. / float(3 * functools.reduce(mul, _pair(patch_size), 1) + embed_dim))  #prompt init per visual prompt tuning

            self.object_tokens = nn.Parameter(torch.zeros(1, self.objects, embed_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.object_tokens, -val, val)

        if self.relations > 0:
            val = math.sqrt(6. / float(3 * functools.reduce(mul, _pair(patch_size), 1) + embed_dim))  #prompt init per visual prompt tuning

            self.relation_tokens = nn.Parameter(torch.zeros(1, self.relations, embed_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.relation_tokens, -val, val)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                use_grad_checkpointing=(use_grad_checkpointing and i>=depth-ckpt_layer), lora = lora, prompts_lora = prompts_lora, objects = objects, relations=relations, prompt_attention = prompt_attention, prompt_attention_full=prompt_attention_full, mask_attention = mask_layers_list[i]
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) and (not isinstance(m, lora_layers.Linear)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, lora_layers.Linear):
            if isinstance(m.lora_A, nn.ParameterList):
                for i in range(len(m.lora_A)):
                    trunc_normal_(m.lora_A[i].data, std=math.sqrt(.02))
                    trunc_normal_(m.lora_B[i].data, std=math.sqrt(.02))
            else:
                trunc_normal_(m.lora_A.data, std=math.sqrt(.02))
                trunc_normal_(m.lora_B.data, std=math.sqrt(.02))

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x, register_blk=-1):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
  
        x = x + self.pos_embed[:,:x.size(1),:]
        x = self.pos_drop(x)
        if self.objects > 0:
            if self.relations > 0:
                object_tokens = self.object_tokens.expand(B, -1, -1)
                relation_tokens = self.relation_tokens.expand(B, -1, -1)

                x = torch.cat((x[:,0,:].unsqueeze(dim = 1), object_tokens, relation_tokens, x[:,1:,:]), dim=1)
            else:
                object_tokens = self.object_tokens.expand(B, -1, -1)

                x = torch.cat((x[:,0,:].unsqueeze(dim = 1), object_tokens, x[:,1:,:]), dim=1)

        for i,blk in enumerate(self.blocks):
            x = blk(x, register_blk==i)
        x = self.norm(x)
        
        return x

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)
        

@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
#     if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
#         model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
#         model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
#     if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
#         model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
#         model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))

            
def interpolate_pos_embed(pos_embed_checkpoint, visual_encoder):        
    # interpolate position embedding
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = visual_encoder.patch_embed.num_patches
    num_extra_tokens = visual_encoder.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)

    if orig_size!=new_size:
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        print('reshape position embedding from %d to %d'%(orig_size ** 2,new_size ** 2))
        
        return new_pos_embed    
    else:
        return pos_embed_checkpoint