import math

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.cuda.amp as amp
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange
from typing import Optional
from ..modules.attention import flash_attention, WanRMSNorm, rope_apply

class WanSelfAttention_modified(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 patch_size=2,
                 block_name=None):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.block_name = block_name
        
        self.save_features = block_name is not None  # 有block_name才保存特征
        self.saved_features = None

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        
        if self.block_name is not None:
            # Only when registered with a block name, the following attr will exist.
            self.inject_kv = False
            self.copy_kv = False
        
            self.query = None
            self.key = None
            self.value = None
        

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)

        if self.block_name is not None:
            if self.inject_kv:
                k[-1:, :, :, :] = self.key[-1:, :, :, :]
                v[-1:, :, :, :] = self.value[-1:, :, :, :]
            elif self.copy_kv:
                self.key = k[-1:, :, :, :].clone()
                self.value = v[-1:, :, :, :].clone()

        x = flash_attention(
            q=q,
            k=k,
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        
        if self.save_features:
        
            f, h, w = grid_sizes[0].tolist() 
            
            self.patch_size = patch_size
            p_h = h // self.patch_size
            p_w = w // self.patch_size
            
            
            self.saved_features = rearrange(
                x[-1], "(t h w) d -> t d h w", 
                t=f, h=p_h, w=p_w
            )
        
        return x
    