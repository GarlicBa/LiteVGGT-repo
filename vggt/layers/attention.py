import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from tqdm.std import tqdm
import matplotlib.pyplot as plt
import numpy as np

import transformer_engine.pytorch as te

XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        kv_group_size: int = 1,
        fused_attn: bool = True,
        rope=None,
        global_merging=None,
        patch_width: int = 37,
        patch_height: int = 28,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.fused_attn = fused_attn

        self.qkv = te.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = te.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = te.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        self.kv_group_size = kv_group_size
        self.attention_layer = te.DotProductAttention(
            num_attention_heads=num_heads,
            kv_channels= (dim // num_heads),
            attention_dropout= self.attn_drop.p if self.training else 0.0,
            attn_mask_type="no_mask",
            qkv_format = "bshd"
        )

    def forward(self, x: Tensor, pos=None, global_merging=False, m_u = None) -> Tensor:

        B, N, C = x.shape
        # (3, B, num_heads, N, head_dim)
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        ) 
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
            
        # bhnd
        B_q, H_q, N_q, D_q = q.shape

        q = q.permute(0, 2, 1, 3)  
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # bnhd
        if global_merging:

            m_a, u_a = m_u

            q_merge_in = q.reshape(B_q, N_q, H_q * D_q)
            k_merge_in = k.reshape(B_q, N_q, H_q * D_q)
            v_merge_in = v.reshape(B_q, N_q, H_q * D_q)

            q_out, k_out, v_out = m_a(
                q_merge_in,
                mode="mean",
                extra_tensors=k_merge_in,
                extra_tensors_2=v_merge_in,
            )

            del q_merge_in, k_merge_in, v_merge_in

            N_m = q_out.shape[1]
            q = q_out.reshape(B_q, N_m, H_q, D_q)
            k = k_out.reshape(B_q, N_m, H_q, D_q)
            v = v_out.reshape(B_q, N_m, H_q, D_q)

            del q_out, k_out, v_out

            N = N_m
       
        #bnhd
        x = self.attention_layer(query_layer=q,
                                 key_layer=k,
                                 value_layer=v)

        del q, k, v

        x = x.reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        if global_merging:
            x = u_a(x)

        return x


class MemEffAttention(Attention):
    def forward(
        self, x: Tensor, attn_bias=None, pos=None, global_merging=False,m_u = None
    ) -> Tensor:
        assert (
            pos is None or self.rope is not None
        ), "Position encoding is only supported with RoPE"
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x, pos=pos, global_merging=global_merging)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = qkv.unbind(2)

        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        # Use scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim**-0.5)
        if attn_bias is not None:
            attn = attn + attn_bias
        attn = F.softmax(attn, dim=-1)
        x = torch.matmul(attn, v)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x