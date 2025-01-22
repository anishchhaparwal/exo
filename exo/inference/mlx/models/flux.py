# Copyright © 2024 Apple Inc.

from dataclasses import dataclass
from typing import List, Optional, Tuple
import mlx.core as mx
import mlx.nn as nn
import math

@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool

class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = nn.RMSNorm(dim)
        self.key_norm = nn.RMSNorm(dim)

    def __call__(self, q: mx.array, k: mx.array) -> tuple[mx.array, mx.array]:
        return self.query_norm(q), self.key_norm(k)

class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x: mx.array, pe: mx.array) -> mx.array:
        H = self.num_heads
        B, L, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = q.reshape(B, L, H, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, H, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, H, -1).transpose(0, 2, 1, 3)
        q, k = self.norm(q, k)
        x = _attention(q, k, v, pe)
        x = self.proj(x)
        return x

@dataclass
class ModulationOut:
    shift: mx.array
    scale: mx.array
    gate: mx.array

class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def __call__(self, x: mx.array) -> Tuple[ModulationOut, Optional[ModulationOut]]:
        x = self.lin(nn.silu(x))
        xs = mx.split(x[:, None, :], self.multiplier, axis=-1)

        mod1 = ModulationOut(*xs[:3])
        mod2 = ModulationOut(*xs[3:]) if self.is_double else None

        return mod1, mod2

def _attention(q: mx.array, k: mx.array, v: mx.array, pe: mx.array):
    B, H, L, D = q.shape

    q = _apply_rope(q, pe)
    k = _apply_rope(k, pe)
    x = mx.fast.scaled_dot_product_attention(q, k, v, scale=D ** (-0.5))

    return x.transpose(0, 2, 1, 3).reshape(B, L, -1)

def _rope(pos: mx.array, dim: int, theta: float):
    scale = mx.arange(0, dim, 2, dtype=mx.float32) / dim
    omega = 1.0 / (theta**scale)
    x = pos[..., None] * omega
    cosx = mx.cos(x)
    sinx = mx.sin(x)
    pe = mx.stack([cosx, -sinx, sinx, cosx], axis=-1)
    pe = pe.reshape(*pe.shape[:-1], 2, 2)

    return pe

class DoubleStreamBlock(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False
    ):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, affine=False, eps=1e-6)
        self.img_attn = SelfAttention(
            dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias
        )

        self.img_norm2 = nn.LayerNorm(hidden_size, affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approx="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(
            dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias
        )

        self.txt_norm2 = nn.LayerNorm(hidden_size, affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approx="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def __call__(
        self, img: mx.array, txt: mx.array, vec: mx.array, pe: mx.array
    ) -> Tuple[mx.array, mx.array]:
        B, L, _ = img.shape
        _, S, _ = txt.shape
        H = self.num_heads

        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = mx.split(img_qkv, 3, axis=-1)
        img_q = img_q.reshape(B, L, H, -1).transpose(0, 2, 1, 3)
        img_k = img_k.reshape(B, L, H, -1).transpose(0, 2, 1, 3)
        img_v = img_v.reshape(B, L, H, -1).transpose(0, 2, 1, 3)
        img_q, img_k = self.img_attn.norm(img_q, img_k)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = mx.split(txt_qkv, 3, axis=-1)
        txt_q = txt_q.reshape(B, S, H, -1).transpose(0, 2, 1, 3)
        txt_k = txt_k.reshape(B, S, H, -1).transpose(0, 2, 1, 3)
        txt_v = txt_v.reshape(B, S, H, -1).transpose(0, 2, 1, 3)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k)

        # run actual attention
        q = mx.concatenate([txt_q, img_q], axis=2)
        k = mx.concatenate([txt_k, img_k], axis=2)
        v = mx.concatenate([txt_v, img_v], axis=2)

        attn = _attention(q, k, v, pe)
        txt_attn, img_attn = mx.split(attn, [S], axis=1)

        # calculate the img bloks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp(
            (1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift
        )

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp(
            (1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift
        )

        return img, txt

class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: List[int]):
        super().__init__()

        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def __call__(self, ids: mx.array):
        n_axes = ids.shape[-1]
        pe = mx.concatenate(
            [_rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            axis=-3,
        )

        return pe[:, None]

class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.out_layer(nn.silu(self.in_layer(x)))

class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def __call__(self, x: mx.array, vec: mx.array):
        shift, scale = mx.split(self.adaLN_modulation(vec), 2, axis=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x

class SingleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: Optional[float] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approx="tanh")
        self.modulation = Modulation(hidden_size, double=False)

    def __call__(self, x: mx.array, vec: mx.array, pe: mx.array):
        B, L, _ = x.shape
        H = self.num_heads

        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift

        q, k, v, mlp = mx.split(
            self.linear1(x_mod),
            [self.hidden_size, 2 * self.hidden_size, 3 * self.hidden_size],
            axis=-1,
        )
        q = q.reshape(B, L, H, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, H, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, H, -1).transpose(0, 2, 1, 3)
        q, k = self.norm(q, k)

        # compute attention
        y = _attention(q, k, v, pe)

        # compute activation in mlp stream, cat again and run second linear layer
        y = self.linear2(mx.concatenate([y, self.mlp_act(mlp)], axis=2))
        return x + mod.gate * y

def timestep_embedding(
    t: mx.array, dim: int, max_period: int = 10000, time_factor: float = 1000.0
    ):
    half = dim // 2
    freqs = mx.arange(0, half, dtype=mx.float32) / half
    freqs = freqs * (-math.log(max_period))
    freqs = mx.exp(freqs)

    x = (time_factor * t)[:, None] * freqs[None]
    x = mx.concatenate([mx.cos(x), mx.sin(x)], axis=-1)

    return x.astype(t.dtype)
    
class Flux(nn.Module):
    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(
                f"Got {params.axes_dim} but expected positional dim {pe_dim}"
            )
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(
            dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim
        )
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
            if params.guidance_embed
            else nn.Identity()
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = [
            DoubleStreamBlock(
                self.hidden_size,
                self.num_heads,
                mlp_ratio=params.mlp_ratio,
                qkv_bias=params.qkv_bias,
            )
            for _ in range(params.depth)
        ]

        self.single_blocks = [
            SingleStreamBlock(
                self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio
            )
            for _ in range(params.depth_single_blocks)
        ]

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def sanitize(self, weights):
        new_weights = {}
        for k, w in weights.items():
            if k.startswith("model.diffusion_model."):
                k = k[22:]
            if k.endswith(".scale"):
                k = k[:-6] + ".weight"
            for seq in ["img_mlp", "txt_mlp", "adaLN_modulation"]:
                if f".{seq}." in k:
                    k = k.replace(f".{seq}.", f".{seq}.layers.")
                    break
            new_weights[k] = w
        return new_weights

    def __call__(
        self,
        img: mx.array,
        img_ids: mx.array,
        txt: mx.array,
        txt_ids: mx.array,
        timesteps: mx.array,
        y: mx.array,
        guidance: Optional[mx.array] = None,
    ) -> mx.array:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = mx.concatenate([txt_ids, img_ids], axis=1)
        pe = self.pe_embedder(ids).astype(img.dtype)

        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        img = mx.concatenate([txt, img], axis=1)
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)

        return img