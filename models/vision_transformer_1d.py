"""
Author: Guoxin Wang
Date: 2025-05-20 13:08:01
LastEditors: Guoxin Wang
LastEditTime: 2025-05-22 12:33:05
FilePath: /MAECG/models/vision_transformer_1d.py
Description:

Copyright (c) 2025 by Guoxin Wang, All Rights Reserved.
"""

import math
from functools import partial
from typing import Any, Callable, Dict, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import (
    AttentionPoolLatent,
    LayerType,
    Mlp,
    PatchDropout,
    get_act_layer,
    get_norm_layer,
    trunc_normal_,
)
from timm.layers.format import Format
from timm.models import load_pretrained, named_apply, register_model
from timm.models.vision_transformer import Block, get_init_weights_vit, global_pool_nlc

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def ncl_to(x: torch.Tensor, fmt: Format):
    if fmt == Format.NLC:
        x = x.transpose(1, 2)
    return x


class PatchEmbed1D(nn.Module):
    """1D Signal to Patch Embedding"""

    output_fmt: Format
    dynamic_sig_pad: torch.jit.Final[bool]

    def __init__(
        self,
        sig_length: int = 480,
        patch_size: int = 32,
        in_chans: int = 1,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        output_fmt: Optional[str] = None,
        bias: bool = True,
        strict_sig_size: bool = True,
        dynamic_sig_pad: bool = False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.sig_length, self.num_patches = self._init_sig_size(sig_length)

        if output_fmt is not None:
            self.output_fmt = Format(output_fmt)
        else:
            self.output_fmt = Format.NCL
        self.strict_sig_size = strict_sig_size
        self.dynamic_sig_pad = dynamic_sig_pad

        self.proj = nn.Conv1d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def _init_sig_size(self, sig_length: int):
        assert self.patch_size
        if sig_length is None:
            return None, None
        num_patches = sig_length // self.patch_size
        return sig_length, num_patches

    def feat_ratio(self) -> int:
        return self.patch_size

    def dynamic_feat_size(self, sig_length: int) -> int:
        """Get grid (feature) size for given signal size taking account of dynamic padding."""
        if self.dynamic_sig_pad:
            return math.ceil(sig_length / self.patch_size)
        else:
            return sig_length // self.patch_size

    def forward(self, x: torch.Tensor):
        B, C, L = x.shape
        if self.sig_length is not None:
            if self.strict_sig_size:
                torch._assert(
                    L == self.sig_length,
                    f"Input length ({L}) doesn't match model ({self.sig_length}).",
                )
            elif not self.dynamic_sig_pad:
                torch._assert(
                    L % self.patch_size == 0,
                    f"Input length ({L}) should be divisible by patch size ({self.patch_size}).",
                )
        if self.dynamic_sig_pad:
            pad_l = (self.patch_size - L % self.patch_size) % self.patch_size
            x = F.pad(x, (0, pad_l))
        x = self.proj(x)
        if self.output_fmt != Format.NCL:
            x = ncl_to(x, self.output_fmt)
        x = self.norm(x)
        return x


class VisionTransformer1D(nn.Module):
    def __init__(
        self,
        sig_length: int = 480,
        patch_size: int = 32,
        in_chans: int = 1,
        num_classes: int = 4,
        global_pool: Literal["", "avg", "avgmax", "max", "token", "map"] = "token",
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        proj_bias: bool = True,
        init_values: Optional[float] = None,
        class_token: bool = True,
        pos_embed: str = "learn",
        reg_tokens: int = 0,
        pre_norm: bool = False,
        final_norm: bool = True,
        fc_norm: bool = False,
        dynamic_sig_pad: bool = False,
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        weight_init: Literal["skip", "jax", "jax_nlhb", "moco", ""] = "",
        fix_init: bool = False,
        embed_layer: Callable = PatchEmbed1D,
        embed_norm_layer: Optional[LayerType] = None,
        norm_layer: Optional[LayerType] = None,
        act_layer: Optional[LayerType] = None,
        block_fn: Type[nn.Module] = Block,
        mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        assert global_pool in ("", "avg", "avgmax", "max", "token", "map")
        assert class_token or global_pool != "token"
        assert pos_embed in ("", "none", "learn")
        use_fc_norm = (
            global_pool in ("avg", "avgmax", "max") if fc_norm is None else fc_norm
        )
        norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        embed_norm_layer = get_norm_layer(embed_norm_layer)
        act_layer = get_act_layer(act_layer) or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = (
            embed_dim  # for consistency with other models
        )
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token

        embed_args = {}
        if embed_norm_layer is not None:
            embed_args["norm_layer"] = embed_norm_layer
        self.patch_embed = embed_layer(
            sig_length=sig_length,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            output_fmt="NLC",
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_sig_pad=dynamic_sig_pad,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches
        reduction = (
            self.patch_embed.feat_ratio()
            if hasattr(self.patch_embed, "feat_ratio")
            else patch_size
        )

        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        )
        self.reg_token = (
            nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        )
        embed_len = num_patches + self.num_prefix_tokens
        if not pos_embed or pos_embed == "none":
            self.pos_embed = None
        else:
            self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    proj_bias=proj_bias,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                )
                for i in range(depth)
            ]
        )
        self.feature_info = [
            dict(module=f"blocks.{i}", num_chs=embed_dim, reduction=reduction)
            for i in range(depth)
        ]
        self.norm = (
            norm_layer(embed_dim) if final_norm and not use_fc_norm else nn.Identity()
        )

        # Classifier Head
        if global_pool == "map":
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
            )
        else:
            self.attn_pool = None
        self.fc_norm = (
            norm_layer(embed_dim) if final_norm and use_fc_norm else nn.Identity()
        )
        self.head_drop = nn.Dropout(drop_rate)
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        if weight_init != "skip":
            self.init_weights(weight_init)
        if fix_init:
            self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, _layer_id):
            param.div_(math.sqrt(2.0 * _layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def init_weights(self, mode: str = "") -> None:
        assert mode in ("jax", "jax_nlhb", "moco", "")
        head_bias = -math.log(self.num_classes) if "nlhb" in mode else 0.0
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        if self.reg_token is not None:
            nn.init.normal_(self.reg_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ("", "avg", "avgmax", "max", "token", "map")
            if global_pool == "map" and self.attn_pool is None:
                assert (
                    False
                ), "Cannot currently add attention pooling in reset_classifier()."
            elif global_pool != "map" and self.attn_pool is not None:
                self.attn_pool = None  # remove attention pooling
            self.global_pool = global_pool
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is None:
            return x

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        # original timm, JAX, and deit vit impl
        # pos_embed has entry for class token, concat then add
        if to_cat:
            x = torch.cat(to_cat + [x], dim=1)
        x = x + self.pos_embed

        return self.pos_drop(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def pool(self, x: torch.Tensor, pool_type: Optional[str] = None) -> torch.Tensor:
        if self.attn_pool is not None:
            x = self.attn_pool(x)
            return x
        pool_type = self.global_pool if pool_type is None else pool_type
        x = global_pool_nlc(
            x, pool_type=pool_type, num_prefix_tokens=self.num_prefix_tokens
        )
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.pool(x)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


@register_model
def vit_1d_atto(
    pretrained: bool = False,
    pretrained_cfg: Optional[Dict[str, Any]] = None,
    pretrained_cfg_overlay: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
    **kwargs: Any,
) -> nn.Module:
    if pretrained_cfg is None:
        pretrained_cfg = {}
    if pretrained_cfg_overlay is None:
        pretrained_cfg_overlay = {}
    model = VisionTransformer1D(
        embed_dim=96,
        depth=12,
        num_heads=2,
        **kwargs,
    )
    if pretrained:
        if pretrained_cfg_overlay.get("file", None):
            pretrained_cfg["file"] = pretrained_cfg_overlay["file"]
            load_pretrained(model, pretrained_cfg, strict=False, cache_dir=cache_dir)
        elif pretrained_cfg_overlay.get("url", None):
            pretrained_cfg["url"] = pretrained_cfg_overlay["url"]
            load_pretrained(model, pretrained_cfg, strict=False, cache_dir=cache_dir)
    return model


@register_model
def vit_1d_tiny(
    pretrained: bool = False,
    pretrained_cfg: Optional[Dict[str, Any]] = None,
    pretrained_cfg_overlay: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
    **kwargs: Any,
) -> nn.Module:
    if pretrained_cfg is None:
        pretrained_cfg = {}
    if pretrained_cfg_overlay is None:
        pretrained_cfg_overlay = {}
    model = VisionTransformer1D(
        embed_dim=192,
        depth=12,
        num_heads=3,
        **kwargs,
    )
    if pretrained:
        if pretrained_cfg_overlay.get("file", None):
            pretrained_cfg["file"] = pretrained_cfg_overlay["file"]
            load_pretrained(model, pretrained_cfg, strict=False, cache_dir=cache_dir)
        elif pretrained_cfg_overlay.get("url", None):
            pretrained_cfg["url"] = pretrained_cfg_overlay["url"]
            load_pretrained(model, pretrained_cfg, strict=False, cache_dir=cache_dir)
    return model


@register_model
def vit_1d_small(
    pretrained: bool = False,
    pretrained_cfg: Optional[Dict[str, Any]] = None,
    pretrained_cfg_overlay: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
    **kwargs: Any,
) -> nn.Module:
    if pretrained_cfg is None:
        pretrained_cfg = {}
    if pretrained_cfg_overlay is None:
        pretrained_cfg_overlay = {}
    model = VisionTransformer1D(
        embed_dim=384,
        depth=12,
        num_heads=6,
        **kwargs,
    )
    if pretrained:
        if pretrained_cfg_overlay.get("file", None):
            pretrained_cfg["file"] = pretrained_cfg_overlay["file"]
            load_pretrained(model, pretrained_cfg, strict=False, cache_dir=cache_dir)
        elif pretrained_cfg_overlay.get("url", None):
            pretrained_cfg["url"] = pretrained_cfg_overlay["url"]
            load_pretrained(model, pretrained_cfg, strict=False, cache_dir=cache_dir)
    return model


@register_model
def vit_1d_base(
    pretrained: bool = False,
    pretrained_cfg: Optional[Dict[str, Any]] = None,
    pretrained_cfg_overlay: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
    **kwargs: Any,
) -> nn.Module:
    if pretrained_cfg is None:
        pretrained_cfg = {}
    if pretrained_cfg_overlay is None:
        pretrained_cfg_overlay = {}
    model = VisionTransformer1D(
        embed_dim=768,
        depth=12,
        num_heads=12,
        **kwargs,
    )
    if pretrained:
        if pretrained_cfg_overlay.get("file", None):
            pretrained_cfg["file"] = pretrained_cfg_overlay["file"]
            load_pretrained(model, pretrained_cfg, strict=False, cache_dir=cache_dir)
        elif pretrained_cfg_overlay.get("url", None):
            pretrained_cfg["url"] = pretrained_cfg_overlay["url"]
            load_pretrained(model, pretrained_cfg, strict=False, cache_dir=cache_dir)
    return model


@register_model
def vit_1d_large(
    pretrained: bool = False,
    pretrained_cfg: Optional[Dict[str, Any]] = None,
    pretrained_cfg_overlay: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
    **kwargs: Any,
) -> nn.Module:
    if pretrained_cfg is None:
        pretrained_cfg = {}
    if pretrained_cfg_overlay is None:
        pretrained_cfg_overlay = {}
    model = VisionTransformer1D(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        **kwargs,
    )
    if pretrained:
        if pretrained_cfg_overlay.get("file", None):
            pretrained_cfg["file"] = pretrained_cfg_overlay["file"]
            load_pretrained(model, pretrained_cfg, strict=False, cache_dir=cache_dir)
        elif pretrained_cfg_overlay.get("url", None):
            pretrained_cfg["url"] = pretrained_cfg_overlay["url"]
            load_pretrained(model, pretrained_cfg, strict=False, cache_dir=cache_dir)
    return model


@register_model
def vit_1d_huge(
    pretrained: bool = False,
    pretrained_cfg: Optional[Dict[str, Any]] = None,
    pretrained_cfg_overlay: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
    **kwargs: Any,
) -> nn.Module:
    if pretrained_cfg is None:
        pretrained_cfg = {}
    if pretrained_cfg_overlay is None:
        pretrained_cfg_overlay = {}
    model = VisionTransformer1D(
        embed_dim=1280,
        depth=32,
        num_heads=16,
        **kwargs,
    )
    if pretrained:
        if pretrained_cfg_overlay.get("file", None):
            pretrained_cfg["file"] = pretrained_cfg_overlay["file"]
            load_pretrained(model, pretrained_cfg, strict=False, cache_dir=cache_dir)
        elif pretrained_cfg_overlay.get("url", None):
            pretrained_cfg["url"] = pretrained_cfg_overlay["url"]
            load_pretrained(model, pretrained_cfg, strict=False, cache_dir=cache_dir)
    return model
