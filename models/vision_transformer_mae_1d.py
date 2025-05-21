from functools import partial
from typing import Any, Dict, Optional, Type

import torch
import torch.nn as nn
from timm.layers import LayerType, get_norm_layer
from timm.models import load_pretrained, register_model
from timm.models.vision_transformer import Block

from .vision_transformer_1d import VisionTransformer1D

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class VisionTransformerMAE1D(VisionTransformer1D):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        sig_length: int = 480,
        patch_size: int = 32,
        in_chans: int = 1,
        embed_dim: int = 96,
        depth: int = 12,
        num_heads: int = 2,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: int = 4.0,
        qkv_bias: bool = True,
        weight_init: Literal["skip", "jax", "jax_nlhb", "moco", ""] = "",
        fix_init: bool = False,
        norm_layer: Optional[LayerType] = None,
        block_fn: Type[nn.Module] = Block,
        norm_pix_loss: bool = False,
        **kwargs,
    ):
        super().__init__(
            sig_length=sig_length,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            block_fn=block_fn,
            **kwargs,
        )
        self.norm_pix_loss = norm_pix_loss
        norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        num_patches = self.patch_embed.num_patches

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.randn(1, num_patches, decoder_embed_dim) * 0.02
        )

        self.decoder_blocks = nn.Sequential(
            *[
                block_fn(
                    dim=decoder_embed_dim,
                    num_heads=decoder_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size * in_chans, bias=True
        )  # decoder to patch
        # --------------------------------------------------------------------------

        if weight_init != "skip":
            self.init_weights(weight_init)
        if fix_init:
            self.fix_init_weight()

    def patchify(self, sig: torch.Tensor):
        """
        x: (N, C, L)
        x: (N, L, patch_size *C)
        """
        N, C, L = sig.shape
        p = self.patch_embed.patch_size
        assert L % p == 0
        l = L // p
        x = sig.reshape(shape=(N, C, l, p))
        x = torch.einsum("nclp->nlpc", x)
        x = x.reshape(shape=(N, l, p * C))
        return x

    def unpatchify(self, x: torch.Tensor, C: int = 1, channel_last=True):
        """
        x: (N, L, patch_size*C)
        x: (N, C, L)
        """
        N, L, _ = x.shape
        p = self.patch_embed.patch_size
        x = x.reshape(shape=(N, L, p, C))
        if channel_last:
            x = x.reshape(shape=(N, L * p, C))
        else:
            x = torch.einsum("nlpc->nclp", x)
            x = x.reshape(shape=(N, C, L * p))
        return x

    def random_masking(self, x: torch.Tensor, mask_ratio: float):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x: torch.Tensor, mask_ratio: float):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        # split cls token and rest
        cls_token = x[:, :1, :]  # [B, 1, D]
        x = x[:, 1:, :]  # [B, num_patches, D]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        x = self.blocks(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply decoder Transformer blocks
        x = self.decoder_blocks(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)
        return x

    def forward_loss(self, x: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor):
        """
        x: [N, 1, L]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(x)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.75):
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(x, pred, mask)
        return loss, pred, mask


@register_model
def vit_mae_1d_atto(
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
    model = VisionTransformerMAE1D(
        embed_dim=96,
        depth=12,
        num_heads=2,
        decoder_embed_dim=128,
        decoder_depth=8,
        decoder_num_heads=16,
        **kwargs,
    )
    if pretrained:
        if pretrained_cfg_overlay.get("path", None):
            pretrained_cfg["file"] = pretrained_cfg_overlay["path"]
        else:
            pretrained_cfg["url"] = (
                "https://huggingface.co/PriceWang/model/resolve/main/dmmecg/vit_tiny_af.pth"
            )
        load_pretrained(model, pretrained_cfg, strict=False, cache_dir=cache_dir)
    return model
