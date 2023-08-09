from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block

from util.pos_embed import get_1d_sincos_pos_embed

__all__ = [
    "ViT1D",
    "MaskedAutoencoderViT1D",
    'mae_vit_atto_patch4',
    'mae_vit_atto_patch8',
    'mae_vit_atto_patch16',
    'mae_vit_atto_patch32',
    'mae_vit_tiny_patch4',
    'mae_vit_tiny_patch8',
    'mae_vit_tiny_patch16',
    'mae_vit_tiny_patch32',
    'mae_vit_small_patch4',
    'mae_vit_small_patch8',
    'mae_vit_small_patch16',
    'mae_vit_small_patch32',
    'mae_vit_base_patch4',
    'mae_vit_base_patch8',
    'mae_vit_base_patch16',
    'mae_vit_base_patch32',
    'mae_vit_large_patch4',
    'mae_vit_huge_patch4',
    'vit_atto_p4',
    'vit_atto_p8',
    'vit_atto_p16',
    'vit_atto_p32',
    'vit_tiny_p4',
    'vit_tiny_p8',
    'vit_tiny_p16',
    'vit_tiny_p32',
    'vit_small_p4',
    'vit_small_p8',
    'vit_small_p16',
    'vit_small_p32',
    'vit_base_p4',
    'vit_base_p8',
    'vit_base_p16',
    'vit_base_p32',
    'vit_large_p4',
    'vit_huge_p4',
]


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dims, activation_layer=nn.ReLU, dropout=0.0):
        super().__init__()
        self.num_layers = len(hidden_dims)
        self.dropout = dropout
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + hidden_dims[:-1], hidden_dims))
        self.activations = nn.ModuleList(activation_layer() for _ in range(self.num_layers - 1))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            # print(layer)
            # print(x.shape)
            x = F.dropout(self.activations[i](layer(x)), p=self.dropout) if i < self.num_layers - 1 else layer(x)
        return x


class PatchEmbed1D(nn.Module):
    """ 1D Signal to Patch Embedding
    Reference: https://github.com/rwightman/pytorch-image-models/blob/main/timm/layers/patch_embed.py
    """

    def __init__(
        self,
        signal_length: int = 480,
        patch_size: int = 4,
        in_chans: int = 1,
        embed_dim: int = 768,
        norm_layer: nn.Module = None,
        channel_last: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.signal_length = signal_length
        self.patch_size = patch_size
        self.grid_size = signal_length // patch_size
        self.num_patches = self.grid_size
        self.channel_last = channel_last
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor):
        B, C, L = x.shape
        assert L == self.signal_length, f"Input signal length ({L}) doesn't match model ({self.signal_length})."
        x = self.proj(x)
        if self.channel_last:
            x = x.transpose(1, 2)  # BCL -> BLC
        x = self.norm(x)
        return x


class ViT1D(nn.Module):

    def __init__(self,
                 mlp_sizes=[128, 128, 1],
                 signal_length: int = 480,
                 patch_size: int = 4,
                 in_chans: int = 1,
                 embed_dim: int = 1024,
                 depth: int = 24,
                 num_heads: int = 16,
                 mlp_ratio: int = 4.,
                 norm_layer: nn.Module = nn.LayerNorm,
                 **kwargs):
        super().__init__(**kwargs)

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed1D(signal_length, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------
        self.init_head(embed_dim, mlp_sizes)
        self.apply(self._init_weights)

    def init_head(self, embed_dim, mlp_sizes):
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), MLP(embed_dim, mlp_sizes))

    def forward(self, signals):
        # signals = signals.transpose(1, 2)

        # embed patches
        x = self.patch_embed(signals)

        # add pos embed
        x = x + self.pos_embed
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = self.head(x.transpose(1, 2))
        return x

    def freeze_backbone(self):
        for _, p in self.named_parameters():
            p.requires_grad = False
        for _, p in self.head.named_parameters():
            p.requires_grad = True

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, signals):
        """
        signals: (N, C, L)
        x: (N, L, patch_size *C)
        """
        N, C, L = signals.shape
        p = self.patch_embed.patch_size
        assert L % p == 0

        l = L // p

        x = signals.reshape(shape=(N, C, l, p))
        x = torch.einsum('nclp->nlpc', x)
        x = x.reshape(shape=(N, l, p * C))
        return x

    def unpatchify(self, x, C: int = 1, channel_last=True):
        """
        x: (N, L, patch_size*C)
        signals: (N, C, L)
        """
        N, L, _ = x.shape
        p = self.patch_embed.patch_size
        x = x.reshape(shape=(N, L, p, C))
        if channel_last:
            signals = x.reshape(shape=(N, L * p, C))
        else:
            x = torch.einsum('nlpc->nclp', x)
            signals = x.reshape(shape=(N, C, L * p))
        return signals


class MaskedAutoencoderViT1D(ViT1D):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self,
                 patch_size: int = 4,
                 in_chans: int = 1,
                 decoder_embed_dim: int = 512,
                 decoder_depth: int = 8,
                 decoder_num_heads: int = 16,
                 norm_pix_loss: bool = False,
                 embed_dim: int = 1024,
                 mlp_ratio: int = 4.,
                 signal_length: int = 480,
                 norm_layer: nn.Module = nn.LayerNorm,
                 **kwargs):
        super().__init__(patch_size=patch_size,
                         in_chans=in_chans,
                         embed_dim=embed_dim,
                         mlp_ratio=mlp_ratio,
                         signal_length=signal_length,
                         norm_layer=norm_layer,
                         **kwargs)
        num_patches = self.patch_embed.num_patches
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)
        ])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def init_head(self, embed_dim, mlp_sizes):
        pass

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches))
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
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

    def forward_encoder(self, x, mask_ratio):
        # No cls token
        # embed patches
        x = self.patch_embed(x)

        # add pos embed
        x = x + self.pos_embed

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x

    def forward_loss(self, signals, pred, mask):
        """
        signals: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(signals)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target)**2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, signals, mask_ratio=0.75):
        # signals = signals.transpose(1, 2)
        latent, mask, ids_restore = self.forward_encoder(signals, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(signals, pred, mask)
        return loss, pred, mask


def mae_vit_atto_patch4_dec128d8b(**kwargs):
    model = MaskedAutoencoderViT1D(patch_size=4,
                                   embed_dim=96,
                                   depth=12,
                                   num_heads=2,
                                   decoder_embed_dim=128,
                                   decoder_depth=8,
                                   decoder_num_heads=16,
                                   mlp_ratio=4,
                                   norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                   **kwargs)
    return model


def mae_vit_atto_patch8_dec128d8b(**kwargs):
    model = MaskedAutoencoderViT1D(patch_size=8,
                                   embed_dim=96,
                                   depth=12,
                                   num_heads=2,
                                   decoder_embed_dim=128,
                                   decoder_depth=8,
                                   decoder_num_heads=16,
                                   mlp_ratio=4,
                                   norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                   **kwargs)
    return model


def mae_vit_atto_patch16_dec128d8b(**kwargs):
    model = MaskedAutoencoderViT1D(patch_size=16,
                                   embed_dim=96,
                                   depth=12,
                                   num_heads=2,
                                   decoder_embed_dim=128,
                                   decoder_depth=8,
                                   decoder_num_heads=16,
                                   mlp_ratio=4,
                                   norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                   **kwargs)
    return model


def mae_vit_atto_patch32_dec128d8b(**kwargs):
    model = MaskedAutoencoderViT1D(patch_size=32,
                                   embed_dim=96,
                                   depth=12,
                                   num_heads=2,
                                   decoder_embed_dim=128,
                                   decoder_depth=8,
                                   decoder_num_heads=16,
                                   mlp_ratio=4,
                                   norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                   **kwargs)
    return model


def mae_vit_tiny_patch4_dec256d8b(**kwargs):
    model = MaskedAutoencoderViT1D(patch_size=4,
                                   embed_dim=192,
                                   depth=12,
                                   num_heads=3,
                                   decoder_embed_dim=256,
                                   decoder_depth=8,
                                   decoder_num_heads=16,
                                   mlp_ratio=4,
                                   norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                   **kwargs)
    return model


def mae_vit_tiny_patch8_dec256d8b(**kwargs):
    model = MaskedAutoencoderViT1D(patch_size=8,
                                   embed_dim=192,
                                   depth=12,
                                   num_heads=3,
                                   decoder_embed_dim=256,
                                   decoder_depth=8,
                                   decoder_num_heads=16,
                                   mlp_ratio=4,
                                   norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                   **kwargs)
    return model


def mae_vit_tiny_patch16_dec256d8b(**kwargs):
    model = MaskedAutoencoderViT1D(patch_size=16,
                                   embed_dim=192,
                                   depth=12,
                                   num_heads=3,
                                   decoder_embed_dim=256,
                                   decoder_depth=8,
                                   decoder_num_heads=16,
                                   mlp_ratio=4,
                                   norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                   **kwargs)
    return model


def mae_vit_tiny_patch32_dec256d8b(**kwargs):
    model = MaskedAutoencoderViT1D(patch_size=32,
                                   embed_dim=192,
                                   depth=12,
                                   num_heads=3,
                                   decoder_embed_dim=256,
                                   decoder_depth=8,
                                   decoder_num_heads=16,
                                   mlp_ratio=4,
                                   norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                   **kwargs)
    return model


def mae_vit_small_patch4_dec256d8b(**kwargs):
    model = MaskedAutoencoderViT1D(patch_size=4,
                                   embed_dim=384,
                                   depth=12,
                                   num_heads=6,
                                   decoder_embed_dim=256,
                                   decoder_depth=8,
                                   decoder_num_heads=16,
                                   mlp_ratio=4,
                                   norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                   **kwargs)
    return model


def mae_vit_small_patch8_dec256d8b(**kwargs):
    model = MaskedAutoencoderViT1D(patch_size=8,
                                   embed_dim=384,
                                   depth=12,
                                   num_heads=6,
                                   decoder_embed_dim=256,
                                   decoder_depth=8,
                                   decoder_num_heads=16,
                                   mlp_ratio=4,
                                   norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                   **kwargs)
    return model


def mae_vit_small_patch16_dec256d8b(**kwargs):
    model = MaskedAutoencoderViT1D(patch_size=16,
                                   embed_dim=384,
                                   depth=12,
                                   num_heads=6,
                                   decoder_embed_dim=256,
                                   decoder_depth=8,
                                   decoder_num_heads=16,
                                   mlp_ratio=4,
                                   norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                   **kwargs)
    return model


def mae_vit_small_patch32_dec256d8b(**kwargs):
    model = MaskedAutoencoderViT1D(patch_size=32,
                                   embed_dim=384,
                                   depth=12,
                                   num_heads=6,
                                   decoder_embed_dim=256,
                                   decoder_depth=8,
                                   decoder_num_heads=16,
                                   mlp_ratio=4,
                                   norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                   **kwargs)
    return model


def mae_vit_base_patch4_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT1D(patch_size=4,
                                   embed_dim=768,
                                   depth=12,
                                   num_heads=12,
                                   decoder_embed_dim=512,
                                   decoder_depth=8,
                                   decoder_num_heads=16,
                                   mlp_ratio=4,
                                   norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                   **kwargs)
    return model


def mae_vit_base_patch8_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT1D(patch_size=8,
                                   embed_dim=768,
                                   depth=12,
                                   num_heads=12,
                                   decoder_embed_dim=512,
                                   decoder_depth=8,
                                   decoder_num_heads=16,
                                   mlp_ratio=4,
                                   norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                   **kwargs)
    return model


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT1D(patch_size=16,
                                   embed_dim=768,
                                   depth=12,
                                   num_heads=12,
                                   decoder_embed_dim=512,
                                   decoder_depth=8,
                                   decoder_num_heads=16,
                                   mlp_ratio=4,
                                   norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                   **kwargs)
    return model


def mae_vit_base_patch32_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT1D(patch_size=32,
                                   embed_dim=768,
                                   depth=12,
                                   num_heads=12,
                                   decoder_embed_dim=512,
                                   decoder_depth=8,
                                   decoder_num_heads=16,
                                   mlp_ratio=4,
                                   norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                   **kwargs)
    return model


def mae_vit_large_patch4_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT1D(patch_size=4,
                                   embed_dim=1024,
                                   depth=24,
                                   num_heads=16,
                                   decoder_embed_dim=512,
                                   decoder_depth=8,
                                   decoder_num_heads=16,
                                   mlp_ratio=4,
                                   norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                   **kwargs)
    return model


def mae_vit_huge_patch4_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT1D(patch_size=4,
                                   embed_dim=1280,
                                   depth=32,
                                   num_heads=16,
                                   decoder_embed_dim=512,
                                   decoder_depth=8,
                                   decoder_num_heads=16,
                                   mlp_ratio=4,
                                   norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                   **kwargs)
    return model



def vit_atto_patch4(**kwargs):
    model = ViT1D(patch_size=4,
                  embed_dim=96,
                  depth=12,
                  num_heads=2,
                  mlp_ratio=4,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6),
                  **kwargs)
    return model


def vit_atto_patch8(**kwargs):
    model = ViT1D(patch_size=8,
                  embed_dim=96,
                  depth=12,
                  num_heads=2,
                  mlp_ratio=4,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6),
                  **kwargs)
    return model


def vit_atto_patch16(**kwargs):
    model = ViT1D(patch_size=16,
                  embed_dim=96,
                  depth=12,
                  num_heads=2,
                  mlp_ratio=4,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6),
                  **kwargs)
    return model


def vit_atto_patch32(**kwargs):
    model = ViT1D(patch_size=32,
                  embed_dim=96,
                  depth=12,
                  num_heads=2,
                  mlp_ratio=4,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6),
                  **kwargs)
    return model


def vit_tiny_patch4(**kwargs):
    model = ViT1D(patch_size=4,
                  embed_dim=192,
                  depth=12,
                  num_heads=3,
                  mlp_ratio=4,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6),
                  **kwargs)
    return model


def vit_tiny_patch8(**kwargs):
    model = ViT1D(patch_size=8,
                  embed_dim=192,
                  depth=12,
                  num_heads=3,
                  mlp_ratio=4,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6),
                  **kwargs)
    return model


def vit_tiny_patch16(**kwargs):
    model = ViT1D(patch_size=16,
                  embed_dim=192,
                  depth=12,
                  num_heads=3,
                  mlp_ratio=4,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6),
                  **kwargs)
    return model


def vit_tiny_patch32(**kwargs):
    model = ViT1D(patch_size=32,
                  embed_dim=192,
                  depth=12,
                  num_heads=3,
                  mlp_ratio=4,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6),
                  **kwargs)
    return model


def vit_small_patch4(**kwargs):
    model = ViT1D(patch_size=4,
                  embed_dim=384,
                  depth=12,
                  num_heads=6,
                  mlp_ratio=4,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6),
                  **kwargs)
    return model


def vit_small_patch8(**kwargs):
    model = ViT1D(patch_size=8,
                  embed_dim=384,
                  depth=12,
                  num_heads=6,
                  mlp_ratio=4,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6),
                  **kwargs)
    return model


def vit_small_patch16(**kwargs):
    model = ViT1D(patch_size=16,
                  embed_dim=384,
                  depth=12,
                  num_heads=6,
                  mlp_ratio=4,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6),
                  **kwargs)
    return model


def vit_small_patch32(**kwargs):
    model = ViT1D(patch_size=32,
                  embed_dim=384,
                  depth=12,
                  num_heads=6,
                  mlp_ratio=4,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6),
                  **kwargs)
    return model


def vit_base_patch4(**kwargs):
    model = ViT1D(patch_size=4,
                  embed_dim=768,
                  depth=12,
                  num_heads=12,
                  mlp_ratio=4,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6),
                  **kwargs)
    return model


def vit_base_patch8(**kwargs):
    model = ViT1D(patch_size=8,
                  embed_dim=768,
                  depth=12,
                  num_heads=12,
                  mlp_ratio=4,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6),
                  **kwargs)
    return model


def vit_base_patch16(**kwargs):
    model = ViT1D(patch_size=16,
                  embed_dim=768,
                  depth=12,
                  num_heads=12,
                  mlp_ratio=4,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6),
                  **kwargs)
    return model


def vit_base_patch32(**kwargs):
    model = ViT1D(patch_size=32,
                  embed_dim=768,
                  depth=12,
                  num_heads=12,
                  mlp_ratio=4,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6),
                  **kwargs)
    return model


def vit_large_patch4(**kwargs):
    model = ViT1D(patch_size=4,
                  embed_dim=1024,
                  depth=24,
                  num_heads=16,
                  mlp_ratio=4,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6),
                  **kwargs)
    return model


def vit_huge_patch4(**kwargs):
    model = ViT1D(patch_size=4,
                  embed_dim=1280,
                  depth=32,
                  num_heads=16,
                  mlp_ratio=4,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6),
                  **kwargs)
    return model


# set recommended archs
mae_vit_atto_patch4 = mae_vit_atto_patch4_dec128d8b  # decoder: 256 dim, 8 blocks
mae_vit_atto_patch8 = mae_vit_atto_patch8_dec128d8b  # decoder: 256 dim, 8 blocks
mae_vit_atto_patch16 = mae_vit_atto_patch16_dec128d8b  # decoder: 256 dim, 8 blocks
mae_vit_atto_patch32 = mae_vit_atto_patch32_dec128d8b  # decoder: 256 dim, 8 blocks
mae_vit_tiny_patch4 = mae_vit_tiny_patch4_dec256d8b  # decoder: 256 dim, 8 blocks
mae_vit_tiny_patch8 = mae_vit_tiny_patch8_dec256d8b  # decoder: 256 dim, 8 blocks
mae_vit_tiny_patch16 = mae_vit_tiny_patch16_dec256d8b  # decoder: 256 dim, 8 blocks
mae_vit_tiny_patch32 = mae_vit_tiny_patch32_dec256d8b  # decoder: 256 dim, 8 blocks
mae_vit_small_patch4 = mae_vit_small_patch4_dec256d8b  # decoder: 256 dim, 8 blocks
mae_vit_small_patch8 = mae_vit_small_patch8_dec256d8b  # decoder: 256 dim, 8 blocks
mae_vit_small_patch16 = mae_vit_small_patch16_dec256d8b  # decoder: 256 dim, 8 blocks
mae_vit_small_patch32 = mae_vit_small_patch32_dec256d8b  # decoder: 256 dim, 8 blocks
mae_vit_base_patch4 = mae_vit_base_patch4_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_base_patch8 = mae_vit_base_patch8_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_base_patch32 = mae_vit_base_patch32_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch4 = mae_vit_large_patch4_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch4 = mae_vit_huge_patch4_dec512d8b  # decoder: 512 dim, 8 blocks

vit_atto_p4 = vit_atto_patch4
vit_atto_p8 = vit_atto_patch8
vit_atto_p16 = vit_atto_patch16
vit_atto_p32 = vit_atto_patch32
vit_tiny_p4 = vit_tiny_patch4
vit_tiny_p8 = vit_tiny_patch8
vit_tiny_p16 = vit_tiny_patch16
vit_tiny_p32 = vit_tiny_patch32
vit_small_p4 = vit_small_patch4
vit_small_p8 = vit_small_patch8
vit_small_p16 = vit_small_patch16
vit_small_p32 = vit_small_patch32
vit_base_p4 = vit_base_patch4
vit_base_p8 = vit_base_patch8
vit_base_p16 = vit_base_patch16
vit_base_p32 = vit_base_patch32
vit_large_p4 = vit_large_patch4
vit_huge_p4 = vit_huge_patch4