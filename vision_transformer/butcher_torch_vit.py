""" Source: https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py
"""

import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn

# from ..ops.misc import Conv2dNormActivation, MLP
# from ..transforms._presets import ImageClassification, InterpolationMode
# from ..utils import _log_api_usage_once
# from ._api import register_model, Weights, WeightsEnum
# from ._meta import _IMAGENET_CATEGORIES
# from ._utils import _ovewrite_named_param, handle_legacy_interface



class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d
    activation_layer: Callable[..., nn.Module] = nn.ReLU


# class (MLP):
#     """Transformer MLP block."""

#     _version = 2

#     def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
#         super().__init__(in_dim, [mlp_dim, in_dim],
#                          activation_layer=nn.GELU, inplace=None, dropout=dropout)

#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.normal_(m.bias, std=1e-6)

#     def _load_from_state_dict(
#         self,
#         state_dict,
#         prefix,
#         local_metadata,
#         strict,
#         missing_keys,
#         unexpected_keys,
#         error_msgs,
#     ):
#         version = local_metadata.get("version", None)

#         if version is None or version < 2:
#             # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
#             for i MLPBlockin range(2):
#                 for type in ["weight", "bias"]:
#                     old_key = f"{prefix}linear_{i+1}.{type}"
#                     new_key = f"{prefix}{3*i}.{type}"
#                     if old_key in state_dict:
#                         state_dict[new_key] = state_dict.pop(old_key)

#         super()._load_from_state_dict(
#             state_dict,
#             prefix,
#             local_metadata,
#             strict,
#             missing_keys,
#             unexpected_keys,
#             error_msgs,
#         )


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))


class VisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        _log_api_usage_once(self)
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.norm_layer = norm_layer

        self.conv_proj = nn.Conv2d(
            in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size)

        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        heads_layers["head"] = nn.Linear(hidden_dim, num_classes)

        self.heads = nn.Sequential(heads_layers)

        # Init the patchify stem
        fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
        nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
        if self.conv_proj.bias is not None:
            nn.init.zeros_(self.conv_proj.bias)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x


# def vit_b_16():
#     return VisionTransformer(
#         image_size=224,
#         patch_size=16,
#         num_layers=12,
#         num_heads=12,
#         hidden_dim=768,
#         mlp_dim=3072,
#     )


if __name__ == "__main__":

    image_size = 224
    patch_size = 16
    num_layers = 2  # default 12
    num_heads = 12
    hidden_dim = 768
    mlp_dim = 1024  # default 3072


    # fake input data
    n = 2  # batch size
    c = 3  # RGB
    h = image_size  # img size
    w = image_size  # img size
    x = torch.rand(n, c, h, w)  # (2, 3, 224, 224)

    # input img embedding
    conv_proj = nn.Conv2d(
        in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size)
    x = conv_proj(x)  # (2, 768, 14, 14)

    n_h = h // patch_size  # 14 = 224 // 16
    n_w = w // patch_size  # 14 = 224 // 16
    x = x.reshape(n, hidden_dim, n_h * n_w)  # (2, 768, 196)

    x = x.permute(0, 2, 1)  # (batch, seq, d_model): (2, 196, 768)

    class_token = torch.zeros(1, 1, hidden_dim)  # (1, 1, 768)
    batch_class_token = class_token.expand(n, -1, -1)  # (2, 1, 768)
    x = torch.cat([batch_class_token, x], dim=1)  # (2, 197, 768)

    seq_length = (image_size // patch_size) ** 2  # 14 * 14 = 196
    seq_length += 1  # 196 + 1 = 197

    # positional embedding  #TODO: why is the embedding in this form?
    pos_embedding = torch.empty(1, seq_length, hidden_dim).normal_(std=0.02)  # from BERT. (1, 197, 768)
    x = x + pos_embedding  # broadcast on the batch dim: (2, 197, 768)

    # encoder_layer: for study purpose, i only do one layer here
    # - attn layer
    ln_1 = nn.LayerNorm(hidden_dim)
    y = ln_1(x)  # (2, 197, 768)
    self_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
    y, _ = self_attention(y, y, y, need_weights=False)
    y = y + x  # (2, 197, 768)
    # - MLP layer
    ln_2 = nn.LayerNorm(hidden_dim)
    z = ln_2(y)  # (2, 197, 768)
    mlp = nn.Sequential(
        nn.Linear(hidden_dim, mlp_dim),
        nn.ReLU(),
        nn.Linear(mlp_dim, hidden_dim),
    )
    z = mlp(z)  # (2, 197, 768)
    x = z + y  # (2, 197, 768)
    del y, z

    # classifier "token" as used by standard language architectures
    x = x[:, 0]  # (2, 768): (batch, d_model)

    # downstream layer
    num_classes = 2  # assuming a downstream binary classification problem
    heads = nn.Linear(hidden_dim, num_classes)
    pred = heads(x)  # (2, 2)
    print(pred)
