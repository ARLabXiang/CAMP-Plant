"""
simvp_full_model.py
-------------------
SimVP backbone augmented with a binary classification head, mirroring the
CAMP_full / TAU_full design on top of the SimVP (gSTA default) architecture.

Architecture:
    input frames
        │
        ▼
    Encoder (SimVP)
        │
        ▼
    Translator (MidMetaNet, gSTA by default)   ← tap point for cls head
        │                                       ├─ avg over T
        │                                       ├─ GAP over (H', W')
        │                                       ├─ FC → ReLU → Dropout → FC
        │                                       └─ cls_logit [B, num_classes]
        ▼
    Decoder (SimVP)
        │
        ▼
    predicted future frames  [B, T, C, H, W]

Mirrors the cls head pattern in CAMP_Model and TAU_Full_Model for apples-to-
apples ablation across backbones.
"""

import torch
import torch.nn as nn

from .simvp_model import SimVP_Model


class SimVP_Full_Model(SimVP_Model):
    r"""SimVP backbone (default gSTA) with a classification head on translator output."""

    def __init__(self, in_shape, hid_S=16, hid_T=256, N_S=4, N_T=4,
                 model_type='gSTA', mlp_ratio=8., drop=0.0, drop_path=0.0,
                 spatio_kernel_enc=3, spatio_kernel_dec=3, act_inplace=True,
                 cls_hidden=256, cls_dropout=0.3, num_classes=1,
                 **kwargs):
        super().__init__(
            in_shape=in_shape, hid_S=hid_S, hid_T=hid_T, N_S=N_S, N_T=N_T,
            model_type=model_type,
            mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path,
            spatio_kernel_enc=spatio_kernel_enc,
            spatio_kernel_dec=spatio_kernel_dec,
            act_inplace=act_inplace, **kwargs,
        )

        # Classification head — mirrors CAMP_Model / TAU_Full_Model head structure
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(hid_S, cls_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cls_dropout),
            nn.Linear(cls_hidden, num_classes),
        )

    def forward(self, x_raw, **kwargs):
        """
        Args:
            x_raw: FloatTensor [B, T, C, H, W] — input RGB sequence.

        Returns:
            Y:         FloatTensor [B, T, C, H, W] — predicted future frames.
            cls_logit: FloatTensor [B, num_classes] — raw classification logits.
        """
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B * T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        hid_full = self.hid(z)                              # [B, T, hid_S, H', W']

        # Classification head: avg over T → GAP → FC stack
        cls_feat = hid_full.mean(dim=1)                     # [B, hid_S, H', W']
        cls_logit = self.cls_head(cls_feat)                 # [B, num_classes]

        # Decoder pathway — identical to SimVP_Model
        hid = hid_full.reshape(B * T, C_, H_, W_)
        Y = self.dec(hid, skip)
        Y = Y.reshape(B, T, C, H, W)

        return Y, cls_logit
