"""
tau_predcls_model.py
--------------------
TAU backbone with a **late-fusion** ("predict-then-classify") classification
head that reads from the predicted future frames `pred_y`, NOT from the
translator latent.

Hypothesis:
    On non-recurrent backbones (TAU, SimVP), tapping the translator latent
    with a classification head (as TAU_full does) catastrophically degrades
    prediction quality (POI_MAE 294 → 547).  The classification gradient
    reshapes the translator's shared features toward drought/control
    discrimination at the expense of leaf-pixel preservation.

    By moving the classification head to AFTER the decoder — operating on
    the predicted RGB frames — we make the classification task semantically
    aligned with prediction:

        "the predicted frames must look enough like a drought-stressed
         (or control) plant to be classified correctly."

    Gradients from cls now flow back through the decoder (which has more
    redundant capacity than the translator) before reaching the translator,
    dampening the destabilization that TAU_full suffered.

Architecture:
    encoder → translator → decoder → pred_y ──→ late_cls_head ──→ cls_logit
                                        │
                                        └─→ MSE + ExGI losses

The late_cls_head is a small CNN classifier that:
    1. Averages pred_y across the time dimension to focus on overall plant state.
    2. Runs a tiny ConvNet (3 stride-2 conv blocks) to extract spatial features.
    3. GAP → FC → ReLU → Dropout → FC → logit.

Total cls_head parameters: ~110k (small relative to the 11.5M backbone).
"""

import torch
import torch.nn as nn

from .simvp_model import SimVP_Model


class TAU_PredCls_Model(SimVP_Model):
    r"""TAU backbone with late-fusion classification head on predicted frames."""

    def __init__(self, in_shape, hid_S=16, hid_T=256, N_S=4, N_T=4,
                 mlp_ratio=8., drop=0.0, drop_path=0.0,
                 spatio_kernel_enc=3, spatio_kernel_dec=3, act_inplace=True,
                 cls_hidden=128, cls_dropout=0.3, num_classes=1,
                 **kwargs):
        # Always TAU — ignore any incoming model_type from the config
        kwargs.pop('model_type', None)

        super().__init__(
            in_shape=in_shape, hid_S=hid_S, hid_T=hid_T, N_S=N_S, N_T=N_T,
            model_type='tau',
            mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path,
            spatio_kernel_enc=spatio_kernel_enc,
            spatio_kernel_dec=spatio_kernel_dec,
            act_inplace=act_inplace, **kwargs,
        )

        T, C, H, W = in_shape

        # Late-fusion classification head: small CNN over time-averaged pred_y
        # Input:  [B, C, H, W]  (after time-average of pred_y)
        # Output: [B, num_classes]
        self.late_cls_head = nn.Sequential(
            # Stride-2 conv block 1: H/2, W/2
            nn.Conv2d(C, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Stride-2 conv block 2: H/4, W/4
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Stride-2 conv block 3: H/8, W/8
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Global avg pool + FC stack
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(128, cls_hidden),
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
            cls_logit: FloatTensor [B, num_classes] — late-fusion classification logits.
        """
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B * T, C, H, W)

        # Standard SimVP/TAU forward pass
        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape
        z = embed.view(B, T, C_, H_, W_)
        hid_full = self.hid(z)
        hid = hid_full.reshape(B * T, C_, H_, W_)
        Y = self.dec(hid, skip)
        Y = Y.reshape(B, T, C, H, W)

        # LATE-FUSION classification head reads from predicted frames Y
        # Average over the temporal dimension to produce a single representative frame.
        avg_pred = Y.mean(dim=1)                    # [B, C, H, W]
        cls_logit = self.late_cls_head(avg_pred)    # [B, num_classes]

        return Y, cls_logit
