"""
simvp_predcls_model.py
----------------------
SimVP backbone with a late-fusion ("predict-then-classify") classification
head that reads from the predicted future frames `pred_y`, NOT from the
translator latent.

This is the SimVP counterpart of TAU_PredCls — same architectural pattern,
different backbone.  Used to test whether the late-fusion failure observed
on TAU generalizes across non-recurrent backbones.
"""

import torch
import torch.nn as nn

from .simvp_model import SimVP_Model


class SimVP_PredCls_Model(SimVP_Model):
    r"""SimVP backbone with late-fusion classification head reading from pred_y."""

    def __init__(self, in_shape, hid_S=16, hid_T=256, N_S=4, N_T=4,
                 model_type='gSTA', mlp_ratio=8., drop=0.0, drop_path=0.0,
                 spatio_kernel_enc=3, spatio_kernel_dec=3, act_inplace=True,
                 cls_hidden=128, cls_dropout=0.3, num_classes=1,
                 **kwargs):
        super().__init__(
            in_shape=in_shape, hid_S=hid_S, hid_T=hid_T, N_S=N_S, N_T=N_T,
            model_type=model_type,
            mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path,
            spatio_kernel_enc=spatio_kernel_enc,
            spatio_kernel_dec=spatio_kernel_dec,
            act_inplace=act_inplace, **kwargs,
        )

        T, C, H, W = in_shape

        # Late-fusion classification head — small CNN over time-averaged pred_y
        self.late_cls_head = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(128, cls_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cls_dropout),
            nn.Linear(cls_hidden, num_classes),
        )

    def forward(self, x_raw, **kwargs):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B * T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape
        z = embed.view(B, T, C_, H_, W_)
        hid_full = self.hid(z)
        hid = hid_full.reshape(B * T, C_, H_, W_)
        Y = self.dec(hid, skip)
        Y = Y.reshape(B, T, C, H, W)

        # LATE-FUSION classification head reads from predicted frames
        avg_pred = Y.mean(dim=1)                    # [B, C, H, W]
        cls_logit = self.late_cls_head(avg_pred)    # [B, num_classes]

        return Y, cls_logit
