"""
tau_full_model.py
-----------------
TAU backbone augmented with a binary classification head, mirroring the
CAMP_full design on top of the TAU (SimVP-family) architecture.

Why: CAMP_full demonstrated that two auxiliary supervisions — ExGI pixel-space
regression and crop-type classification — improve plant-specific metrics on a
PredRNN backbone. To test whether this contribution is backbone-agnostic, we
apply the same two auxiliary heads to TAU (the current SOTA non-recurrent
backbone). This file adds only the classification head to the model. The ExGI
regression loss is computed in the method wrapper on pixel-space predictions
(no model change required for that).

Architecture:
    input frames
        │
        ▼
    Encoder (SimVP)
        │
        ▼
    Translator (TAUSubBlocks stacked via MidMetaNet)   ← tap point for cls head
        │                                              ├─ avg over T
        │                                              ├─ GAP over (H', W')
        │                                              ├─ FC → ReLU → Dropout → FC
        │                                              └─ cls_logit [B, num_classes]
        ▼
    Decoder (SimVP)
        │
        ▼
    predicted future frames  [B, T, C, H, W]

The classification head structure matches CAMP_Model's head exactly (GAP →
Linear → ReLU → Dropout → Linear) so TAU_full vs CAMP_full is a clean
apples-to-apples comparison isolating the *backbone* effect.
"""

import torch
import torch.nn as nn

from .simvp_model import SimVP_Model


class TAU_Full_Model(SimVP_Model):
    r"""TAU backbone (SimVP_Model with model_type='tau') + classification head.

    The classification head taps the translator output (the most semantic
    latent in the SimVP family) rather than the decoded pixel-space prediction,
    so it sees a compact spatiotemporal representation of the whole input.

    Args:
        in_shape      (T, C, H, W) — dataset sequence shape.
        hid_S         spatial hidden channels (default 16).
        hid_T         translator hidden channels (default 256).
        N_S           number of encoder/decoder ConvSC blocks (default 4).
        N_T           number of MidMetaNet (TAU) blocks (default 4).
        cls_hidden    hidden size of the cls head intermediate FC (default 256).
        cls_dropout   dropout inside the cls head (default 0.3).
        num_classes   cls head output dim (default 1 — binary).

        Other args are forwarded to SimVP_Model unchanged.
    """

    def __init__(self, in_shape, hid_S=16, hid_T=256, N_S=4, N_T=4,
                 mlp_ratio=8., drop=0.0, drop_path=0.0,
                 spatio_kernel_enc=3, spatio_kernel_dec=3, act_inplace=True,
                 cls_hidden=256, cls_dropout=0.3, num_classes=1,
                 detach_cls=False,
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

        # When detach_cls=True, the cls head reads from a *detached* copy of
        # the translator latent — gradients from cls_loss train cls_head only,
        # never the backbone. This makes cls a "linear probe" alongside
        # prediction, with no risk of feature corruption.
        self.detach_cls = detach_cls

        # Classification head — mirrors CAMP_Model's head structure
        #   GAP → Flatten → Linear(hid_S, cls_hidden) → ReLU → Dropout → Linear(cls_hidden, num_classes)
        # No sigmoid; BCEWithLogitsLoss applied externally for numerical stability.
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

        # Classification head taps translator output: avg over T → GAP → FC stack.
        # If detach_cls=True, block gradients from flowing back into the backbone
        # (cls_head trains as a linear probe; backbone trains as if no cls existed).
        cls_input = hid_full.detach() if self.detach_cls else hid_full
        cls_feat = cls_input.mean(dim=1)                    # [B, hid_S, H', W']
        cls_logit = self.cls_head(cls_feat)                 # [B, num_classes]

        # Decoder pathway — identical to SimVP_Model
        hid = hid_full.reshape(B * T, C_, H_, W_)
        Y = self.dec(hid, skip)
        Y = Y.reshape(B, T, C, H, W)

        return Y, cls_logit
