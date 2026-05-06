"""
tau_full.py
-----------
TAU_full method: TAU backbone + ExGI regression loss + binary classification
head.  This is the TAU-backbone counterpart of CAMP_full.

Training loss:
    L_total = L_MSE(pred, true)                        # pixel reconstruction
            + alpha          * L_DiffDivReg(pred,true) # TAU native regularizer
            + poi_loss_weight * L_ExGI(pred, true)     # CAMP ExGI supervision
            + cls_loss_weight * L_BCE(cls_logit,label) # CAMP classification

The ExGI loss is reused verbatim from CAMPAblation._exgi_loss, ensuring the
supervision signal is identical across backbones — the only thing that differs
between CAMP_full and TAU_full is the backbone.  This makes the two directly
comparable and tests whether CAMP's contribution is backbone-agnostic.

Plant batches are 3-tuples (batch_x, batch_y, label); this method consumes the
label directly (no PlantMixin needed).  Labels of -1 are masked out of the
classification loss (Komatsuna has no valid labels — though this config is
intended for Arabidopsis).
"""

import torch
import torch.nn as nn

from .tau import TAU
from .camp_ablation import CAMPAblation
from openstl.models.tau_full_model import TAU_Full_Model


class TAUFull(TAU):
    r"""TAU backbone with ExGI regression loss and binary classification head.

    Config hparams consulted:
        use_poi_loss     (bool,  default True)  — enable ExGI MSE loss
        poi_loss_weight  (float, default 0.1)   — weight for ExGI loss
        use_cls          (bool,  default True)  — enable classification loss
        cls_loss_weight  (float, default 1.0)   — weight for classification loss
        cls_hidden       (int,   default 256)   — hidden dim of cls head FC
        cls_dropout      (float, default 0.3)   — dropout in cls head
        num_classes      (int,   default 1)     — cls head output dim
        alpha            (float, default 0.1)   — TAU native DDR weight (inherited)
    """

    def __init__(self, **args):
        super().__init__(**args)
        # Binary classification loss with built-in sigmoid (numerically stable)
        self.bce_criterion = nn.BCEWithLogitsLoss()

    def _build_model(self, **args):
        # Forward classification-head hparams to the model if not already present.
        # This keeps CAMP_full-compatible defaults when the config omits them.
        args = dict(args)
        args.setdefault("cls_hidden",  self.hparams.get("cls_hidden",  256))
        args.setdefault("cls_dropout", self.hparams.get("cls_dropout", 0.3))
        args.setdefault("num_classes", self.hparams.get("num_classes", 1))
        args.setdefault("detach_cls",  self.hparams.get("detach_cls",  False))
        return TAU_Full_Model(**args).to(self.device)

    # ------------------------------------------------------------------ #
    # forward — discards cls_logit for inference (matches SimVP API)     #
    # ------------------------------------------------------------------ #
    def forward(self, batch_x, batch_y=None, **kwargs):
        pre_seq_length  = self.hparams.pre_seq_length
        aft_seq_length  = self.hparams.aft_seq_length

        if aft_seq_length == pre_seq_length:
            pred_y, _ = self.model(batch_x)
        elif aft_seq_length < pre_seq_length:
            pred_y, _ = self.model(batch_x)
            pred_y = pred_y[:, :aft_seq_length]
        else:
            # aft_seq_length > pre_seq_length: autoregressive rollout
            pred_y_chunks = []
            d = aft_seq_length // pre_seq_length
            m = aft_seq_length %  pre_seq_length
            cur_seq = batch_x.clone()
            for _ in range(d):
                cur_seq, _ = self.model(cur_seq)
                pred_y_chunks.append(cur_seq)
            if m != 0:
                cur_seq, _ = self.model(cur_seq)
                pred_y_chunks.append(cur_seq[:, :m])
            pred_y = torch.cat(pred_y_chunks, dim=1)

        return pred_y

    # ------------------------------------------------------------------ #
    # training_step — 3-tuple batch, 4-term loss                         #
    # ------------------------------------------------------------------ #
    def training_step(self, batch, batch_idx):
        batch_x, batch_y, label = batch
        pred_y, cls_logit = self.model(batch_x)

        # Base losses (MSE + TAU native differential divergence regularizer)
        mse_loss = self.criterion(pred_y, batch_y)
        ddr_loss = self.diff_div_reg(pred_y, batch_y)
        total_loss = mse_loss + self.hparams.alpha * ddr_loss

        # ExGI regression loss (reused verbatim from CAMPAblation)
        if self.hparams.get("use_poi_loss", True):
            exgi_loss = CAMPAblation._exgi_loss(pred_y, batch_y)
            poi_weight = self.hparams.get("poi_loss_weight", 0.1)
            total_loss = total_loss + poi_weight * exgi_loss
            self.log("exgi_loss", exgi_loss,
                     on_step=True, on_epoch=True, prog_bar=False)

        # Binary classification loss — only on samples with valid labels (>= 0)
        if self.hparams.get("use_cls", True):
            # label shape: [B, 1]; -1 encodes "unlabeled" (Komatsuna).
            valid_mask = (label.squeeze(-1) >= 0)
            if valid_mask.any():
                valid_logits = cls_logit[valid_mask]
                valid_labels = label[valid_mask].float()
                cls_loss = self.bce_criterion(valid_logits, valid_labels)
                # Track classification accuracy on the valid portion
                with torch.no_grad():
                    pred_class = (torch.sigmoid(valid_logits) > 0.5).float()
                    cls_acc = (pred_class == valid_labels).float().mean()
            else:
                cls_loss = torch.tensor(0.0, device=pred_y.device)
                cls_acc  = torch.tensor(0.0, device=pred_y.device)
            cls_weight = self.hparams.get("cls_loss_weight", 1.0)
            total_loss = total_loss + cls_weight * cls_loss
            self.log("cls_loss", cls_loss,
                     on_step=True, on_epoch=True, prog_bar=False)
            self.log("cls_acc", cls_acc,
                     on_step=True, on_epoch=True, prog_bar=True)

        self.log("train_loss", total_loss,
                 on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    # ------------------------------------------------------------------ #
    # validation_step / test_step — strip label, use inference forward   #
    # ------------------------------------------------------------------ #
    def validation_step(self, batch, batch_idx):
        batch_x, batch_y, _label = batch
        pred_y = self(batch_x, batch_y)
        loss = self.criterion(pred_y, batch_y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def test_step(self, batch, batch_idx):
        batch_x, batch_y, _label = batch
        pred_y = self(batch_x, batch_y)
        outputs = {
            'inputs': batch_x.cpu().numpy(),
            'preds':  pred_y.cpu().numpy(),
            'trues':  batch_y.cpu().numpy(),
        }
        self.test_outputs.append(outputs)
        return outputs
