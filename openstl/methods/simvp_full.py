"""
simvp_full.py
-------------
SimVP_full method: SimVP backbone + ExGI regression loss + binary classification
head.  SimVP-backbone counterpart of CAMP_full / TAU_full.

Training loss:
    L_total = L_MSE(pred, true)                         # pixel reconstruction
            + poi_loss_weight * L_ExGI(pred, true)      # CAMP ExGI supervision
            + cls_loss_weight * L_BCE(cls_logit,label)  # CAMP classification

Note: unlike TAU_full, SimVP has NO native differential divergence regularizer
— so this method only adds the two CAMP auxiliary losses on top of MSE.

Plant batches are 3-tuples (batch_x, batch_y, label); this method consumes the
label directly.  Labels of -1 are masked out of the classification loss.
"""

import torch
import torch.nn as nn

from .simvp import SimVP
from .camp_ablation import CAMPAblation
from openstl.models.simvp_full_model import SimVP_Full_Model


class SimVPFull(SimVP):
    r"""SimVP backbone with ExGI regression loss and binary classification head."""

    def __init__(self, **args):
        super().__init__(**args)
        self.bce_criterion = nn.BCEWithLogitsLoss()

    def _build_model(self, **args):
        args = dict(args)
        args.setdefault("cls_hidden",  self.hparams.get("cls_hidden",  256))
        args.setdefault("cls_dropout", self.hparams.get("cls_dropout", 0.3))
        args.setdefault("num_classes", self.hparams.get("num_classes", 1))
        return SimVP_Full_Model(**args)

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
    # training_step — 3-tuple batch, 3-term loss (MSE + ExGI + BCE)      #
    # ------------------------------------------------------------------ #
    def training_step(self, batch, batch_idx):
        batch_x, batch_y, label = batch
        pred_y, cls_logit = self.model(batch_x)

        # Base MSE loss
        mse_loss = self.criterion(pred_y, batch_y)
        total_loss = mse_loss

        # ExGI regression loss (reused verbatim from CAMPAblation)
        if self.hparams.get("use_poi_loss", True):
            exgi_loss = CAMPAblation._exgi_loss(pred_y, batch_y)
            poi_weight = self.hparams.get("poi_loss_weight", 0.1)
            total_loss = total_loss + poi_weight * exgi_loss
            self.log("exgi_loss", exgi_loss,
                     on_step=True, on_epoch=True, prog_bar=False)

        # Binary classification loss (masking -1 as invalid)
        if self.hparams.get("use_cls", True):
            valid_mask = (label.squeeze(-1) >= 0)
            if valid_mask.any():
                valid_logits = cls_logit[valid_mask]
                valid_labels = label[valid_mask].float()
                cls_loss = self.bce_criterion(valid_logits, valid_labels)
            else:
                cls_loss = torch.tensor(0.0, device=pred_y.device)
            cls_weight = self.hparams.get("cls_loss_weight", 1.0)
            total_loss = total_loss + cls_weight * cls_loss
            self.log("cls_loss", cls_loss,
                     on_step=True, on_epoch=True, prog_bar=False)

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
