"""
simvp_predcls.py
----------------
SimVP_PredCls method: SimVP backbone + ExGI loss + late-fusion classification.

SimVP counterpart of TAUPredCls (same architectural pattern, no DDR loss).

Training loss:
    L_total = L_MSE(pred, true)                         # pixel reconstruction
            + poi_loss_weight * L_ExGI(pred, true)      # CAMP ExGI supervision
            + cls_loss_weight * L_BCE(cls_logit, label) # late-fusion classification
"""

import torch
import torch.nn as nn

from .simvp import SimVP
from .camp_ablation import CAMPAblation
from openstl.models.simvp_predcls_model import SimVP_PredCls_Model


class SimVPPredCls(SimVP):
    r"""SimVP backbone with late-fusion cls head reading from pred_y."""

    def __init__(self, **args):
        super().__init__(**args)
        self.bce_criterion = nn.BCEWithLogitsLoss()

    def _build_model(self, **args):
        args = dict(args)
        args.setdefault("cls_hidden",  self.hparams.get("cls_hidden",  128))
        args.setdefault("cls_dropout", self.hparams.get("cls_dropout", 0.3))
        args.setdefault("num_classes", self.hparams.get("num_classes", 1))
        return SimVP_PredCls_Model(**args)

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

    def training_step(self, batch, batch_idx):
        batch_x, batch_y, label = batch
        pred_y, cls_logit = self.model(batch_x)

        # Base MSE loss (no DDR for SimVP — only TAU has it)
        mse_loss = self.criterion(pred_y, batch_y)
        total_loss = mse_loss

        # ExGI loss
        if self.hparams.get("use_poi_loss", True):
            exgi_loss = CAMPAblation._exgi_loss(pred_y, batch_y)
            poi_weight = self.hparams.get("poi_loss_weight", 0.1)
            total_loss = total_loss + poi_weight * exgi_loss
            self.log("exgi_loss", exgi_loss, on_step=True, on_epoch=True)

        # Late-fusion classification loss
        if self.hparams.get("use_cls", True):
            valid_mask = (label.squeeze(-1) >= 0)
            if valid_mask.any():
                valid_logits = cls_logit[valid_mask]
                valid_labels = label[valid_mask].float()
                cls_loss = self.bce_criterion(valid_logits, valid_labels)
                with torch.no_grad():
                    pred_class = (torch.sigmoid(valid_logits) > 0.5).float()
                    cls_acc = (pred_class == valid_labels).float().mean()
            else:
                cls_loss = torch.tensor(0.0, device=pred_y.device)
                cls_acc  = torch.tensor(0.0, device=pred_y.device)
            cls_weight = self.hparams.get("cls_loss_weight", 1.0)
            total_loss = total_loss + cls_weight * cls_loss
            self.log("cls_loss", cls_loss, on_step=True, on_epoch=True)
            self.log("cls_acc", cls_acc, on_step=True, on_epoch=True, prog_bar=True)

        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y, _label = batch
        pred_y = self(batch_x, batch_y)
        loss = self.criterion(pred_y, batch_y)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
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
