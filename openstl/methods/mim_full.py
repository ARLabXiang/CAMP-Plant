"""
mim_full.py
-----------
MIM_full method: MIM backbone + ExGI regression loss + binary classification head.

Training loss:
    L_total = L_MSE(pred_patch, true_patch)             # inside MIM_Full_Model
            + poi_loss_weight * L_ExGI(pred_pix, true_pix)  # CAMP ExGI
            + cls_loss_weight * L_BCE(cls_logit, label)     # CAMP classification

Plant batches are 3-tuples (batch_x, batch_y, label); this method accepts the
label directly.  Labels of -1 are masked out of the classification loss.

The ExGI loss is computed on pixel-space (decoded) future frames.  The model
returns patch-space predictions, so we call reshape_patch_back before applying
ExGI — the same pattern CAMPAblation uses for its patch-based backbone.
"""

import torch
import torch.nn as nn

from openstl.models.mim_full_model import MIM_Full_Model
from openstl.utils import (
    reshape_patch, reshape_patch_back,
    reserve_schedule_sampling_exp, schedule_sampling,
)
from .mim import MIM
from .camp_ablation import CAMPAblation


class MIMFull(MIM):
    r"""MIM backbone with ExGI regression loss and binary classification head."""

    def __init__(self, **args):
        super().__init__(**args)
        self.bce_criterion = nn.BCEWithLogitsLoss()

    def _build_model(self, **args):
        num_hidden = [int(x) for x in self.hparams.num_hidden.split(',')]
        num_layers = len(num_hidden)
        return MIM_Full_Model(num_layers, num_hidden, self.hparams)

    # ------------------------------------------------------------------ #
    # forward — discards cls_logit for inference (matches PredRNN API)   #
    # ------------------------------------------------------------------ #
    def forward(self, batch_x, batch_y, **kwargs):
        """Inference forward — same patch/mask logic as PredRNN.forward()."""
        if self.hparams.reverse_scheduled_sampling == 1:
            mask_input = 1
        else:
            mask_input = self.hparams.pre_seq_length
        _, img_channel, img_height, img_width = self.hparams.in_shape

        test_ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()
        test_dat = reshape_patch(test_ims, self.hparams.patch_size)

        real_input_flag = torch.zeros(
            (batch_x.shape[0],
             self.hparams.total_length - mask_input - 1,
             img_height // self.hparams.patch_size,
             img_width  // self.hparams.patch_size,
             self.hparams.patch_size ** 2 * img_channel)).to(self.device)

        if self.hparams.reverse_scheduled_sampling == 1:
            real_input_flag[:, :self.hparams.pre_seq_length - 1, :, :] = 1.0

        img_gen, _cls_logit, _loss = self.model(
            test_dat, real_input_flag, return_loss=False)
        img_gen = reshape_patch_back(img_gen, self.hparams.patch_size)
        pred_y = img_gen[:, -self.hparams.aft_seq_length:].permute(0, 1, 4, 2, 3).contiguous()
        return pred_y

    # ------------------------------------------------------------------ #
    # training_step — 3-tuple batch, 3-term loss                         #
    # ------------------------------------------------------------------ #
    def training_step(self, batch, batch_idx):
        batch_x, batch_y, label = batch

        # Patch-embed the full sequence (same as PredRNN / CAMP training)
        ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()
        ims = reshape_patch(ims, self.hparams.patch_size)

        # Scheduled-sampling mask
        if self.hparams.reverse_scheduled_sampling == 1:
            real_input_flag = reserve_schedule_sampling_exp(
                self.global_step, ims.shape[0], self.hparams)
        else:
            self.eta, real_input_flag = schedule_sampling(
                self.eta, self.global_step, ims.shape[0], self.hparams)

        # Forward pass — model-internal MSE on patch-space frames
        img_gen, cls_logit, frame_loss = self.model(
            ims, real_input_flag, return_loss=True)

        total_loss = frame_loss

        # ExGI regression loss — needs pixel-space future frames
        if self.hparams.get("use_poi_loss", True):
            img_gen_decoded = reshape_patch_back(img_gen, self.hparams.patch_size)
            pred_frames = (
                img_gen_decoded[:, -self.hparams.aft_seq_length:]
                .permute(0, 1, 4, 2, 3)
                .contiguous()
            )
            exgi_loss = CAMPAblation._exgi_loss(pred_frames, batch_y)
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
                cls_loss = torch.tensor(0.0, device=batch_y.device)
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
