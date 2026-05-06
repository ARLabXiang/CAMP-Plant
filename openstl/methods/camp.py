"""
camp.py
-------
PyTorch Lightning training/evaluation wrapper for the CAMP model.

CAMP extends the standard PredRNN training loop with:
  • Multi-task loss: L_total = L_MSE(frames) + λ · L_BCE(classification)
  • Support for 3-element batches: (input_frames, target_frames, label)
  • Override of validation_step and test_step to handle the extra label
  • Scheduled sampling for the frame-prediction branch (inherited logic from
    the original PredRNN implementation)
  • Saving of predicted frames, ground-truth frames, and classification
    logits to .npy files after testing (for downstream POI evaluation)

The class inherits from Base_method (a LightningModule) and only overrides
the methods that need to change; everything else (optimizer, scheduler,
logging, checkpointing) comes from Base_method for free.
"""

import numpy as np
import torch

from openstl.models import CAMP_Model
from openstl.utils import (
    reshape_patch,
    reshape_patch_back,
    reserve_schedule_sampling_exp,
    schedule_sampling,
    print_log,
    check_dir,
)
from .base_method import Base_method

import os.path as osp


class CAMP(Base_method):
    """
    Classification-Assisted Multi-task Prediction — Lightning Method.

    Expected hparams (on top of the standard Base_method hparams):
        num_hidden              : comma-separated ints, e.g. '128,128,128,128'
        filter_size             : kernel size for ST-LSTM convolutions (e.g. 5)
        stride                  : convolution stride (e.g. 1)
        patch_size              : patch embedding size (e.g. 2)
        layer_norm              : 0 or 1
        reverse_scheduled_sampling : 0 or 1
        r_sampling_step_1       : iteration for reverse-SS phase 1 end
        r_sampling_step_2       : iteration for reverse-SS phase 2 end
        r_exp_alpha             : decay factor for reverse-SS
        scheduled_sampling      : 0 or 1
        sampling_stop_iter      : iteration at which SS probability reaches 0
        sampling_start_value    : initial probability of using ground truth
        sampling_changing_rate  : per-step change in ground-truth probability
        cls_loss_weight         : λ weight on the classification loss (default 1.0)
        cls_hidden              : hidden size of the classification FC head (256)
        cls_dropout             : dropout rate in the classification head (0.3)
    """

    def __init__(self, **args):
        super().__init__(**args)
        # η is used by standard (non-reverse) scheduled sampling; it decays
        # from sampling_start_value toward 0 as training progresses.
        self.eta = 1.0

    # ---------------------------------------------------------------------- #
    # Model construction                                                      #
    # ---------------------------------------------------------------------- #

    def _build_model(self, **args):
        """Instantiate the CAMP_Model from hparams."""
        # Parse the comma-separated num_hidden string into a list of ints
        num_hidden = [int(x) for x in self.hparams.num_hidden.split(",")]
        num_layers = len(num_hidden)

        return CAMP_Model(
            num_layers     = num_layers,
            num_hidden     = num_hidden,
            configs        = self.hparams,
            cls_hidden     = self.hparams.get("cls_hidden", 256),
            cls_dropout    = self.hparams.get("cls_dropout", 0.3),
        )

    # ---------------------------------------------------------------------- #
    # Shared forward helper                                                   #
    # ---------------------------------------------------------------------- #

    def _preprocess_batch(self, batch_x, batch_y):
        """
        Concatenate input and target along the time axis, apply patch
        embedding (reshape_patch), and build the scheduled-sampling mask
        that will be zeros during inference (no teacher forcing).

        Returns:
            test_dat          : patch-embedded full sequence [B,T,H,W,C_p]
            real_input_flag   : all-zeros mask  [B, aft_len-1, H', W', C_p]
        """
        _, img_channel, img_height, img_width = self.hparams.in_shape

        # Determine how many leading frames always use ground-truth
        mask_input = (
            1 if self.hparams.reverse_scheduled_sampling == 1
            else self.hparams.pre_seq_length
        )

        # Concatenate observed + future along time: [B, T_total, C, H, W]
        # then permute to [B, T_total, H, W, C] for PredRNN convention
        test_ims = (
            torch.cat([batch_x, batch_y], dim=1)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )
        # Apply patch embedding: [B, T, H, W, C] → [B, T, H', W', C_p]
        test_dat = reshape_patch(test_ims, self.hparams.patch_size)

        # Build an all-zeros mask (no scheduled sampling during inference)
        real_input_flag = torch.zeros(
            (
                batch_x.shape[0],
                self.hparams.total_length - mask_input - 1,
                img_height // self.hparams.patch_size,
                img_width  // self.hparams.patch_size,
                self.hparams.patch_size ** 2 * img_channel,
            ),
            device=self.device,
        )

        # For reverse scheduled sampling, the first (pre_seq-1) frames use GT
        if self.hparams.reverse_scheduled_sampling == 1:
            real_input_flag[:, : self.hparams.pre_seq_length - 1] = 1.0

        return test_dat, real_input_flag

    # ---------------------------------------------------------------------- #
    # Lightning forward (used by validation_step and test_step)              #
    # ---------------------------------------------------------------------- #

    def forward(self, batch_x, batch_y, **kwargs):
        """
        Run the model in inference mode (no teacher forcing).

        Returns:
            pred_y    : predicted future frames [B, aft_seq, C, H, W]
            cls_logit : classification logit    [B, 1]
        """
        test_dat, real_input_flag = self._preprocess_batch(batch_x, batch_y)

        # Run CAMP (no loss computation during inference)
        img_gen, cls_logit, _ = self.model(
            test_dat, real_input_flag, return_loss=False
        )

        # Undo patch embedding: [B, T-1, H', W', C_p] → [B, T-1, H, W, C]
        img_gen = reshape_patch_back(img_gen, self.hparams.patch_size)

        # Extract only the future prediction frames and permute back to
        # [B, aft_seq, C, H, W] for metric computation
        pred_y = (
            img_gen[:, -self.hparams.aft_seq_length :]
            .permute(0, 1, 4, 2, 3)
            .contiguous()
        )

        return pred_y, cls_logit

    # ---------------------------------------------------------------------- #
    # Training step                                                           #
    # ---------------------------------------------------------------------- #

    def training_step(self, batch, batch_idx):
        """
        One training iteration.

        The batch contains a 3-tuple: (batch_x, batch_y, label).
        We compute the multi-task loss and log its components.
        """
        batch_x, batch_y, label = batch   # unpack the classification label

        # ---- Patch-embed the full sequence ---------------------------------
        ims = (
            torch.cat([batch_x, batch_y], dim=1)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )
        ims = reshape_patch(ims, self.hparams.patch_size)

        # ---- Build scheduled-sampling mask ---------------------------------
        if self.hparams.reverse_scheduled_sampling == 1:
            real_input_flag = reserve_schedule_sampling_exp(
                self.global_step, ims.shape[0], self.hparams
            )
        else:
            self.eta, real_input_flag = schedule_sampling(
                self.eta, self.global_step, ims.shape[0], self.hparams
            )

        # ---- Forward pass with loss ----------------------------------------
        _, _, loss = self.model(
            ims,
            real_input_flag,
            return_loss       = True,
            label             = label,
            cls_loss_weight   = self.hparams.get("cls_loss_weight", 1.0),
        )

        # Log the combined training loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # ---------------------------------------------------------------------- #
    # Validation step                                                         #
    # ---------------------------------------------------------------------- #

    def validation_step(self, batch, batch_idx):
        """
        Validation uses inference mode (no teacher forcing).

        We report the frame-reconstruction MSE loss only (same metric as the
        baselines) so that the best-checkpoint selection remains comparable.
        """
        batch_x, batch_y, _label = batch   # label not used for val metric

        pred_y, _cls_logit = self(batch_x, batch_y)

        # Frame-level MSE loss (matches Base_method's criterion)
        loss = self.criterion(pred_y, batch_y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    # ---------------------------------------------------------------------- #
    # Test step                                                               #
    # ---------------------------------------------------------------------- #

    def test_step(self, batch, batch_idx):
        """
        Test step collects predicted frames, ground-truth frames, input
        frames, and classification logits for later metric computation and
        POI evaluation.
        """
        batch_x, batch_y, label = batch

        pred_y, cls_logit = self(batch_x, batch_y)

        # Accumulate outputs (moved to CPU to save GPU memory)
        outputs = {
            "inputs"    : batch_x.cpu().numpy(),    # [B, pre_seq, C, H, W]
            "preds"     : pred_y.cpu().numpy(),     # [B, aft_seq, C, H, W]
            "trues"     : batch_y.cpu().numpy(),    # [B, aft_seq, C, H, W]
            "cls_logits": cls_logit.cpu().numpy(),  # [B, 1]
            "labels"    : label.cpu().numpy(),      # [B, 1]
        }
        self.test_outputs.append(outputs)
        return outputs

    # ---------------------------------------------------------------------- #
    # End-of-test aggregation                                                 #
    # ---------------------------------------------------------------------- #

    def on_test_epoch_end(self):
        """
        Concatenate all per-batch test outputs, compute image metrics, log
        them, and save numpy arrays for downstream POI evaluation.
        """
        from openstl.core import metric

        # Concatenate along the batch dimension
        results_all = {}
        for k in self.test_outputs[0].keys():
            results_all[k] = np.concatenate(
                [batch[k] for batch in self.test_outputs], axis=0
            )

        # Compute standard image similarity metrics (MAE, MSE, SSIM, PSNR)
        eval_res, eval_log = metric(
            results_all["preds"],
            results_all["trues"],
            self.hparams.test_mean,
            self.hparams.test_std,
            metrics      = self.metric_list,
            channel_names= self.channel_names,
            spatial_norm = self.spatial_norm,
            threshold    = self.hparams.get("metric_threshold", None),
        )
        results_all["metrics"] = np.array([eval_res["mae"], eval_res["mse"]])

        # Classification accuracy (exclude samples with label == -1)
        labels   = results_all["labels"].squeeze(-1)          # [N]
        logits   = results_all["cls_logits"].squeeze(-1)      # [N]
        valid    = labels >= 0
        if valid.any():
            preds_cls = (logits[valid] > 0).astype(float)     # threshold at 0
            acc = (preds_cls == labels[valid]).mean()
            print_log(f"Classification accuracy: {acc * 100:.2f}%")
            results_all["cls_acc"] = np.array([acc])

        # Only the rank-0 process writes to disk in distributed training
        if self.trainer.is_global_zero:
            print_log(eval_log)
            folder_path = check_dir(
                osp.join(self.hparams.save_dir, "saved")
            )
            # Save arrays for POI evaluation and visualisation
            for key in ["metrics", "inputs", "trues", "preds",
                        "cls_logits", "labels"]:
                if key in results_all:
                    np.save(
                        osp.join(folder_path, key + ".npy"),
                        results_all[key],
                    )

        return results_all
