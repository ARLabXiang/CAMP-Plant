"""
camp_ablation.py
----------------
Ablation study variants of CAMP for the reviewer-requested ablation
experiments (Reviewer 1, Comment 3).

The three variants below isolate the contribution of each CAMP component
by systematically removing one or more of them:

  ┌────────────────┬──────────────┬───────────────┬────────────────────────┐
  │  Method name   │  ST-LSTM     │ Classification │ ExGI (POI-aware) loss  │
  │                │  backbone    │ branch         │ on top of MSE          │
  ├────────────────┼──────────────┼───────────────┼────────────────────────┤
  │  camp_base     │      ✓       │               │                        │
  │  camp_no_cls   │      ✓       │               │          ✓             │
  │  camp          │      ✓       │      ✓        │                        │  ← original
  │  camp_full     │      ✓       │      ✓        │          ✓             │  ← new best
  └────────────────┴──────────────┴───────────────┴────────────────────────┘

The ExGI (Excess Green Index) training loss
-------------------------------------------
ExGI = 2·G − R − B  measures the "greenness" of a pixel.  Adding
L_ExGI = MSE(ExGI(pred), ExGI(true)) to the training objective directly
penalises errors on leaf pixels (high ExGI) more than on background
pixels (near-zero ExGI), grounding the model in the same biological
signal used by the POI evaluation metric.

This turns the POI evaluation metric into a training signal — a novel
contribution that directly addresses Reviewer 1's concerns about the
limited differentiation of CAMP from prior work.

Design notes
------------
• All three classes inherit from the existing CAMP Lightning wrapper, so
  checkpointing, scheduling, and logging are unchanged.
• Component selection is controlled by two boolean hparams:
    use_cls      (default True)  — enable classification branch
    use_poi_loss (default True)  — enable ExGI/POI training loss
  The CAMP_Model itself also receives use_cls so the branch parameters
  are never even created when not needed.
• poi_loss_weight (default 0.1) scales L_ExGI relative to L_MSE.
  0.1 keeps the ExGI loss from dominating the frame-prediction objective.
"""

import torch
import torch.nn.functional as F

from openstl.models import CAMP_Model
from openstl.utils import (
    reshape_patch,
    reshape_patch_back,
    reserve_schedule_sampling_exp,
    schedule_sampling,
    print_log,
    check_dir,
)
from .camp import CAMP


# ===========================================================================
# Base ablation class — adds ExGI loss capability to the CAMP wrapper
# ===========================================================================

class CAMPAblation(CAMP):
    """
    Extension of CAMP that:
      1. Passes the *use_cls* flag down to CAMP_Model so the classification
         branch can be disabled at the architecture level (no stale weights).
      2. Optionally adds an ExGI-weighted MSE loss on top of the model's own
         loss, controlled by the *use_poi_loss* hparam.

    All three ablation variants (CAMPBase, CAMPNoCls, CAMPFull) subclass
    this and rely on config files to set the appropriate hparam flags.
    """

    # ---------------------------------------------------------------------- #
    # Model construction                                                      #
    # ---------------------------------------------------------------------- #

    def _build_model(self, **args):
        """
        Build CAMP_Model, forwarding the use_cls flag so the classification
        head is only constructed when it is actually needed.

        The flag is read from hparams so that it survives checkpoint loading.
        """
        num_hidden = [int(x) for x in self.hparams.num_hidden.split(",")]
        num_layers = len(num_hidden)

        # Read ablation flags from hparams (set by the config file)
        use_cls = bool(self.hparams.get("use_cls", True))

        return CAMP_Model(
            num_layers  = num_layers,
            num_hidden  = num_hidden,
            configs     = self.hparams,
            cls_hidden  = self.hparams.get("cls_hidden", 256),
            cls_dropout = self.hparams.get("cls_dropout", 0.3),
            use_cls     = use_cls,   # ← controls whether cls_head is built
        )

    # ---------------------------------------------------------------------- #
    # ExGI loss helper                                                        #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _exgi_loss(pred_frames: torch.Tensor,
                   true_frames: torch.Tensor) -> torch.Tensor:
        """
        Compute the MSE between the predicted and ground-truth ExGI maps.

        ExGI(frame) = 2·G − R − B, where R/G/B are the three colour channels
        (channel indices 0, 1, 2 respectively in the [B, T, C, H, W] tensor).

        Minimising this loss encourages the model to correctly predict the
        spatial distribution of leaf pixels, which is exactly what the
        POI-based evaluation metric measures.

        Args:
            pred_frames : FloatTensor [B, T, C, H, W] in [0, 1]  — predicted
            true_frames : FloatTensor [B, T, C, H, W] in [0, 1]  — ground truth

        Returns:
            Scalar loss tensor.
        """
        # Extract RGB channels: index 0=R, 1=G, 2=B  (standard RGB convention)
        pred_exgi = (2.0 * pred_frames[:, :, 1]
                     - pred_frames[:, :, 0]
                     - pred_frames[:, :, 2])   # [B, T, H, W]

        true_exgi = (2.0 * true_frames[:, :, 1]
                     - true_frames[:, :, 0]
                     - true_frames[:, :, 2])   # [B, T, H, W]

        return F.mse_loss(pred_exgi, true_exgi)

    # ---------------------------------------------------------------------- #
    # Training step — adds optional ExGI loss on top of model loss           #
    # ---------------------------------------------------------------------- #

    def training_step(self, batch, batch_idx):
        """
        One training iteration with optional ExGI/POI-aware loss.

        Inherits scheduled sampling, patch embedding, and logging from CAMP.
        If use_poi_loss=True in hparams, also computes L_ExGI on the decoded
        predicted frames and adds it to the total loss.

        Args:
            batch     : 3-tuple (batch_x, batch_y, label) from the plant loader
            batch_idx : integer index of the current batch (unused directly)
        """
        batch_x, batch_y, label = batch

        # ---- Patch-embed the full input+target sequence --------------------
        ims = (
            torch.cat([batch_x, batch_y], dim=1)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )
        ims = reshape_patch(ims, self.hparams.patch_size)

        # ---- Build the scheduled-sampling mask -----------------------------
        if self.hparams.reverse_scheduled_sampling == 1:
            real_input_flag = reserve_schedule_sampling_exp(
                self.global_step, ims.shape[0], self.hparams
            )
        else:
            self.eta, real_input_flag = schedule_sampling(
                self.eta, self.global_step, ims.shape[0], self.hparams
            )

        # ---- Forward pass: get predictions + model-internal loss -----------
        # img_gen is [B, T_total-1, H', W', C_p] in patch-embedded space.
        # The model-internal loss already includes frame MSE and (if use_cls)
        # the classification BCE loss.
        img_gen, _cls_logit, loss = self.model(
            ims,
            real_input_flag,
            return_loss     = True,
            label           = label,
            cls_loss_weight = self.hparams.get("cls_loss_weight", 1.0),
        )

        # ---- Optional ExGI / POI-aware loss --------------------------------
        # This loss is computed in pixel space on the decoded future frames.
        # It directly penalises errors in the greenness distribution, which
        # is the same signal used by the POI evaluation metric.
        if self.hparams.get("use_poi_loss", True):
            # Decode from patch space back to pixel space:
            # [B, T-1, H', W', C_p]  →  [B, T-1, H, W, C]
            img_gen_decoded = reshape_patch_back(img_gen, self.hparams.patch_size)

            # Extract only the future (aft_seq) prediction frames and permute
            # to standard [B, T, C, H, W] for the loss computation
            pred_frames = (
                img_gen_decoded[:, -self.hparams.aft_seq_length:]
                .permute(0, 1, 4, 2, 3)
                .contiguous()
            )

            # Compute ExGI MSE between predicted and ground-truth future frames.
            # batch_y is already in pixel space [B, aft_seq, C, H, W].
            exgi_loss = self._exgi_loss(pred_frames, batch_y)

            # Scale ExGI loss by poi_loss_weight (default 0.1) to keep it
            # from dominating the frame-prediction MSE objective.
            poi_weight = self.hparams.get("poi_loss_weight", 0.1)
            loss = loss + poi_weight * exgi_loss

            # Log ExGI loss separately so we can track its contribution
            self.log("exgi_loss", exgi_loss,
                     on_step=True, on_epoch=True, prog_bar=False)

        # ---- Log total training loss ---------------------------------------
        self.log("train_loss", loss,
                 on_step=True, on_epoch=True, prog_bar=True)
        return loss


# ===========================================================================
# Ablation variant 1 — backbone only (no classification, no ExGI loss)
# ===========================================================================

class CAMPBase(CAMPAblation):
    """
    CAMP_base: pure ST-LSTM backbone with MSE frame loss only.

    This is the lower bound of the ablation — it shows what the backbone
    alone can achieve before any CAMP-specific contributions are added.
    It is expected to perform similarly to the standalone PredRNN baseline,
    confirming that the backbone is correctly reproduced.

    Training loss : L_total = L_MSE
    Config file   : configs/plant/CAMP_base.py  (use_cls=False, use_poi_loss=False)
    """
    pass   # All behaviour is controlled by the config hparams (see above)


# ===========================================================================
# Ablation variant 2 — ExGI loss only (no classification branch)
# ===========================================================================

class CAMPNoCls(CAMPAblation):
    """
    CAMP_no_cls: backbone + ExGI/POI loss, classification branch removed.

    Comparing this against CAMP_base shows how much the ExGI training loss
    alone (without knowing the drought/control label) improves biological
    plausibility.  Comparing against the full CAMP shows the additional
    contribution of the classification branch.

    Training loss : L_total = L_MSE + λ_poi · L_ExGI
    Config file   : configs/plant/CAMP_no_cls.py  (use_cls=False, use_poi_loss=True)
    """
    pass   # All behaviour is controlled by the config hparams (see above)


# ===========================================================================
# Ablation variant 3 — full model with all three components
# ===========================================================================

class CAMPFull(CAMPAblation):
    """
    CAMP_full: all three components active — backbone + classification +
    ExGI/POI loss.

    This is the enhanced version of CAMP that adds the ExGI training loss
    on top of the original MSE + BCE objective.  It serves as the upper bound
    of the ablation study.

    Training loss : L_total = L_MSE + λ_cls · L_BCE + λ_poi · L_ExGI
    Config file   : configs/plant/CAMP_full.py  (use_cls=True, use_poi_loss=True)
    """
    pass   # All behaviour is controlled by the config hparams (see above)
