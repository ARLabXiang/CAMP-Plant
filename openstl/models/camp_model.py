"""
camp_model.py
-------------
CAMP (Classification-Assisted Multi-task Prediction) model.

Architecture overview (as described in the paper):
  1. Encoder  — stacked PredRNN (ST-LSTM) cells that process the input
                sequence.  Spatiotemporal memory flows in a zigzag pattern
                across both time and layer depth.
  2. Frame-prediction branch — a final 1×1 convolution that maps the last
                layer's hidden state at each time step back to image space.
                This is identical to the original PredRNN decoder.
  3. Classification branch   — global-average-pools the hidden states
                collected during the *input* phase, then passes them through
                a small fully-connected head to predict a binary
                environmental label (drought vs. control).

The two branches share the same PredRNN encoder, so the classification
signal regularises the spatiotemporal memory and encourages it to retain
physically meaningful information about the environmental condition.

Reference:
    Wang et al. (2017) PredRNN: Recurrent Neural Networks for Predictive
    Learning using Spatiotemporal LSTMs. NeurIPS.
"""

import torch
import torch.nn as nn

from openstl.modules import SpatioTemporalLSTMCell


class CAMP_Model(nn.Module):
    """
    Classification-Assisted Multi-task Prediction model.

    Args:
        num_layers  : Number of stacked ST-LSTM layers.
        num_hidden  : List of hidden channel sizes for each ST-LSTM layer.
        configs     : Namespace / object with the following attributes:
                        in_shape          (T, C, H, W)
                        patch_size        int
                        filter_size       int
                        stride            int
                        layer_norm        bool/int
                        pre_seq_length    int
                        aft_seq_length    int
        cls_hidden  : Hidden size of the intermediate FC layer in the
                      classification head (default: 256).
        cls_dropout : Dropout probability inside the classification head
                      (default: 0.3).
    """

    def __init__(self, num_layers, num_hidden, configs,
                 cls_hidden: int = 256, cls_dropout: float = 0.3,
                 use_cls: bool = True, **kwargs):
        super(CAMP_Model, self).__init__()

        # ------------------------------------------------------------------ #
        # Store configuration                                                 #
        # ------------------------------------------------------------------ #
        T, C, H, W = configs.in_shape

        self.configs        = configs
        self.num_layers     = num_layers
        self.num_hidden     = num_hidden
        # Number of channels after patch embedding (each patch_size×patch_size
        # window of C channels is flattened into one super-pixel)
        self.frame_channel  = configs.patch_size * configs.patch_size * C

        # Spatial size after patch embedding
        height = H // configs.patch_size
        width  = W // configs.patch_size

        # Whether the classification branch is active.  Setting this to False
        # produces the CAMP_no_cls and CAMP_base ablation variants, which use
        # only the frame-prediction loss during training.
        self.use_cls = use_cls

        # ------------------------------------------------------------------ #
        # Encoder: stacked SpatioTemporal LSTM cells                         #
        # ------------------------------------------------------------------ #
        # Each cell takes the hidden state of the cell below (or the input
        # frame for the first cell) and outputs a new hidden state h and two
        # memory tensors (c, m).  The memory tensor m is the "spatiotemporal"
        # memory that flows in a zigzag across both time and depth.
        cell_list = []
        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(
                    in_channel  = in_channel,
                    num_hidden  = num_hidden[i],
                    height      = height,
                    width       = width,
                    filter_size = configs.filter_size,
                    stride      = configs.stride,
                    layer_norm  = configs.layer_norm,
                )
            )
        self.cell_list = nn.ModuleList(cell_list)

        # ------------------------------------------------------------------ #
        # Frame-prediction branch                                             #
        # ------------------------------------------------------------------ #
        # A single 1×1 convolution that maps the last ST-LSTM layer's hidden
        # state back to frame-channel space (inverse of the patch embedding).
        self.conv_last = nn.Conv2d(
            num_hidden[num_layers - 1],
            self.frame_channel,
            kernel_size=1, stride=1, padding=0, bias=False,
        )

        # Reconstruction loss for frame prediction
        self.mse_criterion = nn.MSELoss()

        # ------------------------------------------------------------------ #
        # Classification branch (optional — disabled for ablation variants)  #
        # ------------------------------------------------------------------ #
        # After encoding the *input* frames we global-average-pool the final
        # hidden state (averaged over the pre_seq time steps) to get a
        # [B, num_hidden_last] vector, then pass it through an FC head.
        #
        # Architecture:
        #   GAP → Linear(num_hidden_last, cls_hidden) → ReLU → Dropout
        #       → Linear(cls_hidden, 1) → (sigmoid applied at loss time)
        #
        # When use_cls=False (ablation) the head is not built at all, saving
        # parameters and ensuring it truly has no effect on gradients.
        if self.use_cls:
            self.cls_head = nn.Sequential(
                # Map from spatial feature dimension to a compact vector
                nn.Linear(num_hidden[num_layers - 1], cls_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=cls_dropout),
                # Final binary classification logit (no sigmoid here — BCEWithLogits
                # is used externally for numerical stability)
                nn.Linear(cls_hidden, 1),
            )
            # Binary cross-entropy loss with built-in sigmoid (numerically stable)
            self.bce_criterion = nn.BCEWithLogitsLoss()

    # ---------------------------------------------------------------------- #
    # Forward pass                                                            #
    # ---------------------------------------------------------------------- #

    def forward(self, frames_tensor, mask_true, **kwargs):
        """
        Run the CAMP forward pass.

        Args:
            frames_tensor : FloatTensor [B, T_total, H, W, C]
                            Concatenation of input + target frames
                            (after patch reshape, so C = patch_size² × img_C).
                            T_total = pre_seq_length + aft_seq_length.
            mask_true     : FloatTensor [B, T_mask, H, W, C]
                            Scheduled-sampling mask (1 = use ground-truth,
                            0 = use model prediction).

        Returns:
            next_frames   : FloatTensor [B, T_total-1, H, W, C]
                            Predicted frames for all future time steps.
            cls_logit     : FloatTensor [B, 1]
                            Raw classification logit (before sigmoid).
            loss          : scalar tensor or None
                            Combined MSE + BCE loss if *return_loss* is True
                            (controlled via kwargs), else None.
        """
        # ------------------------------------------------------------------ #
        # Initialise hidden / memory states                                   #
        # ------------------------------------------------------------------ #
        device = frames_tensor.device

        # Permute from [B, T, H, W, C] → [B, T, C, H, W]
        frames   = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch  = frames.shape[0]
        height = frames.shape[3]
        width  = frames.shape[4]

        # Per-layer hidden state (h) and cell state (c)
        h_t = [torch.zeros(batch, self.num_hidden[i], height, width,
                           device=device)
               for i in range(self.num_layers)]
        c_t = [torch.zeros(batch, self.num_hidden[i], height, width,
                           device=device)
               for i in range(self.num_layers)]

        # Spatiotemporal memory shared across layers in zigzag fashion
        memory = torch.zeros(batch, self.num_hidden[0], height, width,
                             device=device)

        # ------------------------------------------------------------------ #
        # Recurrent loop over all time steps                                  #
        # ------------------------------------------------------------------ #
        next_frames = []
        # h_t_last accumulates the top-layer hidden state during the *input*
        # phase; we average these to form the classification feature.
        h_t_last_input_phase = []

        total_steps = self.configs.pre_seq_length + self.configs.aft_seq_length - 1

        for t in range(total_steps):
            # ---- Select the input frame for this time step ---------------
            # During the input phase (t < pre_seq_length) we always feed the
            # ground-truth frame.  During the prediction phase we use the
            # scheduled-sampling mask to mix ground-truth and predictions.
            if self.configs.reverse_scheduled_sampling == 1:
                # Reverse scheduled sampling: use ground-truth first, then
                # progressively switch to model predictions
                if t == 0:
                    net = frames[:, t]
                else:
                    net = (mask_true[:, t - 1] * frames[:, t]
                           + (1 - mask_true[:, t - 1]) * x_gen)
            else:
                # Standard scheduled sampling
                if t < self.configs.pre_seq_length:
                    # Input phase: always use ground-truth
                    net = frames[:, t]
                else:
                    # Prediction phase: mix ground-truth with model output
                    net = (mask_true[:, t - self.configs.pre_seq_length] * frames[:, t]
                           + (1 - mask_true[:, t - self.configs.pre_seq_length]) * x_gen)

            # ---- Run through stacked ST-LSTM cells -----------------------
            # Layer 0 receives the (possibly patch-embedded) input frame
            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)

            # Subsequent layers receive the hidden state from the layer below
            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](
                    h_t[i - 1], h_t[i], c_t[i], memory
                )

            # ---- Frame prediction branch ----------------------------------
            # Decode the top-layer hidden state back to pixel space
            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

            # ---- Collect hidden states for classification -----------------
            # We only use the *input* phase (t < pre_seq_length - 1) hidden
            # states so that the classifier sees only observed information,
            # not its own predictions.
            if t < self.configs.pre_seq_length:
                h_t_last_input_phase.append(h_t[self.num_layers - 1].detach())

        # ------------------------------------------------------------------ #
        # Assemble predicted frames                                           #
        # ------------------------------------------------------------------ #
        # Stack along time: [T-1, B, C, H, W] → [B, T-1, H, W, C]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()

        # ------------------------------------------------------------------ #
        # Classification head (skipped entirely for ablation variants)       #
        # ------------------------------------------------------------------ #
        if self.use_cls:
            # Average the top-layer hidden states across the input phase time
            # steps to get a single [B, num_hidden_last, H, W] feature map
            h_avg = torch.stack(h_t_last_input_phase, dim=0).mean(dim=0)

            # Global average pool over spatial dimensions → [B, num_hidden_last]
            cls_feat = h_avg.mean(dim=[-2, -1])

            # Pass through the FC classification head → [B, 1]
            cls_logit = self.cls_head(cls_feat)
        else:
            # No classification branch: return a zeros tensor as a placeholder
            # so callers don't need to branch on whether cls_logit exists.
            cls_logit = torch.zeros(batch, 1, device=device)

        # ------------------------------------------------------------------ #
        # Compute combined loss (optional)                                    #
        # ------------------------------------------------------------------ #
        loss = None
        if kwargs.get("return_loss", True):
            # Frame reconstruction loss (MSE between predicted and true frames)
            # frames_tensor[:, 1:] are the ground-truth future frames
            frame_loss = self.mse_criterion(next_frames, frames_tensor[:, 1:])

            if self.use_cls:
                # Classification loss — only compute when valid labels are provided
                label = kwargs.get("label", None)
                if label is not None:
                    # Mask out samples where label == -1 (KOMATSUNA, no drought info)
                    valid_mask = (label.squeeze(-1) >= 0)
                    if valid_mask.any():
                        valid_logits = cls_logit[valid_mask]
                        valid_labels = label[valid_mask]
                        cls_loss = self.bce_criterion(valid_logits, valid_labels)
                    else:
                        # No valid labels in this batch → skip classification loss
                        cls_loss = torch.tensor(0.0, device=device)
                else:
                    cls_loss = torch.tensor(0.0, device=device)

                # Combine frame + classification losses.
                # λ = cls_loss_weight balances the two objectives.
                cls_loss_weight = kwargs.get("cls_loss_weight", 1.0)
                loss = frame_loss + cls_loss_weight * cls_loss
            else:
                # Ablation variant with no classification: loss is pure MSE.
                loss = frame_loss

        return next_frames, cls_logit, loss
