"""
mim_full_model.py
-----------------
MIM (Memory-In-Memory) backbone augmented with a binary classification head,
mirroring the CAMP_full / TAU_full design on top of the recurrent ST-LSTM +
MIMBlock/MIMN architecture.

Architecture mirrors MIM_Model exactly for the prediction path.  The only
addition is a classification head that taps the top-layer hidden state
(h_t[num_layers-1]) collected during the *input phase* (t < pre_seq_length),
averaged over time, then pooled and fed through FC layers.  This is the
same pattern CAMP_Model uses on the PredRNN ST-LSTM backbone — enabling
an apples-to-apples backbone ablation.

Forward return:
    (next_frames [B, T-1, H, W, C], cls_logit [B, num_classes], loss [scalar|None])
"""

import torch
import torch.nn as nn

from .mim_model import MIM_Model


class MIM_Full_Model(MIM_Model):
    r"""MIM backbone with a classification head on the top-layer hidden state."""

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super().__init__(num_layers, num_hidden, configs, **kwargs)

        cls_hidden  = getattr(configs, "cls_hidden",  256)
        cls_dropout = getattr(configs, "cls_dropout", 0.3)
        num_classes = getattr(configs, "num_classes", 1)

        top_dim = num_hidden[num_layers - 1]

        # Classification head mirrors CAMP_Model's head:
        #   GAP over spatial → Linear(top_dim, cls_hidden) → ReLU → Dropout → Linear(cls_hidden, num_classes)
        self.cls_head = nn.Sequential(
            nn.Linear(top_dim, cls_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cls_dropout),
            nn.Linear(cls_hidden, num_classes),
        )

    def forward(self, frames_tensor, mask_true, **kwargs):
        """
        Same prediction logic as MIM_Model.forward(), plus:
          • collect the top-layer hidden state h_t[-1] during the input phase
            (t < pre_seq_length),
          • after the loop ends, average the collected states, global-average-
            pool over the spatial dimensions, and run through cls_head.

        Returns:
            next_frames : [B, T-1, H, W, C]   (same as parent)
            cls_logit   : [B, num_classes]
            loss        : scalar or None (same as parent)
        """
        device = frames_tensor.device
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        # Reset stale MIMBlock state (same as parent)
        for layer in self.stlstm_layer:
            if hasattr(layer, 'convlstm_c'):
                layer.convlstm_c = None

        batch  = frames.shape[0]
        height = frames.shape[3]
        width  = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []
        hidden_state_diff = []
        cell_state_diff   = []

        for i in range(self.num_layers):
            zeros = torch.zeros(
                [batch, self.num_hidden[i], height, width], device=device)
            h_t.append(zeros)
            c_t.append(zeros)
            hidden_state_diff.append(None)
            cell_state_diff.append(None)

        st_memory = torch.zeros(
            [batch, self.num_hidden[0], height, width], device=device)

        # Collect top-layer hidden states during input phase for classification
        h_top_input_phase = []

        pre_seq = self.configs.pre_seq_length

        for t in range(pre_seq + self.configs.aft_seq_length - 1):
            # reverse schedule sampling
            if self.configs.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + \
                          (1 - mask_true[:, t - 1]) * x_gen
            else:
                if t < pre_seq:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - pre_seq] * frames[:, t] + \
                          (1 - mask_true[:, t - pre_seq]) * x_gen

            preh = h_t[0]
            h_t[0], c_t[0], st_memory = self.stlstm_layer[0](net, h_t[0], c_t[0], st_memory)

            for i in range(1, self.num_layers):
                if t > 0:
                    if i == 1:
                        hidden_state_diff[i - 1], cell_state_diff[i - 1] = self.stlstm_layer_diff[i - 1](
                            h_t[i - 1] - preh, hidden_state_diff[i - 1], cell_state_diff[i - 1])
                    else:
                        hidden_state_diff[i - 1], cell_state_diff[i - 1] = self.stlstm_layer_diff[i - 1](
                            hidden_state_diff[i - 2], hidden_state_diff[i - 1], cell_state_diff[i - 1])
                else:
                    self.stlstm_layer_diff[i - 1](torch.zeros_like(h_t[i - 1]), None, None)

                h_t[i], c_t[i], st_memory = self.stlstm_layer[i](
                    h_t[i - 1], hidden_state_diff[i - 1], h_t[i], c_t[i], st_memory)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

            # Collect top-layer hidden state during input phase only
            if t < pre_seq:
                h_top_input_phase.append(h_t[self.num_layers - 1])

        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()

        # Classification head: average top-layer hidden state across input phase,
        # then global-average-pool spatially, then FC stack.
        h_avg = torch.stack(h_top_input_phase, dim=0).mean(dim=0)   # [B, C, H, W]
        cls_feat = h_avg.mean(dim=[-2, -1])                         # [B, C]
        cls_logit = self.cls_head(cls_feat)                         # [B, num_classes]

        if kwargs.get('return_loss', True):
            loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        else:
            loss = None

        return next_frames, cls_logit, loss
