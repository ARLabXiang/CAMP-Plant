# configs/plant/TAU_no_cls.py
# ---------------------------
# TAU_no_cls — TAU backbone + ExGI regression loss only (no classification head).
#
# This isolates the effect of the ExGI auxiliary supervision on TAU's strong
# baseline.  Comparing it against TAU baseline answers the question:
#
#     "Does the ExGI biological-grounding signal transfer to a non-PredRNN
#      backbone, independent of the classification head?"
#
# Compared to TAU_full, this config disables the classification loss
# (use_cls=False).  The TAUFull method/model code already handles this case:
# the classification head exists in the model but its outputs are ignored
# during training (no BCE loss term added) — so the cls head receives no
# gradient updates and does not influence prediction quality.
#
# Training loss:
#   L_total = L_MSE                                # frame reconstruction
#           + alpha           * L_DiffDivReg       # TAU native regularizer
#           + poi_loss_weight * L_ExGI             # CAMP ExGI supervision
#
# Dataset guidance: designed for **Arabidopsis** (matches TAU baseline runs).

method = 'tau_full'   # reuse TAUFull method — it handles use_cls=False

# ---- Backbone architecture (identical to TAU for fair comparison) ----------
model_type        = 'TAU'
hid_S             = 64
hid_T             = 256
N_T               = 8
N_S               = 2
spatio_kernel_enc = 3
spatio_kernel_dec = 3
drop_path         = 0.1

# ---- TAU native: differential divergence regularizer weight ----------------
alpha = 0.1

# ---- Auxiliary supervisions ------------------------------------------------
use_poi_loss    = True
poi_loss_weight = 0.1   # β — matches CAMP_full / TAU_full default
use_cls         = False  # ← KEY DIFFERENCE FROM TAU_full
cls_loss_weight = 0.0    # not used when use_cls=False

# ---- Classification head architecture (built but unused) -------------------
# These are still passed to the model so cls_head exists for checkpoint
# compatibility with TAU_full, but its outputs are not in the loss.
cls_hidden  = 256
cls_dropout = 0.3
num_classes = 1

# ---- Training --------------------------------------------------------------
batch_size     = 4
val_batch_size = 4
epoch          = 200
lr             = 1e-3
opt            = 'adam'
sched          = 'onecycle'

# ---- Sequence and image shape ----------------------------------------------
pre_seq_length = 5
aft_seq_length = 5
total_length   = 10
in_shape       = [5, 3, 128, 128]

metrics = ['mse', 'mae', 'ssim', 'psnr']
