# configs/plant/TAU_full.py
# -------------------------
# TAU_full — TAU backbone + ExGI regression loss + binary classification head.
#
# This is the TAU-backbone counterpart of CAMP_full.  It tests whether the two
# auxiliary supervisions introduced in CAMP (ExGI + crop-type classification)
# transfer to TAU, the strongest non-recurrent spatiotemporal backbone.
#
# Training loss:
#   L_total = L_MSE                                # frame reconstruction
#           + alpha           * L_DiffDivReg       # TAU native regularizer
#           + poi_loss_weight * L_ExGI             # CAMP ExGI supervision
#           + cls_loss_weight * L_BCE              # CAMP classification
#
# Dataset guidance: designed for **Arabidopsis** (valid crop-type labels).
# Komatsuna has no valid labels — classification there would hurt, as the
# CAMP_base vs CAMP ablation already showed.

method = 'tau_full'

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

# ---- CAMP auxiliary supervisions -------------------------------------------
use_poi_loss    = True
poi_loss_weight = 0.1   # β — matches CAMP_full default
use_cls         = True
cls_loss_weight = 1.0   # γ — matches CAMP_full default

# ---- Classification head architecture (mirrors CAMP_full) ------------------
cls_hidden  = 256
cls_dropout = 0.3
num_classes = 1         # binary classifier (Arabidopsis: control vs drought)

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
