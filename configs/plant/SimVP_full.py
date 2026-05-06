# configs/plant/SimVP_full.py
# ---------------------------
# SimVP_full — SimVP backbone + ExGI regression loss + binary classification head.
#
# SimVP-backbone counterpart of CAMP_full / TAU_full.  Unlike TAU, SimVP has
# no native auxiliary regularizer, so this variant isolates the effect of
# CAMP's two auxiliary supervisions on a clean strong baseline.
#
# Training loss:
#   L_total = L_MSE                                # frame reconstruction
#           + poi_loss_weight * L_ExGI             # CAMP ExGI supervision
#           + cls_loss_weight * L_BCE              # CAMP classification
#
# Dataset guidance: designed for **Arabidopsis** (valid crop-type labels).

method = 'simvp_full'

# ---- Backbone architecture (identical to SimVP for fair comparison) --------
model_type        = 'gSTA'
hid_S             = 64
hid_T             = 256
N_T               = 8
N_S               = 2
spatio_kernel_enc = 3
spatio_kernel_dec = 3
drop_path         = 0.1

# ---- CAMP auxiliary supervisions -------------------------------------------
use_poi_loss    = True
poi_loss_weight = 0.1
use_cls         = True
cls_loss_weight = 1.0

# ---- Classification head architecture (mirrors CAMP_full) ------------------
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
