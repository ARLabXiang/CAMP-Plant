# configs/plant/TAU_predcls.py
# ----------------------------
# TAU_PredCls — TAU backbone + ExGI loss + late-fusion classification head
# that reads from the predicted future frames (pred_y), not from the
# translator latent.
#
# Hypothesis: this architectural change should restore prediction quality
# (vs TAU_full's catastrophic POI_MAE 547) while keeping the multi-task
# classification capability.
#
# Training loss:
#   L_total = L_MSE                                # frame reconstruction
#           + alpha           * L_DiffDivReg       # TAU native regularizer
#           + poi_loss_weight * L_ExGI             # CAMP ExGI supervision
#           + cls_loss_weight * L_BCE              # late-fusion classification

method = 'tau_predcls'

# ---- Backbone architecture (identical to TAU/TAU_full) ---------------------
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
poi_loss_weight = 0.1
use_cls         = True
cls_loss_weight = 1.0

# ---- Late-fusion classification head architecture --------------------------
# Note: cls_hidden is the FC hidden dim AFTER the small CNN.
# The CNN itself has its own fixed channel sizes (32, 64, 128).
cls_hidden  = 128
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
