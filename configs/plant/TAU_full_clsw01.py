# configs/plant/TAU_full_clsw01.py
# --------------------------------
# TAU_full with reduced classification loss weight (cls_loss_weight=0.1)
# instead of the original 1.0 used in CAMP_full.
#
# Purpose: test the diagnosis that the classification head with cls_weight=1.0
# overpowers TAU's gradient dynamics.  If lowering the weight to 0.1 (matching
# poi_loss_weight) recovers the baseline TAU performance — or improves it —
# this confirms the issue is loss-balance, not the auxiliary signal itself.

method = 'tau_full'

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

# ---- Auxiliary supervisions ------------------------------------------------
use_poi_loss    = True
poi_loss_weight = 0.1
use_cls         = True
cls_loss_weight = 0.1   # ← KEY DIFFERENCE FROM TAU_full (was 1.0)

# ---- Classification head architecture --------------------------------------
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
