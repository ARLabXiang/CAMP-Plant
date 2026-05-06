# configs/plant/SimVP_predcls.py
# ------------------------------
# SimVP_PredCls — SimVP backbone + ExGI loss + late-fusion classification head.
# SimVP counterpart of TAU_PredCls.
#
# Hypothesis: tests whether the late-fusion classification failure observed
# on TAU generalizes across non-recurrent backbones.

method = 'simvp_predcls'

# ---- Backbone architecture (identical to SimVP) ---------------------------
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

# ---- Late-fusion classification head ---------------------------------------
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
