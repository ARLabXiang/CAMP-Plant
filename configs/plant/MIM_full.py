# configs/plant/MIM_full.py
# -------------------------
# MIM_full — MIM backbone + ExGI regression loss + binary classification head.
#
# MIM-backbone counterpart of CAMP_full / TAU_full / SimVP_full.
#
# Training loss:
#   L_total = L_MSE                                # frame reconstruction (patch-space)
#           + poi_loss_weight * L_ExGI             # CAMP ExGI supervision (pixel-space)
#           + cls_loss_weight * L_BCE              # CAMP classification
#
# Dataset guidance: designed for **Arabidopsis** (valid crop-type labels).

method = 'mim_full'

# ---- Backbone architecture (identical to MIM for fair comparison) ----------
num_hidden   = '64,64,64,64'
filter_size  = 5
stride       = 1
patch_size   = 2
layer_norm   = 0

# ---- Scheduled sampling (same as MIM / PredRNN / CAMP) ---------------------
reverse_scheduled_sampling = 0
r_sampling_step_1          = 25000
r_sampling_step_2          = 50000
r_exp_alpha                = 5000
scheduled_sampling         = 1
sampling_stop_iter         = 50000
sampling_start_value       = 1.0
sampling_changing_rate     = 0.00002

# ---- CAMP auxiliary supervisions -------------------------------------------
use_poi_loss    = True
poi_loss_weight = 0.1
use_cls         = True
cls_loss_weight = 1.0

# ---- Classification head architecture (mirrors CAMP_full) ------------------
cls_hidden  = 256
cls_dropout = 0.3
num_classes = 1

# ---- Training (MIM requires batch_size == val_batch_size + drop_last=True) -
batch_size     = 4
val_batch_size = 4
drop_last      = True
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
