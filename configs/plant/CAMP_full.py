# configs/plant/CAMP_full.py
# --------------------------
# Ablation variant: CAMP_full — all three components active.
#
# This is the enhanced CAMP model that adds the ExGI/POI-aware training loss
# on top of the original MSE + BCE objective from the submitted paper.
# It serves as the upper bound of the ablation study and demonstrates that
# training with the same biological signal used for evaluation (POI/ExGI)
# further improves performance — directly addressing Reviewer 1's concern
# about limited novelty beyond "combining known components".
#
# Training loss:
#   L_total = L_MSE                          (frame reconstruction)
#           + cls_loss_weight  * L_BCE        (environmental classification)
#           + poi_loss_weight  * L_ExGI       (POI/ExGI-aware signal)
#
# Ablation table position:
#   Method       | cls branch | ExGI loss | Expected rank
#   -------------|------------|-----------|---------------
#   camp_base    |     ✗      |     ✗     |  worst
#   camp_no_cls  |     ✗      |     ✓     |  middle
#   camp         |     ✓      |     ✗     |  middle
#   camp_full    |     ✓      |     ✓     |  best  ← this config
#
# Controlled flags:
#   use_cls      = True   → classification branch is fully active
#   use_poi_loss = True   → ExGI MSE loss is added in training_step

method = 'camp_full'   # maps to CAMPFull in openstl/methods/__init__.py

# ---- Architecture (identical to full CAMP for a fair comparison) ----------
num_hidden   = '64,64,64,64'
filter_size  = 5
stride       = 1
patch_size   = 2
layer_norm   = 0

# ---- Ablation flags --------------------------------------------------------
use_cls      = True   # enable classification branch
use_poi_loss = True   # enable ExGI/POI training loss

# ---- Loss weights ----------------------------------------------------------
cls_loss_weight = 1.0   # weight for L_BCE (classification loss)
poi_loss_weight = 0.1   # weight for L_ExGI (ExGI/POI loss)
                        # 0.1 keeps ExGI from overshadowing MSE

# ---- Classification head architecture --------------------------------------
cls_hidden  = 256
cls_dropout = 0.3

# ---- Scheduled sampling ----------------------------------------------------
reverse_scheduled_sampling = 0
r_sampling_step_1          = 25000
r_sampling_step_2          = 50000
r_exp_alpha                = 5000
scheduled_sampling         = 1
sampling_stop_iter         = 50000
sampling_start_value       = 1.0
sampling_changing_rate     = 0.00002

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
