# configs/plant/CAMP_no_cls.py
# ----------------------------
# Ablation variant: CAMP_no_cls — backbone + ExGI loss, no classification.
#
# This variant removes the environmental classification branch while keeping
# the ExGI/POI-aware training loss.  It answers the question:
#   "How much does knowing the drought/control condition help frame prediction?"
#
# Expected behaviour:
#   • Should outperform CAMP_base because the ExGI loss adds a biologically
#     grounded training signal without needing irrigation labels.
#   • May underperform the full CAMP on the Arabidopsis dataset (where the
#     drought label is available and informative) but perform similarly on
#     KOMATSUNA (where the label is always -1 / ignored).
#
# Ablation table position:
#   Method       | cls branch | ExGI loss | Expected rank
#   -------------|------------|-----------|---------------
#   camp_base    |     ✗      |     ✗     |  worst
#   camp_no_cls  |     ✗      |     ✓     |  middle  ← this config
#   camp         |     ✓      |     ✗     |  middle
#   camp_full    |     ✓      |     ✓     |  best
#
# Controlled flags:
#   use_cls      = False  → CAMP_Model skips building cls_head entirely
#   use_poi_loss = True   → CAMPAblation adds ExGI MSE loss in training_step

method = 'camp_no_cls'   # maps to CAMPNoCls in openstl/methods/__init__.py

# ---- Architecture (identical to full CAMP for a fair comparison) ----------
num_hidden   = '64,64,64,64'
filter_size  = 5
stride       = 1
patch_size   = 2
layer_norm   = 0

# ---- Ablation flags --------------------------------------------------------
use_cls      = False   # disable classification branch
use_poi_loss = True    # keep ExGI/POI training loss

# ---- ExGI loss weight ------------------------------------------------------
# poi_loss_weight scales L_ExGI relative to L_MSE.
# 0.1 keeps ExGI from dominating while still providing a useful gradient.
poi_loss_weight = 0.1

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
