# configs/plant/CAMP_base.py
# --------------------------
# Ablation variant: CAMP_base — backbone only, no extra branches.
#
# This is the lower-bound ablation that uses only the ST-LSTM backbone with
# a pure MSE frame-prediction loss.  No classification branch and no
# ExGI/POI-aware training loss are included.
#
# Expected behaviour:
#   • Results should be close to the standalone PredRNN baseline, since
#     both use the same ST-LSTM architecture with MSE loss.
#   • Any gap vs PredRNN reflects differences in hyperparameters.
#   • Comparing CAMP_base against higher ablation variants shows how much
#     each added component contributes.
#
# Ablation table position:
#   Method       | cls branch | ExGI loss | Expected rank
#   -------------|------------|-----------|---------------
#   camp_base    |     ✗      |     ✗     |  worst  ← this config
#   camp_no_cls  |     ✗      |     ✓     |  middle
#   camp         |     ✓      |     ✗     |  middle
#   camp_full    |     ✓      |     ✓     |  best
#
# Controlled flags (read by CAMPBase / CAMPAblation):
#   use_cls      = False  → CAMP_Model skips building cls_head entirely
#   use_poi_loss = False  → CAMPAblation skips the ExGI loss in training_step

method = 'camp_base'   # maps to CAMPBase in openstl/methods/__init__.py

# ---- Architecture (identical to full CAMP for a fair comparison) ----------
num_hidden   = '64,64,64,64'
filter_size  = 5
stride       = 1
patch_size   = 2
layer_norm   = 0

# ---- Ablation flags --------------------------------------------------------
# These two flags are the only difference from the full CAMP config.
use_cls      = False   # disable classification branch entirely
use_poi_loss = False   # disable ExGI/POI training loss

# ---- Scheduled sampling (same as CAMP for fair comparison) ----------------
reverse_scheduled_sampling = 0
r_sampling_step_1          = 25000
r_sampling_step_2          = 50000
r_exp_alpha                = 5000
scheduled_sampling         = 1
sampling_stop_iter         = 50000
sampling_start_value       = 1.0
sampling_changing_rate     = 0.00002

# ---- Training (same as CAMP) -----------------------------------------------
batch_size     = 4
val_batch_size = 4
epoch          = 200
lr             = 1e-3
opt            = 'adam'
sched          = 'onecycle'

# ---- Sequence and image shape (same as CAMP) --------------------------------
pre_seq_length = 5
aft_seq_length = 5
total_length   = 10
in_shape       = [5, 3, 128, 128]

metrics = ['mse', 'mae', 'ssim', 'psnr']
