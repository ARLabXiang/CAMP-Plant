# configs/plant/CAMP.py
# ---------------------
# Configuration for training the CAMP model on either the KOMATSUNA or the
# Arabidopsis plant dataset.
#
# To train on KOMATSUNA:
#   python tools/train.py -d komatsuna -m camp --config_file configs/plant/CAMP.py
#
# To train on Arabidopsis:
#   python tools/train.py -d arabidopsis -m camp --config_file configs/plant/CAMP.py
#
# All values here are defaults; any can be overridden on the command line.

# ------------------------------------------------------------------
# Method name (must match the key in openstl/methods/__init__.py)
# ------------------------------------------------------------------
method = 'camp'

# ------------------------------------------------------------------
# PredRNN encoder architecture
# ------------------------------------------------------------------
# Four ST-LSTM layers, each with 64 hidden channels.
# The paper uses 128 hidden channels; reduce to 64 for faster iteration.
num_hidden   = '64,64,64,64'   # comma-separated, one value per layer
filter_size  = 5               # convolution kernel size inside ST-LSTM cells
stride       = 1               # convolution stride (keep at 1 for same-size)
patch_size   = 2               # patch embedding: 2×2 patches → 4× fewer pixels
layer_norm   = 0               # 0 = no layer norm (faster), 1 = with layer norm

# ------------------------------------------------------------------
# Scheduled sampling parameters (controls teacher-forcing decay)
# ------------------------------------------------------------------
# Standard scheduled sampling: start with full ground-truth, gradually
# switch to model predictions over sampling_stop_iter steps.
reverse_scheduled_sampling = 0
r_sampling_step_1          = 25000   # end of reverse-SS phase 1 (unused here)
r_sampling_step_2          = 50000   # end of reverse-SS phase 2 (unused here)
r_exp_alpha                = 5000    # exponential decay factor (unused here)
scheduled_sampling         = 1
sampling_stop_iter         = 50000   # iteration when ground-truth prob → 0
sampling_start_value       = 1.0     # initial ground-truth probability
sampling_changing_rate     = 0.00002 # decrease in GT probability per step

# ------------------------------------------------------------------
# Multi-task loss weighting
# ------------------------------------------------------------------
# The total loss is:  L = L_MSE + cls_loss_weight * L_BCE
# Setting cls_loss_weight = 0 disables the classification branch loss.
cls_loss_weight = 1.0

# ------------------------------------------------------------------
# Classification head architecture
# ------------------------------------------------------------------
cls_hidden  = 256   # number of units in the hidden FC layer
cls_dropout = 0.3   # dropout probability inside the FC head

# ------------------------------------------------------------------
# Training hyper-parameters
# ------------------------------------------------------------------
batch_size     = 4       # small batch for limited plant data
val_batch_size = 4
epoch          = 200
lr             = 1e-3
opt            = 'adam'
sched          = 'onecycle'

# ------------------------------------------------------------------
# Sequence lengths
# ------------------------------------------------------------------
# Input: 5 observed frames; output: 5 predicted frames.
# Together they use all 10 available time steps from KOMATSUNA.
pre_seq_length = 5
aft_seq_length = 5
total_length   = 10   # = pre_seq_length + aft_seq_length

# ------------------------------------------------------------------
# Image shape
# ------------------------------------------------------------------
# [T, C, H, W] — T is pre_seq_length, images are RGB 128×128
in_shape = [5, 3, 128, 128]

# ------------------------------------------------------------------
# Evaluation metrics
# ------------------------------------------------------------------
metrics = ['mse', 'mae', 'ssim', 'psnr']
