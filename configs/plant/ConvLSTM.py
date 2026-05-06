# configs/plant/ConvLSTM.py
# ConvLSTM baseline on plant datasets.
# ConvLSTM was the first method applied to image-level plant growth prediction
# (Sakurai et al. 2019) and serves as the fundamental baseline in the CAMP paper.

method = 'plant_convlstm'

# PredRNN-family architecture (ConvLSTM inherits from PredRNN in OpenSTL)
num_hidden   = '64,64,64,64'
filter_size  = 5
stride       = 1
patch_size   = 2
layer_norm   = 0

# Scheduled sampling
reverse_scheduled_sampling = 0
r_sampling_step_1          = 25000
r_sampling_step_2          = 50000
r_exp_alpha                = 5000
scheduled_sampling         = 1
sampling_stop_iter         = 50000
sampling_start_value       = 1.0
sampling_changing_rate     = 0.00002

# Training
batch_size     = 4
val_batch_size = 4
epoch          = 200
lr             = 1e-3
opt            = 'adam'
sched          = 'onecycle'

# Sequence and image shape
pre_seq_length = 5
aft_seq_length = 5
total_length   = 10
in_shape       = [5, 3, 128, 128]

metrics = ['mse', 'mae', 'ssim', 'psnr']
