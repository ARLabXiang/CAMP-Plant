# configs/plant/MIM.py
# MIM (Memory In Memory) on plant datasets.
#
# MIM is directly cited in the CAMP paper: "Wang et al. (2022) devised a
# model with spatiotemporal long short-term memory (ST-LSTM) and memory in
# memory network (MIM), achieving high similarity between predicted and real
# images."  This makes MIM the most directly motivated new baseline to add.
#
# MIM extends PredRNN with a non-stationary memory module that captures
# higher-order temporal dynamics — potentially useful for the non-linear
# growth patterns of Arabidopsis under drought stress.
#
# Note: MIM requires batch_size == val_batch_size (enforced internally).

method = 'plant_mim'

# MIM uses the same ST-LSTM + scheduled sampling framework as PredRNN
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

# Training — batch_size MUST equal val_batch_size for MIM.
# drop_last=True is required because MIM stores the batch size in its hidden
# state (convlstm_c); an incomplete final batch causes a size mismatch crash.
batch_size     = 4
val_batch_size = 4
drop_last      = True
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
