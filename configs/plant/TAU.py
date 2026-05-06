# configs/plant/TAU.py
# TAU (Temporal Attention Unit) on plant datasets.
#
# TAU extends SimVP with temporal attention and a divergence regularisation
# loss that penalises unrealistic frame-to-frame jumps.  This is particularly
# relevant for plant growth, where changes between consecutive frames should
# be smooth and incremental.
#
# TAU is the current SOTA non-recurrent method.  Adding it strengthens the
# paper: if CAMP outperforms TAU it demonstrates that physically-informed
# multi-task learning beats pure attention-based approaches for plant growth.

method = 'plant_tau'

# TAU uses the same SimVP_Model backbone with model_type='TAU'
model_type       = 'TAU'
hid_S            = 64
hid_T            = 256
N_T              = 8
N_S              = 2
spatio_kernel_enc = 3
spatio_kernel_dec = 3
drop_path        = 0.1

# TAU-specific: weight for the temporal divergence regularisation loss.
# alpha=0.1 is the default from the TAU paper.
alpha = 0.1

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
