# configs/plant/SimVP.py
# SimVP baseline on plant datasets.
# SimVP is the state-of-the-art non-recurrent video prediction method and
# achieves the highest SSIM on KOMATSUNA in the CAMP paper.

method = 'plant_simvp'

# SimVP architecture — gSTA translator (best-performing variant)
model_type       = 'gSTA'
hid_S            = 64    # spatial hidden channels
hid_T            = 256   # temporal hidden channels
N_T              = 8     # number of temporal transformer blocks
N_S              = 2     # number of spatial down/up-sampling stages
spatio_kernel_enc = 3
spatio_kernel_dec = 3
drop_path        = 0.1

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
