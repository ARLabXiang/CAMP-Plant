# configs/plant/PhyDNet.py
# PhyDNet baseline on plant datasets.
# PhyDNet embeds physical constraints via partial-derivative convolutions.
# Including it tests whether a different physical-prior approach (vs CAMP's
# classification-assisted one) is more effective for plant growth.

method = 'plant_phydnet'

# PhyDNet architecture (k=49 moment terms, teacher forcing ratio decays)
# Note: PhyDNet uses its own internal architecture; these are training params.
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
patch_size     = 4    # PhyDNet model requires patch_size in {2, 4}

metrics = ['mse', 'mae', 'ssim', 'psnr']
