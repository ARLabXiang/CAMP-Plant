# configs/plant/TAU_full_detached.py
# -----------------------------------
# TAU_full_detached — TAU + ExGI + classification head with DETACHED gradients.
#
# The classification head is built on the translator latent (same tap point as
# TAU_full) BUT during forward, we apply .detach() before passing the latent
# into cls_head. This means:
#
#   • cls_loss trains the cls_head's weights normally
#   • cls_loss does NOT update the backbone (encoder, translator, decoder)
#   • The backbone trains identically to TAU_no_cls
#
# Expected outcome:
#   - POI_MAE ≈ TAU_no_cls's 296.1 (since backbone gradients are identical)
#   - SSIM    ≈ 0.86 (preserved)
#   - cls accuracy is the NEW informative metric: tells us whether prediction-
#     trained features encode drought/control discrimination as a linear probe.
#
# This is the cleanest architectural fix for non-recurrent backbones —
# guaranteed to preserve prediction quality, while still producing a
# classification readout for deployment + scientific diagnostic.

method = 'tau_full'   # reuse TAUFull method — handles detach_cls flag

# ---- Backbone architecture (identical to TAU/TAU_full) ---------------------
model_type        = 'TAU'
hid_S             = 64
hid_T             = 256
N_T               = 8
N_S               = 2
spatio_kernel_enc = 3
spatio_kernel_dec = 3
drop_path         = 0.1

# ---- TAU native: differential divergence regularizer weight ----------------
alpha = 0.1

# ---- CAMP auxiliary supervisions -------------------------------------------
use_poi_loss    = True
poi_loss_weight = 0.1
use_cls         = True
cls_loss_weight = 1.0   # weight matters less here since cls grad is detached from backbone
detach_cls      = True   # ← KEY DIFFERENCE FROM TAU_full

# ---- Classification head architecture (mirrors CAMP_full) ------------------
cls_hidden  = 256
cls_dropout = 0.3
num_classes = 1

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
