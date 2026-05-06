# configs/plant/SimVP_no_cls.py
# -----------------------------
# SimVP_no_cls — SimVP backbone + ExGI loss only (no classification head).
#
# This isolates the effect of the ExGI auxiliary supervision on SimVP's strong
# baseline.  Comparing it against SimVP baseline answers the question:
#
#     "Does the ExGI biological-grounding signal transfer to a non-PredRNN
#      backbone, independent of the classification head?"
#
# Reuses the SimVPFull method/model code with use_cls=False — the classification
# head exists in the model but its outputs are ignored during training.

method = 'simvp_full'   # reuse SimVPFull method — handles use_cls=False

# ---- Backbone architecture (identical to SimVP for fair comparison) --------
model_type        = 'gSTA'
hid_S             = 64
hid_T             = 256
N_T               = 8
N_S               = 2
spatio_kernel_enc = 3
spatio_kernel_dec = 3
drop_path         = 0.1

# ---- Auxiliary supervisions ------------------------------------------------
use_poi_loss    = True
poi_loss_weight = 0.1   # β — matches CAMP_full / TAU_no_cls default
use_cls         = False  # ← KEY DIFFERENCE FROM SimVP_full
cls_loss_weight = 0.0    # not used when use_cls=False

# ---- Classification head architecture (built but unused) -------------------
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
