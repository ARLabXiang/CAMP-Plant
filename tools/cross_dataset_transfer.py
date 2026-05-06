"""
cross_dataset_transfer.py
-------------------------
Zero-shot cross-dataset transfer evaluation.

For each method that has a checkpoint trained on one plant dataset, this
script runs inference on the OTHER dataset's test set without any
fine-tuning. The pixel space ([0,1] RGB at 128×128, T=5 input → T=5 output)
is identical between Komatsuna and Arabidopsis, so cross-dataset inference
is architecturally well-defined.

Outputs go to:
    work_dirs/transfer_<train>_to_<test>_<method>_ep150/saved/
        preds.npy, trues.npy, inputs.npy, metrics.npy

Then run tools/eval_poi.py on each output dir to add poi_results.npy.
"""

import argparse
import os
import os.path as osp
import sys
import json
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openstl.datasets.dataloader_plant import load_data
from openstl.methods import method_maps
from openstl.utils import print_log


# Methods we will evaluate. Format: (run_dir_key, lightning_class_name).
# `run_dir_key` is the suffix used in `work_dirs/<dataset>_<key>_ep150/`.
# `lightning_class_name` is the Lightning class to call .load_from_checkpoint() on.
METHODS_TO_TRANSFER = [
    # baselines — run dirs are e.g. work_dirs/arabidopsis_tau_ep150/, so the
    # key is just "tau" (without the "plant_" prefix that the config uses).
    ("convlstm",  "PlantConvLSTM"),
    ("predrnn",   "PlantPredRNN"),
    ("phydnet",   "PlantPhyDNet"),
    ("simvp",     "PlantSimVP"),
    ("mim",       "PlantMIM"),
    ("tau",       "PlantTAU"),
    # CAMP family
    ("camp",            "CAMP"),
    ("camp_base",       "CAMPBase"),
    ("camp_no_cls",     "CAMPNoCls"),
    ("camp_full",       "CAMPFull"),
    # New cross-backbone variants
    ("mim_full",        "MIMFull"),
    ("tau_full",        "TAUFull"),
    ("tau_no_cls",      "TAUFull"),       # reuses TAUFull class with use_cls=False
    ("tau_full_clsw01", "TAUFull"),
    ("tau_full_detached","TAUFull"),
    ("tau_predcls",     "TAUPredCls"),
    ("simvp_full",      "SimVPFull"),
    ("simvp_no_cls",    "SimVPFull"),     # reuses SimVPFull class with use_cls=False
    ("simvp_predcls",   "SimVPPredCls"),
]

# Map legacy method-key (used in the run-dir name) → primary name.
# Some run dirs use slightly different keys than the registered method_map keys.
RUN_DIR_KEY_MAP = {
    "convlstm": "convlstm",
    "predrnn":  "predrnn",
    "phydnet":  "phydnet",
    "simvp":    "simvp",
    "mim":      "mim",
    "tau":      "tau",
    "camp":     "camp",
    "camp_base": "camp_base",
    "camp_no_cls": "camp_no_cls",
    "camp_full":   "camp_full",
    "mim_full":    "mim_full",
    "tau_full":    "tau_full",
    "tau_no_cls":  "tau_no_cls",
    "tau_full_clsw01":  "tau_full_clsw01",
    "tau_full_detached":"tau_full_detached",
    "tau_predcls": "tau_predcls",
    "simvp_full":  "simvp_full",
    "simvp_no_cls":"simvp_no_cls",
    "simvp_predcls":"simvp_predcls",
}


DATA_ROOTS = {
    "komatsuna":   "/Users/anniezhoumacpro/Desktop/CAMP agriculture/testdata/data",
    "arabidopsis": "/Users/anniezhoumacpro/Desktop/CAMP agriculture/testdata/water",
}


def find_checkpoint(method_key: str, dataset: str, epoch: int = 150):
    """Return the path to the best-epoch checkpoint for (method, dataset)."""
    run_dir = ROOT / "work_dirs" / f"{dataset}_{method_key}_ep{epoch}"
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    # Prefer the highest "best-*" epoch (lowest val_loss usually wins late)
    best = sorted(ckpt_dir.glob("best-*.ckpt"))
    if best:
        return best[-1]
    last = ckpt_dir / "last.ckpt"
    return last if last.exists() else None


def load_method_class(class_name: str):
    """Return the Lightning class given its name."""
    from openstl.methods import (
        ConvLSTM, PredRNN, PhyDNet, SimVP, MIM, TAU,
        CAMP, CAMPBase, CAMPNoCls, CAMPFull,
        MIMFull, TAUFull, TAUPredCls, SimVPFull, SimVPPredCls,
    )
    from openstl.methods.plant_baselines import (
        PlantConvLSTM, PlantPredRNN, PlantPhyDNet,
        PlantSimVP, PlantTAU, PlantMIM,
    )
    table = {
        "ConvLSTM": ConvLSTM, "PredRNN": PredRNN, "PhyDNet": PhyDNet,
        "SimVP": SimVP, "MIM": MIM, "TAU": TAU,
        "PlantConvLSTM": PlantConvLSTM, "PlantPredRNN": PlantPredRNN,
        "PlantPhyDNet": PlantPhyDNet, "PlantSimVP": PlantSimVP,
        "PlantTAU": PlantTAU, "PlantMIM": PlantMIM,
        "CAMP": CAMP, "CAMPBase": CAMPBase, "CAMPNoCls": CAMPNoCls, "CAMPFull": CAMPFull,
        "MIMFull": MIMFull, "TAUFull": TAUFull, "TAUPredCls": TAUPredCls,
        "SimVPFull": SimVPFull, "SimVPPredCls": SimVPPredCls,
    }
    return table[class_name]


def get_test_loader(dataset: str, batch_size: int, num_workers: int = 2):
    """Return the (test) DataLoader for the given dataset name."""
    # num_workers must be > 0 because openstl.datasets.utils.create_loader
    # passes persistent_workers=True unconditionally.
    _, _, test_loader = load_data(
        dataname=dataset,
        batch_size=batch_size,
        val_batch_size=batch_size,
        data_root=DATA_ROOTS[dataset],
        num_workers=num_workers,
        pre_seq_length=5,
        aft_seq_length=5,
        in_shape=[5, 3, 128, 128],
        distributed=False,
        use_augment=False,
        use_prefetcher=False,
        drop_last=False,
    )
    return test_loader


@torch.no_grad()
def run_inference(model, test_loader, device):
    """Run model on the test loader. Returns (inputs, preds, trues) numpy arrays."""
    model.eval()
    model.to(device)
    all_inputs, all_preds, all_trues = [], [], []
    for batch in test_loader:
        # Plant batches are 3-tuples (x, y, label)
        if len(batch) == 3:
            batch_x, batch_y, _label = batch
        else:
            batch_x, batch_y = batch
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        # Use the method's forward, which handles tuple-output internally
        try:
            pred_y = model(batch_x, batch_y)
        except TypeError:
            pred_y = model(batch_x)
        # If model returns a tuple (pred, cls_logit), take pred only
        if isinstance(pred_y, (tuple, list)):
            pred_y = pred_y[0]
        all_inputs.append(batch_x.cpu().numpy())
        all_preds.append(pred_y.cpu().numpy())
        all_trues.append(batch_y.cpu().numpy())
    inputs = np.concatenate(all_inputs, axis=0)
    preds  = np.concatenate(all_preds,  axis=0)
    trues  = np.concatenate(all_trues,  axis=0)

    # Track numerical failure rate (some methods produce NaN/Inf when the
    # train/test distribution shift is severe — this IS itself a finding).
    nan_mask = np.isnan(preds) | np.isinf(preds)
    nan_frac = nan_mask.mean()

    # Replace NaN/Inf with 0 (a conservative "I don't know" prediction).
    # Then clip to valid [0, 1] range.
    preds = np.where(nan_mask, 0.0, preds)
    preds = np.clip(preds, 0.0, 1.0).astype(np.float32)
    return inputs, preds, trues, nan_frac


def compute_pixel_metrics(preds, trues):
    """Compute MAE, MSE, SSIM, PSNR matching the format used by Base_method.test_step."""
    from openstl.core import metric
    eval_res, _ = metric(
        preds, trues,
        mean=None, std=None,
        metrics=['mse', 'mae', 'ssim', 'psnr'],
        spatial_norm=False,
    )
    return np.array([
        float(eval_res['mae']),
        float(eval_res['mse']),
        float(eval_res.get('ssim', float('nan'))),
        float(eval_res.get('psnr', float('nan'))),
    ])


def transfer_one(method_key, class_name, train_dataset, test_dataset,
                 epoch=150, device="mps"):
    """Run one transfer evaluation: model trained on `train_dataset` evaluated on `test_dataset`."""
    ckpt = find_checkpoint(method_key, train_dataset, epoch)
    if ckpt is None:
        print(f"  [skip] no checkpoint for ({method_key}, {train_dataset})")
        return None

    out_dir = ROOT / "work_dirs" / f"transfer_{train_dataset}_to_{test_dataset}_{method_key}_ep{epoch}/saved"
    if (out_dir / "metrics.npy").exists():
        print(f"  [done] {method_key}: {train_dataset}→{test_dataset} (already evaluated)")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [run]  {method_key}: {train_dataset}→{test_dataset} ...", flush=True)

    # Build test loader for the OTHER dataset
    # MIM requires batch_size to match val_batch_size used in training
    bs = 4
    test_loader = get_test_loader(test_dataset, batch_size=bs, num_workers=2)

    # Load model from checkpoint
    cls = load_method_class(class_name)
    model = cls.load_from_checkpoint(str(ckpt), strict=False)

    # Run inference
    try:
        inputs, preds, trues, nan_frac = run_inference(model, test_loader, device)
    except Exception as e:
        print(f"  [ERROR] {method_key}: {train_dataset}→{test_dataset}: {type(e).__name__}: {e}")
        return None

    # Save outputs in the same format Base_method.test_step does
    np.save(out_dir / "inputs.npy", inputs)
    np.save(out_dir / "preds.npy",  preds)
    np.save(out_dir / "trues.npy",  trues)
    # Save NaN rate as a separate artifact so we can report it later
    np.save(out_dir / "nan_frac.npy", np.array(nan_frac, dtype=np.float32))

    # Compute pixel metrics on the sanitized predictions
    metrics = compute_pixel_metrics(preds, trues)
    np.save(out_dir / "metrics.npy", metrics)

    nan_warn = f" [NaN={100*nan_frac:.1f}%]" if nan_frac > 0 else ""
    print(f"    → {out_dir.name}/  N={len(preds)}  "
          f"MAE={metrics[0]:.2f} MSE={metrics[1]:.2f} "
          f"SSIM={metrics[2]:.4f} PSNR={metrics[3]:.2f}{nan_warn}")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=150)
    ap.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    ap.add_argument("--directions", default="both",
                    choices=["k2a", "a2k", "both"])
    ap.add_argument("--methods", nargs="+", default=None,
                    help="Subset of method keys; default = all")
    args = ap.parse_args()

    methods = METHODS_TO_TRANSFER
    if args.methods:
        methods = [(k, c) for (k, c) in methods if k in args.methods]

    # Direction 1: Komatsuna→Arabidopsis
    if args.directions in ("k2a", "both"):
        print("=" * 60)
        print("DIRECTION 1: Komatsuna-trained → Arabidopsis test")
        print("=" * 60)
        for k, c in methods:
            transfer_one(k, c, "komatsuna", "arabidopsis", args.epoch, args.device)

    # Direction 2: Arabidopsis→Komatsuna
    if args.directions in ("a2k", "both"):
        print()
        print("=" * 60)
        print("DIRECTION 2: Arabidopsis-trained → Komatsuna test")
        print("=" * 60)
        for k, c in methods:
            transfer_one(k, c, "arabidopsis", "komatsuna", args.epoch, args.device)


if __name__ == "__main__":
    main()
