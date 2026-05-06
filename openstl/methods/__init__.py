# Copyright (c) CAIRI AI Lab. All rights reserved

from .convlstm import ConvLSTM
from .e3dlstm import E3DLSTM
from .mau import MAU
from .mim import MIM
from .phydnet import PhyDNet
from .predrnn import PredRNN
from .predrnnpp import PredRNNpp
from .predrnnv2 import PredRNNv2
from .simvp import SimVP
from .tau import TAU
from .mmvp import MMVP
from .swinlstm import SwinLSTM_D, SwinLSTM_B
from .wast import WaST
from .camp import CAMP                          # original CAMP (MSE + BCE)
from .camp_ablation import (                    # ablation study variants
    CAMPAblation, CAMPBase, CAMPNoCls, CAMPFull
)
from .tau_full import TAUFull                   # TAU backbone + ExGI + cls head (translator-tap)
from .tau_predcls import TAUPredCls             # TAU backbone + ExGI + late-fusion cls (pred_y-tap)
from .simvp_full import SimVPFull               # SimVP backbone + ExGI + cls head
from .simvp_predcls import SimVPPredCls         # SimVP backbone + ExGI + late-fusion cls
from .mim_full import MIMFull                   # MIM backbone + ExGI + cls head
from .plant_baselines import (   # Baselines wrapped for 3-tuple plant batches
    PlantConvLSTM, PlantPredRNN, PlantPhyDNet,
    PlantSimVP, PlantTAU, PlantMIM,
)

method_maps = {
    'convlstm': ConvLSTM,
    'e3dlstm': E3DLSTM,
    'mau': MAU,
    'mim': MIM,
    'phydnet': PhyDNet,
    'predrnn': PredRNN,
    'predrnnpp': PredRNNpp,
    'predrnnv2': PredRNNv2,
    'simvp': SimVP,
    'tau': TAU,
    'mmvp': MMVP,
    'swinlstm_d': SwinLSTM_D,
    'swinlstm_b': SwinLSTM_B,
    'swinlstm': SwinLSTM_B,
    'wast': WaST,
    'camp': CAMP,               # original CAMP (MSE + BCE, no ExGI loss)
    # ---- Ablation study variants (Reviewer 1, Comment 3) -----------------
    # Each variant removes one or more CAMP components to isolate contributions.
    # Controlled by use_cls / use_poi_loss flags in the config file.
    'camp_base':    CAMPBase,   # backbone only   — MSE loss
    'camp_no_cls':  CAMPNoCls,  # no cls branch   — MSE + ExGI loss
    'camp_full':    CAMPFull,   # all components  — MSE + BCE + ExGI loss
    'tau_full':     TAUFull,    # TAU backbone    — MSE + DDR + ExGI + BCE (translator-tap cls)
    'tau_predcls':  TAUPredCls, # TAU backbone    — MSE + DDR + ExGI + BCE (late-fusion cls from pred_y)
    'simvp_full':   SimVPFull,  # SimVP backbone  — MSE + ExGI + BCE
    'simvp_predcls': SimVPPredCls, # SimVP backbone — MSE + ExGI + BCE (late-fusion cls)
    'mim_full':     MIMFull,    # MIM backbone    — MSE + ExGI + BCE (patch-space)
    # Plant-dataset wrappers (handle 3-tuple batches with label)
    'plant_convlstm': PlantConvLSTM,
    'plant_predrnn':  PlantPredRNN,
    'plant_phydnet':  PlantPhyDNet,
    'plant_simvp':    PlantSimVP,
    'plant_tau':      PlantTAU,    # new: TAU (temporal attention, SimVP SOTA)
    'plant_mim':      PlantMIM,    # new: MIM (cited in CAMP related work)
}

__all__ = [
    'method_maps', 'ConvLSTM', 'E3DLSTM', 'MAU', 'MIM',
    'PredRNN', 'PredRNNpp', 'PredRNNv2', 'PhyDNet', 'SimVP', 'TAU',
    "MMVP", 'SwinLSTM_D', 'SwinLSTM_B', 'WaST'
]