"""
plant_baselines.py
------------------
Thin wrapper subclasses that make every OpenSTL baseline method compatible
with the plant-growth dataloaders.

Problem
~~~~~~~
All existing OpenSTL methods expect 2-element batches: (batch_x, batch_y).
Our plant dataloaders return 3-element batches: (batch_x, batch_y, label),
where *label* is the drought/control class for the CAMP classification task.
Baseline methods do not use the label, so we just need to strip it before
forwarding to the original logic.

Solution
~~~~~~~~
We define a ``PlantMixin`` that overrides ``training_step``,
``validation_step``, and ``test_step`` to unpack the 3-tuple and discard
the label.  Each concrete baseline class (PlantConvLSTM, PlantPredRNN, …)
inherits from both the mixin and the original method class — the mixin comes
first in the MRO so its overrides win.

All other behaviour (model construction, forward pass, optimiser, scheduler,
metrics) is inherited unchanged from the original method.

Usage
~~~~~
Register the plant baseline methods in method_maps by adding entries like
  'plant_convlstm': PlantConvLSTM
and point the training script to the corresponding config files.
"""

import numpy as np
import torch

from openstl.utils import check_dir, print_log
from openstl.core import metric

import os.path as osp

# Import the original methods we want to wrap
from .convlstm  import ConvLSTM
from .predrnn   import PredRNN
from .phydnet   import PhyDNet
from .simvp     import SimVP
from .tau       import TAU
from .mim       import MIM


# ===========================================================================
# Mixin: strips the label from the 3-element plant batch
# ===========================================================================

class PlantMixin:
    """
    Mixin that overrides the three Lightning steps to accept
    (batch_x, batch_y, label) and forward only (batch_x, batch_y) to the
    original method.  This makes every OpenSTL baseline work unchanged with
    the plant dataset's 3-tuple batches.
    """

    def training_step(self, batch, batch_idx):
        """Unpack 3-tuple, discard label, delegate to parent training_step."""
        batch_x, batch_y, _label = batch
        # Re-pack as 2-tuple so the parent's logic works identically
        return super().training_step((batch_x, batch_y), batch_idx)

    def validation_step(self, batch, batch_idx):
        """Unpack 3-tuple, discard label, delegate to parent validation_step."""
        batch_x, batch_y, _label = batch
        return super().validation_step((batch_x, batch_y), batch_idx)

    def test_step(self, batch, batch_idx):
        """Unpack 3-tuple, discard label, delegate to parent test_step."""
        batch_x, batch_y, _label = batch
        return super().test_step((batch_x, batch_y), batch_idx)


# ===========================================================================
# Concrete wrapper classes — one per baseline
# ===========================================================================

class PlantConvLSTM(PlantMixin, ConvLSTM):
    """ConvLSTM with plant-dataset 3-tuple batch support."""
    pass


class PlantPredRNN(PlantMixin, PredRNN):
    """PredRNN with plant-dataset 3-tuple batch support."""
    pass


class PlantPhyDNet(PlantMixin, PhyDNet):
    """
    PhyDNet with plant-dataset 3-tuple batch support.

    PhyDNet's forward() and training_step() both accept batch_y (for teacher
    forcing), so we need to override training_step explicitly rather than
    just stripping the tuple, because the parent's training_step also calls
    self.model with batch_y.
    """

    def training_step(self, batch, batch_idx):
        batch_x, batch_y, _label = batch
        return super(PlantMixin, self).training_step(
            (batch_x, batch_y), batch_idx
        )

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y, _label = batch
        return super(PlantMixin, self).validation_step(
            (batch_x, batch_y), batch_idx
        )

    def test_step(self, batch, batch_idx):
        batch_x, batch_y, _label = batch
        return super(PlantMixin, self).test_step(
            (batch_x, batch_y), batch_idx
        )


class PlantSimVP(PlantMixin, SimVP):
    """SimVP with plant-dataset 3-tuple batch support."""
    pass


class PlantTAU(PlantMixin, TAU):
    """
    TAU (Temporal Attention Unit) with plant-dataset 3-tuple batch support.

    TAU adds a temporal divergence regularisation term on top of SimVP's MSE
    loss, making it the strongest non-recurrent baseline.  Including it
    strengthens the paper if CAMP outperforms it.
    """
    pass


class PlantMIM(PlantMixin, MIM):
    """
    MIM (Memory In Memory) with plant-dataset 3-tuple batch support.

    MIM is directly cited in the CAMP paper (Wang et al. 2022 used MIM for
    plant growth prediction), making it the most directly relevant new baseline
    to add beyond the original four.
    """
    pass
