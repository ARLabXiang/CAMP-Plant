"""
dataloader_plant.py
-------------------
Dataset classes and data loaders for two plant-growth datasets used in the
CAMP paper:

  1. KOMATSUNADataset  -- multi-view Komatsuna leaf images (public dataset)
  2. ArabidopsisDataset -- RoAD system Arabidopsis images (internal dataset)

Both datasets return tuples of the form:
    (input_frames, target_frames, label)
where
    input_frames  : FloatTensor [pre_seq, C, H, W]  -- observed frames
    target_frames : FloatTensor [aft_seq, C, H, W]  -- future frames to predict
    label         : FloatTensor [1]                  -- environmental condition
                    (0 = control/no-drought, 1 = drought)

For KOMATSUNA there is no drought label, so label is always -1 (ignored during
the classification loss).  For Arabidopsis the label is parsed from the folder
name: W_81.95 → drought (1), W_173.62 → control (0).
"""

import os
import random
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from openstl.datasets.utils import create_loader


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _load_image_rgb(path: str, target_size: int) -> np.ndarray:
    """
    Load an image from *path*, convert it to RGB, resize to
    (target_size × target_size) with bilinear interpolation, and return a
    float32 numpy array in [0, 1] with shape (C, H, W).
    """
    img = Image.open(path).convert("RGB")
    img = img.resize((target_size, target_size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0   # H × W × C  in [0, 1]
    arr = arr.transpose(2, 0, 1)                     # → C × H × W
    return arr


# ===========================================================================
# KOMATSUNA Dataset
# ===========================================================================

class KOMATSUNADataset(Dataset):
    """
    KOMATSUNA multi-view Komatsuna leaf dataset.

    Directory layout expected::

        data_root/
          {DDPP}/                  # e.g. 0000, 0001, …, 0204
            rgb_DD_PP_TTT_VV.png   # DD=dataset, PP=plant, TTT=timestep, VV=view

    Each folder contains 10 time steps × 6 camera views = 60 images.
    We treat **each (folder, view)** combination as one independent time
    series of length 10 and then apply a sliding window of
    (pre_seq_length + aft_seq_length) to generate sequences.

    Train/test split:
      - Dataset IDs 00 and 01 → training
      - Dataset ID  02       → testing
      (controlled via the *split* argument)

    Since KOMATSUNA has no drought / irrigation label, *label* is always
    returned as -1 (the CAMP classification loss ignores it for this dataset).
    """

    # Camera views available in the dataset
    ALL_VIEWS = ["00", "01", "02", "03", "04", "05"]

    # Dataset IDs used for each split
    TRAIN_DATASETS = {"00", "01"}
    TEST_DATASETS  = {"02"}

    def __init__(
        self,
        data_root: str,
        pre_seq_length: int = 5,
        aft_seq_length: int = 5,
        img_size: int = 128,
        split: str = "train",
        use_augment: bool = False,
    ):
        """
        Args:
            data_root     : Path to the directory that contains the numbered
                            plant folders (e.g. '…/testdata/data').
            pre_seq_length: Number of observed input frames.
            aft_seq_length: Number of future frames to predict.
            img_size      : Spatial resolution to resize images to (square).
            split         : 'train' or 'test'.
            use_augment   : Whether to apply random horizontal flip augmentation.
        """
        super().__init__()
        assert split in ("train", "test"), "split must be 'train' or 'test'"

        self.data_root      = data_root
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.seq_length     = pre_seq_length + aft_seq_length
        self.img_size       = img_size
        self.split          = split
        self.use_augment    = use_augment

        # Required by OpenSTL's BaseDataModule
        self.mean      = 0.0
        self.std       = 1.0
        self.data_name = "komatsuna"

        # Build the list of (folder_path, view_id, start_timestep) for every
        # valid sliding-window position.
        self.samples = self._build_sample_list()

    def _build_sample_list(self):
        """
        Enumerate all (folder, view, start_t) triples that form complete
        sequences of length seq_length.
        """
        samples = []

        # Iterate over all numbered folders inside data_root
        for folder_name in sorted(os.listdir(self.data_root)):
            folder_path = os.path.join(self.data_root, folder_name)

            # Skip non-directories and the zip archive
            if not os.path.isdir(folder_path):
                continue

            # Folder name is 4 digits: DDPP  (DD=dataset id, PP=plant id)
            if len(folder_name) != 4 or not folder_name.isdigit():
                continue

            # Determine dataset ID from the first two characters of the folder
            # name, then filter by split
            dataset_id = folder_name[:2]
            if self.split == "train" and dataset_id not in self.TRAIN_DATASETS:
                continue
            if self.split == "test"  and dataset_id not in self.TEST_DATASETS:
                continue

            # Collect all files so we can infer available timesteps and views
            all_files = sorted(
                f for f in os.listdir(folder_path) if f.endswith(".png")
            )
            if not all_files:
                continue

            # Parse unique timestep indices from filenames: rgb_DD_PP_TTT_VV.png
            timesteps = sorted(set(
                f.replace("rgb_", "").replace(".png", "").split("_")[2]
                for f in all_files
            ))
            views = sorted(set(
                f.replace("rgb_", "").replace(".png", "").split("_")[3]
                for f in all_files
            ))

            # Build sliding windows.  With 10 frames and seq_length=10 this
            # yields exactly 1 window; shorter seq_length yields more.
            for view in views:
                n_frames = len(timesteps)
                for start in range(n_frames - self.seq_length + 1):
                    samples.append((folder_path, view, timesteps[start:start + self.seq_length]))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder_path, view, timestep_ids = self.samples[idx]

        # Derive dataset and plant IDs from the folder name for constructing
        # the correct filename prefix.  e.g. folder '0001' → DD='00', PP='01'
        folder_name = os.path.basename(folder_path)
        dd = folder_name[:2]   # dataset id in filename (e.g. '00')
        pp = folder_name[2:]   # plant id in filename (e.g. '01')

        # Load the full sequence of frames
        frames = []
        for t in timestep_ids:
            # Filename: rgb_DD_PP_TTT_VV.png
            filename = f"rgb_{dd}_{pp}_{t}_{view}.png"
            fpath = os.path.join(folder_path, filename)
            frame = _load_image_rgb(fpath, self.img_size)  # C × H × W
            frames.append(frame)

        # Stack into [seq_length, C, H, W]
        frames = np.stack(frames, axis=0)
        frames = torch.tensor(frames, dtype=torch.float32)

        # Optional random horizontal flip augmentation
        if self.use_augment and random.random() > 0.5:
            frames = torch.flip(frames, dims=[-1])  # flip width dimension

        # Split into input and target
        input_frames  = frames[:self.pre_seq_length]   # [pre_seq, C, H, W]
        target_frames = frames[self.pre_seq_length:]   # [aft_seq, C, H, W]

        # KOMATSUNA has no drought label → use -1 as a sentinel so the
        # CAMP method can skip the classification loss for this dataset.
        label = torch.tensor([-1.0], dtype=torch.float32)

        return input_frames, target_frames, label


# ===========================================================================
# Arabidopsis (RoAD) Dataset
# ===========================================================================

class ArabidopsisDataset(Dataset):
    """
    Arabidopsis thaliana time-series dataset captured by the RoAD system.

    Directory layout expected::

        data_root/
          train/                                  # or 'test'
            {GENOTYPE}_W_{WATER}_{REP}/           # e.g. WT-1_W_173.62_1
              GENOTYPE_W_WATER_REP_YYYY-M-D_plant.bmp

    Label encoding (parsed from the folder name):
        W_81.95  → drought  → label = 1
        W_173.62 → control  → label = 0

    The dataset applies a **sliding window** of length (pre_seq + aft_seq)
    over each plant's daily image sequence.  Frames are already 128×128 but
    we optionally resize to a different img_size.
    """

    # Water amounts that map to drought / control labels
    DROUGHT_WATER = "81.95"
    CONTROL_WATER = "173.62"

    def __init__(
        self,
        data_root: str,
        pre_seq_length: int = 5,
        aft_seq_length: int = 5,
        img_size: int = 128,
        split: str = "train",
        use_augment: bool = False,
    ):
        """
        Args:
            data_root     : Path to the water/ directory that contains 'train'
                            and 'test' (or 'dry'/'no_dry') sub-folders.
            pre_seq_length: Number of observed input frames.
            aft_seq_length: Number of future frames to predict.
            img_size      : Spatial resolution for resizing (default 128 px).
            split         : 'train' or 'test'.
            use_augment   : Whether to apply random horizontal flip.
        """
        super().__init__()
        assert split in ("train", "test"), "split must be 'train' or 'test'"

        self.data_root      = data_root
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.seq_length     = pre_seq_length + aft_seq_length
        self.img_size       = img_size
        self.split          = split
        self.use_augment    = use_augment

        # Required by OpenSTL's BaseDataModule
        self.mean      = 0.0
        self.std       = 1.0
        self.data_name = "arabidopsis"

        # Build list of (sorted_image_paths, label) and then expand with
        # sliding windows
        self.samples = self._build_sample_list()

    def _parse_label(self, folder_name: str) -> float:
        """
        Parse the irrigation label from the plant folder name.

        Folder naming convention: GENOTYPE_W_WATERAMOUNT_REPLICATE
        e.g. 'WT-1_W_81.95_1' → drought (1.0)
             'BRI1P-BRI1OX-2_W_173.62_3' → control (0.0)
        """
        if f"W_{self.DROUGHT_WATER}" in folder_name:
            return 1.0   # drought
        elif f"W_{self.CONTROL_WATER}" in folder_name:
            return 0.0   # control
        else:
            # Unknown water amount; default to control and warn
            print(f"[WARNING] Unknown water label in folder: {folder_name}")
            return 0.0

    def _build_sample_list(self):
        """
        Walk through the split directory, collect all plant folders, sort
        their image files chronologically, and generate sliding windows.

        Returns:
            List of (image_path_list, label) tuples, one per window.
        """
        samples = []

        # The combined train/test folders already mix dry and no_dry plants
        split_dir = os.path.join(self.data_root, self.split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Expected split directory not found: {split_dir}\n"
                f"Make sure data_root points to the 'water/' parent folder."
            )

        for plant_folder in sorted(os.listdir(split_dir)):
            plant_path = os.path.join(split_dir, plant_folder)
            if not os.path.isdir(plant_path):
                continue

            # Collect and sort image files by date embedded in the filename.
            # Filename: GENOTYPE_W_WATER_REP_YYYY-M-D_plant.bmp
            image_files = sorted(
                f for f in os.listdir(plant_path)
                if f.lower().endswith(".bmp") or f.lower().endswith(".png")
            )
            if len(image_files) < self.seq_length:
                # Not enough frames for even one window; skip this plant
                continue

            full_paths = [os.path.join(plant_path, f) for f in image_files]
            label      = self._parse_label(plant_folder)

            # Apply sliding window
            for start in range(len(full_paths) - self.seq_length + 1):
                window_paths = full_paths[start : start + self.seq_length]
                samples.append((window_paths, label))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window_paths, label = self.samples[idx]

        # Load every frame in the window
        frames = []
        for fpath in window_paths:
            frame = _load_image_rgb(fpath, self.img_size)  # C × H × W
            frames.append(frame)

        # Stack into [seq_length, C, H, W]
        frames = np.stack(frames, axis=0)
        frames = torch.tensor(frames, dtype=torch.float32)

        # Optional horizontal flip augmentation
        if self.use_augment and random.random() > 0.5:
            frames = torch.flip(frames, dims=[-1])

        # Split into input (observed) and target (future)
        input_frames  = frames[:self.pre_seq_length]   # [pre_seq, C, H, W]
        target_frames = frames[self.pre_seq_length:]   # [aft_seq, C, H, W]

        # Classification label: 0 = control, 1 = drought
        label_tensor = torch.tensor([label], dtype=torch.float32)

        return input_frames, target_frames, label_tensor


# ===========================================================================
# load_data factory function (matches OpenSTL's dataloader.py interface)
# ===========================================================================

def load_data(
    dataname: str,
    batch_size: int,
    val_batch_size: int,
    data_root: str,
    num_workers: int = 4,
    pre_seq_length: int = 5,
    aft_seq_length: int = 5,
    in_shape=None,
    distributed: bool = False,
    use_augment: bool = False,
    use_prefetcher: bool = False,
    drop_last: bool = False,
    **kwargs,
):
    """
    Build train / val / test DataLoaders for the plant datasets.

    Args:
        dataname        : 'komatsuna' or 'arabidopsis'
        batch_size      : Training batch size.
        val_batch_size  : Validation / test batch size.
        data_root       : Root path to the dataset.
                          • For komatsuna  → path to the folder containing the
                            numbered plant directories (e.g. '…/testdata/data')
                          • For arabidopsis → path to the 'water/' directory
                            that holds 'train/' and 'test/' subdirectories.
        num_workers     : DataLoader worker processes.
        pre_seq_length  : Observed sequence length (input to model).
        aft_seq_length  : Prediction horizon (frames to generate).
        in_shape        : Ignored (kept for API compatibility).
        distributed     : Whether to use DistributedSampler.
        use_augment     : Enable random horizontal flip during training.
        use_prefetcher  : Enable CUDA prefetching (requires CUDA).
        drop_last       : Drop last incomplete batch in test loader.
    """
    # Determine the spatial resolution from in_shape if provided
    img_size = in_shape[-1] if in_shape is not None else 128

    if dataname == "komatsuna":
        # KOMATSUNA: train uses dataset IDs 00+01, test uses 02
        train_set = KOMATSUNADataset(
            data_root      = data_root,
            pre_seq_length = pre_seq_length,
            aft_seq_length = aft_seq_length,
            img_size       = img_size,
            split          = "train",
            use_augment    = use_augment,
        )
        test_set = KOMATSUNADataset(
            data_root      = data_root,
            pre_seq_length = pre_seq_length,
            aft_seq_length = aft_seq_length,
            img_size       = img_size,
            split          = "test",
            use_augment    = False,
        )

    elif dataname == "arabidopsis":
        # Arabidopsis: uses pre-split train/ and test/ subdirectories
        train_set = ArabidopsisDataset(
            data_root      = data_root,
            pre_seq_length = pre_seq_length,
            aft_seq_length = aft_seq_length,
            img_size       = img_size,
            split          = "train",
            use_augment    = use_augment,
        )
        test_set = ArabidopsisDataset(
            data_root      = data_root,
            pre_seq_length = pre_seq_length,
            aft_seq_length = aft_seq_length,
            img_size       = img_size,
            split          = "test",
            use_augment    = False,
        )

    else:
        raise ValueError(f"Unknown plant dataname: {dataname!r}. "
                         f"Expected 'komatsuna' or 'arabidopsis'.")

    # Build DataLoaders using OpenSTL's create_loader (handles distributed
    # sampling, prefetching, worker seeding, etc.)
    dataloader_train = create_loader(
        train_set,
        batch_size   = batch_size,
        shuffle      = True,
        is_training  = True,
        pin_memory   = True,
        drop_last    = True,
        num_workers  = num_workers,
        distributed  = distributed,
        use_prefetcher = use_prefetcher,
    )

    # No separate validation split defined — use test set for validation
    dataloader_vali = create_loader(
        test_set,
        batch_size   = val_batch_size,
        shuffle      = False,
        is_training  = False,
        pin_memory   = True,
        drop_last    = False,
        num_workers  = num_workers,
        distributed  = distributed,
        use_prefetcher = use_prefetcher,
    )

    dataloader_test = create_loader(
        test_set,
        batch_size   = val_batch_size,
        shuffle      = False,
        is_training  = False,
        pin_memory   = True,
        drop_last    = drop_last,
        num_workers  = num_workers,
        distributed  = distributed,
        use_prefetcher = use_prefetcher,
    )

    return dataloader_train, dataloader_vali, dataloader_test
