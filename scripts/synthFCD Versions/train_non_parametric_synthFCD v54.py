"""FCD SynthSeg – 7-class FCD lesion segmentation (background + 5 tissue groups + FCD lesion)."""

# ── Standard library ────────────────────────────────────────────────────────
import os
import sys
import glob
import math
import random
import shutil
import datetime
import traceback

from typing import Sequence, Optional
from os     import path, makedirs
from random import shuffle

# ── Third-party libraries ───────────────────────────────────────────────────
import numpy             as np
import pandas            as pd
import torch
import matplotlib.pyplot as plt
import nibabel           as nib

import pytorch_lightning         as pl
from pytorch_lightning.cli       import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers   import CSVLogger, TensorBoardLogger

from torch.utils.data          import Dataset, DataLoader
from torchmetrics.segmentation import DiceScore as dice_compute

import cornucopia       as cc
from cornucopia         import SynthFromLabelTransform, IntensityTransform
from cornucopia.special import IdentityTransform

# NOTE: GaussianSmooth is currently unused → remove if not needed
# from monai.transforms import GaussianSmooth


# ── Project path setup ──────────────────────────────────────────────────────
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)


# ── Local modules (learn2synth) ─────────────────────────────────────────────
from learn2synth.networks      import UNet, SegNet
from learn2synth.train         import SynthSeg
from learn2synth.losses        import (
    DiceLoss, LogitMSELoss, CatLoss, CatMSELoss,
    DiceCELoss, FocalTverskyLoss,
)
from learn2synth               import optim
from learn2synth.parameters    import FCDParameterCalculator
from learn2synth.augmentations import FCDAugmentations

from learn2synth.custom_cc_synthseg import (
    SynthFromLabelTransform as CustomSynthFromLabelTransform,
)

# ── Configuration (single source of truth) ──────────────────────────────────
from learn2synth.configurations import (
    DEFAULT_FOLDER,
    OUTPUT_FOLDER,
    FLAIR_STATS_CSV,
    FLAIR_CLASS_PARAMS,
    flair_file,
    roi_file,
    label_file,
    fusedmask_file,
    INTENSITY_SUBJECTS,
    TRANSMANTLE_SUBJECTS,
    HYPER_SUBJECTS,
    BLUR_SUBJECTS,
    THICKENING_SUBJECTS,
)


# ── FCDDataset ────────────────────────────────────────────────────────────────
# Returns un-augmented volumes plus random augmentation configurations.
# Actual GPU synthesis happens inside Model.synthesize_batch
class FCDDataset(Dataset):
    def __init__(
            self,
            ndim,
            label_paths,
            flair_paths,
            roi_paths,
            fused_paths                  = None,
            native_synthesis: bool       = False,
            fcd_intensity_range          = (0.02, 0.3602),
            fcd_tail_length_range        = (20, 50),
            blur_sigma_range             = (0.7, 1.7),
            zoom_f_range                 = (0.75, 0.95),
            hyper_sigma_range            = (0.0, 0.3),
            trans_sigma_range            = (0.0, 0.3)
    ):
        """
        Args:
            ndim: Dimensions of the input data.
            label_paths, flair_paths, roi_paths: Paths to the respective NIfTI volumes.
            fcd_intensity_range: Range for synthetic lesion intensity factors.
            fcd_tail_length_range: Range for the length of the transmantle tail.
            blur_sigma_range: Range for Gaussian blur augmentation.
            zoom_f_range: Range for cortical thickening (zoom) factor.
            hyper_sigma_range: Range for gray matter hyperintensity noise.
            trans_sigma_range: Range for transmantle signal intensity noise.
        """
        self.ndim                  = ndim
        self.native_synthesis      = native_synthesis

        self.fcd_intensity_range   = fcd_intensity_range
        self.fcd_tail_length_range = fcd_tail_length_range

        # Store augmentation hyperparameters for external configurability
        self.blur_sigma_range      = blur_sigma_range
        self.zoom_f_range          = zoom_f_range
        self.hyper_sigma_range     = hyper_sigma_range
        self.trans_sigma_range     = trans_sigma_range

        self.items                 = []

        # Initialize stateless utility once to minimize instantiation overhead during loading
        self._calc                 = FCDParameterCalculator()

        # Normalise fused_paths — None list when not native_synthesis
        if not fused_paths:
            fused_paths = [None] * len(label_paths)

        for label_path, flair_path, roi_path, fused_path in zip(label_paths, flair_paths, roi_paths, fused_paths):
            subject_num = self._calc.get_subj_num(os.path.dirname(label_path))
            aug_matches = []

            # Determine specific augmentation types based on subject-specific manifests
            if subject_num in BLUR_SUBJECTS:        aug_matches.append('blur')
            if subject_num in THICKENING_SUBJECTS:  aug_matches.append('zoom')
            if subject_num in HYPER_SUBJECTS:       aug_matches.append('hyper')
            if subject_num in TRANSMANTLE_SUBJECTS: aug_matches.append('trans')

            aug_type = '+'.join(aug_matches) if aug_matches else 'combo'
            self.items.append((label_path, flair_path, roi_path, fused_path, aug_type))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        label_path, flair_path, roi_path, fused_path, aug_type = self.items[idx]

        # --- Volume Loading (I/O) ---
        flair_arr = nib.load(flair_path).get_fdata()
        label_arr = nib.load(label_path).get_fdata().astype(int)
        roi_arr   = nib.load(roi_path).get_fdata().astype(int)

        # --- Spatial Standardization ---
        # Resample ROI and labels to match FLAIR space using shared utility instance
        if flair_arr.shape != roi_arr.shape:
            roi_arr   = (self._calc.resample_to_target(roi_arr, flair_arr.shape, True) > 0.5).astype(int)
        if flair_arr.shape != label_arr.shape:
            label_arr = self._calc.resample_to_target(label_arr, flair_arr.shape, True).astype(int)

        # --- Tensor Conversion ---
        # Data is prepared for the CNN pipeline (C, H, W, D format)
        label_tensor = torch.as_tensor(label_arr, dtype=torch.int64).unsqueeze(0)
        flair_tensor = torch.as_tensor(flair_arr, dtype=torch.float32).unsqueeze(0)
        roi_tensor   = torch.as_tensor(roi_arr,   dtype=torch.int64).unsqueeze(0)

        # --- Parameter Exposure ---
        # Construct augmentation parameters from instance ranges.
        # These are passed as tensors to ensure consistency across worker processes.
        aug_params = {
            'int_factor_min':  torch.tensor(self.fcd_intensity_range[0],   dtype=torch.float32),
            'int_factor_max':  torch.tensor(self.fcd_intensity_range[1],   dtype=torch.float32),
            'tail_length_min': torch.tensor(self.fcd_tail_length_range[0], dtype=torch.long),
            'tail_length_max': torch.tensor(self.fcd_tail_length_range[1], dtype=torch.long),

            'blur_sigma_min':  torch.tensor(self.blur_sigma_range[0],      dtype=torch.float32),
            'blur_sigma_max':  torch.tensor(self.blur_sigma_range[1],      dtype=torch.float32),

            'zoom_f_min':      torch.tensor(self.zoom_f_range[0],          dtype=torch.float32),
            'zoom_f_max':      torch.tensor(self.zoom_f_range[1],          dtype=torch.float32),

            'hyper_sigma_min': torch.tensor(self.hyper_sigma_range[0],     dtype=torch.float32),
            'hyper_sigma_max': torch.tensor(self.hyper_sigma_range[1],     dtype=torch.float32),

            'trans_sigma_min': torch.tensor(self.trans_sigma_range[0],     dtype=torch.float32),
            'trans_sigma_max': torch.tensor(self.trans_sigma_range[1],     dtype=torch.float32),
        }

        item = {
            'label_t':    label_tensor,
            'flair_t':    flair_tensor,
            'roi_t':      roi_tensor,
            'aug_type':   aug_type,
            'subject_id': os.path.basename(os.path.dirname(label_path)),
            **aug_params,
        }

        # --- Fused mask (native_synthesis path only) ---
        if self.native_synthesis:
            fused_arr = nib.load(fused_path).get_fdata().astype(int)
            if flair_arr.shape != fused_arr.shape:
                fused_arr = self._calc.resample_to_target(
                    fused_arr, flair_arr.shape, True).astype(int)
            item['fusedmask_t'] = torch.as_tensor(fused_arr, dtype=torch.int64).unsqueeze(0)

        return item


# ── FCDDataModule ─────────────────────────────────────────────────────────────
#
#  Source-aware split policy
#  ─────────────────────────
#  Training  : raw/ + generated/   (all available subjects)
#  Validation: raw/ only           (real subjects with ground-truth labels)
#
#  The split is enforced structurally — by scanning the two source directories
#  independently — rather than post-hoc via a fraction, so generated subjects
#  can never leak into validation regardless of ordering or preshuffle.
#
#  `eval` controls what fraction of *raw* subjects are held out for validation.
#  The remaining raw subjects join training alongside the entire generated pool.
#
#  Configurable via constructor flags:
#    train_subdir      (default "train")       — root under dataset_path
#    raw_subdir        (default "raw")         — val-eligible subfolder
#    extra_subdirs     (default ["generated"]) — train-only subfolders
#    use_extra_data    (default False)          — set True to train on raw + extra
#    val_from_raw_only (default False)          — set True to take validation from raw only
# ──────────────────────────────────────────────────────────────────────────────
class FCDDataModule(pl.LightningDataModule):
    def __init__(self,
                 ndim: int                     = 3,
                 dataset_path: str             = DEFAULT_FOLDER,
                 eval: float                   = 0.04,
                 preshuffle: bool              = False,
                 split_seed: int               = 42,
                 batch_size: int               = 1,
                 shuffle: bool                 = True,
                 num_workers: int              = 4,
                 native_synthesis: bool        = False,
                 train_subdir: str             = 'train',
                 raw_subdir: Optional[str]     = 'raw',
                 extra_subdirs: Optional[list] = None,
                 use_extra_data: bool          = False):
        super().__init__()

        # --- Config ---
        self.ndim             = ndim
        self.dataset_path     = dataset_path
        self.eval_frac        = eval
        self.preshuffle       = preshuffle
        self.split_seed       = split_seed
        self.batch_size       = batch_size
        self.shuffle          = shuffle
        self.num_workers      = num_workers
        self.native_synthesis = native_synthesis
        self.use_extra_data   = use_extra_data

        # --- Resolve directory layout ---
        if raw_subdir is None:
            self.use_extra_data    = False
            self.val_from_raw_only = False
            print("[FCDDataModule] raw_subdir=None — scanning train_subdir directly, use_extra_data forced False.")
        elif extra_subdirs is None:
            extra_subdirs          = ['generated']

        train_root  = path.join(dataset_path, train_subdir)
        raw_root    = train_root if raw_subdir is None else path.join(train_root, raw_subdir)
        extra_roots = [path.join(train_root, s) for s in extra_subdirs] if extra_subdirs else []

        # --- Helper: scan one directory for valid triplets (+ fusedmask when native_synthesis) ---
        def _scan(root: str) -> tuple:
            subject_folders = sorted(glob.glob(path.join(root, 'sub-*')))
            label_paths, flair_paths, roi_paths, fused_paths = [], [], [], []
            dropped = 0
            for subject_dir in subject_folders:
                label_path = path.join(subject_dir, label_file)
                flair_path = path.join(subject_dir, flair_file)
                roi_path   = path.join(subject_dir, roi_file)
                fused_path = path.join(subject_dir, fusedmask_file)

                required = [label_path, flair_path, roi_path]
                if self.native_synthesis:
                    required.append(fused_path)
                if all(path.exists(x) for x in required):
                    label_paths.append(label_path)
                    flair_paths.append(flair_path)
                    roi_paths.append(roi_path)
                    if self.native_synthesis:
                        fused_paths.append(fused_path)
                else:
                    dropped += 1
            if dropped:
                print(f"[FCDDataModule] WARNING: {dropped} incomplete triplets dropped in {root}")
            else:
                print(f"[FCDDataModule] {len(label_paths)} subjects loaded from {root}")
            return label_paths, flair_paths, roi_paths, fused_paths

        # --- Scan raw (val-eligible) subjects ---
        raw_label_paths, raw_flair_paths, raw_roi_paths, raw_fused_paths = _scan(raw_root)
        assert len(raw_label_paths) > 0, (
            f"[FCDDataModule] Fatal: 0 valid triplets in '{raw_root}'. "
            "Check path and file names."
        )

        # --- Scan extra (train-only) subjects ---
        extra_label_paths, extra_flair_paths, extra_roi_paths, extra_fused_paths = [], [], [], []
        if not self.use_extra_data:
            print("[FCDDataModule] use_extra_data=False — training on raw subjects only.")
        else:
            for extra_root in extra_roots:
                if path.isdir(extra_root):
                    e_labels, e_flairs, e_rois, e_fused = _scan(extra_root)
                    extra_label_paths.extend(e_labels)
                    extra_flair_paths.extend(e_flairs)
                    extra_roi_paths.extend(e_rois)
                    extra_fused_paths.extend(e_fused)  # ← fixed
                else:
                    print(f"[FCDDataModule] NOTE: extra subdir not found, skipping: {extra_root}")

        # --- Store split-ready pools ---
        self._raw_label_paths = raw_label_paths
        self._raw_flair_paths = raw_flair_paths
        self._raw_roi_paths   = raw_roi_paths
        self._raw_fused_paths = raw_fused_paths

        self._extra_label_paths = extra_label_paths
        self._extra_flair_paths = extra_flair_paths
        self._extra_roi_paths   = extra_roi_paths
        self._extra_fused_paths = extra_fused_paths

        print(
            f"[FCDDataModule] Source summary: "
            f"{len(raw_label_paths)} raw, {len(extra_label_paths)} generated "
            f"→ val pool = raw only ({len(raw_label_paths)} subjects)"
        )

        if not self.native_synthesis:
            print("[FCDDataModule] Computing FCD augmentation parameters…")
            self._calc = FCDParameterCalculator()
            self.fcd_intensity_range, self.fcd_tail_range = self._calc.calculate_fcd_parameters(
                dataset_path         = raw_root,
                label_file           = label_file,
                flair_file           = flair_file,
                roi_file             = roi_file,
                intensity_subjects   = INTENSITY_SUBJECTS,
                transmantle_subjects = TRANSMANTLE_SUBJECTS,
                auto_resample        = True,
            )
        else:
            print("[FCDDataModule] native_synthesis=True — skipping FCD augmentation parameter computation.")
            self.fcd_intensity_range = (0.0, 0.0)
            self.fcd_tail_range      = (0, 0)

    def setup(self, stage=None):
        if hasattr(self, '_setup_done'):
            return
        self._setup_done = True

        # Copy raw pool (and shuffle if requested) before splitting
        raw_label_paths = list(self._raw_label_paths)
        raw_flair_paths = list(self._raw_flair_paths)
        raw_roi_paths   = list(self._raw_roi_paths)
        raw_fused_paths = list(self._raw_fused_paths)

        if self.preshuffle:
            combined = list(zip(raw_label_paths, raw_flair_paths, raw_roi_paths, raw_fused_paths))
            shuffle(combined)
            raw_label_paths, raw_flair_paths, raw_roi_paths, raw_fused_paths = map(list, zip(*combined))
        else:
            # seeded deterministic shuffle before split
            combined = list(zip(raw_label_paths, raw_flair_paths, raw_roi_paths, raw_fused_paths))
            random.Random(self.split_seed).shuffle(combined)
            raw_label_paths, raw_flair_paths, raw_roi_paths, raw_fused_paths = map(list, zip(*combined))

        def _count(param, total):
            if isinstance(param, float): return int(math.ceil(total * param))
            if isinstance(param, int):   return param
            return 0

        # Split raw pool → val head + train tail
        n_val = _count(self.eval_frac, len(raw_label_paths))

        val_label_paths = raw_label_paths[:n_val]
        val_flair_paths = raw_flair_paths[:n_val]
        val_roi_paths   = raw_roi_paths[:n_val]
        val_fused_paths = raw_fused_paths[:n_val]  # ← fixed

        train_raw_label_paths = raw_label_paths[n_val:]
        train_raw_flair_paths = raw_flair_paths[n_val:]
        train_raw_roi_paths   = raw_roi_paths[n_val:]
        train_raw_fused_paths = raw_fused_paths[n_val:]  # ← fixed

        # Training set = remaining raw + all extra
        train_label_paths = train_raw_label_paths + list(self._extra_label_paths)
        train_flair_paths = train_raw_flair_paths + list(self._extra_flair_paths)
        train_roi_paths   = train_raw_roi_paths   + list(self._extra_roi_paths)
        train_fused_paths = train_raw_fused_paths + list(self._extra_fused_paths)  # ← fixed

        print(
            f"[FCDDataModule] Split: "
            f"train={len(train_label_paths)} ({len(train_raw_label_paths)} raw + {len(self._extra_label_paths)} generated), "
            f"val={len(val_label_paths)} (raw only)"
        )

        kw = dict(
            fcd_intensity_range   = self.fcd_intensity_range,
            fcd_tail_length_range = self.fcd_tail_range,
        )

        self.train_ds = FCDDataset(
            self.ndim, train_label_paths, train_flair_paths, train_roi_paths,
            fused_paths      = train_fused_paths,
            native_synthesis = self.native_synthesis,
            **kw,
        )
        self.eval_ds = FCDDataset(
            self.ndim, val_label_paths, val_flair_paths, val_roi_paths,
            fused_paths      = val_fused_paths,
            native_synthesis = self.native_synthesis,
            **kw,
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0)

    def val_dataloader(self):
        return DataLoader(self.eval_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0)


# ══════════════════════════════════════════════════════════════════════════════
#  SharedSynth  —  geometry + GMM forward pass, intensity kept separate
# ══════════════════════════════════════════════════════════════════════════════
class SharedSynth(torch.nn.Module):
    """
    GMM synthesis + label remapping for the FCD segmentation pipeline.

    Synthetic branch: label map → one-hot → GMM → synthetic FLAIR image
    Real branch:      FLAIR + label + ROI passed through unchanged (no_augs=True)

    IntensityTransform is intentionally excluded — applied downstream after FCD augmentations.
    """

    N_CLASSES = 18  # valid labels are 0..18 inclusive (19 values)

    def __init__(self, synth, target_labels=None, native_synthesis: bool = False):
        super().__init__()
        self.synth            = synth
        self.target_labels    = target_labels or []
        self.native_synthesis = native_synthesis

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_class_params(self, params: dict):
        """Swap the GMM's per-class intensity params (e.g. per-subject stats)."""
        gmm = getattr(self.synth, 'gmm', None)
        if gmm is not None and hasattr(gmm, 'class_params'):
            gmm.class_params = params
        else:
            print('[SharedSynth] Warning: set_class_params called but no GMM found.')

    def forward(self, slab, img, lab, roi=None):
        """
        Parameters
        ----------
        slab : (1, D, H, W) int64   — label map used for GMM synthesis.
                                       Standard path: labels 0..18.
                                       Native path:   labels 0..18 + 21 (fusedmask).
        img  : (1, D, H, W) float32 — real FLAIR, passed through to real branch.
        lab  : (1, D, H, W) int64   — real label map, co-deformed with slab.
                                       Native path: fusedmask passed as both slab and lab.
        roi  : (1, D, H, W) int64   — binary ROI mask, co-deformed with slab.
                                       None on native path — lesion location encoded in slab.

        Returns
        -------
        simg     : (1, D, H, W) float32  — synthetic FLAIR from GMM.
        slab_out : (1, D, H, W) int64    — remapped label map {0..5} standard path,
                                           {0..6} native path (label 21 → class 6).
        rimg     : (1, D, H, W) float32  — real FLAIR, unchanged.
        rlab     : (1, D, H, W) int64    — real label remapped {0..5} standard path,
                                           {0..6} native path.
        rroi     : (1, D, H, W) int64    — ROI mask, unchanged. None on native path.
        """
        img = img.float()

        # Route based on which synthesiser is attached.
        # cornucopia's SynthFromLabelTransform has make_final (legacy/non-FLAIR path).
        # CustomSynthFromLabelTransform (FLAIR path) is a plain nn.Module — no make_final.
        if hasattr(self.synth, 'make_final'):
            return self._forward_standard(slab, img, lab, roi)
        return self._forward_custom(slab, img, lab, roi)

    # ------------------------------------------------------------------
    # Forward paths
    # ------------------------------------------------------------------

    def _forward_standard(self, slab, img, lab, roi):
        """Cornucopia path — full deformation + GMM + intensity. Used when modality != 'flair'.

        native_synthesis=True : label 21 pre-remapped to 19 before cornucopia synthesis
                                so the lesion gets its own GMM channel for unique intensity sampling.
                                Deformed lab (still has 21) is remapped via remap_labels → class 6.
        """
        # Pre-remap label 21 → 19 before cornucopia — avoids out-of-range LUT (sized to 19)
        # and gives the lesion its own GMM channel for unique intensity sampling in simg
        if self.native_synthesis:
            slab_synth = slab.clone()
            slab_synth[slab_synth == 21] = 19
        else:
            slab_synth = slab

        final        = self.synth.make_final(slab_synth, 1)
        final.deform = final.deform.make_final(slab_synth)
        simg, slab_out = final(slab_synth)

        if roi is not None:
            rimg, rlab, rroi = final.deform([img, lab, roi])
            rlab             = final.postproc(rlab)
            return simg, slab_out, rimg, rlab, rroi
        else:
            rimg, rlab = final.deform([img, lab])
            if self.native_synthesis:
                # rlab is deformed fusedmask — label 21 still intact after deformation
                # remap_labels maps 21 → class 6 for both synthetic and real targets
                slab_out = self.remap_labels(rlab)
                rlab_out = self.remap_labels(rlab)
            else:
                rlab_out = final.postproc(rlab)
            return simg, slab_out, rimg, rlab_out, None

    def _forward_custom(self, slab, img, lab, roi):
        """FLAIR path — deformation + per-class GMM. IntensityTransform applied downstream.

        native_synthesis=False : slab is labelmap (labels 0..18), roi co-deformed separately.
        native_synthesis=True  : slab is fusedmask (labels 0..18 + 21), roi absent — lesion
                                 already encoded as label 21 and remapped to class 6.
        """
        if roi is not None:
            oh_slab                                    = self._to_one_hot(slab)  # (19, D, H, W)
            simg, oh_slab_deformed, (rimg, rlab, rroi) = self.synth(oh_slab, coreg=[img, lab, roi])
            slab_deformed                              = oh_slab_deformed.argmax(dim=0, keepdim=True)
            slab_out                                   = self.remap_labels(slab_deformed)
            rlab_out                                   = self.remap_labels(rlab)
            return simg, slab_out, rimg, rlab_out, rroi
        else:
            oh_slab                              = self._to_one_hot(slab, num_classes=22)
            simg, oh_slab_deformed, (rimg, rlab) = self.synth(oh_slab, coreg=[img, lab])
            slab_deformed                        = oh_slab_deformed.argmax(dim=0, keepdim=True)
            slab_out                             = self.remap_labels(slab_deformed)
            rlab_out                             = self.remap_labels(rlab)
            return simg, slab_out, rimg, rlab_out, None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_one_hot(self, label_map: torch.Tensor, num_classes: int = None) -> torch.Tensor:
        if num_classes is None:
            num_classes = self.N_CLASSES + 1  # default: 19, covers labels 0..18
        return (
            torch.nn.functional.one_hot(
                label_map.long().squeeze(0),
                num_classes=num_classes
            )
            .permute(3, 0, 1, 2)
            .float()
        )

    def remap_labels(self, label_map: torch.Tensor) -> torch.Tensor:
        """
        Map sparse label values → consecutive model class indices.

        target_labels = [(1,), (2,), (3,), (4,), (18,)]
        Mapping:
            label 1  → class 1 (White Matter)
            label 2  → class 2 (Cerebral Cortex)
            label 3  → class 3 (Deep Gray Matter)
            label 4  → class 4 (CSF)
            label 18 → class 5 (WM-GM Separator)
            label 21 → class 6 (FCD Lesion)  [native_synthesis path only]
            all else → class 0 (Background)
        """
        max_value = int(label_map.max().item()) + 1
        lut       = torch.zeros(max_value, dtype=torch.long, device=label_map.device)

        for class_index, group in enumerate(self.target_labels, start=1):
            for value in group:
                if value < max_value:
                    lut[value] = class_index

        if self.native_synthesis and 21 < max_value:
            lut[21] = 6                               # FCD lesion remapped from fusedmask label 21

        nb_classes = len(self.target_labels) + 1      # 6: background + 5 tissues
        if self.native_synthesis:
            nb_classes += 1                           # +1 to accommodate class 6 (FCD lesion)

        return torch.clamp(lut[label_map.float().round().long()], 0, nb_classes - 1)


# ══════════════════════════════════════════════════════════════════════════════
#  SynthesisPipelineDebugger
# ──────────────────────────────────────────────────────────────────────────────
#  Saves intermediate NIfTI volumes at every stage of the synthesis pipeline
#  for a user-specified set of subject IDs.  All other subjects are untouched.
#
#  Usage
#  ─────
#  Pass a set of subject IDs (as they appear in `subject_id` batch keys,
#  e.g. "sub-00001") to Model via debug_subject_ids.  A unique save fires
#  once per subject per training run (not once per epoch) to avoid disk bloat.
#  To re-trigger saves (e.g. to see a later epoch), delete the saved folder.
#
#  Output layout
#  ─────────────
#  <OUTPUT_FOLDER>/pipeline_debug/<subject_id>/
#      stage0_input_labelmap.nii.gz      — raw label map from disk (int)
#      stage0_input_flair.nii.gz         — raw real FLAIR from disk (float)
#      stage0_input_roi.nii.gz           — raw binary ROI from disk (int)
#      stage0_input_fusedmask.nii.gz     — fusedmask (native path only, int)
#      stage1_after_synth_simg.nii.gz    — synthetic image from GMM (float, pre-clamp)
#      stage1_after_synth_slab.nii.gz    — deformed+remapped label map (int)
#      stage1_after_synth_rimg.nii.gz    — deformed real FLAIR (float)
#      stage1_after_synth_rlab.nii.gz    — deformed+remapped real label (int)
#      stage1_after_synth_rroi.nii.gz    — deformed ROI (int, non-native only)
#      stage2_after_clamp.nii.gz         — simg after /255 clamp to [0,1] (flair path only)
#      stage3_after_fcd_aug.nii.gz       — after FCDAugmentations (non-native only)
#      stage3_after_fcd_aug_roi.nii.gz   — ROI after thickening aug (non-native only)
#      stage4_after_intensity.nii.gz     — final aug_image_item after IntensityTransform
#      stage5_label_fused_slab.nii.gz    — slab after label fusion (class 6 stamped, non-native)
#      stage5_label_fused_rlab.nii.gz    — rlab after label fusion (class 6 stamped, non-native)
#      summary.txt                        — per-stage tensor stats (min/max/mean/shape)
# ══════════════════════════════════════════════════════════════════════════════
class SynthesisPipelineDebugger:
    """
    Saves intermediate volumes at every synthesis stage for selected subjects.

    Parameters
    ----------
    debug_subject_ids : set[str]
        Subject IDs (e.g. {"sub-00001", "sub-00027"}) for which to save debug
        volumes.  Pass an empty set to disable entirely.
    output_root : str
        Root directory for debug output.  A sub-folder per subject is created.
    save_once_per_subject : bool
        If True (default), each subject is only saved on the first encounter
        during the run.  Set to False to overwrite on every step (verbose).
    """

    def __init__(
        self,
        debug_subject_ids: set,
        output_root: str,
        save_once_per_subject: bool = True,
    ):
        self.debug_subject_ids     = set(debug_subject_ids)
        self.output_root           = output_root
        self.save_once             = save_once_per_subject
        self._saved: set           = set()  # tracks which subjects have already been saved
        self._aug_type_cache: dict = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def is_debug_subject(self, subject_id: str, aug_type: str = "") -> bool:
        if subject_id not in self.debug_subject_ids:
            return False
        if self.save_once and subject_id in self._saved:
            return False
        self._aug_type_cache[subject_id] = aug_type
        return True

    def save_stage0_inputs(
        self,
        subject_id: str,
        label_t: torch.Tensor,
        flair_t: torch.Tensor,
        roi_t: torch.Tensor,
        fusedmask_t: Optional[torch.Tensor] = None,
    ):
        """Stage 0 — raw tensors loaded from disk, before any processing."""
        out_dir = self._subject_dir(subject_id, self._aug_type_cache.get(subject_id, ""))

        self._save_nii(label_t.squeeze(), out_dir, "stage0_input_labelmap.nii.gz", dtype=np.int16)
        self._save_nii(flair_t.squeeze(), out_dir, "stage0_input_flair.nii.gz",    dtype=np.float32)
        self._save_nii(roi_t.squeeze(),   out_dir, "stage0_input_roi.nii.gz",      dtype=np.uint8)
        if fusedmask_t is not None:
            self._save_nii(fusedmask_t.squeeze(), out_dir, "stage0_input_fusedmask.nii.gz", dtype=np.int16)

        self._log_stats(subject_id, "stage0_input_labelmap",  label_t)
        self._log_stats(subject_id, "stage0_input_flair",     flair_t)
        self._log_stats(subject_id, "stage0_input_roi",       roi_t)
        if fusedmask_t is not None:
            self._log_stats(subject_id, "stage0_input_fusedmask", fusedmask_t)

        print(f"[PipelineDebug] {subject_id} | Stage 0 saved → {out_dir}")

    def save_stage1_after_synth(
        self,
        subject_id: str,
        simg: torch.Tensor,
        slab_out: torch.Tensor,
        rimg: torch.Tensor,
        rlab_out: torch.Tensor,
        rroi: Optional[torch.Tensor] = None,
    ):
        """Stage 1 — output of SharedSynth: deformed + GMM-sampled synthetic image."""
        out_dir = self._subject_dir(subject_id, self._aug_type_cache.get(subject_id, ""))

        self._save_nii(simg.squeeze(),     out_dir, "stage1_after_synth_simg.nii.gz", dtype=np.float32)
        self._save_nii(slab_out.squeeze(), out_dir, "stage1_after_synth_slab.nii.gz", dtype=np.int16)
        self._save_nii(rimg.squeeze(),     out_dir, "stage1_after_synth_rimg.nii.gz", dtype=np.float32)
        self._save_nii(rlab_out.squeeze(), out_dir, "stage1_after_synth_rlab.nii.gz", dtype=np.int16)
        if rroi is not None:
            self._save_nii(rroi.squeeze(), out_dir, "stage1_after_synth_rroi.nii.gz", dtype=np.uint8)

        self._log_stats(subject_id, "stage1_simg",     simg)
        self._log_stats(subject_id, "stage1_slab_out", slab_out)
        self._log_stats(subject_id, "stage1_rimg",     rimg)
        self._log_stats(subject_id, "stage1_rlab_out", rlab_out)

        print(f"[PipelineDebug] {subject_id} | Stage 1 saved → {out_dir}")

    def save_stage2_after_fcd_aug(
        self,
        subject_id: str,
        aug_img: torch.Tensor,
        rroi_3d: torch.Tensor,
        choices: list,
    ):
        """Stage 2 — after FCDAugmentations (SynthFCD path only)."""
        out_dir = self._subject_dir(subject_id, self._aug_type_cache.get(subject_id, ""))

        self._save_nii(aug_img.squeeze(), out_dir, "stage2_after_fcd_aug.nii.gz",     dtype=np.float32)
        self._save_nii(rroi_3d.squeeze(), out_dir, "stage2_after_fcd_aug_roi.nii.gz", dtype=np.uint8)

        self._log_stats(subject_id, f"stage2_after_fcd_aug (choices={choices})", aug_img)
        print(f"[PipelineDebug] {subject_id} | Stage 2 (FCD aug: {choices}) saved → {out_dir}")

    def save_stage3_after_intensity(self, subject_id: str, aug_image_item: torch.Tensor):
        """Stage 3 — final aug_image_item after IntensityTransform, normalized to [0,1]."""
        out_dir = self._subject_dir(subject_id, self._aug_type_cache.get(subject_id, ""))
        self._save_nii(aug_image_item.squeeze(), out_dir, "stage3_after_intensity.nii.gz", dtype=np.float32)
        self._log_stats(subject_id, "stage3_after_intensity", aug_image_item)
        print(f"[PipelineDebug] {subject_id} | Stage 3 (intensity aug) saved → {out_dir}")

    def save_stage4_label_fusion(
        self,
        subject_id: str,
        slab_with_fcd: torch.Tensor,
        rlab_with_fcd: torch.Tensor,
    ):
        """Stage 4 — after label fusion: ROI voxels stamped as class 6 (SynthFCD path only)."""
        out_dir = self._subject_dir(subject_id, self._aug_type_cache.get(subject_id, ""))

        self._save_nii(slab_with_fcd.squeeze(), out_dir, "stage4_label_fused_slab.nii.gz", dtype=np.int16)
        self._save_nii(rlab_with_fcd.squeeze(), out_dir, "stage4_label_fused_rlab.nii.gz", dtype=np.int16)

        self._log_stats(subject_id, "stage4_slab_with_fcd", slab_with_fcd)
        self._log_stats(subject_id, "stage4_rlab_with_fcd", rlab_with_fcd)

        print(f"[PipelineDebug] {subject_id} | Stage 4 (label fusion) saved → {out_dir}")

    def save_stage5_after_intensity(self, subject_id: str, real_image_item: torch.Tensor):
        """Stage 5 — final rimg after IntensityTransform, normalized to [0,1]."""
        out_dir = self._subject_dir(subject_id, self._aug_type_cache.get(subject_id, ""))
        self._save_nii(real_image_item.squeeze(), out_dir, "stage5_after_intensity.nii.gz", dtype=np.float32)
        self._log_stats(subject_id, "stage5_after_intensity", real_image_item)
        print(f"[PipelineDebug] {subject_id} | Stage 5 (intensity aug) saved → {out_dir}")

    def mark_saved(self, subject_id: str):
        """Call after all stages are saved to prevent re-saving this subject."""
        self._saved.add(subject_id)
        summary_path = os.path.join(self._subject_dir(subject_id, self._aug_type_cache.get(subject_id, "")), "summary.txt")
        try:
            with open(summary_path, 'a') as f:
                f.write(f"\n[PipelineDebug] All stages saved for {subject_id}.\n")
        except Exception:
            pass

    # ── Internals ─────────────────────────────────────────────────────────────

    def _subject_dir(self, subject_id: str, aug_type: str = "") -> str:
        folder = f"{subject_id}-{aug_type.replace('+', '-')}" if aug_type else subject_id
        d      = os.path.join(self.output_root, folder)
        os.makedirs(d, exist_ok=True)
        return d

    def _save_nii(self, tensor: torch.Tensor, out_dir: str, fname: str, dtype):
        try:
            arr = tensor.detach().cpu().numpy().astype(dtype)
            nib.save(nib.Nifti1Image(arr, np.eye(4)), os.path.join(out_dir, fname))
        except Exception as e:
            print(f"[PipelineDebug] WARNING: could not save {fname}: {e}")

    def _log_stats(self, subject_id: str, stage_name: str, tensor: torch.Tensor):
        try:
            t    = tensor.detach().cpu().float()
            line = (
                f"[{stage_name}] "
                f"shape={tuple(t.shape)}  "
                f"dtype={tensor.dtype}  "
                f"min={t.min().item():.4f}  "
                f"max={t.max().item():.4f}  "
                f"mean={t.mean().item():.4f}  "
                f"unique_vals={len(torch.unique(t))}\n"
            )
            summary_path = os.path.join(self._subject_dir(subject_id, self._aug_type_cache.get(subject_id, "")), "summary.txt")
            with open(summary_path, 'a') as f:
                f.write(line)
        except Exception as e:
            print(f"[PipelineDebug] WARNING: could not log stats for {stage_name}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  Model  —  6-class grouped segmentation (brain structures + FCD lesion)
# ══════════════════════════════════════════════════════════════════════════════
class Model(pl.LightningModule):
    # ── Class-level label definitions — single source of truth ───────────────
    TARGET_LABELS = [
        (1,),   # White Matter
        (2,),   # Cerebral Cortex
        (3,),   # Deep Gray Matter
        (4,),   # CSF
        (18,),  # WM-GM Separator
    ]

    def __init__(
            self,
            ndim: int = 3,
            nb_classes: int = 7,                              # background + 5 tissues + FCD lesion
            seg_nb_levels: int = 6,
            seg_features: Sequence[int] = (16, 32, 64, 128, 256, 512),
            seg_activation: str = 'ReLU',
            seg_nb_conv: int = 2,
            seg_norm: Optional[str] = 'instance',
            loss: str = 'dice_ce',
            alpha: float = 1.0,
            optimizer: str = 'Adam',
            optimizer_options: Optional[dict] = None,
            time_limit_minutes: float = None,
            flair_modality: bool = False,
            flair_stats_csv: Optional[str] = FLAIR_STATS_CSV,
            n_best_batches: int = 2,
            native_synthesis: bool = False,
            lesion_gmm_params: Optional[dict] = None,  # GMM (μ, σ) for label 21 — native path only
            debug_subject_ids: Optional[list] = None,  # subject IDs to save pipeline stages for
    ):
        super().__init__()
        self.save_hyperparameters()

        # ── Native Approach ───────────────────────────────────────────────────────
        self.native_synthesis  = native_synthesis
        self.lesion_gmm_params = lesion_gmm_params or {'mu': (100, 180), 'sigma': (5, 20)}

        # ── Pipeline debugger ─────────────────────────────────────────────────────
        # Saves intermediate NIfTI volumes at every synthesis stage for the
        # specified subjects.  Pass an empty list (or omit) to disable.
        # Example CLI usage:
        #   --model.debug_subject_ids '["sub-00001", "sub-00027", "sub-00065"]'
        modality_tag = "flair" if flair_modality else "random"
        synth_tag    = "native" if native_synthesis else "synthFCD"
        self._pipeline_debugger = SynthesisPipelineDebugger(
            debug_subject_ids=set(debug_subject_ids or []),
            output_root=os.path.join(OUTPUT_FOLDER, f"pipeline_debug_{modality_tag}_{synth_tag}"),
            save_once_per_subject=True,
        )

        self.optimizer_name       = optimizer
        self.optimizer_options    = dict(optimizer_options or {'lr': 1e-4})
        self.time_limit_minutes   = time_limit_minutes
        self.alpha                = alpha
        self.flair_stats_csv      = flair_stats_csv
        self.target_labels        = self.TARGET_LABELS

        # ── Sub-modules ───────────────────────────────────────────────────────
        self.subject_params_cache   = self._load_subject_params()
        seg_net                     = self._build_seg_network(ndim, nb_classes, seg_features, seg_activation, seg_nb_levels, seg_nb_conv, seg_norm)
        synth                       = self._build_synth(flair_modality)
        loss_fn                     = self._build_loss(loss)
        self.network                = SynthSeg(seg_net, synth, loss_fn)
        self.intensity_aug          = self._build_intensity_aug()
        self.rimg_normalizer        = cc.QuantileTransform(clip=True)
        self.fcd_aug                = FCDAugmentations()

        # ── Metrics ───────────────────────────────────────────────────────────
        _m                  = dict(include_background=False, num_classes=nb_classes, input_format='index')
        self.val_dice       = dice_compute(average='micro', **_m)
        self.val_dice_fcd   = dice_compute(average='none',  **_m)

        # ── Manual optimisation ───────────────────────────────────────────────
        self.automatic_optimization = False
        self.network.set_backward(self.manual_backward)

        # ── State ─────────────────────────────────────────────────────────────
        self.n_best_batches    = n_best_batches
        self._val_batch_cache  = []   # top-n_best_batches entries by lowest loss
        self._val_worst_cache  = []   # 1 entry with highest loss

    # ══════════════════════════════════════════════════════════════════════════
    #  Lifecycle hooks
    # ══════════════════════════════════════════════════════════════════════════

    def on_train_start(self):
        # Wire the optimizer getter into SynthSeg so train_step() can call it.
        self.network.optimizers = self.optimizers

        # ── Model Architecture ────────────────────────────────────────────────
        seg = self.network.segnet
        print("\n" + "═" * 60)
        print("DEBUG: Model Architecture")
        print("═" * 60)
        print(f"  Backbone        : UNet")
        print(f"  seg_features    : {self.hparams.seg_features}")
        print(f"  seg_nb_levels   : {self.hparams.seg_nb_levels}")
        print(f"  seg_nb_conv     : {self.hparams.seg_nb_conv}")
        print(f"  seg_norm        : {self.hparams.seg_norm}")
        print(f"  nb_classes      : {self.hparams.nb_classes}")
        modality = "Flair" if self.hparams.flair_modality else "Random Modality"
        print(f"  modality        : {modality}")

        total_params = sum(p.numel() for p in seg.parameters())
        trainable    = sum(p.numel() for p in seg.parameters() if p.requires_grad)
        print(f"  Total params    : {total_params:,}")
        print(f"  Trainable params: {trainable:,}")

        # ── Synthesis Pipeline ────────────────────────────────────────────────
        approach = "Native SynthSeg Approach" if self.hparams.native_synthesis else "SynthFCD Approach"
        print(f"\nDEBUG: Synthesis Pipeline  [{approach}]")
        print("─" * 60)
        print(f"  SharedSynth.synth type : {type(self.network.synth.synth).__name__}")
        gmm = getattr(self.network.synth.synth, 'gmm', None)
        print(f"  GMM type               : {type(gmm).__name__ if gmm else 'None'}")
        print(f"  GMM class_params keys  : {sorted(gmm.class_params.keys()) if gmm and hasattr(gmm, 'class_params') else 'N/A'}")
        print(f"  IntensityAug type      : {type(self.intensity_aug).__name__}")
        print(f"  Subject params cached  : {len(self.subject_params_cache)} subjects")

        if self.hparams.native_synthesis:
            print(f"  Lesion GMM (class 21)  : {self.lesion_gmm_params}  [injected per-step]")
            print(f"  FCDAugmentations       : disabled  [native path — GMM handles lesion appearance]")
        else:
            print(f"  FCDAugmentations       : {type(self.fcd_aug).__name__}")

        # ── Pipeline Debugger ─────────────────────────────────────────────────
        dbg = self._pipeline_debugger
        if dbg.debug_subject_ids:
            print(f"\nDEBUG: Pipeline Debugger ENABLED")
            print(f"─" * 60)
            print(f"  Subjects to debug      : {sorted(dbg.debug_subject_ids)}")
            print(f"  Output root            : {dbg.output_root}")
            print(f"  Save once per subject  : {dbg.save_once}")
            print(f"  Stages saved           : 0=inputs, 1=synth, 2=clamp(flair), "
                  f"3=fcd_aug(non-native), 4=intensity, 5=label_fusion(non-native)")
        else:
            print(f"\n  Pipeline Debugger      : DISABLED (no debug_subject_ids set)")

    # ══════════════════════════════════════════════════════════════════════════
    #  Private builders  (called only from __init__)
    # ══════════════════════════════════════════════════════════════════════════
    def _load_subject_params(self) -> dict:
        """Pre-load per-subject GMM parameters from CSV into a lookup cache."""
        cache = {}
        if not (self.flair_stats_csv and os.path.exists(self.flair_stats_csv)):
            return cache
        try:
            df = pd.read_csv(self.flair_stats_csv)
            df['subject'] = df['subject'].astype(str).str.strip()
            # range(19): covers classes 0–18 inclusive (WM-GM Separator = 18)
            default_keys = set(range(19)) | set(FLAIR_CLASS_PARAMS.keys())

            for subj in df['subject'].unique():
                params = {
                    int(r['class_id']): {
                        'mu':    (float(r['mu_lo']),    float(r['mu_hi'])),
                        'sigma': (float(r['sigma_lo']), float(r['sigma_hi'])),
                    }
                    for _, r in df[df['subject'] == subj].iterrows()
                }
                for cls in default_keys:
                    params.setdefault(
                        cls, FLAIR_CLASS_PARAMS.get(cls, {'mu': (0, 255), 'sigma': (0, 16)})
                    )
                cache[subj] = params

            print(f'[Model] Preloaded per-subject params for {len(cache)} subjects.')
        except Exception as exc:
            print(f'[Model] Warning: failed to parse CSV — {exc}')
        return cache

    def _build_seg_network(self, ndim, nb_classes, features, activation, nb_levels, nb_conv, norm):
        backbone = UNet(ndim, nb_features=features, activation=activation,
                        nb_levels=nb_levels, nb_conv=nb_conv, norm=norm)
        return SegNet(ndim, 1, nb_classes, backbone=backbone, activation=None)

    def _build_synth(self, flair_modality: bool) -> SharedSynth:
        if flair_modality:
            raw = CustomSynthFromLabelTransform(
                num_ch=1, class_params=FLAIR_CLASS_PARAMS, use_per_class_gmm=True,
                gmm_fwhm=10, bias=7, gamma=0.5, motion_fwhm=2.0, resolution=3,
                snr=10, gfactor=3, rotation=15, shears=0.012, zooms=0.15,
                elastic=0.05, elastic_nodes=10, order=3, no_augs=True,
            )
        else:
            raw = SynthFromLabelTransform(
                target_labels=self.target_labels,
                elastic=0.05, elastic_nodes=10, rotation=15, shears=0.012,
                zooms=0.15, resolution=3, motion_fwhm=2.0, snr=10,
                gmm_fwhm=10, gamma=0.5, bias=7, bias_strength=0.5,
            )
            raw.intensity = IdentityTransform()

        return SharedSynth(raw, target_labels=self.target_labels, native_synthesis=self.native_synthesis)

    def _build_loss(self, loss: str):
        options = {
            'dice':          lambda: DiceLoss(activation='Softmax'),
            'logitmse':      lambda: LogitMSELoss(),
            'cat':           lambda: CatLoss(activation='Softmax'),
            'catmse':        lambda: CatMSELoss(activation='Softmax'),
            'dice_ce':       lambda: DiceCELoss(activation='Softmax'),
            'focal_tversky': lambda: FocalTverskyLoss(activation='Softmax'),
        }
        if loss not in options:
            raise ValueError(f"Unsupported loss '{loss}'. Choose from: {list(options)}")
        return options[loss]()

    def _build_intensity_aug(self):
        return IntensityTransform(
            bias=7, bias_strength=0.2, gamma=0.3, motion_fwhm=3,
            resolution=4, snr=20, gfactor=2, order=3,
        )

    # ══════════════════════════════════════════════════════════════════════════
    #  GMM subject param switching
    # ══════════════════════════════════════════════════════════════════════════

    def _set_subject_params(self, subject_id: Optional[str]):
        params = (
            self.subject_params_cache[subject_id]
            if subject_id and subject_id in self.subject_params_cache
            else FLAIR_CLASS_PARAMS
        )
        if self.hparams.flair_modality:
            if self.hparams.native_synthesis:
                params = dict(params)
                if 21 not in params:  # only uses lesion_gmm_params as fallback if 21 not in CSV
                    params[21] = self.lesion_gmm_params
            self.network.synth.set_class_params(params)

    # ══════════════════════════════════════════════════════════════════════════
    #  Augmentation pipeline
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _parse_aug_choices(aug_type: str) -> list:
        if aug_type == 'combo':
            return random.sample(['blur', 'zoom', 'hyper', 'trans'], random.randint(1, 4))
        return aug_type.split('+')

    def _apply_fcd_augmentations(
            self,
            img: torch.Tensor,
            roi: torch.Tensor,
            choices: list,
            params: dict,
    ) -> tuple:
        """
        Apply the FCD augmentation chain with pre-sampled random parameters.

        All random values are drawn here — once per subject, before any
        augmentation method is invoked — so every subject is guaranteed a
        statistically independent sample regardless of augmentation order.

        Parameters
        ----------
        img     : (Z, Y, X) synthetic image tensor on GPU
        roi     : (Z, Y, X) binary ROI mask on GPU
        choices : list of augmentation names, e.g. ['blur', 'hyper']
        params  : dict of (min, max) range tuples from the batch:
                    'int_rng'     → intensity_range for hyper / trans
                    'blur_sigma'  → sigma_range for blur
                    'zoom_f'      → zoom_factor range
                    'hyper_sigma' → sigma_range for hyper
                    'trans_sigma' → sigma_range for trans
                    'tail_length' → (min_int, max_int) for trans tail

        Returns
        -------
        img : augmented image tensor
        roi : (possibly updated) ROI mask tensor
        """
        # ── Pre-sample every random scalar up-front ──────────────────────────────
        # Each draw is independent; order does not affect the values.
        pre = {
            'blur_sigma':       random.uniform(*params['blur_sigma']),
            'zoom_factor':      random.uniform(*params['zoom_f']),
            'hyper_intensity':  random.uniform(*params['int_rng']),
            'hyper_sigma':      random.uniform(*params['hyper_sigma']),
            'trans_intensity':  random.uniform(*params['int_rng']),
            'trans_sigma':      random.uniform(*params['trans_sigma']),
        }

        # ── Apply augmentations using the pre-sampled values ─────────────────────
        for ch in choices:
            if ch == 'zoom':
                img, roi = self.fcd_aug.apply_roi_thickening(
                    img, roi,
                    zoom_factor=pre['zoom_factor'],
                )

            elif ch == 'blur':
                img = self.fcd_aug.apply_roi_augmentations_blured(
                    img, roi,
                    sigma=pre['blur_sigma'],
                )

            elif ch in ('hyper', 'trans'):
                img = self.fcd_aug.apply_roi_augmentations_hyperintensity(
                    img, roi,
                    intensity_factor=pre['hyper_intensity'],
                    sigma=pre['hyper_sigma'],
                )
            # ── Debug: check for NaN/inf after every augmentation step ──────────
            if not torch.isfinite(img).all():
                bad = (~torch.isfinite(img)).sum().item()
                print(f"[NaN DETECTED] after aug='{ch}' | bad_voxels={bad} "
                      f"| params={pre}")
                # Return the pre-augmentation image to avoid propagating NaN
                return img, roi

        return img, roi

    def _process_single_sample(self, batch: dict, i: int):
        """
        Full single-sample synthesis pipeline.
        native_synthesis=False : SharedSynth → FCD aug → IntensityTransform → label fusion
        native_synthesis=True  : SharedSynth (fusedmask) → IntensityTransform  (no aug, no fusion)
        Returns (aug_image, aug_mask, real_image, real_mask) or None if skipped.

        simg normalization
        ──────────────────
        simg exits SharedSynth already in [0, 1] regardless of path:
        - flair_modality=True  : QuantileTransform(clip=True) applied inside
                                 SynthFromLabelTransform after GMM (no_augs=True path)
        - flair_modality=False : cc.IntensityTransform ends with QuantileTransform

        rimg normalization
        ──────────────────
        rimg is the real FLAIR from disk (~0–245). Normalized here via
        QuantileTransform(clip=True) before IntensityTransform, matching simg's scale.

        Debug instrumentation
        ─────────────────────
        If subject_id appears in self._pipeline_debugger.debug_subject_ids, one NIfTI
        volume is saved at every pipeline stage.  This fires once per subject per run
        (controlled by SynthesisPipelineDebugger.save_once_per_subject).
        """
        label_t    = batch['label_t'][i]
        flair_t    = batch['flair_t'][i].float()
        roi_t      = batch['roi_t'][i]
        aug_type   = batch['aug_type'][i]
        subject_id = batch.get('subject_id', [None] * len(batch['label_t']))[i]

        # Validate the actual synthesis input — fusedmask on native path, labelmap otherwise
        input_t = batch['fusedmask_t'][i] if self.hparams.native_synthesis else label_t
        if input_t.sum() == 0 or torch.isnan(input_t.float()).any():
            return None

        self._set_subject_params(subject_id)

        # ── Debug: Stage 0 — raw inputs from disk ─────────────────────────────────
        dbg = self._pipeline_debugger
        is_debug = dbg.is_debug_subject(subject_id, aug_type)
        if is_debug:
            fusedmask_for_debug = (
                batch['fusedmask_t'][i] if self.hparams.native_synthesis else None
            )
            dbg.save_stage0_inputs(
                subject_id, label_t, flair_t, roi_t, fusedmask_for_debug
            )

        # ── Native synthesis path ─────────────────────────────────────────────────
        if self.hparams.native_synthesis:
            fusedmask_t = input_t  # already extracted above

            # fusedmask acts as both slab (GMM input) and lab (real branch label target)
            # roi omitted — lesion is already encoded as label 21 in fusedmask
            simg, slab_out, rimg, rlab_out, _ = self.network.synth(
                fusedmask_t, flair_t, fusedmask_t
            )

            # ── Debug: Stage 1 — after SharedSynth (deform + GMM) ─────────────────
            if is_debug:
                dbg.save_stage1_after_synth(
                    subject_id, simg, slab_out, rimg, rlab_out, rroi=None
                )

            # simg exits SharedSynth already in [0,1] — no manual normalization needed
            simg_3d = simg.squeeze(0).float()

            aug_out = self.intensity_aug(simg_3d.unsqueeze(0))
            aug_image_item = aug_out[0] if isinstance(aug_out, (list, tuple)) else aug_out

            # ── Guard: catch NaN/inf introduced by IntensityTransform ─────────────
            if not torch.isfinite(aug_image_item).all():
                bad = (~torch.isfinite(aug_image_item)).sum().item()
                print(f"[WARN] Stage 3 (IntensityTransform) produced {bad} non-finite voxels "
                      f"for subject={subject_id} — skipping sample")
                return None

            # ── Debug: Stage 3 — after IntensityTransform ─────────────────────────
            if is_debug:
                dbg.save_stage3_after_intensity(subject_id, aug_image_item)
                dbg.mark_saved(subject_id)

            # slab_out and rlab_out already have class 6 from remap_labels — no label fusion needed
            rimg_norm = rimg.float() if rimg.dim() == 4 else rimg.float().unsqueeze(0)
            rimg_norm = self.rimg_normalizer.make_final(rimg_norm)(rimg_norm)
            rimg_out  = self.intensity_aug(rimg_norm)
            rimg_norm = rimg_out[0] if isinstance(rimg_out, (list, tuple)) else rimg_out

            # ── Debug: Stage 5 — rimg after normalization + IntensityTransform ──────
            if is_debug:
                dbg.save_stage5_after_intensity(subject_id, rimg_norm)
                dbg.mark_saved(subject_id)

            return (
                aug_image_item,
                slab_out.long(),
                rimg_norm,
                rlab_out.long(),
            )

        # ── Augmented synthesis path (original) ───────────────────────────────────
        aug_params = {
            'int_rng':     (batch['int_factor_min'][i].item(),  batch['int_factor_max'][i].item()),
            'blur_sigma':  (batch['blur_sigma_min'][i].item(),  batch['blur_sigma_max'][i].item()),
            'zoom_f':      (batch['zoom_f_min'][i].item(),      batch['zoom_f_max'][i].item()),
            'hyper_sigma': (batch['hyper_sigma_min'][i].item(), batch['hyper_sigma_max'][i].item()),
            'trans_sigma': (batch['trans_sigma_min'][i].item(), batch['trans_sigma_max'][i].item()),
        }

        simg, slab, rimg, rlab, rroi = self.network.synth(label_t, flair_t, label_t, roi_t)
        simg_3d = simg.squeeze(0).float()
        slab_3d = slab.squeeze(0).long()
        rroi_3d = (rroi.squeeze(0) > 0).long()

        # ── Debug: Stage 1 — after SharedSynth (deform + GMM) ─────────────────────
        if is_debug:
            dbg.save_stage1_after_synth(subject_id, simg, slab, rimg, rlab, rroi)

        choices = self._parse_aug_choices(aug_type)
        aug_img, rroi_3d = self._apply_fcd_augmentations(simg_3d.clone(), rroi_3d, choices, aug_params)

        # ── Debug: Stage 2 — after FCDAugmentations ───────────────────────────────
        if is_debug:
            dbg.save_stage2_after_fcd_aug(subject_id, aug_img, rroi_3d, choices)

        aug_out = self.intensity_aug(aug_img.float().unsqueeze(0))
        aug_image_item = aug_out[0] if isinstance(aug_out, (list, tuple)) else aug_out

        # ── Guard: catch NaN/inf introduced by IntensityTransform ─────────────────
        if not torch.isfinite(aug_image_item).all():
            bad = (~torch.isfinite(aug_image_item)).sum().item()
            print(f"[WARN] Stage 3 (IntensityTransform) produced {bad} non-finite voxels "
                  f"for subject={subject_id}, aug={choices} — skipping sample")
            return None

        # ── Debug: Stage 3 — after IntensityTransform ─────────────────────────────
        if is_debug:
            dbg.save_stage3_after_intensity(subject_id, aug_image_item)

        slab_with_fcd = slab_3d.clone()
        slab_with_fcd[rroi_3d > 0] = 6

        rlab_with_fcd = rlab.long().squeeze(0).clone()
        rlab_with_fcd[rroi_3d > 0] = 6

        # ── Debug: Stage 4 — after label fusion ───────────────────────────────────
        if is_debug:
            dbg.save_stage4_label_fusion(
                subject_id, slab_with_fcd.unsqueeze(0), rlab_with_fcd.unsqueeze(0)
            )
            dbg.mark_saved(subject_id)

        rimg_norm = rimg.float() if rimg.dim() == 4 else rimg.float().unsqueeze(0)
        rimg_norm = self.rimg_normalizer.make_final(rimg_norm)(rimg_norm)
        rimg_out  = self.intensity_aug(rimg_norm)
        rimg_norm = rimg_out[0] if isinstance(rimg_out, (list, tuple)) else rimg_out

        # ── Debug: Stage 5 — rimg after normalization + IntensityTransform ──────────
        if is_debug:
            dbg.save_stage5_after_intensity(subject_id, rimg_norm)

        return (
            aug_image_item,
            slab_with_fcd.unsqueeze(0),
            rimg_norm,
            rlab_with_fcd.unsqueeze(0),
        )

    def synthesize_batch(self, batch: dict):
        """
        Run the synthesis pipeline for every sample in the batch.
        Returns stacked tensors + a list of subject_ids that survived synthesis
        (samples that returned None are excluded from both).
        """
        results      = []
        subject_ids  = []
        n            = len(batch['label_t'])
        sid_list     = batch.get('subject_id', [None] * n)
        device_type  = 'cuda' if self.device.type == 'cuda' else 'cpu'

        with torch.autocast(device_type=device_type, enabled=False):
            for i in range(n):
                out = self._process_single_sample(batch, i)
                if out is not None:
                    results.append(out)
                    subject_ids.append(sid_list[i])
            if not results:
                return None

        aug_images, aug_masks, real_images, real_masks = zip(*results)
        return (
            torch.stack(aug_images).float(),
            torch.stack(aug_masks).long(),
            torch.stack(real_images).float(),
            torch.stack(real_masks).long(),
            subject_ids,                        # ← list of str|None, same length as batch dim
        )

    # ══════════════════════════════════════════════════════════════════════════
    #  Training
    # ══════════════════════════════════════════════════════════════════════════

    def training_step(self, batch, batch_idx):
        # Periodically free CUDA cache to prevent fragmentation over long runs
        if self.trainer.current_epoch % 10 == 0 and batch_idx == 0:
            torch.cuda.empty_cache()

        result = self.synthesize_batch(batch)
        if result is None:
            return None
        aug_image, aug_mask, real_image, real_mask, _ = result  # subject_ids unused in training

        loss_synth, loss_real = self.network.train_step(
            aug_image, aug_mask, real_image, real_mask)

        loss              = loss_synth + self.alpha * loss_real
        actual_batch_size = aug_image.shape[0]
        self.log('train_loss', loss, prog_bar=True, batch_size=actual_batch_size)
        return loss

    # ══════════════════════════════════════════════════════════════════════════
    #  Validation
    # ══════════════════════════════════════════════════════════════════════════

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            result = self.synthesize_batch(batch)
            if result is None:
                return None
            aug_image, aug_mask, real_image, real_mask, subject_ids = result

            loss_synth, loss_real, pred_synth, pred_real = self.network.eval_for_plot(
                aug_image, aug_mask, real_image, real_mask)

        pred_labels   = pred_real.cpu().argmax(dim=1)
        target_labels = real_mask.cpu().squeeze(1).long()

        self.val_dice.update(pred_labels, target_labels)
        self.val_dice_fcd.update(pred_labels, target_labels)

        loss              = loss_synth + self.alpha * loss_real
        actual_batch_size = aug_image.shape[0]
        self.log('eval_loss', loss, prog_bar=True, batch_size=actual_batch_size)

        # ── Per-subject metrics (every 10 epochs) ─────────────────────────────────
        # Iterates over subjects in the batch individually so each gets its own
        # Dice and loss logged under val_dice_<subject_id> / val_loss_<subject_id>.
        if self.trainer.current_epoch % 10 == 0:
            _m = dict(include_background=False, num_classes=self.hparams.nb_classes,
                      input_format='index')
            for j, subj_id in enumerate(subject_ids):
                if subj_id is None:
                    continue

                subj_pred_labels = pred_labels[j:j+1]
                subj_target      = target_labels[j:j+1]
                subj_pred_logits = pred_real[j:j+1]
                subj_mask        = real_mask[j:j+1]

                # Dice — fresh metric instance to avoid state bleed between subjects
                _dice = dice_compute(average='micro', **_m)
                _dice.update(subj_pred_labels, subj_target)
                subj_dice = _dice.compute()

                # Loss
                subj_loss = self.network.loss(subj_pred_logits, subj_mask)

                self.log(f'val_dice_{subj_id}', subj_dice,        prog_bar=False)
                self.log(f'val_loss_{subj_id}', subj_loss.item(), prog_bar=False)

        # ── Cache for NIfTI diagnostics — best N + worst 1 ────────────────────────
        entry = {
            'pred_synth':    pred_synth.cpu(),
            'pred_labels':   pred_labels,
            'aug_image':     aug_image.cpu(),
            'real_image':    real_image.cpu(),
            'aug_mask':      aug_mask.cpu(),
            'target_labels': target_labels,
            'score':         -loss.item(),      # higher score = lower loss = better
            'batch_idx':     batch_idx,
            'subject_ids':   subject_ids,
        }

        # Best cache: keep top-n_best_batches by highest score (lowest loss)
        self._val_batch_cache.append(entry)
        self._val_batch_cache.sort(key=lambda x: x['score'], reverse=True)
        self._val_batch_cache = self._val_batch_cache[:self.n_best_batches]

        # Worst cache: keep 1 by lowest score (highest loss)
        self._val_worst_cache.append(entry)
        self._val_worst_cache.sort(key=lambda x: x['score'])   # ascending = worst first
        self._val_worst_cache = self._val_worst_cache[:1]

        return loss

    def _log_val_diagnostics(
            self,
            pred_synth,
            pred_labels,
            aug_image,
            real_image,
            aug_mask,
            real_labels,
            subject_id: str = 'unknown',
            rank: str = 'best',
    ):
        """
        Log class-count scalars and save NIfTI samples every 10 epochs.

        NIfTIs are written to:
            <log_dir>/images/epoch-<XXXX>/<rank>_<subject_id>/
                synth-pred.nii.gz
                synth-image.nii.gz
                synth-ref.nii.gz
                real-pred.nii.gz
                real-image.nii.gz
                real-ref.nii.gz
        """
        pred_synth_argmax = pred_synth[0].argmax(dim=0)
        pred_real_argmax  = pred_labels[0]
        self.log('pred_synth_num_classes',
                 float(len(torch.unique(pred_synth_argmax))), prog_bar=False)
        self.log('pred_real_num_classes',
                 float(len(torch.unique(pred_real_argmax))), prog_bar=False)

        if self.trainer.current_epoch % 10 != 0:
            return

        base_dir  = self.trainer.log_dir or self.trainer.default_root_dir
        epoch_dir = os.path.join(base_dir, 'images', f'epoch-{self.trainer.current_epoch:04d}')
        subj_dir  = os.path.join(epoch_dir, f'{rank}_{subject_id}')
        makedirs(subj_dir, exist_ok=True)

        print(f'\n[Saving] NIfTI diagnostics — Epoch {self.trainer.current_epoch}'
              f'  [{rank}]  {subject_id}  →  {subj_dir}')

        save(pred_synth_argmax,                       os.path.join(subj_dir, 'synth-pred.nii.gz'))
        save(pred_real_argmax,                        os.path.join(subj_dir, 'real-pred.nii.gz'))
        save(aug_image[0].squeeze(0),                 os.path.join(subj_dir, 'synth-image.nii.gz'))
        save(real_image[0].squeeze(0),                os.path.join(subj_dir, 'real-image.nii.gz'))
        save(aug_mask[0].squeeze(0).to(torch.uint8),  os.path.join(subj_dir, 'synth-ref.nii.gz'))
        save(real_labels[0].to(torch.uint8),          os.path.join(subj_dir, 'real-ref.nii.gz'))

    def on_validation_epoch_end(self):
        # ── Save NIfTI diagnostics for best N + worst 1 cached batches ───────────
        for bd in self._val_batch_cache:
            subject_id = bd['subject_ids'][0] if bd['subject_ids'] else 'unknown'
            self._log_val_diagnostics(
                bd['pred_synth'].to(self.device),
                bd['pred_labels'],
                bd['aug_image'].to(self.device),
                bd['real_image'].to(self.device),
                bd['aug_mask'].to(self.device),
                bd['target_labels'],
                subject_id = subject_id,
                rank       = 'best',
            )

        for bd in self._val_worst_cache:
            subject_id = bd['subject_ids'][0] if bd['subject_ids'] else 'unknown'
            self._log_val_diagnostics(
                bd['pred_synth'].to(self.device),
                bd['pred_labels'],
                bd['aug_image'].to(self.device),
                bd['real_image'].to(self.device),
                bd['aug_mask'].to(self.device),
                bd['target_labels'],
                subject_id = subject_id,
                rank       = 'worst',
            )

        self._val_batch_cache = []
        self._val_worst_cache = []

        # ── Epoch-level metrics ───────────────────────────────────────────────────
        dice_epoch   = self.val_dice.compute()
        dice_per_cls = self.val_dice_fcd.compute()
        dice_fcd     = dice_per_cls[5] if len(dice_per_cls) > 5 else torch.tensor(0.0)

        self.log('val_dice',     dice_epoch, prog_bar=True)
        self.log('val_dice_fcd', dice_fcd,   prog_bar=False)

        tl = self.trainer.callback_metrics.get('train_loss', -1)
        el = self.trainer.callback_metrics.get('eval_loss', -1)
        print(f"\n{'=' * 40}")
        print(f"EPOCH {self.trainer.current_epoch} SUMMARY:")
        print(f"  Train Loss    : {float(tl):.4f}")
        print(f"  Eval Loss     : {float(el):.4f}")
        print(f"  DICE SCORE    : {dice_epoch:.4f}")
        print(f"  DICE FCD (c6) : {dice_fcd:.4f}")
        print(f"{'=' * 40}\n")

        # Log current LR from the scheduler Lightning manages
        current_lr = self.optimizers().param_groups[0]['lr']
        print(f'  LR            : {current_lr:.2e}')

        self.val_dice.reset()
        self.val_dice_fcd.reset()

    # ══════════════════════════════════════════════════════════════════════════
    #  Optimiser / callbacks / inference
    # ══════════════════════════════════════════════════════════════════════════

    def configure_optimizers(self):
        opt_cls   = getattr(optim, self.optimizer_name)
        optimizer = opt_cls(self.network.segnet.parameters(),
                            **(self.optimizer_options or {}))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=10, factor=0.5, min_lr=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "eval_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def configure_callbacks(self):
        return [TimeLimitCallback(self.time_limit_minutes)] if self.time_limit_minutes else []

    def forward(self, x):
        return self.network.segnet(x)


# ── Helper Functions ──────────────────────────────────────────────────────────
def save(dat, fname):
    dat = dat.detach().cpu().numpy()
    h = nib.Nifti1Header()
    h.set_data_dtype(dat.dtype)
    nib.save(nib.Nifti1Image(dat, np.eye(4), h), fname)


# ══════════════════════════════════════════════════════════════════════════════
#  TimeLimitCallback
# ──────────────────────────────────────────────────────────────────────────────
#  Stops training cleanly after N minutes (epoch boundary, not mid-batch).
#  Reads L2S_TIME_LIMIT_MINUTES from the environment at on_train_start so
#  resuming with a new time budget always takes effect without touching the
#  checkpoint's baked-in hparams.
# ══════════════════════════════════════════════════════════════════════════════
class TimeLimitCallback(Callback):
    def __init__(self, limit_minutes):
        self.limit_minutes_default = limit_minutes
        self.limit_minutes = None
        self.start_time = None

    def on_train_start(self, trainer, pl_module):
        env_val = os.environ.get('L2S_TIME_LIMIT_MINUTES')
        if env_val is not None:
            self.limit_minutes = float(env_val)
            print(f"\n[TimeLimit] Limit set from environment: {self.limit_minutes} mins.")
        else:
            self.limit_minutes = self.limit_minutes_default
            print(f"\n[TimeLimit] Limit set from model hparams: {self.limit_minutes} mins.")
        self.start_time = datetime.datetime.now()
        print(f"[TimeLimit] Training started at {self.start_time}.")

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.limit_minutes and self.start_time:
            elapsed = (datetime.datetime.now() - self.start_time).total_seconds()
            if elapsed > self.limit_minutes * 60:
                print(f"\n[TimeLimit] Time limit reached "
                      f"({elapsed / 60:.1f} > {self.limit_minutes} mins). "
                      f"Stopping after this epoch.")
                trainer.should_stop = True


# ══════════════════════════════════════════════════════════════════════════════
#  LossGraphCallback
# ──────────────────────────────────────────────────────────────────────────────
#  Dual-axis plot: Loss (left) + Dice (right), saved as training_plot.png
#  after every validation epoch. Silent on errors — never blocks training.
# ══════════════════════════════════════════════════════════════════════════════
class LossGraphCallback(Callback):
    """
    Dual-axis loss + dice plot, saved after every validation epoch.

    General behavior:
      - Reads the current CSVLogger metrics.csv.
      - If metrics_history.csv exists in the same log directory, includes it.
      - Plots the combined metrics without owning backup/resume orchestration.
    """

    def __init__(
        self,
        history_filename: str = "metrics_history.csv",
        live_filename: str = "metrics.csv",
        plot_filename: str = "training_plot.png",
    ):
        self.history_filename = history_filename
        self.live_filename = live_filename
        self.plot_filename = plot_filename

    def _get_log_dir(self, trainer):
        """Prefer trainer.log_dir, with a CSVLogger fallback for multi-logger runs."""
        if trainer.log_dir:
            return trainer.log_dir

        loggers = trainer.loggers if isinstance(trainer.loggers, list) else [trainer.logger]
        for logger in loggers:
            if isinstance(logger, CSVLogger):
                return logger.log_dir

        return None

    def _read_csv_if_exists(self, file_path: str):
        if not os.path.exists(file_path):
            return None
        try:
            df = pd.read_csv(file_path)
            return df if not df.empty else None
        except Exception as e:
            print(f"[LossGraph] Failed to read {file_path}: {type(e).__name__}: {e}")
            return None

    @staticmethod
    def _combine_metric_rows(df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge duplicate Lightning metric rows safely.

        Lightning can log different metrics on separate rows for the same
        epoch/step. This keeps the first non-null value for each metric column.
        """
        if df is None or df.empty or "epoch" not in df.columns:
            return df

        df = df.copy()

        if "step" not in df.columns:
            df["step"] = np.nan

        df = df.sort_values(["epoch", "step"], kind="stable")

        def first_non_null(series):
            non_null = series.dropna()
            return non_null.iloc[0] if len(non_null) else np.nan

        grouped = (
            df.groupby(["epoch", "step"], dropna=False, as_index=False)
              .agg(first_non_null)
        )

        return grouped.sort_values(["epoch", "step"], kind="stable").reset_index(drop=True)

    def _load_plot_metrics(self, log_dir: str):
        live_path = os.path.join(log_dir, self.live_filename)
        history_path = os.path.join(log_dir, self.history_filename)

        frames = []

        history = self._read_csv_if_exists(history_path)
        if history is not None:
            frames.append(history)

        live = self._read_csv_if_exists(live_path)
        if live is not None:
            frames.append(live)

        if not frames:
            return None

        metrics = pd.concat(frames, ignore_index=True, sort=False)
        return self._combine_metric_rows(metrics)

    def _make_epoch_metrics(self, metrics: pd.DataFrame):
        if metrics is None or metrics.empty or "epoch" not in metrics.columns:
            return None

        numeric = metrics.copy()
        for col in numeric.columns:
            if col not in ("epoch", "step"):
                numeric[col] = pd.to_numeric(numeric[col], errors="coerce")

        return numeric.groupby("epoch").mean(numeric_only=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        try:
            log_dir = self._get_log_dir(trainer)
            if not log_dir:
                return

            plot_path = os.path.join(log_dir, self.plot_filename)
            metrics = self._load_plot_metrics(log_dir)
            epoch_metrics = self._make_epoch_metrics(metrics)

            if epoch_metrics is None or epoch_metrics.empty:
                return

            fig, ax1 = plt.subplots(figsize=(11, 6))

            if "train_loss" in epoch_metrics:
                ax1.plot(
                    epoch_metrics.index,
                    epoch_metrics["train_loss"],
                    label="Train Loss",
                    color="blue",
                    linestyle="-",
                    alpha=0.75,
                    linewidth=1.8,
                )

            if "eval_loss" in epoch_metrics:
                ax1.plot(
                    epoch_metrics.index,
                    epoch_metrics["eval_loss"],
                    label="Val Loss",
                    color="red",
                    linestyle="--",
                    linewidth=1.8,
                )

            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.set_ylim(bottom=0)
            ax1.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

            ax2 = ax1.twinx()

            if "val_dice" in epoch_metrics:
                ax2.plot(
                    epoch_metrics.index,
                    epoch_metrics["val_dice"],
                    label="Val Dice",
                    color="green",
                    linewidth=2.2,
                )

            if "val_dice_fcd" in epoch_metrics:
                ax2.plot(
                    epoch_metrics.index,
                    epoch_metrics["val_dice_fcd"],
                    label="Val Dice FCD (c6)",
                    color="orange",
                    linewidth=2.2,
                    linestyle="--",
                )

            ax2.set_ylabel("Dice")
            ax2.set_ylim(0, 1)

            if trainer.current_epoch in epoch_metrics.index:
                ax1.axvline(
                    trainer.current_epoch,
                    color="gray",
                    linestyle=":",
                    linewidth=1,
                    alpha=0.7,
                )

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(
                lines1 + lines2,
                labels1 + labels2,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.13),
                ncol=4,
                frameon=True,
            )

            first_epoch = int(epoch_metrics.index.min())
            last_epoch = int(epoch_metrics.index.max())
            fig.suptitle(
                f"Training Metrics | Epochs {first_epoch}–{last_epoch} "
                f"(current: {trainer.current_epoch})"
            )
            fig.tight_layout(rect=[0, 0.08, 1, 0.95])

            try:
                plt.savefig(plot_path, dpi=160, bbox_inches="tight")
            finally:
                plt.close()

            print(f"[LossGraph] Updated plot → {plot_path}")

        except Exception as e:
            print(f"[LossGraph] ❌ Error at epoch {trainer.current_epoch}: "
                  f"{type(e).__name__}: {e}")
            traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
#  EveryEpochCheckpointCallback
# ──────────────────────────────────────────────────────────────────────────────
#  Writes checkpoints/<filename> unconditionally after every training epoch.
#  Does not depend on ModelCheckpoint, metric improvements, or save_last logic.
#  This is the canonical file used for resuming.
#
#  Uses trainer.save_checkpoint() directly — bypasses all of ModelCheckpoint's
#  link/top-k/metric-gating logic.
# ══════════════════════════════════════════════════════════════════════════════
class EveryEpochCheckpointCallback(Callback):
    def __init__(self, filename="resume.ckpt"):
        self.filename = filename

    def on_train_epoch_end(self, trainer, pl_module):
        ckpt_dir = os.path.join(trainer.default_root_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, self.filename)
        try:
            trainer.save_checkpoint(ckpt_path)
            size_mb = os.path.getsize(ckpt_path) / 1e6
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(ckpt_path))
            print(f"[EveryEpoch] ✅ epoch={trainer.current_epoch} → "
                  f"{ckpt_path} ({size_mb:.1f} MB, mtime {mtime})")
        except Exception as e:
            print(f"[EveryEpoch] ❌ Save FAILED at epoch "
                  f"{trainer.current_epoch}: {type(e).__name__}: {e}")
            traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
#  CheckpointTraceCallback
# ──────────────────────────────────────────────────────────────────────────────
#  Diagnostic callback — verifies resume.ckpt and last.ckpt behave as expected
#  each epoch. Fires after EveryEpochCheckpointCallback so resume.ckpt is
#  already written when we inspect it.
# ══════════════════════════════════════════════════════════════════════════════
class CheckpointTraceCallback(Callback):

    def on_validation_epoch_start(self, trainer, pl_module):
        print(f"\n[CKPT TRACE] === Epoch {trainer.current_epoch}: validation starting ===")

    def on_validation_epoch_end(self, trainer, pl_module):
        _, _, free = shutil.disk_usage(OUTPUT_FOLDER)
        print(f"[CKPT TRACE] Epoch {trainer.current_epoch}: validation hooks running. "
              f"Disk free={free / 1e9:.2f}GB, should_stop={trainer.should_stop}")

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        mc = next((cb for cb in trainer.callbacks
                   if type(cb).__name__ == "ModelCheckpoint"), None)
        if mc is None:
            print(f"[CKPT TRACE] Epoch {epoch}: no ModelCheckpoint found!")
            return

        ckpt_dir = mc.dirpath or os.path.join(trainer.log_dir or '.', 'checkpoints')
        print(f"[CKPT TRACE] Epoch {epoch}: post-epoch checkpoint state:")
        print(f"  ModelCheckpoint.last_model_path  = {mc.last_model_path}")
        print(f"  ModelCheckpoint.best_model_path  = {mc.best_model_path}")
        print(f"  ModelCheckpoint.best_model_score = {mc.best_model_score}")

        if os.path.isdir(ckpt_dir):
            for fname, label in [('resume.ckpt', 'resume.ckpt'), ('last.ckpt', 'last.ckpt  ')]:
                fpath = os.path.join(ckpt_dir, fname)
                if os.path.exists(fpath):
                    mb = os.path.getsize(fpath) / 1e6
                    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(fpath))
                    print(f"  {label} on disk: {mb:.1f} MB, mtime {mtime}")
                else:
                    print(f"  {label} DOES NOT EXIST on disk")

            n_ckpts = len([f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')])
            print(f"  total ckpt files   : {n_ckpts}")

        _, _, free = shutil.disk_usage(OUTPUT_FOLDER)
        print(f"  disk free          = {free / 1e9:.2f}GB")

    def on_exception(self, trainer, pl_module, exception):
        print(f"\n[CKPT TRACE] ❌ EXCEPTION: {type(exception).__name__}: {exception}")
        traceback.print_exc()


# ── CLI & Main ────────────────────────────────────────────────────────────────
class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(ModelCheckpoint, "checkpoint")
        parser.set_defaults({
            "checkpoint.monitor": "eval_loss",
            "checkpoint.save_last": True,
            "checkpoint.save_top_k": 1,
            "checkpoint.filename": "checkpoint-{epoch:02d}-{eval_loss:.2f}-{val_dice:.2f}",
            "checkpoint.every_n_epochs": 1,
        })
        parser.link_arguments("model.native_synthesis", "data.native_synthesis")  # ← add


    def instantiate_trainer(self, **kwargs):
        run_name = os.environ.get(
            "L2S_RUN_NAME",
            f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        default_root = kwargs.get("default_root_dir", "experiments")
        save_dir = os.path.join(default_root, run_name)
        makedirs(save_dir, exist_ok=True)

        print(f"[System] Initializing Run: {run_name}")
        print(f"[System] All artifacts will be stored in: {save_dir}")

        logger = [
            TensorBoardLogger(save_dir=default_root, name=run_name, version=''),
            CSVLogger(save_dir=default_root, name=run_name, version=''),
        ]

        cbs = kwargs.get("callbacks", []) or []

        # Registration order matters: EveryEpoch writes resume.ckpt first,
        # then CheckpointTrace inspects it, then LossGraph plots.
        cbs.append(EveryEpochCheckpointCallback(filename="resume.ckpt"))
        cbs.append(LossGraphCallback())
        cbs.append(CheckpointTraceCallback())

        print("\n[CLI] Registered callbacks:")
        for i, cb in enumerate(cbs):
            print(f"  [{i}] {type(cb).__name__}")
        print()

        kwargs["default_root_dir"] = save_dir
        kwargs["enable_progress_bar"] = False
        kwargs["logger"] = logger
        kwargs["callbacks"] = cbs
        return super().instantiate_trainer(**kwargs)


if __name__ == '__main__':
    cli = CLI(Model, FCDDataModule, save_config_kwargs={"overwrite": True})
