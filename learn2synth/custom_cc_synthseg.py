"""
custom_cc_synthseg.py
=====================
FLAIR-specific GMM synthesis components for the SynthFCD pipeline.

Public API
----------
FLAIR_CLASS_PARAMS                  — per-class (μ, σ) ranges for labels 0–18
load_class_params_from_csv()        — load class params from a CSV file
RandomGaussianMixtureTransform      — legacy single-range GMM (all classes share one range)
PerClassGaussianMixtureTransform    — per-class GMM (each class has its own μ/σ range)
SynthFromLabelTransform             — full synthesis: deform + GMM + intensity
"""

import torch
import cornucopia as cc
import pandas as pd


def do_nothing(x):
    return x


# ─────────────────────────────────────────────────────────────────────────────
# CSV loader
# ─────────────────────────────────────────────────────────────────────────────

def load_class_params_from_csv(
        csv_path: str,
        class_id_col: str = "class_id",
        mu_lo_col: str = "mu_lo",
        mu_hi_col: str = "mu_hi",
        sigma_lo_col: str = "sigma_lo",
        sigma_hi_col: str = "sigma_hi",
        fallback: dict | None = None,
) -> dict:
    """Load per-class GMM intensity parameters from a CSV file.

    Missing classes (not present in the CSV) are filled from ``fallback``,
    which defaults to ``FLAIR_CLASS_PARAMS``. This guarantees all 19 classes
    (0–18) are always present in the returned dict.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    class_id_col, mu_lo_col, mu_hi_col, sigma_lo_col, sigma_hi_col : str
        Column names for class ID and parameter bounds.
    fallback : dict, optional
        Per-class params used for any class absent in the CSV.
        Defaults to ``FLAIR_CLASS_PARAMS``.

    Returns
    -------
    dict[int, dict]
        ``{class_id: {"mu": (lo, hi), "sigma": (lo, hi)}, ...}``
    """
    if fallback is None:
        fallback = {}

    df = pd.read_csv(csv_path)

    required = {class_id_col, mu_lo_col, mu_hi_col, sigma_lo_col, sigma_hi_col}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {missing_cols}")

    params = {
        int(row[class_id_col]): {
            "mu":    (float(row[mu_lo_col]),    float(row[mu_hi_col])),
            "sigma": (float(row[sigma_lo_col]), float(row[sigma_hi_col])),
        }
        for _, row in df.iterrows()
    }

    # Fill any classes the CSV didn't cover
    all_classes = set(range(19)) | set(fallback.keys())
    for cls in all_classes:
        params.setdefault(cls, fallback.get(cls, {"mu": (0, 255), "sigma": (0, 16)}))

    return params


# ─────────────────────────────────────────────────────────────────────────────
# GMM transforms
# ─────────────────────────────────────────────────────────────────────────────

class RandomGaussianMixtureTransform(torch.nn.Module):
    """Legacy GMM — samples one shared (μ, σ) range for ALL tissue classes.

    Use ``PerClassGaussianMixtureTransform`` for FLAIR synthesis where each
    tissue class should have its own intensity distribution.
    """

    def __init__(self, mu=255, sigma=16, fwhm=2, background=None, dtype=None):
        super().__init__()
        self.dtype = dtype
        self.background = background
        self.sample = dict(
            mu=cc.random.Uniform.make(cc.random.make_range(0, mu)),
            sigma=cc.random.Uniform.make(cc.random.make_range(0, sigma)),
            fwhm=cc.random.Uniform.make(cc.random.make_range(0, fwhm)),
        )

    def get_parameters(self, x):
        backend = (
            dict(dtype=x.dtype, device=x.device)
            if x.dtype.is_floating_point
            else dict(dtype=self.dtype or torch.get_default_dtype(), device=x.device)
        )
        mu    = torch.as_tensor(self.sample["mu"](len(x))).to(**backend)
        sigma = torch.as_tensor(self.sample["sigma"](len(x))).to(**backend)
        fwhm  = int(self.sample["fwhm"]())
        return mu, sigma, fwhm

    def apply_transform(self, x, parameters):
        mu, sigma, fwhm = parameters
        backend = dict(dtype=mu.dtype, device=x.device)
        if self.background is not None:
            x[self.background] = 0
        y1 = torch.randn(*x.shape, **backend)
        y1 = cc.utils.conv.smoothnd(y1, fwhm=fwhm)
        y1 = y1.mul_(sigma[..., None, None, None]).add_(mu[..., None, None, None])
        return torch.sum(x.to(**backend) * y1, dim=0)[None]

    def forward(self, x):
        return self.apply_transform(x, self.get_parameters(x))


class PerClassGaussianMixtureTransform(torch.nn.Module):
    """Per-class GMM — each tissue class has its own (μ_lo, μ_hi) / (σ_lo, σ_hi).

    At each forward call, one scalar (μ, σ) is independently sampled per class
    from its configured range. The one-hot input mask ensures each voxel picks
    up the intensity noise of exactly its own tissue class.

    Parameters
    ----------
    class_params : dict[int, dict]
        ``{channel_index: {"mu": (lo, hi), "sigma": (lo, hi)}}``.
        Typically ``FLAIR_CLASS_PARAMS`` or a per-subject variant loaded from CSV.
    fwhm : float
        Upper bound for within-class spatial smoothing (makes intensities
        spatially correlated, as in real MRI).
    background : int, optional
        Channel index to zero out before synthesis (suppresses background noise).
    default_mu, default_sigma : tuple
        Fallback ranges for channels absent in ``class_params``.
    dtype : torch.dtype, optional
        Output dtype when input is integer.
    """

    def __init__(
            self,
            class_params: dict,
            fwhm: float = 2,
            background: int | None = None,
            default_mu: tuple = (0, 255),
            default_sigma: tuple = (0, 16),
            dtype=None,
    ):
        super().__init__()
        self.class_params  = class_params
        self.default_mu    = default_mu
        self.default_sigma = default_sigma
        self.background    = background
        self.dtype         = dtype
        self.fwhm_sampler  = cc.random.Uniform.make(cc.random.make_range(0, fwhm))

    @staticmethod
    def _sample_scalar(lo: float, hi: float) -> float:
        if lo >= hi:
            return float(lo)
        return lo + (hi - lo) * torch.rand(1).item()

    def get_parameters(self, x):
        """Sample one (μ, σ) scalar per class channel."""
        mu_vals, sigma_vals = [], []
        for i in range(len(x)):
            p = self.class_params.get(i)
            mu_lo,    mu_hi    = p["mu"]    if p else self.default_mu
            sigma_lo, sigma_hi = p["sigma"] if p else self.default_sigma
            mu_vals.append(self._sample_scalar(mu_lo, mu_hi))
            sigma_vals.append(self._sample_scalar(sigma_lo, sigma_hi))

        backend = (
            dict(dtype=x.dtype, device=x.device)
            if x.dtype.is_floating_point
            else dict(dtype=self.dtype or torch.get_default_dtype(), device=x.device)
        )
        return (
            torch.tensor(mu_vals,    **backend),
            torch.tensor(sigma_vals, **backend),
            int(self.fwhm_sampler()),
        )

    def apply_transform(self, x, parameters):
        """Apply sampled parameters to the one-hot tensor → synthetic image."""
        mu, sigma, fwhm = parameters
        backend = dict(dtype=mu.dtype, device=x.device)

        if self.background is not None:
            x = x.clone()
            x[self.background] = 0

        y1 = torch.randn(*x.shape, **backend)
        if fwhm > 0:
            y1 = cc.utils.conv.smoothnd(y1, fwhm=fwhm)
        y1 = y1.mul_(sigma[..., None, None, None]).add_(mu[..., None, None, None])
        return torch.sum(x.to(**backend) * y1, dim=0)[None]

    def forward(self, x):
        return self.apply_transform(x, self.get_parameters(x))


# ─────────────────────────────────────────────────────────────────────────────
# Full synthesis transform
# ─────────────────────────────────────────────────────────────────────────────

class SynthFromLabelTransform(torch.nn.Module):
    """Synthesize a synthetic MRI from a one-hot label map.

    Pipeline (all three stages are independently configurable):

        one-hot label map
            │
            ▼  [1] Geometric deformation
            │      RandomAffineElasticTransform — same field applied to x and coreg
            │      rlab and rroi rounded to nearest integer after deformation
            │
            ▼  [2] GMM synthesis
            │      PerClassGaussianMixtureTransform  (default, uses FLAIR_CLASS_PARAMS)
            │      RandomGaussianMixtureTransform    (fallback when class_params=None)
            │
            ▼  [3] Intensity augmentation  (disabled when no_augs=True)
            │      cc.IntensityTransform — bias field, gamma, noise, resolution
            │      ends with QuantileTransform → output always [0, 1]
            │
            ▼  [3b] GMM normalization  (only when no_augs=True)
                   QuantileTransform(clip=True) — maps 1st–99th percentile to [0, 1]
                   Replicates what cc.IntensityTransform would have done, so that
                   simg always exits this transform in [0, 1] regardless of path.

    In the FCD training pipeline, ``no_augs=True`` is used so that:
    - Intensity augmentation is applied downstream after FCD augmentations.
    - GMM output is still normalized here to [0, 1] via QuantileTransform(clip=True),
      matching the output range of the no_augs=False path.

    Parameters
    ----------
    num_ch : int
        Number of output image channels (default 1 — single FLAIR channel).
    no_augs : bool
        Disable intensity augmentation only — deformation always runs.
        GMM output is still normalized to [0, 1] via QuantileTransform.
    class_params : dict, optional
        Per-class GMM params. Falls back to the legacy single-range GMM when None.
    use_per_class_gmm : bool
        When True and class_params is provided, use PerClassGaussianMixtureTransform.
    skip_gmm : bool
        Skip GMM entirely — output is the raw intensity-transformed label map.
    """

    def __init__(
            self,
            num_ch: int = 1,
            patch=None,
            rotation: float = 15,
            shears: float = 0.012,
            zooms: float = 0.15,
            elastic: float = 0.05,
            elastic_nodes: int = 10,
            gmm_fwhm: float = 10,
            bias: float = 7,
            gamma: float = 0.6,
            motion_fwhm: float = 3,
            resolution: float = 8,
            snr: float = 10,
            gfactor: float = 5,
            order: int = 3,
            skip_gmm: bool = False,
            no_augs: bool = False,
            class_params: dict | None = None,
            use_per_class_gmm: bool = True,
    ):
        super().__init__()
        self.no_augs = no_augs
        self.num_ch  = num_ch

        # ── Geometric deformation ─────────────────────────────────────────────
        self.deform = cc.RandomAffineElasticTransform(
            elastic, elastic_nodes,
            order=order, bound="zeros",
            rotations=rotation, shears=shears, zooms=zooms, patch=patch,
        )

        # ── GMM ───────────────────────────────────────────────────────────────
        if skip_gmm:
            self.gmm = None
        elif use_per_class_gmm and class_params is not None:
            self.gmm = PerClassGaussianMixtureTransform(
                class_params=class_params, fwhm=gmm_fwhm, background=0,
            )
        else:
            self.gmm = RandomGaussianMixtureTransform(fwhm=gmm_fwhm, background=0)

        # ── Post-GMM intensity augmentation ───────────────────────────────────
        # no_augs disables intensity only — deformation is always active.
        # When no_augs=True, self.normalizer is applied instead to bring the
        # raw GMM output (~0-255) into [0, 1], matching the output range of
        # cc.IntensityTransform (which always ends with QuantileTransform).
        self.intensity  = do_nothing if no_augs else cc.IntensityTransform(
            bias, gamma, motion_fwhm, resolution, snr, gfactor, order,
        )
        self.normalizer = cc.QuantileTransform(clip=True)

    def forward(self, x: torch.Tensor, coreg=None):
        """
        Parameters
        ----------
        x : Tensor (n_classes, D, H, W)
            One-hot label map. n_classes = N_CLASSES + 1 = 19 for labels 0–18.
        coreg : Tensor or list[Tensor], optional
            Extra volumes co-deformed with x (e.g. real FLAIR, label map, ROI).
            coreg order: [rimg, rlab, rroi]
            - rimg : float image — cubic interpolation
            - rlab : integer label map — nearest-neighbour interpolation
            - rroi : binary mask — nearest-neighbour interpolation

        Returns
        -------
        img   : Tensor (1, D, H, W)          — synthetic FLAIR image, always [0, 1]
        x     : Tensor (n_classes, D, H, W)  — deformed one-hot label map
        coreg : Tensor or list[Tensor]        — only when coreg was provided
        """
        # ── Stage 1: Geometric deformation ───────────────────────────────────
        # Freeze the random field relative to x's spatial shape.
        # rimg is deformed with cubic interpolation (order=3).
        # rlab and rroi are deformed with nearest-neighbour (nearest_if_label=True)
        # using the same frozen field — no interpolation artifacts on label maps.
        frozen_deform = self.deform.make_final(x)
        if coreg is not None:
            coreg_list = coreg if isinstance(coreg, (list, tuple)) else [coreg]
            n_lab = x.shape[0]

            # Deform everything together with cubic — preserves original shape behavior
            stacked = torch.cat([x] + coreg_list, dim=0)
            stacked = frozen_deform(stacked)
            x = stacked[:n_lab]
            coreg_out = [stacked[n_lab + i] for i in range(len(coreg_list))]

            # Re-apply same frozen field to rlab (index 1) and rroi (index 2)
            # with nearest-neighbour to fix cubic interpolation artifacts
            frozen_deform.nearest_if_label = True
            if len(coreg_out) > 1:
                coreg_out[1] = frozen_deform(coreg_list[1].round().long())
            if len(coreg_out) > 2:
                coreg_out[2] = frozen_deform(coreg_list[2].round().long())
            frozen_deform.nearest_if_label = False

            coreg = coreg_out if isinstance(coreg, (list, tuple)) else coreg_out[0]
        else:
            x = frozen_deform(x)

        # ── Stage 2: GMM synthesis ────────────────────────────────────────────
        if self.gmm is not None:
            gmm_params = [self.gmm.get_parameters(x) for _ in range(self.num_ch)]
            img = torch.cat(
                [self.intensity(self.gmm.apply_transform(x, gmm_params[i]))
                 for i in range(self.num_ch)],
                dim=0,
            )
        else:
            img = self.intensity(x)

        # ── Stage 3: GMM normalization (no_augs=True path only) ──────────────
        # When no_augs=False, cc.IntensityTransform already ends with
        # QuantileTransform — img is already [0, 1].
        # When no_augs=True, self.intensity is do_nothing — GMM output is raw
        # FLAIR scale (~0-255). Normalize here so simg always exits [0, 1].
        if self.no_augs:
            img = self.normalizer.make_final(img)(img)

        # ── Return ────────────────────────────────────────────────────────────
        if coreg is not None:
            return img, x, coreg
        return img, x