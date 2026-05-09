import torch
import torch.nn.functional as F
import math
import random


class FCDAugmentations:
    """
    FCD appearance augmentations — all GPU-native.

    Design rules for guaranteed per-subject diversity
    --------------------------------------------------
    1. No `seed` / `torch.manual_seed` inside any method.
       Seeding inside augmentation methods resets the global RNG and causes
       every subsequent call in the same process to produce the same sequence.

    2. Every random value is **pre-sampled by the caller** and passed in as an
       explicit argument.  The augmentation methods are therefore pure functions
       of their inputs — no hidden RNG state.

    3. The caller (`_apply_fcd_augmentations`) draws all values from
       `random.uniform` / `random.randint` in a single location, making
       diversity easy to audit and test.

    Migration from the old API
    --------------------------
    Old: aug.apply_roi_augmentations_blured(img, roi, seed=42, sigma_range=(0.3, 1.0))
    New: aug.apply_roi_augmentations_blured(img, roi, sigma=random.uniform(0.3, 1.0))
    """

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers  (unchanged from original)
    # ──────────────────────────────────────────────────────────────────────────

    def gaussian_blur_3d_torch(self, tensor_3d: torch.Tensor, sigma: float,
                               device: torch.device) -> torch.Tensor:
        """
        Separable 3-D Gaussian blur implemented entirely in PyTorch.
        Runs on *device* (GPU or CPU). Returns a (Z,Y,X) float tensor.

        Parameters
        ----------
        tensor_3d : torch.Tensor  shape (Z, Y, X)
        sigma     : float – Gaussian standard deviation in voxels
        device    : torch.device
        """
        if sigma < 1e-4:
            return tensor_3d.clone()

        radius = int(math.ceil(3 * sigma))
        ks = 2 * radius + 1
        coords = torch.arange(ks, dtype=torch.float32, device=device) - radius
        kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()

        t = tensor_3d.float().unsqueeze(0).unsqueeze(0)  # (1,1,Z,Y,X)

        kz = kernel_1d.view(1, 1, ks, 1, 1)
        t = F.conv3d(t, kz, padding=(radius, 0, 0))
        ky = kernel_1d.view(1, 1, 1, ks, 1)
        t = F.conv3d(t, ky, padding=(0, radius, 0))
        kx = kernel_1d.view(1, 1, 1, 1, ks)
        t = F.conv3d(t, kx, padding=(0, 0, radius))

        return t.squeeze(0).squeeze(0)

    def binary_dilation_torch(self, mask_3d: torch.Tensor,
                              iterations: int = 3,
                              device: torch.device = None) -> torch.Tensor:
        """
        Morphological dilation of a binary (Z,Y,X) mask using F.max_pool3d.
        Equivalent to scipy.ndimage.binary_dilation(mask, iterations=iterations).
        """
        t = mask_3d.float().unsqueeze(0).unsqueeze(0)  # (1,1,Z,Y,X)
        for _ in range(iterations):
            t = F.max_pool3d(t, kernel_size=3, stride=1, padding=1)
        return t.squeeze(0).squeeze(0)

    # ──────────────────────────────────────────────────────────────────────────
    # Public augmentation methods  — accept pre-sampled scalars, not ranges
    # ──────────────────────────────────────────────────────────────────────────

    def apply_roi_augmentations_blured(
        self,
        synthetic: torch.Tensor,
        roi: torch.Tensor,
        sigma: float,                   # ← pre-sampled by caller
    ) -> torch.Tensor:
        """
        Gaussian blur inside ROI mask – fully GPU-native.

        Parameters
        ----------
        synthetic : torch.Tensor  (Z, Y, X)
        roi       : torch.Tensor  (Z, Y, X)  binary / integer mask
        sigma     : float  – already sampled from sigma_range by the caller

        Returns
        -------
        augmented : torch.Tensor  (Z, Y, X)
        """
        device = synthetic.device
        augmented = synthetic.clone()
        roi_mask = (roi > 0).float().to(device)

        blurred = self.gaussian_blur_3d_torch(augmented, sigma, device)
        augmented = augmented * (1.0 - roi_mask) + blurred * roi_mask
        return augmented

    def apply_roi_thickening(
        self,
        synthetic: torch.Tensor,
        roi: torch.Tensor,
        zoom_factor: float,             # ← pre-sampled by caller
        bound: str = 'border',
    ):
        """
        Simulate cortical thickening by zooming only the ROI region.

        Parameters
        ----------
        synthetic   : torch.Tensor (Z,Y,X)
        roi         : torch.Tensor (Z,Y,X)
        zoom_factor : float  – already sampled from zoom_range by the caller
                       >1 expands ROI (thickening), <1 shrinks (thinning)
        bound       : str  padding mode for grid_sample

        Returns
        -------
        out          : torch.Tensor (Z,Y,X)
        warped_mask  : torch.Tensor (Z,Y,X)
        """
        device = synthetic.device
        dtype = torch.float32

        img  = synthetic.to(dtype).unsqueeze(0).unsqueeze(0)  # (1,1,Z,Y,X)
        mask = (roi > 0).to(dtype).unsqueeze(0).unsqueeze(0)

        Z, Y, X = synthetic.shape
        
        # Guard: degenerate volume cannot be warped
        if Z < 2 or Y < 2 or X < 2:
            print(f"[WARN] apply_roi_thickening: degenerate shape {synthetic.shape} — returning as-is")
            return synthetic, (roi > 0).float()

        nz = torch.nonzero(mask.squeeze())
        if nz.numel() == 0:
            return synthetic, (roi > 0).float()
        cz, cy, cx = nz.float().mean(dim=0).tolist()

        E = torch.eye(4, device=device, dtype=dtype)
        T1 = E.clone(); T1[:3, 3] = torch.tensor([-cz, -cy, -cx], device=device)
        S  = E.clone(); S[0, 0] = zoom_factor; S[1, 1] = zoom_factor; S[2, 2] = zoom_factor
        T2 = E.clone(); T2[:3, 3] = torch.tensor([cz,  cy,  cx],  device=device)
        A  = T2 @ S @ T1

        z = torch.linspace(0, Z - 1, Z, device=device, dtype=dtype)
        y = torch.linspace(0, Y - 1, Y, device=device, dtype=dtype)
        x = torch.linspace(0, X - 1, X, device=device, dtype=dtype)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        ones   = torch.ones_like(xx)
        coords = torch.stack([zz, yy, xx, ones], dim=-1)
        mapped = coords @ A.T
        mz, my, mx = mapped[..., 0], mapped[..., 1], mapped[..., 2]
        gx = 2 * (mx / (X - 1)) - 1
        gy = 2 * (my / (Y - 1)) - 1
        gz = 2 * (mz / (Z - 1)) - 1
        grid = torch.stack([gx, gy, gz], dim=-1)[None, ...]

        warped_img  = F.grid_sample(img,  grid, mode='bilinear',  padding_mode=bound,   align_corners=True)
        warped_mask = F.grid_sample(mask, grid, mode='nearest',   padding_mode='zeros', align_corners=True)

        out = img * (1 - warped_mask) + warped_img * warped_mask
        return out.squeeze(0).squeeze(0), warped_mask.squeeze(0).squeeze(0)

    def apply_roi_augmentations_hyperintensity(
        self,
        synthetic: torch.Tensor,
        roi: torch.Tensor,
        intensity_factor: float,        # ← pre-sampled by caller
        sigma: float,                   # ← pre-sampled by caller
    ) -> torch.Tensor:
        """
        Hyperintensity augmentation inside ROI – fully GPU-native.

        Parameters
        ----------
        synthetic        : torch.Tensor  (Z, Y, X)
        roi              : torch.Tensor  (Z, Y, X)
        intensity_factor : float  – already sampled from intensity_range by caller
        sigma            : float  – already sampled from sigma_range by caller

        Returns
        -------
        augmented : torch.Tensor  (Z, Y, X)
        """
        device    = synthetic.device
        
        # Guard: if input is already corrupted, bail out cleanly
        if not torch.isfinite(synthetic).all():
            bad = (~torch.isfinite(synthetic)).sum().item()
            print(f"[WARN] hyperintensity received {bad} non-finite voxels in synthetic — returning as-is")
            return synthetic  # don't make it worse; let caller decide
        
        augmented = synthetic.clone()
        roi_mask_t = (roi > 0).float().to(device)
        hyper_map  = self.gaussian_blur_3d_torch(roi_mask_t, sigma, device)
        augmented  = augmented + hyper_map.to(augmented.dtype) * intensity_factor
        return augmented

    def apply_roi_augmentations_transmantle(
        self,
        synthetic: torch.Tensor,
        roi: torch.Tensor,
        labeled_image: torch.Tensor,
        tail_length: int,               # ← pre-sampled by caller
        intensity_factor: float,        # ← pre-sampled by caller
        sigma: float,                   # ← pre-sampled by caller
        tail_dilation_iterations: int = 1,
    ):
        """
        Simulate transmantle sign – fully GPU-native.
        Hyperintense tail from cortex ROI toward nearest lateral ventricle (labels 4 / 43).

        Parameters
        ----------
        synthetic               : torch.Tensor  (Z, Y, X)
        roi                     : torch.Tensor  (Z, Y, X)
        labeled_image           : torch.Tensor  (Z, Y, X)
        tail_length             : int    – already sampled from tail_length_range by caller
        intensity_factor        : float  – already sampled from intensity_range by caller
        sigma                   : float  – already sampled from sigma_range by caller
        tail_dilation_iterations: int    – controls tail thickness

        Returns
        -------
        augmented          : torch.Tensor  (Z, Y, X)
        roi_mask_with_tail : torch.Tensor  (Z, Y, X, float32)
        """
        device    = synthetic.device
        augmented = synthetic.clone()

        roi_mask_t = (roi > 0).float().to(device)
        roi_coords = torch.nonzero(roi_mask_t, as_tuple=False).float()
        if roi_coords.numel() == 0:
            return augmented, roi_mask_t
        roi_centroid = roi_coords.mean(dim=0)

        lab = labeled_image.to(device)
        vent_left_coords  = torch.nonzero(lab == 4,  as_tuple=False).float()
        vent_right_coords = torch.nonzero(lab == 43, as_tuple=False).float()

        c_left  = vent_left_coords.mean(dim=0)  if vent_left_coords.numel()  > 0 else None
        c_right = vent_right_coords.mean(dim=0) if vent_right_coords.numel() > 0 else None

        chosen_centroid = None
        if c_left is not None and c_right is not None:
            d_left  = torch.linalg.norm(roi_centroid - c_left)
            d_right = torch.linalg.norm(roi_centroid - c_right)
            chosen_centroid = c_left if d_left < d_right else c_right
        elif c_left is not None:
            chosen_centroid = c_left
        elif c_right is not None:
            chosen_centroid = c_right

        Z, Y, X = synthetic.shape
        tail_mask = torch.zeros(Z, Y, X, dtype=torch.float32, device=device)

        if chosen_centroid is not None:
            direction = chosen_centroid - roi_centroid
            # Small random perturbation — uses torch.randn (not seeded here)
            noise     = torch.randn(3, device=device) * 0.2
            direction = direction + noise
            direction = direction / (torch.linalg.norm(direction) + 1e-6)

            step_range = torch.arange(1, tail_length, dtype=torch.float32, device=device)
            offsets_f  = roi_centroid.unsqueeze(0) + direction.unsqueeze(0) * step_range.unsqueeze(1)
            offsets    = offsets_f.long()

            valid = (
                (offsets[:, 0] >= 0) & (offsets[:, 0] < Z) &
                (offsets[:, 1] >= 0) & (offsets[:, 1] < Y) &
                (offsets[:, 2] >= 0) & (offsets[:, 2] < X)
            )
            offsets = offsets[valid]
            if offsets.numel() > 0:
                tail_mask[offsets[:, 0], offsets[:, 1], offsets[:, 2]] = 1.0

            tail_mask = self.binary_dilation_torch(
                tail_mask, iterations=tail_dilation_iterations, device=device
            )

        combined_mask = torch.clamp(roi_mask_t + tail_mask, 0.0, 1.0)

        hyper_map = self.gaussian_blur_3d_torch(combined_mask, sigma, device)
        augmented = augmented + hyper_map.to(augmented.dtype) * intensity_factor

        return augmented, combined_mask
