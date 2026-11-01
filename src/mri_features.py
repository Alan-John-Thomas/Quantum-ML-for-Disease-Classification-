"""
Single-scan CPU MRI ROI feature extraction for real-time inference.

Faithful CPU port of src/extract_roi_features.py (which used torch/nilearn):
- nibabel for NIfTI I/O
- scipy.ndimage.map_coordinates for trilinear atlas resampling
  (equivalent to torch grid_sample mode="bilinear", padding_mode="border")
- numpy for per-region mean reduction

The Harvard-Oxford atlas is loaded from the local nilearn cache
(%USERPROFILE%\\nilearn_data\\fsl\\data\\atlases\\HarvardOxford) or from the
directory pointed to by the ATLAS_DIR environment variable. Region names are
parsed from the FSL XML files shipped next to the atlas volumes.

Usage:
    from src.mri_features import extract_roi_vector
    vec = extract_roi_vector("scan_skull_stripped.nii.gz")   # (69,) float32
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates

# --------------------------------------------------------------------------
# Atlas location / loading
# --------------------------------------------------------------------------
def _default_atlas_dir() -> Path:
    env = os.environ.get("ATLAS_DIR")
    if env:
        return Path(env)
    return Path.home() / "nilearn_data" / "fsl" / "data" / "atlases" / "HarvardOxford"


def _xml_names(xml_path: Path) -> list[str]:
    """Parse <label index="i">Name</label> entries -> names ordered by index."""
    tree = ET.parse(xml_path)
    labels = tree.getroot().findall(".//data/label")
    labels.sort(key=lambda el: int(el.attrib["index"]))
    return [el.text.strip() for el in labels]


def load_atlas_volumes(atlas_dir: Path | None = None):
    """Harvard-Oxford cortical (48) + subcortical (21) labels on MNI 2mm grid.

    Mirrors extract_roi_features.load_atlas(): subcortical labels are offset
    and painted into cortical background voxels.

    Returns (label_volume int32 [91,109,91], affine 4x4, names list[str]).
    """
    d = atlas_dir or _default_atlas_dir()
    cort_path = d / "HarvardOxford-cort-maxprob-thr25-2mm.nii.gz"
    sub_path = d / "HarvardOxford-sub-maxprob-thr25-2mm.nii.gz"
    cort_xml = d.parent / "HarvardOxford-Cortical.xml"
    sub_xml = d.parent / "HarvardOxford-Subcortical.xml"
    if not cort_path.exists():
        raise FileNotFoundError(
            f"Harvard-Oxford atlas not found at {cort_path}. "
            "Set ATLAS_DIR to the folder containing the HarvardOxford atlases."
        )

    cort_img = nib.load(str(cort_path))
    lab = np.asanyarray(cort_img.dataobj, dtype=np.int32)
    names = _xml_names(cort_xml) if cort_xml.exists() else [
        f"Region_{i}" for i in range(int(lab.max()))
    ]

    if sub_path.exists():
        sub_img = nib.load(str(sub_path))
        if sub_img.shape == cort_img.shape and np.allclose(
            sub_img.affine, cort_img.affine
        ):
            sub_lab = np.asanyarray(sub_img.dataobj, dtype=np.int32)
            offset = int(lab.max())
            lab = lab + np.where((lab == 0) & (sub_lab > 0), sub_lab + offset, 0)
            if sub_xml.exists():
                names += _xml_names(sub_xml)

    present = np.unique(lab)
    present = present[present > 0]
    names = names[: len(present)]
    return lab, cort_img.affine, names


# --------------------------------------------------------------------------
# Coarse COM + scale alignment (same math as the GPU pipeline)
# --------------------------------------------------------------------------
def brain_moments(data: np.ndarray, affine: np.ndarray, max_pts: int = 60_000):
    """Center-of-mass + per-axis std of brain voxels in world coords."""
    ijk = np.argwhere(data > 0)
    if len(ijk) > max_pts:
        ijk = ijk[np.linspace(0, len(ijk) - 1, max_pts).astype(np.int64)]
    world = (affine[:3, :3] @ ijk.T).T + affine[:3, 3]
    return world.mean(0), world.std(0)


def norm_aff(com, sd):
    """4x4 affine: normalized space -> world (diag(sd) scale + com offset)."""
    W = np.eye(4)
    W[:3, :3] = np.diag(sd)
    W[:3, 3] = com
    return W


def resample_to_atlas_cpu(
    data: np.ndarray,
    scan_affine: np.ndarray,
    atlas_affine: np.ndarray,
    atlas_shape,
    atlas_norm_inv: np.ndarray | None = None,
) -> np.ndarray:
    """Trilinear-resample a scan volume onto the atlas grid.

    Equivalent to the torch grid_sample path: computes scan voxel coords for
    every atlas voxel via M = inv(A_scan) @ W_scan @ atlas_norm_inv @ A_atlas,
    then samples with map_coordinates (order=1, mode='nearest' == border).
    """
    if data.ndim == 4:
        data = data[..., 0]
    data = np.ascontiguousarray(data, dtype=np.float32)

    if atlas_norm_inv is not None:
        com_s, sd_s = brain_moments(data, scan_affine)
        M = (
            np.linalg.inv(scan_affine)
            @ norm_aff(com_s, sd_s)
            @ atlas_norm_inv
            @ atlas_affine
        )
    else:
        M = np.linalg.inv(scan_affine) @ atlas_affine

    ax = [np.arange(s, dtype=np.float32) for s in atlas_shape]
    ii, jj, kk = np.meshgrid(*ax, indexing="ij")
    hom = np.stack([ii, jj, kk, np.ones_like(ii)], axis=-1)  # (...,4)
    p = hom @ M.T.astype(np.float32)  # (...,4) scan voxel coords (i,j,k)

    coords = np.stack([p[..., 0], p[..., 1], p[..., 2]], axis=0).reshape(3, -1)
    out = map_coordinates(data, coords, order=1, mode="nearest")
    return out.reshape(atlas_shape).astype(np.float32)


def roi_means_np(resampled: np.ndarray, lab_flat: np.ndarray, counts: np.ndarray):
    """Z-normalize within the brain mask, then mean intensity per label."""
    mask = resampled > 0.01
    if int(mask.sum()) < 1000:  # degenerate
        return np.full(counts.shape, np.nan, dtype=np.float32)
    mu = resampled[mask].mean()
    sd = max(resampled[mask].std(), 1e-6)
    zn = (resampled - mu) / sd
    sums = np.bincount(
        lab_flat, weights=zn.reshape(-1), minlength=counts.size
    )
    return (sums / np.maximum(counts, 1)).astype(np.float32)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
class AtlasContext:
    """Pre-loaded atlas arrays shared across repeated extractions."""

    def __init__(self, atlas_dir: Path | None = None):
        self.lab_vol, self.affine, self.names = load_atlas_volumes(atlas_dir)
        com_a, sd_a = brain_moments(self.lab_vol > 0, self.affine)
        self.norm_inv = np.linalg.inv(norm_aff(com_a, sd_a))
        present = np.unique(self.lab_vol)
        self.present = present[present > 0]
        self.counts = np.stack(
            [(self.lab_vol == p).sum() for p in self.present]
        ).astype(np.float64)
        # label value -> column index (same as torch.searchsorted)
        self.lab_flat = np.searchsorted(self.present, self.lab_vol.reshape(-1))


_CTX: AtlasContext | None = None


def get_ctx() -> AtlasContext:
    global _CTX
    if _CTX is None:
        _CTX = AtlasContext()
    return _CTX


def extract_roi_vector(nii_path: str | Path, ctx: AtlasContext | None = None):
    """Extract per-region mean intensities from one skull-stripped scan.

    Returns (roi_values float32 [n_regions], region_names list[str]).
    Order = ascending atlas label id — identical to the parquet column order.
    """
    ctx = ctx or get_ctx()
    img = nib.load(str(nii_path))
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    res = resample_to_atlas_cpu(
        data, img.affine, ctx.affine, ctx.lab_vol.shape, ctx.norm_inv
    )
    feats = roi_means_np(res, ctx.lab_flat, ctx.counts)
    return feats, ctx.names


def roi_column_names(names: list[str]) -> list[str]:
    """Build roi_XX_<Safe_Name> column names matching the parquet schema."""
    cols = []
    for c, n in enumerate(names):
        safe = re.sub(r"[^A-Za-z0-9]+", "_", n).strip("_")
        cols.append(f"roi_{c:02d}_{safe}")
    return cols
