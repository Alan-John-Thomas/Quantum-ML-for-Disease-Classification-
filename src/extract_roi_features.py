"""
GPU-accelerated MRI ROI feature extraction — ABIDE I skull-stripped scans.

Reads {control,treatment}/*_skull_stripped.nii.gz, resamples every scan onto
the Harvard-Oxford atlas grid with CUDA (torch.nn.functional.grid_sample,
trilinear), computes per-region mean intensities on GPU, and writes
data/abide_roi_features.parquet matching Contract 1 in
context/team_contracts.md:

    subject_id (int) | dx_group (1=autism, 2=control) | roi_<NN>_<Name> floats

CUDA usage: 3D interpolation + per-region reduction run on the GPU
(~50x faster than scipy CPU resampling). gzip decode stays on CPU.

Usage:
    python src/extract_roi_features.py                 # full 1,004 scans
    python src/extract_roi_features.py --limit 20      # smoke test
    python src/extract_roi_features.py --device cpu    # fallback
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PHENOTYPIC_URL = (
    "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/"
    "Phenotypic_V1_0b_preprocessed1.csv"
)
DX = {"control": 2, "treatment": 1}  # ABIDE DX_GROUP convention (Contract 1)


# --------------------------------------------------------------------------
# Atlas
# --------------------------------------------------------------------------
def load_atlas(no_subcortical: bool):
    """Harvard-Oxford cortical (48) + subcortical (21) labels on MNI 2mm grid.

    Returns (label_volume np.int32 [91,109,91], affine 4x4, names list[str]).
    """
    from nilearn import datasets

    cort = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    cort_img = cort.maps if hasattr(cort.maps, "affine") else nib.load(cort.maps)
    lab = np.asanyarray(cort_img.dataobj, dtype=np.int32)
    names = list(cort.labels[1:])  # labels[0] == 'Background'

    if not no_subcortical:
        sub = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
        sub_img = sub.maps if hasattr(sub.maps, "affine") else nib.load(sub.maps)
        if sub_img.shape == cort_img.shape and np.allclose(
            sub_img.affine, cort_img.affine
        ):
            sub_lab = np.asanyarray(sub_img.dataobj, dtype=np.int32)
            offset = int(lab.max())
            # subcortical fills voxels where cortical map is background
            lab = lab + np.where((lab == 0) & (sub_lab > 0), sub_lab + offset, 0)
            names += list(sub.labels[1:])
        else:
            print("[atlas] subcortical grid mismatch — cortical only")

    # keep only label ids actually present
    present = np.unique(lab)
    present = present[present > 0]
    names = names[: len(present)]
    print(f"[atlas] {len(present)} regions on grid {lab.shape}")
    return lab, cort_img.affine, names


# --------------------------------------------------------------------------
# Coarse alignment (COM + per-axis std) — scans are in native scanner space,
# NOT MNI. Without this, peripheral ROIs sample background. This is a
# moments-based affine fit, not nonlinear registration (documented limitation).
# --------------------------------------------------------------------------
def brain_moments(data, affine, max_pts=60_000):
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


def resample_to_atlas(img, atlas_affine, atlas_shape, device, com_align=True,
                      atlas_norm_inv=None):
    """Resample a scan volume onto the atlas grid. Returns float32 tensor."""
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    # atlas-voxel -> scan-voxel transform
    if com_align:
        com_s, sd_s = brain_moments(data, img.affine)
        M = (np.linalg.inv(img.affine) @ norm_aff(com_s, sd_s)
             @ atlas_norm_inv @ atlas_affine)
    else:
        M = np.linalg.inv(img.affine) @ atlas_affine
    vol = torch.from_numpy(np.ascontiguousarray(data)).to(device)[None, None]
    grid = make_grid(
        torch.from_numpy(M.astype(np.float32)).to(device),
        atlas_shape,
        data.shape,
        device,
    )
    out = F.grid_sample(
        vol, grid, mode="bilinear", padding_mode="border", align_corners=False
    )[0, 0]
    return out
def make_grid(M: torch.Tensor, atlas_shape, scan_shape, device):
    """Normalized sampling grid for grid_sample.

    M: (4,4) atlas-voxel -> scan-voxel transform = inv(A_scan) @ A_atlas.
    Output grid indexed [i,j,k] on the atlas lattice; last dim holds
    normalized scan coords ordered (k-norm, j-norm, i-norm) to match
    grid_sample's (x->W, y->H, z->D) convention for input (Sx,Sy,Sz).
    """
    ax = [torch.arange(s, dtype=torch.float32, device=device) for s in atlas_shape]
    ii, jj, kk = torch.meshgrid(*ax, indexing="ij")
    hom = torch.stack([ii, jj, kk, torch.ones_like(ii)], dim=-1)  # (...,4)
    p = hom @ M.T  # (...,4) scan voxel coords (i,j,k)
    g = torch.empty(*atlas_shape, 3, device=device)
    g[..., 0] = 2.0 * (p[..., 2] + 0.5) / scan_shape[2] - 1.0
    g[..., 1] = 2.0 * (p[..., 1] + 0.5) / scan_shape[1] - 1.0
    g[..., 2] = 2.0 * (p[..., 0] + 0.5) / scan_shape[0] - 1.0
    return g.unsqueeze(0)


def roi_means(resampled, lab_flat, counts, device):
    """Z-normalize within the brain mask, then mean intensity per label."""
    mask = resampled > 0.01
    if int(mask.sum()) < 1000:  # degenerate — return NaNs, imputed later
        return torch.full((counts.numel(),), float("nan"), device=device)
    mu = resampled[mask].mean()
    sd = resampled[mask].std().clamp_min(1e-6)
    zn = (resampled - mu) / sd
    sums = torch.zeros(counts.numel(), device=device)
    sums.index_add_(0, lab_flat, zn.reshape(-1))
    return sums / counts.clamp_min(1)


# --------------------------------------------------------------------------
# Label verification (best-effort, non-fatal offline)
# --------------------------------------------------------------------------
def fetch_phenotypic(data_dir: Path):
    """Download ABIDE phenotypic CSV → {SUB_ID: DX_GROUP} map, or {} offline."""
    csv = data_dir / "Phenotypic_V1_0b_preprocessed1.csv"
    try:
        if not csv.exists():
            print("[labels] downloading phenotypic CSV…")
            urllib.request.urlretrieve(PHENOTYPIC_URL, csv)
        df = pd.read_csv(csv)
        return dict(zip(df["SUB_ID"].astype(int), df["DX_GROUP"].astype(int)))
    except Exception as e:  # noqa: BLE001
        print(f"[labels] phenotypic CSV unavailable ({e}); using folder labels")
        return {}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=r"D:\study\autisim")
    ap.add_argument("--out", default="data/abide_roi_features.parquet")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-subcortical", action="store_true")
    ap.add_argument("--no-com-align", action="store_true",
                    help="disable COM+scale alignment to atlas (NOT recommended)")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"[env] device={device} torch={torch.__version__}")
    if device.type == "cuda":
        print(f"[env] gpu={torch.cuda.get_device_name(0)}")

    root = Path(args.data_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lab_vol, atlas_aff, names = load_atlas(args.no_subcortical)
    com_align = not args.no_com_align
    atlas_norm_inv = None
    if com_align:
        com_a, sd_a = brain_moments(lab_vol > 0, atlas_aff)
        atlas_norm_inv = np.linalg.inv(norm_aff(com_a, sd_a))
        print(f"[align] COM+scale alignment ON (atlas COM={com_a.round(1)})")
    lab_t = torch.from_numpy(lab_vol).to(device)
    present = torch.unique(lab_t)
    present = present[present > 0]
    n_reg = present.numel()
    counts = torch.stack([(lab_t == p).sum() for p in present]).float()
    lab_flat = torch.searchsorted(present, lab_t.reshape(-1))  # label -> col idx

    dx_map = fetch_phenotypic(out_path.parent)

    # collect files
    files = []
    for folder, folder_dx in DX.items():
        for f in sorted((root / folder).glob("*_skull_stripped.nii.gz")):
            sid = int(re.match(r"(\d+)", f.name).group(1))
            files.append((f, sid, folder_dx))
    if args.limit:
        files = files[: args.limit]
    print(f"[scan] {len(files)} files")

    rows, failures, mismatches = [], [], []
    t0 = time.time()
    for n, (path, sid, folder_dx) in enumerate(files, 1):
        try:
            img = nib.load(str(path))
            res = resample_to_atlas(img, atlas_aff, lab_vol.shape, device,
                                    com_align, atlas_norm_inv)
            feats = roi_means(res, lab_flat, counts, device).cpu().numpy()
            dx = dx_map.get(sid, folder_dx)
            if sid in dx_map and dx_map[sid] != folder_dx:
                mismatches.append((sid, folder_dx, dx_map[sid]))
            rows.append({"subject_id": sid, "dx_group": dx, **dict(
                zip(range(n_reg), feats.astype(np.float32)))})
        except Exception as e:  # noqa: BLE001
            failures.append((sid, str(e)))
        if n % 100 == 0 or n == len(files):
            dt = time.time() - t0
            print(f"[{n}/{len(files)}] {dt:.0f}s  ~{dt/n*1000:.0f} ms/scan")

    # assemble: numeric keys -> named columns
    df = pd.DataFrame(rows)
    rename = {}
    for c in df.columns:
        if isinstance(c, int):
            safe = re.sub(r"[^A-Za-z0-9]+", "_", names[c]).strip("_")
            rename[c] = f"roi_{c:02d}_{safe}"
    df = df.rename(columns=rename)

    # Contract 1 invariants
    roi_cols = [c for c in df.columns if c.startswith("roi_")]
    df[roi_cols] = df[roi_cols].fillna(df[roi_cols].median())
    df[roi_cols] = df[roi_cols].fillna(0.0)
    assert df[roi_cols].isna().sum().sum() == 0
    assert np.isfinite(df[roi_cols].to_numpy()).all()
    df = df.sort_values("subject_id").reset_index(drop=True)

    df.to_parquet(out_path, index=False)
    print(f"[done] {df.shape} -> {out_path}  ({out_path.stat().st_size/1e6:.2f} MB)")
    print(f"[classes] {df['dx_group'].value_counts().to_dict()}")
    if failures:
        pd.DataFrame(failures, columns=["subject_id", "error"]).to_csv(
            out_path.parent / "extraction_failures.csv", index=False)
        print(f"[warn] {len(failures)} failures -> extraction_failures.csv")
    if mismatches:
        print(f"[warn] {len(mismatches)} folder-vs-phenotypic label mismatches")
        for s, a, b in mismatches[:20]:
            print(f"   {s}: folder={a} phenotypic={b}")

    # sanity: top separable ROIs by |group mean difference|
    g1 = df[df.dx_group == 1][roi_cols].mean()
    g2 = df[df.dx_group == 2][roi_cols].mean()
    diff = (g1 - g2).abs().sort_values(ascending=False)
    print("[sanity] top-5 separable ROIs:")
    print(diff.head(5).to_string())


if __name__ == "__main__":
    sys.exit(main())
