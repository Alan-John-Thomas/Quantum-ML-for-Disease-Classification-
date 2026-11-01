# Worklog — MRI ROI Feature Extraction (S1-02 / S1-03)

**Owner:** Jeswin | **Written by:** Devin | **Date:** 2026-11-01
**Goal:** turn `D:\study\autisim\` (1,004 skull-stripped `.nii.gz` scans) into
`data/abide_roi_features.parquet` per **Contract 1** in `team_contracts.md`,
ready for Alan's `src/data_loader.py` on `dev`.

---

## 1. What I'm doing and why

The sprint board assigns me the neuroimaging pipeline. The raw scans cannot be
used directly (mixed voxel grids — verified in `02-dataset-notes.md`), so the
job is:

```
1,004 .nii.gz scans  →  resample onto atlas grid (GPU, torch grid_sample)
                     →  Harvard-Oxford atlas → mean intensity per ROI
                     →  1 row per subject: subject_id, dx_group, roi_*
                     →  data/abide_roi_features.parquet  →  push to dev
```

Alan consumes only the parquet — he never touches nibabel/nilearn.

## 2. Hardware / acceleration plan

| Component | Device | Why |
|---|---|---|
| gunzip + NIfTI header parse | CPU (nibabel) | I/O-bound; fast enough |
| **Affine resampling** | **GPU — RTX 3050 6GB, torch 2.5.1+cu121 (already installed)** | `grid_sample` does trilinear resample in ~ms vs ~1–3 s/scan on scipy CPU → ~50× faster |
| **ROI mean reduction** | **GPU** | one `scatter_add` per region vs CPU loop |
| Output assembly | CPU (pandas → pyarrow) | trivial |

CUDA is only used where it actually matters (3D interpolation + reduction).
PyTorch was already installed with working CUDA — zero download cost.

## 3. Pipeline details

### Step 1 — Atlas
`nilearn.datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')`
→ 48 cortical regions, grid 91×109×91 @ 2 mm MNI space.
Optionally + `sub-maxprob-thr25-2mm` (21 subcortical) → up to ~69 ROI columns,
inside the contract's 48–110 range. Atlas downloads once, caches in
`~/nilearn_data` (documented in `03-mri-technical-deepdive.md`).

### Step 2 — Per-scan GPU resampling
For each scan:

1. `nibabel.load` → float32 volume + 4×4 affine `A_scan` (voxel→world mm).
2. **COM+scale alignment**: the repack scans are in native scanner space, not
   MNI — raw affines alone leave atlas regions sampling background (Frontal
   Pole came out ~-100σ). We fit a moments-based world→world transform:
   center-of-mass + per-axis std of brain voxels normalized onto the atlas
   brain's moments. Then `M = A_scan⁻¹ @ T_scan @ T_atlas⁻¹ @ A_atlas`.
   **Empirically validated**: 5-fold LogisticRegression on 100 scans =
   **0.68 acc aligned vs 0.54 unaligned** — kept as default
   (`--no-com-align` disables). This is NOT nonlinear registration.
3. `torch.nn.functional.grid_sample(mode='bilinear', align_corners=False)`
   resamples the scan onto the atlas grid on GPU (custom grid per scan).
4. Label lookup: `atlas_labels` are already on the atlas grid → for each region
   id, `index_add` resampled intensities ÷ voxel count → ROI mean.
5. Per-scan intensity z-normalization over brain voxels (nonzero) before ROI
   means — mitigates inter-site intensity scale differences.

### Step 3 — Row assembly
`{subject_id, dx_group, roi_<atlas_index>_<name>}` → DataFrame → parquet.
- `dx_group`: folder-derived (`treatment`→1, `control`→2) matching ABIDE
  `DX_GROUP` convention and Contract 1.
- Contract invariants checked before write: no NaN/inf (median-impute if any
  region is empty), float32 features, ~1 MB output.

## 4. Known limitations (honest — also for the report/slides)

- **No nonlinear MNI registration**: COM+scale moments alignment only — ROI
  means are *approximate* regional averages, not precise anatomical parcels.
  Peripheral ROIs sample the brain edge; their extreme z-values encode
  size/pose differences (still valid signal — see the 0.68 vs 0.54 ablation).
- **Labels**: `dx_group` taken from official ABIDE phenotypic CSV when
  available (auto-downloaded), folder name otherwise; mismatches logged.
- **Site confound**: 17 sites; scanner artifacts may inflate separability.
- **GPU speedup scope**: CUDA accelerates interpolation/reduction only; gzip
  decode remains CPU — expected wall-clock ~5–15 min for 1,004 scans vs
  ~45–90 min CPU-only.
- ROI count (~48–69) lands inside contract range; PCA happens later in Alan's
  loader (4–6 components → qubits).

## 5. Verification checklist before push

- [ ] row count = number of successfully processed scans (expect ~1,004)
- [ ] `subject_id` unique, matches filenames
- [ ] `dx_group` ∈ {1, 2}; class balance ~50/50
- [ ] `df.isna().sum().sum() == 0`, all features finite
- [ ] file size < ~2 MB (committable)
- [ ] quick sanity: control-vs-autism ROI mean separability nonzero signal
  (t-test on a few ROIs) — proves extraction isn't garbage
- [ ] failed/odd scans logged to `data/extraction_failures.csv`

## 6. Repo artifacts produced

| Path | Purpose |
|---|---|
| `src/extract_roi_features.py` | the extraction script (reproducible, documented) |
| `data/abide_roi_features.parquet` | the handoff deliverable (Contract 1) |
| `data/extraction_failures.csv` | scans that errored (if any) |
| `context/05-mri-extraction-worklog.md` | this file |
