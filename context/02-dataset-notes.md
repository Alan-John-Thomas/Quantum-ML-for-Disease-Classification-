# Dataset Notes — ABIDE I repack (verified locally 2026-11-01)

## What's actually on disk — UPDATED

| Folder | Files | Subject-ID range | Meaning |
|---|---|---|---|
| `D:\study\autisim\control\` | **504** `.nii.gz` | 50030–51570 | Typical controls (ABIDE `DX_GROUP=2`) |
| `D:\study\autisim\treatment\` | **500** `.nii.gz` | 50002–51607 | Autism group (ABIDE `DX_GROUP=1`) |

- **Total: 1,004 skull-stripped structural T1 scans, ~2.1 GB** — both classes
  already local, near-balanced (~50/50). No further scan downloads needed.
- Filename scheme `{SUB_ID}_skull_stripped.nii.gz` uses ABIDE I subject IDs —
  this matches the repackaged ABIDE I distribution circulating on Kaggle
  (control/treatment split already done).
- ⚠️ **Verify labels anyway**: folder names are the only local label source.
  Cross-check `SUB_ID` against the official phenotypic CSV below before
  training — a repackaged dataset may have QC-excluded or relabeled subjects.

## NIfTI header facts (parsed from actual files)

- Format: **NIfTI-1 single-file** (magic `n+1`), gzip-compressed — reads cleanly
  with `nibabel` / `nilearn` on any OS.
- dtype: **float32** everywhere (datatype=16, 32 bpp).
- qform_code=1, sform_code=1 — scanner-space affines present.
- `descrip` field: `2203.12-dirty 2024-02-01` — reprocessed by an external
  skull-stripping pipeline in Feb 2024 (not raw scanner output).
- `pixdim[0] = -1` (qfac=-1) → radiological storage convention; nibabel derives
  the correct affine — **do not manually flip axes**.

## ⚠️ THE critical gotcha: shapes are NOT uniform

Sampled headers show **mixed voxel geometries** — e.g. `(176,256,256)`,
`(160,240,256)`, `(160,480,512)`, `(256,256,124)`, `(105,256,256)`,
`(161,200,200)`. Root cause: ABIDE I aggregates ~17 acquisition sites with
different scanners/protocols; this repack skull-stripped the scans but did
**not** resample them to a common grid.

Consequences:

1. **You cannot `np.stack` raw voxels** — every pipeline needs a
   resample-to-common-grid step first (see `03-mri-technical-deepdive.md`).
2. Voxel sizes also vary between scans — always derive features through the
   affine, never assume fixed mm spacing.
3. **Memory**: largest scans ~160×480×512×4 B ≈ **157 MB decompressed each**;
   naive full load ≈ 40+ GB → stream/batch, never load all 1,004 at once.

## Labels & phenotypic data

- Official file: `Phenotypic_V1_0b_preprocessed1.csv` — public, hosted at
  `https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv`
- Key columns: `SUB_ID`, `DX_GROUP` (1=autism, 2=control), `AGE_AT_SCAN`,
  `SEX`, `SITE_ID`, `FIQ`, `FILE_ID`.
- Safe features: `AGE_AT_SCAN`, `SEX`, `FIQ`, `SITE_ID`.
- **Leakage warning**: `ADOS_*`, `ADI_R_*`, `SRS`, `VINELAND` — only measured
  on autism subjects → using them as features leaks the label. Exclude.
- **Site confound**: scanner site correlates with diagnosis; consider
  site-stratified CV or disclose as limitation.

## Legal/sharing

- ABIDE requires a data-use agreement; **do NOT commit `.nii.gz` to git**.
  Ship the subject-ID list + download instructions in the README — this also
  satisfies the challenge's no-PII rule (scans are anonymized, no PII anyway).

## Environment on this machine (as of 2026-11-01)

- Python **3.10.11** at system level; `nibabel`, `nilearn`, `qiskit`,
  `qiskit-machine-learning`, `scikit-learn`, `pandas`, `numpy` — **none
  installed yet**. See `03-mri-technical-deepdive.md` for the pinned stack.
