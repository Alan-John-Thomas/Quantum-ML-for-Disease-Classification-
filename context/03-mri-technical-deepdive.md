# MRI Technical Deep-Dive — file formats, compatibility, pipeline

> Research summary for the ABIDE-MRI autism-classification option (Option A in
> `01-idea-analysis.md`). Verified against the actual local files on 2026-11-01.

## 1. File format facts

| Property | Value (verified) | Implication |
|---|---|---|
| Container | `.nii.gz` = gzip-compressed NIfTI-1 | `nibabel.load()` handles transparently; no manual gunzip |
| Header | 348 B, magic `n+1` | Single-file NIfTI — compatible with all neuroimaging tools |
| dtype | float32 | Fine for nibabel→numpy→sklearn; halves memory vs float64 |
| dims | **Mixed** (see dataset notes) | Mandatory resample-to-common-grid step |
| Orientation | qfac=-1, qform/sform=1 | Radiological convention; let the affine do the work |
| Per-file size | ~2–3 MB gz, **40–157 MB decompressed** | Stream; ~46 GB if all loaded |

## 2. Windows compatibility matrix

| Tool | Windows support | Needed? |
|---|---|---|
| `nibabel` | ✅ pure-python wheel | Yes — I/O |
| `nilearn` | ✅ pure-python wheel | Yes — masking/resampling/atlas |
| `scikit-learn`, `pandas`, `numpy` | ✅ | Yes |
| `qiskit` 2.x, `qiskit-aer` | ✅ prebuilt wheels | Yes — QSVC/VQC, sim |
| `qiskit-machine-learning` 0.9.x | ✅ (needs Python ≥3.10 — we have 3.10.11, exactly the floor) | Yes |
| FSL / FreeSurfer / ANTs | ❌ native (WSL2 only) | **No** — skull stripping already done; we don't need them. Avoid in README deps or the judges can't reproduce on Windows either |
| HD-BET / deep brain extraction | ⚠️ needs torch, heavy | No — already skull-stripped |

**Takeaway**: a pure-Python stack (nibabel + nilearn + sklearn + qiskit) is
fully reproducible on Windows with `pip install` only — good for the
"another person can reproduce" rubric item. Never depend on FSL/FreeSurfer.

## 3. Version pins (researched)

- `qiskit-machine-learning` latest = **0.9.1** (Aug 2026): requires
  `qiskit>=2.0`, `numpy>=2.0`, `scikit-learn>=1.2`, Python ≥3.10.
- ⚠️ **API gotcha**: in qml 0.9.x, `VQC`/`QSVC` default to **V1 primitives
  with a deprecation-style warning**. Pass V2 primitives explicitly —
  `StatevectorSampler` / `StatevectorEstimator` from `qiskit.primitives` —
  and cite `FidelityQuantumKernel` (fidelity-based kernel) in the writeup.
- `nilearn` current (0.12+) supports numpy 2.x + Python 3.10 — compatible
  with the qiskit stack. One env works; no need to split.
- Suggested `requirements.txt`: `nibabel`, `nilearn`, `scikit-learn`,
  `pandas`, `numpy>=2`, `matplotlib`, `qiskit>=2`, `qiskit-aer`,
  `qiskit-machine-learning>=0.8` (+ pin exact after testing).

## 4. Pipeline design (revised for mixed geometries)

```
.nii.gz (mixed shapes/spaces)
  → resample each scan to a common reference grid
        Option 1 (fast, crude): nilearn.image.resample_img(img,
            target_affine=diag(~3mm), target_shape=(64,76,64))
            — uniform voxels, no true anatomical registration
        Option 2 (better, slower): resample_to_img() to an MNI152
            template (nilearn.datasets.fetch_mni152_template)
            — affine-aligned, still not nonlinear registration
  → extract features:
        a) NiftiLabelsMasker + Harvard-Oxford atlas → per-region means (~100)
        b) global volumetrics (nonzero-voxel count × voxel volume, etc.)
        c) voxel flatten → PCA (baseline/simplest)
  → StandardScaler → PCA to 4–8 components (= 4–8 qubits)
  → ZZFeatureMap → FidelityQuantumKernel + QSVC   (primary model)
       and/or VQC(feature_map=ZZFeatureMap, ansatz=RealAmplitudes)
  → classical baselines on identical features: SVM-RBF, RF, LogReg
```

### Practical notes

- **Resampling is the bottleneck**: ~1,004 resamples ≈ tens of minutes on CPU.
  Cache the extracted feature matrix to `features.parquet`/`.npy` — run once,
  then iterate on the ML side instantly. Judges love a cached artifact.
- **Atlas download**: `fetch_atlas_harvard_oxford` and `fetch_mni152_template`
  download on first call (needs internet once; cached in `~/nilearn_data`).
  For offline reproducibility, commit the fetched atlas files or pre-fetch in
  a setup script.
- **Registration honesty**: option-1 resampling gives coarse alignment only —
  brain anatomy across subjects isn't truly matched. State this plainly in
  limitations; don't claim voxel-level correspondence. For hackathon scale
  this is acceptable and honest.
- **Alternative no-resample route**: brain-volume scalar features (voxel
  count, intensity histograms, moments) need no registration — weak features
  alone, but a legitimate "QC/volumetric" feature block to combine.
- **Batching**: process scans in a loop, write one CSV row per subject;
  3–8 GB peak if you keep ~50 resampled volumes in RAM — safer to write
  features per-scan and concat at the end.
- Subset for iteration: stratified 200-scan subsample for dev loop; full
  1,004 for the final run.

## 5. Quantum-side design notes

- **1 feature = 1 qubit** via rotation encoding (ZZFeatureMap defaults):
  4–8 PCA components → 4–8 qubits → statevector sim is instant.
- Kernel matrix cost: O(n²) fidelity evals. For 1,004 samples that's ~1M
  circuit evals — **too slow**. Use a subsample (~200–400) for QSVC, or
  `sampler` batching + chunked kernel eval; VQC scales better (O(n) per epoch).
- `FidelityQuantumKernel` + `QSVC` is the cleanest story; `PegasosQSVC` is the
  fast-scaling alternative worth mentioning.
- Noise-model run: `AerSimulator` with `ibm_*` noise model → degradation
  curve = strong "real-world" rubric content.

## 6. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Resampling of 1,004 scans takes too long | High | Coarse grid (~3 mm), parallel `joblib`, cache features, subsample |
| Labels in repack don't match phenotypic CSV | Low-Med | Join on SUB_ID early; drop mismatches |
| Site confound inflates accuracy | Med | Site-stratified split; report confound analysis |
| Class balance is ~50/50 → chance-level 50% | — | Report balanced acc/AUC, not raw acc alone |
| numpy 2.x breaks an older dep | Low | Fresh venv + pins above |
| Atlas fetch needs internet during demo | Med | Pre-fetch + commit atlas to repo (small) |

## 7. Suggested repo layout

```
├── context/            ← this shared context
├── data/
│   ├── README.md       ← download instructions + subject list (no .nii.gz!)
│   └── subject_ids.csv ← SUB_ID + label (small, committable)
├── notebooks/
│   ├── 01_extract_features.ipynb   (MRI → features.parquet, cached)
│   ├── 02_classical_baseline.ipynb
│   └── 03_quantum_qsvc_vqc.ipynb
├── src/ (optional .py mirrors for judges who prefer scripts)
├── requirements.txt
└── README.md           ← setup + run + limitations
```
