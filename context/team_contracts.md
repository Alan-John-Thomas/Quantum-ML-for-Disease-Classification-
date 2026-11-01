# Team Division & Interface Contracts

**Project:** Quantum Machine Learning for Autism Spectrum Disorder (ASD) Classification  
**Target Dataset:** ABIDE I Structural Brain MRI (504 Controls, 500 Autism Scans)  

This document formalizes the division of labor between the two team members, defines the interface contracts for all data and code artifacts, and outlines the synchronization points.

---

## 1. Member Ownership Matrix

| Area | Member 1 (Alan — Quantum Lead) | Member 2 (Jeswin — Neuroimaging & Baselines Lead) |
| :--- | :--- | :--- |
| **Primary Focus** | Quantum Circuits, Quantum Kernels, VQC Engine | MRI Preprocessing, ROI Extraction, Baseline Harness |
| **Owned Components** | - Modular data loader for `abide_roi_features.parquet`<br>- PCA dimensionality reduction ($d \to 4-6$ components)<br>- Feature mapping (`ZFeatureMap`, `ZZFeatureMap` linear/full)<br>- Variational ansatz design (`RealAmplitudes`)<br>- Statevector & shot-based Quantum Kernel Gram matrix<br>- VQC architecture & optimizer tuning (`COBYLA`, `SPSA`)<br>- Transpilation & circuit complexity profiling (depth, CNOT count) | - Accessing local MRI scans (`D:\study\autisim\`)<br>- Running `nilearn` resampling & Harvard-Oxford atlas masking<br>- Exporting `data/abide_roi_features.parquet` to `dev`<br>- Generating brain slice visualization overlays for slides<br>- Classical baselines (Logistic Regression, SVM-RBF, RF)<br>- Evaluation metrics & presentation slide design |
| **Shared Reviews** | Verifies feature scaling strictly bounded in $[0, \pi]$ for angle rotation. | Reviews circuit diagrams and benchmark tables for presentation. |

---

## 2. Technical Interface Contracts

### Contract 1: MRI ROI Parquet Handoff Schema (`data/abide_roi_features.parquet`)
Member 2 (Jeswin) extracts features from `.nii.gz` scans on his machine and outputs a single tabular Parquet file committed to `dev`:

```
data/abide_roi_features.parquet
```

**Schema Specifications:**
- **Number of rows:** $\sim 1,004$ (or current processed batch).
- **Identifier Column:**
  - `subject_id` (int or string, e.g. `50002`, `50030`).
- **Target Label Column:**
  - `dx_group` (int: `1` = Autism Spectrum Disorder, `2` = Typical Control).
- **Feature Columns:**
  - Continuous float columns representing regional mean intensities (e.g. `roi_01`, `roi_02`, ... or anatomical names from Harvard-Oxford atlas like `Frontal_Pole`, `Insular_Cortex`, etc.).
  - Recommended ROI count: 48 to 110 regions.

**Invariants:**
- No `NaN`, `null`, or infinite values (impute with median or zero).
- File size is compact ($\sim 300\text{ KB} - 1\text{ MB}$), cleanly committable to git.
- **Zero PII:** Only anonymized numerical identifiers and mean intensities.

---

### Contract 2: Preprocessed Quantum-Ready Data Loader
Member 1 (Alan) consumes `abide_roi_features.parquet` through `src/data_loader.py`:

```python
def load_and_preprocess_data(
    parquet_path: str = "data/abide_roi_features.parquet",
    n_features: int = 4,
    test_size: float = 0.25,
    random_state: int = 42,
    subsample_train: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        X_train: np.ndarray, shape (N_train, n_features), dtype float64, bounded in [0, np.pi]
        X_test:  np.ndarray, shape (N_test, n_features),  dtype float64, bounded in [0, np.pi]
        y_train: np.ndarray, shape (N_train,), dtype int64 (binary: {0, 1})
        y_test:  np.ndarray, shape (N_test,),  dtype int64 (binary: {0, 1})
    """
```

**Invariants:**
- `X_train.shape[1] == X_test.shape[1] == n_features` ($4$ or $6$ qubits).
- `np.all((X_train >= 0.0) & (X_train <= np.pi)) == True`.
- `np.all((X_test >= 0.0) & (X_test <= np.pi)) == True`.
- Target labels mapped: Autism = `1`, Control = `0`.
- Stratified splitting applied (`stratify=y`).
- Scalers fitted on train set only (`fit_transform` on train, `transform` on test).

---

### Contract 3: Quantum Kernel Gram Matrix Schema
Member 1 provides the precomputed kernel computation:

```python
# K_train: shape (N_train, N_train), symmetric: K_train[i, j] == K_train[j, i], diagonal == 1.0
# K_test:  shape (N_test, N_train)
```

**Invariants:**
- Diagonal entries $K_{ii} = 1.0$.
- $0.0 \le K_{ij} \le 1.0$ for all elements.
- The precomputed Gram matrix is passed directly to `sklearn.svm.SVC(kernel='precomputed')`.

---

### Contract 4: Benchmark Evaluation Results Schema
All models log metrics using the standard dictionary format:

```python
{
    "model_name": str,                  # e.g., "Quantum Kernel SVM (ZZ, reps=2)"
    "accuracy": float,                  # e.g., 0.6850
    "balanced_accuracy": float,         # e.g., 0.6830 (critical for clinical datasets)
    "precision": float,                 # e.g., 0.6700
    "recall": float,                    # e.g., 0.7100
    "f1_score": float,                  # e.g., 0.6894
    "roc_auc": float,                   # e.g., 0.7250
    "train_time_sec": float,            # e.g., 3.85
    "inference_time_ms_per_sample": float, # e.g., 0.45
    "circuit_qubits": int | None,       # 4 (or None for classical)
    "circuit_depth": int | None,        # 12 (or None for classical)
    "cnot_count": int | None            # 6 (or None for classical)
}
```

---

## 3. Team Checkpoints & Synchronization Timeline

| Checkpoint | Time Target | Milestone Objective | Deliverable |
| :--- | :--- | :--- | :--- |
| **CP-1** | 12:00 PM | Parquet Feature Handoff | Jeswin pushes `data/abide_roi_features.parquet` to `dev` |
| **CP-2** | 12:45 PM | Baselines & Feature Maps Ready | Classical models trained + transpiled `ZZFeatureMap` logged |
| **CP-3** | 01:30 PM | Mid-Day Sync & Lunch | Quantum Kernel SVM benchmarked on test set |
| **CP-4** | 03:00 PM | VQC Trained & Benchmarks Frozen | VQC loss convergence + consolidated comparison table |
| **CP-5** | 04:00 PM | Presentation & Demo Dry Run | Slides completed, notebook demo verified end-to-end |
