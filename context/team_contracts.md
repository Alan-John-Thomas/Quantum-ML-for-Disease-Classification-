# Team Division & Interface Contracts

This document formalizes the division of labor between the two team members, defines the interface contracts for all data and code artifacts, and outlines the synchronization points.

---

## 1. Member Ownership Matrix

| Area | Member 1 (Quantum Lead) | Member 2 (ML & Evaluation Lead) |
| :--- | :--- | :--- |
| **Primary Focus** | Quantum Circuits, Quantum Kernels, VQC | Clinical Data Pipeline, Classical Baselines, Benchmarking |
| **Owned Components** | - Feature map selection (`ZFeatureMap`, `ZZFeatureMap`)<br>- Variational ansatz design (`RealAmplitudes`, `EfficientSU2`)<br>- Statevector & shot-based Quantum Kernel estimator<br>- VQC architecture & optimizer tuning (`COBYLA`, `SPSA`)<br>- Circuit transpilation & complexity metrics (depth, CNOTs) | - Dataset ingestion (`load_breast_cancer` / Heart Disease)<br>- Normalization & PCA dimensionality reduction<br>- Train/Test splitting (`StratifiedKFold`, `random_state=42`)<br>- Classical baselines (Logistic Regression, SVM-RBF, RF)<br>- Evaluation harness (Accuracy, F1, ROC-AUC, latency)<br>- Presentation slide deck & demo orchestration |
| **Shared Reviews** | Reviews data scaling to verify it adheres to $[0, \pi]$ rotation limits. | Reviews circuit outputs to format tables and plots for presentation. |

---

## 2. Technical Interface Contracts

### Contract 1: Preprocessed Data Schema
Member 2 provides the data loader output consumed directly by Member 1's quantum circuits and classical baselines.

```python
# Function signature:
def load_and_preprocess_data(
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
- `X_train.shape[1] == X_test.shape[1] == n_features` (Default: $4$, optional $6$).
- `np.all((X_train >= 0.0) & (X_train <= np.pi)) == True`.
- `np.all((X_test >= 0.0) & (X_test <= np.pi)) == True`.
- Stratification MUST be applied (`stratify=y`).
- The scaler fitted on `X_train` MUST be used to transform `X_test` (no data leakage).

---

### Contract 2: Model Wrapper Interface (Scikit-Learn Standard)
Every classifier (classical or quantum) MUST implement or conform to standard scikit-learn estimator methods:

```python
class BaseDiseaseClassifier:
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseDiseaseClassifier":
        ...
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns 1D array of shape (N,) with integer class labels {0, 1}."""
        ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns 2D array of shape (N, 2) with predicted class probabilities [P(0), P(1)]."""
        ...
```

---

### Contract 3: Quantum Kernel Gram Matrix Schema
Member 1 provides the precomputed kernel computation:

```python
# Kernel computation schema:
# K_train: shape (N_train, N_train), symmetric: K_train[i, j] == K_train[j, i], diagonal == 1.0
# K_test:  shape (N_test, N_train)
```

**Invariants:**
- Diagonal entries $K_{ii} = 1.0$ (state fidelity with itself is unity).
- $0.0 \le K_{ij} \le 1.0$ for all elements.
- The precomputed Gram matrix is passed directly to `sklearn.svm.SVC(kernel='precomputed')`.

---

### Contract 4: Evaluation Results Format
Member 2's benchmark suite logs model performance using a standard dictionary schema:

```python
{
    "model_name": str,                  # e.g., "Quantum Kernel SVM (ZZ, reps=2)"
    "accuracy": float,                  # e.g., 0.9441
    "precision": float,                 # e.g., 0.9350
    "recall": float,                    # e.g., 0.9500
    "f1_score": float,                  # e.g., 0.9424
    "roc_auc": float,                   # e.g., 0.9812
    "train_time_sec": float,            # e.g., 4.25
    "inference_time_sec": float,        # e.g., 0.82
    "circuit_qubits": int | None,       # 4 (or None for classical)
    "circuit_depth": int | None,        # 12 (or None for classical)
    "cnot_count": int | None            # 6 (or None for classical)
}
```

---

## 3. Team Checkpoints & Synchronization Timeline

| Checkpoint | Time Target | Milestone Objective | Deliverable |
| :--- | :--- | :--- | :--- |
| **CP-1** | 11:15 AM | Data Pipeline & Baseline Lock | Clean `data_loader` and initial Logistic Regression / SVM baseline metrics |
| **CP-2** | 12:30 PM | Quantum Feature Maps Transpiled | Transpiled `ZZFeatureMap` depth & CNOT metrics on 4 qubits |
| **CP-3** | 02:30 PM | Quantum Kernel SVM Evaluated | Fast statevector Gram matrix computed, evaluated against classical SVM |
| **CP-4** | 03:45 PM | VQC Trained & Benchmarks Consolidated | Full comparative table (Classical vs Q-Kernel vs VQC) finalized |
| **CP-5** | 04:15 PM | Presentation & Demo Dry Run | Slides completed, Jupyter notebook demo verified end-to-end |
