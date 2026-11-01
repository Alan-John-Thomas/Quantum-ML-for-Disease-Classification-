"""
Persisted inference pipeline for real-time MRI disease classification.

Fits the exact preprocessing chain used at training time
(StandardScaler -> PCA(4) -> MinMaxScaler[0, pi] -> clip) on the same
stratified 75/25 split (random_state=42), trains the Quantum Kernel SVM
(full entanglement) plus a classical RBF-SVM comparator, and persists
everything to models/pipeline.joblib for instant server startup.

Quantum payload per classification:
  - pca_components: the 4 PCA projections
  - qubit_angles:   the [0, pi] rotation angles fed to ZZFeatureMap
  - statevector:    16 complex amplitudes of the encoded state |phi(x)>
  - bloch_vectors:  per-qubit reduced-state Bloch vectors (x, y, z)
  - kernel_row:     fidelity K(x, x_train_j) against all training states
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace
from qiskit.quantum_info.operators import Pauli
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from src.quantum_kernel import FastQuantumKernelClassifier

MODEL_PATH = Path("models/pipeline.joblib")
PARQUET_PATH = Path("data/abide_roi_features.parquet")

N_QUBITS = 4
LABEL_NAMES = {0: "Typical Control", 1: "Autism Spectrum Disorder"}

_PAULIS = [Pauli("X"), Pauli("Y"), Pauli("Z")]


# --------------------------------------------------------------------------
# Build / persist
# --------------------------------------------------------------------------
def _build_artifacts(parquet_path: Path, random_state: int = 42) -> dict:
    df = pd.read_parquet(parquet_path)
    feature_cols = [c for c in df.columns if c.startswith("roi_")]
    if not feature_cols:
        raise ValueError("No roi_* feature columns in parquet")
    X_raw = df[feature_cols].values
    raw_labels = df["dx_group"].values
    y = (raw_labels == 1).astype(int) if set(np.unique(raw_labels)).issubset({1, 2}) \
        else raw_labels.astype(int)
    subject_ids = df["subject_id"].values

    idx = np.arange(len(df))
    tr_idx, te_idx, y_train, y_test = train_test_split(
        idx, y, test_size=0.25, random_state=random_state, stratify=y
    )
    X_train_raw, X_test_raw = X_raw[tr_idx], X_raw[te_idx]

    std = StandardScaler().fit(X_train_raw)
    pca = PCA(n_components=N_QUBITS, random_state=random_state).fit(
        std.transform(X_train_raw)
    )
    minmax = MinMaxScaler(feature_range=(0.0, np.pi)).fit(
        pca.transform(std.transform(X_train_raw))
    )

    def encode(X):
        v = minmax.transform(pca.transform(std.transform(X)))
        return np.clip(v, 0.0, np.pi)

    X_train = encode(X_train_raw)
    X_test = encode(X_test_raw)

    print("[pipeline] fitting Quantum Kernel SVM (full entanglement)...")
    qsvm = FastQuantumKernelClassifier(
        n_qubits=N_QUBITS, reps=1, entanglement="full", C=1.0
    ).fit(X_train, y_train)
    circ = qsvm.get_circuit_metrics()

    print("[pipeline] fitting classical SVM (RBF, probability)...")
    rbf = SVC(kernel="rbf", probability=True, random_state=random_state).fit(
        X_train, y_train
    )

    return {
        "feature_cols": feature_cols,
        "std_scaler": std,
        "pca": pca,
        "minmax": minmax,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "train_subject_ids": subject_ids[tr_idx],
        "test_subject_ids": subject_ids[te_idx],
        "subject_ids": subject_ids,
        "dx_group": df["dx_group"].values,
        "X_raw": X_raw,
        "qsvm_svc": qsvm.svc,
        "rbf_svc": rbf,
        "circuit_metrics": circ,
        "pca_variance": pca.explained_variance_ratio_.tolist(),
        "n_qubits": N_QUBITS,
        "entanglement": "full",
        "reps": 1,
    }


class InferencePipeline:
    """Loads persisted artifacts (or builds them once) and classifies scans."""

    def __init__(self, parquet_path: Path = PARQUET_PATH,
                 model_path: Path = MODEL_PATH, retrain: bool = False):
        self.parquet_path = Path(parquet_path)
        self.model_path = Path(model_path)
        if self.model_path.exists() and not retrain:
            print(f"[pipeline] loading artifacts <- {self.model_path}")
            art = joblib.load(self.model_path)
        else:
            art = _build_artifacts(self.parquet_path)
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {k: v for k, v in art.items()}, self.model_path
            )
            print(f"[pipeline] artifacts saved -> {self.model_path}")
        self.__dict__.update(art)

        # Rebuild the quantum classifier shell around the persisted SVC and
        # recompute training statevectors (avoids pickling Statevector lists).
        self.qsvm = FastQuantumKernelClassifier(
            n_qubits=art["n_qubits"], reps=art["reps"],
            entanglement=art["entanglement"], C=1.0,
        )
        self.qsvm.X_train_ = self.X_train
        t0 = time.perf_counter()
        self.qsvm.train_states_ = self.qsvm._compute_statevectors(self.X_train)
        self.qsvm.svc = self.qsvm_svc
        print(f"[pipeline] {len(self.X_train)} train statevectors ready "
              f"({time.perf_counter() - t0:.1f}s)")

        self._test_sid_set = set(self.test_subject_ids.tolist())

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------
    def encode(self, roi_vector: np.ndarray) -> np.ndarray:
        """Raw 69-ROI vector -> 4 qubit rotation angles in [0, pi]."""
        v = self.minmax.transform(
            self.pca.transform(self.std_scaler.transform([roi_vector]))
        )
        return np.clip(v, 0.0, np.pi)[0]

    def _statevector(self, angles: np.ndarray) -> Statevector:
        bound = self.qsvm.feature_map.assign_parameters(angles)
        return Statevector.from_instruction(bound)

    @staticmethod
    def _bloch_vectors(sv: Statevector, n_qubits: int) -> list[list[float]]:
        """Per-qubit Bloch vector (x,y,z) from the reduced density matrix."""
        out = []
        for q in range(n_qubits):
            rho = DensityMatrix(
                partial_trace(sv, [j for j in range(n_qubits) if j != q])
            )
            out.append([float(rho.expectation_value(p).real) for p in _PAULIS])
        return out

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------
    def classify_roi_vector(self, roi_vector: np.ndarray,
                            source: str = "roi") -> dict:
        timings = {}
        t0 = time.perf_counter()
        angles = self.encode(roi_vector)
        timings["encode_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        t0 = time.perf_counter()
        sv = self._statevector(angles)
        kernel_row = np.array([
            abs(sv.inner(ts)) ** 2 for ts in self.qsvm.train_states_
        ])
        q_pred = int(self.qsvm_svc.predict(kernel_row.reshape(1, -1))[0])
        q_score = float(self.qsvm_svc.decision_function(
            kernel_row.reshape(1, -1))[0])
        timings["quantum_infer_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        c_pred = int(self.rbf_svc.predict([angles])[0])
        c_proba = self.rbf_svc.predict_proba([angles])[0].tolist()

        amps = sv.data
        basis = [format(i, f"0{self.n_qubits}b") for i in range(len(amps))]
        pcs = self.pca.transform(
            self.std_scaler.transform([roi_vector])
        )[0].tolist()

        return {
            "source": source,
            "quantum": {
                "prediction": q_pred,
                "label": LABEL_NAMES[q_pred],
                "decision_score": q_score,
                "confidence": float(1.0 / (1.0 + np.exp(-q_score))),
            },
            "classical": {
                "prediction": c_pred,
                "label": LABEL_NAMES[c_pred],
                "probabilities": {"control": c_proba[0], "autism": c_proba[1]},
            },
            "agree": q_pred == c_pred,
            "encoding": {
                "pca_components": pcs,
                "qubit_angles": angles.tolist(),
                "angles_deg": (angles * 180.0 / np.pi).tolist(),
            },
            "quantum_state": {
                "basis_states": basis,
                "amplitudes_re": amps.real.tolist(),
                "amplitudes_im": amps.imag.tolist(),
                "probabilities": (np.abs(amps) ** 2).tolist(),
                "bloch_vectors": self._bloch_vectors(sv, self.n_qubits),
            },
            "kernel_row": kernel_row.tolist(),
            "roi_values": np.asarray(roi_vector).tolist(),
            "roi_names": self.feature_cols,
            "timings": timings,
        }

    def classify_subject(self, subject_id: int) -> dict:
        match = np.where(self.subject_ids == subject_id)[0]
        if len(match) == 0:
            raise KeyError(f"subject_id {subject_id} not in dataset")
        row = match[0]
        payload = self.classify_roi_vector(self.X_raw[row], source="subject")
        payload["subject"] = {
            "subject_id": int(subject_id),
            "true_dx_group": int(self.dx_group[row]),
            "true_label": LABEL_NAMES[int(self.dx_group[row] == 1)],
            "split": "test" if subject_id in self._test_sid_set else "train",
        }
        return payload

    def classify_nii(self, nii_path: str | Path) -> dict:
        from src.mri_features import extract_roi_vector

        t0 = time.perf_counter()
        feats, names = extract_roi_vector(nii_path)
        extract_ms = round((time.perf_counter() - t0) * 1000, 2)

        if np.isnan(feats).any():
            raise ValueError("ROI extraction produced NaNs — scan may be corrupt")

        payload = self.classify_roi_vector(feats.astype(np.float64), source="nii")
        payload["timings"]["roi_extraction_ms"] = extract_ms
        payload["scan_file"] = Path(nii_path).name
        return payload

    # ------------------------------------------------------------------
    def subjects_table(self) -> list[dict]:
        return [
            {
                "subject_id": int(sid),
                "dx_group": int(dx),
                "label": LABEL_NAMES[int(dx == 1)],
                "split": "test" if sid in self._test_sid_set else "train",
            }
            for sid, dx in zip(self.subject_ids, self.dx_group)
        ]

    def model_info(self) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "entanglement": self.entanglement,
            "reps": self.reps,
            "circuit_metrics": self.circuit_metrics,
            "pca_variance_pct": [round(v * 100, 2) for v in self.pca_variance],
            "pca_total_variance_pct": round(sum(self.pca_variance) * 100, 2),
            "n_train": len(self.X_train),
            "n_test": len(self.X_test),
            "n_rois": len(self.feature_cols),
            "models": [
                "Quantum Kernel SVM (ZZFeatureMap, full entanglement)",
                "Classical SVM (RBF, probability)",
            ],
        }
