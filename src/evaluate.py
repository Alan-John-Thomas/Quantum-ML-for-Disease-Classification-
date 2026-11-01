"""
Unified Evaluation and Benchmarking Suite.
Trains and compares classical baselines vs. Quantum Kernel SVM on disease classification.
"""

import time
import argparse
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from src.data_loader import load_and_preprocess_data
from src.classical_baselines import get_classical_baselines
from src.quantum_kernel import FastQuantumKernelClassifier
from src.vqc_classifier import VQCClassifier

def evaluate_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    circuit_metrics: dict | None = None
) -> dict:
    """
    Fits and evaluates a single model, recording latency and classification metrics.
    """
    # 1. Fit latency
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    # 2. Inference latency
    t1 = time.perf_counter()
    y_pred = model.predict(X_test)
    inf_time = (time.perf_counter() - t1) / len(X_test)

    # 3. Decision function / Probability for ROC-AUC
    roc_auc = None
    try:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            roc_auc = float(roc_auc_score(y_test, y_prob))
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
            roc_auc = float(roc_auc_score(y_test, y_score))
    except Exception:
        roc_auc = None

    return {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "train_time_sec": round(train_time, 4),
        "inf_latency_ms": round(inf_time * 1000, 2),
        "qubits": circuit_metrics.get("num_qubits") if circuit_metrics else None,
        "circuit_depth": circuit_metrics.get("depth") if circuit_metrics else None,
        "cnot_count": circuit_metrics.get("cx_count") if circuit_metrics else None,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

def format_benchmark_table(results: list[dict]) -> str:
    headers = [
        "Model", "Accuracy", "Balanced Acc", "F1-Score", "ROC-AUC",
        "Train Time (s)", "Qubits", "Circuit Depth", "CNOTs"
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"
    ]
    for r in results:
        auc_str = f"{r['roc_auc']:.3f}" if r['roc_auc'] is not None else "N/A"
        q_str = str(r['qubits']) if r['qubits'] is not None else "-"
        d_str = str(r['circuit_depth']) if r['circuit_depth'] is not None else "-"
        cx_str = str(r['cnot_count']) if r['cnot_count'] is not None else "-"
        row = [
            r['model_name'],
            f"{r['accuracy']:.4f}",
            f"{r['balanced_acc']:.4f}",
            f"{r['f1_score']:.4f}",
            auc_str,
            f"{r['train_time_sec']:.2f}",
            q_str,
            d_str,
            cx_str
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)

def run_benchmarks(
    parquet_path: str = "data/abide_roi_features.parquet",
    n_features: int = 4,
    test_size: float = 0.25,
    random_state: int = 42
) -> list[dict]:
    print("=" * 65)
    print("QUANTUM ML FOR DISEASE CLASSIFICATION: BENCHMARK SUITE")
    print("=" * 65)
    
    # Ingest data
    X_train, X_test, y_train, y_test, pca = load_and_preprocess_data(
        parquet_path=parquet_path,
        n_features=n_features,
        test_size=test_size,
        random_state=random_state
    )
    print(f"Dataset: N_train={len(X_train)}, N_test={len(X_test)}, Qubits={n_features}")
    print(f"PCA Variance Explained: {np.round(pca.explained_variance_ratio_ * 100, 1)}% (Total: {np.sum(pca.explained_variance_ratio_)*100:.1f}%)")
    print("-" * 65)

    results = []

    # 1. Classical Baselines
    classical_models = get_classical_baselines(random_state=random_state)
    for name, model in classical_models.items():
        print(f"Training {name}...")
        res = evaluate_model(model, X_train, y_train, X_test, y_test, name)
        results.append(res)

    # 2. Quantum Kernel SVM (Linear Entanglement)
    print("Training Quantum Kernel SVM (ZZFeatureMap, linear)...")
    qsvm_linear = FastQuantumKernelClassifier(
        n_qubits=n_features, reps=1, entanglement="linear", C=1.0
    )
    circ_metrics_linear = qsvm_linear.get_circuit_metrics()
    res_linear = evaluate_model(
        qsvm_linear, X_train, y_train, X_test, y_test,
        "Quantum Kernel SVM (Linear)",
        circuit_metrics=circ_metrics_linear
    )
    results.append(res_linear)

    # 3. Quantum Kernel SVM (Full Entanglement)
    print("Training Quantum Kernel SVM (ZZFeatureMap, full)...")
    qsvm_full = FastQuantumKernelClassifier(
        n_qubits=n_features, reps=1, entanglement="full", C=1.0
    )
    circ_metrics_full = qsvm_full.get_circuit_metrics()
    res_full = evaluate_model(
        qsvm_full, X_train, y_train, X_test, y_test,
        "Quantum Kernel SVM (Full Entanglement)",
        circuit_metrics=circ_metrics_full
    )
    results.append(res_full)

    # 4. Variational Quantum Classifier (VQC)
    print("Training Variational Quantum Classifier (RealAmplitudes + COBYLA)...")
    vqc = VQCClassifier(
        n_qubits=n_features, reps=1, entanglement="linear", maxiter=25, subsample_train=100, random_state=random_state
    )
    circ_metrics_vqc = vqc.get_circuit_metrics()
    res_vqc = evaluate_model(
        vqc, X_train, y_train, X_test, y_test,
        "Variational Quantum Classifier (VQC)",
        circuit_metrics=circ_metrics_vqc
    )
    results.append(res_vqc)

    print("\n" + "=" * 65)
    print("FINAL COMPARATIVE BENCHMARK TABLE")
    print("=" * 65)
    table_md = format_benchmark_table(results)
    print(table_md)
    print("=" * 65)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_path", type=str, default="data/abide_roi_features.parquet")
    parser.add_argument("--n_features", type=int, default=4)
    args = parser.parse_args()
    run_benchmarks(parquet_path=args.parquet_path, n_features=args.n_features)
