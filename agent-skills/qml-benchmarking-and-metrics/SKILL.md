---
name: qml-benchmarking-and-metrics
description: Evaluate and benchmark quantum and classical disease classifiers on accuracy, precision, recall, F1, ROC-AUC, latency, and generate comparison tables and confusion matrices.
---

# QML Benchmarking & Evaluation Skill

Use this skill when implementing the unified evaluation harness, profiling execution latency, generating comparative benchmark tables for hackathon judges, or rendering diagnostic visualizations.

## 1. Benchmarking Metrics Requirements

Hackathon judges demand multi-faceted evaluation beyond raw accuracy:
1. **Classification Quality:** Accuracy, Precision, Recall, F1-Score, and ROC-AUC (using prediction probabilities or decision function values).
2. **Computational Profiling:** Training duration (seconds) and inference latency per sample (milliseconds).
3. **Quantum Complexity Profile:** Transpiled circuit depth, total gates, and entangling two-qubit ($CX$) gate count.

## 2. Standard Evaluation Harness Pattern

```python
import time
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

def evaluate_classifier(model, X_train, y_train, X_test, y_test, model_name: str, circuit_metrics: dict | None = None) -> dict:
    """
    Fits and evaluates a model, returning standard metrics dictionary.
    """
    # Training Latency
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    # Inference Latency
    t1 = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_time = (time.perf_counter() - t1) / len(X_test)

    # Probability / Decision function for ROC-AUC
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

    metrics = {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "train_time_sec": round(train_time, 4),
        "inference_time_ms_per_sample": round(inference_time * 1000, 2),
        "qubits": circuit_metrics.get("num_qubits") if circuit_metrics else None,
        "circuit_depth": circuit_metrics.get("depth") if circuit_metrics else None,
        "cnot_count": circuit_metrics.get("cx_count") if circuit_metrics else None,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }
    return metrics
```

## 3. Markdown Comparison Table Formatter

Format model results directly into a markdown table suitable for project reports and presentation slides:

```python
def format_benchmark_table(results_list: list[dict]) -> str:
    headers = ["Model", "Accuracy", "F1-Score", "ROC-AUC", "Train Time (s)", "Qubits", "Circuit Depth", "CNOTs"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in results_list:
        auc_str = f"{r['roc_auc']:.3f}" if r['roc_auc'] is not None else "N/A"
        qubits_str = str(r['qubits']) if r['qubits'] is not None else "-"
        depth_str = str(r['circuit_depth']) if r['circuit_depth'] is not None else "-"
        cnot_str = str(r['cnot_count']) if r['cnot_count'] is not None else "-"
        
        row = [
            r['model_name'],
            f"{r['accuracy']:.4f}",
            f"{r['f1_score']:.4f}",
            auc_str,
            f"{r['train_time_sec']:.2f}",
            qubits_str,
            depth_str,
            cnot_str
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
```
