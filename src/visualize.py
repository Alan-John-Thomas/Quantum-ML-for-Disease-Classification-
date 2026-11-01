"""
Visualization Suite for Quantum ML Disease Classification.
Generates publication-quality figures:
1. Circuit diagram of the 4-qubit ZZFeatureMap with two-qubit entanglement.
2. ROC-AUC comparison curves (Classical models vs Quantum Kernel models).
3. Confusion matrices for Clinical diagnosis (Classical RBF-SVM vs Quantum Kernel Full).
4. PCA Scree plot showing brain ROI variance explained across the 4 qubits.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

from src.data_loader import load_and_preprocess_data
from src.classical_baselines import get_classical_baselines
from src.quantum_kernel import FastQuantumKernelClassifier

def ensure_reports_dir(output_dir: str = "reports") -> str:
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def plot_circuit(output_path: str = "reports/circuit_diagram.png", n_qubits: int = 4) -> None:
    """
    Renders and saves the transpiled Qiskit ZZFeatureMap circuit diagram.
    """
    print(f"[Visualize] Generating circuit diagram for {n_qubits} qubits...")
    qsvm = FastQuantumKernelClassifier(n_qubits=n_qubits, reps=1, entanglement="full")
    circuit = qsvm.feature_map
    
    # Render using matplotlib engine
    fig = circuit.draw(output="mpl", style="iqp")
    fig.suptitle(f"Bio-Informed 4-Qubit ZZFeatureMap (Full Entanglement, 12 CNOTs)", fontsize=13, y=0.98)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualize] Saved circuit diagram -> {output_path}")

def plot_pca_variance(pca, output_path: str = "reports/pca_variance.png") -> None:
    """
    Plots the scree plot showing variance explained across the 4 quantum axes.
    """
    print("[Visualize] Generating PCA variance explained plot...")
    var_ratio = pca.explained_variance_ratio_ * 100
    cum_var = np.cumsum(var_ratio)
    components = [f"PC{i+1} (Q{i})" for i in range(len(var_ratio))]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    
    bars = ax1.bar(components, var_ratio, color="#1f77b4", alpha=0.85, label="Individual Variance (%)")
    ax1.set_ylabel("Explained Variance (%)", color="#1f77b4", fontsize=11)
    ax1.set_ylim(0, max(var_ratio) + 8)

    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.8, f"{yval:.1f}%", ha="center", va="bottom", fontsize=10)

    ax2 = ax1.twinx()
    ax2.plot(components, cum_var, color="#d62728", marker="o", linewidth=2.2, label="Cumulative Variance (%)")
    ax2.set_ylabel("Cumulative Variance (%)", color="#d62728", fontsize=11)
    ax2.set_ylim(0, 100)

    for i, txt in enumerate(cum_var):
        ax2.annotate(f"{txt:.1f}%", (components[i], cum_var[i] + 2), fontsize=10, ha="center", color="#d62728")

    plt.title(f"ABIDE I Brain MRI: 69 ROIs Mapped to 4 Qubits (Total: {cum_var[-1]:.1f}%)", fontsize=12, pad=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[Visualize] Saved PCA variance plot -> {output_path}")

def plot_roc_curves(models_dict: dict, X_test: np.ndarray, y_test: np.ndarray, output_path: str = "reports/roc_curves.png") -> None:
    """
    Plots ROC curves for all models on identical test data.
    """
    print("[Visualize] Generating comparative ROC curves...")
    plt.figure(figsize=(8, 6))

    colors = {
        "Logistic Regression": "#7f7f7f",
        "Classical SVM (Linear)": "#bcbd22",
        "Classical SVM (RBF)": "#2ca02c",
        "Random Forest": "#ff7f0e",
        "Quantum Kernel SVM (Linear)": "#17becf",
        "Quantum Kernel SVM (Full Entanglement)": "#9467bd"
    }

    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            continue

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        color = colors.get(name, "#333333")
        lw = 2.5 if "Quantum" in name else 1.5
        linestyle = "-" if "Full" in name else ("--" if "Linear" in name and "Quantum" in name else ":")
        
        plt.plot(fpr, tpr, color=color, lw=lw, linestyle=linestyle, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], color="navy", lw=1.2, linestyle="--", label="Random Chance (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.title("Comparative ROC Curves: Classical Baselines vs. Quantum Kernel SVM", fontsize=12, pad=12)
    plt.legend(loc="lower right", fontsize=9.5)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualize] Saved ROC curves -> {output_path}")

def plot_confusion_matrices(
    classical_model,
    quantum_model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_path: str = "reports/confusion_matrices.png"
) -> None:
    """
    Plots side-by-side confusion matrices for Classical RBF-SVM vs Quantum Kernel SVM (Full).
    """
    print("[Visualize] Generating side-by-side confusion matrices...")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    target_names = ["Control", "Autism"]

    # 1. Classical SVM (RBF)
    y_pred_class = classical_model.predict(X_test)
    cm_class = confusion_matrix(y_test, y_pred_class)
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_class, display_labels=target_names)
    disp1.plot(ax=axes[0], cmap=plt.cm.Blues, colorbar=False)
    axes[0].set_title(f"Classical SVM (RBF)\nAcc: {(cm_class.diagonal().sum()/len(y_test))*100:.1f}%", fontsize=11)

    # 2. Quantum Kernel SVM (Full)
    y_pred_quant = quantum_model.predict(X_test)
    cm_quant = confusion_matrix(y_test, y_pred_quant)
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_quant, display_labels=target_names)
    disp2.plot(ax=axes[1], cmap=plt.cm.Purples, colorbar=False)
    axes[1].set_title(f"Quantum Kernel SVM (Full Entanglement)\nAcc: {(cm_quant.diagonal().sum()/len(y_test))*100:.1f}%", fontsize=11)

    plt.suptitle("Clinical Diagnostic Separation on ABIDE I Structural MRI (N_test=251)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visualize] Saved confusion matrices -> {output_path}")

def generate_all_visualizations(parquet_path: str = "data/abide_roi_features.parquet", n_features: int = 4) -> None:
    """
    Orchestrates generation of all four reporting figures.
    """
    output_dir = ensure_reports_dir()
    
    # 1. Plot circuit diagram
    plot_circuit(os.path.join(output_dir, "circuit_diagram.png"), n_qubits=n_features)

    # 2. Ingest and preprocess data
    X_train, X_test, y_train, y_test, pca = load_and_preprocess_data(
        parquet_path=parquet_path,
        n_features=n_features,
        test_size=0.25,
        random_state=42
    )

    # 3. Plot PCA Scree plot
    plot_pca_variance(pca, os.path.join(output_dir, "pca_variance.png"))

    # 4. Fit models for ROC and Confusion matrices
    print("[Visualize] Fitting classical baselines...")
    classical_models = get_classical_baselines(random_state=42)
    for model in classical_models.values():
        model.fit(X_train, y_train)

    print("[Visualize] Fitting Quantum Kernel (Linear)...")
    qsvm_linear = FastQuantumKernelClassifier(n_qubits=n_features, reps=1, entanglement="linear", C=1.0)
    qsvm_linear.fit(X_train, y_train)

    print("[Visualize] Fitting Quantum Kernel (Full Entanglement)...")
    qsvm_full = FastQuantumKernelClassifier(n_qubits=n_features, reps=1, entanglement="full", C=1.0)
    qsvm_full.fit(X_train, y_train)

    all_models = {
        **classical_models,
        "Quantum Kernel SVM (Linear)": qsvm_linear,
        "Quantum Kernel SVM (Full Entanglement)": qsvm_full
    }

    # 5. Plot ROC Curves
    plot_roc_curves(all_models, X_test, y_test, os.path.join(output_dir, "roc_curves.png"))

    # 6. Plot Confusion Matrices
    plot_confusion_matrices(
        classical_models["Classical SVM (RBF)"],
        qsvm_full,
        X_test,
        y_test,
        os.path.join(output_dir, "confusion_matrices.png")
    )
    print("=" * 65)
    print(f"[Visualize] All figures successfully generated in '{output_dir}/'")
    print("=" * 65)

if __name__ == "__main__":
    generate_all_visualizations()
