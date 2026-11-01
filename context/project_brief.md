# Project Brief: Quantum Machine Learning for Disease Classification

## 1. Hackathon Details
- **Event:** Quantum Hack Day Kochi | Qiskit Fall Fest 2026 (QFF26) × TinkerSpace
- **Track:** Challenge 02 · Quantum Machine Learning | Healthcare
- **Problem Statement:** Quantum Machine Learning for Disease Classification
- **Format:** In-person hackathon, 2-member team sprint
- **Date & Timeline:**
  - 10:00 AM – 01:30 PM: Hackathon Build Sprint I
  - 01:30 PM – 02:00 PM: Lunch & Midpoint Sync
  - 02:00 PM – 04:00 PM: Hackathon Build Sprint II
  - 04:15 PM – 04:30 PM: Project Submission & Demo Preparation
  - 04:30 PM – 05:15 PM: Project Showcase / Demos

---

## 2. Core Problem Formulation

### Clinical Motivation
Clinical diagnostic datasets (e.g., Wisconsin Diagnostic Breast Cancer, Cleveland Heart Disease) comprise multi-dimensional continuous biometric features (cell nuclei radius, perimeter, smoothness, concavity, texture). Classical machine learning models (Logistic Regression, Support Vector Machines, Random Forests) rely on hyperplanes or non-linear kernel tricks (e.g., RBF kernel) in Euclidean space.

### The Quantum Machine Learning Opportunity
Quantum Machine Learning (QML) offers alternate representational mechanisms:
1. **Hilbert Space Feature Mapping:** Non-linear data encoding into quantum states $|\phi(x)\rangle = U_\phi(x)|0\rangle^{\otimes n}$, mapping $n$-dimensional classical features into a $2^n$-dimensional complex Hilbert space.
2. **Entanglement-Induced Correlation:** Entangling gates (e.g., $CX$ pairs in `ZZFeatureMap`) encode pairwise feature interactions directly into quantum state amplitudes and relative phases:
   $$\phi_{\{i,j\}}(x) = 2(\pi - x_i)(\pi - x_j)$$
3. **Dual Exploration Paths:**
   - **Quantum Kernel Estimation (QSVM):** Estimating the transition amplitude (fidelity) between data-encoded states:
     $$K(x_i, x_j) = |\langle \phi(x_i) | \phi(x_j) \rangle|^2$$
     and feeding this precomputed Gram matrix into an optimal margin classifier.
   - **Variational Quantum Classifier (VQC):** Parameterized circuit ansatz $W(\theta)$ trained via classical optimization (COBYLA/SPSA) to minimize cross-entropy loss against measurement observables.

---

## 3. Scope, Realistic Bounds & Jury Stance

> [!IMPORTANT]
> **No Quantum Supremacy Claims:** NISQ simulators with $n \in [4, 6]$ qubits do NOT demonstrate quantum advantage over classical algorithms on tabular medical data. The hackathon jury explicitly evaluates:
> - Clear and mathematically sound problem mapping.
> - Purposeful circuit design (not arbitrarily dropping random gates).
> - Transparent, reproducible comparison against classical baselines.
> - Identification of trade-offs: circuit depth, simulation latency, expressivity, and classical baseline accuracy.

---

## 4. Mandatory Deliverables Checklist

- [ ] **Working Implementation:** Runnable Qiskit 1.x pipeline covering data preparation, classical baselines, quantum kernel estimation, and VQC.
- [ ] **Reproducible Setup:** Clear environment instructions (`qiskit >= 1.0`, `qiskit-aer`, `scikit-learn`, `numpy`).
- [ ] **Circuit Metrics:** Reporting total qubits, circuit depth, 2-qubit CNOT count, and gate breakdown.
- [ ] **Comparative Evaluation Table:** Side-by-side performance of Classical Logistic Regression, Classical SVM (RBF), Quantum Kernel SVM, and VQC on standard test metrics:
  - Accuracy
  - Precision, Recall, F1-Score
  - ROC-AUC
  - Training / Evaluation Latency
- [ ] **Visualizations:**
  - Feature correlation & PCA variance explained plots.
  - Quantum circuit diagrams (transpiled).
  - Confusion matrices for all models.
  - Decision boundary comparison on 2D projections.
- [ ] **Structured Demo & Presentation:** 5-minute concise presentation deck and live notebook/CLI execution.
