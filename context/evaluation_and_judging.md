# Evaluation Criteria & Jury Defense Strategy

This document maps the project deliverables directly to the hackathon judging rubric and prepares structured technical defenses against common evaluator critiques.

---

## 1. Hackathon Judging Rubric Breakdown

### Phase 1: Idea and Feasibility (Preliminary Review)
| Criterion | Weight | How Our Implementation Addresses It |
| :--- | :---: | :--- |
| **Problem Understanding & Relevance** | 20% | Clinical diagnosis on high-dimensional biopsy indicators (WDBC). Clear explanation of why non-linear boundary discovery matters in medical classification. |
| **Innovation & Creativity** | 25% | Dual QML exploration: Quantum Kernel Hilbert space mapping vs. Variational Quantum Classifier (VQC), highlighting structural differences in how quantum states represent clinical correlation. |
| **Feasibility within Timeline** | 20% | 4-qubit PCA reduction + Statevector-accelerated kernel Gram matrix computation guarantees end-to-end execution in < 15 seconds, preventing timeout failures during live evaluation. |
| **Quantum Component Planning** | 25% | Rigorous mathematical mapping of clinical features to relative quantum phases and two-qubit entanglement via second-order Pauli expansion (`ZZFeatureMap`). |
| **Clarity of Presentation** | 10% | Structured repository, clean circuit diagrams, and concise comparative tables. |

---

### Phase 2: Prototype and Implementation (Final Evaluation)
| Criterion | Weight | How Our Implementation Addresses It |
| :--- | :---: | :--- |
| **MVP Functionality & Completion** | 25% | Fully runnable pipeline: data ingestion $\to$ preprocessing $\to$ classical baselines $\to$ Q-Kernel $\to$ VQC $\to$ evaluation report. |
| **Quantum Component Implementation** | 25% | Native Qiskit 1.x implementation with parameterized circuits, transpilation depth and CNOT gate accounting. |
| **Technical Implementation** | 20% | Clean modular separation (data loader, classical models, quantum models, evaluation suite), strict typing, and reproducible random states. |
| **Real-World Impact & Feasibility** | 15% | Honest analysis of NISQ hardware limits, simulation bottlenecks, and clear identification of what parts of the pipeline are quantum vs classical. |
| **Demo and Presentation** | 15% | Live demonstration executing in under 2 minutes, showing loss convergence, confusion matrices, and ROC curves. |

---

## 2. Anticipated Jury Questions & Technical Rebuttals

### Question 1: "Why use a quantum circuit when classical Random Forest or Logistic Regression already achieves > 92% accuracy on this dataset?"
- **Rebuttal:**
  - "We do not claim quantum supremacy on small tabular datasets. The hackathon's explicit problem statement is to investigate how quantum circuits can be integrated into clinical classification workflows."
  - "The primary value here is testing whether quantum kernels can map pairwise feature interactions into Hilbert spaces with different inductive biases than classical RBF kernels."
  - "We provide the classical baseline comparison directly in our benchmark table so judges can transparently see the exact delta in accuracy, F1, and execution latency."

### Question 2: "Is your quantum circuit doing anything meaningful, or did you just append a random circuit?"
- **Rebuttal:**
  - "Our quantum component uses a second-order Pauli-Z feature map (`ZZFeatureMap`). The two-qubit entangling gates ($CX - R_z - CX$) explicitly encode the cross-product terms of clinical features: $\phi_{\{i,j\}}(x) = 2(\pi - x_i)(\pi - x_j)$."
  - "This creates non-classical correlation between features that cannot be factored into independent single-qubit rotations."
  - "We also measure and report circuit depth (transpiled to basis gates) and CNOT count, proving the circuit is parameterized by the data and ansatz weights."

### Question 3: "Why did you use 4 qubits instead of using all 30 features from the dataset?"
- **Rebuttal:**
  - "Simulating 30 qubits requires computing statevectors of dimension $2^{30} \approx 10^9$ complex numbers, which requires gigabytes of RAM and hours of runtime, impossible for interactive hackathon evaluations."
  - "By using PCA to capture the top 4 principal components (~75-80% of total variance), we preserve the primary diagnostic signal while keeping the quantum state space at $2^4 = 16$ dimensions, allowing fast exact statevector simulation and rapid parameter optimization."

### Question 4: "Did your VQC experience barren plateaus during training?"
- **Rebuttal:**
  - "To avoid barren plateaus (exponentially vanishing gradients in deep parameterized circuits), we restricted the ansatz to a shallow 2-layer `RealAmplitudes` circuit with linear entanglement."
  - "Furthermore, we used the `COBYLA` optimizer with gradient-free direct search, which is resilient against noisy gradient estimates in shallow NISQ circuits."
