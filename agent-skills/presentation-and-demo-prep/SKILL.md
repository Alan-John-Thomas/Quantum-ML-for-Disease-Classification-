---
name: presentation-and-demo-prep
description: Prepare structured 5-minute hackathon pitch decks, live demonstration scripts, jury Q&A defense, and submission artifacts.
---

# Presentation & Live Demo Preparation Skill

Use this skill when structuring the 5-minute presentation deck, writing the demo execution script, and preparing team speaking points for the evaluation panel.

## 1. 5-Minute Presentation Structure (Strict Time Budget)

| Slide / Section | Time Budget | Key Talking Points |
| :--- | :---: | :--- |
| **1. Problem & Clinical Motivation** | 45 sec | Medical diagnosis challenge (WDBC), high-dimensional pathology features, non-linear separability. |
| **2. Proposed Solution Architecture** | 45 sec | Hybrid classical-quantum workflow: Standardization $\to$ PCA $\to$ $[0, \pi]$ scaling $\to$ Quantum Hilbert space embedding. |
| **3. Quantum Component Implementation** | 90 sec | - Feature Map: `ZZFeatureMap` with entangling $CX$ gates capturing pairwise feature correlations.<br>- Model A: Quantum Kernel Support Vector Machine.<br>- Model B: Variational Quantum Classifier with `RealAmplitudes` ansatz.<br>- Transpilation profile (depth, gate counts). |
| **4. Benchmark Results & Baseline Comparison** | 60 sec | Transparent comparison against Logistic Regression, Classical SVM (RBF), and Random Forest. Focus on fidelity, expressivity, and trade-offs. |
| **5. Limitations, NISQ Realities & Future Work** | 45 sec | Honest discussion on simulation scaling, barren plateaus, error mitigation on physical QPUs, next steps. |
| **6. Live Demo / Code Walkthrough** | 15 sec | Show running notebook output, confusion matrix, and circuit transpiled visualization. |

## 2. Live Demo Runbook

1. **Pre-Demo Verification:**
   - Pre-run all notebook cells or CLI scripts before standing before judges to avoid network/kernel stalls.
   - Keep a clean cached output window open.
2. **Demo Sequence:**
   - Execute the end-to-end evaluation script:
     ```bash
     python -m src.evaluate --dataset breast_cancer --n_qubits 4
     ```
   - Point out the terminal output:
     - Dataset preprocessed shape and scaling check.
     - Circuit transpiled summary (qubits: 4, depth: 12, CNOTs: 8).
     - Execution time of quantum kernel Gram matrix vs classical models.
     - Final accuracy / F1 score table.
3. **Display Visual Artifacts:**
   - Transpiled quantum circuit diagram.
   - Side-by-side Confusion Matrices.
   - Decision boundary plot in 2D PCA space showing how the quantum kernel separates non-linear clusters.

## 3. Defense Against Tough Jury Questions

- *Question:* "Why not use 30 qubits?"
  - *Answer:* "Exponential simulation complexity ($2^{30}$ amplitudes) is infeasible for local verification. 4 PCA components capture over 75% of variance, balancing diagnostic fidelity with exact statevector simulation speed."
- *Question:* "Is there quantum advantage?"
  - *Answer:* "No. As per track guidelines, the goal is investigating how quantum circuits can be incorporated into ML pipelines. On classical tabular data of this scale, classical algorithms are optimal in wall-clock time; our focus is quantum kernel expressivity and circuit depth optimization."
