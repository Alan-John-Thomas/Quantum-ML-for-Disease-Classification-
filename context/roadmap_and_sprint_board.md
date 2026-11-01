# Hackathon Roadmap & Sprint Task Board

**Project:** Quantum Machine Learning for Autism Spectrum Disorder (ASD) Classification  
**Dataset:** ABIDE I Structural Brain MRI (504 Typical Control + 500 Autism Scans)  
**Architecture:** Decoupled Neuroimaging Feature Extraction $\to$ `abide_roi_features.parquet` $\to$ QML & Classical Baselines  
**Timeline:** 10:00 AM – 04:15 PM (~4.5 hours remaining)

---

## 1. Operating Protocol & Division of Labor

```mermaid
flowchart TD
    subgraph Jeswin_Tasks ["Member 2 (Jeswin) — Neuroimaging Feature Extraction"]
        J1["1. Access Local Scans (D:\\study\\autisim\\)"] --> J2["2. Run nilearn Resampling & Harvard-Oxford Atlas Masking"]
        J2 --> J3["3. Extract Regional Mean Intensities (50–100 ROIs)"]
        J3 --> J4["4. Export & Push data/abide_roi_features.parquet to dev"]
        J4 --> J5["5. Generate 1–2 Brain Slice Visualizations for Demo Slides"]
    end

    subgraph Alan_Tasks ["Member 1 (Alan) — Quantum Machine Learning & Baselines Engine"]
        A1["1. Implement Parquet Data Loader Adapter (with Mock Fallback)"] --> A2["2. Standardize & PCA Reduce (ROIs -> 4-6 components) -> [0, pi] Scaling"]
        A2 --> A3["3. Train Classical Baselines (Logistic Regression, SVM-RBF, Random Forest)"]
        A2 --> A4["4. Design Qiskit 1.x/2.x Feature Maps (ZZFeatureMap, Transpilation Metrics)"]
        A4 --> A5["5. Train Fast Quantum Kernel SVM (Statevector Gram Matrix)"]
        A4 --> A6["6. Train Variational Quantum Classifier (RealAmplitudes + COBYLA)"]
        A3 --> A7["7. Unified Evaluation Suite (Acc, F1, ROC-AUC, Latency Table)"]
        A5 --> A7
        A6 --> A7
    end

    J4 -.-> |Git Pull on dev| A2
```

---

## 2. Live Task Board

### Sprint 1: Data Handoff & Baseline Pipeline (10:00 AM – 01:30 PM)

| Task ID | Workstream | Owner | Specific Deliverables | Status |
| :--- | :--- | :--- | :--- | :---: |
| **S1-01** | Git Setup | Shared | Clean `main` & `dev` branch tracking, context docs | **Completed** |
| **S1-02** | MRI ROI Extraction | **Jeswin** | Run `nilearn` on `D:\study\autisim\`, extract Harvard-Oxford atlas ROI means across 1,004 subjects | **IN PROGRESS** |
| **S1-03** | Parquet Handoff Commit | **Jeswin** | Commit `data/abide_roi_features.parquet` (~500 KB) to `origin/dev` | **BLOCKED ON S1-02** |
| **S1-04** | Modular Data Loader | **Alan** | Build `src/data_loader.py` for `abide_roi_features.parquet` with PCA ($d=4, 6$) and $[0, \pi]$ scaling | **Ready to Build** |
| **S1-05** | Classical Baseline Harness | **Alan** | Benchmark Logistic Regression, Linear SVM, RBF SVM, Random Forest | **Ready to Build** |
| **S1-06** | Feature Map Circuit Design | **Alan** | Transpile `ZFeatureMap` & `ZZFeatureMap` (linear/full), log depth & CNOT counts | **Ready to Build** |

---

### Sprint 2: Quantum Models & Comparative Evaluation (02:00 PM – 04:00 PM)

| Task ID | Workstream | Owner | Specific Deliverables | Status |
| :--- | :--- | :--- | :--- | :---: |
| **S2-01** | Fast Quantum Kernel Estimator | **Alan** | Compute statevector Gram matrix fidelity $K_{ij} = \|\langle \phi_i\|\phi_j\rangle\|^2$, train precomputed SVM | Pending S1 |
| **S2-02** | VQC Implementation | **Alan** | `ZZFeatureMap` + `RealAmplitudes` + `COBYLA` optimizer on Qiskit Aer | Pending S1 |
| **S2-03** | QPU Noise Simulation | **Alan** | Simulate circuit on `AerSimulator` with fake IBM backend noise model | Pending S2-01 |
| **S2-04** | Diagnostic Visualizations | **Jeswin** | Confusion matrices, ROC curves, brain slice overlay plot | Pending S1-03 |
| **S2-05** | Comparative Metrics Table | Shared | Side-by-side benchmark table (Accuracy, Balanced Acc, F1, ROC-AUC, Latency) | Pending S2-02 |

---

### Final Submission & Demo (04:15 PM – 05:15 PM)

| Task ID | Workstream | Owner | Specific Deliverables | Status |
| :--- | :--- | :--- | :--- | :---: |
| **FIN-01** | Repository README | Shared | Setup instructions, Qiskit usage details, reproducibility steps | Pending Sprint 2 |
| **FIN-02** | Presentation Deck | Shared | 5-minute slide deck (Problem, Architecture, Circuits, Benchmarks, Limitations) | Pending Sprint 2 |
| **FIN-03** | Live Demo Script | Shared | Test 60-second end-to-end evaluation runbook for evaluation panel | Pending Sprint 2 |
