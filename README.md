# Quantum Machine Learning for Disease Classification

Collaborative repository for the two-member team participating in **Quantum Hack Day Kochi | Qiskit Fall Fest 2026 (QFF26)** at TinkerSpace, Kochi.

**Challenge Track:** Challenge 02 · Quantum Machine Learning | Healthcare

---

## Shared Team Architecture

This repository maintains shared context, operational contracts, and agent skills to coordinate development across the team:

- **[`context/`](context/README.md):** Shared operational and technical single source of truth:
  - [`project_brief.md`](context/project_brief.md): Challenge background, scope, and mandatory deliverables.
  - [`team_contracts.md`](context/team_contracts.md): Member ownership matrix, data schemas, API contracts, and sync timeline.
  - [`technical_architecture.md`](context/technical_architecture.md): Mathematical formulations, feature map (`ZZFeatureMap`), quantum kernel, and VQC architecture.
  - [`evaluation_and_judging.md`](context/evaluation_and_judging.md): Judging rubric alignment and defense strategy.
  - [`roadmap_and_sprint_board.md`](context/roadmap_and_sprint_board.md): Sprint 1 and Sprint 2 task tracking.
- **[`skills/`](skills/README.md) & [`agent-skills/`](agent-skills/README.md):** Modular agent skills covering:
  - `clinical-data-prep-and-baseline`: Tabular data preprocessing, PCA ($d=4$), scaling to $[0, \pi]$, classical baselines.
  - `qml-circuit-designer`: Qiskit 1.x feature maps, ansatzes, transpilation, and gate metrics.
  - `quantum-kernel-estimator`: Fast statevector Gram matrix calculation and SVM integration.
  - `vqc-classifier`: Variational quantum classifier pipeline and optimizer setup.
  - `qml-benchmarking-and-metrics`: Multi-metric evaluation and markdown table formatting.
  - `presentation-and-demo-prep`: 5-minute pitch deck structure and live demo guide.
  - `team-sync-and-git-workflow`: Interface checks, branch rules, and pre-commit protocols.

---

## Team Division of Labor

- **Member 1 (Quantum Lead):** Quantum circuits, feature maps, kernel estimation, VQC, circuit metrics.
- **Member 2 (Classical ML Lead):** Clinical data ingestion, dimensionality reduction, classical baselines, benchmarking suite.

---

## Real-Time Classification Web App (`q1` branch)

A Flask backend + dashboard that classifies MRI scans live through the quantum kernel pipeline.

```bash
pip install -r requirements.txt
python app/server.py            # http://127.0.0.1:5000  (first run fits + persists models/)
python app/server.py --retrain  # force rebuild of models/pipeline.joblib
```

- **Input:** pick any of the 1,004 ABIDE subjects from the parquet, or drag-and-drop a skull-stripped `.nii` / `.nii.gz` scan (CPU ROI extraction via nibabel + `scipy.map_coordinates`, Harvard-Oxford atlas from the local nilearn cache — set `ATLAS_DIR` to override).
- **Output:** QSVM (full entanglement) prediction + classical RBF-SVM comparison, plus per-stage figures: MRI slices, ROI profile, PCA→qubit angles, 16-state statevector probabilities, per-qubit Bloch vectors, kernel-fidelity histogram, circuit diagram.
- **CSV exports:** `roi_features`, `pca_components`, `qubit_angles`, `statevector`, `kernel_similarities` — via dashboard buttons or `GET /api/export/<job_id>/<kind>.csv`.

| Endpoint | Description |
| --- | --- |
| `GET /api/info` | Circuit metrics, PCA variance, dataset summary |
| `GET /api/subjects` | All subjects with label + train/test split |
| `POST /api/classify` | `{"subject_id": N}` → full quantum payload + figures |
| `POST /api/classify/scan` | multipart `file=.nii[.gz]` → extraction + payload |
| `GET /api/export/<job>/<kind>.csv` | CSV download of encoding artefacts |
