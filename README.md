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
