# Shared Team Context: Quantum ML for Disease Classification

This directory acts as the centralized source of truth for the two-member team participating in **Quantum Hack Day Kochi | QFF26** (Challenge 02: Quantum Machine Learning for Disease Classification).

Both human team members and AI coding agents MUST consult these documents before proposing architectural changes, writing code, or running benchmarks.

---

## Directory Index

| Document | Purpose | Primary Audience |
| :--- | :--- | :--- |
| [`00-challenge-brief.md`](00-challenge-brief.md) | Official challenge text, requirements, and full judging criteria | Both Members |
| [`01-idea-analysis.md`](01-idea-analysis.md) | Initial 6-candidate comparative analysis and research notes | Both Members |
| [`02-dataset-notes.md`](02-dataset-notes.md) | Local MRI dataset audit, NIfTI headers, and phenotypic notes | Jeswin / Devin |
| [`03-mri-technical-deepdive.md`](03-mri-technical-deepdive.md) | Neuroimaging compatibility matrix, resampling traps, and risk register | Jeswin / Devin |
| [`04-dataset-evaluation-and-selection-report.md`](04-dataset-evaluation-and-selection-report.md) | **Adversarial audit, machine constraint check & final dataset decision** | **Both Members** |
| [`project_brief.md`](project_brief.md) | Problem formulation, hackathon scope, deliverables, and judging rules | Both Members |
| [`team_contracts.md`](team_contracts.md) | Division of responsibilities, data schemas, API interfaces, and handoff protocols | Member 1 & Member 2 |
| [`technical_architecture.md`](technical_architecture.md) | Mathematical formulation, circuit topology, classical baselines, and Qiskit 1.x pipeline | Both Members |
| [`evaluation_and_judging.md`](evaluation_and_judging.md) | Judging rubric alignment, metrics requirements, and defense against jury scrutiny | Both Members |
| [`roadmap_and_sprint_board.md`](roadmap_and_sprint_board.md) | Sprint 1 & 2 task breakdown, milestones, and shared task tracking | Both Members |

---

## Operating Protocol for Agents and Collaborators

1. **Strict Separation of Concerns:**
   - **Member 1 (Quantum Track):** Owns quantum circuit design, feature encoding (`ZZFeatureMap`), ansatz optimization, and quantum kernel computation.
   - **Member 2 (Classical ML & Evaluation Track):** Owns data ingestion, PCA dimensionality reduction, classical baseline models (Logistic Regression, SVM RBF, Random Forest), and the benchmarking harness.
2. **Deterministic Data Contract:**
   - All models consume normalized feature vectors of dimension $N \in \{4, 6\}$ scaled strictly to $[0, \pi]$ using `random_state=42`.
3. **No Unsubstantiated Claims:**
   - Maintain objective technical evaluation. Do not claim quantum supremacy or advantage on NISQ-era simulated circuits. Demonstrate clear feature encoding, entanglement verification, and comparative metrics against classical baselines.
