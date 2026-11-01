# Hackathon Challenge Brief — Challenge 02 · Quantum Machine Learning | Healthcare

> Source: pasted by Jeswin from the Qiskit Hack Day challenge description.

## Quantum Machine Learning for Disease Classification

### The problem

Medical datasets often contain multiple features that can help distinguish between
healthy and diseased patients. Machine learning can use these features to support
classification and pattern recognition.

### The challenge

Build a hybrid quantum-classical machine learning model that classifies patients
using an appropriate dataset. The goal is to explore how quantum circuits can be
incorporated into a machine learning pipeline, **not** to prove that quantum
machine learning is inherently superior to classical machine learning.

### Possible approaches

- Variational Quantum Classifiers (VQC)
- Quantum Support Vector Machines (QSVM)
- Quantum kernels
- Hybrid neural networks incorporating quantum circuits

### What to demonstrate

- Dataset selection and preprocessing
- Feature encoding and quantum circuit design
- Integration of the quantum model into the classification pipeline
- Evaluation using appropriate metrics
- Comparison with a classical baseline, where feasible

**Important:** Use publicly available or otherwise appropriately authorized data.
Do not include personally identifiable patient information in the project or demo.

---

## Requirements

### 1. Problem statement

Explain the real-world problem the project aims to solve (~150–300 words):

- What is the problem, and why is it important?
- Who is affected by it?
- What are we trying to optimize, predict, classify, or simulate?

### 2. Proposed solution

Describe the solution in simple, understandable terms:

- Overall approach
- Workflow: Input → Processing → Output
- How the system addresses the problem
- Why this approach was chosen
- A simple architecture/workflow diagram is highly recommended

### 3. Quantum component (MANDATORY)

Must clearly explain:

- Quantum technology used (QAOA, VQE, VQC, QSVM, quantum kernels, QNN,
  quantum simulation, or another Qiskit-supported approach)
- Why it was chosen and why it suits the problem
- Problem mapping: how the real-world problem is represented as a quantum problem
- Quantum circuit or algorithm: what it does and how it contributes
- Role of quantum computing: which part of the system uses it, what it achieves

> Simply adding a Hadamard gate or a random quantum circuit does NOT qualify.
> The quantum component must have a clear purpose.

### 4. Implementation and reproducibility

- Source code (.py, Jupyter notebooks, or both)
- README with setup and execution instructions
- Dependencies and installation instructions
- Dataset, or instructions for accessing it
- Configuration details where applicable
- Code must be organized, documented, and runnable per the README

### 5. Qiskit usage

- Qiskit libraries used
- Quantum circuits created
- Quantum algorithms implemented
- Backend or simulator used
- If hybrid: explain how classical and quantum components interact

---

## Submission checklist

- [ ] All project and team details provided
- [ ] Problem and proposed solution explained
- [ ] Mandatory quantum component documented
- [ ] Complete source code and necessary files
- [ ] Setup instructions and dependencies
- [ ] Qiskit implementation and backend/simulator documented
- [ ] Another person can reproduce the project

Recommended additions: architecture diagram, screenshots, short demo video,
classical baseline comparison, future improvements.

## What makes a strong submission (per organizers)

1. Address the chosen problem clearly
2. Demonstrate a working minimum viable prototype
3. Integrate quantum computing meaningfully
4. Explain technical implementation and design choices
5. Present results and limitations honestly
6. Be reproducible and easy to evaluate

> Teams are NOT expected to prove quantum advantage. A well-designed, honest,
> working prototype beats an ambitious idea without a working implementation.

---

## Judging criteria

### Phase 1: Idea and feasibility

| Criterion | Weight |
|---|---|
| Problem understanding and SDG relevance | 20% |
| Innovation and creativity | 25% |
| Feasibility within hackathon timeline | 20% |
| Quantum component planning | 25% |
| Clarity of presentation | 10% |

### Phase 2: Prototype and implementation

| Criterion | Weight |
|---|---|
| MVP functionality and completion | 25% |
| Quantum component implementation | 25% |
| Technical implementation | 20% |
| Innovation and real-world impact | 15% |
| Demo and presentation | 15% |
