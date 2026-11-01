---
name: vqc-classifier
description: Construct, train, and evaluate Variational Quantum Classifiers (VQC) with parameterized quantum circuits and classical optimizers in Qiskit.
---

# Variational Quantum Classifier (VQC) Skill

Use this skill when developing or debugging Variational Quantum Classifiers (VQC) using Qiskit 1.x and `qiskit_machine_learning` or native Qiskit primitives (`EstimatorV2` / `SamplerV2`).

## 1. Architecture

A VQC combines:
1. **Feature Map:** $U_\phi(x)$ embedding clinical features $x$.
2. **Variational Ansatz:** $W(\theta)$ parameterized by weights $\theta$.
3. **Measurement / Observable:** Expectation value $\langle Z_0 \rangle$ or parity measurement.
4. **Classical Optimizer:** Updates $\theta$ to minimize cross-entropy or mean squared error (e.g. COBYLA, SPSA).

## 2. Implementation Pattern (Qiskit 1.x)

```python
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_aer.primitives import SamplerV2

def build_vqc_model(n_qubits: int = 4, maxiter: int = 50, optimizer_name: str = "COBYLA"):
    feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=1, entanglement='linear')
    ansatz = RealAmplitudes(num_qubits=n_qubits, reps=1, entanglement='linear')
    
    if optimizer_name.upper() == "COBYLA":
        optimizer = COBYLA(maxiter=maxiter)
    elif optimizer_name.upper() == "SPSA":
        optimizer = SPSA(maxiter=maxiter)
    else:
        raise ValueError(f"Unknown optimizer {optimizer_name}")
        
    sampler = SamplerV2()
    
    vqc = VQC(
        sampler=sampler,
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer
    )
    return vqc
```

## 3. Practical Hackathon Guidelines
- **Barren Plateaus:** Keep ansatz shallow (`reps=1` or `reps=2`). Random deep circuits suffer from exponentially vanishing gradients.
- **Iterations:** Limit `maxiter` to 40-80 during demonstrations to keep runtime under 2 minutes while showing clear loss convergence.
- **Optimizer:** Use `COBYLA` for noiseless statevector simulations; switch to `SPSA` if shot noise is introduced.
