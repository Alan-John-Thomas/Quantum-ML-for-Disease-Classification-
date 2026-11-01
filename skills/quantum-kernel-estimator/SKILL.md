---
name: quantum-kernel-estimator
description: Implement and optimize Quantum Kernel estimation (Fidelity Gram matrix) using Qiskit 1.x and integrate with scikit-learn Support Vector Machines.
---

# Quantum Kernel Estimator Skill

Use this skill to construct quantum kernel matrices $K(x_i, x_j) = |\langle \phi(x_i) | \phi(x_j) \rangle|^2$ for clinical classification, interfacing directly with `sklearn.svm.SVC(kernel='precomputed')`.

## 1. Mathematical Formulation

Given a quantum feature map $U_\phi(x)$, the quantum state is $|\phi(x)\rangle = U_\phi(x)|0\rangle^{\otimes n}$. The quantum fidelity kernel is:
$$k(x_i, x_j) = |\langle 0^{\otimes n} | U_\phi^\dagger(x_j) U_\phi(x_i) | 0^{\otimes n} \rangle|^2$$

This kernel maps clinical features into an exponentially large Hilbert space where linear separation may be achieved.

## 2. Fast Exact Computation (Statevector Simulation)

During hackathon sprints, shot-based sampling for $N \times N$ Gram matrices can be computationally heavy. Use `Statevector` computation for rapid, noiseless training and prototyping:

```python
import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin

class FastQuantumKernelClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_qubits: int = 4, reps: int = 2, C: float = 1.0):
        self.n_qubits = n_qubits
        self.reps = reps
        self.C = C
        self.feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=reps, entanglement='linear')
        self.svc = SVC(kernel='precomputed', C=self.C)
        self.X_train_ = None
        self.train_states_ = None

    def _compute_statevectors(self, X: np.ndarray) -> list[Statevector]:
        states = []
        for row in X:
            bound_circuit = self.feature_map.assign_parameters(row)
            states.append(Statevector.from_instruction(bound_circuit))
        return states

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train_ = X
        self.train_states_ = self._compute_statevectors(X)
        n_samples = len(X)
        
        # Compute symmetric training Gram matrix
        K_train = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            K_train[i, i] = 1.0
            for j in range(i + 1, n_samples):
                # Fidelity: |<psi_i | psi_j>|^2
                fidelity = abs(self.train_states_[i].inner(self.train_states_[j])) ** 2
                K_train[i, j] = fidelity
                K_train[j, i] = fidelity
                
        self.svc.fit(K_train, y)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        test_states = self._compute_statevectors(X_test)
        n_test = len(X_test)
        n_train = len(self.X_train_)
        
        K_test = np.zeros((n_test, n_train))
        for i in range(n_test):
            for j in range(n_train):
                fidelity = abs(test_states[i].inner(self.train_states_[j])) ** 2
                K_test[i, j] = fidelity
                
        return self.svc.predict(K_test)
```

## 3. Best Practices & Pitfalls
- **Input Scaling:** Features must be scaled to $[0, \pi]$ or $[-\pi, \pi]$ using `MinMaxScaler`. Unscaled features cause periodic phase wrapping, ruining kernel separation.
- **Complexity Management:** If the dataset has 500 samples, calculating a $500 \times 500$ matrix requires 125,000 inner products. For fast feedback, subsample $N_{train} = 150$, $N_{test} = 50$, or compute on 4-6 qubits.
