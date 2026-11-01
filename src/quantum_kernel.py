"""
Fast Statevector Quantum Kernel Estimator using Qiskit 2.x.
Computes the fidelity Gram matrix K(x_i, x_j) = |<phi(x_i)|phi(x_j)>|^2
and feeds it into scikit-learn Support Vector Classifier with kernel='precomputed'.
"""

import time
import numpy as np
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector
from qiskit import transpile
from qiskit_aer import AerSimulator
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin

class FastQuantumKernelClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_qubits: int = 4,
        reps: int = 1,
        entanglement: str = "linear",
        C: float = 1.0
    ):
        self.n_qubits = n_qubits
        self.reps = reps
        self.entanglement = entanglement
        self.C = C
        
        # Build parameterized Qiskit feature map
        self.feature_map = zz_feature_map(
            feature_dimension=self.n_qubits,
            reps=self.reps,
            entanglement=self.entanglement
        )
        self.svc = SVC(kernel="precomputed", C=self.C)
        self.X_train_ = None
        self.train_states_ = None
        
    def get_circuit_metrics(self) -> dict:
        """
        Transpiles feature map to AerSimulator basis gates to extract complexity metrics.
        """
        sim = AerSimulator()
        transpiled = transpile(self.feature_map, backend=sim, optimization_level=1)
        ops = dict(transpiled.count_ops())
        return {
            "num_qubits": self.n_qubits,
            "depth": transpiled.depth(),
            "gate_count": ops,
            "cx_count": ops.get("cx", 0)
        }

    def _compute_statevectors(self, X: np.ndarray) -> list[Statevector]:
        """
        Maps classical feature rows into 2^n complex statevectors.
        """
        states = []
        for row in X:
            bound_circuit = self.feature_map.assign_parameters(row)
            states.append(Statevector.from_instruction(bound_circuit))
        return states

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FastQuantumKernelClassifier":
        """
        Computes symmetric training Gram matrix and fits precomputed SVC.
        """
        self.X_train_ = np.asarray(X)
        self.train_states_ = self._compute_statevectors(self.X_train_)
        n_samples = len(self.X_train_)

        # Symmetric Gram matrix computation
        K_train = np.zeros((n_samples, n_samples), dtype=np.float64)
        for i in range(n_samples):
            K_train[i, i] = 1.0
            psi_i = self.train_states_[i]
            for j in range(i + 1, n_samples):
                # Fidelity: |<psi_i | psi_j>|^2
                fidelity = abs(psi_i.inner(self.train_states_[j])) ** 2
                K_train[i, j] = fidelity
                K_train[j, i] = fidelity

        self.K_train_ = K_train
        self.svc.fit(K_train, y)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        K_test = self._compute_kernel_test(X_test)
        return self.svc.predict(K_test)

    def decision_function(self, X_test: np.ndarray) -> np.ndarray:
        K_test = self._compute_kernel_test(X_test)
        return self.svc.decision_function(K_test)

    def _compute_kernel_test(self, X_test: np.ndarray) -> np.ndarray:
        """
        Computes rectangular kernel matrix between test samples and training samples.
        """
        test_states = self._compute_statevectors(np.asarray(X_test))
        n_test = len(test_states)
        n_train = len(self.train_states_)
        
        K_test = np.zeros((n_test, n_train), dtype=np.float64)
        for i in range(n_test):
            psi_test = test_states[i]
            for j in range(n_train):
                fidelity = abs(psi_test.inner(self.train_states_[j])) ** 2
                K_test[i, j] = fidelity
                
        return K_test
