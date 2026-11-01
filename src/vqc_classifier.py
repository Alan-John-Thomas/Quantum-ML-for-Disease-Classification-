"""
Variational Quantum Classifier (VQC) using Qiskit.
Constructs a parameterized circuit:
  ZZFeatureMap (data encoding) + RealAmplitudes (variational ansatz)
Optimized via classical COBYLA algorithm to minimize classification error.
"""

import time
import numpy as np
from scipy.optimize import minimize
from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit import transpile
from qiskit_aer import AerSimulator
from sklearn.base import BaseEstimator, ClassifierMixin

class VQCClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_qubits: int = 4,
        reps: int = 1,
        entanglement: str = "linear",
        maxiter: int = 25,
        subsample_train: int = 100,
        random_state: int = 42
    ):
        self.n_qubits = n_qubits
        self.reps = reps
        self.entanglement = entanglement
        self.maxiter = maxiter
        self.subsample_train = subsample_train
        self.random_state = random_state

        # 1. Feature Map (Encoding)
        self.feature_map = zz_feature_map(
            feature_dimension=self.n_qubits,
            reps=self.reps,
            entanglement=self.entanglement
        )

        # 2. Variational Ansatz (Trainable parameters)
        self.ansatz = real_amplitudes(
            num_qubits=self.n_qubits,
            reps=self.reps,
            entanglement=self.entanglement
        )

        # 3. Composed Quantum Neural Circuit
        self.circuit = self.feature_map.compose(self.ansatz)
        self.n_features_ = self.feature_map.num_parameters
        self.n_weights_ = self.ansatz.num_parameters

        # Measurement observable: Pauli Z on qubit 0
        obs_str = "I" * (self.n_qubits - 1) + "Z"
        self.observable = SparsePauliOp.from_list([(obs_str, 1.0)])

        self.weights_ = None
        self.training_loss_history_ = []

    def get_circuit_metrics(self) -> dict:
        """
        Transpiles full VQC circuit to AerSimulator basis to extract gate complexity.
        """
        sim = AerSimulator()
        transpiled = transpile(self.circuit, backend=sim, optimization_level=1)
        ops = dict(transpiled.count_ops())
        return {
            "num_qubits": self.n_qubits,
            "depth": transpiled.depth(),
            "gate_count": ops,
            "cx_count": ops.get("cx", 0)
        }

    def _predict_prob_single(self, x: np.ndarray, theta: np.ndarray) -> float:
        """
        Evaluates expectation value <Z_0> for a single data vector x and weights theta.
        Maps expectation [-1.0, 1.0] to probability [0.0, 1.0].
        """
        # Parameter bindings: first n_features parameters are data x, remaining are weights theta
        param_values = np.concatenate([x, theta])
        bound = self.circuit.assign_parameters(param_values)
        state = Statevector.from_instruction(bound)
        exp_z = state.expectation_value(self.observable).real
        # Map expectation [-1, +1] -> probability [0, 1]
        prob = (1.0 + exp_z) / 2.0
        return float(np.clip(prob, 1e-6, 1.0 - 1e-6))

    def _cost_function(self, theta: np.ndarray, X_sub: np.ndarray, y_sub: np.ndarray) -> float:
        """
        Mean squared error loss across the batch.
        """
        probs = np.array([self._predict_prob_single(x, theta) for x in X_sub])
        loss = float(np.mean((probs - y_sub) ** 2))
        self.training_loss_history_.append(loss)
        return loss

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VQCClassifier":
        """
        Optimizes ansatz parameters using COBYLA on a representative subsample.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Subsample to keep hackathon training within ~15 seconds
        if self.subsample_train is not None and self.subsample_train < len(X):
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(len(X), size=self.subsample_train, replace=False)
            X_sub, y_sub = X[idx], y[idx]
        else:
            X_sub, y_sub = X, y

        # Initial random weights in [-pi, pi]
        rng = np.random.RandomState(self.random_state)
        init_theta = rng.uniform(-np.pi, np.pi, size=self.n_weights_)

        self.training_loss_history_ = []
        opt_res = minimize(
            self._cost_function,
            x0=init_theta,
            args=(X_sub, y_sub),
            method="COBYLA",
            options={"maxiter": self.maxiter, "rhobeg": 0.5}
        )

        self.weights_ = opt_res.x
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns (N, 2) probability array [P(Control), P(Autism)].
        """
        if self.weights_ is None:
            raise RuntimeError("VQCClassifier is not fitted yet!")

        probs_1 = np.array([self._predict_prob_single(x, self.weights_) for x in np.asarray(X)])
        probs_0 = 1.0 - probs_1
        return np.column_stack([probs_0, probs_1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return 2.0 * (probs - 0.5)


if __name__ == "__main__":
    print("[VQC] Running smoke test on synthetic batch...")
    vqc = VQCClassifier(n_qubits=4, reps=1, maxiter=10, subsample_train=30)
    X_dummy = np.random.uniform(0, np.pi, size=(40, 4))
    y_dummy = np.random.randint(0, 2, size=40)
    vqc.fit(X_dummy, y_dummy)
    preds = vqc.predict(X_dummy[:5])
    print("[VQC] Predictions:", preds)
    metrics = vqc.get_circuit_metrics()
    print("[VQC] Circuit Metrics:", metrics)
    print("[VQC] Self-test passed cleanly!")
