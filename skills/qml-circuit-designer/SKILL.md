---
name: qml-circuit-designer
description: Design, transpile, and benchmark parameterized quantum circuits, feature maps, and ansatzes in Qiskit 1.x for clinical classification.
---

# QML Circuit Designer Skill

Use this skill when designing or modifying quantum feature maps (Angle Encoding, `ZFeatureMap`, `ZZFeatureMap`) or parameterized ansatzes (`RealAmplitudes`, `EfficientSU2`) in Qiskit 1.x.

## 1. Feature Map Selection Principles

For clinical tabular datasets (e.g., patient diagnosis indicators):
- **Angle Encoding (`ZFeatureMap`):** Single-qubit rotations $R_z(x_i)$. Low circuit depth ($O(1)$), no entangling gates, useful as a lightweight baseline.
- **Second-Order Pauli-Z Expansion (`ZZFeatureMap`):** Encodes both single-feature non-linearities and pairwise feature correlations:
  $$\phi_{\{i, j\}}(x) = 2(\pi - x_i)(\pi - x_j)$$
  Requires CNOT entangling gates. Standard benchmark for showing meaningful quantum entanglement in hackathon evaluations.
- **Entanglement Topology:** Use `entanglement='linear'` for $N \le 6$ qubits on simulators to minimize circuit depth without sacrificing pairwise interaction representation.

## 2. Standard Implementation (Qiskit 1.x)

```python
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, RealAmplitudes, EfficientSU2
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

def create_feature_map(n_qubits: int = 4, map_type: str = "zz", reps: int = 2) -> QuantumCircuit:
    """
    Creates a parameterized feature map circuit.
    Args:
        n_qubits: Number of qubits (must match input feature count).
        map_type: 'zz' for ZZFeatureMap, 'z' for ZFeatureMap.
        reps: Number of repetitions/layers.
    """
    if map_type.lower() == "zz":
        return ZZFeatureMap(feature_dimension=n_qubits, reps=reps, entanglement='linear')
    elif map_type.lower() == "z":
        return ZFeatureMap(feature_dimension=n_qubits, reps=reps)
    else:
        raise ValueError(f"Unsupported map_type: {map_type}")

def create_ansatz(n_qubits: int = 4, ansatz_type: str = "real_amplitudes", reps: int = 2) -> QuantumCircuit:
    """
    Creates a parameterized variational ansatz for VQC.
    """
    if ansatz_type.lower() == "real_amplitudes":
        return RealAmplitudes(num_qubits=n_qubits, reps=reps, entanglement='linear')
    elif ansatz_type.lower() == "efficient_su2":
        return EfficientSU2(num_qubits=n_qubits, reps=reps, entanglement='linear')
    else:
        raise ValueError(f"Unsupported ansatz_type: {ansatz_type}")
```

## 3. Circuit Transpilation & Complexity Metrics

Judges evaluate circuit feasibility. Always report:
- Total Qubit Count
- Circuit Depth
- Two-Qubit Entangling Gate Count (CNOTs / CX)

```python
def extract_circuit_metrics(circuit: QuantumCircuit) -> dict:
    simulator = AerSimulator()
    transpiled = transpile(circuit, backend=simulator, optimization_level=2)
    return {
        "num_qubits": transpiled.num_qubits,
        "depth": transpiled.depth(),
        "gate_count": dict(transpiled.count_ops()),
        "cx_count": transpiled.count_ops().get("cx", 0)
    }
```
