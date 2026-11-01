---
name: clinical-data-prep-and-baseline
description: Ingest, clean, scale clinical tabular datasets (e.g. Breast Cancer), reduce dimensionality via PCA, and train classical benchmark classifiers.
---

# Clinical Data Preparation & Classical Baselines Skill

Use this skill to preprocess medical diagnostic data, generate quantum-compatible feature vectors, and establish classical baseline benchmarks.

## 1. Preprocessing Requirements for Quantum Encoding

1. **Dimensionality Reduction:**
   NISQ simulators can comfortably handle 4 to 8 qubits. Reduce multi-feature medical datasets (e.g., 30 features in Wisconsin Diagnostic Breast Cancer) using Principal Component Analysis (PCA) to $N \in \{4, 6\}$ dimensions.
2. **Feature Range Scaling:**
   Quantum rotation gates operate over radians. Transform features using `MinMaxScaler(feature_range=(0, np.pi))` or `(-np.pi, np.pi)`.
3. **Stratified Splitting:**
   Maintain class distribution balance across train and test sets using `StratifiedKFold` or `train_test_split(..., stratify=y)`.

## 2. Standard Implementation (`src/data_loader.py`)

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_and_preprocess_data(
    n_features: int = 4,
    test_size: float = 0.25,
    random_state: int = 42,
    subsample_train: int = None
):
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # 1. Standardize original features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. PCA dimensionality reduction
    pca = PCA(n_components=n_features, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    
    # 3. MinMax scale to [0, pi] for angle encoding
    minmax = MinMaxScaler(feature_range=(0, np.pi))
    X_quantum = minmax.fit_transform(X_pca)
    
    # 4. Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X_quantum, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    if subsample_train is not None and subsample_train < len(X_train):
        idx = np.random.RandomState(random_state).choice(len(X_train), subsample_train, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
        
    return X_train, X_test, y_train, y_test
```

## 3. Classical Baselines Harness (`src/classical_baselines.py`)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def train_classical_baselines(X_train, y_train, random_state=42):
    models = {
        "Logistic Regression": LogisticRegression(random_state=random_state),
        "Classical SVM (Linear)": SVC(kernel="linear", probability=True, random_state=random_state),
        "Classical SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=4, random_state=random_state)
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models
```
