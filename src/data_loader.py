"""
Data loader and preprocessing module for Autism Brain MRI classification.
Supports:
1. Real parquet extracted by Jeswin (data/abide_roi_features.parquet).
2. Synthetic fallback matching the exact schema for interactive testing and demonstration.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def generate_mock_abide_data(
    n_samples: int = 200,
    n_rois: int = 64,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generates synthetic data matching the Harvard-Oxford ROI parquet schema.
    Class 1: Autism Spectrum Disorder (ASD)
    Class 2: Typical Control (TC)
    """
    rng = np.random.RandomState(random_state)
    
    # Generate random subject IDs (e.g. 50001, 50002, ...)
    subject_ids = [50000 + i for i in range(n_samples)]
    
    # Balanced binary classes: 1 (Autism) or 2 (Control)
    dx_group = rng.choice([1, 2], size=n_samples, p=[0.5, 0.5])
    
    # Mean regional intensity values (simulating brain MRI ROI extractions)
    # Add subtle mean shift between classes to simulate diagnostic signal
    mean_shift = np.where(dx_group[:, None] == 1, 0.35, -0.35)
    raw_intensities = rng.randn(n_samples, n_rois) + mean_shift
    
    data = {
        "subject_id": subject_ids,
        "dx_group": dx_group
    }
    for i in range(n_rois):
        data[f"roi_{i+1:02d}"] = raw_intensities[:, i]
        
    return pd.DataFrame(data)

def load_and_preprocess_data(
    parquet_path: str = "data/abide_roi_features.parquet",
    n_features: int = 4,
    test_size: float = 0.25,
    random_state: int = 42,
    subsample_train: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, PCA]:
    """
    Loads extracted ROI features, applies standardization, PCA dimensionality reduction,
    and scales features strictly to [0, pi] for quantum angle encoding.

    Returns:
        X_train: (N_train, n_features) bounded in [0, pi]
        X_test:  (N_test, n_features) bounded in [0, pi]
        y_train: (N_train,) binary {0: Control, 1: Autism}
        y_test:  (N_test,) binary {0: Control, 1: Autism}
        pca: Fitted PCA instance for variance profiling
    """
    if os.path.exists(parquet_path):
        print(f"[DataLoader] Ingesting real ABIDE ROI features from: {parquet_path}")
        df = pd.read_parquet(parquet_path)
    else:
        print(f"[DataLoader] Real parquet not found at '{parquet_path}'. Using synthetic ABIDE schema generator.")
        df = generate_mock_abide_data(n_samples=200, n_rois=64, random_state=random_state)
        
    # Extract feature columns and labels
    feature_cols = [c for c in df.columns if c.startswith("roi_") or c.startswith("feat_")]
    if not feature_cols:
        raise ValueError("No ROI feature columns found in dataset!")
        
    X_raw = df[feature_cols].values
    
    # Target label: Map 1 (Autism) -> 1, 2 (Control) -> 0
    # If labels are already 0 and 1, preserve them
    raw_labels = df["dx_group"].values
    if set(np.unique(raw_labels)).issubset({1, 2}):
        y = (raw_labels == 1).astype(int)
    else:
        y = raw_labels.astype(int)
        
    # Step 1: Stratified Train / Test Split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Step 2: Standardize original features (fit only on train to prevent leakage)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train_raw)
    X_test_std = scaler.transform(X_test_raw)
    
    # Step 3: PCA Dimensionality Reduction to target qubit count
    pca = PCA(n_components=n_features, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)
    
    # Step 4: MinMax scale strictly to [0, pi] for quantum angle encoding
    minmax = MinMaxScaler(feature_range=(0.0, np.pi))
    X_train = minmax.fit_transform(X_train_pca)
    X_test = minmax.transform(X_test_pca)
    
    # Clip test features to [0, pi] in case of slight out-of-bound projections
    X_test = np.clip(X_test, 0.0, np.pi)
    
    # Optional subsampling for rapid kernel matrix evaluation
    if subsample_train is not None and subsample_train < len(X_train):
        idx = np.random.RandomState(random_state).choice(len(X_train), subsample_train, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
        
    return X_train, X_test, y_train, y_test, pca
