---
name: team-sync-and-git-workflow
description: Synchronize work between the two-member team, validate cross-module interface contracts, and execute clean Git branching and merging workflows.
---

# Team Sync & Git Workflow Skill

Use this skill when preparing to commit, creating branches, executing handoffs, or validating that quantum and classical components seamlessly interface without breaking the pipeline.

## 1. Pre-Commit Validation Checklist

Before committing or switching branches:
1. **Lint & Run Smoke Test:**
   ```bash
   python -c "from src.data_loader import load_and_preprocess_data; from src.classical_baselines import train_classical_baselines; X_tr, X_te, y_tr, y_te = load_and_preprocess_data(); m = train_classical_baselines(X_tr, y_tr); print('Smoke test passed!')"
   ```
2. **Check Unstaged / Generated Artifacts:**
   Ensure no large `.csv`, `.png`, or virtualenv caches are committed (`git status`).
3. **Verify File Ownership:**
   - Member 1 touches `src/quantum_*.py`, `src/qsvm_*.py`.
   - Member 2 touches `src/data_*.py`, `src/classical_*.py`, `src/evaluate.py`.
   - Shared changes (e.g. `requirements.txt`, `README.md`) must be coordinated.

## 2. Git Branching Commands

```bash
# Check current status & branches
git status
git branch -a

# Create and switch to a new feature branch
git checkout -b <branch-name>

# Stage and commit with conventional commit format
git add .
git commit -m "feat(scope): concise description of changes"

# Push new branch to GitHub
git push -u origin <branch-name>
```

## 3. Resolving Contract Mismatches

- **Symptom:** `ValueError: Matrix dimensions mismatch during kernel dot product`.
  - **Resolution:** Check that `X_test.shape[1] == X_train.shape[1] == n_qubits`. Ensure both sets went through the identical `MinMaxScaler` instance (`scaler.transform(X_test)`, never `fit_transform(X_test)`).
- **Symptom:** `QiskitError: Number of parameters in feature map does not match input vector`.
  - **Resolution:** Verify `feature_dimension=X.shape[1]` when creating `ZZFeatureMap`.
