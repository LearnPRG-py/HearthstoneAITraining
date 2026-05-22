# Hearthstone AI Training Repository

This repository is used for parameter and hyperparameter tuning for Hearthstone AI — an ASL classification model trained across 2208 different words/categories using nearly 108,000 videos.

---

# Current Updates

**Current Top-1 Accuracy:** 40.5%

---

# Current Best Model

For the CI pipeline, models are evaluated on a reduced dataset containing 25,600 videos and trained for only 5 epochs. This allows contributors to quickly validate whether a change improves model quality without requiring a long full-training run.

After training, the model is evaluated on unseen test data.

These minimized runs are referred to as **MTRs (Minimized Training Runs)**.

Changes that pass the MTR pipeline are committed to `origin/staging`. Selected improvements are later merged into `origin/main` after a longer full-scale training run.

## Current Best MTR Results

- **Loss:** 6.9809
- **Accuracy:** 0.0039

---

# Repository Structure

## Editable Files

Only `model.py` should be modified by contributors.

Changes to:
- training scripts
- CI scripts
- helper utilities
- datasets
- evaluation logic

should be requested through an issue instead of being directly edited.

---

# Contributing

1. Make changes only to `model.py`
2. Run:

```bash
python3 commit_helper.py
```

The helper script will automatically:
- verify dependencies
- run formatting checks
- execute the reduced training/evaluation pipeline
- compare results against the previous benchmark
- commit successful improvements
- push approved changes to `origin/staging`

Direct commits to `origin/staging` should normally never be required.

If the helper script is malfunctioning or bypasses are needed, please open an issue immediately.

---

# Reproducibility

The repository uses fixed dataset splits and seeded training behavior for more stable benchmarking across contributors and hardware configurations.

However, exact bit-perfect reproducibility is not guaranteed due to GPU/backend nondeterminism in TensorFlow.

---

# Requirements

Main dependencies:
- TensorFlow 2.16.1
- NumPy 1.26.4
- Pandas 2.2.2
- Matplotlib 3.8.4
- Scikit-learn 1.4.2
- Black 22.3.0
