# Drift-Robust Gas Classification Using Machine Learning

M.Tech Data Science (24DS601 - Machine Learning), Amrita Vishwa Vidyapeetham, Coimbatore.

This project studies how sensor drift affects gas classification on the UCI Gas Sensor Array Drift dataset, and how well classical ML models hold up when they are tested on later batches than they were trained on.

The full step-by-step plan is in [PROJECT_PLAN.md](PROJECT_PLAN.md).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Folders

- `data_raw/Dataset/` - original batch files (not modified)
- `src/` - shared code
- `notebooks/` - one notebook per step
- `results/` - saved tables and figures
- `tests/` - data and leakage checks

## Status

- Step 0: project setup - done
- Step 1: load data, remove Toluene (12,077 samples, 5 gases) - done, see `notebooks/01_load_and_remove_toluene.ipynb`
- Step 2: exploratory data analysis - done, see `notebooks/02_eda.ipynb`
- Step 3: drift visualisation - done, see `notebooks/03_drift_visualisation.ipynb`
- Step 4: scaling pipeline, protocols A/B/C and metrics - done, see `notebooks/04_evaluation_framework.ipynb`
- Step 5: baseline classifiers (kNN, SVM, Decision Tree, Random Forest, Naive Bayes) - done, see `notebooks/05_baseline_models.ipynb`
- Step 6: PCA and LDA comparison - done, see `notebooks/06_pca_lda.ipynb`
- Step 7: boosting (AdaBoost, Gradient Boosting) - done, see `notebooks/07_boosting.ipynb`
- Step 8: hyperparameter tuning with batch-wise cross-validation - done, see `notebooks/08_tuning.ipynb` (tuning is run by `scripts/run_step8_tuning.py`)
- Step 9: stacking ensemble - done, see `notebooks/09_stacking.ipynb` (computed by `scripts/run_step9_stacking.py`)

Results will be added here as each step is completed.
