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

Results will be added here as each step is completed.
