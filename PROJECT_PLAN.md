# Drift-Robust Gas Classification Using Machine Learning — Project Plan (v2)

**Course:** 24DS601 – Machine Learning, M.Tech Data Science, Amrita Vishwa Vidyapeetham, Coimbatore
**Student:** Kiran Gautham (CB.AI.P2DSC26013)
**GitHub:** https://github.com/kirangautham-82899/Drift-Robust-Gas-Classification.git

---

## 1. Project idea (one paragraph)

Metal-oxide gas sensors change as they age. A model trained on early data loses accuracy on later data. This is called **sensor drift**. We train classical ML models on early batches and test them on later batches (a *time-aware* split) so the drift shows up honestly. We measure how much each model suffers, test whether PCA or LDA help, and build a stacking ensemble to see whether it resists drift better. We also show that a random split gives misleadingly high accuracy.

**Motivating question:** *Can a classical ML model keep identifying gases correctly when sensor characteristics change over time?*

## 2. Dataset

UCI Gas Sensor Array Drift Dataset (Vergara et al., 2012). Raw files are in `data_raw/Dataset/batch1.dat … batch10.dat`.

- 16 metal-oxide sensors, 128 features per sample (16 sensors x 8 features), recorded over 36 months (Jan 2008 - Feb 2011).
- Format: LibSVM-style text, `label 1:value 2:value … 128:value`.
- Feature layout and meanings (steady-state ΔR, normalised ΔR, rising/decaying EMA features) are from the UCI documentation. **Step 1 verifies the layout automatically.**
- Gas labels: 1 Ethanol, 2 Ethylene, 3 Ammonia, 4 Acetaldehyde, 5 Acetone, 6 Toluene.

### Toluene is removed (teacher's instruction)
Toluene (label 6) is absent from batches 3, 4 and 5 and has only 5 samples in batch 2. Removing it keeps the class set consistent across batches, so per-batch results are comparable. Raw files are never modified; removal is done in code.

| Batch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | Total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Original | 445 | 1,244 | 1,586 | 161 | 197 | 2,300 | 3,613 | 294 | 470 | 3,600 | 13,910 |
| Toluene removed | 74 | 5 | 0 | 0 | 0 | 467 | 568 | 18 | 101 | 600 | 1,833 |
| **After removal** | **371** | **1,239** | **1,586** | **161** | **197** | **1,833** | **3,045** | **276** | **369** | **3,000** | **12,077** |

Result: 5 classes (Ethanol, Ethylene, Ammonia, Acetaldehyde, Acetone), 12,077 samples.

## 3. Defaults

- Notebooks (one per step) plus a small `src/` folder of shared functions.
- Raw dataset (~24 MB) pushed to GitHub.
- Headline protocol: **A** (train batches 1-3, test batches 4-10). B and C are supporting analysis.
- Headline metrics: **accuracy and macro-F1**, fixed up front.
- Random seed: 42 everywhere.

### Evaluation protocols
- **A:** train on batches 1-3, test on batches 4-10 (the Zeroth review slide).
- **B:** train on batch 1 only, test on each of batches 2-10 separately.
- **C:** rolling, train on batches 1..k-1, test on batch k.

## 4. Rules for every step

1. Test batches are never used for tuning or model selection.
2. Scalers, PCA and LDA are fitted on training data only (the Tier 2 Bonus is the one labelled exception).
3. Every reported number states its protocol (and tier, for Bonus).
4. If a method does not help, we report that.
5. Only **measured** numbers go in the report. No estimated accuracies.
6. After each step: result, how it was tested, a plain-English explanation, likely teacher questions with short answers, then a short human-style commit (no co-author line) pushed to GitHub.

---

## 5. CORE steps (finish these first)

| Step | What is done | How it is checked | Commit message |
|---|---|---|---|
| **0** | Project setup: `git init`, `main` branch, connect GitHub, folders, `.gitignore`, `requirements.txt`, README skeleton | `git status` clean, first push visible on GitHub, packages import | Set up project structure and requirements |
| **1** | Load the 10 batches, remove Toluene | **Automatic checks:** 128 features per row, no NaN/inf, 13,910 rows before and 12,077 after, per-batch counts match the table above, labels only 1-5. If a check fails, stop and fix. | Load all batches and remove Toluene samples |
| **2** | EDA: class balance per batch, feature ranges, summary statistics | Graphs reviewed, counts cross-checked with Step 1 | Add exploratory data analysis |
| **3** | Drift visualisation: sensor means across batches, PCA plot coloured by batch and by gas, signal-vs-time proxy | PCA fitted on training batches only; the Dennler proxy is labelled as a proxy | Add drift visualisation across batches |
| **4** | Scaling pipeline, protocols A/B/C, shared metrics function | Unit tests: scaler never sees test data, train and test batches never overlap | Add scaling pipeline and time-aware evaluation protocols |
| **5** | Baselines: kNN, SVM, Decision Tree, Random Forest, Naive Bayes | **First real numbers**; everything later is compared against these | Add baseline classifiers and results |
| **6** | PCA and LDA (LDA limited to 4 components) on the Step 5 models | Fitted on training data only; with/without comparison | Add PCA and LDA comparison |
| **7** | Boosting: AdaBoost and Gradient Boosting | Same protocol and metrics | Add AdaBoost and Gradient Boosting |
| **8** | Hyperparameter tuning with batch-wise cross-validation on training batches only | Test batches untouched (verified in code) | Tune hyperparameters with batch-wise cross-validation |
| **9** | Stacking ensemble: diverse base models + logistic-regression meta-model, batch-wise folds | Leakage check; if it does not beat the best single model, report that | Add stacking ensemble |
| **10** | Per-batch drift analysis: accuracy and macro-F1 per batch, **bootstrap confidence intervals**, confusion matrices, pooled results for batches 4-10 | Interpretation notes for noisy batches 4 and 5 | Add per-batch drift analysis |
| **11** | Leakage check: random split vs time-aware split | Shows why the random-split number is misleading | Compare random split with time-aware split |
| **12** | Repeated runs over several seeds: mean ± standard deviation for randomised models | Results table with spread | Add repeated runs with mean and std |
| **13** | Core conclusions, Limitations section, README (no viva Q&A sheet, by request) | Every README table checked against the result files by a test | Write README, conclusions and limitations |

## 5b. UI (interactive dashboard for the review/demo)

A Streamlit web app (`app/`) that presents the project visually: data, drift, model results and a live prediction demo. It is built in four small stages, each right after the Core step whose results it needs, and each is committed and pushed like any other step.

### Principles
- **The UI only displays saved results.** All training happens in the notebooks. The notebooks write tables, figures and trained models to `results/`, and the app reads them. So the UI never shows a number that was not measured, and it opens instantly during a demo.
- **Every chart has a plain-English caption** ("What this shows / what to notice") so a beginner, or the teacher, can read it without help.
- **One clear flow:** problem → data → drift → models → results → leakage → demo → limitations.
- **Consistent look:** one colour palette that is colour-blind safe, the same colour per gas and per model on every page, light and dark mode both readable, large readable fonts.
- **Works offline** from the local project folder (no cloud, no login).

### Pages
| Page | What it shows | Source |
|---|---|---|
| 1. Overview | Problem, motivating question, dataset facts, pipeline diagram, the Toluene decision | Static text + Step 1 counts |
| 2. Data Explorer | Samples per batch and per gas (after Toluene removal), feature ranges, a table viewer with batch/gas filters | Steps 1-2 |
| 3. Drift Viewer | PCA scatter coloured by batch vs by gas, sensor-mean trends across batches, a batch slider | Step 3 |
| 4. Model Results | Leaderboard of all models; choose protocol (A/B/C), metric (accuracy, macro-F1) and PCA/LDA option | Steps 5-9 |
| 5. Per-Batch Drift | Accuracy/F1 per batch with confidence intervals for chosen models; the performance-drop line | Step 10 |
| 6. Confusion Matrix | Pick model + batch, see where gases get confused | Step 10 |
| 7. Random vs Time-Aware | Side-by-side bars showing the misleading random-split accuracy against the honest time-aware one | Step 11 |
| 8. Live Demo | Pick a test batch and a sample, or upload a CSV of 128 features; shows the predicted gas, the probabilities and the true label | Trained models in `results/models/` |
| 9. Limitations & About | The limitations list, references, student details | Plan + Step 13 |

### UI steps
| Step | When | What is done | How it is checked | Commit message |
|---|---|---|---|---|
| **U1** | After Step 5 | App skeleton, theme, sidebar navigation; pages 1-3 | App starts without errors; counts on page 2 match Step 1 | Add dashboard skeleton with overview, data and drift pages |
| **U2** | After Step 10 | Pages 4-6 (model results, per-batch drift, confusion matrices) | Numbers shown equal the saved result files; every control changes the chart | Add model results and drift analysis pages |
| **U3** | After Step 12 | Pages 7-8 (random vs time-aware, live demo) | Demo prediction for a known sample equals the notebook's prediction | Add leakage comparison and live prediction demo |
| **U4** | After Step 13 | Page 9, polish, automatic smoke tests (Streamlit's `AppTest`), run instructions and screenshots in the README | Every page loads in the smoke test; the README run command works from a clean clone | Finish dashboard and add run instructions |

Run command (planned): `streamlit run app/app.py`.

### Extra libraries for the UI
`streamlit` and `plotly` (interactive charts), plus `joblib` for loading saved models. **Status: all four dashboard stages (U1-U4) are done.** The packages install with `pip install --use-feature=truststore` on this machine (plain `pip install` failed with an SSL certificate error because this Python has no certificate file of its own).

### Demo flow (suggested, about 5 minutes)
Overview → Drift Viewer (show the batches drifting apart) → Random vs Time-Aware (the "98% is misleading" moment) → Model Results and Per-Batch Drift (which models resist drift best) → Live Demo → Limitations.

### UI risks
- The Live Demo needs trained models saved to disk, so Steps 5-9 must save their models. This is added to those steps.
- Interactive charts can be slow if they plot thousands of points. Plots are subsampled or pre-computed where needed.
- The UI is a presentation layer, not a research result. Marks for the analysis come from the notebooks and report, so the UI never takes priority over Core.

## 6. BONUS steps (only after Core is complete and pushed, and only if the teacher agrees)

Ask the teacher before B2: these methods go beyond the Zeroth review objectives.

| Step | What |
|---|---|
| **B1** | Feature robustness: which of the 8 per-sensor features are most stable across batches |
| **B2** | **Tier 2 (unlabelled adaptation):** per-batch standardisation, CORAL, subspace alignment. Compare **with vs without** per batch, because per-batch scaling can hurt on unusual batches |
| **B3** | **Tier 3 (light supervision):** rolling retraining, recency weighting, a few labelled samples from the new batch, self-training |
| **B4** | Tier 1 → 2 → 3 summary graph; optional appendix with Toluene included as a sensitivity check |

### Tiers (what the model may use from new batches)
1. **Strict:** nothing. Train on early batches, test on later ones.
2. **Unlabelled adaptation:** only the new batch's raw readings, no gas labels.
3. **Light supervision:** the latest labelled batches, or a few labelled samples from the new batch.

## 7. Limitations (stated in the report from the start)

- **5 gases only.** Toluene was excluded on the teacher's instruction.
- **Dennler's baseline-leakage concern cannot be reproduced.** The raw pre-exposure baseline is not in this dataset version, so a proxy is used and labelled as one.
- **Batches 4 and 5 are small and noisy** (161 and 197 samples; as few as 12 per gas). They are used for pooled analysis, not strong individual claims.
- **Perfect accuracy is not achievable** under drift. We report the best honest number per setting.
- **Single dataset,** so conclusions may not transfer to other sensors.
- Stacking may not beat the best single model; if so, that is reported as a finding.

## 8. Expected results (no numbers promised)

No accuracy figures are promised in advance. The first real numbers come from Step 5 and replace any estimate. The things we expect to see qualitatively:

- A random split gives a very high accuracy that is misleading.
- Time-aware accuracy is lower and varies by batch, falling for batches further from training.
- Random Forest, SVM and Gradient Boosting are likely among the strongest; Decision Tree and Naive Bayes among the weakest.
- PCA/LDA may or may not help; either outcome is a valid finding.

## 9. Planned repository layout

```
Drift-Robust-Gas-Classification/
├── README.md
├── PROJECT_PLAN.md
├── requirements.txt
├── .gitignore
├── data_raw/Dataset/           # original batch1.dat … batch10.dat (unchanged)
├── src/                        # shared functions (loading, evaluation, plotting)
├── notebooks/                  # one notebook per step
├── app/                        # Streamlit dashboard (app.py + pages/)
├── results/                    # saved tables, figures and trained models (read by the app)
└── tests/                      # leakage and data-integrity checks
```

## 10. Open items

- Confirm whether the GitHub repo is empty or already has files (checked before the first push; nothing is overwritten).
- Teacher's decision on the Bonus methods.
- Review date, if dated milestones are wanted.
- (done) Installing `streamlit` and `plotly`: works with `pip install --use-feature=truststore`.
- Notebooks must save their models and result tables to `results/` so the app can read them.
