<div align="center">

# Drift-Robust Gas Classification

**Can a model keep recognising gases as its sensors age?**

A time-aware machine-learning study of sensor drift on the UCI Gas Sensor Array Drift dataset,<br>
with an interactive dashboard that shows every result.

[![Open the live app](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://drift-robust-gas-classification-ctfmhqhdsllpacizk7et2b.streamlit.app/)

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-7.1-3F4F75?logo=plotly&logoColor=white)
![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)

[**🚀 Live app**](https://drift-robust-gas-classification-ctfmhqhdsllpacizk7et2b.streamlit.app/) · [Results](#main-results) · [Dashboard](#interactive-dashboard) · [Conclusions](#conclusions) · [Reproduce](#how-to-reproduce)

</div>

---

## The answer in 30 seconds

Metal-oxide gas sensors change as they age, so a model trained on early recordings slowly stops recognising the same gases.
We trained classical models on the **first three batches** and tested them on the **seven later ones**.

| | |
|:--|:--|
| 🎯 **73.2%** | best single model (SVM) on later batches; a soft vote of five models reaches 75.5% |
| 📉 **90.3% → 55.4%** | the SVM on batches close to training vs. batches furthest away |
| 🎭 **97.2%** | what the untuned SVM (72.7% time-aware) *appears* to score with a random split; it hides the drift |
| 🔁 **88.6%** | with recent labelled data (retraining before each batch; mean over batches) |
| 🧪 **79.2% / 92.1%** | bonus: rescaling each new batch on its own (no labels) / plus 25 random labels per batch |

**Short answer: no, not fully.** Drift costs 11-35 points of accuracy, and no classical trick tested here (PCA, LDA,
boosting, tuning, stacking) clearly fixes it. Using the new batches' own readings, or a few of their labels, does help.

## How we test fairly

```mermaid
flowchart LR
    A["Batches 1-3<br/>3,196 recordings"] -->|train| M(("Model"))
    M -->|test| N["Batches 4-6<br/>near in time"]
    M -->|test| F["Batches 8-10<br/>far in time"]
    style A fill:#ccfbf1,stroke:#0d9488,color:#0f172a
    style M fill:#e0e7ff,stroke:#4f46e5,color:#0f172a
    style N fill:#ede9fe,stroke:#7c3aed,color:#0f172a
    style F fill:#fce7f3,stroke:#db2777,color:#0f172a
```

- **Data.** [UCI Gas Sensor Array Drift](https://doi.org/10.24432/C5RP6W): 16 sensors, 128 features per recording, 36 months,
  10 time-ordered batches. Toluene is removed (it is missing from batches 3-5), leaving **12,077 samples and 5 gases**
  over 10 batches of 371, 1,239, 1,586, 161, 197, 1,833, 3,045, 276, 369 and 3,000 samples.
- **Protocols.** **A** (headline): train on batches 1-3, test on 4-10. **B**: train on batch 1, test on each later batch.
  **C**: retrain on all earlier batches before each test batch.
- **No leakage.** Scaling, PCA and LDA live inside the pipeline and see training batches only; tuning uses forward-in-time
  folds inside batches 1-3. Tests prove the test batches cannot change anything that was fitted.
- **Models.** kNN, SVM, Decision Tree, Random Forest, Naive Bayes, AdaBoost and Gradient Boosting, each with no reduction,
  PCA or LDA, plus a stacking ensemble and a soft vote. Scored with accuracy and macro-F1, bootstrap 95% intervals and 10 seeds.

## Main results

Every table below is checked against the saved files in `results/` by `tests/test_readme_numbers.py`.

### Headline: pooled accuracy on batches 4-10

Mean ± standard deviation over 10 seeds, tuned settings, each model with the reduction that won on validation.

| Model | Accuracy (mean ± std) | Macro-F1 |
|---|---|---|
| Soft vote | 75.5% ± 0.14 | 75.5% |
| SVM | 73.2% ± 0.00 | 74.5% |
| Random Forest + PCA | 69.9% ± 0.59 | 67.4% |
| Stack (LR) | 69.5% ± 0.40 | 70.6% |
| Gradient Boosting + PCA | 67.7% ± 0.03 | 67.1% |
| AdaBoost + PCA | 66.2% ± 0.00 | 64.6% |
| kNN + PCA | 64.9% ± 0.00 | 66.3% |
| Decision Tree | 52.6% ± 1.54 | 48.7% |
| Naive Bayes + PCA | 51.4% ± 0.00 | 50.5% |

Guessing the most common gas scores 21-23%. A spread of ± 0.00 means the model has no randomness that the seed
changes (SVM, kNN, Naive Bayes, and AdaBoost here), so the 10 seeds say nothing about its stability.

<p align="center"><img src="results/step10_pooled_ci.png" alt="Pooled accuracy with 95% confidence intervals" width="80%"></p>

### Drift: the further from training, the worse

| Model | Near batches 4-6 | Far batches 8-10 | Change (points) |
|---|---|---|---|
| SVM | 90.3% | 55.4% | -34.9 |
| Soft vote | 85.6% | 66.6% | -19.0 |
| Stack (LR) | 85.3% | 59.1% | -26.1 |
| Random Forest + PCA | 85.3% | 57.3% | -28.0 |
| kNN + PCA | 71.5% | 60.0% | -11.4 |

No model exceeds 67% on the far batches, and batch 10 is hard for all of them (34-62%).

<p align="center"><img src="results/step10_per_batch_ci.png" alt="Accuracy per test batch with 95% bands" width="80%"></p>

### The random-split trap

The same models and settings under three exams. "Random, same sizes" matches protocol A's sizes, so only time order differs.

| Model | Time-aware (A) | Random, same sizes | Random 80/20 |
|---|---|---|---|
| kNN | 69.6% | 98.3% | 99.3% |
| SVM | 72.7% | 97.2% | 98.4% |
| Decision Tree | 49.7% | 95.6% | 97.9% |
| Random Forest | 64.7% | 99.0% | 99.4% |
| Naive Bayes | 56.5% | 64.2% | 65.8% |
| AdaBoost | 47.2% | 85.2% | 86.6% |
| Gradient Boosting | 55.8% | 98.7% | 99.2% |

In a random split, 98% of test recordings have their nearest neighbour in the training set from the same batch (0% under the time-aware split), so models pass by
recognition. It also picks the wrong winner: Gradient Boosting looks like 99% but scores 55.8% under drift.

<p align="center"><img src="results/step11_random_vs_time_aware.png" alt="Random split vs time-aware split" width="80%"></p>

<details>
<summary><b>With recent labelled data (protocol C)</b></summary>

<br>

Mean per-batch accuracy on batches 4-10 when each model is retrained on all earlier batches. A different, easier task,
so it is kept out of the headline.

| Model | Mean accuracy |
|---|---|
| SVM | 88.6% |
| Stack (LR) | 87.0% |
| Soft vote | 85.3% |

</details>

### Bonus: drift adaptation (Step 13)

What if a method may use something from each new batch? **Tier 2** methods use the batch's sensor readings but never its
gas labels; **tier 3** methods also get a few labelled recordings per gas, which are left out of the score. All use the
tuned SVM. A method counts as an improvement only if it also beats the strict SVM on the Step 8 validation folds
(batches 2 and 3), so nothing is chosen by looking at the test batches. This check favours the strict SVM: its settings
were tuned on those same folds, while the MLP and CORAL were not tuned at all (Step 14 tunes them; the verdict holds). These numbers use information protocol A
forbids, so they are kept apart from the headline. "Test" is accuracy over all test recordings, so the large batches 7
and 10 weigh most.

| Method | Tier | Validation | Test, batches 4-10 | Batch 10 |
|---|---|---|---|---|
| SVM (strict) | 1 | 88.6% | 73.2% | 49.6% |
| MLP (strict) | 1 | 80.5% | 75.6% | 58.6% |
| SVM + CORAL | 2 | 80.3% | 77.4% | 52.7% |
| SVM + per-batch standardisation | 2 | 90.6% | 79.2% | 63.3% |
| SVM + per-batch std + self-training | 2 | 90.5% | 80.0% | 65.2% |
| SVM + per-batch std + 3 labels per gas | 3 | 95.3% | 89.1% | 78.5% |
| SVM + per-batch std + 5 labels per gas | 3 | 96.2% | 91.0% | 82.2% |
| SVM + per-batch std + 10 labels per gas | 3 | 97.1% | 94.6% | 89.9% |

**Per-batch standardisation**, which scales each batch with its own mean and spread, is the fair tier-2 winner: +6.0
points over all test recordings with no test labels. The MLP and CORAL look better on test but worse on validation, so
they are not counted. The gain is uneven and depends on how batches are weighted: averaged over the seven test batches
equally, it scores 80.9% against 81.4% for the strict SVM, so it does not help a typical batch. Most of the gain comes
from batch 10, which holds 34% of the test recordings and has exactly 20% of each gas. Per-batch scaling builds each
batch's gas mix into its features: it loses 25 points on batch 8, which is 52% Acetone. A few labels fix that and give
the largest gains (tier-3 rows: mean of 10 seeds). The "labels per gas" rows pick a fixed number of recordings of each
gas, which needs the labels to be known first; the fair version is a fixed budget picked at random, below.

**Using the same labels more cleverly did not help.** With 25 random labels per new batch (the same count as 5 per gas,
picked without knowing the gas) the SVM scores 96.2% on validation and 92.1% on test (94.1% averaged over batches). Labelling the recordings the model is least sure about (active learning), weighting the
labels, self-training, the soft vote and four other SVM settings never beat that on validation. Some reach 93.0% on
test, but only a choice made by looking at the test batches would pick them. More labels is what helps.

**The hardest gas, Acetaldehyde.** The strict SVM finds only 52.6% of it, and in batch 10 just 11.0%: there it calls
most Acetaldehyde Ethanol. In the training data, the three closest pairs of gases all include Acetaldehyde, and it is
the smallest class (11.9%). Recall = share of the real Acetaldehyde found; precision = how often "Acetaldehyde" is right.

| Method | Acetaldehyde recall | Acetaldehyde precision | Recall, batch 10 | Accuracy, all gases |
|---|---|---|---|---|
| SVM (strict) | 52.6% | 57.2% | 11.0% | 73.2% |
| SVM, Acetaldehyde weighted x3 | 50.0% | 54.9% | 8.7% | 72.5% |
| SVM + per-batch standardisation | 33.2% | 73.9% | 19.2% | 79.2% |
| SVM + per-batch std + Ethanol/Acetaldehyde specialist | 38.0% | 71.8% | 27.3% | 80.1% |
| SVM + per-batch std + 10 labels per gas | 96.2% | 93.7% | 98.4% | 94.7% |
| SVM + per-batch std + 10 per gas, 20 Acetaldehyde | 99.4% | 93.1% | 99.4% | 95.2% |

Without labels nothing fixes it, and per-batch standardisation even makes it worse, although overall accuracy rises.
Class weights barely change this SVM: with C=100 the training batches are almost separable, so only 7 of its 213
support vectors sit at the limit where class weights act.
Labels from the new batches do: 10 per gas plus 20 Acetaldehyde finds almost all of it (tier-3 rows: mean of 5 seeds).

<p align="center"><img src="results/step13_methods.png" alt="Drift adaptation, validation vs test" width="80%"></p>

### Robustness checks (Step 14)

Four checks of whether the findings above hold up. Rows inside the batch files are not in time order (batches 1-6 are
sorted by gas, batch 10 is shuffled), so checks that need the order of recordings inside a batch are replaced by ones
that do not.

**Honest intervals: resample whole batches.** Recordings in one batch share sensor state, so the row bootstrap of Step 10
is far too narrow. Resampling the seven test batches instead asks how much each result depends on which batches were
tested.

| Model | Pooled accuracy | 95% interval, rows resampled | 95% interval, batches resampled | Difference vs SVM, batches resampled |
|---|---|---|---|---|
| Soft vote | 75.3% | 74.4-76.2 | 66.1-87.6 | +2.1 (-4.3 to +10.8) |
| SVM | 73.2% | 72.4-74.0 | 55.8-89.2 | |
| Stack (LR) | 70.2% | 69.2-71.1 | 60.2-82.9 | -3.0 (-12.0 to +5.3) |
| Random Forest + PCA | 69.5% | 68.6-70.5 | 58.3-83.0 | -3.7 (-12.3 to +3.5) |
| kNN + PCA | 64.9% | 63.9-65.9 | 60.2-71.3 | -8.3 (-19.3 to +4.8) |
| Naive Bayes + PCA | 51.4% | 50.4-52.4 | 37.1-66.2 | -21.8 (-29.3 to -16.8) |
| Decision Tree | 49.7% | 48.8-50.7 | 38.9-68.8 | -23.5 (-33.8 to -15.5) |

With honest intervals, only Naive Bayes and the Decision Tree are clearly worse than the SVM; the soft vote, the stack,
Random Forest, the boosting models and kNN cannot be told apart from it on seven batches.

**Per-batch standardisation from a small sample.** The full method needs a whole new batch before it predicts anything.
Here each new batch is scaled with the statistics of only some of its recordings, drawn at random (mean of 10 seeds).

| Recordings used per new batch | Validation | Test, batches 4-10 | Worst seed | Batch 10 |
|---|---|---|---|---|
| 10 | 79.8% | 69.4% | 59.8% | 53.5% |
| 25 | 83.4% | 76.4% | 73.8% | 58.9% |
| 50 | 87.1% | 77.3% | 75.1% | 61.5% |
| 100 | 88.7% | 78.2% | 76.8% | 62.3% |
| 200 | 89.7% | 78.4% | 77.3% | 61.8% |
| all | 90.6% | 79.2% | 79.2% | 63.3% |

About 100 unlabelled recordings per batch keep most of the gain; with 10 it is worse than the strict SVM (73.2%).

**MLP and CORAL tuned on the same validation folds as the SVM.** This removes the advantage the strict SVM had in the
Step 13 validation check.

| Method | Validation, untuned | Validation, tuned | Test, tuned |
|---|---|---|---|
| MLP, tuned on validation | 80.5% | 81.1% | 77.2% |
| SVM + CORAL, tuned on validation | 80.3% | 88.5% | 77.5% |
| SVM (strict), for reference | | 88.6% | 73.2% |

Tuning makes CORAL almost tie the strict SVM on validation, but neither beats it there, so neither counts as an
improvement. The Step 13 verdict stands.

**Other training periods.** The tuned models (Step 8 settings) and the per-batch-standardised SVM, trained on six
different sets of early batches. Batch 3 helped choose the settings, so it is never a test batch.

| Training batches | Test batches | SVM (strict) | SVM rank among 7 strict models | SVM + per-batch std | SVM (strict), mean over batches | SVM + per-batch std, mean over batches |
|---|---|---|---|---|---|---|
| 1-2 | 4-10 | 50.2% | 5 | 74.7% | 58.5% | 76.5% |
| 1-3 | 4-10 | 73.2% | 1 | 79.2% | 81.4% | 80.9% |
| 2-3 | 4-10 | 71.2% | 1 | 78.1% | 80.9% | 79.1% |
| 1-4 | 5-10 | 73.3% | 1 | 77.5% | 80.4% | 78.0% |
| 2-4 | 5-10 | 71.1% | 1 | 77.8% | 81.5% | 78.2% |
| 1-5 | 6-10 | 72.5% | 1 | 77.3% | 75.6% | 73.0% |

The findings mostly hold. In 48 of 48 runs a model scores lower on batches 8-10 than on its first test batches before
batch 8 (at most three).
The strict SVM is the best strict model in 5 of 6 periods; trained on batches 1-2 only, it falls to 50.2%. Per-batch
standardisation wins over all test recordings in every period, but averaged over batches only in 1 of 6, the same
pattern as in Step 13.

<p align="center"><img src="results/step14_robustness.png" alt="Batch-level intervals and training periods" width="80%"></p>

## Conclusions

1. **Drift is real and large.** The tuned SVM falls from 90.3% to 55.4%. Errors are structured: the SVM tends to label other
   gases as Ethanol, and Ammonia holds up best: only 1 of the 10 analysed models finds fewer than 40% of Ammonia recordings in any batch,
   against 4 to 8 of 10 for each other gas.
2. **The best honest accuracy is about 73-75%, with wide uncertainty.** The soft vote leads when pooled, but batch by
   batch it wins 3, loses 3 and ties 1 against the SVM, so it is not clearly better. Resampling whole batches (Step 14)
   gives the SVM a 95% interval of 55.8-89.2%, and only Naive Bayes and the Decision Tree are clearly worse than it.
3. **PCA and LDA are not a drift fix.** They help weak models and hurt strong ones.
4. **Boosting and tuning did not help clearly.** 5 of 21 model/reduction combinations improved, 7 stayed about the same and
   9 got worse. Validation on nearby batches does not predict later batches (correlation -0.14; validation was on average
   8.9 points too optimistic), itself a sign of drift.
5. **Stacking did not beat the best single model** (69.5% vs 73.2%).
6. **Random splits mislead**: 96-99% instead of 50-73%, the drift disappears and the ranking changes.
7. **Recent labels help most** (protocol C: 85-89%), but they cost labelling every new batch.
8. **Adapting to each new batch helps** (bonus, Step 13): per-batch standardisation lifts the SVM to 79.2% over all test
   recordings without labels (but not on the per-batch average), and 25 random labels per batch reach 92.1%, against
   82.3% for protocol C on the same measure (94.1% vs 88.6% averaged over batches), with far fewer labels.
9. **The drift findings survive other training periods** (Step 14): accuracy falls on later batches in 48 of 48 runs,
   and the SVM is the best strict model in 5 of 6 periods. Per-batch standardisation works from about 100 unlabelled
   recordings per batch, but it only beats the strict SVM averaged over batches in 1 of 6 periods.

## Interactive dashboard

**Try it live: [drift-robust-gas-classification-ctfmhqhdsllpacizk7et2b.streamlit.app](https://drift-robust-gas-classification-ctfmhqhdsllpacizk7et2b.streamlit.app/)**

Nine pages that show the data, the drift and every result, plus a live demo where you pick a recording and watch the models
vote. It only displays what the notebooks measured, and tests check the numbers on screen.

```bash
pip install -r app/requirements.txt
streamlit run app/app.py
```

**Overview** · **Data explorer** · **Drift viewer** (with a play-through-time animation) · **Model results** ·
**Per-batch drift** · **Confusion matrices** · **Random vs time-aware** · **Live demo** · **Limitations**

<details>
<summary><b>Deploy it for free</b></summary>

<br>

Streamlit apps need a long-running server, so serverless hosts such as Vercel cannot run them. Use
[Streamlit Community Cloud](https://share.streamlit.io):

1. Sign in with GitHub and choose **Create app → Deploy a public app from GitHub**.
2. Pick this repository, branch `main`, main file path `app/app.py`.
3. Under **Advanced settings** choose Python 3.11, pick a `*.streamlit.app` address and deploy.

The cloud installs the lean `app/requirements.txt` and reads the theme from `.streamlit/config.toml`.

</details>

## Limitations

- **Five gases only**, and **gas concentration is not in this dataset version**, so it cannot be separated from ageing.
- **Small, uneven batches** (4, 5, 8 and 9 hold 161-369 samples) make single-batch results noisy.
- **The headline uses one training period** (batches 1-3); Step 14 checks five others with the same settings, which
  were tuned on batches 1-3.
- **Row-bootstrap intervals are too narrow** (Steps 10-13): they treat recordings of one batch as independent. Step 14's
  batch-level intervals are honest but rest on only seven test batches.
- **No time order inside batches**: the files sort or shuffle recordings, so drift within a batch, a moving-block
  bootstrap and recording-by-recording adaptation cannot be studied.
- **Tuned models are tested on protocol A only**, with small grids and two validation folds.
- **The Dennler et al. (2022) baseline check** needs raw data this version lacks; we use clearly labelled proxies.
- **One dataset**: conclusions may not transfer to other sensors.

## How to reproduce

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

python -m pytest tests -q            # full test suite, about 2 min
streamlit run app/app.py             # dashboard
```

All results are already saved in `results/`, so the notebooks can simply be read. To recompute them:

<details>
<summary><b>Step-by-step commands and timings</b></summary>

<br>

```bash
# Steps 1-7: run notebooks/01 ... 07 in order (07 is the slowest, about 25 min)
python scripts/run_step8_tuning.py     # about 19 min, then notebook 08
python scripts/run_step9_stacking.py   # about 11 min, then notebook 09
#                                        notebook 10 (under a minute)
python scripts/run_step11_leakage.py   # about 20 min, then notebook 11
python scripts/run_step12_repeats.py   # about 6 min,  then notebook 12
python scripts/run_step13_adaptation.py # about 6 min, then notebook 13 (bonus)
python scripts/run_step14_robustness.py # about 3 min, writes results/step14_* (no notebook)
python scripts/make_dashboard_cache.py # optional: saves predictions so the dashboard opens faster
```

Run a notebook headless with `jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb`.
On Windows, if pip fails with `CERTIFICATE_VERIFY_FAILED`, add `--use-feature=truststore`.

</details>

## Where each number comes from

<details>
<summary><b>Result files and notebooks</b></summary>

<br>

| Numbers | File | Notebook |
|---|---|---|
| Sample counts, Toluene removal | `results/step1_counts.csv` | `01_load_and_remove_toluene` |
| Untuned baselines | `results/step5_protocolA_summary.csv` | `05_baseline_models` |
| PCA / LDA effect | `results/step6_protocolA_summary.csv` | `06_pca_lda` |
| Boosting | `results/step7_protocolA_all_models.csv` | `07_boosting` |
| Tuning, validation vs test | `results/step8_tuned_vs_untuned.csv` | `08_tuning` |
| Stack, soft vote, protocol C | `results/step9_stacking_metrics.csv` | `09_stacking` |
| Intervals, near vs far, confusion matrices | `results/step10_*.csv` | `10_drift_analysis` |
| Random vs time-aware split | `results/step11_accuracy_table.csv` | `11_leakage_check` |
| Headline mean ± std over seeds | `results/step12_summary.csv` | `12_repeated_runs` |
| Bonus drift adaptation | `results/step13_summary.csv` | `13_drift_adaptation` |
| Bonus: using 25 labels per batch | `results/step13_label_strategies_summary.csv` | `13_drift_adaptation` |
| Bonus: Acetaldehyde | `results/step13_acetaldehyde_summary.csv` | `13_drift_adaptation` |
| Robustness checks | `results/step14_*_summary.csv`, `results/step14_batch_bootstrap.csv` | `scripts/run_step14_robustness.py` |

</details>

<details>
<summary><b>Repository layout</b></summary>

<br>

```
app/          Streamlit dashboard (app.py, design.py, ui.py, motion.js, views/)
data_raw/     original batch files, never modified
notebooks/    one notebook per step, with results and explanations
results/      saved tables, figures and trained models
scripts/      long-running, restartable jobs (Steps 8, 9, 11, 12, 13, 14)
src/          loading, evaluation, models, tuning, stacking, analysis, adaptation, robustness
tests/        data-integrity, leakage, dashboard and README-number checks
```

</details>

## References

<details>
<summary><b>Six papers</b></summary>

<br>

1. Vergara, A., Vembu, S., Ayhan, T., Ryan, M. A., Homer, M. L., & Huerta, R. (2012). Chemical gas sensor drift compensation
   using classifier ensembles. *Sensors and Actuators B: Chemical*, 166-167, 320-329. doi:10.1016/j.snb.2012.01.074
2. Vergara, A. (2012). Gas Sensor Array Drift Dataset. UCI Machine Learning Repository. doi:10.24432/C5RP6W
3. Dennler, N., Rastogi, S., Fonollosa, J., van Schaik, A., & Schmuker, M. (2022). Drift in a popular metal oxide sensor
   dataset reveals limitations for gas classification benchmarks. *Sensors and Actuators B: Chemical*, 361, 131668.
   doi:10.1016/j.snb.2022.131668
4. Zhang, Y., et al. (2022). TDACNN: Target-domain-free domain adaptation CNN for drift compensation in gas sensors.
   *Sensors and Actuators B: Chemical*, 361, 131739. doi:10.1016/j.snb.2022.131739
5. Jiang, K., et al. (2025). Gas sensor drift compensation using semi-supervised ensemble classifiers with multi-level
   features and center loss. *ACS Sensors*, 10(4), 2906-2918. doi:10.1021/acssensors.4c03655
6. Lin, J., & Zhan, X. (2026). Sensor-drift compensation in electronic-nose-based gas recognition using knowledge
   distillation. *Informatics*, 13(1), 15. doi:10.3390/informatics13010015

</details>
