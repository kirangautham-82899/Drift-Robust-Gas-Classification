import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st
from common import read_result
from ui import callout, cards, hero, section

summary = read_result("step12_summary.csv").rename(columns={"Unnamed: 0": "model"}).set_index("model")
near_far = read_result("step10_near_far.csv").set_index("model")
svm, vote = summary.loc["SVM"], summary.loc["Soft vote"]
nf = near_far.loc["SVM"]

hero("About", "Limitations and about this project",
     "What the results can and cannot tell you, how the project was built, and where the data comes from.", compact=True)

section("What we found")
callout(f"Gas-sensor drift is **real and large**: the best single model (SVM) falls from {nf['near (4-6) %']:.1f}% on batches close to the training period to "
        f"{nf['far (8-10) %']:.1f}% on far batches, and its overall accuracy on later batches is **{svm['mean accuracy %']:.1f}%**. A plain average of five models scores "
        f"{vote['mean accuracy %']:.1f}% ± {vote['std (points)']:.2f}, mostly because of one batch. Squeezing the features (PCA, LDA), boosting, tuning and stacking did "
        "not clearly beat the SVM. A random split makes models look almost perfect (96-99%) and hides the drift.", title="Summary")

section("What this does not tell you", "Read the numbers with these limits in mind.")
limits = [
    ("Five gases only", "Toluene was excluded on the teacher's instruction, so the results apply to Ethanol, Ethylene, Ammonia, Acetaldehyde and Acetone."),
    ("The Dennler baseline check is a proxy", "The raw pre-exposure baseline of Dennler et al. (2022) is not in this dataset version, so we use proxies and label them as proxies."),
    ("Small, uneven batches", "Batches 4, 5, 8 and 9 have 161-369 recordings (as few as 12 per gas), so their individual results are noisy, and the gas mix differs between batches."),
    ("Gas concentration is missing", "Concentration changed between batches and is not in this dataset version, so it cannot be separated from sensor ageing."),
    ("One training period", "Only batches 1-3 are used as training data. Seeds and bootstrap intervals do not capture the effect of choosing a different period."),
    ("Intervals are too narrow", "The bootstrap treats recordings as independent, but recordings made close together are alike, so the true uncertainty is larger."),
    ("Tuned models: protocol A only", "Protocols B and C test batches 2 and 3, which tuning used for validation. The stack has no protocol B."),
    ("No perfect accuracy", "Under drift no model keeps more than 67% on the far batches. One dataset, so conclusions may not transfer to other sensors."),
]
for i in range(0, len(limits), 2):
    cards(limits[i:i + 2], numbered=False)
    st.write("")

section("Objectives from the Zeroth review")
st.dataframe(pd.DataFrame([
    ("Preprocess (scaling, cleaning)", "Done", "Data explorer, Step 1, 2, 4"),
    ("Analyse drift batch by batch", "Done", "Drift viewer, Per-batch drift"),
    ("PCA and LDA: effect on drift robustness", "Done", "Model results"),
    ("kNN, SVM, Decision Tree, Random Forest, Naive Bayes, boosting", "Done", "Model results"),
    ("Stacking ensemble", "Done (did not beat the SVM)", "Model results, Per-batch drift"),
    ("Time-aware evaluation, not only random splits", "Done", "Random vs time-aware"),
    ("Accuracy, precision, recall, F1, confusion matrix, degradation over time", "Done", "Model results, Confusion matrices, Per-batch drift"),
], columns=["Objective", "Status", "Where to see it"]), width="stretch", hide_index=True)

section("How it was built")
st.markdown(
    "- Every step of the project is a notebook with results and plain-English explanations; long computations are scripts that save their progress.\n"
    "- Scalers, PCA, LDA and tuning only ever see the training batches, and tests prove the test batches cannot change them.\n"
    "- This dashboard only displays results saved by those notebooks. Automated tests check that its pages run and that the numbers it shows match the saved results "
    "and the numbers in the project README.\n"
    "- Code, notebooks, results and tests: [GitHub repository](https://github.com/kirangautham-82899/Drift-Robust-Gas-Classification)."
)

section("Data and references")
st.markdown(
    "1. Vergara, A., Vembu, S., Ayhan, T., Ryan, M. A., Homer, M. L., & Huerta, R. (2012). Chemical gas sensor drift compensation using classifier ensembles. "
    "*Sensors and Actuators B: Chemical*, 166-167, 320-329.\n"
    "2. Vergara, A. (2012). Gas Sensor Array Drift Dataset. UCI Machine Learning Repository. [doi:10.24432/C5RP6W](https://doi.org/10.24432/C5RP6W)\n"
    "3. Dennler, N., Rastogi, S., Fonollosa, J., van Schaik, A., & Schmuker, M. (2022). Drift in a popular metal oxide sensor dataset reveals limitations for gas "
    "classification benchmarks. *Sensors and Actuators B: Chemical*, 361, 131668.\n"
    "4. Zhang, Y., et al. (2022). TDACNN: Target-domain-free domain adaptation CNN for drift compensation in gas sensors. *Sensors and Actuators B: Chemical*, 361, 131739."
)
st.caption("Kiran Gautham (CB.AI.P2DSC26013) · M.Tech Data Science, 24DS601 Machine Learning · Amrita Vishwa Vidyapeetham, Coimbatore")
