import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from common import batch_counts, load_data, read_result
from ui import batch_strip, callout, cards, hero, kpis, link, section, stepper

df = load_data()
counts = batch_counts()
summary = read_result("step12_summary.csv").rename(columns={"Unnamed: 0": "model"}).set_index("model")
near_far = read_result("step10_near_far.csv").set_index("model")
exams = read_result("step11_accuracy_table.csv").set_index("model")

svm, vote = summary.loc["SVM"], summary.loc["Soft vote"]
nf = near_far.loc["SVM"]
random_claim = float(exams.loc["SVM", "random, same sizes"])
honest = float(exams.loc["SVM", "time-aware (A)"])

hero("M.Tech Data Science · 24DS601 Machine Learning",
     "Can a model keep recognising gases as its sensors age?",
     "Gas sensors drift: months later the same gas gives different readings. We trained classical machine-learning models on early "
     "recordings and tested them on later ones to measure how much accuracy is lost, and why a random split would hide it.",
     chips=[f"{len(df):,} recordings", f"{df['gas'].nunique()} gases", f"{df['batch'].nunique()} batches over 36 months", "16 sensors · 128 features"])

kpis([
    dict(label="Best single model", value=f"{svm['mean accuracy %']:.1f}%", note="SVM, tested on batches 4-10", accent=True),
    dict(label="Close to the training period", value=f"{nf['near (4-6) %']:.1f}%", note="SVM on batches 4-6"),
    dict(label="Far from the training period", value=f"{nf['far (8-10) %']:.1f}%", note="SVM on batches 8-10",
         tag=f"▼ {abs(nf['change (points)']):.1f} points lost", tone="bad"),
    dict(label="What a random split claims", value=f"{random_claim:.1f}%", note="the same SVM, with mixed batches",
         tag=f"inflated by {random_claim - honest:.1f} points", tone="warn"),
])

section("The story in three steps")
cols = cards([
    ("Sensors age", "The same gas produces different numbers months later. Open the drift viewer to see it in the raw readings and on a map."),
    ("Test on the future", "We train on batches 1-3 and test on batches 4-10, like an exam on material the model has never seen."),
    ("Measure the damage", f"Accuracy falls from about {nf['near (4-6) %']:.0f}% to about {nf['far (8-10) %']:.0f}% for the best single model. "
                           "Which models cope best, and where do the mistakes happen?"),
])
with cols[0]:
    link("views/drift_viewer.py", "Open the drift viewer →")
with cols[1]:
    link("views/random_vs_time_aware.py", "Why random splits mislead →")
with cols[2]:
    link("views/model_results.py", "See the model results →")

section("How we test fairly", "Recordings come in 10 batches, in time order. A random split mixes them, which is far too easy.")
batch_strip([1, 2, 3], [4, 5, 6, 7, 8, 9, 10])

section("The pipeline")
stepper([("Load", "10 batch files, 128 numbers per recording"), ("Clean", "Toluene removed, checks on every file"),
         ("Scale", "so large and small numbers count equally"), ("Reduce (optional)", "PCA or LDA squeeze 128 numbers into a few"),
         ("Classify", "kNN, SVM, trees, boosting, stacking"), ("Test later", "accuracy batch by batch")])

section("What we found", "The short version; every page has the evidence.")
callout(f"The best single model, the SVM, is right **{svm['mean accuracy %']:.1f}%** of the time on later batches. A plain average of five models "
        f"(the soft vote) scores **{vote['mean accuracy %']:.1f}% ± {vote['std (points)']:.2f}**, but mostly because of one batch, so we do not call it "
        "a general win. Squeezing the features (PCA, LDA), boosting, tuning and stacking did not give a clear improvement over the SVM.")

section("Explore")
a, b, c = st.columns(3)
with a:
    link("views/data_explorer.py", "Data explorer: the 12,077 recordings")
    link("views/drift_viewer.py", "Drift viewer: how the readings change")
with b:
    link("views/model_results.py", "Model results: the leaderboard")
    link("views/per_batch_drift.py", "Per-batch drift: with confidence intervals")
with c:
    link("views/confusion_matrices.py", "Confusion matrices: where mistakes happen")
    link("views/live_demo.py", "Live demo: try a recording yourself")

with st.expander("Why is Toluene removed?"):
    st.markdown(
        "Toluene is missing from batches 3, 4 and 5 and has only 5 samples in batch 2, so the set of gases would change from batch to batch. "
        "Removing it (on the teacher's instruction) keeps the same five gases in every batch. The original files are never changed; Toluene is "
        "dropped in code."
    )
    st.dataframe(counts, width="stretch")

st.caption("Kiran Gautham (CB.AI.P2DSC26013) · Amrita Vishwa Vidyapeetham, Coimbatore · Data: UCI Gas Sensor Array Drift Dataset (Vergara et al., 2012)")
