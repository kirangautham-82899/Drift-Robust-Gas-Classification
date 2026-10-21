import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from common import batch_counts, load_data, read_result

df = load_data()
counts = batch_counts()

st.title("Can a model keep recognising gases as its sensors age?")
st.markdown(
    "Gas sensors change as they get older, so a program trained on early recordings becomes less accurate on later ones. "
    "This is called **sensor drift**. This dashboard shows the data, the drift, and how well classical machine-learning "
    "models cope with it. Every chart has a short **\"What to notice\"** box in plain English."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Samples", f"{len(df):,}", help="After removing Toluene (13,910 before).")
c2.metric("Gases", df["gas"].nunique(), help=", ".join(sorted(df["gas"].unique())))
c3.metric("Batches", df["batch"].nunique(), help="Recording periods in time order, Jan 2008 - Feb 2011.")
c4.metric("Features", 128, help="16 sensors x 8 features each.")

st.subheader("The problem in plain English")
st.markdown(
    "- A set of 16 gas sensors measures a gas and a program learns which gas it is.\n"
    "- Months later the same sensors give *different numbers* for the same gas, because they have aged.\n"
    "- A program that learned from the old numbers then makes more mistakes. How many more, and can we reduce them?"
)

st.subheader("How we test it fairly")
st.markdown(
    "The recordings come in 10 batches, in time order. A random split mixes the batches, which is too easy. "
    "We train on the **early** batches and test on the **later** ones, like an exam on material the model has never seen:"
)
st.markdown(
    "**Train:** :green-background[ batch 1 ] :green-background[ batch 2 ] :green-background[ batch 3 ] "
    "&nbsp;&nbsp;&nbsp; **Test:** :orange-background[ batch 4 ] :orange-background[ batch 5 ] :orange-background[ batch 6 ] "
    ":orange-background[ batch 7 ] :orange-background[ batch 8 ] :orange-background[ batch 9 ] :orange-background[ batch 10 ]"
)

st.subheader("The pipeline")
steps = [("1. Load", "10 batch files, 128 numbers per recording"), ("2. Clean", "Toluene removed, checks on every file"),
         ("3. Scale", "so large and small numbers count equally"), ("4. Reduce (optional)", "PCA or LDA squeezes 128 numbers into a few"),
         ("5. Classify", "kNN, SVM, trees, boosting, stacking"), ("6. Test on later batches", "accuracy per batch")]
cols = st.columns(len(steps))
for col, (title, text) in zip(cols, steps):
    with col.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(text)

with st.expander("Why is Toluene removed?"):
    st.markdown(
        "Toluene is missing from batches 3, 4 and 5 and has only 5 samples in batch 2, so the set of gases would change from "
        "batch to batch. Removing it (on the teacher's instruction) keeps the same five gases in every batch. "
        "The original files are never changed; Toluene is dropped in code."
    )
    st.dataframe(counts, width="stretch")

st.subheader("Headline result")
summary = read_result("step12_summary.csv").rename(columns={"Unnamed: 0": "model"}).set_index("model")
near_far = read_result("step10_near_far.csv").set_index("model")
svm, vote = summary.loc["SVM"], summary.loc["Soft vote"]
m1, m2, m3 = st.columns(3)
m1.metric("Best single model (SVM)", f"{svm['mean accuracy %']:.1f}%", help="Pooled accuracy on test batches 4-10, trained on batches 1-3.")
m2.metric("SVM on near batches 4-6", f"{near_far.loc['SVM', 'near (4-6) %']:.1f}%")
m3.metric("SVM on far batches 8-10", f"{near_far.loc['SVM', 'far (8-10) %']:.1f}%",
          delta=f"{near_far.loc['SVM', 'change (points)']:.1f} points")
st.caption(
    f"A plain average of five models (soft vote) scores {vote['mean accuracy %']:.1f}% ± {vote['std (points)']:.2f}. "
    f"The SVM is right about {near_far.loc['SVM', 'near (4-6) %']:.0f}% of the time on batches close to the training period "
    f"and only {near_far.loc['SVM', 'far (8-10) %']:.0f}% on the far ones. "
    "These numbers come from the saved results of the notebooks."
)

st.divider()
st.caption("Kiran Gautham (CB.AI.P2DSC26013) · 24DS601 Machine Learning · Amrita Vishwa Vidyapeetham, Coimbatore · "
           "Data: UCI Gas Sensor Array Drift Dataset (Vergara et al., 2012)")
