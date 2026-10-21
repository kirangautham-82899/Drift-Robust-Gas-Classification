import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from common import GASES, confusion, notice, predictions

preds = predictions()
models = [c for c in preds.columns if c not in ("batch", "label")]
GROUPS = {"All test batches (4-10)": list(range(4, 11)), "Near batches (4-6)": [4, 5, 6], "Far batches (8-10)": [8, 9, 10],
          **{f"Batch {b}": [b] for b in range(4, 11)}}

st.title("Confusion matrices")
st.caption("Where do the mistakes happen? Each row is the **true** gas, each column the gas the model **predicted**. "
           "The numbers are percentages of that true gas, so a bright diagonal means correct.")

c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
first = c1.selectbox("Model", models, index=models.index("SVM"))
second = c2.selectbox("Compare with", ["(none)"] + models, index=1 + models.index("Soft vote"))
group = c3.selectbox("Test batches", list(GROUPS), index=0)
as_counts = c4.checkbox("Show counts", value=False)
rows = preds[preds["batch"].isin(GROUPS[group])]
st.caption(f"{len(rows):,} test recordings in this selection.")


def matrix_figure(model):
    cm = confusion(rows["label"], rows[model])
    shown = cm if as_counts else cm / cm.sum(axis=1, keepdims=True) * 100
    fig = px.imshow(shown, x=GASES, y=GASES, text_auto=".0f", color_continuous_scale="Blues",
                    zmin=0, zmax=None if as_counts else 100, aspect="auto",
                    labels={"x": "predicted gas", "y": "true gas", "color": "count" if as_counts else "% of true gas"})
    fig.update_layout(title=f"{model}  (accuracy {(rows['label'] == rows[model]).mean() * 100:.1f}%)", height=430,
                      margin=dict(l=10, r=10, t=50, b=10), coloraxis_showscale=False)
    return fig, cm


chosen = [first] + ([second] if second != "(none)" and second != first else [])
cols = st.columns(len(chosen))
matrices = {}
for col, m in zip(cols, chosen):
    fig, cm = matrix_figure(m)
    matrices[m] = cm
    col.plotly_chart(fig, width="stretch")

cm = matrices[first]
rates = cm / cm.sum(axis=1, keepdims=True)
off = rates.copy()
np.fill_diagonal(off, 0)
t, p = np.unravel_index(off.argmax(), off.shape)
recall = pd.Series(np.diag(rates) * 100, index=GASES)
notice(
    f"For **{first}** on {group.lower()}: the most common mistake is **{GASES[t]}** recordings being labelled **{GASES[p]}** "
    f"({off[t, p] * 100:.0f}% of the {int(cm[t].sum()):,} {GASES[t]} recordings). The gas it finds best is {recall.idxmax()} "
    f"({recall.max():.0f}% found); the one it misses most is {recall.idxmin()} ({recall.min():.0f}% found). Small batches contain few recordings "
    "of some gases, so single cells there are noisy."
)

# ------------------------------------------------------------------------------------- recall per gas and batch
st.header("How often is each gas found in each batch?")
model_r = st.selectbox("Model", models, index=models.index(first), key="recall_model")
table = {}
for b in range(4, 11):
    d = preds[preds["batch"] == b]
    cmb = confusion(d["label"], d[model_r])
    table[b] = np.diag(cmb) / cmb.sum(axis=1) * 100
heat = pd.DataFrame(table, index=GASES)
fig = px.imshow(heat, text_auto=".0f", color_continuous_scale="RdYlGn", zmin=0, zmax=100, aspect="auto",
                labels={"x": "Test batch", "y": "", "color": "% found"})
fig.update_xaxes(type="category")
fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, width="stretch")
worst_gas, worst_batch = heat.stack().idxmin()[0], heat.stack().idxmin()[1]
notice(
    f"Different gases fail in different batches. For {model_r}, the weakest cell is {worst_gas} in batch {worst_batch} "
    f"({heat.stack().min():.0f}% found). The most stable gas for this model is {heat.min(axis=1).idxmax()} (never below "
    f"{heat.min(axis=1).max():.0f}% found). Cells based on few recordings (for example Ethylene in batches 8 and 9, with 30 and 55 "
    "recordings) are noisy."
)
