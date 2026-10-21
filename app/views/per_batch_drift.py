import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from common import MODEL_COLORS, notice, read_result, rgba

ci = read_result("step10_per_batch_ci.csv")
for col in ["accuracy", "acc_low", "acc_high", "macro_f1", "f1_low", "f1_high"]:
    ci[col] = ci[col].astype(float) * 100
per = ci[ci["group"].str.startswith("batch")].assign(batch=lambda d: d["group"].str.split().str[1].astype(int))
models = list(ci["model"].unique())

st.title("Per-batch drift")
st.caption("How does accuracy change as the test batches get further from the training batches (1-3)? "
           "Shaded bands and bars show 95% bootstrap confidence intervals.")
st.warning("The intervals treat every recording as independent, but recordings made close together are alike, so the real "
           "uncertainty is larger than the bands show. Batches 4, 5, 8 and 9 are small (161-369 samples) and noisy.",
           icon=":material/warning:")

# ------------------------------------------------------------------------------------- per batch lines
st.header("Accuracy on each test batch")
c1, c2, c3 = st.columns([3, 1, 1])
chosen = c1.multiselect("Models", models, default=["SVM", "Soft vote", "Stack (LR)", "Random Forest + PCA", "kNN + PCA"])
metric = c2.radio("Score", ["Accuracy", "Macro-F1"])
bands = c3.checkbox("Show confidence bands", value=True)
col, lo, hi = ("accuracy", "acc_low", "acc_high") if metric == "Accuracy" else ("macro_f1", "f1_low", "f1_high")
if not chosen:
    st.warning("Pick at least one model.")
else:
    fig = go.Figure()
    for m in chosen:
        d = per[per["model"] == m].sort_values("batch")
        colour = MODEL_COLORS.get(m, "#888888")
        if bands:
            fig.add_trace(go.Scatter(x=list(d["batch"]) + list(d["batch"])[::-1], y=list(d[hi]) + list(d[lo])[::-1],
                                     fill="toself", fillcolor=rgba(colour, 0.15), line=dict(width=0), hoverinfo="skip",
                                     showlegend=False))
        fig.add_trace(go.Scatter(x=d["batch"], y=d[col], mode="lines+markers", name=m, line=dict(color=colour)))
    fig.update_xaxes(title="Test batch (time order, trained on batches 1-3)", dtick=1)
    fig.update_yaxes(title=f"{metric} (%)", range=[0, 102])
    fig.update_layout(height=470)
    st.plotly_chart(fig, width="stretch")
    sub = per[per["model"].isin(chosen)]
    worst = sub.loc[sub[col].idxmin()]
    best_b10 = per[(per["batch"] == 10)].sort_values(col).iloc[-1]
    notice(
        f"Accuracy falls on later batches for every model, but not smoothly: the lowest point among the selected lines is "
        f"{worst['model']} on batch {worst['batch']} ({worst[col]:.1f}%). Batch 10 is hard for every model ({per[per['batch'] == 10][col].min():.0f}% to "
        f"{per[per['batch'] == 10][col].max():.0f}%; best: {best_b10['model']}). The bands are widest on the small batches 4, 8 and 9."
    )

# ------------------------------------------------------------------------------------- heatmap
st.header("Every model on every batch")
order = list(ci[ci["group"] == "4-10 (all)"].sort_values("accuracy", ascending=False)["model"])
heat = per.pivot(index="model", columns="batch", values="accuracy").loc[order]
fig = px.imshow(heat, text_auto=".0f", color_continuous_scale="Viridis", zmin=20, zmax=100, aspect="auto",
                labels={"x": "Test batch", "y": "", "color": "accuracy %"})
fig.update_xaxes(type="category")
fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, width="stretch")
notice("Models are sorted by their pooled accuracy on batches 4-10. The best model is different in different batches, and no model "
       "is strong on batch 10.")

# ------------------------------------------------------------------------------------- near vs far
st.header("How much accuracy is lost as time passes? Batches 4-6 compared with 8-10")
nf = read_result("step10_near_far.csv").set_index("model")
for c in ["near (4-6) %", "far (8-10) %", "change (points)", "CI low", "CI high"]:
    nf[c] = nf[c].astype(float)
nf = nf.sort_values("change (points)", ascending=False)
fig = go.Figure(go.Bar(x=nf["change (points)"], y=nf.index, orientation="h",
                       marker_color=[MODEL_COLORS.get(m, "#888888") for m in nf.index],
                       error_x=dict(type="data", symmetric=False, array=nf["CI high"] - nf["change (points)"],
                                    arrayminus=nf["change (points)"] - nf["CI low"]),
                       text=[f"{v:.1f}" for v in nf["change (points)"]], textposition="outside"))
fig.update_xaxes(title="Change in accuracy from near to far batches (percentage points, 95% CI)", range=[-45, 3])
fig.update_layout(height=430, margin=dict(l=10, r=30, t=10, b=10))
st.plotly_chart(fig, width="stretch")
svm, vote = nf.loc["SVM"], nf.loc["Soft vote"]
notice(
    f"The tuned SVM is best on the near batches ({svm['near (4-6) %']:.1f}%) but loses {abs(svm['change (points)']):.1f} points on the far ones "
    f"({svm['far (8-10) %']:.1f}%). The soft vote loses {abs(vote['change (points)']):.1f} points and has the best accuracy on the far batches "
    f"({vote['far (8-10) %']:.1f}%). No model keeps more than {nf['far (8-10) %'].max():.0f}% on the far batches. Part of the drop may come from the "
    "changing gas mix and from gas concentration, which this dataset does not contain."
)

# ------------------------------------------------------------------------------------- real differences
st.header("Which differences are real? Compared with the tuned SVM, pooled over batches 4-10")
paired = read_result("step10_paired_vs_svm.csv").set_index("model")
paired = paired.rename(columns={"accuracy %": "accuracy (%)", "difference vs SVM (points)": "difference vs SVM (points)",
                                "CI low": "CI low", "CI high": "CI high", "verdict": "verdict"})
st.dataframe(paired.astype({"accuracy (%)": float}), width="stretch")
acc = per.pivot(index="model", columns="batch", values="accuracy").round(1)
wins, losses, ties = int((acc.loc["Soft vote"] > acc.loc["SVM"]).sum()), int((acc.loc["Soft vote"] < acc.loc["SVM"]).sum()), int((acc.loc["Soft vote"] == acc.loc["SVM"]).sum())
notice(
    "A model is only called better or worse when its whole confidence interval is above or below zero. Tuning the SVM made no clear "
    "difference (untuned SVM: not clearly different), and every other model is clearly worse than the tuned SVM except the soft vote. "
    f"**Caution about the soft vote:** its interval excludes zero only if recordings are independent, which they are not. Batch by batch it beats the "
    f"SVM on {wins} batches, loses on {losses} and ties on {ties}, and its pooled lead comes mostly from batch 10, so we do not claim it is generally better."
)
