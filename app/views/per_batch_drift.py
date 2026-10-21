import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from common import ANALYSIS_MODELS, read_result
from design import cat, model_color, rgba, sequential_scale, style, tint, tok
from ui import callout, caution, figure, hero, section

ci = read_result("step10_per_batch_ci.csv")
for col in ["accuracy", "acc_low", "acc_high", "macro_f1", "f1_low", "f1_high"]:
    ci[col] = ci[col].astype(float) * 100
per = ci[ci["group"].str.startswith("batch")].assign(batch=lambda d: d["group"].str.split().str[1].astype(int))

hero("The results", "Per-batch drift",
     "How does accuracy change as the test batches get further from the training batches (1-3)? Shaded bands and bars show 95% bootstrap confidence intervals.",
     compact=True)
caution("The intervals treat every recording as independent, but recordings made close together are alike, so the real uncertainty is larger than the bands show. "
        "Batches 4, 5, 8 and 9 are small (161-369 samples) and noisy.")

# ------------------------------------------------------------------------------------- per batch lines
section("Accuracy on each test batch")
c1, c2, c3 = st.columns([3, 1, 1])
chosen = c1.multiselect("Models (up to 6)", ANALYSIS_MODELS, default=["SVM", "Soft vote", "Stack (LR)", "Random Forest + PCA", "kNN + PCA"], max_selections=6)
metric = c2.radio("Score", ["Accuracy", "Macro-F1"])
bands = c3.checkbox("Show confidence bands", value=True)
col, lo, hi = ("accuracy", "acc_low", "acc_high") if metric == "Accuracy" else ("macro_f1", "f1_low", "f1_high")
if not chosen:
    st.warning("Pick at least one model.")
else:
    fig = go.Figure()
    for m in chosen:
        d = per[per["model"] == m].sort_values("batch")
        colour = model_color(m)
        dash = "dash" if m == "SVM (untuned)" else "solid"
        if bands:
            fig.add_trace(go.Scatter(x=list(d["batch"]) + list(d["batch"])[::-1], y=list(d[hi]) + list(d[lo])[::-1], mode="lines", fill="toself",
                                     fillcolor=rgba(colour, 0.14), line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=d["batch"], y=d[col], mode="lines+markers", name=m, line=dict(color=colour, width=2, dash=dash),
                                 marker=dict(color=colour, size=8, line=dict(color=tok()["surface"], width=2))))
    style(fig, height=470, x_title="Test batch (time order, trained on batches 1-3)", y_title=f"{metric} (%)", hover="x unified")
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(range=[0, 102])
    table = per[per["model"].isin(chosen)].pivot(index="batch", columns="model", values=col).round(1)
    figure(fig, table)
    sub = per[per["model"].isin(chosen)]
    worst = sub.loc[sub[col].idxmin()]
    b10 = per[per["batch"] == 10]
    callout(f"Accuracy falls on later batches for every model, but not smoothly: the lowest point among the selected lines is **{worst['model']}** on batch "
            f"{worst['batch']} ({worst[col]:.1f}%). Batch 10 is hard for every model ({b10[col].min():.0f}% to {b10[col].max():.0f}%; best: "
            f"{b10.sort_values(col).iloc[-1]['model']}). The bands are widest on the small batches 4, 8 and 9.")

# ------------------------------------------------------------------------------------- heatmap
section("Every model on every batch", "Models are sorted by their pooled accuracy on batches 4-10.")
order = list(ci[ci["group"] == "4-10 (all)"].sort_values("accuracy", ascending=False)["model"])
heat = per.pivot(index="model", columns="batch", values="accuracy").loc[order]
fig = px.imshow(heat, text_auto=".0f", color_continuous_scale=sequential_scale(), zmin=20, zmax=100, aspect="auto",
                labels={"x": "Test batch", "y": "", "color": "accuracy %"})
fig.update_xaxes(type="category", side="top")
style(fig, height=440, legend=False)
fig.update_xaxes(showgrid=False, showline=False)
fig.update_yaxes(showgrid=False, showline=False)
figure(fig, heat.round(1))
callout("The best model is different in different batches, and no model is strong on batch 10.")

# ------------------------------------------------------------------------------------- near vs far
section("How much accuracy is lost as time passes?", "Batches 4-6 compared with batches 8-10, with 95% confidence intervals.")
nf = read_result("step10_near_far.csv").set_index("model")
for c in ["near (4-6) %", "far (8-10) %", "change (points)", "CI low", "CI high"]:
    nf[c] = nf[c].astype(float)
nf = nf.sort_values("change (points)", ascending=False)
fig = go.Figure(go.Bar(x=nf["change (points)"], y=nf.index, orientation="h", width=0.6, marker=dict(color=cat(0), line=dict(color=tok()["surface"], width=2)),
                       marker_cornerradius=4, error_x=dict(type="data", symmetric=False, array=nf["CI high"] - nf["change (points)"],
                                                           arrayminus=nf["change (points)"] - nf["CI low"], color=tok()["ink2"], thickness=1.5),
                       text=[f"{v:.1f}" for v in nf["change (points)"]], textposition="outside", textfont=dict(color=tok()["ink"]),
                       hovertemplate="%{y}: %{x:.1f} points<extra></extra>"))
style(fig, height=440, legend=False, x_title="Change in accuracy from near to far batches (percentage points)", grid_x=True)
fig.update_xaxes(range=[-46, 4])
fig.update_yaxes(showgrid=False)
figure(fig, nf[["near (4-6) %", "far (8-10) %", "change (points)", "CI low", "CI high"]])
svm, vote = nf.loc["SVM"], nf.loc["Soft vote"]
callout(f"The tuned SVM is best on the near batches (**{svm['near (4-6) %']:.1f}%**) but loses {abs(svm['change (points)']):.1f} points on the far ones "
        f"({svm['far (8-10) %']:.1f}%). The soft vote loses {abs(vote['change (points)']):.1f} points and has the best accuracy on the far batches "
        f"({vote['far (8-10) %']:.1f}%). No model keeps more than {nf['far (8-10) %'].max():.0f}% on the far batches.")
caution("Part of the drop may come from the changing gas mix and from gas concentration, which this dataset does not contain.")

# ------------------------------------------------------------------------------------- real differences
section("Which differences are real?", "Each model compared with the tuned SVM, pooled over batches 4-10.")
paired = read_result("step10_paired_vs_svm.csv").set_index("model")
paired["accuracy %"] = paired["accuracy %"].astype(float)
st.dataframe(paired, width="stretch")
acc = per.pivot(index="model", columns="batch", values="accuracy").round(1)
wins, losses, ties = (int((acc.loc["Soft vote"] > acc.loc["SVM"]).sum()), int((acc.loc["Soft vote"] < acc.loc["SVM"]).sum()),
                      int((acc.loc["Soft vote"] == acc.loc["SVM"]).sum()))
callout("A model is only called better or worse when its whole confidence interval is above or below zero. Tuning the SVM made no clear difference "
        "(untuned SVM: not clearly different), and every other model is clearly worse than the tuned SVM except the soft vote.")
caution(f"The soft vote's interval excludes zero only if recordings are independent, which they are not. Batch by batch it beats the SVM on **{wins}** "
        f"batches, loses on **{losses}** and ties on **{ties}**, and its pooled lead comes mostly from batch 10, so we do not claim it is generally better.")
