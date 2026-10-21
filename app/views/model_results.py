import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from common import (FAMILIES, FAMILY_COLORS, MODEL_COLORS, REDUCTION_NAMES, REDUCTIONS, all_results, leaderboard,
                    notice, read_result)

results = all_results()

st.title("Model results")
st.caption("How well do the models recognise gases in batches they have never seen? Models are trained on the early "
           "batches and tested on later ones. Pick what to compare.")

PROTOCOLS = {"A": "A: train on batches 1-3, test on 4-10 (the headline)",
             "B": "B: train on batch 1 only, test on each of 2-10",
             "C": "C: retrain on all earlier batches, test on 2-10"}
c1, c2 = st.columns([2, 1])
protocol = c1.radio("Test protocol", list(PROTOCOLS), format_func=PROTOCOLS.get)
metric = c2.radio("Score", ["Accuracy", "Macro-F1"], help="Macro-F1 gives every gas the same weight, which matters because the gas mix differs a lot between batches.")
c3, c4 = st.columns([2, 1])
reduction = c3.radio("Features given to the model", REDUCTIONS, format_func=REDUCTION_NAMES.get, horizontal=True)
settings = c4.radio("Model settings", ["Default", "Tuned"], horizontal=True,
                    help="Tuned settings were chosen on batches 1-3 only (Step 8) and were tested on protocol A only.")
if settings == "Tuned" and protocol != "A":
    st.info("Tuned settings were only tested on protocol A: protocols B and C test batches 2 and 3, which the tuning used "
            "for validation. Showing the default settings instead.", icon=":material/info:")
    settings = "Default"
if protocol != "A":
    st.warning("For protocols B and C the score is the **average over the test batches**. It is not comparable with "
               "protocol A, where all test samples are pooled.", icon=":material/warning:")

score_col = "accuracy" if metric == "Accuracy" else "macro_f1"
table = leaderboard(results, protocol, reduction, settings)
ranked = table.sort_values(score_col)

st.header("Leaderboard")
fig = go.Figure(go.Bar(x=ranked[score_col], y=ranked.index, orientation="h",
                       marker_color=[FAMILY_COLORS[m] for m in ranked.index],
                       text=[f"{v:.1f}%" for v in ranked[score_col]], textposition="outside"))
fig.update_xaxes(range=[0, 105], title=f"{metric} (%)")
fig.update_layout(height=380, margin=dict(l=10, r=30, t=10, b=10))
st.plotly_chart(fig, width="stretch")

best, worst = ranked.index[-1], ranked.index[0]
notice(
    f"With {REDUCTION_NAMES[reduction].lower()} and {settings.lower()} settings, the best model is **{best}** "
    f"({ranked.loc[best, score_col]:.1f}%) and the weakest is **{worst}** ({ranked.loc[worst, score_col]:.1f}%). "
    "These are single runs; gaps of a few points are not conclusive (see the Per-batch drift page for confidence intervals). "
    "Guessing the most common gas would score about 21-23%."
)
with st.expander("All scores for this selection"):
    shown = table.rename(columns={"accuracy": "Accuracy %", "macro_precision": "Macro precision %",
                                  "macro_recall": "Macro recall %", "macro_f1": "Macro-F1 %"}).round(1)
    st.dataframe(shown.sort_values("Accuracy %", ascending=False), width="stretch")

# ----------------------------------------------------------------------------------------------- PCA / LDA / tuning
st.header("Does squeezing the features (PCA, LDA) or tuning help? Protocol A")
a = results[(results["protocol"] == "A") & (results["test_batch"] == "pooled")]
heat = a.pivot_table(index="model", columns=["settings", "reduction"], values="accuracy").reindex(FAMILIES)
heat = heat[[(s, r) for s in ["Default", "Tuned"] for r in REDUCTIONS]]
labels = [f"{s} · {'all features' if r == 'None' else r}" for s, r in heat.columns]
fig = px.imshow(heat.to_numpy(), x=labels, y=heat.index, text_auto=".1f", color_continuous_scale="Blues", zmin=40, zmax=80,
                aspect="auto", labels={"color": "accuracy %"})
fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, width="stretch")
better = (heat["Tuned"] - heat["Default"]).stack()
notice(
    "Each cell is the pooled accuracy (%) on test batches 4-10. Squeezing the features with PCA or LDA helps some models and hurts others "
    "(with default settings it hurts SVM and kNN and helps the Decision Tree and Naive Bayes + LDA), so it is not a general fix. Tuning "
    f"improved the result by more than 0.5 points in {int((better > 0.5).sum())} of {len(better)} cases, left it about the same in "
    f"{int((better.abs() <= 0.5).sum())} and made it worse in {int((better < -0.5).sum())}."
)

# ----------------------------------------------------------------------------------------------- headline with seeds
st.header("Headline: tuned models and ensembles on protocol A (mean of 10 random seeds)")
summary = read_result("step12_summary.csv").rename(columns={"Unnamed: 0": "model"}).set_index("model")
summary = summary.sort_values("mean accuracy %")
fig = go.Figure(go.Bar(x=summary["mean accuracy %"], y=summary.index, orientation="h",
                       marker_color=[MODEL_COLORS.get(m, "#888888") for m in summary.index],
                       error_x=dict(type="data", array=summary["std (points)"], visible=True),
                       text=[f"{v:.1f}%" for v in summary["mean accuracy %"]], textposition="outside"))
fig.update_xaxes(range=[0, 105], title="Pooled accuracy on batches 4-10 (%), error bars = standard deviation over seeds")
fig.update_layout(height=420, margin=dict(l=10, r=30, t=10, b=10))
st.plotly_chart(fig, width="stretch")
vote, svm = summary.loc["Soft vote"], summary.loc["SVM"]
notice(
    f"Each model uses the reduction (none or PCA) that scored best on validation. The **soft vote** (a plain average of five models) "
    f"scores {vote['mean accuracy %']:.1f}% ± {vote['std (points)']:.2f}, the best single model **SVM** {svm['mean accuracy %']:.1f}%, "
    f"and the learned **stack** {summary.loc['Stack (LR)', 'mean accuracy %']:.1f}% ± {summary.loc['Stack (LR)', 'std (points)']:.2f}, "
    "which did not beat SVM. The soft vote's lead is mostly due to batch 10 (batch by batch it wins on 3 batches, loses on 3 and ties on 1), "
    "so it should not be taken as a general win."
)

c = read_result("step9_stacking_metrics.csv")
c = c[(c["protocol"] == "C") & (c["test_batch"] != "pooled")]
mean_c = c.groupby("model")["accuracy"].mean().astype(float) * 100
st.subheader("With recent labelled data (protocol C, test batches 4-10)")
cols = st.columns(3)
for col, m in zip(cols, ["SVM", "Stack (LR)", "Soft vote"]):
    col.metric(m, f"{mean_c[m]:.1f}%", help="Mean per-batch accuracy when retrained on all earlier batches.")
st.caption("When each model is retrained on all earlier batches, accuracy is much higher than in protocol A. That uses labelled data "
           "from the batches just before the test batch, so it is a different (easier) task and is not mixed into the headline.")
