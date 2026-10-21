import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from common import FAMILIES, REDUCTION_NAMES, REDUCTIONS, all_results, leaderboard, read_result
from design import cat, sequential_scale, style, tint, tok
from ui import callout, caution, figure, hero, kpis, section

results = all_results()

hero("The results", "Model results",
     "How well do the models recognise gases in batches they have never seen? Models are trained on the early batches and tested on later ones. "
     "Pick what to compare.", compact=True)

PROTOCOLS = {"A": "A: train on batches 1-3, test on 4-10 (the headline)",
             "B": "B: train on batch 1 only, test on each of 2-10",
             "C": "C: retrain on all earlier batches, test on 2-10"}
c1, c2 = st.columns([2, 1])
protocol = c1.radio("Test protocol", list(PROTOCOLS), format_func=PROTOCOLS.get)
metric = c2.radio("Score", ["Accuracy", "Macro-F1"],
                  help="Macro-F1 gives every gas the same weight, which matters because the gas mix differs a lot between batches.")
c3, c4 = st.columns([2, 1])
reduction = c3.radio("Features given to the model", REDUCTIONS, format_func=REDUCTION_NAMES.get, horizontal=True)
settings = c4.radio("Model settings", ["Default", "Tuned"], horizontal=True,
                    help="Tuned settings were chosen on batches 1-3 only (Step 8) and were tested on protocol A only.")
if settings == "Tuned" and protocol != "A":
    st.info("Tuned settings were only tested on protocol A: protocols B and C test batches 2 and 3, which the tuning used for validation. "
            "Showing the default settings instead.", icon=":material/info:")
    settings = "Default"
if protocol != "A":
    caution("For protocols B and C the score is the **average over the test batches**. It is not comparable with protocol A, where all test samples are pooled.")

score_col = "accuracy" if metric == "Accuracy" else "macro_f1"
table = leaderboard(results, protocol, reduction, settings)
ranked = table.sort_values(score_col)
best, worst = ranked.index[-1], ranked.index[0]

# ----------------------------------------------------------------------------------------------- leaderboard
section("Leaderboard", f"{REDUCTION_NAMES[reduction]}, {settings.lower()} settings. The best model is highlighted.")
fig = go.Figure(go.Bar(x=ranked[score_col], y=ranked.index, orientation="h", width=0.6,
                       marker=dict(color=[cat(0) if m == best else tint() for m in ranked.index],
                                   line=dict(color=tok()["surface"], width=2)), marker_cornerradius=4,
                       text=[f"{v:.1f}%" for v in ranked[score_col]], textposition="outside", textfont=dict(color=tok()["ink"]),
                       hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
style(fig, height=380, legend=False, x_title=f"{metric} (%)", grid_x=True)
fig.update_xaxes(range=[0, 108])
fig.update_yaxes(showgrid=False)
shown = table.rename(columns={"accuracy": "Accuracy %", "macro_precision": "Macro precision %", "macro_recall": "Macro recall %",
                              "macro_f1": "Macro-F1 %"}).round(1).sort_values("Accuracy %", ascending=False)
figure(fig, shown, table_label="View all scores as a table")
callout(f"The best model is **{best}** ({ranked.loc[best, score_col]:.1f}%) and the weakest is **{worst}** ({ranked.loc[worst, score_col]:.1f}%). "
        "These are single runs, so gaps of a few points are not conclusive (the Per-batch drift page shows confidence intervals). "
        "Guessing the most common gas would score about 21-23%.")

# ----------------------------------------------------------------------------------------------- PCA / LDA / tuning
section("Does squeezing the features (PCA, LDA) or tuning help?", "Pooled accuracy (%) on test batches 4-10, protocol A.")
a = results[(results["protocol"] == "A") & (results["test_batch"] == "pooled")]
heat = a.pivot_table(index="model", columns=["settings", "reduction"], values="accuracy").reindex(FAMILIES)
heat = heat[[(s, r) for s in ["Default", "Tuned"] for r in REDUCTIONS]]
labels = [f"{s} · {'all features' if r == 'None' else r}" for s, r in heat.columns]
fig = px.imshow(heat.to_numpy(), x=labels, y=heat.index, text_auto=".1f", color_continuous_scale=sequential_scale(), zmin=40, zmax=80,
                aspect="auto", labels={"color": "accuracy %"})
style(fig, height=400, legend=False)
fig.update_xaxes(showgrid=False, showline=False, side="top")
fig.update_yaxes(showgrid=False, showline=False)
figure(fig, heat.round(1).set_axis(labels, axis=1))
change = (heat["Tuned"] - heat["Default"]).stack()
callout("Squeezing the features with PCA or LDA helps some models and hurts others (with default settings it hurts SVM and kNN and helps the Decision Tree "
        "and Naive Bayes + LDA), so it is not a general fix. Tuning improved the result by more than 0.5 points in "
        f"**{int((change > 0.5).sum())} of {len(change)}** cases, left it about the same in {int((change.abs() <= 0.5).sum())} and made it worse in "
        f"{int((change < -0.5).sum())}.")

# ----------------------------------------------------------------------------------------------- headline with seeds
section("Headline: tuned models and ensembles", "Protocol A, mean of 10 random seeds. Error bars show the standard deviation over the seeds.")
summary = read_result("step12_summary.csv").rename(columns={"Unnamed: 0": "model"}).set_index("model").sort_values("mean accuracy %")
best_m = summary.index[-1]
fig = go.Figure(go.Bar(x=summary["mean accuracy %"], y=summary.index, orientation="h", width=0.6,
                       marker=dict(color=[cat(0) if m == best_m else tint() for m in summary.index], line=dict(color=tok()["surface"], width=2)),
                       marker_cornerradius=4, error_x=dict(type="data", array=summary["std (points)"], visible=True, color=tok()["ink2"], thickness=1.5),
                       text=[f"{v:.1f}%" for v in summary["mean accuracy %"]], textposition="outside", textfont=dict(color=tok()["ink"]),
                       hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
style(fig, height=430, legend=False, x_title="Pooled accuracy on batches 4-10 (%)", grid_x=True)
fig.update_xaxes(range=[0, 108])
fig.update_yaxes(showgrid=False)
figure(fig, summary[["mean accuracy %", "std (points)", "min %", "max %", "mean macro-F1 %"]].sort_values("mean accuracy %", ascending=False))
vote, svm, stack = summary.loc["Soft vote"], summary.loc["SVM"], summary.loc["Stack (LR)"]
callout(f"Each model uses the reduction (none or PCA) that scored best on validation. The **soft vote** (a plain average of five models) scores "
        f"{vote['mean accuracy %']:.1f}% ± {vote['std (points)']:.2f}, the best single model **SVM** {svm['mean accuracy %']:.1f}%, and the learned **stack** "
        f"{stack['mean accuracy %']:.1f}% ± {stack['std (points)']:.2f}, which did not beat the SVM.")
caution("The soft vote's lead is mostly due to batch 10: batch by batch it beats the SVM on 3 batches, loses on 3 and ties on 1, so it should not be "
        "taken as a general win.")

c = read_result("step9_stacking_metrics.csv")
c = c[(c["protocol"] == "C") & (c["test_batch"] != "pooled")]
mean_c = c.groupby("model")["accuracy"].mean().astype(float) * 100
section("With recent labelled data", "Protocol C, test batches 4-10: each model retrained on all earlier batches.")
kpis([dict(label=m, value=f"{mean_c[m]:.1f}%", note="mean accuracy per batch", accent=(m == "SVM")) for m in ["SVM", "Stack (LR)", "Soft vote"]])
callout("Accuracy is much higher than in protocol A, because every model sees labelled recordings from the batches just before the test batch. "
        "That is a different, easier task, so it is not mixed into the headline numbers.")
