import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import all_results, read_result
from design import EXAM_SLOT, cat, line_marker, style, tok
from ui import callout, caution, cards, figure, hero, kpis, section

FAMILIES = ["kNN", "SVM", "Decision Tree", "Random Forest", "Naive Bayes", "AdaBoost", "Gradient Boosting"]
EXAMS = {"time-aware (A)": ("time-aware", "Time-aware: train on batches 1-3, test on 4-10"),
         "random, same sizes": ("random, same sizes", "Random, same sizes as the time-aware exam"),
         "random 80/20": ("random 80/20", "Random 80/20 split of all recordings")}

table = read_result("step11_accuracy_table.csv").set_index("model").astype(float)
per_batch_raw = read_result("step11_random_vs_time_aware.csv")
twin = read_result("step11_nearest_neighbour.csv").set_index("exam")
twin_cols = list(twin.columns)

hero("Evaluation trap", "A random split makes the models look almost perfect",
     "If recordings from the same session land on both sides of the split, a model can pass by recognising near-twins. "
     "Here the **same models with the same settings** take three different exams.", compact=True)

callout("Imagine an exam in which many questions are your practice questions with the numbers barely changed. You would score "
        "very high without learning anything that helps next year. A random split does exactly that to a gas-sensor model: "
        "recordings made minutes apart, in the same session, end up in both the training and the test set.",
        title="The idea in plain English")

model = st.selectbox("Choose a model to look at", FAMILIES, index=FAMILIES.index("SVM"))
ta, same, r80 = (float(table.loc[model, c]) for c in ["time-aware (A)", "random, same sizes", "random 80/20"])
kpis([
    dict(label="Honest result (time-aware)", value=f"{ta:.1f}%", note="trained on batches 1-3, tested on 4-10", accent=True),
    dict(label="Random split, same sizes", value=f"{same:.1f}%", note="same amount of training data", tag=f"+{same - ta:.1f} points", tone="warn"),
    dict(label="Random 80/20 split", value=f"{r80:.1f}%", note="the usual textbook split", tag=f"+{r80 - ta:.1f} points", tone="warn"),
])

# ------------------------------------------------------------------------------------------------ all models
section("Seven models, three exams", "Pooled accuracy. The random bars are the average of 3 random seeds (they differ by at most 2.5 points).")
fig = go.Figure()
for col, (key, label) in EXAMS.items():
    fig.add_trace(go.Bar(x=[m.replace(" ", "<br>", 1) for m in table.index], y=table[col], name=label, marker_color=cat(EXAM_SLOT[key]),
                         marker_line=dict(color=tok()["surface"], width=2), marker_cornerradius=4))
fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.06)
style(fig, height=430, y_title="Pooled accuracy (%)", hover="x unified")
fig.update_yaxes(range=[0, 105])
figure(fig, table[list(EXAMS)].round(1).rename(columns={k: v[1] for k, v in EXAMS.items()}).rename_axis("model"))
best_random = table["random 80/20"].idxmax()
callout(f"Five of the seven models reach 96-99% when the exam is a random split, but only 50-73% when they are tested on later batches. "
        f"The random exam would crown **{best_random}** ({table.loc[best_random, 'random 80/20']:.1f}%); under drift it scores "
        f"{table.loc[best_random, 'time-aware (A)']:.1f}%. **Naive Bayes** barely changes only because it is weak either way.")

# ------------------------------------------------------------------------------------------------ per batch
section("A random split makes the drift disappear", f"Accuracy of the {model} on each batch.")
rand = per_batch_raw[(per_batch_raw["model"] == model) & (per_batch_raw["scheme"] == "random 80/20")]
rand_acc = {b: float(rand[f"acc_batch{b}"].mean()) * 100 for b in range(1, 11)}
res = all_results()
ta_rows = res[(res["model"] == model) & (res["reduction"] == "None") & (res["settings"] == "Default") & (res["protocol"] == "A")
              & (res["test_batch"] != "pooled")]
ta_acc = {int(b): float(a) for b, a in zip(ta_rows["test_batch"], ta_rows["accuracy"])}
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(rand_acc), y=list(rand_acc.values()), name="Random 80/20 split", mode="lines+markers",
                         **line_marker(cat(EXAM_SLOT["random 80/20"]))))
fig.add_trace(go.Scatter(x=list(ta_acc), y=list(ta_acc.values()), name="Time-aware (train on batches 1-3)", mode="lines+markers",
                         **line_marker(cat(EXAM_SLOT["time-aware"]))))
style(fig, height=380, x_title="Batch (time order)", y_title="Accuracy (%)", hover="x unified")
fig.update_xaxes(dtick=1)
fig.update_yaxes(range=[0, 104])
frame = pd.DataFrame({"random 80/20 (%)": pd.Series(rand_acc), "time-aware (%)": pd.Series(ta_acc)}).round(1).rename_axis("batch")
figure(fig, frame)
callout(f"In the random split the {model} is right {min(rand_acc.values()):.0f}-{max(rand_acc.values()):.0f}% of the time in **every** batch, even the "
        f"furthest ones. Tested honestly it falls to {min(ta_acc.values()):.0f}% on its worst batch. The decline we measured in the other "
        "pages simply never shows up in a random split.")

# ------------------------------------------------------------------------------------------------ twins
section("Why does this happen? Near-twins", "How close is each test recording to its nearest recording in the training set?")
dist_col, share_col = twin_cols[0], twin_cols[1]
exam_names = {"random 80/20": "Random 80/20", "random, same sizes as protocol A": "Random, same sizes", "time-aware (protocol A)": "Time-aware"}
fig = go.Figure(go.Bar(y=[exam_names[e] for e in twin.index][::-1], x=twin[dist_col].astype(float).tolist()[::-1], orientation="h",
                       marker_color=cat(0), marker_line=dict(color=tok()["surface"], width=2), marker_cornerradius=4, width=0.5,
                       text=[f"{v:.2f}" for v in twin[dist_col].astype(float)][::-1], textposition="outside", textfont=dict(color=tok()["ink"])))
style(fig, height=250, legend=False, x_title="Typical distance to the nearest training recording (smaller = more similar)", grid_x=True)
fig.update_xaxes(range=[0, float(twin[dist_col].astype(float).max()) * 1.18])
fig.update_yaxes(showgrid=False)
figure(fig, twin.rename(columns={dist_col: "median distance", share_col: "nearest training recording is from the same batch (%)"}).round(2))
r80_share = float(twin.loc["random 80/20", share_col])
callout(f"In a random split, **{r80_share:.0f}%** of the test recordings have their nearest training recording in the **same batch**, "
        f"and it is about {float(twin.loc['time-aware (protocol A)', dist_col]) / float(twin.loc['random 80/20', dist_col]):.0f} times closer than in the time-aware exam. "
        "A model can score highly just by recognising that near-twin. In the time-aware exam no such twin exists, so the model must really generalise across the drift.")

# ------------------------------------------------------------------------------------------------ takeaways
section("Three things to remember")
gb = table.loc["Gradient Boosting"]
gap_small, gap_big = (table["random 80/20"] - table["random, same sizes"]).abs().agg(["min", "max"])
cards([
    ("It is not extra training data", f"A random split with exactly the same amount of data as the time-aware exam scores almost the same as the "
     f"80/20 split (within {gap_small:.1f}-{gap_big:.1f} points). The jump comes from ignoring time."),
    ("It hides the drift", "In a random split every batch looks easy, so the sensor ageing that the rest of this dashboard measures stays invisible."),
    ("It picks the wrong model", f"Gradient Boosting looks like {gb['random 80/20']:.1f}% in a random split but is {gb['time-aware (A)']:.1f}% under drift, "
     f"while SVM, which is not the top model in a random split, is the best single model at {table.loc['SVM', 'time-aware (A)']:.1f}%."),
])
caution("The distances are measured in a scaled 128-dimensional space and are only a diagnostic; no model used them. This page shows that "
        "leakage happens, not exactly which properties of the recordings cause it.")
