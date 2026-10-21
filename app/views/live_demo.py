import html
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import DEMO_MODELS, GASES, MAX_UPLOAD_ROWS, demo_predict, load_data, parse_upload, read_result
from design import gas_color, style, tok
from src import data_loader as dl
from ui import callout, caution, figure, hero, html_block, kpis, section

hero("Try it yourself", "Which gas is this recording?",
     "Pick a recording from the later batches, or upload your own, and see what a model trained on **batches 1-3 only** makes of it. "
     "Then look at how the other models voted.", compact=True)

df = load_data()
test = df[df["batch"] >= 4]
per_batch = read_result("step10_per_batch_ci.csv")


def model_accuracy(model, batch):
    row = per_batch[(per_batch["model"] == model) & (per_batch["group"] == f"batch {batch}")]
    return float(row["accuracy"].iloc[0]) * 100 if len(row) else None


def probability_chart(probs, predicted):
    t = tok()
    order = list(range(5))[::-1]
    fig = go.Figure(go.Bar(y=[GASES[i] for i in order], x=[probs[i] * 100 for i in order], orientation="h",
                           marker_color=[gas_color(GASES[i]) for i in order], marker_line=dict(color=t["surface"], width=2),
                           marker_cornerradius=4, width=0.6,
                           text=[f"{probs[i] * 100:.0f}%" for i in order], textposition="outside", textfont=dict(color=t["ink"]),
                           hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
    style(fig, height=300, legend=False, x_title="Model confidence (%)", grid_x=True)
    fig.update_xaxes(range=[0, 118])
    fig.update_yaxes(showgrid=False)
    return fig


c1, c2 = st.columns([1, 1])
source = c1.radio("Recording from", ["The later batches (4-10)", "My own CSV file"], horizontal=True)
model = c2.selectbox("Model", DEMO_MODELS, index=0,
                     help="All models were trained on batches 1-3 only. The soft vote is a plain average of five models.")

# ---------------------------------------------------------------------------------------------------- pick a recording
if source.startswith("The later"):
    section("1. Pick a recording")
    a, b, c = st.columns([1, 1, 2])
    batch = a.selectbox("Batch", sorted(test["batch"].unique()), index=6)
    gas_filter = b.selectbox("Gas", ["Any gas"] + GASES)
    rows = test[test["batch"] == batch]
    if gas_filter != "Any gas":
        rows = rows[rows["gas"] == gas_filter]
    n = len(rows)
    st.session_state["rec"] = min(max(int(st.session_state.get("rec", 1)), 1), n)
    if n > 1:
        c.slider(f"Recording number (1 to {n})", 1, n, key="rec")
    else:
        c.caption("Only one recording matches.")
    c.button("Surprise me", icon=":material/casino:", on_click=lambda: st.session_state.update(rec=random.randint(1, n)))
    row = rows.iloc[st.session_state["rec"] - 1]
    X = row[dl.FEATURE_COLS].to_numpy(dtype=float).reshape(1, -1)
    true_gas = row["gas"]

    label, probs = demo_predict(model, X)
    predicted = GASES[int(label[0]) - 1]
    correct = predicted == true_gas

    section("2. What the model says")
    left, right = st.columns([1, 1.25])
    verdict = '<span class="verdict ok">✓ Correct</span>' if correct else '<span class="verdict no">✗ Wrong</span>'
    with left:
        html_block(
            f'<div class="result"><div class="small">{html.escape(model)} says this is</div>'
            f'<div class="big"><span class="swatch" style="background:{gas_color(predicted)}"></span>{html.escape(predicted)}</div>'
            f'<div class="small" style="margin-top:.5rem">The recording really is: '
            f'<strong>{html.escape(true_gas)}</strong></div>{verdict}</div>')
        acc = model_accuracy(model, batch)
        if acc is not None:
            st.caption(f"Over all {int((test['batch'] == batch).sum()):,} recordings of batch {batch} this model is right {acc:.0f}% of the time.")
    with right:
        figure(probability_chart(probs[0], predicted),
               pd.DataFrame({"gas": GASES, "probability (%)": (probs[0] * 100).round(1)}))

    section("3. How the other models vote")
    votes = []
    for m in DEMO_MODELS:
        lab, _ = demo_predict(m, X)
        pg = GASES[int(lab[0]) - 1]
        votes.append({"model": m, "predicts": pg, "result": "✓ correct" if pg == true_gas else "✗ wrong"})
    vote_frame = pd.DataFrame(votes)
    n_right = int((vote_frame["result"] == "✓ correct").sum())
    st.dataframe(vote_frame, width="stretch", hide_index=True)
    callout(f"{n_right} of the {len(DEMO_MODELS)} models identify this recording correctly. A model that is right on one recording can be wrong on the "
            "next: later batches have drifted away from what the models saw in training, and accuracy changes from batch to batch "
            "(see the Per-batch drift page).")

# ---------------------------------------------------------------------------------------------------- upload
else:
    section("1. Upload your recordings")
    st.markdown(
        f"Upload a CSV with **128 numeric columns** (the raw sensor features, in the dataset's order, named `f1` to `f128` or without names), "
        f"optionally preceded by a gas label column (1 = Ethanol, 2 = Ethylene, 3 = Ammonia, 4 = Acetaldehyde, 5 = Acetone). "
        f"At most {MAX_UPLOAD_ROWS:,} rows. The model scales the numbers itself, so please upload them unscaled."
    )
    sample = test[test["batch"] == 10].head(5)[dl.FEATURE_COLS].reset_index(drop=True)
    st.download_button("Download a sample file (5 recordings from batch 10)", sample.to_csv(index=False).encode("utf-8"),
                       file_name="sample_recordings.csv", mime="text/csv", icon=":material/download:")
    upload = st.file_uploader("CSV file", type=["csv"])
    if upload is not None:
        try:
            frame = pd.read_csv(upload)
            X, labels = parse_upload(frame)
        except Exception as exc:      # unreadable file or unexpected shape: say why in plain words
            st.error(str(exc) if isinstance(exc, ValueError) else "Sorry, that file could not be read as a CSV table.")
        else:
            label, probs = demo_predict(model, X)
            predicted = [GASES[int(v) - 1] for v in label]
            result = pd.DataFrame({"row": np.arange(1, len(X) + 1), "predicted gas": predicted,
                                   "confidence (%)": (probs.max(axis=1) * 100).round(0).astype(int)})
            if labels is not None:
                result["true gas"] = [GASES[int(v) - 1] for v in labels]
                result["result"] = np.where(result["predicted gas"] == result["true gas"], "✓ correct", "✗ wrong")
            section("2. What the model says", f"{len(X):,} recordings, model: {model}.")
            if labels is not None:
                share = float((np.asarray(label) == labels).mean()) * 100
                kpis([dict(label="Recordings", value=f"{len(X):,}"), dict(label="Correct", value=f"{share:.1f}%", accent=True,
                                                                          note="against the labels you supplied")])
            counts = result["predicted gas"].value_counts().reindex(GASES, fill_value=0)
            fig = go.Figure(go.Bar(x=GASES, y=counts.to_numpy(), marker_color=[gas_color(g) for g in GASES],
                                   marker_line=dict(color=tok()["surface"], width=2), marker_cornerradius=4))
            style(fig, height=300, legend=False, y_title="Number of recordings")
            figure(fig, counts.rename("recordings").to_frame())
            st.dataframe(result, width="stretch", hide_index=True)

caution("These models were trained on batches 1-3 only. On the later batches the best of them is right about 73-75% of the time overall, and as little "
        "as 34-62% on batch 10, so treat any single answer as a guess, not a measurement.")
