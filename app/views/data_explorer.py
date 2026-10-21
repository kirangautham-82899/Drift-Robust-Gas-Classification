import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from common import GASES, batch_counts, load_data, read_result
from design import cat, gas_color, log_axis, sequential_scale, style, tok
from src import data_loader as dl
from src import eda
from ui import callout, figure, hero, kpis, section

df = load_data()
counts = batch_counts()

hero("The data", "Data explorer",
     f"What is in the dataset: **{len(df):,} recordings** of 5 gases in 10 time-ordered batches, after Toluene was removed. "
     "Each recording is 128 numbers from 16 sensors.", compact=True)

# The same checks as Step 1: the counts must match exactly.
checks_ok = counts["after"].to_dict() == dl.EXPECTED_AFTER and len(df) == sum(dl.EXPECTED_AFTER.values()) and set(df["label"]) <= {1, 2, 3, 4, 5}
callout(f"{len(df):,} samples, the per-batch counts match the Step 1 checks, and only gases 1-5 are present." if checks_ok
        else "The counts do not match the Step 1 checks.",
        title="Data check passed" if checks_ok else "Data check FAILED", kind="good" if checks_ok else "bad")

kpis([dict(label="Recordings", value=f"{len(df):,}", accent=True), dict(label="Gases", value=df["gas"].nunique()),
      dict(label="Largest batch", value=f"{int(counts['after'].max()):,}", note=f"batch {int(counts['after'].idxmax())}"),
      dict(label="Smallest batch", value=f"{int(counts['after'].min()):,}", note=f"batch {int(counts['after'].idxmin())}")])

# ---------------------------------------------------------------- samples per batch
section("Samples per batch", "The batches are very different in size.")
by_gas = df.groupby(["batch", "gas"]).size().unstack(fill_value=0)[GASES]
fig = go.Figure()
for g in GASES:
    fig.add_trace(go.Bar(x=by_gas.index, y=by_gas[g], name=g, marker_color=gas_color(g),
                         marker_line=dict(color=tok()["surface"], width=2)))
fig.update_layout(barmode="stack", bargap=0.3)
style(fig, height=400, x_title="Batch (time order)", y_title="Number of recordings", hover="x unified")
fig.update_xaxes(dtick=1)
figure(fig, by_gas.assign(total=by_gas.sum(axis=1)).rename_axis("batch"))
big, small = counts["after"].nlargest(2), counts["after"].nsmallest(2)
callout(f"The largest batches are batch {big.index[0]} ({big.iloc[0]:,} samples) and batch {big.index[1]} ({big.iloc[1]:,}); the smallest are batch "
        f"{small.index[0]} ({small.iloc[0]:,}) and batch {small.index[1]} ({small.iloc[1]:,}). Results on the small batches are noisy, so we do not read "
        "much into them.")

# ---------------------------------------------------------------- gas mix
section("Gas mix inside each batch", "Each row adds up to 100%.")
pct = by_gas.div(by_gas.sum(axis=1), axis=0) * 100
fig = px.imshow(pct.round(0), text_auto=".0f", color_continuous_scale=sequential_scale(), aspect="auto",
                labels={"x": "Gas", "y": "Batch", "color": "% of batch"})
fig.update_yaxes(type="category")
style(fig, height=430, legend=False)
fig.update_xaxes(showgrid=False, showline=False)
fig.update_yaxes(showgrid=False, showline=False)
figure(fig, pct.round(1).rename_axis("batch"))
batch_top, gas_top = pct.stack().idxmax()
callout(f"Overall the gases are moderately balanced, but single batches are very different: in batch {batch_top}, {gas_top} makes up "
        f"{pct.stack().max():.0f}% of the recordings. Batch 10 is perfectly balanced ({pct.loc[10].min():.0f}% each). Because the mix changes, "
        "we report macro-F1 as well as accuracy.")

# ---------------------------------------------------------------- feature scales
section("The 128 numbers per recording", "Each sensor gives 8 numbers: the steady-state response (dR), the normalised response, and 3 rising and "
        "3 falling moving-average features. They are on very different scales.")
info = eda.feature_info()
stds = df[dl.FEATURE_COLS].std()
fig = go.Figure()
for i, t in enumerate(eda.FEATURE_TYPES):
    idx = info.index[info["type"] == t]
    fig.add_trace(go.Bar(x=idx + 1, y=stds.iloc[idx], name=t, marker_color=cat(i)))
fig.update_layout(bargap=0.15)
style(fig, height=400, x_title="Feature number (8 per sensor)", y_title="Spread (standard deviation, log scale)")
log_axis(fig)
figure(fig, stds.rename("standard deviation").to_frame().assign(type=info["type"].to_numpy()).rename_axis("feature"))
callout(f"The widest feature has a spread about **{stds.max() / stds.min():,.0f} times** larger than the narrowest. Programs that use distances "
        "(kNN, SVM, PCA) would listen almost only to the biggest numbers, so the data is scaled first.")
with st.expander("Value range of each feature type (all sensors together)"):
    st.dataframe(read_result("step2_feature_type_summary.csv").rename(columns={"type": "feature type"}), width="stretch", hide_index=True)

# ---------------------------------------------------------------- browse
section("Browse the data")
f1, f2, f3 = st.columns(3)
batches = f1.multiselect("Batches", sorted(df["batch"].unique()), default=[1])
gases = f2.multiselect("Gases", GASES, default=GASES)
sensor = f3.selectbox("Sensor (shows its 8 features)", list(range(1, 17)), index=0)
cols = [f"f{(sensor - 1) * 8 + k}" for k in range(1, 9)]
view = df[df["batch"].isin(batches) & df["gas"].isin(gases)][["batch", "gas"] + cols]
view = view.rename(columns=dict(zip(cols, eda.FEATURE_TYPES)))
st.caption(f"{len(view):,} recordings shown")
st.dataframe(view, width="stretch", hide_index=True)
