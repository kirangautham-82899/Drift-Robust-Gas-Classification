import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.express as px
import streamlit as st
from common import GAS_COLORS, GASES, batch_counts, load_data, notice, read_result
from src import data_loader as dl
from src import eda

df = load_data()
counts = batch_counts()

st.title("Data explorer")
st.caption("What is in the dataset: 12,077 recordings of 5 gases in 10 time-ordered batches, after Toluene was removed.")

# The same checks as Step 1: the counts must match exactly.
checks_ok = counts["after"].to_dict() == dl.EXPECTED_AFTER and len(df) == sum(dl.EXPECTED_AFTER.values()) \
    and set(df["label"]) <= {1, 2, 3, 4, 5}
if checks_ok:
    st.success(f"Data check passed: {len(df):,} samples, per-batch counts match the Step 1 checks, only gases 1-5 present.",
               icon=":material/check_circle:")
else:
    st.error("Data check FAILED: the counts do not match the Step 1 checks.")

# ---------------------------------------------------------------- samples per batch
st.header("Samples per batch")
by_gas = df.groupby(["batch", "gas"]).size().unstack(fill_value=0)[GASES]
long = by_gas.reset_index().melt(id_vars="batch", var_name="gas", value_name="samples")
fig = px.bar(long, x="batch", y="samples", color="gas", color_discrete_map=GAS_COLORS,
             category_orders={"gas": GASES}, labels={"batch": "Batch (time order)", "samples": "Number of samples"})
fig.update_xaxes(type="category")
st.plotly_chart(fig, width="stretch")
big, small = counts["after"].nlargest(2), counts["after"].nsmallest(2)
notice(
    f"The batches are very different in size. The largest are batch {big.index[0]} ({big.iloc[0]:,} samples) and batch "
    f"{big.index[1]} ({big.iloc[1]:,}); the smallest are batch {small.index[0]} ({small.iloc[0]:,}) and batch "
    f"{small.index[1]} ({small.iloc[1]:,}). Results on the small batches are noisy, so we do not read much into them."
)

# ---------------------------------------------------------------- gas mix
st.header("Gas mix inside each batch")
pct = by_gas.div(by_gas.sum(axis=1), axis=0) * 100
fig = px.imshow(pct.round(0), text_auto=".0f", color_continuous_scale="Blues", aspect="auto",
                labels={"x": "Gas", "y": "Batch", "color": "% of batch"})
fig.update_yaxes(type="category")
st.plotly_chart(fig, width="stretch")
batch_top, gas_top = pct.stack().idxmax()
notice(
    f"Each row adds up to 100%. Overall the gases are moderately balanced, but single batches are very different: "
    f"in batch {batch_top}, {gas_top} makes up {pct.stack().max():.0f}% of the recordings. Batch 10 is perfectly balanced "
    f"({pct.loc[10].min():.0f}% each). Because the mix changes, we report macro-F1 "
    "as well as accuracy."
)

# ---------------------------------------------------------------- feature scales
st.header("The 128 numbers per recording")
st.markdown(
    "Each recording has 8 numbers per sensor: the steady-state response (**dR**), the normalised response, and 3 rising and "
    "3 falling moving-average features. They are on very different scales."
)
info = eda.feature_info()
stds = df[dl.FEATURE_COLS].std()
fig = px.bar(x=list(range(1, 129)), y=stds.to_numpy(), color=info["type"].to_numpy(), log_y=True,
             labels={"x": "Feature number (8 per sensor)", "y": "Spread (standard deviation, log scale)", "color": "Feature type"})
st.plotly_chart(fig, width="stretch")
notice(
    f"The widest feature has a spread about {stds.max() / stds.min():,.0f} times larger than the narrowest. Programs that use "
    "distances (kNN, SVM, PCA) would listen almost only to the biggest numbers, so the data is scaled first."
)
with st.expander("Value range of each feature type (all sensors together)"):
    st.dataframe(read_result("step2_feature_type_summary.csv").rename(columns={"type": "feature type"}), width="stretch", hide_index=True)

# ---------------------------------------------------------------- browse
st.header("Browse the data")
f1, f2, f3 = st.columns(3)
batches = f1.multiselect("Batches", sorted(df["batch"].unique()), default=[1])
gases = f2.multiselect("Gases", GASES, default=GASES)
sensor = f3.selectbox("Sensor (shows its 8 features)", list(range(1, 17)), index=0)
cols = [f"f{(sensor - 1) * 8 + k}" for k in range(1, 9)]
view = df[df["batch"].isin(batches) & df["gas"].isin(gases)][["batch", "gas"] + cols]
view = view.rename(columns=dict(zip(cols, eda.FEATURE_TYPES)))
st.caption(f"{len(view):,} recordings shown")
st.dataframe(view, width="stretch", hide_index=True)
