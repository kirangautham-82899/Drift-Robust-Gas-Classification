import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from common import FEATURE_NAMES, GAS_COLORS, GASES, centroid_shift, feature_column, load_data, notice, pca_projection

df = load_data()

st.title("Drift viewer")
st.caption("The same gas looks different in later batches. Three ways to see it.")

tab1, tab2, tab3 = st.tabs(["Same gas, different batch", "Where the batches sit (PCA)", "How far has each gas moved?"])

# ======================================================================== tab 1
with tab1:
    c1, c2, c3 = st.columns([1, 1, 2])
    gas = c1.selectbox("Gas", GASES, index=3)
    offset = c2.radio("Reading", list(FEATURE_NAMES), format_func=lambda k: FEATURE_NAMES[k], horizontal=False) - 1
    sensors = c3.multiselect("Sensors", list(range(1, 17)), default=[1, 5, 9, 13])
    if not sensors:
        st.warning("Pick at least one sensor.")
    else:
        sub = df[df["gas"] == gas]
        fig = go.Figure()
        ratios = {}
        for s in sensors:
            means = sub.groupby("batch")[feature_column(s, offset)].mean()
            ratios[s] = means.max() / means.min()
            fig.add_trace(go.Scatter(x=means.index, y=means.values, mode="lines+markers", name=f"sensor {s}"))
        fig.update_yaxes(type="log", title=f"Average {FEATURE_NAMES[offset + 1]} (log scale)")
        fig.update_xaxes(title="Batch (time order)", dtick=1)
        st.plotly_chart(fig, width="stretch")
        worst = max(ratios, key=ratios.get)
        notice(
            f"For {gas}, the average reading of a sensor changes a lot from batch to batch. Among the selected sensors, "
            f"sensor {worst} has its highest batch average {ratios[worst]:,.0f} times larger than its lowest. The changes are not a "
            "smooth decline: values go up and down. Part of this may be gas concentration, which changed between batches and is "
            "not in this dataset, so it cannot be separated from sensor ageing."
        )

# ======================================================================== tab 2
with tab2:
    proj, var = pca_projection()
    c1, c2 = st.columns([1, 2])
    colour_by = c1.radio("Colour the points by", ["Batch (time)", "Gas"])
    lo, hi = c2.select_slider("Show batches", options=list(range(1, 11)), value=(1, 10))
    show_path = st.checkbox("Draw the path of each gas's batch average", value=False)
    shown = proj[(proj["batch"] >= lo) & (proj["batch"] <= hi)]
    if colour_by == "Gas":
        fig = px.scatter(shown, x="PC1", y="PC2", color="gas", color_discrete_map=GAS_COLORS,
                         category_orders={"gas": GASES}, opacity=0.5, render_mode="webgl", hover_data=["batch"])
    else:
        fig = px.scatter(shown, x="PC1", y="PC2", color="batch", color_continuous_scale="Viridis",
                         range_color=(1, 10), opacity=0.5, render_mode="webgl", hover_data=["gas"])
    fig.update_traces(marker=dict(size=5))
    if show_path:
        cent = shown.groupby(["gas", "batch"])[["PC1", "PC2"]].mean().reset_index()
        for g in GASES:
            c = cent[cent["gas"] == g]
            fig.add_trace(go.Scatter(x=c["PC1"], y=c["PC2"], mode="lines+markers+text", text=c["batch"].astype(str),
                                     textposition="top right", line=dict(color=GAS_COLORS[g], width=2),
                                     marker=dict(size=9, color=GAS_COLORS[g], line=dict(width=1, color="black")),
                                     name=f"{g} average", showlegend=False))
    fig.update_layout(height=560)
    st.plotly_chart(fig, width="stretch")
    notice(
        f"This is a 2-D map of the 128 numbers (PC1 and PC2 together keep {sum(var) * 100:.0f}% of the information). The map was "
        "built **only from batches 1-3**, then every batch was placed on it. In the training batches the gases form fairly separate "
        "bands. In later batches the gases overlap and many points, especially in batch 10, land in places the training batches "
        "never covered. A model trained on batches 1-3 has never seen those regions. A 2-D map hides detail, so overlap here does "
        "not prove the gases are inseparable in the full data."
    )

# ======================================================================== tab 3
with tab3:
    shift, between = centroid_shift()
    chosen = st.multiselect("Gases", GASES, default=GASES)
    if chosen:
        fig = go.Figure()
        for g in chosen:
            fig.add_trace(go.Scatter(x=shift.index, y=shift[g], mode="lines+markers", name=g,
                                     line=dict(color=GAS_COLORS[g])))
        fig.add_hline(y=between, line_dash="dash", line_color="black",
                      annotation_text=f"typical distance between two different gases ({between:.1f})", annotation_position="top left")
        fig.update_xaxes(title="Batch (time order)", dtick=1)
        fig.update_yaxes(title="Distance from the gas's average in batches 1-3")
        st.plotly_chart(fig, width="stretch")
        far = (shift > between).sum().sum()
        batch_max, gas_max = shift.stack().idxmax()
        notice(
            f"For each gas we take its average position in the 128-number space in the training batches, then measure how far its "
            f"average in each batch has moved. If a gas has moved further than the typical distance between two different gases "
            f"({between:.1f}, dashed line), a model built on the training batches can easily confuse it with another gas. "
            f"This happens in {far} of the {shift.size} gas-and-batch combinations. The largest move is {gas_max} in batch "
            f"{batch_max} ({shift.stack().max():.1f}; batch {batch_max} has only {int((df['batch'] == batch_max).sum())} samples, so "
            f"this may partly be noise). Ammonia stays closest to home (never above {shift['Ammonia'].max():.1f})."
        )
