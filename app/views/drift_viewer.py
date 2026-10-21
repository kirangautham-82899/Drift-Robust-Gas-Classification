import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from common import FEATURE_NAMES, GASES, centroid_shift, feature_column, load_data, pca_projection
from design import GAS_SYMBOL, cat, gas_color, line_marker, log_axis, ordinal_scale, style, tok
from ui import callout, caution, figure, hero

df = load_data()

hero("The problem", "Drift viewer",
     "The same gas looks different in later batches. Three ways to see it: in the raw readings, on a map of all 128 numbers, "
     "and as a distance from where each gas started.", compact=True)

tab1, tab2, tab3 = st.tabs(["Same gas, different batch", "Where the batches sit", "How far has each gas moved?"])

# ======================================================================== tab 1
with tab1:
    c1, c2, c3 = st.columns([1, 1, 2])
    gas = c1.selectbox("Gas", GASES, index=3)
    offset = c2.radio("Reading", list(FEATURE_NAMES), format_func=lambda k: FEATURE_NAMES[k]) - 1
    sensors = c3.multiselect("Sensors (up to 6)", list(range(1, 17)), default=[1, 5, 9, 13], max_selections=6)
    if not sensors:
        st.warning("Pick at least one sensor.")
    else:
        sub = df[df["gas"] == gas]
        fig = go.Figure()
        ratios, means_table = {}, {}
        for i, s in enumerate(sensors):
            means = sub.groupby("batch")[feature_column(s, offset)].mean()
            ratios[s] = means.max() / means.min()
            means_table[f"sensor {s}"] = means
            fig.add_trace(go.Scatter(x=means.index, y=means.values, mode="lines+markers", name=f"sensor {s}", **line_marker(cat(i))))
        style(fig, height=430, x_title="Batch (time order)", y_title=f"Average {FEATURE_NAMES[offset + 1]} (log scale)", hover="x unified")
        log_axis(fig)
        fig.update_xaxes(dtick=1)
        figure(fig, pd.DataFrame(means_table).round(2).rename_axis("batch"))
        worst = max(ratios, key=ratios.get)
        callout(f"For {gas}, the average reading of a sensor changes a lot from batch to batch. Among the selected sensors, sensor {worst} has its "
                f"highest batch average **{ratios[worst]:,.0f} times** larger than its lowest. The changes are not a smooth decline: values go up and down.")
        caution("Part of this may be gas concentration, which changed between batches and is not in this dataset, so it cannot be separated from sensor ageing.")

# ======================================================================== tab 2
with tab2:
    proj, var = pca_projection()
    c1, c2 = st.columns([1, 2])
    colour_by = c1.radio("Colour the points by", ["Batch (time)", "Gas"])
    lo, hi = c2.select_slider("Show batches", options=list(range(1, 11)), value=(1, 10))
    show_path = st.checkbox("Draw the path of each gas's batch average", value=False)
    shown = proj[(proj["batch"] >= lo) & (proj["batch"] <= hi)]
    if colour_by == "Gas":
        fig = go.Figure()
        for g in GASES:                                 # colour AND marker shape, so no gas depends on colour alone
            s = shown[shown["gas"] == g]
            fig.add_trace(go.Scattergl(x=s["PC1"], y=s["PC2"], mode="markers", name=g, customdata=s["batch"],
                                       hovertemplate=g + " · batch %{customdata}<extra></extra>",
                                       marker=dict(color=gas_color(g), symbol=GAS_SYMBOL[g], size=5, opacity=0.45)))
    else:
        fig = px.scatter(shown, x="PC1", y="PC2", color="batch", color_continuous_scale=ordinal_scale(), range_color=(1, 10),
                         opacity=0.6, render_mode="webgl", hover_data=["gas"])
        fig.update_traces(marker=dict(size=6))
        fig.update_layout(coloraxis_colorbar=dict(title="batch", thickness=12, len=0.7))
    if show_path:
        cent = shown.groupby(["gas", "batch"])[["PC1", "PC2"]].mean().reset_index()
        for g in GASES:
            c = cent[cent["gas"] == g]
            fig.add_trace(go.Scatter(x=c["PC1"], y=c["PC2"], mode="lines+markers+text", text=c["batch"].astype(str),
                                     textposition="top right", textfont=dict(color=tok()["ink"], size=11),
                                     line=dict(color=gas_color(g), width=2),
                                     marker=dict(size=9, color=gas_color(g), symbol=GAS_SYMBOL[g], line=dict(width=2, color=tok()["surface"])),
                                     name=f"{g} average", showlegend=False, hoverinfo="skip"))
    style(fig, height=560, legend=colour_by == "Gas", x_title="PC1", y_title="PC2", grid_x=True)
    figure(fig, shown.groupby(["batch", "gas"])[["PC1", "PC2"]].mean().round(2), table_label="View the batch averages as a table")
    callout(f"This is a 2-D map of the 128 numbers (PC1 and PC2 together keep **{sum(var) * 100:.0f}%** of the information). The map was built **only from "
            "batches 1-3**, then every batch was placed on it. In the training batches the gases form fairly separate bands. In later batches the gases "
            "overlap and many points, especially in batch 10, land in places the training batches never covered. A model trained on batches 1-3 has "
            "never seen those regions.")
    caution("A 2-D map hides detail, so overlap here does not prove that the gases are inseparable in the full data.")

# ======================================================================== tab 3
with tab3:
    shift, between = centroid_shift()
    chosen = st.multiselect("Gases", GASES, default=GASES)
    if chosen:
        fig = go.Figure()
        for g in chosen:
            fig.add_trace(go.Scatter(x=shift.index, y=shift[g], mode="lines+markers", name=g, **line_marker(gas_color(g))))
        fig.add_hline(y=between, line_dash="dash", line_color=tok()["ink2"], line_width=1,
                      annotation_text=f"typical distance between two different gases ({between:.1f})", annotation_position="top left",
                      annotation_font=dict(color=tok()["ink2"], size=12))
        style(fig, height=430, x_title="Batch (time order)", y_title="Distance from the gas's average in batches 1-3", hover="x unified")
        fig.update_xaxes(dtick=1)
        figure(fig, shift.round(2).rename_axis("batch"))
        far = int((shift > between).sum().sum())
        batch_max, gas_max = shift.stack().idxmax()
        callout(f"For each gas we take its average position in the 128-number space in the training batches, then measure how far its average in each batch "
                f"has moved. If a gas has moved further than the typical distance between two different gases (**{between:.1f}**, dashed line), a model "
                f"built on the training batches can easily confuse it with another gas. This happens in **{far} of the {shift.size}** gas-and-batch "
                f"combinations. Ammonia stays closest to home (never above {shift['Ammonia'].max():.1f}).")
        caution(f"The largest move is {gas_max} in batch {batch_max} ({shift.stack().max():.1f}), but batch {batch_max} has only "
                f"{int((df['batch'] == batch_max).sum())} samples, so this may partly be noise.")
