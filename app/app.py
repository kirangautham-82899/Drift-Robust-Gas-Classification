"""Drift-Robust Gas Classification: interactive dashboard.

Run from the project root:  streamlit run app/app.py
"""
import streamlit as st

from design import apply_theme
from ui import sidebar_brand

st.set_page_config(page_title="Gas sensor drift", page_icon=":material/air:", layout="wide")
apply_theme()
sidebar_brand()

pages = {
    "Start": [st.Page("views/overview.py", title="Overview", icon=":material/home:", default=True)],
    "The data": [
        st.Page("views/data_explorer.py", title="Data explorer", icon=":material/table_chart:"),
        st.Page("views/drift_viewer.py", title="Drift viewer", icon=":material/show_chart:"),
    ],
    "The results": [
        st.Page("views/model_results.py", title="Model results", icon=":material/leaderboard:"),
        st.Page("views/per_batch_drift.py", title="Per-batch drift", icon=":material/timeline:"),
        st.Page("views/confusion_matrices.py", title="Confusion matrices", icon=":material/grid_on:"),
        st.Page("views/random_vs_time_aware.py", title="Random vs time-aware", icon=":material/call_split:"),
    ],
    "Try it": [st.Page("views/live_demo.py", title="Live demo", icon=":material/science:")],
    "About": [st.Page("views/about.py", title="Limitations and about", icon=":material/info:")],
}
st.navigation(pages).run()
