"""Drift-Robust Gas Classification: interactive dashboard.

Run from the project root:  streamlit run app/app.py
"""
import streamlit as st

st.set_page_config(page_title="Gas sensor drift", page_icon=":material/air:", layout="wide")

pages = [
    st.Page("views/overview.py", title="Overview", icon=":material/home:", default=True),
    st.Page("views/data_explorer.py", title="Data explorer", icon=":material/table_chart:"),
    st.Page("views/drift_viewer.py", title="Drift viewer", icon=":material/show_chart:"),
    st.Page("views/model_results.py", title="Model results", icon=":material/leaderboard:"),
    st.Page("views/per_batch_drift.py", title="Per-batch drift", icon=":material/timeline:"),
    st.Page("views/confusion_matrices.py", title="Confusion matrices", icon=":material/grid_on:"),
]

with st.sidebar:
    st.caption("Drift-Robust Gas Classification Using Machine Learning")
    st.caption("24DS601 - Machine Learning, M.Tech Data Science")

st.navigation(pages).run()
