"""Drift-Robust Gas Classification: interactive dashboard.

Run from the project root:  streamlit run app/app.py
"""
import streamlit as st

from common import read_result
from design import apply_theme
from ui import footer, motion, page_steps, sidebar_brand, sidebar_panel

st.set_page_config(page_title="Gas sensor drift", page_icon=":material/air:", layout="wide")
apply_theme()
motion()
sidebar_brand()

pages = {
    "Start": [st.Page("views/overview.py", title="Overview", icon=":material/space_dashboard:", default=True)],
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
page = st.navigation(pages)

counts = read_result("step1_counts.csv")
counts = counts[counts["batch"] != "Total"]
ci = read_result("step10_per_batch_ci.csv")
svm = ci[(ci["model"] == "SVM") & ci["group"].str.startswith("batch")]
svm = svm.assign(batch=svm["group"].str.split().str[1].astype(int)).sort_values("batch")
sidebar_panel([(f"{int(counts['after'].astype(int).sum()):,}", "recordings"), ("5", "gases"), (str(len(counts)), "batches"), ("16", "sensors")],
              list(svm["accuracy"].astype(float) * 100))

page.run()
page_steps([p for group in pages.values() for p in group], page)
footer()
