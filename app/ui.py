"""Reusable page components: hero banner, KPI cards, callouts, section headings, chart + table view."""
import html
import re

import pandas as pd
import streamlit as st

from design import CHART_CONFIG, apply_theme  # noqa: F401  (apply_theme re-exported for the entry point)


def inline(text):
    """Escape HTML, then turn **bold** into <strong>."""
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html.escape(text, quote=False))


def html_block(markup):
    st.markdown(markup, unsafe_allow_html=True)


def hero(kicker, title, subtitle, chips=(), compact=False):
    chip_html = "".join(f'<span class="chip">{html.escape(c)}</span>' for c in chips)
    html_block(
        f'<div class="hero{" compact" if compact else ""}"><div class="kicker">{html.escape(kicker)}</div>'
        f'<h1>{html.escape(title)}</h1><p>{inline(subtitle)}</p>'
        f'{f"<div class=chips>{chip_html}</div>" if chips else ""}</div>'
    )


def section(title, subtitle=None):
    sub = f"<p>{inline(subtitle)}</p>" if subtitle else ""
    html_block(f'<div class="section"><h2>{html.escape(title)}</h2>{sub}</div>')


def kpis(items):
    """items: dicts with label, value and optionally note, tag, tone ('bad' | 'good' | 'warn'), accent (bool)."""
    cards = []
    for it in items:
        tag = f'<span class="tag {it.get("tone", "")}">{html.escape(it["tag"])}</span>' if it.get("tag") else ""
        note = f'<div class="note">{inline(it["note"])}</div>' if it.get("note") else ""
        cards.append(f'<div class="kpi{" accent" if it.get("accent") else ""}"><div class="label">{html.escape(it["label"])}</div>'
                     f'<div class="value">{html.escape(str(it["value"]))}</div>{note}{tag}</div>')
    html_block(f'<div class="kpis n{len(items)}">{"".join(cards)}</div>')


def callout(text, title="What to notice", kind=""):
    """A short plain-English box. kind: '' (blue) | 'warn' | 'good' | 'bad'."""
    html_block(f'<div class="callout {kind}"><div class="title">{html.escape(title)}</div><div class="body">{inline(text)}</div></div>')


def caution(text):
    callout(text, title="Keep in mind", kind="warn")


def cards(items, numbered=True):
    """Equal-height story cards in columns. items: (title, text)."""
    cols = st.columns(len(items))
    for i, (col, (title, text)) in enumerate(zip(cols, items), start=1):
        num = f'<div class="num">{i}</div>' if numbered else ""
        col.markdown(f'<div class="card">{num}<h3>{html.escape(title)}</h3><p>{inline(text)}</p></div>', unsafe_allow_html=True)
    return cols


def stepper(items):
    parts = [f'<div class="step"><div class="n">STEP {i}</div><div class="t">{html.escape(t)}</div><div class="d">{html.escape(d)}</div></div>'
             for i, (t, d) in enumerate(items, start=1)]
    html_block(f'<div class="stepper">{"".join(parts)}</div>')


def batch_strip(train, test):
    cells = "".join(f'<div class="cell train">batch {b}</div>' for b in train)
    cells += '<div class="gap">→</div>'
    cells += "".join(f'<div class="cell test">batch {b}</div>' for b in test)
    html_block(f'<div class="strip">{cells}</div><div class="legend-line"><span style="color:#1baf7a;font-weight:700">■</span> train '
               f'&nbsp;&nbsp; <span style="color:#2a78d6;font-weight:700">■</span> test (later in time)</div>')


def figure(fig, table=None, table_label="View as a table", key=None):
    """A styled chart plus a table twin (every value is reachable without hovering)."""
    st.plotly_chart(fig, config=CHART_CONFIG, theme=None, width="stretch", key=key)
    if table is not None:
        with st.expander(table_label):
            st.dataframe(table, width="stretch")


def link(path, label):
    """A page link that quietly does nothing when the page is not registered (for example in a single-page test)."""
    try:
        st.page_link(path, label=label)
    except Exception:
        pass


def sidebar_brand():
    with st.sidebar:
        st.markdown('<div class="sidebar-brand">Gas sensor drift</div>'
                    '<div class="sidebar-sub">Drift-robust gas classification with machine learning<br>'
                    '24DS601 · M.Tech Data Science</div>', unsafe_allow_html=True)


def table_df(df, **kwargs):
    return pd.DataFrame(df, **kwargs)
