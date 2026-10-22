"""Reusable page components: hero banner, KPI cards, callouts, section headings, chart + table view."""
import html
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from design import CHART_CONFIG, apply_theme  # noqa: F401  (apply_theme re-exported for the entry point)

# Five sensor-response curves drawn behind the landing hero; each one drifts a little further from the first.
WAVE = ('<svg class="wave" viewBox="0 0 400 220" aria-hidden="true">'
        + "".join(f'<path stroke="{c}" stroke-opacity="{o}" d="M0 {150 - 6 * i} C 60 {150 - 6 * i}, 80 {40 + 14 * i}, 140 {40 + 14 * i} '
                  f'S 220 {170 - 4 * i}, 270 {120 - 8 * i} S 350 {60 + 10 * i}, 400 {70 + 12 * i}"/>'
                  for i, (c, o) in enumerate([("#4f46e5", 1), ("#7c3aed", .8), ("#0d9488", .7), ("#eb6834", .6), ("#e87ba4", .55)]))
        + "</svg>")


def inline(text):
    """Escape HTML, then turn **bold** into <strong>."""
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html.escape(text, quote=False))


def html_block(markup):
    st.markdown(markup, unsafe_allow_html=True)


def hero(kicker, title, subtitle, chips=(), compact=False):
    chip_html = "".join(f'<span class="chip">{html.escape(c)}</span>' for c in chips)
    html_block(
        f'<div class="hero{" compact" if compact else ""}">{"" if compact else WAVE}<div class="kicker">{html.escape(kicker)}</div>'
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
    """A short plain-English box. kind: '' (indigo) | 'warn' | 'good' | 'bad'."""
    html_block(f'<div class="callout {kind}"><div class="title">{html.escape(title)}</div><div class="body">{inline(text)}</div></div>')


def caution(text):
    callout(text, title="Keep in mind", kind="warn")


def cards(items, numbered=True):
    """Equal-height story cards in columns. items: (title, text)."""
    cols = st.columns(len(items))
    for i, (col, (title, text)) in enumerate(zip(cols, items), start=1):
        num = f'<div class="num">{i}</div>' if numbered else ""
        col.markdown(f'<div class="card" style="animation-delay:{(i - 1) * 0.08:.2f}s">{num}<h3>{html.escape(title)}</h3>'
                     f'<p>{inline(text)}</p></div>', unsafe_allow_html=True)
    return cols


def stepper(items):
    parts = [f'<div class="step"><div class="n" data-n="{i}">STEP {i}</div><div class="t">{html.escape(t)}</div><div class="d">{html.escape(d)}</div></div>'
             for i, (t, d) in enumerate(items, start=1)]
    html_block(f'<div class="stepper">{"".join(parts)}</div>')


def batch_strip(train, test):
    cells = "".join(f'<div class="cell train" style="animation-delay:{i * 0.04:.2f}s">batch {b}</div>' for i, b in enumerate(train))
    cells += '<div class="gap">→</div>'
    cells += "".join(f'<div class="cell test" style="animation-delay:{(i + len(train)) * 0.04:.2f}s">batch {b}</div>' for i, b in enumerate(test))
    html_block(f'<div class="strip">{cells}</div><div class="legend-line"><span style="color:#0d9488;font-weight:800">●</span> train &nbsp;&nbsp; '
               f'<span style="color:#4f46e5;font-weight:800">●</span> test, near in time &nbsp;&nbsp; '
               f'<span style="color:#6d28d9;font-weight:800">●</span> test, far in time</div>')


def vote_cards(votes, colours):
    """One small card per model: ✓/✗ mark, model name and the gas it predicted. votes: dicts with model, predicts, correct."""
    parts = [f'<div class="vote {"ok" if v["correct"] else "no"}" style="animation-delay:{i * 0.04:.2f}s">'
             f'<div class="mark">{"✓" if v["correct"] else "✗"}</div><div><div class="m">{html.escape(v["model"])}</div>'
             f'<div class="g"><i style="background:{colours[v["predicts"]]}"></i>{html.escape(v["predicts"])}</div></div></div>'
             for i, v in enumerate(votes)]
    html_block(f'<div class="votes">{"".join(parts)}</div>')


def meter(share, label):
    """A thin animated progress bar with a caption, for a share between 0 and 1."""
    html_block(f'<div class="meter"><span style="width:{share * 100:.1f}%"></span></div>'
               f'<div style="color:var(--muted);font-size:.84rem">{inline(label)}</div>')


def figure(fig, table=None, table_label="View as a table", key=None):
    """A styled chart plus a table twin (every value is reachable without hovering)."""
    st.plotly_chart(fig, config=CHART_CONFIG, theme=None, width="stretch", key=key)
    if table is not None:
        with st.expander(table_label, icon=":material/table_view:"):
            st.dataframe(table, width="stretch")


def link(path, label):
    """A page link that quietly does nothing when the page is not registered (for example in a single-page test)."""
    try:
        st.page_link(path, label=label)
    except Exception:
        pass


ASSETS = Path(__file__).resolve().parent / "assets"
DATASET = "https://doi.org/10.24432/C5RP6W"


def sidebar_brand():
    """The logo above the page list; when the sidebar is collapsed only the round mark stays in the header."""
    st.logo(str(ASSETS / "logo.svg"), size="large", icon_image=str(ASSETS / "mark.svg"))


def sparkline(values, width=232, height=58):
    """A small inline SVG line for the sidebar: the line draws itself, the last point pulses."""
    lo, hi = min(values) - 5, max(values) + 5
    xs = [6 + i * (width - 12) / (len(values) - 1) for i in range(len(values))]
    ys = [height - 6 - (v - lo) / (hi - lo) * (height - 12) for v in values]
    line = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f} {y:.1f}" for i, (x, y) in enumerate(zip(xs, ys)))
    area = f"{line} L{xs[-1]:.1f} {height} L{xs[0]:.1f} {height} Z"
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.4" fill="#fff" stroke="#4f46e5" stroke-width="1.6"/>' for x, y in zip(xs[:-1], ys[:-1]))
    return (f'<svg class="spark" viewBox="0 0 {width} {height}" preserveAspectRatio="none" aria-hidden="true">'
            '<defs><linearGradient id="sparkfill" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#6366f1" stop-opacity=".28"/>'
            '<stop offset="1" stop-color="#6366f1" stop-opacity="0"/></linearGradient>'
            '<linearGradient id="sparkline" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#4f46e5"/><stop offset="1" stop-color="#db2777"/>'
            f'</linearGradient></defs><path class="area" d="{area}" fill="url(#sparkfill)"/>'
            f'<path class="line" d="{line}" fill="none" stroke="url(#sparkline)" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>'
            f'{dots}<circle class="end" cx="{xs[-1]:.1f}" cy="{ys[-1]:.1f}" r="4" fill="#db2777"/></svg>')


def sidebar_panel(facts, svm_by_batch):
    """Below the page list: dataset facts, the headline drift in one sparkline, and the data source.

    facts: (value, label) pairs. svm_by_batch: SVM accuracy (%) on test batches 4-10, in order."""
    first, last = svm_by_batch[0], svm_by_batch[-1]
    tiles = "".join(f"<div><b>{html.escape(str(v))}</b><span>{html.escape(k)}</span></div>" for v, k in facts)
    with st.sidebar:
        st.markdown(
            f'<div class="side-label">At a glance</div><div class="side-facts">{tiles}</div>'
            f'<div class="side-card"><div class="side-head"><span>SVM accuracy · batch 4→10</span>'
            f'<em>▼ {first - last:.0f} pts</em></div>{sparkline(svm_by_batch)}'
            f'<div class="side-scale"><span>{first:.0f}%</span><span>{last:.0f}%</span></div></div>'
            f'<a class="side-author" href="{DATASET}" target="_blank" rel="noopener"><span class="avatar">UCI</span>'
            '<span><b>Open dataset</b><small>UCI Gas Sensor Array Drift</small></span><i>↗</i></a>',
            unsafe_allow_html=True)


MOTION = (Path(__file__).resolve().parent / "motion.js").read_text(encoding="utf-8")


def motion():
    """Load the interface motion script (count-up numbers, scroll reveal, spotlight, ripples, progress bar)."""
    st.html(f"<script>{MOTION}</script>", unsafe_allow_javascript=True)


def page_steps(ordered, current):
    """'Previous' and 'Next' cards at the end of a page, so the story can be read in order."""
    i = next((k for k, p in enumerate(ordered) if p.url_path == current.url_path), None)
    if i is None:
        return
    html_block(f'<div class="steps-head">Page {i + 1} of {len(ordered)}</div>')
    left, right = st.columns(2)
    if i > 0:
        with left:
            st.page_link(ordered[i - 1], label=f"← Previous · {ordered[i - 1].title}")
    if i < len(ordered) - 1:
        with right:
            st.page_link(ordered[i + 1], label=f"Next · {ordered[i + 1].title} →")


def footer():
    html_block('<div class="foot"><div><strong>Gas sensor drift</strong> · drift-robust gas classification with classical machine learning</div>'
               '<div>Data: UCI Gas Sensor Array Drift Dataset (Vergara et al., 2012)</div></div>')


def table_df(df, **kwargs):
    return pd.DataFrame(df, **kwargs)
