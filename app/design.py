"""Design system for the dashboard: colour tokens, page CSS and chart styling.

The categorical palette is the validated reference palette (every hard check passes in light and dark mode, see
the dataviz guidance). Colour follows the entity: a gas or a model keeps the same colour on every page and in every
filter. Single-series charts use one colour. Heatmaps use one blue ramp (sequential) or blue-red with a grey middle
(diverging). Text always wears text colours, never a series colour.
"""
import streamlit as st

FONT = '"Source Sans 3", "Source Sans Pro", system-ui, -apple-system, "Segoe UI", sans-serif'

TOKENS = {
    "light": dict(page="#f9f9f7", surface="#fcfcfb", raised="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", border="rgba(11,11,11,0.10)", mid="#f0efec", accent="#2a78d6",
                  good="#006300", bad="#d03b3b", warn="#b97f00", shadow="0 1px 2px rgba(11,11,11,0.05), 0 4px 14px rgba(11,11,11,0.04)"),
    "dark": dict(page="#0d0d0d", surface="#1a1a19", raised="#222221", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", border="rgba(255,255,255,0.10)", mid="#383835", accent="#3987e5",
                 good="#0ca30c", bad="#ec6a6a", warn="#fab219", shadow="0 1px 2px rgba(0,0,0,0.4), 0 4px 14px rgba(0,0,0,0.25)"),
}

# Validated categorical slots (fixed order, never cycled)
CATEGORICAL = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
# Sequential single hue (blue 100 -> 700); diverging blue <-> red with a grey middle
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING_POLES = {"light": ("#e34948", "#2a78d6"), "dark": ("#e66767", "#3987e5")}

GAS_SLOT = {"Ethanol": 0, "Ethylene": 1, "Ammonia": 2, "Acetaldehyde": 3, "Acetone": 4}
GAS_SYMBOL = {"Ethanol": "circle", "Ethylene": "square", "Ammonia": "diamond", "Acetaldehyde": "triangle-up", "Acetone": "x"}
GASES = list(GAS_SLOT)

# 8 coloured models; SVM (untuned) shares the SVM colour (dashed); Decision Tree is the neutral "other"
MODEL_SLOT = {"SVM": 0, "Soft vote": 1, "Stack (LR)": 2, "Random Forest + PCA": 3, "kNN + PCA": 4,
              "Gradient Boosting + PCA": 5, "AdaBoost + PCA": 6, "Naive Bayes + PCA": 7}
EXAM_SLOT = {"time-aware": 0, "random, same sizes": 1, "random 80/20": 2}


def mode():
    """'light' or 'dark', following the theme the viewer has selected."""
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def tok():
    return TOKENS[mode()]


def cat(i):
    return CATEGORICAL[mode()][i]


def gas_color(gas):
    return cat(GAS_SLOT[gas])


def model_color(name):
    if name == "SVM (untuned)":
        return cat(MODEL_SLOT["SVM"])
    if name in MODEL_SLOT:
        return cat(MODEL_SLOT[name])
    return tok()["muted"]          # Decision Tree and anything else: neutral


def rgba(hex_colour, alpha):
    h = hex_colour.lstrip("#")
    return f"rgba({int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}, {alpha})"


def sequential_scale():
    """One blue ramp. Light mode: light = low. Dark mode: near-zero recedes into the dark surface."""
    ramp = SEQUENTIAL if mode() == "light" else SEQUENTIAL[::-1]
    return [[i / (len(ramp) - 1), c] for i, c in enumerate(ramp)]


ORDINAL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]    # starts no lighter than step 250


def ordinal_scale():
    """Ordered categories (for example batch 1-10): one blue ramp whose lightest step still stands out from the surface."""
    ramp = ORDINAL if mode() == "light" else ORDINAL[::-1]
    return [[i / (len(ramp) - 1), c] for i, c in enumerate(ramp)]


def tint(alpha=0.38):
    """The accent colour as a translucent tint, for the 'everything else' bars when one bar is highlighted."""
    return rgba(tok()["accent"], alpha)


def diverging_scale():
    low, high = DIVERGING_POLES[mode()]
    return [[0, low], [0.5, tok()["mid"]], [1, high]]


# ------------------------------------------------------------------------------------------------ page CSS
def css():
    t = tok()
    return f"""
<style>
:root {{
  --page:{t['page']}; --surface:{t['surface']}; --raised:{t['raised']}; --ink:{t['ink']}; --ink2:{t['ink2']};
  --muted:{t['muted']}; --grid:{t['grid']}; --border:{t['border']}; --accent:{t['accent']}; --good:{t['good']};
  --bad:{t['bad']}; --warn:{t['warn']}; --shadow:{t['shadow']}; --mid:{t['mid']};
}}
html, body, [class*="css"], .stApp {{ font-family:{FONT}; }}
.block-container {{ max-width:1180px; padding-top:1.6rem; padding-bottom:4rem; }}
[data-testid="stToolbar"], [data-testid="stDecoration"], footer {{ visibility:hidden; height:0; }}
h1, h2, h3 {{ letter-spacing:-0.015em; }}

@keyframes rise {{ from {{ opacity:0; transform:translateY(8px); }} to {{ opacity:1; transform:none; }} }}
@media (prefers-reduced-motion: reduce) {{ * {{ animation:none !important; transition:none !important; }} }}

/* ---------- hero ---------- */
.hero {{ position:relative; overflow:hidden; border-radius:20px; padding:2.4rem 2.6rem; margin:0 0 1.6rem 0; color:#fff;
  background:linear-gradient(135deg,#0d366b 0%,#184f95 48%,#2a78d6 100%); box-shadow:var(--shadow); animation:rise .5s ease both; }}
.hero::before {{ content:""; position:absolute; inset:-40% -10% auto auto; width:60%; height:140%;
  background:radial-gradient(closest-side, rgba(255,255,255,.22), rgba(255,255,255,0)); pointer-events:none; }}
.hero::after {{ content:""; position:absolute; left:0; right:0; bottom:0; height:4px;
  background:linear-gradient(90deg,#eb6834,#eda100,#1baf7a,#e87ba4,#6250d6); opacity:.9; }}
.hero .kicker {{ font-size:.78rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; opacity:.8; }}
.hero h1 {{ color:#fff; font-size:clamp(1.8rem,3.4vw,2.9rem); line-height:1.1; font-weight:760; margin:.45rem 0 .7rem 0; padding:0; max-width:22ch; }}
.hero.compact {{ padding:1.5rem 2rem; }}
.hero.compact h1 {{ font-size:clamp(1.5rem,2.6vw,2.15rem); max-width:none; margin:.3rem 0 .4rem 0; }}
.hero p {{ color:rgba(255,255,255,.9); font-size:1.08rem; line-height:1.55; max-width:64ch; margin:0; }}
.hero.compact p {{ font-size:1rem; }}
.chips {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:1.1rem; }}
.chip {{ background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.28); color:#fff; border-radius:999px;
  padding:.25rem .8rem; font-size:.85rem; font-weight:600; backdrop-filter:blur(2px); }}

/* ---------- sections ---------- */
.section {{ margin:2.6rem 0 .9rem 0; }}
.section h2 {{ font-size:1.5rem; font-weight:720; margin:0; padding:0; color:var(--ink); display:flex; align-items:center; gap:.6rem; }}
.section h2::before {{ content:""; display:inline-block; width:6px; height:1.4rem; border-radius:3px; background:var(--accent); }}
.section p {{ color:var(--ink2); margin:.35rem 0 0 0; max-width:75ch; line-height:1.55; }}

/* ---------- KPI cards ---------- */
.kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:.4rem 0 1rem 0; }}
.kpis.n1 {{ grid-template-columns:1fr; }} .kpis.n2 {{ grid-template-columns:repeat(2,1fr); }} .kpis.n3 {{ grid-template-columns:repeat(3,1fr); }}
@media (max-width:980px) {{ .kpis.n4 {{ grid-template-columns:repeat(2,1fr); }} }}
@media (max-width:640px) {{ .kpis, .kpis.n2, .kpis.n3, .kpis.n4 {{ grid-template-columns:1fr; }} }}
.kpi {{ background:var(--surface); border:1px solid var(--border); border-radius:16px; padding:1.05rem 1.2rem; box-shadow:var(--shadow);
  transition:transform .15s ease, box-shadow .15s ease; animation:rise .55s ease both; }}
.kpi:hover {{ transform:translateY(-2px); }}
.kpi .label {{ color:var(--ink2); font-size:.86rem; font-weight:600; }}
.kpi .value {{ color:var(--ink); font-size:2.35rem; line-height:1.15; font-weight:740; letter-spacing:-0.02em; margin:.15rem 0 .1rem 0; }}
.kpi .note {{ color:var(--muted); font-size:.84rem; }}
.kpi .tag {{ display:inline-block; margin-top:.5rem; border-radius:999px; padding:.12rem .6rem; font-size:.78rem; font-weight:700;
  border:1px solid currentColor; }}
.kpi .tag.bad {{ color:var(--bad); }} .kpi .tag.good {{ color:var(--good); }} .kpi .tag.warn {{ color:var(--warn); }}
.kpi.accent {{ border-top:4px solid var(--accent); }}

/* ---------- cards, callouts, notes ---------- */
.card {{ background:var(--surface); border:1px solid var(--border); border-radius:16px; padding:1.15rem 1.3rem; box-shadow:var(--shadow);
  height:100%; animation:rise .55s ease both; }}
.card .num {{ display:inline-flex; align-items:center; justify-content:center; width:2rem; height:2rem; border-radius:50%;
  background:var(--accent); color:#fff; font-weight:700; margin-bottom:.6rem; }}
.card h3 {{ font-size:1.12rem; margin:.1rem 0 .35rem 0; padding:0; color:var(--ink); }}
.card p {{ color:var(--ink2); margin:0; line-height:1.55; font-size:.97rem; }}
.callout {{ background:var(--surface); border:1px solid var(--border); border-left:5px solid var(--accent); border-radius:14px;
  padding:.9rem 1.2rem 1rem 1.2rem; margin:.7rem 0 1.2rem 0; box-shadow:var(--shadow); }}
.callout .title {{ font-size:.72rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; color:var(--accent); margin-bottom:.25rem; }}
.callout .body {{ color:var(--ink2); line-height:1.6; font-size:1rem; }}
.callout .body strong {{ color:var(--ink); }}
.callout.warn {{ border-left-color:var(--warn); }} .callout.warn .title {{ color:var(--warn); }}
.callout.good {{ border-left-color:var(--good); }} .callout.good .title {{ color:var(--good); }}
.callout.bad {{ border-left-color:var(--bad); }} .callout.bad .title {{ color:var(--bad); }}

/* ---------- batch strip & stepper ---------- */
.strip {{ display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin:.4rem 0 .2rem 0; }}
.cell {{ min-width:3.1rem; text-align:center; border-radius:10px; padding:.55rem .6rem; font-weight:700; font-size:.9rem; color:#fff; }}
.cell.train {{ background:#1baf7a; }} .cell.test {{ background:#2a78d6; }}
.strip .gap {{ width:1.2rem; text-align:center; color:var(--muted); }}
.legend-line {{ color:var(--ink2); font-size:.9rem; margin-top:.2rem; }}
.stepper {{ display:grid; grid-template-columns:repeat(6,1fr); gap:12px; }}
@media (max-width:1100px) {{ .stepper {{ grid-template-columns:repeat(3,1fr); }} }}
@media (max-width:560px) {{ .stepper {{ grid-template-columns:repeat(2,1fr); }} }}
.step {{ background:var(--surface); border:1px solid var(--border); border-radius:14px; padding:.9rem 1rem; position:relative; box-shadow:var(--shadow); }}
.step .n {{ font-size:.72rem; font-weight:800; letter-spacing:.1em; color:var(--accent); }}
.step .t {{ font-weight:700; color:var(--ink); margin:.1rem 0 .2rem 0; }}
.step .d {{ color:var(--ink2); font-size:.88rem; line-height:1.45; }}

/* ---------- verdict / result card (live demo) ---------- */
.result {{ background:var(--surface); border:1px solid var(--border); border-radius:18px; padding:1.3rem 1.5rem; box-shadow:var(--shadow); }}
.result .small {{ color:var(--ink2); font-size:.85rem; font-weight:600; }}
.result .big {{ font-size:2.6rem; font-weight:760; letter-spacing:-0.02em; color:var(--ink); line-height:1.15; }}
.result .swatch {{ display:inline-block; width:14px; height:14px; border-radius:4px; margin-right:.5rem; vertical-align:middle; }}
.verdict {{ display:inline-flex; align-items:center; gap:.4rem; border-radius:999px; padding:.2rem .8rem; font-weight:700; font-size:.9rem;
  border:1px solid currentColor; margin-top:.4rem; }}
.verdict.ok {{ color:var(--good); }} .verdict.no {{ color:var(--bad); }}

/* ---------- Streamlit widgets ---------- */
[data-testid="stSidebar"] {{ border-right:1px solid var(--border); }}
[data-testid="stSidebarNav"] a {{ border-radius:10px; }}
.sidebar-brand {{ font-weight:760; font-size:1.05rem; color:var(--ink); line-height:1.25; margin:.2rem 0 .1rem 0; }}
.sidebar-sub {{ color:var(--muted); font-size:.82rem; line-height:1.4; }}
[data-testid="stExpander"] {{ border:1px solid var(--border); border-radius:14px; background:var(--surface); }}
[data-testid="stPlotlyChart"] {{ background:var(--surface); border:1px solid var(--border); border-radius:16px; padding:0; box-shadow:var(--shadow); overflow:hidden; }}
.stTabs [data-baseweb="tab"] {{ font-weight:650; }}
@media (max-width:640px) {{
  .hero {{ padding:1.5rem 1.2rem; border-radius:16px; }} .kpi .value {{ font-size:1.9rem; }} .block-container {{ padding-left:1rem; padding-right:1rem; }}
  .result .big {{ font-size:2rem; }}
}}
</style>
"""


def apply_theme():
    st.markdown(css(), unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------------ chart styling
CHART_CONFIG = {"displayModeBar": False, "responsive": True}


def style(fig, height=420, legend=True, x_title=None, y_title=None, hover="closest", grid_x=False):
    """Apply the shared chart look: thin marks, hairline solid grid, recessive axes, transparent surface."""
    t = tok()
    fig.update_layout(
        template="none", height=height, hovermode=hover,
        font=dict(family=FONT, size=13, color=t["ink2"]),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=18, r=24, t=44 if legend else 22, b=22),
        hoverlabel=dict(bgcolor=t["raised"], bordercolor=t["axis"], font=dict(color=t["ink"], family=FONT, size=13)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title=None, bgcolor="rgba(0,0,0,0)",
                    font=dict(color=t["ink2"], size=13)),
    )
    axis = dict(showgrid=True, gridcolor=t["grid"], gridwidth=1, zeroline=False, showline=True, linecolor=t["axis"],
                ticks="", tickfont=dict(color=t["muted"], size=12), title_font=dict(color=t["ink2"], size=13), automargin=True)
    fig.update_xaxes(**{**axis, "showgrid": grid_x}, title_text=x_title)
    fig.update_yaxes(**axis, title_text=y_title)
    return fig


def log_axis(fig):
    """A clean log y axis: labels only at powers of ten (1k, 10k, 100k), no cluttering minor labels."""
    fig.update_yaxes(type="log", dtick=1, tickformat="~s", minor=dict(showgrid=False, ticks=""))
    return fig


def line_marker(colour, size=8):
    """A 2px line with markers that have a 2px ring in the surface colour."""
    return dict(line=dict(color=colour, width=2), marker=dict(color=colour, size=size, line=dict(color=tok()["surface"], width=2)))
