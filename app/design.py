"""Design system for the dashboard: colour tokens, page CSS and chart styling.

The interface uses one light "aurora" theme: a cool off-white page with soft pastel light, white glass surfaces, slate
text and an indigo-to-violet accent. The chart palette is separate and unchanged.

The categorical palette is the validated reference palette (every hard check passes in light and dark mode, see
the dataviz guidance). Colour follows the entity: a gas or a model keeps the same colour on every page and in every
filter. Single-series charts use one colour. Heatmaps use one blue ramp (sequential) or blue-red with a grey middle
(diverging). Text always wears text colours, never a series colour.
"""
import streamlit as st

FONT = '"Plus Jakarta Sans", "Inter", system-ui, -apple-system, "Segoe UI", sans-serif'
FONT_URL = "https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap"

TOKENS = {
    "light": dict(page="#f6f7fb", surface="#ffffff", raised="#ffffff", ink="#0f172a", ink2="#475569", muted="#7c8799",
                  grid="#e9ecf3", axis="#d3d8e3", border="rgba(15,23,42,0.08)", mid="#eef0f6", accent="#4f46e5",
                  accent2="#7c3aed", teal="#0d9488", good="#047857", bad="#dc2626", warn="#b45309",
                  shadow="0 1px 2px rgba(15,23,42,0.04), 0 8px 24px -6px rgba(15,23,42,0.08)",
                  shadow_hi="0 2px 4px rgba(15,23,42,0.05), 0 22px 44px -12px rgba(79,70,229,0.28)"),
    "dark": dict(page="#0d0d0d", surface="#1a1a19", raised="#222221", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", border="rgba(255,255,255,0.10)", mid="#383835", accent="#818cf8",
                 accent2="#a78bfa", teal="#2dd4bf", good="#0ca30c", bad="#ec6a6a", warn="#fab219",
                 shadow="0 1px 2px rgba(0,0,0,0.4), 0 4px 14px rgba(0,0,0,0.25)",
                 shadow_hi="0 2px 4px rgba(0,0,0,0.4), 0 18px 40px rgba(0,0,0,0.35)"),
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
    """The app ships one light theme (see .streamlit/config.toml), so pages and charts always use the light tokens.
    The dark tokens are kept so a dark theme can be switched back on here."""
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


def tint(alpha=0.32):
    """The first series colour as a translucent tint, for the 'everything else' bars when one bar is highlighted."""
    return rgba(cat(0), alpha)


def diverging_scale():
    low, high = DIVERGING_POLES[mode()]
    return [[0, low], [0.5, tok()["mid"]], [1, high]]


# ------------------------------------------------------------------------------------------------ page CSS
def css():
    t = tok()
    variables = "".join(f"--{k.replace('_', '-')}:{v};" for k, v in t.items())
    return f"<style>@import url('{FONT_URL}');\n:root {{ {variables} --font:{FONT}; }}\n{STYLES}</style>"


# Plain CSS (not an f-string): colours come from the variables above.
STYLES = """
/* ================================================================ base & backdrop */
html, body, .stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button,
.stMarkdown, [data-testid="stCaptionContainer"], [data-baseweb="select"] * { font-family:var(--font); }
.stApp { background:var(--page); color:var(--ink); }
.stApp::before { content:""; position:fixed; inset:0; z-index:0; pointer-events:none;
  background:
    radial-gradient(42rem 28rem at 8% -6%, rgba(129,140,248,.20), transparent 70%),
    radial-gradient(36rem 26rem at 96% 4%, rgba(244,114,182,.13), transparent 70%),
    radial-gradient(40rem 30rem at 70% 105%, rgba(45,212,191,.13), transparent 70%); }
[data-testid="stMain"], [data-testid="stAppViewContainer"] > section { position:relative; z-index:1; }
.block-container { max-width:1220px; padding-top:1.4rem; padding-bottom:5rem; }
[data-testid="stToolbar"], [data-testid="stDecoration"], footer { visibility:hidden; height:0; }
[data-testid="stHeader"] { background:transparent; }
h1, h2, h3 { letter-spacing:-0.025em; color:var(--ink); }
::selection { background:rgba(79,70,229,.18); }
a { color:var(--accent); }

@keyframes rise { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:none; } }
@keyframes floaty { 0%,100% { transform:translate(0,0) scale(1); } 50% { transform:translate(-14px,10px) scale(1.06); } }
@keyframes pulse { 0% { box-shadow:0 0 0 0 rgba(79,70,229,.45); } 70% { box-shadow:0 0 0 9px rgba(79,70,229,0); } 100% { box-shadow:0 0 0 0 rgba(79,70,229,0); } }
@keyframes shimmer { from { background-position:0% 50%; } to { background-position:200% 50%; } }
@keyframes dash { to { stroke-dashoffset:0; } }
@keyframes grow { from { transform:scaleX(0); } to { transform:scaleX(1); } }
@media (prefers-reduced-motion: reduce) { * { animation:none !important; transition:none !important; } }

/* ================================================================ hero */
.hero { position:relative; overflow:hidden; isolation:isolate; border-radius:28px; padding:2.6rem 2.8rem 2.4rem; margin:0 0 1.8rem 0;
  background:linear-gradient(135deg, rgba(255,255,255,.92), rgba(255,255,255,.74));
  border:1px solid rgba(255,255,255,.9); box-shadow:var(--shadow), inset 0 1px 0 #fff; animation:rise .45s cubic-bezier(.2,.7,.2,1) both; }
.hero::before { content:""; position:absolute; z-index:-1; width:30rem; height:30rem; right:-7rem; top:-12rem; border-radius:50%;
  background:radial-gradient(circle at 30% 30%, rgba(129,140,248,.55), rgba(167,139,250,.28) 45%, transparent 70%); animation:floaty 14s ease-in-out infinite; }
.hero::after { content:""; position:absolute; z-index:-1; width:24rem; height:24rem; right:14rem; bottom:-15rem; border-radius:50%;
  background:radial-gradient(circle, rgba(45,212,191,.35), rgba(244,114,182,.16) 50%, transparent 70%); animation:floaty 18s ease-in-out infinite reverse; }
.hero .kicker { display:inline-flex; align-items:center; gap:.55rem; font-size:.74rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase;
  color:var(--accent); background:rgba(79,70,229,.08); border:1px solid rgba(79,70,229,.16); padding:.32rem .8rem .32rem .6rem; border-radius:999px; }
.hero .kicker::before { content:""; width:7px; height:7px; border-radius:50%; background:var(--accent); animation:pulse 2.2s infinite; }
.hero h1 { font-size:clamp(2rem,3.9vw,3.3rem); line-height:1.06; font-weight:800; margin:1rem 0 .9rem 0; padding:0; max-width:19ch;
  letter-spacing:-0.035em;
  background:linear-gradient(100deg, #1e1b4b 0%, #3730a3 22%, #6d28d9 42%, #0f766e 58%, #3730a3 78%, #1e1b4b 100%); background-size:300% auto;
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; animation:shimmer 14s ease-in-out infinite alternate; }
.hero p { color:var(--ink2); font-size:1.1rem; line-height:1.65; max-width:62ch; margin:0; }
.hero p strong { color:var(--ink); }
.hero.compact { padding:1.8rem 2.2rem 1.7rem; border-radius:24px; }
.hero.compact h1 { font-size:clamp(1.65rem,2.8vw,2.4rem); max-width:none; margin:.8rem 0 .5rem 0; }
.hero.compact p { font-size:1.02rem; }
.hero.compact::before { width:22rem; height:22rem; top:-11rem; right:-5rem; }
.hero.compact::after { width:16rem; height:16rem; bottom:-11rem; right:12rem; }
.hero .wave { position:absolute; right:2.2rem; top:50%; transform:translateY(-50%); width:min(34%,380px); opacity:.95; z-index:-1; }
.hero .wave path { fill:none; stroke-width:2.6; stroke-linecap:round; stroke-dasharray:900; stroke-dashoffset:900; animation:dash 2.6s ease forwards; }
.hero .wave path:nth-child(2) { animation-delay:.25s; } .hero .wave path:nth-child(3) { animation-delay:.5s; }
.hero .wave path:nth-child(4) { animation-delay:.75s; } .hero .wave path:nth-child(5) { animation-delay:1s; }
.hero:has(.wave) p, .hero:has(.wave) .chips { max-width:min(62ch, 60%); }
@media (max-width:980px) { .hero .wave { display:none; } .hero:has(.wave) p, .hero:has(.wave) .chips { max-width:none; } }
.chips { display:flex; flex-wrap:wrap; gap:8px; margin-top:1.4rem; }
.chip { display:inline-flex; align-items:center; gap:.5rem; background:rgba(255,255,255,.85); border:1px solid var(--border); color:var(--ink);
  border-radius:999px; padding:.38rem .95rem .38rem .75rem; font-size:.86rem; font-weight:600; box-shadow:0 1px 2px rgba(15,23,42,.04);
  transition:transform .2s ease, box-shadow .2s ease; }
.chip:hover { transform:translateY(-2px); box-shadow:var(--shadow); }
.chip::before { content:""; width:8px; height:8px; border-radius:50%; background:var(--dot, var(--accent)); }
.chip:nth-child(1) { --dot:#4f46e5; } .chip:nth-child(2) { --dot:#eb6834; } .chip:nth-child(3) { --dot:#1baf7a; } .chip:nth-child(4) { --dot:#e87ba4; }

/* ================================================================ sections */
.section { margin:3rem 0 1rem 0; animation:rise .4s ease both; }
.section h2 { font-size:1.55rem; font-weight:800; margin:0; padding:0; color:var(--ink); display:flex; align-items:center; gap:.7rem; }
.section h2::before { content:""; flex:none; width:.95rem; height:.95rem; border-radius:5px;
  background:linear-gradient(135deg, var(--accent), var(--accent2) 60%, var(--teal)); box-shadow:0 4px 12px rgba(79,70,229,.35); transform:rotate(45deg) scale(.8); }
.section p { color:var(--ink2); margin:.45rem 0 0 1.65rem; max-width:78ch; line-height:1.6; }

/* ================================================================ KPI cards */
.kpis { display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin:.4rem 0 1.2rem 0; }
.kpis.n1 { grid-template-columns:1fr; } .kpis.n2 { grid-template-columns:repeat(2,1fr); } .kpis.n3 { grid-template-columns:repeat(3,1fr); }
@media (max-width:980px) { .kpis.n4 { grid-template-columns:repeat(2,1fr); } }
@media (max-width:640px) { .kpis, .kpis.n2, .kpis.n3, .kpis.n4 { grid-template-columns:1fr; } }
.kpi { position:relative; overflow:hidden; background:var(--surface); border:1px solid var(--border); border-radius:20px; padding:1.2rem 1.35rem 1.25rem;
  box-shadow:var(--shadow); transition:transform .25s cubic-bezier(.2,.7,.2,1), box-shadow .25s ease, border-color .25s ease;
  animation:rise .4s cubic-bezier(.2,.7,.2,1) both; }
.kpi:nth-child(2) { animation-delay:.07s; } .kpi:nth-child(3) { animation-delay:.14s; } .kpi:nth-child(4) { animation-delay:.21s; }
.kpi::after { content:""; position:absolute; width:9rem; height:9rem; right:-3.5rem; top:-3.5rem; border-radius:50%;
  background:radial-gradient(circle, rgba(129,140,248,.16), transparent 70%); transition:transform .4s ease; }
.kpi:hover { transform:translateY(-4px); box-shadow:var(--shadow-hi); border-color:rgba(79,70,229,.25); }
.kpi:hover::after { transform:scale(1.5); }
.kpi .label { color:var(--ink2); font-size:.84rem; font-weight:650; display:flex; align-items:center; gap:.45rem; }
.kpi .value { color:var(--ink); font-size:2.45rem; line-height:1.1; font-weight:800; letter-spacing:-0.035em; margin:.4rem 0 .2rem 0;
  font-variant-numeric:tabular-nums; }
.kpi .note { color:var(--muted); font-size:.84rem; line-height:1.4; }
.kpi .tag { display:inline-flex; align-items:center; margin-top:.7rem; border-radius:999px; padding:.18rem .65rem; font-size:.76rem; font-weight:700; }
.kpi .tag.bad { color:var(--bad); background:rgba(220,38,38,.08); } .kpi .tag.good { color:var(--good); background:rgba(4,120,87,.09); }
.kpi .tag.warn { color:var(--warn); background:rgba(180,83,9,.09); }
.kpi.accent { background:linear-gradient(150deg, #4f46e5 0%, #6d28d9 70%, #7c3aed 100%); border-color:transparent;
  box-shadow:0 18px 40px -14px rgba(79,70,229,.6); }
.kpi.accent::after { background:radial-gradient(circle, rgba(255,255,255,.28), transparent 70%); }
.kpi.accent .label, .kpi.accent .note { color:rgba(255,255,255,.82); } .kpi.accent .value { color:#fff; }

/* ================================================================ cards, callouts */
.card { position:relative; background:var(--surface); border:1px solid var(--border); border-radius:20px; padding:1.3rem 1.4rem; box-shadow:var(--shadow);
  height:100%; animation:rise .4s ease both; transition:transform .25s cubic-bezier(.2,.7,.2,1), box-shadow .25s ease, border-color .25s ease; }
.card:hover { transform:translateY(-4px); box-shadow:var(--shadow-hi); border-color:rgba(79,70,229,.22); }
.card .num { display:inline-flex; align-items:center; justify-content:center; width:2.3rem; height:2.3rem; border-radius:12px;
  background:linear-gradient(135deg, var(--accent), var(--accent2)); color:#fff; font-weight:800; margin-bottom:.85rem; box-shadow:0 8px 18px -6px rgba(79,70,229,.6); }
.card h3 { font-size:1.12rem; font-weight:750; margin:.1rem 0 .4rem 0; padding:0; color:var(--ink); }
.card p { color:var(--ink2); margin:0; line-height:1.6; font-size:.96rem; }
.callout { position:relative; background:linear-gradient(135deg, rgba(79,70,229,.065), rgba(124,58,237,.035)); border:1px solid rgba(79,70,229,.14);
  border-radius:18px; padding:1rem 1.3rem 1.05rem 3.4rem; margin:.9rem 0 1.3rem 0; animation:rise .4s ease both; --tone:var(--accent); }
.callout::before { content:"i"; position:absolute; left:1.05rem; top:1rem; width:1.55rem; height:1.55rem; border-radius:50%; background:var(--tone);
  color:#fff; font-weight:800; font-size:.85rem; display:flex; align-items:center; justify-content:center; font-family:Georgia, serif; font-style:italic;
  box-shadow:0 4px 12px -2px var(--tone); }
.callout .title { font-size:.72rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; color:var(--tone); margin:.12rem 0 .3rem 0; }
.callout .body { color:var(--ink2); line-height:1.65; font-size:.99rem; }
.callout .body strong { color:var(--ink); }
.callout.warn { --tone:var(--warn); background:linear-gradient(135deg, rgba(245,158,11,.09), rgba(251,191,36,.04)); border-color:rgba(217,119,6,.2); }
.callout.warn::before { content:"!"; font-family:var(--font); font-style:normal; }
.callout.good { --tone:var(--good); background:linear-gradient(135deg, rgba(16,185,129,.09), rgba(45,212,191,.04)); border-color:rgba(4,120,87,.18); }
.callout.good::before { content:"✓"; font-family:var(--font); font-style:normal; }
.callout.bad { --tone:var(--bad); background:linear-gradient(135deg, rgba(239,68,68,.08), rgba(244,114,182,.04)); border-color:rgba(220,38,38,.2); }
.callout.bad::before { content:"✕"; font-family:var(--font); font-style:normal; }

/* ================================================================ batch timeline & stepper */
.strip { position:relative; display:grid; grid-template-columns:repeat(3,1fr) 2.2rem repeat(7,1fr); gap:8px; align-items:stretch; margin:.6rem 0 .4rem 0;
  background:var(--surface); border:1px solid var(--border); border-radius:20px; padding:1rem; box-shadow:var(--shadow); }
.cell { text-align:center; border-radius:14px; padding:.8rem .3rem; font-weight:750; font-size:.88rem; color:#fff; cursor:default;
  transition:transform .2s cubic-bezier(.2,.7,.2,1), box-shadow .2s ease; animation:rise .4s ease both; }
.cell:hover { transform:translateY(-5px) scale(1.04); }
.cell.train { background:linear-gradient(160deg, #34d399, #0d9488); box-shadow:0 8px 18px -10px #0d9488; }
.cell.test { background:linear-gradient(160deg, #818cf8, #4f46e5); box-shadow:0 8px 18px -10px #4f46e5; }
.cell.test:nth-of-type(n+9) { background:linear-gradient(160deg, #a78bfa, #6d28d9); }
.strip .gap { display:flex; align-items:center; justify-content:center; color:var(--muted); font-size:1.3rem; font-weight:700; }
@media (max-width:760px) { .strip { grid-template-columns:repeat(5,1fr); } .strip .gap { display:none; } }
.legend-line { color:var(--ink2); font-size:.9rem; margin:.5rem 0 0 .2rem; }
.stepper { position:relative; display:grid; grid-template-columns:repeat(6,1fr); gap:14px; }
.stepper::before { content:""; position:absolute; left:4%; right:4%; top:1.45rem; height:3px; border-radius:3px; transform-origin:left;
  background:linear-gradient(90deg, #34d399, #4f46e5, #7c3aed, #e87ba4); opacity:.55; animation:grow 1.2s ease both; }
@media (max-width:1100px) { .stepper { grid-template-columns:repeat(3,1fr); } .stepper::before { display:none; } }
@media (max-width:560px) { .stepper { grid-template-columns:repeat(2,1fr); } }
.step { position:relative; background:var(--surface); border:1px solid var(--border); border-radius:18px; padding:3.1rem 1rem 1rem; box-shadow:var(--shadow);
  transition:transform .25s cubic-bezier(.2,.7,.2,1), box-shadow .25s ease; animation:rise .4s ease both; }
.step:hover { transform:translateY(-4px); box-shadow:var(--shadow-hi); }
.step:nth-child(2) { animation-delay:.06s; } .step:nth-child(3) { animation-delay:.12s; } .step:nth-child(4) { animation-delay:.18s; }
.step:nth-child(5) { animation-delay:.24s; } .step:nth-child(6) { animation-delay:.3s; }
.step .n { position:absolute; top:.75rem; left:1rem; display:flex; align-items:center; justify-content:center; width:1.75rem; height:1.75rem; border-radius:50%;
  font-size:0; background:linear-gradient(135deg, var(--accent), var(--accent2)); box-shadow:0 0 0 4px #fff, 0 6px 14px -4px rgba(79,70,229,.6); }
.step .n::after { content:attr(data-n); font-size:.78rem; font-weight:800; color:#fff; }
.step .t { font-weight:750; color:var(--ink); margin:.1rem 0 .25rem 0; }
.step .d { color:var(--ink2); font-size:.87rem; line-height:1.5; }

/* ================================================================ live demo result & votes */
.result { position:relative; overflow:hidden; background:var(--surface); border:1px solid var(--border); border-radius:24px; padding:1.5rem 1.6rem;
  box-shadow:var(--shadow-hi); animation:rise .4s ease both; }
.result::before { content:""; position:absolute; inset:auto -20% -60% auto; width:18rem; height:18rem; border-radius:50%;
  background:radial-gradient(circle, var(--glow, rgba(79,70,229,.25)), transparent 70%); }
.result .small { position:relative; color:var(--ink2); font-size:.85rem; font-weight:650; }
.result .big { position:relative; font-size:2.8rem; font-weight:800; letter-spacing:-0.035em; color:var(--ink); line-height:1.15; margin:.25rem 0; }
.result .swatch { display:inline-block; width:18px; height:18px; border-radius:6px; margin-right:.6rem; vertical-align:middle; box-shadow:0 0 0 4px rgba(255,255,255,.9), 0 4px 10px rgba(15,23,42,.2); }
.verdict { position:relative; display:inline-flex; align-items:center; gap:.4rem; border-radius:999px; padding:.3rem .95rem; font-weight:750; font-size:.9rem; margin-top:.8rem; }
.verdict.ok { color:var(--good); background:rgba(4,120,87,.1); } .verdict.no { color:var(--bad); background:rgba(220,38,38,.09); }
.votes { display:grid; grid-template-columns:repeat(auto-fill, minmax(200px, 1fr)); gap:12px; margin:.4rem 0 1rem 0; }
.vote { display:flex; align-items:center; gap:.8rem; background:var(--surface); border:1px solid var(--border); border-radius:16px; padding:.8rem 1rem;
  box-shadow:var(--shadow); transition:transform .2s ease, box-shadow .2s ease; animation:rise .4s ease both; }
.vote:hover { transform:translateY(-3px); box-shadow:var(--shadow-hi); }
.vote .mark { flex:none; width:2rem; height:2rem; border-radius:10px; display:flex; align-items:center; justify-content:center; font-weight:800; color:#fff; }
.vote.ok .mark { background:linear-gradient(135deg, #34d399, #059669); } .vote.no .mark { background:linear-gradient(135deg, #fb7185, #dc2626); }
.vote .m { font-size:.8rem; color:var(--muted); font-weight:650; } .vote .g { font-weight:750; color:var(--ink); display:flex; align-items:center; gap:.4rem; }
.vote .g i { width:9px; height:9px; border-radius:3px; display:inline-block; }
.meter { height:10px; border-radius:999px; background:var(--mid); overflow:hidden; margin:.5rem 0 .2rem 0; }
.meter > span { display:block; height:100%; border-radius:999px; background:linear-gradient(90deg, #34d399, #4f46e5); transform-origin:left; animation:grow 1s cubic-bezier(.2,.7,.2,1) both; }

/* ================================================================ batch explorer (overview) */
.bx { display:grid; grid-template-columns:1.1fr 1fr 1fr; gap:14px; }
@media (max-width:820px) { .bx { grid-template-columns:1fr; } }
.bx .pane { background:var(--surface); border:1px solid var(--border); border-radius:20px; padding:1.15rem 1.3rem; box-shadow:var(--shadow); animation:rise .45s ease both; }
.bx .pane:nth-child(2) { animation-delay:.05s; } .bx .pane:nth-child(3) { animation-delay:.1s; }
.bx .h { color:var(--ink2); font-size:.82rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; }
.bx .v { font-size:2.1rem; font-weight:800; letter-spacing:-0.03em; color:var(--ink); margin:.25rem 0 .1rem; font-variant-numeric:tabular-nums; }
.bx .s { color:var(--muted); font-size:.86rem; line-height:1.45; }
.role { display:inline-flex; align-items:center; gap:.4rem; border-radius:999px; padding:.18rem .7rem; font-size:.76rem; font-weight:750; margin-top:.55rem; }
.role.train { color:#0f766e; background:rgba(13,148,136,.11); } .role.test { color:var(--accent); background:rgba(79,70,229,.1); }
.mix { display:flex; height:14px; border-radius:999px; overflow:hidden; margin:.7rem 0 .6rem; box-shadow:inset 0 0 0 1px rgba(15,23,42,.04); }
.mix span { height:100%; transition:flex .5s ease; }
.mixlegend { display:flex; flex-wrap:wrap; gap:.35rem .8rem; font-size:.8rem; color:var(--ink2); }
.mixlegend b { display:inline-block; width:8px; height:8px; border-radius:3px; margin-right:.3rem; }

/* ================================================================ footer */
.steps-head { margin:3.2rem 0 .2rem; font-size:.7rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; color:var(--muted);
  display:flex; align-items:center; gap:.7rem; }
.steps-head::after { content:""; flex:1; height:1px; background:linear-gradient(90deg, rgba(15,23,42,.12), transparent); }
[data-testid="stElementContainer"]:has(.steps-head) ~ * [data-testid="stPageLink"] a { width:100%; padding:1rem 1.25rem; border-radius:18px;
  background:linear-gradient(135deg, rgba(255,255,255,.95), rgba(245,243,255,.9)); }
[data-testid="stElementContainer"]:has(.steps-head) ~ * [data-testid="stColumn"]:last-child [data-testid="stPageLink"] a { justify-content:flex-end; }
[data-testid="stElementContainer"]:has(.steps-head) ~ * [data-testid="stPageLink"] a p { font-size:1rem; }

.foot { margin-top:1.6rem; padding:1.4rem 1.6rem; border-radius:20px; background:rgba(255,255,255,.7); border:1px solid var(--border);
  display:flex; flex-wrap:wrap; gap:.6rem 1.6rem; align-items:center; justify-content:space-between; color:var(--muted); font-size:.85rem; }
.foot strong { color:var(--ink); }

/* ================================================================ Streamlit chrome: sidebar */
[data-testid="stSidebar"] { background:
    radial-gradient(18rem 14rem at 0% 0%, rgba(129,140,248,.16), transparent 70%),
    radial-gradient(16rem 14rem at 100% 100%, rgba(45,212,191,.12), transparent 70%),
    linear-gradient(180deg, #ffffff 0%, #f7f6ff 100%);
  border-right:1px solid var(--border); box-shadow:6px 0 30px -18px rgba(15,23,42,.18); }
[data-testid="stSidebarHeader"] { padding:1.3rem 1.1rem .6rem 1.25rem; height:auto; align-items:center; }
[data-testid="stSidebarHeader"] img, [data-testid="stSidebarLogo"], [data-testid="stLogo"] { height:2.85rem !important; max-width:none; width:auto;
  transition:transform .3s cubic-bezier(.2,.7,.2,1), filter .3s ease; }
[data-testid="stSidebarHeader"] a:hover img, [data-testid="stSidebarHeader"] img:hover { transform:scale(1.03); filter:drop-shadow(0 6px 14px rgba(79,70,229,.25)); }
[data-testid="stHeader"] [data-testid="stLogo"] { height:2.4rem !important; }
[data-testid="stSidebarCollapseButton"] button { border-radius:10px; transition:background .2s ease; }
[data-testid="stSidebarCollapseButton"] button:hover { background:rgba(79,70,229,.1); }
[data-testid="stSidebarNav"] { padding:0 .75rem; }
[data-testid="stSidebarNavItems"] { padding-top:.4rem; }
[data-testid="stNavSectionHeader"] { display:flex; align-items:center; gap:.6rem; margin:1rem .55rem .35rem !important; padding:0 !important; min-height:0; }
[data-testid="stNavSectionHeader"] p { font-size:.68rem !important; font-weight:800 !important; letter-spacing:.14em; text-transform:uppercase;
  color:var(--muted) !important; white-space:nowrap; }
[data-testid="stNavSectionHeader"] > span { flex:none; }
[data-testid="stNavSectionHeader"]::after { content:""; flex:1; height:1px; background:linear-gradient(90deg, rgba(15,23,42,.1), transparent); order:1; }
[data-testid="stNavSectionHeader"] > div { order:2; opacity:.5; }
[data-testid="stSidebarNavSeparator"] { display:none; }
[data-testid="stSidebarNavLink"] { position:relative; overflow:hidden; display:flex; align-items:center; gap:.2rem; border-radius:14px; margin:3px 0;
  padding:.42rem .6rem .42rem .45rem; min-height:2.6rem; transition:background .25s ease, transform .25s cubic-bezier(.2,.7,.2,1), box-shadow .25s ease; }
[data-testid="stSidebarNavLink"] > span:first-child { display:flex; align-items:center; justify-content:center; flex:none; width:1.95rem; height:1.95rem;
  margin-right:.45rem; border-radius:10px; background:rgba(79,70,229,.07); transition:all .25s ease; }
[data-testid="stSidebarNavLink"] [data-testid="stIconMaterial"] { color:#6366f1 !important; font-size:1.12rem; }
[data-testid="stSidebarNavLink"] p { font-weight:600; color:var(--ink2); font-size:.93rem; transition:color .2s ease; }
[data-testid="stSidebarNavLink"]:hover { background:rgba(255,255,255,.95); transform:translateX(4px); box-shadow:0 6px 16px -10px rgba(15,23,42,.3); }
[data-testid="stSidebarNavLink"]:hover > span:first-child { background:linear-gradient(135deg, rgba(79,70,229,.16), rgba(124,58,237,.12)); transform:rotate(-6deg) scale(1.06); }
[data-testid="stSidebarNavLink"]:hover p { color:var(--ink); }
[data-testid="stSidebarNavLink"][aria-current="page"] { background:linear-gradient(120deg, #4f46e5 0%, #6d28d9 70%, #7c3aed 100%);
  box-shadow:0 12px 24px -12px rgba(79,70,229,.8); transform:none; }
[data-testid="stSidebarNavLink"][aria-current="page"] > span:first-child { background:rgba(255,255,255,.2); box-shadow:inset 0 0 0 1px rgba(255,255,255,.25); }
[data-testid="stSidebarNavLink"][aria-current="page"] [data-testid="stIconMaterial"],
[data-testid="stSidebarNavLink"][aria-current="page"] p, [data-testid="stSidebarNavLink"][aria-current="page"] span { color:#fff !important; }
[data-testid="stSidebarNavLink"][aria-current="page"] p { font-weight:750; }
[data-testid="stSidebarNavLink"][aria-current="page"]::after { content:""; position:absolute; right:.8rem; top:50%; width:6px; height:6px; margin-top:-3px;
  border-radius:50%; background:#fff; animation:pulse-w 2.4s infinite; }
@keyframes pulse-w { 0%,100% { box-shadow:0 0 0 3px rgba(255,255,255,.25); } 50% { box-shadow:0 0 0 7px rgba(255,255,255,0); } }
[data-testid="stSidebarUserContent"] { padding:.4rem 1.1rem 1.5rem; }
.side-label { font-size:.68rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); margin:.6rem .15rem .5rem;
  display:flex; align-items:center; gap:.6rem; }
.side-label::after { content:""; flex:1; height:1px; background:linear-gradient(90deg, rgba(15,23,42,.1), transparent); }
.side-facts { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
.side-facts div { background:rgba(255,255,255,.88); border:1px solid var(--border); border-radius:14px; padding:.6rem .75rem;
  transition:transform .2s ease, box-shadow .2s ease, border-color .2s ease; }
.side-facts div:hover { transform:translateY(-2px); box-shadow:var(--shadow); border-color:rgba(79,70,229,.25); }
.side-facts b { display:block; font-size:1.15rem; font-weight:800; color:var(--ink); letter-spacing:-0.02em; font-variant-numeric:tabular-nums; }
.side-facts span { font-size:.72rem; color:var(--muted); font-weight:650; }
.side-card { position:relative; margin-top:.75rem; padding:.85rem .9rem .7rem; border-radius:16px; background-color:rgba(255,255,255,.92);
  border:1px solid var(--border); box-shadow:0 1px 2px rgba(15,23,42,.04); overflow:hidden;
  background-image:radial-gradient(220px circle at var(--mx,-999px) var(--my,-999px), rgba(99,102,241,.1), transparent 60%); }
.side-head { display:flex; justify-content:space-between; align-items:center; gap:.4rem; font-size:.72rem; white-space:nowrap; font-weight:700; color:var(--ink2); }
.side-head em { font-style:normal; color:#db2777; background:rgba(219,39,119,.09); border-radius:999px; padding:.08rem .5rem; font-size:.7rem; white-space:nowrap; }
.spark { width:100%; height:58px; display:block; margin:.45rem 0 .1rem; overflow:visible; }
.spark .line { stroke-dasharray:420; stroke-dashoffset:420; animation:dash 1.8s .2s cubic-bezier(.4,0,.2,1) forwards; }
.spark .area { opacity:0; animation:fadein .8s 1.2s ease forwards; }
.spark .end { transform-box:fill-box; transform-origin:center; animation:beat 1.8s 1.8s ease-in-out infinite; }
@keyframes fadein { to { opacity:1; } }
@keyframes beat { 0%,100% { transform:scale(1); } 50% { transform:scale(1.5); } }
.side-scale { display:flex; justify-content:space-between; font-size:.7rem; font-weight:700; color:var(--muted); }
.side-author { display:flex; align-items:center; gap:.7rem; margin-top:.75rem; padding:.65rem .75rem; border-radius:16px; text-decoration:none !important;
  border:1px solid var(--border); background:rgba(255,255,255,.7); transition:all .25s cubic-bezier(.2,.7,.2,1); }
.side-author:hover { background:#fff; transform:translateY(-2px); box-shadow:var(--shadow-hi); border-color:rgba(79,70,229,.3); }
.side-author .avatar { flex:none; width:2.2rem; height:2.2rem; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:.78rem;
  font-weight:800; color:#fff; background:linear-gradient(135deg, #f472b6, #7c3aed 55%, #4f46e5); box-shadow:0 0 0 3px #fff, 0 6px 14px -6px rgba(124,58,237,.7); }
.side-author b { display:block; color:var(--ink); font-size:.86rem; } .side-author small { color:var(--muted); font-size:.72rem; font-weight:600; }
.side-author > span:nth-child(2) { flex:1; line-height:1.25; }
.side-author i { font-style:normal; color:var(--accent); font-weight:800; transition:transform .25s ease; }
.side-author:hover i { transform:translate(3px,-3px); }

/* ================================================================ Streamlit widgets */
[data-testid="stWidgetLabel"] p { font-weight:650; color:var(--ink2); font-size:.88rem; }
/* radios become option chips: every option is a small card, the chosen one glows */
[data-testid="stRadioGroup"] { gap:6px; }
[data-testid="stRadioOption"] { margin:0; padding:.5rem .85rem; border-radius:12px; background:var(--surface); border:1px solid var(--border);
  transition:all .2s ease; cursor:pointer; box-shadow:0 1px 2px rgba(15,23,42,.03); }
[data-testid="stRadioOption"]:hover { border-color:rgba(79,70,229,.35); transform:translateY(-1px); }
[data-testid="stRadioOption"][data-selected="true"] { background:linear-gradient(135deg, rgba(79,70,229,.1), rgba(124,58,237,.06));
  border-color:rgba(79,70,229,.45); box-shadow:0 6px 16px -8px rgba(79,70,229,.55); }
[data-testid="stRadioOption"][data-selected="true"] p { color:var(--accent); font-weight:700; }
/* select, multiselect, inputs */
[data-testid="stSelectbox"] [role="group"], [data-testid="stMultiSelect"] [role="group"]:not([aria-label]), [data-baseweb="input"],
[data-testid="stNumberInput"] > div { border-radius:12px !important; background:var(--surface);
  transition:box-shadow .2s ease, border-color .2s ease; box-shadow:0 1px 2px rgba(15,23,42,.03); }
[data-testid="stSelectbox"] [role="group"]:hover, [data-testid="stMultiSelect"] [role="group"]:not([aria-label]):hover { border-color:rgba(79,70,229,.4); }
[data-testid="stSelectbox"] [role="group"]:focus-within, [data-testid="stMultiSelect"] [role="group"]:not([aria-label]):focus-within {
  box-shadow:0 0 0 4px rgba(79,70,229,.14); }
[data-testid="stMultiSelect"] [data-tag] { background:linear-gradient(135deg, rgba(79,70,229,.13), rgba(124,58,237,.1)) !important; border-radius:8px !important;
  color:#3730a3 !important; font-weight:650; }
[data-testid="stMultiSelect"] [data-tag] * { color:#3730a3 !important; }
/* sliders */
[data-testid="stSlider"] [role="slider"] { box-shadow:0 0 0 5px rgba(79,70,229,.16), 0 4px 10px rgba(79,70,229,.4); }
/* buttons */
.stButton > button, .stDownloadButton > button, [data-testid="stBaseButton-secondary"] { border-radius:12px; font-weight:700; border:1px solid rgba(79,70,229,.25);
  background:linear-gradient(135deg, #ffffff, #f5f3ff); color:var(--accent); transition:all .2s ease; box-shadow:0 1px 2px rgba(15,23,42,.04); }
.stButton > button:hover, .stDownloadButton > button:hover { transform:translateY(-2px); color:#fff; border-color:transparent;
  background:linear-gradient(135deg, #4f46e5, #7c3aed); box-shadow:0 12px 24px -10px rgba(79,70,229,.7); }
.stButton > button:active { transform:translateY(0) scale(.98); }
/* pills & segmented control */
[data-testid="stButtonGroup"] button { border-radius:999px !important; font-weight:650; transition:all .2s ease; }
[data-testid="stButtonGroup"] button:hover { transform:translateY(-1px); }
[data-testid="stButtonGroup"] button[kind$="Active"], [data-testid="stButtonGroup"] button[aria-checked="true"] {
  background:linear-gradient(135deg, #4f46e5, #7c3aed) !important; color:#fff !important; border-color:transparent !important;
  box-shadow:0 8px 18px -8px rgba(79,70,229,.7); }
[data-testid="stButtonGroup"] button[kind$="Active"] p, [data-testid="stButtonGroup"] button[aria-checked="true"] p { color:#fff !important; }
/* tabs as a pill bar */
.stTabs [role="tablist"] { gap:4px; background:rgba(255,255,255,.75); border:1px solid var(--border); padding:5px; border-radius:16px;
  box-shadow:var(--shadow); width:fit-content; max-width:100%; overflow-x:auto; }
.stTabs [role="tablist"]::after, .stTabs [role="tablist"]::before { display:none; }
.stTabs [data-testid="stTab"] { border-radius:12px; padding:.55rem 1.05rem; height:auto; color:var(--ink2); transition:all .2s ease; }
.stTabs [data-testid="stTab"] > div:not([data-testid]) { display:none; }
.stTabs [data-testid="stTab"] p { font-weight:700; }
.stTabs [data-testid="stTab"]:hover { background:rgba(79,70,229,.07); }
.stTabs [data-testid="stTab"]:hover p { color:var(--accent); }
.stTabs [data-testid="stTab"][aria-selected="true"] { background:linear-gradient(135deg, #4f46e5, #7c3aed); box-shadow:0 8px 18px -8px rgba(79,70,229,.7); }
.stTabs [data-testid="stTab"][aria-selected="true"] p { color:#fff; }
.stTabs [role="tabpanel"] { padding-top:1.3rem; }
/* expanders, tables, charts, alerts */
[data-testid="stExpander"] details { border:1px solid var(--border); border-radius:16px; background:rgba(255,255,255,.8); box-shadow:0 1px 2px rgba(15,23,42,.03);
  transition:box-shadow .2s ease; }
[data-testid="stExpander"] details:hover { box-shadow:var(--shadow); }
[data-testid="stExpander"] summary { border-radius:16px; font-weight:650; }
[data-testid="stExpander"] summary:hover { color:var(--accent); }
[data-testid="stDataFrame"] { border-radius:14px; overflow:hidden; border:1px solid var(--border); box-shadow:var(--shadow); }
[data-testid="stPlotlyChart"] { background:var(--surface); border:1px solid var(--border); border-radius:22px; padding:0; box-shadow:var(--shadow);
  overflow:hidden; transition:box-shadow .3s ease; }
[data-testid="stPlotlyChart"]:hover { box-shadow:var(--shadow-hi); }
[data-testid="stAlert"] > div { border-radius:16px; }
[data-testid="stFileUploaderDropzone"] { border-radius:18px; border:2px dashed rgba(79,70,229,.3); background:linear-gradient(135deg, rgba(79,70,229,.04), rgba(45,212,191,.04)); }
[data-testid="stFileUploaderDropzone"]:hover { border-color:var(--accent); }
[data-testid="stCaptionContainer"] { color:var(--muted); }
/* page links as navigation cards */
[data-testid="stPageLink"] a { display:flex; border-radius:14px; padding:.6rem .85rem; background:rgba(255,255,255,.85); border:1px solid var(--border);
  margin:.55rem 0 .2rem; transition:all .22s cubic-bezier(.2,.7,.2,1); box-shadow:0 1px 2px rgba(15,23,42,.03); }
[data-testid="stPageLink"] a:hover { transform:translateY(-2px); border-color:rgba(79,70,229,.35); box-shadow:var(--shadow-hi); background:#fff; }
[data-testid="stPageLink"] a p { font-weight:700; color:var(--accent) !important; white-space:normal; }

/* ================================================================ motion helpers (motion.js) */
.rv { animation:none !important; opacity:0; transform:translateY(22px);
  transition:opacity .5s cubic-bezier(.2,.7,.2,1), transform .5s cubic-bezier(.2,.7,.2,1); }
.rv.rv-in { opacity:1; transform:none; }
.kpi:not(.accent), .card, .step, .bx .pane, .vote { background-image:radial-gradient(280px circle at var(--mx,-999px) var(--my,-999px), rgba(99,102,241,.09), transparent 60%); }
.kpi { will-change:transform; }
.hero::before { translate:calc(var(--px,0) * -36px) calc(var(--py,0) * -28px); transition:translate .6s cubic-bezier(.2,.7,.2,1); }
.hero::after { translate:calc(var(--px,0) * 30px) calc(var(--py,0) * 24px); transition:translate .6s cubic-bezier(.2,.7,.2,1); }
.hero .wave { translate:calc(var(--px,0) * 14px) calc(var(--py,0) * 10px); transition:translate .5s ease; }
[data-testid="stButtonGroup"] button, .stButton > button, .stDownloadButton > button, [data-testid="stTab"], [data-testid="stRadioOption"],
[data-testid="stPageLink"] a { position:relative; overflow:hidden; }
.ripple { position:absolute; border-radius:50%; pointer-events:none; background:radial-gradient(circle, rgba(99,102,241,.35), rgba(99,102,241,0) 70%);
  transform:scale(0); animation:ripple .65s ease-out forwards; }
[aria-current="page"] .ripple, [aria-selected="true"] .ripple { background:radial-gradient(circle, rgba(255,255,255,.5), rgba(255,255,255,0) 70%); }
@keyframes ripple { to { transform:scale(1); opacity:0; } }
#gas-progress { position:fixed; top:0; left:0; right:0; height:3px; z-index:999991; transform-origin:left; transform:scaleX(0);
  background:linear-gradient(90deg, #0d9488, #4f46e5, #7c3aed, #db2777); box-shadow:0 0 12px rgba(124,58,237,.5); transition:transform .12s linear; }
#gas-top { position:fixed; right:1.6rem; bottom:1.6rem; z-index:999990; width:2.9rem; height:2.9rem; border-radius:50%; border:0; cursor:pointer; color:#fff;
  display:flex; align-items:center; justify-content:center; background:linear-gradient(135deg, #4f46e5, #7c3aed);
  box-shadow:0 14px 30px -10px rgba(79,70,229,.8); opacity:0; transform:translateY(16px) scale(.8); pointer-events:none;
  transition:opacity .3s ease, transform .3s cubic-bezier(.2,.7,.2,1); }
#gas-top.show { opacity:1; transform:none; pointer-events:auto; }
#gas-top:hover { transform:translateY(-3px); box-shadow:0 18px 34px -10px rgba(79,70,229,.9); }
/* accessibility and finish */
:focus-visible { outline:3px solid rgba(79,70,229,.45) !important; outline-offset:2px; }
* { scrollbar-width:thin; scrollbar-color:rgba(79,70,229,.25) transparent; }
[data-testid="stStatusWidget"] { border-radius:999px; background:rgba(255,255,255,.9); box-shadow:var(--shadow); padding:.15rem .6rem; }
[data-testid="stElementContainer"]:has(.stHtml script) { display:none; }
/* keep the page crisp while Streamlit re-runs after a click (by default it greys out the old content) */
[data-stale="true"] { opacity:1 !important; transition:none !important; }
.hero, .kpi, .card, .step, .pane, .vote { contain:layout paint; }

@media (max-width:640px) {
  #gas-top { right:1rem; bottom:1rem; }
  .hero { padding:1.6rem 1.25rem; border-radius:20px; } .kpi .value { font-size:2rem; } .block-container { padding-left:1rem; padding-right:1rem; }
  .result .big { font-size:2.1rem; } .section p { margin-left:0; }
}
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
        margin=dict(l=22, r=26, t=56 if legend else 30, b=30),
        hoverlabel=dict(bgcolor=t["raised"], bordercolor=t["axis"], font=dict(color=t["ink"], family=FONT, size=13)),
        showlegend=legend, transition=dict(duration=250, easing="cubic-out"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title=None, bgcolor="rgba(0,0,0,0)",
                    font=dict(color=t["ink2"], size=13), itemclick="toggle", itemdoubleclick="toggleothers"),
    )
    axis = dict(showgrid=True, gridcolor=t["grid"], gridwidth=1, zeroline=False, showline=True, linecolor=t["axis"],
                ticks="", tickfont=dict(color=t["muted"], size=12), title_font=dict(color=t["ink2"], size=13), automargin=True,
                showspikes=hover == "closest", spikecolor=t["axis"], spikethickness=1, spikedash="dot", spikemode="across")
    fig.update_xaxes(**{**axis, "showgrid": grid_x}, title_text=x_title)
    fig.update_yaxes(**axis, title_text=y_title)
    return fig


def log_axis(fig):
    """A clean log y axis: labels only at powers of ten (1k, 10k, 100k), no cluttering minor labels."""
    fig.update_yaxes(type="log", dtick=1, tickformat="~s", minor=dict(showgrid=False, ticks=""))
    return fig


def line_marker(colour, size=8):
    """A 2.5px line with markers that have a 2px ring in the surface colour."""
    return dict(line=dict(color=colour, width=2.5),
                marker=dict(color=colour, size=size, line=dict(color=tok()["surface"], width=2)))
