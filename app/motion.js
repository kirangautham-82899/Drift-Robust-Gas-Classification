// Interface motion for the dashboard: number count-up, scroll reveal, cursor spotlight, card tilt, hero parallax,
// click ripples, a reading-progress bar and a back-to-top button. Loaded once by app.py through st.html; it only
// decorates what Streamlit renders and never changes a number or a widget value.
(() => {
  if (window.__gasMotion) return;
  window.__gasMotion = true;
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const doc = window.document;

  // ------------------------------------------------------------------ count-up for KPI values and stat tiles
  const NUM = /^([^\d-]*)(-?[\d,]*\.?\d+)(.*)$/s;
  const ease = (t) => 1 - Math.pow(1 - t, 3);
  function countUp(el) {
    if (el.__counting) return;
    const final = el.textContent.trim();
    if (el.dataset.counted === final) return;
    const m = final.match(NUM);
    el.dataset.counted = final;
    if (!m || reduce) return;
    const [, pre, raw, post] = m;
    const target = parseFloat(raw.replace(/,/g, ""));
    const decimals = (raw.split(".")[1] || "").length;
    const commas = raw.includes(",");
    const fmt = (v) => {
      const s = v.toFixed(decimals);
      return commas ? Number(s).toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals }) : s;
    };
    const start = performance.now(), dur = 800;
    el.__counting = true;
    const tick = (now) => {
      const t = Math.min(1, (now - start) / dur);
      el.textContent = pre + fmt(target * ease(t)) + post;
      if (t < 1) requestAnimationFrame(tick);
      else { el.textContent = final; el.__counting = false; }
    };
    requestAnimationFrame(tick);
  }

  // ------------------------------------------------------------------ reveal on scroll (only for content below the fold)
  const REVEAL = ".section, .card, .kpis, .callout, .stepper, .strip, .bx, .votes, .result, .foot, [data-testid='stPlotlyChart'], [data-testid='stExpander'], [data-testid='stDataFrame']";
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) if (e.isIntersecting) { e.target.classList.add("rv-in"); io.unobserve(e.target); }
  }, { threshold: 0.08, rootMargin: "0px 0px -40px 0px" });
  function reveal(el) {
    if (el.__rv || reduce || el.closest("[data-testid='stSidebar']")) return;
    el.__rv = true;
    if (el.getBoundingClientRect().top > window.innerHeight * 0.92) { el.classList.add("rv"); io.observe(el); }
  }

  function scan(root) {
    root.querySelectorAll?.(".kpi .value").forEach(countUp);
    root.querySelectorAll?.(REVEAL).forEach(reveal);
  }
  // Collect added nodes and handle them once per animation frame; skip chart and table internals (hovering a chart
  // adds and removes nodes constantly).
  let pending = new Set(), queued = false;
  const flush = () => {
    queued = false;
    for (const n of pending) { if (!n.isConnected) continue; scan(n); if (n.matches?.(REVEAL)) reveal(n); if (n.matches?.(".kpi .value")) countUp(n); }
    pending.clear();
  };
  new MutationObserver((muts) => {
    for (const m of muts) {
      const t = m.target.nodeType === 1 ? m.target : m.target.parentElement;
      if (!t || t.closest?.(".js-plotly-plot, [data-testid='stDataFrame'], #gas-progress, #gas-top")) continue;
      if (m.type === "characterData") { pending.add(t); continue; }
      m.addedNodes.forEach((n) => { if (n.nodeType === 1) pending.add(n); else if (n.parentElement) pending.add(n.parentElement); });
    }
    if (pending.size && !queued) { queued = true; requestAnimationFrame(flush); }
  }).observe(doc.body, { childList: true, subtree: true, characterData: true });
  scan(doc);

  // ------------------------------------------------------------------ cursor spotlight, card tilt, hero parallax
  const SPOT = ".kpi, .card, .step, .pane, .vote, .result, .side-card";
  doc.addEventListener("pointermove", (e) => {
    const card = e.target.closest?.(SPOT);
    if (card) {
      const r = card.getBoundingClientRect();
      card.style.setProperty("--mx", `${e.clientX - r.left}px`);
      card.style.setProperty("--my", `${e.clientY - r.top}px`);
      if (!reduce && card.classList.contains("kpi")) {
        const rx = ((e.clientY - r.top) / r.height - 0.5) * -7, ry = ((e.clientX - r.left) / r.width - 0.5) * 7;
        card.style.transform = `perspective(700px) translateY(-4px) rotateX(${rx}deg) rotateY(${ry}deg)`;
      }
    }
    const hero = e.target.closest?.(".hero");
    if (hero && !reduce) {
      const r = hero.getBoundingClientRect();
      hero.style.setProperty("--px", ((e.clientX - r.left) / r.width - 0.5).toFixed(3));
      hero.style.setProperty("--py", ((e.clientY - r.top) / r.height - 0.5).toFixed(3));
    }
  }, { passive: true });
  doc.addEventListener("pointerout", (e) => {
    const card = e.target.closest?.(".kpi");
    if (card && !card.contains(e.relatedTarget)) card.style.transform = "";
  }, { passive: true });

  // ------------------------------------------------------------------ ripple on anything clickable
  const RIPPLE = "[data-testid='stButtonGroup'] button, .stButton > button, .stDownloadButton > button, [data-testid='stTab'], " +
                 "[data-testid='stRadioOption'], [data-testid='stSidebarNavLink'], [data-testid='stPageLink'] a";
  doc.addEventListener("pointerdown", (e) => {
    const el = e.target.closest?.(RIPPLE);
    if (!el || reduce) return;
    const r = el.getBoundingClientRect(), size = Math.max(r.width, r.height) * 2.2;
    const s = doc.createElement("span");
    s.className = "ripple";
    s.style.cssText = `width:${size}px;height:${size}px;left:${e.clientX - r.left - size / 2}px;top:${e.clientY - r.top - size / 2}px`;
    el.appendChild(s);
    setTimeout(() => s.remove(), 650);
  }, { passive: true });

  // ------------------------------------------------------------------ reading progress and back-to-top
  const bar = doc.createElement("div"); bar.id = "gas-progress"; doc.body.appendChild(bar);
  const top = doc.createElement("button"); top.id = "gas-top"; top.setAttribute("aria-label", "Back to top"); top.innerHTML =
    // Markup is written with \x3c instead of "<": the HTML sanitiser drops a script whose text looks like it contains tags.
    '\x3csvg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">\x3cpath d="M12 19V5M5 12l7-7 7 7"/>\x3c/svg>';
  doc.body.appendChild(top);
  let scroller = null;
  const update = (el) => {
    const max = el.scrollHeight - el.clientHeight;
    const pos = el === doc.documentElement ? window.scrollY : el.scrollTop;
    bar.style.transform = `scaleX(${max > 0 ? pos / max : 0})`;
    top.classList.toggle("show", pos > 500);
  };
  doc.addEventListener("scroll", (e) => {
    const el = e.target === doc ? doc.documentElement : e.target;
    if (el.closest?.("[data-testid='stSidebar']") || el.closest?.(".js-plotly-plot") || el.closest?.("[data-testid='stDataFrame']")) return;
    if (el.scrollHeight - el.clientHeight < 200 && el !== doc.documentElement) return;
    scroller = el; update(el);
  }, { passive: true, capture: true });
  top.addEventListener("click", () => (scroller || doc.documentElement).scrollTo({ top: 0, behavior: reduce ? "auto" : "smooth" }));
  // Navigating to another page resets the scroll position, so reset the bar too.
  let path = location.pathname;
  setInterval(() => { if (location.pathname !== path) { path = location.pathname; bar.style.transform = "scaleX(0)"; top.classList.remove("show"); } }, 400);
})();
