"""
The dashboard, as one file.

The previous dashboard was a separate program in a separate language, deployed somewhere. That
is the right shape for something several people watch and the wrong shape for the question this
one answers, which is "what did it just do, on my machine, now". A page that needs a build step
and a deploy is a page nobody opens while debugging.

So: one HTML file with the payload inside it. No build, no server required, no network. It opens
from the filesystem and it works on a plane. Every pixel is drawn by hand into a canvas rather
than by a charting library, because a library here means either a CDN request the page cannot
make offline or a vendored megabyte in the repository.

What it draws is the uncomfortable half, the same as the payload it reads. Anyone can plot an
equity curve. This one leads with calibration, model disagreement, the share of each prediction
that is reducible uncertainty, and every refusal with its reason, because those are the numbers
that say whether the equity curve means anything.
"""

"""
    embed_json(payload)

A payload as JSON safe to place inside a `<script>` tag.

`</script>` inside a string ends the tag no matter where it appears, including in the middle of
a JSON string literal. The browser then parses the rest of the payload as HTML. Escaping the
slash is what stops a symbol or an error message from silently breaking the page.
"""
function embed_json(payload::AbstractDict)
    text = JSON3.write(payload)
    return replace(text, "</" => "<\\/", "\\u2028" => "", "\\u2029" => "")
end

const DASHBOARD_PAGE_TEMPLATE = raw"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
__REFRESH__<title>__TITLE__</title>
<style>
:root {
  color-scheme: light dark;
  --ground: #f4f6f8;
  --panel: #ffffff;
  --edge: #dbe1e8;
  --ink: #16202b;
  --ink-soft: #5b6b7d;
  --ink-faint: #8c9bab;
  --accent: #0f7d8c;
  --accent-soft: rgba(15, 125, 140, 0.16);
  --warn: #b0741a;
  --warn-soft: rgba(176, 116, 26, 0.16);
  --bad: #a83232;
  --bad-soft: rgba(168, 50, 50, 0.16);
  --good: #2f7a4f;
  --grid: rgba(91, 107, 125, 0.16);
}
@media (prefers-color-scheme: dark) {
  :root {
    --ground: #0e1419;
    --panel: #161e26;
    --edge: #26323d;
    --ink: #e4ecf3;
    --ink-soft: #9aabbb;
    --ink-faint: #6a7b8c;
    --accent: #3fbecf;
    --accent-soft: rgba(63, 190, 207, 0.18);
    --warn: #e0a445;
    --warn-soft: rgba(224, 164, 69, 0.18);
    --bad: #e06c6c;
    --bad-soft: rgba(224, 108, 108, 0.18);
    --good: #5fbe87;
    --grid: rgba(154, 171, 187, 0.14);
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--ground);
  color: var(--ink);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1180px; margin: 0 auto; padding: 32px 24px 72px; }
header { display: flex; flex-wrap: wrap; gap: 16px 28px; align-items: baseline;
  padding-bottom: 20px; border-bottom: 1px solid var(--edge); margin-bottom: 28px; }
h1 { font-size: 22px; margin: 0; letter-spacing: -0.01em; font-weight: 650; }
h1 .sym { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--accent); }
.meta { color: var(--ink-faint); font-size: 13px; display: flex; gap: 18px; flex-wrap: wrap;
  margin-left: auto; font-variant-numeric: tabular-nums; }
h2 { font-size: 13px; text-transform: uppercase; letter-spacing: 0.09em;
  color: var(--ink-soft); margin: 40px 0 14px; font-weight: 600; }
h2:first-of-type { margin-top: 0; }
.note { color: var(--ink-faint); font-size: 13px; margin: -6px 0 14px; max-width: 68ch; }
.cards { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(158px, 1fr)); }
.card { background: var(--panel); border: 1px solid var(--edge); border-radius: 8px;
  padding: 14px 16px; }
.card .label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.07em;
  color: var(--ink-faint); }
.card .value { font: 600 24px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace;
  font-variant-numeric: tabular-nums; margin-top: 6px; overflow-wrap: anywhere; }
.card .value.word { font-size: 17px; letter-spacing: 0.01em; }
.card .hint { font-size: 12px; color: var(--ink-faint); margin-top: 4px; }
.card.flag-good .value { color: var(--good); }
.card.flag-warn .value { color: var(--warn); }
.card.flag-bad .value { color: var(--bad); }
.panel { background: var(--panel); border: 1px solid var(--edge); border-radius: 8px;
  padding: 16px; }
.chart { position: relative; width: 100%; }
canvas { display: block; width: 100%; }
.legend { display: flex; gap: 16px; flex-wrap: wrap; font-size: 12px; color: var(--ink-soft);
  margin-top: 10px; font-variant-numeric: tabular-nums; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.swatch { width: 11px; height: 11px; border-radius: 2px; display: inline-block; }
.split { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px;
  font-variant-numeric: tabular-nums; }
th { text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--ink-faint); font-weight: 600; padding: 8px 12px; border-bottom: 1px solid var(--edge); }
td { padding: 7px 12px; border-bottom: 1px solid var(--edge); white-space: nowrap; }
tr:last-child td { border-bottom: 0; }
td.num { text-align: right; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.pill { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 11px;
  font-weight: 600; letter-spacing: 0.03em; }
.pill.buy { background: var(--accent-soft); color: var(--accent); }
.pill.sell { background: var(--bad-soft); color: var(--bad); }
.pill.no_trade, .pill.hold { background: var(--warn-soft); color: var(--warn); }
.why { color: var(--ink-faint); font-size: 12px; }
.filters { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
button.filter { font: inherit; font-size: 12px; padding: 4px 12px; border-radius: 999px;
  border: 1px solid var(--edge); background: transparent; color: var(--ink-soft);
  cursor: pointer; }
button.filter[aria-pressed="true"] { background: var(--accent-soft); color: var(--accent);
  border-color: transparent; }
footer { margin-top: 48px; padding-top: 18px; border-top: 1px solid var(--edge);
  color: var(--ink-faint); font-size: 12px; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1><span class="sym" id="symbol"></span> <span id="headline"></span></h1>
  <div class="meta">
    <span id="scored"></span><span id="generated"></span><span id="schema"></span>
  </div>
</header>

<h2>Is it calibrated</h2>
<p class="note">Whether the stated uncertainty is honest. A model that is right on average and
wrong about how sure it was is not usable, because the risk engine sizes on the spread.</p>
<div class="cards" id="calibration-cards"></div>

<h2>What it predicted, and what happened</h2>
<div class="panel">
  <div class="chart"><canvas id="fan" height="320"></canvas></div>
  <div class="legend">
    <span><i class="swatch" style="background:var(--accent-soft)"></i>90% credible band</span>
    <span><i class="swatch" style="background:var(--accent)"></i>posterior mean</span>
    <span><i class="swatch" style="background:var(--ink-faint)"></i>realised return</span>
  </div>
</div>

<div class="split" style="margin-top:16px">
  <div class="panel">
    <h2 style="margin-top:0">Coverage</h2>
    <p class="note">Nominal against empirical. Below the diagonal is overconfident: the
    intervals are too narrow and the losses will be bigger than the model said.</p>
    <div class="chart"><canvas id="coverage" height="260"></canvas></div>
  </div>
  <div class="panel">
    <h2 style="margin-top:0">Who was carrying the opinion</h2>
    <p class="note">Pool weights over time. A fused number says what the system believed; only
    the components say why.</p>
    <div class="chart"><canvas id="weights" height="260"></canvas></div>
    <div class="legend" id="weights-legend"></div>
  </div>
</div>

<h2>Decisions</h2>
<p class="note">Every bar the system acted on or refused to, with the gate that stopped it.
Declining because the uncertainty was too high is an outcome, not a missing trade.</p>
<div class="filters" id="filters"></div>
<div class="panel scroll"><table id="decisions">
  <thead><tr>
    <th>As of</th><th>Action</th><th class="num">P(up)</th><th class="num">Mean</th>
    <th class="num">Requested</th><th class="num">Approved</th><th>Why not</th>
  </tr></thead>
  <tbody></tbody>
</table></div>

<h2>Risk limits in force</h2>
<div class="cards" id="limits"></div>

<footer id="footer"></footer>
</div>

<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
"use strict";
const DATA = JSON.parse(document.getElementById("payload").textContent);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const fmt = (value, digits = 4) =>
  value === null || value === undefined || !isFinite(value) ? "—" : Number(value).toFixed(digits);
const pct = (value, digits = 1) =>
  value === null || value === undefined || !isFinite(value) ? "—" : (100 * value).toFixed(digits) + "%";

document.getElementById("symbol").textContent = DATA.symbol || "";
document.getElementById("headline").textContent = "posterior review";
document.getElementById("scored").textContent =
  DATA.n_scored + " scored of " + DATA.n_examples + " examples";
document.getElementById("generated").textContent = (DATA.generated_at || "").replace("T", " ");
document.getElementById("schema").textContent = "schema " + DATA.schema;

/* Cards ------------------------------------------------------------------ */
const cal = DATA.calibration || {};
function card(label, value, hint, flag) {
  const el = document.createElement("div");
  el.className = "card" + (flag ? " flag-" + flag : "");
  el.innerHTML = '<div class="label"></div><div class="value"></div><div class="hint"></div>';
  el.querySelector(".label").textContent = label;
  const slot = el.querySelector(".value");
  // A word does not fit the width a number does, and clipping the verdict is the one thing
  // on this page that must stay readable.
  if (!isFinite(parseFloat(value))) slot.classList.add("word");
  slot.textContent = value;
  el.querySelector(".hint").textContent = hint || "";
  return el;
}
const cards = document.getElementById("calibration-cards");
cards.append(
  card("Verdict", cal.overconfident ? "overconfident" : "calibrated",
       cal.overconfident ? "intervals too narrow" : "intervals hold up",
       cal.overconfident ? "bad" : "good"),
  card("Interval error", fmt(cal.interval_error, 4), "mean |nominal − empirical|",
       cal.interval_error > 0.1 ? "warn" : null),
  card("Expected calib. error", fmt(cal.expected_calibration_error, 4), "probability bins"),
  card("Brier score", fmt(cal.brier_score, 4), "lower is better"),
  card("PIT KS", fmt(cal.pit_ks, 4), "uniformity of the PIT"),
  card("Mean log score", fmt(cal.mean_log_score, 4), "higher is better"),
  card("Sharpness", fmt(cal.sharpness, 5), "mean predictive sd"),
  card("Bias", fmt(cal.bias, 5), "mean signed error"),
);

/* Canvas plumbing -------------------------------------------------------- */
function setup(canvas, pad) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || canvas.parentElement.clientWidth;
  const height = Number(canvas.getAttribute("height"));
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return { ctx, width, height, pad,
           iw: width - pad.l - pad.r, ih: height - pad.t - pad.b };
}
function axes(view, xLabels, yTicks, yFormat) {
  const { ctx, pad, iw, ih } = view;
  ctx.strokeStyle = css("--grid");
  ctx.fillStyle = css("--ink-faint");
  ctx.lineWidth = 1;
  ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
  yTicks.forEach((tick) => {
    const y = pad.t + ih - tick.at * ih;
    ctx.beginPath();
    ctx.moveTo(pad.l, Math.round(y) + 0.5);
    ctx.lineTo(pad.l + iw, Math.round(y) + 0.5);
    ctx.stroke();
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(yFormat(tick.value), pad.l - 8, y);
  });
  ctx.textBaseline = "top";
  xLabels.forEach((label) => {
    // The first and last labels sit on the edges, and a centred one there hangs off the
    // canvas and is cut in half.
    ctx.textAlign = label.at <= 0.01 ? "left" : label.at >= 0.99 ? "right" : "center";
    ctx.fillText(label.text, pad.l + label.at * iw, pad.t + ih + 8);
  });
}

/* Fan chart -------------------------------------------------------------- */
const series = DATA.series || [];
(function fan() {
  const canvas = document.getElementById("fan");
  const view = setup(canvas, { l: 62, r: 12, t: 12, b: 26 });
  if (!series.length) return;
  const { ctx, pad, iw, ih } = view;
  const values = [];
  series.forEach((point) => {
    values.push(point.lower, point.upper, point.mean);
    if (point.outcome !== null && point.outcome !== undefined) values.push(point.outcome);
  });
  const finite = values.filter((value) => isFinite(value));
  let lo = Math.min.apply(null, finite), hi = Math.max.apply(null, finite);
  if (lo === hi) { lo -= 1e-4; hi += 1e-4; }
  const span = hi - lo;
  lo -= span * 0.06; hi += span * 0.06;
  const x = (index) => pad.l + (series.length === 1 ? 0.5 : index / (series.length - 1)) * iw;
  const y = (value) => pad.t + ih - ((value - lo) / (hi - lo)) * ih;

  const ticks = [0, 0.25, 0.5, 0.75, 1].map((at) => ({ at, value: lo + at * (hi - lo) }));
  const labels = [0, 0.5, 1].map((at) => ({
    at, text: (series[Math.round(at * (series.length - 1))].as_of || "").slice(0, 10),
  }));
  axes(view, labels, ticks, (value) => value.toFixed(4));

  ctx.beginPath();
  series.forEach((point, index) => {
    const px = x(index), py = y(point.upper);
    index === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  });
  for (let index = series.length - 1; index >= 0; index -= 1) {
    ctx.lineTo(x(index), y(series[index].lower));
  }
  ctx.closePath();
  ctx.fillStyle = css("--accent-soft");
  ctx.fill();

  ctx.beginPath();
  series.forEach((point, index) => {
    const px = x(index), py = y(point.mean);
    index === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  });
  ctx.strokeStyle = css("--accent");
  ctx.lineWidth = 1.5;
  ctx.stroke();

  ctx.fillStyle = css("--ink-faint");
  series.forEach((point, index) => {
    if (point.outcome === null || point.outcome === undefined || !isFinite(point.outcome)) return;
    ctx.beginPath();
    ctx.arc(x(index), y(point.outcome), 1.4, 0, Math.PI * 2);
    ctx.fill();
  });
})();

/* Coverage --------------------------------------------------------------- */
(function coverage() {
  const canvas = document.getElementById("coverage");
  const view = setup(canvas, { l: 46, r: 12, t: 12, b: 26 });
  const points = (cal.coverage || []).slice().sort((a, b) => a.level - b.level);
  if (!points.length) return;
  const { ctx, pad, iw, ih } = view;
  const x = (value) => pad.l + value * iw;
  const y = (value) => pad.t + ih - value * ih;
  axes(view,
       [0, 0.5, 1].map((at) => ({ at, text: at.toFixed(1) })),
       [0, 0.25, 0.5, 0.75, 1].map((at) => ({ at, value: at })),
       (value) => value.toFixed(2));

  ctx.strokeStyle = css("--grid");
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(x(0), y(0)); ctx.lineTo(x(1), y(1)); ctx.stroke();
  ctx.setLineDash([]);

  ctx.beginPath();
  points.forEach((point, index) => {
    const px = x(point.level), py = y(point.empirical);
    index === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  });
  ctx.strokeStyle = css("--accent");
  ctx.lineWidth = 1.8;
  ctx.stroke();
  points.forEach((point) => {
    ctx.beginPath();
    ctx.arc(x(point.level), y(point.empirical), 3, 0, Math.PI * 2);
    ctx.fillStyle = point.empirical < point.level ? css("--bad") : css("--good");
    ctx.fill();
  });
})();

/* Model weights ---------------------------------------------------------- */
(function weights() {
  const canvas = document.getElementById("weights");
  const view = setup(canvas, { l: 46, r: 12, t: 12, b: 26 });
  const names = DATA.models || [];
  if (!series.length || !names.length) return;
  const { ctx, pad, iw, ih } = view;
  const palette = [css("--accent"), css("--warn"), css("--good"), css("--bad"),
                   css("--ink-soft")];
  axes(view,
       [0, 0.5, 1].map((at) => ({
         at, text: (series[Math.round(at * (series.length - 1))].as_of || "").slice(0, 7),
       })),
       [0, 0.5, 1].map((at) => ({ at, value: at })),
       (value) => value.toFixed(1));

  const x = (index) => pad.l + (series.length === 1 ? 0.5 : index / (series.length - 1)) * iw;
  const y = (value) => pad.t + ih - value * ih;
  const floor = new Array(series.length).fill(0);
  names.forEach((name, model) => {
    ctx.beginPath();
    series.forEach((point, index) => {
      const px = x(index), py = y(floor[index]);
      index === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    });
    for (let index = series.length - 1; index >= 0; index -= 1) {
      const weight = (series[index].weights || [])[model] || 0;
      ctx.lineTo(x(index), y(floor[index] + weight));
    }
    ctx.closePath();
    ctx.fillStyle = palette[model % palette.length];
    ctx.globalAlpha = 0.55;
    ctx.fill();
    ctx.globalAlpha = 1;
    series.forEach((point, index) => {
      floor[index] += (point.weights || [])[model] || 0;
    });
  });

  const legend = document.getElementById("weights-legend");
  legend.textContent = "";
  names.forEach((name, model) => {
    const span = document.createElement("span");
    const swatch = document.createElement("i");
    swatch.className = "swatch";
    swatch.style.background = palette[model % palette.length];
    span.append(swatch, document.createTextNode(
      name + "  " + fmt((DATA.reliabilities || [])[model], 3)));
    legend.append(span);
  });
})();

/* Decisions -------------------------------------------------------------- */
const decisions = DATA.decisions || [];
const byTime = new Map(series.map((point) => [point.as_of, point]));
let active = "all";
function renderDecisions() {
  const body = document.querySelector("#decisions tbody");
  body.textContent = "";
  const rows = decisions.filter((row) => active === "all" || row.action === active);
  if (!rows.length) {
    const cell = document.createElement("td");
    cell.colSpan = 7;
    cell.className = "why";
    cell.textContent = decisions.length
      ? "No decisions of that kind."
      : "No decisions in this payload. Pass a portfolio to dashboard_payload to record them.";
    const row = document.createElement("tr");
    row.append(cell);
    body.append(row);
    return;
  }
  rows.slice(0, 500).forEach((row) => {
    const point = byTime.get(row.as_of) || {};
    const tr = document.createElement("tr");
    const cells = [
      ["", (row.as_of || "").replace("T", " ")],
      ["action", row.action],
      ["num", pct(point.probability_up, 1)],
      ["num", fmt(point.mean, 5)],
      ["num", pct(row.requested, 1)],
      ["num", pct(row.approved, 1)],
      ["why", (row.failures || []).join(", ") || (row.reason || "")],
    ];
    cells.forEach(([kind, text]) => {
      const td = document.createElement("td");
      if (kind === "action") {
        const pill = document.createElement("span");
        pill.className = "pill " + text;
        pill.textContent = (text || "").replace("_", " ");
        td.append(pill);
      } else {
        td.className = kind;
        td.textContent = text;
      }
      tr.append(td);
    });
    body.append(tr);
  });
}
(function filters() {
  const holder = document.getElementById("filters");
  const counts = new Map();
  decisions.forEach((row) => counts.set(row.action, (counts.get(row.action) || 0) + 1));
  const kinds = ["all"].concat(Array.from(counts.keys()).sort());
  kinds.forEach((kind) => {
    const button = document.createElement("button");
    button.className = "filter";
    button.type = "button";
    button.setAttribute("aria-pressed", String(kind === "all"));
    button.textContent = kind === "all"
      ? "all  " + decisions.length
      : kind.replace("_", " ") + "  " + counts.get(kind);
    button.addEventListener("click", () => {
      active = kind;
      holder.querySelectorAll("button").forEach((other) =>
        other.setAttribute("aria-pressed", String(other === button)));
      renderDecisions();
    });
    holder.append(button);
  });
})();
renderDecisions();

/* Limits ----------------------------------------------------------------- */
(function limits() {
  const holder = document.getElementById("limits");
  const labels = {
    max_position_weight: ["Max position", pct],
    max_portfolio_exposure: ["Max exposure", pct],
    min_probability_positive: ["Min P(up)", pct],
    max_probability_large_loss: ["Max P(large loss)", pct],
    max_model_uncertainty: ["Max uncertainty", (value) => fmt(value, 4)],
    risk_budget_per_trade: ["Risk per trade", pct],
  };
  Object.entries(DATA.limits || {}).forEach(([key, value]) => {
    const [label, format] = labels[key] || [key, (v) => fmt(v, 4)];
    holder.append(card(label, format(value, 2), ""));
  });
})();

document.getElementById("footer").textContent =
  "Generated by BayesTrade. Paper and backtest only: no order path exists in this build.";

addEventListener("resize", () => { clearTimeout(window.__redraw);
  window.__redraw = setTimeout(() => location.reload(), 250); });
</script>
</body>
</html>
"""

"""
    dashboard_page(payload; title)

The payload and the page that draws it, as one self-contained HTML string.

No external stylesheet, no script from a CDN, no font from a network. It opens from the
filesystem, which is the only property that matters for something meant to be looked at while
debugging.
"""
function dashboard_page(
        payload::AbstractDict; title::AbstractString = "", refresh_seconds::Real = 0,
    )
    heading = isempty(title) ?
        string(get(payload, "symbol", "BayesTrade"), " review") : String(title)
    refresh_seconds >= 0 ||
        throw(ArgumentError("refresh_seconds must be non-negative, got $refresh_seconds"))
    # A meta refresh rather than a poll: the page holds its whole payload, so there is nothing
    # to fetch incrementally, and reloading is both simpler and honest about what it does.
    refresh = refresh_seconds > 0 ?
        string("<meta http-equiv=\"refresh\" content=\"", round(Int, refresh_seconds), "\">\n") : ""
    return replace(
        DASHBOARD_PAGE_TEMPLATE,
        "__PAYLOAD__" => embed_json(payload),
        "__TITLE__" => replace(heading, "&" => "&amp;", "<" => "&lt;", ">" => "&gt;"),
        "__REFRESH__" => refresh,
    )
end

"""
    write_dashboard_page(payload, path; title)

Write the page somewhere a browser can open it, and return the path.
"""
function write_dashboard_page(
        payload::AbstractDict, path::AbstractString; title::AbstractString = "",
        refresh_seconds::Real = 0,
    )
    mkpath(dirname(abspath(path)))
    write(
        path,
        dashboard_page(payload; title = title, refresh_seconds = refresh_seconds),
    )
    return abspath(path)
end

"""
    serve_dashboard(producer; port, host, title, refresh_seconds, open_browser)

Serve the dashboard on this machine, rebuilding it on every request.

In the HTTP extension, and raises with an explanation when HTTP is not loaded. `producer` is
called per request, so a session that is still running shows its current state rather than the
state it had when the server started.
"""
function serve_dashboard(args...; kwargs...)
    return throw(
        ArgumentError(
            "serve_dashboard lives in a package extension: run `using HTTP` first, " *
                "after adding HTTP to your environment with `Pkg.add(\"HTTP\")`",
        ),
    )
end
