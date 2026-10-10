"""Build docs/analysis/cost_latency.html (one page) and cost_latency.pdf.

All figures are estimates computed from the assumptions below; edit them and re-run:
    .venv/bin/python docs/analysis/make_report.py
"""
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- assumptions
# Judge prompt, from the regulatory example (Qwen tokenizer; other tokenizers differ by ~10-20 %).
SYS_TOKENS = 700        # instructions (~250) + 7 class definitions (348 measured) + output format
MSG_TOKENS = 50         # measured: test mean 45, OOD mean 54
WRAP_TOKENS = 30        # speaker tag, delimiters
IN_TOKENS = SYS_TOKENS + MSG_TOKENS + WRAP_TOKENS
LABEL_TOKENS = 20       # {"label": "reg_bi"} plus a few tokens
VOLUME_PER_DAY = 1_000_000
DAYS = 30

# name, $/M in, $/M out, thinking tokens, base TTFT s, output tok/s, benchmark first-chunk s (reasoning tasks), colour, note
JUDGES = [
    dict(name="Llama 4 Maverick", sub="open weights, DeepInfra", pin=0.20, pout=0.80, think=0, ttft=0.83, tps=27,
         bench="0.83 s (DeepInfra) to 1.39 s (Azure)", col="#5b7aa6"),
    dict(name="Gemini 2.5 Flash", sub="thinking off", pin=0.30, pout=2.50, think=0, ttft=0.46, tps=189,
         bench="0.46 s", col="#c8a24d"),
    dict(name="Gemini 3.1 Pro", sub="thinking on, ~300 tokens", pin=2.00, pout=12.00, think=300, ttft=1.0, tps=114,
         bench="24.8 s on long reasoning tasks", col="#a8822e"),
    dict(name="Claude Opus 5.5", sub="effort low, ~200 thinking tokens", pin=4.00, pout=20.00, think=200, ttft=1.5, tps=71,
         bench="8.7 s at low effort on long tasks", col="#96151d"),
]
# CLM: frozen Qwen3-8B encoder + two 512-d heads; one forward pass, no generation.
CLM_LAT_MEASURED = 0.08     # s, warm server end to end, Apple M5 Pro 24 GB (encoder 54 ms, heads 0.3 ms)
GPU_HOURLY = 0.80           # $/h, 24 GB L4-class cloud GPU on demand (estimate)
GPU_MSGS_PER_S = 20         # conservative: measured 22 msg/s on the M5 Pro; L4 prefill should be similar or faster
HOURS_MONTH = 730

monthly_vol = VOLUME_PER_DAY * DAYS
for j in JUDGES:
    out = LABEL_TOKENS + j["think"]
    j["per_call"] = IN_TOKENS * j["pin"] / 1e6 + out * j["pout"] / 1e6
    j["per_m"] = j["per_call"] * 1e6
    j["month"] = j["per_call"] * monthly_vol
    j["lat"] = j["ttft"] + out / j["tps"]
    j["out"] = out

need_gpus = max(1, math.ceil(VOLUME_PER_DAY / 86400 * 2.5 / GPU_MSGS_PER_S))   # 2.5x peak-to-average headroom
clm_month = need_gpus * GPU_HOURLY * HOURS_MONTH
clm_per_m = clm_month / monthly_vol * 1e6
clm_full_util_per_m = GPU_HOURLY / (GPU_MSGS_PER_S * 3600) * 1e6
CLM = dict(name="Trained CLM", sub=f"{need_gpus}× L4 GPU, always on", per_m=clm_per_m, month=clm_month,
           lat=CLM_LAT_MEASURED, col="#0b2a4a")
breakeven_day = {j["name"]: clm_month / j["per_call"] / DAYS for j in JUDGES}
rows = JUDGES + [CLM]


def money(x):
    if x >= 1000:
        return f"${x/1000:,.1f}K" if x < 1e6 else f"${x/1e6:,.2f}M"
    if x >= 10:
        return f"${x:,.0f}"
    return f"${x:,.2f}"


def secs(x):
    return f"{x*1000:.0f} ms" if x < 1 else f"{x:.1f} s"


def bar_chart(items, key, fmt, lo, hi, title, unit_ticks):
    """Horizontal log-scale bar chart as inline SVG."""
    W, rowh, left, right, top = 560, 27, 118, 62, 24
    H = top + rowh * len(items) + 22
    span = math.log10(hi) - math.log10(lo)
    x = lambda v: left + (math.log10(max(v, lo)) - math.log10(lo)) / span * (W - left - right)
    s = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="{title}">',
         f'<text x="0" y="13" class="ct">{title}</text>']
    for t, lab in unit_ticks:
        s.append(f'<line x1="{x(t):.1f}" y1="{top-4}" x2="{x(t):.1f}" y2="{H-20}" class="grid"/>'
                 f'<text x="{x(t):.1f}" y="{H-7}" class="tick" text-anchor="middle">{lab}</text>')
    for i, it in enumerate(items):
        y = top + i * rowh
        v = it[key]
        s.append(f'<text x="{left-8}" y="{y+13}" class="lab" text-anchor="end">{it["name"]}</text>'
                 f'<rect x="{left}" y="{y+3}" width="{x(v)-left:.1f}" height="16" rx="3" fill="{it["col"]}"/>'
                 f'<text x="{x(v)+5:.1f}" y="{y+15}" class="val">{fmt(v)}</text>')
    s.append("</svg>")
    return "".join(s)


cost_chart = bar_chart(rows, "per_m", money, 1, 30000, "Cost per 1M classifications (log scale)",
                       [(1, "$1"), (10, "$10"), (100, "$100"), (1000, "$1K"), (10000, "$10K")])
lat_chart = bar_chart(rows, "lat", secs, 0.01, 30, "Latency per decision (log scale)",
                      [(0.01, "10 ms"), (0.1, "100 ms"), (1, "1 s"), (10, "10 s")])

judge_rows = "".join(
    f'<tr><td><i class="sw" style="background:{j["col"]}"></i><b>{j["name"]}</b><br><span class="m">{j["sub"]}</span></td>'
    f'<td>${j["pin"]:.2f} / ${j["pout"]:.2f}</td><td>{IN_TOKENS} / {j["out"]}</td><td>{money(j["per_m"])}</td>'
    f'<td><b>{money(j["month"])}</b></td><td>{secs(j["lat"])}</td><td class="m">{j["bench"]}</td>'
    f'<td>{j["per_m"]/clm_per_m:,.0f}×</td></tr>' for j in JUDGES)
clm_row = (f'<tr class="clm"><td><i class="sw" style="background:{CLM["col"]}"></i><b>Trained CLM</b><br>'
           f'<span class="m">{CLM["sub"]}</span></td><td>${GPU_HOURLY:.2f}/GPU-h</td><td>~80 / 0</td>'
           f'<td>{money(clm_per_m)}</td><td><b>{money(clm_month)}</b></td><td>{secs(CLM_LAT_MEASURED)} <span class="m">measured</span></td>'
           f'<td class="m">54 ms encoder + 0.3 ms heads</td><td>1×</td></tr>')
be = breakeven_day["Llama 4 Maverick"]

html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cost and Latency</title>
<style>
:root {{ --navy:#0b2a4a; --brand:#96151d; --gold:#c8a24d; --bg:#f6f4f1; --ink:#1d2733; --mute:#5d6a78; --line:#dcd6cc; --card:#fff; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#0f1720; --ink:#e7ebf0; --mute:#9aa7b5; --line:#2a3644; --card:#16212d; }} }}
:root[data-theme="dark"] {{ --bg:#0f1720; --ink:#e7ebf0; --mute:#9aa7b5; --line:#2a3644; --card:#16212d; }}
@page {{ size: Letter landscape; margin: 0.32in; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:10.5px/1.38 -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Arial, sans-serif; }}
.page {{ max-width:1060px; margin:0 auto; padding:14px 16px; }}
header {{ display:flex; justify-content:space-between; align-items:flex-end; border-bottom:3px solid var(--navy); padding-bottom:6px; }}
h1 {{ font-size:19px; margin:0; color:var(--navy); letter-spacing:-.2px; }}
header .r {{ text-align:right; color:var(--mute); font-size:9.5px; }}
.lede {{ margin:7px 0 8px; font-size:11.5px; }}
.lede b {{ color:var(--brand); }}
.kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin-bottom:8px; }}
.kpi {{ background:var(--card); border:1px solid var(--line); border-top:3px solid var(--gold); border-radius:6px; padding:6px 9px; }}
.kpi .v {{ font-size:18px; font-weight:700; color:var(--navy); }}
.kpi .k {{ color:var(--mute); font-size:9.5px; }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:6px; padding:8px 10px; }}
.chart {{ width:100%; height:auto; display:block; }}
.chart .ct {{ font-size:11px; font-weight:700; fill:var(--navy); }}
.chart .lab {{ font-size:10px; fill:var(--ink); }} .chart .val {{ font-size:10px; font-weight:600; fill:var(--ink); }}
.chart .tick {{ font-size:8.5px; fill:var(--mute); }} .chart .grid {{ stroke:var(--line); stroke-width:1; }}
h2 {{ font-size:11.5px; color:var(--navy); margin:8px 0 4px; text-transform:uppercase; letter-spacing:.5px; }}
table {{ width:100%; border-collapse:collapse; font-size:9.8px; }}
th {{ text-align:left; background:var(--navy); color:#fff; font-weight:600; padding:4px 6px; }}
td {{ padding:3.5px 6px; border-bottom:1px solid var(--line); vertical-align:top; }}
tr.clm td {{ background:rgba(200,162,77,.14); }}
.m {{ color:var(--mute); font-size:9px; }}
.sw {{ display:inline-block; width:8px; height:8px; border-radius:2px; margin-right:5px; }}
.cols3 {{ display:grid; grid-template-columns:1.15fr 1fr 1fr; gap:8px; margin-top:8px; }}
ul {{ margin:2px 0 0; padding-left:15px; }} li {{ margin:1px 0; }}
.src {{ margin-top:6px; color:var(--mute); font-size:8.3px; line-height:1.3; }}
.src a {{ color:var(--mute); }}
@media (max-width:760px) {{ .kpis, .grid2, .cols3 {{ grid-template-columns:1fr; }} .tablewrap {{ overflow-x:auto; }} header {{ display:block; }} header .r {{ text-align:left; }} }}
@media print {{ body {{ background:#fff; }} .page {{ padding:0; max-width:none; zoom:.87; }} }}
</style></head>
<body><div class="page">
<header><div><h1>Cost and latency: prompt-based LLM judges vs a trained CLM classifier</h1>
<div class="m">Worked example: the 5-class regulatory screener (one decision per chat message)</div></div>
<div class="r">CLASP · estimates as of 2026-10-10<br>Volume scenario: {VOLUME_PER_DAY:,} classifications / day</div></header>

<p class="lede">At this volume, the trained CLM costs about <b>{money(clm_month)} a month</b> on {"one rented GPU" if need_gpus == 1 else f"{need_gpus} rented GPUs"}.
A prompt-based judge costs <b>{money(JUDGES[0]['month'])} to {money(JUDGES[3]['month'])}</b> a month, depending on the model.
The CLM answers in <b>{secs(CLM_LAT_MEASURED)}</b>, while the judges take <b>{secs(JUDGES[1]['lat'])} to {secs(JUDGES[3]['lat'])}</b>.
The CLM runs one encoder pass and generates nothing, so its cost does not depend on prompt length or reasoning tokens.</p>

<div class="kpis">
<div class="kpi"><div class="v">{JUDGES[0]['per_m']/clm_per_m:,.0f}× – {JUDGES[3]['per_m']/clm_per_m:,.0f}×</div><div class="k">cheaper than the judges per classification, at this volume</div></div>
<div class="kpi"><div class="v">{JUDGES[1]['lat']/CLM_LAT_MEASURED:,.0f}× – {JUDGES[3]['lat']/CLM_LAT_MEASURED:,.0f}×</div><div class="k">faster per decision (CLM measured on an Apple M5 Pro)</div></div>
<div class="kpi"><div class="v">~{be/1000:,.0f}K / day</div><div class="k">break-even volume against the cheapest judge (Llama 4 Maverick)</div></div>
<div class="kpi"><div class="v">0.983 / 0.800</div><div class="k">CLM macro-F1 measured: in-distribution test / hard out-of-distribution. Judge accuracy not measured.</div></div>
</div>

<div class="grid2">
<div class="card">{cost_chart}</div>
<div class="card">{lat_chart}</div>
</div>

<h2>Per-model estimate</h2>
<div class="tablewrap"><table>
<tr><th>Approach</th><th>Price in / out ($ per 1M tokens)</th><th>Tokens in / out per call</th><th>Per 1M calls</th><th>Per month ({monthly_vol/1e6:.0f}M calls)</th><th>Latency (estimate)</th><th>Published first-token benchmark</th><th>Cost vs CLM</th></tr>
{judge_rows}{clm_row}
</table></div>

<div class="cols3">
<div><h2>How the numbers are built</h2><ul>
<li><b>Judge input</b> is {IN_TOKENS} tokens: about 250 for instructions, 348 for the 7 class definitions (measured), plus a {MSG_TOKENS}-token message (measured mean 45 in test, 54 out of distribution).</li>
<li><b>Judge output</b> is a {LABEL_TOKENS}-token JSON label. The reasoning models also pay for thinking tokens at the output price, and Opus 5.5 always thinks.</li>
<li><b>Judge latency</b> is time to first token plus output tokens divided by the published speed. Benchmarks on long reasoning tasks are far slower, so treat them as an upper bound.</li>
<li><b>CLM cost</b> assumes {need_gpus} always-on 24 GB L4-class GPU{"s" if need_gpus > 1 else ""} at ${GPU_HOURLY:.2f}/h. Each handles {GPU_MSGS_PER_S} msg/s against an average load of {VOLUME_PER_DAY/86400:.1f} msg/s with 2.5× peak headroom. At full utilisation the cost falls to {money(clm_full_util_per_m)} per 1M.</li>
</ul></div>
<div><h2>What moves the result</h2><ul>
<li><b>Prompt size.</b> Adding few-shot examples to reach 2,000 input tokens raises every judge's cost by about 2.5×. Prompt caching only helps prefixes above the provider's minimum cacheable length.</li>
<li><b>Batch APIs</b> from Anthropic and Google halve the price, but only for offline jobs that can wait hours.</li>
<li><b>Low volume favours the judges.</b> Below about {be/1000:,.0f}K classifications a day, the rented GPUs cost more than the Llama judge. A CLM on hardware you already own costs only the electricity.</li>
<li><b>Provider spread.</b> Llama 4 Maverick input prices range from $0.15 to $0.50 per 1M tokens, and speed ranges from 12 to 272 tokens/s.</li>
</ul></div>
<div><h2>Not in the per-call figures</h2><ul>
<li><b>Accuracy.</b> The CLM's quality is measured. None of the judges have been evaluated on this task yet. The next step is to run them on the same test and out-of-distribution files.</li>
<li><b>CLM one-off costs.</b> These are labelled data and a few minutes of head training on cached embeddings. Embedding the training set takes minutes to hours.</li>
<li><b>Judge one-off costs.</b> These are prompt engineering and a labelled set to validate the prompt against.</li>
<li><b>Data control.</b> The CLM keeps messages in-house. The hosted judges send every message to a third party.</li>
</ul></div>
</div>

<div class="src"><b>Sources (retrieved 2026-10-10; prices and speeds change often).</b>
Prices: Anthropic pricing for Claude Opus 5.5; Google Gemini API pricing via
<a href="https://www.cloudzero.com/blog/gemini-pricing">cloudzero.com</a> and <a href="https://www.morphllm.com/gemini-api-pricing">morphllm.com</a>;
Llama 4 Maverick via <a href="https://artificialanalysis.ai/models/llama-4-maverick/providers">artificialanalysis.ai</a> and <a href="https://pricepertoken.com/pricing-page/model/meta-llama-llama-4-maverick">pricepertoken.com</a>.
Speed and first-token latency: <a href="https://artificialanalysis.ai/providers/anthropic">artificialanalysis.ai/providers/anthropic</a>,
<a href="https://artificialanalysis.ai/providers/google">/providers/google</a>, <a href="https://artificialanalysis.ai/models/llama-4-maverick/providers">/models/llama-4-maverick/providers</a> (medians over 72 h, 10K-token workload).
CLM latency and accuracy: this repo, Tests/METRICS.md (warm server, Apple M5 Pro, 24 GB). GPU price and CLM throughput on an L4 are estimates, not measurements.</div>
</div></body></html>
"""

out_html = os.path.join(HERE, "cost_latency.html")
open(out_html, "w").write(html)
print("wrote", out_html)
for r in rows:
    print(f'{r["name"]:18s} per1M={money(r["per_m"]):>8s} month={money(r["month"]):>8s} lat={secs(r["lat"])}')
print("breakeven/day", {k: round(v) for k, v in breakeven_day.items()})

try:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome")
        pg = b.new_page()
        pg.emulate_media(color_scheme="light")
        pg.goto("file://" + out_html)
        pg.evaluate("document.fonts.ready")
        pdf = os.path.join(HERE, "cost_latency.pdf")
        pg.pdf(path=pdf, format="Letter", landscape=True, print_background=True, prefer_css_page_size=True)
        b.close()
    print("wrote", pdf)
except ImportError:
    print("playwright not installed: pip install -r scripts/demo/requirements-demo.txt to also write the PDF")
