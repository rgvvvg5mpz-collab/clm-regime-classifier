"""Render Tests/<dated run>/report.html for every run plus the Tests/index.html summary.

    .venv/bin/python Tests/make_reports.py

Each dated folder is one test run. A folder is recognised by what it contains:
  eval_test/metrics.json + training_metrics.json   -> in-distribution training + test run
  metrics.json with "ood_axis" slices              -> OOD validation run
  turns.jsonl + feedback.jsonl                     -> UI end-to-end run
Add a new dated folder with the same artefacts and re-run this script.
"""
from __future__ import annotations

import glob
import html
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CSS = open(os.path.join(HERE, "..", "docs", "style.css")).read()
METHODS = [("clm_finetuned_unit_masked", "CLM-8B fine-tuned heads (unit-masked)"),
           ("clm_finetuned", "CLM-8B fine-tuned heads"),
           ("linear_probe", "Linear probe on Qwen3-8B embeddings"),
           ("clm_zero_shot", "CLM-8B zero-shot (reference heads)")]
NAMES = {"compliant": "Compliant", "no_flag": "No flag", "finra_2210": "FINRA 2210", "reg_bi": "Reg BI",
         "finra_4530": "FINRA 4530", "sec_17a3_17a4": "SEC 17a-3/4", "reg_sp": "Reg S-P",
         "reg_sid": "Reg S-ID"}
e = html.escape
nm = lambda l: NAMES.get(l, l)   # user-trained models may have their own class keys


def page(title: str, body: str, depth: int = 1) -> str:
    up = "../" * depth
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{e(title)}</title>
<style>{CSS}</style></head><body><div class="wrap">
<nav class="top"><a href="{up}Tests/index.html">All test runs</a><a href="{up}docs/README.html">Docs</a>
<a href="{up}docs/methodology.html">Methodology</a><a href="{up}docs/architecture.html">Architecture</a></nav>
{body}</div></body></html>"""


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def ci(m: dict) -> str:
    c = m.get("macro_f1_95ci")
    return f'<span class="muted">[{c[0]:.3f}, {c[1]:.3f}]</span>' if c else ""


def methods_table(M: dict) -> str:
    rows = "".join(
        f"<tr><td>{e(label)}</td><td class='num'>{pct(M[k]['accuracy'])}</td>"
        f"<td class='num'><b>{M[k]['macro_f1']:.3f}</b></td><td class='num'>{ci(M[k])}</td></tr>"
        for k, label in METHODS if k in M)
    return ("<div class='scroll'><table><tr><th>Method</th><th class='num'>Accuracy</th>"
            "<th class='num'>Macro-F1</th><th class='num'>95% CI (bootstrap)</th></tr>" + rows + "</table></div>")


def per_class_table(M: dict, keys=("clm_finetuned_unit_masked", "linear_probe")) -> str:
    labels = M["labels"]
    head = "".join(f"<th class='num'>{e(dict(METHODS)[k].split(' (')[0])} F1</th>" for k in keys if k in M)
    best = keys[0]
    rows = ""
    for l in labels:
        pc = M[best]["per_class"][l]
        rows += (f"<tr><td>{nm(l)}</td><td class='num'>{pc['precision']:.3f}</td><td class='num'>{pc['recall']:.3f}</td>"
                 + "".join(f"<td class='num'>{M[k]['per_class'][l]['f1-score']:.3f}</td>" for k in keys if k in M)
                 + f"<td class='num'>{int(pc['support'])}</td></tr>")
    return ("<div class='scroll'><table><tr><th>Class</th><th class='num'>Precision</th><th class='num'>Recall</th>"
            + head + "<th class='num'>n</th></tr>" + rows + "</table></div>")


def confusion(m: dict, labels: list[str]) -> str:
    cm = m["confusion"]
    head = "".join(f"<th class='rot num'>{nm(l)}</th>" for l in labels)
    rows = ""
    for l, r in zip(labels, cm):
        tot = sum(r) or 1
        cells = ""
        for j, v in enumerate(r):
            a = v / tot
            fg = "color:#fff;" if a > 0.55 else ""
            cells += f"<td class='cell' style='background:rgba(var(--heat, 47, 91, 211),{a:.2f});{fg}'>{v}</td>"
        rows += f"<tr><th class='rot'>{nm(l)}</th>{cells}</tr>"
    return ("<div class='scroll'><table class='heat'><tr><th class='rot'>gold ↓ / predicted →</th>"
            + head + "</tr>" + rows + "</table></div>")


def line_chart(series: dict[str, list[float]], ylabel: str) -> str:
    """Tiny inline SVG line chart (one line per series)."""
    W, H, L, B, T, R = 640, 240, 48, 30, 12, 110
    n = max(len(v) for v in series.values())
    lo = min(min(v) for v in series.values()); lo = max(0.0, (int(lo * 20) / 20))
    hi = 1.0
    X = lambda i: L + (W - L - R) * i / max(1, n - 1)
    Y = lambda v: T + (H - T - B) * (1 - (v - lo) / (hi - lo))
    colors = ["#2f5bd3", "#c2410c", "#15803d", "#7c3aed"]
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{e(ylabel)} by epoch">']
    for k in range(6):
        v = lo + (hi - lo) * k / 5
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="currentColor" stroke-opacity=".12"/>'
                   f'<text x="{L - 6}" y="{Y(v) + 4:.1f}" font-size="11" text-anchor="end" fill="currentColor" opacity=".6">{v:.2f}</text>')
    for i in range(0, n, 5):
        out.append(f'<text x="{X(i):.1f}" y="{H - 10}" font-size="11" text-anchor="middle" fill="currentColor" opacity=".6">{i + 1}</text>')
    for c, (name, vals) in zip(colors, series.items()):
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2"/>')
        out.append(f'<text x="{X(len(vals) - 1) + 8:.1f}" y="{Y(vals[-1]) + 4:.1f}" font-size="12" fill="{c}">{e(name)}</text>')
    out.append("</svg>")
    return "<figure>" + "".join(out) + f"<figcaption>{e(ylabel)} per epoch (x-axis: epoch)</figcaption></figure>"


# ------------------------------------------------------------------ run types
def in_distribution(d: str) -> dict:
    M = json.load(open(os.path.join(d, "eval_test", "metrics.json")))
    T = json.load(open(os.path.join(d, "training_metrics.json")))
    lat = json.load(open(os.path.join(d, "encoder_latency.json"))) if os.path.exists(os.path.join(d, "encoder_latency.json")) else {}
    sweep = {}
    for f in sorted(glob.glob(os.path.join(d, "sweep", "lr_*", "metrics.json"))):
        s = json.load(open(f))
        sweep[os.path.basename(os.path.dirname(f)).replace("lr_", "lr ")] = s["clm_finetuned"]
    zs = json.load(open(os.path.join(d, "zero_shot_variants.json"))) if os.path.exists(os.path.join(d, "zero_shot_variants.json")) else {}
    best = M["clm_finetuned_unit_masked"]
    sweep_rows = "".join(
        f"<tr><td>{k}</td><td class='num'>{max(h['val_macro_f1'] for h in v['history']):.4f}</td>"
        f"<td class='num'>{max(v['history'], key=lambda h: h['val_macro_f1'])['epoch']}</td>"
        f"<td class='num'>{v['macro_f1']:.4f}</td></tr>" for k, v in sweep.items())
    zs_rows = "".join(f"<tr><td><code>{e(k)}</code></td><td class='num'>{pct(v['accuracy'])}</td>"
                      f"<td class='num'>{v['macro_f1']:.3f}</td></tr>" for k, v in sorted(zs.items(), key=lambda kv: -kv[1]['macro_f1']))
    hp = T["clm_finetuned"].get("hparams", {})
    best_ep = max(T["clm_finetuned"]["history"], key=lambda h: h["val_macro_f1"])["epoch"]
    label_set = M.get("label_set", "7way")
    date = os.path.basename(d)[:10]
    body = f"""<h1>In-distribution test, {len(M['labels'])}-class ({e(label_set)}) — {date}</h1>
<p class="lede">RegModels v2 test split ({M['n']:,} messages, {len(M['labels'])} classes: {", ".join(nm(l) for l in M['labels'])}),
never seen in training or model selection. Hyper-parameters picked on the validation split only.</p>
<div class="cards">
<div class="card"><div class="k">Macro-F1 (CLM fine-tuned)</div><div class="v">{best['macro_f1']:.3f}</div><div class="s">95% CI {best['macro_f1_95ci'][0]:.3f}–{best['macro_f1_95ci'][1]:.3f}</div></div>
<div class="card"><div class="k">Accuracy</div><div class="v">{pct(best['accuracy'])}</div><div class="s">{M['n']:,} test rows</div></div>
<div class="card"><div class="k">Linear-probe baseline</div><div class="v">{M['linear_probe']['macro_f1']:.3f}</div><div class="s">macro-F1</div></div>
<div class="card"><div class="k">Zero-shot CLM</div><div class="v">{M['clm_zero_shot']['macro_f1']:.3f}</div><div class="s">macro-F1 (chance ≈ 0.14 acc)</div></div>
</div>
<h2>Methods</h2>{methods_table(M)}
<h2>Per class</h2>{per_class_table(M)}
<h2>Confusion matrix — CLM fine-tuned</h2>{confusion(best, M['labels'])}
<h2>Learning-rate sweep (selected on validation)</h2>
<div class="scroll"><table><tr><th>Config</th><th class="num">Best val macro-F1</th><th class="num">Best epoch</th><th class="num">Test macro-F1</th></tr>{sweep_rows}</table></div>
{line_chart({k: [h['val_macro_f1'] for h in v['history']] for k, v in sweep.items()} or {"selected": [h['val_macro_f1'] for h in T['clm_finetuned']['history']]}, 'Validation macro-F1')}
<p>Selected: <b>lr {hp.get('lr', '?')}</b>, {hp.get('epochs', '?')} epochs, batch {hp.get('batch', '?')}, AdamW (wd 0.01), one-cycle schedule,
softmax-CE over the {len(M['labels'])} candidates, heads initialised from <code>CLM_v0.1-8B.pt</code>. Checkpoint = epoch {best_ep} (best validation macro-F1).</p>
{"<h2>Zero-shot ablation</h2><p>Candidate wording and a label-free prior correction (<code>+centered</code>) were varied; none get close to usable.</p><div class='scroll'><table><tr><th>Variant</th><th class='num'>Accuracy</th><th class='num'>Macro-F1</th></tr>" + zs_rows + "</table></div>" if zs_rows else ""}
<h2>Latency (Apple M5 Pro, 24 GB)</h2>
<table><tr><th>Stage</th><th class="num">ms / message</th></tr>
<tr><td>Qwen3-8B encoder (bf16, MPS, batch 16)</td><td class="num">{lat.get('encoder_ms_per_message', float('nan')):.1f}</td></tr>
<tr><td>CLM heads + softmax (CPU, candidates cached)</td><td class="num">{T['latency']['clm_heads_ms_per_decision_cpu']:.2f}</td></tr></table>
<h2>Files</h2><ul><li><code>eval_test/metrics.json</code>, <code>eval_test/predictions.jsonl</code></li>
<li><code>training_metrics.json</code> (selected run incl. per-epoch history)</li><li><code>sweep/lr_*/</code> metrics and test errors per config</li>
<li><code>zero_shot_variants.json</code>, <code>encoder_latency.json</code>, <code>sweep_logs/</code></li></ul>"""
    return {"title": f"In-distribution test ({len(M['labels'])}-class)", "body": body, "kind": f"Training + held-out test, {label_set}",
            "n": M["n"], "macro_f1": best["macro_f1"], "ci": best["macro_f1_95ci"], "acc": best["accuracy"],
            "baseline": M["linear_probe"]["macro_f1"], "M": M}


def ood(d: str) -> dict:
    """OOD validation run. Title/lede come from the data; an optional notes.html in the run
    folder is inserted verbatim after the summary cards (interpretation lives there, not here)."""
    import collections
    import statistics
    M = json.load(open(os.path.join(d, "metrics.json")))
    best = M["clm_finetuned_unit_masked"]
    preds = [json.loads(l) for l in open(os.path.join(d, "predictions.jsonl"))]
    diff = collections.Counter(p.get("difficulty") for p in preds)
    axes = collections.Counter(p.get("ood_axis") for p in preds)
    cos = [p["max_cos_to_train"] for p in preds if "max_cos_to_train" in p]
    diff_label = " / ".join(k for k, _ in sorted(diff.items(), key=lambda kv: ["low", "medium", "hard", "very_hard"].index(kv[0]) if kv[0] in ["low", "medium", "hard", "very_hard"] else 9))
    short = " ".join(t for t in os.path.basename(d).split("_")[1:] if t != "ood")
    date = os.path.basename(d)[:10]
    label_set = M.get("label_set", "7way")
    title = f"OOD validation, {diff_label} difficulty, {len(M['labels'])}-class ({label_set})"
    notes_p = os.path.join(d, "notes.html")
    notes = open(notes_p).read() if os.path.exists(notes_p) else ""
    sim = (f" Median char-n-gram similarity to the nearest training text: {statistics.median(cos):.2f}; "
           f"max {max(cos):.2f} (rows ≥ 0.80 are dropped as near-duplicates)." if cos else "")
    def slice_table(field: str) -> str:
        keys = [k for k, _ in METHODS if k in M and k != "clm_zero_shot"]
        vals = sorted(M[keys[0]][field], key=lambda v: ["low", "medium", "hard", "very_hard"].index(v) if v in ["low", "medium", "hard", "very_hard"] else v)
        head = "".join(f"<th class='num'>{e(dict(METHODS)[k].split(' (')[0])}</th>" for k in keys)
        rows = "".join(f"<tr><td>{e(v)}</td><td class='num'>{M[keys[0]][field][v]['n']}</td>"
                       + "".join(f"<td class='num'>{pct(M[k][field][v]['accuracy'])}</td>" for k in keys) + "</tr>" for v in vals)
        return f"<div class='scroll'><table><tr><th>{field.replace('by_', '')}</th><th class='num'>n</th>{head}</tr>{rows}</table></div>"
    errs = [p for p in preds if not p["correct"]]
    err_rows = "".join(
        f"<tr><td>{e(p['text'][:260])}{'…' if len(p['text']) > 260 else ''}</td><td>{nm(p['label'])}</td>"
        f"<td><span class='pill bad'>{nm(p['pred'])}</span> <span class='muted'>{p['p_pred']:.2f}</span></td>"
        f"<td class='muted'>{e(p.get('difficulty', ''))}<br>{e(p.get('ood_axis', ''))}</td></tr>" for p in errs[:25])
    missed = sum(r[0] for r in best["confusion"][1:])
    n_viol = M["n"] - M["label_counts"].get(M["labels"][0], 0)
    body = f"""<h1>{e(title)} — {date}</h1>
<p class="lede">{M['n']} messages written by Claude Fable 5.1 to be out of distribution:
{", ".join(f"{v} {k}" for k, v in sorted(diff.items()))} difficulty; axes {", ".join(f"{k} {v}" for k, v in sorted(axes.items()))};
{M['n'] // len(M['labels'])} per class.{sim} Spec: <code>regime_clf/ood/SPEC.md</code>.</p>
<div class="cards">
<div class="card"><div class="k">Macro-F1 (CLM fine-tuned, unit-masked)</div><div class="v">{best['macro_f1']:.3f}</div><div class="s">95% CI {best['macro_f1_95ci'][0]:.3f}–{best['macro_f1_95ci'][1]:.3f}</div></div>
<div class="card"><div class="k">Accuracy</div><div class="v">{pct(best['accuracy'])}</div><div class="s">{M['n']} rows</div></div>
<div class="card"><div class="k">Linear probe</div><div class="v">{M['linear_probe']['macro_f1']:.3f}</div><div class="s">macro-F1, same embeddings</div></div>
<div class="card"><div class="k">Missed violations</div><div class="v">{missed}</div><div class="s">of {n_viol} in-class violations → predicted {nm(M['labels'][0]).lower()}</div></div>
</div>
{notes}
<h2>Methods</h2>{methods_table(M)}
<h2>Per class</h2>{per_class_table(M)}
<h2>Confusion matrix — CLM fine-tuned, unit-masked</h2>{confusion(best, M['labels'])}
<h2>By difficulty</h2>{slice_table('by_difficulty')}
<h2>By OOD axis</h2>{slice_table('by_ood_axis')}
<h2>By speaker</h2>{slice_table('by_unit')}
{"<h2>By source label (out-of-class rows are gold " + nm(M['labels'][0]).lower() + ")</h2>" + slice_table('by_source_label') if 'by_source_label' in best and len(best['by_source_label']) > len(M['labels']) else ""}
<h2>Sample errors ({len(errs)} total; first 25)</h2>
<div class="scroll"><table><tr><th>Text</th><th>Gold</th><th>Predicted</th><th>Difficulty / axis</th></tr>{err_rows}</table></div>
<h2>Files</h2><ul><li><code>metrics.json</code>: all methods, slices, CIs</li>
<li><code>predictions.jsonl</code>: every row with gold, prediction, Fable's rationale</li>
<li><code>notes.html</code>: interpretation (optional)</li>
<li>Data: <code>{e(M['data'])}</code> (built by <code>regime_clf/ood/build_ood.py</code>)</li></ul>"""
    return {"title": f"OOD {short} ({len(M['labels'])}-class)", "body": body, "kind": f"OOD validation ({diff_label}), {label_set}",
            "n": M["n"], "macro_f1": best["macro_f1"], "ci": best["macro_f1_95ci"], "acc": best["accuracy"],
            "baseline": M["linear_probe"]["macro_f1"], "M": M}


def ui_e2e(d: str) -> dict:
    turns = [json.loads(l) for l in open(os.path.join(d, "turns.jsonl"))]
    fb = [json.loads(l) for l in open(os.path.join(d, "feedback.jsonl"))]
    t = turns[0]
    checks = [
        ("UI loads, presets populate, provider switch shows Base URL", True),
        ("Chat via OpenAI-compatible endpoint (app/mock_llm.py)", bool(t["assistant"]["text"])),
        (f"User message screened → {NAMES[t['user']['verdict']['label']]} ({t['user']['verdict']['confidence']:.3f})",
         t["user"]["verdict"]["label"] == "finra_4530"),
        (f"Reply screened → {NAMES[t['assistant']['verdict']['label']]} ({t['assistant']['verdict']['confidence']:.3f})",
         t["assistant"]["verdict"]["label"] == "finra_2210"),
        ("Turn written to app/data/turns.jsonl", True),
        (f"False-positive flag written to app/data/feedback.jsonl ({fb[0]['predicted_label']} → {fb[0]['correct_label']})",
         fb[0]["kind"] == "false_positive"),
    ]
    rows = "".join(f"<tr><td>{e(c)}</td><td>{'<span class=\"pill good\">pass</span>' if ok else '<span class=\"pill bad\">fail</span>'}</td></tr>"
                   for c, ok in checks)
    lat = t["latency_s"]
    body = f"""<h1>Chat UI end-to-end — 2026-10-05</h1>
<p class="lede">Manual browser run of <code>app/server.py</code> against the offline mock LLM. The flag submitted here was a
deliberate test of the feedback path (the reply really was a FINRA 2210 problem).</p>
<table><tr><th>Check</th><th>Result</th></tr>{rows}</table>
<h2>Latency of the screened turn</h2>
<table><tr><th>Stage</th><th class="num">seconds</th></tr>
<tr><td>Screen user message (includes one-off classifier load)</td><td class="num">{lat['screen_user']:.2f}</td></tr>
<tr><td>LLM call (mock)</td><td class="num">{lat['llm']:.3f}</td></tr>
<tr><td>Screen reply (warm)</td><td class="num">{lat['screen_reply']:.3f}</td></tr></table>
<h2>Files</h2><ul><li><code>turns.jsonl</code>, <code>feedback.jsonl</code> — copies of what the server wrote</li></ul>"""
    passed = sum(ok for _, ok in checks)
    return {"title": "Chat UI end-to-end", "body": body, "kind": "UI end-to-end",
            "n": len(checks), "summary": f"{passed}/{len(checks)} checks passed"}


def main():
    runs = []
    for d in sorted(glob.glob(os.path.join(HERE, "20*"))):
        if os.path.exists(os.path.join(d, "eval_test", "metrics.json")):
            r = in_distribution(d)
        elif os.path.exists(os.path.join(d, "feedback.jsonl")):
            r = ui_e2e(d)
        elif os.path.exists(os.path.join(d, "metrics.json")):
            r = ood(d)
        else:
            continue
        open(os.path.join(d, "report.html"), "w").write(page(r["title"], r["body"], depth=2))
        runs.append((os.path.basename(d), r))
        print("wrote", os.path.join(os.path.basename(d), "report.html"))
    rows = ""
    for name, r in runs:
        if "macro_f1" in r:
            ma = r["M"]["clm_finetuned_unit_masked"]["per_class"]["macro avg"]
            res = (f"<td class='num'>{pct(r['acc'])}</td><td class='num'>{ma['precision']:.3f}</td><td class='num'>{ma['recall']:.3f}</td>"
                   f"<td class='num'><b>{r['macro_f1']:.3f}</b> <span class='muted'>[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]</span></td>"
                   f"<td class='num'>{r['baseline']:.3f}</td>")
        else:
            res = f"<td colspan='5'>{e(r['summary'])}</td>"
        rows += (f"<tr><td><a href='{name}/report.html'>{name[:10]}</a></td><td>{e(r['title'])}</td>"
                 f"<td class='muted'>{e(r['kind'])}</td><td class='num'>{r['n']:,}</td>{res}</tr>")
    scored = [(name, r) for name, r in runs if "macro_f1" in r]
    headline = "; ".join(f"{r['kind'].lower()}: <b>{r['macro_f1']:.3f}</b> macro-F1 (n={r['n']:,})" for _, r in scored)
    headline = f"Fine-tuned CLM-8B regime classifier. {headline}. Plan for the OOD numbers, not the in-distribution one."
    body = f"""<h1>Test runs</h1>
<p class="lede">One dated folder per run. Each has a <code>report.html</code> and the raw JSON/JSONL it was built from.
Regenerate with <code>.venv/bin/python Tests/make_reports.py</code>.</p>
<div class="scroll"><table><tr><th>Date</th><th>Run</th><th>Type</th><th class="num">n</th>
<th class="num">Accuracy</th><th class="num">Macro precision</th><th class="num">Macro recall</th><th class="num">Macro-F1 [95% CI]</th><th class="num">Probe macro-F1</th></tr>{rows}</table></div>
<p class="muted">Precision, recall and F1 are macro-averaged over classes (each class counts equally). Per-class figures are in each run's report and in <a href="METRICS.md">METRICS.md</a>.</p>
<div class="note"><b>Headline.</b> {headline}</div>"""
    open(os.path.join(HERE, "index.html"), "w").write(page("Test runs", body, depth=1))
    print("wrote index.html")
    write_metrics_md(runs)


def write_metrics_md(runs) -> None:
    """Tests/METRICS.md: accuracy / precision / recall / F1 for every scored run, overall and per class."""
    lines = ["# Metrics summary", "",
             "Generated by `Tests/make_reports.py` from each run's `metrics.json`. Model = fine-tuned CLM-8B heads "
             "(speaker-masked); probe = logistic regression on the frozen Qwen3-8B embeddings. Precision/recall/F1 are "
             "macro-averaged unless a class is named. 95% CIs are percentile bootstraps (1,000 resamples).", "",
             "## Overall", "",
             "| Run | Label set | n | Accuracy | Macro precision | Macro recall | Macro-F1 [95% CI] | Probe acc | Probe macro-F1 |",
             "|---|---|---:|---:|---:|---:|---|---:|---:|"]
    scored = [(name, r) for name, r in runs if "M" in r]
    for name, r in scored:
        M = r["M"]; b = M["clm_finetuned_unit_masked"]; p = M["linear_probe"]
        ma, pa = b["per_class"]["macro avg"], p["per_class"]["macro avg"]
        lines.append(f"| [{name}]({name}/report.html) | {M.get('label_set', '7way')} | {M['n']:,} | {b['accuracy']:.3f} | "
                     f"{ma['precision']:.3f} | {ma['recall']:.3f} | **{b['macro_f1']:.3f}** [{b['macro_f1_95ci'][0]:.3f}, {b['macro_f1_95ci'][1]:.3f}] | "
                     f"{p['accuracy']:.3f} | {p['macro_f1']:.3f} |")
    lines += ["", "## Per class (fine-tuned CLM heads)", ""]
    for name, r in scored:
        M = r["M"]; b = M["clm_finetuned_unit_masked"]
        lines += [f"### {name}", "", "| Class | Precision | Recall | F1 | n |", "|---|---:|---:|---:|---:|"]
        for l in M["labels"]:
            c = b["per_class"][l]
            lines.append(f"| {nm(l)} | {c['precision']:.3f} | {c['recall']:.3f} | {c['f1-score']:.3f} | {int(c['support'])} |")
        lines.append("")
    open(os.path.join(HERE, "METRICS.md"), "w").write("\n".join(lines))
    print("wrote METRICS.md")


if __name__ == "__main__":
    main()
