// Data curation tab (step 0): prompt Claude Code to research and build a labelled dataset.
// Talks to /api/curate/*; see app/curation.py.
(() => {
  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
  const C = { job: null, last: null, polling: false };
  const ICON = { tool: "→", say: "“", done: "✓", error: "✕", denied: "⚠", info: "•", start: "▶" };
  const TEMPLATES = {
    generic: "Research how support teams triage customer emails, then build a labelled dataset of short customer emails " +
             "with the classes: billing, technical_issue, account_access, feedback, and a negative class other. " +
             "Write realistic, varied emails (different tones, lengths, typos) and include hard cases near class boundaries.",
    sentiment: "Build a product-review sentiment dataset with classes positive, negative, mixed and a negative class neutral " +
               "(factual, no opinion). Cover several product categories and include sarcasm and mixed reviews as hard cases.",
    regulatory: "Example use case: research FINRA Rule 2210 (communications with the public) and Regulation Best Interest, " +
                "then write assistant replies from a broker-dealer's chat assistant labelled finra_2210 (misleading, promissory or " +
                "unbalanced statements), reg_bi (personalised recommendations or steering to house products) and a negative class " +
                "no_flag (balanced, educational answers). speaker = assistant for every row. Focus on blunt, short recommendations " +
                "that a model might confuse with promissory language.",
  };

  async function api(path, body) {
    const r = await fetch(path, body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
    return data;
  }

  async function checkStatus() {
    const box = $("curStatus"); box.textContent = "Checking Claude Code…";
    try {
      const s = await api("/api/curate/status");
      box.replaceChildren();
      if (s.available && s.logged_in) {
        box.append(el("div", "issue ok", `✓ Claude Code ${s.version || ""} ready (${s.auth || "logged in"})`));
        $("curStart").disabled = false;
      } else {
        box.append(el("div", "issue warn", s.help || "Claude Code is not available."));
        $("curStart").disabled = true;
      }
    } catch (e) { box.textContent = "Could not check: " + e.message; }
  }

  function renderJob(j) {
    const feed = $("curFeed"); feed.replaceChildren();
    const head = el("div", "small");
    head.append(el("b", "", j.status === "running" ? "Working… " : j.status === "done" ? "Finished. " : j.status === "failed" ? "Failed. " : j.status + ". "),
      el("span", "muted", `job ${j.id} · model ${j.model}` + (j.turns != null ? ` · ${j.turns} turns` : "") +
        (j.cost_usd != null ? ` · $${Number(j.cost_usd).toFixed(2)}` : "") + ` · folder ${j.workdir}`));
    feed.append(head);
    const list = el("ol", "feed");
    j.events.slice(-60).forEach((ev) => {
      const li = el("li", "ev ev-" + ev.kind);   // prefixed: a bare "info" class would pick up the round i-icon style
      li.append(el("span", "ico", ICON[ev.kind] || "•"), el("span", "", ev.text));
      list.append(li);
    });
    feed.append(list);
    list.scrollTop = list.scrollHeight;
    if (j.error) feed.append(el("div", "issue err", j.error));
    $("curCancel").hidden = j.status !== "running";
    $("curStart").disabled = j.status === "running";
  }

  function renderResult(j) {
    const box = $("curResult"); box.replaceChildren();
    const s = j.summary; if (!s) return;
    $("c-step3").hidden = false;
    s.errors.forEach((m) => box.append(el("div", "issue err", "✕ " + m)));
    s.warnings.forEach((m) => box.append(el("div", "issue warn", "! " + m)));
    if (!s.n_rows) return;
    box.append(el("p", "", `${s.n_rows.toLocaleString()} rows · ` + Object.entries(s.by_label).map(([l, n]) => `${l}: ${n}`).join(" · ")));
    if (s.classes) {
      const t = el("table"); const h = el("tr"); ["Class", "Speaker", "Negative", "Description"].forEach((x) => h.append(el("th", "", x))); t.append(h);
      Object.entries(s.classes).forEach(([k, v]) => { const r = el("tr"); r.append(el("td", "", k), el("td", "", v.speaker || "either"), el("td", "", v.negative ? "yes" : ""), el("td", "small", v.description || "")); t.append(r); });
      box.append(el("h3", "", "Classes"), el("div", "scroll")); box.lastChild.append(t);
    }
    const pv = el("table"); const ph = el("tr"); ["Text", "Label", "Speaker", "Source"].forEach((x) => ph.append(el("th", "", x))); pv.append(ph);
    s.preview.forEach((r) => { const tr = el("tr"); tr.append(el("td", "", String(r.text).slice(0, 220)), el("td", "", r.label), el("td", "", r.speaker || ""), el("td", "small muted", String(r.source || "").slice(0, 60))); pv.append(tr); });
    box.append(el("h3", "", "Preview (first 12 rows)"), el("div", "scroll")); box.lastChild.append(pv);
    if (s.notes) { const d = el("details"); d.append(el("summary", "", "Claude Code's notes (sources, judgement calls)"), el("pre", "notes", s.notes)); box.append(d); }
    const row = el("div", "row wrap");
    const send = el("button", "", "Send to Train →");
    send.dataset.tip = "Stages this dataset on the Train tab with the class table prefilled from classes.json; you then validate and train as usual.";
    send.disabled = s.errors.length > 0;
    send.onclick = async () => {
      try {
        const u = await api(`/api/curate/jobs/${j.id}/to_train`, {});
        window.trainTabs.showUpload(u, `curated_${j.id.slice(-5)}`);
        document.querySelector('.tab[data-view="trainView"]').click();
        if (window.toast) window.toast("Curated dataset staged on the Train tab: validate, then train");
      } catch (e) { alert("Could not send: " + e.message); }
    };
    const dl = el("a", "", "Download dataset.jsonl"); dl.href = `/api/curate/jobs/${j.id}/download`;
    dl.dataset.tip = "The raw file Claude Code wrote. It is also kept in the job folder with classes.json, NOTES.md and the prompt log.";
    row.append(send, dl);
    box.append(row);
    $("curFollowRow").hidden = false;
  }

  async function poll(id) {
    if (C.polling) return; C.polling = true;
    const tick = async () => {
      let j; try { j = await api(`/api/curate/jobs/${id}`); } catch (e) { C.polling = false; return; }
      C.last = j; renderJob(j);
      if (j.status === "running") { setTimeout(tick, 1200); return; }
      C.polling = false; renderResult(j);
      if (window.toast) window.toast(j.status === "done" ? "Curation finished: review the dataset below" : "Curation " + j.status);
    };
    tick();
  }

  async function start(followUp) {
    const prompt = (followUp ? $("curFollow").value : $("curPrompt").value).trim();
    if (!prompt) { alert("Write a prompt first."); return; }
    try {
      const j = await api("/api/curate/start", { prompt, model: $("curModel").value || null, max_turns: +$("curTurns").value,
        budget_usd: +$("curBudget").value, rows_per_class: +$("curRows").value, resume_of: followUp && C.last ? C.last.id : null });
      C.job = j; $("c-step2").hidden = false; $("curResult").replaceChildren(); if (followUp) $("curFollow").value = "";
      poll(j.id);
    } catch (e) { alert("Could not start: " + e.message); }
  }

  $("curTemplate").onchange = (e) => { if (TEMPLATES[e.target.value]) $("curPrompt").value = TEMPLATES[e.target.value]; };
  $("curStart").onclick = () => start(false);
  $("curFollowBtn").onclick = () => start(true);
  $("curCancel").onclick = async () => { if (C.last) { await api(`/api/curate/jobs/${C.last.id}/cancel`, {}); } };
  $("curRecheck").onclick = checkStatus;
  document.querySelectorAll('[data-view="curateView"]').forEach((b) => b.addEventListener("click", () => {
    checkStatus();
    if (!C.last) api("/api/curate/jobs").then((js) => { if (js.length) { C.last = js[0]; $("c-step2").hidden = false; renderJob(js[0]); if (js[0].status === "running") poll(js[0].id); else renderResult(js[0]); } });
  }));
})();
