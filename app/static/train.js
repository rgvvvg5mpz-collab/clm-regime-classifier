// Train / Post-train tabs. Talks to /api/train/*; see app/training.py.
(() => {
  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
  const pct = (x) => (100 * x).toFixed(1) + "%";
  const NEG_HINTS = ["compliant", "no_flag", "none", "no_issue", "ok", "negative", "clean", "not_flagged", "nothing"];
  const KNOWN = {   // prefill for labels that match the shipped taxonomy
    finra_2210: ["FINRA Rule 2210", "assistant"], reg_bi: ["Regulation Best Interest", "assistant"],
    finra_4530: ["FINRA Rule 4530 customer complaint", "client"], sec_17a3_17a4: ["SEC Rules 17a-3 / 17a-4 books and records", "client"],
    reg_sp: ["Regulation S-P privacy", "client"], reg_sid: ["Regulation S-ID identity theft red flags", "client"],
  };
  const T = { upload: null, validated: false, job: null };
  const P = { upload: null, job: null, models: null };

  async function api(path, body) {
    const r = await fetch(path, body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
    return data;
  }
  async function upload(file) {
    const fd = new FormData(); fd.append("file", file);
    const r = await fetch("/api/train/upload", { method: "POST", body: fd });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
    return data;
  }
  function issues(box, v) {
    box.replaceChildren();
    v.errors.forEach((m) => box.append(el("div", "issue err", "✕ " + m)));
    v.warnings.forEach((m) => box.append(el("div", "issue warn", "! " + m)));
    if (v.ok) box.append(el("div", "issue ok", `✓ ${v.n_clean} rows ready; split ${v.split_source}: ` +
      ["train", "val", "test"].map((s) => `${s} ${Object.values(v.split_counts[s]).reduce((a, b) => a + b, 0)}`).join(", ")));
  }

  // ---------------------------------------------------------------- Train tab
  async function loadExamples() {
    try {
      const ex = await api("/api/train/examples");
      ex.filter((e) => e.kind === "train").forEach((e) => $("trainExample").add(new Option(`${e.name} (${e.rows.toLocaleString()} rows${e.has_spec ? ", classes prefilled" : ""})`, e.name)));
      ex.filter((e) => e.kind === "expert").forEach((e) => $("postExample").add(new Option(`${e.name} (${e.rows} rows)`, e.name)));
    } catch (e) { /* examples are optional */ }
  }
  async function doUpload() {
    const f = $("trainFile").files[0];
    const ex = $("trainExample").value;
    if (!f && !ex) { alert("Choose an example dataset or a CSV / JSONL file first."); return; }
    $("trainSummary").textContent = "Loading…";
    try {
      T.upload = f ? await upload(f) : await api("/api/train/use_example", { name: ex }); T.validated = false;
      const u = T.upload;
      $("trainSummary").replaceChildren(
        el("div", "", `${u.n_rows.toLocaleString()} rows, ${Object.keys(u.labels).length} labels` +
          (u.has_speaker ? ", speaker column found" : ", no speaker column") + (u.has_split ? ", split column found" : "")),
        el("div", "muted", Object.entries(u.labels).map(([l, n]) => `${l}: ${n}`).join(" · ")));
      buildSpec(u);
      if (u.spec) applySpec(u.spec);
      $("t-step2").hidden = false; $("t-step3").hidden = false; $("t-step4").hidden = true; $("t-step5").hidden = true;
      $("t-step1").classList.add("done"); ["t-step2", "t-step3", "t-step4", "t-step5"].forEach((i) => $(i).classList.remove("done"));
      $("trainName").value = (f ? f.name : ex).replace(/\.[^.]+$/, "").replace(/_sample$/, "").replace(/[^A-Za-z0-9_-]+/g, "_").slice(0, 30);
    } catch (e) { $("trainSummary").textContent = "Upload failed: " + e.message; }
  }

  function buildSpec(u) {
    const t = $("classSpec"); t.replaceChildren();
    const h = el("tr"); ["Label", "Rows", "Display name", "Description (what the model scores against)", "Speaker", "Negative"].forEach((x) => h.append(el("th", "", x))); t.append(h);
    Object.entries(u.labels).forEach(([label, n]) => {
      const r = el("tr"); r.dataset.label = label;
      r.append(el("td", "", label), el("td", "num", String(n)));
      const name = el("input"); name.name = "name"; name.value = KNOWN[label]?.[0] || label.replace(/_/g, " ");
      const desc = el("textarea"); desc.name = "description"; desc.rows = 2;
      desc.placeholder = NEG_HINTS.includes(label.toLowerCase()) ? "e.g. None of the other classes applies: an ordinary, unremarkable text" : "One plain sentence defining this class, e.g. 'The text expresses …'";
      const sp = el("select"); sp.name = "speaker"; [["either", "either"], ["client", "user turn"], ["assistant", "assistant turn"]].forEach(([v, t]) => sp.add(new Option(t, v)));
      sp.dataset.tip = "Which speaker's turns can receive this class. Leave 'either' for data that is not conversational.";
      sp.value = KNOWN[label]?.[1] || "either";
      const neg = el("input"); neg.type = "radio"; neg.name = "negative"; neg.value = label;
      neg.dataset.tip = "Mark the one class that means 'none of the above'. Required: the model always picks a class, so it needs this one to abstain.";
      if (label === u.suggested_negative) { neg.checked = true; desc.value = "None of the other classes applies."; }
      [name, desc, sp, neg].forEach((x) => { const c = el("td"); c.append(x); r.append(c); });
      t.append(r);
    });
  }
  function applySpec(spec) {   // prefilled classes from an example's .spec.json
    $("classSpec").querySelectorAll("tr[data-label]").forEach((r) => {
      const c = spec[r.dataset.label]; if (!c) return;
      r.querySelector("[name=name]").value = c.name || r.dataset.label;
      r.querySelector("[name=description]").value = c.description || "";
      r.querySelector("[name=speaker]").value = c.speaker || "either";
      r.querySelector("[name=negative]").checked = !!c.negative;
    });
  }
  function readSpec() {
    const spec = {};
    $("classSpec").querySelectorAll("tr[data-label]").forEach((r) => {
      spec[r.dataset.label] = { name: r.querySelector("[name=name]").value, description: r.querySelector("[name=description]").value,
        speaker: r.querySelector("[name=speaker]").value, negative: r.querySelector("[name=negative]").checked };
    });
    return spec;
  }
  async function doValidate() {
    if (!T.upload) return;
    $("trainValidation").textContent = "Validating…";
    try {
      const v = await api("/api/train/validate", { upload_id: T.upload.upload_id, spec: readSpec() });
      issues($("trainValidation"), v);
      T.validated = v.ok; $("t-step4").hidden = !v.ok; $("t-step2").classList.toggle("done", v.ok); $("t-step3").classList.toggle("done", v.ok);
    } catch (e) { $("trainValidation").textContent = "Validation failed: " + e.message; }
  }
  async function doTrain() {
    if (!T.validated) return;
    try {
      T.job = await api("/api/train/start", { name: $("trainName").value, epochs: +$("trainEpochs").value, lr: +$("trainLr").value,
        init: $("trainInit").value, upload_id: T.upload.upload_id });
      $("trainStart").disabled = true;
      poll(T.job.id, $("trainJob"), (j) => { $("trainStart").disabled = false; if (j.status === "done") { $("t-step4").classList.add("done"); $("t-step5").hidden = false; renderResults($("trainResults"), j.result); $("t-step5").scrollIntoView({ block: "start" }); if (window.toast) window.toast("Training finished: results are below"); } });
    } catch (e) { $("trainJob").textContent = "Could not start: " + e.message; }
  }

  // ---------------------------------------------------------------- shared: job polling + results
  function poll(id, box, done) {
    const tick = async () => {
      let j; try { j = await api(`/api/train/jobs/${id}`); } catch (e) { box.textContent = "Lost the job: " + e.message; return; }
      renderJob(box, j);
      if (j.status === "done" || j.status === "failed") { done(j); return; }
      setTimeout(tick, 1500);
    };
    tick();
  }
  function renderJob(box, j) {
    box.replaceChildren();
    const p = j.progress || {};
    let line = `${j.status} · ${j.stage}`;
    let frac = null;
    if (p.stage === "embedding") { line += ` ${p.what}: ${p.done}/${p.total}`; frac = p.total ? p.done / p.total : 0; }
    if (p.stage === "training") { line += ` epoch ${p.epoch}/${p.epochs} · loss ${p.loss} · val macro-F1 ${p.val_macro_f1}`; frac = p.epoch / p.epochs; }
    box.append(el("div", j.status === "failed" ? "issue err" : "", line));
    if (frac !== null) { const pr = el("div", "prog"); const s = el("span"); s.style.width = (100 * frac).toFixed(0) + "%"; pr.append(s); box.append(pr); }
    if (j.error) box.append(el("div", "issue err", j.error));
    const pre = el("pre", "", (j.log || []).slice(-12).join("\n")); box.append(pre);
  }
  const TIPS = {
    "CLM heads, speaker-masked (the model)": "The trained model as it runs in the chat: classes a speaker cannot receive are masked out before the softmax (no effect when every class applies to either speaker).",
    "CLM heads, unmasked": "The same heads without the speaker mask. If this is lower, the mask is doing useful work.",
    "Linear probe baseline": "Logistic regression on the same frozen Qwen3-8B embeddings. A sanity check: if it beats the heads, the heads are undertrained.",
    "Zero-shot reference heads": "The published CLM heads with your class descriptions and no training. Expected to be near chance.",
  };
  function metricRow(label, m) {
    const r = el("tr"); const ma = m.per_class["macro avg"];
    if (TIPS[label]) r.dataset.tip = TIPS[label];
    r.append(el("td", "", label), el("td", "num", pct(m.accuracy)), el("td", "num", ma.precision.toFixed(3)), el("td", "num", ma.recall.toFixed(3)), el("td", "num", m.macro_f1.toFixed(3)));
    return r;
  }
  function confusion(m, labels, names) {
    const t = el("table", "cm"); const h = el("tr"); h.append(el("th", "", "gold ↓ / predicted →"));
    labels.forEach((l) => h.append(el("th", "", names[l] || l))); t.append(h);
    m.confusion.forEach((row, i) => {
      const r = el("tr"); r.append(el("th", "", names[labels[i]] || labels[i]));
      const tot = row.reduce((a, b) => a + b, 0) || 1;
      row.forEach((v, j) => { const c = el("td", "cell", String(v)); c.style.background = `rgba(var(--heat, 150, 21, 29), ${(v / tot).toFixed(2)})`; if (v / tot > 0.55) c.style.color = "#fff";
        c.dataset.tip = `${v} ${names[labels[i]] || labels[i]} turn${v === 1 ? "" : "s"} predicted as ${names[labels[j]] || labels[j]} (${(100 * v / tot).toFixed(0)}% of that class)`; r.append(c); });
      t.append(r);
    });
    return t;
  }
  function renderResults(box, res) {
    box.replaceChildren();
    const m = res.metrics, labels = m.labels, names = m.names || {};
    const t = el("table"); const h = el("tr"); ["Method", "Accuracy", "Macro precision", "Macro recall", "Macro-F1"].forEach((x) => h.append(el("th", "", x))); t.append(h);
    t.append(metricRow("CLM heads, speaker-masked (the model)", m.clm_finetuned_unit_masked));
    t.append(metricRow("CLM heads, unmasked", m.clm_finetuned));
    if (m.linear_probe) t.append(metricRow("Linear probe baseline", m.linear_probe));
    if (m.clm_zero_shot) t.append(metricRow("Zero-shot reference heads", m.clm_zero_shot));
    box.append(el("p", "", `Checkpoint ${res.checkpoint} · test rows ${m.n}` + (m.clm_finetuned_unit_masked.macro_f1_95ci ? ` · macro-F1 95% CI [${m.clm_finetuned_unit_masked.macro_f1_95ci.map((x) => x.toFixed(3)).join(", ")}]` : "")));
    box.append(t, el("h3", "", "Confusion matrix"), confusion(m.clm_finetuned_unit_masked, labels, names));
    const pc = el("table"); const ph = el("tr"); ["Class", "Precision", "Recall", "F1", "n"].forEach((x) => ph.append(el("th", "", x))); pc.append(ph);
    labels.forEach((l) => { const c = m.clm_finetuned_unit_masked.per_class[l]; const r = el("tr"); r.append(el("td", "", names[l] || l), el("td", "num", c.precision.toFixed(3)), el("td", "num", c.recall.toFixed(3)), el("td", "num", c["f1-score"].toFixed(3)), el("td", "num", String(c.support))); pc.append(r); });
    box.append(el("h3", "", "Per class"), pc);
    if (res.gate) box.append(el("h3", "", "Gate against the active model"), gateTable(res.gate, labels, names));
    const row = el("div", "row");
    if (res.report) { const a = el("a", "", "Open full report"); a.href = "/" + res.report; a.target = "_blank"; row.append(a); }
    const act = el("button", "", res.gate && !res.gate.passed ? "Promote anyway (needs a reason)" : "Activate this model");
    act.dataset.tip = res.gate && !res.gate.passed ? "The gate found regressions; promoting requires a reason, which is logged with the promotion." : "Make this checkpoint the one that screens the chat. The encoder stays loaded, so it is instant.";
    act.onclick = async () => {
      let reason = null;
      if (res.gate && !res.gate.passed) { reason = prompt("The gate found regressions. Reason for promoting anyway:"); if (!reason) return; }
      try { const r = await api("/api/train/activate", { checkpoint: res.checkpoint, reason }); act.replaceWith(el("span", "thanks", `Active: ${r.active.split("/").pop()}`)); loadModels(); fetch("/api/config").then((x) => x.json()).then((c) => { state.config = c; state.labels = c.labels; }); if (window.refreshActiveModel) window.refreshActiveModel(); if (window.toast) window.toast("Model activated: it now screens the chat"); }
      catch (e) { alert("Could not activate: " + e.message); }
    };
    row.append(act); box.append(row);
  }
  function gateTable(g, labels, names) {
    const w = el("div");
    w.append(el("div", g.passed ? "issue ok" : "issue err", g.passed ? "✓ No regressions versus the active model" : "✕ Regressions: " + g.regressions.join("; ")));
    w.append(el("p", "muted small", "The 'test' row is the candidate's own held-out split; if the active model was trained on overlapping data (e.g. a shipped model vs an example sample of the same data) its score there is inflated. The OOD rows are a fair comparison for both."));
    const t = el("table"); const h = el("tr"); ["Set", "n", "Active macro-F1", "Candidate macro-F1", "Δ", "Per-class recall (active → candidate)"].forEach((x) => h.append(el("th", "", x))); t.append(h);
    Object.entries(g.sets).forEach(([s, c]) => {
      const d = c.candidate.macro_f1 - c.active.macro_f1; const r = el("tr");
      r.dataset.tip = { test: "The candidate's own held-out split.", ood_hard: "Regulatory example only: 315 hard / very hard out-of-distribution messages, never used for training.", ood_low_medium: "Regulatory example only: 210 low / medium out-of-distribution messages, never used for training." }[s] || s;
      r.append(el("td", "", s), el("td", "num", String(c.n)), el("td", "num", c.active.macro_f1.toFixed(3)), el("td", "num", c.candidate.macro_f1.toFixed(3)),
        el("td", "num delta " + (d >= -0.01 ? "up" : "down"), (d >= 0 ? "+" : "") + d.toFixed(3)),
        el("td", "small", labels.slice(1).map((l) => `${names[l] || l}: ${c.active.recall[l].toFixed(2)} → ${c.candidate.recall[l].toFixed(2)}`).join(" · ")));
      t.append(r);
    });
    w.append(t); return w;
  }

  // ---------------------------------------------------------------- Post-train tab
  async function loadModels() {
    try {
      P.models = await api("/api/train/models");
      const t = $("modelTable"); t.replaceChildren();
      const h = el("tr"); ["Model", "Version", "Parent", "Source", "Classes", "Test macro-F1", "Gate", ""].forEach((x) => h.append(el("th", "", x))); t.append(h);
      P.models.models.slice().reverse().forEach((m) => {
        const r = el("tr"); const active = m.checkpoint === P.models.active;
        r.append(el("td", "", (active ? "● " : "") + m.name), el("td", "num", "v" + (m.version || 1)),
          el("td", "small muted", (m.parent || "").split("/").pop()), el("td", "muted", m.source), el("td", "small", m.labels.join(", ")),
          el("td", "num", m.metrics?.test_macro_f1 != null ? m.metrics.test_macro_f1.toFixed(3) : "-"),
          el("td", "", m.gate ? (m.gate.passed ? "passed" : "failed") : "-"));
        const c = el("td"); if (!active) { const b = el("button", "linkbtn", "activate"); b.dataset.tip = "Make this version the one that screens the chat."; b.onclick = async () => { try { await api("/api/train/activate", { checkpoint: m.checkpoint }); loadModels(); loadActive(); if (window.refreshActiveModel) window.refreshActiveModel(); if (window.toast) window.toast(`Activated ${m.name}`); } catch (e) { alert(e.message); } }; c.append(b); }
        r.append(c); t.append(r);
      });
    } catch (e) { $("modelTable").textContent = "Could not load models: " + e.message; }
  }
  async function loadActive() {
    try {
      const m = await api("/api/train/models"); const a = m.models.find((x) => x.checkpoint === m.active);
      $("postActive").textContent = a ? `${a.name} v${a.version || 1} (${a.source}) · classes: ${a.labels.join(", ")}` : m.active;
      const fb = await api("/api/train/feedback");
      $("postFeedback").replaceChildren(
        el("div", "", `${fb.n} expert flags from the chat usable for this model` + (fb.skipped ? ` (${fb.skipped} skipped: label outside this model)` : "")),
        el("div", "muted", Object.entries(fb.by_label).map(([l, n]) => `${l}: ${n}`).join(" · ") || "none yet"));
      $("postName").value = (a ? a.name : "model") + "_post_" + new Date().toISOString().slice(5, 10).replace("-", "");
    } catch (e) { $("postActive").textContent = "Could not load: " + e.message; }
  }
  async function postUpload() {
    const f = $("postFile").files[0]; const ex = $("postExample").value;
    if (!f && !ex) { alert("Choose an example or a file first."); return; }
    $("postUploadInfo").textContent = "Loading…";
    try {
      const u = f ? await upload(f) : await api("/api/train/use_example", { name: ex });
      const m = await api("/api/train/models"); const a = m.models.find((x) => x.checkpoint === m.active);
      const unknown = Object.keys(u.labels).filter((l) => !a.labels.includes(l));
      if (unknown.length) { $("postUploadInfo").textContent = `Labels not in the active model: ${unknown.join(", ")}. Expert rows must use the model's own labels (${a.labels.join(", ")}).`; return; }
      const spec = {}; a.labels.forEach((l, i) => spec[l] = { name: a.names?.[l] || l, description: a.classes[l], speaker: "either", negative: i === 0 });
      Object.keys(u.labels).forEach((l) => { if (!spec[l]) spec[l] = { name: l, description: l, speaker: "either", negative: false }; });
      const v = await api("/api/train/validate", { upload_id: u.upload_id, spec, expert: true });
      issues($("postUploadInfo"), v);
      P.upload = v.ok ? u : null;
    } catch (e) { $("postUploadInfo").textContent = "Upload failed: " + e.message; }
  }
  async function postStart() {
    try {
      P.job = await api("/api/train/start", { post_train: true, name: $("postName").value, epochs: +$("postEpochs").value, lr: +$("postLr").value,
        init: $("postMode").value, use_feedback: true, extra_upload_id: P.upload?.upload_id || null, include_base_data: $("postBase").checked, oversample: +$("postOversample").value });
      $("postStart").disabled = true;
      poll(P.job.id, $("postJob"), (j) => { $("postStart").disabled = false; if (j.status === "done") { $("p-results").hidden = false; renderResults($("postResults"), j.result); loadModels(); $("p-results").scrollIntoView({ block: "start" }); if (window.toast) window.toast(j.result.gate && !j.result.gate.passed ? "Candidate trained: the gate found regressions" : "Candidate trained: gate passed"); } });
    } catch (e) { $("postJob").textContent = "Could not start: " + e.message; }
  }

  loadExamples();
  $("trainExample").onchange = () => { if ($("trainExample").value) { $("trainFile").value = ""; doUpload(); } };
  $("postExample").onchange = () => { if ($("postExample").value) { $("postFile").value = ""; postUpload(); } };
  $("trainUpload").onclick = doUpload; $("trainValidate").onclick = doValidate; $("trainStart").onclick = doTrain;
  $("postUpload").onclick = postUpload; $("postStart").onclick = postStart;
  document.querySelector('.tab[data-view="postView"]').addEventListener("click", () => { loadActive(); loadModels(); });
  window.trainTabs = { loadModels, loadActive };
})();
