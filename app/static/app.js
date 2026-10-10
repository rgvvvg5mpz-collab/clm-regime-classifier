// Regime screening chat - plain JS, no build step.
// Two ways to drive a conversation: a live LLM (Anthropic / OpenAI-compatible), or a scripted
// transcript (bundled demo file or an uploaded .json, format in app/transcripts/FORMAT.md)
// replayed turn by turn through the same screening pipeline.
const $ = (id) => document.getElementById(id);
const state = { config: null, labels: {}, history: [], sessionId: newId(), busy: false,
                transcript: null, convIdx: 0, turnIdx: 0 };
// Display names for "expected" labels that the loaded model may not cover.
// One-line triggers for the diagram / table on the "The classes" tab.
const CLASS_INFO = {
  finra_2210: ["FINRA Rule 2210", "Misleading, promissory or unbalanced statements; performance projections; missing disclosures"],
  reg_bi: ["Regulation Best Interest", "Personalised recommendation of a security, strategy, account or rollover; steering to house products"],
  finra_4530: ["FINRA Rule 4530", "Written grievance: sales practice, unauthorised trades, account errors, theft or forgery"],
  sec_17a3_17a4: ["SEC Rules 17a-3 / 17a-4", "Off-channel contact requests, \u201cdon\u2019t record this\u201d, account updates that must be captured"],
  reg_sp: ["Regulation S-P", "Shares SSN, date of birth or account numbers; privacy or opt-out requests; data incidents"],
  reg_sid: ["Regulation S-ID", "Identity-theft red flags: account-takeover signals, suspicious ID details, bypassing verification"],
};
const SPEAKER_CLASSES = { client_message: ["finra_4530", "sec_17a3_17a4", "reg_sp", "reg_sid"],
                          assistant_response: ["finra_2210", "reg_bi"] };
const ALL_NAMES = { none: "nothing", compliant: "nothing", no_flag: "nothing",
  finra_2210: "FINRA 2210", reg_bi: "Reg BI", finra_4530: "FINRA 4530",
  sec_17a3_17a4: "SEC 17a-3/4", reg_sp: "Reg S-P", reg_sid: "Reg S-ID" };

function newId() { return Math.random().toString(36).slice(2, 10); }
// Tooltips: one floating box for every element with data-tip, positioned in the viewport so
// sidebars, tables and scroll containers never clip it. Works on hover and keyboard focus.
(() => {
  const box = document.createElement("div"); box.id = "tipbox"; box.setAttribute("role", "tooltip");
  document.addEventListener("DOMContentLoaded", () => document.body.append(box));
  let current = null;
  function show(target) {
    const text = target.getAttribute("data-tip"); if (!text) return;
    current = target; box.textContent = text; box.classList.add("show");
    const r = target.getBoundingClientRect(), pad = 8;
    box.style.left = "0px"; box.style.top = "0px";
    const w = box.offsetWidth, h = box.offsetHeight;
    let left = Math.min(Math.max(pad, r.left), window.innerWidth - w - pad);
    let top = r.bottom + 7;
    if (top + h > window.innerHeight - pad) top = r.top - h - 7;
    box.style.left = left + "px"; box.style.top = Math.max(pad, top) + "px";
  }
  function hide() { current = null; box.classList.remove("show"); }
  document.addEventListener("mouseover", (e) => { const t = e.target.closest("[data-tip]"); if (t && t !== current) show(t); else if (!t) hide(); });
  document.addEventListener("mouseout", (e) => { const t = e.target.closest("[data-tip]"); if (t && !t.contains(e.relatedTarget)) hide(); });
  document.addEventListener("focusin", (e) => { const t = e.target.closest("[data-tip]"); if (t) show(t); });
  document.addEventListener("focusout", hide);
  document.addEventListener("scroll", hide, true);
  document.addEventListener("click", hide, true);
})();
let toastTimer = null;
function toast(msg) {   // small confirmation in the corner
  const t = $("toast"); if (!t) return;
  t.textContent = msg; t.classList.add("show");
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove("show"), 2600);
}
async function refreshActiveModel() {   // header badge + sidebar selector: which checkpoint screens the chat
  try {
    const m = await api("/api/train/models");
    const a = m.models.find((x) => x.checkpoint === m.active);
    const b = $("activeModel");
    if (b) b.replaceChildren("Model: ", el("b", "", a ? `${a.name} v${a.version || 1}` : m.active.split("/").pop()));
    const sel = $("modelSelect");
    if (sel) {
      sel.replaceChildren();
      m.models.slice().reverse().forEach((x) => sel.add(new Option(`${x.name} v${x.version || 1} · ${x.labels.length} classes${x.metrics?.test_macro_f1 != null ? ` · F1 ${x.metrics.test_macro_f1.toFixed(2)}` : ""}`, x.checkpoint)));
      sel.value = m.active;
    }
  } catch { /* cosmetic */ }
}
async function switchModel(checkpoint) {
  try {
    await api("/api/train/activate", { checkpoint });
    state.config = await api("/api/config"); state.labels = state.config.labels;
    updateScopeNote(); refreshActiveModel();
    if (!$("aboutView").hidden) renderAbout();
    if (window.trainTabs) { window.trainTabs.loadModels(); window.trainTabs.loadActive(); }
    toast(`Screening with ${checkpoint.split("/").pop()}`);
  } catch (e) { alert("Could not switch model: " + e.message); }
}
function updateScopeNote() {
  const names = (u) => (state.config.unit_classes[u] || []).slice(1).map((k) => state.labels[k] || k).join(", ") || "nothing";
  const note = $("scopeNote");
  if (note) note.textContent = `User turns can be flagged as: ${names("client_message")}. Assistant turns as: ${names("assistant_response")}.`;
}
window.refreshActiveModel = refreshActiveModel;
function ss(key, val) {   // sessionStorage, tolerant of blocked storage
  try { if (val === undefined) return sessionStorage.getItem(key); sessionStorage.setItem(key, val); }
  catch { return null; }
}

async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
  return data;
}

// ---------------------------------------------------------------- provider panel
function applyPreset(p) {
  $("provider").value = p.provider;
  $("baseUrl").value = p.base_url || "";
  $("model").value = p.model || "";
  $("apiKey").dataset.env = p.api_key_env || "";
  syncProvider();
  if (p.provider === "scripted" && p.transcript) loadTranscriptUrl(p.transcript);
}
async function loadTranscriptList() {   // transcript files in the project
  try {
    const files = (await api("/api/files")).filter((f) => f.kind === "transcript");
    const sel = $("transcriptFile"); sel.replaceChildren();
    files.forEach((f) => sel.add(new Option(`${f.name} (${f.count} conversations)`, f.path)));
    const def = files.find((f) => f.path.endsWith("demo_conversations.json"));
    if (def) sel.value = def.path;
  } catch { /* optional */ }
}
function syncProvider() {
  const prov = $("provider").value;
  const scripted = prov === "scripted";
  $("baseUrlRow").style.display = prov === "openai_compatible" ? "" : "none";
  $("modelRow").style.display = scripted ? "none" : "";
  $("apiKeyRow").style.display = scripted ? "none" : "";
  $("systemRow").style.display = scripted ? "none" : "";
  $("transcriptPanel").style.display = scripted ? "" : "none";
  $("composer").style.display = scripted ? "none" : "";
  $("playbar").style.display = scripted ? "" : "none";
  $("apiKey").placeholder = $("apiKey").dataset.env
    ? `uses ${$("apiKey").dataset.env} if empty` : "optional";
}
function providerPayload() {
  return {
    provider: $("provider").value, model: $("model").value.trim() || "transcript",
    base_url: $("provider").value === "openai_compatible" ? $("baseUrl").value.trim() || null : null,
    api_key: $("apiKey").value || null, api_key_env: $("apiKey").dataset.env || null,
  };
}

async function loadConfig() {
  state.config = await api("/api/config");
  state.labels = state.config.labels;
  const sel = $("preset");
  state.config.presets.forEach((p, i) => sel.add(new Option(p.name, i)));
  sel.onchange = () => { resetChat(); applyPreset(state.config.presets[sel.value]); };
  $("system").value = state.config.system_prompt || "";
  updateScopeNote();
  $("apiKey").value = ss("apiKey") || "";
  applyPreset(state.config.presets[0]);
  refreshActiveModel();
  pollHealth();
}
async function pollHealth() {   // the server loads the encoder in the background at startup
  let h; try { h = await api("/api/health"); } catch { setTimeout(pollHealth, 3000); return; }
  if (h.state === "ready" || h.classifier_loaded) { setClf("on", h.seconds ? `Model ready (loaded in ${h.seconds}s)` : "Model ready"); return; }
  if (h.state === "loading") { setClf("busy", "Loading the encoder\u2026 (first run downloads ~16 GB)"); setTimeout(pollHealth, 2500); return; }
  if (h.state === "missing_model") { setClf("off", "Model file missing: run scripts/setup.sh"); return; }
  if (h.state === "error") { setClf("off", "Load failed: " + (h.detail || "see server log")); return; }
  setClf("off"); setTimeout(pollHealth, 2500);
}

function setClf(s, text) {
  $("clfDot").className = "dot" + (s === "on" ? " on" : s === "busy" ? " busy" : "");
  $("clfStatus").textContent = text || { on: "Model ready", busy: "Loading the encoder\u2026", off: "Model not loaded" }[s];
  $("warmup").style.display = s === "off" ? "" : "none";
}

async function warmup() {
  setClf("busy");
  try { const r = await api("/api/warmup", {}); setClf("on", `Classifier ready (${r.seconds}s)`); }
  catch (e) { setClf("off", "Load failed: " + e.message); }
}

// ---------------------------------------------------------------- transcripts
async function loadTranscriptUrl(url) {   // project-relative paths go through the files API
  if (!url.startsWith("/")) url = "/api/files/transcript?path=" + encodeURIComponent(url);
  try {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    setTranscript(await r.json(), url.split("/").pop());
  } catch (e) { $("transcriptInfo").textContent = "Could not load transcript: " + e.message; }
}

function validateTranscript(t) {
  if (!t || !Array.isArray(t.conversations) || !t.conversations.length) throw new Error("no 'conversations' array");
  t.conversations.forEach((c, i) => {
    if (!Array.isArray(c.turns) || !c.turns.length) throw new Error(`conversation ${i + 1} has no turns`);
    c.turns.forEach((u, j) => {
      const want = j % 2 === 0 ? "user" : "assistant";
      if (u.role !== want) throw new Error(`conversation ${i + 1}, turn ${j + 1}: expected role '${want}', got '${u.role}'`);
      if (typeof u.content !== "string") throw new Error(`conversation ${i + 1}, turn ${j + 1}: missing content`);
    });
  });
}

function flagCount(c) {   // turns the transcript's author expects to be flagged
  return c.turns.filter((u) => u.expected && !["none", "compliant", "no_flag"].includes(u.expected)).length;
}

function setTranscript(t, name) {
  validateTranscript(t);
  state.transcript = t;
  const sel = $("conv");
  sel.replaceChildren();
  t.conversations.forEach((c, i) => {
    const n = flagCount(c);
    sel.add(new Option(`${i + 1}. ${c.title || c.id || "conversation"} (${n ? `${n} expected flag${n === 1 ? "" : "s"}` : "clean"})`, i));
  });
  const total = t.conversations.reduce((a, c) => a + flagCount(c), 0);
  const info = $("transcriptInfo");
  info.textContent = `${t.conversations.length} conversations \u00b7 ${total} turns expected to be flagged`;
  info.dataset.tip = (t.title || name) + (t.description ? ": " + t.description : "");
  // Start on the conversation with the most expected flags so a demo doesn't open on a clean one.
  const busiest = t.conversations.reduce((b, c, i) => flagCount(c) > flagCount(t.conversations[b]) ? i : b, 0);
  sel.value = busiest;
  selectConversation(busiest);
}

function selectConversation(i) {
  state.convIdx = Number(i); state.turnIdx = 0;
  resetChat();
  const c = state.transcript.conversations[state.convIdx];
  $("scenario").textContent = c.scenario || "";
  updatePlaybar();
}

function updatePlaybar() {
  const c = state.transcript?.conversations[state.convIdx];
  const left = c ? Math.ceil((c.turns.length - state.turnIdx) / 2) : 0;
  $("playNext").disabled = !c || left === 0 || state.busy;
  $("playAll").disabled = !c || left === 0 || state.busy;
  $("playStatus").textContent = c ? (left ? `${left} exchange${left === 1 ? "" : "s"} left` : "End of conversation") : "";
}

async function playNext() {
  const c = state.transcript?.conversations[state.convIdx];
  if (!c || state.turnIdx >= c.turns.length || state.busy) return;
  const u = c.turns[state.turnIdx], a = c.turns[state.turnIdx + 1];
  state.turnIdx += a ? 2 : 1;
  await runTurn(u.content, { scripted_reply: a ? a.content : null,
    expected: { user: u.expected || null, assistant: a ? (a.expected || null) : null },
    notes: { user: u.note || null, assistant: a ? (a.note || null) : null } });
  updatePlaybar();
}

async function playAll() {
  const c = state.transcript?.conversations[state.convIdx];
  while (c && state.turnIdx < c.turns.length) await playNext();
}

function onUpload(ev) {
  const f = ev.target.files[0];
  if (!f) return;
  const rd = new FileReader();
  rd.onload = () => {
    try { setTranscript(JSON.parse(rd.result), f.name); }
    catch (e) { $("transcriptInfo").textContent = `Could not read ${f.name}: ${e.message}`; }
  };
  rd.readAsText(f);
  ev.target.value = "";
}

// ---------------------------------------------------------------- "The classes" tab
function svgEl(tag, attrs = {}, text) {
  const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
}
function wrapText(e, text, maxChars) {   // crude word-wrap into <tspan>s
  const words = text.split(" "); const lines = [""];
  for (const w of words) {
    if ((lines[lines.length - 1] + " " + w).trim().length > maxChars) lines.push(w);
    else lines[lines.length - 1] = (lines[lines.length - 1] + " " + w).trim();
  }
  const x = e.getAttribute("x");
  lines.forEach((l, i) => e.append(svgEl("tspan", { x, dy: i ? "1.25em" : 0 }, l)));
  return lines.length;
}

function renderAbout() {
  // Built from the selected checkpoint: its class list, descriptions and speaker masks.
  const labels = state.labels, negative = Object.keys(labels)[0];
  const covered = (k) => k in labels;
  const desc = state.config.descriptions || {};
  const info = (k) => [labels[k] || CLASS_INFO[k]?.[0] || k, desc[k] || CLASS_INFO[k]?.[1] || ""];
  const uc = state.config.unit_classes || {};
  const lanes = { client_message: (uc.client_message || []).slice(1), assistant_response: (uc.assistant_response || []).slice(1) };
  // For the regulatory example only: also draw the regimes this checkpoint leaves out, greyed.
  const isExample = Object.keys(labels).slice(1).every((k) => k in CLASS_INFO);
  const extra = isExample ? Object.keys(CLASS_INFO).filter((k) => !covered(k)) : [];
  extra.forEach((k) => (SPEAKER_CLASSES.client_message.includes(k) ? lanes.client_message : lanes.assistant_response).push(k));
  const nCov = Object.keys(labels).length - 1;
  const modelName = (state.config.active || "the selected model").split("/").pop().replace(/\.pt$/, "");
  $("aboutTitle").textContent = `What ${modelName} looks for`;
  $("aboutLede").textContent = `This model assigns each turn one of ${nCov} classes, or “${labels[negative]}” when none applies. ` +
    `Which classes a turn can receive depends on who is speaking.` +
    (extra.length ? " Greyed classes are outside this model’s scope: that content is left unflagged." : "");

  const W = 900, lane = 400, boxH = 78, gap = 14, top = 150, left = [40, 460];
  const rows = Math.max(1, lanes.client_message.length, lanes.assistant_response.length);
  const H = top + rows * (boxH + gap) + 120;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "How a turn is routed to a class" });
  const defs = svgEl("defs");
  const m = svgEl("marker", { id: "arr", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto" });
  m.append(svgEl("path", { d: "M0,0 L10,5 L0,10 z", class: "arrfill" })); defs.append(m); svg.append(defs);

  svg.append(svgEl("rect", { x: W / 2 - 170, y: 16, width: 340, height: 54, rx: 10, class: "dbox" }));
  svg.append(svgEl("text", { x: W / 2, y: 38, "text-anchor": "middle", class: "dt" }, "One turn of a conversation"));
  svg.append(svgEl("text", { x: W / 2, y: 58, "text-anchor": "middle", class: "ds" }, "scored on its own, with the speaker known"));
  [["client_message", "A user turn…", left[0]], ["assistant_response", "An assistant turn…", left[1]]].forEach(([unit, title, x]) => {
    svg.append(svgEl("path", { d: `M${W / 2 + (x < W / 2 ? -40 : 40)} 70 C${W / 2 + (x < W / 2 ? -40 : 40)} 100 ${x + lane / 2} 90 ${x + lane / 2} 112`, class: "dedge" }));
    svg.append(svgEl("text", { x: x + lane / 2, y: 134, "text-anchor": "middle", class: "dlane" }, title));
    if (!lanes[unit].length) svg.append(svgEl("text", { x: x + lane / 2, y: top + 30, "text-anchor": "middle", class: "ds" }, "no classes apply to this speaker"));
    lanes[unit].forEach((k, i) => {
      const y = top + i * (boxH + gap), on = covered(k), [nm, ds] = info(k);
      const box = svgEl("rect", { x, y, width: lane, height: boxH, rx: 9, class: on ? "dbox dclass" : "dbox doff" });
      box.append(svgEl("title", {}, nm + ": " + ds + (on ? "" : " (not covered by the selected model: left unflagged)")));
      svg.append(box);
      svg.append(svgEl("text", { x: x + 14, y: y + 24, class: on ? "dt" : "dt doff-t" }, nm + (on ? "" : "  → out of scope, no flag")));
      const d = svgEl("text", { x: x + 14, y: y + 44, class: on ? "ds" : "ds doff-t" });
      wrapText(d, ds.length > 125 ? ds.slice(0, 122) + "…" : ds, 62); svg.append(d);
    });
  });
  const ny = top + rows * (boxH + gap) + 10;
  svg.append(svgEl("rect", { x: 40, y: ny, width: W - 80, height: 60, rx: 9, class: "dbox dneg" }));
  svg.append(svgEl("text", { x: W / 2, y: ny + 25, "text-anchor": "middle", class: "dt" }, `${labels[negative]}: the negative class, nothing to flag`));
  const nd = desc[negative] || "none of the classes above applies";
  svg.append(svgEl("text", { x: W / 2, y: ny + 45, "text-anchor": "middle", class: "ds" }, nd.length > 110 ? nd.slice(0, 107) + "…" : nd));
  $("classDiagram").replaceChildren(svg);

  const tbl = $("classTable"); tbl.replaceChildren();
  const hdr = el("tr"); ["Class", "Speaker", "Description the model scores against", "This model"].forEach((h) => hdr.append(el("th", "", h))); tbl.append(hdr);
  const rowFor = (k, speaker) => {
    const r = el("tr"); const on = covered(k), [nm, ds] = info(k);
    r.append(el("td", "", nm), el("td", "", speaker), el("td", "", ds));
    const c = el("td"); c.append(el("span", "pill " + (on ? "good" : "muted"), on ? "covered" : "out of scope → " + labels[negative])); r.append(c);
    return r;
  };
  const both = lanes.assistant_response.filter((k) => lanes.client_message.includes(k));
  lanes.assistant_response.filter((k) => !both.includes(k)).forEach((k) => tbl.append(rowFor(k, "assistant")));
  lanes.client_message.filter((k) => !both.includes(k)).forEach((k) => tbl.append(rowFor(k, "user")));
  both.forEach((k) => tbl.append(rowFor(k, "either")));
  const neg = el("tr"); neg.append(el("td", "", labels[negative]), el("td", "", "either"),
    el("td", "", desc[negative] || "None of the classes applies"), el("td", "", "always"));
  tbl.append(neg);
}

function showView(id) {
  document.querySelectorAll(".tab").forEach((b) => {
    b.classList.toggle("on", b.dataset.view === id);
    const v = $(b.dataset.view); if (v) v.hidden = b.dataset.view !== id;
  });
  // the sidebar (conversation + screening model) only where it is used; curation and training get the full width
  document.body.classList.toggle("no-sidebar", id === "curateView" || id === "trainView");
  // the sidebar's conversation controls belong to the Chat & flag tab only
  const chat = id === "chatView";
  $("convSection").hidden = !chat; $("chatOnly").hidden = !chat;
  const hint = $("sideHint");
  hint.hidden = chat;
  hint.textContent = { postView: "Post-training starts from the model selected above.",
                       aboutView: "The diagram describes the model selected above." }[id] || "";
  if (id === "aboutView" && state.config) renderAbout();
}

// ---------------------------------------------------------------- rendering
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function expectedLine(expected, v) {   // tooltip explains the comparison
  const line = expectedLineInner(expected, v);
  line.dataset.tip = "What the transcript's author expected for this turn versus what the model said. 'Out of scope' means the expected class is not one this model covers, so it is deliberately not flagged.";
  return line;
}
function expectedLineInner(expected, v) {
  // Compare the transcript's expectation with the verdict; labels the model doesn't cover are "out of scope".
  const exp = expected || "none";
  const nothing = ["none", "compliant", "no_flag"].includes(exp);
  const inScope = nothing || exp in state.labels;
  let cls = "match", text;
  if (!inScope) { cls = "scope"; text = `expected ${ALL_NAMES[exp] || exp} (out of scope for this model, so not flagged)`; }
  else if (nothing && !v.flagged) text = "expected nothing: agrees";
  else if (!nothing && v.label === exp) text = `expected ${ALL_NAMES[exp]}: agrees`;
  else { cls = "miss"; text = `expected ${ALL_NAMES[exp] || exp}, model said ${v.flagged ? (ALL_NAMES[v.label] || v.rule) : "nothing"}`; }
  return el("span", "expected " + cls, text);
}

function verdictBlock(turnId, target, text, v, meta = {}) {
  const box = el("div", "verdict");
  const chip = el("span", "chip " + (v.flagged ? "flag" : "ok"),
    v.flagged ? `⚑ ${v.rule}` : "✓ No issue");
  chip.dataset.tip = v.flagged
    ? `The model's verdict for this ${target === "user" ? "user" : "assistant"} turn. Probability ${(v.confidence * 100).toFixed(1)}% that it is ${v.rule}; 'details' shows every class.`
    : `Nothing to flag among the classes this ${target === "user" ? "user" : "assistant"} turn can trigger (probability ${(v.confidence * 100).toFixed(1)}%).`;
  const conf = el("span", "muted", `${(v.confidence * 100).toFixed(0)}%`);
  conf.dataset.tip = "Confidence: the softmax probability of the chosen class. Below ~60% the runner-up is close; open 'details'.";
  box.append(chip, conf);
  if (meta.expected !== undefined && meta.expected !== null) box.append(expectedLine(meta.expected, v));

  const probs = el("div", "probs");
  if (meta.note) probs.append(el("div", "note-line", "Transcript note: " + meta.note));
  Object.entries(v.probabilities).filter(([, p]) => p > 0)
    .sort((a, b) => b[1] - a[1]).forEach(([k, p]) => {
      const row = el("div", "probrow");
      const bar = el("div", "bar"); const fill = el("span"); fill.style.width = (p * 100).toFixed(1) + "%";
      bar.append(fill);
      row.append(el("span", "", state.labels[k] || k), bar, el("span", "muted", (p * 100).toFixed(1) + "%"));
      probs.append(row);
    });
  const toggle = el("button", "linkbtn", "details");
  toggle.type = "button"; toggle.dataset.tip = "Show the probability of every class for this turn" + (meta.note ? ", and the transcript author's note" : "") + ".";
  toggle.onclick = () => probs.classList.toggle("open");

  const kind = v.flagged ? "false_positive" : "false_negative";
  const flagBtn = el("button", "linkbtn", v.flagged ? "Flag false positive" : "Flag false negative");
  flagBtn.type = "button";
  flagBtn.dataset.tip = v.flagged ? "Disagree? Record that this turn should not have been flagged (or should be a different class). Becomes expert data for Post-train."
                                  : "Disagree? Record that this turn should have been flagged, and with which class. Becomes expert data for Post-train.";
  flagBtn.onclick = () => openFlag(box, flagBtn, { turnId, target, text, v, kind });
  box.append(toggle, flagBtn, probs);
  return box;
}

function openFlag(box, btn, ctx) {
  if (box.querySelector(".flagform")) return;
  const form = $("flagTpl").content.firstElementChild.cloneNode(true);
  const sel = form.querySelector("select");
  const unit = ctx.target === "user" ? "client_message" : "assistant_response";
  const allowed = state.config.unit_classes[unit];
  const negative = allowed[0];
  allowed.filter((k) => ctx.kind === "false_positive" ? k !== ctx.v.label : k !== negative)
    .forEach((k) => sel.add(new Option(state.labels[k] || k, k)));
  if (ctx.kind === "false_positive") sel.value = negative;
  form.querySelector(".cancel").onclick = () => form.remove();
  form.onsubmit = async (e) => {
    e.preventDefault();
    try {
      await api("/api/feedback", {
        turn_id: ctx.turnId, target: ctx.target, kind: ctx.kind, text: ctx.text,
        predicted_label: ctx.v.label, correct_label: sel.value || null,
        note: form.querySelector("[name=note]").value || null });
      form.replaceWith(el("span", "thanks", "Flag recorded"));
      btn.remove(); toast("Flag recorded: it will appear on the Post-train tab");
    } catch (err) { alert("Could not save flag: " + err.message); }
  };
  box.append(form);
}

function addBubble(role, text) {
  $("empty")?.remove();
  const b = el("div", "msg " + role);
  if (role === "user" || role === "assistant") b.append(el("div", "who", role === "user" ? "User" : "Assistant"));
  if (text) b.append(document.createTextNode(text));
  $("messages").append(b);
  b.scrollIntoView({ block: "end" });
  return b;
}

function resetChat() {
  state.history = []; state.sessionId = newId();
  if (!$("messages").querySelector(".msg") && $("empty")?.querySelector(".welcome")) return;   // nothing played yet: keep the welcome card
  const e = el("div", "empty"); e.id = "empty";
  e.append(el("p", "muted", $("provider").value === "scripted" ? "Press Play next exchange or Play all." : "Type a message below."));
  $("messages").replaceChildren(e);
}

// ---------------------------------------------------------------- one screened turn
async function runTurn(text, extra = {}) {
  const prov = providerPayload();
  if (prov.provider !== "scripted" && !prov.model) { alert("Set a model name first."); return; }
  ss("apiKey", $("apiKey").value);
  state.busy = true; $("send").disabled = true; updatePlaybarSafe();
  const userB = addBubble("user", text);
  const pending = addBubble("typing", prov.provider === "scripted" ? "Screening…" : "Screening and asking the model…");
  if ($("clfDot").className === "dot") setClf("busy");
  try {
    const turn = await api("/api/chat", {
      session_id: state.sessionId, provider: prov, history: state.history,
      message: text, system: $("system").value,
      scripted_reply: extra.scripted_reply ?? null, expected: extra.expected ?? null });
    setClf("on");
    userB.append(verdictBlock(turn.turn_id, "user", text, turn.user.verdict,
      { expected: extra.expected?.user, note: extra.notes?.user }));
    pending.remove();
    const reply = turn.assistant.text;
    state.history.push({ role: "user", content: text });
    if (reply) {
      const botB = addBubble("assistant", "");
      const withhold = turn.assistant.verdict.flagged && state.config.on_flagged_response === "withhold";
      if (withhold) {
        const w = el("span", "withheld", "Response withheld by screening. ");
        const show = el("button", "linkbtn", "Show anyway");
        show.onclick = () => { w.replaceWith(document.createTextNode(reply)); };
        w.append(show); botB.append(w);
      } else {
        botB.append(document.createTextNode(reply));
      }
      botB.append(verdictBlock(turn.turn_id, "assistant", reply, turn.assistant.verdict,
        { expected: extra.expected?.assistant, note: extra.notes?.assistant }));
      botB.scrollIntoView({ block: "end" });
      state.history.push({ role: "assistant", content: reply });
    }
  } catch (err) {
    pending.remove();
    addBubble("error", "Error: " + err.message);
    if ($("clfDot").className.includes("busy")) setClf("off");
  } finally {
    state.busy = false; $("send").disabled = false; updatePlaybarSafe();
    if (prov.provider !== "scripted") $("input").focus();
  }
}
function updatePlaybarSafe() { if (state.transcript) updatePlaybar(); }

async function send(e) {
  e?.preventDefault();
  const text = $("input").value.trim();
  if (!text || state.busy) return;
  $("input").value = "";
  await runTurn(text);
}

document.querySelectorAll(".tab, .topbar .flow button").forEach((b) => b.onclick = () => showView(b.dataset.view));
$("modelSelect").onchange = (e) => switchModel(e.target.value);
$("wPlay").onclick = () => { if (state.transcript) playAll(); else showView("chatView"); };
$("wTrain").onclick = () => showView("trainView");
$("wAbout").onclick = () => showView("aboutView");
$("composer").onsubmit = send;
$("input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) send(e); });
$("provider").onchange = () => { syncProvider(); if ($("provider").value === "scripted" && !state.transcript) loadTranscriptUrl("examples/regulatory/transcripts/demo_conversations.json"); };
$("warmup").onclick = warmup;
$("newChat").onclick = () => { resetChat(); if (state.transcript) { state.turnIdx = 0; updatePlaybar(); } };
$("conv").onchange = (e) => selectConversation(e.target.value);
$("upload").onchange = onUpload;
$("transcriptFile").onchange = async (e) => {
  try { setTranscript(await api("/api/files/transcript?path=" + encodeURIComponent(e.target.value)), e.target.value.split("/").pop()); }
  catch (err) { $("transcriptInfo").textContent = "Could not load transcript: " + err.message; }
};
loadTranscriptList();
$("playNext").onclick = playNext;
$("playAll").onclick = playAll;
$("restart").onclick = () => selectConversation(state.convIdx);
loadConfig().catch((e) => addBubble("error", "Could not load config: " + e.message));
