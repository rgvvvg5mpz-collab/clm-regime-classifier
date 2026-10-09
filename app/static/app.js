// Regime screening chat - plain JS, no build step.
// Two ways to drive a conversation: a live LLM (Anthropic / OpenAI-compatible), or a scripted
// transcript (bundled demo file or an uploaded .json, format in app/transcripts/FORMAT.md)
// replayed turn by turn through the same screening pipeline.
const $ = (id) => document.getElementById(id);
const state = { config: null, labels: {}, history: [], sessionId: newId(), busy: false,
                transcript: null, convIdx: 0, turnIdx: 0 };
// Display names for "expected" labels that the loaded model may not cover.
const ALL_NAMES = { none: "nothing", compliant: "nothing", no_flag: "nothing",
  finra_2210: "FINRA 2210", reg_bi: "Reg BI", finra_4530: "FINRA 4530",
  sec_17a3_17a4: "SEC 17a-3/4", reg_sp: "Reg S-P", reg_sid: "Reg S-ID" };

function newId() { return Math.random().toString(36).slice(2, 10); }
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
  const names = (u) => state.config.unit_classes[u].slice(1).map((k) => state.labels[k] || k).join(", ");
  const note = $("scopeNote");
  if (note) note.textContent = `Your message is checked against ${names("client_message")}. The model's reply is checked against ${names("assistant_response")}.`;
  $("apiKey").value = ss("apiKey") || "";
  applyPreset(state.config.presets[0]);
  const h = await api("/api/health");
  setClf(h.classifier_loaded ? "on" : "off");
}

function setClf(s, text) {
  $("clfDot").className = "dot" + (s === "on" ? " on" : s === "busy" ? " busy" : "");
  $("clfStatus").textContent = text || { on: "Classifier ready", busy: "Loading Qwen3-8B…", off: "Classifier not loaded" }[s];
  $("warmup").style.display = s === "off" ? "" : "none";
}

async function warmup() {
  setClf("busy");
  try { const r = await api("/api/warmup", {}); setClf("on", `Classifier ready (${r.seconds}s)`); }
  catch (e) { setClf("off", "Load failed: " + e.message); }
}

// ---------------------------------------------------------------- transcripts
async function loadTranscriptUrl(url) {
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
  $("transcriptInfo").textContent = `${t.title || name}: ${t.conversations.length} conversations, ` +
    `${total} turns the author expects flagged` + (t.description ? ` - ${t.description}` : "");
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

// ---------------------------------------------------------------- rendering
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function expectedLine(expected, v) {
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
  chip.title = `confidence ${(v.confidence * 100).toFixed(1)}%`;
  box.append(chip, el("span", "muted", `${(v.confidence * 100).toFixed(0)}%`));
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
  toggle.type = "button";
  toggle.onclick = () => probs.classList.toggle("open");

  const kind = v.flagged ? "false_positive" : "false_negative";
  const flagBtn = el("button", "linkbtn", v.flagged ? "Flag false positive" : "Flag false negative");
  flagBtn.type = "button";
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
      form.replaceWith(el("span", "thanks", "Flag recorded - thank you"));
      btn.remove();
    } catch (err) { alert("Could not save flag: " + err.message); }
  };
  box.append(form);
}

function addBubble(role, text) {
  $("empty")?.remove();
  const b = el("div", "msg " + role, text);
  $("messages").append(b);
  b.scrollIntoView({ block: "end" });
  return b;
}

function resetChat() {
  state.history = []; state.sessionId = newId();
  $("messages").replaceChildren(el("div", "empty", "New chat started."));
}

// ---------------------------------------------------------------- one screened turn
async function runTurn(text, extra = {}) {
  const prov = providerPayload();
  if (prov.provider !== "scripted" && !prov.model) { alert("Set a model name first."); return; }
  ss("apiKey", $("apiKey").value);
  state.busy = true; $("send").disabled = true; updatePlaybarSafe();
  const userB = addBubble("user", text);
  const pending = addBubble("assistant typing", prov.provider === "scripted" ? "Screening…" : "Screening and asking the model…");
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

$("composer").onsubmit = send;
$("input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) send(e); });
$("provider").onchange = () => { syncProvider(); if ($("provider").value === "scripted" && !state.transcript) loadTranscriptUrl("/static/transcripts/demo_conversations.json"); };
$("warmup").onclick = warmup;
$("newChat").onclick = () => { resetChat(); if (state.transcript) { state.turnIdx = 0; updatePlaybar(); } };
$("conv").onchange = (e) => selectConversation(e.target.value);
$("upload").onchange = onUpload;
$("playNext").onclick = playNext;
$("playAll").onclick = playAll;
$("restart").onclick = () => selectConversation(state.convIdx);
loadConfig().catch((e) => addBubble("error", "Could not load config: " + e.message));
