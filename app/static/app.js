// Regime screening chat - plain JS, no build step.
const $ = (id) => document.getElementById(id);
const state = { config: null, labels: {}, history: [], sessionId: newId(), busy: false };

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
}
function syncProvider() {
  const oa = $("provider").value === "openai_compatible";
  $("baseUrlRow").style.display = oa ? "" : "none";
  $("apiKey").placeholder = $("apiKey").dataset.env
    ? `uses ${$("apiKey").dataset.env} if empty` : "optional";
}
function providerPayload() {
  return {
    provider: $("provider").value, model: $("model").value.trim(),
    base_url: $("provider").value === "openai_compatible" ? $("baseUrl").value.trim() || null : null,
    api_key: $("apiKey").value || null, api_key_env: $("apiKey").dataset.env || null,
  };
}

async function loadConfig() {
  state.config = await api("/api/config");
  state.labels = state.config.labels;
  const sel = $("preset");
  state.config.presets.forEach((p, i) => sel.add(new Option(p.name, i)));
  sel.onchange = () => applyPreset(state.config.presets[sel.value]);
  applyPreset(state.config.presets[0]);
  $("system").value = state.config.system_prompt || "";
  $("apiKey").value = ss("apiKey") || "";
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

// ---------------------------------------------------------------- rendering
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function verdictBlock(turnId, target, text, v) {
  const box = el("div", "verdict");
  const chip = el("span", "chip " + (v.flagged ? "flag" : "ok"),
    v.flagged ? `⚑ ${v.rule}` : "✓ No issue");
  chip.title = `confidence ${(v.confidence * 100).toFixed(1)}%`;
  box.append(chip, el("span", "muted", `${(v.confidence * 100).toFixed(0)}%`));

  const probs = el("div", "probs");
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
  const allowed = ctx.target === "user"
    ? ["compliant", "finra_4530", "sec_17a3_17a4", "reg_sp", "reg_sid"]
    : ["compliant", "finra_2210", "reg_bi"];
  allowed.filter((k) => ctx.kind === "false_positive" ? k !== ctx.v.label : k !== "compliant")
    .forEach((k) => sel.add(new Option(state.labels[k] || k, k)));
  if (ctx.kind === "false_positive") sel.value = "compliant";
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

// ---------------------------------------------------------------- send
async function send(e) {
  e?.preventDefault();
  const text = $("input").value.trim();
  if (!text || state.busy) return;
  const prov = providerPayload();
  if (!prov.model) { alert("Set a model name first."); return; }
  ss("apiKey", $("apiKey").value);
  state.busy = true; $("send").disabled = true; $("input").value = "";
  const userB = addBubble("user", text);
  const pending = addBubble("assistant typing", "Screening and asking the model…");
  if ($("clfDot").className === "dot") setClf("busy");
  try {
    const turn = await api("/api/chat", {
      session_id: state.sessionId, provider: prov, history: state.history,
      message: text, system: $("system").value });
    setClf("on");
    userB.append(verdictBlock(turn.turn_id, "user", text, turn.user.verdict));
    pending.remove();
    const reply = turn.assistant.text;
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
    botB.append(verdictBlock(turn.turn_id, "assistant", reply, turn.assistant.verdict));
    botB.scrollIntoView({ block: "end" });
    state.history.push({ role: "user", content: text }, { role: "assistant", content: reply });
  } catch (err) {
    pending.remove();
    addBubble("error", "Error: " + err.message);
    if ($("clfDot").className.includes("busy")) setClf("off");
  } finally {
    state.busy = false; $("send").disabled = false; $("input").focus();
  }
}

$("composer").onsubmit = send;
$("input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) send(e); });
$("provider").onchange = syncProvider;
$("warmup").onclick = warmup;
$("newChat").onclick = () => {
  state.history = []; state.sessionId = newId();
  $("messages").replaceChildren(el("div", "empty", "New chat started."));
};
loadConfig().catch((e) => addBubble("error", "Could not load config: " + e.message));
