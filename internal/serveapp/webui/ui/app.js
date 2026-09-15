"use strict";
const $ = id => document.getElementById(id);
const KEY = "goinfer.apikey";

// --- auth -------------------------------------------------------------------
$("key").value = sessionStorage.getItem(KEY) || "";
$("key").addEventListener("change", () => sessionStorage.setItem(KEY, $("key").value));
function headers(extra) {
  const h = Object.assign({"Content-Type": "application/json"}, extra || {});
  const k = ($("key").value || "").trim();
  if (k) h["Authorization"] = "Bearer " + k;
  return h;
}

// --- tabs -------------------------------------------------------------------
function tab(name) {
  for (const n of ["chat", "models"]) {
    $("tab-" + n).setAttribute("aria-selected", String(n === name));
    $("pane-" + n).hidden = n !== name;
  }
}
$("tab-chat").onclick = () => tab("chat");
$("tab-models").onclick = () => tab("models");

// --- models -----------------------------------------------------------------
// Runtime stats as first-class chrome (decision 4): model, and whatever the
// server's own decode_path names, in the header — not buried in a status line.
// models is the last /v1/models list; the header stats follow whichever one is selected.
let models = [];
function showStats() {
  const stats = $("stats");
  stats.textContent = "";
  const cur = models.find(m => m.id === $("model").value) || models[0];
  if (!cur) { stats.textContent = "no model loaded"; return; }
  const add = (label, val) => {
    const s = document.createElement("span");
    const b = document.createElement("b"); b.textContent = val;
    s.appendChild(document.createTextNode(label + " ")); s.appendChild(b);
    stats.appendChild(s);
  };
  add("model", cur.id);
  if (cur.decode_path) add("path", cur.decode_path);
  showContext();
}

// --- context meter (W8) -------------------------------------------------------------
// How full the conversation is, against the context window the server enforces (/v1/models publishes
// context_window from the same function that rejects an oversized prompt). "Used" is the token usage
// the server reported for the latest reply — prompt plus completion, i.e. what the conversation
// occupied when that reply finished — so it is approximate by design: the next request also carries
// your new message, and sends earlier replies without their thinking (W6). It is shown as "about".
const CTX_WARN = 0.8, CTX_FULL = 0.95;
function contextUsed() {
  for (let i = transcript.length - 1; i >= 0; i--) {
    const u = transcript[i].usage;
    if (transcript[i].role === "assistant" && u) return u.prompt_tokens + u.completion_tokens;
  }
  return 0;
}
function showContext() {
  const cur = models.find(m => m.id === $("model").value);
  const win = cur && Number.isFinite(cur.context_window) && cur.context_window > 0 ? cur.context_window : 0;
  const used = contextUsed();
  const box = $("ctx");
  if (!win || !used) { box.hidden = true; return; }
  const frac = used / win;
  const pct = Math.min(100, Math.round(frac * 100));
  box.hidden = false;
  box.classList.toggle("warn", frac >= CTX_WARN && frac < CTX_FULL);
  box.classList.toggle("full", frac >= CTX_FULL);
  $("ctx-fill").style.width = pct + "%";
  $("ctx-bar").setAttribute("aria-valuenow", String(pct));
  let text = "about " + used.toLocaleString("en-US") + " of " + win.toLocaleString("en-US") + " tokens used (" + pct + "%)";
  if (frac >= CTX_FULL) text += " — the next message will likely not fit. Start a new chat, or delete earlier exchanges.";
  else if (frac >= CTX_WARN) text += " — nearing this model's context limit.";
  $("ctx-text").textContent = text;
}
// loadModels refreshes the model list. The current choice survives a refresh; pick, when given and
// listed, replaces it (W5 selects the model it just loaded).
async function loadModels(pick) {
  try {
    const r = await fetch("/v1/models", {headers: headers()});
    if (!r.ok) throw new Error("HTTP " + r.status);
    const j = await r.json();
    models = Array.isArray(j.data) ? j.data : [];
    const sel = $("model");
    const keep = pick || sel.value;
    sel.textContent = "";
    for (const m of models) {
      const o = document.createElement("option");
      o.value = m.id; o.textContent = m.id;
      sel.appendChild(o);
    }
    if (models.some(m => m.id === keep)) sel.value = keep;
    showStats();
  } catch (e) {
    $("stats").textContent = e.message;
  }
}
$("model").addEventListener("change", showStats);
loadModels();

// --- chat -------------------------------------------------------------------
// transcript is the conversation — one entry per turn: {role, content, model?, meta?, state?}, where
// state is "generating" (streaming now), "stopped" (Stop pressed) or "interrupted" (the page was
// reloaded mid-stream). What the API is sent is DERIVED from it (apiMessages), so the log on screen,
// what is saved, and what the model sees cannot drift apart.
const transcript = [];
let ac = null;
let generating = null;   // the in-progress assistant entry
const apiMessages = () => {
  // A past reply goes back WITHOUT its thinking (W6) — what Qwen3's own template does with history, and
  // it keeps a long reasoning trace from eating the context window on every later turn.
  const msgs = transcript.filter(m => m !== generating)
    .map(m => ({role: m.role, content: m.role === "assistant" ? answerOf(m.content) : m.content}));
  const sys = systemText();
  return sys ? [{role: "system", content: sys}, ...msgs] : msgs;   // W4
};

// --- thinking (W6) --------------------------------------------------------------
// Reasoning models put their thinking in the SAME content stream as the answer — the server does not
// split it — so the page folds it. Two shapes are recognised, both measured from real serve output:
//
//   Qwen3 and kin   "<think>\n…</think>\n\nanswer". The opening tag counts only as the very first thing
//                   in the reply, so an answer that merely mentions the tag is not folded. A "</think>" with
//                   NO opening tag also folds everything before it: some templates open the thinking in
//                   the prompt, so the model's output starts inside it.
//   gpt-oss         "<|channel|>analysis<|message|>…<|end|><|start|>assistant<|channel|>final<|message|>answer".
//
// splitThinking returns null for a reply with no thinking, else {thinking, answer, open}, where open means
// the model is still thinking. live says the reply is still streaming: only then is a half-arrived tag
// held back instead of shown, since after the stream "<thi" is just what the model said.
const THINK_OPEN = "<think>", THINK_CLOSE = "</think>";
const H_ANALYSIS = "<|channel|>analysis<|message|>", H_FINAL = "<|channel|>final<|message|>";
const H_END = "<|end|>", H_START = "<|start|>assistant";
const H_TAIL = /(<\|return\|>|<\|end\|>)\s*$/;

// heldBack is how many trailing characters of s could be the start of tag — hidden while live, so a
// closing tag split across two chunks never flashes on screen.
function heldBack(s, tag) {
  for (let n = Math.min(tag.length - 1, s.length); n > 0; n--) if (tag.startsWith(s.slice(-n))) return n;
  return 0;
}
const trimLive = (s, tag, live) => live ? s.slice(0, s.length - heldBack(s, tag)) : s;

function splitThinking(text, live) {
  const t = text.trimStart();
  if (live && t && t.length < H_ANALYSIS.length && (THINK_OPEN.startsWith(t) || H_ANALYSIS.startsWith(t) || H_FINAL.startsWith(t))) {
    return {thinking: "", answer: "", open: true, pending: true};   // could still be a tag, or not: show nothing yet
  }
  if (t.startsWith(THINK_OPEN)) {
    const body = t.slice(THINK_OPEN.length), i = body.indexOf(THINK_CLOSE);
    if (i < 0) return {thinking: trimLive(body, THINK_CLOSE, live), answer: "", open: live};
    return {thinking: body.slice(0, i), answer: body.slice(i + THINK_CLOSE.length).trimStart(), open: false};
  }
  if (t.startsWith(H_ANALYSIS)) {
    const body = t.slice(H_ANALYSIS.length), i = body.indexOf(H_END);
    if (i < 0) return {thinking: trimLive(body, H_END, live), answer: "", open: live};
    const rest = body.slice(i + H_END.length), f = rest.indexOf(H_FINAL);
    // between the two channels the model is still, as far as the reader can tell, thinking
    if (f < 0) return {thinking: body.slice(0, i), answer: "", open: live && H_START.startsWith(rest.trim().slice(0, H_START.length)) };
    return {thinking: body.slice(0, i), answer: trimLive(rest.slice(f + H_FINAL.length).replace(H_TAIL, ""), H_END, live), open: false};
  }
  if (t.startsWith(H_FINAL)) return {thinking: "", answer: trimLive(t.slice(H_FINAL.length).replace(H_TAIL, ""), H_END, live), open: false};
  const c = text.indexOf(THINK_CLOSE);
  if (c >= 0) return {thinking: text.slice(0, c), answer: text.slice(c + THINK_CLOSE.length).trimStart(), open: false};
  return null;
}

// answerOf is a reply without its thinking: what later turns send back. copyOf is what Copy copies —
// the answer, or the whole reply when there is no answer to copy (stopped mid-thought).
function answerOf(text) {
  const p = splitThinking(text, false);
  return p ? p.answer : text;
}
const copyOf = text => answerOf(text) || text;

// renderReply draws a model reply into out: straight through Markdown.render when there is no
// thinking, otherwise a folded <details> (collapsed by default) above the answer. The <details> is
// built once and then UPDATED, so a reader who opens it mid-stream keeps it open while tokens arrive.
// Both parts go through Markdown.render — thinking is model output too, and gets no more trust.
// e is the transcript entry: its thought (seconds) labels the fold, and a finished reply with no state
// that thought but never answered says so rather than showing an empty bubble.
function renderReply(out, text, live, e) {
  const p = splitThinking(text, live);
  let box = out.firstElementChild;
  const folded = box && box.classList.contains("think");
  if (!p) {
    if (folded) out.replaceChildren();
    Markdown.render(out, text);
    return;
  }
  if (p.pending) {
    out.replaceChildren();
    return;
  }
  if (!folded) {
    box = document.createElement("details");
    box.className = "think";
    const sum = document.createElement("summary");
    const body = document.createElement("div");
    body.className = "think-body md";
    box.appendChild(sum); box.appendChild(body);
    const ans = document.createElement("div");
    ans.className = "think-answer";
    out.replaceChildren(box, ans);
  }
  const [sum, body] = box.children, ans = out.children[1];
  sum.textContent = p.open ? "Thinking…" : (e && Number.isFinite(e.thought) ? "Thought for " + e.thought.toFixed(e.thought < 10 ? 1 : 0) + "s" : "Thinking");
  box.classList.toggle("live", p.open);
  Markdown.render(body, p.thinking);
  if (p.answer || live || !p.thinking || (e && e.state)) {
    Markdown.render(ans, p.answer);
  } else {
    ans.replaceChildren();
    const n = document.createElement("p");
    n.className = "note";
    n.textContent = "No answer after the thinking — the model stopped before it answered.";
    ans.appendChild(n);
  }
}

// --- copy (W2) ----------------------------------------------------------------
// The raw text each message was rendered from. A message's Copy button copies THIS — the Markdown
// source — not the rendered text, so pasting it elsewhere keeps its formatting.
const sources = new WeakMap();

// copyText tries the async clipboard API, then falls back to a selected textarea. The fallback is not
// an edge case: the async API only exists in a SECURE context, and the page opened from another
// machine at a plain-http LAN address (a phone on the LAN, W16) is not one.
async function copyText(s) {
  if (navigator.clipboard && window.isSecureContext) {
    try { await navigator.clipboard.writeText(s); return; } catch { /* fall through */ }
  }
  const ta = document.createElement("textarea");
  ta.value = s;
  ta.setAttribute("readonly", "");
  ta.style.position = "fixed"; ta.style.top = "0"; ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.select();
  let ok = false;
  try { ok = document.execCommand("copy"); } catch { ok = false; }
  ta.remove();
  if (!ok) throw new Error("copy refused");
}

function flash(btn, label) {
  if (!btn.dataset.label) btn.dataset.label = btn.textContent;
  btn.textContent = label;
  clearTimeout(btn.flashTimer);
  btn.flashTimer = setTimeout(() => { btn.textContent = btn.dataset.label; }, 1500);
}

// One delegated handler for every Copy button in the log. Code-block buttons are rebuilt on each
// streamed frame by Markdown.render, so binding them individually would drop handlers mid-stream.
$("log").addEventListener("click", async e => {
  const btn = e.target.closest("button.md-copy, button.msg-copy");
  if (!btn) return;
  const text = btn.classList.contains("md-copy")
    ? btn.closest(".md-code")?.querySelector("pre code")?.textContent
    : sources.get(btn.closest(".msg")?.children[1]);
  if (text == null) return;
  try { await copyText(text); flash(btn, "Copied"); }
  catch { flash(btn, "Copy failed"); }
});

// --- persistence (W3) ------------------------------------------------------------
// The conversation survives a reload, in localStorage under a versioned key. Every failure mode is
// non-fatal: storage blocked, quota exceeded, or a value this page cannot read — the chat keeps
// working, and an unreadable value is set aside under STORE+".unreadable" rather than destroyed.
// One conversation is shared by every tab on this origin; an idle tab follows another tab's changes
// (the "storage" listener below). Separate conversations are W9.
const STORE = "goinfer.chat.v1";

function storeFailed() {
  $("store-note").textContent = "This conversation can't be kept across a reload (browser storage is full or blocked) — " +
    "it stays on screen until you close or reload the page.";
  $("store-note").hidden = false;
}

function save() {
  try {
    localStorage.setItem(STORE, JSON.stringify({ v: 1, messages: transcript }));
    $("store-note").hidden = true;
  } catch {
    storeFailed();
  }
}

// --- system prompt (W4) -----------------------------------------------------------
// A SETTING, not part of a conversation: it has its own key, survives New chat and reloads, and is
// prepended to every request (apiMessages) when non-blank, trimmed. The route accepts a system message
// for every family — templates without a system role fold it into a user turn server-side. It is not
// written into the transcript, because it is applied per request, not said once. W9 (separate
// conversations) may make it per-conversation.
const SYSTEM_STORE = "goinfer.system.v1";
const systemText = () => $("system").value.trim();
function showSystemState() { $("system-state").textContent = systemText() ? "· active" : ""; }
function saveSystem() {
  try {
    if (systemText()) localStorage.setItem(SYSTEM_STORE, $("system").value);
    else localStorage.removeItem(SYSTEM_STORE);
  } catch {
    storeFailed();
  }
}
function loadSystem() {
  try { $("system").value = localStorage.getItem(SYSTEM_STORE) || ""; } catch { /* blocked */ }
  showSystemState();
}

function loadStored() {
  let raw;
  try { raw = localStorage.getItem(STORE); } catch { return []; }   // storage blocked
  if (!raw) return [];
  let data = null;
  try { data = JSON.parse(raw); } catch { /* unreadable */ }
  if (!data || data.v !== 1 || !Array.isArray(data.messages)) {
    try { localStorage.setItem(STORE + ".unreadable", raw); localStorage.removeItem(STORE); } catch { /* leave it */ }
    return [];
  }
  const out = [];
  for (const m of data.messages) {
    // Validate every field: this is data, and it is rendered.
    if (!m || (m.role !== "user" && m.role !== "assistant") || typeof m.content !== "string") continue;
    const e = { role: m.role, content: m.content };
    if (typeof m.model === "string") e.model = m.model;
    if (typeof m.meta === "string") e.meta = m.meta;
    if (m.state === "stopped" || m.state === "interrupted") e.state = m.state;
    if (m.state === "generating") e.state = "interrupted";   // it was streaming when the page went away
    if (typeof m.thought === "number" && Number.isFinite(m.thought) && m.thought >= 0) e.thought = m.thought;   // W6
    const u = m.usage, tok = x => Number.isSafeInteger(x) && x >= 0;
    if (u && tok(u.prompt_tokens) && tok(u.completion_tokens)) e.usage = {prompt_tokens: u.prompt_tokens, completion_tokens: u.completion_tokens};   // W8
    out.push(e);
  }
  return out;
}

function metaText(e) {
  if (e.state === "interrupted") return (e.meta ? e.meta + " · " : "") + "interrupted — the page was reloaded while this was generating";
  if (e.state === "stopped") return "stopped" + (e.meta ? " · " + e.meta : "");
  return e.meta || "";
}

function addMeta(content, text) {
  if (!text) return;
  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = text;
  content.parentElement.appendChild(meta);
}

// renderEntry draws one finished turn exactly as a live one is drawn: the user's text as text, the
// model's through Markdown.render. Restored content gets no more trust than fresh model output.
function renderEntry(e) {
  if (e.role === "user") {
    const b = bubble("you", "you amb-surface");
    b.textContent = e.content;
    addActions(b, e.content, e);
    return;
  }
  const out = bubble(e.model || "assistant", "bot amb-surface-convex amb-elevation-1");
  out.classList.add("md");
  renderReply(out, e.content, false, e);
  addMeta(out, metaText(e));
  addActions(out, e.content ? copyOf(e.content) : "", e);
}

function showTranscript(list) {
  transcript.length = 0;
  transcript.push(...list);
  $("log").replaceChildren();
  for (const e of transcript) renderEntry(e);
  showContext();
}

// addActions records a finished message's source and adds its action row: Copy (W2), and W7's Edit on
// the user's messages, Regenerate on the last reply, and Delete on both. entry is the message's
// transcript entry — the row acts on the conversation through it, never through the DOM.
const entries = new WeakMap();   // .msg element -> its transcript entry
function addActions(content, src, entry) {
  sources.set(content, src);
  const msg = content.parentElement;
  entries.set(msg, entry);
  const row = document.createElement("div");
  row.className = "actions";
  const button = (cls, label, aria) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = cls;
    b.textContent = label;
    b.setAttribute("aria-label", aria);
    row.appendChild(b);
  };
  if (entry.role === "user") button("msg-edit", "Edit", "Edit message");
  else button("msg-regen", "Regenerate", "Regenerate reply");
  if (src) button("msg-copy", "Copy", "Copy message");
  button("msg-delete", "Delete", "Delete this exchange");
  msg.appendChild(row);
  syncActions();
}

// syncActions keeps every row truthful: Regenerate only on the reply that is LAST in the conversation
// (regenerating an earlier one would silently discard everything after it — that is what Edit is
// for), and nothing that changes the conversation while a reply is generating or a message is being
// edited.
function syncActions() {
  const last = transcript[transcript.length - 1];
  const busy = !!ac || !!editing;
  for (const msg of document.querySelectorAll("#log .msg")) {
    const e = entries.get(msg);
    if (!e) continue;
    const regen = msg.querySelector(".msg-regen");
    if (regen) regen.hidden = e !== last;
    for (const b of msg.querySelectorAll(".msg-edit, .msg-regen, .msg-delete")) b.disabled = busy;
  }
}

// --- regenerate, edit, delete (W7) ------------------------------------------------
// History is addressable now: each action edits the transcript, re-renders the log from it, saves, and
// (for Regenerate and Edit) asks for a new reply to the conversation as it now stands.
let editing = null;   // {entry, msg} while a message is open for editing

function rerender() {
  showTranscript(transcript.slice());
  save();
}

function regenerate(entry) {
  if (ac || editing || entry !== transcript[transcript.length - 1] || entry.role !== "assistant") return;
  transcript.pop();
  rerender();
  generate();
}

// deleteExchange removes a user message together with its reply — deleting half an exchange would
// leave two user turns in a row, which some chat templates refuse. No undo exists, so it asks first.
function deleteExchange(entry) {
  if (ac || editing) return;
  const i = transcript.indexOf(entry);
  if (i < 0) return;
  const start = entry.role === "assistant" && transcript[i - 1]?.role === "user" ? i - 1 : i;
  const end = transcript[start].role === "user" && transcript[start + 1]?.role === "assistant" ? start + 2 : start + 1;
  if (!confirm(end - start > 1 ? "Delete this message and its reply?" : "Delete this message?")) return;
  transcript.splice(start, end - start);
  rerender();
}

function startEdit(entry, msg) {
  if (ac || editing) return;
  const box = msg.children[1];
  const ta = document.createElement("textarea");
  ta.className = "edit-box";
  ta.value = entry.content;
  ta.setAttribute("aria-label", "Edit message");
  const row = document.createElement("div");
  row.className = "edit-row";
  const saveBtn = document.createElement("button");
  saveBtn.type = "button"; saveBtn.className = "edit-save"; saveBtn.textContent = "Save & send";
  const cancel = document.createElement("button");
  cancel.type = "button"; cancel.className = "edit-cancel"; cancel.textContent = "Cancel";
  row.appendChild(cancel); row.appendChild(saveBtn);
  box.replaceChildren(ta, row);
  msg.querySelector(".actions").hidden = true;
  editing = {entry, msg};
  syncActions();
  $("send").disabled = true;
  ta.focus();
  ta.setSelectionRange(ta.value.length, ta.value.length);
  saveBtn.onclick = () => finishEdit(ta.value);
  cancel.onclick = () => finishEdit(null);
  ta.addEventListener("keydown", e => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); finishEdit(ta.value); }
    if (e.key === "Escape") { e.preventDefault(); finishEdit(null); }
  });
}

// finishEdit(null) cancels. Saving replaces the message, drops everything after it — asking first when
// that is more than the one reply the edit is about to replace — and generates a new reply.
function finishEdit(value) {
  if (!editing) return;
  const {entry} = editing;
  if (value === null) {
    editing = null;
    $("send").disabled = false;
    showTranscript(transcript.slice());
    return;
  }
  const text = value.trim();
  if (!text) return;   // an empty message is not a message; keep the box open
  const i = transcript.indexOf(entry);
  const later = transcript.length - i - 1;
  if (later > 1 && !confirm("Saving will remove the " + later + " messages after this one. Continue?")) return;
  editing = null;
  $("send").disabled = false;
  entry.content = text;
  transcript.length = i + 1;
  rerender();
  generate();
}

$("log").addEventListener("click", e => {
  const btn = e.target.closest("button.msg-regen, button.msg-edit, button.msg-delete");
  if (!btn || btn.disabled) return;
  const msg = btn.closest(".msg"), entry = entries.get(msg);
  if (!entry) return;
  if (btn.classList.contains("msg-regen")) regenerate(entry);
  else if (btn.classList.contains("msg-edit")) startEdit(entry, msg);
  else deleteExchange(entry);
});

function bubble(who, cls) {
  const d = document.createElement("div");
  d.className = "msg ambient amb-rounded " + cls;
  const h = document.createElement("div");
  h.className = "who"; h.textContent = who;
  const b = document.createElement("div");
  d.appendChild(h); d.appendChild(b);
  $("log").appendChild(d);
  d.scrollIntoView({block: "end"});
  // Content goes in via textContent (the user's own text, errors) or Markdown.render (model
  // output, W1) — never innerHTML. TestWebUI_noHTMLStringSinks enforces that for the whole page.
  return b;
}

async function send() {
  const text = $("prompt").value.trim();
  if (!text || ac || editing) return;
  if (!$("model").value) { $("chat-status").textContent = "no model loaded"; return; }

  // What you type is recessed (amb-surface-concave, decision 3); the echo of
  // it in the log is a quieter flat surface — neither is the "product".
  const mine = bubble("you", "you amb-surface");
  mine.textContent = text;
  const u = {role: "user", content: text};
  transcript.push(u);
  addActions(mine, text, u);
  $("prompt").value = "";
  await generate();
}

// generate asks for a reply to the conversation as it stands: after send(), and after W7's Regenerate
// and Edit, which change the transcript first.
async function generate() {
  if (ac || editing) return;
  const model = $("model").value;
  if (!model) { $("chat-status").textContent = "no model loaded"; return; }
  const messages = apiMessages();          // taken BEFORE the assistant entry exists
  save();
  // What the engine produces sits proud (amb-surface-convex, decision 3).
  const out = bubble(model, "bot amb-surface-convex amb-elevation-1");
  out.classList.add("md");
  const entry = {role: "assistant", content: "", model, state: "generating"};
  transcript.push(entry);
  generating = entry;
  $("send").disabled = true; $("stop").hidden = false; $("newchat").disabled = true;
  $("chat-status").textContent = "generating…";

  ac = new AbortController();
  syncActions();
  const started = performance.now();
  let got = 0, acc = "", lastSave = 0;
  // Render model output as Markdown (W1). Throttled to one parse per animation frame: a fast token
  // stream would otherwise re-parse the whole answer once per token. The final render after the
  // stream (or after Stop) guarantees the last chunk is shown even if no frame ran after it.
  let paintQueued = false;
  let live = true, thinkFrom = 0;
  const paint = () => {
    paintQueued = false;
    // W6: time the thinking from its first token to the first frame that sees it closed
    const p = splitThinking(acc, live);
    if (p && p.thinking && !thinkFrom) thinkFrom = performance.now();
    if (p && thinkFrom && !p.open && entry.thought === undefined) entry.thought = (performance.now() - thinkFrom) / 1000;
    renderReply(out, acc, live, entry);
    out.scrollIntoView({block: "end"});
  };
  try {
    const r = await fetch("/v1/chat/completions", {
      method: "POST", headers: headers(), signal: ac.signal,
      body: JSON.stringify({
        model, messages, stream: true,
        stream_options: {include_usage: true},   // W8: real token counts, for the meter and the stats
        temperature: parseFloat($("temp").value) || 0,
        max_tokens: parseInt($("max").value, 10) || 512,
      }),
    });
    if (!r.ok) {
      const body = await r.text();
      // W8: the wall, said in words a user can act on — the server's own message stays underneath.
      if (r.status === 400 && body.includes("context_length_exceeded")) {
        throw new Error("This conversation no longer fits in " + model + "'s context window. Start a new chat, or delete earlier exchanges to make room.\n\n" + body.slice(0, 400));
      }
      throw new Error(body.slice(0, 400) || ("HTTP " + r.status));
    }
    for await (const ev of sse(r)) {
      if (ev.data === "[DONE]") break;
      let j; try { j = JSON.parse(ev.data); } catch { continue; }
      if (j.usage && Number.isSafeInteger(j.usage.prompt_tokens) && Number.isSafeInteger(j.usage.completion_tokens)) {
        entry.usage = {prompt_tokens: j.usage.prompt_tokens, completion_tokens: j.usage.completion_tokens};
      }
      const d = j.choices && j.choices[0] && j.choices[0].delta;
      if (d && d.content) {
        acc += d.content; got++;
        entry.content = acc;
        if (!paintQueued) { paintQueued = true; requestAnimationFrame(paint); }
        // Save the partial answer about once a second, so a reload mid-stream keeps what arrived (W3).
        const now = performance.now();
        if (now - lastSave > 1000) { lastSave = now; save(); }
      }
    }
    live = false;
    paint();
    const s = (performance.now() - started) / 1000;
    // Per-response stats live WITH the response, not in a status line that
    // the next turn overwrites (decision 4).
    // W8: the server's completion_tokens when it reported usage. Counting chunks undercounts — a token held
    // back for a partial UTF-8 rune or stop string arrives merged into the next chunk.
    const toks = entry.usage ? entry.usage.completion_tokens : got;
    entry.meta = toks
      ? toks + " tok · " + (toks / s).toFixed(1) + " tok/s · " + s.toFixed(1) + "s"
      : "no output";
    delete entry.state;
    renderReply(out, acc, false, entry);   // the entry is final now: "no answer" can be judged
    addMeta(out, entry.meta);
    addActions(out, copyOf(acc), entry);
    showContext();
    $("chat-status").textContent = "";
  } catch (e) {
    if (e.name === "AbortError") {
      live = false;
      entry.content = acc;
      entry.state = "stopped";
      paint();
      if (got) entry.meta = got + " tok";
      addMeta(out, metaText(entry));
      $("chat-status").textContent = "stopped";
      addActions(out, acc ? copyOf(acc) : "", entry);   // a stopped answer is still worth copying, and regenerating
    } else {
      // A failed request is not a turn: drop it, as the old history never recorded one either.
      transcript.splice(transcript.indexOf(entry), 1);
      out.textContent = String(e.message || e); out.parentElement.classList.add("err"); $("chat-status").textContent = "";
    }
  } finally {
    generating = null;
    save();
    ac = null; $("send").disabled = false; $("stop").hidden = true; $("newchat").disabled = false;
    syncActions();
  }
}
$("send").onclick = send;

// New chat (W3): with the conversation now persistent, it needs a way to start over. Asks first,
// because clearing has no undo. Disabled while a reply is generating.
$("newchat").onclick = () => {
  if (ac) return;
  if (editing) finishEdit(null);
  if (transcript.length && !confirm("Start a new chat? This conversation will be cleared.")) return;
  transcript.length = 0;
  $("log").replaceChildren();
  try { localStorage.removeItem(STORE); } catch { /* blocked: nothing was stored */ }
  $("store-note").hidden = true;
  showContext();
  $("chat-status").textContent = "";
};

// A reload mid-stream: save what has arrived. pagehide fires where beforeunload is unreliable (mobile).
addEventListener("pagehide", () => { if (generating) save(); });

// Another tab changed the conversation or the system prompt: follow it — unless this tab is mid-reply
// (its own save wins then), or, for the system prompt, the user is typing in that box right now.
addEventListener("storage", e => {
  if ((e.key === SYSTEM_STORE || e.key === null) && document.activeElement !== $("system")) loadSystem();
  if ((e.key === STORE || e.key === null) && !ac && !editing) showTranscript(loadStored());
});

$("system").addEventListener("input", () => { showSystemState(); saveSystem(); });

loadSystem();

// Restore the conversation this browser was having. A "generating" entry becomes "interrupted"; saving
// right away writes that back, so the state is recorded rather than re-derived on every load.
showTranscript(loadStored());
if (transcript.some(e => e.state === "interrupted")) save();
$("stop").onclick = () => ac && ac.abort();
$("prompt").addEventListener("keydown", e => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); send(); }
});

// Minimal SSE reader over fetch. The server sends "event:"/"data:" frames separated by a
// blank line; we only need the last data line per frame.
async function* sse(resp) {
  const rd = resp.body.getReader(), dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const {value, done} = await rd.read();
    if (done) break;
    buf += dec.decode(value, {stream: true});
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const frame = buf.slice(0, i); buf = buf.slice(i + 2);
      let ev = "message", data = "";
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) ev = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (data) yield {event: ev, data};
    }
  }
}

// --- pull -------------------------------------------------------------------
$("list").onclick = async () => {
  const repo = $("repo").value.trim();
  if (!repo) return;
  $("list").disabled = true;
  $("files-card").hidden = false;
  $("files").textContent = "listing…";
  try {
    const r = await fetch("/web/models/list", {method: "POST", headers: headers(), body: JSON.stringify({repo})});
    const j = await r.json();
    if (!r.ok) throw new Error((j.error && j.error.message) || ("HTTP " + r.status));
    renderFiles(repo, j.files || []);
  } catch (e) {
    $("files").textContent = "";
    const p = document.createElement("p"); p.className = "err"; p.textContent = String(e.message || e);
    $("files").appendChild(p);
  } finally { $("list").disabled = false; }
};

function renderFiles(repo, files) {
  const host = $("files"); host.textContent = "";
  if (!files.length) { host.textContent = "no .gguf files in this repo"; return; }
  const t = document.createElement("table");
  const head = document.createElement("tr");
  for (const h of ["File", "Size", ""]) { const th = document.createElement("th"); th.textContent = h; head.appendChild(th); }
  t.appendChild(head);
  for (const f of files) {
    const tr = document.createElement("tr");
    const a = document.createElement("td"); a.className = "f"; a.textContent = f.path;
    const b = document.createElement("td"); b.className = "n"; b.textContent = f.human;
    const c = document.createElement("td"); c.className = "n";
    const btn = document.createElement("button"); btn.className = "ambient amb-surface-convex amb-elevation-1 amb-rounded go"; btn.textContent = "Pull";
    btn.onclick = () => pull(repo, f.path);
    c.appendChild(btn);
    tr.appendChild(a); tr.appendChild(b); tr.appendChild(c);
    t.appendChild(tr);
  }
  host.appendChild(t);
}

async function pull(repo, file) {
  $("pull-card").hidden = false;
  $("pull-what").textContent = repo + " · " + file;
  $("bar").style.width = "0%";
  const st = $("pull-status"); st.className = "note"; st.textContent = "starting…";
  document.querySelectorAll("#files button").forEach(b => b.disabled = true);
  try {
    const r = await fetch("/web/models/pull", {method: "POST", headers: headers(), body: JSON.stringify({repo, file})});
    if (!r.ok) {
      let m = "HTTP " + r.status;
      try { const j = await r.json(); if (j.error && j.error.message) m = j.error.message; } catch {}
      throw new Error(m);
    }
    for await (const ev of sse(r)) {
      const j = JSON.parse(ev.data);
      if (ev.event === "start") {
        st.textContent = j.human + (j.sha256 ? " · sha256 " + j.sha256.slice(0, 16) + "…" : " · no sha256 published");
      } else if (ev.event === "progress") {
        if (j.total > 0) $("bar").style.width = (100 * j.done / j.total).toFixed(1) + "%";
        st.textContent = j.human + " · " + j.rate + (j.eta ? " · eta " + j.eta : "");
      } else if (ev.event === "done") {
        $("bar").style.width = "100%";
        st.className = "note ok";
        st.textContent = "";
        st.appendChild(document.createTextNode(
          (j.verified ? "sha256 verified" : "no sha256 published — NOT verified") + " · " + j.elapsed + " · "));
        const code = document.createElement("code"); code.textContent = j.path;
        st.appendChild(code);
        // The file is on disk, but the server has not loaded it. Say so rather than letting the
        // green tick imply the model is live — and offer the load right here (W5).
        offerLoad(st, j.path);
        loadModels();
      } else if (ev.event === "error") {
        throw new Error(j.message);
      }
    }
  } catch (e) {
    st.className = "note err"; st.textContent = String(e.message || e);
  } finally {
    document.querySelectorAll("#files button").forEach(b => b.disabled = false);
  }
}

// --- load (W5) --------------------------------------------------------------
// The pull flow used to end on "restart the server with --model <path>". Now it ends on a button that
// asks the server to load the file it just downloaded. The server confines that route to regular .gguf
// files inside its own pull cache, so the page can load what it pulled and nothing else.
function offerLoad(host, path) {
  const row = document.createElement("div");
  row.className = "load-row";
  const msg = document.createElement("span");
  msg.className = "note"; msg.id = "load-status";
  msg.textContent = "Downloaded, not loaded yet.";
  const btn = document.createElement("button");
  btn.type = "button"; btn.id = "load-now";
  btn.className = "ambient amb-surface-convex amb-elevation-1 amb-rounded go";
  btn.textContent = "Load it now";
  btn.onclick = () => loadPulled(path, btn, msg);
  row.appendChild(btn); row.appendChild(msg);
  host.appendChild(row);
}

// servedName is the name the server gives a loaded file: its base name without the extension.
const servedName = path => path.split(/[\\/]/).pop().replace(/\.gguf$/i, "");

async function loadPulled(path, btn, msg) {
  btn.disabled = true;
  msg.className = "note"; msg.textContent = "loading…";
  const loaded = async (id, text) => {
    await loadModels(id);
    msg.className = "note ok"; msg.textContent = text;
    btn.hidden = true;
  };
  try {
    const r = await fetch("/web/models/load", {method: "POST", headers: headers(), body: JSON.stringify({path})});
    if (!r.ok) {
      let m = "HTTP " + r.status;
      try { const j = await r.json(); if (j.error && j.error.message) m = j.error.message; } catch {}
      // Already loaded is the outcome the user wanted, not a failure: select it and say so.
      if (r.status === 409 && /already loaded/.test(m)) {
        const id = servedName(path);
        await loaded(id, "Already loaded as " + id + " — selected in Chat.");
        return;
      }
      throw new Error(m);
    }
    let finished = false;
    for await (const ev of sse(r)) {
      const j = JSON.parse(ev.data);
      if (ev.event === "start") {
        msg.textContent = "loading " + j.name + "…";
      } else if (ev.event === "progress") {
        msg.textContent = "loading… " + j.elapsed;
      } else if (ev.event === "done") {
        finished = true;
        await loaded(j.id, "Loaded " + j.id + " in " + j.elapsed + " — selected in Chat.");
      } else if (ev.event === "error") {
        throw new Error(j.message);
      }
    }
    // A stream that ends without done or error was cut off (the server went away mid-load).
    if (!finished) throw new Error("the load stream ended without a result — check the server, then refresh the model list");
  } catch (e) {
    msg.className = "note err"; msg.textContent = String(e.message || e);
    btn.disabled = false;
  }
}
