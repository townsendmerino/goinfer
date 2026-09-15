"use strict";
const $ = id => document.getElementById(id);

// --- theme (W15) --------------------------------------------------------------------------
// System (the default: follows prefers-color-scheme), Light or Dark. A choice is a data-theme attribute on
// <html>; the stylesheet does the rest. Applied first thing, so a page reloaded in Dark does not repaint.
const THEME_STORE = "goinfer.theme.v1";
function applyTheme(t) {
  if (t !== "light" && t !== "dark") t = "system";
  if (t === "system") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.dataset.theme = t;
  $("theme").value = t;
}
function loadTheme() {
  let t = "system";
  try { t = localStorage.getItem(THEME_STORE) || "system"; } catch { /* blocked */ }
  applyTheme(t);
}
loadTheme();
$("theme").addEventListener("change", () => {
  applyTheme($("theme").value);
  try {
    if ($("theme").value === "system") localStorage.removeItem(THEME_STORE);
    else localStorage.setItem(THEME_STORE, $("theme").value);
  } catch { /* the choice still applies to this page */ }
});
addEventListener("storage", e => { if (e.key === THEME_STORE || e.key === null) loadTheme(); });
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
  showAttach();
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
// --- errors that say what to do (W13) ------------------------------------------------------
// The server's failures are typed — a status, and a JSON body — and each one has a different remedy, so
// they are explained, not pasted: a bad key, a full queue and a halted server used to read alike as a
// red box of JSON. problemFor turns one into {title, detail, actions}; the server's own message stays in
// the detail. Shapes, from internal/serveapp: writeErr's {"error":{"message","type"}}; the halt gate's
// {"error":"halted","reason"}; 429 carries Retry-After.
function serverMessage(text) {
  try {
    const j = JSON.parse(text);
    if (j && j.error && typeof j.error === "object" && typeof j.error.message === "string") return {message: j.error.message};
    if (j && j.error === "halted") return {halted: true, message: typeof j.reason === "string" ? j.reason : ""};
  } catch { /* not JSON */ }
  return {message: String(text || "").slice(0, 400)};
}

// problemFor describes a failed request. status 0 means no HTTP answer at all (the network failed).
function problemFor(status, bodyText, {model = "", retryAfter = ""} = {}) {
  const m = serverMessage(bodyText);
  const retry = "retry", key = "key", models = "models", fresh = "newchat";
  if (status === 0) return {title: "Can't reach the server.", detail: "The request got no answer. Check that goinfer serve is still running, then retry.", actions: [retry]};
  if (status === 401) return {title: "The server needs an API key.", detail: "Enter it in the Server API key field on the Models tab, then retry.", actions: [key, retry]};
  if (status === 404) return {title: "The model " + JSON.stringify(model) + " isn't loaded.", detail: m.message, actions: [models, retry]};
  if (status === 413) return {title: "This request is larger than the server accepts.", detail: m.message + (m.message ? " " : "") + "Remove an image, or start a new chat.", actions: [fresh]};
  if (status === 429) return {title: "The model is busy: its request queue is full.", detail: "Try again in a moment" + (retryAfter ? " (the server suggests " + retryAfter + " s)" : "") + ".", actions: [retry]};
  if (status === 503 && m.halted) return {title: "The server has halted new generations.", detail: (m.message ? "Reason: " + m.message + ". " : "") + "Someone with admin access has to resume it before anything can be generated.", actions: [retry]};
  if (status === 503) return {title: "The server is at capacity.", detail: (m.message ? m.message + ". " : "") + "Try again in a moment" + (retryAfter ? " (the server suggests " + retryAfter + " s)" : "") + ".", actions: [retry]};
  if (status === 400 && /context_length_exceeded/.test(bodyText)) {
    return {title: "This conversation no longer fits in " + model + "'s context window. Start a new chat, or delete earlier exchanges to make room.", detail: m.message, actions: [fresh]};   // W8
  }
  if (status >= 400 && status < 500) return {title: "The server rejected the request.", detail: m.message, actions: []};
  if (status >= 500) return {title: "The server hit an error.", detail: m.message, actions: [retry]};
  return {title: "The request failed (HTTP " + status + ").", detail: m.message, actions: []};
}

// showProblem renders a problem into a message bubble: title, the detail as text, and a button per remedy.
// retryFn is what Retry does; the bubble removes itself first, since it is not part of the conversation.
function showProblem(content, p, retryFn) {
  const msg = content.parentElement;
  msg.classList.add("err");
  const t = document.createElement("p"); t.className = "err-title"; t.textContent = p.title;
  const parts = [t];
  if (p.detail) { const d = document.createElement("p"); d.className = "err-detail"; d.textContent = p.detail; parts.push(d); }
  const labels = {retry: "Retry", key: "Enter API key", models: "Refresh models", newchat: "New chat"};
  if (p.actions.length) {
    const row = document.createElement("div"); row.className = "err-actions";
    for (const a of p.actions) {
      const b = document.createElement("button");
      b.type = "button"; b.className = "err-" + a; b.textContent = labels[a];
      b.onclick = () => {
        // the key field lives on the Models tab, hidden while chatting — show it, then focus it
        if (a === "key") { tab("models"); $("key").scrollIntoView({block: "center"}); $("key").focus(); return; }
        if (a === "models") { loadModels(); return; }
        if (a === "newchat") { msg.remove(); $("newchat").click(); return; }
        if (ac || editing) return;
        msg.remove();
        retryFn();
      };
      row.appendChild(b);
    }
    parts.push(row);
  }
  content.replaceChildren(...parts);
}

// A reply that was cut off keeps what arrived; this says why, and that Regenerate is the retry.
class StreamProblem extends Error {}

async function loadModels(pick) {
  try {
    let r;
    try { r = await fetch("/v1/models", {headers: headers()}); }
    catch { throw new Error("can't reach the server — is goinfer serve running?"); }
    if (r.status === 401) throw new Error("API key needed — enter it on the Models tab");   // W13
    if (!r.ok) throw new Error(problemFor(r.status, await r.text()).title);
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

// --- images (W11) -------------------------------------------------------------------------
// Attach one image to the next message — Attach button, drop, or paste — on a model whose /v1/models
// entry says vision: true; the control is hidden otherwise. The server decodes PNG and JPEG only, and a
// photo straight off a phone would fill browser storage in a few messages, so anything that is not
// already a small PNG/JPEG is drawn onto a canvas, scaled to IMAGE_MAX_SIDE on its longest side, and
// re-encoded as JPEG (transparency flattened onto white). Vision towers resize to their own input
// size anyway (SigLIP: 896 px), so this loses nothing the model would have used.
const IMAGE_MAX_SIDE = 1344, IMAGE_KEEP_BYTES = 1_500_000, IMAGE_MAX_INPUT = 25_000_000;
// A stored or pending image is rendered, so it must be exactly this: a PNG/JPEG base64 data URI. Never a
// remote URL (a stored conversation must not be able to make the page fetch anything) or a script scheme.
const IMAGE_OK = /^data:image\/(png|jpeg);base64,[A-Za-z0-9+/]+={0,2}$/;
const IMAGE_MAX_URI = 12_000_000;
const imageOK = u => typeof u === "string" && u.length <= IMAGE_MAX_URI && IMAGE_OK.test(u);
let pendingImage = null;   // {url, info}

const visionOK = () => models.find(m => m.id === $("model").value)?.vision === true;

function showAttach() {
  const ok = visionOK();
  $("attach").hidden = !ok;
  // The thumbnail exists only while an image is pending: the page carries no <img> the user did not add.
  const box = $("attach-preview");
  box.hidden = !pendingImage;
  box.querySelector("img")?.remove();
  if (pendingImage) {
    const img = document.createElement("img");
    img.alt = "Image to send"; img.src = pendingImage.url;
    box.prepend(img);
    $("attach-info").textContent = pendingImage.info;
  }
  const note = $("attach-note");
  let text = "";
  if (pendingImage && !ok) text = "The selected model can't see images — remove the image, or choose a model that can.";
  else if (!ok && transcript.some(m => m.image)) text = "This model can't see images, so the images in this conversation won't be sent to it.";
  else if (ok && pendingImage && transcript.some(m => m.image)) text = "The model sees one image per request: this one replaces the earlier image in what is sent.";
  note.textContent = text;
  note.hidden = !text;
}

function imageNote(text) {
  $("attach-note").textContent = text;
  $("attach-note").hidden = !text;
}

// loadImage decodes a File in an <img> (the browser's own decoder, so anything it can show works), then
// keeps it as-is or re-encodes it. Resolves to {url, info} or throws a message a user can act on.
function loadImage(file) {
  return new Promise((resolve, reject) => {
    if (!file || !/^image\//.test(file.type)) return reject(new Error("That isn't an image file."));
    if (file.size > IMAGE_MAX_INPUT) return reject(new Error("That image is too large (over 25 MB)."));
    const src = URL.createObjectURL(file);
    const img = new Image();
    img.onerror = () => { URL.revokeObjectURL(src); reject(new Error("That file couldn't be read as an image.")); };
    img.onload = () => {
      const w = img.naturalWidth, h = img.naturalHeight;
      const keep = (file.type === "image/png" || file.type === "image/jpeg") && Math.max(w, h) <= IMAGE_MAX_SIDE && file.size <= IMAGE_KEEP_BYTES;
      if (keep) {
        const fr = new FileReader();
        fr.onload = () => { URL.revokeObjectURL(src); imageOK(fr.result) ? resolve({url: fr.result, info: w + "×" + h}) : reject(new Error("That image couldn't be prepared.")); };
        fr.onerror = () => { URL.revokeObjectURL(src); reject(new Error("That file couldn't be read.")); };
        fr.readAsDataURL(file);
        return;
      }
      const scale = Math.min(1, IMAGE_MAX_SIDE / Math.max(w, h));
      const cw = Math.max(1, Math.round(w * scale)), ch = Math.max(1, Math.round(h * scale));
      const c = document.createElement("canvas");
      c.width = cw; c.height = ch;
      const g = c.getContext("2d");
      g.fillStyle = "#fff"; g.fillRect(0, 0, cw, ch);
      g.drawImage(img, 0, 0, cw, ch);
      URL.revokeObjectURL(src);
      const url = c.toDataURL("image/jpeg", 0.9);
      imageOK(url) ? resolve({url, info: cw + "×" + ch + (scale < 1 ? " (scaled from " + w + "×" + h + ")" : " (converted to JPEG)")}) : reject(new Error("That image couldn't be prepared."));
    };
    img.src = src;
  });
}

async function attachFile(file) {
  if (ac || editing) return;
  if (!visionOK()) { imageNote("The selected model can't see images."); return; }
  try {
    pendingImage = await loadImage(file);
    showAttach();
  } catch (e) {
    imageNote(String(e.message || e));
  }
}

$("attach").onclick = () => $("attach-file").click();
$("attach-file").addEventListener("change", () => { const f = $("attach-file").files[0]; $("attach-file").value = ""; if (f) attachFile(f); });
$("attach-remove").onclick = () => { pendingImage = null; showAttach(); };
$("prompt").addEventListener("paste", e => {
  const f = [...(e.clipboardData?.files || [])].find(f => f.type.startsWith("image/"));
  if (f) { e.preventDefault(); attachFile(f); }
});
// Drop anywhere on the chat pane. dragover must be cancelled for drop to fire at all.
$("pane-chat").addEventListener("dragover", e => { if ([...(e.dataTransfer?.types || [])].includes("Files")) e.preventDefault(); });
$("pane-chat").addEventListener("drop", e => {
  const f = [...(e.dataTransfer?.files || [])].find(f => f.type.startsWith("image/"));
  if (!f) return;
  e.preventDefault();
  attachFile(f);
});
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
  // W11: the server takes ONE image per request, so only the most recent image in the conversation is
  // sent, as an image_url part on the message it came with — and none at all to a model that cannot see.
  const list = transcript.filter(m => m !== generating);
  const lastImage = visionOK() ? list.findLastIndex(m => m.role === "user" && m.image) : -1;
  const msgs = list.map((m, i) => {
    if (m.role === "assistant") return {role: "assistant", content: answerOf(m.content)};
    if (i !== lastImage) return {role: "user", content: m.content};
    const parts = [];
    if (m.content) parts.push({type: "text", text: m.content});
    parts.push({type: "image_url", image_url: {url: m.image}});
    return {role: "user", content: parts};
  });
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

// --- persistence (W3) and conversations (W9) -------------------------------------------
// Every conversation survives a reload, in localStorage, ONE KEY PER CONVERSATION (CHAT_PREFIX + id):
// {v: 2, id, title, titled, updated, messages}. One key each, rather than one key for all, so the
// once-a-second save while a reply streams rewrites only the conversation being written, and so a
// conversation that cannot be read costs only itself. There is no separate index to fall out of step:
// the list is read from the keys. Every failure mode is non-fatal — storage blocked, quota exceeded,
// or a value this page cannot read (set aside under its key + ".unreadable", never destroyed).
//
// Which conversation a tab shows is per TAB (sessionStorage), so two tabs can have different chats
// open; a new tab opens the most recently updated one. W3's single conversation (LEGACY_STORE) is
// migrated into a conversation of its own, once, and opened.
const CHAT_PREFIX = "goinfer.chat.v2.";
const LEGACY_STORE = "goinfer.chat.v1";
const CURRENT_CHAT = "goinfer.chat.current";
const CHAT_ID_OK = /^[a-z0-9]{1,32}$/;
// titled says where the title came from: "first" = provisional, from the first message, model not yet
// asked; "tried" = provisional, and the model's attempt gave nothing usable; "model"; "user" (renamed).
const TITLED = new Set(["first", "tried", "model", "user"]);

const newChatId = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
const freshChat = () => ({id: newChatId(), title: "", titled: "", updated: 0, stored: false});
let currentChat = freshChat();

// provisionalTitle is the instant title: the first message, one line, cut to fit the list.
function provisionalTitle(text) {
  const t = text.replace(/\s+/g, " ").trim();
  return t.length > 48 ? t.slice(0, 47) + "…" : t;
}

function storeFailed() {
  $("store-note").textContent = "This conversation can't be kept across a reload (browser storage is full or blocked) — " +
    "it stays on screen until you close or reload the page.";
  $("store-note").hidden = false;
}

function rememberCurrent() {
  try { sessionStorage.setItem(CURRENT_CHAT, currentChat.id); } catch { /* blocked */ }
}

// save writes the current conversation. An empty new chat is not written — it is not a conversation
// until something is said in it. bump=false keeps its place in the list (a rename is not activity).
function save(bump = true) {
  if (!transcript.length && !currentChat.stored) return;
  if (!currentChat.title) {
    const first = transcript.find(m => m.role === "user");
    if (first) { currentChat.title = provisionalTitle(first.content); currentChat.titled = "first"; }
  }
  if (bump || !currentChat.updated) currentChat.updated = Date.now();
  const c = currentChat;
  try {
    localStorage.setItem(CHAT_PREFIX + c.id, JSON.stringify({v: 2, id: c.id, title: c.title, titled: c.titled, updated: c.updated, messages: transcript}));
    $("store-note").hidden = true;
    if (!c.stored) { c.stored = true; renderChatList(); }
  } catch {
    storeFailed();
  }
  rememberCurrent();
}

// parseMessages validates stored messages field by field: this is data, and it is rendered.
// hadGenerating reports a reply that was streaming when the page went away.
function parseMessages(list) {
  const out = [];
  let hadGenerating = false;
  for (const m of list) {
    if (!m || (m.role !== "user" && m.role !== "assistant") || typeof m.content !== "string") continue;
    const e = { role: m.role, content: m.content };
    if (m.role === "user" && imageOK(m.image)) e.image = m.image;   // W11: anything else is dropped, not rendered
    if (typeof m.model === "string") e.model = m.model;
    if (typeof m.meta === "string") e.meta = m.meta;
    if (m.state === "stopped" || m.state === "interrupted" || m.state === "failed" || m.state === "cancelled") e.state = m.state;
    if (typeof m.note === "string" && m.note.length <= 500) e.note = m.note;   // W13
    if (m.state === "generating") { e.state = "interrupted"; hadGenerating = true; }   // it was streaming when the page went away
    if (typeof m.thought === "number" && Number.isFinite(m.thought) && m.thought >= 0) e.thought = m.thought;   // W6
    const u = m.usage, tok = x => Number.isSafeInteger(x) && x >= 0;
    if (u && tok(u.prompt_tokens) && tok(u.completion_tokens)) e.usage = {prompt_tokens: u.prompt_tokens, completion_tokens: u.completion_tokens};   // W8
    out.push(e);
  }
  return {messages: out, hadGenerating};
}

// setAside moves an unreadable value out of the live key, keeping it.
function setAside(key, raw) {
  try { localStorage.setItem(key + ".unreadable", raw); localStorage.removeItem(key); } catch { /* leave it */ }
}

// readChat returns a stored conversation, or null when there is none (or it could not be read, in
// which case it has been set aside). withMessages=false skips message validation, for the list.
function readChat(id, withMessages = true) {
  if (!CHAT_ID_OK.test(id)) return null;
  const key = CHAT_PREFIX + id;
  let raw;
  try { raw = localStorage.getItem(key); } catch { return null; }
  if (!raw) return null;
  let d = null;
  try { d = JSON.parse(raw); } catch { /* unreadable */ }
  if (!d || d.v !== 2 || d.id !== id || !Array.isArray(d.messages)) { setAside(key, raw); return null; }
  const c = {
    id, stored: true,
    title: typeof d.title === "string" ? d.title.slice(0, 200) : "",
    titled: TITLED.has(d.titled) ? d.titled : "user",
    updated: Number.isFinite(d.updated) ? d.updated : 0,
  };
  if (withMessages) Object.assign(c, parseMessages(d.messages));
  return c;
}

// listChats reads every stored conversation's metadata, most recently updated first.
function listChats() {
  const ids = [];
  try {
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (k && k.startsWith(CHAT_PREFIX) && CHAT_ID_OK.test(k.slice(CHAT_PREFIX.length))) ids.push(k.slice(CHAT_PREFIX.length));
    }
  } catch { return []; }
  return ids.map(id => readChat(id, false)).filter(Boolean).sort((a, b) => b.updated - a.updated);
}

// migrateLegacy turns W3's single conversation into a W9 conversation, once. Returns its id, or null.
function migrateLegacy() {
  let raw;
  try { raw = localStorage.getItem(LEGACY_STORE); } catch { return null; }
  if (!raw) return null;
  let d = null;
  try { d = JSON.parse(raw); } catch { /* unreadable */ }
  if (!d || d.v !== 1 || !Array.isArray(d.messages)) { setAside(LEGACY_STORE, raw); return null; }
  const {messages} = parseMessages(d.messages);
  const c = freshChat();
  const first = messages.find(m => m.role === "user");
  const value = {v: 2, id: c.id, title: first ? provisionalTitle(first.content) : "", titled: "first", updated: Date.now(), messages};
  try {
    if (messages.length) localStorage.setItem(CHAT_PREFIX + c.id, JSON.stringify(value));
    localStorage.removeItem(LEGACY_STORE);
  } catch { return null; }   // quota: leave the legacy value where it is, try again next load
  return messages.length ? c.id : null;
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
// --- sampling controls (W10) ---------------------------------------------------------
// Every sampling field /v1/chat/completions accepts, as a SETTING like the system prompt: kept in its own
// key, across reloads and new chats, followed across tabs. A blank optional field is not sent, so the
// server's default applies (a blank seed means a fresh random one — M-03). Ranges are OpenAI's documented
// ones; the server rejects some of them itself (temperature < 0, top_p outside [0,1]) and accepts the
// rest unchecked, so the page checks them all and REFUSES to send an out-of-range value rather than
// clamping it into one the user did not choose.
const SAMPLING_STORE = "goinfer.sampling.v1";
function rangeField(lo, hi, integer, label) {
  return t => {
    const v = Number(t);
    if (t.trim() === "" || !Number.isFinite(v) || (integer && !Number.isInteger(v)) || v < lo || v > hi) {
      return {error: label + " must be " + (integer ? "a whole number" : "a number") + " from " + lo + " to " + hi};
    }
    return {value: v};
  };
}
// stopField reads one stop sequence per line; "\n" and "\t" stand for a newline and a tab.
function stopField(t) {
  const list = t.split("\n").map(l => l.replace(/\r$/, "")).filter(l => l !== "")
    .map(l => l.replace(/\\(\\|n|t)/g, (_, c) => c === "n" ? "\n" : c === "t" ? "\t" : "\\"));
  return list.length ? {value: list} : {};
}
const SAMPLING = [
  {id: "temp", key: "temperature", def: "0.7", required: true, parse: rangeField(0, 2, false, "Temperature")},
  {id: "max", key: "max_tokens", def: "512", required: true, parse: rangeField(1, 131072, true, "Max tokens")},
  {id: "top-p", key: "top_p", def: "", parse: rangeField(0, 1, false, "top_p")},
  {id: "top-k", key: "top_k", def: "", parse: rangeField(0, 1000000, true, "top_k")},
  {id: "seed", key: "seed", def: "", parse: rangeField(-Number.MAX_SAFE_INTEGER, Number.MAX_SAFE_INTEGER, true, "Seed")},
  {id: "freq-pen", key: "frequency_penalty", def: "", parse: rangeField(-2, 2, false, "Frequency penalty")},
  {id: "pres-pen", key: "presence_penalty", def: "", parse: rangeField(-2, 2, false, "Presence penalty")},
  {id: "stop", key: "stop", def: "", parse: stopField},
];

// readSampling returns {params} to merge into a request, or {error, field} for the first bad field. It also
// marks every field valid or invalid, so the reason is visible where it is.
function readSampling() {
  const params = {};
  let first = null;
  for (const f of SAMPLING) {
    const el = $(f.id), t = el.value;
    const r = (t.trim() === "" && !f.required) ? {} : f.parse(t);
    el.setAttribute("aria-invalid", String(!!r.error));
    el.classList.toggle("invalid", !!r.error);
    if (r.error && !first) first = {error: r.error, field: el};
    if (r.value !== undefined) params[f.key] = r.value;
  }
  $("sampling-error").hidden = !first;
  $("sampling-error").textContent = first ? first.error : "";
  return first || {params};
}

// samplingOK is the guard every action that asks for a reply runs FIRST, before it changes the
// conversation: a regenerate that removed the old reply and then could not send would lose it.
function samplingOK() {
  const r = readSampling();
  if (!r.error) return true;
  if (!r.field.closest("details") || r.field.closest("details").open === false) $("sampling-box").open = true;
  r.field.focus();
  $("chat-status").textContent = "Fix the settings first: " + r.error + ".";
  return false;
}

function showSamplingState() {
  const custom = SAMPLING.filter(f => $(f.id).value !== f.def).length;
  $("sampling-state").textContent = custom ? "· " + custom + " changed" : "";
  readSampling();
}
function saveSampling() {
  const v = {};
  for (const f of SAMPLING) v[f.id] = $(f.id).value;
  try {
    if (SAMPLING.every(f => v[f.id] === f.def)) localStorage.removeItem(SAMPLING_STORE);
    else localStorage.setItem(SAMPLING_STORE, JSON.stringify({v: 1, fields: v}));
  } catch {
    storeFailed();
  }
}
function loadSampling() {
  let stored = null;
  try { stored = JSON.parse(localStorage.getItem(SAMPLING_STORE) || "null"); } catch { /* unreadable: defaults */ }
  const fields = stored && stored.v === 1 && stored.fields && typeof stored.fields === "object" ? stored.fields : {};
  for (const f of SAMPLING) {
    const t = fields[f.id];
    $(f.id).value = typeof t === "string" && t.length <= 4000 ? t : f.def;   // data, and shown: strings only
  }
  showSamplingState();
}

function loadSystem() {
  try { $("system").value = localStorage.getItem(SYSTEM_STORE) || ""; } catch { /* blocked */ }
  showSystemState();
}

// fillUserBubble shows a message the user sent: its text as text, and its image (W11), if any.
function fillUserBubble(b, e) {
  b.textContent = e.content;
  if (e.image) {
    const img = document.createElement("img");
    img.className = "uimg";
    img.alt = "Attached image";
    img.src = e.image;   // imageOK-validated: a PNG/JPEG data URI, never a URL that fetches
    b.appendChild(img);
  }
}

function metaText(e) {
  if (e.state === "interrupted") return (e.meta ? e.meta + " · " : "") + "interrupted — the page was reloaded while this was generating";
  if (e.state === "stopped") return "stopped" + (e.meta ? " · " + e.meta : "");
  // W13: why a reply is incomplete, and what to do — kept with the reply, not in a status line
  if (e.state === "failed") return (e.meta ? e.meta + " · " : "") + "incomplete — " + (e.note || "the reply did not finish") + ". Regenerate to try again.";
  if (e.state === "cancelled") return (e.meta ? e.meta + " · " : "") + "cancelled by the server" + (e.note ? " (" + e.note + ")" : "") + ".";
  return (e.meta || "") + (e.note ? " · " + e.note : "");
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
    fillUserBubble(b, e);
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
  showAttach();
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
  for (const b of $("chat-list").querySelectorAll("button")) b.disabled = busy;   // W9
  showExport();   // W14
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
  if (!samplingOK()) return;   // W10: before the reply is removed
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
  if (!text && !entry.image) return;   // an empty message is not a message (unless it carries an image, W11)
  const i = transcript.indexOf(entry);
  const later = transcript.length - i - 1;
  if (!samplingOK()) return;   // W10: before anything is dropped
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
  if ((!text && !pendingImage) || ac || editing) return;
  if (!$("model").value) { $("chat-status").textContent = "no model loaded"; return; }
  if (!samplingOK()) return;   // W10: before the message is added, so a bad setting loses nothing
  if (pendingImage && !visionOK()) { showAttach(); $("chat-status").textContent = "The selected model can't see images."; return; }   // W11

  // What you type is recessed (amb-surface-concave, decision 3); the echo of
  // it in the log is a quieter flat surface — neither is the "product".
  const mine = bubble("you", "you amb-surface");
  const u = {role: "user", content: text};
  if (pendingImage) u.image = pendingImage.url;
  fillUserBubble(mine, u);
  transcript.push(u);
  addActions(mine, text, u);
  $("prompt").value = "";
  pendingImage = null;
  showAttach();
  await generate();
}

// generate asks for a reply to the conversation as it stands: after send(), and after W7's Regenerate
// and Edit, which change the transcript first.
async function generate() {
  if (ac || editing) return;
  if (titleAC) titleAC.abort();   // W9: a title is never worth making someone wait for their reply
  const model = $("model").value;
  if (!model) { $("chat-status").textContent = "no model loaded"; return; }
  if (!samplingOK()) return;
  const {params: sampling} = readSampling();
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
  let got = 0, acc = "", lastSave = 0, problem = null, finish = "";
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
    let r;
    try {
     r = await fetch("/v1/chat/completions", {
      method: "POST", headers: headers(), signal: ac.signal,
      body: JSON.stringify({
        model, messages, stream: true,
        stream_options: {include_usage: true},   // W8: real token counts, for the meter and the stats
        ...sampling,                              // W10: temperature, max_tokens and whatever else is set
      }),
     });
    } catch (e) {
      if (e.name === "AbortError") throw e;
      problem = problemFor(0, "");   // no HTTP answer at all
      throw e;
    }
    if (!r.ok) {
      problem = problemFor(r.status, await r.text(), {model, retryAfter: r.headers.get("Retry-After") || ""});   // W13 (and W8's wall)
      throw new Error(problem.title);
    }
    for await (const ev of sse(r)) {
      if (ev.data === "[DONE]") break;
      let j; try { j = JSON.parse(ev.data); } catch { continue; }
      // W13: the server reports a failure DURING a stream as an error object, not an HTTP status
      if (j.error) throw new StreamProblem("the server reported an error: " + (typeof j.error === "object" && typeof j.error.message === "string" ? j.error.message : String(j.error)));
      const fr = j.choices && j.choices[0] && j.choices[0].finish_reason;
      if (typeof fr === "string" && fr) finish = fr;
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
    delete entry.note;
    // W13: how the reply ended, when that is something to act on
    if (finish === "length") entry.note = "stopped at the Max tokens limit — raise it to get more";
    if (finish === "cancelled") entry.state = "cancelled";
    renderReply(out, acc, false, entry);   // the entry is final now: "no answer" can be judged
    addMeta(out, metaText(entry));
    addActions(out, copyOf(acc), entry);
    showContext();
    if (currentChat.titled === "first") {
      const first = transcript.find(m => m.role === "user");
      if (first) queueMicrotask(() => autoTitle(currentChat.id, first.content, model));
    }
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
    } else if (acc) {
      // W13: cut off part-way — by a server error or a dropped connection. What arrived is kept (it was
      // being saved as it streamed anyway), marked incomplete, and Regenerate is the retry.
      live = false;
      entry.content = acc;
      entry.state = "failed";
      entry.note = e instanceof StreamProblem ? e.message : "the connection to the server was lost";
      if (got) entry.meta = got + " tok";
      paint();
      addMeta(out, metaText(entry));
      addActions(out, copyOf(acc), entry);
      $("chat-status").textContent = "";
    } else {
      // A request that produced nothing is not a turn: drop it, and say what happened and what to do.
      transcript.splice(transcript.indexOf(entry), 1);
      const p = problem || (e instanceof StreamProblem ? {title: "The server hit an error.", detail: e.message, actions: ["retry"]} : problemFor(0, ""));
      showProblem(out, p, () => generate());
      $("chat-status").textContent = "";
    }
  } finally {
    generating = null;
    save();
    ac = null; $("send").disabled = false; $("stop").hidden = true; $("newchat").disabled = false;
    syncActions();
    renderChatList();   // W9: this conversation moves to the top, and the list is usable again
  }
}
$("send").onclick = send;

// --- the conversation list (W9) ------------------------------------------------------------
function openChat(id) {
  if (ac || editing) return;
  const c = readChat(id);
  if (!c) { renderChatList(); return; }
  currentChat = {id: c.id, title: c.title, titled: c.titled, updated: c.updated, stored: true};
  rememberCurrent();
  showTranscript(c.messages);
  $("store-note").hidden = true;
  $("chat-status").textContent = "";
  // A reply that was streaming when the page went away is now "interrupted"; write that back, so the
  // state is recorded rather than re-derived on every load.
  if (c.hadGenerating) save(false);
  renderChatList();
}

function startFresh() {
  currentChat = freshChat();
  rememberCurrent();
  showTranscript([]);
  $("store-note").hidden = true;
  $("chat-status").textContent = "";
  renderChatList();
}

// New chat starts another conversation; the one you were in stays in the list. Nothing is lost, so it
// does not ask (W3's New chat cleared the only conversation, and had to). On an empty new chat it
// does nothing but put you in the message box.
$("newchat").onclick = () => {
  if (ac) return;
  if (editing) finishEdit(null);
  if (!transcript.length && !currentChat.stored) { $("prompt").focus(); return; }
  startFresh();
};

function deleteChat(id) {
  if (ac || editing) return;
  if (!confirm("Delete this conversation? This cannot be undone.")) return;
  try { localStorage.removeItem(CHAT_PREFIX + id); } catch { /* blocked */ }
  if (id !== currentChat.id) { renderChatList(); return; }
  const next = listChats()[0];
  if (next) openChat(next.id); else startFresh();
}

// setTitle changes a conversation's title without moving it in the list. Returns false when the
// conversation no longer exists.
function setTitle(id, title, titled) {
  if (id === currentChat.id) {
    currentChat.title = title; currentChat.titled = titled;
    if (currentChat.stored) save(false);
    renderChatList();
    return true;
  }
  let d;
  try { d = JSON.parse(localStorage.getItem(CHAT_PREFIX + id)); } catch { return false; }
  if (!d || d.v !== 2 || d.id !== id) return false;
  d.title = title; d.titled = titled;
  try { localStorage.setItem(CHAT_PREFIX + id, JSON.stringify(d)); } catch { storeFailed(); return false; }
  renderChatList();
  return true;
}

let renaming = null;   // id of the conversation whose title is being edited in the list
function startRename(id, item) {
  if (ac || editing || renaming) return;
  const c = id === currentChat.id ? currentChat : readChat(id, false);
  if (!c) return;
  renaming = id;
  const input = document.createElement("input");
  input.type = "text"; input.className = "chat-rename-box"; input.maxLength = 200;
  input.value = c.title; input.setAttribute("aria-label", "Conversation title");
  item.replaceChildren(input);
  input.focus(); input.select();
  let done = false;
  const finish = keep => {
    if (done) return;
    done = true; renaming = null;
    const t = input.value.replace(/\s+/g, " ").trim();
    if (keep && t) setTitle(id, t, "user"); else renderChatList();
  };
  input.addEventListener("keydown", e => {
    if (e.key === "Enter") { e.preventDefault(); finish(true); }
    if (e.key === "Escape") { e.preventDefault(); finish(false); }
  });
  input.addEventListener("blur", () => finish(true));
}

function renderChatList() {
  if (renaming) return;   // do not pull the box out from under someone typing a title
  const ul = $("chat-list");
  const chats = listChats();
  if (!currentChat.stored) chats.unshift({id: currentChat.id, title: "", pending: true});
  const busy = !!ac || !!editing;
  const items = chats.map(c => {
    const li = document.createElement("li");
    li.className = "chat-item" + (c.id === currentChat.id ? " current" : "");
    li.dataset.id = c.id;
    const open = document.createElement("button");
    open.type = "button"; open.className = "chat-open"; open.disabled = busy;
    open.textContent = c.pending ? "New chat" : (c.title || "Untitled");
    if (c.id === currentChat.id) open.setAttribute("aria-current", "true");
    li.appendChild(open);
    if (!c.pending) {
      for (const [cls, label, aria] of [["chat-rename", "Rename", "Rename conversation"], ["chat-delete", "Delete", "Delete conversation"]]) {
        const b = document.createElement("button");
        b.type = "button"; b.className = cls; b.textContent = label; b.disabled = busy;
        b.setAttribute("aria-label", aria);
        li.appendChild(b);
      }
    }
    return li;
  });
  ul.replaceChildren(...items);
  showExport();   // W14
}

$("chat-list").addEventListener("click", e => {
  const btn = e.target.closest("button");
  const li = btn && btn.closest("li.chat-item");
  if (!btn || !li || btn.disabled) return;
  const id = li.dataset.id;
  if (btn.classList.contains("chat-open")) { if (id !== currentChat.id) openChat(id); }
  else if (btn.classList.contains("chat-rename")) startRename(id, li);
  else if (btn.classList.contains("chat-delete")) deleteChat(id);
});

// --- export (W14) -----------------------------------------------------------------------------
// The open conversation, to a file: Markdown to read, JSON to keep or process. No share link — that would
// be a server-side copy, and an anti-goal. Both are built from the transcript, the same source the
// screen and storage use. The system prompt is a page SETTING, not recorded with messages (W4), so it is
// exported labelled as what was set when the file was made.
const EXPORT_FORMAT = "goinfer.chat";

// exportName turns a title into a safe file name: letters, digits and dashes, never a path.
function exportName(title, ext, when = new Date()) {
  const slug = (title || "").normalize("NFKD").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 60).replace(/-+$/, "");
  const day = when.toISOString().slice(0, 10);
  return (slug || "conversation") + "-" + day + "." + ext;
}

function chatMarkdown(when = new Date()) {
  const lines = ["# " + (currentChat.title || "Conversation"), ""];
  lines.push("*Exported from goinfer on " + when.toISOString().slice(0, 16).replace("T", " ") + " UTC · " + transcript.length + " message" + (transcript.length === 1 ? "" : "s") + "*", "");
  const sys = systemText();
  if (sys) lines.push("> **System prompt at export time** (a setting of the page, not recorded per message):", ">", ...sys.split("\n").map(l => "> " + l), "");
  for (const e of transcript) {
    if (e.role === "user") {
      lines.push("## You", "");
      if (e.content) lines.push(e.content, "");
      if (e.image) lines.push("![attached image](" + e.image + ")", "");
      continue;
    }
    lines.push("## " + (e.model || "Assistant"), "");
    const p = splitThinking(e.content, false);
    if (p && p.thinking) lines.push("<details><summary>Thinking</summary>", "", p.thinking.trim(), "", "</details>", "");
    const answer = p ? p.answer : e.content;
    if (answer) lines.push(answer, "");
    const meta = metaText(e);
    if (meta) lines.push("*" + meta + "*", "");
  }
  return lines.join("\n");
}

function chatJSON(when = new Date()) {
  return JSON.stringify({
    format: EXPORT_FORMAT, version: 1, exported_at: when.toISOString(),
    title: currentChat.title || "", system_prompt_at_export: systemText() || null,
    messages: transcript,
  }, null, 2) + "\n";
}

function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], {type}));
  const a = document.createElement("a");
  a.href = url; a.download = name; a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

function showExport() {
  const empty = !transcript.length;
  $("export-md").disabled = empty || !!ac;
  $("export-json").disabled = empty || !!ac;
}
$("export-md").onclick = () => { if (transcript.length && !ac) download(exportName(currentChat.title, "md"), chatMarkdown(), "text/markdown;charset=utf-8"); };
$("export-json").onclick = () => { if (transcript.length && !ac) download(exportName(currentChat.title, "json"), chatJSON(), "application/json"); };

// --- generated titles (W9) --------------------------------------------------------------------
// After a conversation's first reply, the model is asked once, in the background, for a short title.
// Until then — and for good, if the model gives nothing usable (a reasoning model can spend its whole
// budget thinking) — the title is the first message. The request is cancelled the moment a reply is
// asked for, so it never holds the model while you are waiting; a rename always wins over it.
const TITLE_PROMPT = "Write a short title, at most six words, for a conversation that begins with the message below. " +
  "Reply with the title only — no quotes, no punctuation at the end.\n\n";
let titleAC = null;

// cleanTitle turns a model's reply into a title, or "" when it is not one.
function cleanTitle(text) {
  const line = (answerOf(text || "").split("\n").map(l => l.trim()).find(Boolean) || "");
  if (/<\||<\/?think/.test(line)) return "";
  // strip wrapping markup and a final period until nothing changes: "**Title**." has the period outside
  let t = line.replace(/^(title\s*:\s*)/i, ""), prev;
  do { prev = t; t = t.replace(/^[#>*_`"'“”‘’\s]+|[*_`"'“”‘’\s.。]+$/g, ""); } while (t !== prev);
  t = t.replace(/\s+/g, " ").trim();
  if (t.length > 60) t = t.slice(0, 59) + "…";
  return t;
}

async function autoTitle(id, first, model) {
  const mine = new AbortController();
  titleAC = mine;
  let settled = false;
  try {
    const r = await fetch("/v1/chat/completions", {
      method: "POST", headers: headers(), signal: mine.signal,
      body: JSON.stringify({model, stream: false, temperature: 0, max_tokens: 24,
        messages: [{role: "user", content: TITLE_PROMPT + first.slice(0, 1000)}]}),
    });
    const j = r.ok ? await r.json() : null;
    const t = cleanTitle(j?.choices?.[0]?.message?.content);
    settled = true;
    const c = id === currentChat.id ? currentChat : readChat(id, false);
    if (!c || c.titled !== "first") return;   // deleted, or renamed while we waited
    setTitle(id, t || c.title, t ? "model" : "tried");
  } catch {
    // cancelled by a new reply (it will be asked again after that one), or the request failed
    if (!mine.signal.aborted && !settled) {
      const c = id === currentChat.id ? currentChat : readChat(id, false);
      if (c && c.titled === "first") setTitle(id, c.title, "tried");
    }
  } finally {
    if (titleAC === mine) titleAC = null;
  }
}

// A reload mid-stream: save what has arrived. pagehide fires where beforeunload is unreliable (mobile).
addEventListener("pagehide", () => { if (generating) save(); });

// Another tab changed a conversation or the system prompt. The list always follows. The conversation
// on screen follows too — unless this tab is mid-reply (its own save wins then) or editing a message;
// for the system prompt, unless the user is typing in that box right now.
addEventListener("storage", e => {
  if ((e.key === SYSTEM_STORE || e.key === null) && document.activeElement !== $("system")) loadSystem();
  // W10: likewise, unless the user is in one of the sampling fields right now
  if ((e.key === SAMPLING_STORE || e.key === null) && !SAMPLING.some(f => document.activeElement === $(f.id))) loadSampling();
  if (e.key === null || (e.key && e.key.startsWith(CHAT_PREFIX))) {
    renderChatList();
    if ((e.key === null || e.key === CHAT_PREFIX + currentChat.id) && !ac && !editing) {
      const c = readChat(currentChat.id);
      if (c) {
        Object.assign(currentChat, {title: c.title, titled: c.titled, updated: c.updated, stored: true});
        showTranscript(c.messages);
      }
    }
  }
});

$("system").addEventListener("input", () => { showSystemState(); saveSystem(); });
for (const f of SAMPLING) $(f.id).addEventListener("input", () => { showSamplingState(); saveSampling(); });
$("sampling-reset").onclick = () => {
  for (const f of SAMPLING) $(f.id).value = f.def;
  showSamplingState(); saveSampling();
  $("chat-status").textContent = "";
};

loadSystem();
loadSampling();

// Open a conversation: the one migrated from W3's single store (once), else the one this tab had open,
// else the most recently updated, else a new one.
(() => {
  const migrated = migrateLegacy();
  let id = migrated;
  if (!id) { try { id = sessionStorage.getItem(CURRENT_CHAT); } catch { /* blocked */ } }
  if (!(id && readChat(id, false))) id = listChats()[0]?.id;
  if (id) openChat(id); else startFresh();
})();
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
