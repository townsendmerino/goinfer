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
async function loadModels() {
  try {
    const r = await fetch("/v1/models", {headers: headers()});
    if (!r.ok) throw new Error("HTTP " + r.status);
    const j = await r.json();
    const sel = $("model");
    sel.textContent = "";
    for (const m of (j.data || [])) {
      const o = document.createElement("option");
      o.value = m.id; o.textContent = m.id;
      sel.appendChild(o);
    }
    const n = (j.data || []).length;
    const stats = $("stats");
    stats.textContent = "";
    if (!n) {
      stats.textContent = "no model loaded";
    } else {
      const add = (label, val) => {
        const s = document.createElement("span");
        const b = document.createElement("b"); b.textContent = val;
        s.appendChild(document.createTextNode(label + " ")); s.appendChild(b);
        stats.appendChild(s);
      };
      add("model", j.data[0].id);
      if (j.data[0].decode_path) add("path", j.data[0].decode_path);
    }
  } catch (e) {
    $("stats").textContent = e.message;
  }
}
loadModels();

// --- chat -------------------------------------------------------------------
const history = [];
let ac = null;

function bubble(who, cls) {
  const d = document.createElement("div");
  d.className = "msg ambient amb-rounded " + cls;
  const h = document.createElement("div");
  h.className = "who"; h.textContent = who;
  const b = document.createElement("div");
  d.appendChild(h); d.appendChild(b);
  $("log").appendChild(d);
  d.scrollIntoView({block: "end"});
  return b;                       // textContent only — never innerHTML
}

async function send() {
  const text = $("prompt").value.trim();
  if (!text || ac) return;
  const model = $("model").value;
  if (!model) { $("chat-status").textContent = "no model loaded"; return; }

  // What you type is recessed (amb-surface-concave, decision 3); the echo of
  // it in the log is a quieter flat surface — neither is the "product".
  bubble("you", "you amb-surface").textContent = text;
  history.push({role: "user", content: text});
  $("prompt").value = "";
  // What the engine produces sits proud (amb-surface-convex, decision 3).
  const out = bubble(model, "bot amb-surface-convex amb-elevation-1");
  $("send").disabled = true; $("stop").hidden = false;
  $("chat-status").textContent = "generating…";

  ac = new AbortController();
  const started = performance.now();
  let got = 0, acc = "";
  try {
    const r = await fetch("/v1/chat/completions", {
      method: "POST", headers: headers(), signal: ac.signal,
      body: JSON.stringify({
        model, messages: history, stream: true,
        temperature: parseFloat($("temp").value) || 0,
        max_tokens: parseInt($("max").value, 10) || 512,
      }),
    });
    if (!r.ok) throw new Error((await r.text()).slice(0, 400) || ("HTTP " + r.status));
    for await (const ev of sse(r)) {
      if (ev.data === "[DONE]") break;
      let j; try { j = JSON.parse(ev.data); } catch { continue; }
      const d = j.choices && j.choices[0] && j.choices[0].delta;
      if (d && d.content) { acc += d.content; out.textContent = acc; got++; out.scrollIntoView({block: "end"}); }
    }
    history.push({role: "assistant", content: acc});
    const s = (performance.now() - started) / 1000;
    // Per-response stats live WITH the response, not in a status line that
    // the next turn overwrites (decision 4).
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = got
      ? got + " tok · " + (got / s).toFixed(1) + " tok/s · " + s.toFixed(1) + "s"
      : "no output";
    out.parentElement.appendChild(meta);
    $("chat-status").textContent = "";
  } catch (e) {
    if (e.name === "AbortError") { $("chat-status").textContent = "stopped"; history.push({role: "assistant", content: acc}); }
    else { out.textContent = String(e.message || e); out.parentElement.classList.add("err"); $("chat-status").textContent = ""; }
  } finally {
    ac = null; $("send").disabled = false; $("stop").hidden = true;
  }
}
$("send").onclick = send;
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
        // The file is on disk, but this server has not loaded it — serving it is a separate,
        // deliberately-gated action. Say so rather than letting the green tick imply the model
        // is now live.
        const hint = document.createElement("div");
        hint.className = "note"; hint.style.marginTop = "6px";
        hint.textContent = "Downloaded, not loaded — restart the server with --model <path> to serve it.";
        st.appendChild(hint);
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
