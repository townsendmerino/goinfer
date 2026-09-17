// webui_app_gate.mjs — end-to-end gate for the web UI page itself (docs/tasks/task-web-ui-2026-09.md).
//
//   node scripts/webui_app_gate.mjs        # exit 0 pass / 1 a check failed / 2 could not drive a browser
//
// Loads the SHIPPED internal/serveapp/webui/index.html from file:// in headless Chrome and drives the
// page through its own code: fetch is replaced with a fake that streams SSE, and the gate calls the
// page's real send(). Nothing here reimplements page logic. Each W-item that ships adds its checks.
//
//   W1 — model output streams through Markdown.render; a hostile <img onerror> stays inert.
//   W2 — Copy on code blocks and messages: exact text (raw Markdown for messages), both clipboard
//        paths (async API; textarea fallback when the API rejects or does not exist, as on
//        http://<lan-ip>), failure feedback, feedback reset, copy on a STOPPED answer, and delegation
//        surviving a re-render that rebuilds every code-block button.
//   W3 — the conversation survives a reload: what is saved, what is restored (rendered the same way,
//        Copy still copying the source), a reload MID-STREAM keeping the partial answer as
//        "interrupted", unreadable storage set aside rather than destroyed, quota failure shown and
//        survived, New chat (confirmed and cancelled), another tab's change followed, and hostile
//        stored content rendered inert. Phases are separated by real page reloads.
//   W4 — the system prompt: sent first only when non-blank and trimmed, never written into the saved
//        transcript, kept across New chat and a reload, a hostile prompt inert, and another tab's change
//        followed without clobbering a box the user is typing in.
//   W5 — load from the page: a finished pull offers "Load it now", which posts exactly the pulled path,
//        shows the heartbeat, then selects the loaded model (header stats following) so the next chat
//        goes to it. Errors, a load already running, and a stream cut off mid-load are shown and leave
//        the button usable; "already loaded" selects instead of failing; hostile paths and errors inert.
//   W6 — thinking folded above the answer, for the Qwen3 <think> shape (and its template-opened variant)
//        and gpt-oss's channels: no raw tag ever on screen while streaming, a reader's open fold kept
//        open, Copy and the next request carry only the answer, a thought-but-never-answered reply says
//        so, a mention of the tag is not folded, hostile content inert, and the fold rebuilt after reload.
//   W7 — Regenerate (last reply only, including a stopped one), Edit (in place; Escape cancels, Ctrl+Enter
//        saves; asks before dropping more than the one reply; resends from the edited message) and Delete
//        (the whole exchange, after asking): exactly what is sent, shown and saved after each; all of them
//        disabled while generating or editing; hostile edits inert; and the actions restored after reload.
//   W8 — the context meter against /v1/models' context_window: usage requested and used (stats count the
//        server's completion_tokens), warn at 80%, "will not fit" at 95%, capped bar, re-measured on delete
//        and model switch, hidden with no window; context_length_exceeded explained (and only it);
//        usage saved, restored, and unreadable stored usage ignored.
//   W9 — separate conversations: an empty chat is not stored; provisional title from the first message,
//        then ONE background title request (non-streaming, cleaned, hostile-inert, cancelled by a reply and
//        re-asked, never retried once unusable, never overriding a rename); most-recent-first list; open,
//        busy-disabled, rename (Escape/Enter/empty), delete (asks; current falls to the next); another
//        tab's conversation listed; unreadable/mislabelled keys set aside; per-tab reopen after reload;
//        and the W3-era single conversation migrated once.
//   W10 — sampling controls: exactly what is sent (and not sent) for defaults, set and cleared fields; stop
//        sequences parsed; the title request unaffected; each out-of-range value marked, explained, and
//        refused by Send, Regenerate and Edit before anything changes; saved, followed across tabs
//        (not over a field being typed in), reset, restored after reload, unreadable values ignored.
//   W11 — images: Attach only on vision models; picker, paste and drop; small PNG kept, WebP converted and a
//        large image scaled (checked by decoding the result); exactly the content parts sent, only the
//        newest image, none to a text-only model; non-images and corrupt files refused; image-only send and
//        edit; nothing attachable mid-reply; saved and restored; hostile stored images never rendered or sent.
//   W13 — errors that say what to do: REAL serve error bodies replayed (401, 404, 429, 503 capacity, 503 halted,
//        400 context) — each explained with its remedy; Retry / Enter API key / Refresh models / New chat each
//        do it; an unreachable server; a mid-reply server error or dropped connection keeps the partial answer,
//        labelled; finish_reason length and cancelled explained; the model list's own errors; persisted.
//   W14 — export: the exact Markdown (turns, image, folded thinking, stats and state, labelled system prompt)
//        and JSON (the stored conversation, format tag, version); safe file names; disabled when empty or busy.
//   W15 — theme: System follows the (emulated) preference either way, Light and Dark override it, saved and
//        followed across tabs, restored first thing after reload; WCAG AA contrast measured for every text
//        colour on every real surface (gradient stops included) in each mode.
//   W16 — phone layout at an emulated 400 and 360 px, crowded with the content most likely to break it: no
//        sideways page scroll, nothing past the edge outside its own scroll box, every control >= 24x24 px,
//        header items not overlapping, key controls on screen; the desktop header still one row.
//   W17 — keyboard: Ctrl/Cmd+Enter always sends; "Enter sends" as a saved setting (Shift/Alt+Enter never send);
//        never while an input method composes (isComposing, keyCode 229); the edit box follows the setting; ↑ in
//        an empty box edits the last message; Esc stops a reply and still cancels an edit.
//   W18 — which model answered: model and compute path on each reply, saved; a divider exactly where the model
//        changes, following Regenerate, left behind by no failed request; path in the export; hostile or
//        over-long stored labels inert or dropped.
//   W27–W29 — replies as server jobs, replaying REAL serve jobs traffic: a text reply is a job streamed from its
//        events (usage and end from the real closing chunks); a waiting reply shows its place in line from the real
//        queue field; Stop DELETEs the job; the real 429; images stay on the streaming route (list disabled there);
//        leaving mid-reply keeps the job running, marked in the list, and re-attaches on return; finished, failed,
//        cancelled and lost jobs are filled in or labelled; Resume after an unreachable server; a reload mid-reply.
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { readFileSync } from "node:fs";
import { openPage, report, finishOrExit } from "./webui-gate/cdp.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const page = await openPage(pathToFileURL(resolve(here, "..", "internal", "serveapp", "webui", "index.html")).href);

// Shared by every phase: result recording, the model option, and a fetch stub that streams SSE.
// (No template literals in page code: this text is itself inside Node template strings.)
const prelude = String.raw`
  const results = [];
  const check = (name, ok, why) => results.push(ok ? { name, ok: true } : { name, ok: false, why: String(why) });
  const wait = ms => new Promise(r => setTimeout(r, ms));
  if (typeof send !== "function" || typeof Markdown !== "object") {
    return [{ name: "page scripts loaded", ok: false, why: "send()/Markdown missing — index.html did not load its ui/ scripts" }];
  }
  if (window.__pwned === undefined) window.__pwned = 0;
  const sel = $("model");
  const opt = document.createElement("option");
  opt.value = opt.textContent = "gate-model";
  sel.appendChild(opt);
  sel.value = "gate-model";

  // fetch stub: streams text as SSE deltas in 3-char chunks (cutting through fences and emphasis like
  // a real token stream). With hang=true it stops mid-answer and waits for the request to be aborted.
  const enc = new TextEncoder();
  function streamAnswer(text, { hang = false } = {}) {
    window.fetch = async (url, opts) => { window.__lastBody = opts && opts.body ? JSON.parse(opts.body) : null; return new Response(new ReadableStream({ async start(c) {
      opts?.signal?.addEventListener("abort", () => { try { c.error(new DOMException("aborted", "AbortError")); } catch {} });
      for (let k = 0; k < text.length; k += 3) {
        c.enqueue(enc.encode("data: " + JSON.stringify({ choices: [{ delta: { content: text.slice(k, k + 3) } }] }) + "\n\n"));
        await wait(2);
      }
      if (hang) return;              // never closes: only Stop ends it
      c.enqueue(enc.encode("data: [DONE]\n\n"));
      c.close();
    } }), { status: 200, headers: { "Content-Type": "text/event-stream" } }); };
  }
  const lastBot = () => { const b = document.querySelectorAll("#log .msg.bot"); return b[b.length - 1]; };
  const lastYou = () => { const b = document.querySelectorAll("#log .msg.you"); return b[b.length - 1]; };
  // W9: the conversation on screen is stored under its own key
  const chatKey = () => "goinfer.chat.v2." + currentChat.id;
  const stored = () => { try { return JSON.parse(localStorage.getItem(chatKey())); } catch { return "UNPARSEABLE"; } };
  const putStored = messages => localStorage.setItem(chatKey(), JSON.stringify({ v: 2, id: currentChat.id, title: "t", titled: "user", updated: Date.now(), messages }));

  // W9 asks the model for a title in the background, as a NON-streaming chat request. Every fetch stub a
  // phase installs is wrapped so that request is answered here — recorded in __titleBodies, answered with
  // __titleReply — and never reaches the stub, whose __lastBody the other checks read.
  if (!window.__titleWrapped) {
    window.__titleWrapped = true;
    window.__titleBodies = [];
    window.__titleReply = { content: "Gate title" };
    let inner = window.fetch;
    // W27: text replies go through the jobs API. This emulates it over whatever stub a phase installed: the
    // submit is answered by the stub (as a chat request — so the stub still decides what "the model" does, and
    // records __lastBody), a successful stream is kept behind a job id, and /events hands it back. Phases that
    // test the jobs API itself replace window.__jobsEmulator with their own behaviour.
    window.__jobs = {};
    let jobSeq = 0;
    const emulateJobs = async (url, opts) => {
      if (url === "/v1/jobs" && opts && opts.method === "POST") {
        const r = await inner("/v1/chat/completions", opts);
        if (!r.ok) return r;
        const id = "job_" + (++jobSeq).toString(16).padStart(16, "0");
        window.__jobs[id] = { response: r, status: "running", cancelled: false };
        return new Response(JSON.stringify({ id, status: "pending" }), { status: 202, headers: { "Content-Type": "application/json" } });
      }
      const m = typeof url === "string" && url.match(/^\/v1\/jobs\/(job_[0-9a-f]+)(\/events)?$/);
      if (!m) return null;
      const job = window.__jobs[m[1]];
      if (!job) return new Response(JSON.stringify({ error: { message: "job not found" } }), { status: 404 });
      if (opts && opts.method === "DELETE") { job.cancelled = true; return new Response(JSON.stringify({ id: m[1], cancelled: true }), { status: 200 }); }
      if (m[2]) { const r = job.response; job.response = null; return r || new Response("data: [DONE]\n\n", { status: 200 }); }
      return new Response(JSON.stringify({ id: m[1], status: job.status }), { status: 200 });
    };
    window.__jobsEmulator = emulateJobs;
    const wrapped = async (url, opts) => {
      if (window.__jobsEmulator) { const j = await window.__jobsEmulator(url, opts); if (j) return j; }
      let b = null; try { b = opts && opts.body ? JSON.parse(opts.body) : null; } catch {}
      if (url === "/v1/chat/completions" && b && b.stream !== true) {
        window.__titleBodies.push(b);
        const t = window.__titleReply;
        if (t.hang) await new Promise((_, no) => opts?.signal?.addEventListener("abort", () => { window.__titleAborts = (window.__titleAborts || 0) + 1; no(new DOMException("aborted", "AbortError")); }));
        if (t.delay) await wait(t.delay);
        if (t.status) return new Response("{}", { status: t.status });
        return new Response(JSON.stringify({ choices: [{ message: { role: "assistant", content: t.content } }] }), { status: 200 });
      }
      return inner(url, opts);
    };
    Object.defineProperty(window, "fetch", { configurable: true, get: () => wrapped, set: f => { inner = f; } });
  }
`;
// Every phase ends with the same rendering check, because a property check cannot see this class of bug:
// an element can carry the hidden attribute (el.hidden === true) while a CSS display rule keeps it on
// screen. The context meter (W8) and the image preview's Remove button (W11) both shipped that way — every
// check that read .hidden passed. So: nothing marked hidden may take up any space.
const RENDER_CHECK = String.raw`
  const leaks = [...document.querySelectorAll("[hidden]")].filter(el => el.getClientRects().length > 0).map(el => el.id || el.className || el.tagName);
  check("render: nothing marked hidden is on screen (" + document.querySelectorAll("[hidden]").length + " hidden elements)", leaks.length === 0, JSON.stringify(leaks));
`;
const phase = body => "(async () => {" + prelude + body + RENDER_CHECK + "\n  return results;\n})()";

// ---- phase 1: W1 + W2 in a fresh page, then what W3 saved -----------------------------------------
const phase1 = phase(String.raw`

  // ---- W1: streamed Markdown, end to end ----------------------------------------------------------
  const code = "func main() {\n\tfmt.Println(\"hi\")\n}";
  const answer = "Here is **bold** and \x60inline\x60.\n\n\x60\x60\x60go\n" + code + "\n\x60\x60\x60\n\n- one\n- two\n\n<img src=x onerror=\"window.__pwned=1\">";
  const prompt = "show me markdown";
  streamAnswer(answer);
  $("prompt").value = prompt;
  await send();
  await wait(200);
  const bot = lastBot(), out = bot.children[1];
  check("W1 bubble is marked .md", out.classList.contains("md"), out.className);
  check("W1 strong rendered", out.querySelector("strong")?.textContent === "bold", out.querySelector("strong")?.textContent);
  check("W1 inline code rendered", out.querySelector("p code")?.textContent === "inline", out.querySelector("p code")?.textContent);
  check("W1 code block keeps tabs and newlines", out.querySelector("pre code")?.textContent === code, JSON.stringify(out.querySelector("pre code")?.textContent));
  check("W1 language label", out.querySelector(".md-lang")?.textContent === "go", out.querySelector(".md-lang")?.textContent);
  check("W1 list rendered", [...out.querySelectorAll("li")].map(l => l.textContent).join("|") === "one|two", [...out.querySelectorAll("li")].map(l => l.textContent));
  check("W1 hostile <img> not created", document.querySelectorAll("#log img").length === 0, document.querySelectorAll("#log img").length);
  check("W1 per-response stats still attached", /tok · .*tok\/s · .*s$/.test(bot.querySelector(".meta")?.textContent || ""), bot.querySelector(".meta")?.textContent);

  // ---- W2: copy ------------------------------------------------------------------------------------
  const you = lastYou();
  check("W2 user message has one Copy action", you.querySelectorAll(".actions .msg-copy").length === 1, you.querySelectorAll(".actions .msg-copy").length);
  check("W2 model message has one Copy action", bot.querySelectorAll(".actions .msg-copy").length === 1, bot.querySelectorAll(".actions .msg-copy").length);
  check("W2 code block has one Copy button", out.querySelectorAll(".md-code .md-copy").length === 1, out.querySelectorAll(".md-code .md-copy").length);

  // path 1: async clipboard API
  let clip = null;
  const clipDesc = Object.getOwnPropertyDescriptor(Navigator.prototype, "clipboard");
  const fakeClipboard = { writeText: async s => { clip = s; } };
  Object.defineProperty(navigator, "clipboard", { value: fakeClipboard, configurable: true });
  check("W2 file:// counts as a secure context (so path 1 is the one exercised)", window.isSecureContext === true, window.isSecureContext);

  out.querySelector(".md-copy").click(); await wait(50);
  check("W2 code Copy copies the exact code", clip === code, JSON.stringify(clip));
  check("W2 code Copy shows Copied", out.querySelector(".md-copy").textContent === "Copied", out.querySelector(".md-copy").textContent);

  bot.querySelector(".msg-copy").click(); await wait(50);
  check("W2 message Copy copies the raw Markdown, not the rendered text", clip === answer, JSON.stringify(clip)?.slice(0, 80));
  you.querySelector(".msg-copy").click(); await wait(50);
  check("W2 user Copy copies the prompt", clip === prompt, JSON.stringify(clip));

  // delegation survives a re-render that rebuilds every code-block button
  Markdown.render(out, answer);
  clip = null;
  out.querySelector(".md-copy").click(); await wait(50);
  check("W2 Copy still works on a button rebuilt by re-render (delegated handler)", clip === code, JSON.stringify(clip));

  // path 2: API present but rejecting -> textarea fallback
  let exec = null;
  const realExec = document.execCommand.bind(document);
  fakeClipboard.writeText = async () => { throw new Error("denied"); };
  document.execCommand = cmd => { exec = { cmd, value: document.querySelector("textarea[readonly]")?.value }; return true; };
  clip = null;
  out.querySelector(".md-copy").click(); await wait(50);
  check("W2 rejected clipboard API falls back to the textarea path", exec?.cmd === "copy" && exec?.value === code, JSON.stringify(exec));
  check("W2 fallback textarea is removed afterwards", document.querySelectorAll("textarea[readonly]").length === 0, document.querySelectorAll("textarea[readonly]").length);

  // path 3: no clipboard API at all (insecure context, e.g. http://<lan-ip>) -> textarea fallback
  Object.defineProperty(navigator, "clipboard", { value: undefined, configurable: true });
  exec = null;
  bot.querySelector(".msg-copy").click(); await wait(50);
  check("W2 missing clipboard API (insecure context) still copies via fallback", exec?.value === answer, JSON.stringify(exec)?.slice(0, 80));

  // both refuse -> visible failure, then the label resets
  document.execCommand = () => false;
  const failBtn = you.querySelector(".msg-copy");
  failBtn.click(); await wait(50);
  check("W2 total failure shows Copy failed", failBtn.textContent === "Copy failed", failBtn.textContent);
  await wait(1600);
  check("W2 feedback label resets", failBtn.textContent === "Copy", failBtn.textContent);

  // restore the real clipboard plumbing
  document.execCommand = realExec;
  if (clipDesc) delete navigator.clipboard;
  Object.defineProperty(navigator, "clipboard", { value: fakeClipboard, configurable: true });
  fakeClipboard.writeText = async s => { clip = s; };

  // a STOPPED answer still gets Copy, and copies what arrived
  const partial = "partial answer that never finishes ";
  streamAnswer(partial, { hang: true });
  $("prompt").value = "this will be stopped";
  const running = send();
  await wait(400);
  $("stop").click();
  await running;
  await wait(100);
  const stopped = lastBot();
  check("W2 stopped answer has a Copy action", stopped.querySelectorAll(".actions .msg-copy").length === 1, stopped.querySelectorAll(".actions .msg-copy").length);
  clip = null;
  stopped.querySelector(".msg-copy")?.click(); await wait(50);
  check("W2 stopped answer copies what arrived", clip === partial, JSON.stringify(clip));

  check("no payload executed anywhere", window.__pwned === 0, window.__pwned);

  // ---- W3: what was saved -------------------------------------------------------------------------
  const st = stored();
  const msgs = st && st.messages;
  check("W3 conversation saved under a versioned key", st && st.v === 2 && st.id === currentChat.id && Array.isArray(msgs), JSON.stringify(st)?.slice(0, 120));
  check("W3 saved exactly the four turns", msgs && msgs.map(m => m.role).join(",") === "user,assistant,user,assistant", msgs && msgs.map(m => m.role));
  check("W3 saved the full answer and its stats", msgs && msgs[1].content === answer && /tok\/s/.test(msgs[1].meta || "") && !msgs[1].state, JSON.stringify(msgs && msgs[1])?.slice(0, 160));
  check("W3 saved the stopped answer as stopped, with what arrived", msgs && msgs[3].content === partial && msgs[3].state === "stopped", JSON.stringify(msgs && msgs[3]));
  check("W3 nothing left marked generating", msgs && !msgs.some(m => m.state === "generating"), JSON.stringify(msgs));
`);

// ---- phase 2: after a reload — restored; then start a reply and reload mid-stream --------------------
const phase2 = phase(String.raw`
  const bots = document.querySelectorAll("#log .msg.bot"), yous = document.querySelectorAll("#log .msg.you");
  check("W3 restored all four bubbles", bots.length === 2 && yous.length === 2, bots.length + " bot, " + yous.length + " you");
  const first = bots[0].children[1];
  check("W3 restored answer re-renders as Markdown", first.classList.contains("md") && first.querySelector("pre code")?.textContent === "func main() {\n\tfmt.Println(\"hi\")\n}", first.innerText.slice(0, 80));
  check("W3 restored answer keeps its stats line", /tok · .*tok\/s/.test(bots[0].querySelector(".meta")?.textContent || ""), bots[0].querySelector(".meta")?.textContent);
  check("W3 restored stopped answer is labelled stopped", /^stopped/.test(bots[1].querySelector(".meta")?.textContent || ""), bots[1].querySelector(".meta")?.textContent);
  check("W3 restored user prompt as plain text", yous[0].children[1].textContent === "show me markdown", yous[0].children[1].textContent);
  let clip = null;
  Object.defineProperty(navigator, "clipboard", { value: { writeText: async s => { clip = s; } }, configurable: true });
  bots[0].querySelector(".msg-copy").click(); await wait(50);
  check("W3 Copy on a restored answer copies its raw Markdown", typeof clip === "string" && clip.startsWith("Here is **bold**"), JSON.stringify(clip)?.slice(0, 60));
  check("W3 no uncaught state after restore (New chat enabled, Send enabled)", !$("newchat").disabled && !$("send").disabled, $("newchat").disabled + "/" + $("send").disabled);

  // start a reply that never finishes, let the periodic save run, then return WITHOUT pressing Stop —
  // the gate reloads the page mid-stream next
  streamAnswer("an answer the reload interrupts ", { hang: true });
  $("prompt").value = "this reply gets interrupted";
  send();
  await wait(1500);
  // since W29 a text reply is a job: New chat stays usable during it, and leaving detaches rather than stops
  check("W3/W29 during a job-backed reply New chat stays usable (it detaches the job, it does not stop it)", $("newchat").disabled === false && !!(generating && generating.job), $("newchat").disabled + " / job " + (generating && generating.job));
`);

// ---- phase 3: after a mid-stream reload; then corrupt the store --------------------------------------
const phase3 = phase(String.raw`
  const bots = document.querySelectorAll("#log .msg.bot");
  const last = bots[bots.length - 1];
  check("W3 mid-stream reload kept the partial answer", bots.length === 3 && last.children[1].textContent.startsWith("an answer the reload interrupts"), bots.length + " / " + last?.children[1].textContent);
  // since W27 a text reply is a server job, so a reply the reload cut off is "not finished here", with Resume
  check("W3/W27 it is labelled as not finished here, with a way to resume", /not finished here — the job may still be running on the server\./.test(last.querySelector(".meta")?.textContent || "") && !!last.querySelector(".msg-resume"), last.querySelector(".meta")?.textContent);
  const st = stored();
  check("W3 interrupted state written back to storage", st.messages[st.messages.length - 1].state === "interrupted", JSON.stringify(st.messages[st.messages.length - 1]));

  window.__corruptKey = chatKey();
  sessionStorage.setItem("gate.corruptKey", chatKey());
  localStorage.setItem(chatKey(), "{this is not json");
`);

// ---- phase 4: after a reload with unreadable storage; quota; New chat; tab sync ---------------------
const phase4 = phase(String.raw`
  const corrupt = sessionStorage.getItem("gate.corruptKey");
  check("W3 unreadable storage: page loads with an empty log", document.querySelectorAll("#log .msg").length === 0, document.querySelectorAll("#log .msg").length);
  check("W3 unreadable storage is set aside, not destroyed", !!corrupt && localStorage.getItem(corrupt + ".unreadable") === "{this is not json", localStorage.getItem(corrupt + ".unreadable"));
  check("W3 unreadable value removed from the live key", localStorage.getItem(corrupt) === null, localStorage.getItem(corrupt));

  // quota: every write throws — the chat must keep working and say so
  const realSet = Storage.prototype.setItem;
  Storage.prototype.setItem = function () { throw new DOMException("full", "QuotaExceededError"); };
  streamAnswer("still works without storage");
  $("prompt").value = "storage is full";
  await send(); await wait(150);
  check("W3 quota failure: the reply still renders", lastBot()?.children[1].textContent === "still works without storage", lastBot()?.children[1].textContent);
  check("W3 quota failure: the note is shown", !$("store-note").hidden && /can't be kept across a reload/.test($("store-note").textContent), $("store-note").hidden + " " + $("store-note").textContent);
  Storage.prototype.setItem = realSet;

  // New chat (W9 semantics): starts another conversation, keeps this one, asks nothing
  let asked = 0;
  window.confirm = () => { asked++; return false; };
  save();   // storage works again: write the quota-era conversation
  const before = chatKey();
  $("newchat").click();
  check("W3/W9 New chat clears the log", document.querySelectorAll("#log .msg").length === 0, document.querySelectorAll("#log .msg").length);
  check("W3/W9 New chat keeps the previous conversation stored, and asks nothing", asked === 0 && JSON.parse(localStorage.getItem(before))?.messages?.length === 2 && chatKey() !== before, asked + " asked / " + localStorage.getItem(before)?.slice(0, 60));
  check("W3 New chat hides the storage note", $("store-note").hidden, $("store-note").hidden);
  window.confirm = () => true;

  // another tab writes the conversation this tab has open: this idle tab follows it
  putStored([
    { role: "user", content: "from the other tab" },
    { role: "assistant", content: "**synced**", model: "other-model", meta: "3 tok" } ]);
  window.dispatchEvent(new StorageEvent("storage", { key: chatKey() }));
  await wait(50);
  check("W3 idle tab follows another tab's conversation", lastBot()?.children[1].querySelector("strong")?.textContent === "synced" && lastYou()?.children[1].textContent === "from the other tab", lastBot()?.innerText);

  // hostile stored content, rendered on the next load
  putStored([
    { role: "user", content: "<img src=x onerror=\"window.__pwned=1\">" },
    { role: "assistant", content: "[x](javascript:window.__pwned=1)\n\n<script>window.__pwned=1<\/script>", model: "<b onmouseover=window.__pwned=1>m</b>", meta: "<img src=x onerror=window.__pwned=1>" },
    { role: "system", content: "not a turn this page renders" },
    { role: "assistant", content: 42 } ]);
`);

// ---- phase 5: after a reload with hostile stored content --------------------------------------------
const phase5 = phase(String.raw`
  await wait(300);
  check("W3 hostile stored content: no element outside the page's own", document.querySelectorAll("#log img, #log script, #log b").length === 0, document.querySelectorAll("#log img, #log script, #log b").length);
  check("W3 hostile stored content: no javascript: link", ![...document.querySelectorAll("#log a")].some(a => /^javascript:/i.test(a.getAttribute("href") || "")), [...document.querySelectorAll("#log a")].map(a => a.getAttribute("href")));
  check("W3 hostile stored content: canary never fired", window.__pwned === 0, window.__pwned);
  check("W3 invalid stored entries skipped (system role, non-string content)", document.querySelectorAll("#log .msg").length === 2, document.querySelectorAll("#log .msg").length);
  check("W3 stored model name shown as text", document.querySelector("#log .msg.bot .who")?.textContent === "<b onmouseover=window.__pwned=1>m</b>", document.querySelector("#log .msg.bot .who")?.textContent);
`);

// ---- phase 6: W4 — the system prompt ---------------------------------------------------------------
const phase6 = phase(String.raw`
  const sys = $("system"), state = $("system-state");
  const setSys = v => { sys.value = v; sys.dispatchEvent(new Event("input", { bubbles: true })); };
  const roles = () => (window.__lastBody?.messages || []).map(m => m.role).join(",");
  window.confirm = () => true;
  $("newchat").click();
  check("W4 system prompt box present, collapsed, inactive on a fresh profile", sys && !$("system-box").open && state.textContent === "" && sys.value === "", sys?.value + "|" + state?.textContent);

  streamAnswer("one");
  $("prompt").value = "no system prompt yet";
  await send(); await wait(100);
  check("W4 no system message sent when the box is empty", roles() === "user", roles());

  setSys("Answer in French.");
  check("W4 summary shows active once set", state.textContent === "· active", state.textContent);
  check("W4 saved under its own key as typed", localStorage.getItem("goinfer.system.v1") === "Answer in French.", localStorage.getItem("goinfer.system.v1"));
  streamAnswer("deux");
  $("prompt").value = "second";
  await send(); await wait(100);
  const body = window.__lastBody;
  check("W4 system message sent FIRST, then the history", roles() === "system,user,assistant,user" && body.messages[0].content === "Answer in French.", roles() + " / " + JSON.stringify(body?.messages?.[0]));
  const st = stored();
  check("W4 system prompt NOT written into the saved transcript", !st.messages.some(m => m.role === "system" || m.content === "Answer in French."), JSON.stringify(st.messages.map(m => m.role)));

  setSys("   \n\t  ");
  check("W4 whitespace-only counts as empty: inactive", state.textContent === "", state.textContent);
  check("W4 whitespace-only removes the stored key", localStorage.getItem("goinfer.system.v1") === null, localStorage.getItem("goinfer.system.v1"));
  streamAnswer("trois");
  $("prompt").value = "third";
  await send(); await wait(100);
  check("W4 whitespace-only prompt is not sent", !roles().startsWith("system"), roles());

  setSys("   Be brief.  ");
  streamAnswer("quatre");
  $("prompt").value = "fourth";
  await send(); await wait(100);
  check("W4 system prompt is sent trimmed", window.__lastBody.messages[0].role === "system" && window.__lastBody.messages[0].content === "Be brief.", JSON.stringify(window.__lastBody.messages[0]));

  $("newchat").click();
  check("W4 New chat clears the conversation but keeps the system prompt", document.querySelectorAll("#log .msg").length === 0 && sys.value === "   Be brief.  " && localStorage.getItem("goinfer.system.v1") === "   Be brief.  ", sys.value + " / " + localStorage.getItem("goinfer.system.v1"));

  setSys("<img src=x onerror=\"window.__pwned=1\">");
  await wait(200);
  check("W4 hostile system prompt stays inert text", document.querySelectorAll("img").length === 0 && window.__pwned === 0, document.querySelectorAll("img").length + " img, pwned " + window.__pwned);

  // another tab changes the system prompt
  sys.blur();
  localStorage.setItem("goinfer.system.v1", "From tab two.");
  window.dispatchEvent(new StorageEvent("storage", { key: "goinfer.system.v1" }));
  check("W4 idle box follows another tab's system prompt", sys.value === "From tab two." && state.textContent === "· active", sys.value);
  // A user can only type in the box when its <details> is open — and a closed <details> hides its
  // content from focus, so focus() would silently do nothing and this check would test an idle box.
  $("system-box").open = true;
  sys.focus();
  check("W4 precondition: the system box really has focus", document.activeElement === sys, document.activeElement?.id);
  sys.value = "typing here";
  localStorage.setItem("goinfer.system.v1", "From tab three.");
  window.dispatchEvent(new StorageEvent("storage", { key: "goinfer.system.v1" }));
  check("W4 another tab's change does NOT clobber a box being typed in", sys.value === "typing here", sys.value);
  sys.blur();
  setSys("Be brief.");
`);

// ---- phase 7: after a reload — the system prompt persisted ------------------------------------------
const phase7 = phase(String.raw`
  check("W4 system prompt restored after reload", $("system").value === "Be brief.", $("system").value);
  check("W4 restored prompt shows active", $("system-state").textContent === "· active", $("system-state").textContent);
  streamAnswer("après");
  $("prompt").value = "after reload";
  await send(); await wait(100);
  check("W4 restored prompt is sent", window.__lastBody?.messages?.[0]?.role === "system" && window.__lastBody.messages[0].content === "Be brief.", JSON.stringify(window.__lastBody?.messages?.[0]));
`);

// ---- phase 8: W5 — load what the pull downloaded ----------------------------------------------------
// fetch is routed by URL: a pull that finishes, a load whose outcome each step picks, and a /v1/models
// that lists the model only once a load has succeeded — so "selected" means the page really refreshed.
const phase8 = phase(String.raw`
  const until = async (cond, ms = 3000) => { const t0 = Date.now(); while (!cond() && Date.now() - t0 < ms) await wait(10); return cond(); };
  const PULLED = "/home/u/.cache/goinfer/models/o/r/tiny-Q4_K_M.gguf";
  const sse = frames => new Response(new ReadableStream({ async start(c) {
    for (let f of frames) {
      if (f === "hold") { await new Promise(r => { window.__releaseLoad = r; }); continue; }
      if (typeof f === "function") f = f();
      c.enqueue(enc.encode("event: " + f[0] + "\ndata: " + JSON.stringify(f[1]) + "\n\n"));
      await wait(5);
    }
    c.close();
  } }), { status: 200, headers: { "Content-Type": "text/event-stream" } });
  const jsonErr = (status, message) => new Response(JSON.stringify({ error: { message } }), { status, headers: { "Content-Type": "application/json" } });
  let listed = [{ id: "gate-model", decode_path: "cpu" }];
  let loadMode = "ok", pulledPath = PULLED;
  const loadCalls = [];
  const routed = async (url, opts) => {
    if (url === "/v1/models") return new Response(JSON.stringify({ object: "list", data: listed }), { status: 200 });
    if (url === "/web/models/pull") return sse([["start", { human: "1 MB", sha256: "ab" }], ["done", { path: pulledPath, elapsed: "1s", verified: true }]]);
    if (url === "/web/models/load") {
      loadCalls.push({ method: opts?.method, body: opts?.body, ctype: opts?.headers?.["Content-Type"] });
      switch (loadMode) {
        case "ok":
          return sse([["start", { name: "tiny-Q4_K_M" }], ["progress", { elapsed: "2s" }], "hold", () => { listed = listed.concat([{ id: "tiny-Q4_K_M", decode_path: "resident" }]); return ["done", { id: "tiny-Q4_K_M", elapsed: "4s" }]; }]);
        case "error": return sse([["start", { name: "x" }], ["error", { message: "out of memory <img src=x onerror=\"window.__pwned=1\">", status: 400 }]]);
        case "running": return jsonErr(409, "a model load is already running");
        case "dup": return jsonErr(409, "model \"tiny-Q4_K_M\" already loaded");
        case "cut": return sse([["start", { name: "x" }]]);
      }
    }
    return jsonErr(404, "unrouted " + url);
  };
  window.fetch = routed;
  const st = $("pull-status");

  await pull("o/r", "tiny-Q4_K_M.gguf");
  const btn = () => document.getElementById("load-now"), msg = () => document.getElementById("load-status");
  check("W5 the pull no longer dead-ends on restart-the-server", !/restart the server/i.test(st.textContent), st.textContent);
  check("W5 a finished pull offers Load it now", btn()?.textContent === "Load it now" && btn().type === "button" && !btn().disabled, btn()?.outerHTML);
  check("W5 and says the model is not loaded yet", msg()?.textContent === "Downloaded, not loaded yet.", msg()?.textContent);

  btn().click();
  await until(() => window.__releaseLoad);
  check("W5 load POSTs exactly the path the pull returned", loadCalls.length === 1 && loadCalls[0].method === "POST" && loadCalls[0].body === JSON.stringify({ path: PULLED }) && loadCalls[0].ctype === "application/json", JSON.stringify(loadCalls));
  check("W5 button disabled while loading", btn().disabled === true, btn().disabled);
  check("W5 heartbeat shown while loading", msg().textContent === "loading… 2s", msg().textContent);
  window.__releaseLoad();
  await until(() => btn().hidden);
  check("W5 loaded model is selected in Chat", $("model").value === "tiny-Q4_K_M" && [...$("model").options].map(o => o.value).join("|") === "gate-model|tiny-Q4_K_M", $("model").value + " / " + [...$("model").options].map(o => o.value));
  check("W5 header stats follow the selected model", /model tiny-Q4_K_M/.test($("stats").textContent) && /path resident/.test($("stats").textContent), $("stats").textContent);
  check("W5 success reported and the button retired", msg().className === "note ok" && msg().textContent === "Loaded tiny-Q4_K_M in 4s — selected in Chat." && btn().hidden, msg().className + " / " + msg().textContent);
  streamAnswer("hello from the loaded model");
  $("prompt").value = "hi";
  await send(); await wait(100);
  check("W5 the next chat goes to the loaded model", window.__lastBody?.model === "tiny-Q4_K_M", window.__lastBody?.model);

  $("model").value = "gate-model";
  $("model").dispatchEvent(new Event("change"));
  check("W5 changing the selection updates the header stats", /model gate-model/.test($("stats").textContent), $("stats").textContent);
  window.fetch = routed;
  // the second option, so a refresh that forgot the choice (and fell back to the first) shows
  $("model").value = "tiny-Q4_K_M";
  await loadModels();
  check("W5 a refresh keeps the user's selection", $("model").value === "tiny-Q4_K_M", $("model").value);
  $("model").value = "gate-model";

  // failures: each keeps the button usable and changes nothing else
  for (const [mode, want] of [["error", /^out of memory <img/], ["running", /^a model load is already running$/], ["cut", /ended without a result/]]) {
    loadMode = mode;
    await pull("o/r", "tiny-Q4_K_M.gguf");
    btn().click();
    await until(() => msg().className === "note err");
    check("W5 load failure (" + mode + ") shown as an error", msg().className === "note err" && want.test(msg().textContent), msg().className + " / " + msg().textContent);
    check("W5 load failure (" + mode + ") leaves the button usable", !btn().disabled && !btn().hidden, btn().disabled + "/" + btn().hidden);
    check("W5 load failure (" + mode + ") leaves the selection alone", $("model").value === "gate-model", $("model").value);
  }
  await wait(200);
  check("W5 hostile load error stays inert", document.querySelectorAll("img").length === 0 && window.__pwned === 0, document.querySelectorAll("img").length + " img, pwned " + window.__pwned);

  // already loaded is what the user wanted: select it, say so
  loadMode = "dup";
  await pull("o/r", "tiny-Q4_K_M.gguf");
  btn().click();
  await until(() => btn().hidden || msg().className === "note err");
  check("W5 already-loaded selects the model instead of failing", $("model").value === "tiny-Q4_K_M" && msg().className === "note ok" && msg().textContent === "Already loaded as tiny-Q4_K_M — selected in Chat.", $("model").value + " / " + msg().className + " / " + msg().textContent);

  // a hostile path from the pull is shown as text, never markup
  pulledPath = "/c/<img src=x onerror=\"window.__pwned=1\">.gguf";
  await pull("o/r", "x.gguf");
  await wait(200);
  check("W5 hostile pulled path stays inert", st.querySelector("code")?.textContent === pulledPath && document.querySelectorAll("img").length === 0 && window.__pwned === 0, st.innerHTML);
`);

// ---- phase 9: W6 — thinking folded above the answer ---------------------------------------------------
// The reply shapes are the ones real serve output has (Qwen3-1.7B and gpt-oss-20b, captured 2026-09-14).
const phase9 = phase(String.raw`
  const until = async (cond, ms = 3000) => { const t0 = Date.now(); while (!cond() && Date.now() - t0 < ms) await wait(10); return cond(); };
  let clip = null;
  Object.defineProperty(navigator, "clipboard", { value: { writeText: async s => { clip = s; } }, configurable: true });
  // On-screen text of the reply being streamed that shows a raw tag fragment, sampled on every DOM change.
  // Only the LAST reply: an earlier one may legitimately show "<think>" (the mention check below).
  let flashed = [];
  const watch = new MutationObserver(() => {
    const txt = lastBot()?.children[1]?.textContent || "";
    if (/<\/?think>|<\|/.test(txt) || /<\/?th?i?n?k?$/.test(txt.trimEnd())) flashed.push(txt.slice(-40));
  });
  watch.observe($("log"), { childList: true, subtree: true, characterData: true });
  const turn = async (text, prompt) => { streamAnswer(text); $("prompt").value = prompt; await send(); await wait(150); return lastBot(); };

  // manual stream: the gate decides what arrives and when
  let push = null, end = null;
  const manualStream = () => { window.fetch = async (url, opts) => { window.__lastBody = JSON.parse(opts.body); return new Response(new ReadableStream({ start(c) {
    push = t => c.enqueue(enc.encode("data: " + JSON.stringify({ choices: [{ delta: { content: t } }] }) + "\n\n"));
    end = () => { c.enqueue(enc.encode("data: [DONE]\n\n")); c.close(); };
    opts?.signal?.addEventListener("abort", () => { try { c.error(new DOMException("aborted", "AbortError")); } catch {} });
  } }), { status: 200, headers: { "Content-Type": "text/event-stream" } }); }; };
  const frame = () => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));

  // ---- Qwen3 shape, streamed in 3-char chunks (so both tags arrive split) ----
  const qwen = "<think>\nLet me add 2 and 2.\n</think>\n\nThe answer is **4**.";
  let bot = await turn(qwen, "what is 2+2");
  let box = bot.querySelector("details.think");
  check("W6 thinking is folded into a <details>", !!box, bot.children[1].innerHTML.slice(0, 120));
  check("W6 fold is collapsed by default", box && box.open === false, box?.open);
  check("W6 finished fold is labelled with its duration", /^Thought for \d+(\.\d)?s$/.test(box?.querySelector("summary")?.textContent || ""), box?.querySelector("summary")?.textContent);
  check("W6 thinking text is inside the fold", box?.querySelector(".think-body")?.textContent.trim() === "Let me add 2 and 2.", JSON.stringify(box?.querySelector(".think-body")?.textContent));
  const ans = bot.querySelector(".think-answer");
  check("W6 the answer renders as Markdown below the fold", ans?.querySelector("strong")?.textContent === "4" && ans.textContent.trim() === "The answer is 4.", ans?.textContent);
  check("W6 no raw tag ever reached the screen while streaming", flashed.length === 0, JSON.stringify(flashed.slice(0, 3)));
  bot.querySelector(".msg-copy").click(); await wait(50);
  check("W6 Copy copies the answer without the thinking", clip === "The answer is **4**.", JSON.stringify(clip));
  const saved = stored().messages.at(-1);
  check("W6 the saved reply keeps the thinking and its duration", saved.content === qwen && typeof saved.thought === "number", JSON.stringify(saved).slice(0, 120));
  await turn("ok", "and 3+3?");
  const hist = window.__lastBody.messages.filter(m => m.role === "assistant");
  check("W6 the next request sends the earlier reply WITHOUT its thinking", hist.at(-1)?.content === "The answer is **4**." && !JSON.stringify(window.__lastBody).includes("think>"), JSON.stringify(hist.at(-1)));

  // ---- streaming behaviour, chunk by chunk ----
  manualStream();
  $("prompt").value = "think slowly";
  flashed = [];
  const running = send();
  await until(() => push);
  push("<th"); await frame();
  bot = lastBot();
  check("W6 a half-arrived opening tag shows nothing", bot.children[1].textContent === "", JSON.stringify(bot.children[1].textContent));
  push("ink>\nfirst step </thi"); await frame();
  box = bot.querySelector("details.think");
  check("W6 while thinking: Thinking… and marked live", box?.querySelector("summary")?.textContent === "Thinking…" && box.classList.contains("live"), box?.querySelector("summary")?.textContent);
  check("W6 a half-arrived closing tag is held back", box?.querySelector(".think-body")?.textContent.trim() === "first step", JSON.stringify(box?.querySelector(".think-body")?.textContent));
  box.open = true;
  push("nk-not-a-tag second step"); await frame();
  check("W6 held-back text that turns out not to be a tag is shown", /first step <\/think-not-a-tag second step/.test(box.querySelector(".think-body").textContent), box.querySelector(".think-body").textContent);
  check("W6 the fold is updated in place, so a reader's open survives new tokens", bot.querySelector("details.think") === box && box.open === true, box.open);
  push("</think>\n\nDone."); await frame();
  check("W6 closing the thinking relabels the fold and shows the answer", /^Thought for/.test(box.querySelector("summary").textContent) && !box.classList.contains("live") && bot.querySelector(".think-answer").textContent.trim() === "Done.", box.querySelector("summary").textContent + " / " + bot.querySelector(".think-answer")?.textContent);
  end(); await running; await wait(100);
  check("W6 the reader's open fold stays open after the reply finishes", bot.querySelector("details.think") === box && box.open === true, box.open);
  check("W6 no raw tag reached the screen (chunked)", flashed.length === 0, JSON.stringify(flashed.slice(0, 3)));

  // ---- other shapes ----
  bot = await turn("Reasoning here.\n</think>\n\nAnswer.", "template opened the thinking");
  check("W6 a closing tag with no opening folds what came before it", bot.querySelector(".think-body")?.textContent.trim() === "Reasoning here." && bot.querySelector(".think-answer")?.textContent.trim() === "Answer.", bot.children[1].textContent);

  const harmony = "<|channel|>analysis<|message|>Compute 17*23.<|end|><|start|>assistant<|channel|>final<|message|>It is **391**.<|return|>";
  bot = await turn(harmony, "gpt-oss");
  check("W6 gpt-oss: analysis channel folded, final channel is the answer", bot.querySelector(".think-body")?.textContent.trim() === "Compute 17*23." && bot.querySelector(".think-answer strong")?.textContent === "391", bot.children[1].textContent);
  check("W6 gpt-oss: no channel markers on screen", !/<\|/.test(bot.textContent), bot.textContent.slice(0, 120));
  bot.querySelector(".msg-copy").click(); await wait(50);
  check("W6 gpt-oss: Copy copies the final answer", clip === "It is **391**.", JSON.stringify(clip));

  // streamed by hand, with every frame settled before the stream ends: no animation-frame paint is left
  // pending to redraw the reply, so the note can only come from the page's own final render
  push = null;   // so the wait below is for THIS stream, not the finished one above
  manualStream();
  $("prompt").value = "gpt-oss, analysis only";
  const r1 = send();
  await until(() => push);
  push("<|channel|>analysis<|message|>Provide answer."); await frame(); await frame();
  end(); await r1; await wait(100);
  bot = lastBot();
  check("W6 a finished reply that thought but never answered says so", /No answer after the thinking/.test(bot.querySelector(".think-answer")?.textContent || ""), bot.children[1].textContent);

  bot = await turn("Qwen wraps reasoning in <think> tags.", "mention the tag");
  check("W6 an answer that only MENTIONS the tag is not folded", !bot.querySelector("details.think") && bot.children[1].textContent.includes("<think> tags"), bot.children[1].innerHTML.slice(0, 120));

  bot = await turn("<think><img src=x onerror=\"window.__pwned=1\"></think>**ok** <img src=x onerror=\"window.__pwned=1\">", "hostile");
  await wait(200);
  check("W6 hostile markup in thinking and answer stays inert", document.querySelectorAll("#log img").length === 0 && window.__pwned === 0 && bot.querySelector(".think-answer strong")?.textContent === "ok", document.querySelectorAll("#log img").length + " img, pwned " + window.__pwned);

  // ---- stopped mid-thought ----
  streamAnswer("<think>\na long line of reasoning that is still going ", { hang: true });
  $("prompt").value = "stop me";
  const r2 = send(); await wait(300); $("stop").click(); await r2; await wait(100);
  bot = lastBot();
  check("W6 stopped mid-thought: stopped, and no 'no answer' note", /^stopped/.test(bot.querySelector(".meta")?.textContent || "") && !/No answer/.test(bot.textContent), bot.querySelector(".meta")?.textContent + " / " + bot.children[1].textContent.slice(0, 60));
  bot.querySelector(".msg-copy").click(); await wait(50);
  check("W6 stopped mid-thought: Copy copies what arrived", /^<think>\na long line/.test(clip || ""), JSON.stringify(clip)?.slice(0, 40));

  // ---- REAL serve output, replayed chunk for chunk (scripts/webui-gate/captured-thinking.json) ----
  const CAPTURED = ${JSON.stringify(JSON.parse(readFileSync(resolve(here, "webui-gate", "captured-thinking.json"), "utf8")))};
  for (const rep of CAPTURED.replies) {
    window.fetch = async (url, opts) => { window.__lastBody = JSON.parse(opts.body); return new Response(new ReadableStream({ async start(c) {
      for (const d of rep.deltas) { c.enqueue(enc.encode("data: " + JSON.stringify({ choices: [{ delta: { content: d } }] }) + "\n\n")); await wait(1); }
      c.enqueue(enc.encode("data: [DONE]\n\n")); c.close();
    } }), { status: 200, headers: { "Content-Type": "text/event-stream" } }); };
    flashed = [];
    $("prompt").value = rep.prompt;
    await send(); await wait(150);
    bot = lastBot();
    const tag = "W6 real " + rep.model + ": ";
    check(tag + "thinking folded, collapsed", !!bot.querySelector("details.think") && bot.querySelector("details.think").open === false && bot.querySelector(".think-body").textContent.trim().length > 20, bot.children[1].textContent.slice(0, 80));
    check(tag + "no raw tag or channel marker reached the screen while streaming", flashed.length === 0, JSON.stringify(flashed.slice(0, 3)));
    if (rep.answer) {
      check(tag + "the answer below the fold is exactly the model's answer", bot.querySelector(".think-answer").textContent.trim() === rep.answer, JSON.stringify(bot.querySelector(".think-answer").textContent));
    } else {
      check(tag + "no final channel arrived, and the page says there is no answer", /No answer after the thinking/.test(bot.querySelector(".think-answer").textContent), bot.querySelector(".think-answer").textContent);
    }
  }
  watch.disconnect();

  // leave one finished Qwen-shaped reply last for the reload phase
  await turn(qwen, "for the reload");
`);

// ---- phase 10: after a reload — the fold is rebuilt from the saved reply -----------------------------
const phase10 = phase(String.raw`
  let bot = lastBot();
  let box = bot?.querySelector("details.think");
  check("W6 restored reply is folded again", !!box && box.open === false && bot.querySelector(".think-answer strong")?.textContent === "4", bot?.children[1]?.textContent);
  check("W6 restored fold keeps its duration label", /^Thought for \d+(\.\d)?s$/.test(box?.querySelector("summary")?.textContent || ""), box?.querySelector("summary")?.textContent);
  // a stored duration that is not a number is ignored, not rendered
  const st = stored();
  st.messages.at(-1).thought = "<img src=x onerror=\"window.__pwned=1\">";
  localStorage.setItem(chatKey(), JSON.stringify(st));
  window.dispatchEvent(new StorageEvent("storage", { key: chatKey() }));
  await wait(200);
  box = lastBot()?.querySelector("details.think");
  check("W6 a non-numeric stored duration falls back to a plain label", box?.querySelector("summary")?.textContent === "Thinking" && document.querySelectorAll("img").length === 0 && window.__pwned === 0, box?.querySelector("summary")?.textContent);
`);

// ---- phase 11: W7 — regenerate, edit, delete -----------------------------------------------------------
const phase11 = phase(String.raw`
  const until = async (cond, ms = 3000) => { const t0 = Date.now(); while (!cond() && Date.now() - t0 < ms) await wait(10); return cond(); };
  const idle = () => until(() => $("stop").hidden);
  let asked = [], answer = true;
  window.confirm = m => { asked.push(m); return answer; };
  $("newchat").click();
  const bots = () => [...document.querySelectorAll("#log .msg.bot")], yous = () => [...document.querySelectorAll("#log .msg.you")];
  const sent = () => window.__lastBody.messages.filter(m => m.role !== "system").map(m => m.role[0] + ":" + m.content).join("|");
  const shown = () => [...document.querySelectorAll("#log .msg")].map(m => (m.classList.contains("you") ? "u:" : "a:") + (m.querySelector(".think-answer") || m.children[1]).textContent.trim()).join("|");
  const saved = () => stored().messages.map(m => m.role[0] + ":" + m.content).join("|");
  const visibleRegen = () => [...document.querySelectorAll("#log .msg-regen")].filter(b => !b.hidden);
  const turn = async (q, a) => { streamAnswer(a); $("prompt").value = q; await send(); await idle(); };
  await turn("q1", "a1"); await turn("q2", "a2"); await turn("q3", "a3");

  check("W7 your messages have Edit, Copy, Delete", yous().every(m => [...m.querySelectorAll(".actions button")].map(b => b.textContent).join(",") === "Edit,Copy,Delete"), yous().map(m => [...m.querySelectorAll(".actions button")].map(b => b.textContent)));
  check("W7 Regenerate is offered only on the last reply", visibleRegen().length === 1 && visibleRegen()[0].closest(".msg") === bots().at(-1), visibleRegen().length);

  // ---- regenerate ----
  streamAnswer("a3 again");
  visibleRegen()[0].click(); await idle();
  check("W7 regenerate asks again with the history up to the last question", sent() === "u:q1|a:a1|u:q2|a:a2|u:q3", sent());
  check("W7 regenerate replaces the last reply, on screen and saved", shown() === "u:q1|a:a1|u:q2|a:a2|u:q3|a:a3 again" && saved() === "u:q1|a:a1|u:q2|a:a2|u:q3|a:a3 again", shown() + " / " + saved());

  // ---- nothing changes the conversation while a reply is generating ----
  streamAnswer("a slow reply ", { hang: true });
  visibleRegen()[0].click();
  await until(() => !$("stop").hidden); await wait(100);
  const acting = [...document.querySelectorAll("#log .msg-edit, #log .msg-regen, #log .msg-delete")];
  check("W7 edit, regenerate and delete are disabled while generating", acting.length > 0 && acting.every(b => b.disabled), acting.filter(b => !b.disabled).map(b => b.textContent));
  asked = [];
  yous()[0].querySelector(".msg-delete").click();
  check("W7 a disabled Delete does nothing", asked.length === 0 && yous().length === 3, asked.length + " asked, " + yous().length + " yous");
  // The actions refuse on their own too, not only through a disabled button: W17's keyboard shortcuts
  // will call them with no button in between.
  const snapshot = JSON.stringify(transcript);
  regenerate(transcript[transcript.length - 1]);
  deleteExchange(transcript[0]);
  check("W7 regenerate() and deleteExchange() refuse while generating, when called directly", JSON.stringify(transcript) === snapshot && asked.length === 0, asked.length + " asked");
  $("stop").click(); await idle(); await wait(50);
  const stoppedRegen = bots().at(-1).querySelector(".msg-regen");
  check("W7 a stopped reply can be regenerated", stoppedRegen && !stoppedRegen.hidden && !stoppedRegen.disabled, stoppedRegen?.outerHTML);
  streamAnswer("a3 final");
  stoppedRegen.click(); await idle();
  check("W7 regenerating a stopped reply replaces it", shown() === "u:q1|a:a1|u:q2|a:a2|u:q3|a:a3 final" && !/stopped/.test(bots().at(-1).textContent), shown());

  // ---- edit ----
  const beforeEdit = window.__lastBody;
  yous()[1].querySelector(".msg-edit").click();
  let ta = document.querySelector("#log textarea.edit-box");
  check("W7 Edit opens the message in place, focused, with its text", ta && ta.value === "q2" && document.activeElement === ta && yous()[1].querySelector(".actions").hidden, ta?.value);
  check("W7 while editing: Send and the other actions are disabled", $("send").disabled && [...document.querySelectorAll("#log .msg-edit, #log .msg-delete")].every(b => b.disabled), $("send").disabled);
  $("prompt").value = "sneaks in while editing";
  await send();
  check("W7 the prompt cannot send while a message is being edited", window.__lastBody === beforeEdit && yous().length === 3 && $("prompt").value === "sneaks in while editing", yous().length);
  $("prompt").value = "";
  window.dispatchEvent(new StorageEvent("storage", { key: chatKey() }));
  check("W7 another tab's change does not close an open edit", document.querySelector("#log textarea.edit-box") === ta, !!document.querySelector("#log textarea.edit-box"));
  ta.value = "changed but cancelled";
  ta.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
  check("W7 Escape cancels: nothing changed, nothing sent", shown() === "u:q1|a:a1|u:q2|a:a2|u:q3|a:a3 final" && window.__lastBody === beforeEdit && !$("send").disabled, shown());

  yous()[1].querySelector(".msg-edit").click();
  ta = document.querySelector("#log textarea.edit-box");
  ta.value = "   ";
  document.querySelector("#log .edit-save").click();
  check("W7 saving an empty edit keeps the box open", document.querySelector("#log textarea.edit-box") === ta && saved().includes("u:q2|"), saved());
  ta.value = "q2 edited";
  asked = []; answer = false;
  document.querySelector("#log .edit-save").click();
  check("W7 an edit that would drop later turns asks first, naming how many", asked.length === 1 && /remove the 3 messages/.test(asked[0]), JSON.stringify(asked));
  check("W7 declining keeps the conversation and the edit", document.querySelector("#log textarea.edit-box") === ta && saved() === "u:q1|a:a1|u:q2|a:a2|u:q3|a:a3 final", saved());
  answer = true;
  streamAnswer("a2 new");
  ta.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", ctrlKey: true, bubbles: true }));
  await idle();
  check("W7 saving an edit resends from that message", sent() === "u:q1|a:a1|u:q2 edited", sent());
  check("W7 saving an edit drops what came after it, on screen and saved", shown() === "u:q1|a:a1|u:q2 edited|a:a2 new" && saved() === "u:q1|a:a1|u:q2 edited|a:a2 new", shown() + " / " + saved());

  asked = [];
  yous()[1].querySelector(".msg-edit").click();
  document.querySelector("#log textarea.edit-box").value = "<img src=x onerror=\"window.__pwned=1\">";
  streamAnswer("a2 newer");
  document.querySelector("#log .edit-save").click(); await idle(); await wait(100);
  check("W7 editing the last question replaces only its reply, without asking", asked.length === 0 && bots().length === 2, asked.length + " asked");
  check("W7 a hostile edit is shown as text and stays inert", yous()[1].children[1].textContent === "<img src=x onerror=\"window.__pwned=1\">" && document.querySelectorAll("#log img").length === 0 && window.__pwned === 0, yous()[1].children[1].innerHTML);

  // ---- delete ----
  asked = []; answer = false;
  bots()[0].querySelector(".msg-delete").click();
  check("W7 Delete asks first", asked.length === 1 && /message and its reply/.test(asked[0]), JSON.stringify(asked));
  check("W7 declining keeps the exchange", yous().length === 2 && bots().length === 2, yous().length + "/" + bots().length);
  answer = true;
  bots()[0].querySelector(".msg-delete").click();
  check("W7 deleting a reply removes the whole exchange, on screen and saved", shown().startsWith("u:<img") && saved() === "u:<img src=x onerror=\"window.__pwned=1\">|a:a2 newer", shown() + " / " + saved());
  await turn("q9", "a9");
  yous().at(-1).querySelector(".msg-delete").click();
  check("W7 deleting your message removes its reply too, and Regenerate moves to the new last reply", bots().length === 1 && visibleRegen().length === 1 && visibleRegen()[0].closest(".msg") === bots()[0], shown());
  await turn("next question", "next answer");
  check("W7 after deleting, the next request carries the history as it now stands", sent() === "u:<img src=x onerror=\"window.__pwned=1\">|a:a2 newer|u:next question", sent());

  // New chat while editing cancels the edit
  yous()[0].querySelector(".msg-edit").click();
  $("newchat").click();
  check("W7 New chat while editing cancels the edit and clears", !document.querySelector("#log textarea.edit-box") && document.querySelectorAll("#log .msg").length === 0 && !$("send").disabled, document.querySelectorAll("#log .msg").length);

  // leave a small conversation for the reload phase
  await turn("keep me", "kept");
  await turn("second", "last reply");
`);

// ---- phase 12: after a reload — the actions come back with the conversation ------------------------------
const phase12 = phase(String.raw`
  const until = async (cond, ms = 3000) => { const t0 = Date.now(); while (!cond() && Date.now() - t0 < ms) await wait(10); return cond(); };
  const bots = [...document.querySelectorAll("#log .msg.bot")];
  const regen = [...document.querySelectorAll("#log .msg-regen")].filter(b => !b.hidden);
  check("W7 after reload: Regenerate is back, on the last reply only", bots.length === 2 && regen.length === 1 && regen[0].closest(".msg") === bots[1] && !regen[0].disabled, regen.length);
  check("W7 after reload: Edit and Delete are back on every message", document.querySelectorAll("#log .msg-edit").length === 2 && document.querySelectorAll("#log .msg-delete").length === 4, document.querySelectorAll("#log .msg-edit").length + "/" + document.querySelectorAll("#log .msg-delete").length);
  streamAnswer("regenerated after reload");
  regen[0].click();
  await until(() => $("stop").hidden && /regenerated after reload/.test(document.querySelector("#log .msg.bot:last-of-type")?.textContent || ""));
  const sent = window.__lastBody.messages.filter(m => m.role !== "system").map(m => m.role[0] + ":" + m.content).join("|");
  check("W7 after reload: Regenerate sends the restored history", sent === "u:keep me|a:kept|u:second", sent);
`);

// ---- phase 13: W8 — context meter, real token counts, the wall in words ----------------------------------
const W8_PRELUDE = String.raw`
  const until = async (cond, ms = 3000) => { const t0 = Date.now(); while (!cond() && Date.now() - t0 < ms) await wait(10); return cond(); };
  const idle = () => until(() => $("stop").hidden);
  window.confirm = () => true;
  const MODELS = [{ id: "gate-model", context_window: 1000 }, { id: "no-window" }, { id: "big", context_window: 4000 }];
  let nextReply = { text: "ok", usage: null, status: 200, body: "" };
  window.fetch = async (url, opts) => {
    if (url === "/v1/models") return new Response(JSON.stringify({ object: "list", data: MODELS }), { status: 200 });
    window.__lastBody = JSON.parse(opts.body);
    const r = nextReply;
    if (r.status !== 200) return new Response(r.body, { status: r.status, headers: { "Content-Type": "application/json" } });
    return new Response(new ReadableStream({ async start(c) {
      for (let k = 0; k < r.text.length; k += 3) {
        c.enqueue(enc.encode("data: " + JSON.stringify({ choices: [{ delta: { content: r.text.slice(k, k + 3) } }] }) + "\n\n"));
        await wait(2);
      }
      if (r.usage) c.enqueue(enc.encode("data: " + JSON.stringify({ choices: [], usage: r.usage }) + "\n\n"));
      c.enqueue(enc.encode("data: [DONE]\n\n"));
      c.close();
    } }), { status: 200, headers: { "Content-Type": "text/event-stream" } });
  };
  const reply = async (q, text, usage) => { nextReply = { text, usage, status: 200 }; $("prompt").value = q; await send(); await idle(); await wait(50); };
  const meter = () => ({ hidden: $("ctx").hidden, text: $("ctx-text").textContent, width: $("ctx-fill").style.width, now: $("ctx-bar").getAttribute("aria-valuenow"), warn: $("ctx").classList.contains("warn"), full: $("ctx").classList.contains("full") });
`;
const phase13 = phase(W8_PRELUDE + String.raw`
  $("newchat").click();
  await loadModels("gate-model");
  check("W8 no meter before any reply has reported usage", $("ctx").hidden, JSON.stringify(meter()));

  await reply("first", "a reply streamed in many small chunks", { prompt_tokens: 300, completion_tokens: 100 });
  check("W8 the request asks the server for usage", window.__lastBody?.stream_options?.include_usage === true, JSON.stringify(window.__lastBody?.stream_options));
  let m = meter();
  check("W8 meter shows used of window, from the server's usage", !m.hidden && m.text === "about 400 of 1,000 tokens used (40%)" && m.width === "40%" && m.now === "40" && !m.warn && !m.full, JSON.stringify(m));
  const meta = document.querySelector("#log .msg.bot:last-of-type .meta")?.textContent || "";
  check("W8 per-reply stats count the server's completion_tokens, not stream chunks", /^100 tok · /.test(meta), meta);

  await reply("second", "more", { prompt_tokens: 700, completion_tokens: 120 });
  m = meter();
  check("W8 at 80% or more the meter warns", m.warn && !m.full && /about 820 of 1,000 tokens used \(82%\) — nearing this model's context limit\.$/.test(m.text), JSON.stringify(m));
  await reply("third", "even more", { prompt_tokens: 900, completion_tokens: 60 });
  m = meter();
  check("W8 at 95% or more it says the next message will likely not fit, and what to do", m.full && !m.warn && /\(96%\) — the next message will likely not fit\. Start a new chat, or delete earlier exchanges\.$/.test(m.text), JSON.stringify(m));
  await reply("fourth", "over", { prompt_tokens: 1200, completion_tokens: 50 });
  m = meter();
  check("W8 past 100% the bar is capped, the number is not", m.width === "100%" && /about 1,250 of 1,000 tokens used \(100%\)/.test(m.text), JSON.stringify(m));

  document.querySelectorAll("#log .msg.bot")[3].querySelector(".msg-delete").click();
  m = meter();
  check("W8 deleting the last exchange moves the meter back to the reply before it", m.full && /about 960 of 1,000/.test(m.text), JSON.stringify(m));

  $("model").value = "no-window"; $("model").dispatchEvent(new Event("change"));
  check("W8 a model that publishes no context_window shows no meter", $("ctx").hidden, JSON.stringify(meter()));
  $("model").value = "big"; $("model").dispatchEvent(new Event("change"));
  m = meter();
  check("W8 switching model re-measures against its window", !m.hidden && m.text === "about 960 of 4,000 tokens used (24%)" && !m.warn && !m.full, JSON.stringify(m));
  $("model").value = "gate-model"; $("model").dispatchEvent(new Event("change"));

  // a server that sends no usage chunk: stats fall back to counting chunks, the meter keeps the last known
  await reply("old server", "abcdefghi", null);
  const meta2 = document.querySelector("#log .msg.bot:last-of-type .meta")?.textContent || "";
  check("W8 without usage, stats fall back to counting chunks", /^3 tok · /.test(meta2), meta2);
  check("W8 without usage, the meter keeps the last reported figure", /about 960 of 1,000/.test(meter().text), meter().text);

  // the wall
  const turns = document.querySelectorAll("#log .msg").length;
  nextReply = { status: 400, body: JSON.stringify({ error: { message: "prompt is 1003 tokens but the model's context window is 1000 (context_length_exceeded) <img src=x onerror=\"window.__pwned=1\">", type: "invalid_request_error" } }) };
  $("prompt").value = "one too many";
  await send(); await idle(); await wait(100);
  const errBubble = document.querySelector("#log .msg.err:last-of-type")?.children[1];
  check("W8 context_length_exceeded is explained in words a user can act on", /^This conversation no longer fits in gate-model's context window\. Start a new chat, or delete earlier exchanges to make room\./.test(errBubble?.textContent || "") && /1003 tokens/.test(errBubble.textContent), errBubble?.textContent?.slice(0, 160));
  await wait(150);
  check("W8 the server's message in that error stays inert", document.querySelectorAll("#log img").length === 0 && window.__pwned === 0, document.querySelectorAll("#log img").length);
  nextReply = { status: 400, body: JSON.stringify({ error: { message: "temperature must be between 0 and 2" } }) };
  $("prompt").value = "other 400";
  await send(); await idle(); await wait(50);
  check("W8 other 400s are not dressed up as the context wall", !/no longer fits/.test(document.querySelector("#log .msg.err:last-of-type")?.textContent || ""), document.querySelector("#log .msg.err:last-of-type")?.textContent);

  check("W8 usage is saved with the reply", stored().messages.filter(x => x.usage).map(x => x.usage.prompt_tokens + x.usage.completion_tokens).join(",") === "400,820,960", JSON.stringify(stored().messages.map(x => x.usage)));
`);

// ---- phase 14: after a reload — the meter comes back; hostile stored usage is ignored ----------------------
const phase14 = phase(W8_PRELUDE + String.raw`
  await loadModels("gate-model");
  let m = meter();
  check("W8 after reload the meter is restored from the saved usage", !m.hidden && /about 960 of 1,000 tokens used \(96%\)/.test(m.text) && m.full, JSON.stringify(m));
  const st = stored();
  const lastWithUsage = st.messages.map((x, i) => x.usage ? i : -1).filter(i => i >= 0).pop();
  st.messages[lastWithUsage].usage = { prompt_tokens: "900<img src=x onerror=window.__pwned=1>", completion_tokens: -5 };
  localStorage.setItem(chatKey(), JSON.stringify(st));
  window.dispatchEvent(new StorageEvent("storage", { key: chatKey() }));
  await wait(100);
  m = meter();
  check("W8 unreadable stored usage is ignored, falling back to the previous reply's", /about 820 of 1,000/.test(m.text) && document.querySelectorAll("img").length === 0 && window.__pwned === 0, JSON.stringify(m));
  $("newchat").click();
  check("W8 New chat hides the meter", $("ctx").hidden, JSON.stringify(meter()));
`);

// ---- phase 15: W9 — conversations, the list, generated titles --------------------------------------------
const W9_PRELUDE = String.raw`
  const until = async (cond, ms = 3000) => { const t0 = Date.now(); while (!cond() && Date.now() - t0 < ms) await wait(10); return cond(); };
  const idle = () => until(() => $("stop").hidden);
  let asked = [], answer = true;
  window.confirm = m => { asked.push(m); return answer; };
  const items = () => [...document.querySelectorAll("#chat-list li.chat-item")];
  const titles = () => items().map(li => li.querySelector(".chat-open")?.textContent).join("|");
  const currentTitle = () => document.querySelector("#chat-list li.current .chat-open")?.textContent;
  const v2keys = () => Object.keys(localStorage).filter(k => k.startsWith("goinfer.chat.v2.") && !k.endsWith(".unreadable"));
  const item = t => items().find(li => li.querySelector(".chat-open")?.textContent === t);
  const turn = async (q, a) => { streamAnswer(a); $("prompt").value = q; await send(); await idle(); await wait(30); };
`;
const phase15 = phase(W9_PRELUDE + String.raw`
  for (const k of v2keys()) localStorage.removeItem(k);
  startFresh();
  const headBox = document.querySelector("#chats .chats-head").getBoundingClientRect(), listBox = $("chat-list").getBoundingClientRect();
  check("W9 the conversation list is laid out below its heading, full width — not squeezed into a row beside it", listBox.top >= headBox.bottom - 1 && listBox.width > headBox.width * 0.9, JSON.stringify({ headBottom: headBox.bottom, listTop: listBox.top, listW: listBox.width, headW: headBox.width }));
  check("W9 a fresh start lists one pending New chat, current, with no rename or delete", titles() === "New chat" && items()[0].classList.contains("current") && !items()[0].querySelector(".chat-rename, .chat-delete"), titles());
  const freshId = currentChat.id;
  $("newchat").click();
  check("W9 New chat on an empty chat does nothing", currentChat.id === freshId && items().length === 1, currentChat.id + " / " + items().length);
  save();   // save() has many callers (reload, tab sync, W7) — on an empty new chat it must write nothing
  check("W9 an empty chat is not stored, even when save() is called", v2keys().length === 0, v2keys());

  // ---- titles ----
  window.__titleBodies.length = 0;
  window.__titleReply = { content: "Capital of France", delay: 150 };
  streamAnswer("Paris.");
  $("prompt").value = "What is the capital of France?";
  await send(); await idle();
  check("W9 first message gives an instant provisional title", titles() === "What is the capital of France?" && stored().titled === "first", titles());
  check("W9 the chat is stored once something is said", v2keys().length === 1 && !items()[0].querySelector(".chat-open").textContent.includes("New chat"), v2keys());
  await until(() => titles() === "Capital of France");
  const tb = window.__titleBodies[0];
  check("W9 after the first reply the model is asked for a title — once, non-streaming, short", window.__titleBodies.length === 1 && tb.stream === false && tb.max_tokens === 24 && tb.messages[0].content.endsWith("What is the capital of France?") && tb.model === "gate-model", JSON.stringify(tb)?.slice(0, 200));
  check("W9 the generated title replaces the provisional one, and is saved", titles() === "Capital of France" && stored().title === "Capital of France" && stored().titled === "model", titles() + " / " + JSON.stringify(stored()?.titled));
  check("W9 the title request never becomes the chat's last request", window.__lastBody?.stream === true, JSON.stringify(window.__lastBody)?.slice(0, 80));
  await turn("And of Spain?", "Madrid.");
  await wait(100);
  check("W9 later replies do not ask for a title again", window.__titleBodies.length === 1, window.__titleBodies.length);

  // a reply that is not a usable title: provisional stays, and it is not retried
  $("newchat").click();
  check("W9 New chat adds a pending chat on top and keeps the old one", titles() === "New chat|Capital of France", titles());
  window.__titleReply = { content: "<think>\nThe user wants a title. Let me think about" };
  await turn("x".repeat(30) + " a long first message that will not fit in the list", "ok");
  await until(() => stored()?.titled === "tried");
  check("W9 a thinking-only title reply keeps the provisional title, marked tried", stored().titled === "tried" && currentTitle() === ("x".repeat(30) + " a long first message th").slice(0, 47) + "…", currentTitle() + " / " + stored()?.titled);
  const n = window.__titleBodies.length;
  await turn("again", "ok"); await wait(100);
  check("W9 a tried title is not asked for again", window.__titleBodies.length === n, window.__titleBodies.length + " vs " + n);

  $("newchat").click();
  window.__titleReply = { content: "Title: \"**Rust ownership rules**.\"\nSome explanation after" };
  await turn("explain ownership", "ok");
  await until(() => stored()?.titled === "model");
  check("W9 the model's title is cleaned: first line, no label, quotes, markup or final period", currentTitle() === "Rust ownership rules", currentTitle());

  $("newchat").click();
  window.__titleReply = { content: "<think>\nThe user asks about bread.\n</think>\n\nSourdough starter trouble" };
  await turn("my starter smells odd", "ok");
  await until(() => stored()?.titled !== "first");
  check("W9 a reasoning model that finishes thinking gets its title, without the thinking", currentTitle() === "Sourdough starter trouble" && stored().titled === "model", currentTitle());

  $("newchat").click();
  window.__titleReply = { content: "<img src=x onerror=\"window.__pwned=1\">" };
  await turn("hostile title", "ok");
  await until(() => stored()?.titled === "model");
  await wait(150);
  check("W9 a hostile title is shown as text and stays inert", currentTitle() === "<img src=x onerror=\"window.__pwned=1\">" && document.querySelectorAll("img").length === 0 && window.__pwned === 0, currentTitle());

  // a reply started while the title request is out cancels it; it is asked again after that reply
  $("newchat").click();
  window.__titleReply = { hang: true };
  const before = window.__titleBodies.length;
  await turn("first of two", "one");
  await until(() => window.__titleBodies.length === before + 1);
  window.__titleReply = { content: "Asked again" };
  window.__titleAborts = 0;
  streamAnswer("two", { hang: true });
  $("prompt").value = "second of two";
  const r2 = send();
  await until(() => !$("stop").hidden);
  check("W9 starting a reply cancels the title request that is out", window.__titleAborts === 1, window.__titleAborts);
  $("stop").click(); await r2; await idle();
  await turn("third", "three");
  await until(() => currentTitle() === "Asked again");
  check("W9 a cancelled title is asked for again after a later reply", window.__titleBodies.length === before + 2 && currentTitle() === "Asked again", window.__titleBodies.length - before + " / " + currentTitle());

  // ---- the list: order, switching, busy ----
  check("W9 the list is most recently updated first", titles() === "Asked again|<img src=x onerror=\"window.__pwned=1\">|Sourdough starter trouble|Rust ownership rules|" + ("x".repeat(30) + " a long first message th").slice(0, 47) + "…|Capital of France", titles());
  item("Capital of France").querySelector(".chat-open").click();
  check("W9 opening a conversation shows it and marks it current", currentTitle() === "Capital of France" && [...document.querySelectorAll("#log .msg.you")].map(m => m.children[1].textContent).join("|") === "What is the capital of France?|And of Spain?", currentTitle());
  check("W9 the open conversation is remembered for this tab", sessionStorage.getItem("goinfer.chat.current") === currentChat.id, sessionStorage.getItem("goinfer.chat.current"));
  check("W9 opening a conversation does not reorder the list", titles().startsWith("Asked again|"), titles());
  await turn("And of Italy?", "Rome.");
  check("W9 replying moves a conversation to the top, with its history intact", titles().startsWith("Capital of France|") && window.__lastBody.messages.filter(m => m.role !== "system").length === 5, titles());

  streamAnswer("slow ", { hang: true });
  $("prompt").value = "busy";
  const run = send();
  await until(() => !$("stop").hidden); await wait(50);
  // (while a JOB-backed reply runs the list stays usable and opening another chat detaches it — W29's phase
  // covers that, and the streamed route, where the list is still disabled)
  $("stop").click(); await run; await idle();

  // ---- rename ----
  const renameOrder = titles();
  item("Rust ownership rules").querySelector(".chat-rename").click();
  let box = document.querySelector("#chat-list .chat-rename-box");
  check("W9 Rename opens the title for editing, focused", box && box.value === "Rust ownership rules" && document.activeElement === box, box?.value);
  box.value = "discarded";
  box.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
  check("W9 Escape cancels a rename", titles() === renameOrder, titles());
  item("Rust ownership rules").querySelector(".chat-rename").click();
  box = document.querySelector("#chat-list .chat-rename-box");
  box.value = "  Borrow   checker notes ";
  box.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
  const renamedId = item("Borrow checker notes")?.dataset.id;
  const renamed = renamedId && JSON.parse(localStorage.getItem("goinfer.chat.v2." + renamedId));
  check("W9 Enter saves a rename — tidied, marked user, without moving it in the list", renamed && renamed.title === "Borrow checker notes" && renamed.titled === "user" && titles() === renameOrder.replace("Rust ownership rules", "Borrow checker notes"), titles());
  item("Borrow checker notes").querySelector(".chat-rename").click();
  box = document.querySelector("#chat-list .chat-rename-box");
  box.value = "   ";
  box.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
  check("W9 an empty rename keeps the old title", !!item("Borrow checker notes"), titles());
  // rename the OPEN conversation while it is not the most recent: a rename is not activity
  items()[2].querySelector(".chat-open").click();
  const orderOpen = titles(), openTitle = currentTitle();
  document.querySelector("#chat-list li.current .chat-rename").click();
  box = document.querySelector("#chat-list .chat-rename-box");
  box.value = "Renamed while open";
  box.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
  check("W9 renaming the open conversation does not move it either", titles() === orderOpen.replace(openTitle, "Renamed while open") && currentTitle() === "Renamed while open", titles() + " vs " + orderOpen);

  // a rename made while a title request is out wins over it
  $("newchat").click();
  window.__titleReply = { content: "Model would say this", delay: 400 };
  await turn("race the title", "ok");
  document.querySelector("#chat-list li.current .chat-rename").click();
  box = document.querySelector("#chat-list .chat-rename-box");
  box.value = "Mine";
  box.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
  await wait(600);
  check("W9 a rename wins over a title that arrives after it", currentTitle() === "Mine" && stored().titled === "user", currentTitle());

  // ---- delete ----
  asked = []; answer = false;
  const curBefore = currentChat.id;
  item("Borrow checker notes").querySelector(".chat-delete").click();
  check("W9 Delete asks first; declining keeps it", asked.length === 1 && /cannot be undone/.test(asked[0]) && !!item("Borrow checker notes"), JSON.stringify(asked));
  answer = true;
  const delId = item("Borrow checker notes").dataset.id;
  item("Borrow checker notes").querySelector(".chat-delete").click();
  check("W9 deleting another conversation removes it and its storage, and leaves you where you are", !item("Borrow checker notes") && localStorage.getItem("goinfer.chat.v2." + delId) === null && currentChat.id === curBefore, titles());
  const nextUp = items().find(li => !li.classList.contains("current"))?.querySelector(".chat-open").textContent;
  document.querySelector("#chat-list li.current .chat-delete").click();
  check("W9 deleting the open conversation opens the most recent remaining one", currentTitle() === nextUp && document.querySelectorAll("#log .msg").length > 0, currentTitle() + " vs " + nextUp);

  // ---- another tab, and what storage can hold ----
  localStorage.setItem("goinfer.chat.v2.othertab1", JSON.stringify({ v: 2, id: "othertab1", title: "From another tab", titled: "user", updated: Date.now() + 1000, messages: [{ role: "user", content: "hi" }] }));
  localStorage.setItem("goinfer.chat.v2.brokenchat", "{not json");
  localStorage.setItem("goinfer.chat.v2.Bad-Id!", JSON.stringify({ v: 2, id: "Bad-Id!", title: "invalid id", updated: 1, messages: [] }));
  localStorage.setItem("goinfer.chat.v2.mismatch", JSON.stringify({ v: 2, id: "notmismatch", title: "wrong id inside", updated: 1, messages: [] }));
  const openId = currentChat.id;
  window.dispatchEvent(new StorageEvent("storage", { key: "goinfer.chat.v2.othertab1" }));
  check("W9 a conversation added in another tab appears in the list, without switching this tab", titles().startsWith("From another tab|") && currentChat.id === openId, titles());
  check("W9 unreadable or mislabelled stored conversations are not listed, and are set aside", !/invalid id|wrong id inside/.test(titles()) && localStorage.getItem("goinfer.chat.v2.brokenchat.unreadable") === "{not json" && localStorage.getItem("goinfer.chat.v2.brokenchat") === null && localStorage.getItem("goinfer.chat.v2.mismatch.unreadable") !== null, titles());
  localStorage.removeItem("goinfer.chat.v2.Bad-Id!");
  sessionStorage.setItem("gate.w9.open", currentChat.id);
  sessionStorage.setItem("gate.w9.titles", titles());
`);

// ---- phase 16: after a reload — same conversation open, same list; then plant a W3-era store ------------
const phase16 = phase(W9_PRELUDE + String.raw`
  check("W9 after reload this tab reopens the conversation it had open", currentChat.id === sessionStorage.getItem("gate.w9.open"), currentChat.id);
  check("W9 after reload the list and its titles are unchanged", titles() === sessionStorage.getItem("gate.w9.titles"), titles());
  sessionStorage.setItem("gate.w9.count", String(v2keys().length));
  localStorage.setItem("goinfer.chat.v1", JSON.stringify({ v: 1, messages: [
    { role: "user", content: "a conversation from before W9" }, { role: "assistant", content: "kept", meta: "1 tok" } ] }));
  sessionStorage.removeItem("goinfer.chat.current");
`);

// ---- phase 17: after a reload — the W3 conversation migrated into a conversation of its own -------------
const phase17 = phase(W9_PRELUDE + String.raw`
  check("W9 a W3-era conversation is migrated and opened", [...document.querySelectorAll("#log .msg")].map(m => m.children[1].textContent).join("|") === "a conversation from before W9|kept" && currentTitle() === "a conversation from before W9", currentTitle());
  check("W9 migration removes the old key and adds exactly one conversation", localStorage.getItem("goinfer.chat.v1") === null && v2keys().length === Number(sessionStorage.getItem("gate.w9.count")) + 1, v2keys().length);
  window.__titleBodies.length = 0;
  await turn("carry on", "sure");
  await until(() => window.__titleBodies.length === 1);
  check("W9 a migrated conversation gets its generated title after its next reply", window.__titleBodies.length === 1 && window.__titleBodies[0].messages[0].content.endsWith("a conversation from before W9"), window.__titleBodies.length);
`);

// ---- phase 18: W10 — sampling controls ----------------------------------------------------------------------
const W10_PRELUDE = String.raw`
  const until = async (cond, ms = 3000) => { const t0 = Date.now(); while (!cond() && Date.now() - t0 < ms) await wait(10); return cond(); };
  const idle = () => until(() => $("stop").hidden);
  window.confirm = () => true;
  const setField = (id, v) => { $(id).value = v; $(id).dispatchEvent(new Event("input", { bubbles: true })); };
  const KEYS = ["top_p", "top_k", "seed", "frequency_penalty", "presence_penalty", "stop"];
  const sentSampling = () => { const b = window.__lastBody || {}; const o = {}; for (const k of ["temperature", "max_tokens", ...KEYS]) if (k in b) o[k] = b[k]; return JSON.stringify(o); };
  const turn = async (q, a) => { streamAnswer(a); $("prompt").value = q; await send(); await idle(); await wait(30); };
`;
const phase18 = phase(W10_PRELUDE + String.raw`
  $("sampling-reset").click();
  $("newchat").click();
  check("W10 defaults: temperature 0.7, max 512, the rest blank, nothing marked changed", $("temp").value === "0.7" && $("max").value === "512" && ["top-p", "top-k", "seed", "freq-pen", "pres-pen", "stop"].every(id => $(id).value === "") && $("sampling-state").textContent === "", $("sampling-state").textContent);
  await turn("defaults", "ok");
  check("W10 with defaults, only temperature and max_tokens are sent", sentSampling() === JSON.stringify({ temperature: 0.7, max_tokens: 512 }), sentSampling());

  setField("top-p", "0.9"); setField("top-k", "40"); setField("seed", "42");
  setField("freq-pen", "0.5"); setField("pres-pen", "-0.5");
  setField("stop", "###\n\\n\n\nEND");
  check("W10 the summary counts what changed", $("sampling-state").textContent === "· 6 changed", $("sampling-state").textContent);
  await turn("all set", "ok");
  check("W10 every set field is sent, as numbers, stop as a list with \\n unescaped and blank lines dropped", sentSampling() === JSON.stringify({ temperature: 0.7, max_tokens: 512, top_p: 0.9, top_k: 40, seed: 42, frequency_penalty: 0.5, presence_penalty: -0.5, stop: ["###", "\n", "END"] }), sentSampling());
  await until(() => window.__titleBodies.length > 0);
  const tb = window.__titleBodies.at(-1);
  check("W10 the background title request keeps its own settings", tb && tb.temperature === 0 && tb.max_tokens === 24 && !KEYS.some(k => k in tb), JSON.stringify(tb)?.slice(0, 160));

  setField("top-k", ""); setField("stop", "");
  await turn("two cleared", "ok");
  const sb = JSON.parse(sentSampling());
  check("W10 a cleared field is not sent at all", !("top_k" in sb) && !("stop" in sb) && sb.seed === 42, sentSampling());

  // ---- invalid values: refused before anything changes ----
  for (const [id, v, msg] of [
    ["top-p", "1.5", "top_p must be a number from 0 to 1"],
    ["top-k", "-1", "top_k must be a whole number from 0 to 1000000"],
    ["top-k", "2.5", "top_k must be a whole number from 0 to 1000000"],
    ["seed", "abc", "Seed must be a whole number"],
    ["seed", "99999999999999999999", "Seed must be a whole number"],
    ["freq-pen", "3", "Frequency penalty must be a number from -2 to 2"],
    ["pres-pen", "-2.01", "Presence penalty must be a number from -2 to 2"],
    ["temp", "", "Temperature must be a number from 0 to 2"],
    ["temp", "2.5", "Temperature must be a number from 0 to 2"],
    ["max", "0", "Max tokens must be a whole number from 1 to 131072"],
  ]) {
    const good = $(id).value;
    setField(id, v);
    const bubbles = document.querySelectorAll("#log .msg").length, last = window.__lastBody;
    $("sampling-box").open = false;
    $("prompt").value = "should not send";
    await send(); await wait(30);
    check("W10 " + id + "=" + JSON.stringify(v) + ": marked invalid, and says why", $(id).getAttribute("aria-invalid") === "true" && $(id).classList.contains("invalid") && $("sampling-error").textContent.startsWith(msg) && !$("sampling-error").hidden, $("sampling-error").textContent);
    check("W10 " + id + "=" + JSON.stringify(v) + ": Send refuses — nothing sent, nothing added, the message kept, the field focused", window.__lastBody === last && document.querySelectorAll("#log .msg").length === bubbles && $("prompt").value === "should not send" && document.activeElement === $(id) && /^Fix the settings first/.test($("chat-status").textContent), document.activeElement?.id + " / " + $("chat-status").textContent);
    setField(id, good);
  }
  setField("top-k", "-3");
  const lastDirect = window.__lastBody, bubblesDirect = document.querySelectorAll("#log .msg").length;
  await generate();
  check("W10 generate() refuses a bad setting even when called directly", window.__lastBody === lastDirect && document.querySelectorAll("#log .msg").length === bubblesDirect, document.querySelectorAll("#log .msg").length + " vs " + bubblesDirect);
  setField("top-k", "");
  check("W10 fixing the value clears the mark and the message", [...document.querySelectorAll(".sampling-grid input, #temp, #max, #stop")].every(el => el.getAttribute("aria-invalid") !== "true") && $("sampling-error").hidden, $("sampling-error").textContent);
  check("W10 the closed Sampling box was opened to show the bad field", $("sampling-box").open, $("sampling-box").open);
  $("prompt").value = "";

  // regenerate and edit refuse BEFORE they change the conversation
  setField("top-p", "7");
  const before = stored().messages.length, last = window.__lastBody;
  const regen = [...document.querySelectorAll("#log .msg-regen")].find(b => !b.hidden);
  regen.click(); await wait(50);
  check("W10 Regenerate with a bad setting keeps the reply and sends nothing", stored().messages.length === before && document.querySelectorAll("#log .msg.bot").length === before / 2 && window.__lastBody === last, stored().messages.length + " vs " + before);
  const you = [...document.querySelectorAll("#log .msg.you")][0];
  you.querySelector(".msg-edit").click();
  document.querySelector("#log textarea.edit-box").value = "edited";
  document.querySelector("#log .edit-save").click(); await wait(50);
  check("W10 saving an edit with a bad setting drops nothing and keeps the edit open", stored().messages.length === before && !!document.querySelector("#log textarea.edit-box") && window.__lastBody === last, stored().messages.length + " / " + !!document.querySelector("#log textarea.edit-box"));
  document.querySelector("#log .edit-cancel").click();
  setField("top-p", "0.9");

  // ---- persistence, tab sync, reset ----
  const st = JSON.parse(localStorage.getItem("goinfer.sampling.v1"));
  check("W10 settings are saved as typed", st && st.v === 1 && st.fields["top-p"] === "0.9" && st.fields.seed === "42" && st.fields["pres-pen"] === "-0.5", JSON.stringify(st));
  document.activeElement?.blur();   // the refused Regenerate above left focus in top_p, on purpose
  check("W10 precondition: no sampling field has focus", !["temp", "max", "top-p", "top-k", "seed", "freq-pen", "pres-pen", "stop"].some(id => document.activeElement === $(id)), document.activeElement?.id);
  localStorage.setItem("goinfer.sampling.v1", JSON.stringify({ v: 1, fields: { ...st.fields, seed: "7" } }));
  window.dispatchEvent(new StorageEvent("storage", { key: "goinfer.sampling.v1" }));
  check("W10 an idle page follows another tab's settings", $("seed").value === "7", $("seed").value);
  $("sampling-box").open = true;
  $("seed").focus();
  $("seed").value = "123";
  localStorage.setItem("goinfer.sampling.v1", JSON.stringify({ v: 1, fields: { ...st.fields, seed: "8" } }));
  window.dispatchEvent(new StorageEvent("storage", { key: "goinfer.sampling.v1" }));
  check("W10 another tab's change does not overwrite a field being typed in", $("seed").value === "123", $("seed").value);
  $("seed").blur();
  setField("seed", "42");
  $("sampling-reset").click();
  check("W10 Reset restores the defaults and forgets the saved settings", $("temp").value === "0.7" && $("top-p").value === "" && $("seed").value === "" && localStorage.getItem("goinfer.sampling.v1") === null && $("sampling-state").textContent === "", localStorage.getItem("goinfer.sampling.v1"));

  // leave custom settings for the reload phase, plus one hostile stored value
  setField("temp", "0"); setField("top-k", "5"); setField("stop", "STOP");
  const saved = JSON.parse(localStorage.getItem("goinfer.sampling.v1"));
  saved.fields.seed = { toString: 1 }; saved.fields["freq-pen"] = "x".repeat(5000);
  localStorage.setItem("goinfer.sampling.v1", JSON.stringify(saved));
`);

// ---- phase 19: after a reload — settings restored; unreadable stored values ignored ------------------------
const phase19 = phase(W10_PRELUDE + String.raw`
  check("W10 after reload the settings are restored", $("temp").value === "0" && $("top-k").value === "5" && $("stop").value === "STOP" && $("sampling-state").textContent === "· 3 changed", $("temp").value + "/" + $("top-k").value + "/" + $("sampling-state").textContent);
  check("W10 stored values that are not short strings fall back to defaults", $("seed").value === "" && $("freq-pen").value === "", JSON.stringify([$("seed").value, $("freq-pen").value.length]));
  $("newchat").click();
  await turn("after reload", "ok");
  check("W10 the restored settings are what is sent", sentSampling() === JSON.stringify({ temperature: 0, max_tokens: 512, top_k: 5, stop: ["STOP"] }), sentSampling());
  $("sampling-reset").click();
`);

// ---- phase 20: W11 — images for vision models ------------------------------------------------------------
const W11_PRELUDE = String.raw`
  const until = async (cond, ms = 4000) => { const t0 = Date.now(); while (!cond() && Date.now() - t0 < ms) await wait(10); return cond(); };
  const idle = () => until(() => $("stop").hidden);
  window.confirm = () => true;
  const MODELS = [{ id: "vis", vision: true }, { id: "txt", vision: false }];
  // /v1/models is answered only while a model is picked; every chat request is set up with stream() first.
  // (Not a router wrapped around streamAnswer's stub: the prelude's title wrapper already sits on
  // window.fetch as a getter/setter, and a second wrapper that captures it recurses into itself.)
  const stream = (text, opts) => streamAnswer(text, opts);
  const pick = async id => {
    window.fetch = async () => new Response(JSON.stringify({ object: "list", data: MODELS }), { status: 200 });
    await loadModels(id); await wait(20);
  };
  const makeFile = (type, w, h, name) => new Promise(res => {
    const c = document.createElement("canvas"); c.width = w; c.height = h;
    const g = c.getContext("2d"); g.fillStyle = "#e11"; g.fillRect(0, 0, w / 2, h); g.fillStyle = "#12e"; g.fillRect(w / 2, 0, w / 2, h);
    c.toBlob(b => res(new File([b], name, { type })), type, 0.95);
  });
  const dt = f => { const d = new DataTransfer(); d.items.add(f); return d; };
  const viaInput = f => { $("attach-file").files = dt(f).files; $("attach-file").dispatchEvent(new Event("change")); };
  const viaPaste = f => $("prompt").dispatchEvent(new ClipboardEvent("paste", { clipboardData: dt(f), bubbles: true, cancelable: true }));
  const viaDrop = f => { const d = dt(f); $("pane-chat").dispatchEvent(new DragEvent("dragover", { dataTransfer: d, bubbles: true, cancelable: true })); $("pane-chat").dispatchEvent(new DragEvent("drop", { dataTransfer: d, bubbles: true, cancelable: true })); };
  const dims = url => new Promise(res => { const i = new Image(); i.onload = () => res(i.naturalWidth + "x" + i.naturalHeight); i.onerror = () => res("undecodable"); i.src = url; });
  const imageParts = () => (window.__lastBody?.messages || []).flatMap(m => Array.isArray(m.content) ? m.content.filter(p => p.type === "image_url") : []);
  const shape = () => (window.__lastBody?.messages || []).filter(m => m.role !== "system").map(m => m.role[0] + ":" + (Array.isArray(m.content) ? "[" + m.content.map(p => p.type === "text" ? "text:" + p.text : "image").join(",") + "]" : m.content)).join("|");
  const turn = async (q, a) => { stream(a); $("prompt").value = q; await send(); await idle(); await wait(30); };
`;
const phase20 = phase(W11_PRELUDE + String.raw`
  $("sampling-reset").click();
  $("newchat").click();
  await pick("txt");
  check("W11 a text-only model shows no Attach control", $("attach").hidden && $("attach-preview").hidden, $("attach").hidden);
  await pick("vis");
  check("W11 a vision model shows Attach", !$("attach").hidden, $("attach").hidden);
  check("W11 nothing pending: no <img> anywhere on the page", document.querySelectorAll("img").length === 0, document.querySelectorAll("img").length);

  // ---- attach by the file picker: a small PNG is kept exactly ----
  const small = await makeFile("image/png", 200, 100, "small.png");
  viaInput(small);
  await until(() => pendingImage);
  check("W11 a small PNG is attached as-is, with a preview", pendingImage?.url.startsWith("data:image/png;base64,") && pendingImage.info === "200×100" && !$("attach-preview").hidden && $("attach-preview").querySelector("img")?.src === pendingImage.url, pendingImage?.info);
  const smallURL = pendingImage.url;
  stream("red and blue");
  $("prompt").value = "what colors?";
  await send(); await idle(); await wait(30);
  const you1 = [...document.querySelectorAll("#log .msg.you")].at(-1);
  check("W11 the sent message shows its image", you1.querySelector("img.uimg")?.src === smallURL && you1.children[1].textContent === "what colors?", you1.children[1].innerHTML.slice(0, 80));
  check("W11 the request carries text then the image as content parts", shape() === "u:[text:what colors?,image]" && imageParts()[0].image_url.url === smallURL, shape());
  check("W11 sending clears the pending image and its preview", pendingImage === null && $("attach-preview").hidden && !$("attach-preview").querySelector("img"), String(pendingImage));
  check("W11 the image is saved with its message", stored().messages[0].image === smallURL, String(stored().messages[0].image).slice(0, 40));

  await turn("which is on the left?", "red");
  check("W11 a follow-up still sends the conversation's image, on the message it came with", shape() === "u:[text:what colors?,image]|a:red and blue|u:which is on the left?" && imageParts().length === 1, shape());

  // ---- paste a WebP: converted to JPEG ----
  const webp = await makeFile("image/webp", 300, 300, "x.webp");
  viaPaste(webp);
  await until(() => pendingImage);
  check("W11 a pasted WebP is converted to JPEG (the server decodes PNG and JPEG only)", pendingImage?.url.startsWith("data:image/jpeg;base64,") && /300×300 \(converted to JPEG\)/.test(pendingImage.info) && await dims(pendingImage.url) === "300x300", pendingImage?.info);
  $("attach-remove").click();
  check("W11 Remove drops the pending image", pendingImage === null && $("attach-preview").hidden && document.querySelectorAll("#attach-preview img").length === 0, String(pendingImage));

  // ---- drop a large PNG: scaled ----
  const big = await makeFile("image/png", 3000, 1000, "big.png");
  viaDrop(big);
  await until(() => pendingImage);
  check("W11 a dropped large image is scaled to 1344 on its longest side, as JPEG", pendingImage?.url.startsWith("data:image/jpeg;base64,") && pendingImage.info === "1344×448 (scaled from 3000×1000)" && await dims(pendingImage.url) === "1344x448", pendingImage?.info);
  check("W11 attaching a second image says the model will see only this one", !$("attach-note").hidden && /one image per request/.test($("attach-note").textContent), $("attach-note").textContent);
  const bigURL = pendingImage.url;
  await turn("and this one?", "blue");
  check("W11 only the newest image is sent; the earlier message goes as plain text", imageParts().length === 1 && imageParts()[0].image_url.url === bigURL && shape().startsWith("u:what colors?|a:red and blue|u:which is on the left?|a:red|u:[text:and this one?,image]"), shape());

  // ---- things that are not images ----
  viaInput(new File(["hello"], "a.txt", { type: "text/plain" }));
  await until(() => !$("attach-note").hidden);
  check("W11 a non-image file is refused with a reason", pendingImage === null && $("attach-note").textContent === "That isn't an image file.", $("attach-note").textContent);
  viaInput(new File([new Uint8Array(25_000_001)], "huge.png", { type: "image/png" }));
  await until(() => /too large/.test($("attach-note").textContent));
  check("W11 a file over 25 MB is refused before it is decoded", pendingImage === null && $("attach-note").textContent === "That image is too large (over 25 MB).", $("attach-note").textContent);
  viaInput(new File(["not really a png"], "x.png", { type: "image/png" }));
  await until(() => /couldn't be read/.test($("attach-note").textContent));
  check("W11 a corrupt image is refused with a reason", pendingImage === null && $("attach-note").textContent === "That file couldn't be read as an image.", $("attach-note").textContent);

  // ---- image-only message ----
  viaInput(small); await until(() => pendingImage);
  $("prompt").value = "";
  stream("an image");
  await send(); await idle(); await wait(30);
  check("W11 an image can be sent with no text", shape().endsWith("u:[image]") && [...document.querySelectorAll("#log .msg.you")].at(-1).querySelector("img.uimg"), shape());

  // ---- a model that cannot see ----
  viaInput(small); await until(() => pendingImage);
  await pick("txt");
  check("W11 switching to a text-only model with an image pending hides Attach and says why", $("attach").hidden && !$("attach-note").hidden && /can't see images — remove the image/.test($("attach-note").textContent), $("attach-note").textContent);
  const last = window.__lastBody, nMsgs = document.querySelectorAll("#log .msg").length;
  $("prompt").value = "try anyway";
  await send(); await wait(30);
  check("W11 Send refuses an image to a text-only model: nothing sent, nothing added, message kept", window.__lastBody === last && document.querySelectorAll("#log .msg").length === nMsgs && $("prompt").value === "try anyway", shape());
  $("attach-remove").click();
  check("W11 a text-only model is told the conversation's images won't be sent", /won't be sent to it/.test($("attach-note").textContent), $("attach-note").textContent);
  await turn("text only now", "ok");
  check("W11 a text-only model is sent no image parts at all", imageParts().length === 0 && !JSON.stringify(window.__lastBody).includes("data:image"), shape().slice(0, 120));
  viaInput(small); await wait(200);
  check("W11 attaching on a text-only model is refused", pendingImage === null && /can't see images/.test($("attach-note").textContent), $("attach-note").textContent);
  await pick("vis");

  // ---- busy, and editing a message that has an image ----
  stream("slow ", { hang: true });
  $("prompt").value = "busy";
  const run = send();
  await until(() => !$("stop").hidden);
  viaDrop(small); await wait(200);
  check("W11 nothing can be attached while a reply is generating", pendingImage === null, String(pendingImage));
  $("stop").click(); await run; await idle();
  const withImage = [...document.querySelectorAll("#log .msg.you")].find(m => m.querySelector("img.uimg") && m.children[1].textContent === "");
  withImage.querySelector(".msg-edit").click();
  document.querySelector("#log textarea.edit-box").value = "";
  stream("edited");
  document.querySelector("#log .edit-save").click();
  await idle(); await wait(30);
  check("W11 an image-only message can be edited and resent with no text — the image stays", !document.querySelector("#log textarea.edit-box") && shape().endsWith("u:[image]") && [...document.querySelectorAll("#log .msg.you")].at(-1).querySelector("img.uimg"), shape().slice(-80));

  // leave a conversation with one valid image, and plant hostile stored images, for the reload
  const msgs = stored().messages.slice(0, 2);
  msgs.push({ role: "user", content: "remote", image: "https://tracker.invalid/p.png" },
            { role: "user", content: "script", image: "javascript:window.__pwned=1" },
            { role: "user", content: "svg", image: "data:image/svg+xml;base64,PHN2ZyBvbmxvYWQ9d2luZG93Ll9fcHduZWQ9MT4=" },
            { role: "user", content: "junk", image: "data:image/png;base64,AAAA\"><img src=x onerror=window.__pwned=1>" },
            { role: "assistant", content: "img on a reply", image: smallURL });
  putStored(msgs);
`);

// ---- phase 21: after a reload — images restored; hostile stored images never rendered or sent -------------
const phase21 = phase(W11_PRELUDE + String.raw`
  await pick("vis");
  await wait(200);
  const imgs = [...document.querySelectorAll("#log img")];
  check("W11 after reload the valid image is shown again, on its message", imgs.length === 1 && imgs[0].closest(".msg.you")?.children[1].textContent === "what colors?" && imgs[0].src.startsWith("data:image/png;base64,"), imgs.map(i => i.src.slice(0, 30)));
  check("W11 remote, script, SVG and malformed stored images are dropped, not rendered", !imgs.some(i => !/^data:image\/(png|jpeg);base64,/.test(i.getAttribute("src"))) && window.__pwned === 0, imgs.length + " img, pwned " + window.__pwned);
  await turn("after reload", "ok");
  check("W11 after reload only the valid image is sent", imageParts().length === 1 && imageParts()[0].image_url.url.startsWith("data:image/png;base64,") && !/tracker\.invalid|javascript:|svg\+xml/.test(JSON.stringify(window.__lastBody)), shape().slice(0, 160));
`);

// ---- phase 22: W13 — errors that say what to do --------------------------------------------------------------
// The HTTP errors are REAL serve responses, replayed byte for byte (scripts/webui-gate/captured-errors.json).
const W13_PRELUDE = String.raw`
  const until = async (cond, ms = 4000) => { const t0 = Date.now(); while (!cond() && Date.now() - t0 < ms) await wait(10); return cond(); };
  const idle = () => until(() => $("stop").hidden);
  window.confirm = () => true;
  const CAPTURED = ${JSON.stringify(JSON.parse(readFileSync(resolve(here, "webui-gate", "captured-errors.json"), "utf8")))};
  const reply = c => { window.fetch = async (url, opts) => { window.__lastBody = opts?.body ? JSON.parse(opts.body) : null; window.__calls = (window.__calls || 0) + 1; return new Response(c.body, { status: c.status, headers: c.headers }); }; };
  const lastErr = () => [...document.querySelectorAll("#log .msg.err")].at(-1);
  const errText = () => { const e = lastErr(); return e ? { title: e.querySelector(".err-title")?.textContent, detail: e.querySelector(".err-detail")?.textContent || "", actions: [...e.querySelectorAll(".err-actions button")].map(b => b.textContent).join(",") } : null; };
  const ask = async q => { $("prompt").value = q; await send(); await idle(); await wait(30); };
  const lastBotMeta = () => [...document.querySelectorAll("#log .msg.bot")].at(-1)?.querySelector(".meta")?.textContent || "";
  // a stream of the given SSE data lines; {drop:true} ends it with a network error instead of closing
  const streamOf = (lines, { drop = false } = {}) => { window.fetch = async (url, opts) => { window.__lastBody = JSON.parse(opts.body); return new Response(new ReadableStream({ async start(c) {
    for (const l of lines) { c.enqueue(enc.encode("data: " + (typeof l === "string" ? l : JSON.stringify(l)) + "\n\n")); await wait(5); }
    if (drop) { c.error(new TypeError("network error")); return; }
    c.enqueue(enc.encode("data: [DONE]\n\n")); c.close();
  } }), { status: 200, headers: { "Content-Type": "text/event-stream" } }); }; };
  const delta = t => ({ choices: [{ delta: { content: t }, finish_reason: null }] });
  const fin = r => ({ choices: [{ delta: {}, finish_reason: r }] });
`;
const phase22 = phase(W13_PRELUDE + String.raw`
  $("sampling-reset").click();
  $("newchat").click();
  const byCase = Object.fromEntries(CAPTURED.responses.map(r => [r.case, r]));
  const want = {
    "no API key": ["The server needs an API key.", "Enter it in the Server API key field on the Models tab, then retry.", "Enter API key,Retry"],
    "unknown model": ["The model \"gate-model\" isn't loaded.", "model \"gone\" not found (served: q)", "Refresh models,Retry"],
    "queue full": ["The model is busy: its request queue is full (max 1).", "Try again in a moment (the server suggests 1 s).", "Retry"],   // W28: the real 429 names its depth
    "in-flight cap": ["The server is at capacity.", "server at capacity (max in-flight requests reached); retry. Try again in a moment (the server suggests 1 s).", "Retry"],
    "halted": ["The server has halted new generations.", "Reason: maintenance window. Someone with admin access has to resume it before anything can be generated.", "Retry"],
    "prompt too large": ["This conversation no longer fits in gate-model's context window. Start a new chat, or delete earlier exchanges to make room.", "prompt is too large for the model's context window of 32768 tokens (context_length_exceeded)", "New chat"],
  };
  for (const [name, [title, detail, actions]] of Object.entries(want)) {
    reply(byCase[name]);
    await ask("provoke: " + name);
    const e = errText();
    check("W13 real " + byCase[name].status + " (" + name + "): says what happened and what to do", e && e.title === title && e.detail === detail && e.actions === actions, JSON.stringify(e));
  }
  check("W13 no raw JSON is shown for any of them", ![...document.querySelectorAll("#log .msg.err")].some(m => /\{"error"/.test(m.textContent)), [...document.querySelectorAll("#log .msg.err")].map(m => m.textContent.slice(0, 40)));
  check("W13 a failed request is not a turn: no empty replies were saved", stored().messages.every(m => m.role === "user"), JSON.stringify(stored().messages.map(m => m.role)));

  // ---- the buttons do what they say ----
  reply(byCase["no API key"]);
  await ask("needs a key");
  lastErr().querySelector(".err-key").click();
  check("W13 Enter API key opens the Models tab, where the key field is, and puts the cursor in it", !$("pane-models").hidden && $("pane-chat").hidden && document.activeElement === $("key"), document.activeElement?.id + " / models hidden " + $("pane-models").hidden);
  document.activeElement.blur();
  tab("chat");
  reply(byCase["queue full"]);
  await ask("busy");
  const errCount = document.querySelectorAll("#log .msg.err").length;
  streamAnswer("worked on retry");
  lastErr().querySelector(".err-retry").click();
  await until(() => !$("stop").hidden); await idle(); await wait(30);
  check("W13 Retry removes the error and asks again with the same conversation", document.querySelectorAll("#log .msg.err").length === errCount - 1 && lastBot().children[1].textContent === "worked on retry" && window.__lastBody.messages.at(-1).content === "busy", document.querySelectorAll("#log .msg.err").length + " / " + lastBot()?.children[1].textContent);
  let modelsFetched = 0;
  reply(byCase["unknown model"]);
  await ask("model gone");
  window.fetch = async url => { if (url === "/v1/models") modelsFetched++; return new Response(JSON.stringify({ object: "list", data: [{ id: "gate-model" }] }), { status: 200 }); };
  lastErr().querySelector(".err-models").click();
  await until(() => modelsFetched > 0);
  check("W13 Refresh models asks the server for the model list", modelsFetched === 1, modelsFetched);
  reply(byCase["prompt too large"]);
  await ask("too long");
  const chatBefore = currentChat.id;
  lastErr().querySelector(".err-newchat").click();
  check("W13 New chat on the context wall starts a new conversation", currentChat.id !== chatBefore && document.querySelectorAll("#log .msg").length === 0, currentChat.id === chatBefore);

  // ---- no HTTP answer at all ----
  window.fetch = async () => { throw new TypeError("Failed to fetch"); };
  await ask("server down");
  check("W13 an unreachable server says so, with Retry", JSON.stringify(errText()) === JSON.stringify({ title: "Can't reach the server.", detail: "The request got no answer. Check that goinfer serve is still running, then retry.", actions: "Retry" }), JSON.stringify(errText()));

  // ---- failures part-way through a reply: what arrived is kept ----
  streamOf([delta("Half an ans"), delta("wer"), { error: { message: "generation failed: out of memory <img src=x onerror=\"window.__pwned=1\">", type: "api_error" } }]);
  await ask("error mid-stream");
  check("W13 a server error mid-reply keeps what arrived, marked incomplete with the reason", lastBot().children[1].textContent === "Half an answer" && /incomplete — the server reported an error: generation failed: out of memory .*\. Regenerate to try again\.$/.test(lastBotMeta()), lastBot()?.children[1].textContent + " / " + lastBotMeta());
  check("W13 and it is saved as failed, with the partial answer", stored().messages.at(-1).state === "failed" && stored().messages.at(-1).content === "Half an answer", JSON.stringify(stored().messages.at(-1)));
  const regen = [...document.querySelectorAll("#log .msg-regen")].find(b => !b.hidden);
  check("W13 an incomplete reply can be regenerated", regen && !regen.disabled && regen.closest(".msg") === lastBot(), !!regen);
  await wait(150);
  check("W13 a hostile server message stays inert", document.querySelectorAll("#log img").length === 0 && window.__pwned === 0, document.querySelectorAll("#log img").length);
  streamOf([delta("Cut off by the net")], { drop: true });
  await ask("connection drops");
  check("W13 a dropped connection mid-reply keeps what arrived and says the connection was lost", lastBot().children[1].textContent === "Cut off by the net" && /incomplete — the connection to the server was lost\. Regenerate to try again\.$/.test(lastBotMeta()), lastBotMeta());
  streamOf([{ error: { message: "generation failed: boom", type: "api_error" } }]);
  await ask("error before any text");
  check("W13 a server error before any text is an error message with Retry, not an empty reply", JSON.stringify(errText()) === JSON.stringify({ title: "The server hit an error.", detail: "the server reported an error: generation failed: boom", actions: "Retry" }) && stored().messages.at(-1).role === "user", JSON.stringify(errText()));

  // ---- how a reply ended ----
  streamOf([delta("truncated"), fin("length")]);
  await ask("long answer");
  check("W13 a reply that hit Max tokens says so, and how to get more", /· stopped at the Max tokens limit — raise it to get more$/.test(lastBotMeta()) && !stored().messages.at(-1).state, lastBotMeta());
  streamOf([delta("cancel"), fin("cancelled")]);
  await ask("cancelled");
  check("W13 a reply the server cancelled says so", /cancelled by the server\.$/.test(lastBotMeta()) && stored().messages.at(-1).state === "cancelled", lastBotMeta());
  streamOf([delta("normal"), fin("stop")]);
  await ask("normal");
  check("W13 a normal stop adds nothing", !/incomplete|cancelled|Max tokens/.test(lastBotMeta()), lastBotMeta());

  // ---- the model list ----
  window.fetch = async () => new Response(CAPTURED.responses[0].body, { status: 401, headers: CAPTURED.responses[0].headers });
  await loadModels();
  check("W13 the model list says an API key is needed, not HTTP 401", $("stats").textContent === "API key needed — enter it on the Models tab", $("stats").textContent);
  window.fetch = async () => { throw new TypeError("Failed to fetch"); };
  await loadModels();
  check("W13 the model list says when the server can't be reached", $("stats").textContent === "can't reach the server — is goinfer serve running?", $("stats").textContent);
  window.fetch = async () => new Response(JSON.stringify({ object: "list", data: [{ id: "gate-model" }] }), { status: 200 });
  await loadModels("gate-model");
`);

// ---- phase 23: after a reload — incomplete and cancelled replies keep their labels ----------------------------
const phase23 = phase(W13_PRELUDE + String.raw`
  const metas = [...document.querySelectorAll("#log .msg.bot .meta")].map(m => m.textContent);
  check("W13 after reload an incomplete reply is still labelled, with its reason", metas.some(t => /incomplete — the server reported an error: generation failed: out of memory/.test(t)) && metas.some(t => /incomplete — the connection to the server was lost/.test(t)), JSON.stringify(metas));
  check("W13 after reload the cancelled and max-tokens labels are still there", metas.some(t => /cancelled by the server\.$/.test(t)) && metas.some(t => /stopped at the Max tokens limit/.test(t)), JSON.stringify(metas));
  check("W13 after reload no error message came back — they were never part of the conversation", document.querySelectorAll("#log .msg.err").length === 0, document.querySelectorAll("#log .msg.err").length);
  const st = stored(); st.messages.at(-1).state = "exploded"; st.messages.at(-1).note = "x".repeat(600);
  localStorage.setItem(chatKey(), JSON.stringify(st));
  window.dispatchEvent(new StorageEvent("storage", { key: chatKey() }));
  await wait(50);
  check("W13 an unknown stored state or an oversized note is ignored", !/exploded|xxxx/.test([...document.querySelectorAll("#log .msg.bot .meta")].at(-1)?.textContent || ""), [...document.querySelectorAll("#log .msg.bot .meta")].at(-1)?.textContent);
  save();
  check("W13 an unknown stored state is dropped, not written back on the next save", stored().messages.at(-1).state === undefined && stored().messages.at(-1).note === undefined, JSON.stringify(stored().messages.at(-1)).slice(0, 120));
`);

// ---- phase 24: W14 — export the conversation ---------------------------------------------------------------
const phase24 = phase(String.raw`
  const until = async (cond, ms = 4000) => { const t0 = Date.now(); while (!cond() && Date.now() - t0 < ms) await wait(10); return cond(); };
  window.confirm = () => true;
  // capture what a download would save, instead of saving it
  const downloads = [];
  const realCreate = URL.createObjectURL;
  URL.createObjectURL = blob => { const u = realCreate(blob); downloads.push({ url: u, blob }); return u; };
  const realClick = HTMLAnchorElement.prototype.click;
  HTMLAnchorElement.prototype.click = function () { const d = downloads.find(x => x.url === this.href); if (d) { d.name = this.download; d.rel = this.rel; } };
  const lastDownload = async () => { const d = downloads.at(-1); return d && { name: d.name, type: d.blob.type, text: await d.blob.text() }; };

  $("newchat").click();
  check("W14 export is disabled on an empty chat", $("export-md").disabled && $("export-json").disabled, $("export-md").disabled);
  $("export-md").click();
  check("W14 an empty chat exports nothing", downloads.length === 0, downloads.length);

  // a conversation with every shape an export has to carry
  const c = document.createElement("canvas"); c.width = 4; c.height = 4;
  const png = c.toDataURL("image/png");
  const id = "exportcase1";
  const messages = [
    { role: "user", content: "Explain *ownership*.", image: png },
    { role: "assistant", content: "<think>\nThe user wants Rust.\n</think>\n\nOwnership means **one owner**.", model: "qwen3", meta: "12 tok · 30.0 tok/s · 0.4s" },
    { role: "user", content: "More?" },
    { role: "assistant", content: "Half of an answ", model: "qwen3", meta: "4 tok", state: "failed", note: "the connection to the server was lost" },
  ];
  localStorage.setItem("goinfer.chat.v2." + id, JSON.stringify({ v: 2, id, title: "../../Rust: ownership & borrowing?", titled: "user", updated: Date.now(), messages }));
  openChat(id);
  $("system").value = "Be brief.\nUse examples."; $("system").dispatchEvent(new Event("input", { bubbles: true }));
  check("W14 export is enabled once there is a conversation", !$("export-md").disabled && !$("export-json").disabled, $("export-md").disabled);

  $("export-md").click();
  await until(() => downloads.length === 1);
  const md = await lastDownload();
  const day = new Date().toISOString().slice(0, 10);
  check("W14 Markdown file name is the title made safe, plus the date — never a path", md.name === "rust-ownership-borrowing-" + day + ".md" && md.type.startsWith("text/markdown"), md.name + " / " + md.type);
  const lines = md.text.split("\n");
  check("W14 Markdown starts with the title and when it was exported", lines[0] === "# ../../Rust: ownership & borrowing?" && /^\*Exported from goinfer on \d{4}-\d\d-\d\d \d\d:\d\d UTC · 4 messages\*$/.test(lines[2]), lines.slice(0, 3).join(" | "));
  const body = md.text.slice(md.text.indexOf("\n> **System prompt"));
  const expected = [
    "> **System prompt at export time** (a setting of the page, not recorded per message):", ">", "> Be brief.", "> Use examples.", "",
    "## You", "", "Explain *ownership*.", "", "![attached image](" + png + ")", "",
    "## qwen3", "", "<details><summary>Thinking</summary>", "", "The user wants Rust.", "", "</details>", "", "Ownership means **one owner**.", "", "*12 tok · 30.0 tok/s · 0.4s*", "",
    "## You", "", "More?", "",
    "## qwen3", "", "Half of an answ", "", "*4 tok · incomplete — the connection to the server was lost. Regenerate to try again.*", "",
  ].join("\n");
  check("W14 Markdown carries each turn: your text and image, the model's name, thinking folded, the answer, and each reply's stats and state", body === "\n" + expected, JSON.stringify(body).slice(0, 400));

  $("export-json").click();
  await until(() => downloads.length === 2);
  const js = await lastDownload();
  let parsed = null; try { parsed = JSON.parse(js.text); } catch {}
  check("W14 JSON file name and type", js.name === "rust-ownership-borrowing-" + day + ".json" && js.type === "application/json", js.name + " / " + js.type);
  check("W14 JSON is the conversation as stored, with a format tag, version and export time", parsed && parsed.format === "goinfer.chat" && parsed.version === 1 && !isNaN(Date.parse(parsed.exported_at)) && parsed.title === "../../Rust: ownership & borrowing?" && JSON.stringify(parsed.messages) === JSON.stringify(stored().messages), js.text.slice(0, 200));
  check("W14 JSON labels the system prompt as the one set at export time", parsed && parsed.system_prompt_at_export === "Be brief.\nUse examples.", parsed && JSON.stringify(parsed.system_prompt_at_export));

  // no system prompt, no title
  $("system").value = ""; $("system").dispatchEvent(new Event("input", { bubbles: true }));
  const id2 = "exportcase2";
  localStorage.setItem("goinfer.chat.v2." + id2, JSON.stringify({ v: 2, id: id2, title: "", titled: "user", updated: Date.now(), messages: [{ role: "user", content: "日本語だけ" }] }));
  openChat(id2);
  $("export-md").click();
  await until(() => downloads.length === 3);
  const md2 = await lastDownload();
  check("W14 a title with nothing file-safe falls back to 'conversation', and no system prompt means no quote block", md2.name === "conversation-" + day + ".md" && md2.text.startsWith("# Conversation\n") && !md2.text.includes("System prompt"), md2.name + " / " + md2.text.slice(0, 60));
  $("export-json").click();
  await until(() => downloads.length === 4);
  check("W14 JSON with no system prompt says null, not an empty string", JSON.parse((await lastDownload()).text).system_prompt_at_export === null, "");

  // busy
  streamAnswer("slow ", { hang: true });
  $("prompt").value = "busy";
  const run = send();
  await until(() => !$("stop").hidden);
  check("W14 export is disabled while a reply is generating", $("export-md").disabled && $("export-json").disabled, $("export-md").disabled);
  $("stop").click(); await run; await until(() => $("stop").hidden);
  check("W14 and enabled again afterwards", !$("export-md").disabled, $("export-md").disabled);
  URL.createObjectURL = realCreate; HTMLAnchorElement.prototype.click = realClick;
`);

// ---- phases 25–27: W15 — theme, and contrast measured in every mode ----------------------------------------
// Contrast is computed from what the browser actually resolved: each text colour against the real
// backgrounds it sits on — the page, a card, a message, a text field (every stop of its gradient) — and white
// on the Send button. WCAG AA for text: 4.5:1. The browser's colour scheme is switched from Node (DevTools
// Emulation), between phases, because a page cannot emulate its own media query.
const W15_PRELUDE = String.raw`
  const parse = c => {
    let m = c.match(/^rgba?\(([\d.]+),\s*([\d.]+),\s*([\d.]+)(?:,\s*([\d.]+))?\)$/);
    if (m) return { r: m[1] / 255, g: m[2] / 255, b: m[3] / 255, a: m[4] === undefined ? 1 : +m[4] };
    m = c.match(/^color\(srgb ([\d.e-]+) ([\d.e-]+) ([\d.e-]+)(?: \/ ([\d.]+))?\)$/);
    if (m) return { r: +m[1], g: +m[2], b: +m[3], a: m[4] === undefined ? 1 : +m[4] };
    return null;
  };
  const lum = c => { const f = v => v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); return 0.2126 * f(c.r) + 0.7152 * f(c.g) + 0.0722 * f(c.b); };
  const ratio = (a, b) => { const x = lum(a), y = lum(b); return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05); };
  // every solid colour a surface is painted with: its background-color if opaque, and each gradient stop
  const surfaceColors = el => {
    const cs = getComputedStyle(el), out = [];
    const bg = parse(cs.backgroundColor); if (bg && bg.a > 0.99) out.push(bg);
    for (const m of cs.backgroundImage.matchAll(/(rgba?\([^)]*\)|color\(srgb [^)]*\))/g)) { const c = parse(m[1]); if (c && c.a > 0.99) out.push(c); }
    return out;
  };
  // resolve a token where el sits (a <textarea> cannot hold a probe element, so its parent stands in)
  const tokenOn = (el, token) => { const host = /^(TEXTAREA|INPUT|SELECT)$/.test(el.tagName) ? el.parentElement : el; const s = document.createElement("span"); s.style.color = "var(" + token + ")"; host.appendChild(s); const c = parse(getComputedStyle(s).color); s.remove(); return c; };
  const TEXT = ["--gi-ink", "--gi-muted", "--gi-accent", "--gi-err", "--gi-ok", "--gi-warn-ink"];
  const worst = () => {
    const surfaces = { page: document.body, card: document.querySelector(".card"), header: document.querySelector("header"), message: document.querySelector("#log .msg.bot") || document.querySelector(".card"), "text field": $("prompt") };
    let low = { r: 99 };
    for (const [sn, el] of Object.entries(surfaces)) for (const bg of surfaceColors(el)) for (const t of TEXT) {
      const fg = tokenOn(el, t), r = ratio(fg, bg); if (r < low.r) low = { r, what: t + " on " + sn + " " + JSON.stringify([fg, bg].map(c => [c.r, c.g, c.b].map(v => Math.round(v * 255)))) };
    }
    const field = $("prompt"), typed = parse(getComputedStyle(field).color);   // what you type, on the field itself
    for (const bg of surfaceColors(field)) { const r = ratio(typed, bg); if (r < low.r) low = { r, what: "typed text in the message field" }; }
    const send = $("send"), ink = parse(getComputedStyle(send).color);
    for (const bg of surfaceColors(send)) { const r = ratio(ink, bg); if (r < low.r) low = { r, what: "Send label on its button" }; }
    return low;
  };
  const isDark = () => lum(parse(getComputedStyle(document.body).backgroundColor)) < 0.2;
  const setTheme = v => { $("theme").value = v; $("theme").dispatchEvent(new Event("change")); };
  // Found live 2026-09-16: dark mode's shading read as nearly flat — WCAG text contrast (worst(),
  // above) held throughout, because that regression was never in a text colour. AmbientCSS renders
  // elevation as box-shadow layers, each pure black (a drop shadow) or pure white (a highlight) at
  // some alpha; the alpha itself IS the perceptibility (RGB is always 0 or 255 either way). The
  // highlight carried nearly all of the lost contrast when this broke (measured: 0.976 fixed vs
  // 0.38 broken on .card's own strongest highlight layer) — the drop-shadow layers barely moved.
  const strongestHighlightAlpha = el => {
    let max = 0;
    for (const m of getComputedStyle(el).boxShadow.matchAll(/rgba\(255, 255, 255, ([\d.]+)\)/g)) max = Math.max(max, +m[1]);
    return max;
  };
`;
const phase25 = phase(W15_PRELUDE + String.raw`
  // a message on screen, so a message surface is measured too
  streamAnswer("Some **answer** text.");
  $("prompt").value = "contrast"; await send(); await new Promise(r => setTimeout(r, 150));
  check("W15 default is System, with no theme stored", $("theme").value === "system" && !document.documentElement.hasAttribute("data-theme") && localStorage.getItem("goinfer.theme.v1") === null, $("theme").value);
  check("W15 System under a light preference is light", !isDark() && getComputedStyle(document.documentElement).colorScheme === "light", getComputedStyle(document.body).backgroundColor);
  let w = worst();
  check("W15 light: every text colour on every surface meets WCAG AA (worst " + w.r.toFixed(2) + ":1, " + w.what + ")", w.r >= 4.5, w.r.toFixed(2) + " " + w.what);
  setTheme("dark");
  check("W15 choosing Dark switches the page to dark, saved", isDark() && document.documentElement.dataset.theme === "dark" && localStorage.getItem("goinfer.theme.v1") === "dark" && getComputedStyle(document.documentElement).colorScheme === "dark", getComputedStyle(document.body).backgroundColor);
  w = worst();
  check("W15 Dark chosen: every text colour on every surface meets WCAG AA (worst " + w.r.toFixed(2) + ":1, " + w.what + ")", w.r >= 4.5, w.r.toFixed(2) + " " + w.what);
  setTheme("system");
  check("W15 back to System forgets the choice and follows the (light) preference", !isDark() && localStorage.getItem("goinfer.theme.v1") === null && !document.documentElement.hasAttribute("data-theme"), localStorage.getItem("goinfer.theme.v1"));
  localStorage.setItem("goinfer.theme.v1", "<script>");
  window.dispatchEvent(new StorageEvent("storage", { key: "goinfer.theme.v1" }));
  check("W15 an unknown stored theme is treated as System", $("theme").value === "system" && !document.documentElement.hasAttribute("data-theme"), $("theme").value + " / " + document.documentElement.getAttribute("data-theme"));
  localStorage.removeItem("goinfer.theme.v1");
`);
const phase26 = phase(W15_PRELUDE + String.raw`
  check("W15 System under a dark preference is dark, with nothing stored", isDark() && $("theme").value === "system" && getComputedStyle(document.documentElement).colorScheme === "dark", getComputedStyle(document.body).backgroundColor);
  let w = worst();
  check("W15 dark (system): every text colour on every surface meets WCAG AA (worst " + w.r.toFixed(2) + ":1, " + w.what + ")", w.r >= 4.5, w.r.toFixed(2) + " " + w.what);
  const hi = strongestHighlightAlpha(document.querySelector(".card"));
  check("W15 dark: a card's shading is actually perceptible, not just WCAG-legal — its strongest highlight layer is well above the 0.38 alpha the old dimming produced", hi > 0.7, hi);
  setTheme("light");
  check("W15 choosing Light overrides a dark system preference", !isDark() && getComputedStyle(document.documentElement).colorScheme === "light", getComputedStyle(document.body).backgroundColor);
  w = worst();
  check("W15 Light chosen under a dark system: WCAG AA holds (worst " + w.r.toFixed(2) + ":1, " + w.what + ")", w.r >= 4.5, w.r.toFixed(2) + " " + w.what);
  window.__otherTab = true;
  localStorage.setItem("goinfer.theme.v1", "dark");
  window.dispatchEvent(new StorageEvent("storage", { key: "goinfer.theme.v1" }));
  check("W15 another tab's theme choice is followed", isDark() && $("theme").value === "dark", $("theme").value);
`);
const phase27 = phase(W15_PRELUDE + String.raw`
  check("W15 after reload a chosen theme is applied before anything else", document.documentElement.dataset.theme === "dark" && $("theme").value === "dark" && isDark(), document.documentElement.getAttribute("data-theme"));
  setTheme("system");
`);

// ---- phases 28–30: W16 — phone layout ----------------------------------------------------------------------
// The viewport is emulated from Node (a real mobile metrics override, so the layout viewport is what a phone
// gives, and it can be blown wider by content — which is exactly the bug). The page is filled with the
// content most likely to break a narrow layout: an unbroken 300-character URL, a very long code line, a wide
// table, an error with buttons, a pending image, an open edit box.
const W16_BODY = String.raw`
  const until = async (cond, ms = 4000) => { const t0 = Date.now(); while (!cond() && Date.now() - t0 < ms) await wait(10); return cond(); };
  window.confirm = () => true;
  window.fetch = async () => new Response(JSON.stringify({ object: "list", data: [{ id: "a-model-with-a-rather-long-name-q4_k_m", vision: true, context_window: 8192, decode_path: "cuda-resident (int4)" }] }), { status: 200 });
  await loadModels("a-model-with-a-rather-long-name-q4_k_m");
  const url = "https://example.com/" + "a".repeat(300);
  const id = "phonecase";
  localStorage.setItem("goinfer.chat.v2." + id, JSON.stringify({ v: 2, id, title: "A very long conversation title that will not fit on a phone screen at all", titled: "user", updated: Date.now(), messages: [
    { role: "user", content: "Look at " + url },
    { role: "assistant", model: "a-model-with-a-rather-long-name-q4_k_m", content: "<think>\nhm\n</think>\n\nSee " + url + "\n\n\x60\x60\x60\n" + "x".repeat(400) + "\n\x60\x60\x60\n\n| " + Array.from({ length: 12 }, (_, i) => "column" + i).join(" | ") + " |\n|" + "---|".repeat(12) + "\n| " + Array.from({ length: 12 }, () => "value").join(" | ") + " |", meta: "96 tok · 31.2 tok/s · 3.1s", usage: { prompt_tokens: 7000, completion_tokens: 900 } },
    { role: "user", content: "second" },
    { role: "assistant", model: "m", content: "cut off", meta: "2 tok", state: "failed", note: "the connection to the server was lost" } ] }));
  openChat(id);
  showProblem(bubble("m", "bot amb-surface-convex amb-elevation-1"), problemFor(401, "{}"), () => {});
  const cv = document.createElement("canvas"); cv.width = 2; cv.height = 2;
  pendingImage = { url: cv.toDataURL("image/png"), info: "2×2" }; showAttach();
  $("system-box").open = true; $("sampling-box").open = true;
  [...document.querySelectorAll("#log .msg.you")][1].querySelector(".msg-edit").click();
  await wait(150);

  const W = innerWidth;
  check("W16 @" + W + ": the page does not scroll sideways (layout " + document.documentElement.scrollWidth + " px)", document.documentElement.scrollWidth <= W && W === window.__expectW, document.documentElement.scrollWidth + " vs " + W + " (expected " + window.__expectW + ")");
  // anything past the right edge must be inside a container that scrolls on its own
  const scroller = el => { for (let e = el.parentElement; e; e = e.parentElement) { const o = getComputedStyle(e).overflowX; if (o === "auto" || o === "scroll" || o === "hidden") return true; } return false; };
  const escapes = [...document.querySelectorAll("body *")].filter(e => { if (!e.getClientRects().length) return false; const b = e.getBoundingClientRect(); return (b.right > W + 1 || b.left < -1) && !scroller(e); }).map(e => e.tagName + "." + e.className);
  check("W16 @" + W + ": nothing reaches past the screen edge, except inside its own scroll box (code, tables)", escapes.length === 0, JSON.stringify(escapes.slice(0, 5)));
  const code = document.querySelector("#log pre"), table = document.querySelector("#log table");
  // really scroll them: a container that merely clips would report the same sizes, and hide the rest of the line
  code.scrollLeft = 1e6; table.scrollLeft = 1e6;
  check("W16 @" + W + ": the long code line and the wide table can be scrolled to their end, inside themselves", code.scrollLeft > 0 && table.scrollLeft > 0, code.scrollLeft + " / " + table.scrollLeft);
  code.scrollLeft = 0; table.scrollLeft = 0;
  const controls = [...document.querySelectorAll("button, select, input:not([type=file]), textarea, summary")].filter(e => e.getClientRects().length && getComputedStyle(e).visibility !== "hidden");
  const tiny = controls.map(e => { const b = e.getBoundingClientRect(); return { e: e.id || e.className || e.tagName, w: Math.round(b.width), h: Math.round(b.height) }; }).filter(x => x.w < 24 || x.h < 24);
  check("W16 @" + W + ": every visible control is at least 24×24 px (WCAG 2.5.8) — " + controls.length + " checked", tiny.length === 0, JSON.stringify(tiny.slice(0, 6)));
  const rect = sel => document.querySelector(sel).getBoundingClientRect();
  const head = ["header .brand", "header nav", "#stats", ".book-link", ".theme-pick"].map(sel => [sel, rect(sel)]);
  const overlaps = [];
  for (let i = 0; i < head.length; i++) for (let j = i + 1; j < head.length; j++) { const [a, A] = head[i], [b, B] = head[j]; if (A.left < B.right - 1 && B.left < A.right - 1 && A.top < B.bottom - 1 && B.top < A.bottom - 1) overlaps.push(a + " × " + b); }
  check("W16 @" + W + ": header items do not overlap", overlaps.length === 0, JSON.stringify(overlaps));
  check("W16 @" + W + ": the model stats get a row of their own, not a squeezed column", rect("#stats").width >= rect("header").width * 0.8, Math.round(rect("#stats").width) + " of " + Math.round(rect("header").width));
  const inView = ["#send", "#prompt", "#model", "#tab-chat", "#tab-models", "#theme", "#newchat"].filter(sel => { const b = rect(sel); return b.left < 0 || b.right > W || b.width === 0; });
  check("W16 @" + W + ": Send, the message box, model, tabs, theme and New chat are fully on screen", inView.length === 0, JSON.stringify(inView));
  check("W16 @" + W + ": the theme select is usable, not collapsed", rect("#theme").width >= 60, rect("#theme").width);
  check("W16 @" + W + ": the disclosure toggles keep their triangle (display list-item)", [...document.querySelectorAll("details > summary")].filter(e => e.getClientRects().length).every(e => getComputedStyle(e).display === "list-item"), [...document.querySelectorAll("details > summary")].map(e => getComputedStyle(e).display));
  finishEdit(null); pendingImage = null; showAttach();
`;
const phase28 = phase("window.__expectW = 400;" + W16_BODY);
const phase29 = phase("window.__expectW = 360;" + W16_BODY);
const phase30 = phase(String.raw`
  window.fetch = async () => new Response(JSON.stringify({ object: "list", data: [{ id: "m", decode_path: "cpu" }] }), { status: 200 });
  await loadModels("m");
  const top = sel => document.querySelector(sel).getBoundingClientRect().top;
  check("W16 desktop: the header is still one row", Math.abs(top("header .brand") - top("header nav")) < 12 && Math.abs(top("#stats") - top("header nav")) < 12, [top("header .brand"), top("#stats"), top("header nav")].map(Math.round));
  check("W16 desktop: the phone spacing does not apply (cards keep 18 px padding, the header 20 px)", getComputedStyle(document.querySelector(".card")).paddingLeft === "18px" && getComputedStyle(document.querySelector("header")).paddingLeft === "20px", getComputedStyle(document.querySelector(".card")).paddingLeft + " / " + getComputedStyle(document.querySelector("header")).paddingLeft);
  check("W16 desktop: the content column is still 920 px wide", Math.round(document.querySelector("main").getBoundingClientRect().width) === 920, document.querySelector("main").getBoundingClientRect().width);
`);

// ---- phases 31–32: W17 — Enter sends (a setting), ↑ edits last, Esc stops ------------------------------
const W17_PRELUDE = String.raw`
  const until = async (cond, ms = 4000) => { const t0 = Date.now(); while (!cond() && Date.now() - t0 < ms) await wait(10); return cond(); };
  const idle = () => until(() => $("stop").hidden);
  window.confirm = () => true;
  // count sends by the request body streamAnswer's stub records — not by wrapping fetch again, which the
  // prelude's title wrapper (a getter/setter on window.fetch) turns into a loop through itself
  let sent = 0;
  const answer = t => streamAnswer(t);
  // a keydown as a browser delivers it; ime:true is a key pressed while an input method is composing
  const key = (el, k, { ctrl = false, meta = false, shift = false, alt = false, ime = false, code229 = false } = {}) => {
    const ev = new KeyboardEvent("keydown", { key: k, ctrlKey: ctrl, metaKey: meta, shiftKey: shift, altKey: alt, isComposing: ime, bubbles: true, cancelable: true });
    if (code229) Object.defineProperty(ev, "keyCode", { get: () => 229 });
    el.dispatchEvent(ev);
    return ev.defaultPrevented;
  };
  const typeAndPress = async (text, opts) => { const before = window.__lastBody; $("prompt").value = text; const p = key($("prompt"), "Enter", opts); await wait(60); await idle(); await wait(30); if (window.__lastBody !== before) sent++; return p; };
`;
const phase31 = phase(W17_PRELUDE + String.raw`
  $("newchat").click();
  answer("ok");
  check("W17 default: Enter does not send, and the hint says Ctrl/Cmd+Enter", !$("enter-sends").checked && /Ctrl\/Cmd\+Enter to send/.test($("prompt").placeholder), $("prompt").placeholder);
  let prevented = await typeAndPress("plain enter");
  check("W17 default: Enter makes a new line — nothing sent, the key not swallowed", sent === 0 && !prevented && $("prompt").value === "plain enter", sent + " sent, prevented " + prevented);
  await typeAndPress("ctrl enter", { ctrl: true });
  check("W17 Ctrl+Enter sends", sent === 1 && $("prompt").value === "", sent);
  await typeAndPress("cmd enter", { meta: true });
  check("W17 Cmd+Enter sends", sent === 2, sent);

  $("enter-sends").click();
  check("W17 turning on Enter sends is saved, and the hint changes", $("enter-sends").checked && localStorage.getItem("goinfer.keys.v1") === "enter" && /Enter to send, Shift\+Enter for a new line/.test($("prompt").placeholder), $("prompt").placeholder);
  await typeAndPress("enter now sends");
  check("W17 with the setting on, Enter sends", sent === 3, sent);
  prevented = await typeAndPress("shift enter", { shift: true });
  check("W17 Shift+Enter is still a new line", sent === 3 && !prevented && $("prompt").value === "shift enter", sent);
  prevented = await typeAndPress("alt enter", { alt: true });
  check("W17 Alt+Enter does not send", sent === 3 && !prevented, sent);
  prevented = await typeAndPress("にほんご", { ime: true });
  check("W17 Enter while an input method is composing never sends (isComposing)", sent === 3 && !prevented && $("prompt").value === "にほんご", sent);
  prevented = await typeAndPress("かな", { code229: true });
  check("W17 ... nor when the browser reports the composing key code 229", sent === 3 && !prevented, sent);
  $("prompt").value = "";

  // ↑ edits the last message
  $("prompt").value = "draft";
  prevented = key($("prompt"), "ArrowUp");
  check("W17 ↑ with text in the box does nothing (the cursor moves, as usual)", !prevented && !document.querySelector("#log textarea.edit-box"), prevented);
  $("prompt").value = "";
  prevented = key($("prompt"), "ArrowUp");
  const box = document.querySelector("#log textarea.edit-box");
  const lastMine = [...document.querySelectorAll("#log .msg.you")].at(-1);
  check("W17 ↑ in an empty box opens your LAST message for editing, focused", prevented && box && box.closest(".msg") === lastMine && box.value === "enter now sends" && document.activeElement === box, box?.value);
  // the edit box follows the same setting
  prevented = key(box, "Enter", { shift: true });
  check("W17 in the edit box, Shift+Enter is a new line", !prevented && document.querySelector("#log textarea.edit-box") === box, prevented);
  prevented = key(box, "Enter", { ime: true });
  check("W17 in the edit box, Enter while composing does not save", !prevented && document.querySelector("#log textarea.edit-box") === box, prevented);
  box.value = "edited by keyboard";
  answer("after edit");
  const beforeEdit = window.__lastBody;
  key(box, "Enter");
  await until(() => !document.querySelector("#log textarea.edit-box")); await idle(); await wait(30);
  if (window.__lastBody !== beforeEdit) sent++;
  check("W17 in the edit box, Enter saves and resends (setting on)", [...document.querySelectorAll("#log .msg.you")].at(-1).children[1].textContent === "edited by keyboard" && sent === 4, sent);
  $("enter-sends").click();
  key($("prompt"), "ArrowUp");
  const box2 = document.querySelector("#log textarea.edit-box");
  prevented = key(box2, "Enter");
  check("W17 with the setting off, plain Enter in the edit box is a new line", !prevented && document.querySelector("#log textarea.edit-box") === box2, prevented);
  key(box2, "Escape");
  check("W17 Esc still cancels an edit", !document.querySelector("#log textarea.edit-box"), !!document.querySelector("#log textarea.edit-box"));

  // rename box and an input method
  document.querySelector("#chat-list li.current .chat-rename").click();
  const rn = document.querySelector("#chat-list .chat-rename-box");
  rn.value = "名前";
  prevented = key(rn, "Enter", { ime: true });
  check("W17 the rename box does not commit on Enter while composing", !prevented && document.querySelector("#chat-list .chat-rename-box") === rn, prevented);
  key(rn, "Escape");

  // Esc stops a reply
  streamAnswer("slow ", { hang: true });
  $("prompt").value = "to be stopped";
  const run = send();
  await until(() => !$("stop").hidden);
  prevented = key(document.body, "Escape");
  // with a deadline: if Esc does NOT stop it, the stream never ends, and this must fail rather than hang
  const stoppedInTime = await Promise.race([run.then(() => true), wait(5000).then(() => false)]);
  if (!stoppedInTime) { $("stop").click(); await run; }
  await idle();
  check("W17 Esc stops a reply that is generating", stoppedInTime && prevented && /^stopped/.test([...document.querySelectorAll("#log .msg.bot")].at(-1).querySelector(".meta")?.textContent || ""), [...document.querySelectorAll("#log .msg.bot")].at(-1).querySelector(".meta")?.textContent);
  check("W17 Esc with nothing generating does nothing", !key(document.body, "Escape"), "");
  $("enter-sends").click();   // on, for the reload phase
`);
const phase32 = phase(W17_PRELUDE + String.raw`
  check("W17 after reload the Enter-sends setting is kept", $("enter-sends").checked && /Enter to send/.test($("prompt").placeholder), $("enter-sends").checked);
  localStorage.removeItem("goinfer.keys.v1");
  window.dispatchEvent(new StorageEvent("storage", { key: "goinfer.keys.v1" }));
  check("W17 another tab's change to the setting is followed", !$("enter-sends").checked && /Ctrl\/Cmd\+Enter/.test($("prompt").placeholder), $("enter-sends").checked);
`);

// ---- phases 33–34: W18 — which model answered each turn --------------------------------------------------
const W18_PRELUDE = String.raw`
  const until = async (cond, ms = 4000) => { const t0 = Date.now(); while (!cond() && Date.now() - t0 < ms) await wait(10); return cond(); };
  const idle = () => until(() => $("stop").hidden);
  window.confirm = () => true;
  const MODELS = [{ id: "alpha-4b", decode_path: "cpu (int8)" }, { id: "beta-9b", decode_path: "cuda-resident (int4)" }, { id: "gamma", }];
  const pick = async id => { window.fetch = async () => new Response(JSON.stringify({ object: "list", data: MODELS }), { status: 200 }); await loadModels(id); };
  const turn = async (q, a) => { streamAnswer(a); $("prompt").value = q; await send(); await idle(); await wait(30); };
  const labels = () => [...document.querySelectorAll("#log .msg.bot .who")].map(w => w.firstChild?.textContent + "|" + (w.querySelector(".who-path")?.textContent || ""));
  const dividers = () => [...document.querySelectorAll("#log .model-switch")].map(d => d.textContent);
  const order = () => [...document.querySelectorAll("#log > *")].map(el => el.classList.contains("model-switch") ? "—" : el.classList.contains("you") ? "u" : "a").join("");
`;
const phase33 = phase(W18_PRELUDE + String.raw`
  $("sampling-reset").click();
  $("newchat").click();
  await pick("alpha-4b");
  await turn("one", "first");
  check("W18 a reply is labelled with its model and the compute path it ran on", JSON.stringify(labels()) === JSON.stringify(["alpha-4b|cpu (int8)"]), JSON.stringify(labels()));
  check("W18 the path is saved with the reply", stored().messages[1].path === "cpu (int8)" && stored().messages[1].model === "alpha-4b", JSON.stringify(stored().messages[1]));
  await turn("two", "second");
  check("W18 no divider while the model stays the same", dividers().length === 0, JSON.stringify(dividers()));
  await pick("beta-9b");
  await turn("three", "third");
  check("W18 a divider marks where the model changed, just before the new model's reply", JSON.stringify(dividers()) === JSON.stringify(["Model changed: alpha-4b → beta-9b"]) && order() === "uauau—a", order() + " " + JSON.stringify(dividers()));
  check("W18 the new reply carries the new model and its path", labels().at(-1) === "beta-9b|cuda-resident (int4)", JSON.stringify(labels()));
  check("W18 the divider is a separator, and is not part of the conversation", document.querySelector("#log .model-switch").getAttribute("role") === "separator" && stored().messages.length === 6, stored().messages.length);

  // regenerate the last reply on the first model: relabelled, and the divider goes
  await pick("alpha-4b");
  streamAnswer("third, again");
  [...document.querySelectorAll("#log .msg-regen")].find(b => !b.hidden).click();
  await until(() => !$("stop").hidden); await idle(); await wait(30);
  check("W18 regenerating on another model relabels the reply, and the divider follows the conversation", labels().at(-1) === "alpha-4b|cpu (int8)" && dividers().length === 0, JSON.stringify(labels()) + " " + JSON.stringify(dividers()));

  // a model that publishes no path
  await pick("gamma");
  await turn("four", "fourth");
  check("W18 a model with no decode_path is labelled by name only", labels().at(-1) === "gamma|" && stored().messages.at(-1).path === undefined && dividers().length === 1, JSON.stringify(labels()));

  // a failed request to a different model leaves no divider behind
  await pick("beta-9b");
  window.fetch = async () => new Response(JSON.stringify({ error: { message: "boom" } }), { status: 500 });
  $("prompt").value = "fails"; await send(); await idle(); await wait(30);
  check("W18 a request that failed before any text leaves no model divider", dividers().length === 1 && !!document.querySelector("#log .msg.err"), JSON.stringify(dividers()));

  // export
  const downloads = [];
  const realCreate = URL.createObjectURL;
  URL.createObjectURL = blob => { downloads.push(blob); return realCreate(blob); };
  const realClick = HTMLAnchorElement.prototype.click; HTMLAnchorElement.prototype.click = function () {};
  $("export-md").click();
  await until(() => downloads.length);
  const md = await downloads[0].text();
  URL.createObjectURL = realCreate; HTMLAnchorElement.prototype.click = realClick;
  check("W18 the Markdown export names each reply's model and path", md.includes("\n## alpha-4b · cpu (int8)\n") && md.includes("\n## gamma\n"), md.slice(0, 300));

  // hostile stored labels
  const st = stored();
  st.messages[1].model = "<img src=x onerror=\"window.__pwned=1\">";
  st.messages[1].path = "<b onmouseover=window.__pwned=1>p</b>";
  st.messages[3].path = "x".repeat(300);
  localStorage.setItem(chatKey(), JSON.stringify(st));
`);
const phase34 = phase(W18_PRELUDE + String.raw`
  await wait(200);
  const whos = [...document.querySelectorAll("#log .msg.bot .who")];
  // replies: hostile-model, alpha, alpha, gamma — two model changes; the failed request left its user message, no reply
  check("W18 after reload labels and dividers come back from the saved conversation", whos.length === 4 && dividers().length === 2 && order() === "uau—auau—au", order() + " " + JSON.stringify(labels()));
  check("W18 hostile stored model and path names are shown as text", whos[0].firstChild.textContent === "<img src=x onerror=\"window.__pwned=1\">" && whos[0].querySelector(".who-path").textContent === "<b onmouseover=window.__pwned=1>p</b>" && document.querySelectorAll("#log img, #log b").length === 0 && window.__pwned === 0, whos[0].innerHTML);
  check("W18 an over-long stored path is dropped", !whos[1].querySelector(".who-path"), whos[1].innerHTML);
`);

// ---- phases 35–36: W27–W29 — replies as server jobs: re-attach, place in line, carry on in the background -----
// The jobs traffic replayed here is REAL serve output (scripts/webui-gate/captured-jobs.json).
const W27_PRELUDE = String.raw`
  const until = async (cond, ms = 4000) => { const t0 = Date.now(); while (!cond() && Date.now() - t0 < ms) await wait(10); return cond(); };
  const idle = () => until(() => $("stop").hidden);
  window.confirm = () => true;
  const CAP = ${JSON.stringify(JSON.parse(readFileSync(resolve(here, "webui-gate", "captured-jobs.json"), "utf8")))};
  const JOB = JSON.parse(CAP.submit.body).id;
  // a scripted jobs server: every request is logged; per-phase code sets how each job answers
  const calls = [];
  const jobs = {};   // id -> {status, queue?, events: [lines] | "hang", result, usage, error}
  let nextSubmit = null;
  const sseResponse = (lines, signal) => new Response(new ReadableStream({ async start(c) {
    signal?.addEventListener("abort", () => { try { c.error(new DOMException("aborted", "AbortError")); } catch {} });
    if (lines === "hang") return;
    for (const l of lines) { c.enqueue(enc.encode(l + "\n\n")); await wait(3); }
    c.close();
  } }), { status: 200, headers: { "Content-Type": "text/event-stream" } });
  window.__jobsEmulator = async (url, opts) => {
    const method = (opts && opts.method) || "GET";
    if (typeof url !== "string" || !url.startsWith("/v1/jobs")) return null;
    calls.push(method + " " + url);
    if (url === "/v1/jobs" && method === "POST") {
      window.__lastBody = JSON.parse(opts.body);
      const s = nextSubmit || { status: 202, body: CAP.submit.body };
      if (s.status !== 202) return new Response(s.body, { status: s.status, headers: s.headers || {} });
      return new Response(s.body, { status: 202, headers: { "Content-Type": "application/json" } });
    }
    const m = url.match(/^\/v1\/jobs\/(job_[0-9a-f]+)(\/events)?$/);
    const j = m && jobs[m[1]];
    if (!j) return new Response(CAP.unknown_job.body, { status: 404 });
    if (method === "DELETE") { j.deleted = true; return new Response(JSON.stringify({ id: m[1], cancelled: true }), { status: 200 }); }
    if (m[2]) return sseResponse(typeof j.events === "function" ? j.events() : j.events, opts && opts.signal);
    const rec = { id: m[1], model: "gate-model", status: j.status };
    if (j.queue) rec.queue = j.queue;
    if (j.result) rec.result = j.result;
    if (j.usage) rec.usage = j.usage;
    if (j.error) rec.error = j.error;
    return new Response(JSON.stringify(rec), { status: 200 });
  };
  const realEvents = CAP.events.split("\n").filter(l => l.startsWith("data: "));
  const lastBotEl = () => [...document.querySelectorAll("#log .msg.bot")].at(-1);
  const ask = async q => { $("prompt").value = q; await settle(send()); await idle(); await wait(40); };
  // every await on a reply has a deadline: a reply that never ends must fail the phase, not hang the gate
  function settle(p, ms = 6000) { return Promise.race([p, wait(ms).then(() => { check("deadline: a reply settled within " + ms + " ms", false, "it did not — the page is stuck"); })]); }
`;
const phase35 = phase(W27_PRELUDE + String.raw`
  $("sampling-reset").click();
  $("newchat").click();

  // ---- W27: a text reply is a job, streamed from its events — real serve output ----
  jobs[JOB] = { status: "running", events: realEvents };
  await ask("say hello in french");
  check("W27 a text reply is submitted as a job, then streamed from that job's events", calls[0] === "POST /v1/jobs" && calls.includes("GET /v1/jobs/" + JOB + "/events") && window.__lastBody.messages.at(-1).content === "say hello in french", JSON.stringify(calls));
  check("W27 the real job stream renders the answer, and its closing chunks give the stats (usage) and a normal end", lastBotEl().children[1].textContent.trim() === "Bonjour!" && /^\d+ tok · /.test(lastBotEl().querySelector(".meta").textContent) && stored().messages.at(-1).usage?.prompt_tokens === JSON.parse(realEvents.at(-2).slice(6)).usage.prompt_tokens, lastBotEl().textContent.slice(0, 80) + " / " + JSON.stringify(stored().messages.at(-1).usage));
  check("W27 the job id is saved with the reply", stored().messages.at(-1).job === JOB && !stored().messages.at(-1).state, JSON.stringify(stored().messages.at(-1)).slice(0, 120));

  // ---- W28: waiting in line — the real waiting status's queue field ----
  calls.length = 0;
  jobs[JOB] = { status: "pending", queue: { position: 2, waiting: 3, running: true }, events: "hang" };
  $("prompt").value = "wait your turn";
  const waitRun = send();
  await until(() => lastBotEl()?.querySelector(".queue-note"));
  check("W28 a waiting reply says where it stands in line", lastBotEl().querySelector(".queue-note")?.textContent === "Waiting for its turn: 2nd in line (3 waiting, 1 running)." && $("chat-status").textContent === "waiting in line…", lastBotEl().children[1].textContent);
  jobs[JOB].queue = CAP.waiting_status.queue;   // the REAL field: position 1 of 1, one running
  await until(() => /1st in line/.test(lastBotEl().querySelector(".queue-note")?.textContent || ""), 2500);
  check("W28 the place in line updates as the queue moves (real status: 1st of 1)", lastBotEl().querySelector(".queue-note")?.textContent === "Waiting for its turn: 1st in line (1 waiting, 1 running).", lastBotEl().children[1].textContent);

  // ---- W27: Stop cancels the job on the server ----
  $("stop").click(); await settle(waitRun); await idle();
  check("W27 Stop on a job-backed reply cancels the job on the server (DELETE), not just the stream", jobs[JOB].deleted === true && /^stopped/.test(lastBotEl().querySelector(".meta").textContent) && !lastBotEl().querySelector(".queue-note"), JSON.stringify(calls));

  // ---- a full queue at submit: the real 429 ----
  nextSubmit = { status: 429, body: CAP.full_queue_submit.body, headers: { "Retry-After": CAP.full_queue_submit.retry_after } };
  $("prompt").value = "queue is full"; await send(); await idle(); await wait(30);
  check("W27 a submit refused by a full queue gets W13's explanation, from the real 429, with W28's depth", [...document.querySelectorAll("#log .msg.err")].at(-1)?.querySelector(".err-title")?.textContent === "The model is busy: its request queue is full (max 2).", [...document.querySelectorAll("#log .msg.err")].at(-1)?.textContent);
  nextSubmit = null;

  // ---- a reply with an image stays on the streaming route — and stays tied to the tab ----
  window.fetch = async () => new Response(JSON.stringify({ object: "list", data: [{ id: "gate-model", vision: true }] }), { status: 200 });
  await loadModels("gate-model");
  const cv = document.createElement("canvas"); cv.width = 2; cv.height = 2;
  pendingImage = { url: cv.toDataURL("image/png"), info: "2×2" };
  calls.length = 0;
  streamAnswer("streamed ", { hang: true });
  $("prompt").value = "with an image";
  const imgRun = send();
  await until(() => !$("stop").hidden); await wait(50);
  check("W27 a reply carrying an image uses /v1/chat/completions, not a job", calls.length === 0 && Array.isArray(window.__lastBody.messages.at(-1).content), JSON.stringify(calls));
  check("W27/W9 during a streamed (non-job) reply the conversation list is disabled", [...document.querySelectorAll("#chat-list button")].every(b => b.disabled) && $("newchat").disabled, [...document.querySelectorAll("#chat-list button")].filter(b => !b.disabled).length);
  const streamedChat = currentChat.id;
  openChat("nonexistent");
  check("W9 openChat refuses during a streamed reply, even called directly", currentChat.id === streamedChat, currentChat.id);
  $("stop").click(); await settle(imgRun); await idle();

  // ---- W29: a job-backed reply carries on in the background ----
  // (a fresh conversation: one that holds an image keeps using the streaming route, since a job cannot carry it)
  $("newchat").click();
  const JOB2 = "job_" + "b".repeat(32);
  nextSubmit = { status: 202, body: JSON.stringify({ id: JOB2, status: "pending" }) };
  jobs[JOB2] = { status: "running", events: "hang" };
  $("prompt").value = "a long answer";
  const bgRun = send();
  await until(() => generating && generating.job === JOB2);
  await wait(60);
  const bgChat = currentChat.id;
  check("W29 while a job-backed reply runs, the conversation list and New chat stay usable", !$("newchat").disabled && [...document.querySelectorAll("#chat-list li:not(.current) button")].every(b => !b.disabled), $("newchat").disabled);
  $("newchat").click();
  await settle(bgRun); await idle();
  // the detached job's catch runs AFTER the switch, against a global status element — it must not clobber
  // the conversation now on screen with the OLD reply's outcome (a real "stopped" text on a fresh, idle chat)
  check("W29 leaving a job-backed reply does not leave the new conversation's status saying it was stopped", $("chat-status").textContent === "", JSON.stringify($("chat-status").textContent));
  check("W29 leaving the conversation does not cancel the job", !jobs[JOB2].deleted && currentChat.id !== bgChat, JSON.stringify(calls.filter(c => c.startsWith("DELETE"))));
  const kept = JSON.parse(localStorage.getItem("goinfer.chat.v2." + bgChat));
  check("W29 the reply is kept as still generating, with its job id", kept.messages.at(-1).job === JOB2 && kept.messages.at(-1).state === "generating" && kept.running === true, JSON.stringify(kept.messages.at(-1)).slice(0, 120));
  const badge = [...document.querySelectorAll("#chat-list li")].find(li => li.dataset.id === bgChat)?.querySelector(".chat-running");
  check("W29 the list marks that conversation as having a reply in progress", badge?.textContent === "reply in progress", [...document.querySelectorAll("#chat-list li")].map(li => li.textContent).join(" | "));

  // come back while it is still running: re-attach, replaying from the start
  jobs[JOB2] = { status: "running", events: ["data: " + JSON.stringify({ choices: [{ delta: { content: "The whole " } }] }), "data: " + JSON.stringify({ choices: [{ delta: { content: "answer." } }] }), "data: " + JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] }), "data: [DONE]"] };
  openChat(bgChat);
  await until(() => !generating && /The whole answer\./.test(lastBotEl()?.textContent || ""));
  await idle(); await wait(40);
  check("W27/W29 opening the conversation again re-attaches: the stream replays and the reply completes", lastBotEl().children[1].textContent.trim() === "The whole answer." && !JSON.parse(localStorage.getItem("goinfer.chat.v2." + bgChat)).messages.at(-1).state, lastBotEl().textContent.slice(0, 80));

  // finished while away: take the job's record (the REAL finished status)
  const JOB3 = CAP.finished_status.id;
  nextSubmit = { status: 202, body: JSON.stringify({ id: JOB3, status: "pending" }) };
  jobs[JOB3] = { status: "running", events: "hang" };
  $("prompt").value = "answer while I am away";
  const awayRun = send();
  await until(() => generating && generating.job === JOB3); await wait(40);
  const awayChat = currentChat.id;
  $("newchat").click(); await settle(awayRun); await idle();
  jobs[JOB3] = { status: "done", result: CAP.finished_status.result, usage: CAP.finished_status.usage, events: [] };
  openChat(awayChat);
  await until(() => /Bonjour!/.test(lastBotEl()?.textContent || ""));
  await wait(40);
  check("W29 a reply that finished while you were away is filled in from the job's real record", lastBotEl().children[1].textContent.trim() === "Bonjour!" && /finished while you were away/.test(lastBotEl().querySelector(".meta").textContent) && stored().messages.at(-1).usage?.completion_tokens === CAP.finished_status.usage.completion_tokens, lastBotEl().textContent.slice(0, 100));

  // a job the server no longer has
  const JOB4 = "job_" + "d".repeat(32);
  const lostId = "lostjob1";
  localStorage.setItem("goinfer.chat.v2." + lostId, JSON.stringify({ v: 2, id: lostId, title: "lost", titled: "user", updated: Date.now(), running: true, messages: [{ role: "user", content: "q" }, { role: "assistant", model: "gate-model", content: "part", state: "generating", job: JOB4 }] }));
  openChat(lostId);
  await until(() => /no longer has this job/.test(lastBotEl()?.querySelector(".meta")?.textContent || ""));
  check("W27 a job the server no longer has (the real 404) is labelled, not left spinning", /not finished here — the server no longer has this job/.test(lastBotEl().querySelector(".meta").textContent), lastBotEl().querySelector(".meta").textContent);

  // a failed job, and a cancelled one (real record)
  const JOB5 = "job_" + "e".repeat(32);
  jobs[JOB5] = { status: "failed", error: "generation failed: out of memory", result: { content: "", finish_reason: "" } };
  const failId = "failjob1";
  localStorage.setItem("goinfer.chat.v2." + failId, JSON.stringify({ v: 2, id: failId, title: "fail", titled: "user", updated: Date.now(), messages: [{ role: "user", content: "q" }, { role: "assistant", model: "gate-model", content: "", state: "generating", job: JOB5 }] }));
  openChat(failId);
  await until(() => /incomplete/.test(lastBotEl()?.querySelector(".meta")?.textContent || ""));
  check("W29 a job that failed while away is shown as failed, with the server's reason", /incomplete — the server reported an error: generation failed: out of memory/.test(lastBotEl().querySelector(".meta").textContent), lastBotEl().querySelector(".meta")?.textContent);
  const JOB6 = CAP.cancelled_status.id;
  jobs[JOB6] = { status: CAP.cancelled_status.status, error: CAP.cancelled_status.error };
  const cancId = "cancjob1";
  localStorage.setItem("goinfer.chat.v2." + cancId, JSON.stringify({ v: 2, id: cancId, title: "canc", titled: "user", updated: Date.now(), messages: [{ role: "user", content: "q" }, { role: "assistant", model: "gate-model", content: "", state: "generating", job: JOB6 }] }));
  openChat(cancId);
  await until(() => /cancelled by the server/.test(lastBotEl()?.querySelector(".meta")?.textContent || ""));
  check("W29 a job cancelled while queued (real record) says so, with its reason", lastBotEl().querySelector(".meta").textContent === "cancelled by the server (cancelled before a turn was granted).", lastBotEl().querySelector(".meta")?.textContent);

  // Resume: the server was unreachable when the conversation opened
  const JOB7 = "job_" + "f".repeat(32);
  const resId = "resumejob";
  localStorage.setItem("goinfer.chat.v2." + resId, JSON.stringify({ v: 2, id: resId, title: "resume", titled: "user", updated: Date.now(), messages: [{ role: "user", content: "q" }, { role: "assistant", model: "gate-model", content: "partial", state: "generating", job: JOB7 }] }));
  const emu = window.__jobsEmulator;
  window.__jobsEmulator = async () => { throw new TypeError("Failed to fetch"); };
  openChat(resId);
  await wait(150);
  check("W27 with the server unreachable, the reply stays 'not finished here' and offers Resume", /not finished here — the job may still be running/.test(lastBotEl().querySelector(".meta").textContent) && !!lastBotEl().querySelector(".msg-resume"), lastBotEl().querySelector(".meta")?.textContent);
  window.__jobsEmulator = emu;
  jobs[JOB7] = { status: "running", events: ["data: " + JSON.stringify({ choices: [{ delta: { content: "Resumed in full." } }] }), "data: [DONE]"] };
  lastBotEl().querySelector(".msg-resume").click();
  await until(() => /Resumed in full\./.test(lastBotEl()?.textContent || "") && !generating);
  await idle(); await wait(40);
  check("W27 Resume re-attaches once the server is back, replaying the whole reply", lastBotEl().children[1].textContent.trim() === "Resumed in full." && !lastBotEl().querySelector(".msg-resume"), lastBotEl().textContent.slice(0, 80));

  // leave a job running mid-reply for the reload phase
  const JOB8 = "job_" + "a".repeat(32);
  nextSubmit = { status: 202, body: JSON.stringify({ id: JOB8, status: "pending" }) };
  jobs[JOB8] = { status: "running", events: ["data: " + JSON.stringify({ choices: [{ delta: { content: "first half " } }] })].concat(["hangmarker"]) };
  jobs[JOB8].events = "hang";
  $("prompt").value = "survive a reload";
  send();
  await until(() => generating && generating.job === JOB8);
  await wait(1200);   // let the periodic save run
  sessionStorage.setItem("gate.w27.chat", currentChat.id);
`);
const phase36 = phase(W27_PRELUDE + String.raw`
  const id = sessionStorage.getItem("gate.w27.chat");
  check("W27 after a reload mid-reply, the conversation is back with its reply marked not finished here", currentChat.id === id && /not finished here/.test(lastBotEl()?.querySelector(".meta")?.textContent || "") && stored().messages.at(-1).job === "job_" + "a".repeat(32), lastBotEl()?.querySelector(".meta")?.textContent);
  jobs["job_" + "a".repeat(32)] = { status: "running", events: ["data: " + JSON.stringify({ choices: [{ delta: { content: "The reply, " } }] }), "data: " + JSON.stringify({ choices: [{ delta: { content: "replayed after the reload." } }] }), "data: " + JSON.stringify({ choices: [], usage: { prompt_tokens: 9, completion_tokens: 6, total_tokens: 15 } }), "data: [DONE]"] };
  lastBotEl().querySelector(".msg-resume").click();
  await until(() => /replayed after the reload\./.test(lastBotEl()?.textContent || "") && !generating);
  await idle(); await wait(40);
  check("W27 re-attaching after the reload replays the job from the start and completes the reply", lastBotEl().children[1].textContent.trim() === "The reply, replayed after the reload." && stored().messages.at(-1).usage?.completion_tokens === 6 && !stored().messages.at(-1).state, lastBotEl().textContent.slice(0, 100));
  // a stored job id that is not a job id is never requested, and never offered for Resume
  calls.length = 0;
  const badId = "badjobid";
  localStorage.setItem("goinfer.chat.v2." + badId, JSON.stringify({ v: 2, id: badId, title: "bad", titled: "user", updated: Date.now(), messages: [{ role: "user", content: "q" }, { role: "assistant", model: "gate-model", content: "x", state: "generating", job: "../../admin/halt" }] }));
  openChat(badId);
  await wait(200);
  check("W27 a stored job id that is not a job id is dropped: nothing requested, no Resume", calls.length === 0 && !lastBotEl().querySelector(".msg-resume") && stored().messages.at(-1).job === undefined, JSON.stringify(calls));
`);

// ---- phase 37: W30 — a Batch tab over J4's /v1/files + /v1/batches --------------------------------
const phase37 = phase(String.raw`
  const until = async (cond, ms = 4000) => { const t0 = Date.now(); while (!cond() && Date.now() - t0 < ms) await wait(10); return cond(); };
  // capture what a download would save, instead of saving it (same technique as W14's phase 24)
  const downloads = [];
  const realCreate = URL.createObjectURL;
  URL.createObjectURL = blob => { const u = realCreate(blob); downloads.push({ url: u, blob }); return u; };
  HTMLAnchorElement.prototype.click = function () { const d = downloads.find(x => x.url === this.href); if (d) { d.name = this.download; d.rel = this.rel; } };
  const lastDownload = async () => { const d = downloads.at(-1); return d && { name: d.name, type: d.blob.type, text: await d.blob.text() }; };
  const viaInput = f => { const d = new DataTransfer(); d.items.add(f); $("batch-file").files = d.files; $("batch-file").dispatchEvent(new Event("change")); };

  tab("batch");
  check("W30 the Batch tab is its own pane, Chat and Models hidden", !$("pane-batch").hidden && $("pane-chat").hidden && $("pane-models").hidden && $("tab-batch").getAttribute("aria-selected") === "true", $("pane-batch").hidden);
  check("W30 Run is disabled with no file chosen", $("batch-run").disabled, $("batch-run").disabled);

  // ---- the happy path: upload, run, watch progress, download both outputs ----
  const calls = [];
  let batchPolls = 0;
  window.fetch = async (url, opts) => {
    calls.push((opts && opts.method || "GET") + " " + url);
    if (url === "/v1/files" && opts.method === "POST") {
      window.__uploadHeaders = opts.headers; window.__uploadIsFormData = opts.body instanceof FormData;
      return new Response(JSON.stringify({ id: "file_in1", object: "file", bytes: 123, created_at: 1, filename: "in.jsonl", purpose: "batch" }), { status: 200 });
    }
    if (url === "/v1/batches" && opts.method === "POST") {
      window.__lastBatchBody = JSON.parse(opts.body);
      return new Response(JSON.stringify({ id: "batch_1", object: "batch", status: "in_progress", request_counts: { total: 3, completed: 0, failed: 0 } }), { status: 200 });
    }
    if (url === "/v1/batches/batch_1") {
      batchPolls++;
      if (batchPolls === 1) return new Response(JSON.stringify({ id: "batch_1", status: "in_progress", request_counts: { total: 3, completed: 1, failed: 0 } }), { status: 200 });
      if (batchPolls === 2) return new Response(JSON.stringify({ id: "batch_1", status: "in_progress", request_counts: { total: 3, completed: 2, failed: 0 } }), { status: 200 });
      return new Response(JSON.stringify({ id: "batch_1", status: "completed", request_counts: { total: 3, completed: 2, failed: 1 }, output_file_id: "file_out1", error_file_id: "file_err1" }), { status: 200 });
    }
    if (url === "/v1/files/file_out1/content") return new Response('{"custom_id":"a","response":{"status_code":200,"body":{}}}\n{"custom_id":"b","response":{"status_code":200,"body":{}}}\n', { status: 200 });
    if (url === "/v1/files/file_err1/content") return new Response('{"custom_id":"c","error":{"code":"api_error","message":"boom"}}\n', { status: 200 });
    return new Response(JSON.stringify({ error: { message: "unexpected " + url } }), { status: 500 });
  };
  viaInput(new File(['{"custom_id":"a","method":"POST","url":"/v1/chat/completions","body":{}}\n'], "in.jsonl", { type: "application/jsonl" }));
  check("W30 a chosen file enables Run", !$("batch-run").disabled, $("batch-run").disabled);
  $("batch-run").click();
  await until(() => calls.includes("POST /v1/files"));
  check("W30 the file is uploaded as multipart, not JSON (no Content-Type forced, real FormData body)", calls[0] === "POST /v1/files" && window.__uploadIsFormData === true && !("Content-Type" in (window.__uploadHeaders || {})), JSON.stringify(calls) + " / formdata " + window.__uploadIsFormData + " / headers " + JSON.stringify(window.__uploadHeaders));
  await until(() => window.__lastBatchBody);
  check("W30 the batch is created against the uploaded file, on the one supported endpoint", window.__lastBatchBody?.input_file_id === "file_in1" && window.__lastBatchBody?.endpoint === "/v1/chat/completions", JSON.stringify(window.__lastBatchBody));
  check("W30 Run and the file input are disabled while it runs", $("batch-run").disabled && $("batch-file").disabled, $("batch-run").disabled);
  await until(() => batchPolls >= 1);
  await wait(1100);
  check("W30 the progress bar and status follow the real request_counts while it polls", parseFloat($("batch-bar").style.width) > 0 && parseFloat($("batch-bar").style.width) < 100 && /\d+ of 3 done/.test($("batch-status").textContent), $("batch-bar").style.width + " / " + $("batch-status").textContent);
  await until(() => /finished/.test($("batch-status").textContent), 6000);
  check("W30 the finished line names ok vs failed, and the bar reaches 100%", $("batch-status").textContent === "finished — 3 of 3 done (2 ok, 1 failed)" && parseFloat($("batch-bar").style.width) === 100 && $("batch-status").className === "note err", $("batch-status").textContent + " / " + $("batch-bar").style.width + " / " + $("batch-status").className);
  check("W30 Run and the file input re-enable once it's done, Cancel hides", !$("batch-run").disabled && !$("batch-file").disabled && $("batch-cancel").hidden, $("batch-run").disabled + " / " + $("batch-cancel").hidden);
  const pollsAtFinish = batchPolls;
  await wait(2200);
  check("W30 polling really stops once finished — no further GET after the terminal status", batchPolls === pollsAtFinish, batchPolls + " vs " + pollsAtFinish + " at finish");
  check("W30 both downloads are offered when there is an error file", !$("batch-dl-output").hidden && !$("batch-dl-errors").hidden, $("batch-dl-output").hidden + " / " + $("batch-dl-errors").hidden);

  $("batch-dl-output").click();
  await until(() => downloads.length === 1);
  let d = await lastDownload();
  check("W30 the results download is the real file content, under a name naming this batch", d?.name === "batch_1_output.jsonl" && d?.text === '{"custom_id":"a","response":{"status_code":200,"body":{}}}\n{"custom_id":"b","response":{"status_code":200,"body":{}}}\n', d?.name + " / " + d?.text?.length);
  $("batch-dl-errors").click();
  await until(() => downloads.length === 2);
  d = await lastDownload();
  check("W30 the errors download is the real error file content", d?.name === "batch_1_error.jsonl" && d?.text === '{"custom_id":"c","error":{"code":"api_error","message":"boom"}}\n', d?.name + " / " + d?.text);

  // ---- no failures at all: only the results download is offered ----
  batchPolls = 0;
  window.fetch = async (url, opts) => {
    if (url === "/v1/files" && opts.method === "POST") return new Response(JSON.stringify({ id: "file_in2" }), { status: 200 });
    if (url === "/v1/batches" && opts.method === "POST") return new Response(JSON.stringify({ id: "batch_2", status: "in_progress", request_counts: { total: 1, completed: 0, failed: 0 } }), { status: 200 });
    if (url === "/v1/batches/batch_2") return new Response(JSON.stringify({ id: "batch_2", status: "completed", request_counts: { total: 1, completed: 1, failed: 0 }, output_file_id: "file_out2" }), { status: 200 });
    return new Response("{}", { status: 200 });
  };
  viaInput(new File(["{}"], "clean.jsonl"));
  $("batch-run").click();
  await until(() => /finished/.test($("batch-status").textContent), 6000);
  check("W30 no error file: only the results download is offered, and the line reads clean", !$("batch-dl-output").hidden && $("batch-dl-errors").hidden && $("batch-status").className === "note ok", $("batch-dl-errors").hidden + " / " + $("batch-status").className);

  // ---- cancel while it runs ----
  let cancelled = false;
  window.fetch = async (url, opts) => {
    if (url === "/v1/files" && opts.method === "POST") return new Response(JSON.stringify({ id: "file_in3" }), { status: 200 });
    if (url === "/v1/batches" && opts.method === "POST") return new Response(JSON.stringify({ id: "batch_3", status: "in_progress", request_counts: { total: 2, completed: 0, failed: 0 } }), { status: 200 });
    if (url === "/v1/batches/batch_3/cancel" && opts.method === "POST") { cancelled = true; return new Response(JSON.stringify({ id: "batch_3", status: "cancelling" }), { status: 200 }); }
    if (url === "/v1/batches/batch_3") return new Response(JSON.stringify({ id: "batch_3", status: cancelled ? "cancelled" : "in_progress", request_counts: { total: 2, completed: cancelled ? 1 : 0, failed: 0 } }), { status: 200 });
    return new Response("{}", { status: 200 });
  };
  viaInput(new File(["{}"], "cancel-me.jsonl"));
  $("batch-run").click();
  await until(() => !$("batch-cancel").hidden);
  $("batch-cancel").click();
  await until(() => cancelled);
  check("W30 Cancel calls the real cancel route for this batch", cancelled, cancelled);
  await until(() => /^cancelled/.test($("batch-status").textContent), 4000);
  check("W30 a cancelled batch says so, and stops polling", $("batch-status").textContent.startsWith("cancelled — ") && $("batch-cancel").hidden, $("batch-status").textContent);

  // ---- an upload that fails leaves the tab usable, with the server's own reason ----
  window.fetch = async (url, opts) => {
    if (url === "/v1/files" && opts.method === "POST") return new Response(JSON.stringify({ error: { message: "request body is 99999999 bytes, which exceeds the limit" } }), { status: 413 });
    return new Response("{}", { status: 200 });
  };
  viaInput(new File(["{}"], "big.jsonl"));
  $("batch-run").click();
  await until(() => /exceeds the limit/.test($("batch-status").textContent));
  check("W30 an upload failure is shown with the server's own reason, and re-enables the tab", $("batch-status").className === "note err" && !$("batch-run").disabled && !$("batch-file").disabled, $("batch-status").textContent);
`);

// ---- phase 38: W31 — cancel a backgrounded job-backed reply from the chat list, without opening it --
const phase38 = phase(W27_PRELUDE + String.raw`
  // a conversation with a job-backed reply running, left in the background (W29's own setup)
  $("newchat").click();
  const JOB9 = "job_" + "c".repeat(32);
  nextSubmit = { status: 202, body: JSON.stringify({ id: JOB9, status: "pending" }) };
  jobs[JOB9] = { status: "running", events: "hang" };
  $("prompt").value = "a long answer, left running";
  const bgRun = send();
  await until(() => generating && generating.job === JOB9);
  await wait(60);
  const bgChat2 = currentChat.id;
  renderChatList();
  const ownRow = () => [...document.querySelectorAll("#chat-list li")].find(li => li.dataset.id === bgChat2);
  check("W31 no Cancel on the conversation you are ACTUALLY viewing (Stop already covers it)", !ownRow()?.querySelector(".chat-cancel-job") && !ownRow()?.querySelector(".chat-running"), ownRow()?.textContent);
  $("newchat").click();
  // the SAME synchronous tick: detachReply() has aborted the stream but the abort has not resolved
  // yet (that is a microtask), so ac is still non-null here — the Cancel button just created for the
  // conversation we left must render disabled, not race its own still-resolving detach.
  const row = () => [...document.querySelectorAll("#chat-list li")].find(li => li.dataset.id === bgChat2);
  check("W31 the newly-offered Cancel starts disabled — its own detach has not settled yet", row()?.querySelector(".chat-cancel-job")?.disabled === true, row()?.querySelector(".chat-cancel-job")?.disabled);
  await settle(bgRun); await idle();

  check("W31 the backgrounded reply offers Cancel, next to its badge", !!row()?.querySelector(".chat-cancel-job"), row()?.textContent);

  // busy (mid-edit in the CURRENT conversation) disables it, same as rename/delete
  nextSubmit = null;   // JOB9's submit response above must not leak into this NEW conversation's own send
  jobs[JOB] = { status: "running", events: realEvents };
  await ask("hi");
  lastYou().querySelector(".msg-edit").click();
  check("W31 Cancel is disabled while editing, like the list's other buttons", row().querySelector(".chat-cancel-job").disabled, row().querySelector(".chat-cancel-job").disabled);
  document.querySelector(".edit-box")?.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
  renderChatList();

  row().querySelector(".chat-cancel-job").click();
  await until(() => jobs[JOB9].deleted === true);
  check("W31 Cancel calls DELETE on the real job id for that OTHER conversation, not the one on screen", jobs[JOB9].deleted === true && currentChat.id !== bgChat2, JSON.stringify(calls.filter(c => c.startsWith("DELETE"))));
  await until(() => !row()?.querySelector(".chat-cancel-job"));
  check("W31 the badge and Cancel disappear immediately — no need to reopen it to see the outcome", !row()?.querySelector(".chat-running") && !row()?.querySelector(".chat-cancel-job"), row()?.textContent);
  const bgStored = JSON.parse(localStorage.getItem("goinfer.chat.v2." + bgChat2));
  check("W31 the conversation is marked cancelled in storage, matching what a real reopen would show", bgStored.running === false && bgStored.job === undefined && bgStored.messages.at(-1).state === "cancelled", JSON.stringify(bgStored.messages.at(-1)));

  calls.length = 0;
  openChat(bgChat2);
  await wait(150);
  check("W31 reopening a cancelled conversation does not try to re-attach (it is not 'interrupted')", calls.length === 0 && !lastBotEl().querySelector(".msg-resume") && lastBotEl().querySelector(".meta").textContent === "cancelled by the server.", calls.join(",") + " / " + lastBotEl()?.querySelector(".meta")?.textContent);

  // a hostile/unreadable stored job id offers no Cancel button at all — never DELETEs a caller-shaped id
  $("newchat").click();
  const hostileId = "hostilebg";
  localStorage.setItem("goinfer.chat.v2." + hostileId, JSON.stringify({ v: 2, id: hostileId, title: "bg", titled: "user", updated: Date.now(), running: true, job: "../../admin/halt", messages: [{ role: "user", content: "q" }, { role: "assistant", model: "m", content: "", state: "generating", job: "../../admin/halt" }] }));
  renderChatList();
  const hostileRow = [...document.querySelectorAll("#chat-list li")].find(li => li.dataset.id === hostileId);
  check("W31 a hostile stored job id gets the badge but never a Cancel button", !!hostileRow?.querySelector(".chat-running") && !hostileRow?.querySelector(".chat-cancel-job"), hostileRow?.textContent);

  // a race: storage is legitimate when the button renders, but changes to something hostile before
  // the click resolves — the function re-reads storage fresh rather than trusting the render, so it
  // must still refuse. (cancelBackgroundJob is invoked directly here: the DOM cannot rebuild a
  // now-invalid button to click, since a render never offers one on a hostile value in the first
  // place — see the check just above — so this is the only way to exercise that re-read at all.)
  calls.length = 0;
  const raceId = "racebg";
  localStorage.setItem("goinfer.chat.v2." + raceId, JSON.stringify({ v: 2, id: raceId, title: "race", titled: "user", updated: Date.now(), running: true, job: "../../admin/halt", messages: [{ role: "user", content: "q" }, { role: "assistant", model: "m", content: "", state: "generating", job: "../../admin/halt" }] }));
  const fakeBtn = document.createElement("button");
  await cancelBackgroundJob(raceId, fakeBtn);
  check("W31 a job id that turned hostile between render and click is refused, never requested", calls.length === 0, JSON.stringify(calls));
`);

// ---- phase 39: W32 — the Models tab lists what is resident, with an Unload per row -----------------
const phase39 = phase(String.raw`
  const until = async (cond, ms = 4000) => { const t0 = Date.now(); while (!cond() && Date.now() - t0 < ms) await wait(10); return cond(); };
  tab("models");
  const GATE_MODEL = { id: "gate-model", object: "model", decode_path: "cuda-resident (int4)", prefill_batched: true, prefill_path: "batched", quant: "int8int8", resident_bytes: 4_500_000_000, vision: false };
  const EMBED_MODEL = { id: "embed-model", object: "model" };   // no decode_path: never listed as resident
  const calls = [];
  const serveModels = list => { window.fetch = async (url, opts) => { calls.push((opts && opts.method || "GET") + " " + url); if (url === "/v1/models") return new Response(JSON.stringify({ object: "list", data: list }), { status: 200 }); return new Response("{}", { status: 200 }); }; };

  serveModels([GATE_MODEL, EMBED_MODEL]);
  await loadModels();
  check("W32 the resident list shows only entries with a decoder, not the embedding-only one", $("resident-list").children.length === 1 && !$("resident-card").hidden, $("resident-list").textContent);
  const row = () => $("resident-list").children[0];
  check("W32 a row names the model (marked as the one in Chat), its quant, decode path and resident size", row().querySelector(".resident-name").textContent === "gate-model — in Chat" && /int8int8/.test(row().querySelector(".note").textContent) && /cuda-resident \(int4\)/.test(row().querySelector(".note").textContent) && /4\.2 GB/.test(row().querySelector(".note").textContent), row().textContent);
  check("W32 the Unload button names the same free-able size", row().querySelector(".resident-unload").textContent === "Unload (4.2 GB)", row().querySelector(".resident-unload").textContent);

  // declining the confirm sends nothing; the wording states the general guarantee, since another
  // client's in-flight work on this model is invisible to this page. unloadModel is invoked through
  // its own onclick, not a synthetic .click(), so each scenario's promise can be awaited directly —
  // a real event-dispatch .click() gives no handle back, and the confusable "unloading…" placeholder
  // this button sets before its own fetch resolves is otherwise indistinguishable from a settled one.
  let confirmMsg = "";
  window.confirm = m => { confirmMsg = m; return false; };
  calls.length = 0;
  await row().querySelector(".resident-unload").onclick();
  check("W32 declining the confirm sends nothing", calls.filter(c => c.startsWith("POST /web/models/unload")).length === 0, JSON.stringify(calls));
  check("W32 the confirm states the drain guarantee, not a claim about what is running elsewhere", /in flight elsewhere will finish/.test(confirmMsg), confirmMsg);

  // this page's OWN active reply on the model gets the specific wording instead
  generating = { model: "gate-model" };
  await row().querySelector(".resident-unload").onclick();
  check("W32 unloading the model you are mid-reply on warns about YOUR reply specifically", /Your current reply will finish first/.test(confirmMsg), confirmMsg);
  generating = null;

  // accepted, freed:true — the row disappears (refreshed from the real, now-shorter /v1/models)
  window.confirm = () => true;
  calls.length = 0;
  window.fetch = async (url, opts) => {
    calls.push((opts && opts.method || "GET") + " " + url);
    if (url === "/web/models/unload") { window.__lastUnload = JSON.parse(opts.body); return new Response(JSON.stringify({ id: "gate-model", status: "unloaded", freed: true }), { status: 200 }); }
    if (url === "/v1/models") return new Response(JSON.stringify({ object: "list", data: [EMBED_MODEL] }), { status: 200 });
    return new Response("{}", { status: 200 });
  };
  await row().querySelector(".resident-unload").onclick();
  check("W32 Unload posts exactly the row's own name", window.__lastUnload && window.__lastUnload.name === "gate-model", JSON.stringify(window.__lastUnload));
  check("W32 a freed unload refreshes the list from the real /v1/models — the row is gone", $("resident-card").hidden && $("resident-list").children.length === 0, $("resident-list").textContent);

  // accepted, freed:false — unloaded (unrouted) but shared with a sibling, so memory was not freed.
  // Captured from the TEST's own fetch stub, at the instant unloadModel's own follow-up GET
  // /v1/models fires: by then note.textContent already holds the final text (it is set before that
  // call), but a moment later renderResidentModels rebuilds the list and the note element is gone —
  // there is no later, safer point to read it from outside app.js itself.
  serveModels([GATE_MODEL, EMBED_MODEL]);
  await loadModels();
  window.fetch = async (url, opts) => {
    if (url === "/web/models/unload") return new Response(JSON.stringify({ id: "gate-model", status: "unloaded", freed: false }), { status: 200 });
    if (url === "/v1/models") {
      window.__capturedNote = $("resident-list").querySelector(".resident-note")?.textContent || "";
      return new Response(JSON.stringify({ object: "list", data: [EMBED_MODEL] }), { status: 200 });
    }
    return new Response("{}", { status: 200 });
  };
  await row().querySelector(".resident-unload").onclick();
  check("W32 'unloaded' is not 'freed': a shared model says its memory was not freed", /not freed/.test(window.__capturedNote), window.__capturedNote);

  // 202 — the drain is still running; the page must not pretend it already finished
  serveModels([GATE_MODEL, EMBED_MODEL]);
  await loadModels();
  window.fetch = async (url, opts) => {
    if (url === "/web/models/unload") return new Response(JSON.stringify({ id: "gate-model", status: "unloading", freed: false }), { status: 202 });
    if (url === "/v1/models") {
      window.__capturedNote = $("resident-list").querySelector(".resident-note")?.textContent || "";
      return new Response(JSON.stringify({ object: "list", data: [EMBED_MODEL] }), { status: 200 });
    }
    return new Response("{}", { status: 200 });
  };
  await row().querySelector(".resident-unload").onclick();
  check("W32 a 202 says it is still draining — no spinner that resolves as if it were already done", /unloading/.test(window.__capturedNote) && /in-flight/.test(window.__capturedNote), window.__capturedNote);

  // an unload failure leaves the row usable, not silently dropped
  serveModels([GATE_MODEL, EMBED_MODEL]);
  await loadModels();
  window.fetch = async (url) => {
    if (url === "/web/models/unload") return new Response(JSON.stringify({ error: { message: "model not found" } }), { status: 404 });
    if (url === "/v1/models") return new Response(JSON.stringify({ object: "list", data: [GATE_MODEL, EMBED_MODEL] }), { status: 200 });
    return new Response("{}", { status: 200 });
  };
  await row().querySelector(".resident-unload").onclick();
  check("W32 a failed unload shows the server's reason and re-enables the button, without touching the list", /model not found/.test(row().querySelector(".resident-note")?.textContent || "") && !row().querySelector(".resident-unload").disabled && $("resident-list").children.length === 1, row().querySelector(".resident-note")?.textContent);

  // no decoder anywhere: no card, nothing to unload
  serveModels([EMBED_MODEL]);
  await loadModels();
  check("W32 no resident card when nothing has a decoder", $("resident-card").hidden, $("resident-card").hidden);
`);

// ---- phase 40: W20 — attach a text/source file: read, fence, prepend (client-side only) -----------
const phase40 = phase(W27_PRELUDE + String.raw`
  const dt = f => { const d = new DataTransfer(); d.items.add(f); return d; };
  const viaInput = files => { const d = new DataTransfer(); for (const f of files) d.items.add(f); $("doc-file").files = d.files; $("doc-file").dispatchEvent(new Event("change")); };
  const viaDrop = f => { const d = dt(f); $("pane-chat").dispatchEvent(new DragEvent("dragover", { dataTransfer: d, bubbles: true, cancelable: true })); $("pane-chat").dispatchEvent(new DragEvent("drop", { dataTransfer: d, bubbles: true, cancelable: true })); };
  const textFile = (name, text) => new File([text], name, { type: "text/plain" });
  const binFile = (name, bytes) => new File([new Uint8Array(bytes)], name, { type: "application/octet-stream" });
  const chips = () => [...$("doc-preview").children];

  $("sampling-reset").click();
  $("newchat").click();

  viaInput([textFile("main.go", "package main\n\nfunc main() {}\n")]);
  await until(() => chips().length === 1);
  check("W20 attaching a text file adds one chip naming it and its length", chips()[0].querySelector(".doc-name").textContent === "main.go (29 chars)" && !$("doc-preview").hidden, chips()[0]?.textContent);

  viaInput([textFile("notes.txt", "hello")]);
  await until(() => chips().length === 2);
  check("W20 a second file adds a second chip, in order, the first untouched", chips().length === 2 && chips()[0].querySelector(".doc-name").textContent.startsWith("main.go") && chips()[1].querySelector(".doc-name").textContent.startsWith("notes.txt"), chips().map(c => c.textContent).join(" | "));

  chips()[0].querySelector("button").click();
  check("W20 removing one chip leaves the other, not the one clicked", chips().length === 1 && chips()[0].querySelector(".doc-name").textContent.startsWith("notes.txt"), chips().map(c => c.textContent).join(" | "));

  // a binary file (a NUL byte) is refused, not silently mangled — the pending list is unchanged
  viaInput([binFile("a.bin", [0, 1, 2, 3])]);
  await until(() => /binary/.test($("doc-note").textContent));
  check("W20 a binary file (a NUL byte) is refused as not text, and nothing is added", /binary/.test($("doc-note").textContent) && chips().length === 1, $("doc-note").textContent + " / " + chips().length);

  // invalid UTF-8 (a lone continuation byte) is refused too — a NUL-free binary would sail through
  // the first check, so this is a DIFFERENT defect the first check cannot catch
  viaInput([binFile("bad.txt", [0xc3, 0x28])]);
  await until(() => /UTF-8/.test($("doc-note").textContent));
  check("W20 invalid UTF-8 is refused, distinct from the NUL-byte check", /UTF-8/.test($("doc-note").textContent) && chips().length === 1, $("doc-note").textContent);

  // oversized is refused with a specific reason, naming the file
  viaInput([textFile("huge.txt", "x".repeat(262_145))]);
  await until(() => /huge\.txt/.test($("doc-note").textContent));
  check("W20 an oversized file is refused by name, not silently truncated", /huge\.txt/.test($("doc-note").textContent) && /larger than/.test($("doc-note").textContent) && chips().length === 1, $("doc-note").textContent);

  // more than the per-message cap: the first DOC_MAX_FILES-1 more are accepted, the rest refused with a note
  viaInput([1, 2, 3, 4, 5, 6].map(n => textFile("f" + n + ".txt", "n" + n)));
  await until(() => /Up to/.test($("doc-note").textContent));
  check("W20 more than the per-message cap is refused with a note, not silently dropped", chips().length === 5 && /Up to 5 files/.test($("doc-note").textContent), chips().length + " / " + $("doc-note").textContent);

  // clear back down to exactly the two files this phase sends
  while (chips().length) chips()[0].querySelector("button").click();
  viaInput([textFile("a.go", "package a\n"), textFile("weird.xyz", "???")]);
  await until(() => chips().length === 2);

  // send: the fenced content is PREPENDED to the typed text, in attachment order — the same string
  // sent, shown in the user's own bubble, and saved (W20 never touches the server: no new route, no
  // content-part — it is exactly what apiMessages would have carried had the user typed it by hand)
  jobs[JOB] = { status: "running", events: realEvents };
  $("prompt").value = "explain this";
  await settle(send()); await idle(); await wait(30);
  const wantPrefix = "\`a.go\`:\n\`\`\`go\npackage a\n\`\`\`\n\n\`weird.xyz\`:\n\`\`\`\n???\n\`\`\`\n\nexplain this";
  check("W20 the sent message is the fenced files then the typed text, languages guessed from extension (unknown: bare fence)", window.__lastBody?.messages?.at(-1)?.content === wantPrefix, JSON.stringify(window.__lastBody?.messages?.at(-1)?.content));
  // the user's own message is TEXT, never Markdown (W3: "the user's text as text") — so the fence
  // markers show up literally, exactly as sent and saved, not as a rendered code block
  check("W20 the user's own bubble shows the literal fenced text, matching what was sent and saved", lastYou().children[1].textContent === wantPrefix && stored().messages.at(-2).content === wantPrefix, lastYou().children[1].textContent);
  check("W20 sending clears the pending files", chips().length === 0 && $("doc-preview").hidden, chips().length);

  // sending with attachments and NO typed text at all is allowed (images already work this way)
  viaInput([textFile("only.go", "package only\n")]);
  await until(() => chips().length === 1);
  jobs[JOB] = { status: "running", events: realEvents };
  $("prompt").value = "";
  await settle(send()); await idle(); await wait(30);
  check("W20 an attachment with no typed text still sends — the fenced file alone", stored().messages.at(-2).content === "\`only.go\`:\n\`\`\`go\npackage only\n\`\`\`\n\n", JSON.stringify(stored().messages.at(-2).content));

  // a non-image file dropped on the chat pane attaches the same way the file picker does
  viaDrop(textFile("dropped.py", "print(1)\n"));
  await until(() => chips().length === 1);
  check("W20 dropping a non-image file onto the chat pane attaches it", chips()[0]?.querySelector(".doc-name")?.textContent.startsWith("dropped.py"), chips()[0]?.textContent);
  chips()[0].querySelector("button").click();

  // an image dropped on the pane goes to the OTHER attach path silently — no doc chip, and no
  // "binary file" rejection note either: that message would be true (image bytes aren't text) but
  // wrong to show, since the user never asked to attach it as a file in the first place
  $("doc-note").textContent = ""; $("doc-note").hidden = true;
  viaDrop(new File([new Uint8Array([0x89, 0x50, 0x4e, 0x47])], "pic.png", { type: "image/png" }));
  await wait(80);
  check("W20 dropping an IMAGE never reaches the doc path — no chip, no spurious rejection note", chips().length === 0 && $("doc-note").hidden && !$("doc-note").textContent, chips().length + " / " + $("doc-note").textContent);

  // busy (mid-generation) refuses new attachments, same as the image path
  jobs[JOB] = { status: "running", events: "hang" };
  $("prompt").value = "long running";
  const bgRun = send();
  await until(() => generating && generating.job === JOB);
  viaInput([textFile("busy.go", "package busy\n")]);
  await wait(50);
  check("W20 attaching while a reply is generating is refused, same as image attach", chips().length === 0, chips().length);
  $("stop").click(); await settle(bgRun); await idle();
`);

// Found live 2026-09-16, not from the task doc's W-list: an unbounded #log made a streaming reply
// grow the WHOLE PAGE, so the composer below it kept sliding further down and off-screen ("losing
// the footer") while app.js's own scrollIntoView re-scrolled the window every frame to chase it
// ("jumps around"). Fixed by bounding #log (overflow-y:auto, scroll-behavior:smooth) and scrolling
// its own scrollTop instead of the window. A long synthetic reply (600 chunks) is used, not a short
// captured one, because the panel has to genuinely overflow its bound for any of this to be
// checking something real rather than a page that never grew past one screen anyway.
const phase41 = phase(W27_PRELUDE + String.raw`
  $("sampling-reset").click();
  $("newchat").click();

  const N = 600;
  const chunks = [];
  for (let i = 0; i < N; i++) {
    chunks.push("data: " + JSON.stringify({ choices: [{ delta: { content: "lorem" + i + " " }, finish_reason: null, index: 0 }], created: 1, id: JOB, model: "q", object: "chat.completion.chunk" }));
  }
  chunks.push("data: " + JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop", index: 0 }], created: 1, id: JOB, model: "q", object: "chat.completion.chunk" }));
  chunks.push("data: [DONE]");
  jobs[JOB] = { status: "running", events: chunks };

  const log = $("log");
  const pageScroll0 = document.scrollingElement.scrollTop;

  $("prompt").value = "write something long";
  const run = send();
  // Sample once #log has GENUINELY overflowed its own bound, not at an arbitrary character count —
  // a short prefix of the reply is well under the 60vh cap and the checks below would be checking
  // nothing real if it hadn't. If it never does (e.g. the bound itself regresses), this check alone
  // catches that; the composer-position checks below exist to catch a DIFFERENT class of
  // regression — the panel scrolling correctly but something still moving the page around it —
  // and are read against a baseline taken here, once the panel is doing its real job, not before.
  await until(() => log.scrollHeight > log.clientHeight + 20, 4000);
  const composerTop0 = $("send").getBoundingClientRect().top;

  check("scroll panel: #log is bounded and genuinely overflowing mid-stream, not just tall", log.scrollHeight > log.clientHeight + 20, log.scrollHeight + " / " + log.clientHeight);
  check("scroll panel: #log stays scrolled to its OWN bottom while streaming", log.scrollTop + log.clientHeight >= log.scrollHeight - 2, log.scrollTop + "+" + log.clientHeight + " vs " + log.scrollHeight);
  check("scroll panel: the page itself has not scrolled while the reply streams in", document.scrollingElement.scrollTop === pageScroll0, document.scrollingElement.scrollTop + " vs " + pageScroll0);

  await wait(120);   // several more frames of streaming, sampled mid-flight, not just at one instant
  check("scroll panel: the composer has not moved since the panel started overflowing — the page around #log stays put", Math.abs($("send").getBoundingClientRect().top - composerTop0) < 1, $("send").getBoundingClientRect().top + " vs " + composerTop0);
  check("scroll panel: #log is still following its own bottom a moment later, not falling behind", log.scrollTop + log.clientHeight >= log.scrollHeight - 2, log.scrollTop + "+" + log.clientHeight + " vs " + log.scrollHeight);

  await settle(run); await idle();

  check("scroll panel: still scrolled to #log's own bottom once the reply finishes", log.scrollTop + log.clientHeight >= log.scrollHeight - 2, log.scrollTop + "+" + log.clientHeight + " vs " + log.scrollHeight);
  check("scroll panel: the composer still has not moved after the reply finishes", Math.abs($("send").getBoundingClientRect().top - composerTop0) < 1, $("send").getBoundingClientRect().top + " vs " + composerTop0);

  // opening a different (empty) conversation, then reopening this one, lands scrolled to its
  // newest message too — not just the live-streaming case above
  const thisChat = currentChat.id;
  $("newchat").click();
  openChat(thisChat);
  check("scroll panel: reopening a conversation shows it scrolled to the newest message", log.scrollTop + log.clientHeight >= log.scrollHeight - 2, log.scrollTop + "+" + log.clientHeight + " vs " + log.scrollHeight);
`);

// ---- phase 42: repo search-as-you-type (HF search, GGUF only for now) --------------------------
const phase42 = phase(String.raw`
  const until = async (cond, ms = 4000) => { const t0 = Date.now(); while (!cond() && Date.now() - t0 < ms) await wait(10); return cond(); };
  tab("models");
  const setRepo = v => { $("repo").value = v; $("repo").dispatchEvent(new Event("input")); };
  let calls = [];
  const stub = (repos, status = 200) => { window.fetch = async (url, opts) => {
    calls.push({ url, method: (opts && opts.method) || "GET", body: opts && opts.body ? JSON.parse(opts.body) : null });
    if (url === "/web/models/search") return new Response(JSON.stringify(status === 200 ? { repos } : { error: { message: "HuggingFace returned " + status } }), { status });
    return new Response("{}", { status: 200 }); // covers /web/models/list, from clicking a suggestion below
  }; };

  // below the 4-char minimum: never fetches at all
  stub([{ repo: "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF", downloads: 12345, likes: 10 }]);
  setRepo("abc");
  await wait(400);
  check("search: below the 4-char minimum never fetches", !calls.some(c => c.url === "/web/models/search"), JSON.stringify(calls));

  // rapid typing before the debounce fires coalesces into ONE request, for the latest text only —
  // both values here clear the 4-char minimum on their own, so this isolates the debounce from
  // that other gate rather than accidentally retesting it (an earlier draft used "qwe" then
  // "qwen", where "qwe" alone never fires regardless of debounce — it is 3 characters)
  calls = [];
  setRepo("qwen"); await wait(60); setRepo("qwen2");
  await wait(400);
  let hits = calls.filter(c => c.url === "/web/models/search");
  check("search: rapid typing coalesces into ONE request, for the latest text", hits.length === 1 && hits[0].body.query === "qwen2" && hits[0].body.kind === "gguf", JSON.stringify(hits));

  // a match renders as a suggestion row; picking it fills the box AND lists it immediately —
  // the point of offering one is fewer clicks than typing the whole name, not just spelling help
  await until(() => $("repo-suggest").children.length === 1);
  check("search: a match renders as a suggestion naming the repo, and the box says it is expanded", $("repo-suggest").children[0].textContent.includes("Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF") && !$("repo-suggest").hidden && $("repo").getAttribute("aria-expanded") === "true", $("repo-suggest").textContent);
  calls = [];
  $("repo-suggest").querySelector("button").click();
  await wait(30);
  check("search: picking a suggestion fills the box, closes the list, and lists it immediately", $("repo").value === "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF" && $("repo-suggest").hidden && calls.some(c => c.url === "/web/models/list" && c.body.repo === "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF"), $("repo").value + " / " + JSON.stringify(calls));

  // no matches: a note, not a stale or silently-empty list
  stub([]);
  setRepo("zzzz"); await wait(60); setRepo("zzzznomatch");
  await wait(400);
  check("search: no matches shows a note, not a stale or empty-looking list", $("repo-suggest").hidden && !$("repo-suggest-note").hidden && /No matching/.test($("repo-suggest-note").textContent), $("repo-suggest-note").textContent);

  // a failed search is not fatal to the box: a quiet note, nothing left stale on screen
  stub([], 502);
  setRepo("errq"); await wait(60); setRepo("errq123");
  await wait(400);
  check("search: a failed search shows a quiet note instead of breaking the box", $("repo-suggest").hidden && /Couldn't search HuggingFace/.test($("repo-suggest-note").textContent), $("repo-suggest-note").textContent);

  // Escape closes the list without touching what was typed
  stub([{ repo: "a/b", downloads: 1, likes: 1 }]);
  setRepo("escq"); await wait(60); setRepo("escq123");
  await until(() => !$("repo-suggest").hidden);
  $("repo").dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
  check("search: Escape closes the suggestions without changing the typed text", $("repo-suggest").hidden && $("repo").value === "escq123", $("repo").value + " / " + $("repo-suggest").hidden);

  // THE RACE: starting a new search aborts the previous request's own signal (not just a sequence
  // counter's job) — and even if a stale request's response arrives AFTER a newer one's, it must
  // never overwrite what the newer one already rendered. Two independent guards (abort + a
  // sequence number the async handler checks itself), because nothing guarantees the event loop
  // delivers an abort rejection before a response that was already in flight when it fired.
  const resolvers = [];
  const signals = [];
  window.fetch = async (url, opts) => {
    if (url !== "/web/models/search") return new Response("{}", { status: 200 });
    signals.push(opts.signal);
    return new Promise(resolve => resolvers.push({ query: JSON.parse(opts.body).query, resolve }));
  };
  setRepo("race"); await wait(60); setRepo("race1"); await wait(310);   // debounce fires: request #1 pending
  setRepo("race12"); await wait(310);                                   // debounce fires again: request #2 pending, #1 superseded
  check("search: starting a new search aborts the previous request's own AbortSignal", signals.length === 2 && signals[0].aborted === true && signals[1].aborted === false, signals.map(s => s && s.aborted));
  // Resolve OUT OF ORDER: the newer query's response arrives FIRST, the stale one SECOND.
  resolvers[1].resolve(new Response(JSON.stringify({ repos: [{ repo: "newer/match", downloads: 2, likes: 2 }] }), { status: 200 }));
  await wait(30);
  resolvers[0].resolve(new Response(JSON.stringify({ repos: [{ repo: "stale/match", downloads: 1, likes: 1 }] }), { status: 200 }));
  await wait(30);
  check("search: a stale response that resolves AFTER a newer one never overwrites it", $("repo-suggest").textContent.includes("newer/match") && !$("repo-suggest").textContent.includes("stale/match"), $("repo-suggest").textContent);
`);

const all = [];
const PHASES = [phase1, phase2, phase3, phase4, phase5, phase6, phase7, phase8, phase9, phase10, phase11, phase12, phase13, phase14, phase15, phase16, phase17, phase18, phase19, phase20, phase21, phase22, phase23, phase24,
  // headless Chrome's own default is a DARK preference — so the light phase must set light explicitly
  async () => page.cdp("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: "light" }] }),
  phase25,
  async () => page.cdp("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: "dark" }] }),
  phase26, phase27,
  async () => { await page.cdp("Emulation.setDeviceMetricsOverride", { width: 400, height: 860, deviceScaleFactor: 2, mobile: true }); },
  phase28,
  async () => { await page.cdp("Emulation.setDeviceMetricsOverride", { width: 360, height: 740, deviceScaleFactor: 2, mobile: true }); },
  phase29,
  async () => { await page.cdp("Emulation.setDeviceMetricsOverride", { width: 1100, height: 900, deviceScaleFactor: 1, mobile: false }); },
  phase30, phase31, phase32, phase33, phase34, phase35, phase36, phase37, phase38, phase39, phase40, phase41, phase42];
let n = 0;
for (const prog of PHASES) {
  if (typeof prog === "function") { await prog(); continue; }   // a Node-side step between phases, not a phase
  n++;
  if (n > 1) await page.reload();
  all.push(...finishOrExit("webui app gate (phase " + n + ")", await page.evaluate(prog), all));
}
page.close();
report("webui app gate", all, page.exceptions);
