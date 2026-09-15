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
  const stored = () => { try { return JSON.parse(localStorage.getItem("goinfer.chat.v1")); } catch { return "UNPARSEABLE"; } };
`;
const phase = body => "(async () => {" + prelude + body + "\n  return results;\n})()";

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
  check("W3 conversation saved under a versioned key", st && st.v === 1 && Array.isArray(msgs), JSON.stringify(st)?.slice(0, 120));
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
  check("W3 New chat is disabled while a reply is generating", $("newchat").disabled === true, $("newchat").disabled);
`);

// ---- phase 3: after a mid-stream reload; then corrupt the store --------------------------------------
const phase3 = phase(String.raw`
  const bots = document.querySelectorAll("#log .msg.bot");
  const last = bots[bots.length - 1];
  check("W3 mid-stream reload kept the partial answer", bots.length === 3 && last.children[1].textContent.startsWith("an answer the reload interrupts"), bots.length + " / " + last?.children[1].textContent);
  check("W3 it is labelled interrupted", /interrupted — the page was reloaded/.test(last.querySelector(".meta")?.textContent || ""), last.querySelector(".meta")?.textContent);
  const st = stored();
  check("W3 interrupted state written back to storage", st.messages[st.messages.length - 1].state === "interrupted", JSON.stringify(st.messages[st.messages.length - 1]));

  localStorage.setItem("goinfer.chat.v1", "{this is not json");
`);

// ---- phase 4: after a reload with unreadable storage; quota; New chat; tab sync ---------------------
const phase4 = phase(String.raw`
  check("W3 unreadable storage: page loads with an empty log", document.querySelectorAll("#log .msg").length === 0, document.querySelectorAll("#log .msg").length);
  check("W3 unreadable storage is set aside, not destroyed", localStorage.getItem("goinfer.chat.v1.unreadable") === "{this is not json", localStorage.getItem("goinfer.chat.v1.unreadable"));
  check("W3 unreadable value removed from the live key", localStorage.getItem("goinfer.chat.v1") === null, localStorage.getItem("goinfer.chat.v1"));

  // quota: every write throws — the chat must keep working and say so
  const realSet = Storage.prototype.setItem;
  Storage.prototype.setItem = function () { throw new DOMException("full", "QuotaExceededError"); };
  streamAnswer("still works without storage");
  $("prompt").value = "storage is full";
  await send(); await wait(150);
  check("W3 quota failure: the reply still renders", lastBot()?.children[1].textContent === "still works without storage", lastBot()?.children[1].textContent);
  check("W3 quota failure: the note is shown", !$("store-note").hidden && /can't be kept across a reload/.test($("store-note").textContent), $("store-note").hidden + " " + $("store-note").textContent);
  Storage.prototype.setItem = realSet;

  // New chat, cancelled: nothing cleared
  window.confirm = () => false;
  $("newchat").click();
  check("W3 New chat cancelled keeps the conversation", document.querySelectorAll("#log .msg").length === 2, document.querySelectorAll("#log .msg").length);
  // New chat, confirmed
  window.confirm = () => true;
  localStorage.setItem("goinfer.chat.v1", JSON.stringify({ v: 1, messages: [{ role: "user", content: "x" }] }));
  $("newchat").click();
  check("W3 New chat confirmed clears the log", document.querySelectorAll("#log .msg").length === 0, document.querySelectorAll("#log .msg").length);
  check("W3 New chat confirmed clears storage", localStorage.getItem("goinfer.chat.v1") === null, localStorage.getItem("goinfer.chat.v1"));
  check("W3 New chat hides the storage note", $("store-note").hidden, $("store-note").hidden);

  // another tab writes a conversation: this idle tab follows it
  localStorage.setItem("goinfer.chat.v1", JSON.stringify({ v: 1, messages: [
    { role: "user", content: "from the other tab" },
    { role: "assistant", content: "**synced**", model: "other-model", meta: "3 tok" } ] }));
  window.dispatchEvent(new StorageEvent("storage", { key: "goinfer.chat.v1" }));
  await wait(50);
  check("W3 idle tab follows another tab's conversation", lastBot()?.children[1].querySelector("strong")?.textContent === "synced" && lastYou()?.children[1].textContent === "from the other tab", lastBot()?.innerText);

  // hostile stored content, rendered on the next load
  localStorage.setItem("goinfer.chat.v1", JSON.stringify({ v: 1, messages: [
    { role: "user", content: "<img src=x onerror=\"window.__pwned=1\">" },
    { role: "assistant", content: "[x](javascript:window.__pwned=1)\n\n<script>window.__pwned=1<\/script>", model: "<b onmouseover=window.__pwned=1>m</b>", meta: "<img src=x onerror=window.__pwned=1>" },
    { role: "system", content: "not a turn this page renders" },
    { role: "assistant", content: 42 } ] }));
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
  localStorage.setItem("goinfer.chat.v1", JSON.stringify(st));
  window.dispatchEvent(new StorageEvent("storage", { key: "goinfer.chat.v1" }));
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
  window.dispatchEvent(new StorageEvent("storage", { key: "goinfer.chat.v1" }));
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
  localStorage.setItem("goinfer.chat.v1", JSON.stringify(st));
  window.dispatchEvent(new StorageEvent("storage", { key: "goinfer.chat.v1" }));
  await wait(100);
  m = meter();
  check("W8 unreadable stored usage is ignored, falling back to the previous reply's", /about 820 of 1,000/.test(m.text) && document.querySelectorAll("img").length === 0 && window.__pwned === 0, JSON.stringify(m));
  $("newchat").click();
  check("W8 New chat hides the meter", $("ctx").hidden, JSON.stringify(meter()));
`);

const all = [];
for (const [i, prog] of [phase1, phase2, phase3, phase4, phase5, phase6, phase7, phase8, phase9, phase10, phase11, phase12, phase13, phase14].entries()) {
  if (i > 0) await page.reload();
  all.push(...finishOrExit("webui app gate (phase " + (i + 1) + ")", await page.evaluate(prog), all));
}
page.close();
report("webui app gate", all, page.exceptions);
