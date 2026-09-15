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
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { openPage, report, finishOrExit } from "./webui-gate/cdp.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const page = await openPage(pathToFileURL(resolve(here, "..", "internal", "serveapp", "webui", "index.html")).href);

const program = String.raw`(async () => {
  const results = [];
  const check = (name, ok, why) => results.push(ok ? { name, ok: true } : { name, ok: false, why: String(why) });
  const wait = ms => new Promise(r => setTimeout(r, ms));
  if (typeof send !== "function" || typeof Markdown !== "object") {
    return [{ name: "page scripts loaded", ok: false, why: "send()/Markdown missing — index.html did not load its ui/ scripts" }];
  }
  window.__pwned = 0;
  const sel = $("model");
  const opt = document.createElement("option");
  opt.value = opt.textContent = "gate-model";
  sel.appendChild(opt);
  sel.value = "gate-model";

  // fetch stub: streams text as SSE deltas in 3-char chunks (cutting through fences and emphasis like
  // a real token stream). With hang=true it stops mid-answer and waits for the request to be aborted.
  const enc = new TextEncoder();
  function streamAnswer(text, { hang = false } = {}) {
    window.fetch = async (url, opts) => new Response(new ReadableStream({ async start(c) {
      opts?.signal?.addEventListener("abort", () => { try { c.error(new DOMException("aborted", "AbortError")); } catch {} });
      for (let k = 0; k < text.length; k += 3) {
        c.enqueue(enc.encode("data: " + JSON.stringify({ choices: [{ delta: { content: text.slice(k, k + 3) } }] }) + "\n\n"));
        await wait(2);
      }
      if (hang) return;              // never closes: only Stop ends it
      c.enqueue(enc.encode("data: [DONE]\n\n"));
      c.close();
    } }), { status: 200, headers: { "Content-Type": "text/event-stream" } });
  }
  const lastBot = () => { const b = document.querySelectorAll("#log .msg.bot"); return b[b.length - 1]; };
  const lastYou = () => { const b = document.querySelectorAll("#log .msg.you"); return b[b.length - 1]; };

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
  return results;
})()`;

const results = finishOrExit("webui app gate", await page.evaluate(program));
page.close();
report("webui app gate", results, page.exceptions);
