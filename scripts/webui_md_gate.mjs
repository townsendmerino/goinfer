// webui_md_gate.mjs — the W1 security and correctness gate for the web UI's Markdown renderer
// (docs/tasks/task-web-ui-2026-09.md W1, §6.2).
//
//   node scripts/webui_md_gate.mjs          # exit 0 = pass, 1 = a check failed, 2 = could not run
//
// It drives headless Chrome over the DevTools protocol and loads scripts/webui-md-gate/harness.html
// from file:// — which loads the SHIPPED internal/serveapp/webui/ui/markdown.js, not a copy. Then, in
// that real browser:
//
//   1. HOSTILE INPUT. Every payload is rendered into a node ATTACHED to the document, so a payload
//      that did create a live <img onerror> or <script> would actually fire. Then it asserts:
//        - every element is in the renderer's allow-list, with only allow-listed attributes;
//        - every href is http:, https: or mailto:, and every class is one the renderer emits;
//        - no payload ran: window.__pwned is still 0 after the page has had time to load anything.
//   2. PATHOLOGICAL INPUT. Inputs built to trigger regex backtracking must each render within a time
//      budget — the renderer runs on every streamed chunk, so a slow case freezes the page.
//   3. CORRECTNESS. Common Markdown renders to the expected element structure.
//
// file:// rather than a server on purpose: it needs no port, and it works on a box whose sandbox
// blocks headless Chrome from loopback HTTP (nobara-pc, 2026-09-14).
import { spawn, execSync } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const harness = pathToFileURL(resolve(here, "webui-md-gate", "harness.html")).href;
const chromeBin = process.env.CHROME_BIN || ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"]
  .find(b => { try { execSync(`command -v ${b}`, { stdio: "ignore", shell: "/bin/sh" }); return true; } catch { return false; } }) || "google-chrome";

const sleep = ms => new Promise(r => setTimeout(r, ms));
const port = 9400 + Math.floor(Math.random() * 400);
const chrome = spawn(chromeBin, ["--headless=new", "--disable-gpu", "--no-sandbox", "--allow-file-access-from-files",
  `--remote-debugging-port=${port}`, "--remote-allow-origins=*", `--user-data-dir=${mkdtempSync(join(tmpdir(), "mdgate-"))}`,
  "about:blank"], { stdio: "ignore" });
const bail = (code, msg) => { console.error(msg); try { chrome.kill(); } catch {} process.exit(code); };
chrome.on("error", e => bail(2, `could not start ${chromeBin}: ${e.message}`));

let target;
for (let i = 0; i < 80 && !target; i++) {
  try { target = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).find(t => t.type === "page"); } catch {}
  if (!target) await sleep(250);
}
if (!target) bail(2, "chrome exposed no page target");
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((ok, no) => { ws.addEventListener("open", ok); ws.addEventListener("error", () => no(new Error("devtools socket error"))); setTimeout(() => no(new Error("devtools socket timeout")), 10000); })
  .catch(e => bail(2, e.message));
let seq = 0; const waiting = new Map(); const exceptions = [];
ws.addEventListener("message", ev => {
  const m = JSON.parse(ev.data);
  if (m.id && waiting.has(m.id)) { waiting.get(m.id)(m); waiting.delete(m.id); return; }
  if (m.method === "Runtime.exceptionThrown") exceptions.push(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
});
const send = (method, params = {}) => new Promise(r => { const id = ++seq; waiting.set(id, r); ws.send(JSON.stringify({ id, method, params })); });
await send("Runtime.enable"); await send("Page.enable");
await send("Page.navigate", { url: harness });
await sleep(1500);

// ---- everything below runs inside the page ------------------------------------------------------
const program = String.raw`(async () => {
  const results = [];
  const fail = (name, why) => results.push({ name, ok: false, why });
  const pass = name => results.push({ name, ok: true });
  if (typeof Markdown !== "object" || typeof Markdown.render !== "function") {
    return [{ name: "renderer loaded", ok: false, why: "Markdown.render not defined — harness did not load ui/markdown.js" }];
  }
  window.__pwned = 0;
  const stage = document.getElementById("stage");
  const ATTRS = { a: new Set(["class", "href", "rel", "target"]), ol: new Set(["class", "start"]) };
  const CLASS_OK = /^(md-code|md-lang|md-table|al-(left|center|right)|language-[A-Za-z0-9_+#.-]{1,32})$/;

  function audit(root) {
    const bad = [];
    for (const e of root.querySelectorAll("*")) {
      const tag = e.tagName.toLowerCase();
      if (!Markdown.TAGS.has(tag)) bad.push("element <" + tag + ">");
      const allowed = ATTRS[tag] || new Set(["class"]);
      for (const at of e.attributes) {
        if (!allowed.has(at.name)) bad.push("attribute " + at.name + " on <" + tag + ">");
        if (at.name === "href" && !/^(https?:|mailto:)/i.test(e.getAttribute("href"))) bad.push("href " + JSON.stringify(e.getAttribute("href")));
        if (at.name === "class") for (const c of at.value.split(/\s+/).filter(Boolean)) if (!CLASS_OK.test(c)) bad.push("class " + JSON.stringify(c));
      }
    }
    return bad;
  }

  // 1. hostile input — rendered into ATTACHED nodes so anything live would actually fire
  const hostile = {
    "raw script tag": "<script>window.__pwned=1<\/script>",
    "img onerror": '<img src=x onerror="window.__pwned=1">',
    "svg onload inside strong": "**<svg onload=window.__pwned=1>**",
    "iframe in table cell": "| a |\n|---|\n| <iframe src=\"javascript:window.__pwned=1\"></iframe> |",
    "html anchor with javascript": '<a href="javascript:window.__pwned=1">x</a>',
    "md link javascript": "[click](javascript:window.__pwned=1)",
    "md link mixed-case scheme": "[click](JaVaScRiPt:window.__pwned=1)",
    "md link leading space": "[click]( javascript:window.__pwned=1)",
    "md link data url": "[click](data:text/html,<script>window.__pwned=1<\/script>)",
    "md link vbscript": "[click](vbscript:msgbox(1))",
    "md link entity-encoded scheme": "[click](&#106;avascript:window.__pwned=1)",
    "md link tab in scheme": "[click](java\tscript:window.__pwned=1)",
    "md image javascript": "![a](javascript:window.__pwned=1)",
    "md image remote (must not load)": "![pixel](https://tracker.invalid/p.png)",
    "code span with html": "\x60<b onmouseover=window.__pwned=1>x</b>\x60",
    "fence with hostile lang": "\x60\x60\x60html\"><script>window.__pwned=1<\/script>\nx\n\x60\x60\x60",
    "link text with html": "[<img src=x onerror=window.__pwned=1>](https://example.com)",
    "bare url with markup": "https://example.com/<script>window.__pwned=1<\/script>",
    "html comment": "<!-- <script>window.__pwned=1<\/script> -->",
    "style tag": "<style>body{display:none}</style>",
    "form + input": "<form action=https://evil.invalid><input name=k></form>",
    "heading with handler": "# <span onclick=window.__pwned=1>t</span>",
    "blockquote with meta refresh": "> <meta http-equiv=refresh content=0;url=https://evil.invalid>",
    "list item with object": "- <object data=javascript:window.__pwned=1></object>",
  };
  const holder = document.createElement("div");
  stage.appendChild(holder);
  for (const [name, src] of Object.entries(hostile)) {
    const box = document.createElement("div");
    holder.appendChild(box);
    try { Markdown.render(box, src); } catch (e) { fail("hostile: " + name, "render threw " + e.message); continue; }
    const bad = audit(box);
    if (bad.length) fail("hostile: " + name, bad.join("; ")); else pass("hostile: " + name);
  }
  // give any live payload the chance to run before checking
  await new Promise(r => setTimeout(r, 800));
  if (window.__pwned !== 0) fail("no payload executed", "window.__pwned = " + window.__pwned);
  else pass("no payload executed");
  if (document.querySelectorAll("#stage img, #stage script, #stage iframe, #stage object, #stage style, #stage form").length)
    fail("no dangerous elements anywhere in the stage", "found one");
  else pass("no dangerous elements anywhere in the stage");
  const imgLink = [...holder.querySelectorAll("a")].find(a => a.href.startsWith("https://tracker.invalid/"));
  if (!imgLink) fail("remote image becomes a link, not an image", "no link to the image URL");
  else pass("remote image becomes a link, not an image");

  // 2. pathological input — bounded time on every streamed chunk
  const slow = {
    "50k asterisks": "*".repeat(50000),
    "50k underscores": "_".repeat(50000),
    "20k open brackets": "[".repeat(20000),
    "20k link openers": "[a](".repeat(5000),
    "10k backticks": "\x60".repeat(10000),
    "unclosed strong x5000": "**a ".repeat(5000),
    "alternating emphasis": "*a_".repeat(10000),
    "deep list nesting": Array.from({ length: 300 }, (_, i) => " ".repeat(i * 2) + "- x").join("\n"),
    "5k-row table": "| a | b |\n|---|---|\n" + "| 1 | 2 |\n".repeat(5000),
    "long line of pipes": "|".repeat(50000),
    "escaped chars": "\\*".repeat(20000),
  };
  for (const [name, src] of Object.entries(slow)) {
    const box = document.createElement("div");
    const t0 = performance.now();
    try { Markdown.render(box, src); } catch (e) { fail("pathological: " + name, "threw " + e.message); continue; }
    const ms = performance.now() - t0;
    if (ms > 1000) fail("pathological: " + name, ms.toFixed(0) + " ms (budget 1000 ms)");
    else pass("pathological: " + name + " (" + ms.toFixed(0) + " ms)");
  }

  // 3. correctness — element structure, adjacent text merged
  const sig = n => {
    if (n.nodeType === 3) return JSON.stringify(n.data);
    const cls = n.className ? "." + n.className.split(/\s+/).join(".") : "";
    return n.nodeName.toLowerCase() + cls + "(" + [...n.childNodes].map(sig).join(",") + ")";
  };
  const shape = src => { const box = document.createElement("div"); Markdown.render(box, src); box.normalize(); return [...box.childNodes].map(sig).join(","); };
  const cases = [
    ["Hello **world**", 'p("Hello ",strong("world"))'],
    ["*em* and _em_", 'p(em("em")," and ",em("em"))'],
    ["snake_case_name and 2*3*4", 'p("snake_case_name and 2*3*4")'],
    ["~~gone~~", 'p(del("gone"))'],
    ["a \x60code\x60 b", 'p("a ",code("code")," b")'],
    ["\\*not em\\*", 'p("*not em*")'],
    ["line one\nline two", 'p("line one",br(),"line two")'],
    ["# Title", 'h1("Title")'],
    ["### Sub ###", 'h3("Sub")'],
    ["\x60\x60\x60go\nfmt.Println(1)\n\x60\x60\x60", 'div.md-code(span.md-lang("go"),pre(code.language-go("fmt.Println(1)")))'],
    ["\x60\x60\x60\nunclosed while streaming", 'div.md-code(pre(code("unclosed while streaming")))'],
    ["~~~\ntilde fence\n~~~", 'div.md-code(pre(code("tilde fence")))'],
    ["- a\n- b", 'ul(li("a"),li("b"))'],
    ["1. a\n2. b", 'ol(li("a"),li("b"))'],
    ["3. c\n4. d", 'ol(li("c"),li("d"))'],
    ["- a\n  - b\n- c", 'ul(li("a",ul(li("b"))),li("c"))'],
    ["- a\n\n- b", 'ul(li(p("a")),li(p("b")))'],
    ["> quoted", 'blockquote(p("quoted"))'],
    ["---", "hr()"],
    ["see [docs](https://example.com)", 'p("see ",a("docs"))'],
    ["visit https://example.com.", 'p("visit ",a("https://example.com"),".")'],
    ["[bad](javascript:x)", 'p("bad (javascript:x)")'],
    ["| a | b |\n|:--|--:|\n| 1 | 2 |", 'table.md-table(thead(tr(th.al-left("a"),th.al-right("b"))),tbody(tr(td.al-left("1"),td.al-right("2"))))'],
    ["para\n\n\x60\x60\x60\ncode\n\x60\x60\x60\n\nafter", 'p("para"),div.md-code(pre(code("code"))),p("after")'],
    ["<b>not bold</b>", 'p("<b>not bold</b>")'],
  ];
  for (const [src, want] of cases) {
    let got;
    try { got = shape(src); } catch (e) { got = "THREW " + e.message; }
    if (got === want) pass("renders: " + JSON.stringify(src));
    else fail("renders: " + JSON.stringify(src), "got " + got + "  want " + want);
  }
  return results;
})()`;

const r = await send("Runtime.evaluate", { expression: program, awaitPromise: true, returnByValue: true, timeout: 60000 });
ws.close(); chrome.kill();
// A destroyed execution context means the page NAVIGATED or reloaded mid-gate. Nothing in the harness
// navigates, so a payload did it (a live <meta http-equiv=refresh>, say): that is a security failure
// and exits 1, not "could not run".
const cdpErr = r.error?.message || r.result?.exceptionDetails?.exception?.description || r.result?.exceptionDetails?.text || "";
if (/context was destroyed|navigated|target closed|Cannot find context/i.test(cdpErr)) {
  console.log("  FAIL  the harness page navigated away while rendering hostile input — a payload took effect (" + cdpErr + ")");
  console.log("webui markdown gate: FAILED (page hijacked)");
  process.exit(1);
}
if (!Array.isArray(r.result?.result?.value)) {
  console.error("gate program did not complete: " + (cdpErr || JSON.stringify(r).slice(0, 600)));
  process.exit(2);
}
const results = r.result.result.value;
let failed = 0;
for (const x of results) {
  if (!x.ok) failed++;
  console.log((x.ok ? "  ok    " : "  FAIL  ") + x.name + (x.ok ? "" : "\n          " + x.why));
}
for (const e of exceptions) { failed++; console.log("  FAIL  uncaught page exception: " + e); }
console.log(`webui markdown gate: ${results.length - (failed - exceptions.length)} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
