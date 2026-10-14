// webui_csp_smoke.mjs — the web UI under its Content-Security-Policy, as a server sends it, in headless Chrome.
//
//   node scripts/webui_csp_smoke.mjs http://127.0.0.1:8123/        # exit 0 pass / 1 a check failed / 2 no browser
//
// scripts/webui_app_gate.mjs opens index.html from file://, where no policy applies: an inline handler or a script the
// policy blocks would pass that gate and break the served page. This opens the page from a server, so the browser enforces
// the header, and checks two things. First, that everything the page does still works with no violation reported: its
// scripts and stylesheet, its style attributes, the model list and a reply over fetch, an attached image (a data: URI made
// through a blob: URL), and the HTML preview (a data: frame). Second, that the policy is in force, by trying what it
// forbids — an inline script, an inline handler, and a script, image, request, frame and stylesheet on another origin — and
// requiring a violation for each; without those, "no violation" could mean "no policy".
//
// TestWebUI_contentPolicyInBrowser runs it against the real page handlers with a stub model behind them. Run it against a
// live `serve -web` to check the same with a real model.
import { openPage, report, finishOrExit } from "./webui-gate/cdp.mjs";

const url = process.argv[2];
if (!url) { console.error("usage: node scripts/webui_csp_smoke.mjs <url of a serve -web>"); process.exit(2); }

const sleep = ms => new Promise(r => setTimeout(r, ms));
// within resolves to what p does, or to { stalled: true } after ms: nothing below may wait on the browser without a bound.
const within = (p, ms) => Promise.race([p, sleep(ms).then(() => ({ stalled: true }))]);

const page = await openPage(url, { settleMs: 500 });
// The page must have arrived before anything is asked of the browser: while a page load is pending, some of its
// commands never answer. A browser that cannot load the page from this server could not be driven, which is exit 2 and
// a loud skip in go test; it says nothing about the policy.
const notDriven = why => { console.error("the browser did not load " + url + " (" + why + "): the content-policy smoke did NOT run"); page.close(); process.exit(2); };
const arrived = async () => {
  const where = await within(page.evaluate("location.href + ' ' + document.readyState"), 15000);
  if (where.stalled) notDriven("no answer in 15 s");
  if (where.error || where.hijacked || !String(where.value).startsWith(url)) notDriven(JSON.stringify(where));
};
await arrived();
// Record violations from before the page's first script runs, then load the page again with the recorder in place.
const recorder = await within(page.cdp("Page.addScriptToEvaluateOnNewDocument", { source: `
  window.__csp = [];
  document.addEventListener("securitypolicyviolation", e => window.__csp.push({
    directive: e.effectiveDirective, blocked: e.blockedURI, at: (e.sourceFile || "") + ":" + e.lineNumber, sample: e.sample }));
` }), 15000);
if (recorder.stalled) notDriven("the violation recorder could not be installed in 15 s");
await page.reload();
await sleep(2500);
await arrived();

// A 2x2 PNG.
const PNG = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAFElEQVR4nGP8z8DwnwEEmBigAAAfCgIBjw0G0QAAAABJRU5ErkJggg==";

const program = `(async () => {
  const results = window.__results = [];   // readable from outside if this program stalls
  const check = (name, ok, why) => results.push(ok ? { name, ok: true } : { name, ok: false, why: String(why) });
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const until = async (cond, ms) => { const t0 = Date.now(); while (!cond() && Date.now() - t0 < ms) await sleep(100); return cond(); };
  const within = (p, ms) => Promise.race([p, sleep(ms).then(() => "stalled")]);
  const seen = () => window.__csp.map(v => v.directive + " " + v.blocked + " (" + v.at + ")");
  if (!Array.isArray(window.__csp)) return [{ name: "the violation recorder is installed", ok: false, why: "window.__csp missing" }];

  // ---- the policy is there ----
  const head = await within(fetch("/"), 10000);
  const policy = head === "stalled" ? "(no answer in 10 s)" : head.headers.get("content-security-policy") || "";
  check("the page is served with a policy that names every source", policy.includes("default-src 'none'") && policy.includes("script-src 'self'"), JSON.stringify(policy));

  // ---- what the page loads ----
  check("the page's scripts ran", typeof send === "function" && typeof Markdown === "object" && typeof copySafe === "function", typeof send + "/" + typeof Markdown);
  check("the stylesheet applied", getComputedStyle(document.documentElement).getPropertyValue("--gi-accent").trim() !== "", "no --gi-accent on :root");
  const styled = document.querySelector('[style*="flex:0 0 110px"]');
  check("a style attribute in the page applied", !!styled && getComputedStyle(styled).flexBasis === "110px", styled ? getComputedStyle(styled).flexBasis : "no such element");
  await until(() => $("model").value !== "", 20000);
  const model = $("model").value;
  check("the model list arrived over fetch", model !== "", $("status") ? $("status").textContent : "no model");

  // ---- an image: a data: URI, made through a blob: URL ----
  const bytes = Uint8Array.from(atob(${JSON.stringify(PNG)}), c => c.charCodeAt(0));
  const loads = src => new Promise(ok => { const i = new Image(); i.onload = () => ok(i.naturalWidth > 0); i.onerror = () => ok(false); i.src = src; setTimeout(() => ok(false), 5000); });
  check("a data: image loads", await loads("data:image/png;base64," + ${JSON.stringify(PNG)}), "blocked or broken");
  const blobURL = URL.createObjectURL(new Blob([bytes], { type: "image/png" }));
  check("a blob: image loads", await loads(blobURL), "blocked or broken");
  URL.revokeObjectURL(blobURL);
  const vision = model !== "" && typeof visionOK === "function" && visionOK();
  if (vision) {
    await within(attachFile(new File([bytes], "dot.png", { type: "image/png" })), 10000);
    await until(() => !$("attach-preview").hidden, 5000);
    const shown = $("attach-preview").querySelector("img");
    check("an attached image is shown in the page", !!shown && await until(() => shown.naturalWidth > 0, 5000), shown ? "naturalWidth " + shown.naturalWidth : $("attach-preview").textContent);
  }

  // ---- a reply, streamed over fetch ----
  if (model !== "") {
    if ($("max")) { $("max").value = "24"; $("max").dispatchEvent(new Event("input", { bubbles: true })); }
    $("prompt").value = "Say hello in three words.";
    const before = document.querySelectorAll("#log .msg.bot").length;
    await within(send(), 60000);
    const done = await until(() => $("stop-btn").hidden && document.querySelectorAll("#log .msg.bot").length > before, 20000);
    const bot = [...document.querySelectorAll("#log .msg.bot")].pop();
    check("a reply streamed into the page" + (vision ? " (with the image)" : ""), done && !!bot && bot.children[1].textContent.trim() !== "", bot ? bot.textContent.slice(0, 200) : "no reply");
  }

  // ---- the HTML preview: a sandboxed data: frame ----
  const box = document.createElement("div");
  box.className = "md";
  $("log").appendChild(box);
  Markdown.render(box, "\\x60\\x60\\x60html\\n<p>hello from the preview</p>\\n\\x60\\x60\\x60");
  box.querySelector("button.md-preview").click();
  const frame = box.querySelector("iframe.md-preview-frame");
  const loaded = frame ? await new Promise(ok => { frame.addEventListener("load", () => ok(true)); setTimeout(() => ok(false), 5000); }) : false;
  check("the HTML preview's frame loaded", loaded && !frame.hidden && frame.getClientRects().length > 0, frame ? "no load event" : "no frame");
  await sleep(300);

  check("the page did all of that with no policy violation", window.__csp.length === 0, seen().join("; "));

  // ---- the policy is in force: each of these must be refused ----
  const refused = async (name, directive, act, stillClean) => {
    const n = window.__csp.length;
    let threw = "";
    try { await within(act(), 5000); } catch (e) { threw = String(e); }
    await until(() => window.__csp.slice(n).some(v => v.directive.startsWith(directive)), 3000);
    const got = window.__csp.slice(n).filter(v => v.directive.startsWith(directive));
    check("refused: " + name, got.length > 0 && stillClean(), got.length ? "it took effect anyway" : "no " + directive + " violation" + (threw ? " (threw " + threw + ")" : ""));
  };
  const elsewhere = "http://127.0.0.1:9/";   // another origin; nothing needs to answer, the request must not be made
  await refused("an inline script", "script-src", () => { const s = document.createElement("script"); s.textContent = "window.__inline = 1"; document.head.appendChild(s); }, () => window.__inline === undefined);
  await refused("an inline event handler", "script-src", () => { const b = document.createElement("button"); b.setAttribute("onclick", "window.__handler = 1"); document.body.appendChild(b); b.click(); b.remove(); }, () => window.__handler === undefined);
  await refused("a script from another origin", "script-src", () => { const s = document.createElement("script"); s.src = elsewhere + "x.js"; document.head.appendChild(s); }, () => true);
  await refused("an image from another origin", "img-src", () => { new Image().src = elsewhere + "x.png"; }, () => true);
  await refused("a request to another origin", "connect-src", () => fetch(elsewhere + "v1/models"), () => true);
  await refused("a frame from another origin", "frame-src", () => { const f = document.createElement("iframe"); f.src = elsewhere; document.body.appendChild(f); }, () => true);
  await refused("a stylesheet from another origin", "style-src", () => { const l = document.createElement("link"); l.rel = "stylesheet"; l.href = elsewhere + "x.css"; document.head.appendChild(l); }, () => true);
  return results;
})()`;
// Under go test's three-minute limit, so a stall is reported with the checks that finished before it.
let out = await within(page.evaluate(program, 150000), 150000);
if (out.stalled) {
  const partial = await within(page.evaluate("window.__results || []"), 5000);
  out = { value: [...(Array.isArray(partial.value) ? partial.value : []), { name: "the smoke ran to its end", ok: false, why: "it stalled after the checks above" }] };
}
page.close();
report("webui content-policy smoke", finishOrExit("webui content-policy smoke", out, []), page.exceptions);
