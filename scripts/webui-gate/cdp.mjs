// cdp.mjs — the shared headless-Chrome driver for the web UI's browser gates
// (scripts/webui_md_gate.mjs, scripts/webui_app_gate.mjs).
//
// Pages are loaded from file://, never from a server: it needs no port, and it works on a box whose
// sandbox blocks headless Chrome from loopback HTTP (nobara-pc, 2026-09-14). The page under test
// references its assets relatively, which is what makes that possible.
//
// Exit-code contract every gate keeps: 0 pass, 1 a check failed, 2 the browser could not be driven.
// A page that NAVIGATES while a gate runs is exit 1, not 2 — nothing in a gate navigates, so a payload
// or a bug did, and that is a failure of the thing under test.
import { spawn, execSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const sleep = ms => new Promise(r => setTimeout(r, ms));

function chromeBinary() {
  if (process.env.CHROME_BIN) return process.env.CHROME_BIN;
  for (const b of ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"]) {
    try { execSync(`command -v ${b}`, { stdio: "ignore", shell: "/bin/sh" }); return b; } catch { /* next */ }
  }
  return "google-chrome";
}

// openPage launches Chrome, navigates to url, and returns { evaluate, close }. evaluate(expr) runs an
// async expression in the page and resolves to its returned value, or to { hijacked } / { error }.
//
// ONE BROWSER PER RUN, AND IT IS REALLY GONE AFTERWARDS (found 2026-09-14, W7). This used to pick a random
// port and kill the spawned process on close — but `google-chrome` is a launcher script, so that killed the
// script and left the browser running. Every gate run leaked a Chrome (181 processes had piled up), and a
// later run whose random port collided with a leaked one drove THAT browser: an old page, with an old
// run's localStorage, so "fresh profile" checks failed intermittently. Now Chrome picks a free port itself
// (port 0) and writes it into the run's own profile directory, so a run can only reach its own browser;
// the browser is started as its own process group and the whole group is killed; the profile is removed.
export async function openPage(url, { settleMs = 1500 } = {}) {
  const bin = chromeBinary();
  const profile = mkdtempSync(join(tmpdir(), "webui-gate-"));
  const chrome = spawn(bin, ["--headless=new", "--disable-gpu", "--no-sandbox", "--allow-file-access-from-files",
    "--remote-debugging-port=0", "--remote-allow-origins=*",
    `--user-data-dir=${profile}`, "about:blank"], { stdio: "ignore", detached: true });
  const stop = () => {
    try { process.kill(-chrome.pid, "SIGKILL"); } catch { /* already gone */ }
    try { rmSync(profile, { recursive: true, force: true }); } catch { /* best effort */ }
  };
  process.on("exit", stop);   // also on a check failure's process.exit, and on die()
  const die = (code, msg) => { console.error(msg); process.exit(code); };
  chrome.on("error", e => die(2, `could not start ${bin}: ${e.message}`));

  let target;
  for (let i = 0; i < 80 && !target; i++) {
    try {
      const port = readFileSync(join(profile, "DevToolsActivePort"), "utf8").split("\n")[0].trim();
      target = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).find(t => t.type === "page");
    } catch { /* not up yet */ }
    if (!target) await sleep(250);
  }
  if (!target) die(2, "chrome exposed no page target");

  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((ok, no) => {
    ws.addEventListener("open", ok);
    ws.addEventListener("error", () => no(new Error("devtools socket error")));
    setTimeout(() => no(new Error("devtools socket did not open in 10s")), 10000);
  }).catch(e => die(2, e.message));

  let seq = 0;
  const waiting = new Map();
  const exceptions = [];
  ws.addEventListener("message", ev => {
    const m = JSON.parse(ev.data);
    if (m.id && waiting.has(m.id)) { waiting.get(m.id)(m); waiting.delete(m.id); return; }
    if (m.method === "Runtime.exceptionThrown") {
      exceptions.push(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
    }
  });
  const send = (method, params = {}) => new Promise(r => { const id = ++seq; waiting.set(id, r); ws.send(JSON.stringify({ id, method, params })); });
  await send("Runtime.enable");
  await send("Page.enable");
  await send("Page.navigate", { url });
  await sleep(settleMs);

  return {
    exceptions,
    async evaluate(expression, timeout = 60000) {
      const r = await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true, timeout });
      const err = r.error?.message || r.result?.exceptionDetails?.exception?.description || r.result?.exceptionDetails?.text || "";
      if (/context was destroyed|navigated|target closed|Cannot find context/i.test(err)) return { hijacked: err };
      if (err) return { error: err };
      return { value: r.result?.result?.value };
    },
    // reload is a DELIBERATE navigation between gate phases (W3: does state survive it?). It is the one
    // navigation a gate may perform; one that happens inside evaluate() is still reported as a hijack.
    async reload() {
      await send("Page.reload", { ignoreCache: true });
      await sleep(settleMs);
    },
    // cdp sends one raw DevTools command — for what evaluate() cannot do from inside the page: emulating
    // prefers-color-scheme or a phone-width viewport (W15, W16), or capturing a screenshot. Resolves to the
    // command's result, or throws with the protocol's error message.
    async cdp(method, params = {}) {
      const r = await send(method, params);
      if (r.error) throw new Error(method + ": " + r.error.message);
      return r.result;
    },
    close() { try { ws.close(); } catch { /* closed */ } stop(); },
  };
}

// report prints a results array ({name, ok, why}) plus uncaught page exceptions, and exits.
export function report(title, results, exceptions = []) {
  let failed = 0;
  for (const x of results) {
    if (!x.ok) failed++;
    console.log((x.ok ? "  ok    " : "  FAIL  ") + x.name + (x.ok ? "" : "\n          " + x.why));
  }
  for (const e of exceptions) { failed++; console.log("  FAIL  uncaught page exception: " + e); }
  console.log(`${title}: ${results.filter(x => x.ok).length} passed, ${failed} failed`);
  process.exit(failed ? 1 : 0);
}

// finishOrExit handles an evaluate() outcome that is not a results array.
//
// EXIT 2 IS ONLY FOR "THE BROWSER COULD NOT BE DRIVEN" — openPage failing to start Chrome or reach it.
// Once the page is loaded and running, anything that stops a gate program from returning results is
// exit 1: an exception inside it means the page was not in the state the gate expects (a missing
// restored bubble dereferenced as undefined, say), which is a failure of the page. This matters
// because go test turns exit 2 into a SKIP — found by mutation (W3): with restore-on-load deleted,
// the gate threw, exited 2, and a broken restore would have reached CI as a skipped test.
export function finishOrExit(title, out, earlier = []) {
  // print what earlier phases established before exiting, so one bad phase does not erase the rest
  const flush = () => { for (const x of earlier) console.log((x.ok ? "  ok    " : "  FAIL  ") + x.name + (x.ok ? "" : "\n          " + x.why)); };
  if (out.hijacked || out.error || !Array.isArray(out.value)) flush();
  if (out.hijacked) {
    console.log(`  FAIL  the page navigated away during the gate — something took effect (${out.hijacked})`);
    console.log(`${title}: FAILED (page hijacked)`);
    process.exit(1);
  }
  if (out.error) {
    console.log(`  FAIL  the gate program threw — the page is not in the state the gate expects:\n          ${out.error.split("\n")[0]}`);
    console.log(`${title}: FAILED (gate program threw)`);
    process.exit(1);
  }
  if (!Array.isArray(out.value)) {
    console.log(`  FAIL  the gate program returned no results: ${JSON.stringify(out.value).slice(0, 300)}`);
    console.log(`${title}: FAILED (no results)`);
    process.exit(1);
  }
  return out.value;
}
