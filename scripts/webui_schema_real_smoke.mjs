// webui_schema_real_smoke.mjs — the response-schema control (W36) against a REAL `serve -web`, in headless Chrome.
//
//   node scripts/webui_schema_real_smoke.mjs http://127.0.0.1:8123/        # exit 0 pass / 1 a check failed / 2 no browser
//
// Not part of CI: it needs a running server with GLM-OCR loaded (a vision model that reads an image into a schema), a GPU or
// a lot of patience, and takes a minute or two. scripts/webui_app_gate.mjs is the CI gate (a fake fetch); this is what shows
// the page's request is accepted by the real server and the reply comes back constrained. It drives the page through its own
// code, as the gate does: attachFile() with the committed rendered invoice, the schema typed into the Sampling box's field,
// send() with an EMPTY message (the server's GLM-OCR rule then builds the extraction prompt from the schema), then reads what
// the page rendered.
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { openPage, report, finishOrExit } from "./webui-gate/cdp.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const url = process.argv[2] || "http://127.0.0.1:8123/";
const png = readFileSync(resolve(root, "testdata", "glm_ocr", "invoice.png")).toString("base64");
const schema = readFileSync(resolve(root, "testdata", "glm_ocr", "invoice.schema.json"), "utf8");

const page = await openPage(url, { settleMs: 3000 });
const out = await page.evaluate(`(async () => {
  const results = [];
  const check = (name, ok, why) => results.push(ok ? { name, ok: true } : { name, ok: false, why: String(why) });
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const until = async (cond, ms) => { const t0 = Date.now(); while (!cond() && Date.now() - t0 < ms) await sleep(250); return cond(); };
  if (typeof send !== "function") return [{ name: "page loaded", ok: false, why: "send() missing: is -web on, and is this the page?" }];

  await until(() => $("model").value !== "", 20000);
  const model = $("model").value;
  check("a model is loaded and selected", model !== "", model);
  await until(() => !$("attach").hidden, 20000);
  check("the model is a vision model: Attach image is offered", !$("attach").hidden, "attach hidden");

  const bytes = Uint8Array.from(atob(${JSON.stringify(png)}), c => c.charCodeAt(0));
  await attachFile(new File([bytes], "invoice.png", { type: "image/png" }));
  check("the invoice is attached", !$("attach-preview").hidden, "no preview");

  $("sampling-box").open = true;
  const set = (id, v) => { $(id).value = v; $(id).dispatchEvent(new Event("input", { bubbles: true })); };
  set("temp", "0");
  set("max", "2048");   // the default 512 would cut a long extraction off
  set("schema", ${JSON.stringify(schema)});
  check("the summary says a schema is on", $("sampling-state").textContent.includes("schema on"), $("sampling-state").textContent);
  check("no sampling error", $("sampling-error").hidden, $("sampling-error").textContent);

  const t0 = Date.now();
  $("prompt").value = "";           // empty: the extraction prompt is built server-side from the schema
  await send();
  const done = await until(() => $("stop").hidden, 280000);
  const secs = ((Date.now() - t0) / 1000).toFixed(0);
  check("the reply finished (" + secs + " s)", done, "still generating");

  const bot = [...document.querySelectorAll("#log .msg.bot")].pop();
  const code = bot && bot.querySelector(".md-code pre code");
  check("the reply is shown as a JSON code block", !!code, bot && bot.textContent.slice(0, 200));
  let doc = null, perr = "";
  try { doc = JSON.parse(code ? code.textContent : ""); } catch (e) { perr = String(e); }
  check("the reply parses as JSON", doc !== null, perr + " | " + (code ? code.textContent.slice(0, 160) : ""));
  const want = ["invoice_number", "date", "due_date", "vendor", "bill_to", "currency", "paid", "line_items", "subtotal", "tax", "total"];
  check("the reply has exactly the schema's keys, in order", doc && JSON.stringify(Object.keys(doc)) === JSON.stringify(want), doc && Object.keys(doc).join(","));
  check("the reply read the invoice: number, six line items, total", doc && doc.invoice_number === "INV-2026-0417" && Array.isArray(doc.line_items) && doc.line_items.length === 6 && doc.total === 1140.55, doc && JSON.stringify([doc.invoice_number, (doc.line_items || []).length, doc.total]));
  const entry = (typeof transcript !== "undefined") ? transcript[transcript.length - 1] : null;
  check("the reply is stored as structured", entry && entry.structured === true && entry.state === undefined, JSON.stringify(entry && { structured: entry.structured, state: entry.state }));
  check("reply text: " + (code ? code.textContent.replace(/\\s+/g, " ").slice(0, 300) : "(none)"), true);
  return results;
})()`, 330000);

const results = finishOrExit("webui schema real smoke", out);   // exits 1 itself if the page threw or navigated
page.close();
report("webui schema real smoke", results, page.exceptions);
