# Web UI: a response-schema control (W36), 2026-10-02

The `-web` UI could attach an image to a vision model but had no way to constrain a reply, so GLM-OCR extraction through the page was
unconstrained OCR or a hand-typed prompt with no grammar (found by the O5 agent, `docs/tasks/task-glm-ocr-2026-10.md`). The server already
honoured `response_format` on `/v1/chat/completions` and on the `/v1/jobs` route text replies take. This adds the control.

## What was built (`internal/serveapp/webui/`)

- A **Response schema** textarea in the Sampling box (`index.html`, `ui/app.css`, `ui/app.js`). It takes a JSON Schema, or a whole `response_format`
  object pasted from an API call. Either is sent as `response_format: {type: "json_schema", json_schema: {name, schema}}` (a missing name becomes
  `"response"`) with every reply request: the chat route (images) and the jobs route (text). The background title request is not constrained by it.
- Validation is the page's job only as far as "is a JSON object"; the server compiles the schema and refuses one it cannot enforce, with the reason.
  An invalid value is marked, explained in the shared error line, opens the Sampling box, and Send, Regenerate and Edit refuse before anything changes
  (the W10 behaviour for every sampling field).
- Kept across reloads and followed across tabs like the other settings, but with a 65,536-character limit (the others are held to 4,000, and a real
  schema is longer). Reset clears it. The summary shows **`· schema on`** with the box closed, because a schema changes the shape of every reply.
- A reply generated under a schema is stored `structured: true` (only the boolean `true`, only on a reply, is accepted back from storage) and shown as a
  JSON code block, so its indentation survives Markdown and Copy on the block copies the JSON. The fence is longer than any run of backticks in the reply.
- The help line under the box says the two things that bite: Max tokens defaults to 512, which cuts off a long extraction, and a document-reading model
  such as GLM-OCR reads an attached image into the schema when the message is left empty.

## Evidence

- **The CI browser gate** (`scripts/webui_app_gate.mjs`, headless Chrome, a fake fetch) grew two phases, 19a and 19b, with a real reload between them:
  **543 passed / 0 failed before, 585 / 0 after** (42 new checks: what is sent for no schema, a JSON Schema and a pasted `response_format`, with and without a name;
  the jobs route; the title request; code-block rendering, indentation, a line of backticks and inline backticks; storage as structured; seven invalid inputs refused
  with the box opened and the message kept; clearing; and after a reload the restored schema, the rebuilt block, the hostile-stored-value rules, an over-limit
  stored schema dropped, and a 4,000-plus-character schema kept). `internal/serveapp` `TestWebUI_*` pass.
- **Red/green:** four mutations of the page, each run against the whole gate (`mutations.log`): `response_format` never sent, 9 checks red; no code block, 4 red;
  the stored-field whitelist loosened, 1 red; the fence forced to three backticks, 1 red. **The first version of the fence check stayed green under that mutation**:
  backticks in the middle of a one-line JSON string can never close a fence (a fence must start a line), and a reply the grammar kept valid cannot contain a line of
  only backticks, so the longer fence is a defence for a reply that is not clean JSON. The check was rewritten to use text that has such a line, and then went red.
- **A real server, a real browser** (`scripts/webui_schema_real_smoke.mjs`, `real_smoke.log`; not CI): `serve -web` on the CUDA build with GLM-OCR resident (int4,
  context 8192), headless Chrome on the page, the committed rendered invoice attached through the page's own `attachFile()`, the invoice schema typed into the field, Max tokens
  2048, an EMPTY message. **12 checks passed**: the reply finished in 47 s, rendered as a JSON code block with exactly the schema's keys in order, invoice number
  INV-2026-0417, six line items, total 1140.55, stored as structured. The invoice is a procedurally rendered document, not a scan; one run; the time is exploratory.

## Not done

- The schema is not validated beyond "a JSON object" in the page (the server owns that), so an unsupported keyword shows up as the server's 400 message in the reply bubble.
- No schema library, examples picker or "generate from JSON" helper; the field is a textarea with one example in its placeholder.
- No per-field confidence display (the server can report it with `goinfer_confidence`; the page does not ask for it).
- A reply that hits Max tokens mid-JSON is shown as a (truncated) code block with the existing "incomplete" note; the page does not try to repair it.
