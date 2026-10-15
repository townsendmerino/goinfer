# Id round trip, step 0: do generated ids survive decode, echo and re-render? (2026-10-15)

nobara-pc, RTX 2070 SUPER, driver 595.91.07. serve's handler at `origin/main` 4d393367 (the test binary was built at 1a510c14, that commit plus this record's method; the harness runs the handler in process),
Mellum2.1-thinking int4 `.giw` from `~/models` (`--model ~/models/mellum2.1-thinking`, as the night). One run, 45 replies, 613 s of
replay wall, under the timing lock. Raw replay output: `id-roundtrip-step0/run.jsonl.gz` (generated ids and text per reply, the
same content class as the committed rollouts), its log `run.log`, the grade `analysis.txt` (printed by `analyze.py`). The method,
registered before the run, is the last section.

## Result

**41 of 45 greedy replies re-render to the ids the model generated (91%, Wilson 95% interval 79% to 96%).** All 4 that do not are
**non-canonical ids**: the echoed text re-renders to exactly the decoded text, and only the split into tokens differs. No reply
re-rendered to different text, so the template and serve's parse and emit lose nothing on this sample, and no client edit is
in play (below). A one-line check predicts the outcome exactly: every reply whose generated ids are the tokenizer's own encoding
of their text, run by run between special tokens, round-trips (41 of 41), and every reply with a non-canonical run does not (0 of 4).

| | round trip holds | 95% interval |
|---|---|---|
| all replies | 41 of 45 | 79% to 96% |
| replies of 448 tokens or less | 35 of 36 | 86% to 100% |
| replies over 448 tokens | 6 of 9 | 35% to 88% |
| replies with a backslash in the tool-call JSON | 4 of 6 | 30% to 90% |
| replies with `./...` | 4 of 4 | 51% to 100% |
| replies with non-ASCII | 1 of 4 | 5% to 70% |
| replies with code (`func `, `:=`, a fence, `package `, a here-document) | 15 of 19 | 57% to 91% |
| replies in none of the four | 22 of 22 | 85% to 100% |

The reply categories are confounded with length: the replies that carry non-ASCII or code are the long ones, and a long reply
has more places to go non-canonical. **The divergence points themselves** say more:

| reply | length | diverges at | part | generated | canonical re-encoding |
|---|---|---|---|---|---|
| `write` r2 #4 | 825 | 606 | tool-call JSON, a JSON escape | `\\` `t` `tests` | `\\` `tt` `ests` |
| `write` r2 #5 | 2,547 | 178 | reasoning, JSON quoted in prose | `"]}` `".` | `"]` `}".` |
| `fix` r3 #1 | 370 | 215 | reasoning, JSON quoted in prose | `"],` `"` `type` | `"],"` `type` |
| `fix` r3 #9 | 2,780 | 1,439 | reasoning, a space before a word | ` ` `Goal` | ` Goal` |

None diverged at `./...` or at non-ASCII. The night's `./...` and non-ASCII break points are not reproduced at n = 45 greedy; they
are the same mechanism (a split the BPE would not choose) at a different place.

**By part** (runs of generated ids between special tokens that are the tokenizer's own encoding of their text): reasoning
canonical throughout in 42 of 45 replies (93%, 82% to 98%), tool-call JSON in 43 of 44 (98%, 88% to 100%), the rest in 45 of 45
(the rest is the separators between the parts in all but one reply; one reply ended in a plain answer, and it round-tripped).

**What the ring would have made of the four.** Rolled back from the newest position: 2,369 and 1,341 (cold under the 448-position
rollback) and 219 and 155 (partial reuse). That is 2 cold continuations in 45 replies, and the night saw 15 in 243 plain steps.

**Against the night** (the night's bound from token counts, its replies sampled at temperature 1.0, serve's default when Codex sends
none): replies over 448 tokens diverged there in 21 of 49 (43%) and here in 3 of 9 (33%); replies of 448 or less in 8 of 194
(4%) and here 1 of 36 (3%). The two agree within these intervals. Greedy is the case measured; whether sampling at 1.0 produces
more non-canonical splits is not measured here.

## Causes

| cause | replies | what fixes it |
|---|---|---|
| **non-canonical ids** (text identical, split differs) | **4 of 4 mismatches** | reusing the generated ids |
| rendering difference (template, or serve's parse and emit) | 0 of 45 | nothing needed on this sample |
| client edits | none seen | nothing: an edited reply must re-tokenize |

**Client edits, read from the archive and the live capture, not from the replay** (the replay echoes as Codex does by
construction):
- **Reasoning:** all 298 reasoning items in the night's rollouts carry `content` text equal to the decoded `encrypted_content`,
  which is serve's own bytes. serve reads `encrypted_content` first (`internal/serveapp/reasoning_items.go:reasoningItemText`), so an edit to the
  visible text could not reach the prompt anyway.
- **Calls and messages:** in the 2026-10-14 live capture, the items Codex sent back equal the items serve emitted key for key,
  less `status` (2 reasoning, 2 function calls). The night's 213 full reuses re-rendered exactly, so Codex changed nothing there.
  Function-call arguments have no twin to compare, so an edit there would be invisible until a gate checks it (§(c)).

## (d) Codex compactions under smaller output caps (archive only, `id-roundtrip-step0/caps.py`, output `caps.txt`)

connect sets Codex's compaction trigger to the window less the output cap. The model is `diagnose_after_compactions.py`'s
(trajectories held fixed, only the trigger moves). As run it gives 15 compactions against the real 20, so read the rows against
each other.

| output cap | trigger | modelled compactions | recorded replies over the cap | compaction summaries over the cap |
|---|---|---|---|---|
| 4,096 (as run) | 12,288 | 15 | 0 of 278 (2 reached 4,096 and stopped there) | 0 of 20 |
| 3,072 | 13,312 | **12** | **12 of 278** | 1 of 20 |
| 2,048 | 14,336 | **9** | **18 of 278** | 4 of 20 |

The model does not say what a truncated reply does next. A truncated call is lost and its turn is redone, and a truncated summary
loses part of the history. Those replies are 4% and 6% of the run, so the compaction saving is an upper bound.

**Owner decision (2026-10-15): the cap stays at 4,096 for Codex.** A smaller cap truncates compaction summaries (1 of 20 at
3,072, 4 of 20 at 2,048), and a truncated summary costs more than the compactions saved. Recorded in the I3 notes too
(`i3-night-2026-10-13.md`, § "After run: the compactions and the cold continuations, diagnosed", part 4).

## (a) Design (owner decision, 2026-10-15): the reply's ids ride in its own opaque field, signed; no server-side store

**Decided:** the opaque-field design, not the store, with conditions (a) to (c) below. The store design and the reasons it lost
are kept after them. **There is no Chat Completions store for now.** It is revisited if opencode runs show cold or partial
continuations that this cause explains.

**What is carried, and where.** When a reply is committed, serve has its generated ids: the whole assistant turn, from the first
generated id to the last before the stop token, covering reasoning, calls and text. On `/v1/responses` the reasoning item's
`encrypted_content` carries them, in a new format beside the current one:
- the current `goinfer-reasoning-v1:<base64 text>` stays readable, and carries text only;
- `goinfer-reasoning-v2:<base64>` encodes: the format version, the reasoning text (which the field already carries), the ids
  (varint-packed: about 1.5 bytes an id, so 4,096 ids are about 8 KB before base64), the identity of the served model and its
  tokenizer, the key id, and the MAC.

`content` keeps the readable reasoning text as today.

**Condition a: signed with a key the server holds.**
- **The MAC.** HMAC-SHA256 over the version, the key id, the model and tokenizer identity, the request's D4 namespace, and the
  packed ids.
- **Why refusing special ids is not enough.** An unsigned blob would let a client choose an unusual tokenization of
  harmless-looking text. The model would then be conditioned on a token sequence it never produced, with a text check that
  passes.
- **How the key is held.** By default it is 32 bytes from `crypto/rand`, generated at server start and never written anywhere.
  A restart, or a second server, makes every older blob fail the check. It then falls back to re-tokenizing, counted under
  `key`, which is today's behaviour. An operator who wants ids to survive restarts or span servers configures the key with a
  flag naming a file (32 bytes or more, mode 0600, refused otherwise). It is not an env read (`docs/env-vars.md`, the policy).
- **The key id** is the first 8 bytes of SHA-256 of the key. It lets a blob from a different key be counted as `key`, not as a
  forgery.

**Condition b: the ids must decode to exactly the echoed text, or the turn re-tokenizes as today, and every outcome is counted.**
- **Where the check runs:** on the assistant turn's own segments, as the template renders them from the echoed items.
- **What it requires:**
  - the carried ids, split at their special ids, give the same sequence of special ids as the template's special segments for
    that turn;
  - each run of non-special ids decodes to exactly the text of the template's content segment in that place: reasoning, the
    rendered call JSON, the answer text;
  - no special id inside a content run.
- **When it passes,** the turn's ids are the carried ids. That would have given 45 of 45 here: in this replay, every rendered
  turn's text equalled the decoded generated text.
- **When it fails,** the turn is tokenized exactly as today, and the reason is counted:

  | reason | what failed |
  |---|---|
  | `v1` | the blob carries no ids |
  | `parse` | a malformed blob |
  | `mac` | a bad or missing signature |
  | `key` | another key id |
  | `model` | another model or tokenizer |
  | `ns` | another namespace |
  | `special` | the special ids differ, or a special id sits inside content |
  | `decode` | the text differs, which is a client edit |

- **Where the count lands:**
  - `-log-requests` gains `ids=<used>/<carried>` (turns that used their ids, of the turns that carried a blob) and
    `ids_fallback=<reason>,...`;
  - the totals by reason go in the admin socket's `status` report.
- **A fallback is the cold path by construction:** the prompt ids are the ones a request carrying no blob gets today.

**Condition c: replies with no reasoning item.**

| case | carrier | covered? |
|---|---|---|
| Responses, a reply that reasons (Codex x Mellum2.1 here) | the reasoning item's `encrypted_content` | yes |
| Responses, a reply with no reasoning (a non-thinking model, thinking off, `reasoning.effort: none`) | a carrier-only reasoning item (empty `content`, `encrypted_content` holding the ids), emitted only when the request asked for `include: ["reasoning.encrypted_content"]`, which says the client round-trips it | yes for clients that ask, as Codex does; no for the others |
| Responses, a client that drops reasoning items, or does not ask for `encrypted_content` | none (call and message items have no opaque field; `call_id` and the item `id` are echoed but are identifiers, and are not used as carriers) | **no** |
| Anthropic `/v1/messages`, a turn with a thinking block | the block's `signature`, empty today | yes, if the client echoes thinking blocks back (Claude Code does on tool turns; to be checked per client) |
| Anthropic, a turn with no thinking block (thinking off, a non-thinking model) | none: `text` and `tool_use` blocks have no opaque field. A `redacted_thinking` block carries opaque `data`, but sending one to a client that did not ask for thinking is unverified | **no** |
| Chat Completions | none, and no store for now | **no** |

**The step-0 sample's share without a carrier: 0 of 45 replies and 0 of 4 misses.** Every reply carried a reasoning item: 44
reasoning then a function call, 1 reasoning then a message. The night's rollouts agree: 298 of 298 responses had one. The cases
marked **no** are other workloads (other clients, thinking off, Chat Completions), and this sample cannot size them.

**Cold against warm, under this design.**
- **Prompt ids.** Within one key's lifetime they are a function of the request bytes alone: the same echoed items give the same
  ids, whatever the KV holds. With a configured key that holds across restarts and servers too. With the default key, a restart
  turns old blobs into counted `key` fallbacks, which tokenize exactly as a cold request with no blob.
- **Outputs.** Byte identity of outputs between warm and cold is not given by equal ids. A warm request reuses rows that decode
  wrote, while a cold one prefills the same ids in batch: `docs/tasks/task-kv-architecture-2026-10.md` §5.6 property 3, true for
  every continuation on main today. The gates below hold prompt-id identity as the bar and report output identity. If byte
  identity is wanted, the next turn would re-prefill the reply's rows instead of reusing them, costing the reply's length in
  prefill (a median of 184 tokens here, about 2 s on this load).

**The store design, not chosen.** A per-reply id store keyed by the echoed rendering also makes the ids independent of the KV.
But it makes them depend on the store's own contents: eviction, restart, which server answered, and what else ran (a problem for
KV2's schedule-independence gate). It also has to be namespaced for D4. The signed blob has none of these, because the ids arrive
in the requester's own request.

## (b) Fit with KV2 and D4

- **The chain.** KV1's index keys blocks by the chain hash of token ids. With the carried ids used, the chain runs through the
  reply instead of breaking inside it, so a conversation's next turn finds its own blocks. On CUDA the follow-up then diverges
  only after the reply, inside the ring's 448-position rollback. That removes this cause of the cold continuations whether or
  not the rollback is ever raised.
- **Across conversations.** §5.6 property 3 accepts decode-class rows only as a continuation of the same chain. KV2 therefore
  never copies a reply's decode-written rows to another conversation, even when the ids are equal, because the requester's cold
  class for those rows is a prefill class. What KV2 shares stays the prefill-class preamble. The design does not widen sharing
  and needs no change to KV2's gates.
- **D4.**
  - The namespace is inside the MAC. A blob minted in one namespace and presented in another falls back (`ns`, counted), so one
    namespace's generations never decide another's tokenization.
  - With serve's single API key there is one namespace today, so this is a no-op until namespaces multiply. It is in the MAC
    from the start so it does not become a format change later.
- **Schedule independence (KV2's gate) holds under this design.** The ids come from the request, not from shared state that
  other conversations could change.

## (c) Gates — DRAFT, not registered

**G-ids-0, day, CPU, seconds (unit, through the route).** Render the next request for every reply of this replay (`run.jsonl.gz`)
with the feature on.
- **Rule:**
  - every reply whose echo equals the emitted items gives exactly the slot's ids;
  - every **planted defect** gives today's tokenization (the id path rejected) with no error:
    - (1) one character of the reasoning changed;
    - (2) one character of a call's `arguments` changed, including a backslash-escape edit (`\\t` to `\t`);
    - (3) one character of a final answer changed;
    - (4) the reasoning item dropped;
    - (5) a blob whose ids decode to the same text but put a special id (`<|im_end|>`, `<tool_call>`) inside content;
    - (6) a blob or store entry written under another namespace.
- **Void:** the replay's own validity fails (a slot that does not start with the route's rendered prompt), or a planted defect
  fails to change the rendered text.
- **Stopping rule:** all cases, once.
- **Cost:** seconds; no model weights, tokenizer and template only.

**G-ids-1, day, GPU, about 11 minutes (this replay, feature on).**
- **Rule:** the round trip holds in 45 of 45, against today's 41 of 45. Each of the 4 mismatches above must now hold, and no
  reply may regress.
- **Void:** a prompt-count mismatch other than the constant 18; the windowed-KV setting other than `--windowed-kv=false`; a
  render error.
- **Stopping rule:** the same 600 s budget and order (`select.txt`).
- **Cost:** this run, 613 s plus the load.

**G-ids-2, day, GPU, about 5 minutes (prompt-id identity, cold against warm).**
- **Rule:** for the 4 non-canonical replies, the next request's prompt ids are identical when it runs warm (after its reply) and
  cold (after a server restart, blob form; after a cleared KV with the store kept, store form).
- **Report beside it, not a bar:** the outputs, compared byte for byte. They may differ under §5.6 property 3. If the owner
  chooses byte identity, the gate becomes byte identity with the reply's rows re-prefilled.
- **Void:** either arm not cold or not warm as stated (its `cached_tokens`).
- **Stopping rule:** 4 pairs.
- **Cost:** 8 requests at 60 to 120 s cold.

**G-ids-3, night (live).**
- **Arms:** `mellum-16k-codex-f`, 5 tasks x 3, two arms (feature off, the do-nothing arm; feature on), interleaved by task in one
  session.
- **Rule:** the feature-on arm's cold plain continuations are at most 1 against the off arm's count, and both are reported.
  Compactions and the pooled pass rate are reported beside, with an ambiguous → parked band for the owner to set.
- **Void:** a deadline skip, a sandbox self-test failure, the expert-cache cap differing between arms.
- **Stopping rule:** fixed N.
- **Cost:** about 2 x 2.5 h. Over the 3-hour job cap, so two jobs on two nights, or 5 tasks x 2.

## (e) Logging generated ids in night runs (APPROVED 2026-10-15): the mechanism

**What exists:**
- `-log-requests`: one line, counts and times;
- `cached_tokens` and `prefill_reused_tokens` in usage: the reused length, no position;
- the matrix's `wire_proxy.py`, which counted events and kept no bodies;
- no route returns ids.

**Built now, harness only (this PR): request and response bodies in night runs.**
- `matrix/wire_proxy.py` takes an optional fourth argument, a directory. Each POST's body goes to `<seq>.req.json.gz` and the
  response's whole byte stream to `<seq>.resp.gz`, and the wire log line names the pair (`"bodies": "<seq>"`).
- `matrix/matrix.py` passes `bodies-<config>/` in the run's output directory when the configuration has `"bodies": true`.
- Smoke-tested against a stub upstream: both files are written per request, the log line names them, and the bytes relayed are
  unchanged.
- With bodies, a grade can compare every echoed item with the emitted one (the client-edit check, per request) and re-render
  every prompt offline with this record's renderer.
- **The bodies hold the harness's prompts and the model's replies.** They stay in the run's output, beside the wire log, and a
  grade commits them only as `codex-rollouts.tar.gz` was committed: as a tarball, by decision.

**To build with proposal 2 (serve; specified here, not built):**
1. **Reuse fields on the `-log-requests` line** (`internal/serveapp/reqlog.go:reqTrace`, filled where the first generation
   reports in). Fields, in order after `total=`:
   - `reused=<n>`: the generation's `PrefillReused`;
   - `slot=<n>`: the length of the bound slot's id record before the request, which needs a decoder accessor beside
     `PrefillReused` on `decoder.Generation`;
   - `div=<n|->`: the first divergent position when reuse ended before the slot's end, else `-`;
   - `canon=<0|1>`: whether this reply's generated ids are canonical, run by run between special tokens (one `Encode` of each
     run's decoded text at commit; the check that predicted the outcome 45 of 45 here);
   - with proposal 2, `ids=<used>/<carried>` and `ids_fallback=<reasons|->` (condition b).

   These land in the night's `logs/serve-<config>.log`, archived as `serve-<config>.log.gz` as now. They carry no content.
2. **The generated ids, for measurement runs only.** A serve flag, `--log-token-ids <file>`, appends one JSON line per
   generation:
   - `t`, `route`, the response id;
   - `prompt_len`, `prompt_sha256` (SHA-256 of the prompt ids, which the bodies let a grade re-render and check);
   - `reused`, `slot`;
   - `gen` (the generated ids);
   - `canon`.

   It is a flag, not an env read, and it is off by default. In the matrix, a configuration with `"token_ids": true` adds
   `--log-token-ids <out>/token-ids-<config>.jsonl` to the serve arguments. The grade archives the file as
   `token-ids-<config>.jsonl.gz` beside `serve-<config>.log.gz`, under the same rule as the bodies.
3. **The night's grade then measures this record's question directly, on every request.** It reads the next request's body,
   renders it, and compares the result with the logged `gen` after the logged prompt. Bodies and `token_ids` are switched on
   together, for the runs that measure this.

## Method (written and committed before the run)

**The question** (owner, step 0 of proposal 2 in `i3-night-2026-10-13.md` § "After run: the compactions and the cold
continuations, diagnosed", part 3): replay archived Codex prompts greedily, capture the generated ids, and count how often the
reply, decoded and echoed back the way Codex echoes it, re-renders to the same ids. Break it down by JSON escapes, `./...`,
non-ASCII and code, and by part (reasoning, answer, tool-call JSON).

**What serve exposes today.** Nothing that carries generated ids on `/v1/responses`. `-log-requests` writes one line per request
(token counts and times, no ids, no reuse figure). `/v1/chat/completions` with `logprobs` returns each token's text, not its id,
and renders a different prompt. `usage.input_tokens_details.cached_tokens` gives the reused length only. So the replay runs
serve's own handler in process and reads the slot's id record after each request.

**Instrument.** `id-roundtrip-step0/idstep0_test.go.txt`, a throwaway test copied into `internal/serveapp/` for the run and not
committed there. It builds the server as `Main` does from serve's flags (`newServer`, `routes`) and POSTs each request to
`/v1/responses` through the real mux, streamed, as Codex sends it. After the request it reads the decoder's committed id record
(`resIDs`, the prompt followed by the generated ids, which is what prefix reuse compares) and checks that it starts with the
prompt as the route renders it. It then builds the next request as Codex would: the same input, then this reply's output items
less `status` (what Codex echoes; checked on the 2026-10-14 live capture, where the echoed reasoning and function_call items equal
the emitted ones key for key apart from `status`), then the archived items Codex appended before its next request (tool output,
inserted messages), with their call ids moved onto this reply's calls. It renders that through the route's own renderer
(`kv0Render`, KV0's) and compares token ids with the slot.

**Prompts.** `id-roundtrip-step0/prep.py` rebuilds each request from `codex-rollouts.tar.gz`: the first developer message and
the tool list from the live 2026-10-14 capture (same Codex 0.162.1, same base instructions in all 15 sessions), then the
rollout's items. Its check: rebuilding the live session's third request gives that request's captured input exactly. Requests
before each session's first compaction only (129; a compaction request's reply is not echoed as an assistant turn). **Render
check (CPU, done before the run):** all 129 render to the night's `input_tokens` less a constant 18, so every token after the
fixed preamble renders as on the night; the 18 are in the tool list or first developer message, which the night did not record.

**Sample and cost.** Greedy (`temperature` 0), `max_output_tokens` 4,096 as a guard (Codex sends none. Corrected after the run: this
said the night's largest reply was 3,584, which is its largest compaction summary; two night replies reached 4,096 and one 4,083). Every pre-compaction request of five sessions, in order (`select.txt`): `write` r2, `fix` r3, `write` r3, `edit` r2,
`search` r2, 52 requests, chosen for the categories their night replies carried (dots 8 of 9, non-ASCII 8 of 11, code 23 of 44,
escapes 13 of 30, of the 129). `scripts/power.py binomial --p 0.15 --half-width 0.10` gives N = 49 for a 95% half-width of 10
points; intervals are reported as Wilson intervals because n·p < 10. Load as the night (`--ctx 16384 --moe-cache-experts`,
KV f32) **with `--windowed-kv=false`**: the greedy reply differs from the archived one the next prompt carries, so under the ring
nearly every request would roll back past 448 and re-prefill 6K to 12K tokens cold; without it each request prefills its
growth. The ring changes which positions are kept, not the ids a request generates. **Estimate:** one cold 6.1K-token prefill
(about 60 s), then per request about 560 tokens of prefill (about 6 s) and about 430 generated (about 8 s): about 13 minutes for
52. **Stopping rule:** no request starts after 600 s of replay wall; the order is fixed above, so the cut does not depend on any
outcome. Under the timing lock (label `step0`).

**Per reply, recorded:** match (the next prompt's ids start with the slot's whole id record), first divergent position and the
tokens there, whether each run of generated ids between special tokens is the tokenizer's own (canonical) encoding of its text,
and whether the re-rendered text equals the decoded text. **Causes:** non-canonical ids (texts equal, ids differ: fixable by
reusing ids), a rendering difference (the re-rendered text differs from the decoded text: template or serve's parse and emit),
and client edits (Codex's echo differs from what serve emitted; read from the live capture and the archive, not the replay).
