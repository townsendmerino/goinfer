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

## (c) Gates — REGISTERED 2026-10-15, for the proposal-2 build (the design in (a))

Registered in a commit of its own, before any of the feature is built. The commit is named at the end of this section. The
fixtures are this replay's: `run.jsonl.gz`, the requests `prep.py` rebuilds, the order `select.txt`, and the echo rule of the
method below. "Today's tokenization" means the next request rendered with the carrier removed, on the same build. All four gates
share three definitions:
- a **fallback** re-tokenizes exactly as today;
- every fallback is **counted** with its reason in `ids_fallback` and the `status` totals, per condition b;
- a missing count fails the gate.

**G-ids-0. Day, CPU, through the route's renderer: tokenizer and template, no weights.**
- **Rule (all of it, no tolerance):**
  1. **Carried and untouched.** For each of the 45 replies, the next request carries the reply's items as emitted, with a v2
     blob signed by the test key. Its prompt ids start with the slot's id record (the prompt, then every generated id), and the
     line reads `ids=1/1` with no fallback. That includes the 4 non-canonical replies, which give 41 of 45 today.
  2. **Planted defects, each applied to every reply it fits.** Each must give today's tokenization exactly, the named reason,
     and no error:

     | defect | expected reason |
     |---|---|
     | (a) one character of the reasoning text inside the blob changed | `decode` |
     | (b) one character of a call's `arguments` changed | `decode` |
     | (c) an escape edited in `arguments` (`\\t` to `\t`), where one occurs | `decode` |
     | (d) one character of the final answer changed (the one reply with a message) | `decode` |
     | (e) **forged:** ids that decode exactly to the echoed text, in a non-canonical split chosen by the client, with no MAC | `mac` |
     | (f) **forged:** the same ids with a MAC from the wrong computation (random bytes) | `mac` |
     | (g) the reply's own ids, signed by a **different key** | `key` |
     | (h) a blob signed with the right key whose ids put a special id inside a content run while decoding to the same text | `special` |
     | (i) the right ids and key, minted for another namespace | `ns` |
     | (j) the right ids and key, minted for another model or tokenizer identity | `model` |
     | (k) a v1 blob (text, no ids) | `v1` |
     | (l) a truncated or garbled blob | `parse` |
     | (m) the reasoning item dropped | no carrier, so `ids=0/0` and today's tokenization; the reasoning is not rendered, as today |

  3. **Edits to the visible text only.** An edit to the reasoning item's visible `content`, with the blob untouched, renders
     the blob's text and uses the ids, as serve reads `encrypted_content` first today.
- **Void:** the fixture fails its own validity (a slot that does not start with the route's rendered prompt), or a planted edit
  leaves the rendered text unchanged (the defect did not plant).
- **Stopping rule:** every case once.
- **Cost basis:** the render check in the method ran 129 prompts in 2.6 s. This is 45 x 15 renders, about 15 s.

**G-ids-1. Day, GPU: the step-0 replay with the feature on.**
- **Rule:** in every completed reply, the next request re-renders to the generated ids. Today that is 41 of 45, and the 4
  misses must now hold. The archived history in each generation prompt carries the night's v1 blobs, so `v1` fallbacks there
  are expected and reported. There are zero fallbacks of any other reason.
- **Void:** the prompt-count offset is not the constant -18; any render error; the slot-prefix validity fails; the flags are not
  this run's (`--ctx 16384 --moe-cache-experts --windowed-kv=false`, KV f32).
- **Stopping rule:** this run's: the fixed order in `select.txt`, and no request starts after 600 s of replay wall.
- **Cost basis:** this run, 613 s of replay plus about 60 s of load, under the timing lock.

**G-ids-2. Night, GPU: a fallback is the cold path, and prompt ids do not depend on the KV.**
- **Arms.** Each of the 4 misses (`write` r2 #4 and #5, `fix` r3 #1 and #9) is run in its next request under these arms, in this
  order:

  | arm | what the next request carries | how it is run |
  |---|---|---|
  | A-warm | the valid carried blob | right after its reply |
  | A-cold | the valid carried blob | cold |
  | B | the forged blob of G-ids-0 (e) | cold |
  | C | the different-key blob of G-ids-0 (g) | cold |
  | D | the carrier removed | cold, twice (D1, D2) |

  "Cold" means the slot last held another session's first request, so only the shared preamble can be reused.
- **Rule:**
  - B and C: counted with `mac` and `key`; their prompt ids identical to D's; their generated output byte-identical to D's.
  - A-cold: its prompt ids equal to A-warm's (hard), and different from D's at the non-canonical split.
  - The outputs of A-warm against A-cold are reported, not barred: §5.6 property 3 allows a continuation to differ from cold.
- **Void:**
  - D1 and D2 differ (the cold path is not reproducible, so no output comparison counts);
  - an arm is not cold or not warm as stated, by its `cached_tokens` (cold: at most the preamble it shares with the previous
    prompt; warm: the whole previous prompt plus the reply);
  - a server restart inside the gate (it changes the default key).
- **Stopping rule:** 4 replies x 6 arms = 24 requests, once.
- **Cost basis:** each cold arm prefills 1K to 6K tokens past the shared preamble, at this load's 100 tokens/s (the night's
  median cold rate), plus the reply. That is about 20 cold arms x 35 s, plus 4 warm, so about 15 minutes with the load.

**G-ids-3. Night, live.**
- **Arms:** `mellum-16k-codex-f`, 5 tasks x 3, feature off (the do-nothing arm) and on, interleaved by task in one session, with
  `"bodies": true` (and `"token_ids": true` once (e)'s flag exists).
- **Rule:**
  - the on arm's cold plain continuations are at most 1, and at most the off arm's;
  - the on arm has zero `mac`, `key`, `model` and `ns` fallbacks (one server, one key: any of them means the carrier or the MAC
    is broken);
  - `decode` fallbacks are reported, each attributed from the bodies (a client edit or a defect);
  - compactions and the pooled pass rate are reported beside, with an ambiguous → parked band for the owner to set.
- **Void:** a deadline skip; a sandbox self-test failure; the expert-cache cap differing between the arms; a server restart
  inside an arm.
- **Stopping rule:** fixed N, 15 runs an arm.
- **Cost basis:** the after run, 2 h 28 min for 15 runs, so about 2.5 h an arm. That is two night jobs under the 3-hour cap.

**Registration commit: `dd04f022`** (the gates above, as committed there; the design they test is `96fe0edd`).

### (c) Amendment 1, before any code (2026-10-15): the gates in the tiered form

The owner's tiered testing rule (2026-10-15) applies to new gates: tier 0 (seconds: deterministic counters and bit-identity on
tiny fixtures) is the default; tier 1 (10 minutes or less, by day, one small real model at the smallest size that exercises
the mechanism); tier 2 (night) only for a published number or a decision on a small effect, with the reason tiers 0 and 1
cannot answer it and a wall-clock budget. An unchanged arm is not re-measured: its last readings are cited. Each check stops
when its answer is clear. The gates above are regrouped as follows. Every rule, void condition and planted defect above keeps
its wording unless this amendment changes it.

**Tier 0 (seconds, CPU, every change).**
- **G-ids-0, unchanged:** 45 renders x 15 cases, about 15 s.
- **G-ids-2's fallback half, moved here from the night:** arms B (forged, G-ids-0 (e)), C (different key, (g)) and D (no
  carrier, run twice), on a tiny fixture model whose greedy output is deterministic, with a reply whose tokenization is made
  non-canonical by construction (a planted split the BPE would not produce). Rule: B and C counted `mac` and `key`, their
  prompt ids equal to D's, and their generated output byte-identical to D's. Void: D1 and D2 differ. Counters read: the fallback
  counts by reason, the ids actually prefilled, and the rows the slot reused.
- **Prompt ids do not depend on the KV (G-ids-2's A-arms), on the tiny fixture:** the same request with the valid blob, sent
  right after its reply (warm) and after another session has taken the slot (cold), gives identical prompt ids; the cold
  arm's rows-reused counter shows the slot really was cold.

**Tier 1 (by day, under the timing lock, 10 minutes or less): G-ids-1, trimmed, which also takes G-ids-2's real-model
check.** Mellum2.1 is the smallest model on this box that produced the non-canonical splits; the mechanism needs a real model's
generation, which no tiny fixture reproduces.
- **Cells:** the 4 misses (`write` r2 #4 and #5, `fix` r3 #1 and #9), and 8 controls drawn from the 41 that held, in `select.txt`'s
  order: 2 each from JSON escapes, `./...` and code; the 1 non-ASCII reply that held; and 1 from none of the four.
- **Rule:** all 12 re-render to their generated ids with the feature on, with zero fallbacks except `v1` on archived history.
  For the 4 misses, the next request's prompt ids are equal warm (right after the reply) and cold (after another session's
  request), and differ from the no-carrier arm's at the non-canonical split.
- **Void:** G-ids-1's void conditions, and an arm whose `cached_tokens` contradicts its stated warm or cold state.
- **Stopping rule:** every cell once.
- **Cost basis:** step 0's 13.6 s a request, so 12 replays plus 8 next requests is about 4.5 minutes, plus about 60 s of
  load: under 6 minutes.
- G-ids-1's full 45-request replay (about 11 minutes, over the tier-1 cap) and G-ids-2's 24-request night job are withdrawn.
  The tier-0 checks above cover their deterministic part, and the trimmed replay covers the real-model part.

**Tier 2 (night, one job): G-ids-3, for the reuse rate on real Codex traffic only.**
- **Why tiers 0 and 1 cannot answer it:** the number is how often a live Codex session's follow-ups stay warm. That depends on
  Codex's own echoing, compaction and turn structure, which a replay of recorded prompts does not reproduce; a by-day replay of
  all 285 recorded follow-ups would take about 65 minutes and still not include Codex's behaviour. It is also a number the
  release notes may quote.
- **The off arm is not re-run.** It is the after run of 2026-10-15 (serve at `695ce16c`; 55 of 285 follow-ups cold), cited
  from `i3-night-2026-10-13.md`. That holds only if nothing on the reuse path (the windowed-KV ring, prefix reuse, the
  Responses render of reasoning items, the slot choice) has landed between `695ce16c` and the commit under test. If something
  has, for example KV2's merge, the off arm is re-run in the same job, interleaved by task, and the budget doubles.
- **Rule:** G-ids-3's rule above, read against the cited (or re-run) off arm.
- **Stopping rule:** fixed N, 15 runs. If the first 5 runs show no cold plain continuation and no fallback other than `decode`,
  the answer is clear and the job stops there; that is a reading of "stop as soon as the answer is clear", registered here
  before any run.
- **Budget:** the on arm alone, about 2 h 30 min (the after run's 2 h 28 min for 15 runs), or about 50 minutes if it stops
  at 5; with a re-run off arm, about 5 hours, so two night jobs.
- **Fit guard, by day before queueing:** Mellum2.1 at `--ctx 16384 --moe-cache-experts` on this card. Its last successful
  load is the after run (62 expert slots).

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
