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

## (a) How cold and warm prefills stay identical

**What the owner's store gives.** With the generated ids kept per reply, in a store keyed by the exact echoed rendering and
outliving the KV blocks, a warm request and a cold one tokenize the echoed reply identically. The token sequence then no longer
depends on what the KV holds. That part holds.

**Where it cannot hold as stated.**
- **Equal ids do not give byte-identical outputs.** A warm request reuses the reply's rows as decode wrote them; a cold one
  prefills the same ids in batch. `docs/tasks/task-kv-architecture-2026-10.md` §5.6 property 3 already says a continuation is not
  bit-identical to cold. That is true today for every full reuse, and the store neither adds nor removes it. A gate that demands
  byte-identical outputs warm against cold fails on main today without any id store. What the store can promise is identical
  prompt ids. Byte identity would need the next turn to re-prefill the reply's rows instead of reusing them, at the cost of the
  reply's length in prefill (a median of 184 tokens here, about 2 s on this load).
- **The store brings in a new dependence of its own.** Today the prompt ids are a function of the request bytes alone. With a
  store, they are a function of the request bytes and the store's contents. An evicted entry, a restart or a second server
  tokenizes the same request differently from an identical earlier one. That is the cold/warm split moved from the KV to the store.
  It is narrow, because only the non-canonical replies (4 of 45 here) need an entry at all. The canonicity check above, run once at
  commit, says which ones those are, and a canonical reply needs nothing stored. Entries are small too: about 4 bytes an id,
  about 3 KB for a 700-token reply.

**A better answer where the protocol allows it: carry the ids in the reply's own opaque field.** Responses has one:
`encrypted_content`, which serve already writes as the reasoning's bytes and which came back byte for byte in all 298 of the
night's items. Anthropic's `thinking.signature` is another, which serve sends empty today. If that blob carried the reply's
generated ids (the whole assistant turn's, reasoning, calls and text), plus a hash of the text they decode to, the prompt ids
would again be a function of the request bytes alone. Cold equals warm by construction, across restarts, evictions and servers,
and no store is needed. serve uses the ids only when the echoed items re-render to exactly the text the ids decode to; otherwise
it tokenizes as today. Chat Completions has no opaque field, so it keeps the store, with the dependence stated. Two conditions
apply to the blob:
- **Special ids.** A client can forge a blob. Decode-equal text is not enough, because a special id decodes to its surface form
  (`<|im_end|>`). serve must accept only ids that are not special inside content segments, in the template's own segmentation,
  or a blob forges turn boundaries that `EncodeSegments` exists to stop.
- **A dropped reasoning item.** A client that drops the reasoning item loses the ids, and the reply falls back to today's
  behaviour, not to anything worse.

## (b) Fit with KV2 and D4

- **The chain.** KV1's index keys blocks by the chain hash of token ids. With the ids reused, the chain runs through the reply
  instead of breaking inside it, so a conversation's next turn finds its own blocks. On CUDA the follow-up then diverges only
  after the reply, inside the ring's 448-position rollback, which removes this cause of the cold continuations whether or not the
  rollback is ever raised.
- **Across conversations.** §5.6 property 3 accepts decode-class rows only as a continuation of the same chain. So KV2 will not
  copy a reply's decode-written rows to another conversation, even when the store makes the ids equal: the requester's cold class
  for those rows is a prefill class. What KV2 can share across conversations stays the prefill-class preamble. The id reuse does
  not widen sharing and needs no change to KV2's gates.
- **D4.** A server-side store must be keyed by (namespace, hash of the echoed rendering). Otherwise one namespace's generations
  decide how another's identical text is tokenized. That is cross-namespace influence on outputs, and a timing probe: whether reuse
  happened reveals that someone generated that exact text. The blob form needs no namespace, because the ids come in the
  requester's own request.
- **The schedule-independence gate.** KV2's gate (N conversations sharing a preamble, every reply equal to that conversation's
  reply alone) holds under the blob form. Under a shared store whose eviction depends on what else ran it can fail: the schedule
  decides an eviction, and the eviction decides the tokenization. A store that is per namespace and never evicts before the
  conversation's KV does is the minimum for the store form.

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

## (e) Logging generated ids in future night runs

**What exists:**
- `-log-requests`: one line, counts and times;
- `cached_tokens` and `prefill_reused_tokens` in usage: the reused length, no position inside the reply;
- `kv0_proxy.py`, which saves whole request bodies outside the repository;
- the matrix's `wire_proxy.py`, which counts events and keeps no bodies;
- no route returns ids.

**Proposed, none built:**
1. **A reuse field on the `-log-requests` line:** the reused length, the first divergent position, and that position's offset
   into the previous turn's reply. It is cheap, carries no content, and would have turned the night's bound into a measurement.
   It goes in the log line, not an env read (the env-read policy).
2. **A canonicity bit per reply on the same line** (the check above, at commit): whether the reply's ids are the tokenizer's own.
   With 1, it attributes every partial and cold continuation without ids.
3. **The ids themselves, for a measurement run only:** a request-scoped vendor field on the response (as `prefill_reused_tokens`
   is), off unless asked, captured by a body-saving proxy (`kv0_proxy.py`'s shape) with the bodies kept outside the repository.
   The night harness would switch its proxy to save bodies for the run that measures this, not for every night.


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
