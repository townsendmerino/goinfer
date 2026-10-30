# gpt-oss's forced tool call that came back empty at ~12k tokens (2026-10-30, nobara-mm01)

Assigned by the coordinator under Francis's standing rule: BIG1's gpt-oss-20b serve check
(`docs/measurements/big01-moe-serve-2026-10-30/README.md`, commit `5df3b7801`) had a cold forced call
(`tool_choice: "required"`, `max_tokens` 128, a ~12k-token prompt) return no call, no text and no usage after 312 s.
A forced call should never come back empty.

## The cause, read from the code before any card use

The tree is `origin/main` at `98858151a`.

1. On gpt-oss a must-call `tool_choice` is post-checked, not constrained. `forcedCallEnforcement` returns "post-check" for
   a template with no JSON call form, which harmony is. The reply runs free, and `runToolTurn` returns `errNoForcedCall`
   when it holds no parsed call.
2. serve's reasoning budget (`internal/serveapp/budget.go:server.applyBudget`, the room rule) gives the analysis channel at
   most three quarters of `max_tokens`, so 96 of 128.
3. On harmony the budget's forced close (`chat/budget.go:harmonyBudgetFor`) is `<|end|><|start|>assistant<|channel|>final<|message|>`.
   That is the final channel, a prose answer. A harmony call goes on the commentary channel
   (`<|channel|>commentary to=functions.NAME …`), so once the budget fires on a must-call turn the call cannot be written.
4. At ~12k tokens the reasoning ran past 96 tokens. The first token came at 311.9 s and the stream ended at 315.2 s, about
   3.3 s at gpt-oss's 32.6 tok/s. The budget closed the reasoning into final, and the reply held no call.
5. The chat route then sent `sseErr("generation failed: " + errNoForcedCall)` and `[DONE]`, with no usage chunk. BIG1's
   probe at `5df3b7801` skipped error frames, hence `tool_calls: []`, `text: ''`, `usage: null`.
6. The 330-token forced call passed because its reasoning ended under 96 tokens on its own.

A second defect sits beside it. `errNoForcedCall` does not look at the finish reason, so a turn that `max_tokens` cut off
before the call is reported as "the model answered without one", and gets no usage.

## Reproduction, registered before the run

`run.sh` loads gpt-oss-20b on nobara's RTX 2070 SUPER with BIG1's flags (`-backend cuda -require-backend
-moe-cache-experts -ctx 32768`), using goinfer-serve built from `origin/main`. `repro.py` then sends BIG1's own prompts:
it imports `probe.py` from a checkout at `5df3b7801`, so the 12k prompt is the failing one byte for byte. Every request is
`tool_choice: "required"` at temperature 0. Every SSE frame is kept: reasoning, content, calls, finish reasons, usage and
error frames.

| case | prompt | `max_tokens` | budget fires at | prediction |
|---|---|---:|---:|---|
| `short128` | BIG1's 330-token forced call | 128 | 96 | P0: a call, as in BIG1 (the control) |
| `short24` | the same | 24 | 18 | P1: an error frame naming `errNoForcedCall`, no usage, reasoning cut at the budget, no call |
| `long128` | BIG1's ~12k harness prompt | 128 | 96 | P2: the same as P1, reproducing BIG1 |
| `long1024` | the same, reusing `long128`'s prefix | 1024 | 768 | P3: a call to a listed tool |

- **Decision rule:** the cause is confirmed if P1, P2 and P3 all hold. If `long1024` also comes back empty, the budget is not
  the cause: stop and read again before any fix.
- **Void:** a card with any CUDA context besides the desktop compositor's at the start, a 1-minute load above 8.0, or
  MemAvailable under the load's 13 GB plus 2 GB.
- **Stopping rule:** one run.
- **Cost basis:** BIG1's figures. A 14 s load, seconds for the short cases, about 312 s for the cold 12k prefill, and seconds
  for `long1024` (the prefix is reused). About 6 minutes under the exclusive timing lock, in a gap tack's runs leave.

## The fix, if the cause holds (a red-first PR)

1. **On a must-call harmony turn, the budget's forced close opens the call** instead of final:
   `<|end|><|start|>assistant<|channel|>commentary to=functions.`, with the recipient text tokenized by the model's own
   tokenizer. The model then writes the tool name and its arguments, and the post-check reads them as before.
2. **A post-checked turn with no call reports why.** On `finish == "length"` it says that `max_tokens` ran out before the
   call, and the streamed error carries usage.

Tier-0 tests go first, red on today's tree. The budget's forced sequence on a must-call harmony turn must end in the
commentary call header, and a post-check turn cut off by length must name the cause. Then this run's `long128` is repeated
on the fixed build.

## Run A, 2026-10-30 14:19–14:25 PDT: P2 did not hold, so the cause is not confirmed

Run on main's goinfer-serve (`98858151a`, sha256 `0b09c2aab2cc`), the RTX 2070 SUPER, driver 595.112.07. The start was clean:
load 2.22, MemAvailable 63.5 GB, only the desktop compositor on the card. The 22 s load reported fit and swap guard as
BIG1's did. Raw files are in `raw/a-main/`.

| case | result | against the prediction |
|---|---|---|
| `short128` | `run_command {"command":"ls -R"}`, finish `tool_calls`, 30 completion tokens, 25 chars of reasoning | P0 holds |
| `short24` | an error frame naming `errNoForcedCall`, no usage, no call | the outcome P1 named, but not by the budget: the reasoning (25 chars) closed before the trigger at token 12, and `max_tokens` 24 then cut the call off. That is the second defect, a length cut-off reported as "answered without one" |
| `long128` | a call (`run_command`), finish `tool_calls`, 10,932 prompt tokens, 45 completion tokens, 36 chars of reasoning | **P2 fails**: BIG1's failure did not reproduce |
| `long1024` | a call, 53 completion tokens, the prefix reused | P3 holds, but it decides nothing once P2 failed |

**By the decision rule, the budget is not confirmed as BIG1's cause.** The reasoning at 10.9k tokens was 36 characters, far
under the 96-token budget, so the budget never fired. Both defects read from the code are still there: the budget's close into
final on a must-call turn, and the length cut-off reported as "answered without one". But run A does not show either one
causing BIG1's empty call.

**What differed from BIG1.**
- **The server's history (the leading hypothesis):** BIG1's harness turn was its fifth request. It came after `cold_12k`
  (12,524 tokens over `cuda/`) and `append_12k` (12,557, of which 12,525 were reused). So the one GPU KV slot held about
  12.5k rows of another conversation when the 10.9k-token harness prompt prefilled cold over it. Run A's slot had held only
  a few hundred rows. A read past the new prompt's end into stale rows would be invisible in run A and fatal in BIG1.
  gpt-oss's sliding-window layers make that read plausible.
- **The binary:** BIG1 ran branch build `00584c35` (`~/tack-runs/bin/goinfer-serve-cuda-00584c35`).

## Runs B and C, registered 2026-10-30 14:30 PDT before they run

Both use BIG1's order without its short and 4k steps: `cold12k` (BIG1's `cold_12k` prompt, no tools, `max_tokens` 48), then
`append12k`, then `long128`. The prompts come from the `5df3b7801` checkout, as in run A.

- **Run B:** main's build. **Run C:** BIG1's build `00584c35`, run only if run B's `long128` calls.
- **Decision rule:**
  - B's `long128` empty: the preceding longer conversation is what breaks it (the stale-KV hypothesis). The next step is a
    tier-0 or tier-1 test of the cold prefill over a slot that held a longer conversation.
  - B calls and C is empty: something fixed it between `00584c35` and main. Bisect only if Francis wants the commit named.
  - Both call: BIG1's failure is unreproduced. It is recorded as such, and only the two defects read from the code are fixed.
- **Void:** as in run A. **Stopping rule:** one run each.
- **Cost basis:** about 11 minutes each, mostly the two cold prefills of about 5 minutes each (BIG1: 311 s and 312 s).

## Run B, 2026-10-30 14:27–14:38 PDT: BIG1 reproduced

Run on main's build (`0b09c2aab2cc`, the same binary as run A), clean at its start (load 0.52, only the compositor on the
card). Raw files are in `raw/b-main/`.
- `cold12k`: 12,524 prompt tokens, finish `length` at 48 tokens, prose.
- `append12k`: 12,557 tokens, 12,525 of them reused, finish `length`.
- **`long128`: the same empty failure as BIG1.** An `errNoForcedCall` frame, no usage, no call.
  - Its expert-cache line is `834422 hits, 227338 misses`, equal to BIG1's failing turn to the token, so the reproduction is
    deterministic.
  - Its reasoning is the same 36 characters as run A's, where the call parsed (`We need to read cuda/pack_stream.go.`). The
    reasoning budget did not fire in either run, so **the budget is ruled out as BIG1's cause**. The two runs diverge after
    the reasoning, where the call is written.

Under the decision rule, run C is not run. Run B's `long128` differs from run A's in two ways at once:
1. **Stale rows:** the slot held 12,557 rows of another conversation, and `long128` prefilled 10,932 cold over it. In run A the
   slot held a few hundred.
2. **A boundary snapshot:** run B took a prefix snapshot at the 10,880-row boundary during the cold prefill (`kv snapshot: took
   10880 rows`). Run A took none, because its prefill reused 311 tokens and so was not a cold first prefill.

## Run D, registered 2026-10-30 14:40 PDT before it runs

On a fresh server, `long128` is the first request. It is a cold first prefill, so the snapshot is taken at the boundary, but
the slot holds no stale rows. Same build as runs A and B.
- **D empty:** the prefill split at the snapshot boundary is the cause, not stale rows. The next step is the code path that
  takes the snapshot mid-prefill on a sliding-window model.
- **D calls:** stale rows from a longer earlier conversation are the cause.
- **Void, stopping rule:** as in run A, with one run. **Cost basis:** one cold 10.9k prefill, about 5 minutes.

## Run D, 2026-10-30 14:39–14:45 PDT: the same failure on a fresh server, so stale rows are ruled out

Clean start: load 1.07, only the compositor on the card. Raw files are in `raw/d-main/`. `long128` was the first request: an
`errNoForcedCall` frame, no usage, no call, the same 36 characters of reasoning. A 10,880-row snapshot was taken at the
boundary, and the expert-cache line read 834,310 hits and 227,450 misses.

Runs A and D now differ in two ways:
1. **The mark:** D's cold prefill took the boundary snapshot at 10,880; A's took none.
2. **The start:** A's prefill began at 311 (its tools prefix was reused), D's at 0, so the batched prefill's passes fall on a
   different grid and the logits can differ slightly.

Either way the model writes the same reasoning, then something the post-check does not parse as a call. The post-check streams
no prose, so what it wrote is not in the record yet.

## Run E, registered 2026-10-30 14:46 PDT before it runs

On a fresh server, `long128auto`: `long128`'s prompt under `tool_choice: "auto"`. Harmony renders the same prompt for either
choice, and the same build writes the same tokens. With no post-check, the reply comes back whole, either as a call the
parser read or as the text it could not.
- **E's reply holds text that is a call in a form the parser misses** (another header order, a recipient it does not read, a
  missing `<|constrain|>`): the defect is in the harmony parse. The fix makes it read that form, with a tier-0 test on the
  exact text, red first.
- **E's reply is prose, or a final-channel answer:** the model chose not to call on this pass grid. That is a model choice
  under a post-check that cannot force one. The fix is the forced call itself: open the commentary header when the reply
  leaves the analysis channel on a must-call turn, not only when the budget is due.
- **E calls:** `auto` and `required` decode differently after all, and this needs reading again.
- **Void, stopping rule:** as before, with one run. **Cost basis:** one cold 10.9k prefill, about 5 minutes.

## Run E, 2026-10-30 14:46–14:51 PDT: the decode ran out of `max_tokens` inside the call

Clean start: load 1.38, only the compositor on the card. Raw files are in `raw/e-main/`.
- **`long128auto`:** finish `length` at exactly 128 completion tokens (10,932 prompt tokens, none reused). The reasoning is the same
  36 characters (about 9 tokens), with no content and no call.
- **Reading:** about 119 tokens went into a tool call that had not reached `<|call|>` when `max_tokens` stopped it. The parser
  shows an unfinished call in neither content nor calls, so the reply looks empty under `auto` as well.
- **BIG1's chain, then:**
  1. On a cold prefill from position 0 (BIG1's harness turn, runs B and D), the model writes a call longer than 128 tokens.
     Prefilled from 311 (run A), the same turn wrote a 45-token call.
  2. `max_tokens` cuts the call off.
  3. The post-check finds no call and reports `errNoForcedCall`, "the model answered without one". That is the second defect
     read from the code.
  4. The streaming route sends it as an error frame with no usage and no finish reason. Nothing tells the client that the
     budget ran out.
- The reasoning budget's close into final is a real defect, but not this one: the budget never fired.

## Run F, registered 2026-10-30 14:52 PDT before it runs

On a fresh server, `long1024`: the cold turn with room for the call to finish. Its arguments show whether the call was only long
or degenerate (a repeating run).
- **F calls, with arguments under about 1,000 tokens:** the call was long. The fix is the second defect alone: name the length
  cut-off, with usage and finish `length`.
- **F's call repeats, or runs to 1,024:** the cold-from-0 pass grid produces a degenerate decode, a fidelity question about
  that path. It is reported, and the length fix still goes in.
- **Void, stopping rule:** as before, with one run. **Cost basis:** about 5 minutes.

## Run F, 2026-10-30 14:52–14:58 PDT: the cold turn's call is degenerate

Clean start: load 2.26, only the compositor on the card. Raw files are in `raw/f-main/`.
- **`long1024`, cold:** a call, finish `tool_calls`, 340 completion tokens (10,932 prompt, none reused). The call is
  `run_command {"cmd":["bash","-lc","-lc","-lc", …, "-c","-1","-lc","-1", … "-1", …]}`, a repeating run of `"-lc"` and
  then `"-1"`.
- **The schema:** neither argument fits it. `run_command` takes `command`, a string. Run A's passing 45-token call,
  `{"cmd":["bash","-lc","-lc","cat","cuda/pack_stream.go"]}`, has the same wrong name and the start of the repeat. It was
  short enough to finish only because its prefill began at 311.

**So, against run F's rule, the second branch:**
1. **The decode is degenerate** at greedy on this 10.9k-token prompt. Prefilled cold from 0, it ran past 128 tokens; from 311
   it happened to close sooner. Whether that is goinfer's numerics or gpt-oss's own greedy behavior on this prompt, these
   runs cannot tell. That takes a reference decode of the same prompt (a peer, or the CPU path), which these runs do not
   include. It is reported, not fixed here.
2. **The empty reply is goinfer's defect, whatever the decode.** A forced call cut off by `max_tokens` came back as
   "answered without one", with no usage and no finish reason. The fix names the cut-off and sends usage on the chat stream.
3. **Also fixed, found on the way:** a must-call harmony turn whose reasoning outruns the budget was forced into the final
   channel, where no call can be written. That path cannot fire on BIG1's prompt, since its reasoning is about 9 tokens.
   tier-0 tests cover it.
4. **Not fixed, for a decision:** the post-check accepts a call whose arguments do not match the tool's schema (`cmd`, an
   array, against `command`, a string). Whether a post-checked must-call turn should also validate the arguments is a
   contract question.

## The fix, and its checks (2026-10-30, CPU only)

1. **A post-checked must-call turn cut off by `max_tokens` says so.** `runToolTurn` wraps `errNoForcedCall`: "max_tokens (N) ran
   out after M tokens, before the call was written; raise max_tokens". `errors.Is` still matches, so every route keeps its
   status and shape. The chat route's streamed error now sends the usage chunk first, when the client asked for it.
2. **On a must-call harmony turn, the reasoning budget opens a call instead of the final answer.**
   `chat.Template.NewCallBudgetFor` forces `<|end|><|start|>assistant<|channel|>commentary to=functions.`, encoded by the model's
   own tokenizer. `server.applyBudget` uses it when `genRequest.callPostCheck` is set. Every other template, and every turn
   that may answer in prose, keeps the old close.

**Red first, on the branch tip before the fix** (the new tests copied onto it; `raw/tests/red.log`, `red-budget.log`):
- `TestForcedCall_postCheckCutOffNamesTheCause` fails on all three routes, streamed and not, each reporting "answered without
  one". The chat stream also has no usage chunk.
- `TestApplyBudget_harmonyMustCallOpensTheCall`: the must-call budget forced `[6 7 8 3 9 5]`, which is
  `<|end|><|start|>assistant<|channel|>final<|message|>`.

**Green on the fix** (`raw/tests/green2.log`, `green-chat.log`, `touched2.log`, `lint.log`):
- Both new tests pass, with every forced-call, budget, harmony and stream-usage serve test.
- `TestNewCallBudgetFor` and `TestHarmonyBudget_realVocabulary` pass. The latter ran against gpt-oss's own tokenizer from
  `~/models`, not skipped.
- `scripts/touched.py` passes 15 of 15. Its first run caught `TestStreamSurfaces_allSendUsageBeforeDone`, the usage-before-[DONE]
  scan: the usage chunk now goes before the error frame, so the exit stays an error exit.
- `go run ./cmd/lint` passes 7 of 7.
- The test fixture `harmonyTokenizer` gained the recipient text's bytes, one merge and its four markers as added tokens, so it
  encodes. Without them `NewCallBudgetFor` returns nil, and the test would have checked nothing.

**Not done here:**
- **On the device:** `long128` repeated on the fixed build. The card is held for tack until about 20:15; tier 0 already pins
  both changes.
- **For decisions:**
  - whether the cold-from-0 decode's repetition is goinfer's numerics or gpt-oss's own (it needs a reference decode);
  - whether a post-checked call should be validated against its tool's schema. Decided 2026-10-30 (Francis, via the
    coordinator): no contract change now; recorded as a known gap in `docs/QUEUE.md`, "Known gap: a post-checked must-call
    tool turn does not check the call's arguments".
