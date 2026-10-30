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
