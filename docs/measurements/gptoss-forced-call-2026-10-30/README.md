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
