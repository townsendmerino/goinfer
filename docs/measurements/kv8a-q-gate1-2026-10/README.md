# KV8a-Q gate 1 — where the next question's request diverges, on goinfer's Qwen3.5 rendering (2026-10-24)

Gate 1 of KV8a-Q (`docs/tasks/task-kv-architecture-2026-10.md` § "KV8a-Q"), the stop point: render a conversation with a
question, a tool loop, a final answer and a new question, reasoning sent back, through goinfer's Qwen3.5 template, and record
where the new question's request diverges from the one before. The rule: if it does not diverge right after the previous
question, KV8a-Q is closed.

Desk only. No model weights are loaded; the tokenizer and chat template come from the Qwen3.5-9B `.giw` on the Mac's
`~/models`.

## Verdict

**The template claim holds:** the new question's request leaves the previous request right after the previous question, at its
`<|im_start|>assistant\n`, in all three thinking modes. **But the copy KV8a-Q would add is already on main, on CUDA.** Serve's
`KVScope.Keep` for every request of a question's tool loop is that same point, and B35 snapshots a recurrent model's state at
Keep. The section's premise ("the request falls back to KV7's prefix snapshot at the system boundary, or to 0") does not
describe main. Its save point ("end-of-prefill state") would also miss: end of prefill is 2 or 4 ids past the divergence.

## How it was run

- **Fixture** (`make_bodies.py`): one conversation as the six `/v1/chat/completions` requests a client sends, with two tools.
  With one tool under `auto`, a request that ends with a user message forces that tool (`forcedTool`), and the route then
  renders thinking off.
  - `r1`: Q1. `r2`, `r3`: Q1's tool loop.
  - `r4`: the final answer plus Q2. `r5`: Q2's tool loop.
  - `r6`: the final answer plus Q3.
- **Modes:**
  - `on`: `enable_thinking: true`, every assistant turn's `reasoning_content` sent back.
  - `off`: `enable_thinking: false`, no reasoning.
  - `default`: neither. This is the template's default, and it renders identically to `off`.
- **Instrument** (`internal/serveapp/kv8aq_gate1_test.go:TestKV8aQ_gate1Scopes`): posts each body to the real chat route
  (`server.handleChat`) on a scripted model that keeps recurrent snapshots (`loadedModel.scriptedRecurrent`). It captures the
  prompt ids and the `decoder.KVScope` the route hands the decoder. The system-and-tools boundary comes from
  `loadedModel.prefixBoundaryOf`. The route's ids equal those of the KV0 study renderer (`TestKV0_tokenizeBodies`) for every
  body.
- **Analysis** (`diverge.py`): the first id at which each request differs from the previous one, and the deepest held point
  that could serve it. A recurrent state cannot roll back, so a point serves only at or before the divergence.

```sh
D=$PWD/docs/measurements/kv8a-q-gate1-2026-10
for m in on off default; do
  python3 $D/make_bodies.py $D/bodies-$m $m
  KV8AQ_BODIES=$D/bodies-$m KV8AQ_MODEL=$HOME/models/Qwen3.5-9B-Q4_K_M.int4.metal.v17.giw KV8AQ_OUT=$D/scopes-9b-$m.jsonl \
    python3 scripts/timing_lock.py run --shared --label kv8aq-g1 -- go test -count=1 -run '^TestKV8aQ_gate1Scopes$' -v ./internal/serveapp/
  python3 $D/diverge.py $D/scopes-9b-$m.jsonl > $D/diverge-9b-$m.txt
done
```

## Readings (`diverge-9b-on.txt`; `off` and `default` in their files)

| request | ids (on) | Keep (on) | diverges from the previous at | deepest point that serves it |
|---|---|---|---|---|
| r1 Q1 | 201 | 199 | — | cold |
| r2 loop | 263 | 199 | 201: the whole of r1 is a prefix | live state / KV8a at 201 |
| r3 loop | 328 | 199 | 263: the whole of r2 is a prefix | live state / KV8a at 263 |
| **r4 Q2** | 332 | 330 | **199**, right after Q1's `<|im_start|>assistant\n`; 129 of r3's ids not shared | **r1–r3's Keep, 199** |
| r5 loop | 393 | 330 | 332: the whole of r4 is a prefix | live state / KV8a at 332 |
| **r6 Q3** | 410 | 408 | **330**, right after Q2's `<|im_start|>assistant\n`; 63 of r5's ids not shared | **r4–r5's Keep, 330** |

- **The divergence:** at the next question, r3 goes on `<think>\nI need both temperatures…` and r4 goes on `<tool_call>…`. The
  history render drops the think blocks of every assistant turn since the previous question. With thinking off the
  divergence is the same id (199 and 330). There the dropped text is the empty `<think>\n\n</think>\n\n` block.
- **Keep:** each request's Keep is where `chat.NextTurns` (the next request's shape: a reply, then a user message) parts from
  it. Inside a tool loop that is the loop's question. For a request ending in a user message it is that message's assistant
  header, before the generation tail. So Keep always names the point the next question continues from.
- **End of prefill is past it:** a request ending in Q1 prefills to 201 (on: `<think>\n`) or 203 (off:
  `<think>\n\n</think>\n\n`). That is 2 or 4 ids beyond the 199 that r4 shares. A `question` copy saved at end of prefill, as
  the section says, could never be restored.
- **The boundary** (system and tools) is at 181 in every request. It plays no part for a recurrent model; see below.

## What main already does with this (code reading, not run)

- **The snapshot:** for a recurrent model, `decoder/kvsnap.go:Model.kvBoundaryPlan` plans the prefill's snapshot at
  `KVScope.Keep`, rounded down to the resident's pass grid, and never at `KVScope.Boundary`. `Model.kvPrefillMark` takes it
  during r1's cold prefill.
- **The tool loop leaves it alone:** r2 and r3 continue from the live state past it, so they take no new snapshot, and r1's
  stays in the host store unless the store's budget evicts it.
- **The restore:** r4 restores it and re-prefills from there. That is B35 (`docs/queue-engineering.md` § B35), on main. Its
  end-to-end gate measured plain chat turns 2 and 3 on nobara, not a question after a tool loop.
- **B35's limits, which are what KV8a-Q could still add:**
  - **CUDA only.** `cuda/kvsnap.go:cudaResident.KVSnapshotRecurrent` is true only with no Mamba-2 layer. Metal and the CPU
    keep no recurrent snapshot (`Model.KeepsRecurrentSnapshots` false), so on Metal the new question goes cold.
  - **Not Mamba-2.** Nemotron-H and Granite-4.0-H are refused on CUDA too.
  - **A long prompt.** A prompt of at least `kvSnapMinPrompt` (2048) ids, and a kept point of at least `kvSnapMinShared` (512)
    after rounding. Agent clients' system prompts clear both.
  - **Rounded to the pass grid.** The warm turn re-prefills up to one 512-row pass. B35 records that as its follow-up (a cut
    at Keep).
- **The second question saves only the prefix through Q1, whoever holds it:** system, tools and Q1. Q1's whole tool loop is
  re-prefilled, because its rendering changed. In this fixture r4 re-prefills 133 of 332 ids. A copy at Q2 covers Q1's whole
  loop only from Q3 on (r6: 80 of 410). Gate 5's bar ("the second question's time to first token falls by at least half its
  cold time") therefore depends on how the system prompt's size compares with the loop's. It would need a third question to
  measure the case the copy is for.

## Not established here

- Whether B35's snapshot is still held when the second question arrives in a real session. That depends on the host budget
  and on other conversations.
- Whether B35's restore after a tool loop matches a cold prefill end to end. Its gate ran plain chat turns only.
- Other routes. `/v1/responses` (Codex) and `/v1/messages` (Claude Code) compute Keep with the same
  `loadedModel.recurrentKeep`, but this fixture posts chat completions only.
