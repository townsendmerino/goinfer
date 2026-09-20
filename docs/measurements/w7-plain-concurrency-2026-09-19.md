# R12 item (iv), simplified — W7: goinfer's single worker vs llama-server's slots under concurrency

`docs/tasks/red-october.md` R12's measurement brief (iv), W7: "four interleaved conversations
from the W4 transcript on both boxes, goinfer's single worker vs llama-server's slots — the first
concurrency number on the page, expected to be a loss and recorded at full value."

**This run is a simplified variant, not the exact W7 as specified** — see "Why simplified" below.
The core finding holds regardless: **goinfer collapses under concurrency (60.1 → 36.1 → 36.4
tok/s at 1/2/4 clients, a real loss that plateaus rather than keeps falling); llama-server scales
up (84.8 → 95.9 → 149.7 tok/s over the same range, 1.77× from 1 to 4 clients).** By 4 clients
llama-server is **4.11× goinfer's aggregate throughput** on identical hardware and the identical
checkpoint. This sizes exactly what the brief says it would: "the batched multi-request decode
item that `task-work-queue-2026-09.md` J8 names."

## Why simplified

The brief's own W4 transcript fixture (`scripts/w4_transcript_base.json`) forces a named
`tool_choice` every turn, which needs a model whose chat template AND training genuinely support
tool-calling. Investigated directly, not assumed:

- The memory-safe local model (`qwen2.5-coder-1.5b-instruct`, added to `bench_peer_transcript.py`
  earlier this session specifically because the 7B model caused two near-total memory exhaustions
  under concurrent load) does **not** reliably satisfy llama-server's tool-call parser — verified
  with the model's own default template (wraps the JSON in markdown fences, no `tool_calls` field)
  and again with the *real* Qwen tool-calling jinja template, extracted directly from the sibling
  `qwen2.5-7b-instruct` checkpoint's own GGUF metadata via a minimal hand-written GGUF reader (no
  internet fetch) and supplied via `--chat-template-file`: the model then emits the *correct JSON
  content* but still omits the `<tool_call>` wrapper tags the parser requires. A genuine
  model-capability gap (the coder variant was not tuned for this exact format), not a config
  problem — confirmed by testing the 7B instruct model's own default template, which *does* work
  correctly (proper `tool_calls` field, empty `content`).
- The 7B model that does satisfy the parser carries the demonstrated memory risk noted above.

Given the user's explicit choice (asked directly, given this exact tradeoff) to simplify rather
than risk the 7B model or abandon W7, this run replaces the tool-calling transcript with six plain
multi-turn text questions (`scripts/bench_w7_plain.py`, new script) — no tools, no forced
`tool_choice` — on the safe 1.5B model. This answers the core W7 question (concurrent decode
throughput, goinfer's single worker vs llama-server's per-client slots) without either blocker, at
the cost of not being byte-for-byte the W4 transcript's agentic-tool-use shape. A future pass with
a genuinely tool-call-capable, memory-safe model (if one becomes available) would close this gap
properly.

## Method

- Model: `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` on both engines (same weights, per this
  session's own established convention).
- goinfer: Metal backend, `bench-cur/serve-metal` rebuilt fresh from HEAD (`9b5d2f58`) before the
  run.
- llama-server: `0.3.0 (build 10621, commit c1d0e7a00)`, `-np N -cb` (continuous batching enabled,
  N server slots matching the client count).
- 6-turn plain-text conversation per client (a real multi-turn shape — each turn genuinely
  different, building on the model's own prior reply, not N repeats of one message), max_tokens=128
  per turn, greedy (temperature=0).
- Same isolation discipline as `bench_peer_transcript.py`: a **fresh server per (engine,
  concurrency level)** so no level inherits a warmer cache than `clients=1` saw, and each client's
  first turn is nonce-prefixed so N clients replaying the same fixture don't share cache credit.
- **Not session-interleaved between engines** (all goinfer levels ran, then all llamacpp levels) —
  a deviation from this repo's own "same-session interleaved" peer-comparison discipline, noted
  rather than silently accepted. Session-to-session drift on this box is characterized elsewhere at
  ~3.5%; the effect measured here (4.11× at n=4) is roughly two orders of magnitude larger, so this
  is not expected to change the qualitative finding, but the exact ratios should not be quoted to
  more precision than that caveat allows.
- Two real bugs found and fixed in the harness itself while building it (both now fixed in
  `scripts/bench_w7_plain.py`, not just worked around for this one run):
  1. **llama-server accepts TCP connections well before the model finishes loading.** A
     socket-connect readiness check (the same `wait_port` pattern `bench_peer_transcript.py`
     already uses for goinfer) is not readiness for llama-server — a request in that window gets
     HTTP 503 `{"error":{"message":"Loading model",...}}` (reproduced directly, isolated from the
     concurrency logic). Fixed: poll `/health` for the real `{"status":"ok"}` it returns once
     loaded.
  2. **The original script only wrote results at the very end.** The 503 above crashed the first
     attempt after goinfer's full 3-level sweep had already completed, losing all of it. Fixed:
     save after every `(engine, level)` cell completes, and resume from a partial file on restart
     (skip cells already recorded) — the same discipline `bench_peer.py`/`bench_peer_transcript.py`
     already use for exactly this reason.

## Data

| clients | goinfer tok/s | llamacpp tok/s | llamacpp / goinfer |
|---|---|---|---|
| 1 | 60.08 | 84.82 | 1.41× |
| 2 | 36.14 | 95.85 | 2.65× |
| 4 | 36.39 | 149.72 | **4.11×** |

goinfer: **60.08 → 36.14 (0.60×) → 36.39 (flat)** — a real loss at 2 clients that then *plateaus*
rather than keeps collapsing (36.14 ≈ 36.39, within noise). llama-server: **84.82 → 95.85 (1.13×)
→ 149.72 (1.56× more)** — genuinely scales with concurrency, 1.76× aggregate from 1 to 4 clients.

**Per-client latency shape explains the mechanism, not just the aggregate number.** goinfer's
per-turn latencies at n=4 range 2.5–17.8 s across clients and turns, growing with both client count
*and* turn number (later turns carry more accumulated history) — consistent with a single global
worker serving one request at a time through a FIFO admission queue: N clients divide one
worker's time N ways, and prompt growth compounds on top. llama-server's per-turn latencies at n=4
are **nearly identical across all 4 clients at every turn** (e.g. turn 2: 3.35–3.44 s for every
client) — consistent with continuous batching genuinely co-processing multiple clients' tokens in
one forward pass, so every client gets comparable service time regardless of how many others are
active. This is a mechanistic difference, not just a throughput number: goinfer serializes,
llama-server batches.

## What is and isn't established

**Established:** on this hardware, this checkpoint, this workload shape, goinfer's concurrent
decode throughput is a real, substantial loss against llama-server's continuous batching — the
gap widens with client count (1.41× → 4.11×), and goinfer's own aggregate plateaus rather than
recovers. The per-client latency pattern is consistent with (not proof of) FIFO single-worker
serialization on goinfer's side and real request batching on llama-server's side. **Not
established:** the exact ratio the brief's own W4 tool-calling transcript would show (a different,
heavier request shape — larger prompts from tool schemas/results, forced-grammar generation on
goinfer's side) — this run is a lower bound on how much of a problem this is, not the specific
number the brief asked for. Not session-interleaved between engines (see Method) — the qualitative
finding is robust to that, the precise ratios carry the same ~3.5% same-box caveat as any other
non-interleaved comparison on this repo's own record.

## Next step

R12's own text: this sizes "the batched multi-request decode item that `task-work-queue-2026-09.md`
J8 names" — a real architectural gap (goinfer has no continuous-batching / multi-request decode
path at all), not a tuning question. Closing it is a genuinely different, larger scope than this
measurement brief. A cleaner re-run with the exact W4 tool-calling transcript, on a model that
satisfies both engines, remains open if one becomes available.
