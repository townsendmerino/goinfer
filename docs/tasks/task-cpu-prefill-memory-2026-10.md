# Task: bound the CPU prefill's memory, and return it after a long prompt (G40, 2026-10)

> **Status: DIAGNOSED AND REGISTERED 2026-10-14, before any code.**
>
> - Queue entry: `docs/queue-correctness.md` G40, on branch `kv9-step0-2026-10` until it merges.
> - The measurement that found it: `docs/measurements/kv9-2026-10/README.md`, Finding 3, on the same branch.
> - The code was read at `main` `afd83410` and not run.
> - The gates below are registered. Code starts after this commit.

## 1. What was measured

Measured by `serve -backend cpu` on the M1 Pro 16 GB, with int4 weights from the CPU sidecar and the default f32 KV.

| model | prompt (tokens) | server footprint after the prefill | of which the session's KV | swap growth |
|---|---|---|---|---|
| Qwen2.5-Coder-0.5B | 5,778 | 1,030 MiB | 135 MiB | none |
| Qwen2.5-Coder-0.5B | 20,193 | 3,129 MiB | 473 MiB | 1.27 GB |
| Qwen2.5-Coder-1.5B | 5,778 | 2,050 MiB | 316 MiB | none |

**The footprint did not fall in 60 s of idle.** Later sessions' KV copies landed inside it without moving it. The
20K prefill on the 0.5B put a desktop-loaded 16 GB Mac 1.27 GB into swap. That breaks the never-swap rule
(`docs/tasks/task-never-swap-2026-09.md`).

**A 1.5B at 20K was not run:** by §2's arithmetic it would hold about 3.6 GiB live before garbage.

## 2. Diagnosis: where the high-water comes from

### The path

1. A CPU request reaches `decoder/model.go:Model.generateInto`.
2. That calls `decoder/forwardn.go:Model.prefillLogits` with the whole uncached suffix of the prompt.
3. `prefillLogits` calls `decoder/forwardn.go:Model.forwardLayersN` once, over every row.
4. `forwardLayersN` calls `decoder/forwardn.go:Model.embedN`, then `decoder/forwardn.go:Model.runLayersFromEmbedN`.

Nothing on this path chunks. The only CPU chunking in the tree is MC3's, and it is for GPU residents (`Options.ResidentPrefillChunk`).

### Sized by the whole prompt, live for the whole sweep

`runLayersFromEmbedN` allocates its working set once, sized by K (the suffix length), and holds it across every layer.

| buffer | shape | 0.5B at K = 20,193 | 1.5B at K = 20,193 |
|---|---|---|---|
| `h` (from `embedN`), `norm`, `att`, `mlpOut` | 4 × K × hidden, f32 | 276 MiB | 473 MiB |
| `q`, `ctx` | 2 × K × max query width, f32 | 138 MiB | 237 MiB |
| `k`, `v` | 2 × K × KV width, f32 | 20 MiB | 39 MiB |
| `gate`, `up` | 2 × K × intermediate, f32 | 749 MiB | 1,380 MiB |
| **activations in all** | 15,360 / 27,648 floats a row | **1,183 MiB** | **2,130 MiB** |
| `hq` (`decoder/weightmat.go:w4a8Act`) | K × hidden, int8, plus scales | 17 MiB | 30 MiB |
| the down projection's quantized input | K × intermediate, int8 | 94 MiB | 173 MiB |

The last row needs explaining. `decoder/weightmat.go:matmul` takes a `linalg.Workspace` from
`decoder/weightmat.go:matmulWSPool`. The workspace's int8 activation buffer grows to the largest call it serves and
stays at that size. It stays in the `sync.Pool` until two collections pass without a `Get`.

### Already bounded, not the cause

- **Attention.** `decoder/scratch.go:prefillAttnScratchBudget` caps the head-worker pool at 256 MiB. Its scores
  buffer is tiled by rows (`decoder/scratch.go:attnRowTileK`), and the worker count falls as the prompt grows
  (`decoder/scratch.go:prefillAttnWorkersK`).
- **The LM head.** `prefillLogits` runs it on the last row only.

### The KV regrows as it is written

`decoder/kvcache.go:KVCache.Append` grows each attention layer's `keys` and `vals` with `append`, one row at a time.
serve's sessions are made with no capacity hint, so the slices start empty:

- `internal/serveapp/sessions.go:sessionLRU.newSession` passes the LRU's hint;
- serve sets that hint to 0 where it builds the LRU (the "capHint 0: KV grows on demand" line in `internal/serveapp/main.go`).

Above a few hundred elements Go grows a slice by about 1.25× each time, so writing N rows allocates about 5 × N rows
in all, and 4 × N of that is garbage. For a 20K prefill that is about 1.9 GiB of garbage on the 0.5B and 4.4 GiB on
the 1.5B, spread over the sweep.

### Go keeps it

At the default `GOGC=100` the heap may grow to twice the live heap of the previous collection before the next one, so
the regrowth garbage accumulates on top of the live set. After the request, the runtime's background scavenger returns
freed memory to the OS only slowly, and only down to a retention target near the heap goal. Two observations confirm
this:

- the footprint held for 60 s after the request;
- `GOGC=10` cut the 6K one-session footprint from 1,039 to 684 MiB.

`debug.FreeOSMemory` forces a collection and returns every free span at once. The GPU residency build already calls
it (`decoder/residency.go:releaseHostMemory`). The CPU path never does.

The pooled workspaces survive one more collection in the pool's victim cache, and the second collection frees them.

### The accounting

At the 20K prefill's peak, the 0.5B holds about 2.0 GiB live:

- 1,183 MiB of activations;
- 473 MiB of KV;
- up to 256 MiB of attention;
- about 110 MiB of quantized inputs.

The measured 3,129 MiB is that plus about 1 GiB of KV-regrowth garbage held under the GC goal. The 6K cell matches
the same way: about 0.75 GiB live against 1,030 MiB measured.

This accounting is arithmetic. Gate G40-2's peak sampling is what will show it on the running process.

## 3. The fix

### (a) Chunk the prefill

`Model.prefillLogits` runs the batched pass over the suffix in chunks of **C = 2,048 rows**. It is a constant, not a
knob: `testdata/env_reads.txt` only shrinks, and C is not a per-model choice.

- Each chunk calls `forwardLayersN` on its rows, appending to the same cache.
- Only the last chunk's last row reaches the LM head.
- The embedding is taken per chunk.
- The context is still checked per layer.

**The fast-attention decision is taken once, for the whole suffix.** Today `runLayersFromEmbedN` turns fast attention
off when `K < fastAttnMinPrompt`, keyed on the suffix length (its comment says why). Chunked, a short last chunk would
fall below the floor and switch to the exact path, and its rows would then differ from today's.

The floor's test therefore moves to `prefillLogits`, on the suffix's length, and every chunk is handed the result.
Every other caller of `runLayersFromEmbedN`, including verify, keeps the floor exactly where it is. The floor's
meaning does not change: it still keys on the suffix being prefilled.

**Scope.** Text prefill through `prefillLogits` and its callers: generation, and the speculative draft's and target's
prefills.

- A multimodal prefill (`decoder/forwardn.go:Model.prefillLogitsVLSpans`) keeps one pass: an image's bidirectional
  block must sit inside one pass. Chunking it at span boundaries is a later step.
- Verify (`forwardN`, small K) does not change.

**Memory after it:** the working set is C rows: 131 MiB on the 0.5B and 236 MiB on the 1.5B, against 1.2 and 2.1 GiB
at 20K today. The quantized down-projection input in the pooled workspace is C rows too.

**Amendment, 2026-10-15, after G40-1's first read: a sliding-window cache under fast attention is not chunked.**

- **What G40-1 found.** On `gemma2-tiny` (sliding window 4), the chunked prefill differed from the single pass in 446
  of the last row's logits at a suffix of 529, with C = 256. That held cold and warm, at suffixes 529 and 1,368: only
  where the suffix takes fast attention.
- **Where it differs.** The first difference was layer 1's keys, at the second chunk's rows. So layer 0, the
  sliding-window layer, gave a different attention output. The fast local-layer attention depends on where the pass's
  assembled key window starts, and a chunk boundary moves it.
- **What still holds.** The exact path chunked identically on the same fixture. The dense (`llama-tiny`,
  `llama-attnfa-tiny`) and MoE (`mixtral-tiny`) fixtures chunked identically in every case, including a short last
  chunk inside a fast suffix.
- **The change.** `prefillLogits` keeps the single pass for a cache with sliding-window layers whenever the suffix
  takes fast attention. G40 therefore does not bound that case's working set: Gemma 2, 3 and 4, and every other family
  with sliding-window layers, prefilled fast on the CPU. Making the fast local attention chunk-invariant is a separate
  item.
- **What does not change:** no bar, and no other part of the design. G40-1 now also asserts the number of passes each
  chunked arm ran: several where it chunks, one for a windowed cache under fast attention.

### (b) Reserve the KV once

Before the sweep, each attention layer's `keys` and `vals` are grown once, to the capacity the suffix needs. Rows are
then appended as today, but nothing is reallocated. The storage layout and the stored values do not change. An int8
cache reserves its int8 arrays the same way.

### (c) Return memory after a long prompt

After a prefill of at least **4,096 rows**, `prefillLogits` returns memory to the OS once, before decode starts, as
`releaseHostMemory` does after a GPU build. That is one forced collection and scavenge.

- Its cost is read in G40-3; it is expected in the tens of milliseconds beside a prefill of tens of seconds.
- A threshold keeps it off short prompts and decode, where a forced collection per request would cost something
  for nothing.

### Not in scope

- The per-row MoE FFN's allocations (`decoder/mlp.go:moeMLP` with a nil scratch). They are per row and freed per row;
  chunking bounds what accumulates.
- CPU f16 KV, which is `docs/tasks/task-cpu-kv-f16-2026-10.md`.

### Interplay with KV2 (`kv2-2026-10`)

KV2's CPU class map records which prefill class wrote each row; a pass of 512 rows or more is the fast class. Under
(a) the class is the suffix's, so the rule KV2 records must key on the suffix length, not on each chunk's. Whichever
of G40 and KV2 merges second carries that one-line change and its test.

## 4. Gates (registered 2026-10-14, before any code)

Each gate names its tier, instrument, stopping rule and cost (CLAUDE.md, "Gate tiers").

### G40-1. Parity, bit-identical (day, tiny fixtures and the 0.5B; seconds to minutes)

**What is compared.** The chunked prefill against today's single pass, from the same cache state, in one test
process. All of these must be bit-identical:

- the last row's logits;
- every K and V element of every layer;
- for a sliding-window family, the ring contents.

**Cells.**

- Fixtures:
  - dense: `llama-tiny`, `llama-attnfa-tiny`;
  - sliding window: `gemma2-tiny`;
  - MoE (per-row FFN): `mixtral-tiny`.
- The real Qwen2.5-Coder-0.5B, under the day rule of no checkpoint above 1.5B.

**Suffix lengths** (C = 2,048; the test lowers C through an unexported test variable so the fixtures cross chunk
boundaries cheaply, and the 0.5B runs at the real C):

- C − 1, C and C + 1;
- 2C + 17, whose last chunk is under 512 rows inside a fast suffix;
- 3C + 600.

Each length runs:

- cold (start position 0) and warm (a prior turn's cache, start position > 0);
- under fast attention (the default) and under `ExactPrefill`.

**Also:** the forward goldens and the parity manifest pass unchanged (`scripts/refresh_parity_hashes.sh`, whose
goldens are the independent proof).

**Plants, each of which must go red:**

- a chunk that applies the floor to its own length (the last chunk exact under a fast suffix);
- an off-by-one in a chunk's start position;
- the KV reservation sized one row short, if the reserve is written to assume exact capacity.

**Stopping rule:** every cell runs.

### G40-2. Memory: peak, swap and return (night, Mac; about 20 minutes)

**Instrument.** `docs/measurements/kv9-2026-10/kv9_mem.py` (KV9's harness) with one addition: during each request it
samples the server's footprint (phys_footprint, `footprint -p`) and RSS every 250 ms and keeps the maximum.

**Cells.** One cold prompt of 20,193 tokens (the KV9 preamble at `e42db592`) through `serve -backend cpu` at its
defaults:

- on Qwen2.5-Coder-0.5B and Qwen2.5-Coder-1.5B;
- with the fixed binary and with `main` before the fix, one after the other in one night. The before arm of the
  1.5B is skipped if its own §2 arithmetic says it would swap; the record then says so.

**Precondition:** the cell refuses to start with less than 1 GiB of swap free or 4 GiB of memory available, and a
refusal voids the cell rather than failing it.

**PASS needs all of these, on both models:**

| criterion | 0.5B | 1.5B |
|---|---|---|
| peak footprint during the request ≤ the footprint after load + the session's KV by geometry + 768 MiB | ≤ 1,307 MiB | ≤ 1,939 MiB |
| peak RSS ≤ the weight file's size + the KV by geometry + 768 MiB (RSS counts the mapped weights, which footprint does not) | | |
| swap used at the end ≤ swap used at the start + 32 MiB (32 MiB is the noise KV9's quiet cells showed, −32 to +0 MiB) | | |
| the footprint 10 s after the request ≤ the load footprint + KV + 256 MiB: memory returned | | |

**FAIL:** any criterion missed on the fixed binary.

**Ambiguous, parked:** a peak within 64 MiB of its bound. That is re-read with the sampling interval at 100 ms; it
is never re-rolled.

**Prediction (written now):**

- 0.5B peak footprint 900 to 1,150 MiB (0.7);
- 1.5B peak 1,500 to 1,850 MiB (0.6);
- no swap growth on either (0.8);
- the before arm of the 0.5B reproduces KV9's 3.1 GiB within 10% (0.7).

### G40-3. Speed, non-regression (in process by day, then served at night)

The chunks stream each weight once per chunk instead of once per prompt. At 2,048 rows a chunk's matmuls are
compute-bound, so the cost is expected to be small, but it is not assumed.

**Instrument.** A whole-prefill in-process A/B, chunked against a single pass in one process, interleaved
(TE5(b)), on the 0.5B and 1.5B at suffixes of 6,000 and 20,000 tokens. The 1.5B at 20K runs at night only, because
the single-pass arm is the thing that swaps.

**The quantity:** prefill wall time, chunked ÷ single pass, as a median over paired runs. The release call's
forced-collection time is reported beside it.

**Bars.**

- **Ships:** ≤ 1.03 at every cell.
- **Parked:** the worst cell in (1.03, 1.05]. The one registered alternative is C = 4,096, read once; nothing else
  is tried.
- **Killed at this C:** any cell above 1.05.

A run that does not resolve a direction by day goes to the night as a served two-arm gate (`bench_peer.py`,
`BENCH_ENGINES=goinfer,goinfer_old`) with the same bars.

**Prediction:** every cell within 1.00 ± 0.02 (0.7); the forced collection under 100 ms at 20K (0.8).

### Done

G40-1 and G40-2 pass and G40-3 ships. G40-2's before-and-after record goes beside KV9's, and G40 in
`docs/queue-correctness.md` closes with a pointer to it.

## 5. Size, and what it touches

About a day with its gates. It edits the decoder core, so the parity manifest goes stale and is refreshed with the
goldens:

- `decoder/forwardn.go`: `prefillLogits`, and the floor moved out of `runLayersFromEmbedN` for this caller;
- `decoder/kvcache.go`: the reserve;
- the memory return (wherever `releaseHostMemory` is shared from).

No new production environment read.
