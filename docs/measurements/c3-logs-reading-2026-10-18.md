# Tack's E0 run C3 on goinfer f20f0d93: two findings read from its logs and the code (2026-10-18)

Logs only; nothing was run for this note. Sources: tack's C3 output and the servers' load logs on nobara (under `~/tack-runs/e0/`), `internal/kvmgr`, `decoder/cachestate.go`, `cuda/prefill.go`, and the `kv3-tiled-2026-10` branch.
Nothing here is built. Where a number is not in the logs, the note says so and names the cheapest run that would give it.

## 1. Qwen3.6-35B-A3B (`--moe-cache-experts`): the serving swap watch halted the whole arm

**Which guard fired: the server's own `VmSwap`, not the system backstop.** The load log reads `swap guard: TRIPPED — server swap grew 0.90 GB over baseline (0.90 GB used now); refusing new requests until it recovers`. The run that built the
sidecar earlier the same day tripped the same way, at +0.83 GB. The per-process threshold is 512 MB. The system backstop (+4096 MB over a baseline of 9.82 GB of system swap) did not fire. Every request after the trip returned `503 halted: swap guard tripped`, so all of
the arm's cells (TTFT at 4096, 8192 and 31744 tokens, decode, the check) are void.

**When.** The TRIPPED line comes before `loaded "qwen" … in 1m5s` and before `[cuda] C′ cache: 64 slots/layer would need 4.5 GB VRAM … capping to 60`: the swap grew *during the model load*, which on this path runs after the server is up.
The same log says `swap guard (load): not armed for …int4.e4h.cuda.v17.giw — only a direct .gguf build is watched` and `swap guard: baseline 0.00 GB (server swap)`. So the serving watch took its baseline before the load began, and swap the load caused was charged to serving.
A page swapped out stays in swap until it is touched, so `VmSwap` never fell back toward that baseline: "until it recovers" could not happen, and the halt lasted for the life of the server.

**What was priced.** `fit: 32768-token cap needs 1.2 GB KV (weights 18.2 GB, budget 26.6 GB, 25.4 GB left for a request's own prefill)`. The `.giw` guard (`decoder/fitguard.go:decideGIWFit`) prices KV and prefill scratch only. It does not price the C′ expert cache: the model's routed experts are 18.12 GB (`docs/tasks/task-memory-campaign-2026-10.md`, MEM0's table), and on the C′ path they are packed into a page-locked host copy that cannot be reclaimed
(MEM10). (The line `stream-weights: cache ready (18691 MB)` in the earlier sidecar-build log is the `.giw` file being written, not that copy.) And "reading its 19.6 GB once before the load" brings the whole `.giw` into page cache beside it. The 70% budget (26.6 GB) was compared with KV plus scratch (about 1.2 GB), not with
18.1 GB pinned plus 19.6 GB cached. The box already carried 9.8 GB of system swap (7.2 GB of it in zram) before the load started.

**The 33.1 GB peak: the split is not in these logs.** The two terms are known (about 18.1 GB anonymous and pinned; the 19.6 GB mapped file, some part of it resident), but they sum to 37.7 GB, more than 33.1 GB, so some of the file's pages had been evicted by the time the peak
was read, and the peak cannot be split between them from the logs. A sampled load gives the split: `docs/measurements/cuda-build-heap-2026-10-18/build_peak.py` (RssAnon, RssFile and VmSwap every 50 ms) with `-serve-option moe-cache-experts`, about 4 minutes, as a night job.

**Which term let it through:**
1. The load is unwatched on the `.giw` path (B33).
2. The serving watch's baseline predates the load.
3. The host guard does not count the pinned expert copy, or the page cache read beside it.

**Proposed fix, not built, in order of cost:**
1. **Re-baseline the serving watch when the load completes** (arm it from the "loaded" hook). Smallest change; it removes the false permanent halt. The load itself stays unwatched until B33.
2. **B33** (`docs/queue-engineering.md`): arm the load-time watch on the `.giw` path with an abort point in the CUDA build, so a swap-growing C′ stream stops with a named error.
3. **Price the pinned expert copy** in `decideGIWFit` when `--moe-cache-experts` is on (the `.giw`'s expert bytes), so a box that cannot hold 18.1 GB pinned beside its page cache is refused or told to use a smaller cache.
4. **Do not pre-read the whole `.giw`** when the C′ stream will read it anyway, or drop the range from page cache (`posix_fadvise` DONTNEED) after it is copied, so the cache and the pinned copy are not both held.

## 2. Qwen3.5-9B on the same build

Model facts (GGUF metadata): 32 layers, `full_attention_interval` 4 (so 8 full-attention layers and 24 Gated-DeltaNet layers), head dim (`attention.key_length`) **256**, 16 heads, 4 KV heads.

### (a) Prefix reuse: the probe cannot show what tack's loop needs, and the code does serve the append-only case

- **Tack's "reused" cell** (`internal/eval/runner.go` in tack: `measure("reused", …, prefix+tailAt(10+rep))`) sends "the previous prompt with only its last 512 tokens changed". For a recurrent model that is the one request that can never be reused: continuing at the unchanged prefix needs the
  recurrent state as it was 512 tokens before the end, and the state is mutated in place with no history. The cache-state grid says so (`decoder/cachestate.go`, `recurrentCells`: rewind inexact, snapshot refused, share "only a strict extension of the whole sequence continues, from exactly where the state stopped; anything else is cold").
  So cold equal to reused at every size (15.3 vs 15.7 s at 4k, 38.8 vs 38.8 s at 8k, 421.3 vs 421.5 s at 31.7k) is the designed behaviour for that request. Qwen3-8B and Qwen3-4B differ only because their positional KV rewinds exactly.
- **The append-only request is served.** `internal/kvmgr/slots.go:scoreRecurrent` reuses the whole held sequence when the request's tokens start with all of them and are longer. So a follow-up whose ids are (previous prompt ids, the tokens the model generated, new text) skips the
  held part, with no end-of-prompt snapshot (there is none for recurrent state, and the model keeps one slot). This is a code reading; it has not been run on the 9B.
- **Where the real loop can still miss.** The held sequence is the prompt *plus the generated tokens*. The next request is re-rendered by the chat template from the message list. If that history drops earlier reasoning, re-renders a tool call's arguments with different whitespace, or tokenizes the reply
  text into different ids than were sampled, the request is not an extension of the held ids at some token. The rule then returns 0 (cold) with no partial credit, because the state cannot be rewound to that token. Whether tack's loop hits this is an empirical question.
- **To settle it** (scope, not run): a three-turn chat through the chat endpoint at 4k and 8k prompt tokens with tack's real system prompt and tool form, each turn appending the previous reply, comparing time-to-first-token for turns 2 and 3 against a cold request of the same length. About 5 minutes of GPU on f20f0d93 or a newer build.
- **If the real loop does miss** and a fix is wanted, the fix is a recurrent-state checkpoint: a host copy of the DeltaNet state and conv windows at the end of each prompt (and at the end of its reply, if the next request matches it), restored for a request that shares that prefix. The state of 24 layers is tens of MB, so a host copy and a copy back take milliseconds. Pass-boundary alignment
  matters for bit-identity with a cold run (the KV7 restore test's rule). Effort: M to L; it needs the snapshot format work, which today refuses recurrent state. Not scoped further until the three-turn run says it is needed.

### (b) Cold prefill 260 → 75 tok/s between 4k and 31k: the same 48 KiB shared-memory limit as Gemma 4's finding 1

- The 9B's load log says `prefill path: batched (weight-stationary CUDA passes of up to 512 rows); attention: attn_fused (L2, head dim 64/128, M>=16) else attn_batched (exact)`. Head dim 256 is not served by the fused attention (64 or 128 only), so the 8 full-attention layers take
  the exact `attn_batched`, whose dynamic shared memory is `(nKeys+128)*4` bytes. The device limit is `singleBlockAttnShmemLimit`, 48 KiB or 49,152 B, which allows **12,160 attended keys**. Gemma's 50,300 B is 12,447 keys on the same formula. Past that bound `cuda/prefill.go:cudaResident.checkPrefillShmem`
  declines the batched pass (`errPrefillDeclined`) and the prompt is prefilled by the sequential KV-only path.
- **The throughput numbers fit.** 4,093 tokens in 15.3 s is 267 tok/s, and 8,182 in 38.8 s is 211 tok/s (batched, under the bound; the fall is the attention's quadratic growth). 31,744 tokens in 421.3 s is **75.3 tok/s**, close to the 65 tok/s decode rate:
  the whole 31.7k prompt ran sequentially, as an up-front check on the prompt length would have it.
- **Not captured:** the decline line itself (`cuda prefill: layer N attention at … attended keys needs … B of shared memory …`) is not in the logs tack saved, since the load logs end at shutdown. One request of about 13,000 tokens on the 9B, reading the server's stderr, confirms it in about 3 minutes.
- **A fix already exists, unmerged.** Branch `kv3-tiled-2026-10` (17 commits ahead of main; the owner's decision is recorded in `docs/measurements/kv3-tiled-exact-2026-10-13.md` §11) has `cuda/prefill_splitkv.cu`: three row-batched kernels (scores, softmax, V-sum) that run an exact-attention layer past the shared-memory bound,
  bit-identical to `attn_batched`, routed exactly where `splitKVRequired` holds and no fused attention serves the head dim, with the score scratch bounded at 64 MiB. That is the split-KV prefill path for Gemma, and it serves any layer with head dim 256, so it would carry the 9B's 8 attention layers.
  Two things are not known. First, past the bound the branch runs every pass of the prompt on the exact kernels (the tensor-core GEMM missed its fidelity bar there), so the speed past 12k is that path's and has not been measured for a DeltaNet hybrid. Second, the branch predates the Gated-DeltaNet batched
  prefill and the `.giw` changes, so it has to be brought onto main, and a Qwen3.5 tiny fixture must prove the route (bit-identity below the bound with the route forced; above the bound against the per-token path).
- **Estimate (not built):** merge or rebase of the branch onto current main with its conflicts: S to M. A Qwen3.5 tier-0 gate (tiny fixture, route forced, bit-identity): S. A tier-1 timing of the 9B at 13k, 20k and 31k tokens, cold, against the sequential path: about 15 minutes. The gain is bounded by what the exact path costs per row; if it
  is far from the batched rate, the alternative is a fused-attention variant for head dim 256 (the fused kernel stages 64 keys at a time and has no length bound), which is a new kernel: L.
