# P28: Gemma 4 E2B time to first token on Metal

Queue item P28 (`docs/queue-performance.md`). Tier 1, by day, on the MacBook (M1 Pro, 16 GB, macOS 26.6.2). A pinned
`serve-metal` built from main `784e7c82`, each model at `-ctx 16384 --quant int4`, holding the timing lock.

## Step 1, 2026-10-17 07:34-07:40 PDT: prefill rate and path, against Qwen2.5-Coder-1.5B

`run-p28.py`, as registered in P28. One untimed 16-token warm-up per model, then one cold request per length on a fresh
lead token. Records: `step1/`.

| model | prompt tokens | wall | prefill rate |
|---|---|---|---|
| Gemma 4 E2B (`gemma-4-E2B_q4_0-it.gguf`) | 4,096 | 28.4 s | 144 tok/s |
| Gemma 4 E2B | 8,192 | 238.5 s | 34 tok/s |
| Qwen2.5-Coder-1.5B q4_k_m | 4,097 | 4.6 s | 900 tok/s |
| Qwen2.5-Coder-1.5B q4_k_m | 8,193 | 10.3 s | 796 tok/s |

- **Paths, from each server's banner:** both decode `metal-resident (int4)`, and neither banner has a decline line.
  - E2B: `prefill path: batched f16-MMA from 16 prompt tokens (a Gemma 4 E-model, S9 step 2); layer-major on decode's
    kernels below it (bit-identical to sequential)`.
  - The 1.5B: `batched f16-MMA from 32 prompt tokens; below it, decode rows on the batched step kernels`.
- **Reading:**
  - E2B is not prefilling one token at a time at 4,096 tokens. It runs the batched pass, at 6.2x the 1.5B's time.
  - From 4,096 to 8,192 tokens E2B's time grows 8.4x, against the 1.5B's 2.3x. That is more than a quadratic term
    alone would give (4x).
- **Coldness:** the 8,192 requests reused 4 (E2B) and 24 (1.5B) leading tokens, from a shared lead word and the chat
  template. Under 0.3% of either prompt.
- **Unexplained:** tack's E0 run put E2B at 99.8 s for a 4,084-token prompt (41 tok/s). This run took 28.4 s for 4,096
  tokens. Its 8,192 reading (238.5 s) matches tack's (238.2 s). Tack's prompts (Messages route, its own content) and build
  (goinfer 0e598d3f) differ from this run's. To be checked when tack shares its inputs: the prompt's exact token count
  and the build.

## Read from the code before step 2 (main 784e7c82)

- **The kernel each layer takes:** the batched pass takes the fused (flash) attention kernel only for a head dim that is a
  multiple of 8 and at most 128 (`metal/prefill.go:resident.prefillAttnKernels`, `metal/prefill.go:resident.prefillExactAttn`).
  Gemma 4's layer head dims are above that, so E2B's attention layers run the exact kernel, `attention_prefill`.
- **What the exact kernel does:** it keeps one score per key in threadgroup memory, `sc[4096]`, so it holds at most 4,096
  keys (`metal/prefill.go:prefillExactAttnMaxKeys`). It does not write a score matrix to device memory.
- **What happens past 4,096 keys:** `metal/backend.go:metalResident.batchedPrefill` declines the prompt ("the exact prefill
  attention kernel holds 4096"). The E-model branch of `metal/backend.go:metalResident.PrefillLast` drops that error and
  runs the layer-major pass on decode's kernels (bit-identical to sequential). Nothing logs it: the banner's prefill path
  line does not mention the bound.
- **Prediction:** the knee is at 4,097 keys, a fall from the batched pass to the layer-major one, rather than memory
  pressure. Step 2 tests that; it does not assume it.

## Step 2 (registered 2026-10-17, before it runs): where the time goes

`run-p28b.py`, by day when Francis says now, holding the timing lock.

- **Arms:** E2B cold at 4,096, 6,144 and 8,192 prompt tokens. Each runs on a fresh server, so each length has its own
  lifetime peak footprint and nothing is reused, with one untimed warm-up first. About 8 minutes in all.
- **Read per length:**
  - prefill tok/s;
  - `vm_stat`'s pageouts, swap-ins and swap-outs, compressions and decompressions, and the compressor's pages, before and
    after the request;
  - `vm.swapusage` before and after;
  - the server's lifetime peak physical footprint (`proc_pid_rusage`, `ri_lifetime_max_phys_footprint`).
- **Kernel per layer type:** from the code, above. The build exposes no per-phase prefill timing in serve; its only
  prefill trace is a test hook.
- **The hypothesis under test:** memory pressure from a non-flash attention path on the wide-head layers.
  - It holds if the slow lengths show swap or compressor growth, or a footprint that climbs with length beyond KV's own
    share.
  - It falls if 6,144 and 8,192 run at the layer-major pass's rate with flat counters. The cause is then the exact
    kernel's 4,096-key bound and the fallback it forces.
  - Either way, the result is reported before any fix.

## Step 2, read 2026-10-17 17:27-17:34 PDT: the 4,096-key bound, not memory

Run as registered (`run-p28b.py`, the pinned `serve-metal` at `784e7c82`, `-ctx 16384`, `--quant int4`), on Francis's
"now", holding the timing lock. Records: `step2/` (`results.json`, each length's serve log, `run.log`).

| prompt tokens | prefill | rate | peak footprint | swap-outs | compressions |
|---|---:|---:|---:|---:|---:|
| 4,096 | 28.5 s | 143.5 tok/s | 1,626 MB | 0 | 59,462 |
| 6,144 | 164.1 s | 37.5 tok/s | 1,358 MB | 0 | 0 |
| 8,192 | 238.6 s | 34.3 tok/s | 1,564 MB | 0 | 0 |

- **The hypothesis falls:** the slow lengths show flat counters. Neither has a swap-out or a compression, and swap used stays
  at 458 MB throughout. The peak footprint does not climb with length: it is lower at 6,144 and 8,192 than at 4,096.
  4,096 has the only compressor activity, and it runs at the batched pass's rate.
- **The knee is past 4,096 keys,** at the layer-major pass's rate: 37.5 and 34.3 tok/s, about 4× under the batched
  pass's 143.5. That is what the code predicts.
  - E2B's head dims are above the fused kernel's 128 (`metal/prefill.go:resident.prefillAttnKernels`), so the batched
    pass runs the exact `attention_prefill` kernel.
  - That kernel holds at most 4,096 scores (`metal/prefill.go:prefillExactAttnMaxKeys`), so `PrefillLast` drops past it to
    the layer-major pass.
- **Nothing logged the change in this build.** All three servers print the same banner line (`prefill path: batched
  f16-MMA from 16 prompt tokens ...`), and the long requests' `-log-requests` lines carry no `prefill_fallback` field. The
  pinned build `784e7c82` predates `292211eb`, which logs the decline (main from 2026-10-17 08:14).

### Fixes, scoped (not built; owner to decide)

1. **Log the decline (small).** `metalResident.PrefillLast`'s E-model branch keeps the error it drops now, reports it
   once on stderr with the bound and the head dim, and sets the request's `prefill_fallback`, as the other prefill
   declines do. No numerics change. A tier 0 test: a pass over 4,096 keys on a wide-head fixture reports the reason.
2. **Remove the bound in the exact kernel (the smaller kernel change).** Tile `attention_prefill`'s scores over key
   blocks of at most 4,096. One way is two passes: the row's max over all tiles, then exp, sum and PV. It keeps the exact
   kernel's summation order, so the output stays bit-identical to the layer-major pass it replaces, past 4,096. The
   decode kernels already tile their scores this way (`attnScoreTileBound`). Expected: 6,144 and 8,192 run near the
   batched pass's rate instead of 4× under it. It does not speed up prompts of 4,096 tokens or fewer.
3. **A fused (flash) kernel for head dims 256 and 512 (larger).** Raise `attention_prefill_fused`'s `ATTN_MAXHD` above 128.
   That means a new threadgroup and register budget per head dim, and tuning, and the online-softmax rescale is not
   bit-identical (the P19 category). It is the only option that also speeds up E2B below 4,097 keys, where step 1 put the
   batched pass at 144 tok/s against Qwen2.5-1.5B's 900. It removes the bound too.

Smaller first: (1), then (2). (3) is the one that would close most of the gap to the 1.5B, and it is its own task.


## Fix 2's reading (registered 2026-10-17, before it runs): does the tiled kernel remove the knee

Fixes 1 and 2 are on main at `27e917d7`. Past 4,096 keys the exact kernel now runs `attention_prefill_tiled`, so E2B's
batched pass no longer declines there.

`run-p28c.py`: step 2's run unchanged except the server, a `serve-metal` pinned at `27e917d7` (`-ctx 16384 --quant int4`).
It is a Mac night job (`night.py`, holding the timing lock), about 10 minutes. Step 2's own run took 7.3 minutes, and the
fix can only shorten it. Fit-checked by day at that build: `fit: 16384-token cap needs 0.3 GB KV (weights 3.5 GB, budget
4.4 GB, 4.1 GB left for a request's own prefill)`, with the banner naming the tiled kernel past 4,096 keys.

- **Arms:** E2B cold at 4,096, 6,144 and 8,192 prompt tokens, each on a fresh server, with step 2's counters and footprint.
  The script also records each request's `-log-requests` line and whether the server printed the layer-major note.
- **Path (read first; pass or fail):**
  - It passes if the 6,144 and 8,192 request lines carry no `prefill_fallback` and the servers print no layer-major note.
  - If either appears, the batched pass still declines in production. That is reported as the result, and the rates
    are not graded.
- **4,096 (control; the kernel there is unchanged):**
  - Read against step 2's 143.5 tok/s. That comparison is cross-session, so it is a check, not a ratio.
  - Outside about ±5%, the difference is investigated before the long rows are quoted.
- **6,144 and 8,192 (the effect), each against its step 2 rate (37.5 and 34.3 tok/s):**

  | verdict | band | 6,144 | 8,192 |
  |---|---|---|---|
  | the knee is gone | at least 2× | ≥ 75.0 tok/s | ≥ 68.6 tok/s |
  | ambiguous, parked | 1.2–2× | | |
  | no effect | under 1.2× | ≤ 45.0 tok/s | ≤ 41.2 tok/s |

  - The reading also states each rate as a fraction of 4,096's. Attention's quadratic term should leave it under 1.
  - Step 2 and this run are separate sessions. The bands sit far outside cross-session drift (3.6% for goinfer,
    `docs/measurements/noise-registry.md`, TE3), so a cross-session reading can decide this question. It does not make a
    ratio quotable.
- **Footprint:** the tiled kernel allocates no device memory beyond a 4-byte tile argument. The peak footprint is
  reported, not graded.
