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
