# KS-F2 — gpt-oss and `attendQueryP` CPU attention, bit-identical (registered 2026-10-25, before any code)

KS-F2 of the kernel survey (`docs/tasks/task-kernel-survey-2026-10.md` § "KS-F2"). Francis's decision of 2026-10-25 (6:33–6:38 am, relayed by
the coordinator): the bit-identical step goes ahead with its gates registered first. KS-O4 (changes output) and Step 2 (memory) are a
separate PR with their own fidelity gate.

## The change (no number moves)

Two CPU attention functions keep their math and change their schedule:
- `decoder/forward_gptoss.go:attendSinksToken` (gpt-oss, the `attnSinksF64` order);
- `decoder/attention.go:attendQueryP` (the `attnQuery` order: Qwen3.5/3.6 attention layers, Granite-4.0-H, Nemotron-H, LFM2, Llama 4).

1. **Heads fan out** over the attention worker pool (`runSplitAligned`), one head per task range. Each worker has its own score buffer (a
   `decodeScratch` slot), and each head writes its own disjoint slice of ctx. A head's arithmetic is the same on any worker.
2. **Eight keys are scored together** as eight independent float64 chains. Each chain sums its key's d = 0..hd−1 in order, in the same
   expression form (`acc += float64(q[d]) * float64(k[d])`), so each arch contracts it as it contracts the one-key loop. The scale stays
   float64, and the max is taken in key order.
3. **AV through `addScaled`** in both functions. Its expression (`dst[i] += scale * src[i]`) is the one gpt-oss's inline loop uses, so each
   arch's own contraction is kept: fused on arm64, not on amd64.
4. **Scores come from `decodeScratch`**: gpt-oss's per-call `make([]float32, nKeys)` goes.

The exp-and-sum loop and the AV loop keep key order.

## Gates (registered before the code)

| gate | tier | what | bar |
|---|---|---|---|
| G0 | 0, the Mac (arm64), and nobara's CPU (amd64) through the B-run | `decoder/ksf2_identity_test.go`: the new functions against the old ones kept verbatim in the test file. Random q/k/v over shapes that cover hd 64/80/128, GQA groups 1/2/8, key counts 1–300 (not multiples of 8 included), windows, a wrapped ring (`attendQueryP`), softcap on and off, and sinks. The cache's KV rows are compared after the appends, and every ctx word is compared with `math.Float32bits`. Run at GOMAXPROCS 1 and 8 | every word equal, every cell; a skip is a failure |
| G1a | 0, the Mac | `go run ./cmd/gate identity origin/main <branch> -backend cpu -assets tiny -steps 64`: full logits, prefill plus 64 greedy steps, byte-compared between builds, for every tiny family. gpt-oss, `qwen3_5_moe`, `granitemoehybrid` and `lfm2` (each step reads the KV rows held from the steps before) must be among the IDENTICAL cells | every cell IDENTICAL; none WITHIN TOLERANCE, none NOT RUN among the four |
| G1b | 0, nobara CPU (amd64) | the same, on nobara, through its `night.py` queue | the same |
| G2 | timed, nobara CPU, the timing lock, by day | `decoder/ksf2_speed_test.go:TestKSF2_cpuDecodeSpeed` built as a test binary at `origin/main` and at the branch (the same file, copied into the old worktree), run A B B A over 10 pairs after 2 warm runs each. Cells: gpt-oss-20b (`gpt-oss-20b-MXFP4.gguf`, int4, CPU) at depth 128 and 2048, and Qwen3.5-9B (`Qwen3.5-9B-Q4_K_M.gguf`, int4, CPU) at depth 2048. Each run prefills to the depth, then times 32 greedy decode tokens | **SHIP** iff G0 and G1 pass, memory is as stated below, and in every cell at least 9 of 10 pairs are faster (the small-gains rule). **FAIL** if any cell has at most 1 of 10 pairs faster. **PARK** otherwise. The median per-cell ratio is reported beside it |

- **Memory, stated and not gated:**
  - **Removed:** gpt-oss's per-call score allocation, nKeys × 4 bytes per layer per token.
  - **Added:** a score buffer per worker slot, held in `decodeScratch`: at most `maxAttnWorkers` (6) × nKeys × 4 bytes per cache, grown with
    the context (0.79 MB at 32k).
- **The KS-F2 survey estimate,** for comparison and not a bar: 1.12× at depth 128 on gpt-oss, about 1.5× at 2k.

**Amended before any G2 data (2026-10-25), by Francis's standing merge rule of 8:46–8:48 am (relayed by the coordinator):**
- **The rule:** a lossless change merges when it is consistently faster, median ≥ 1.05× with at least 9 of 10 pairs faster, and its PR
  states the memory cost (device and host bytes, and any effect on the longest context that fits).
- **Why it needs more here:** KS-F2 is not opt-in. It replaces the default CPU attention for every family that uses the two functions, so
  it is a default change, which also needs the gain judged against its memory trade-off, and Francis's call.
- **How G2 is graded:** against both bars.
  - The registered small-gains reading (SHIP, PARK or FAIL above) is kept as recorded.
  - The merge bar's reading is "met" iff every cell has median ≥ 1.05× and at least 9 of 10 pairs faster (`grade_g2.py`).
- **Memory, for the PR:** host bytes only (the CPU path). It adds at most 6 × the key count × 4 bytes of score buffers per cache (0.79 MB at
  32k) and removes gpt-oss's per-call score allocation. No effect on the longest context that fits: the fit plan does not price decode
  scratch of this size.

## Pre-registration (`docs/measurements/noise-registry.md` §8)

- **Tier:** day.
- **Testing tier:**
  - G0 and G1 are tier 0 (bit-identity on tiny fixtures, seconds to minutes).
  - G2 is tier 1, one model per function on one box. Its wall clock is stated below. A kernel speed question starts in-process
    (TE5(b)), and this is a whole-token A/B at two builds, interleaved, which resolves direction with the small-gains rule.
- **Interleaving:** A B B A over 10 pairs, per cell, in one session on one box.
- **Cell validity:** each run records the load average and `MemAvailable` at its start. A run started above 8.0 load, or below the fit
  plan's need plus 2 GB, is void and re-run in order.
- **Reused readings:** none.
- **Fit check:** a load-only start of each model through the test binary before the timed runs, as SP0's fit check did.
  - SP0's CUDA load of gpt-oss-20b peaked at 23.4 GB of host memory on nobara (62 GB). The CPU load is measured by this load-only start.
  - On the Mac, gpt-oss-20b's CPU path is off-limits by the Mac's memory rule, so G2 runs on nobara only.
- **Never-swap:** the test binary runs directly, and its `VmHWM` and peak `VmSwap` are sampled from `/proc/<pid>/status`.
- **Stopping rule:** fixed. 10 pairs per cell, no early stop.
- **Ambiguous band:** PARK, 2 to 8 of 10 pairs faster in some cell. The owner decides.
- **Speed bar:** the small-gains rule. The owner's default (ship ≥ 1.02×) is reported beside it.
- **Cost (G2, estimated):** each run is a process (load, prefill, 32 tokens).
  - gpt-oss-20b depth 128: about 10 s load, 2 s prefill, 5 s decode, so about 17 s a run.
  - gpt-oss-20b depth 2048: prefill about 20–30 s, decode about 9 s, so about 45 s.
  - Qwen3.5-9B depth 2048: about 35 s.
  - 24 runs per cell (10 pairs and 4 warm-ups), so about 7 + 18 + 14 min.
  - **About 40 min**, plus two builds. The start and expected finish are stated when it is queued.
- **First loads:** the first CPU int4 load of each model writes its CPU sidecar `.giw` beside the `.gguf` in nobara's `~/models`
  (about 11 GB for gpt-oss-20b and 5 GB for the 9B; 87 GB free). The fit step absorbs it, before any timed run.

## Results so far

| gate | where | result |
|---|---|---|
| G0 | the Mac, arm64 and amd64 (Rosetta), GOMAXPROCS 1 and 8 | **PASS.** `TestKSF2_sinksBitIdentical`: 256 cells, 48 with the heads fanned out. `TestKSF2_queryPBitIdentical`: 768 cells, 132 fanned out. Every ctx word and KV row equal on both arches. A planted defect (two of the eight key rows swapped in `dotsF64x8`) turned both tests red |
| G1a | the Mac (Darwin/arm64), CPU | **PASS** (2026-10-25 06:55, `origin/main` `8135a0c3` against `f8a1eda2`, 64 steps): 33 families IDENTICAL, 0 within tolerance, 0 different, 10 not run (fixtures absent or not tiny). gpt-oss, `granitemoehybrid`, `lfm2`, `qwen3_5_moe`, `nemotron_h` and `llama4_text` are all IDENTICAL |
| G1b | nobara CPU (amd64) | queued |
| G2 | nobara CPU | queued |

**What G1a needed first.** The identity gate had never run gpt-oss, on any revision.
- **The cause:** its tiny GGUF has no embedded tokenizer, so `decoder.Load`'s default sidecar refused it.
- **The fix:** the asset now carries `DirectLoad` through the dumper's cell spec (`cmd/gate/identity_assets.go:identityAsset`). It is
  the only asset that sets it.
- **The Granite fixture:** `granitemoehybrid`'s `testdata/granite-tiny` is gitignored. It was cloned into the worktree from the main
  checkout.
