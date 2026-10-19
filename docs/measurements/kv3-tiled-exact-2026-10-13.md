# KV3: a tiled exact attention for CUDA's batched prefill, past 12,160 keys (registered 2026-10-13, before any code)

Owner decision 2026-10-13 (`docs/tasks/task-kv-architecture-2026-10.md` §8 Q7): build the tiled exact kernel the census recommended.
Branch `kv3-tiled-2026-10`, from `kv3-cuda-2026-10` (it needs that branch's guard and admission figure). nobara-pc, RTX 2070 SUPER.

## 1. Facts this rests on

1. **The census** (`docs/measurements/kv3-long-prompt-census-2026-10-13.md`): Gemma-3-4B and Gemma-4-E4B decline past 12,160 attended keys on the first
   global layer and fall to the per-token path: 9.17 and 13.78 ms a token, so at least 183 s and 276 s for a 20,000-token prompt, against 0.87 and 2.43 ms a
   token batched at 1.5k. The paged-expert models do not decline (their cost is expert paging).
2. **Why they decline.** The exact batched attention (`attn_batched`, grid `nH × M`, block 128) keeps one row's scores in shared memory,
   `(nWin + 128)·4` bytes, so it cannot pass 48 KiB (12,160 keys); the fused attention serves head dim 64 and 128 only. Gemma-3 is head dim 256; Gemma-4's
   global layers are 512 (local 256).
3. **Decode already has the kernel shape.** `splitKVAttnDecode` (`cuda/resident.go`) is three launches (`splitkv_scores`, `splitkv_softmax`, `splitkv_vsum` in
   `cuda/decode_splitkv.cu`) that hold the scores in a global buffer and are **bit-identical to `attn_batched(M=1)`**, held by `TestSplitKV_bitIdentical`.
   It splits only the independent axes (keys for the scores, output dims for the V fold); the softmax keeps `attn_batched`'s 128-wide partition and
   tree, and each output dim keeps the whole ascending-key fold. Decode takes it exactly where the single block cannot run (`splitKVRequired`).
4. **The exact class is one class.** `attn_batched`, `gemv_w4a8_rn` and decode's split-KV agree bit for bit (KV0's record); a batched prefill that
   runs only exact-class kernels produces the bits of the per-token path.

## 2. The change

- **Three row-batched kernels** in a new `cuda/prefill_splitkv.cu` and its own `.ptx` (the `attn_block.cu` isolation pattern: no shipped PTX changes):
  `splitkv_scores_rows`, `splitkv_softmax_rows`, `splitkv_vsum_rows`. Each is the decode kernel with a row index in the grid's Z axis. Row `m` of a pass at
  `startPos` attends `nKeys = startPos + m + 1`, window start as `attn_batched` derives it from that `nKeys`, and its own `nWin`; the instruction sequence
  per (row, head) is the decode kernel's, so identity is by construction; only the addressing (a row's `q`, score slice and `ctx` row) is new.
  The sink is passed as decode passes it. Head dims 64 to 512 (the q staging is `hd·4` bytes of shared; the float4 K read needs `hd` a multiple of 4,
  which the families here satisfy and the host checks).
- **Rows are chunked** so the score scratch stays bounded: `R = max(1, budget / (nH · maxNWin · 4))` rows a launch group, with `budget = 64 MiB`. The
  scratch is one device buffer allocated with the pass's other scratch.
- **Routing.** A layer whose exact attention would exceed the shared-memory bound, and that the fused attention does not serve, takes the row-batched
  kernels. `checkPrefillShmem` stops declining for it and keeps declining for what these kernels do not serve (image blocks, `attn_img_batched`; the verify
  lane). A layer under the bound is untouched: `attn_batched` as today. So **no existing prefill's bits change**.
- **Admission.** `PrefillScratchBytes` includes the score scratch for the pass, and `TestKV3_prefillScratchBytes_matchesThePass` is extended to hold it.
- **Regeneration.** `./build_ptx.sh prefill_splitkv` with the pinned `nvidia-cuda-nvrtc-cu12==12.6.85` environment, so the new `.ptx` is reproducible and no other
  `.ptx` is rebuilt.

## 3. Gates (registered)

| | gate | tier | pass |
|---|---|---|---|
| G1 | the row kernels equal `attn_batched` row by row: random q, K and V, head dim 64, 128, 256 and 512, MHA and GQA, sliding window, sink, `startPos` 0 and not 0, f32 and f16 KV, spans below the bound (a test seam forces the new route there) | day | every output float bit-identical |
| G2 | no change below the bound: the existing digest runs at 300, 3,900 and 12,000 tokens against `kv3-cuda-2026-10`; the `cuda` identity suites the branch already runs | day | bit-identical |
| G3 | batched equals per-token past the bound, on a tiny head-dim-256 fixture at 12,400 to 16,384 tokens (the fixture's context raised as `kv0LoadCtx` does), logits and every row's KV | day | bit-identical |
| G4 | the same on the real models: Gemma-3-4B at 12,400 tokens and Gemma-4-E4B at 12,400 tokens (global layers head dim 512), batched against per-token, last-row logits and a KV digest | night (the per-token side is at least 2 and 3 minutes a model at 12,400 tokens) | bit-identical |
| G5 | speed: time to first token on Gemma-3-4B and Gemma-4-E4B at 12,400 and 20,000 tokens, batched against the per-token path, same session, interleaved | night; one exploratory cell by day, never quoted | **PASS** at 3x or more at 12,400 tokens on both; **AMBIGUOUS** 1.5x to 3x (parked: tune the V-sum's occupancy, a rows axis on the dims split, then re-read once); below 1.5x the kernel is not kept |
| G6 | no regression where it does not route: TTFT at K = 512 and 3,900 on Gemma-3-4B against `kv3-cuda-2026-10`, and the Qwen3-4B pair the branch already runs | night | 0.97 or better |
| G7 | admission: the pass's own scratch (a test seam reports it) is within the registered 5% of `PrefillScratchBytes` at 16 and at 512 rows, tiny and real | day | within the margin, over-pricing only |

**Expectation, stated so it can be wrong.** Each (row, head) reads its whole key range from the cache with no reuse across rows beyond the GQA group, so the cost is
cache bandwidth, not the fused kernel's 64-key tiles. A 20,000-token prompt reads about 10 TB of K and as much of V across the six global layers of Gemma-3-4B (8 heads, head dim 256, f32) before L2 reuse:
tens of seconds at this card's bandwidth, not the 17 s of the batched rate at 1.5k. G5 measures it; this note does not predict a number.

## 4. Not in this work

Gated-DeltaNet hybrids (no decline was logged for Qwen3.6 at 12,242 keys), MLA, the image-block prefill, the all-logits, embedding and residual tails, and a fused
attention for head dim 256 or 512 (a larger numerics project: it is not bit-identical to decode). An unresolved G5 is not a reason to re-register the bar.

## 5. Estimate

Build and G1 to G3 and G7 by day: one session. G4 to G6 on the night queue under the timing lock, about 40 minutes estimated (two real models, per-token sides at
2 to 5 minutes each, interleaved with the batched side, two lengths).

## 6. Amendment 1 (2026-10-13, before any code or run)

Three facts read from the tree while preparing the build; none comes from a run of the new kernels.

1. **No tiny fixture has head dim 256.** G3 as registered names one. G3 runs on `llama-attnfa-tiny` (head dim 128, GQA 4:2, two layers) with
   `GOINFER_CUDA_FAST_PREFILL=0`, at a context raised past the bound: with the levers off the pass is exact class and its attention takes the row kernels
   past 12,160 keys, so batched against per-token is still a bit-identity. Head dims 256 and 512 are held at the kernel level by G1 and on the real models by G4.
2. **Gemma-3's batched pass is not exact class as a whole.** Above the 512-token floor its projections run the tensor-core GEMM (the fast lever); only its
   attention is exact, because no fused attention serves head dim 256. So "batched equals per-token, bit for bit" is true for Gemma-4-E4B (an E-model forces the
   exact kernels) and for Gemma-3 only with `GOINFER_CUDA_FAST_PREFILL=0`. **G4 becomes:** Gemma-4-E4B at 12,400 tokens, default settings, bit-identical; Gemma-3-4B at
   12,400 tokens with the levers off, bit-identical; and Gemma-3-4B at default settings, a fidelity reading against the per-token path (last-row logits cosine
   0.99 or better and equal argmax: the fast GEMM's own class past the bound, which the lever's existing gate covers only to 3,900 tokens). The kernel's own
   identity at default settings is G1 plus a new **G2b (day):** on a full model pass below the bound, forcing the row kernels (a test seam) against
   `attn_batched` with every other kernel unchanged, logits and KV bit-identical.
3. **The score scratch depends on the attended span, and `PrefillScratchBytes(rows)` does not see it.** A pass allocates the score scratch only when a layer
   routes to the row kernels, sized `R·nH·maxNWin` floats plus `R·nH` inverses. Admission reads it through a second optional method,
   `PrefillSpanScratchBytes(rows, keys)`, added to the request's price with `keys` the request's prompt length; `PrefillScratchBytes` is unchanged, so its
   existing equality test stands. **G7 becomes:** the pass's allocations equal `PrefillScratchBytes(M) + PrefillSpanScratchBytes(M, keys)` exactly, at spans
   below and past the bound.

Any pass can route, not only a Gemma's: a suffix of 15 rows or fewer past 12,160 keys is exact class on every family (it declined to the per-token path before, a
few tokens). The routing rule is the registered one; this is its consequence.

## 7. Amendment 2 (2026-10-13, after the build, before the night jobs)

1. **The module is built with this box's ambient NVRTC, 12.9.86, not 12.6.85.** §2 named the pin of the audited set (`glue`, `moe`, `gemv_fwd` and their kin). Every
   other production module, `decode_splitkv.ptx` and `prefill_batched.ptx` among them, is built at 12.9.86, and a new module follows its siblings:
   `./build_ptx.sh prefill_splitkv`, no other `.ptx` rebuilt. The kernels have no bare multiply-add (they are in `lintedKernels`), so the compiler version is not a
   numerics choice, and G1 measures the result.
2. **Head dim must be a multiple of 4**, as `attn_batched`'s float4 read of `q` already requires (a head dim of 6 faults in `attn_batched` itself). G1's smallest head dim is 20.
3. **The image-block pass still declines past the bound, by name** (an MLA model does not batch at any length, a static decline). An E-model's layers and a Gated-DeltaNet
   hybrid's attention layers go through the same launch site as any exact layer, so they route too: the E-model is read by the Gemma-4-E4B smoke below, the hybrid is read
   from the code only, and the census's inference that Qwen3.6 did not decline is still unmeasured.

## 8. Day readings (2026-10-13)

Branch `kv3-tiled-2026-10`. Logs: `kv3-tiled-exact-2026-10-13/` (`day-gates.log`, `g3-tiny.log`, `digest-tiled.txt`, `gemma3-forced-day.log`, the two smoke JSONs).

| gate | reading | verdict |
|---|---|---|
| G1 | 360 cases (180 on each KV type: head dim 20, 64, 128, 256, 512; MHA and GQA; a causal past; a window opening partway; with and without a sink; groups of 1, 7 and every row), every float bit-identical to `attn_batched`. A softmax launched at block 64 turns 180 of them red. | **PASS** |
| G2 | digests at 300, 3,900, 4,100 and 12,000 tokens on the 1.5B equal `kv3-cuda-2026-10`'s (`digest-tiled.txt` against that branch's `digest-new.txt`) | **PASS** |
| G2b | tiny, f32 and f16 KV, levers off and tensor-core GEMM alone, 300, 1,000 and 3,900 tokens: forced route against `attn_batched`, 0 of 256 logits and 0 of up to 3,993,600 KV elements differ, 12 of 12 cells | **PASS** |
| G3 | tiny, levers off, 12,400 and 16,384 tokens, f32 and f16 KV: batched against per-token, 0 logits and 0 of up to 16,777,216 KV elements differ, 4 of 4 cells; without the module the same pass declines by name | **PASS** |
| G7 | the pass's allocations equal `PrefillScratchBytes(M) + PrefillSpanScratchBytes(M, keys)` in 16 of 16 cells (levers on and off, start 0 and 12,300, 8 to 512 rows); the span term is present exactly when the pass is exact past the bound | **PASS** |

**Not gates (measurements and smokes, one run each, exploratory, not to be quoted as results):**

- *Real weights, below the bound:* on Gemma-3-4B one 512-row pass whose last row attends 12,032 keys, `attn_batched` against the forced row kernels, three interleaved pairs:
  0 of 262,144 logits and 0 of 35,651,584 KV elements differ in every pass; 1.25 s against 1.40 s a pass (all 34 layers forced, three pairs, the same sign each time).
- *Served smoke, one request each:* Gemma-3-4B, 12,241 tokens: 25.8 s, no decline (the census: unfinished at 150 s). Gemma-4-E4B, 12,215 tokens: 55.9 s, no decline (the census:
  unfinished at 150 s). G5 measures the ratio.

**A harness defect found and fixed in G1 before it was read.** The first runs failed in about half the invocations, always on the reference's side: the row kernels matched a
float64 host computation and `attn_batched` did not. The test uploads `K`, `V` and `q` and launches the reference at once on its own stream; a host-to-device copy from pageable
memory returns once it is staged, so the reference read the tail of `q` before it landed. A context synchronize and a read-back of `q` after the uploads fixed it: 12 of 12 runs
clean. `TestAttnBatched_bitIdentical` has the same shape with buffers small enough that it has not shown; it is not changed here.

## 9. Amendment 3 (2026-10-13, before the night jobs): G4 to G6 as they are queued

- **G6 named the wrong model.** "The Qwen3-4B pair the branch already runs" is the 1.5B: `kv3-cuda-2026-10`'s TTFT job runs Qwen2.5-Coder-1.5B. G6 is K = 512 and K = 3,900 (the
  1.5B's prompt texts) on the 1.5B and on Gemma-3-4B, this branch's serve against `kv3-cuda-2026-10`'s, three alternating pairs, 0.97 or better at both depths on both models
  (`ttft_ab_models.py`). Both binaries carry the aligned tail rule (`kv3-cuda-2026-10-13.md` §2.5), so the pair differs only in this branch's change.
- **G5's design:** per model, three processes in order batched, per-token (the same binary with `GOINFER_BATCHED_PREFILL=0`), batched; in each, one request of about 12,400 tokens and
  one of about 20,000, `max_tokens` 1, a unique prompt each. The ratio at a length is per-token over the median of the two batched readings. The bars are §3's, read at 12,400 tokens
  (`ttft_tiled.py`). A batched arm in which the server logs a decline is a FAIL whatever its time.
- **G4** is `TestKV3Tiled_batchedEqualsPerToken_real` at 12,400 tokens. It was run by day at 1,200 tokens with the route forced, to check the plumbing only: Gemma-3-4B with the
  levers off and Gemma-4-E4B (global layers of head dim 512) each 0 of 262,144 logits and 0 KV elements differing from the per-token path; Gemma-3-4B at default settings cosine 0.9963,
  argmax equal. Those are not G4's readings.
- **The whole `./cuda/` suite on this branch is queued too** (`JOB=suite`, 60 minutes, the owner's call on 2026-10-13), in the pinned worktree with the ignored fixtures linked from
  the primary checkout, as `kv3-cuda-suite` does on the branch below this one. A test red in both is that branch's; one red only here is this branch's.
- **Estimates** (from the census and the day smokes): G4 20 minutes, G5 45, G6 15. Jobs: `run-night.sh` with `JOB=identity`, `speed`, `ttft`.


## 10. Night readings (2026-10-14, graded as registered)

The jobs ran on the night of 2026-10-13 from the pinned worktree at `a0ce6e40` (code identical to the serve binary's `12d9d8fb`: the two commits between them change
docs and `run-night.sh` only); the base binary is `kv3-cuda-2026-10` at `5282df9a`. Driver 595.91.07, RTX 2070 SUPER, the timing lock held by the runner. Logs and
provenance: `kv3-tiled-exact-2026-10-13/night-2026-10-14/` (the suite log as `cuda-suite.log.gz`).

| gate | reading | verdict |
|---|---|---|
| G4, Gemma-4-E4B, default, 12,400 tokens | 0 of 262,144 logits and 0 of 355,532,800 KV elements differ from the per-token path; 35 row groups, 0 tensor-core launches | **PASS** |
| G4, Gemma-3-4B, levers off, 12,400 tokens | 0 of 262,144 logits and 0 of 863,436,800 KV elements differ; 25 row groups, 0 tensor-core launches | **PASS** |
| G4, Gemma-3-4B, default, 12,400 tokens (amendment 1: cosine 0.99 or better and equal argmax) | **cosine 0.986525**, argmax equal; 25 row groups, 5,950 tensor-core launches; every logit differs (worst 6.12) | **FAIL** |
| G5, time to first token at 12,400 tokens, per-token over the median batched | Gemma-3-4B **5.68x** (130.39 s against 22.92 and 23.02 s); Gemma-4-E4B **3.31x** (186.32 s against 56.43 and 56.28 s); no decline logged in any batched arm | **PASS** |
| G6, old over new, median of three pairs | 1.5B: 1.006 at K = 512, 1.000 at 3,900; Gemma-3-4B: 0.994 at 512, 0.999 at 3,900 (lowest single pair 0.991) | **PASS** |

**G4 is FAILED as registered.** The test asserts what amendment 1 registered for the Gemma-3 default arm (cosine 0.99 and equal argmax, last-row logits, against the
per-token path), so the bar stays and the arm is recorded red. The two exact arms are bit-identical, so the row kernels themselves are not in question.

**Where the 0.0135 comes from, and whether it is this branch's.** One reading by day (2026-10-14, the same test at 12,000 tokens with the row kernels forced,
`day-2026-10-14-gemma3-12000.log`; the prompt is the 12,400-token prompt's first 12,000 tokens): levers off, 0 logits and 0 of 835,584,000 KV elements differ; default,
**cosine 0.990073**, argmax equal, 5,712 tensor-core launches. Below the bound the forced row kernels equal `attn_batched` bit for bit with the tensor-core GEMM on (G2b,
and the real-weights 512-row reading in §8), so 0.990073 is what the kernel set `kv3-cuda-2026-10` already ships produces at 12,000 tokens on Gemma-3. The run
overlapped another session's process holding about 50 GB of host memory (swap grew from 8.5 to 27.7 GB), so its wall times are contaminated and not quoted; its
difference counts and cosines are deterministic and stand. Read together:

- the whole difference is the tensor-core GEMM's: both exact arms are bit-identical at both lengths, and of the two levers the arms toggle, the fused attention does
  not serve head dim 256, so the GEMM is the only kernel that differs between them;
- the fast lever's distance from the per-token path grows with depth on Gemma-3 (0.9963 at 1,200 tokens by day in §9, 0.9901 at 12,000, 0.9865 at 12,400; one prompt
  each, a prefix of the same text), and it is already within 0.0001 of the bar below the bound, where nothing in this branch changes a bit;
- on `kv3-cuda-2026-10` and on `main` a Gemma-3 prompt past 12,160 keys declines to the per-token path, so the default arm's 0.9865 is not reachable there. This branch
  is the first to run the lever at that depth: the cause is pre-existing levers behaviour, the exposure is this branch's.

The fast lever's own gate (the L2/L3 CUDA gate, `docs/completed/task-prefill-gap.md`) was graded on the Qwen models to K = 3,900 by agreement, hard flips and mean KL,
not by a last-row cosine against the per-token path, and never on Gemma-3. Amendment 1's 0.99 is the KV3 fidelity bar carried over from `kv3-cuda-2026-10-13.md` §2.3.
**An owner decision is open; the bar is not moved here.** The routes the evidence leaves: (a) a Gemma-3 pass that routes to the row kernels runs with the levers off
(exact, G4's levers-off arm; that arm took 31.9 s at 12,400 tokens in the test rig against the default's 23.2 s and the per-token 138.4 s, a test-rig figure, not a
G5 reading); (b) register a fidelity gate for the tensor-core GEMM at depth on Gemma-3 (several prompts, the near-tie rule, a parked band) and decide by it; one prompt
already sits on 0.99 at 12,000, below the bound, so that gate also reads `main`'s Gemma-3 prefill between 3,900 and 12,160 tokens. Either way G5's Gemma-3 ratio was
read at default settings and would need re-reading under (a).

**Vacuity.** G4 asserts at least one row-batched launch group in every arm (25 and 35) and no tensor-core launch in an arm graded exact. G5's batched arms logged no
`batched prefill declined` line (the string the decoder writes on a decline, which the script matches), the serve banner reads `prefill path: batched` and the
per-token arm's reads `sequential — GOINFER_BATCHED_PREFILL=0`; Gemma-3's attention at head dim 256 has no fused kernel, so a batched Gemma-3 pass past 12,160 keys can
only be the row kernels. G6 ran the base binary at `kv3-cuda-2026-10`'s tip, so the pair differs only in this branch.

**The whole `./cuda/` suite: TIMEOUT, not graded.** It reached the job's 2-hour limit while `TestSpecPagerInteraction` was loading its 20 GB checkpoint: 394 PASS,
4 FAIL, 47 SKIP; 37 of the 482 tests the run's tags build reached no verdict (`TestSpecPagerInteraction` started, 36 never did). (Corrected in §12: this line first read "41 of 486", a count from a listing built with `realckpt`, which the run does not use.) The four failures are the four of `kv3-cuda-2026-10`'s suite of the same night, with the same
messages and the same figures, so none is this branch's (§9's rule): `TestResidentDrafter_extendContext` (the second resident's allocation fails out of memory),
`TestGemma4_26B_cache_B` (the 26B's resident declines, 4.00 GB free in both), `TestP20ExpertLocality` (3.98 GB free in both) (all three print the test's own
`ISOLATED-RUN` note: they fail in the one-process heavy tier and pass alone), and `TestGemma4EModelPrefill_batchedBitIdentical` at 18 and 96 rows (`PrefillLast chunk 5:
cuda prefill: empty prompt` in both: the 16-row tail rule emptied a pass at a `GOINFER_PREFILL_CHUNK` under 16, a defect of the branch below). Every test this branch
adds ran and passed (`TestPrefillSplitKV_rowsEqualAttnBatched` and the four tiny `TestKV3Tiled_*`).

**After the suite (2026-10-14, by day).** `kv3-cuda-2026-10` fixed the tail rule (`prefillTailCut` gives nothing when the pass before the last would drop under 16 rows)
and is merged here; `TestGemma4EModelPrefill_batchedBitIdentical` (18 and 96 rows) and `TestKV3_chunkerKeepsPassStartsOnTheGrid` (chunks 1 to 79) pass on this branch
with the fixtures present, at the merge `d05fc2da` (`suite-after-merge-day.log`). The three out-of-memory reds are `needsFreshProcess` tests, which fail in any one-process run of the package
and pass alone; `TestResidentDrafter_extendContext` passed alone on this branch in the day check of the re-queued job's plumbing. What the suite has not read green is
re-queued as `kv3-tiled-suite-rest` (`run-night.sh`, `JOB=suite-rest`), in `cmd/gate gpu`'s partition: the 36 unreached tests other than the fresh-process one (40 as first queued; §12)
(`suite-rest-main.txt`) as one process with the fresh-process tests deferred, then `TestGemma4_26B_cache_B`, `TestP20ExpertLocality`,
`TestResidentDrafter_extendContext` and `TestSpecPagerInteraction` each as its own process, then the drain group (skipped in a one-process run), with a reconcile of
each part. The tests this branch does not touch that the 2026-10-13 run read green are not repeated; `kv3-cuda-2026-10`'s partitioned suite reads the merged fix.

## 11. Owner decision on G4's Gemma-3 default arm (2026-10-14): the exact kernels past the bound

**Decision (the owner, 2026-10-14, option (a) of §10):** a prompt whose attention takes the row-batched kernels runs every pass on the exact kernels, the
tensor-core GEMM included, so it is bit-identical to the per-token path. The 0.99 bar is not moved; G4's default arm stays recorded as FAILED for the build it read.

**The rule as built** (`cuda/prefill_splitkv.go:cudaResident.prefillPastBoundExact`, read by prefillCore's `forceExactKernels` and by `PrefillSpanScratchBytes`): a prompt has a layer
past `attn_batched`'s shared-memory bound (its window-clamped span over 12,160 keys) that no fused attention serves (head dim other than 64 and 128, or the fused lever
off). It is judged on the whole prompt's length, so every chunk of such a prompt is exact, not only the passes past the bound. Below the bound, and on a prompt whose
past-bound layers the fused attention serves (the 1.5B and the other head-dim-64/128 families: `kv3-cuda-2026-10-13.md` §2.3's fidelity class), nothing changes. The
test hook that forces the route below the bound does not trigger it, so G2b's mixed-class cells still run the GEMM. Families it reaches beyond Gemma-3: Gemma 2 and dense
Gemma 4 (head dim 256 and 512), Phi-3 (96), and any family run with the fused attention off and the GEMM on; Gemma-4 E-models were exact already.

**A limit stated, not fixed:** a request that reuses a cached prefix computed by an earlier request below the bound (the fast kernels) keeps those rows, so its output is
exact only from the reused length on. Fresh prompts, which G4 grades, are exact.

**Its cost.** In the test rig at 12,400 tokens, G4's levers-off arm (what the default now runs) took 31.9 s against the default's 23.2 s, per-token 138.4 s (night of
2026-10-13). G5's Gemma-3-4B 5.68x was read with the GEMM on; it is re-read below. The day readings: the new `TestKV3Tiled_pastBoundPromptRunsExact_tiny` (the tiny
fixture with the fused attention off and the GEMM on, the shape of a head-dim-256 model) failed before the change, 350 tensor-core launches past the bound and 256 of 256
logits differing from the levers-off pass (`exact-past-bound-red-before.log`), and passes after it, 0 tensor-core launches and 0 logits and 0 of 12,697,600 KV elements
differing, with the GEMM still running at 3,900 tokens; the other tiled tests pass unchanged (`exact-past-bound-green-after.log`).

**Re-registered for the night, before the run (same bars as §3 and amendment 1, nothing else changes):**

- **G4, Gemma-3-4B at 12,400 tokens, both arms graded bit for bit** (the test's default arm is graded exact when the prompt is past the bound): logits and every K and
  V row equal the per-token path, 0 tensor-core launches. PASS or FAIL; there is no fidelity band.
- **G5 re-read, Gemma-3-4B only**, the serve binary built at the pinned commit, the §9 design (batched, per-token, batched; 12,400 and 20,000 tokens). **PASS** at 3x
  or more at 12,400 tokens; **AMBIGUOUS** 1.5x to 3x (parked as §3 says); below 1.5x the route is not kept as the default for this family. Gemma-4-E4B's 3.31x stands:
  the change does not touch an E-model.
- Job `kv3-tiled-exact` (`run-night.sh`, `JOB=exact`), about 20 minutes.

**Not taken: option (b)**, a fidelity gate for the tensor-core GEMM at depth on Gemma-3, is filed as its own item in `docs/tasks/task-kv-architecture-2026-10.md`.

## 12. Readings of the re-registered jobs (2026-10-14, graded against §11)

Both jobs ran on 2026-10-14 from the pinned worktree at `f7530d01` (the exact-past-the-bound rule and the merged tail-rule fix), the serve binary `serve-cuda-exact`
built at that commit. The owner started the queue by day, at 11:20 PDT; the runner held the timing lock, load average 0.34 at the start. Driver 595.91.07,
RTX 2070 SUPER. Logs and provenance: `kv3-tiled-exact-2026-10-13/queue-2026-10-14/` (`exact/`, `suite-rest/`).

| gate | reading | verdict |
|---|---|---|
| G4, Gemma-3-4B, levers off, 12,400 tokens | 0 of 262,144 logits and 0 of 863,436,800 KV elements differ from the per-token path; 25 row groups, 0 tensor-core launches | **PASS** |
| G4, Gemma-3-4B, default, 12,400 tokens (§11: graded bit for bit) | 0 of 262,144 logits and 0 of 863,436,800 KV elements differ; 25 row groups, 0 tensor-core launches; batched 31.88 s, per-token 138.4 s in the test rig | **PASS** |
| G5 re-read, Gemma-3-4B, per-token over the median batched at 12,400 tokens | **4.13x** (130.47 s against 31.49 and 31.63 s); at 20,000 tokens 3.48x (219.3 s against 62.96 and 62.96 s); no decline logged in either batched arm | **PASS** (bar 3x) |

**Vacuity.** The test asserts at least one row-group launch in each arm and no tensor-core launch in an arm graded exact; both arms print 25 row groups and 0
launches, and the two arms are now the same computation, so their equal 31.88 s is expected. The batched serve logs read `prefill path: batched` and log no decline,
the per-token one reads `sequential — GOINFER_BATCHED_PREFILL=0`.

**What the exact rule costs.** Against §10's G5 (the GEMM on, the same prompts and design), the batched arm went from 22.92 and 23.02 s to 31.49 and 31.63 s at 12,400
tokens (1.37x slower) and the ratio from 5.68x to 4.13x. The per-token arm read 130.47 s against 130.39 s the night before, so the box was in the same state for the
two readings; this comparison is across sessions and is context, not a graded ratio. Gemma-4-E4B's 3.31x stands (§11).

**The suite's remainder (`kv3-tiled-suite-rest`): no test failed; the job's red is a harness defect, fixed.** The job exited 1 on its own reconcile, `main listed=40
reached=36`. Every part ran: the main process 30 PASS, 0 FAIL, 6 SKIP; the four fresh-process tests each alone, 4 PASS (`TestGemma4_26B_cache_B`,
`TestP20ExpertLocality`, `TestResidentDrafter_extendContext`, `TestSpecPagerInteraction`); the drain group 2 PASS, 2 SKIP.

- **Why 36 of 40.** The four listed names that reached no verdict are `TestAudioEncoder_realMatchesCPU`, `TestAudioEncoderSpeed_record`,
  `TestWhisperSmall_clipsOnDevice` (`cuda/audio_encoder_real_test.go`, `//go:build cuda && realckpt`) and `TestLoRACostMeasure` (`cuda && goinfer_testhooks &&
  realckpt`). `suite-rest-main.txt` was built from a test listing that included `realckpt` (487 names at `f7530d01`; 486 at `a0ce6e40`, the source of §10's
  first figure), while the 2026-10-13 suite and this job both build `-tags 'cuda goinfer_testhooks'` (483 names at `f7530d01`), as `cmd/gate gpu`'s CUDA cells do.
  Those four were never in the 2026-10-13 suite's build, so they were not unreached tests of it, and `-run` could not match them. Under the run's own tags the
  2026-10-13 suite had 482 tests, 445 verdicts, and 37 without one; this job gave a verdict to every one of the 37 that it listed (36 here, `TestSpecPagerInteraction` alone).
- **The fix** (`run-night.sh`, `JOB=suite-rest`): before any device time the job lists the package under its own tags (`go test -list`, a compile) and exits 2 naming
  any listed test that build does not contain; `KV3_TILED_REST_LIST_ONLY=1` stops after that check. The four names are removed from `suite-rest-main.txt` (36 names;
  the job's `-run` pattern lists exactly 36). Dry run, no device (`suite-rest/list-check-dry-run.txt`): the list as queued exits 2 naming the four, the fixed list
  exits 0, 36 listed.
- **The skips, read rather than counted.** Main: `TestSpecResidentPrefillRegression`, `TestThetaAB`, `TestThetaProbe_CUDA`, `TestVsumMechanism`,
  `TestVsumSplitFidelity` and `TestVsumSplitGateVsReference` are opt-in probes that skip without their own variable in any run. Drain: `TestAllocFloor` and
  `TestMoERouteDemandThreshold` skip because the desktop compositor holds a CUDA context on the device; they need an exclusive device, which no queued run has. None
  of the eight touches prefill attention. They are skips, not passes.
- **Not re-queued.** With the list corrected the same run reconciles 36 of 36, and nothing it would read is unread: the `./cuda/` package under `cuda goinfer_testhooks`
  has now given a verdict on every test across the two runs, with no failure that is this branch's (§10's four: the three fresh-process tests pass alone here, the
  tail-rule test passed by day after the merge). The four `realckpt` tests are outside the suite as registered (§9) and outside `cmd/gate gpu`'s CUDA cells; three
  skip without their own variables, and none reaches the prefill code this branch changes.

**Where this leaves the branch.** On the build that carries §11's rule every registered gate reads PASS (G4's Gemma-3 default arm stays recorded FAILED for the build
§10 read), and the suite has no red of its own. It merges after `kv3-cuda-2026-10`, in a release later than v0.23.0.

## 13. Brought onto main, with the image-block twin and the family gates (2026-10-18)

The branch (`kv3-tiled-2026-10`) was merged onto main with one trivial conflict (the FMA lint's kernel list), after the owner's yes of 2026-10-18, with three additions the owner named:

- **The image-block twin.** The three row-batched kernels take the image-block table (`blocks`, `nBlocks`; a null table with 0 blocks is the causal case, the formula `attn_batched` has): a row whose own position is inside a block attends the whole block, every other row stays causal, and the
  window start comes from the CAUSAL count, never the widened one (the decoupling `attn_img_prefill.cu`'s header explains). `prefillCore` routes an image-block pass whose widest layer is past the shared-memory bound to them; `checkPrefillShmemImgBlocks` no longer declines a layer they serve.
  `prefill_splitkv.ptx` and its f16 twin were regenerated with NVRTC 12.9.86, the version the file already recorded.
- **Gates, each bit-equal to the exact kernel, with plants:**
  1. `TestPrefillSplitKV_rowsEqualAttnImgBatched`: the kernels against `attn_img_batched`, head dims 64, 128, 256 and 512, one and two blocks, a plain and a sliding-window layer with the block well over a window into the sequence, groups of 1, 7 and every row. Plant in the test: the block table withheld must differ.
  2. `TestKV3Families_forcedRouteEqualsAttnBatched`: on the Gemma 4 dense fixtures (`gemma4-dense-twogeom-tiny`: head dim 16 and a global head dim of 512, K = V; `gemma4-dense-scaled`: 256 and 512, 12 layers, a sliding window) and on the Qwen3.5 hybrid (`qwen35-tiny`), the route forced below the bound against `attn_batched`, with the tensor-core GEMM off and on, at 300, 1,000 and 2,600 tokens.
     Result: 0 logits and 0 KV elements differ in all 18 cases (KV elements compared per case from 19,200 to 37,273,600).
  3. `TestKV3Families_pastBoundBatchedEqualsPerToken`: at 12,400 tokens, past the bound, batched on the row kernels against the per-token path on `gemma4-dense-twogeom-tiny` (26,188,800 KV elements, 0 differ, 0 of 256 logits) and `qwen35-tiny` (793,600 KV elements, 0 differ, 0 of 256 logits); the negative arm: with the module absent, the same prompt declines by name.
  4. `TestKV3Families_imageBlockForcedRouteEqualsAttnImgBatched`: Gemma 3 VL, two blocks, the route forced against `attn_img_batched`: 0 of 262 logits, 0 of 44,800 KV elements differ. `TestKV3Families_imageBlockPastTheBound`: an image pass that starts past 12,160 keys prefills with no decline and finite logits (there is no per-token oracle for a bidirectional block, so the numbers rest on gates 1 and 4), and declines by name with the module absent.
  5. **Plants in the kernel,** each regenerated, run and restored: the window start from the widened count (the coupled formula) turns the image kernel gate and the image forced-route gate red; the V-sum ignoring the block table does the same; a V-sum that accumulates with a separate multiply and add turns the kernel gates and the Gemma 4 forced-route gates red.
- **Not covered, said plainly.** The Qwen3.5 gate is the MoE hybrid fixture (`qwen35-tiny`, model type `qwen3_5_moe_text`); there is no tiny dense Qwen3.5 fixture, so the 9B's dense feed-forward is not in it. The `PrefillSpanScratchBytes` admission figure takes no image blocks, so an image pass's score scratch is bounded by the 64 MiB budget rather than priced (`prefillSplitKVBudget`). The tensor-core GEMM stays off for the whole of a prompt past the bound (owner decision, §11).

