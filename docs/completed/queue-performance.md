# Performance queue — ARCHIVED 2026-08-31

> **This is the closed record. The live queue is [`docs/queue-performance.md`](../queue-performance.md),
> and it is empty.** Everything here is done, refuted, withdrawn, or moved to the track that owns
> it; nothing below is waiting on anyone.
>
> Archived under `docs/README.md`'s rule that finished work is kept **for the reasoning rather than
> the outcome**, negative results included — several entries here exist precisely because something
> was rebuilt that had already been measured and rejected.
>
> **Its citations are now retired from the live gate.** `docs/completed/` is excluded from
> `scripts/queue_citation_lint.py` by design, so the `path:line` references below are no longer
> re-keyed when code moves and will drift. That is the documented trade for archiving, not a defect
> — but it means a citation here is evidence of what was true when written, not a live pointer.
>
> **Two items left this queue unfinished rather than closed**: P10 (DSpark / DFlash block drafters)
> and P15 (DFlash 2) moved to the speculation track, [`docs/spec/08-dspark-dflash.md`](../spec/08-dspark-dflash.md).
> Their entries below are the state at the move. **Do not read them as closed.**


Throughput, latency, kernels, residency, memory. Anything whose success criterion is a **measured number** — a benchmark, a profile, a bytes-per-token figure. If the question is *how fast* or *how much memory*, it belongs here.

> **One of four queues.** The work list is split by *success criterion*, not by component:
> [performance](queue-performance.md) · [correctness](queue-correctness.md) ·
> [engineering](queue-engineering.md) · [release](queue-release.md).
> [`QUEUE.md`](QUEUE.md) is the index over all four and holds the cross-cutting sweeps.
>
> **Task docs are NOT queues.** `docs/task-*.md` are *design records* — why a thing is built as it
> is — and they are cited from 88 code comments. A queue entry cannot carry that, so the task docs
> stay put and the queues hold only the open work.
>
> Entries keep the section they were filed under (`In flight`, `Queued`, …) and their original IDs,
> so a citation to an ID still finds it.






## ~~G15 · CPU prefill falls off an INT4-SPECIFIC cliff at ~1.5k-3k tokens~~ — **WITHDRAWN 2026-08-25: THE CLIFF WAS A MEASUREMENT ARTIFACT**

**There is no int4 cliff. The finding was wrong and is withdrawn, not amended.** Kept in full
because a withdrawn measurement that leaves no trace is how the same wrong number gets rediscovered.

**What happened.** One measurement — int4, 3020 tokens, through `cmd/serve` — returned **1587.1 s**.
Three later measurements of the same thing return ~350 s:

| tokens | int4 (ORIGINAL, via serve) | int4 (RE-RUN, via serve, verified-idle box) | int8int8 (via serve) | int4 (direct `forwardLayersN`) |
|---|---|---|---|---|
| 170 | 4.5s | 4.1s | 3.3s | — |
| 620 | 24.4s | 22.5s | 19.7s | — |
| 1520 | 99.9s | 100.3s | 93.2s | — |
| 3020 | **1587.1s** | **355.5s** | 334.9s | **348.9s** |

Per-step exponents:

| step | int4 ORIGINAL | int4 RE-RUN | int8int8 |
|---|---|---|---|
| 170→620 | n^1.31 | n^1.32 | n^1.38 |
| 620→1520 | n^1.57 | n^1.67 | n^1.73 |
| 1520→3020 | **n^4.03** | **n^1.84** | n^1.86 |

The first three points reproduce within 3%. Only the 3020 point moved, and it moved by 4.5×. **int4
and int8int8 scale identically at ~n^1.85** — which is just attention's own O(n²)-ish cost, and
unremarkable. The n^4.03 "cliff", the "int4-specific" framing, and every conclusion drawn from them
are void.

**How it was caught:** a direct-call instrument built to profile the cliff failed to reproduce it
(348.9 s vs 1587.1 s). The first instinct was that the *instrument* was unrepresentative — the
[synthetic-reproduces-shape-not-pressure] trap. It was the opposite: the instrument was right and
the original number was bad. **A microbench disagreeing with the real path is a question, not a
verdict on the microbench.**

**What the artifact probably was.** Not established, and deliberately not asserted. The leading
candidate is an orphaned prefill from a client killed minutes earlier, burning a core throughout —
the G18 defect, which was still unfixed when that number was taken, and which by construction hits
the longest-running point hardest. That fits the shape (only the largest K affected) but was not
proven, and the honest record is "contaminated, cause not established".

**What survives:**
- **G16 stands entirely** — the single-threading is real, independently measured, and reproduced
  here. It was never derived from the cliff.
- **G17 stands** — the label was wrong on its own terms.
- **The dsh conclusion stands.** An ~8k-token agent prompt extrapolates to ~2200 s at n^1.85, still
  far past any harness idle timeout, so the Tier-0 verdict does not depend on the withdrawn number.
- **The real curve is still bad news for CPU agent work**, just for an ordinary reason: prefill is
  superlinear and single-threaded, so multi-thousand-token prompts are minutes. That is G16's lever,
  not a quant choice.

**Process change this earns.** Every timing measurement here must record machine state at the moment
it is taken — a pre-flight check that nothing else is running, and the load average alongside the
number. This artifact survived three documents and several hours of reasoning because the number
arrived with no context to contradict it. Retroactive reconstruction was only possible because the
sequence happened to be re-runnable.

## G16 · CPU batched prefill is single-threaded — ~4-5x left on the pure-Go lane — `mac`, **DONE 2026-08-26** (surfacing + lever; see G20 for the long-prompt half)

**Independent of G15 and true pre-cliff**, which is why it is filed separately: at 170 tokens,
int8int8 prefills at 51.5 tok/s while **five performance cores sit idle**.

CPU sampled every 2s during a large int8int8 prefill: `99.6 111.0 270.2 99.7 168.4 99.1 102.5` on a
box with **8 logical / 6 performance cores**. Predominantly one core with brief bursts. int4 is the
same (~105% throughout). This is the exact lane the pure-Go, single-binary pitch lives on.

**The bounded question:** did the batched-prefill matmuls simply never get the parallel dispatch the
decode path has? `runLayersFromEmbedN`'s own comment says the batched path runs "each layer's whole
K-row batch through ONE aikit matmul dispatch per head", and explicitly scopes threading batched
attention's heads OUT ("A1 move (a)... explicitly out of scope"). So the non-parallelism may be
deliberate and merely undocumented at the serve layer — establish that first; it changes this from
a bug to a roadmap item.

**BOUNDED QUESTION ANSWERED 2026-08-25 (mac, no hardware needed). Resolution: deliberate then,
never surfaced — a roadmap-plus-surfacing item, not a bug.**

It splits cleanly in two, and only one half is serial:

1. **Prefill ATTENTION heads are serial BY DESIGN, and the design is documented** — but in a
   *design doc*, as a deferral. `runLayersFromEmbedN` constructs `newHeadWorkerPool(1, K, maxKeys,
   hd)` and says so outright: *"this stays serial by construction (pool len 1 always takes
   attendBatchedHeads's serial branch)"*, citing
   `docs/prompts/attention-a1-bit-identical-restructure.md`'s scope-out. That doc's wording is a
   deferral, not a rejection: *"Prefill/verify (M=K) fast-pathing beyond what (b)/(c) give for free
   — they share the kernels, and that shared speedup is welcome, but **no M>1-specific work
   here**."* A1 declined to do it, not to have it.
2. **The weight matmuls DO parallelize.** aikit's `MatmulBTW8A8Batch` fans out over the
   concatenated column space via `ws.parallel(totalN, …)` above the MAC threshold (and `MatmulBT`
   is row-parallel above `SetParallelThreshold`). So the qkv and gate/up groups are threaded
   already.

**That reconciles the CPU fingerprint exactly**, and explains the falling rate in BOTH quants
without appealing to the cliff: the ~100% floor is the serial attention, the 270%/168% bursts are
the parallel matmul groups, and since serial attention is O(K²) while the parallel matmuls are
O(K), attention's share grows with prompt length — so effective tok/s falls as K rises in int4 and
int8int8 alike. (Strongly indicated by code + fingerprint; the same G15 pprof confirms or kills it
for free, so it is not worth a separate measurement.)

**Never surfaced anywhere a user could see it.** Grepped `README.md`, `docs/ARCHITECTURE.md`,
`docs/benchmarks.md`, `docs/env-vars.md`, `docs/api-tiers.md` for any single-threaded/serial
disclosure: **zero hits**. The decision lives only in a design doc and a code comment, while
`/health` advertised "batched" — which is precisely the gap G17 just closed at the serve layer.

**Both cheap halves are DONE 2026-08-25 (`0c84d1d`).** `docs/benchmarks.md` now carries the
absolute prefill numbers beside the v0.5.0 relative speedup, states the path is single-threaded, and
tells readers to size long-prompt expectations from the absolute table; `docs/task-attention-decode-cost.md`
records the deferral's measured cost and its non-goals bullet now points at it, so the standing
decision cannot be quoted without its price. **What remains in G16 is only the lever itself.**

**So this item becomes two cheap things and one real one:** surface the fact (the G17 pattern,
already built); record in the A1 campaign doc that the deferral has a measured cost; and then the
actual lever — thread prefill attention's heads under A1's bit-identity constraint, which
explicitly permits splitting independent outputs across workers (*"Parallelism may only split
independent outputs across workers/registers — heads, …"*), so the guarantee is not in the way.

**LEVER LANDED 2026-08-26.** Prefill attention's heads now run in parallel, bit-identical to the
serial path (`TestPrefillAttnPoolInvariance`: exact float equality at K=64/512/1200, serial vs 6
workers — not a tolerance, which would silently accept the reassociation A1 exists to prevent).
The pool is BUDGETED because a slot's `scores` buffer is K*nKeys floats, quadratic in prompt length.

**Measured, dense 1.5B `int8int8`, K=4096, M1 Pro:**

| workers | elapsed | rate | vs serial |
|---|---|---|---|
| 1 (the pre-G16 path) | 596.9s | 6.9 tok/s | — |
| 3 (what the 256 MiB budget grants at this K) | 250.2s | 16.4 tok/s | 2.39x |
| 6 (forced past the budget) | **180.4s** | **22.7 tok/s** | **3.31x** |

**Read the absolute numbers with suspicion and the 6-vs-3 comparison with confidence.** The serial
arm ran at ambient load 4.06 on a swapping box, so 3.31x is likely flattering and must be
re-measured on an idle machine before it is quoted anywhere. The 6-vs-3 gap is robust in the
opposite direction: the 6-worker arm ran at the HIGHEST ambient load of the three (7.92, rising to
10.43) and still won by 1.39x, so an idle box can only widen it.

**The owed 1520/3020 arms, measured (and these are the numbers to quote):**

| K | workers=1 (serial) | workers=6 (what the budget grants here — the shipped default) | speedup |
|---|---|---|---|
| 1520 | 89.7s (16.9 tok/s) | **33.8s (44.9 tok/s)** | **2.65x** |
| 3020 | 333.3s (9.1 tok/s) | **101.6s (29.7 tok/s)** | **3.28x** |

**These are the trustworthy pair, and the reason is the baseline.** The serial K=3020 arm came back
at **333.3s**, matching three independent prior measurements of the same thing (335.5s via serve,
337.4s direct int8int8, 348.9s direct int4) — the very quantity the withdrawn G15 cliff put at
1587.1s. A speedup is only as good as what it is measured against, and this denominator has now
been confirmed four times. Both `workers=6` arms also ran at HIGHER ambient load than their serial
counterparts (3.84→6.41 vs 1.39→2.70), so the ratios are conservative.

The K=4096 arms above stand as the worker-count sweep (they answer G20's go/no-go); the K=3020 pair
is the headline.

**The budget's edge, stated because it is where the motivating case sits.** At 256 MiB the pool is
6 workers through K=3020 and falls to **1** at K=8192 — one slot alone is 272 MB there. So this
speeds up short and mid prompts and does nothing for the ~8k agent transcripts that motivated G16.
That is the honest scope of what landed. See G20.

## G20 · Tile prefill attention's `scores` over K so long prompts can parallelize — `mac`, **DONE 2026-08-26**

**Substantiated by measurement, not assumed.** G16's lever is capped by memory, not by diminishing
returns: at K=4096, going from 3 workers to 6 bought a further **1.39x** (250.2s → 180.4s) — and
did so while handicapped by higher ambient load. The win has not saturated, so the memory budget is
the binding constraint on it, which is exactly the condition that makes tiling worth building.

**The problem:** `headWorkerScratch.scores` is `K*nKeys` floats — quadratic in prompt length. Per
slot that is 40.7 MB at K=3020, **272 MB at K=8192**, 4.2 GB at 32k. So `prefillAttnWorkers` must
collapse to one worker exactly where parallelism would pay most.

**The change:** compute attention in tiles over the K (query) dimension so a slot's scratch is
`tile*nKeys` rather than `K*nKeys`, making per-slot cost linear in nKeys and independent of prompt
length. Then the worker count is bounded by cores, not by prompt size.

**The constraint that governs it:** A1's bit-identity guarantee. Tiling the QUERY dimension splits
independent outputs (each query row's attention is its own reduction over keys), which is what A1
permits — *"Parallelism may only split independent outputs across workers/registers — heads, layers'
KV groups, individual QK scores, individual AV dims"*. It must NOT tile the KEY dimension, which
would re-associate the softmax denominator and the AV fold — the exact thing acc64 exists to
prevent. `TestPrefillAttnPoolInvariance` extends to cover it: same prompt, tiled vs untiled,
bit-identical.

**LANDED.** `attendOneHead` walks its query rows in tiles (`attnRowTile`, 8 MiB of `scores` per
slot); pool slots are sized for one tile instead of the whole prompt.

**Bit-identical, gated:** `TestPrefillAttnRowTileInvariance` compares tiles 1/7/64/333 against the
untiled shape (tile = K) at K=64/512/1200, float-for-float. The hazard was never the matmuls — it
was the position mapping: the softmax indexes `startPos+row`, `treeRowPos[row]` and
`treeMask[row]` by the GLOBAL row while buffers index within the tile, and an off-by-tile there is a
silent attention-mask bug that still produces fluent output. Exact equality is what catches that;
a tolerance would not.

**Per-slot scratch is now flat instead of quadratic**, which is the whole point:

| K | tile | per-slot | 6 workers | before G20 (per-slot / workers) |
|---|---|---|---|---|
| 3020 | 694 | 11.6 MB | 69.7 MB | 40.7 MB / 6 |
| 8192 | 256 | **16.2 MB** | 97.5 MB | **272 MB / 1** |
| 32768 | 64 | **40.1 MB** | 240.4 MB | **4.2 GB / 1** |

**Measured payoff at the size G20 exists for** — K=8192, dense 1.5B `int8int8`, M1 Pro:

| workers | elapsed | rate |
|---|---|---|
| 1 (serial) | 2381.8s | 3.4 tok/s |
| 6 (tiled) | **607.6s** | **13.5 tok/s** |

**3.92x**, and the load record says it is conservative rather than flattering: the SERIAL arm ran
8.09 → 1.70 (the box quieting) while the parallel arm ran 4.40 → 8.62 (busying), so the slow arm had
the better conditions. It also lands within 13% of the n^1.85 extrapolation from the four-times-
confirmed 333.3s at K=3020, which corroborates both numbers. Per the repo's own bench discipline the
arms could not be interleaved (a 40-minute serial arm), so treat 3.92x as indicative-but-conservative
rather than gold-standard.

**One G16 assertion was invalidated and REPLACED, not relaxed.** `TestPrefillAttnWorkerBudget` used
to assert that K=32768 falls back to one worker because a slot would be 4 GB. After tiling no such
slot exists. The property it protected — per-slot growth must stay bounded — is now asserted
directly against the real tiled size by `TestAttnRowTileBoundsScratch`.



## G17 · `prefill path: batched` advertises a speedup it does not deliver — FIXED 2026-08-25

C2 class: a claim readers hear as "fast" that names only a shape. `/health` and `/v1/models`
reported `batched (CPU forwardLayersN, one weight stream for the whole prompt)` while the measured
best-case CPU rate is the same order as the model's DECODE rate, on one thread.

**Fixed** in `decoder/residency.go`'s `PrefillPath`: the string now says it describes weight
streaming and not throughput, and that the path is single-threaded. Deliberately not gated on G15
or G16 — the wording is wrong today regardless of how either investigation lands.


## P14 — the CPU gap to llama.cpp is KERNEL ARITHMETIC, not threading and not quant format (2026-08-19)

aikit pushed back on "kernel quality" as too vague and proposed a specific, well-founded
alternative: the `int4ParThreshold` bug class that cost gemma4-26B a measured 2.3x when decode-time
M=1 matmuls fell under aikit's 16.78M-MAC serial default. Close enough to this 2.55x to be worth
ruling out properly. **Measured on Qwen3.8-27B; it is ruled out.**

**Every projection is far above the threshold.** hidden 5120, key/value_head_dim 128/128,
num_key/value_heads 16/48, head_dim 256, 24q/4kv, ffn 17408, vocab 248320:

| projection | MACs | vs `int4ParThreshold` (1<<20) |
|---|---|---|
| DeltaNet `in_proj_qkv` [10240,5120] | 52.4M | x50 |
| DeltaNet `in_proj_z` / `out_proj` | 31.5M | x30 |
| softmax `q_proj` [12288,5120] | 62.9M | x60 |
| softmax `k`/`v_proj` [1024,5120] | 5.24M | x5 |
| FFN gate/up/down [17408,5120] | 89.1M | x85 |
| LM head [248320,5120] | 1271M | x1212 |
| `in_proj_a`/`b` [48,5120] | 0.25M | BELOW — but f32, and 0.002% of per-token MACs |

**And the profile shows the opposite of the gemma4 signature**: 832% CPU (the fan-out IS firing),
`runtime.park_m` 1.22%, no `chanrecv` in the top 60 cumulative. Time is in `dotW4A8FoldAVX2` (58.3%
flat), `dequantI8AVX2` (2.9%), `dotFMA` (1.0%).

**What the numbers say instead.** 25.62 GMAC per token, 16.8 GMAC/s achieved:

```
goinfer  17.9 GB heap / 1.524 s = 11.7 GB/s   at 832% CPU
ollama   16.5 GB      / 0.597 s = 27.6 GB/s
DDR4-3200 dual-channel peak     ~51   GB/s
```

Neither engine saturates memory bandwidth, so neither is bandwidth-bound — the kernel is
COMPUTE-limited and doing ~2.4x more work per weight byte. aikit's own byte-count analysis supports
this from the other side: int4's 5.0 bits/weight (one f32 scale per 32) against Q4_K_M's ~4.5
(6-bit scales shared across a 256-weight superblock) is ~11%, nowhere near 2.55x. But that superblock
layout also means far FEWER scale loads per weight, which is an arithmetic difference, not a byte one.

**Agreed conclusion: do NOT start a Q4_K-style kernel rewrite.** The next step is a micro-benchmark
of `dotW4A8FoldAVX2` on ONE projection shape against its own theoretical ops/byte — per-group scale
handling, nibble unpack, accumulator width — before anyone redesigns a format. A format change is
the expensive answer to a question that has not been asked yet.

### P14 ops/byte RESULT (2026-08-31) — the penalty splits in two, and only HALF of it is the format

The question is asked. `aikit/linalg` gained the benchmark on both architectures
(`p14_w4a8_opsperbyte_bench_test.go`, `..._arm64_test.go`): `dotI8AVX2`/`dotI8SDOT` is the same MAC
count on the same activations with no unpack and no per-group scale, so the gap is a **difference,
not an estimate**. L1-resident, at the real Qwen3.8-27B projection dims.

| K=5120 (hidden) | int8 | W4A8 | penalty per MAC |
|---|---|---|---|
| arm64, M1 Pro | 50.07 GMAC/s | **25.09** | **2.00×** |
| amd64, Ryzen 3700X | 51.33 GMAC/s | **17.04** | **3.01×** |

Consistent across K=768 / 5120 / 17408 (arm64 1.99–2.00×, amd64 2.71–3.01×). **int8 matches across
both machines at ~50 GMAC/s**, so the baselines are sound and the W4A8 difference is real.

**Two independent costs, and the split is the finding:**

1. **~2× is inherent to W4A8** — a nibble unpack and a per-group scale fold against a MAC that is
   otherwise one instruction. arm64 pays it after two rounds of tuning, so it is not slack.
2. **A further ~1.5× is AVX2-specific.** arm64's W4A8 is **1.47× faster than amd64's** at the same
   shape on near-identical int8 baselines. That is kernel headroom, not a format problem.

**This sharpens P14's conclusion rather than overturning it.** A Q4_K-style format change targets
term 1, which is real but is the *smaller, harder* half and is already well-tuned on one
architecture. Term 2 needs no format change at all — and **amd64 is where P14's peer gap was
measured** (goinfer 11.7 GB/s of weights vs ollama 27.6). Bringing AVX2's W4A8 to arm64's ratio is
~+47% on the kernel that P14 showed *is* the end-to-end bottleneck: goinfer's whole-decode weight
throughput (11.7 GB/s) sits at ~90% of this kernel's isolated rate (10.65 GB/s), so there is
almost no composition overhead left — the kernel is the ceiling.

**Prior art incorporated rather than re-derived** (`docs/task-w4a8-neon-bandwidth.md`): that page's
Gate 1 items 1+2 already measured **dropping the centering subtract as a 3% REGRESSION** on arm64
(v1 209 ns/call vs v2 215 ns), because the kernel is **issue-limited** — instructions added anywhere
in the call cost real time. Centering is a closed question; item 3's split-half repack is the lever
that page identifies. The arm64 number here reproduces that page's 24.50 GMAC/s within 2.4%
(204.1 ns vs 209 ns), which is a free validation that both measurements describe the same kernel.

### The AVX2 term, attacked and REFUTED (2026-08-31) — the serial fold is not the ceiling

The obvious candidate for the 1.5× AVX2-specific term was the fold's **single accumulator**:
`dotW4A8FoldAVX2` ends each iteration in `VFMADD231PS Y11, Y9, Y10`, reading the previous
iteration's `Y10`, so one ~5-cycle dependency chain spans the whole row. aikit's
`TestW4A8IssueWidthProbe` reports idle issue slots — the signature of a latency-bound chain — and
the identical change on arm64 (`dotW4A8FoldSDOT2Acc`) is on record as a **real 1.4–1.75× win**.

Built (`dotW4A8Fold2AccAVX2`, aikit `231f989`) and **measured negative**:

| Ryzen 7 3700X, K=5120, order-alternated | pass 1 | pass 2 |
|---|---|---|
| 1Acc `dotW4A8FoldAVX2` | 17.26 | 17.38 GMAC/s |
| 2Acc `dotW4A8Fold2AccAVX2` | 17.45 | 17.41 GMAC/s |

~0.5%, inside noise. Correctness clean against the scalar oracle (~1e-7 across nGroups
2–160), so this is a refuted hypothesis and not a broken kernel.

**Why it was still right to build it.** aikit's priors note records the issue-width probe as *"a
hint, never load-bearing"*, and cites this very mechanism as the fix a mistrusted "not
issue-limited" reading would have talked someone out of. The probe motivated a cheap attempt; it
did not get to substitute for the A/B.

**Leading explanation, UNVERIFIED — no uop counters were read.** ~20 vector instructions per
32-MAC group, and Zen 2 cracks every 256-bit AVX2 op into two 128-bit uops ⇒ ~40 uops/group, a
6.7–10 cycle/group floor at 4–6 uops/cycle. The measurement is 1.854 ns/group = 6.7–8.2 cycles
depending on clock — already at that floor, so breaking a 5-cycle chain frees nothing. arm64 NEON
is 128-bit natively and pays no such split, which is also the leading explanation for its 1.47×
advantage on the same algorithm.

### Item 3 ported to AVX2 — 1.12×, hot AND cold (2026-08-30)

Built (`dotW4A8SplitHalfAVX2`, aikit `7e1af80`). The split-half layout deletes the two
`VPUNPCK` shuffles per group — one 16-byte load yields two contiguous 16-weight halves with no
interleave to undo — taking the prologue from 8 shuffle/logic ops to 6.

| Ryzen 7 3700X, K=5120, order-alternated | canonical | split-half | ratio |
|---|---|---|---|
| hot (L1-resident) | 17.09 / 17.11 | **19.20 / 19.23** | **1.12×** |
| cold (17,408 distinct rows, ~45 MB, past L3) | 16.19 / 16.35 | **18.41 / 18.22** | **1.11–1.14×** |

Cold is decode's real pattern — one weight row per output, never reused within a token — so the
win is not a hot-only artifact. Correctness gated against the scalar oracle reading the
**canonical** layout, so a repack bug and a kernel bug cannot cancel (rel-err ≤ 3.7e-07).

**The two ISAs disagree exactly as their diagnosed bottlenecks predict**, which is the strongest
evidence either diagnosis has:

| lever | arm64 | amd64 AVX2 |
|---|---|---|
| accumulator chains | **1.41–1.47×** | ~1% and ~0.5% — two dead ends |
| split-half prologue | **flat 1.000×** alone | **1.12×** |
| both together | 1.60–1.75×, shipped via `.giw` kind 4 | — |

arm64 is latency-bound, so shortening the prologue was invisible until 2Acc removed the stall.
AVX2 is port-bound, so the accumulator fix does nothing and the prologue fix works.

**Where it leaves the numbers:** the W4A8-vs-int8 penalty falls 3.01× → 2.69×, and the
amd64-behind-arm64 gap 1.47× → 1.31×. P14 established this kernel *is* the end-to-end CPU decode
bottleneck (whole-decode weight throughput ~90% of the kernel's isolated rate), so most of the 12%
should transfer — **should**, not does: that is an end-to-end measurement nobody has run.
**RUN 2026-08-30: it transfers, at +2.10% decode, below the bar that would have paid for it — see
"Item 3 — WIRED IN" below.**

**Not wired into dispatch, and the reason is a hard constraint.** ~~Not wired~~ — **SUPERSEDED
2026-08-30, see "Item 3 — WIRED IN" below; the constraint below is still exactly right and is how
it was built.** `packed` must be split-half, and the canonical packer feeds `.giw` kind=3's
zero-copy mmap load path — changing it would silently misdecode existing bundles, no error, wrong
numbers. Production needs a load-time repack, the pattern arm64 and the GPU backends already use,
plus the memory trade that carries (a second in-memory copy per int4 tensor, or freeing the
canonical one after repack).

**So the AVX2 lever is fewer instructions per group — the unpack prologue — not chain depth.**
That is what the split-half repack attacks, and `docs/task-w4a8-neon-bandwidth.md` already names
item 3's repack as "the one real lever Gate 0 identified, unattempted". This result reaches the
same conclusion from the amd64 side by a different route, which is the strongest form the
recommendation has had.

**Limits.** Micro-benchmarks, so they prove the primitive and never the composition — P14's 2.4×
peer ratio is end-to-end and is not restated by any ratio here. One box each; the M1 Pro ran at load
1.89 rather than idle, though its int8 baseline landing within 2.5% of the Ryzen's argues against
meaningful contamination. And this does **not** separate nibble unpack from per-group scale within
the 2×; that split needs a kernel variant, which is what item 3 would build.

### Item 3 — WIRED IN and MEASURED end-to-end, 2026-08-30. Verdict: real, parked, default OFF.

The end-to-end measurement the section above called "nobody has run" has now been run, and the
"not wired into dispatch" paragraph is superseded — it is wired, and the constraint it names was
handled rather than hit.

**aikit `db03fd2`** adds `RepackW4A8SplitHalf` (portable), a `q4SplitHalf` field, an opt-in
`RepackInt4SplitHalf()` gated on amd64 + AVX2 + group=32 + cols%32==0, and an amd64
`MatmulBTW4A8Into` that dispatches to the split-half kernel **at M=1 only**. Scales are NOT
repacked: split-half permutes nibbles *within* a group and never reorders groups, so one scale
array serves both layouts. **goinfer** calls it from `repackW4A8IfEligible`, at exactly the two
sites the arm64 row4 repack already used (`quantizeWM` / `streamQuantized`) — the
GGUF/safetensors paths, never the `.giw` loader, for the reason that function's comment gives.

**The `.giw` kind=3 constraint was met by construction, not avoided.** The repack allocates a
second buffer and never writes through the canonical bytes, which for a kind=3 tensor are a
zero-copy mmap alias of the file; rewriting them in place would silently misdecode every
existing bundle. `TestWeightMatSplitHalf_canonicalUntouched` pins both halves of that (bytes
unchanged, and the new layout is a distinct allocation).

**Result — `docs/measurements/w4a8-splithalf-decode-ab-PREREGISTERED.md`.** Qwen2.5-Coder-1.5B
int4, Ryzen 7 3700X, `BenchmarkDecode`, interleaved ON/OFF/ON/OFF in one session, **same binary
both arms**:

```
  ON  median 18.685 tok/s   range 18.61-18.77
  OFF median 18.300 tok/s   range 18.23-18.41     ranges DO NOT overlap
  effect +2.10%             floor 0.75%   pre-registered ship bar 4%
```

**+2.10% is real and is not enough.** It clears the floor with no overlap between arms, so the
1.12× kernel win does survive composition — it just arrives at the token level as ~2%, because
the kernel is ~half of decode. The pre-registration fixed +4% in advance as the price of the
memory, so this lands in the band it had already named AMBIGUOUS → PARKED. Default is **OFF**,
opt-in with `GOINFER_W4A8_SPLITHALF=1`; kernel, repack, wiring and tests all stay.

**A narrowing found after the fact, by CI rather than by the A/B.** aikit's canonical W4A8 dot
prefers its AVX-512 VNNI tier (added aikit v1.29.0) and the split-half kernel is AVX2-only, so on
a VNNI host the repack would swap a faster kernel for a slower one — a pessimization that nothing
would have failed on, since the AVX2 kernel is correct, only slower. `RepackInt4SplitHalf` now
declines on VNNI hosts (aikit `8fed687`). The Ryzen 7 3700X above is Zen 2 (AVX2, no VNNI), so
the +2.10% stands as measured — but it describes **AVX2-without-VNNI hosts**, not amd64 at large.
The equivalence test caught it as a numeric divergence (rel 1.09e-4 on a VNNI runner, passing at
every shape on the box); the divergence was the symptom and the downgrade was the actual bug.

**The memory it is short against, computed rather than read off RSS:** 196 tensors repacked, 0
skipped, **+624.8 MiB** of duplicate nibbles — 781 MiB → 1.37 GiB of int4 weights on that model,
+80%. Canonical is never dropped, because M>1 prefill and every non-AVX2 path still read it.

**Two guards worth reusing.** `TestW4A8SplitHalfFires_onBenchModel` fails on *zero tensors
repacked* — without it, a wrong quant or a wrong load path gives two identical arms and a
confident "flat" that measures nothing. And `TestW4A8SplitHalfWiring_matchesCanonical` drives
the repack through `quantizeWM` + `matmul()` rather than calling the kernel directly, because
aikit's own kernel test cannot show that goinfer's load path reaches the repack at all.

**The aikit bump this needed carries a numerics exposure that is NOT this change's, and is owed
to T3.** Reaching `RepackInt4SplitHalf` meant moving goinfer from aikit v1.28.0 to v1.30.0, which
crosses **v1.29.0 — the release that added the AVX-512 VNNI tier** for the int8/W4A8 kernels.
aikit validates VNNI against its scalar oracle at a **1e-5 relative tolerance, not exactly**, and
the AVX2 tier is validated the same way, so the two tiers are not bit-identical to each other. On
a VNNI host, goinfer's kernels therefore now accumulate differently than they did at v1.28.0.

Nothing measured in this repo is affected — the bench box is Zen 2 and the MacBook is arm64,
neither has VNNI — and the deps_hash refresh below is proof only on arm64, where the VNNI path
cannot execute. So this is **unproven rather than disproven**, and it belongs to the owed T3
re-validation rather than to this entry. Noted here because the bump is what introduced it and
that would otherwise be invisible.

Related and pre-existing: the manifest's `aikit_version` reads **v1.19.0** against a `go.mod` that
said v1.28.0 before this and v1.30.0 after — the exact hand-maintained-literal drift
`parity-coverage-policy.md` documents. Deliberately NOT corrected here: `aikit_version` is mixed
into `deps_hash`, so editing it re-stales every family and demands the full T3 run, and
hand-editing it before a refresh is the recorded mistake the refresh script's pre-flight exists to
abort.

**Reopen if** the kernel beats 1.12×, or if a decode-only build can drop canonical — that would
turn an ~80% increase into roughly zero and change the answer, not the measurement.

**An inconsistency this exposed, recorded rather than resolved.** The arm64 row4 repack sitting
at the same two call sites costs MORE memory — it duplicates the scales as well as the nibbles,
+0.625 B/weight (~100%) against split-half's +0.5 (~80%) — and it ships **default-ON**, having
never been put to a memory bar like the +4% this one was held to. Both cannot be right. Either
row4 owes the same end-to-end A/B and the same explicit trade (it has a load-time/RSS delta
recorded, but no "is the tok/s worth the bytes" verdict), or +4% is the wrong bar and this
result was parked against a number that is too strict. **Not resolved here, and deliberately not
resolved by adjusting the bar after seeing the result** — that is precisely the move the
pre-registration exists to prevent. It is a question about row4, and it should be answered by
measuring row4, not by re-reading this.

## P13 — the safetensors loader keeps the whole source mapping resident (FOUND 2026-08-19)

**Measured while adding the Qwen3.8 GGUF loader**, by loading the SAME model both ways and reading
`/proc/self/status`:

| path | Go heap | **RSS** | decode |
|---|---|---|---|
| safetensors (55.6 GB bf16) | 17.9 GB | **46.8 GB** | 0.656 tok/s |
| GGUF (16.5 GB Q4_K_M) | **17.9 GB** | **24.5 GB** | **1.109 tok/s** |

**The Go heap is IDENTICAL** — the quantized model is the same size either way, so this is not a
representation difference. The ~22 GB gap is the mmap'd SOURCE file staying resident: the
safetensors loader maps 55.6 GB of bf16, touches all of it to quantize, and never releases it. On a
62 GB box that leaves almost no headroom, and the dead bf16 pages compete with the hot quantized
weights for cache — which is the most plausible reading of the 1.69x decode difference between two
paths holding an identical heap.

**The fix is to drop the mapping once the weights are copied** (`madvise(MADV_DONTNEED)` on the
source range, or close the mapping outright), which needs one check first: whether any tensor still
ALIASES the mapping rather than owning a copy. The f32 vectors (norms, biases, the small DeltaNet
gates) are the candidates — if they alias, they must be copied before the mapping goes, and they are
small enough that copying them is free.

**Do not assume the 1.69x transfers**: it was measured on a box where 46.8 GB of 62 GB was resident.
On a machine with room to spare, the dead mapping costs address space and page-cache pressure but
may cost little wall-clock. Re-measure after the fix rather than claiming the number.

### DONE 2026-08-31 — released at end of load, and the aliasing fear was mostly unfounded

**The precondition resolved better than the entry expected.** The gate was "whether any tensor still
ALIASES the mapping", with f32 norms/biases/DeltaNet gates named as the candidates. The rule turns
out to be a dtype rule, taken from aikit's reader rather than from inspecting checkpoints: **BF16 and
F16 are widened into fresh storage on read** ("the result then does not alias the file"), while every
other dtype is served by `reinterpretLE`, which takes a zero-copy view when aligned — the common case.

Measured across five real checkpoints, **modern HF checkpoints are BF16 throughout**:

| checkpoint | size | F32 bytes (the aliasing set) |
|---|---|---|
| qwen3.8-27b (the one P13 measured) | 55.6 GB | **0** |
| qwen3.6-35b-a3b | 71.9 GB | **0** |
| gemma-4-26b-a4b-it | 51.6 GB | **0** |
| mellum2-unq | 24.3 GB | **0** |
| qwen3.5-0.8b | 1.7 GB | 0.01 MB (`linear_attn.A_log`) |

So on the checkpoints that matter there is nothing to copy first. `loadWeights` now closes the source
at end of load when `mmapAliasRisk` finds no risky dtype; `GOINFER_P13_OFF=1` restores the old
behaviour.

**Measured, paired, same checkpoint** (Mellum2 4-layer slice, 4.25 GB bf16 source, int8int8):

| arm | RSS after load | mapping retained |
|---|---|---|
| release ON | **1.22 GB** | no |
| release OFF (old) | 3.12 GB | yes |

**1.90 GB freed, 61% less resident after load.** The decode-throughput half of the original finding
is deliberately NOT restated: 1.69x came from a box at 46.8 GB of 62 GB, and this arm ran with room
to spare, so it measures the mapping and not the pressure. Whether throughput moves on a loaded box
is a separate run.

**The gate is tested, not assumed** (`decoder/p13_mmap_release_test.go`): a table-driven dtype
classification so a new dtype must be classified deliberately rather than inheriting "safe", plus an
end-to-end check that the mapping is gone AND a forward pass still succeeds — a wrongly-released
mapping is a use-after-free that would surface as garbage output far from the loader, not as a load
error.

## P12 — the qwen35 family's projections were f32 at every quant (FIXED 2026-08-19)

**Found by benchmarking Qwen3.8-27B against Ollama on the same box**, which goinfer lost 4.07x. The
cause was not kernel quality, it was BYTES.

`deltaNetWeights` and `qwenAttnWeights` held `[]float32` and went through `linalg.MatmulBT`
regardless of `Options.Quant` — "parity-first", from the qwen3_5_moe bring-up, inherited by every
family on that path (Qwen3.5-MoE, Qwen3-Next, Qwen3.8) and never revisited. So a 27.8B Qwen3.8 at
`Quant:"int4"` streamed **~29 GB of f32 attention weights per token** (22.1 GB across 48 DeltaNet
layers + 6.7 GB across 16 softmax layers) while its FFN was a tidy ~9.5 GB of int4. Decode at this
size is memory-bandwidth-bound, so that WAS the speed.

| Qwen3.8-27B, CPU, 32 tokens | before | after | |
|---|---|---|---|
| decode | 0.411 tok/s | **0.656 tok/s** | **1.60x** |
| TTFT (7-token prompt) | 73.2 s | **9.9 s** | **7.4x** |
| load (bf16 safetensors → int4) | 145.2 s | **68.0 s** | 2.1x |

Decode gained 1.60x rather than the ~2.4x the byte count predicts: the DeltaNet recurrence is scalar
work that did not change, and W4A8 pays an activation-quantization cost f32 SIMD does not. TTFT
gained 7.4x because prefill is almost pure matmul.

**Design: quantize only when asked.** `WeightMat` stays f32 when `Options.Quant` is unset, so every
unquantized parity gate is bit-unchanged — the DeltaNet golden, qwen3_5_moe forward parity and the
Qwen3.8 tiny golden all pass untouched, which is what made this safe to do at all.

**What it costs, recorded rather than glossed:** the family's T3 oracle moved, because more of the
model is now quantized. `TestQwen35Real_gate2FullModel` (int8 vs bf16) went 66/80 -> **62/80**
argmax, cosine min 0.99333 -> **0.99069**, mean 0.99837 -> **0.99644**, worst divergence gap 0.0045
-> **0.0073** of range. Still inside the >=0.98 floor with every divergence a near-tie, re-run green
on the real 35B, and the manifest says so in those words.

**Evidence the new int4 numbers are RIGHT and not merely different**: int4-vs-f32 on qwen35-tiny is
cosine **0.997923** with matching argmax — BETTER than mixtral (0.987977) and phi3 (0.992787), both
long-shipping int4 families. The real 27.8B still answers correctly at int4 (trigram 0.797, three
correct Paris landmarks with correct detail). The int4 golden was re-baked on that evidence; the
diff changed `qwen35-tiny` ONLY, which is independent confirmation the change is scoped.

**Still open (the remaining 2.55x to Ollama):** the DeltaNet recurrence is scalar Go, llama.cpp's
k-quant kernels use hand-tuned AVX2, and goinfer cannot read this family's GGUF at all
(`architecture "qwen35" unsupported` — the dense loader was deferred at bring-up, so it re-quantizes
55.6 GB of bf16 on every load against Ollama's 0.7 s mmap). The CUDA DeltaNet kernel is the next
track, and this change is its prerequisite: resident runners consume `WeightMat` + the W4A8/W8A8
kernels, which f32 slices would have had to become anyway.

## In flight

**G24 · Attention A3 — an f32 attention path IS worth its divergence flag** — `mac`, **DONE 2026-08-26.**

> **Renumbered from G23 on 2026-08-26.** Two entries briefly held G23: this one, claimed at
> `53e1c6d`, and `queue-engineering.md`'s T3-cosine-bar item, written concurrently on the other
> box. IDs are meant to be globally addressable — "see G23" must resolve to one place — so this
> one moved. It moved rather than the other because the other is cited from a PUSHED commit body
> (`dc35a31`), which can never be corrected, while every reference to this one was in files that
> could be updated in a single pass. Commits before this point cite it as G23; the artifacts are
> `decoder/g24_attnkernel_test.go` and `docs/measurements/attention-a3-kernel-ratio-2026-08-26.md`.

**Answer: 2.28× end-to-end at K=8192** (602.9s → 264.6s), from an 8.14× kernel ratio; divergence
cosine 0.9976, stable across depth. Shipped as `--cpu-fast-attention`, off by default, MoE refused,
speculative verify structurally excluded. Full record:
`docs/measurements/attention-a3-kernel-ratio-2026-08-26.md`.

`docs/task-attention-decode-cost.md` closed A2/A3 with a named reopening trigger: *"the 32k regime
is where A3 might still earn its keep (revisit with A1's long-context numbers in hand)."* **Those
numbers now exist**, from the G20 work — K=8192 prefill, dense 1.5B, M1 Pro:

| function | flat |
|---|---|
| `MatmulAVAcc64` | **51.1%** |
| `MatmulQKAcc64` | **18.7%** |
| `dotI8SDOT` (weight matmul) | 12.7% |

**~70% of long-context prefill is attention in the acc64 (f64) path**, whose own comment calls it
"~3.7× slower than f32". That is the trigger firing.

**What is NOT yet established, and is this item's whole job:** how much of that an f32 path actually
recovers. Two reasons the naive estimate is wrong:

1. **acc64 skips work f32 must do.** `MatmulQK/AVAcc64` read K/V *directly* by stride, "skipping a
   kh gather entirely" and "skipping a vt gather+transpose". The f32 branch pays both. So the
   end-to-end delta is smaller than the kernel ratio, by an amount nobody has measured.
2. **The f32 branch is single-threaded BY CONSTRUCTION.** Its per-kv-group `kh`/`vt` gather is
   shared mutable state — "this stays single-threaded (the gather itself is shared, mutable state a
   concurrent split would race on)" — so it runs through `pool[0]` alone. An end-to-end A/B would
   compare parallel-acc64 against serial-f32 and measure the confound, not the kernel.

**So: measure the KERNELS at the real tiled shapes** (kt=256, hd=128, nKeys=8192 — what G20's tiling
actually calls), and derive the ceiling from the profile share rather than running a rigged race.

**Decision rule, fixed before the number arrives.** A3 costs a permanently-supported,
not-bit-identical `--cpu-fast-attention` flag (the `--metal-fast-prefill` precedent) and it breaks
A1's stated guarantees — spec-decode verify == sequential greedy, and decode == prefill — for
anything that enables it. Near the 3.7× kernel ratio, that surface is earned. Near 1.3×, it is not,
and this item closes as a measured negative. **If it is built, the f32 path must also be made to
fan out** (gather once per kv-group, then split the group's query heads across workers reading it) —
otherwise a faster kernel loses to parallel acc64 anyway.

**P16 · Re-anchor every `linux`-box measurement to the Nobara 44 / driver 595.91.07 stack** —
`linux`, **DONE 2026-08-31. Every leg is re-anchored or deliberately retired; none owes a run.**

> **Closed on a decision, not a sweep.** Six of the seven legs were already discharged and the
> status line below had not caught up (table under "STILL STALE"). The seventh — the v0.11.0
> qualification — was **retired rather than re-measured**, on the same basis §B was: it was a
> go/no-go for a tag that shipped 2026-08-10, and its numbers were never independent — the table
> states they are §B6/§B7 by code-identity, and both are now re-anchored. Re-running it would
> restate current anchors under an old tag's name.
>
> **The retirement found a live trap, which is the part worth keeping.** That table also claimed to
> "double as v1.0's sweep **iff** the code delta between the two tags stays zero". Measured:
> v0.11.0 → HEAD is **802 commits, 467 Go files, +48,477 / −2,066 lines**. Reusing it for v1.0
> would have qualified the release against a codebase that no longer exists. The clause is struck
> at its own site in `benchmarks.md`. **v1.0 needs its own sweep, and that is a release-queue item,
> not a performance-queue one.**

**What moved, from `dnf history` (transactions 96-102, 2026-08-25 22:2x PDT), not from memory:**
a Nobara **43 → 44** upgrade of ~7,100 packages that carried the NVIDIA driver
`595.58.03-1.fc43` → **`595.91.07-2.fc44`** (txn 99), the running kernel
`7.0.5-200.nobara.fc43` → **`7.2.0-202.nobara.fc44`** (txn 98), glibc → **`2.43-8.fc44`**, and the
whole graphics stack. `nvidia-smi` now reports **CUDA 13.2**. **2026-08-26 04:44 PDT was the first
boot on it.**

**The decision was taken on the driver alone (2026-08-25); the scope is wider.** Kernel, libc and
loader moved with it, so CPU-side dispatch and scheduling are in scope, not only the GPU rows. The
re-anchor decision does not change — the list of what must be re-measured does.

**STAGE 2 (timing) DONE, 2026-08-26 — 33/33 cells, zero errors, landed as `benchmarks.md` §B8.**
Gate PASS at `a161bd6` 13:44Z, sweep 13:44→14:23Z. The competitive picture is unchanged by the
upgrade: 0.5B **1.24×** and 1.5B **1.13×** at 128, parity at 7B/128 (**1.00×**), losing with depth
to **0.71×** at 3900 on every model.

**The peer was the control, and it held to <0.5%.** Across a distro major upgrade, Ollama
reproduced its 2026-08-09 greedy numbers in all seven comparable cells within 0.7% (largest 0.63%;
0.5B@2048 to the decimal), on a box whose documented between-session drift is ~3.5%. So the stack did not move decode
throughput for an engine that did not change, and a goinfer-side delta above ~1% is attributable
rather than dismissible. goinfer's own cells moved +0.7% to +4.3% except the two @512 cells —
**not claimed as a speedup**: that cell carries the set's widest spread and the old 0.5B row was
non-monotonic (512 read *below* 2048), so the honest reading is that the old 512 cells were suspect.

**STILL STALE, and none of it is discharged by this run:** §B (WebGPU int8/q8_0 — §B8 is q4_K_M),
§B4, §B6, §B7, the v0.11.0 qualification, and §B5's sampled / phi3-mini / gemma3-1b cells.

> **^ THAT LIST WAS FOUR ITEMS OUT OF DATE WHEN IT WAS READ ON 2026-08-31.** Every leg on it had
> already been discharged, most of them on 2026-08-27, and the line was never updated:
>
> | leg | actual state |
> |---|---|
> | §B (WebGPU int8/q8_0) | **RETIRED 2026-08-27** — withdrawn with a costed rationale, not owed |
> | §B4 | **re-anchored** §B4.1, and §B4.2 added 2026-08-31 |
> | §B5 sampled / phi3-mini / gemma3-1b | **re-anchored** §B5.1 — it carries "the rows §B5 could not carry forward" by name |
> | §B6 | **re-anchored** §B6.3 |
> | §B7 | **re-anchored** §B7.1 |
> | v0.11.0 qualification | **RETIRED 2026-08-31** — see below |
>
> Same defect as A2, D3b and A10: the work landed, the status line did not move. `benchmarks.md`'s
> own re-anchor banner had already been corrected to say "the only leg still owing a run is the
> v0.11.0 qualification" — so the two documents disagreed, and the queue was the stale one.

**A harness defect this run exposed, now fixed.** `bench_peer.py` stamped NO provenance into its
results file — no driver, no commit, no load average, nothing but the numbers. The page it feeds is
provenance-gated, and the re-anchor this task exists to perform was needed *because* a driver
version had not been attached to the numbers it governed; the instrument could not attach it either.
It now writes a header (driver, distro, kernel, commit + tree-dirty, peer version, binary mtimes)
plus load average and GPU temperature at every cell, and **refuses to start on a non-idle box**
rather than warning — a warning inside a 39-minute detached run is read, if at all, after the
numbers already exist and have been believed. This run's own machine state is therefore **attested
by the operator, not instrument-read**; its archived results header is marked `RECONSTRUCTED` and
separates the two field by field. Decision (Francis, 2026-08-26): keep the numbers, fix the harness.

**STAGE 1 (parity) RESULT, 2026-08-26 — the JIT is CLEAN; two allocation pins moved.** The gate ran
48 min and came back **RED on exactly two tests**, both allocation-accounting pins, and **green on
everything that asserts a forward**: the heavy tier (real models, 249 tests, 2758s) PASSED, CUDA
graphs replay is bit-exact under forced capture, and **24/24 PTX regenerate byte-identically** at
their recorded NVRTC. So the driver's new compiler did not move the numerics — which is the question
that had to be answered before any tok/s, and it is answered.

The two reds were `TestAllocFloor` and `TestMoERouteDemandThreshold`, and they are **one finding**:

| component | pinned | re-measured 2026-08-26 | |
|---|---|---|---|
| device floor (`TestAllocFloor`) | 54,263,808 | **1,769,472** | MOVED (3 processes, byte-identical) |
| moe_route residual (`TestMoERouteFirstLaunchReservation`) | 138,412,032 | **138,412,032** | unchanged, PASSES |
| demand (balloon bisection) | — | **140,181,504** | = floor + residual, **to the byte** |

`1,769,472 + 138,412,032 = 140,181,504`. The identity CLOSES, so per A11's own recorded instruction
a **component** moved and the demand pin is downstream of it — update the component, not the number.
This is explicitly **not** the branch that would require re-deriving A1/A5/A7/A9. Both pins
re-derived and now green; the safety direction is the same as 2026-08-21 and in the safe sense — a
smaller floor means MORE headroom, and the margin clears the worst-regime demand by 250.3 MiB where
it cleared by 200.2 MiB before.

Worth noting as a result in its own right: the new driver hands back ~52.5 MB of VRAM that the old
one reported free and would not allocate. On an 8 GB card carrying a 26B, that is not noise.

Found while re-deriving and filed separately as **G20**: the demand gate's WARM branch has been
unreachable since 2026-08-21.

**Do the parity half FIRST, before any tok/s.** The `cuda/` backend is **driver-JIT**: the driver
compiles the frozen PTX at load, so this upgrade changed the *compiler*, not just the runtime. A
throughput re-measure on a stack whose bit-identity has not been re-established measures the wrong
thing. Smoke-checked already on 2026-08-26: `TestGemvW4A8Batched_bitIdentical` passes at M=1/8/13/100
(a real `--- PASS`, not an `ok` over skips) — that is one kernel, not the gate.

**Scope of the marked rows — 78, in seven sections of `docs/benchmarks.md`:** §B (2), §B2 (6),
§B4 (8), §B5 (19), the v0.11.0 qualification (2), §B6 (16), §B7 (25). Rows already RETIRED or
HISTORICAL in §B2 stay that way. §A (Apple Silicon CPU) and §B3 (Metal) are `mac` rows and are
**unaffected** — do not touch them.

**Harness:** `scripts/bench_peer.py`, peer Ollama **v0.32.5** at `~/ollama-0325` (client confirmed
0.32.5 on 2026-08-26), both engines over their own HTTP server, decode-only, interleaved cell by
cell with a server restart between cells. **Not `bench_compare.sh`** — it never drives the peer, and
that is exactly the defect that retired the old §B2 ratio.

**Gate on re-entering a row:** the full Methodology provenance — machine, checkpoint+quant, greedy
or explicit sampling, pinned versions, **the new driver AND distro/kernel**, date, thermal note,
local-disk path under `~/models` (**never `/Volumes/`, and never `/srv/models` — the archive is not
a bench surface on the box it is local to either**). Peer comparisons same-session interleaved;
drift between sessions is ~3.5% on this box.

**`bench_peer.py` does NOT re-anchor everything — its 33 cells cover the §B/§B2/§B5-style peer
rows only.** §B4 (26B host↔VRAM streaming), §B6 (split-KV) and §B7 (deep context) are separate legs
with their own procedures. **Do not mark those re-anchored off this run**; they stay STALE until
their own leg is re-run. (Carried from the pre-reboot handoff note in `~/bench-reanchor/`.)

**Row-count reconciliation.** That handoff note counts ~74 rows over nine sections; the box now in
`benchmarks.md` counts 78 over seven. Same rows, two conventions: this count includes four
blockquoted §B2 rows the other skips, folds §B6.1/§B6.2 into §B6 and §B7.1 into §B7, and draws the
§B5 header boundary one row differently. Neither is wrong — do not "fix" one to match the other.

**The run is detached and does not need the session that started it.** A supervisor script in
Francis's `bench-reanchor` directory on the `linux` box runs both stages in order, refuses to start
stage 2 unless stage 1 exits green, and appends every transition to a STATUS file beside it. If it
dies, re-running it is the whole recovery: stage 1 is idempotent and stage 2 resumes from the cells
already in its results JSON. A state note in the same directory carries the rest cold. That
directory is outside the repo, so it resolves on that box only — hence prose here rather than a
path anyone else's clone would fail to follow.

**Two traps already paid for, 2026-08-26, both worth inheriting.** (1) The gate's verdict has
THREE states, and an untracked file in the worktree makes it **INCONCLUSIVE, not PASS** — the first
attempt wrote its log into `docs/measurements/` and had to be aborted six minutes in. Gate logs
belong outside the tree and get committed **after** the run. (2) Killing that gate left a
`cuda.test` process holding **5.3 GB of VRAM**; the next run would have measured a card that was
not idle. Check `nvidia-smi --query-compute-apps` before relaunching anything.

**Do not re-baseline a floor because a number moved.** If a row lands materially below its
pre-upgrade value, that is a finding to be explained by mechanism, not a new baseline to bless.

**A11 · moe_route's demand threshold — RESOLVED 2026-08-12. The identity now CLOSES; the old pin was
the outlier** — `linux`, **CLOSED**, and it merges with A9-RESID

**A11 and A9-RESID are ONE finding with two observations.** `TestMoERouteDemandThreshold` failed with
both bounds moved by exactly **+589,824 B**, deterministically. That number was already on the record:
A9-RESID called it "baseline drift". It is not drift. It is the amount by which
**demand = floor + residual** failed to close at `MOE_MAX_E=512` while closing exactly at 256:

    256:  151,191,552 + 54,525,952  = 205,717,504   measured 205,717,504   EXACT
    512:  151,191,552 + 138,412,032 = 289,603,584   measured 289,013,760   short by 589,824

**The measurement now reads 289,603,584 — the closed form, to the byte.**

**Both components re-measured, and BOTH HELD** — so this is a fourth outcome, not one of the three
that were pre-registered (floor moved / residual moved / identity wrong):

| component | recorded | re-measured | |
|---|---|---|---|
| floor (allocate-until-failure, fresh context: 7,665,287,168 reported − 7,514,095,616 obtained) | 151,191,552 | **151,191,552** | unchanged |
| residual (`cuMemGetInfo` either side of first launch) | 138,412,032 | **138,412,032** | unchanged |
| demand (balloon bisection) | 289,013,760 | **289,603,584** | = floor + residual |

Nothing about the machine or the kernel moved — consistent with `aikit/gpu` never leaving `v0.28.0`
and the gate's own 21/21 byte-identical PTX. **The OLD PIN was the outlier**, recorded from the one
measurement that did not close, with the 589,824 misattributed to drift rather than read as *a
failure to close*. A9-RESID's "CLOSED — baseline variance" verdict was wrong on the mechanism while
being harmless in effect.

**The pins are RE-DERIVED, not edited to match** (`cuda/moe_route_demand_test.go`), with the identity
recorded at the pin site as their justification, plus the instruction for next time: if these move
again, check the identity first — if `floor + residual` still equals the demand, the components moved
and the pin is downstream of them.

**BLAST RADIUS — narrower than the first filing implied.**

- **A1 is NOT at risk.** Its closed form does not rest on the demand threshold: it was confirmed by
  predicting allocation at 16, 30 and 34 slots and free-after-`allocSlots` at 34, **all to the byte**.
  Nothing in this touches it, and A11 must not be read as reopening the A-chain.
- **In scope, and now corrected:** A9's demand figure, and A10's decomposition — which is *vindicated*
  rather than revisited. The identity it proposed now closes at both `MOE_MAX_E` values instead of
  one.
- **Nothing that ships is affected.** The margin check passes with `slotMarginBytes` 402,653,184 ≥
  289,603,584, **clear by 107.8 MiB**, and the 33-slot cap is a binary search against the margin, not
  against these pins. This was a record-versus-machine mismatch, never a defect.

**The lesson worth keeping: a residue that "is exactly the drift we measured elsewhere" deserves more
suspicion than a residue that is merely small.** 589,824 B against 289 MiB is 0.2% — easy to wave
through. Its appearing *twice, exactly* is what made it a mechanism, and the second observation is
what resolved the first.


## Queued

**A1 · Why the 26B forward failed at 34 slots** — `linux`, **CLOSED**

Waste is **allocation granularity**, fully accounted. Buffers are 123,904 × {1, 2, 8, 16} bytes per
slot, 4 per layer, 30 layers; each is rounded up **independently** to the driver's 2 MiB quantum.

    x = n × 123,904 / 2,097,152
    Q(n)           = ceil(x) + ceil(2x) + ceil(8x) + ceil(16x)      per-layer quanta
    Requirement(n) = 30 × Q(n) × 2,097,152

`roundedPredicted` matched measured actual **to the byte** at 16 and 30 slots, with structure
asserted before totals at both (120 allocations, 4 distinct sizes, 30 occurrences each, distinct
sum = n × 3,345,408), and predicted post-`allocSlots` free matched to the byte at 34 — a slot count
never used to derive the form.

**No memory was ever unaccounted.** The delta was real; the prior closed form under-predicted it by
ignoring the quantum. Machine state for every figure here: free 3,847,880,704 B, margin
402,653,184 B (384 MiB), quantum 2,097,152 B, 30 MoE layers.

| n | Q(n) | requirement | + margin | vs free | free after `allocSlots` |
|---|---|---|---|---|---|
| 30 | 50 | 3,145,728,000 | 3,548,381,184 | 299,499,520 spare | 702,152,704 |
| 32 | 53 | 3,334,471,680 | 3,737,124,864 | 110,755,840 spare | 513,409,024 |
| 33 | 54 | 3,397,386,240 | 3,800,039,424 | 47,841,280 spare | 450,494,464 |
| 34 | 58 | 3,649,044,480 | 4,051,697,664 | **203,816,960 over** | 198,836,224 |

**Corrected cap is 33.** At n = 34, x crosses 2 and all four buffers tip at once — a 4-quanta step.
34 is the worst boundary in the range, and it is the value the README used to recommend.

Known and durable: driver quantum is **2 MiB** (not next-power-of-two — 5→6, 6→6, 9→10 MiB
measured); sub-quantum requests are pool-served but **not free**. Both asserted in
`cuda/allocgran_test.go` with their measurements.

**A2 · 26B documentation correction** — `linux`, **DONE 2026-08-31.** Published as
`docs/benchmarks.md` §B4.2. The pre-registered prediction at the bottom of this entry **resolved,
and it half-missed** — see the verdict below.

> **THE PREDICTION RESOLVED WITHOUT ANYONE NOTICING, which is the part worth recording.** This
> entry pre-registered *"~74-78% hit rate, ~15.0-15.8 tok/s"* for 30 slots. §B4.1's re-anchor run
> measured exactly that configuration on 2026-08-27 — 76.1% hit rate, **16.12 tok/s** — and the
> result was never carried back here. It sat for four days as an open pre-registration whose
> answer was already in the repo.
>
> | | pre-registered | measured @30 | verdict |
> |---|---|---|---|
> | hit rate | 74-78% | **76.1%** | **in band** |
> | decode | 15.0-15.8 tok/s | **16.12** | **ABOVE band, by ~2%** |
>
> **Half-confirmed, and the half that missed is the informative one.** The hit-rate curve was
> predicted well, so the LRU model behind it is sound. Throughput came in above the band, which
> means the prediction implicitly assumed decode scales with hit rate more weakly than it does —
> the interpolation from 16 slots (57.3%, 11.42) to 38 (81.6%, 16.98) was treated as closer to
> linear in slots than it is. Recorded as a miss rather than rounded into a hit: 16.12 is outside
> 15.0-15.8, and a band you widen after seeing the number is not a band.

> **AND THE CAP MOVED, so this entry's headline figure was stale before it was published.** "The
> corrected cap is 33" is keyed to A1's machine state — free 3,847,880,704 B on driver
> `595.58.03`. The current stack (`595.91.07`, Nobara 44) reports less free and caps the same
> 48-slot request at **30**. The closed form is unchanged and still exact; the number it produces
> is a function of free VRAM, which is not a constant of the codebase. §B4.2 therefore publishes
> the requirement column and the free-VRAM threshold per slot count, and names the cap PER STACK
> rather than as a single number.

**Original entry follows.**

**A2 (original) · 26B documentation correction** — `linux`, **unblocked by A1**

38 slots is unreachable with correct accounting. The published 16.98 tok/s was measured at a cap
that shouldn't have been granted — it worked with ~133 MB leftover, equal to the forward's demand
within measurement error. The README currently instructs `GOINFER_MOE_CACHE_SLOTS=48`.

The corrected-cap figure is **33**, and the leftover-VRAM column can now be filled from the closed
form (the table under A1). Publish what the corrected cap delivers, with that fourth column for
**leftover VRAM after `allocSlots`** — the figure that distinguishes a safe operating point from a
lucky one. Record 16.98 as **measured-but-unsafe rather than retracted**; the correction is a few
percent, not a repudiation.

The hit-rate curve is worth publishing alongside, since it explains the flag better than an
instruction does. **The leftover-VRAM column now falls out of the closed form** — it is what
distinguishes a safe operating point from a lucky one, and it is the column whose absence let 38 be
published:

| slots/layer | LRU hit rate | decode | requirement (rounded) | leftover after `allocSlots` |
|---|---|---|---|---|
| 8 (default) | 0% — **inert** | ~5 tok/s | 838,860,800 | 3,009,019,904 |
| 16 | 57.3% | 11.33 tok/s | 1,698,693,120 | 2,149,187,584 |
| 30 | *not yet measured* | *not yet measured* | 3,145,728,000 | 702,152,704 |
| 33 | *not yet measured* | *not yet measured* | 3,397,386,240 | 450,494,464 |
| 34 | — | **0 tok/s** | 3,649,044,480 | 198,836,224 — **below the 289,013,760 demand** |
| 38 | 81.6% | 16.98 tok/s | 3,900,702,720 | **negative** — unreachable |

Machine state: free 3,847,880,704 B at `allocSlots`, 30 MoE layers. Leftover = free − requirement;
the 384 MiB margin is what the cap additionally reserves, so a row is grantable when
requirement + 402,653,184 ≤ free. The 8/16 rows' leftovers are confirmed by measurement (the A1
instrument read 16 slots' consumption to the byte); 30/33/34/38 are computed from the same form that
matched at 16, 30 and 34.

At the default of 8 the cache is **inert** — the routed set exactly fills it and nothing survives to
the next token.

Pre-registered prediction for 30 slots, to test the curve rather than assume it: **~74–78% hit
rate, ~15.0–15.8 tok/s**.

### B. Enforcement gaps — things that exist but aren't composed into a decision

**A3 · Make the launch OOM say what it is** — `linux`, **DONE `e42e83e`**

The one failure that has resisted a day of investigation also produces the least informative
message: a raw `cuLaunchKernel: CUDA_ERROR_OUT_OF_MEMORY` with nothing tying it to the cache
setting the user chose. The decline floor added in `a15a394` does not catch it — that fires below
`topK`, and this dies at 34 slots with `topK` of 8.

Two changes, both error handling rather than prediction, useful whatever A1 turns out to be:

1. **Name the kernel in the launch error.** One line. As a side effect it collapses much of A1's
   candidate space — a router or eviction kernel failing means something very different from the
   main expert GEMV failing, and right now the message does not distinguish them.
2. **Catch the launch OOM on the resident MoE path and reframe it**: name the configured slot
   count, say it is the likely cause, suggest lowering it. That converts a fatal driver error into
   an actionable decline — the pattern applied everywhere else this fortnight and conspicuously
   absent exactly where a day was spent.

Deliberately its own item rather than folded into A1, so the next investigation starts from a
message rather than a symptom.

**Landing shape, fixed.** Shipped error text is: kernel name, **requested** slot count, **effective**
slot count after capping, cause, remedy. Naming only the effective count sends a user who set 48 to
lower it to 40 — which caps to the same value and fails identically, making the advice look wrong.
**No VRAM readings in user-facing error text**: those are instrumentation, they move with the probe,
and a number whose meaning depends on where it was taken does not belong in a message the reader
cannot situate. `pipeName` must be **total** — no panic on any shape, including nil, unexported, and
a pipeline not held in a named field; it runs only when something has already failed, so it must not
be able to turn a diagnosable error into a crash. staticcheck ST1005: lowercase, unpunctuated.

Split from the A1 VRAM instrumentation before committing — the two are currently interleaved in
`cuda/resident.go` and `cuda/backend.go`, and one ships while the other does not.

**A4 · Do the two cap implementations differ?** — `linux`, **CLOSED, refuted**

Both copies apply the 384 MiB margin; they do not disagree. The benchmark's cap of 38 came from a
machine state with more free VRAM, not from a second implementation computing a different answer.

This mattered because a numeric disagreement between the copies would have accounted for a claimed
cap of 38 against an observed 34 with **no unaccounted VRAM at all** — it had to be excluded before
any accounting branch was believed. That they agree numerically does not make the duplication
harmless: see A5, and the `capSlots` row under sibling drift in `parity-coverage-policy.md`.

**A7 · Confirm the corrected cap by run** — `linux`, **DONE 2026-08-12, every figure as predicted**

Pre-registered before the run: free after `allocSlots` at 33 slots is 450,494,464 B exactly, and the
forward succeeds. Both held.

| reading (real 26B, 33 slots) | measured | predicted |
|---|---|---|
| free before `allocSlots` | 3,847,880,704 | 3,847,880,704 |
| free after `allocSlots` | **450,494,464** | **450,494,464** (exact) |
| free at first launch | 450,494,464 | — |
| free before the last launch | 312,082,432 | — |
| tokens generated | **4** | > 0 (34 slots gives 0) |

**The cross-validation is the valuable part.** The decrement from first launch to last launch is
**138,412,032 B — exactly the `moe_route` residual pinned from the synthetic fresh-context harness**,
reproduced here on a completely different path: real model, real expert cache, real decode. Two
independent routes to the same byte figure.

So the corrected cap of **33 is confirmed by run**, not only by formula, and A2 can publish it.

What this run does *not* do is narrow the demand: 33 passes with 450,494,464 free and 34 fails with
198,836,224, which brackets the 289,013,760 threshold without tightening it. The balloon search is
still the only measurement of the demand itself.

Note the margin reading has changed. The concern written here in advance was that 33 clears the
*margin* by only 47,841,280 B — but the margin is not the binding quantity; the **demand** is, and 33
clears that by 161,480,704 B. The old framing would have called 33 marginal when it is not.

**A8 · Is `fRoute` the first launch?** — `linux`, **CLOSED**

`fRoute` is **not** the first launch of the token — `ropeKV` (from `gmod`) and `fAttn` (from the
glue module) precede it. But it is plausibly the first launch out of **`moePTX`**, so a lazily
deferred module load is attributable to it exactly as a first-launch would be. A9 stands.

**A5 · The corrected cap must be a SEARCH, not a division** — `linux`, **DONE `6091e7a`**

When A1's fix lands, do not write it as a division with a correction term. Per-buffer 2 MiB
rounding makes the requirement a **step function of slot count**, and

    fit := int((free - marginBytes) / nLayers / perLayer)

cannot invert a step function — it is wrong precisely at the boundaries the failure lives on. A
division plus a fudge term reproduces the class A5 exists to close.

The requirement is monotone non-decreasing in n, so binary-search it:

    largest n such that
      nLayers × Σᵢ ceil(n·pᵢ / 2 MiB) × 2 MiB + marginBytes ≤ free

where pᵢ are the per-slot per-buffer byte strides (4 buffers per MoE layer).

Land it in ONE implementation — `allocSlots` calls `capSlots`, not a copy — with the gate pointed
at the shipping path and a mutation check. And correct, in the same change, wherever it was written
down that `cuda/slotcap_test.go` corroborates production sizing: it corroborates a parallel copy.

**Landed and verified on hardware through the shipping auto-cap path**, no manual slot count:
requesting 128 on the real 26B logs `capping to 33` and generates 4 tokens, where it previously
logged `capping to 34` and generated 0. Free after `allocSlots` is 450,494,464 B — byte-identical to
A7's manually-set 33-slot run, so the search lands exactly where the measurement said.

Two withdrawn claims went with it. `cuda/slotcap_test.go` said it "corroborates the sizing" (it
corroborated the *copying*; the answer was wrong) and that the agreement placed the discrepancy
"downstream of the sizing decision" (the sizing decision **was** the defect). The gate now asserts
properties rather than a remembered number: search ≠ division at the 26B configuration, monotonicity
in n, the 4-quanta step at 33→34, and that no returned count leaves n+1 also fitting.

**README retraction, done in the same change — revert-includes-claims.** The README said the slot
count was "a manual workaround for a safety net that is not holding". That was accurate and stopped
being accurate the moment this landed. It is not deleted: it now records what the old behaviour was,
names both costs the cap was missing with their measured figures, and gives a reader who hit the
failure a way to tell whether their build has the fix (`capping to 33` has it, `34` does not).

**Mutation deltas, re-derived.** The margin mutation was first documented as 38, carried over from
the raw-sum derivation without re-deriving under the rounding form; the gate caught it. The real
answer is **37** — 33 → 37, not 34 → 38. Same delta, different endpoints, and only one is real.

**A9 · The deferred fixed cost paid after the cap is computed** — `linux`, **CLOSED — cause
established, after one reopen**

Ran 2026-08-12, **before A5 landed**, so no cap override was needed and 34 was reachable.

**Answer: `moe_route`'s first launch demands 289,013,760 B (275.6 MiB) of free VRAM, and retains
138,412,032 B (132.0 MiB) of it.**

| quantity | bytes | MiB |
|---|---|---|
| highest observed FAIL | 286,916,608 | 273.6 |
| **lowest observed PASS (the demand)** | **289,013,760** | **275.6** |
| residual after a successful launch | 138,412,032 | 132.0 |

Peak demand is **2.09× the residual**, and the ~150 MiB difference is transient and **still
unnamed** — the one loose end this item leaves.

**It closed once too early.** The first answer was the 132 MiB residual, taken as the cause. It is
not: free before the failing launch was 198,836,224 B, which *exceeds* 138,412,032 by 60,424,192 B.
The reservation fit and the launch failed anyway. The tell was in the numbers already recorded —
free after the failure (265,945,088) was 67,108,864 B **above** the pre-attempt level, and an unwind
cannot return more than was taken, so something that pre-existed the attempt had been released. The
demand was measured directly rather than inferred: balloon the device to a chosen free level, launch
`moe_route`, bisect, one fresh context per trial.

**The arithmetic closes now:**

- 26B at 34 slots: 198,836,224 free against 275.6 MiB demand — short by **90,177,536 B**. Fails.
- post-trim 265,945,088 — still short by **23,068,672 B**, which is why trimming did not save it.
- 26B at 33 slots: 450,494,464 free — clear by **161,480,704 B**. (A7 still runs; see below.)

**Capacity, not contiguity.** Three identical repeats only exclude run-to-run noise, since a
deterministic balloon produces a deterministic layout. The discriminating control is balloon *shape*:
re-run filling with many 2 MiB blocks instead of a few large ones — same free bytes, very different
arrangement — and the threshold is **identical to the byte**. (Contiguity was refuted earlier in this
campaign against a different observation; that refutation was about slot buffers and did not carry
here, so it was re-tested rather than reused.)

**Enumerated, not sampled** (`TestKernelLocalMemoryCensus`): `LOCAL_SIZE_BYTES` for all **37 entry
points across all 12 embedded modules**. Three declare per-thread local memory — `moe_route` at
**4416** B/thread, `rope_kv` and `rope_kv_batched` at 32 each. Two kernels was a sample and the
sibling-drift class is about exactly that.

**All three figures are pinned**, not asserted non-zero — a gate that only checks "a reservation
exists" lets a future `MOE_MAX_E` change double a hidden cost while staying green, which is the
exercised-but-never-triggered shape inside the gate written for this finding. Mutation-checked: each
pin perturbed by one byte, each fails red.

**A9-FIX · The fix is ORDERING, not a bigger margin** — `linux`, **DONE `0103b49`**

Adding 275.6 MiB (or 132, or any measured figure) to `slotMarginBytes` is the correction-term mistake
relocated from A5 to A9. It buries a named consumer inside an unnamed constant, and the next deferred
cost — a new kernel with per-thread scratch, a driver that reserves differently — reopens it silently.

**Structural fix: pay the deferred reservation BEFORE taking the free reading that sizes the cache.**
Force the flagged kernels to launch once during `BuildResident`, ahead of `allocSlots` at
`cuda/backend.go:1115`. The cap is then correct **by construction**, because the free reading it is
computed from already includes every fixed cost that will ever be paid.

**Known-unbounded, recorded rather than filed as a defect.** Module load is **resident-zero and
transient-nonzero**: `CompileLibrary` + `NewComputePipeline` cost 0 B by both instruments at 7.6 GB
free, but `cuModuleLoadData` *fails* with `CUDA_ERROR_OUT_OF_MEMORY` under memory pressure — measured
incidentally when a test's module setup was placed behind a balloon. The transient is unbounded and
unmeasured. A9-FIX's ordering argument is unaffected, because forcing happens while ~3.8 GB is free;
what nothing currently prevents is a *later* module load paying that transient under pressure.

**On iterating the census rather than naming `moe_route` — how this actually landed, and why.**
Mechanical iteration turned out not to be available: forcing a reservation requires *launching*, and
launching requires valid arguments, which is per-kernel knowledge. A zero-block launch would have
avoided that (no thread runs, so no argument is dereferenced) and was tested — **rejected by aikit's
geometry validation**, `invalid launch geometry (grid 0x1x1)`.

So the fix forces `moe_route` **by name**, which is sound for a reason that is itself measured: the
backing store is **shared and sized by the largest kernel**, so forcing the maximum forces the pool
for every kernel. The naming is kept honest by moving the assumption into a check —
`TestKernelLocalMemoryCensus` enumerates every entry point in every embedded module and **fails,
naming `cuda/backend.go`**, if any kernel declares more per-thread scratch than `moe_route`. That is
the enumerate-the-members remedy applied where enumeration cannot be mechanical: the *selection* is
checked even though the *launch* is hand-written.

**Measured result.** Cap moves 33 → 31 and the decrement from first launch to last launch goes from
138,412,032 B to **0** — nothing is taken after the sizing decision, which is the whole property. Free
after `allocSlots` rises to 501,415,936 B; 4 tokens generate as before. **The trade is two slots**, and
that is the point: the margin no longer silently absorbs 132 MiB it was never sized for, so 384 MiB
now means 384 MiB.

**A9-MARGIN · Re-derive `slotMarginBytes` now that it covers only what it names** — `linux`,
**ANSWERED by A10: do not lower it. 151,191,552 B of the margin is a driver floor, not slack.**

Three runs on the real 26B with A9-FIX in place, varying only the margin:

| margin | cap granted | outcome | leftover after `allocSlots` |
|---|---|---|---|
| 384 MiB (shipped) | 31 | 4 tokens | 501,415,936 |
| 128 MiB | **33** | **4 tokens** | 312,672,256 |
| 32 MiB | 34 | **allocation FAILS**, declines to CPU | — |

So the two slots A9-FIX cost **are recoverable** at a 128 MiB margin, on this card, at this free
level. **But do not take that as the recommendation**, because the reason 34 fails is not the one the
margin models — see the next item. The margin is currently doing a job nobody specified, and lowering
it to 128 MiB would work here for a reason that is not understood. Measure the *servability*
constraint first.

What this run did establish: with the reservation paid up front, the decrement after `allocSlots` is
**0 at every cap tested**, so post-sizing consumption is genuinely nil and the margin is not covering
launch growth. Its remaining job is whatever the next item names.

**A10 · The ~150 MiB driver allocation floor** — `linux`, **CLOSED 2026-08-12 — fully decomposed,
nothing unattributed.** `44,236,800 B` once per device + `106,954,752 B` per context =
`151,191,552 B`, exactly. See "RESOLVED 2026-08-12" below.

> **The header used to read "THE OPEN CUDA ITEM" and the body has said RESOLVED since 2026-08-12.**
> Corrected 2026-08-31. Same defect as D3b's: the resolution was appended and the verdict line was
> left, so the item read as open to anyone who did not scroll. **D3b, which this entry said it
> blocked, shipped on 2026-08-20** (`8f3c5e7`) — so the dependency below was already discharged
> twice over by the time anyone read it as live.

Raising the expert-cache default above `topK` requires `cuda/backend.go`'s stated precondition,
"fixing the margin FIRST" — read as *derived rather than asserted*. **A10 is 151,191,552 B of the
402,653,184 B margin, 37% of it, unattributed**, and no derivation can be written while more than a
third of what the margin covers is unexplained. So A10 is on D3b's critical path.
**(Historical — both are closed: A10 was decomposed 2026-08-12 and D3b shipped 2026-08-20.)**

**FIRST DISCRIMINATOR RUN 2026-08-12 — the floor is NOT local-memory backing.**

Launched `shared_gate_combine` — zero declared per-thread scratch — alone, from a freshly loaded
module, in the balloon harness:

    allocation floor                151,191,552
    zero-local kernel LAUNCHES at   153,288,704   (one 2 MiB quantum above the floor)
    moe_route demand                289,013,760   = floor + 137,822,208
    moe_route residual              138,412,032   (differs by 589,824, the A9-RESID drift)

**The bracket check refused to search**, because the low end *succeeded* — which is the answer. A
kernel with no local memory launches at the lowest free the balloon can reach, so **the floor is not
a launch demand at all**: it is the allocator's own limit, and `moe_route`'s 289 MiB is
`floor + its backing store`. Local-memory backing is fully accounted for by the residual, and the
floor sits underneath everything, kernel-independent.

**Candidate space halved without proposing a mechanism.** Ruled out: anything scaling with a kernel's
declared local memory. Remaining: a per-context or per-device allocator reserve.

**MODEL TESTED, 2026-08-12 — the reporting gap is CONFIRMED, exactly.**

> `cuMemGetInfo` reports **151,191,552 B more free than is allocatable, by anyone**.
> `usable = reported_free − 151,191,552`.

Cross-checked **with no launch and no bisection**: allocate directly in a fresh context until even a
1 MiB request fails, and compare what was obtained against what was reported at the start.

    reported free at start   7,665,287,168
    total obtained           7,514,095,616
    SHORTFALL                  151,191,552   ← the floor, to the byte

No kernel involved, so this is not about launches at all. It also reproduces the 34-slot failure
directly: usable at 198,836,224 reported is **47,644,672**, against `moe_route`'s 137,822,208 backing
store — short by 90 MiB, which is the figure A9 measured.

**The CONTEXT discriminator did not resolve, and is blocked by the API.** With the child holding
everything down to the floor, the parent process **cannot create a context at all** —
`cuDevicePrimaryCtxRetain: CUDA_ERROR_OUT_OF_MEMORY` at 151,191,552 B reported free. That is a real
finding (the floor is not available for context setup either) but it means the arm cannot measure
what it was built for. The in-process arm is blocked too: gocudrv exposes only primary-context
retain, not `cuCtxCreate`.

**RESOLVED 2026-08-12 — the reserve is PER-CONTEXT, and the derivation survives anyway because
goinfer has exactly one.**

The first attempt failed because the child drained to the floor and left the parent nothing for
context setup. Giving it room — child balloons to ~300 MiB reported free and holds, parent reads free
via `nvidia-smi` (no context needed), retains the primary context, reads again:

    child holds, reporting          312,672,256 B free
    parent BEFORE its context       313,524,224 B free
    parent AFTER  its context       205,717,504 B free
    DELTA — what a 2nd context cost 107,806,720 B  (102.8 MiB)

**Per-process: a second context pays again.** So `slotMarginBytes` is **not a device constant** in
general — N contexts cost roughly N × this.

**A10 IS NOW FULLY DECOMPOSED — nothing is unattributed.** The gap splits exactly into a
**once-per-device** portion and a **per-context** one, and the split is measured rather than fitted:

| quantity | bytes | MiB |
|---|---|---|
| gap in a single process (its first context) | 151,191,552 | 144.1875 |
| marginal cost, **2nd** context | **106,954,752** | **102.0000** |
| marginal cost, **3rd** context | **106,954,752** | **identical, to the byte** |
| device-once portion (gap − per-context) | **44,236,800** | 42.1875 |

    44,236,800 + 106,954,752 = 151,191,552   EXACT

Two independent additional contexts cost the same 102.00 MiB, so the per-context term is a constant
and the residue is the once-per-device setup. **`usable = reported_free − 44,236,800 − 102 MiB × contexts`**,
which for goinfer's single context is the 151,191,552 already in use.

**A measurement defect was fixed to get here, and it mattered.** The first run reported the marginal
cost as 107,806,720 B — it took `pre` from `nvidia-smi` and `post` from `cuMemGetInfo`, so the delta
silently carried ~832 KiB of *instrument disagreement* as if it were context cost. That is the
851,968 B discrepancy visible in the earlier record. Reading both sides with the same instrument gives
**exactly 102.00 MiB**, and the decomposition only closes with the corrected figure. Same shape as the
measurement-shape class: the number was real and the comparison was not like-for-like.

**Why the derivation holds for goinfer regardless: it creates exactly ONE context, and cannot create
a second.** `cuda/backend.go:586` is the only production call of `CreateSystemDefaultDevice`, aikit
retains the **primary** context (`cuDevicePrimaryCtxRetain`, refcounted per device per process, so
repeat calls do not make a second), and **`cuCtxCreate` is not bound by gocudrv at all**. The
single-context premise is therefore enforced by the dependency, not merely by current usage.

**Original entry follows.**

**A10 (original) · The ~150 MiB driver allocation floor** — `linux`, **THE OPEN CUDA ITEM.** Mechanism measured,
cause unattributed. *(Historical — superseded by the decomposition above.)*

**Status: measured, not explained.** ~150 MiB that `cuMemGetInfo` reports as free and `cuMemAlloc`
will not hand out — 150,601,728 B at `MOE_MAX_E=512`, 151,191,552 B at 256, i.e. **constant across
the one parameter tested**, within the known 589,824 B baseline drift. What it *is* remains
unattributed: driver reserve, allocator bookkeeping, or something else. It is not fragmentation
(refused at any request size down to 1 MiB) and not capacity (free was 2.71× the request).

**Two figures are RETIRED, and why matters.** A9 twice recorded a "~150 MiB transient, still unnamed"
and a "peak is 2.09× the residual" ratio. Both dissolve: **demand = floor + residual**, exact at
`MOE_MAX_E=256` and off by exactly the baseline drift at 512. So the transient was never transient —
it is this floor — and the 2.09× was never a property of anything, being `(floor + residual)/residual`,
which reads **3.77×** at 256 for the same system. A ratio between two quantities that scale
differently is not a constant of the thing.

The original finding follows.

**Original: the total fitting does not imply the allocations succeed.**

**The mechanism.** `cuMemGetInfo` reports **151,191,552 B (144.2 MiB) more free than `cuMemAlloc`
will hand out** — at *any* request size down to 1 MiB. Measured directly by draining the device in
shrinking blocks (`TestAllocFloor`, seconds, no model):

    1024 MiB blocks exhausted -> free 1,222,836,224
     ...
      32 MiB blocks exhausted -> free   182,648,832
      16 MiB blocks exhausted -> free   165,871,616
       8 MiB blocks exhausted -> free   157,483,008
       4 MiB blocks exhausted -> free   153,288,704
       2 MiB blocks exhausted -> free   151,191,552
       1 MiB blocks exhausted -> free   151,191,552   <- FLOOR

**The ladder reproduces both 26B failures exactly.** The 32 MiB rung exhausts at **182,648,832** —
the precise free figure at which the group-by-group order refused a 67 MB request. The 4–8 MiB rungs
bracket 155,385,856, where largest-first refused a 4.2 MB one. Nothing about either failure is
mysterious once the floor is named.

**The rule, and it fits every observation:** *leftover after `allocSlots` must exceed the floor.*

| cap | leftover | vs floor 151,191,552 | outcome |
|---|---|---|---|
| 31 (shipped) | 501,415,936 | clear | works |
| 33 | 312,672,256 | clear | works |
| 34 | 61,014,016 | **below** | fails mid-allocation |

**This retires a figure A9-MARGIN nearly recommended.** A 128 MiB margin is **134,217,728 — below the
floor**. The cap-33 run under it worked only because *that cap's leftover* happened to be 312 MiB;
the margin itself was unsafe. That was luck, and it is now something a test can see:
`TestAllocFloor` asserts `slotMarginBytes ≥ floor` (shipped 402,653,184, clear by 251,461,632).
Mutation-checked at 128 MiB → red.

**The ordering hypothesis was REFUTED, and the change was kept anyway on different grounds.**
Largest-first was pre-registered to complete at 34 slots. It did not — it failed on a 4,212,736 B
request with 155,385,856 B free, a ratio of **36.88**, which no contiguity account survives. It *is*
kept, because measured on its own merit it drains **27 MiB further** before hitting the floor at zero
cost. The code comment says so rather than carrying the refuted rationale.

**A9-MARGIN is unblocked and its answer is: do not lower it.** The margin's job is now decomposed —
151,191,552 B of it is the unallocatable floor and the remaining 251,461,632 B is genuine headroom.
Any reduction has to stay above the floor, which leaves far less room than the "recover two slots"
framing suggested.

**A9-RESID · The 589,824 B is baseline variance, not reservation variance** — `linux`, **CLOSED —
BUT THE MECHANISM WAS WRONG; see A11 (2026-08-12)**

> The verdict "baseline variance" was harmless in effect and wrong in cause. 589,824 B is the amount
> by which `demand = floor + residual` failed to close at `MOE_MAX_E=512`, and the demand now measures
> the closed form to the byte with both components unchanged. It was a **failure to close**, read as
> drift. Merged into A11, which is the resolved record.

The launch-configuration branch is **refuted**. The reservation is **138,412,032 B at every
configuration tested** — nE ∈ {1, 8, 128, 512}, k ∈ {1, 2, 8}, a 512× span in nE — which is what a
compile-time property should do, and confirms the driver sizes the backing store from the kernel's
declared footprint rather than from anything passed at launch.

So the 576 KiB is the other branch: **the pre-launch free-VRAM baseline itself moves**. It reproduced
directly — the same harness reported free before the first `moe_route` as 7,662,600,192 in one build
and 7,663,190,016 in another, **a difference of exactly 589,824 B**, with the reservation identical in
both.

**Caveat worth carrying.** Every figure in A1/A2/A5/A7 is anchored to a pre-`allocSlots` free of
3,847,880,704 B, and that anchor is now known to drift by ~576 KiB. That is well under the 2,097,152 B
quantum, so it can only change a cap decision when a requirement lands within 576 KiB of the
free-minus-margin boundary — not the case at any figure recorded here, but it is why the cap should
never be quoted as a property of the card alone.

**Why the ordering fix is better than a margin bump, stated so nobody later "simplifies" it into one.**
Peak demand is 289,013,760 and residual is 138,412,032 — a ratio of **2.09×**. Forcing early pays the
275.6 MiB *peak* while ~3.8 GB is still free, and the free reading taken afterwards then sees only the
132 MiB *residual*. **One reordering covers both quantities, and neither has to be represented by a
constant.** A margin bump would have to be sized against the peak, permanently, while the peak is
transient — it would reserve 275.6 MiB forever to cover something that is only briefly needed.

**A9-SPEC · Specialize `MOE_MAX_E` at JIT time** — `linux`, **CLOSED — not worth doing, on
measurement.**

**Measured basis for closing.** The allocation floor is **150,601,728 B at `MOE_MAX_E=512` and
151,191,552 B at 256** — invariant within 589,824 B, which is exactly the A9-RESID baseline drift. At
256 the floor is already **74% of total demand** (151,191,552 of 205,717,504). Driving the residual
all the way to zero would still leave the floor, so the reclaim is **bounded near one slot** — and
the measured 512→256 step already buys exactly one (cap 31 → 32).

Against that: a second specialized module, selection logic keyed on expert count at load, and a
dependency on the pinned 12.6.85 NVRTC path for every future rebuild. **Closed on the numbers, not
on preference.** Reopen only if the floor changes or a device shows a materially different ratio.

**The frozen-artifact decision is not part of this item.** The standing constraint is that frozen
artifacts are not regenerated and new kernels get their own files — so the shape is a **second,
specialized module alongside `moe.ptx`, selected by expert count at load**, never a rebuild of the
audited artifact. Recorded so it is not re-litigated as a freeze exception.

**Measured, 2026-08-12** — `moe.cu` compiled at `MOE_MAX_E=256` to a scratch PTX through the pinned
12.6.85 NVRTC (`cuda/testdata/moe.ptx` untouched), then read in the balloon harness:

| | MOE_MAX_E=512 (shipped) | MOE_MAX_E=256 | saved |
|---|---|---|---|
| residual reservation | 138,412,032 | **54,525,952** | 83,886,080 |
| launch demand | 289,013,760 | **205,717,504** | 83,296,256 |
| ratio (residual) | — | **0.394** | not 0.5 |

**0.394, not a halving** — which settles A9-MULT by measurement rather than by refuting a derivation.

**And it names the transient.** The two reductions are nearly identical, because
**demand = allocation floor + residual**:

    MOE_MAX_E=256:  151,191,552 + 54,525,952 = 205,717,504  — EXACT
    MOE_MAX_E=512:  151,191,552 + 138,412,032 = 289,603,584  — measured 289,013,760, off by 589,824,
                    which is exactly the baseline drift A9-RESID measured

So the "~150 MiB transient, still unnamed" that A9 recorded twice **is the A10 allocation floor**, and
the "peak is 2.09× the residual" ratio was never a property of anything — it is
`(floor + residual) / residual`, and at 256 it reads 3.77×. Both figures are retired.

**What the reclaim actually buys: ONE slot.** With A9-FIX the residual is charged before sizing, so
free before `allocSlots` rises from 3,709,468,672 to 3,793,354,752 — and the cap moves **31 → 32**.
83.9 MB is *less than one slot* (30 layers × 3,345,408 = 100,362,240 raw), so this is a boundary
effect, not a proportional win. Worth knowing before anyone budgets a second module for it.

**Not extrapolable to 128**, and now moot: 0.394 at one halving does not predict the next (the
derivation A9-MULT withdrew), and the floor caps the payoff regardless. The harness keeps
`GOINFER_MOE_PTX_FILE`, so re-measuring is a two-minute job if the basis for closing ever changes.

**A9-MULT · The halving was DERIVED and is now withdrawn** — `linux`, **CLOSED, refuted**

"`MOE_MAX_E` 256 → 512 doubled the cost from ~66 to 132 MiB" assumed the backing store is linear in
per-thread bytes with a constant occupancy multiplier. Checked: `moe_route` declares **4416**
B/thread (not the 4096 that "two `float[512]`" implies), and 4416 × 40 SMs × 1024 threads/SM =
180,879,360 B ≠ the measured 138,412,032 (**ratio 0.7652**). The occupancy factor is **not**
`SMs × maxThreadsPerSM`, so proportionality in local-bytes is unverified and the halving does not
follow.

**A second, independent reason.** The residual is **exactly 66 quanta** — 138,412,032 / 2,097,152 = 66
— so it passes through the same 2 MiB rounding A1 closed on. A quantity that is both occupancy-scaled
by an unknown factor *and* quantum-rounded cannot be halved by halving its input, even if the
occupancy factor were linear. Two independent reasons, which is why A9-SPEC's reclaim has to be
**measured** rather than predicted.

Withdrawn rather than restated. The replacement is a measurement at a lower `MOE_MAX_E`, which
A9-SPEC needs anyway.

**The named mechanism was wrong.** moePTX's *module* memory is **0 B** — at `CompileLibrary`, at
`NewComputePipeline`, and at the first launch of a module kernel that declares no scratch — with both
instruments agreeing, so it is a real zero and not a blind spot. A9's *shape* (a deferred fixed cost
invisible to the cap) is confirmed; the thing paying it is local memory, not code.

Gated by `TestMoERouteFirstLaunchReservation` (`cuda/moe_route_reservation_test.go`, seconds, no
fixture), which asserts `shared_gate_combine` reserves 0 and `moe_route` reserves more than 0 — so a
future change to `MOE_MAX_E` cannot silently move a 132 MiB fixed cost.

**Price of the router cap, recorded not proposed.** Raising `MOE_MAX_E` 256 → 512 doubled this
reservation from ~66 MiB to 132 MiB. That halving is **derived from the form, not measured**. It is
written down as the VRAM price of the cap, not as an argument to change it.

**The forcing mechanism that did not fire.** `CUDA_MODULE_LOADING=EAGER` was the intended way to pay
the load early. Readings are **byte-identical with and without it**, so it does not engage on this
driver and path — and the 26B run made under it forced nothing. Its null was uninformative, and would
have been read as "module load excluded" had the cheap control not been run. **A forcing mechanism
has to be shown to fire before a null from it means anything.**

**What this leaves for A5 — corrected, because the earlier statement is now checkable and false.**
This entry previously recorded A5 as *necessary but not sufficient*, on the reasoning that the
rounding fix alone would not have prevented the failure. With the demand measured, **it would have**:
the corrected cap picks 33, which leaves 450,494,464 B against a demand of 289,013,760 B. **A5 alone
avoids this failure.**

What A5 does not avoid is the **class**. It works only because `slotMarginBytes` (402,653,184)
happens to exceed the peak demand (289,013,760) — a relationship **nobody chose, nothing checked, and
`MOE_MAX_E` has already moved once**. That is a stronger reason to keep A9-FIX than the one written
here before, not a weaker one: the fix is not needed to make 33 work, it is needed so that the next
`MOE_MAX_E`, the next kernel with per-thread scratch, or the next driver does not silently reintroduce
a cap whose forward cannot run.

The relationship is now pinned (`slotMarginBytes ≥ measured peak demand`, clear by 113,639,424 B) so
it is at least checked rather than merely true. **`max`, not `Σ`** — launching the whole census gives
a threshold and residual identical to `moe_route` alone, to the byte, so the driver shares one backing
store sized by the largest kernel.

**The regime is part of that claim.** It was measured with the census launched **sequentially in one
context**, which is what goinfer does: batch-1, single stream, one resident model. Under concurrent
residency on separate streams there is no reason the bound stays `max` — and the assertion would then
be **wrong without failing**, the worse of the two ways to be wrong. Recorded next to the claim, and
in the gate's own failure message, the way the measured-quantities rule requires. Concurrent streams
or multi-model residency reopens it.

Historical framing, kept because the trigger was rewritten twice:

**Measured, 2026-08-12 — the pre-launch probe.** Free VRAM read immediately *before* every
`cuLaunchKernel` of the token, at 34 slots:

| reading | value |
|---|---|
| free after `allocSlots` | 198,836,224 (= predicted, to the byte) |
| free before each of the 20 launches, `fRms` … `fRouterF32` | 198,836,224, **constant** |
| free before the failing `fRoute` | **198,836,224** |
| free reported by `describeLaunchErr`, after the failure | 265,945,088 |

So **nothing is consumed between launches**, and the "64 MiB released" was an artifact of probe
position: the block appears only after the failed attempt unwinds. Settled — do not carry it as an
observation.

**Two supersessions this produced.** First, the earlier ~100 MiB threshold was already wrong (the
closed form predicts 198,836,224, so a large reading is the expected case). Second, **the decrement
trigger that replaced it is blind to the thing A9 is about.** It read free after `allocSlots` → free
at first launch → free at failing launch, expecting a deferred module load to appear as a gap. All
gaps came back 0 — but under the driver's default `CUDA_MODULE_LOADING=LAZY`, `moePTX` materialises
*during* the launch that fails, which is after the last pre-launch reading and before the
post-failure one. **No difference of those three readings can contain it.** The zero is the
instrument's blind spot, not a result.

A9 therefore runs on its own merits, and it is the only instrument that can see the cost at all.

Rationale: `fRoute` is the first kernel launched out of `moePTX` (`ropeKV` comes from `gmod`,
`fAttn` from the glue module), so a lazily-deferred module load is attributable to it exactly as a
first-launch would be. The cap is computed from a free-VRAM reading taken **before** that load. That
cost is invisible to before/after readings around `allocSlots`, and invisible to a between-slot-count
delta, because it does not scale with slots.

It is **additive with the rounding shortfall, not an alternative to it**: rounding eats into the
headroom the 384 MiB margin was sized to provide, and the module load then spends from what remains.

**Mechanism, now located precisely.** `CompileLibrary(moePTX)` runs at `cuda/backend.go:714`;
`allocSlots` runs at `cuda/backend.go:1115`. Under lazy loading the module's *device* memory is not
taken at 591 — it is taken at the first launch of one of its kernels, which is `fRoute`, long after
the cap was computed from the free reading at 793. Corroborating: the failed attempt released
exactly 2^26 B while unwinding, which reads as a driver-side code/constant block rather than as
application scratch.

Experiment: force `moePTX` to load while free VRAM is still at its full ~3.8 GB, then re-run at 34
slots. The cheapest forcing is `CUDA_MODULE_LOADING=EAGER` in the environment — driver-level, read at
context creation, and **read-only on the allocation path**, so it changes when the cost is paid
without changing any goinfer code. Branches, pre-registered:

- `fRoute` launches after the forced load → module load was the mechanism; the fix shape is to size
  the cache **after** deferred fixed costs are paid, not before.
- `fRoute` still fails → module load excluded; candidate list reopens one entry shorter.
- the forced load itself fails → same finding, relocated to where it is visible. That is a result.

**Outcome against those branches: none of them, because the forcing mechanism never fired.** The
question was settled instead by measuring each step directly on a fresh context, which needed no
model and took under a second — the mechanism question was never model-dependent, and trying to
answer it inside a five-minute 26B load is what made it look expensive.

Read-only on the allocation path. The reordering is an **experiment first and a fix only after it
answers**.

**Sequencing constraint, honoured.** A9 reproduced at 34 slots and A5 fixes the cap to 33, which
makes 34 unreachable. A9 ran **before A5 landed**, so no override was needed. Recorded because a run
at the new cap would simply pass and look like confirmation, leaving no trace of the loss.

**P1 · KV re-gather and V re-transpose on every decode token** — **LANDED `97f824a`, 2026-08-15**.
Was `decoder/forwardn.go:739` (retargeted 2026-08-24 after later edits shifted the line).

Was estimated ~10–15% of per-token traffic at 4k+ context — the largest single item in the group.

**The aikit-side blocker resolved same-day it was checked.** "It needs a new aikit row-pitch API" was
future tense; aikit `v1.18.0` shipped `linalg.MatmulBTAcc64Strided` (in the tree via the v1.19.0
bump, `fb8e26b`) — its own test is titled `TestMatmulBTAcc64Strided_bitIdenticalToPacked` and its
doc comment says *"(P1 step 3)"*, built against goinfer's exact KV cache layout
(`[nKeys, kvDim=nKV·hd]`) and both stride shapes `attendBatchedHeads` needed:

- **K re-copy (QKᵀ):** strided rows, contiguous elements — `bOff=kvh·hd, bRowStride=kvDim,
  bElemStride=1` reads `keys` in place, no `kh` scratch.
- **V re-transpose (scores·V):** contiguous rows, strided elements — `bOff=kvh·hd, bRowStride=1,
  bElemStride=kvDim` reads `vals` "as if transposed", no `vt` scratch.

**The freeze itself did not block this** — Francis's 2026-08-12 re-declaration (no version gate, a
goldens run with printed composition instead) meant landing didn't wait for a v1.0 unfreeze event.
Verified: `decoder/attend_strided_test.go` transcribes the old gather math and checks the new strided
calls byte-for-byte across 6 shapes (mutation-checked — a swapped stride flips it red); the named
end-to-end gates (`TestForwardN_matchesSequential` argmax-exact/cosine 1.000000/max-diff 0.00e+00,
`TestSpeculativeGreedyParity`) stayed green; the goldens-proof requirement ran clean — 22 passed,
0 failed, 19 f32 + 3 quantized, zero drift across dense/MoE/MLA.

**Measured, not left as an estimate — and the honest number is FLAT at this model size.**
`BenchmarkDecodeAtDepth` (new), qwen2.5-coder-0.5b int8int8, depth 2048, M1 Pro: before 9.049 tok/s,
after 9.078 tok/s (+0.3%) — a single unreplicated before/after pair, not the full interleaved
discipline this project's decode A/Bs otherwise use, and not distinguishable from the ~5%
session-level drift measured on a similar benchmark previously. **Correctness does not depend on
this number and stands regardless; the throughput claim does, and is not being made at 0.5B.** A
proper interleaved multi-visit A/B, ideally at a larger model where the estimate's own "rising with
model size and context" should make the effect more visible, is open follow-up work.

*Unrelated finding surfaced along the way, confirmed NOT caused by this change (isolated worktree
bisect against `6a8a54f`, pre-dating both this and the aikit bump):* `TestDecodeParityInt4` diverges
from its recorded golden on the real qwen2.5-coder-0.5b int4 checkpoint. Pre-existing on `main`,
needs its own investigation — filed as its own item, not blocking P1. **Investigated and closed
2026-08-15 (`8f63a7d`): bisected to `7deb368` (2026-06-14, aikit 1.7.3→1.8.1's W4A8 in-register
scale-fold), a STALE GOLDEN rather than a defect — the new kernel matches an f32 forward for 11
leading ids where the pinned golden matched 5. The "confirmed not caused by this change" above
holds, and by a wider margin than the two-point check could show.**

**P2 · Scalar `int8→f32` widen on the LM head** — **DONE, landed via the ordinary aikit release
cadence.** aikit `linalg/quant.go:138` (`q8Span`).

**Resolved 2026-08-15.** aikit shipped the exact fix this entry specified — `dequantRowInt8(deq, bq,
1.0)`, the scale-1.0 route below, verbatim — as `2f0c65f perf(linalg): SIMD widen in q8Span — ~2×
faster q8 LM head, bit-identical (P2)`, first in aikit `v1.18.0`. goinfer bumped `v1.17.1 → v1.19.0` (`fb8e26b`, goldens-refreshed at
`88ac2cd`), picking it up. **This entry's own analysis is why the
shipped fix is correct**, not incidentally: it worked out that the naive substitution
(`DequantizeRowsInt8Into` with the real scale) is a *silent* numerics change, and that passing `scale
= 1.0` is the one substitution that is bit-identical by construction — and that is precisely the
route aikit's commit took, asserted with its own bit-identity test
(`TestQ8Span_bitIdenticalToScalarWiden`, serial/parallel/SIMD-tail/prefill, every output float
compared by raw bits).

**goinfer-side proof, not just trust in the upstream claim:** the bump's `deps_hash` refresh ran the
forward goldens on **both** architectures — 19 f32 argmax+cosine goldens green on arm64 (the
FMA-contracting arch, where a reassociation would show first) and the prior box run on amd64 — no
argmax/cosine breach anywhere. Consistent with, but independent confirmation of, aikit's own
byte-comparison.

**The "leave it UNRELEASED, planned for v1.0" framing below is superseded by events, not reversed.**
It predates aikit actually resuming ordinary releases (v1.17.0 onward) and goinfer bumping through
them in the normal course — exactly what E6 (closed 2026-08-12) already settled: a release needs a
reason a consumer can receive, and this one had two (P2 itself, plus the bit-identical encoder GELU
fix riding the same bump). No decision reversed; the precondition just arrived sooner than the note
assumed.

**Still owed — the goinfer-side magnitude, not the aikit-side one.** aikit's own perfgate measured
**Δ ≈ −50% on the widen** (its own microbenchmark, LM-head shapes, M1 Pro) — real, but internal to
`q8Span`. What is NOT yet measured: goinfer's own end-to-end decode/LM-head tok/s delta from the bump
(P9/P10-style A/B, box + Mac). That is the number worth banking before the win goes in a release note.

**Original analysis, kept — it is the record of why the fix had to be exactly this shape:**

**The bit-identity condition, checked in source rather than assumed.** It splits in two, and the
half that matters is the one the original wording did not cover:

- *The widen kernel itself is exact.* `dequantI8AVX2` (VPMOVSXBD → VCVTDQ2PS → VMULPS) and
  `dequantI8NEON` (SXTL/SXTL2 → SCVTF → FMUL) both compute `float32(q[i]) * scale` elementwise, with
  no reduction and no reassociation freedom. `int8 → float32` is exact for all 256 values.
- **But the shipped call site does not apply the scale per element.** `q8Span` widens *without* the
  scale — `deq[k] = float32(bq[k])` — and applies it **after** the dot:
  `dst[i,j] = dotF32(a_i, deq_j) · s_j`. So the naive substitution changes

      dot(a, widen(q)) · s        one rounding of the scale, at the end
      dot(a, widen(q) · s)        one rounding PER ELEMENT

  which are mathematically equal and **not bit-equal**. Swapping in `DequantizeRowsInt8Into` with the
  real scale is a silent numerics change, exactly the kind that reaches a release looking like a pure
  speedup.

**The route that IS bit-identical: pass `scale = 1.0`.** Multiplication by 1.0 is exact in IEEE-754
for every finite value (and preserves ±0, inf, NaN), so `float32(q[k]) * 1.0` equals `float32(q[k])`
bit for bit on both kernels, and the scale stays where `q8Span` already applies it. Then the
structural argument holds and no parity run is needed.

Mechanics: `dequantRowInt8` is unexported and `DequantizeRowsInt8Into` is the whole-matrix form
taking a per-row `scales` slice, so this needs either an exported per-row entry or a ones-filled
slice. The `len(q) < 8` (amd64) / `< 16` (arm64) and `!hasAVX2` fallbacks all route to
`dequantRowInt8Scalar`, which is the same expression — no additional argument needed there.

**The magnitude is still an ESTIMATE and should be measured before the E6 decision.** "Several
ms/token at large vocab" was a verifier's reading. The package comment measures the same widen at
~190 ms per CodeRankEmbed forward for 113 M elements (~1.7 ns/element), and an LM head at Gemma's
262,144 × 2,560 would be 671 M elements — two orders of magnitude larger, which suggests the LM head
does **not** go through `q8Span` on the paths that matter. Establish which path the LM head actually
takes before quoting a number, and measure it there.

**P3 · Gemma final-logit softcap, serial O(vocab) `tanh` on the sampling path** — **DONE `4c26a58`**

Measured rather than estimated: the loop costs **1.43 ms/sampled token** at Gemma's 262,144 vocab and
**640 µs** parallelised — a **2.3×** on the loop, saving ~0.85 ms/token.

**The 10–30% estimate needed qualifying, not correcting.** 0.85 ms is ~28% of a 3 ms decode step and
**under 1% of the 26B's ~80 ms**. The share depends entirely on which model you run, so the loop
figure is what is recorded — it is the part that does not.

Greedy decoding does not pay it at all (`ForwardArgmax` reduces on-device and reads back 4 B), which
confirms the audit's "sampling only".

The threshold is measured, and the small end is a **loss**: 8,192 elements parallelise at 0.95×.
Hence `softcapParallelMin = 32768` rather than an unconditional fan-out.

Bit-identity is **structural** — each output element is a pure function of the input at the same
index, so there is no accumulation order to perturb. Gated at exact equality across sizes straddling
the threshold and GOMAXPROCS ∈ {1, 3, 16}, with lengths that do not divide evenly.

**Two of five siblings fixed** — see B6. The other three are frozen or on hold.

**P4 · Metal RoPE dispatched twice per layer — DONE, MEASURED NET-ZERO. Do not re-queue as a win.**

Grid-merge (2→1 dispatch/layer) is bit-identical and already implemented on branch `metal-rope-merge`
(`d682315`; snapshot-golden byte-exact) — **but that branch is not on origin and the commit resolves
in no clone here, so this claim is unverifiable from any machine but the mac. Push it or restate the
claim.** The audit re-surfaced this as "estimated a few %" **not knowing
that branch existed** — a measurement that wasn't composed into the queue (the class this file exists to
prevent). Dispatch census (2026-08-12) measured `rope` = 56/token = exactly 2/layer, so the merge
removes 28/token = **8.3% of the 338 dispatches/token**. But re-A/B'd on the current binary
(`TestZZ_metalDepthBench`, qwen2.5-coder-1.5b W4A8, M1 Pro): before 59.7/49.1/28.4/18.4 vs after
61.0/46.5/26.9/18.4 tok/s at 128/512/2048/4000 — **net-zero, within noise**. 8.3% fewer dispatches, 0%
tok/s. Correct and harmless; kept on the branch as a measured record, not merged (no speedup to bank).
See ollama-chase §A2-Metal.

**P5 · Metal `quant_vec` fused into the o-proj GEMV — PREDICTED NET-ZERO (do not build standalone)**

Dispatch census (2026-08-12): exactly **one** `quant_vec` dispatch/layer = 28/token (the o-proj input
quant; the other GEMVs already fuse theirs — so the swiglu half the "~5–6%" estimate worried about is
not a `quant_vec` dispatch and is out of scope). Fusing it removes 28/token = **8.3% of 338** — the
**same magnitude and mechanism as P4** (one small per-layer dispatch), and **P4 measured net-zero**. So
P5 is predicted net-zero by direct analogy; the fusion is more invasive than P4's merge, so it is not
worth a standalone build for a tok/s win. Only reconsider inside a **megakernel collapse** (many
dispatches at once), which is the actual Metal-decode lever (with int4 unpack / bandwidth). If ever
built, A/B it — do not assume the estimate.

**P6 · `moeMLP` allocates ~7–8 MB/token** — **DONE `eea7f29`** (`decoder/mlp.go:92`)

By skipping the `decodeScratch` invariant its dense sibling honours. **See B6.**

**PRICED (2026-08-12) — the freeze is a cost, not a prohibition, and the cost is 6 seconds.**
`decoder/mlp.go` is in the `core` shared set and `decoder/weightmat.go` in `quant`, and **all 23
families use both**, so an exception re-stales the entire matrix. But the sanctioned instrument is
`scripts/refresh_parity_hashes.sh` — the goldens-gated refresh, precedent **`ecc5af2`** (default-off
diagnostic hooks: a core-file change that is non-numeric by construction, refreshed behind the
goldens) — **not**
`scripts/parity_sweep.sh`'s T3 oracle sweep, because these are allocation changes rather than arithmetic.

Measured on `linux-62gb`: **19 goldens pass, 11 skip, 0 fail, 6.09 s wall.** One machine, no model
zoo, no HF venv. (18 of the 23 manifest rows name `linux-62gb`; only `gemma4` names
`macbook-arm64`, and that is its *oracle*, not its golden — `TestGemma4MoE_forwardParity` ran here.)

**Coverage is good for P6.** Nine MoE goldens actually RAN: `TestGemma4MoE_forwardParity`,
`TestGemma4MoEKV_forwardParity`, `TestGemma4MoEUnified_forwardParity`, `TestMixtral_forwardParity`,
`TestGlm4Moe_textParity`, `TestQwen35_forwardParity`, `TestDeepseek_textParity`,
`TestKimi_textParity`, `TestLlama4_textParity`. `TestQwen2Moe_forwardParity` skipped.

**Verdict: P6 can land now under the exception.** Do not refresh the hash without running the
script — it refuses on any golden failure and on a vacuous all-skipped run, which is the whole point.

**P7 · W4A8 allocates a fresh `Workspace` per projection per token** — **DONE `91f359f`**, verified by the int4 goldens

**RESOLVED BY READING THE SIBLING — and the answer is neither branch as posed.** No concurrency
argument is needed; the tree already contains one.

**Concurrent decode streams DO exist**, and W8A8 has **no latent race**. `decodeScratch`'s own doc
settles it: *"One lives on each KVCache — a cache is one generation stream, so the buffers are never
shared concurrently."* The Workspace W8A8 reuses (`ws *linalg.Workspace`, `decoder/scratch.go:45`)
lives inside that per-stream struct, so W8A8's "fix" was never a *shared* Workspace — it is a
**per-stream** one, race-free by the same property that makes every other scratch buffer safe.

**So the per-call Workspace comment is accurate and irrelevant to the fix.** `matmul` — the free
function, for callers with no scratch — keeps a per-call Workspace for W8A8 *and* W4A8 alike
(`decoder/weightmat.go`, the same per-call-Workspace pattern twice). The divergence is elsewhere: `matmulInto`
special-cases `isW8A8(w)` and falls through to `matmul` for everything else, so W4A8 never reaches
the per-stream Workspace even though its six call sites already pass one.

**Verdict: P7 is a straightforward divergence repair.** Route W4A8 through
`linalg.MatmulBTW4A8Into(ws, ...)` in `matmulInto`, exactly as W8A8 does. Race-free by the same
argument, no new one required.

**THE FREEZE IS NOT P7'S BLOCKER, and waiting for the unfreeze is not a route to it.** The goldens'
numeric protection is **f32-only**; P7 is an **int4** path. Lifting `6edd1ca` adds no coverage
whatsoever to W4A8, so P7 would be **just as blocked at v1.0** as it is today. It is blocked on Q1(c)
— authoring int4 goldens — and on nothing else.

**Landed once Q1(c) existed** — `91f359f`. `matmulInto` now dispatches on *"does this weight have an
Into form that takes a Workspace"* rather than on `isW8A8`. All **23 int4 rows pass** across 16
architectures; before `1d0d1ed` nothing in the tree could have told a correct W4A8 change from a
broken one, and the goldens-gated refresh would have gone green either way. That is the whole
argument for Q1(c), demonstrated on its first customer.

**Historical: blocked ONLY by Q1.** The goldens give no numeric proof on this path — every golden that runs is
f32, and W4A8 is precisely the path being changed — so the 6-second refresh would be **vacuous
exactly where it matters**. P7 lands when Q1 gives int4 a golden, or behind a real T3 quant gate.

**See B6** — and note the "sibling" framing was loose in a way the audit did not capture: the pair is
not W8A8-fixed / W4A8-unfixed, it is `matmulInto` covering one quantization and silently delegating
the rest.

**P9 · aikit v1.17.0 cost ~3% of DECODE throughput on this shape — CLOSED 2026-08-12 by aikit
v1.17.1** — `linux`

**Four statements, kept separate on purpose. Collapsing them would make the record claim more than
the work did.**

1. **The A/B measured a decode regression.** v1.16.0 against v1.17.0, interleaved, pre-registered
   2.0% floor: **−2.96%**, above the floor, per-visit medians not overlapping.
2. **aikit v1.17.1 fixed it.** The same instrument, the same floor, re-run with the expectation
   written down first: **+0.43%**, branch 1, **flat**.

   **SCOPE OF THAT "FIXED", added 2026-08-12 — it says less than it appears to.** +0.43% is a
   **benchmark-level** number, and the changed int8 code is only part of what `BenchmarkDecode`
   spends time in. So what was confirmed is that **the benchmark-level regression is gone**, not that
   the changed code is unchanged in cost. A residual effect inside the int8 path smaller than
   `0.43% ÷ (that path's share of decode runtime)` is entirely consistent with this result.

   **THE SHARE IS NOW MEASURED (2026-08-12), and it is small: 6.48%.** `BenchmarkDecode` was
   profiled — `linalg.MatmulBTW8A8Into` is **6.48%** of decode runtime (`w8a8Span` 5.22%,
   `dotI8AVX2` 5.13% flat), on the v1.17.1 build. The changed int8 code is a **sixteenth** of what
   this benchmark spends time in, so benchmark-level figures divide by ~0.065 to become statements
   about that code:

   | benchmark-level | ÷ 6.48% → within the int8 path |
   |---|---|
   | v1.17.0 regression **−2.96%** | **≈ −46%** |
   | v1.17.1 result **+0.43%** (median) | ≈ +6.6% |
   | v1.17.1 bootstrap 95% CI **[−2.52%, +3.73%]** | **[−39%, +58%]** |
   | the 2.0% floor | **≈ 31%** |

   **What that does to statement 2, plainly: the flat verdict is much weaker than it looks.** A
   residual of up to **~31%** inside the int8 path would have been *undetectable* by that A/B, and
   the bootstrap interval on the measured delta spans **−39% to +58%** within the path. So
   "v1.17.1 fixed it" means **the benchmark-level regression is gone** — nothing more. It is *not*
   evidence that the int8 path returned to its v1.16.0 cost, and it never was.

   **A corroboration worth recording, because it is independent.** The v1.17.0 regression converts
   to **≈ −46% within the int8 path**, derived end-to-end from goinfer's benchmark and a profile
   divisor. aikit measured that kernel directly at **+49% slower** in its worst (serial) case. Two
   unrelated methods, ~3 points apart — which raises confidence in the divisor and in the −2.96%
   alike.

   **Caveats on the divisor, stated with it.** Measured **under a profiler** and applied to
   **unprofiled** runs, so it is rough. And only the **v1.17.1** build is profiled — the v1.17.0
   build's share would likely be *larger* (a slower kernel takes a bigger slice), which would make
   the −46% an *over*-estimate; 6.48% is the conservative choice for the *flat* claim, which is
   where it matters most here.

   **And the flat verdict needs its delta and uncertainty printed beside it**, for the same reason:
   a floor is a practical-significance threshold, not a detection limit. "Flat" means *no effect
   exceeding the declared threshold*, never *no difference*. 17 of 36 pairwise comparisons separate, where
   18 is exactly none. `w8a8Span`'s executable body in v1.17.1 is byte-identical to v1.16.0.
3. **The locus was INFERRED, not measured.** "The int8 kernel at M=1" was recorded as inference at
   the time, explicitly labelled, with an ablation named as the thing that would settle it. The
   ablation was never run here. Upstream's revert later confirmed the inference was right — **and
   that does not retroactively make it evidence.** This measurement established a direction and a
   magnitude; it never located a cause. Anyone reading this entry as "the A/B found the int8 kernel"
   has upgraded a guess to a finding, which is the error the labelling existed to prevent.
4. **The upstream report produced a fix in a patch release the same day.** aikit v1.17.1 reverts the
   eight-column span, and its commit message adopts the method — *"interleaved with a pre-registered
   2% floor and warm-up discard — a better methodology than the one that shipped the regression"* —
   and states the mechanism this A/B could not: the two forms walk memory differently, so the
   eight-column kernel wins when B is cache-resident and loses when B is streamed. Both production
   callers stream.

**That last point is the case for the methodology, and it is written down for the next time a
careful A/B looks expensive.** The regression shipped from a real measurement taken at ONE shape.
What caught it was not a better benchmark but a *disciplined* one — interleaved rather than
sequential, floor fixed before the data, warm-up discard defined in advance, and the limits stated
rather than the result rounded. The first attempt at this number, two runs separated in time, was
worthless and would have been reported as −4% had it not been checked. **The extra ~40 minutes of
machine time is the entire reason a regression reached a patch release instead of a user.**

**Session drift makes the point concretely:** this box ran ~0.93–0.97 tok/s during the v1.17.0 A/B
and ~0.98–1.03 during the v1.17.1 one — a **5% shift, larger than either effect under test**. Any
before/after comparison spanning them would have been dominated by whichever session it straddled.

**STILL OWED — DECODE ONLY.** Every number here is decode. `linalg/matmul_blocked.go` is **unchanged
in v1.17.1**, so v1.17.0's f32 blocked-matmul rework is still live and **unmeasured in both
versions**. That path is a prefill shape this instrument barely exercises. **A prefill measurement
gates cutting a goinfer tag that carries this bump** — a release characterizing one phase while
silently carrying an unmeasured change to another is a claim by omission.

Full records: `docs/measurements/aikit-v1.17.0-decode-ab.md` and `-v1.17.1-decode-ab.md`, each
carrying its pre-registration, its raw samples, and its own weaknesses.

<details><summary>The original v1.17.0 finding, as recorded before the fix</summary>

Not a product claim and deliberately not in the CHANGELOG: an engineering finding, recorded with its
method and its limits so it is not lost. The bump (`f33fcaf`) is the **only** compiled-code change
between the arms, so the effect is attributable to it.

**Result: −2.96%**, against a noise floor of **2.0% pre-registered before the comparison ran**.

| arm | median | mean | sd | min | max |
|---|---|---|---|---|---|
| pre (`aikit v1.16.0`) | 0.9662 | 0.9674 | 0.0051 | 0.9621 | 0.9745 |
| post (`aikit v1.17.0`) | 0.9380 | 0.9364 | 0.0220 | 0.8988 | 0.9631 |

**Method**, because the first attempt at this number was confounded and the design is the finding as
much as the figure. `BenchmarkDecode`, DeepSeek-V2-Lite-Chat-Q4_K_M at `Quant: "int8int8"` (W8A8),
`-benchtime 30x`, batch=1 greedy, Ryzen 7 3700X / GOMAXPROCS 16. Both arms in detached worktrees,
`GOWORK=off`, no `replace`, aikit from the module cache. **Interleaved pre/post/pre/post in one
session** — not two runs separated in time, which is what made the first attempt worthless. The
**first sample of every visit is discarded** as warm-up, defined from an independent 8-sample
characterization and applied identically to both arms. 6 retained per arm.

**Why the direction survives the noise.** The post arm's sd is 4× the pre arm's, concentrated
entirely in one visit (per-visit sd 0.0055 then 0.0343), and one post sample reaches into the pre
range. But the **per-visit means do not overlap** — pre {0.9658, 0.9690} against post {0.9351,
0.9378} — and 34 of 36 pairwise comparisons put post below pre. The direction is solid; treat the
**magnitude as ~3% ± a point**, not as 2.96%.

**Locus NOT isolated — this is the open part.** v1.17.0 brings two new kernels onto goinfer's paths
(see `f33fcaf`): a new AVX2 int8 routine behind `w8a8Span`, and a reworked inner loop in the blocked
f32 matmul. This benchmark is int8int8 at M=1, so the int8 kernel is the *likely* locus and the f32
blocked path is barely exercised — **that is inference, not measurement.** An ablation would pin it;
nobody has run one. Do not repeat the inference as a finding.

**Scope.** One model, one quantization, batch=1, one box. It says nothing about prefill, where the
changed f32 blocked path actually lives and where the upstream optimisation is presumably aimed — a
decode regression is consistent with a prefill win. Measure that before concluding the change is bad
rather than mis-shaped for this workload.

*Action:* report upstream with the method above. Not urgent — 3% of decode on one shape, against a
bump whose bit-identity is gated and green.

*(That action was taken. It produced aikit v1.17.1 the same day — see statement 4 above.)*

</details>

**P8 · `sampleChunked` allocates a full-vocab `[]float64` and rebuilds the goroutine pool twice per
sampled token** — `decoder/sampler_chunked.go:188`. **TRIED AND REVERTED — the allocation removal
costs 5–6% throughput.**

**Not frozen** (checked, not assumed): `decoder/sampler_chunked.go` and `decoder/sampler.go` are
absent from `testdata/parity_manifest.json`'s 21 files, so this does not re-stale any family's
`deps_hash`. Sampling is not forward numerics. (`decoder/mlp.go` and `decoder/weightmat.go` — P6 and
P7 — **are** in it, confirming those two are genuinely blocked.)

**The change:** hang the full-vocabulary `exp()` scratch off the `Sampler` and reuse it, rather than
`make([]float64, vocab)` per draw. Safe by the type's existing contract — a `Sampler` holds a
`*rand.Rand` and appends to `history` without a lock, so it was never usable across goroutines.

**Measured, `BenchmarkFilterNew262k`, 5 runs of 400 iterations each:**

| | ns/op median | B/op |
|---|---|---|
| before | 6,344,898 | 2,150,668 |
| after | 6,682,883 | **58,732** |

Allocation drops **97%** and throughput drops **5.3% median / 6.3% min**, with the two distributions
**not overlapping** (before max 6,391,328 < after min 6,649,446). So this is not noise.

**Reverted.** P8 was filed as a jitter reducer, and paying 5–6% of throughput for it is a bad trade
against jitter nobody has measured. **No mechanism proposed** — the obvious guesses (page-fault
behaviour on fresh spans, aliasing or bounds-check effects from a field-derived slice) are exactly
the premature-mechanism shape, and none was tested.

**Two discriminators run, both NEGATIVE.** Medians, `BenchmarkFilterNew262k`, 4 runs × 400 iterations:

| variant | GOGC default | GOGC=off |
|---|---|---|
| baseline (`make` per call) | 6,221,779 | 6,215,477 |
| pooled, inline arg | 6,690,158 | — |
| pooled + hoisted local | 6,585,193 | 6,700,647 |

- **(a) codegen — NOT the cause.** Hoisting the field-derived slice into a local at function entry
  recovers ~1.6 of ~7 points. The gap does not close, so **no systematic grep for the pattern is
  warranted** — that follow-up was conditional on (a) closing it, and it did not.
- **(b) GC — EXCLUDED.** The gap survives `GOGC=off` and is slightly *wider* there (ratio 1.078
  against 1.058).

**Cause unidentified, and no mechanism is proposed.** What remains untested is memory/page behaviour,
which is where the guesses point and precisely why they are not written down as findings.

**What would make this landable:** measure the jitter P8 exists to reduce, so the trade has two
numbers rather than one. Until then the allocation stays.

> **MEASURE-DON'T-ASSUME, inherited by every drafter decision below (2026-08-25).** The
> adaptive-width ship-gates measured something that sits UNDER this whole line of work, not just
> under that campaign: **"structured output is more predictable, so a drafter accepts more of it"
> DID NOT TRANSFER.** On CUDA-resident Qwen3-4B + DFlash, a suite built specifically to force
> prose→structured transitions found acceptance *falling* across the boundary — width and
> commits both dropped (5.3→4.0 @ 2.83 tok/round; 6.5→4.0 @ 2.17) — and **no verify width, and
> no adaptive schedule, beat running NO drafter at all** on that suite (best arm 0.943×). The
> assumption is load-bearing in several places here (front-loading, traffic-class routing, the
> grammar prior, DFlash 2's suffix-decay rationale) and it came from a fork's result on
> different hardware with a different drafter. **Treat it as a hypothesis per pairing, not a
> property of text.** Full evidence: `docs/measurements/adaptive-width-shipgates-2026-08-25.md`.

**P15 · DFlash 2 — P10's successor upstream, aimed at P10's two measured weaknesses** — `linux`,
filed 2026-08-20 from [inco.ai/blog/dflash2](https://inco.ai/blog/dflash2/); **gates before code**.
Upstream adds two modules to the v1 drafter P10 shipped: a two-tap dynamic depthwise convolution
against **suffix decay** (the cause behind our measured front-loaded acceptance — positions 12–15
bought 0.09 tokens for 9.4 ms, which P10 mitigated by capping verify width at 7–8; upstream's v2
block width lands at 7–8, convergent with our sweep) and a **2.0 M-param path-selection head**
(top-16 candidates per position, pairwise bilinear scoring, one serial walk) — the licensed
analogue of the DSpark Markov head whose +41% chat α was the DSpark path's surviving rationale.
Claims, theirs until reproduced: mean acceptance **+21% over v1**, DSpark beaten by 1+ tokens,
MT-Bench 4.10 on the Qwen3.8-27B pair — which, if it transfers to our chat suite, clears the ~3.0
break-even the acceptance guard polices and turns chat from bounded-loss to a win. **If gate 2
confirms, the DSpark exploration path is mooted, license issue included.** Verify stays lossless;
the P10 substrate (capture seam, `attn_block_full`, batched verify, `DFlashContext`, resident
trunk, guard, harness) carries over; claimed overhead is 1.3%/cycle. **Order of work:** (0)
per-repo license audit of `incoai/Qwen3.8-27B-DFlash2` + `Muse-Glimmer-30B-DFlash2` — the post
states none, and the HF list endpoint has already lied to this program once — plus a sweep for a
resident-capable v2 pair (the post's Qwen3.5-4B table implies a small checkpoint exists); (1)
reference-loop acceptance on a released pair, their code first, ours second, agreement required;
(2) tensor dump accounted to the byte — settles v1↔v2 trunk compatibility and where the dynamic
conv kernels come from; (3) wall-clock on whatever venue survives (0) — both released targets are
too big for the 2070S, and the CPU re-run stays pre-registered as a predicted **no** with one term
moved by P14. **Sequencing: lands BEFORE gate 4's width router**, which would otherwise be
calibrated to v1's acceptance curve and rebuilt. Design page:
[`docs/spec/08-dspark-dflash.md` §"DFlash 2"](spec/08-dspark-dflash.md) — context there, this
entry is the claimable work. Training code is not released, so thinking-mode targets have no
re-finetune escape this time; the harness's existing hard-errors stand guard on the two known
input-distribution traps.

> **MEASURE-DON'T-ASSUME, inherited by every drafter decision below (2026-08-25).** The
> adaptive-width ship-gates measured something that sits UNDER this whole line of work, not just
> under that campaign: **"structured output is more predictable, so a drafter accepts more of it"
> DID NOT TRANSFER.** On CUDA-resident Qwen3-4B + DFlash, a suite built specifically to force
> prose→structured transitions found acceptance *falling* across the boundary — width and
> commits both dropped (5.3→4.0 @ 2.83 tok/round; 6.5→4.0 @ 2.17) — and **no verify width, and
> no adaptive schedule, beat running NO drafter at all** on that suite (best arm 0.943×). The
> assumption is load-bearing in several places here (front-loading, traffic-class routing, the
> grammar prior, DFlash 2's suffix-decay rationale) and it came from a fork's result on
> different hardware with a different drafter. **Treat it as a hypothesis per pairing, not a
> property of text.** Full evidence: `docs/measurements/adaptive-width-shipgates-2026-08-25.md`.

**P10 · DSpark / DFlash block drafters — the α lever 05/06 named, arriving pretrained** — `linux`
next (resident CUDA go/no-go), then `mac`, **increments 1–2 DONE 2026-08-15 (`linux-62gb`)**.
Increment 2 landed against **DFlash, not DSpark** — every DSpark drafter checkpoint is unlicensed
(see the correction below), so on Francis's call it pivoted to `z-lab/Qwen3-4B-DFlash-b16` (MIT).
Fixtures + f32 conversion + CONFIRMED shapes are in the design page; **first acceptance signal:
10/15 accepted (11 tok/verify) on a chat-templated code prompt, 0/15 on a bare completion prompt**
— a smoke reading, not kill-gate 2, but 3.7× the ≥3.0 bar. **Increments 3–4 follow DFlash**
unless the DSpark license resolves. **Increment 3: KILL-GATES 1 AND 2 BOTH CLEARED 2026-08-15.**
Gate 1: cosine 1.0 per layer + 15/15 drafted ids vs the reference, mutation-verified.
Gate 2: **6.14 tok/verify on code, 7.32 on math** (int8 target, 160 tokens, bar >=3.0),
with code landing within 4% of the bf16 reference's 6.37. Chat is ~2.2 and roughly a wash,
as upstream also reports. **int8 costs only ~4% of acceptance, so a quantized resident
target does NOT crater this drafter — increment 4's economics survive.**
**PROBE B ANSWERED 2026-08-16 (`linux-62gb`): acceptance is NOT a Qwen3-4B artifact.** The
capture seam was the only thing blocking a second pairing, and wiring it (one shared helper +
a bitwise placement gate, mutation-tested) unblocked three: `qwen3_5_moe`, `gemma4`, `gpt-oss`.
Measured at gate 2's recorded length, code suite, int8, greedy: **Qwen3.6-35B-A3B 6.78 tok/verify
vs Qwen3-4B 6.14** — and the 4B **re-measured to 6.14 exactly**, the control that says the
harness changes are numerically inert. Depth ratios are near-identical (6/40 vs 5/36) and probe A
established depth ratio as the mechanism, so this is theory confirmed, not luck. All four z-lab
drafters now load and run the SHARED trunk at 5/6/8 layers, 5/6/8 taps, three hidden widths and
two block widths. **gpt-oss is blocked on a missing harmony chat template** (not on the seam) —
`chat.Detect` correctly refuses it rather than silently rendering it as Gemma-4, verified against
the real 16.7 KB template. **Two MORE retracted conclusions en route**, both in the COMPARISON
rather than the measurement: a baseline taken from a number the doc had already retired (2.90),
and a matched control at a single length that the next length reversed (the two pairings rank in
opposite orders at maxNew 16 vs 48). Plus a killed-healthy-run caused by an unbuffered `grep`.
The harness now hard-errors on a too-short `maxNew` and on an undetected chat template.

**BIGGEST LEVER, MEASURED 2026-08-16 (`linux-62gb`): cap the VERIFY WIDTH, not the block.**
Gate 3's 1.29×-on-code (under its own bar) verified all 16 drafted positions. It did not have to.
Capping how many the TARGET verifies — drafter untouched, still drafting its full trained block —
measures **code 1.74× at width 7, math 2.25× at 8, chat 0.92× at 4**, against 1.29× / 1.54× /
0.46× at 16. Every k=16 row reproduces its recorded figure, anchoring the sweep. **Gate 3 goes
from "pass on math only" to clearing on code AND math.** Mechanism: positions 12–15 gain **0.09
accepted tokens between them** while costing 9.4 ms of verify per round. **This kills DSpark's
rationale** — the doc projected DSpark 1.75× vs DFlash 1.29× on code and treated that as the
pivot's justification; measured DFlash-at-7 is **1.74×**, so the advantage was block width, not
DSpark, and DFlash is the LICENSED one. Also found: a single fitted α understates narrow-width
acceptance by 7.5–13.4% (acceptance is FRONT-LOADED, not position-independent), and the optimum
width TRACKS acceptance (8/7/4) — so gate 4's router should select WIDTH, not just fire/skip,
from the α̂ signal it already computes. Still a projection: acceptance measured, wall-clock not;
**gate 3 should run per-suite at these widths.**

**P10 SHIPPED 2026-08-17 (`linux-62gb`): block drafting is a production feature.** `serve
--drafter <dir>` — the drafter proposes a block per round, the target verifies it in one batched
pass, output is token-identical to plain greedy. **Gate 3 measured end to end: code 1.44×, math
1.58×** against `Model.Generate` (≥1.3× bar). Built: a non-causal attention kernel
(`attn_block_full`, its own PTX, existing PTX byte-unchanged), the drafter resident on device
(cosine 0.9973 vs the CPU trunk), incremental context K/V (bit-identical 4+4 vs 8), a batched
verify head gated on ARGMAX equality (weaker bar than bit-identity, and what makes batching the
head admissible), batched capture (bit-identical, 1.09 ms/block vs 2.79 per-token), and the
`decoder.BlockDrafterWeights` / `ResidentDrafterHost` interfaces so DFlash and DSpark both work
with no type switch and Metal needs only to implement two interfaces.

**THE MEASUREMENT LESSONS COST MORE THAN THE BUILD.** Three published figures were wrong and each
was corrected by a measurement, not review: (1) gate 3's baseline was a
`PrefillLastNArgmax(M=1)` loop that downloads 608 KB/token where `Generate` uses a 4-byte GPU
argmax — every ratio was 10–35% optimistic; (2) a single timing run reported math at 1.50× when
the steady state is 1.58× (acceptance is deterministic, wall-clock is not — the first run is the
cold tail); (3) chat measured 0.96× against the wrong baseline and **0.61× against the right
one**, which twice reversed a "the router is not mandatory" conclusion. **All three failures
returned CORRECT OUTPUT**, because block drafting is lossless — no correctness test can catch any
of them.

**Two safeguards ship because of that.** A runtime acceptance guard stops drafting below
break-even (~3.0 tok/round), turning chat from 0.61× to **0.92×** and a thinking-mode target from
0.82× to 0.99×; and a startup WARNING when the served template leaves thinking mode on, since
that halves acceptance (5.76 → 3.00 tok/round) and is otherwise completely silent. **Open:** the
residual ~7% on short chat (the guard's 6-round decision window can't be recovered after the
fact) wants a predictor that decides BEFORE drafting — gate 4's router, now with a measured 7%
behind it.

**The process finding is worth more than the number:** three separate harness errors each
produced a confident wrong result first (raw-vs-chat prompt, 32-vs-160 tokens, and
thinking-mode template), all input-distribution mistakes with gate 1 green throughout. See
the design page.

**Increment 4 is now SCOPED, and it is bigger than the plan assumed — two prerequisites,
both measured rather than guessed.** (1) The drafter must be **GPU-resident too**: the CPU
trunk costs 1.4 s/block (ctx 64) against ~100 ms of decode bought, a >10x net loss and the
same wall Lever 2 hit. (2) resident hidden capture, now **BUILT and gated**
(`SetHiddenCapture`/`HiddenCapture` + `TestResidentHiddenCapture`: cosine 0.998-0.999 vs the
CPU seam, off-by-one tap correctly rejected at 0.264). **My earlier "the resident runner has
no capture seam" was wrong** — `cuda/resident.go` already had a `layerCap` debug probe doing
every layer with a sync each; the grep was for the CPU seam's identifiers and read a missing
NAME as a missing capability. So increment 4's remainder is the resident bidirectional block
trunk, THEN gate 3. **Gate 3 now PROJECTS from measured inputs, and narrowed twice:** verify
M=16 is 46.4 ms (3.86x cheaper than 16 decodes), and the draft term — measured off the
target's own per-layer cost, since the drafter is exactly 5 of the target's layers + fc
(537.4 M to the digit) — is **~6.6 ms, not the 2.7 ms first modelled**. That puts **code at
1.29x (UNDER the 1.3x bar) and math at ~1.54x**. Fund the trunk against "~1.5x on math-like
traffic, break-even on code", not a flat speedup — and read the chat figure through the ROUTED
baseline below, not as a standing 2x loss. Landed on the way: a `DFlashContext` K/V cache (−50% at ctx 512+,
gate-1 bit-identical) that the GPU port should carry.
`decoder/dflash.go` (trunk + loader) matches the reference at cosine 1.00000000 on every
layer, and end-to-end through goinfer's own Qwen3-4B the drafted ids are **15/15 exact** on
both traces. The gate is falsifiable — 4 wiring mutations all rejected (design page).
Remaining in 3: the `Drafter` wiring into the existing verify + the one CPU wall-clock
measurement (which the design page predicts will lose; the entry is the GPU paths).

**DECISION 2026-08-15 (Francis) — explore the DSpark path; `license=None` accepted for
exploration.** The original pivot to DFlash was for licensing only (see the CORRECTION below);
with that lifted for a spike, `dspark_*_block7` is in scope. Upstream issue drafted at
[`docs/prompts/dspark-license-issue.md`](prompts/dspark-license-issue.md) — for eventual
distribution, non-blocking. Reframe in the design page §"Performance levers and the DSpark pivot".

- **Evaluate against ROUTED economics, not the raw 0.46x.** Gate 4 forbids indiscriminate
  firing, so chat floors at **1.0x** — real economics are **1.29–1.54x structured / 1.0x chat**,
  a win on every class. The "2x loss on chat" framing was the un-routed baseline.
- **Why DSpark:** (1) its **confidence head is a built-in acceptance router** — the chat fix is
  in the checkpoint, not a separate build DFlash needs; (2) **7-token blocks** waste less verify
  than DFlash's 16 on low-acceptance traffic. ~~(3) it likely avoids DFlash's largest build~~ —
  **CONFIRMED FALSE 2026-08-15**: DSpark's block attention is also bidirectional
  (`is_causal=False`; its training mask allows `q_block_id == kv_block_id` with no `q>=k`), so it
  needs the **identical** non-causal cross-attention kernel at M=7 instead of M=16, plus a Markov
  head and a 7-step serial matvec chain. Its trunk is 537 M — the same as DFlash's entire drafter
  — with a further 778 M of embed+lm_head that are frozen copies of the target's. **The build is
  larger, not smaller; the pivot rests on reasons 1 and 2.**
- **Two cheap probes before any trunk:** verify-amortization + acceptance on a 12B/26B target,
  which settles the mac premise (`cuda/spec_verify_ceiling_test.go` now takes a
  `GOINFER_CUDA_MODEL` override). The **7.32 math figure is an int8-predictability artifact** on
  2 prompts — int8 flattens the tail, making high-entropy text more draftable. Do not quote it as
  beating upstream; the trustworthy cross-check is code 6.14 vs 6.37 bf16.
- **DSPARK GATE 2 MEASURED 2026-08-15 — it wins on every class, including chat.**
  `scripts/ref_dspark_accept.py` drives **DeepSpec's own** loop on `dspark_qwen3_4b_block7` +
  Qwen3-4B (bf16, greedy, non-thinking, 160 tok, the DFlash suites): **α = 5.76 code / 5.73 math
  / 3.04 chat**, beating the transferred projection on all three (+41% on chat — the Markov head
  is the likely cause, and it is exactly what DFlash lacks). With the measured verify curve that
  projects **2.12× / 2.15× / 1.29×**, against DFlash's 1.29× / 1.54× / **0.46×**. **No routing is
  needed to avoid a loss.** The confidence head is *not* a fire/don't-fire gate — gating slightly
  lowers acceptance (3.04→2.96) but cuts the proposal 6.96→4.87, so verify falls 26.3→21.2 ms and
  chat nets 1.11×→1.29×. It is adaptive block length buying back verify, i.e. 04 learned.
- **Sequencing:** reframe (free) → re-price the DSpark build now the kernel is known to be needed
  → DSpark gates 1–2 on the existing `HiddenCapture` seam + DFlash harness, confidence head gating
  chat → bigger-target probes → file the licence issue, keep apache-2.0 `PARO` as fallback.
  Confidence-head threshold + block length are an **E9 autoresearch** target once the path runs.

Design page:
[`docs/spec/08-dspark-dflash.md`](spec/08-dspark-dflash.md) — the context lives there; this entry is
the claimable work.

**Increment 1 (license + checkpoint audit) is done — not a kill, but a real flag, not a clean
pass.** 3 of the 4 named checkpoints ship with **no license at all** (bare HF pages, no card); only
`z-lab/Qwen3-4B-DFlash-b16` is documented (MIT). The audit also found more official pairs than
originally scoped (deepseek-ai `dspark_qwen3_{8b,14b}_block7` + full-size `DeepSeek-V4-*-DSpark`,
all MIT; z-lab ~20 more DFlash targets) and a first-party NVIDIA Nemotron pair the doc didn't know
about. Full table and detail in the design page. **Resolve the license gap on the 4 originally-named
checkpoints before increment 2 touches them** — increment 2 is next, `linux`.

> **CORRECTION (2026-08-15, increment 2).** The clause above — "deepseek-ai
> `dspark_qwen3_{8b,14b}_block7` + full-size `DeepSeek-V4-*-DSpark`, **all MIT**" — is **wrong**, and
> it is the sentence that sent increment 2 at the 8b as a "licensed" checkpoint. Re-checked per-repo
> against the HF detail endpoint: **`dspark_qwen3_{4b,8b,14b}_block7` and `dspark_gemma4_12b_block7`
> all carry NO license and ship exactly three files** (`.gitattributes`, `config.json`,
> `model.safetensors`) — no `LICENSE`, no `README`. Only `DeepSeek-V4-{Flash,Pro}-DSpark` are MIT
> with a `LICENSE` file, and those are **166.9 GB / 892.8 GB full-size V4 models**, not standalone
> drafters. **So there is no licensed, feasible DSpark drafter checkpoint** — the "MIT" attached to a
> list whose other members were not. The list endpoint reports `license=None` even for the two that
> ARE MIT, so an enumeration pass reads them all the same way; only the per-repo detail endpoint
> separates them. Increment 2 therefore downloaded no weights. See the design page for what the
> protocol extraction achieved without them, and the decision now owed.

Two pretrained lossless block drafters (DeepSeek **DSpark**: ~5-layer backbone over our
`ForwardCapture` seam + rank-256 Markov head + confidence head, 7-token blocks; z-lab **DFlash**:
one-pass block diffusion over 16 tokens, reuses the target's embed + lm_head) with HF checkpoints
for targets we run (Gemma-4 12B, Qwen3-4B; newer pairs likely — increment 1 audits). Surfaced via
`ARahim3/mlx-dspark` (M4 Pro, batch-1 — our regime): DSpark ~1.4–1.6×, DFlash up to ~2.1× on
code/math and ~0.98× on open chat, **their numbers, not ours**.

Why this is not "EAGLE again": the spec program's own scorecard says the lever is **α, and a draft
that isn't paid per token** — EAGLE lost at ~1.6 tok/verify with a head forward per drafted token;
these draft per *block*, and DFlash's reported ~6.0 tok/verify on structured traffic lands exactly
where α̂_grammar ≈ 0.20 is our calibrated floor. The verify side is already paid for: resident CUDA
batched M=k verify measured 2.5–3.6× cheaper than k decodes at k=4–8 (D1), and a 7/16-token
**linear** block is the short-draft shape 07's large-dim re-measure said amortizes (trees don't).
Substrate all exists (Drafter/router/rollback/seam); the build is two drafter forwards (DFlash's
bidirectional block pass is the one genuinely new type) + loaders + fixtures.

**Kill-gates, pre-registered in the design page:** (1) reference-parity on dumped fixtures BEFORE
any acceptance run — the 05 lesson, priced in; (2) ≥3.0 tok/verify on ≥1 suite, else stop or back
to gate 1; (3) end-to-end ≥1.3× vs plain resident decode on ≥1 GPU backend (the 07 funding bar),
lossless gates green; (4) mixed-workload router number must not regress vs n-gram-only. CPU:
predict the 05 verdict repeats (~0.5·step verify node, ~2.2× perfect-drafter ceiling); measure once
while the CPU wiring is up, to document rather than assume. Increment 1 (license + checkpoint
audit) is free — kill there costs nothing. Not release-gating; queues behind the replace-free tag
(C3) like everything else new.

**Metal leg (`mac`) — MEASURED 2026-08-16, verdict: not ready, two independent blockers.**
Ceiling ~1.13× even at draft_ms=0 (vs CUDA's 1.74×) — Metal's batched-dispatch fixed cost (W=80ms)
dwarfs CUDA's (8.77ms) and doesn't amortize at any tested width. Separately, and more
fundamentally: the batched-verify kernel this needs (`metal.PrefillLast`) is declined by default
in production because it's not bit-identical to decode (54% divergence, §A2-Metal) — P10's
lossless contract can't be met on Metal until that's fixed, independent of the timing. Full
measurement + method in `docs/spec/08-dspark-dflash.md` under "Metal". Also corrects the
handoff prompt (`docs/prompts/metal-verify-curve.md`): it pointed at `gpu/` (WebGPU, no
`PrefillLast`), the actual harness lives in `metal/` (native, has `PrefillLast`).

**P11 · Metal batched prefill as a TTFT lever — MEASURED 2026-08-16, WIRED IN 2026-08-16
(`mac`), shipped as opt-in.** `metal/prefill_ttft_test.go`. Same kernel P10 evaluated and found
marginal for verify (tiny M) is a real win at real prefill sizes: **3.9–4.6× TTFT** across P ∈
{128,512,1024,2048} (qwen2.5-coder-1.5b int4) — a 2048-token prompt drops from ~51s to ~13s.
Consistent across the whole range, not a one-length artifact; per-token cost columns confirm the
mechanism (sequential grows with depth, batched stays near-flat, cross-validating P10's
`C≈5.9ms/row` fit from an unrelated test run). Full table + writeup:
`docs/ollama-chase.md` §"Medium / larger", "Metal batched prefill as a TTFT lever".

**The blocker is the SAME §A2-Metal divergence P10 hit** (54% stream divergence, f16-MMA vs
int8-decode) — this does not fix or route around it. What differs is the bar: P10 needed the
kernel to be the exact default verify oracle (no opt-out under 00-core's lossless contract); a
prompt-ingestion path ships as an **explicit opt-in lane** with the divergence disclosed in the
flag's own `--help` text, a materially smaller ask than unifying activation precision
project-wide.

**Landed:** `--metal-fast-prefill` (`internal/serveapp/main.go`) sets the same env var
`metal/backend.go`'s `PrefillLast` already gated on (`GOINFER_METAL_BATCHED_PREFILL`) at
startup — no frozen-core touch (`decoder/model.go`'s generic batched-prefill gate needed no
changes; it already tries `PrefillLast` by default and gracefully falls back on Metal's own
decline, so making Metal not decline was the entire wiring job). `metal/backend.go`'s decline
error now names the flag. **Verified end-to-end, not just built:** ran the real server, same
~1450-token prompt, `/v1/chat/completions`, off vs on — **68.5s → 18.3s (3.74×)**, response
identical ("done" both times) — confirming the flag actually reaches the real HTTP path, not
just the synthetic benchmark. `go test ./internal/serveapp/...` green; full build/vet clean.
