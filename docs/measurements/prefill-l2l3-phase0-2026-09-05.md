# CUDA prefill L2/L3 — Phase 0: baseline and the ceiling arithmetic (2026-09-05)

**The arithmetic that sets the bands, written before either kernel exists, plus the "before" rows
every later ratio is taken against.** `docs/task-prefill-gap.md` §4 L2/L3 pre-registers the ship /
ambiguous / park bands; this doc records what the ceilings actually are when computed from this
box's measured category times and this card's published peaks, and flags **one place where §4 L3's
stated ceiling does not reconcile with the kernel's own profiled figure** (§3.3 below).

## Provenance

| | |
|---|---|
| box | `nobara-pc`, RTX 2070 SUPER (8 GB, sm_75), **driver 595.91.07**, Nobara 44, kernel 7.2.0-202.fc44 |
| goinfer | `9eb6ef43` (tip of main; tree clean at baseline) |
| go | go1.27.0 |
| NVRTC | 12.6.85 and 12.9.86 both present; `build_ptx.sh` regen verified **byte-identical** (24/24 PTX, `gate gpu` check 4) |
| models | **S** = `~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` (28L, nH=12, nKV=2, hd=128, hidden=1536, inter=8960)<br>**D7** = `~/models/qwen2.5-7b-instruct-q4_k_m.gguf` (28L, nH=28, nKV=4, hd=128, hidden=3584, inter=18944)<br>**0.5B** = `~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf` (24L, nH=14, nKV=2, **hd=64**, hidden=896, inter=4864) |
| model path | `~/models` on local NVMe — **not** `/srv/models` (CLAUDE.md: a row measured off the archive is void) |
| peer | Ollama v0.32.5 (`~/ollama-0325`) |
| thermal / load | at baseline start: GPU idle 51 °C, 300 MHz, 464 MiB used (kwin only); loadavg 0.05 / 0.14 / 0.08 |
| logs | `~/goinfer-logs/prefill-l2l3/` (outside the worktree — a gate log inside the tree turns PASS into INCONCLUSIVE) |

Dims are read from `Model.Dims()` on the real checkpoints, not assumed. **0.5B carries hd=64**, so
both `hd ∈ {64, 128}` branches of the L2 design are exercised by a real model rather than only by
synthetic tests.

## 1. Card peaks, stated once so every ratio below can be checked

RTX 2070 SUPER: 2560 CUDA cores, boost 1770 MHz.

| pipe | peak | derivation |
|---|---|---|
| FP32 FMA | 9.06 TFLOP/s = **4.53 TMAC/s** | 2560 × 1.77 GHz × 2 |
| INT8 `dp4a` | **18.1 TMAC/s** | 2560 × 1.77 GHz × 4 MAC/instr (INT pipe, full rate on Turing) |
| f16 tensor, **f32 accumulate** | ~28.7 TFLOP/s = **14.4 TMAC/s** | GeForce Turing halves the f32-accumulate tensor rate |
| f16 tensor, f16 accumulate | ~57.4 TFLOP/s = 28.7 TMAC/s | not the relevant peak here — the online softmax accumulates in f32 |
| INT8 tensor (IMMA) | ~114.8 TOP/s = **57.4 TMAC/s** | ≈ 3.17 × dp4a, which is §4 L3's "dp4a is ~⅓ of IMMA" ✓ |

**§4 L2 quotes "~70 TFLOPS f16 tensor".** That is the f16-accumulate figure. The L2 design
accumulates the online softmax in f32, so the honest ceiling for this kernel is **~28.7 TFLOP/s**,
half of it. Every L2 ratio below uses 14.4 TMAC/s. This makes the counted ceiling *smaller* than
§4 L2 assumed, and the band still clears comfortably (§2).

## 2. L2 — the attention term

Causal QKᵀ + PV work for S at K=3900, counting each as one MAC per (layer, head, hd, key, query)
with the causal half:

```
28 layers × 12 heads × 128 hd × (3900² / 2) × 2 products = 6.54e11 MAC = 0.654 TMAC
```

which matches §4 L2's "~0.66 TMAC". Measured attention category at K=3900 is **3.007 s**
(`cuda-prefill-attention-share-2026-09-01.md:25`).

| quantity | value |
|---|---|
| achieved | 0.654 TMAC / 3.007 s = **0.218 TMAC/s** |
| of FP32 FMA peak (what the kernel uses today) | **4.8%** |
| of f16 tensor f32-accumulate peak | **1.5%** |

**The kernel is latency-shaped, not compute-shaped** — which is what `attn_batched`'s structure
predicts: one block per (head, query row), the whole K prefix re-read per query, a serial
`nKeys`-long FMA chain for AV.

Amdahl, with attention at 55.0% of `catSum` at K=3900 — end-to-end from a category speedup `A`:

| A (attention category) | e2e |
|---|---|
| 2.08 | **1.40× — the ship bar** |
| 4 | 1.70× |
| 6.6 (5% of tensor peak) | 1.88× |
| 13 (10% of tensor peak) | 2.03× |
| ∞ | 2.22× (the cap) |

**So the ≥1.4× e2e ship bar needs only ≈2.1× on the attention category**, against a kernel
currently at 1.5% of the relevant tensor peak. The bar is not the binding risk; the binding risk
is whether a fused schedule actually converts a latency bound into a throughput one on this card.

**A capability win that is not a speed win, and is worth recording separately.** Today's launch
sizes dynamic shared memory as `(maxNWin + 128) × 4` (`cuda/resident.go:155`), so
`checkPrefillShmem` declines any layer attending more than **12,160 keys** — past that a prompt
falls back to the sequential per-token path. A 64-query × 64-key tiled kernel's shared memory is
**constant in K**, so it removes that ceiling entirely. That is independent of any ratio measured
below.

## 3. L3 — the weight term

### 3.1 The work

Per-token matmul MACs for S, counted per layer from the real dims (embedding lookup is not a
matmul; the LM head runs on one row only under `tailLastLogits` and is ~0.005% of the total at
K=3900, so it is noted and dropped):

```
q 1536×1536 + k 1536×256 + v 1536×256 + o 1536×1536   =  5,505,024
gate/up/down 3 × 1536×8960                            = 41,287,680
per layer 46,792,704 × 28 layers                      = 1.310e9 MAC/token
```

§4's "1.5B × 3900 = ~5.9 TMAC" uses the full parameter count; the matmul-only figure is
**1.310e9 × 3900 = 5.11 TMAC**. Both are recorded; 5.11 is the one the gemv category actually does.

### 3.2 The achieved rate

| K | gemv category | work | achieved | of dp4a peak | of IMMA peak |
|---|---|---|---|---|---|
| 512 | 307.7 ms | 0.671 TMAC | 2.18 TMAC/s | **12.0%** | 3.8% |
| 3900 | 2.396 s | 5.11 TMAC | 2.13 TMAC/s | **11.8%** | 3.7% |

Flat across a 7.6× depth range, as a weight-stationary GEMV should be.

### 3.3 §4 L3's stated ceiling does not reconcile — flagged, not yet resolved

§4 L3 sets its band from "the kernel is at **~54% of dp4a** and dp4a is ~⅓ of IMMA, so **~5×** is
the counted ceiling on the category". Two independent figures disagree with the 54%:

- **This arithmetic: 11.8–12.0% of dp4a peak**, at both depths.
- **The kernel's own profiled header**, `cuda/gemv_w4a8_batched.cu:27`, records attribution (2)
  "needs IMMA" being refuted at **"7.9% of dp4a peak — compute ceiling unused"**. `gemv_w4a8_rn`
  (what `bGemvB` launches today) is ~1.3× that kernel (4.41 → 3.38 ms, `cuda/prefill.go:969`),
  which lands at ~10.3% — consistent with 11.8%, not with 54%.

54% is close to the complement of the `ncu` "Compute 46%" line in the same header, which is a
*throughput-utilisation* percentage and not a fraction of dp4a MAC peak; that is the most likely
provenance of the number, but it is a guess and is labelled as one.

**What this changes:** the counted ceiling is **larger**, not smaller — perfect IMMA would be
~27×, and a good MMA GEMM at 30–50% of IMMA peak is 8–13× on the category. So §4 L3's
**≥2.5× ship bar stands unchanged and is, if anything, conservative**; nothing about the
pre-registration needs to move, and it is not being moved. What is corrected is the *reason*
written beside it.

**The caution that survives, and it is the important half.** The same header refutes IMMA
explicitly — *"NOT IMMA (compute ceiling unused)"* — because the kernel is **L1TEX-latency-bound**
(DRAM 1.7%, L2 6%, No Eligible 71%, IPC 1.07/4, ~17.8 cycle scoreboard stalls), not compute-bound.
A kernel at 12% of a compute peak is not made fast by a bigger compute peak. The case for L3 is
therefore **not** "use tensor cores because the ALU is saturated" — it plainly is not — but that an
MMA GEMM changes the *memory schedule* at the same time: activations staged once in shared memory
per block-K-step and reused across the N-slice, weights read once per block, register-blocked
accumulators. That attacks the measured bound. **This is a hypothesis with a named refutation risk,
and Phase 2 measures it rather than asserting it** — five attributions on this exact kernel have
already been recorded as conclusions and then refuted by the next measurement, and this would be
the sixth.

## 3.4 `gate gpu` at this sha — what it covered, and what it did not

The brief requires `gate gpu` PASS at the sha before any timing. What actually happened, in full,
because a partial gate reported as a pass is the exact thing this repo's gate machinery exists to
prevent:

**Run 1 was an INVOCATION ERROR, not a tree failure, and it is recorded because the mistake is
recurrent.** `-logdir` was pointed at a directory that did not exist; the gate never created it, so
every cell's `go test -json` stream went nowhere and read back empty. That produced **10 FAIL cells
with "NO ASSERTION LINE MATCHED"** and three "CELL MATCHED NO TESTS" notices — a tree-wide red from
a typo. `cmd/gate/main.go`'s provenance header carries the note that this same invocation error
produced a false "15 BLOCKER(S)" three separate times; this is the fourth. Nothing in run 1's
go-test cells is evidence about anything.

**Run 2 was correct and was stopped deliberately at 1h05m**, by decision, once it had covered the
groups this work depends on and was into a slow tail of VRAM-probe and teardown cells:

| group | verdict |
|---|---|
| 0 clean GPU | PASS |
| 1 seam | PASS |
| 2a CUDA kernel-level suite | **PASS** (9.2 s; 6 skips, all `GOINFER_HEAVY_TESTS` bandwidth/real-weight cells) |
| 2b resident PARITY gates (the forward is asserted here) | **PASS** |
| 2c heavy tier (real models) | **258 tests finished, 0 failures — INCOMPLETE, killed in `TestSpecPagerInteraction`** |
| 2d CUDA graphs bit-exactness (forced capture) | **NOT RUN** |
| 3 cgo-free build | PASS — from run 1, which is valid: this cell shells `go build` and its output never went through the broken JSON path |
| 4 PTX reproduces byte-identically at its recorded NVRTC | **PASS, 24/24** — same, valid from run 1; independently re-verified here by regenerating `attn_block.ptx` and diffing (identical) |
| W WebGPU suite (vulkan) | **NOT VALIDLY RUN in either run.** Run 1's FAIL is an artifact of the logdir bug; run 2 never reached it |
| 5 repo hygiene / citation lint | **FAIL (real)** — 4 stale `path:line` indices in `docs/QUEUE.md` for citations `docs/task-prefill-gap.md` added in the §3.1 commit. Pre-existing at `9eb6ef43`, fixed with `--update`, not by this work |

**So the honest statement is:** the CUDA kernel suite and the resident parity gates — the two groups
that would catch a moved forward — are green at `9eb6ef43`, and 258 heavy-tier tests passed with
zero failures. The graphs cell and the heavy tail did not run. No measurement below is affected by
the two cells that did not run, and neither is any claim about the fused kernel; but the gate was
not run to completion and this doc does not say it was.

## 4. Baselines

Filled in below from this session's runs. Every later ratio in Phases 1–4 is taken against these.

### 4.1 `TestPrefillDecomp`, S, K ∈ {128, 512, 2048, 3900}

_(pending)_

### 4.2 `TestPrefillTTFT`, S to K=3900 and D7 at K ∈ {512, 2048}

_(pending)_

### 4.3 Peer row, `bench_peer_prefill.py` vs Ollama v0.32.5

_(pending)_

## Not claimed

- No L2 or L3 kernel exists at the time this section was written. Every ratio in §2 and §3 is
  arithmetic on measured category times and published card peaks — an Amdahl ceiling, not a win.
- The card peaks are published figures for this SKU, not measured on this box. They bound the
  ratios; they are not themselves measurements taken here.
- §3.3 flags a discrepancy in a *stated ceiling*. It does not change any band, and it does not
  claim the 54% figure is wrong about anything measurable — only that it does not reconcile with
  two other figures and that the band's arithmetic does not depend on it.
