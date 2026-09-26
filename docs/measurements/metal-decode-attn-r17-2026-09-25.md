# Metal decode attention at depth — R17 (2026-09-25)

R17 (`docs/tasks/red-october.md`) is pre-registered: ship ≥ 2.5× / park 1.5–2.5× / kill < 1.5× on the in-sequence
attention work at 3900 keys on the 1.5B (the no-op method of `metal-decode-decomp-2026-09-25.md`), with the
teacher-forced fidelity gate, ≤ 3% at 128 keys, sustained timing and a confirmation run as preconditions.

## Step 0 — a larger `attention_fa` split count: 1.20× at best, KILL band (2026-09-25)

**What was tested.** Whether `attention_fa` is latency-bound on too few threadgroups in flight: at 3900 keys it runs
28 threadgroups on the 1.5B (`attnFASplitFor`: S = 14 for nKV = 2, from a 2×-core-count rule). A test-only override
(`resident.attnFASplitOverride`, zero in production, read inside `attnFASplitFor` so the dispatch grid and the
per-step uniform agree) swept S; the partial buffer was re-allocated for S ≤ 64. Kernels take nSplit at runtime.

**How.** `TestMetalDecodeDecomp` with `GOINFER_METAL_DDECOMP_SPLITS=0,20,28,32,48,64` (0 = production's rule):
per split count, the production decode token with and without the attention pipelines no-op'd, arms interleaved rep
by rep, 5 reps × 20 tokens. M1 Pro, 1.5B q4_k_m, goinfer `143314c1` + the uncommitted override and sweep;
16:59:15–17:00:43 local, idle at start (load1 1.98). Raw: [`step0-split-sweep.log`](metal-decode-attn-r17-2026-09-25/step0-split-sweep.log).

| S | attention @ 2048 keys | @ 3900 keys | vs S=14 @ 3900 | full token @ 3900 |
|---:|---:|---:|---:|---:|
| 14 (production) | 4.984 ms | 8.649 ms | 1.00× | 20.982 ms |
| 20 | 5.462 | 9.221 | 0.94× | 21.560 |
| 28 | 4.918 | 8.076 | 1.07× | 20.419 |
| **32** | **4.598** | **7.235** | **1.20×** | **19.588** |
| 48 | 4.855 | 7.277 | 1.19× | 19.622 |
| 64 | 5.554 | 8.006 | 1.08× | 20.330 |

Per-rep spreads are small (e.g. S=32 at 3900: 7.2–7.6 ms).

**Outcome: KILL band as an R17 candidate** (best 1.20× < 1.5×; not taken to a confirmation run). What it says:

- **The "too few threadgroups in flight" reading is largely refuted.** More than doubling the threadgroups buys at
  most 1.2×, and the response is not monotonic (S=20 is slower than 14; S=64 gives back most of S=32's gain). The
  per-key cost is inside each simdgroup's dependent chain — consistent with the other two reasons the kernel read
  ranked: the online-softmax work is repeated for every key (a `simd_sum`, two `exp`s and a rescale per key per head,
  on every lane) rather than amortized over a block, and few loads are in flight per iteration. Both are what the
  prototype's block-of-32, lane-per-key-softmax shape changes; neither is touched by more splits.
- **A small real gain exists outside R17's band:** S=32 is 1.20× on attention and ~7% on the whole 1.5B token at
  3900. It changes the split-merge order, so taking it on its own would need the teacher-forced fidelity gate; it is
  recorded, not proposed, since R17's decision is about the kernel shape.

## Step 2 — the block-of-32 prototype: speed in the ship band, fidelity gate NOT passed — and R2's own gate does not pass either once its arms are clean (2026-09-25)

**Status: stopped for an owner decision.** Precondition 1 (fidelity) is not met as registered, so the prototype is not
graded on speed and no confirmation run was made. The same corrected gate also fails the shipped `attention_fa`, which
has been default-on since R2 (2026-09-21) on a PASS that this section finds was produced by a harness defect.

### The kernel

`attention_fa_blk<G>` in [`metal/decode_attn_r17_test.go`](../../metal/decode_attn_r17_test.go) (test-only; production
never compiles it). It replaces `attention_fa`'s first pass with the same signature, grid, and partial layout, and reuses
`attention_fa_combine` unchanged. Per simdgroup:

- keys in blocks of 32: 32 coalesced half4 K loads, and per head 32 independent `simd_sum`s landing one score per lane;
- one online-softmax step per block per head (`simd_max`, one `exp` per lane, `simd_sum`, one rescale);
- V accumulated as `simd_shuffle(p, j) · v_j`;
- G as a template parameter (instantiated for 6 and 7), so every per-head loop has a constant bound;
- a guard-free path for full blocks, with only the tail block bounds-checked.

It is not bit-identical: block-wise softmax reassociates, as `attention_fa` already does.

### Exploratory speed (these select; they do not grade)

`TestR17AttentionProto`: the production decode token, with the attention pipelines swapped (current / prototype at
split count S / all no-op), arms interleaved rep by rep, 5 reps × 20 tokens, in-sequence attention = full − no-op.
M1 Pro, idle-gated at load1 < 2.0 (1.50–1.77 at start), models from `~/models` (the 7B from its
`.int4.metal.giw` sidecar, aliased), resident context 4096, goinfer `3391eade` + the uncommitted test file.
Run 17:18:50–17:25:54 local. Raw: [`step2-explore2.log`](metal-decode-attn-r17-2026-09-25/step2-explore2.log) (the
first, S ∈ {0, 32} run, 17:16:46: [`step2-explore1.log`](metal-decode-attn-r17-2026-09-25/step2-explore1.log)).

| arm | 1.5B @2048: attention (speedup) | 1.5B @3900 | 7B @2048 | 7B @3900 |
|---|---:|---:|---:|---:|
| current `attention_fa` (S=14 / 7) | 4.874 ms | 8.574 ms | 13.601 ms | 24.670 ms |
| prototype, production split rule | 2.156 (2.30×) | 3.040 (2.82×) | 5.907 (2.30×) | 9.092 (2.71×) |
| prototype S=8 | 2.228 (2.28×) | 3.281 (2.61×) | — | — |
| **prototype S=16** | **1.770 (2.77×)** | **2.537 (3.37×)** | **4.898 (2.66×)** | **7.072 (3.49×)** |
| prototype S=24 | 2.367 (2.08×) | 3.990 (2.15×) | — | — |
| prototype S=32 | 2.497 (1.98×) | 2.941 (2.90×) | 6.686 (2.03×) | 7.004 (3.52×) |
| prototype S=48 | 3.488 (1.39×) | 3.618 (2.36×) | — | — |

Speedups are medians of the per-rep paired ratios. On the 1.5B at 3900, S=16's reps are 3.3–3.4. The 7B's
reps are noisier (e.g. 3.1–3.7 at 3900). The full token at 3900 drops from 20.86 to 14.82 ms on the 1.5B
(Ollama: 13.06) and from 69.85 to 52.30 ms on the 7B. **Selection: S=16**, the best or near-best at both depths on
both models. The response to S is not monotonic (S=24 is worse than 16 and 32), so a production rule would need its own
sweep. The 0.5B (head dim 64) never reaches `attention_fa`; the prototype is hd=128 only, so it is out of reach there.

**After-idle timing (precondition 3).** One token after 2 s idle, per arm per rep. Differencing single tokens
(full − no-op) is not resolvable: it produced negative "attention" times, because single post-idle tokens jitter by
several ms. The full token is readable. After idle, the prototype's full token is *lower* than the current one's
(1.5B @3900, medians over reps: 19.8–26.8 ms across the prototype arms vs 30.0 ms). So the win is not an idle-only win; the old kernel pays a larger ramp-up penalty.

### Fidelity

**Single-step logits probe (not a verdict).** [`step2-fidelity-probe2.log`](metal-decode-attn-r17-2026-09-25/step2-fidelity-probe2.log)
compares each arm's logits on the synthetic decode step against the exact kernel and against `attention_fa`. The
differences are bimodal:

- either ~1e-6 max|diff|;
- or a jump to ~1.5 with an argmax flip.

The **current kernel jumps the same way when only its split count changes**: on the 7B at S=32 it gets cosine
0.9899 against the exact kernel, the same flipped argmax as the prototype; on the 1.5B at S=16 it gets 0.9974. So this
is one rounding crossing amplified by the per-tensor int8 activation quantization on a synthetic prompt. It is not a
prototype defect.

**The teacher-forced gate (precondition 1).** `TestR17_decodeFidelityGate` is R2's construction with the prototype as
the candidate. It is refactored to share `runDecodeFidelityGate` with `TestR2_decodeFidelityGate`, and the refactor is
inert: the first R2 re-run reproduced the 2026-09-21 record on every prompt. The first R17 run PASSED
([`step2-fidelity-gate.log`](metal-decode-attn-r17-2026-09-25/step2-fidelity-gate.log)). But its **shipped arm
differed from R2's shipped arm** on prompts 3, 4, 5, 7, 8 and 10 (for example, prompt 3: 82.8% vs 79.7% agreement).
The shipped arm runs the exact kernel with `attention_fa` off, so it should be identical in both runs.

**Mechanism: the pipelined executor leaks the previous arm's kernel into each arm's first decode step.** `execLoop`
(`metal/model.go`) pre-encodes token t+1's command buffer right after committing t, reading the resident's state at
that moment: `decodeAttnFA`, the `pAttnFA` pipeline, `canUseAttnFA` at `curNKeys`, and the split grid. It keeps that
buffer for the next job. `PrefillLast` does not go through the executor. So the first `Forward` after an arm switch runs
a command buffer encoded under the previous arm. Localised by [`step2-state-leak3.log`](metal-decode-attn-r17-2026-09-25/step2-state-leak3.log),
using exact → X → exact on one input, compared bit for bit:

- X = exact: 0 of 16 rows differ.
- X = `attention_fa` or the prototype: 15 of 16 rows differ (the prefill seed row is identical).
- A second exact run matches the first exact run.

Restoring the KV cache, or `r.ctx` plus the partial buffer, does not remove it
([`step2-state-leak-bisect.log`](metal-decode-attn-r17-2026-09-25/step2-state-leak-bisect.log)). Nor does restoring
**all 64 changed buffers**, out of 614 reachable from the resident
([`step2-state-leak-allbufs.log`](metal-decode-attn-r17-2026-09-25/step2-state-leak-allbufs.log)). `stopExec()` after
each toggle, which drops the pre-encoded buffer, does remove it: 0 rows in every case
([`step2-state-leak-flush.log`](metal-decode-attn-r17-2026-09-25/step2-state-leak-flush.log)). `runR2GateCell` and the
timing harness's arm switch now flush.

(The timing and probe numbers above are not affected in substance. Every arm's value is a median over 20 tokens at
one fixed position, or the last of three warm tokens at that position, and only the first token after a switch was
contaminated. The after-idle token is preceded by a warm token of the same arm.)

**Clean gates, same session** ([`step2-fidelity-gate-flushed.log`](metal-decode-attn-r17-2026-09-25/step2-fidelity-gate-flushed.log),
18:03:10 and 18:06:59 local, idle at 1.75 and 1.92). The shipped arm is now identical in both runs:

| S, set "a", K=3900, 10 prompts × 64 | hard flips | agreement | pooled mean KL | lower KL on | verdict |
|---|---:|---:|---:|---:|---|
| shipped exact kernel (W4A8) | 69 | 80.47% | 0.4852 | — | — |
| **R17 prototype, S=16** | 69 (critA ✓) | 80.00%, d=23 (critB ✓) | 0.4871 | 4/10 (critC ✗) | **DOES NOT PASS** |
| **shipped `attention_fa` (R2's gate, re-run clean)** | 73 (critA ✓) | 80.16%, d=16 (critB ✓) | 0.4872 | 4/10 (critC ✗) | **DOES NOT PASS** |

The prototype and `attention_fa` are indistinguishable from each other (KL 0.4871 vs 0.4872), and both sit about 0.4%
above the exact kernel in pooled KL. That fails critC, which has no noise allowance: the candidate's pooled KL must be
≤ the exact arm's, and lower on at least half the prompts. **R2's recorded PASS (0.4859 ≤ 0.4861, 6/10) was measured
on contaminated arms.** Every arm after the first began with one step of the other arm's kernel, and a 0.0002 margin
is well inside what that moves.

### What this leaves open (owner decisions, not taken here)

1. **`attention_fa` default-on rests on a gate that, run clean, does not pass.** Revert the default, or amend §3.2's
   critC with a mechanism. Re-baselining the bar because a number moved is ruled out by this repo's own discipline.
2. **R17's fate follows from (1).** The prototype is as faithful as the shipped `attention_fa`. Whatever bar is judged
   right for `attention_fa` decides the prototype too.
3. **The executor leak is a production issue, not only a harness one.** It follows from the mechanism but was not
   measured here. In production the toggle is fixed, but the encode-time attention plan (`attention_fa` vs exact, and
   the split grid) comes from the *previous* token's `curNKeys`. So the first decode step after a new prefill runs the
   previous request's plan. Two consequences:
   - A short request (< 1536 keys) that follows a long one runs `attention_fa` for that step, below its floor, with a
     split grid that disagrees with the uniform whenever nKeys < 448. The extra threadgroups then read past the `q`
     region. Their writes stay inside the partial allocation and are ignored by the combine.
   - Output depends on the previous request.

   R1's lane gate (`r1_gate3_test.go`, `decodeLaneW4F16`) toggles encode-time state the same way. R1 was killed on
   speed, so nothing shipped on it.

## Is the critC failure the bar or the kernel? Kernel accuracy vs float64, and the gate's null distribution (2026-09-25, evening)

**Question.** Is a critC failure by a non-bit-identical kernel evidence of lost accuracy, or can the gate not tell
rounding from accuracy at all? Two direct measurements answer it: the kernels' own error against float64 on real
inputs, and the gate's verdicts on changes that are accuracy-neutral by construction.

### Kernel accuracy against a float64 reference on real inputs

`TestR17KernelAccuracy` ([`metal/decode_attn_r17_test.go`](../../metal/decode_attn_r17_test.go)) works as follows:

- It uses 3 real prompts from set "a", prefilled at K=3900, and one decode step at pos K encoded **layer by layer**
  with the exact kernel in the trunk.
- After each layer it takes that layer's post-RoPE q (`r.qkv`) and its f16 K/V cache as inputs. Every kernel under
  test is dispatched standalone on those same buffers.
- The reference is attention in float64 on the host, from the same f32 q and f16 K/V, so it measures kernel
  arithmetic only.
- Capture sanity: the standalone exact dispatch is bit-identical to what the layer wrote into `r.ctx` in **84/84
  layers**, on both models.

Models are loaded from the `.int4.metal.giw` sidecars in `~/models` (the same int4 weights).

| per-head relative L2 error vs float64 | 1.5B median | p99 | max | 7B median | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| shipped exact `attention` | 5.66e-7 | 5.78e-6 | 1.37e-5 | 7.49e-7 | 9.34e-6 | 2.23e-5 |
| shipped `attention_fa` (production S) | 1.77e-7 | 1.54e-6 | 4.17e-5 | 2.77e-7 | 2.45e-6 | 1.28e-5 |
| **R17 prototype S=16** | **1.72e-7** | **1.43e-6** | **3.53e-5** | **2.42e-7** | **2.22e-6** | **1.68e-5** |

- **The prototype is closer to float64 than the exact kernel on 939/1008 heads (1.5B) and 2212/2352 (7B).**
- The median gain is about 3× and holds in every one of the 28 layers on the 1.5B.
- On the 7B, the prototype also wins on the worst head and on the worst absolute error (5.2e-5 vs 1.28e-4).
- The one statistic where exact wins: the 1.5B's single worst head (prompt 3, layer 0, head 1). It is the worst
  head for every kernel. The split kernels are 2.6× worse there.
- `precise::exp` in every split-kernel `exp` does not change that head (3.525e-5 before and after), so fast-math
  `exp` is not the cause. The median improves only from 1.72e-7 to 1.66e-7.
- Even that head's error is about 100× below the step of the per-tensor int8 quantizer `ctx` passes through next.

Why the exact kernel is the least accurate: it sums all 3901 keys **serially in f32**, per output element. The split
kernels sum shorter chains and merge the partials.

Raw: [`step2-kernel-accuracy-1.5b.log`](metal-decode-attn-r17-2026-09-25/step2-kernel-accuracy-1.5b.log),
[`-perlayer`](metal-decode-attn-r17-2026-09-25/step2-kernel-accuracy-1.5b-perlayer.log),
[`-precise`](metal-decode-attn-r17-2026-09-25/step2-kernel-accuracy-1.5b-precise.log); the 7B is the last section of
[`step2-nudge-gates.log`](metal-decode-attn-r17-2026-09-25/step2-nudge-gates.log).

**Does the gate's CPU reference share the exact kernel's error?** The S-K3900 reference computes attention as
`decoder/attention.go` `attendQuery` does: scores and softmax in f64, then a serial f32 V accumulation in the same
key order as the exact kernel. Emulating it on the same captured inputs (fused and unfused) gives the following
([`-sharedbias`](metal-decode-attn-r17-2026-09-25/step2-kernel-accuracy-1.5b-sharedbias.log)):

- The exact kernel's error vector leans slightly toward the emulation's error: median cosine +0.065, positive on 58%
  of heads.
- The split kernels' error vectors are uncorrelated with it (about 0, 47–52% positive).
- Even so, **the split kernels are closer to the emulated reference arithmetic than the exact kernel is, on about 84%
  of heads.**

So there is a weak shared direction, but it is too small to make the exact kernel the nearer one to the reference.

**How far each arm moves the output from the exact kernel** ([`-magnitude`](metal-decode-attn-r17-2026-09-25/step2-kernel-accuracy-1.5b-magnitude.log);
median relative L2 of out − out_exact):

| arm | moved from exact by | its own error vs f64 (median) |
|---|---:|---:|
| `attention_fa` / prototype, every S | 5.9–6.1e-7 | 1.7–1.8e-7 |
| exact-nudge k=+1 (softmax scale × (1+2⁻²³)) | 1.6e-7 | 5.8e-7 |
| exact-null (4-term q·k partial sums reversed) | 2.3e-7 | 5.6e-7 |
| exact-nudge k=−3 | 3.7e-7 | 6.8e-7 |
| exact-nudge k=+16 | 1.7e-6 | 1.9e-6 |

The nulls graded first were 2–4× *smaller* perturbations than the candidates. Matching the candidates' size needs
|k| ≈ 5–7, and at that size a nudge is also slightly *less* accurate than exact, which makes it a conservative control.

### Set A's S-K3900 references do not match 4 of the 10 prompts

The gate reads its prompts from the frozen snapshot `testdata/prefill-gate-prose-a/`, added in `b0bdf43d` on
2026-09-09 at 12:14. The set-A S-K3900 reference files (`~/goinfer-logs/prefill-ref/S-K3900-p*.bin`) were generated
on **2026-09-05** (file times 14:40–14:55; see `prefill-ref-gen-2026-09-05.log`) from the *live* docs of the time. The
docs changed in between. The first byte at which each snapshot differs from the same file at `29d40c2d` (2026-09-05
14:26) was checked; a 3900-token prompt covers roughly its first 12–16 KB:

| prompt | file | first differing byte | size change | exact-arm KL |
|---:|---|---:|---:|---:|
| 1 | `audit-2026-09-02.md` | 9,172 | +29 B | 0.2374 |
| 2 | `QUEUE.md` | 2,894 | +10.9 KB | 1.6347 |
| 5 | `benchmarks.md` | 1,356 | +27.2 KB | 0.9456 |
| 8 | `legacy-benchmarks.md` | 2,180 | +0.6 KB | 1.7227 |
| 9 | `task-zeno-compare.md` | 2,179 | 0 (a same-length edit) | 0.0407 |
| 3, 4, 6, 7, 10 | — | ≥ 18,329, or identical | — | 0.033–0.072 |

**Prompts 1, 2, 5 and 8 teacher-force the reference's continuation onto different text.** They carry 4.54 of the
4.85 pooled KL and nearly all of its ±0.01 per-prompt swings. So on set A, the pooled verdicts are dominated by a
data defect. That includes R2's 2026-09-21 PASS, today's clean FAIL, and the null battery pooled over all 10
prompts. None of them is a fidelity verdict.

Set B's S-K3900 references (`prefill-ref-b/`, written 2026-09-09 12:29, fourteen minutes after the snapshot commit)
were generated from the snapshot, which has not changed since. So set B is valid and unused for any Metal
decode-attention decision.

Harness fixes (test-only):
- `runDecodeFidelityGate` hardcoded set A's directory, so `GOINFER_PREFILL_GATE_PROMPTS=b` would have scored set-B
  prompts against set-A logits. It now uses `refDirFor(home, label)`.
- It now prints a **prompt-identity check** per prompt, KL(reference row 0 ‖ prefill seed), and errors when any prompt
  exceeds 1.0.

### The gate's null distribution

Every run is `TestR17_decodeFidelityGate` with the executor flushed and the exact arm unchanged; the exact arm is
identical at printed precision in all runs. Candidates:
- `exact-null`: the shipped kernel with each 4-term q·k partial sum reversed.
- `exact-nudge(k)`: its softmax scale × (1 + k·2⁻²³).
- `exact-vchunk(C)`: its V sum accumulated in C-key chunks.
- `exact-vrev8`: its 8-term V groups added in reverse.
- The split family: `attention_fa` at S = 8/14/20/28/32 and the prototype at S = 8/14/16/32.

Table: [`null_table.py`](metal-decode-attn-r17-2026-09-25/null_table.py). Raw:
[`step2-null-gates.log`](metal-decode-attn-r17-2026-09-25/step2-null-gates.log),
[`step2-nudge-gates.log`](metal-decode-attn-r17-2026-09-25/step2-nudge-gates.log),
[`step2-nudge-matched-gates.log`](metal-decode-attn-r17-2026-09-25/step2-nudge-matched-gates.log),
[`step2-vsum-gates.log`](metal-decode-attn-r17-2026-09-25/step2-vsum-gates.log). All runs were on set A (calibration,
not a decision) between 18:36 and 20:20.

Ratio = candidate mean KL / exact-arm mean KL over the prompts used. "Strict critC fails" counts runs failing
`mean ≤ exact AND lower on ≥ half the prompts`.

| group | n | all 10 prompts: ratio (mean) · strict critC fails | valid six (3,4,6,7,9,10): ratio (mean) · fails |
|---|---:|---|---|
| exact, small rounding nulls (reversed q·k; k = ±1..3) | 7 | 0.995–1.003 (0.999) · 3/7 | 0.995–1.013 (1.003) · 5/7 |
| exact, magnitude-matched scale nudges (k = ±5..7) | 6 | 0.998–1.004 (1.000) · 3/6 | 0.992–1.027 (1.007) · 4/6 |
| exact, 8-term V groups reversed | 1 | 1.000 · 1/1 | 0.989 · 0/1 |
| exact, chunked V sum (C = 32/64/128/256) | 4 | 0.996–1.000 (0.998) · 1/4 | 0.998–1.009 (1.004) · 3/4 |
| split family (`attention_fa` ×5 S, prototype ×4 S) | 9 | 1.001–1.005 (1.003) · 9/9 | 1.009–1.030 (1.018) · 9/9 |

**Reading.**

1. **Strict critC cannot judge a kernel that is not bit-identical.** The shipped kernel's own rounding-level variants
   fail it 8 of 18 times over all 10 prompts, and 12 of 18 times on the valid six. Its pooled "≤ exact" clause has no
   noise allowance, and its sign clause fails an equal-quality arm at about coin-flip rate. The criterion was written
   on 2026-09-05 for a candidate predicted to be *better* (f16-activation prefill). This supplies the mechanism an
   amendment needs. It is not a number moving after the fact: the nulls are accuracy-neutral by construction.
2. **No end-to-end accuracy loss is resolvable.** On the valid six prompts the split family sits at 1.018×. That is
   inside the range of the exact kernel's same-size perturbations (0.992–1.027×). The family is roughly *one* draw,
   not nine: every split kernel lands near float64, so each one's move away from exact is mostly the same vector (the
   exact kernel's own error, removed). CUDA's R6 lane, measured 5–7× more accurate than its exact kernel against f64,
   showed the same ~2% (1.0205× on its S cell) and shipped under `≤ 1.10×`.
3. **The shared-serial-sum explanation is refuted.** The chunked-V exact variants remove the serial f32 V-sum error,
   which is the part of the exact kernel's arithmetic the CPU reference shares, and they stay with the nulls (1.004×).
4. **The only direct accuracy evidence favours the prototype:** about 3× closer to float64 than the shipped kernel at
   the kernel level, on both models.

**Caveat.** The valid-six figures are a post-hoc subset of a development set: calibration, not a decision. A decision
belongs on set B under a bar registered first.

### Corrections to earlier statements in this session

- "KL is convex, so a strict ≤ is biased toward the unperturbed arm": withdrawn. The data show no general upward bias
  on all ten prompts (the exact-kernel nulls average 0.999–1.000×).
- R2's "attention_fa matches the exact kernel to ≤ 1e-5 on identical inputs" was measured on Gaussian embeddings at
  1600 keys, not on real prompts at 3900. It is superseded by the float64 measurement above.
- The strict critC is the owner's explicit 2026-09-18 policy for R1/R2/R6 (`red-october.md`), not a copying slip.
  Changing it is the owner's decision.

## Production fix: the executor encodes each command buffer for its own job's key count (2026-09-25)

**Defect** (see "Mechanism" under Step 2). `execLoop` (`metal/model.go`) encoded token t+1's command buffer
right after committing token t. The two attention decisions an encode bakes in, attention_fa's depth gate and its
split grid, read the key count *at that moment*, which was the previous job's. Nothing dropped a pre-encoded buffer
when a new request began, because `PrefillLast` does not go through the executor. Production consequences:

- the first decode step at 1536 keys ran the shipped kernel instead of attention_fa;
- the first decode step of every request ran the plan of wherever the previous request stopped. After a long
  request that meant attention_fa below its floor, with a split grid sized for the old depth and a split uniform
  sized for the new one, so extra threadgroups read past `q` whenever the new request had fewer than 32·S keys;
- the first decode step after a `ForwardBatch` declined attention_fa at any depth.

So output depended on the previous request.

**Fix.**
- `attnPlan` holds everything an encode bakes in that depends on the key count or on a runtime toggle:
  attention_fa on or off, its split count, and the f16 lane.
- The executor encodes each buffer for a known key count through `encNKeys` / `planNKeys`: the job's own
  count when it encodes fresh, and pos+2 when it pre-encodes for the predicted next job.
- Before committing, it compares the buffer's plan with `attnPlanFor(job.pos+1)`. On a mismatch it drops the buffer
  uncommitted and re-encodes, the same path a head-mode switch already took.
- The dispatch grid and `setPos`'s split uniform are both `attnFASplitFor` of the same key count, so they agree by
  construction.
- Steady decode never mismatches, and costs one struct comparison per token.

**Test** (`TestExecutorAttnPlan`, `metal/exec_plan_test.go`, committed fixture `testdata/llama-attnfa-tiny`, about
6 s). A sequence of requests runs through the production adapter (`metalResident.Forward` → executor), never
flushed, and each is compared bit for bit with the same request run synchronously:
- a floor crossing;
- 400-, 700- and 1400-key prefilled requests, each after a long one;
- long prefilled requests, each after a short one;
- decode after `ForwardBatch`.

**All 7 cases failed on the unfixed executor, each at exactly its first stale step** (positions 1535, 400, 1600, 700,
1600, 1400, 1556), and all 7 pass with the fix. Two sizes were discarded as insensitive, because the stale plan ran
and the output coincided on this fixture: a request decoded from position 0 (one key, where softmax is 1.0) and
prefilled requests of 101 and 1001 keys.

**Cost: none measurable.** Setup:
- `TestEncodeAhead` on the 1.5B (int8int8, positions 30–98), best of 60 tokens per arm;
- the unfixed and fixed test binaries alternated old/new three times;
- 20:49–20:50 local, load1 1.5–2.0.

Pipelined decode is **12.83 ms/token with the fix against 12.86 without** (medians of 3). The fix is equal or faster
in every pair, by 0.01–0.03 ms, which is noise. Encode-ahead parity is 10/10 in all six runs. Raw:
[`exec-fix-ab.log`](metal-decode-attn-r17-2026-09-25/exec-fix-ab.log). The Metal suite is green both ways:
- untagged: [`exec-fix-suite-untagged.log`](metal-decode-attn-r17-2026-09-25/exec-fix-suite-untagged.log);
- `goinfer_testhooks`: 316 passes, 79 skips, 0 failures,
  [`exec-fix-suite-tagged.log`](metal-decode-attn-r17-2026-09-25/exec-fix-suite-tagged.log).
