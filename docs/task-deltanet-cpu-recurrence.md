# Task: the DeltaNet CPU recurrence — Gate D0

> Opened 2026-08-25 from `docs/prompts/deltanet-cpu-recurrence.md`. Prior art: `docs/queue-performance.md`
> already named this lever ("the DeltaNet recurrence is scalar Go"); `docs/deltanet-residency-plan.md`
> proved the compute parallelizes (WebGPU resident at 11.4-12.2x CPU decode); the 35B diagnostic
> (`docs/task-zeno-compare.md`) measured DeltaNet at ~19% of the decode token, the largest component
> no perf campaign had visited before this one.

## Gate D0 — the sixth split, real 35B-A3B checkpoint, 2026-08-25

Same env-gated component-stub method as every prior split in this repo (`GOINFER_DELTANET_TIMING=1`),
instrumenting `gatedDeltaNetStep` (`decoder/deltanet.go`) into three buckets:

- **proj**: the three W4A8/W8A8-quantized projections (`inProjQKV`, `inProjZ`, `outProj`).
- **recurrence**: section 3's per-value-head delta-rule state update loop — the suspect.
- **other**: the depthwise conv, the small f32 gate projections (`inProjB`/`inProjA`), `l2normScaled`,
  and the gated RMSNorm.

Real decode, `Qwen3.5-35B-A3B-Q4_K_M.int4.giw` (kind-3, resident, no paging — isolates the recurrence's
own cost from any paging effect), 19-token prompt + 40 generated tokens, 30 DeltaNet layers ×
59 forward passes = 1770 steps:

| bucket | ms/step | % of DeltaNet step | method |
|---|--:|--:|---|
| proj (3 quantized projections) | 1.472 | 37.6% | wrap the three `matvecWM` calls |
| **recurrence (section 3)** | **1.649** | **42.1%** | wrap the per-value-head state-update loop |
| other (conv, gates, l2norm, gated RMSNorm) | 0.792 | 20.2% | wrap the remaining sections |

Sums to 100.0% (no missing component, same bar every prior split in this repo has held to). The
recurrence is not a surprise-projections case like the MoE admit-time split was — it genuinely
dominates, as the queue entry predicted, though at 42.1% of the DeltaNet step rather than "nearly
all of it."

**Rate check: confirms scalar-chain-speed behavior.** The recurrence did Σ 927,989,760 state-element
updates (`nv·hk·hv` per step, accumulated) at **3.145 ns/element**. Each element does ~3 MACs
(the `kv` accumulation, the `S` update, the `o` accumulation) — **~1.05 ns/MAC, at or below the
~1.4 ns/MAC serial-chain signature** (`docs/task-attention-decode-cost.md`'s A0 finding: the speed of
an unoverlapped scalar/f64 FMA dependency chain on this box, ~4 cycles/MAC at ~3GHz). **Gate D0's own
test (Gate D0 §2) is satisfied: the recurrence runs at scalar-chain speed. The A1 playbook transfers.**

**Exactness surface, enumerated before designing:**

- The DeltaNet golden and qwen3_5_moe forward parity are bit-unchanged under quantization
  (established fact per the brief) — treated as exact, not tolerance-based, matching every other
  gate in this family.
- `matvec`'s own comment (`decoder/deltanet.go`) already distinguishes "SIMD reassociation only (no
  precision change of kind)" — the constraint it's protecting is float32 summation order: addition is
  not associative, so a kernel that changes the ORDER terms are summed in in a reduction is not
  automatically bit-identical to the scalar reference, even though it's mathematically equivalent.
- **The recurrence's own reduction (over `kd`, the key/head dimension) is per-`vd` and untouched by
  D1(a)'s proposed vectorization.** For a fixed `kd`, `S[kd*hv+vd]` is contiguous across `vd` — the
  state is stored with `hv` as the fast dimension specifically so this is true. D1(a) (SIMD across
  independent state elements) means running MULTIPLE `vd` lanes' accumulations *in parallel*, each
  lane's own `kd` fold order completely unchanged (still 0, 1, 2, ..., hk-1, in the same order, just N
  lanes doing it simultaneously instead of one `vd` at a time). This is bit-identical by construction,
  exactly as the brief's own framing claims — not a hopeful assumption, a property of the loop
  structure as written. The `q`/`k` vectors are read the same way regardless of how many `vd` lanes
  run per instruction; only the *order operations issue in*, not the *order they sum in*, changes.

## Amdahl projection, stated before building anything

Recurrence is 42.1% of the DeltaNet step, which is itself ~19% of the decode token: **8.0% of the
total token**, before any speedup.

| target on recurrence alone | new recurrence share | end-to-end speedup |
|--:|--:|--:|
| 2x | 4.0% | 1.042x |
| 3x (this brief's working target) | 2.67% | 1.056x |
| 5x | 1.6% | 1.068x |

**Held to honestly: even a 5x win on the recurrence alone is a ~6-7% end-to-end decode speedup on
this specific 35B-A3B checkpoint**, not a headline number — Amdahl's law is unforgiving when the
target component is 8% of the whole, however dominant it looks *within* its own layer type. This is
real, and the same order of magnitude as several other levers this campaign has shipped, but it
should be sized honestly against the effort of a new NEON kernel + threading path before committing
further, not oversold on the 42.1%-of-DeltaNet figure alone.

## PARKED, 2026-08-25 — not worth building blind at this ceiling

D0's own bar is fully met — scalar-chain speed confirmed, exactness surface enumerated, projection
derived and stated. But the ~6-7% end-to-end ceiling on the 35B is below this repo's own bar for
building a kernel without a stronger number: `docs/task-w4a8-neon-bandwidth.md`'s uncentered-Σact
retry was judged not-worth-building-blind at ~10% — this sits below that, with MORE implementation
cost (a new hand-tuned NEON kernel in `aikit/linalg`, its own threading path, its own
release/cross-family-golden discipline, permanent maintenance), not less. **Parked, not abandoned:**
the diagnosis is banked value — exactness settled, the scalar-chain confirmation, the 42.1% split all
stand — so if this lever is ever funded, D1(a) starts shovel-ready, not from zero.

**The revisit trigger is family-dependent, not "if the 35B needs 6% someday."** 8.0% is the 35B-A3B's
own number — a MoE model where DeltaNet's own layers are a large share of the token but the *whole
token* is dominated by MoE routing/GEMV, diluting the recurrence's share of the total. Qwen3.8-27B is
a **dense** DeltaNet hybrid with no MoE bucket to dilute against, and its own quantization entry
(`docs/queue-performance.md`'s P12) already recorded the symptom directly: decode gained only 1.60x
from quantizing the big projections, against the ~2.4x the byte count alone predicted, "because the
DeltaNet recurrence is scalar work that did not change" (P12 also names W4A8's own activation-quant
overhead as a second, un-split contributor to that same shortfall). Backing out the Amdahl arithmetic
from those measured numbers (`0.411 → 0.656 tok/s`, i.e. `2.433s → 1.524s`/token) against the ideal
2.4x prediction (`1.014s`/token) puts the **combined unshrunk cost (recurrence + W4A8 quant overhead,
not separately split by this arithmetic) at ~33.5% of Qwen3.8's post-quantization token** — several
times the 35B's clean, isolated 8.0% figure, consistent with "almost certainly a multiple." This is a
real signal from real numbers, not a guess, but it conflates two costs; it is NOT a clean recurrence-
only share the way the 35B's 42.1%-of-DeltaTet split is. **The honest revisit trigger: Qwen3.8 (or any
dense Gated-DeltaNet hybrid) becoming a priority** — at that point, re-run this exact sixth-split
method (`GOINFER_DELTANET_TIMING=1`, already built and committed) directly on that checkpoint to get
the clean, isolated number before deciding, the same discipline this parked entry itself followed.

## Not in scope (unchanged from the brief)

CUDA/WebGPU DeltaNet residency (separate track); the qwen35-family dense GGUF loader gap; chunked-scan
batched prefill; anything MoE/paging/kind-4; attention; W4A8; quantizing the recurrence's activations
or state.
