# A/B pre-registration — W4A8 split-half repack, end-to-end decode (amd64)

**Written 2026-08-30, BEFORE any comparison sample of this A/B exists.** The kernel-level
result (P14 item 3, 1.12× on the W4A8 dot span, hot and cold) is already recorded and is the
*input* to this question, not an answer to it.

## The question

P14 item 3 built `dotW4A8SplitHalfAVX2` and measured **1.12× on the kernel**. The repack that
feeds it is now wired into goinfer's int4 load path (`quantizeWM` / `streamQuantized`). Two
things are still unknown, and only the first is a performance question:

1. **Does 1.12× on the kernel become anything at the token level?**
2. **Is it worth the memory?** The repack keeps canonical and adds a second copy of every
   eligible tensor's nibbles: **+0.5 bytes/weight** against the 0.625 an int4 tensor already
   costs (0.5 nibbles + 0.125 scales), so int4 weight bytes grow **~80%**.

## The expectation, stated up front because that is the risk

**I expect a small positive effect, well under the kernel's 1.12×, and quite possibly under
this instrument's noise floor.** The kernel win is confined to the W4A8 matmul at M=1; a decode
token also pays attention, norms, RoPE, the KV cache, and the sampler, none of which this
touches. Knowing the expected answer is exactly when a reading bends toward it, so every branch
is fixed below before any data.

## Instrument

`BenchmarkDecode`, **Qwen2.5-Coder-1.5B-Instruct Q4_K_M** (`~/models/qwen15/`, local NVMe — the
bench set, never `/srv/models`) at `Quant: "int4"` (W4A8), `-benchtime 30x`, batch=1 greedy.
Ryzen 7 3700X, linux/amd64. **Both arms are the SAME BINARY**, selected by
`GOINFER_W4A8_SPLITHALF_OFF` at process start — so nothing but the layout differs, and no
rebuild sits between the arms.

**Interleaved on/off/on/off in one session on one box.** Session drift on this box is ~3.5%,
larger than the effect this is looking for; a sequential before/after would report the box's
mood. (`BenchmarkDecode`'s own doc comment records the incident that established this.)

**The arms must actually differ, and that is checked, not assumed.**
`TestW4A8SplitHalfFires_onBenchModel` reports **196 tensors repacked, 0 skipped** on this exact
model+quant. A zero there would make both arms the same work and any "flat" result void.

**Warm-up discard: the first sample of each visit**, carried over unchanged from the recorded
`BenchmarkDecode` calibration rather than re-derived — re-deriving a discard rule against data
I have an expectation about is the freedom this rule exists to remove. Applied identically to
both arms.

**Statistic: median of the retained samples per arm.**

**Floor: derived from a characterization pass run BEFORE the first comparison sample, and fixed
here in writing once derived.** The recorded 2.0% floor belongs to a *different* cell
(DeepSeek-V2-Lite at W8A8), and carrying a floor across a model and a quantization would be
borrowing a number that was never measured for this shape.

> **CHARACTERIZATION — filled in before the A/B, from a single-arm run:**
> **Run 2026-08-30 21:32:57–21:33:28 PDT, box quiet (load 0.01), 2 visits x 4 samples, all
> `GOINFER_W4A8_SPLITHALF_OFF=1`.** Retained after the pre-registered first-sample discard:
> 18.27, 18.43, 18.29, 18.33, 18.34, 18.31 tok/s.
> Mean **18.3283**, sd **0.0560**, relative sd **0.306%**.
>
> **FLOOR = 0.75%** (2.4 sigma, the same multiplier the recorded `BenchmarkDecode` calibration
> used). This instrument is ~2.7x tighter than the W8A8 cell's 2.0%, which is exactly why the
> floor had to be re-derived here rather than borrowed.
>
> **Known limitation of this floor, stated now rather than after seeing the result:** it comes
> from a 31-SECOND window, so it characterizes short-term sample noise and NOT the ~3.5%
> session drift this box shows over longer spans. It is a valid floor only because the A/B is
> interleaved and runs in a comparably short window. It must NOT be used to call a difference
> real across separated sessions.

## The decision rule. All branches fixed now

Stated as the *shipping* decision, because that is what is actually being argued about:

1. **Gain ≥ floor, and ≥ +4%** → **ship default-on.** The memory trade is accepted on the same
   terms the arm64 row4 repack already shipped under — row4 costs *more* (+0.625 B/weight,
   ~100%, because it duplicates the scales too) and has been default-on since 2026-08-24.
2. **Gain ≥ floor but < +4%** → **AMBIGUOUS → PARKED.** Keep the code and the wiring, flip the
   default to OFF, and record it as a measured-but-not-worth-80%-memory result. This band is
   named in advance precisely because it is where motivated reasoning lives.
3. **Within ±floor (flat)** → **do not ship default-on.** The kernel win does not survive
   composition at the token level. A result, not an absence of one.
4. **Slower by ≥ floor** → the repack costs more than it saves end-to-end; default OFF and the
   mechanism gets investigated before anything else is built on it.

## A SECOND pre-registration that can disagree with the first

The stopping-rule failure of 2026-08-28 was caught because two pre-registered things disagreed.
So, independently of the A/B: from a CPU profile of this exact cell, let **X** = the fraction of
decode time inside the split-half-eligible W4A8 dot. The kernel's measured 1.12× predicts an
end-to-end speedup of

```
  predicted = 1 / ( (1 - X) + X/1.12 )
```

> **PREDICTION — filled in before the A/B, from a profile:**
> **From the two profiles above (`-benchtime 1x` minus load, vs `-benchtime 1000x`),
> 2026-08-30 21:34:42-21:35:59 PDT, split-half ON:**
>
> ```
>   dotW4A8SplitHalfAVX2   224.76s - 2.01s  = 222.75s   (decode only)
>   total samples          466.44s - 26.10s = 440.34s   (decode only)
>   X = 222.75 / 440.34    = 50.6%
> ```
>
> X is measured on the ON arm, so the prediction runs the other way — the canonical kernel
> would cost 1.12x more: `predicted = X*1.12 + (1-X)` = **+6.1%**.
>
> **This is an UPPER BOUND, and the reason is structural.** X is a share of CPU SAMPLES, not of
> wall clock. The kernel is the fan-out-parallel part of a token (the profile ran at 661% CPU);
> serial work contributes ~1 CPU-second per wall-second while the kernel contributes ~8. So the
> kernel's wall-clock share is necessarily SMALLER than its 50.6% sample share, and the true
> prediction is "**at most about +6%**". A measured result below +6% is therefore consistent
> with the model rather than a contradiction of it; a result ABOVE +6% is not, and would mean
> one of the two instruments is wrong.

The A/B and this prediction answer different questions (what the box does vs. what the profile
says the box should do). **If they disagree by more than the floor, the disagreement is the
result** and gets written up as such — neither number silently wins.

## What this cannot answer

- **Decode only, M=1 only.** The split-half kernel is dispatched at M=1 and nowhere else, so
  this says nothing about prefill in either direction.
- **One model, one quantization, one box, batch=1.** A win here is not a goinfer speedup claim
  and does not enter `docs/benchmarks.md` or any release note on this evidence.
- **amd64 only.** `RepackInt4SplitHalf` is a no-op on arm64, which has row4 instead.
- **Load time is not measured here.** The repack is O(bytes) per tensor at load; this
  instrument starts timing after the model is resident.
- **Memory is measured analytically, not by RSS.** The expected delta is computed from the
  shapes goinfer repacked (Σ rows × ⌈cols/2⌉ over the 196 accepted tensors). An OS RSS reading
  is corroboration at most: RSS reports what survived reclaim, not what was asked for, and it
  has already inverted once on this project under exactly this kind of question.

---

# Result — appended after the run. Branch 2: AMBIGUOUS → PARKED

**Run 2026-08-30 21:36:54–21:37:56 PDT**, box quiet (load 0.01), interleaved ON/OFF/ON/OFF, one
session, one binary. Raw, in the order run. **Bold = the pre-registered warm-up discard** (first
sample of each visit).

| visit | arm | samples (tok/s) |
|---|---|---|
| 1 | ON  | **18.73**, 18.77, 18.67, 18.65 |
| 1 | OFF | **18.27**, 18.27, 18.41, 18.34 |
| 2 | ON  | **18.69**, 18.70, 18.61, 18.70 |
| 2 | OFF | **18.29**, 18.31, 18.29, 18.23 |

```
  ON  median  18.685 tok/s        range 18.61 - 18.77
  OFF median  18.300 tok/s        range 18.23 - 18.41
                                  ^ the two ranges DO NOT OVERLAP
  effect     +2.10%               floor 0.75%,  ship bar 4%
```

Paired by visit, as the matched-observation rule requires rather than pooling:
**visit 1 +1.80%, visit 2 +2.24%.** The paired and pooled readings agree, so the pooling makes
no difference here — which is worth stating, since it is only checkable by having done both.

## The verdict, and it is the pre-registered one

**+2.10% is REAL and is NOT ENOUGH.** It clears the 0.75% floor with no overlap between the
arms at all, so this is not a noise result: the split-half layout genuinely makes decode faster.
It also falls short of the +4% that was fixed in advance as the price of an ~80% increase in
int4 weight bytes. That is branch 2 verbatim — the band the pre-registration named as the place
motivated reasoning lives, entered with the rule already written down.

**Acted on:** `w4a8SplitHalfRepackEnabled` now defaults **OFF**, opt-in via
`GOINFER_W4A8_SPLITHALF=1`. The kernel, the repack, the wiring and the tests all stay. (The A/B
above was run with the opposite spelling, `GOINFER_W4A8_SPLITHALF_OFF=1`, because at the time
the default was ON; the flag was renamed when the default flipped. Anyone reproducing the raw
table above needs that.)

**The memory cost, measured rather than derived:** `TestW4A8SplitHalfFires_onBenchModel` reports
**196 tensors repacked, 0 skipped, +624.8 MiB** of duplicate nibbles — taking this model's int4
weights from 781 MiB to 1.37 GiB. Counted from the shapes goinfer actually repacked, not read
off an RSS gauge.

## The two pre-registrations did not disagree, and that is a weaker statement than it sounds

Predicted **≤ +6.1%**; measured **+2.10%**. Consistent — the measurement sits below the bound,
exactly where the bound's stated bias (CPU-sample share overstates wall-clock share for the
fan-out-parallel kernel) says it should.

But note what this does *not* establish. The prediction was an upper bound, so a wide range of
outcomes would have been "consistent" with it; only a result above +6.1% could have falsified
it. It did its job — it was fixed in advance and it could have fired — yet agreement here is
much weaker evidence than disagreement would have been. Recorded that way deliberately.

## Scope

Decode only, M=1 only, one model, one quantization, batch=1, one box, amd64 only. Not a goinfer
speedup claim; does not enter `docs/benchmarks.md`; not in any release note. Says nothing about
prefill, which the split-half kernel is never dispatched for.

**And the hardware class is narrower than "amd64" — discovered AFTER this A/B ran.** aikit's
canonical W4A8 dot prefers its AVX-512 VNNI tier where the host has one, and the split-half
kernel exists at AVX2 only, so on a VNNI machine the repack would swap a faster kernel for a
slower one. aikit now declines there. The Ryzen 7 3700X this ran on is Zen 2 — AVX2, no VNNI —
so it sits squarely in the eligible class and the +2.10% stands as measured. But the population
that figure describes is **AVX2 hosts WITHOUT VNNI**, not amd64 generally, and on newer amd64
the answer is not a smaller win, it is no win and a declined repack. That narrowing was found by
CI on a VNNI runner, not by this measurement, which could not have seen it.

**And the quantization is not the default one.** `BenchmarkDecode` ships at `int8int8`; this cell
had to set `GOINFER_BENCH_QUANT=int4` to reach the W4A8 kernel at all. So the +2.10% is available
only to int4 deployments — for anyone running the default W8A8 path the repack is not a smaller
win, it is *no* win, because nothing dispatches to it. That narrows who the ~80% memory question
is even addressed to, and is part of why it parks rather than ships.
