# Parity-hunt playbook — how to chase a backend-vs-reference numeric divergence

> The durable *method* distilled from the hunts this session (Gemma-on-Metal, CUDA-MoE
> gates). **Principle-level on purpose** — it says *how to investigate*, never *what the
> answer is*, so it doesn't rot the way a specific prompt does. When a backend produces
> wrong (or suspiciously-off) output on a real model, start here instead of re-deriving the
> traps. Task prompts in `docs/prompts/` are ephemeral (mark SUPERSEDED when their hunt
> ends); this is the standing reference.

## Prime directive

**A hypothesis — yours, mine, or the one written into a prompt — is a *target to test*, not
a truth to trust.** Every wrong turn this session was a believed hypothesis; every
resolution was a measurement. The int16-overflow theory, the "Gemma is quant-hostile"
theory, the "Fork 2 is quant-or-GEMV" framing — all wrong, all named confidently. The
GELU-tanh `exp` overflow was found by *looking*, not guessing. Localize by measurement,
then read the code at the localized point; do not assume the mechanism.

## Reference discipline — compare against the right thing

- **Use the quantization twin, not a mismatched reference.** For an int4 GPU backend,
  compare vs **CPU-int4** (`MatmulBTW4A8`, the exact quant twin), *not* CPU-int8 — or you
  conflate "what int4 costs" with "what the backend broke."
- **Split the gap with a three-way trace.** `(backend-int4 vs CPU-int8)` =
  `(CPU-int4 vs CPU-int8)` [the quant cost] + `(backend vs CPU-int4)` [the backend's own
  contribution]. Only the second term is a bug. This is what turned Gemma's 0.818 into
  "int4 costs 0.104, Metal adds 0.104" — two different problems that a single number hid.
- **Two references that agree, one that diverges = the diverging one is wrong.** CPU-int4
  and CPU-int8 tracking each other while the backend flips a channel *proves* it's the
  backend's compute, not quant sensitivity.

## Metric discipline — no single number is trustworthy

- **Report norm AND cosine — never cosine alone.** A cosine between near-zero vectors is
  dominated by rounding; it flops to ~0 while the true vector is fine. The whole "layer-0
  output is orthogonal" localization was a cosine on a **near-zero BOS attention-sink V** —
  one `‖·‖` print dissolved it.
- **Never trust an all-dims cosine alone.** It hides a handful of catastrophic channels
  among thousands of fine ones — Gemma's trunk was 0.981 all-dims while **6 of 2560
  channels were sign-flipped**, and the final norm amplified exactly those into the head.
  Rank channels **by |magnitude| AND by |divergence| separately** (they're different
  questions — this was the 443-vs-1698 reconciliation), and dump specific channels' raw
  signed values on a fixed shared set so two harnesses compare apples-to-apples.
- **Argmax agreement is a red herring at small N.** Under forced-lockstep greedy on a
  high-margin trajectory, ~60% is the floor even for a healthy model (the known-good control
  scored 15/24 on one id set, 20/24 on another — ±21 points from trajectory choice alone).
  Gate on **ΔNLL of the forced token** (`< ~0.3 nats` = functionally fine; `≫ 1` = broken) —
  the tokenizer-free "is it actually broken" metric — not argmax.

## Probe discipline — where you tap decides what you see

- **`pos 0` is BOS is the attention sink** — the most numerically pathological token.
  Choosing it to "eliminate RoPE / window / scale" (clever) *maximally exposes* sink
  degeneracy. Always also probe an **ordinary token at `pos > 0`**; a "catastrophe" that
  vanishes there was a sink artifact.

## Oracle discipline — borrow a known-correct backend

If one backend is correct by construction (CUDA's `__dp4a` keeps int32; CPU-f32), it's the
**ground-truth answer for the exact channels another backend fails**. Don't re-hunt the bug
on the clean backend — diff its treatment of the culprit channels against the broken one;
the difference *is* the mechanism. (CUDA nailing channel 443 at 0.95% with no sign-flip
told the Metal fix precisely what "correct" looked like.)

## Bisect discipline — localize, then read the code

Localize by measurement in halves — **trunk vs final-norm**, then **attention-contribution
vs MLP-contribution** (before the residual add), reference **backend-int4 vs CPU-int4**,
reporting norm+cosine+raw on the fixed channel set. Once localized to an op, **read that
op** — the mechanism is routinely something no fork you named covered (an activation
function's internal `exp`, a bounce-buffer D2H, an int8-head pin that was dead code).

## Gate discipline — a green you didn't break is not a green

- **Break-it-first.** Before trusting a gate, break the exact thing it must catch and
  confirm it goes RED, then revert. Half the gates this session were *vacuous* until broken:
  the 3%-near-tie rule couldn't see a wrong-but-similar-magnitude expert (needed a
  **calibrated cosine floor**); the unit "kernel" tests compiled **inline copies**, green
  while the shipped kernel was broken.
- **Guard the NaN hole.** `NaN < minCos` is false, so a NaN cosine silently never updates a
  running min — the gate can't fail on the *worst* inputs. Count NaNs and fail on
  `nan > 0` before the floor; `mustFinite` before every threshold.
- **A gate that doesn't run is not a gate.** The whole Metal module was uncompiled in CI;
  the prefill NaN and three stale bindings rode along silently. Build + vet + the
  device-free gate must actually execute (CI or a scripted hand-run on real hardware).

## Geometry & belief discipline

- **Measure the second geometry.** Hardcoded-shape assumptions hide on the tested size and
  break silently on another — `Kwords%32` (fine on 1.5B, garbage on 0.5B), the As-cap
  (fine on `nH·hd == H`, broke on Qwen3's `nH·hd ≠ H`). Run a *different* geometry before
  shipping — it's the single highest-value pre-ship check.
- **Measure the mechanism, don't believe it.** "Provably free" gets measured (the
  double-quant crush was measured-free, not assumed); "strictly less lossy" gets measured
  (int4-direct was verified byte-identical yet *worse* — the real defect was a per-row
  stride, invisible to a row-0 dequant check).
- **A suspiciously-large win is a symptom.** +22.9% where you expected +4–6% meant something
  was pathologically wrong (a pageable bounce-buffer D2H) — chase the too-good number.

## Permanent tooling (reuse, don't rebuild)

The capture seams (`subCapture` / `ForwardSubCapture` / `LayerKVForTest`), the three-way
trace, per-channel norm+cosine+raw dumps on a fixed set, and the ΔNLL metric are standing
infrastructure — the next hunt starts from them, not from scratch.

## Worked example (the shape of a hunt)

Gemma-on-Metal: "orthogonal layer-1 KV ⇒ layer-0 output broken" (six confident hypotheses) →
**norm print** showed it was a near-zero BOS sink (measurement dissolved the localization) →
**ordinary-token probe** showed the trunk healthy at 0.984 → the **three-way trace** split
the 0.818 into int4-cost + a real Metal-only −0.104 → **magnitude-vs-divergence ranking**
put the bug in mid channels (1698/1723/227), not the massive one (443) → CUDA **oracle**
confirmed the fix target → **within-layer bisect** localized the MLP → **reading the op**
found GELU-tanh's `exp` overflowing on the massive activation. One-line fix; every step a
measurement, not a guess.
