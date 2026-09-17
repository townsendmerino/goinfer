# split-KV at 8000: the re-anchor FAILED as a re-anchor, and that failure is the finding

Ran 2026-09-12, nobara-pc. Pre-registration: `splitkv-8000-reanchor-PREREGISTERED.md`, written
before any cell. Raw: `goinfer-logs/splitkv-8000-reanchor-run3-20260912*.json`.

## Results

Mode `force` (`GOINFER_SPLITKV_MIN_KEYS=0`) ÷ off (`GOINFER_SPLITKV_ATTN=0`). Paired, arms adjacent,
order alternating, fresh `serve` per arm, warm request discarded. Binary `serve-cuda-1f224682`
(the anchor, not the commit field). Driver `595.91.07`, RTX 2070 SUPER, 51–52 °C, preflight green.

| cell | on | off | ratio | role |
|---|--:|--:|--:|---|
| 1.5B:128 | 222.36 | 228.12 | **0.9747** | smoke — must lose, and does |
| mistral-7b:3900 | 48.15 | 47.02 | **1.0240** | validity check |
| mistral-7b:8000 | 34.73 | 33.55 | **1.0354** | the decisive cell |
| 1.5B:8000 | 83.57 | 60.06 | **1.3915** | positive control |

## Verdict against the pre-registered rule — read this before the finding

**Primary metric: AMBIGUOUS → PARKED.** mistral-7b at 8000 is 1.0354, strictly inside the
pre-registered ambiguous band (1.00, 1.05). The rule was written so a marginal result could not be
argued into a conclusion, and it lands marginal. On its own terms this run does not say the `never`
class is wrong.

**Validity check: FAILED, which voids the re-anchor.** The pre-registration named a second, independent
condition: *if mistral-7b at 3900 does not reproduce phi3-mini's direction, mistral is not a valid
stand-in for the class and the 8000 cell says nothing about `never`.* phi3-mini at 3900 is **0.746**,
a heavy loss. mistral-7b at 3900 is **1.0240**, a win. Opposite directions. **mistral is not a
stand-in for phi3-mini, so `splitkvNever` remains un-re-anchored** and the question this run was
built to answer is still open.

The two pre-registered checks disagreed, which is what the corollary that required a second one is
for. The validity check dominates: the primary number is moot when the substitution was invalid.

## The finding: `splitkvNever` is keyed on the wrong variable

`cuda/resident.go:223-223` states the rule and its reasoning: *"at/above this many query heads the
single-block kernel already fills the device, so split-KV is pure cost. Anchored at phi3-mini's
measured nH=32 ('never') and lowered to 24…"*

Two production models **at the anchor's own nH=32** measure **opposite signs**. This is not an
extrapolation failing away from its anchor; the anchor value itself admits both outcomes, so nH
cannot be what determines the sign.

| model | nH | nKV | hd | KV floats/key (nKV·hd) | @3900 |
|---|--:|--:|--:|--:|--:|
| phi3-mini (MHA) | 32 | 32 | 96 | **3072** | 0.746 |
| mistral-7b (GQA 4:1) | 32 | 8 | 128 | **1024** | 1.0240 |

**Mechanism, plausible and not proven here:** split-KV buys occupancy, which only helps a
latency-bound kernel. MHA moves 3× the KV bytes per key that this GQA model does, so phi3-mini is
nearer bandwidth-saturated and extra blocks cannot help it. The discriminator looks like KV traffic
per key, not query-head count.

**Why this matters beyond a 2–3% cell:** D7 (Qwen2.5-7B) is nH=28 ≥ 24, so the shipped gate gives it
`splitkvNever` — and D7 is GQA, on mistral's side of the split, not phi3-mini's. D7 at depth 8000 is
exactly the cell P24 opened over (0.49 retention against llama.cpp's 0.73). Its model is not on this
box, so that pairing is inference, not measurement.

## The positive control answered the doc's OPEN item outright

`task-decode-splitkv-attention.md` §OPEN says the ladder stops at ~3900. For 1.5B it now does not:

| depth | 2048 | 3900 | **8000** |
|---|--:|--:|--:|
| 1.5B force-ratio | 1.189 | 1.282 | **1.3915** |

Monotone. Where split-KV wins, the win keeps growing past the last measured point — so for the
geometries in the table the extrapolation was conservative, not wrong.

## Limitations — two of these are defects in this run's own design

1. **No A/A floor for mistral-7b — SUPPLIED 2026-09-12, see `splitkv-aa-floor-2026-09-12.md`: floor 0.142%, effect confirmed at 17-26x it across two runs. The limitation below stood when written and is now closed.** §B6.3 characterises a per-cell floor in advance; this
   pre-registration omitted it. Within-arm spreads were 0.00–0.04 tok/s (≈0.08%) and §B6.3's measured
   floors ran 0.03–0.82%, so a 2.4–3.5% effect plausibly clears — but "plausibly clears a floor I did
   not measure" is not "clears the floor", and the mistral cells are small enough for that to matter.
   The 1.5B control at 1.39 is far outside any plausible floor; the mistral cells are not.
2. **The one overlapping cell did not cleanly reproduce.** 1.5B:128 reads 0.9747 here against
   §B6.3's 0.933 — same geometry, depth and driver, different binary (`1f224682` vs `bb42106`, with
   the L1/L2 int4 work and aikit v1.41.0 in between). Direction matches; magnitude does not. Enough
   to trust the sign, not enough to call this a reproduction of §B6.3.
3. ~~**D7's own model is absent** from this box, so its classification alongside mistral rests on its
   published config, not a measurement here.~~ **FALSE, corrected 2026-09-13.**
   `~/models/qwen2.5-7b-instruct-q4_k_m.gguf` (4.68 GB, dated Jun 8) has been present the whole time:
   nH=28, nKV=4, hd=128, ctx 32768, **kvFloatsPerKey = 512**. The claim came from
   `ls ~/models/ | grep -iE "7b" | head -4`, and qwen2.5-7b sorts fifth. D7 was directly measurable
   throughout, so the mistral substitution was not forced — though mistral remains the better
   *dissociation* control, since it holds nH=32 identical to phi3-mini where D7 is nH=28.
4. One geometry stands in for a class. A third GQA model with nH ≥ 24 would separate "GQA vs MHA"
   from "mistral specifically".

## What would settle it

An A/A floor for mistral-7b at 3900/8000, plus one more nH ≥ 24 GQA geometry. If both hold, the fix
is not to move `splitkvMaxHeads` but to re-key the `never` class on KV traffic per key — and the
gate's own comment, which reasons entirely in query heads, needs rewriting with it.
