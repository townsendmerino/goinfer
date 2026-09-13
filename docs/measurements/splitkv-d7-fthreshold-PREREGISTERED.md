# PRE-REGISTERED — D7 (Qwen2.5-7B), with the DRAM-roof fraction f as the SOLE discriminator

Written 2026-09-13 **before any cell ran**. Third geometry in the split-KV sign investigation, and
the first with its discriminator named in advance instead of chosen from a menu.

## Why f, and why only f

`splitkv-mechanism-ncu-2026-09-12.md` PARKED my mechanism: it was operationalised as
max(L1TEX, L2, DRAM) >= 1.5x and came back 1.21x. Review (2026-09-13) identified the defect, and it
is not that the idea was wrong — **L1TEX and L2 measure cache re-read pressure**, which is a
different bottleneck (mistral's GQA re-reads each KV head 4x through L1), and the max operator
folded it into the comparison. The discriminator is how close the single-block kernel already runs
to the DRAM roof:

    f  =  t_roof / t_actual        t_roof = 2 * nKeys * nKV * hd * 4 B / BW      BW = 448 GB/s

This reproduces both existing measurements from geometry: phi3-mini 65.3% predicted vs 67.83%
measured, mistral-7b 25.9% vs 26.88%. **One metric, named now, no menu.**

## Geometry, and the roofs computed before measuring

| model | nH | nKV | hd | floats/key | roof @3900 | roof @8000 | measured f @3900 |
|---|--:|--:|--:|--:|--:|--:|--:|
| D7 Qwen2.5-7B | 28 | 4 | 128 | **512** | 35.7 µs | 73.1 µs | *this run* |
| mistral-7b | 32 | 8 | 128 | 1024 | 71.3 µs | 146.3 µs | 26.9% |
| phi3-mini | 32 | 32 | 96 | 3072 | 213.9 µs | — (4k model) | 67.8% |

## Primary test — the three-point ordering, which is stronger than any threshold

If f is the discriminator, then across the three geometries at 3900 the force-ratio must be
**monotone decreasing in f**:

    ratio(D7, 512/key)  >  ratio(mistral, 1024/key) = 1.0240  >  ratio(phi3, 3072/key) = 0.746

- **CONFIRMED** — the ordering holds AND D7's ratio exceeds mistral's by more than the A/A floor
  measured in this same run.
- **REFUTED** — the ordering breaks. f is not the discriminator; nH is already dead either way
  (that rests on phi3-vs-mistral at identical nH=32), so a break here means the replacement key is
  still unknown and the gate should stay conservative.
- **AMBIGUOUS → PARKED** — D7 lands above mistral but within the A/A floor of it. Ordering
  directionally right, resolution insufficient; record and stop.

## Secondary, pre-registered separately so it can disagree

The review proposed a bar of **f < 0.40** for "split-KV wins", as the midpoint between the measured
0.26 (wins) and 0.65 (loses). Prediction: D7's f lands near **13%** (roof 35.7 µs against a kernel
expected in the 250-300 µs range, as mistral's was), comfortably inside. If D7's f is >= 0.40 while
its ratio still wins, the 0.40 bar is wrong even though the ordering may survive — and that is a
different finding from the ordering failing.

## Cells

`force` (`SPLITKV_MIN_KEYS=0`) ÷ off, at 3900 and 8000. Plus **both A/A polarities at both depths**
in the same run — not optional after the last one, where the ON floor came in 3.2x the OFF floor.
Plus one ncu pass at 3900, decode-only, grid asserted `(28,1,1)` before any number is read: the
prefill-vs-decode mix-up cost a full profile last time and is only caught by the grid check.

`prompts.json`'s `7B:3900` (3912 tokens) and `7B:8000` are already calibrated against this exact
checkpoint, so there is no calibration stage — which also removes the failure that refused run 1.

## Standing caveat

D7 is nH=28, not 32. It does not hold nH constant against phi3-mini the way mistral does, so it
strengthens the f story but is NOT the cleanest nH control. The nH refutation stands on
phi3-vs-mistral; this run is about the replacement key, not the dead one. Llama-2-7B (4096
floats/key, predicted a LARGER loss than phi3) would anchor the far end and is not on this box.
