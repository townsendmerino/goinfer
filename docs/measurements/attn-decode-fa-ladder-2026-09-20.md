# attn_decode_fa kernel ladder (R6 step 2): S is fixed at 16

Recorded 2026-09-20 **before any fidelity scoring**, because `attn-decode-fa-fidelity-PREREGISTERED.md` requires the lane's S to
come from this kernel record and not from gate numbers. Kernel: `cuda/decode_fa.cu` (`fa_partial_{64,128,256}` +
`fa_combine`); instrument: `cuda/flash_decode_ladder_test.go` (`TestFlashDecodeKernelLadder`), one layer's decode attention on
real geometry and a real-size synthetic f32 KV, exact split-KV (3 launches) vs the lane at each S. RTX 2070 SUPER, driver
`595.91.07`, `nobara-pc`.

## Numbers that decide (ncu `gpu__time_duration.sum`, GPU-side, so independent of host load)

Measured while a 4-hour CPU reference build was running on this box (host load ~8); ncu's device timing is not affected by that,
the wall-clock ladder (below) is, and the two agree where both were taken.

| geometry, depth | exact (scores+softmax+vsum) | S=4 | S=8 | **S=16** | S=32 |
|---|---:|---:|---:|---:|---:|
| D7 (nH=28 nKV=4 hd=128), 8000 | 439.4 us | 203.3 (2.16x) | 126.9 (3.46x) | **107.7 (4.08x)** | not run |
| 1.5B (nH=12 nKV=2 hd=128), 3900 | 146.3 us | 98.0 (1.49x) | 56.7 (2.58x) | **42.3 (3.46x)** | 43.6 (3.35x) |
| 1.5B, 8000 | 291.0 us | 188.3 (1.55x) | 102.5 (2.84x) | **69.8 (4.17x)** | 66.6 (4.37x) |

S=1 is 0.41-0.59x of exact (a single CTA per kv head cannot fill the card), S=2 1.14x/0.81x; the win comes from parallelism across
keys. **S=16 is the operating point:** best or within 3% of best everywhere measured, and S=32 loses at 3900 (the combine grows
with S: 9.2 us at S=16, 14.8 us at 32). One S is registered for the gate, the same for both cells: **S = 16**.

The combine kernel was restructured once during this ladder (weights and the global max computed once per head in shared memory, a
main loop with no load-dependent branch): **23.6 -> 9.2 us at S=16 on D7**. Both figures are ncu-measured; the numbers above use the
final kernel. Partial kernel at D7@8000, S=16: 98.5 us, i.e. 32.8 MB of K+V at ~333 GB/s, about 74% of the card's 448 GB/s peak.

## Wall-clock ladder (loaded box, indicative only)

D7 at 8000 wall: exact 419.6 us, S=16 120.4 us (3.49x, before the combine fix); 3900: 209.4 -> 74.1 (2.83x); 2048: 112.5 -> 49.7
(2.26x, best S=8 44.9). 0.5B (hd=64) at 8000: 236.3 -> 48.0 (4.92x); 3900: 95.9 -> 32.1 (2.99x); 2048: 48.2 -> 31.1 (1.55x). The
1.5B wall readings at 8000 were visibly contaminated by host load (S=16 read 403 us against ncu's 70 us) and are NOT used.
These are for shape only; the speed decision is served tok/s on an idle box (`attn-decode-fa-fidelity-PREREGISTERED.md`).

## What this does and does not say

- It says the lane's *attention* is 3.5-4.4x faster than the exact path's at 3900-8000 on the two Qwen geometries measured, and that
  at 2048 the gain is 1.4-2.5x. It is a KERNEL number for ONE layer.
- **Arithmetic projection, not a measurement:** at D7@8000 the exact path's attention is 28 x 0.44 ms = 12.3 ms of a 25.9 ms token
  (38.6 tok/s); replacing it with 28 x 0.108 ms = 3.0 ms would give ~16.6 ms = ~60 tok/s (band: >= 50 ships). At 1.5B@3900 the same
  arithmetic gives ~196 tok/s (band: >= 170 ships) from 125. Neither is served; the served numbers decide.
- It says nothing about fidelity (`TestFlashDecodeVsF64` shows agreement with an f64 oracle to ~6e-7 relative on synthetic K/V;
  the end-to-end wiring check shows the lane's logit deviation from exact is the same size as the accepted V-sum spike's on the same
  chaotic synthetic input; the registered fidelity gate on held-out set B decides), and nothing about geometries with hd not in
  {64,128,256}, a GQA group over 8, or an attention sink (all decline to the exact path).
- Below the lane's attended-span floor (`GOINFER_CUDA_FLASH_DECODE_MIN_KEYS`, default 1024, to be set from the served ladder) the exact
  path still runs.
