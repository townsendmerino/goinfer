# R5 phase 1: a 32-row query tile for `attn_fused` is a regression; a 128-row tile is 2.40x faster on attention and bit-identical

Pre-registration: `attn-fused-tile-PREREGISTERED.md` (with its dated addendum, written after the phase-1 timing and before the 128-row run). RTX 2070 SUPER, driver 595.91.07, idle box, dense qwen2.5-coder-1.5B int4, `TestPrefillDecomp`
(attention category; gemv category = control), arms interleaved as separate processes A,B,C / A,B repeated for 3 rounds, same binary, arm chosen by `GOINFER_CUDA_ATTN_FUSED_TILE`. Raw logs beside this file.
Kernel: `cuda/attn_fused_bm.cu` (`template <HD, BM, BN>`; `attn_fused.cu`/PTX untouched; own PTX module).

## Result (attention ms, median of 3 rounds; every arm's round-to-round spread < 0.5%)

| K | 64x64 (shipped) | 32x64 | 32x32 | **128x64** |
|---|---:|---:|---:|---:|
| 512 | 12.8 | 33.0 (2.6x slower) | 19.0 (1.5x slower) | **6.55 (1.95x faster)** |
| 1024 | 46.1 | 120.6 (2.6x slower) | 65.3 (1.4x slower) | **23.6 (1.95x faster)** |
| 2048 | 210.6 | 546.1 (2.6x slower) | 268.9 (1.28x slower) | **96.6 (2.18x faster)** |
| 3900 | 807 | 2307 (**2.86x slower**) | 963 (**1.19x slower**) | **337 (2.40x faster)** |

The gemv control is 528-531 ms in every arm at K=3900 (within ±0.5%). Whole prefill at K=3900, 128x64: 529 + 337 + 87 = **~950 ms vs 1430 ms (1.50x)**.

## Verdict under the registered rules

- **Phase 1 (BM=32): NULL, worse than null.** Both 32-row arms are below the 1.05x line by a wide margin (they are 0.35x and 0.84x). The lever R5 named is dead; **phase 2 (a designed FA-style tile) was not started** per the rule.
- **Diagnostic A3 (128x64), against the addendum's fixed lines: CONFIRMS the staging mechanism** (0.417x of A0 at K=3900, threshold <= 0.85x; gemv control unmoved). The mechanism, now supported rather than guessed: the kernel is bound by K/V *staging*
  (f32 global load -> f16 -> shared, per block per key tile), a per-block cost, so query rows per block is the lever and its direction is UP, opposite to R5's brief (which assumed occupancy was the wall). 32x64 doubles staging per row *and* halves
  threads per block to hide it (2.86x slower); 32x32 halves shared memory but not staging (1.19x slower); 128x64 halves staging per row (2.40x faster). Not measured: an ncu breakdown of the staging share (this is inferred from the sign and size of three arms).
- **Shipping is NOT decided by this record.** The addendum says a confirmation ships nothing by itself. For orientation only: R5's phase-2 band would call 2.40x "parked" (1.5-2.5x; ship needs >= 2.5x, i.e. <= 320 ms) — but that band was written for a designed tile,
  and this arm is a drop-in whose output is bit-identical to the shipped kernel, so it carries no fidelity risk. Whether that changes the decision is the owner's call, on a fresh pre-registration; it is not made here.

## Correctness evidence

- `TestAttnFusedTile_32x64BitIdentical` (both the 32x64 and the 128x64 arms): equal to the shipped 64x64 kernel bit for bit (`math.Float32bits`) on 162 shapes (hd 64/128; M 1,7,31,32,33,64,65,200,517; startPos 0,5,1024; MHA, GQA 12/2 and 8/2, with sinks), window=0. **Not mutation-checked**
  (the kernel bodies share one template, so a rounding change would move all arms; the arms differ only in tile size).
- `TestAttnFused_vsF16Reference` extended to all four arms with a reference that walks each arm's own tiles: every arm >= 0.99999 worst cosine on the six cases (windowed and sink cases included; 128x64 == 64x64 to the printed digits).
  Windowed rows are NOT expected bit-identical across BM (tile grouping starts at the block's first row) and were not tested for it.
- The fidelity gate (§3) was not run: no arm is default. `go test -short ./cuda/...`, vet, gofmt, staticcheck clean.

## Not established / open

- **hd64 (0.5B) and windowed models (gemma3) were not timed at 128x64**; only hd128 dense 1.5B was. The 128-row tile has a bigger tail-waste for small M (a 130-row chunk is two 128-row blocks): K=512 is already faster (1.95x), the M in 16..127 band was not swept.
- Occupancy of 128x64 (256 threads x 178 regs = one block/SM; the shipped arm is two 128-thread blocks/SM, same register total) was reasoned, not read from ncu. 256-row tiles (512 threads) exceed the register file at 178 regs/thread and were not built.
- Peer prefill (Ollama) was not re-measured; the 1.5B TTFT@3900 would move 1.43 s -> ~0.95 s if this shipped, against R5's 0.71-0.84 s target.
- Default is unchanged (`attnTile` 0); the knob is experiment-only.
