# P26c: where the CPU's ~10 s of image prefill went, and the fix (2026-10-06)

After P26a (the image prefill runs the batched hybrid forward) a 684-token image prompt on Qwen3.5-0.8B at int4 still took 11 s on the CPU, 62 tokens/s. The question was what the 11 s is. Method: the real checkpoint at serve's default quantization (`profile_test.go.txt`, a scratch
test, kept as text), `prefillQwen35VLBatched` over a synthetic 684-token prompt with a 662-token image run, twice (warm-up and timed), under `go test -cpuprofile`; the rate reproduced the serve run's (62.3 and 61.8 tokens/s).

## What the profile said

The matmuls are not the problem: the W4A8 AVX2 tile kernel is ~55 CPU-seconds over the two runs, spread across all 16 threads, about 2 s of wall per run. The caller's own goroutine, which is where the wall time goes, spent about 6.8 s per run in **`deltaNetCore` run once per token on one thread**:
`deltaNetRecurrence` ~4.9 s (scalar loops over each value head's 128x128 state, 16 heads, 18 DeltaNet layers, 684 tokens) and the conv's `silu` ~2 s (6,144 channels per token). Two further things were not worth chasing: the activation quantization at load (not part of the prefill) and attention (~0.7 s).

## The fix

`deltaNetCoreN` (`decoder/deltanet_n.go`) is `deltaNetCore` over K rows with the independent parts fanned out, and no float operation of any output element reordered: the conv (+SiLU) and the L2-normalised q and k per row in parallel (a row's window is only earlier `mixed` inputs, all known up front); the recurrence with the **value heads in parallel**, each head walking the K tokens in order with `deltaNetRecurrence`'s own loop body; the gated RMSNorm per row and head in parallel; and the two small gate projections per row in parallel. The per-token loop stays for the capture hook and the timing counters.

## Results (exploratory, one run per cell)

| | before (P26a) | after |
|---|---|---|
| isolated prefill, 684 tokens, int4 | 11.0 s, 62 tok/s | **4.6 s, 149 tok/s** |
| `serve` TTFT, new image (CPU backend) | 13.8 s (38.3 s before P26a) | **7.7 s** |
| `serve` TTFT, resend (tower output cached) | 10.7 s (34.5 s before P26a) | **4.4 s** |

The four replies are identical to the previous run's, as the bit-identity gate requires. What is left of the 4.6 s is the W4A8 matmuls (about 2 s of wall, compute-bound; a Ryzen 7 3700X has no VNNI), the batched attention, and elementwise ops; no single piece is large, and the matmul kernel is aikit's. 3.3 s of the 7.7 s is the CPU vision tower.

## Gates

`TestDeltaNetCoreN_bitIdentical`: exact bit equality of the outputs, the recurrent state and the conv window against the per-token loop, from a fresh and from a non-empty starting state, for K from 2 to 37 (below, at and above the conv kernel), with the fan-out forced onto the tiny fixture and counted so it cannot pass without forking; four mutants (a changed recurrence step, the conv activation dropped, the conv-window tail off by one, the fan-out disabled) each turn it red. The HF-golden tiny-fixture tests, `TestQwen35VLReal_G2` and `TestServe_qwen35Image_G4` (real 0.8B, exact reply text against HF) pass unchanged.
