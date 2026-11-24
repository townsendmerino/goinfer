# Metal int4 prefill GEMM: next-slab register prefetch (perf review finding 7): NOT SHIPPED, slower on 10 of 10 pairs (2026-11-23)

**Result: the prefetch is 2.5% slower; the kernel is not merged.** A bit-identical speed change, so the four-condition rule decides
(CLAUDE.md, "Measurement discipline"), with no separate registration: bit-identical outputs, memory not higher, faster on at least
9 of 10 timed pairs, replacing code. It met the first, second and fourth and failed the third on every pair.

**The change** ([`prefetch.patch`](prefetch.patch), the branch's `134bf7c1d` against `bb76811c6`): `gemm_w4f16_store` and the int4
`gemm_w4f16_swiglu` read slab k0+32's weight words, group scale and activations into registers while slab k0's MMAs run, and stage
them after the next barrier. Same values in the same K order. `TestGemmPrefetch_bitIdentical` (in the patch) ran it beside the in-slab
form on the Mac: every output bit equal over M and N off the tile, one-slab and 8,960-deep K, all three epilogue modes and the swiglu
form. Threadgroup and device memory unchanged.

**The timed run, 2026-11-23 09:40–09:58 PST** (MacBook Pro M1 Pro, macOS 26.7.3, on AC throughout, the exclusive timing lock):
[`ttft_ab.py`](ttft_ab.py), WG0's TTFT cell (`webgpu-baseline-2026-11/wg0_ttft.py`: one server per cell, the cache check, 6
unique-prefix requests, median, its void rule) at the qwen2.5-coder 1.5B Q4_K_M's `1.5B:3900` prompt, `-quant int4`, ctx 4096, from
`~/models`. Two serve builds differing only in the kernel (sha256 in [`results/binaries.txt`](results/binaries.txt)), A B B A × 5 =
10 pairs. Graded by [`grade.py`](grade.py). Every cell valid, `metal-resident (int4)`, cache check 0.017 on both arms.

| | base | prefetch |
|---|---|---|
| TTFT, median over the 10 blocks | 4,171 ms (4,168–4,179) | 4,278 ms (4,273–4,289) |
| base ÷ prefetch per pair | | median **0.975** (0.972–0.977); prefetch faster on **0 of 10** |

**What it means.** Holding the next slab in registers (2 words, a scale and 4 half4 a thread, live across 16 accumulator tiles)
costs more than the load latency it hides, presumably occupancy; the kernel's single-buffered shape stays. Not tried here: a
threadgroup-memory double buffer, the review's other form, which doubles the staging arrays (about 17 KB a threadgroup) and so
also trades occupancy for overlap. The review's own expectation was modest (llama.cpp's Metal GEMM has the same single-buffered shape,
and the TTFT row is already 0.98 of Ollama's).

Raw: [`results/`](results/) (`provenance.txt`, `run.log`, `ttft.json`, `grade.txt`, `serve-logs/`).
