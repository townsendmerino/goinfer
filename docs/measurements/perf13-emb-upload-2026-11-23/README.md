# Perf review finding 13: the CUDA per-token embedding upload (2026-11-23)

`cudaResident.launchToken` uploaded each token's embedding row with `gpu.Upload`: a device synchronize, a pageable copy and a second
synchronize, each a job on gocudrv's command thread. The finding: a copy on the stream instead should save the two syncs (20–50 µs a
token, under 1–3%). The change is bit-identical, so it ships only by the four-condition rule (CLAUDE.md, "Measurement discipline"):
outputs bit-identical, memory not higher, faster on at least 9 of 10 timed pairs, a replacement rather than a second path.

## Instrument

`backend/cuda/resident_decode_bench_test.go:BenchmarkResidentDecode` at depth 128, 128 decoded tokens a cell, serve's f16 KV (the
serve-auto scratch init, `docs/measurements/phi3-serve-gap-2026-11-22/serve_auto_scratch_test.go.txt`, built in by `build_bins.sh`).
Two test binaries in one session: A from the base commit, B from the change. `run_ab.py`: A B B A × 5 blocks per model, Qwen2.5-Coder
0.5B q4_k_m (`~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf`), then Phi-3 mini q4k (`~/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf`),
under the exclusive timing lock; each cell waits for a 1-minute load under 1.0 and is void above load 8.0, under 5000 MiB available,
or beside a foreign build, test or serve process. nobara (RTX 2070 SUPER, driver 595.112.07), greedy, local NVMe.

## Attempt 1: pinned staging, a copy on the stream, an event guarding the staging row (`9d802ecfe`, against `26a73a8b0`)

2026-11-23, 10:08–10:26 PST, 40 cells, none void, load 0.89–0.99 at every start, 60 GB available. Records: `ab1/`.

| model | A tok/s | B tok/s | B faster | median B ÷ A | range |
|---|---|---|---|---|---|
| Qwen2.5-Coder 0.5B | 387.5–389.4 | 384.9–387.4 | 0 of 10 | 0.9928 | 0.9895–0.9990 |
| Phi-3 mini q4k | 126.5–126.6 | 125.8–126.2 | 0 of 10 | 0.9964 | 0.9945–0.9976 |

**Slower on every pair:** about +19 µs a token on the 0.5B (a 3.5 KiB row) and +28 µs on Phi-3 (12 KiB). It fails condition 3, and
condition 2 too (12 KiB more pinned host memory). Not shipped. The cost grows with the row, not with the host-side calls, which points
at the device (a reading, not profiled): the in-stream copy from pinned memory goes through a copy engine that the first kernel then
waits on, which the driver's synchronous copy of a small pageable buffer may avoid.

## Attempt 2: the copy on the stream straight from the caller's row (`9fd27ac46`, against `26a73a8b0`): ships

`gpudev.UploadPageableAsync`: `cuMemcpyHtoDAsync` from the pageable row on `r.stream`, issued on the calling thread (no command-thread
hop), with no staging buffer and no event. The driver stages a pageable source before it returns, so the row may be reused at once;
`TestLaunchToken_queuedRowsReadTheirOwnEmbedding` poisons each row after its launch and queues the tokens behind memsets, so a copy
that read its source late, or out of stream order, fails it. Tier 0 green three times over before the timed run (`ab2/tier0-tests.log`).

2026-11-23, 12:14–12:24 PST, 40 cells, none void, load 0.85–0.98 at every start, 60 GB available. Records: `ab2/` (`b3_chain.sh` ran the
tests, the build and the A/B in order). Binary A is attempt 1's.

| model | A tok/s | B tok/s | B faster | median B ÷ A | range |
|---|---|---|---|---|---|
| Qwen2.5-Coder 0.5B | 387.4–389.9 | 392.5–395.5 | 10 of 10 | 1.0116 | 1.0074–1.0180 |
| Phi-3 mini q4k | 126.3–126.6 | 127.1–127.3 | 10 of 10 | 1.0039 | 1.0039–1.0071 |

**The four conditions hold:** outputs bit-identical (the same bytes reach `r.x` before the same launches; the decode cost golden, the
graphs self-test and the E-model parity tests unchanged); memory not higher (no allocation added; device and host unchanged); faster on
10 of 10 pairs on both models; it replaces the synchronous upload. About 31 µs a token saved on each model (from the median tok/s), inside the
finding's 20–50 µs. A decode token's counted host waits drop from 3 to 2 (`backend/cuda/testdata/sb11_decode_llama_tiny.json`).
