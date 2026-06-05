# Task (goinfer): Phase 0 — profile the decode loop (measure only)

> **For:** Claude Code, in `~/tmcode/goinfer` (with `~/tmcode/aikit` in the
> workspace — the hot kernel may be in `aikit/linalg`).
> **Goal:** find out *where the decode time goes* so the perf campaign
> (`docs/perf-campaign.md`) optimizes the right thing. **This is measurement
> only — do NOT optimize any non-test code in this task.**

## 1. A decode benchmark

Add a Go benchmark (e.g. `decoder/decode_bench_test.go`) that:
- Loads a model (Qwen2.5-Coder-0.5B GGUF at `int8int8`). **Skip cleanly** if the
  asset isn't present, like the other model-dependent tests.
- Builds a KV cache, prefills a short fixed prompt, then times **N single-token
  decode steps** in the benchmark loop (the steady-state decode path —
  `forward`/`runLayers` + sampler, greedy/seeded so it's deterministic).
- Reports `b.ReportMetric(tokPerSec, "tok/s")` and uses `-benchmem`.

Run it:
```bash
go test ./decoder -run '^$' -bench BenchmarkDecode -benchmem \
  -cpuprofile cpu.out -memprofile mem.out -benchtime 5s
```
(Profiles in a 0.5B decode are dominated by steady-state; 5s is plenty.)

## 2. Report the breakdown

From `go tool pprof -top cpu.out` and `-top -alloc_space mem.out`, capture:

- **CPU, flat + cum %**, bucketed into: the **matmul kernel**
  (`aikit/linalg.MatmulBTW8A8` / dot kernels), **`mallocgc` + GC**
  (`runtime.mallocgc`, `gcBgMarkWorker`, etc.), **goroutine dispatch / scheduling**
  (from `parallelCols` — `runtime.goready`, `chanrecv`, `morestack`),
  **attention** (KV cache read, RoPE), **activation quantization**
  (`QuantizeRowInt8`), and **everything else**.
- **`allocs/op` and `bytes/op`** from `-benchmem`, and the **top alloc sites**
  from the mem profile (expectation to confirm/deny: the per-layer
  `make([]float32, hidden)` / `append([]float32(nil), …)` in `runLayers`).
- **Effective bandwidth:** `int8_model_bytes × tok/s` in GB/s, vs this Mac's
  rated memory bandwidth (note the exact chip — M1/M2/M3, base/Pro/Max). This
  says how far below the roofline we are.

## 3. Write findings + recommend the next phase

Fill in the **Findings log** section of `docs/perf-campaign.md` with the numbers
and a one-paragraph read: which phase has the payoff —
- kernel dominates → Phase 2 (aikit/linalg kernel tuning),
- big alloc/GC share → Phase 1 (kill per-token allocations),
- dispatch/scheduling share → Phase 3 (coarser parallelism + QKV fusion).

If two are comparable, say so and rank them.

## 4. Constraints / done

- [ ] Only test/benchmark code added; **no changes to the forward path or
      kernels** in this task.
- [ ] Benchmark skips cleanly without the model asset; `gofmt` / `go vet` /
      `go test ./...` green.
- [ ] CPU + alloc breakdown, `allocs/op`, and effective GB/s recorded in
      `docs/perf-campaign.md`; next phase recommended with a reason.
- [ ] Commit the benchmark (it's the regression guard the campaign will reuse).
