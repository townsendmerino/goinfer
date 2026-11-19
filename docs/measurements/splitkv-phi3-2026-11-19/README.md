# Split-KV for Phi-3 decode on CUDA at depth (registration)

Registered 2026-11-19, before any data. Review perf finding 1 (the 2026-11-19 perf review), brief by the coordinator at Francis's word of
2026-11-19 13:45 PST. Tier 1: one small real model, the question is a direction at two depths.

**Question.** Phi-3-mini's decode attention on CUDA takes `attn_batched` (32 blocks), because `cudaResident.faEligible` refuses head size 96
and `splitkvThreshold` returns `splitkvNever` for nKV*hd = 3072 (`splitkvNeverKVFloats`). The never class was measured at f32 KV
(`splitkv-d7-fthreshold-2026-09-13.md`: f = 67.8 %, split-KV 0.746 at 3900). Phi-3's KV is f16 under `auto` now, which halves the bytes per
key the rule stands on (predicted f ~34 %, under that record's 0.40 bar). Does forcing split-KV make decode at depth faster?

**Arms.** A = the default load. B = the same with `GOINFER_SPLITKV_MIN_KEYS=0` (load-time knob, `decoder/knobs.go:cudaKnobs`). One test
binary for both, built once from `origin/main` 31f97ca5a: `go test -c -tags 'cuda goinfer_testhooks'` in `backend/cuda`.

**Instrument.** `backend/cuda/resident_decode_bench_test.go:BenchmarkResidentDecode`, `-test.benchtime 128x` (positions depth..depth+127,
inside Phi-3's 4096), `GOINFER_BENCH_QUANT=q4k` (the served default; per-32 activations follow from it), KV `auto` (f16; the identity run
prints the delivered precision; see Amendment 1). Checkpoint `~/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf` (local NVMe; its `q4k cuda` .giw
beside it). `run-sweep.sh` / `run_ab.py` beside this file; one process per cell.

**Cells and order.** `GOINFER_BENCH_DEPTH` 128, then 3900. Per depth five blocks of A B B A = 10 pairs (A1,B1) and (B2,A2). Each block holds
the timing lock exclusively; other sessions may take the card between blocks. Metric: tok/s per cell; per pair B/A.

**Bar.** WIN: at 3900 B faster on >= 9 of 10 pairs with median B/A >= 1.05, and at 128 median B/A >= 0.98. AMBIGUOUS (parked): median at 3900
in 1.00-1.05, or 7-8 of 10 pairs faster. KILL: anything under that, or the 128 median under 0.98.

**Void cells.** A cell is void, listed and re-run in the same order (never averaged, never dropped) if at its start: 1-minute load > 8.0;
MemAvailable < 5 GiB (the 2.8 GB .giw plus 2 GB); another session's build or test is running (a `compile`, `link`, `cgo`, `vet`, `nvcc`,
`ptxas`, foreign `*.test`, `serve`, `ollama` or `llama-server` process; `gopls` is listed but does not void); or a CUDA compute process other
than `kwin_wayland` holds the card. The desktop is logged in (kwin's ~17 MiB context); recorded, not a void.

**Fidelity (step 2).** One scratch test, arm A then arm B in one process (sequential loads): teacher-forced prose positions 3700-3899 and a
64-token free greedy run from 3900, full logits each step; plus the same at 64-127 / 128. Reported: bit-identical steps, argmax agreement,
max |dlogit|, min cosine, free-run common prefix, and split-KV launch counters (A must be 0, B > 0, or the cell is vacuous). If B is not
bit-identical, the speed verdict stands but a default change needs a registered fidelity gate and Francis's call.

**Kill from the finding (step 3, only if WIN).** nsys or the kernel trace: `attn_batched` under 30 % of the 3900 token, or above 75 % of
DRAM peak, means the lever is smaller than the finding claims; reported beside the verdict.

**Stopping rule.** Fixed N, no early stop. Not repeated for a better read. No exclusive-lock cell after 21:30 PT.

**Cost.** Per cell: load plus two prefills (the benchmark's N=1 probe and the 128x run). Estimated ~25 s at 128 and ~2 min at 3900, so
~10 + ~40 min of lock time plus waits for other holders; the first block's walls replace the estimate in the status file.

**Outcome.** WIN: no code change here; the threshold choice (re-key the never class on bytes per key, or leave it) goes to Francis with the
speed and fidelity numbers. AMBIGUOUS: parked with the numbers. KILL: the never class stands at f16; recorded as a negative result.

## Amendment 1 (2026-11-19 14:01 PST, before any f16 data)

The instrument as registered timed the wrong cache. `backend/cuda`'s test suite pins `kvF16Auto` false in
`kv16_default_test.go`'s `init`, so under the test binary `auto` delivers **f32** ("auto is f32 on CUDA in this build"), while serve's
`auto` gives Phi-3 f16 (`decoder/kvresolve.go:cudaKVF16AutoCleared`). The first identity run and six completed depth-128 cells (a seventh was stopped mid-run) ran on f32. They are
kept, ungraded, under `raw/void-f32/` and are void by instrument, not by load. (One of the six, d128 b1 c2, would also have been void: a
foreign `go`/`compile` was running at its start under the exclusive lock.)

Change: the binary adds one scratch test file, `scratch_test.go.txt` beside this README (built in as
`backend/cuda/zz_splitkv_phi3_scratch_test.go`, not committed as a test). Its `init` runs after the suite's and, with
`SPLITKV_SERVE_AUTO=1`, sets `kvF16Auto` back to true, which is serve's `auto`. `run_ab.py` sets that for both arms and marks a cell VOID if its
log lacks the init's marker line. The identity test fails unless the delivered KV is f16. The bench, the arms, the order, the bar and the
stopping rule are unchanged. The binary is built from this commit with that one file added. Each depth restarts at block 1.
