# Phi-3 mini on CUDA: the served path against in-process decode, same session (registered 2026-11-22)

**Status: REGISTERED 2026-11-22 14:30 PST, before any timed run.** By day, under nobara-pc's exclusive timing lock.

**Question.** Does serve decode Phi-3 mini slower than the resident decode loop it drives? The split-KV record
([`splitkv-phi3-2026-11-19/record.md`](../splitkv-phi3-2026-11-19/record.md)) read about 126 tok/s in-process at depth 128, and the
11-19 peer cell read 114.4 served. Those are two sessions and two instruments, so they are not a ratio (noise registry, TE3).
This run measures both arms in one session, interleaved.

**Arms** (one commit, `origin/main` at this branch's base; the model and quant both sides: `~/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf`
at `q4k`, f16 KV, CUDA-resident):
- **A, in-process:** `BenchmarkResidentDecode` (the tree's), 128 timed `ForwardArgmax` tokens after a prefill of D positions. It is
  built with [`serve_auto_scratch_test.go.txt`](serve_auto_scratch_test.go.txt), which restores serve's f16 `auto` KV, as the split-KV
  record did. A cell without its marker is VOID.
- **B, served:** `bench_peer.py`'s goinfer arm alone, the instrument of the 11-19 peer cell: serve over HTTP, greedy, 3 runs ×
  8 completions × 64 tokens, the decode rate `(n − 1) / (t_last − t_first)` per completion. Its token gate, swap void and decode-path
  check (`cuda-resident (q4k)`) apply. A cell's figure is the mean of its 3 runs.

**Shape.** Depths 128 and 3,900. Each depth gets 2 blocks of A B B A ([`run_gap.py`](run_gap.py)), so 4 pairs a depth. Every cell
waits for a 1-minute load under 1.0 first, the same gate for both arms. A cell is VOID if it starts above load 8.0 or beside a foreign
build, test or serve process; it is re-run in place and listed, and three voids in a row stop the run.

**Metric.** The pair ratio r = B ÷ A (served ÷ in-process).

**Rule, per depth.** Noise: the 11-19 instruments' own run spreads were 0.1–0.2% (served) and a 0.3% pair range (in-process).
- **GAP:** every pair r < 0.98. The served path loses more than noise there, so the next step is to find where (serve loop, sampling,
  detokenize, HTTP). A fix ships as ordinary engineering if it is bit-identical (CLAUDE.md: 9 of 10 timed pairs faster, memory not
  higher, it replaces code).
- **NO GAP:** every pair r ≥ 0.98. The 126-against-114 reading came from comparing across sessions and instruments; nothing is pursued.
- **AMBIGUOUS:** otherwise. It is reported, and not pursued at that depth.

**Stopping rule:** fixed, 2 blocks a depth, no early stop. Nothing on the GPU or under the lock after 21:30 PST; the driver stops
itself at 21:25.

**Tier:** day. Testing tier 1 (one small real model). It decides a direction (whether to work on the serve path), so it is registered.

**Cost:** cells read 6–7 s (A, 128), about 97 s (A, 3,900, a sequential prefill), about 22 s (B, 128) and about 82 s (B, 3,900) on
11-19, plus up to 2 min of idle gate each. That is 16 cells, about 40 min in all. The start and finish are in the results.

## Results

### Run, 2026-11-22 14:31–14:48 PST: **GAP at both depths** by the registered rule

- **The run.** One process under the exclusive timing lock (nobara-pc, RTX 2070 SUPER, driver 595.112.07), 16 cells, no void cell, no
  re-run. Every cell started at a 1-minute load of 0.83–0.99 (after its settle under 1.0), with 61,102–61,200 MiB available and system
  swap flat at 519 MiB, held from before. No foreign process was present. kwin_wayland held 19 MiB of the card, and the GPU read
  48–60 °C.
- **Build.** Both binaries come from `896638446` (this branch's base, `origin/main`):
  - serve: `CGO_ENABLED=0 go build -tags cuda ./backend/cuda/cmd/serve`, sha256 `c69859c6…`;
  - the test binary: `go test -c -tags 'cuda goinfer_testhooks'` in `backend/cuda`, with the scratch file added as
    `zz_gap_serve_auto_test.go`, sha256 `7246ac86…`. Every A cell printed the scratch's f16 marker.
- **The fit check** (serve's own banner, [`raw/fit-phi3.log`](raw/fit-phi3.log)): `cuda-resident (q4k)`, f16 KV, 2 KV slots of 4 asked
  at ctx 4096, a KV pool of 8,192 tokens in blocks of 64.

| depth | pair ratios B ÷ A (block 1; block 2) | in-process A, tok/s | served B, tok/s | per-token gap, ms |
|--:|---|---|---|--:|
| 128 | 0.9051, 0.9066; 0.9049, 0.9051 | 126.3–126.6 | 114.31–114.58 | 0.82–0.83 |
| 3,900 | 0.9495, 0.9489; 0.9498, 0.9494 | 62.21–62.27 | 59.06–59.09 | 0.85–0.86 |

- **Verdict:** GAP at 128 and at 3,900. Every pair is under 0.98, and the pairs agree within 0.002.
- **The gap is a constant per-token cost, not a depth-dependent one.** The served token is 0.82–0.86 ms longer than the in-process one
  at both depths, while the in-process token itself grows from 7.9 to 16.1 ms between them. That points at something added once per
  token around the same kernels, not at attention or the KV layout.
- **Records:** [`raw/cells.tsv`](raw/cells.tsv) (one row per cell with its start state), [`raw/run.log`](raw/run.log), and the
  per-cell logs and bench_peer result files in `raw/cell-logs.tar.gz`.

### Diagnosis, 2026-11-22 17:07–17:20 PST: serve decodes Phi-3's Q5_K and Q6_K tensors as int8, because it loads them from the CUDA q4k `.giw`

Each step was one process at a time under the exclusive timing lock, every cell started under load 1.0, all at depth 128 (the gap is
constant). Instruments: [`diag_scratch_test.go.txt`](diag_scratch_test.go.txt), built in as `zz_gap_diag_test.go` beside the f16 scratch;
[`run_diag.sh`](run_diag.sh); [`client_ab.py`](client_ab.py) and `sseclient/main.go.txt`; [`cost_probe.py`](cost_probe.py). Logs are
in `raw/diag/`.

1. **Not the KV layout, the slots, the head table or the decoder's generate loop.** In-process, A B C D D C B A: these all read 126.4
   to 126.7 tok/s:
   - the benchmark's load;
   - serve's options by hand: ctx 4096, 4 slots asked (3 given), f16 auto, windowed KV, the int8 head table;
   - `ForwardArgmax` and `decoder.Generate` (`raw/diag/diag1.log`, `diag2.log`).

   `EnableResidentConcurrency(2)` returns 1 for this load, so serve's batcher is not in play for Phi-3 at q4k either.
2. **Not HTTP and not the measuring client.** On one served model, A B B A × 2: bench_peer's own Python `post_stream` read
   114.47–114.62 tok/s, and a minimal Go SSE client 114.41–114.59 (`client-ab.log`).
3. **It is inside the served generation.** A request asking for its cost (`goinfer: {cost: true}`) reports a decode wall of
   558.5 ms for 64 tokens, 8.73 ms a token: 114.6 tok/s, inside the server. The served path is the fast greedy one (`fArg`, the fused
   argmax, once a token, and 3 syncs).
4. **The difference is the GEMV kinds** (launches a token, from the cost counter on both sides):

| | serve (`.giw` route) | in-process `decoder.Load(gguf)` |
|---|---|---|
| `gemvQ4K` | 112 | 112 |
| `gemvQ5K` | 0 | 96 |
| `gemvQ6K` | 0 | 17 |
| `gemvW8` (int8) | 113 | 0 |
| decode wall a token | 8.73 ms | 7.91 ms |

- **The mechanism, in the code.**
  - Serve loads a q4k `.gguf` through the q4k `.giw` sidecar it writes on the first load. This box's CUDA one is
    `Phi-3-mini-4k-instruct-q4.q4k.cuda.v17.giw`, 2,811,139,646 bytes, the size of the int8-widened sidecars (`raw/diag/sidecars.txt`).
  - `q4kNative56` writes the native Q5_K/Q6_K kinds (11, 12) for the cpu-amd64 target only; every other target widens them to int8.
  - The CUDA resident takes native blocks only through `Model.OpenKQuant`, which returns nil for a `.giw` load.
  - So every served CUDA load of a Q5_K/Q6_K GGUF at q4k, from its second load on, decodes those tensors from int8 copies. That undoes
    v0.25.0's "Q5_K and Q6_K run natively on Metal and CUDA under `--quant q4k`" for serve on CUDA: about 0.84 ms a token on Phi-3
    mini, and the 374.7 MB of device memory that entry names.
- **Not fixed here.** Any fix changes served outputs: from the int8 copies to the file's own blocks, the outputs a direct `.gguf` load
  already gives. The brief allowed a fix by day only if it was bit-identical. The options, set out for Francis in the status file:
  - (a) the CUDA q4k sidecar carries the native kinds, or records their source tensors so the resident reads them from the GGUF;
  - (b) serve's CUDA q4k route opens the GGUF's Q5_K/Q6_K blocks beside the sidecar, as `OpenKQuant` does for a heap load;
  - (c) leave it, and record that the served CUDA path is int8 for those tensors.
