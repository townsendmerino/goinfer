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

(pending)
