# CPU KV cache: f16 against auto's f32 (perf finding 12), registered 2026-11-23 08:30 PST

**Status: REGISTERED, before any run.** Owner's go: Francis, 2026-11-23, relayed by the coordinator. By day on nobara-pc (Ryzen 7 3700X,
16 threads), under the exclusive timing lock, with the box otherwise idle.

**Question.** On the CPU, `--kv auto` keeps an f32 KV cache (`internal/runtime/mem/kv.go`: "auto is f32 on the CPU"); f16 is delivered on
request. Does f16 decode faster at depth, at fidelity no worse than f32's, on Phi-3 mini (`q4k`, its default) and Qwen2.5-Coder-0.5B
(`int4`)? Finding 12 (the 2026-11-19 perf review) prices Phi-3's 3,900-position cache at 3.07 GB f32 against 1.53 GB f16.

**Speed.** `BenchmarkDecodeAtDepth/grouped_on` (production's grouped attention; `GOINFER_BENCH_KV` picks the arm's `KVQuant`, f32 or
f16), Backend cpu, at each model's default quant. 32 timed tokens a cell, one process a cell, depths 128 and 3,900, A B B A × 3 blocks
(6 pairs) per model and depth. Every cell waits for load under 1.0 and records its start state. A cell is VOID if it starts above 8.0 or
with less memory than its fit need, or if swap grows during it; a void cell is re-run in place and listed.
- **Metric:** the pair ratio f16 ÷ f32 of tok/s.
- **Bar** (the owner's default): SHIP at a median ≥ 1.02 at 3,900 and ≥ 0.98 at 128; PARKED from 1.00 to 1.02; OFF under 1.00.
- **Finding 12's kill, read from the f32 arm:** if the 3,900 token costs under 1.3× the 128 token, KV traffic is not the cost, and the
  record says so whatever f16 reads.

**Fidelity.** `TestCPUKVF16GateVsReference` (new, `decoder/`): both arms at the default quant, teacher-forced on the reference tokens of
held-out set B (K = 3,900, 64 positions × 10 prompts), each scored against the CPU f32 reference.
- **References:** `P3M-K3900` exists (2026-11-22). `Q05-K3900` is built first, by `TestPrefillGateReference` with its existing row.
- **Criteria,** the KV gates' (S4's, unchanged), f16 against f32: (a) hard flips ≤ f32 + 2√f32; (b) mean agreement ≥ f32 − 1.0 pt and
  f16 ≥ f32 on at least half the prompts; (c) mean KL ≤ 1.10 × f32's.
- **AMBIGUOUS → PARKED:** (b) within its last 0.2 pt, or (c) in 1.05–1.10×. A re-run needs a named mechanism.
- **Preconditions** (a failed one voids the cell): the f16 arm's cache holds f16 rows; its decode rows differ from f32's on every prompt;
  A/A is bit-identical on the first prompt.

**Host bytes:** each arm's KV bytes at the cell's depth, from the cache, and its process's VmHWM.

**Stopping rule:** fixed N at both instruments; no early stop.

**What each outcome means:**
- Fidelity PASS and speed SHIP on a model: evidence for Francis to make `auto` mean f16 on the CPU for that family. No default moves here.
- Fidelity FAIL or PARKED: `auto` stays f32 for it, and the record names the criterion.
- Speed PARKED or OFF, with fidelity passing: f16 stays on request, and the record states its memory saving alone.

**Cost:** speed 48 cells, about 50 s each plus about 30 s of settle, so about 65 min. Fidelity: the 0.5B's references about 10 min, the
gate about 30 min for Phi-3 and 5 for the 0.5B. In all, about 2 h by day, finishing well before 21:30.

## Running notes (not yet the result)

- **Cost estimate was low** (written 2026-11-23 09:31 PST): each 3,900 cell pays its own batched prefill before its 32 timed tokens
  (about 116 s on the 0.5B), so the speed half runs past its 65-minute estimate. The driver was made resumable (it skips cells it holds a
  reading for) so the outer 150-minute timeout can be followed by a re-launch without re-running a cell. No cell, arm or bar changed.
- **Mechanism, read from the code after the 0.5B's cells:** single-token decode on an f16 global layer goes through
  `KVCache.dequantGlobalLayer`, which widens the layer's whole history into f32 scratch on one goroutine every token, and attention then
  reads the f32 scratch (`decoder/program_cpu.go:attendToken`'s `cache.quant.widened()` case). Per element per token that is a 2-byte
  read, a 4-byte write and a 4-byte read against f32's one 4-byte read, serial ahead of the parallel head workers.

## Speed: Qwen2.5-Coder-0.5B (int4), complete, read 2026-11-23 09:36 PST

24 cells, 08:41–09:28 PST, A B B A × 3 blocks at each depth, one process a cell, the decoder test binary built at `8bf0687b1`. Start load
0.85–0.98 on every cell (settle under 1.0), no swap growth, no void cells. Cells: `speed/q05-cells.tsv`; logs: `speed/q05-logs.tar.gz`.

| depth | f32 tok/s (mean) | f16 tok/s (mean) | pair ratios f16 ÷ f32 | median | bar | reading |
|---|---|---|---|---|---|---|
| 128 | 55.13 | 53.90 | 0.978 0.988 0.986 0.946 0.981 0.989 | **0.983** | ≥ 0.98 | holds |
| 3,900 | 23.31 | 17.17 | 0.720 0.729 0.728 0.723 0.788 0.733 | **0.729** | ≥ 1.02 | **OFF** (under 1.00) |

**0.5B speed: OFF.** Finding 12's kill does not fire: the f32 token at 3,900 costs 2.37× the token at 128, so depth is a real cost here; f16
makes it worse, by the mechanism in the running notes. Peak RSS is the same for both arms (964 MB): it is set by the load, not by a cache this
small, so the host-bytes reading comes from the fidelity gate's cache count.

## Pause between models: lock sequencing, coordinator, 2026-11-23 09:35 PST

At the coordinator's word (09:35 PST, lock sequencing: four other sessions were queued behind this run), the run was stopped at 09:35:54
after the 0.5B half completed and five Phi-3 depth-128 cells had run, and the lock released. Those five cells (`speed/p3m-interrupted-2026-11-23.tsv`:
block 1's A 9.018, B 8.046, B 8.102, A 9.037, and block 2's first A 9.014) are **set aside, not used**: Phi-3's half re-runs from its first
cell as its own same-session A B B A comparison once the lock queue drains, by day and done before 21:30, or tomorrow by day if it cannot be.
Pausing between models changes no registered rule: each model's comparison is self-contained (its own pairs, its own bar), and no cell of
one model is paired with a cell of the other.

## Amendment: Phi-3 not run (Francis, 2026-11-23, relayed by the coordinator at 10:39 PST; the relay was stamped ~11:00), written 10:40 PST before any further run

Francis chose option (a). **Phi-3 mini's half is not run, speed or fidelity.** The reason is the mechanism in the running notes: CPU
single-token decode on an f16 cache widens each layer's whole history into f32 scratch on one goroutine every token
(`KVCache.dequantGlobalLayer`), about 10 bytes moved per element against f32's 4, so f16 cannot win on CPU decode at any depth on this read
path. The five Phi-3 cells that ran before the pause stay set aside. **`auto` stays f32 on the CPU.** What still runs from this
registration: the 0.5B's fidelity gate (`TestCPUKVF16GateVsReference`, cell `Q05:3900`, bars and preconditions as above), so the record
holds f16's fidelity for the follow-on below.

## Profile: attention's share of CPU decode at depth 3,900 (measure-first gate), registered 10:40 PST before the run

Francis, same decision: before building a fused f16 read into CPU attention, measure whether attention is enough of decode to be worth it.
- **Instrument:** `TestCPUAttnShareProbe` (`decoder/`): Phi-3 mini q4k on the CPU, the current f32 KV path (`auto`), prefilled to 3,900 by
  the benchmark's batched path, then 32 greedy decode tokens with the CPU cost counters on. **Share** = the `attention` operation's time
  (the q·k and p·v loops over the cache and softmax) ÷ the decode wall time.
- **One run, under the exclusive lock**, load settled under 1.0 at its start, start load and memory recorded. Vacuity: the report must
  hold a timed attention operation.
- **Kill:** share under 25%: stop, record it, build nothing. Share 25% or more: build the fused f16 read (attention reads the f16 rows in
  place inside the parallel head workers; aikit fused f16 dot and axpy kernels on amd64 and arm64 with scalar fallbacks), under its own
  registration, committed before any timed run.

## Fidelity: the 0.5B at K = 3,900, read 2026-11-23 13:03 PST — PASSES

`TestCPUKVF16GateVsReference`, cell `Q05:3900`, set B (10 prompts × 64 positions), binary at `d1cdda1bb`, 12:42–12:54 PST under the shared
lock. Log: `q05-fidelity.log`. Preconditions held: the arms' caches are f16 and f32, f16's rows differ from f32's on 10 of 10 prompts, A/A
bit-identical over 64 rows.

| | f32 (auto) | f16 | bar | |
|---|---|---|---|---|
| (a) hard flips | 29 | 26 | ≤ 29 + 2√29 | pass (strict too) |
| (b) mean agreement | 86.72% | 87.50% (+0.78 pt) | ≥ f32 − 1.0 pt, f16 ≥ f32 on ≥ half | pass, 9/10 prompts |
| (c) mean KL | 0.160735 | 0.164572 | ≤ 1.10× | pass, 1.024× (not in the 1.05–1.10 parked band) |

**Host bytes**, read from the caches at depth 3,963: f32 97,394,688 B of K/V, f16 48,697,344 B (exactly half).

## Profile: attention's share of decode, read 2026-11-23 13:03 PST — 65.9%, over the 25% kill: the build goes ahead

`TestCPUAttnShareProbe`, Phi-3 mini q4k, CPU, f32 KV, depth 3,900, 32 tokens, binary at `d1cdda1bb`, 12:56–13:03 PST under the exclusive
lock (start 1-min load 0.93, 60.6 GB available, swap 910 → 860 MiB, peak RSS 6.15 GB). Log: `profile.log`.

| operation | ms/token | share of decode wall |
|---|---|---|
| attention | 210.97 | **65.9%** |
| ffn | 67.07 | 20.9% |
| proj_qkv | 28.29 | 8.8% |
| proj_o | 8.54 | 2.7% |
| lm_head | 3.74 | 1.2% |
| the rest | 1.03 | 0.3% |

Decode 320.22 ms/token (3.12 tok/s); the timed operations sum to 99.8% of wall. Attention reads 3,081 MB a token, so it runs at about
14.6 GB/s: it is not at this box's memory bandwidth on f32, which bounds what halving its bytes alone can buy. The fused f16 read's own
registration says what it must show.
