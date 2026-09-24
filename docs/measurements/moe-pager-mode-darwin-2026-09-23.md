# S5 — MoE pager mode on darwin: pool vs mmap, real M35, local disk, 2026-09-23

This is the measurement the S5 rule (`docs/tasks/task-never-swap-2026-09.md`) registered, run for
real. It supersedes the correctness-only SMB run (`moe-pager-m35-smb-2026-09-23.md`, whose timings
were void); every number here is from a checkpoint on **local disk** (`~/models`, pulled with
`models-pull`, byte-identical to the archive: 22,146,332,580 bytes). It also carries S4 item 4's
GC-at-scale check and the S4 item 5 predictor check, because one M35 run produces all three.

**Provenance.** MacBook (darwin/arm64, 16 GB) · goinfer `79dddbcc` + the uncommitted changes in this
commit (see "What changed") · `cmd/serve` CPU build · `qwen3.6-35b-a3b-int4.giw` (canonical int4, NOT
the rule's "kind 5" row4 layout — that would need a local 22 GB transcode from the `.gguf`, which does
not fit this disk; stated deviation) · `-stream-weights -weight-cache 1.5 -ctx 1024` · greedy, 32
tokens, ~110-token prompt (depth ≈ 128, the prompt ends mid-sentence so the model keeps generating) ·
harness `scripts/moe_pager_ab.py` (copy in the data directory), one server start per run.
**Budget fixed at S4's live figure:** a load-only probe with `-weight-cache` unset resolved **1.5 GB**
(live-available 2.7 GB on this machine at that moment), and that figure was used for every arm.
`GOMEMLIMIT=off` for every A/B arm (see the S4 item 4 result below for why).

**Guards on every run:** S3's in-process tripwire (armed by default in `goinfer-serve`) and the
harness's own out-of-process kill switch (swap-used > baseline + 1 GB). Neither fired in any run.

**Order:** pool, mmap, mmap, pool, pool, mmap — interleaved and balanced so order and page-cache
warming cancel across arms; 20 s settle between runs. n = 3 per arm.

## Result 1 — S5 rule: pool vs mmap at an equal budget

| run | mode | decode tok/s (token 1→32) | time to first token | swap-used Δ | page-ins | dirty footprint @ token 32 | file-backed clean mapped @ 32 |
|---|---|---|---|---|---|---|---|
| r1 | pool | 2.315 | 28.7 s | 0 MB | 277k | 6147 MB | 1129 MB |
| r2 | mmap | 2.290 | 29.4 s | 0 MB | 1285k | 4989 MB | 4094 MB |
| r3 | mmap | 2.014 | 33.9 s | 0 MB | 1445k | 4987 MB | 3749 MB |
| r4 | pool | 2.130 | 30.1 s | 0 MB | 343k | 6392 MB | 1096 MB |
| r5 | pool | 2.372 | 27.1 s | 0 MB | 277k | 6417 MB | 1148 MB |
| r6 | mmap | 2.363 | 28.4 s | 0 MB | 1259k | 4988 MB | 4106 MB |

- **Throughput: pool mean 2.272 tok/s (sd 0.10), mmap mean 2.222 (sd 0.15) → pool/mmap = 1.02×.**
  The +2% is inside the ~4–7% run-to-run spread: the honest reading is "no measurable slowdown from
  pool mode", not "pool is faster". Bar: ≥ 0.9×. **Met.**
- **The budget: mmap mode does not enforce it.** After decoding, mmap mode holds **3.7–4.1 GB** of
  file-backed pages against a 1.5 GB budget (pool: ~1.1 GB, the non-expert part). That is the
  registered "`MADV_DONTNEED` is a no-op on darwin" premise, now measured on M35. Pool mode's cost is
  exactly what the design says: **+1.2–1.4 GB of owned anonymous buffers** (dirty footprint 6.1–6.4
  GB vs 5.0 GB) — the budget, as real dirty memory, by construction. So the trade is: mmap's excess
  is clean file-backed pages the kernel can drop for free; pool's is dirty memory that is swap-eligible
  but capped and predictable.
- **Swap: zero growth in all six runs, both modes.** mmap mode did not swap either. The rule's
  "if mmap mode's RSS also holds → repeat under a 6 GB memory hog" branch is **not triggered** (its
  RSS did not hold — see the budget bullet), and that arm was **not run**. The claim "pool is safer
  under pressure" is therefore still untested here; what is measured is "pool holds its budget, mmap
  does not, at no throughput cost, on a warm-ish page cache".
- **Page-ins** were ~4.5× higher in mmap mode (1.26–1.44M vs 277–343k) at the same decode rate, so on
  this run I/O was not the bottleneck for either arm; compute was. That will not hold on a cold
  cache or a slower disk.

**Decision (as registered): pool ships as the darwin default.** `--moe-pager` now defaults to `pool`
on darwin and `mmap` elsewhere (`moePagerDefault`, `internal/serveapp/main.go`); the banner already
names the mode (`expert paging (pread)`), and the real binary was confirmed to select it with no
flag. Reversible with `--moe-pager=mmap`.

## Result 2 — S4 item 4: GOMEMLIMIT at a scale where it binds → DROPPED

The earlier 1.5B result was a null because the limit never bound. On M35 it binds hard: the Go heap is
~5.7 GB (see "Open finding" below) and `applyGoMemLimit` chose 1.7–2.5 GB.

| arm | decode tok/s | time to first token | GC cycles in the run | cumulative GC CPU |
|---|---|---|---|---|
| `GOMEMLIMIT=off` (r1, r4, r5) | 2.13–2.37 (mean 2.27) | ~29 s | 13–14 | 0% |
| default, limit 2.50–2.52 GB (d1, d2) | **1.079, 1.079** | **70–73 s** | **8,642–8,688** | 8–9% |

The limit sat below the live heap, so the GC ran continuously (~70 cycles/s): **throughput fell 52%
and time to first token rose 2.4×.** The brief's own rule — *"kept only if ... GC CPU does not rise
more than 10% — a soft limit that costs throughput is dropped, and the reason recorded"* — decides
it. The feature is **removed** (`applyGoMemLimit`, its `Load` hook, and both tests), not made opt-in:
setting a limit below the live heap is exactly what its own doc warned against, and the "projected
live heap" it would need is not knowable at load time. `GOMEMLIMIT` set by the caller still works
(it is the runtime's own knob). n = 2 for the limit-on arm; the effect is 50%, far outside any noise.
(An earlier limit-on run returned zero tokens for an unrelated reason — my prompt, below — and is
kept only as `discarded-limit-on-zero-token.*` for its GC trace: 7,561 GC cycles in 105 s.)

## Result 3 — S4 item 5: the predictor against a real rate

Banner prediction at the 1.5 GB budget: **2.62–2.68 tok/s** (a prior; compute time deliberately
omitted, so an upper bound). Measured: pool 2.27 (0.85×; range 0.79–0.89×), mmap 2.22. **All six
runs fell below the prediction**, so the "upper bound" property held 6/6, and it over-predicted by
~15–20% on this model/budget — consistent with ~65 ms/token of omitted non-I/O time. Both prediction
and measurement cleared the 2 tok/s gate here, but two runs (2.01, 2.13) were close to it. One model,
one budget, one machine state: this checks that the number is the right order and the right side of
the bound, not that the hit-rate curve (borrowed from CUDA's C′ cache) is right.

## What did not run, and why

- **S6's M26 (N=8) Metal cell — not run.** It is the R11(c) configuration that produced three
  swap-spiral / kernel-panic incidents on this machine, and the machine's headroom is worse now than
  when I started: live-available memory was 2.7 GB during these runs and system swap-used drifted from
  ~0.7 GB to 2.3 GB over the session (these 22 GB runs push other processes' memory out and macOS does
  not give it back). Running it needs a fresh boot or closed apps first.
- **The 6 GB memory-hog arm** (not triggered, above) and **footprint snapshots on a cold page cache**.
- The rule's **"kind 5" (row4) layout** — canonical int4 was used (deviation, above).

## Open finding — ~5 GB of anonymous memory in a "streamed" M35

Both modes carry **~5.0 GB of dirty (anonymous) footprint** (mmap arm, before pool's buffers) and a
Go heap of ~5.7 GB, for a model whose weights are supposed to be file-backed aliases. The load
takes the heap to ~3.1 GB and serving a request to ~5.7 GB. Something outside the expert pager
(candidates: the qwen3.5 hybrid's non-serializable/heap-resident DeltaNet and attention weights,
the ~248k-vocab embedding/head, per-request scratch) is the dominant anonymous term, larger than
the pager budget being tuned. **Not investigated.** It is also why any Go memory limit was doomed
here and it bounds how much the pager budget can matter for swap on this model.

## Harness bugs found and fixed while measuring (so the data can be trusted)

- A first attempt returned **0 tokens in both arms**. Cause: my prompt ended "Summarize the trade-off
  in one short paragraph." and this instruct model emitted its stop token as token 1
  (`finish_reason: stop`, empty text — reproduced with `curl -N`). Not a pager or server fault; the
  prompt now ends mid-sentence, and the driver aborts on a zero-token run.
- `ps` prints `-` for fault counters on macOS; faults/page-ins come from `top -l 1 -pid`.
- The served model name for a `.gguf` drops its extension; the harness reads it from the banner.
- A crashed harness could orphan a multi-GB server; it now kills its child on any exit.

## Caveats, all of them

n = 3 per arm and one machine; the throughput difference is inside the noise; page cache was warm
(the CRC pass and prior runs had touched much of the file), which flatters mmap mode and hides I/O;
the prompt is ~110 tokens, decode timing excludes prefill; the pager budget (1.5 GB) is small
relative to the dense footprint, so its effect on total memory is modest on this model.

## What changed in the tree

`--moe-pager` default → pool on darwin (+ test, mutation-checked); `applyGoMemLimit` removed (+ its two
tests); `scripts/moe_pager_ab.py` added. Raw runs (JSON, per-2 s samples CSV, server logs with the
GC-trace lines filtered) are in `moe-pager-mode-darwin-2026-09-23/`.
