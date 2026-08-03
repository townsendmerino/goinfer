# CUDA Graphs for the resident CUDA decode — built, bounded, SAFE-GATED (2026-08-03)

**Status: shippable behind a safe-gate (was: parked).** The §8-step-1 runtime tenancy gate is now
built (`cuda/graphs_safe.go`, `admitGraphs`): `GOINFER_CUDA_GRAPHS=1` is promoted to on ONLY under
`CU_COMPUTEMODE_EXCLUSIVE_PROCESS` or active MPS (`CUDA_MPS_PIPE_DIRECTORY`), then confirmed by a
startup bit-exactness self-test (live vs replay, one forward); a DEFAULT-compute-mode box declines to
the live path with a logged reason — "byte-identical or decline, never silently mis-run" restored.
`GOINFER_CUDA_GRAPHS_UNSAFE=1` force-enables for on-box benchmarking (loud warning, self-test still
runs). Gated by `TestGraphsSafeGate` (this box is DEFAULT → declines; unsafe-override enables + the
self-test passes bit-exact, confirming the capture is correct and the divergence is churn-only).

**MEASURED, and it changes the value case: graphs give ~1.01× on the real 1.5B decode** (220.9 →
223.3 tok/s, `TestGraphsDecodeSpeedup`, greedy). The **~1.4–1.7× did NOT transfer from the tiny
model** — that number was dispatch-dominated because each tiny launch does almost no work; at real
model size the CPU dispatch **overlaps** GPU compute, so eliminating it is off the critical path.
The lever therefore only pays where dispatch actually dominates: the **26B** (30 layers × MoE's
per-token per-expert launch explosion → the ~19 ms-of-~29 ms figure that motivated this), which is
exactly the **hardware-mismatch model** (8 GB paging 13 GB) that stays disclosed-with-caveats
regardless. So: the safe-gate is correct and worth having (graphs are now safe if a dispatch-bound
deployment ever wants them), but **graphs are not the decode win for the dense flagships that fit and
that goinfer actually wins with.** Do NOT flip any default. Measuring the 26B specifically (heavy
streaming load) would confirm the MoE case but does not change the decision, since that model's
ceiling is set by memory, not dispatch.

**Original parked status (historical).** The code is committed, off by default, and the graphs-off
path is byte-identical to before. Graph replay is **~1.5× faster, and bit-exact under EXCLUSIVE GPU tenancy or
under CUDA MPS — but NOT bit-exact under time-sliced multi-context sharing** (two separate CUDA
contexts, no MPS) on this Turing box (RTX 2070 SUPER). The capture is proven topologically correct (a
DAG dump, §5); the divergence is an inter-context time-slicing effect, **confirmed on this hardware by
an MPS A/B** (§5.1): same churn, MPS-off diverges, MPS-on bit-exact ×10. Because a goinfer inference
backend must be "byte-identical or decline, never silently mis-run" and an operator cannot always
guarantee exclusive/MPS tenancy, it is not shipped — but the safe operating condition is now concrete
and testable, not "revisit on Ampere."

This document is the complete record so the investigation can be resumed cold.

---

## 1. Why graphs at all

Resident CUDA decode on the 2070 runs ~59 ms/token. The measured decomposition (see
`docs/task-moe-streaming.md` and the memory `gpu-decode-fusion` / `moe-hostvram-streaming`):

| component | ~ms/token | nature |
|---|---|---|
| compute floor | ~17 | irreducible GEMV/attention work |
| dispatch / launch overhead | ~19 | **~600 kernel launches × ~32 µs** |
| readback drain | ~12 | 30 host round-trips, fixed sync latency |
| expert-miss DMA | ~11 | host→VRAM expert streaming |

The dispatch line is the target. `cuda/launch_cost_test.go` (Step 0, committed earlier as `b2227ec`)
bounded the per-launch host cost at **~10 µs for a 1×1 grid, ~11 µs for a 512-block grid** — i.e.
**FFI/purego-bound, grid-independent**, not GPU-side dispatch. Every `cuLaunchKernel` in the cgo-free
stack is a dlopen'd-symbol crossing + Go-side arg packing. That is exactly what **CUDA graphs**
collapse: capture a static run of launches once, replay it as a **single** driver call. Batching
(fewer, larger launches) was bounded out at ~1.04× because it only cuts dispatch *count*; graphs cut
the *crossings*.

The launch-cost test also found **~96 % of per-layer launches are static** (only `rope_kv` and
`attention` bind per-token `pos`/`nKeys`), so nearly all of them are capturable.

## 2. What was built

### 2a. aikit primitive (`gpu/v0.21.0`, commit `b4fb5c1`)
A thin wrapper over gocudrv's own graph API — **no fork**:
- `Queue.Capture(issue func() error) (*Graph, error)` — `BeginCapture(ThreadLocal)` → `issue()` →
  `EndCapture` → `Instantiate`.
- `Graph.Replay() error` — `exec.Launch(stream)`, one driver call, stream-ordered.
- `Graph.Close()`.
Tested in `aikit/gpu/cuda_graph_test.go`: a captured `vadd` replays bit-identically AND reads current
buffer contents across replays (the property the whole design needs — topology fixed, contents vary).

### 2b. goinfer forward restructure (`cuda/resident.go`, `cuda/backend.go`)
The per-layer body of `launchToken` was factored into three **graph-static** segment methods, so a
single source of truth serves both paths — the live path calls them in sequence (byte-identical to the
pre-graph code), the graph path replays captured versions:

- **`segA`** — QKV projection (fused K1 super-kernel or rmsnorm+quant+GEMVs) + QK-norm + K=V v-norm.
  Ends before `rope_kv`.
- *gap (live): `rope_kv` + `attention`* — bind `pos`/`nKeys`; attention's shared-mem grows with the
  attended span. Never capturable.
- **`segB`** — context-quant + o-proj + the MLP up to the router readback. For a dense layer this is
  the whole MLP (no segC). For MoE it is `moeMLPPre` / `gemma4MoeMLPPre` (router logits → top-k, left
  on device).
- *gap (live): `g4x2` accumulator clear (H2D) + `loadRoutedExperts` routing readback (D2H) + expert
  slot DMA (H2D)* — host copies, not capturable.
- **`segC`** — the post-readback MoE half (expert loop + join). `moeMLPPost` / `gemma4MoeMLPPost`.
  nil for a dense layer.

`moeMLP` was split into `moeMLPPre`/`moeMLPPost`; `gemma4MoeMLP` into `gemma4MoeMLPPre`/`Post`, exactly
at the readback point.

Capture happens once at build (`captureGraphs`, on the executor thread, after `fuseQKV` is finalized
since segA/segB branch on it). Each MoE layer captures 3 graphs, each dense layer 2 (no segC). For the
26B that's ~48×3 ≈ 144 graphs. `cudaLayer.gSegA/gSegB/gSegC *gpu.Graph`; freed in `Close()` before
context teardown.

### Env flags (all off by default)
- `GOINFER_CUDA_GRAPHS=1` — enable capture + replay. Off ⇒ `launchToken` is byte-identical.
  Incompatible with the `g4cap` diagnostic (it syncs inside a segment); capture is gated on `!g4cap`,
  replay's `useGraphs` on `!subCap`.
- `GOINFER_CUDA_GRAPHS_SYNC=1` — DEBUG: `stream.Sync()` after each segment replay (bisect probe).
- `GOINFER_CUDA_GRAPHS_ONLY=A|B|C|AB…` — DEBUG: replay only the named segments, issue the rest live
  (segment-localization probe).

## 3. The fail-fast bound (before touching launchToken)

`cuda/graph_bound_test.go` — `TestCUDA_graphReplayBound`. Captures a representative 16-kernel static
chain, times live-issue vs graph-replay per iteration, and spot-checks bit-exactness. Result on the
tiny fixture:

```
K=16 launches/segment
  live chain:   183.87 µs/iter  (11.49 µs/launch)
  graph replay:  15.16 µs/iter  (12.13× cheaper than live)
  saved/segment: 168.71 µs
  projection (48 layers × 3 segments = 144 replays/token): ~24.3 ms/token reclaimed
  VERDICT: replay is 12.1× cheaper — crossing collapse is REAL. ~24ms off ~59ms ≈ 1.70× speedup.
```

The captured chain was **bit-exact** to the live chain. So the win is real (~1.4–1.7×, stronger than
the 1.3× Step-0 estimate which conservatively assumed only crossing-*count* reduction). The build was
justified. This is the key lesson-that-paid-off: **bound the replay win at real segment size before
the invasive refactor** — `launch_cost` had only bounded a *single* live launch, never replay.

## 4. The correctness gate and what it caught

Gate design (the sharpened version): bit-exact graph-vs-live at every position, across a spread of
attention geometries (40-token prompt crossing the sliding window), with three arms:

1. **Isolation ×20** — `TestGemma4Graphs_bitExact_tiny{,Cache,CacheReuse}`. **PASS**, byte-identical,
   cache off / on / with LRU reuse.
2. **`CUDA_LAUNCH_BLOCKING=1` control arm** — serialize every launch. **PASS**, agrees bit-exact.
3. **Concurrent GPU load from another process** — **FAIL.** Graph replay diverges from live.

Arms 1 and 2 are the ones that *look* like proof and would have shipped the bug.

**Methodology correction (important — two traps, both about serialization hiding a race):**
- `CUDA_LAUNCH_BLOCKING=1` is a **one-way** test. It *serializes* every launch, which is exactly the
  condition under which a race or a missing dependency looks correct. Disagreement would prove a bug;
  **agreement proves nothing.** Do not read arm 2 as evidence of correctness.
- The same trap recurs at the test level: any per-iteration `Sync()` between interleaved ops hides an
  inter-operation race. An early round of "clean under churn" volume micro-tests (segA ×6000,
  live→graph ×9000) were **invalid** for this reason — they synced each iteration. The tell was that a
  per-layer drain probe made the full-forward divergence *vanish*: the race is sync-maskable, so any
  test that syncs can't see it. The **valid** evidence is the off-vs-off control, the no-sync
  non-commutative ordering test, and the DAG dump (§5) — none of which rely on a masking sync.

### The divergence, controlled
`TestGemma4Graphs_sameModelUnderLoad` isolates graph-vs-live on **one** model instance (identical
weights, `cacheSlots`, buffers — no build confound): run the prompt once replaying, once live.

Under a leak-proof concurrent churn (`timeout`-bounded), back-to-back on the **same fixture, same
load, only variable = graphs on/off**:

```
CONTROL: graphs-OFF cache gate ×15 under churn   → PASS (bit-exact)
FINDING: graphs-ON  same-model ×15 under churn    → FAIL
         SAME MODEL pos 5 logit 1: graphs 0.423619986 != live 0.194403574
```

Properties of the divergence:
- **Only under concurrent multi-context GPU load.** Isolation is clean.
- **Live is canonical.** Live stays at the isolation value (e.g. 0.8337 at pos 0); graph moves
  (0.7396, 1.031, 0.4236). The graphs-off forward is bit-exact under the *same* churn (control).
- **Deterministic-ish** (specific recurring wrong values like `1.031`), not random garbage.
- **Sync-maskable.** Any sync between the interleaved ops (a per-layer drain, per-replay sync) makes it
  vanish — the fingerprint of a race, and the reason arm 2 and the synced volume tests were blind.

### The critical control (the finding is real)
`TestGemma4Graphs_offVsOffControl` — two **graphs-OFF** runs of the same 40-token prompt, compared,
under the *same* leak-proof churn where graphs-ON diverges: **bit-identical ×10**. So the live forward
is deterministic under this load; the graphs-on divergence is a genuine graph effect, not
comparison noise on a busy GPU.

## 5. Root-cause elimination — the DAG dump settles it

The earlier version of this section leaned on behavioral inference and a leading hypothesis. The
gocudrv fork + `AIKIT_GRAPH_DUMP` hook (§9) let us dump the **actual captured DAGs** and settle the
central question directly rather than infer it.

**Result — every captured graph is a strict linear chain (a total order).** For the tiny g4moe model
(2 layers × 3 segments = 6 graphs), `cuGraphGetNodes`/`GetEdges` + `cuGraphDebugDotPrint`:

```
segA: 2 nodes,  1 edge   0->1
segB: 15 nodes, 14 edges 0->1->2->...->14
segC: 11 nodes, 10 edges 0->1->2->...->10
```

Every edge is `k -> k+1`: `edges == nodes-1` AND the edge set is a path — no forks, no joins, no gaps.
So the capture is **topologically correct and fully serialized**: no missing dependency, and nothing
that can run concurrently. The "missing edge" class is refuted by direct inspection, not inference.

| hypothesis | ruled out by | verdict |
|---|---|---|
| **Missing dependency edge** (capture serialized by timing, not by an explicit edge → fewer edges than the real deps) | **DAG dump: all 6 graphs are strict `0->1->..->n-1` chains, `edges==nodes-1`, no forks/joins** | NOT it (direct) |
| Build confound (two builds get different `cacheSlots`) | `TestGemma4Graphs_sameModelUnderLoad` — one instance, toggle `r.graphs` — still diverges | NOT it |
| Inter-op stream ordering (graph replay ↔ adjacent live launch not serialized) | `TestCUDA_graphLiveNoSyncOrdering` — non-commutative live/graph chain, NO sync between, 24000 ops under churn → exact `2^20-1` | NOT it |
| Intra-graph node race | segA replayed 6000× under churn with only a trailing sync (which can't hide an intra-graph race) → clean; corroborated by the DAG being linear | NOT it |
| Kernel-arg lifetime / stale-or-mutable pointer | every device buffer is allocated in setup *before* `captureGraphs` (allocSlots precedes capture); a stale pointer would be wrong in isolation or random, not reproducible-under-contention | NOT it |
| gocudrv wrapper bug | single `c.exec` executor runs BeginCapture + all launches + EndCapture (one thread → complete capture); wrapper is a clean pass-through | NOT it |
| Stray-process contamination (the trap that bit once — §7) | re-confirmed under `timeout`-bounded churn, verified 0 leftover procs, + the off-vs-off control | finding survives |

### Conclusion: a driver/hardware execution boundary, on direct evidence
A **topologically-correct, fully-serialized** graph, replayed, diverges from the equivalent serial live
launches **only** under concurrent multi-context GPU load. With the capture proven correct, the
difference is in *execution*: `cuGraphLaunch` of a baked graph vs a stream of individual
`cuLaunchKernel`s, under context contention on this Turing card (RTX 2070, no MPS). The mechanism is
**inter-context time-slicing**: without MPS, two separate CUDA contexts (churn + decode) time-slice via
context switch; an individual launch re-establishes its state each call and survives a switch, a baked
graph replay spanning a switch window does not. This fits every observation (concurrent-context-only,
graph-specific, sync-maskable, needs the long forward, deterministic-ish), and — unlike a bare
hypothesis — it is now **confirmed on this hardware** (§5.1).

### 5.1 MPS A/B — the mechanism, confirmed on the 2070
CUDA MPS supports Volta+ (the 2070 is compute 7.5). Under MPS, clients share **one** server context, so
GPU work runs concurrently (spatial) with **no inter-context switching** — exactly the condition the
hypothesis says the replay needs. Run on this box, same leak-proof churn, same `sameModelUnderLoad`
discriminator, back to back in one session, with MPS confirmed active (`get_server_list` returns a
server PID; `nvidia-smi` shows `nvidia-cuda-mps-server`):

```
[MPS OFF]  graphs-ON same-model ×10 under churn  → FAIL (pos 0: graphs 1.031 != live 0.8337)
[MPS ON]   graphs-ON same-model ×10 under churn  → PASS (bit-exact ×10)
```

Only variable = MPS. MPS **removes** the divergence → the cause is inter-context time-slicing, not
concurrency per se and not our capture. Reproduce:
```
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
mkdir -p "$CUDA_MPS_LOG_DIRECTORY"; nvidia-cuda-mps-control -d      # start
echo get_server_list | nvidia-cuda-mps-control                     # verify a server PID prints
# ... run churn + TestGemma4Graphs_sameModelUnderLoad (both inherit the pipe dir) ...
echo quit | nvidia-cuda-mps-control                                # stop
```
So the parked result is not "driver boundary, revisit on Ampere" — it is **"graphs are bit-exact under
exclusive tenancy OR MPS; unsafe only under time-sliced multi-context sharing,"** verified here.

## 6. Decision: park dormant

The perf win (~1.5×) is real, and the code is bit-exact under **exclusive tenancy or MPS** (§5.1). The
hazard is confined to **time-sliced multi-context sharing** — but there it is **invisible** (wrong
tokens, no error) and an operator cannot always guarantee exclusive/MPS tenancy, which violates the
repo's "byte-identical or decline, never mis-run" discipline. So:

- Committed, **off by default**, graphs-off path **byte-identical** (verified: cacheExperts,
  cacheReuse, `TestRealForwardParity` real dense checkpoint all pass graphs-off).
- **Not advertised** as a feature; no README/benchmarks claim.
- Tests + this doc are the boundary markers.

Not shipped opt-in because a time-sliced-shared-GPU operator could enable it and get silently-wrong
tokens; not reverted because the `segA/B/C` refactor and the primitive are reusable and the ~1.5× is
now shippable behind a **runtime tenancy/MPS gate** (§8) rather than blocked on new hardware.

## 7. Lessons

### 7.0 The instrument that verifies a concurrency property often destroys it
The durable, generalizable finding — the concurrency analogue of "isolation proves the primitive,
never the composition." **The tools you naturally reach for to check a concurrent result are the ones
that serialize it, so they report "correct" precisely by removing the concurrency under test.** Two
errors in this one investigation, same shape:
- **`CUDA_LAUNCH_BLOCKING=1`** serializes every launch. Its PASS is not evidence of correctness — it is
  the definition of hiding a race. It is a *one-way* test: only disagreement is informative.
- **Syncing every iteration** of a micro-test to read the buffer is the obvious way to verify each
  step — and it serializes the exact inter-op interleaving whose concurrency you are testing. My first
  "clean under churn" volume tests (segA ×6000, live→graph ×9000) were invalid for this reason.

The tell here was that a per-layer *drain* probe made the real divergence vanish — a result being
sync-*maskable* means every synced test is blind to it. Valid concurrency evidence must **not** contain
a serializing sync inside the window under test: the off-vs-off control, the no-sync non-commutative
ordering test, the DAG dump, and the MPS A/B all satisfy this; the blocking arm and the synced volume
tests did not. When testing a concurrency change, list every `Sync`/barrier/blocking flag in the test
and ask which one you are hiding behind.

### 7.1 Reap background processes

During the investigation, two backgrounded churn processes (`go test … -count=100 &`) survived
`kill $CHURN; wait` and kept hammering the GPU. They then contaminated a *default-path* regression run
— `TestRealForwardParity` (real multi-GB checkpoint) hard-failed under the surprise contention (alloc
corruption, not compute divergence), which momentarily looked like the refactor had broken the
graphs-off path. It had not. **Always bound churn with `timeout`, and verify
`nvidia-smi --query-compute-apps=pid` shows zero leftovers before trusting a GPU test result.** The
controlled re-run (§4) uses `timeout`-bounded churn for exactly this reason. Also note that a
`go test … &` parent, when `kill`ed, can leave its compiled test binary child running — kill the child
PID too. And **never** `pkill -f mps` (or any short pattern): the bare substring matches unrelated
processes and can destabilize the session — use exact PIDs or a specific pattern like
`nvidia-cuda-mps`.

## 8. How to resume

0. **Already done — the mechanism is confirmed (§5.1).** MPS on the 2070 removes the divergence
   (MPS-off fails, MPS-on bit-exact ×10, same churn). The safe operating condition is exclusive tenancy
   OR MPS. Ampere+ re-validation is now optional confirmation, not the gating unknown.
1. **The shippable path is a runtime tenancy/MPS gate, not new hardware.** To promote graphs from
   parked to opt-in-safe, add a load-time check that enables `GOINFER_CUDA_GRAPHS` only when the
   process has exclusive use of the device OR MPS is active (`CUDA_MPS_PIPE_DIRECTORY` set + a live
   `get_server_list`, or a probe). Then a shared-GPU deployment without MPS declines graphs (staged/
   non-graph path) instead of silently mis-running — restoring "byte-identical or decline." Measure the
   real 26B tok/s (step 3) before flipping any default.
   The DAG-dump tooling remains available: `AIKIT_GRAPH_DUMP=1` (+ optional `AIKIT_GRAPH_DUMP_DIR`)
   logs each captured graph's node/edge counts and writes Graphviz DOT. It needs the gocudrv
   introspection bindings, which live UNCOMMITTED in `/home/francis/mycode/gocudrv-fork`
   (`cuGraphGetNodes`/`GetEdges`/`DebugDotPrint` + a `Graph.Topology()`/`DebugDotPrint()` method) and
   the `AIKIT_GRAPH_DUMP` hook in `aikit/gpu/cuda.go`, wired via a `go.work` `replace` +
   `use`. On this box the dump already proved all 6 tiny-model graphs are strict linear chains (§5).
2. **If it still diverges on Ampere+**, the capture is proven correct (linear DAG), so it is a genuine
   CUDA graph + concurrent-context execution issue worth a minimal driver-bug repro
   (`TestCUDA_graphLiveNoSyncOrdering` extended toward the failing composition) and a report.
3. **To promote to a shipped feature** (only after step 1 passes on target hardware): keep it opt-in
   with a doc caveat, OR make it default and add a load-time GPU-capability gate. Then measure the real
   26B tok/s (`cuda/gemma4_26b_cache_test.go` pattern, `GOINFER_HEAVY_TESTS`, `GOINFER_CUDA_GRAPHS=1`)
   against the ~1.4–1.7× prediction. Bit-exact gate that must pass first:
   `TestGemma4Graphs_bitExact_{tiny,scaled}` under churn on the target GPU.

## 9. File / commit map

- `aikit/gpu/cuda.go` — `Queue.Capture`, `Graph.Replay/Close` (`gpu/v0.21.0`, `b4fb5c1`);
  `Queue.UploadAsync` (`gpu/v0.20.0`, `133c87a`, banked, unused here).
- `aikit/gpu/cuda_graph_test.go` — primitive tests incl. `TestCUDA_graphDependentChain`.
- `cuda/resident.go` — `segA/segB/segC`, `moeMLPPre/Post`, `gemma4MoeMLPPre/Post`, `captureGraphs`,
  `cudaLayer.gSegA/B/C`, `cudaResident.graphs/graphsSync/graphMask`, `launchToken` loop, `Close`.
- `cuda/backend.go` — env wiring + `captureGraphs` call after `fuseQKV`.
- `cuda/graph_bound_test.go` — the fail-fast replay-cost bound.
- `cuda/gemma4_graphs_test.go` — bit-exact gate: `bitExact` (isolation), `sameModelUnderLoad`,
  `offVsOffControl` (the critical control).
- `cuda/graph_sharedmem_test.go` — kernel-class isolation (elementwise vs reduction).
- `cuda/graph_segisolate_test.go` — single-segment high-volume replay (intra-graph, trailing-sync-valid).
- `cuda/graph_nosync_test.go` — no-sync non-commutative live↔graph ordering (the VALID ordering test).
- `cuda/graph_locate_test.go` — per-layer divergence localizer (its per-layer drain also demonstrated
  the sync-maskability). Uses `cudaResident.layerCap`.
- `cuda/go.mod` — pinned to `gpu/v0.21.0` with this landing (the pin that "rides Step 2").
- **Investigation tooling (UNCOMMITTED, for the Ampere+ resume test):**
  `/home/francis/mycode/gocudrv-fork` (gocudrv copy + `cuGraphGetNodes`/`GetEdges`/`DebugDotPrint`
  bindings + `Graph.Topology()`/`DebugDotPrint()`), the `AIKIT_GRAPH_DUMP` hook in `aikit/gpu/cuda.go`,
  and the `go.work` `use …/aikit/aikit/gpu` + `replace gocudrv => …/gocudrv-fork` that wire them in.
  Revert the `go.work` edits to fall back to the module cache; the fork stays as dormant tooling.
