# CUDA `--kv i8`, S6: int8 KV on the tensor-core prefill tiles, registered before any code (nobara-sb01, 2026-11-08 09:50 PST)

**Why.** Francis, 2026-11-08 09:30, with S4's speed reading in view: fix the int8 prefill next, because long context is int8's whole point.
S4 (`s4-registration.md`, "Speed results") measured a cold first request with an int8 cache **20x to 26x slower than f16 at 16,384 keys**
(7B: 314 s against 16 s). An f16 or f32 cache prefills through the fused tensor-core tiles (`attn_fused.cu`, `attn_fused_bm.cu`). Those have
no int8 build, so an int8 cache is left without them (`backend/cuda/backend.go`, the `!r.kvI8` at the tile load) and prefills through
`attn_batched` or the row-batched split-KV kernels.

**The change.**
- The tiles stage K and V into shared memory as f16 and run `mma.sync` m16n8k8 on f16. They read the cache only at that staging:
  - `attn_fused.cu` through `KV_LDH`;
  - `attn_fused_bm.cu` through 16-byte vector loads (`stage_tile_vec`, `KV_VW = 16 / sizeof(KV_T)`) and its scalar edge path.
- An int8 build (`-DKV_I8`) stages each element as `__float2half_rn((float)c * sc)`, with `sc` the element's (pool row, KV head) scale from
  the K or V scale pool (`KV_SCALE_PARAMS`).
  - A new macro `KV_LDHS(p, sc)` in `kvtype.cuh` expands to `KV_LDH(p)` in the f32 and f16 builds, so their PTX is byte-unchanged.
  - The vector path gets an int8 unpack of 16 lanes, one scale a row.
- Every other operation in the tile is the f16 build's, in the same order.
- `_kv8` PTX twins of both modules are built with `build_ptx.sh` at each base module's own recorded NVRTC
  (`TestKV8_glueTwinIsBuiltAtTheAuditedToolchain` gets the two pairs).
- The tile load site binds them on an int8 cache, and the launch sites pass the layer's scale pools (`kvScaled`). Nothing else in prefill
  changes: the quantizing store (`kv8_store`), the split-KV and `attn_batched` routes, decode, and pricing.

**Not in scope:**
- The windowed ring and other families (S3c and S5 stay parked).
- An image block (prefill already routes it around the tile).
- Any `auto` change: int8 stays opt-in.

## Gates (in this order)

### G1, exact-input identity (tier 0, hard; CI with a device)

`TestKV8_exactInputIdentity_fusedPrefill`, on the shape of `TestKV16_exactInputIdentity_prefillReaders`:
- An f16 resident and an int8 resident of the same fixture are loaded. The int8 cache's K/V are drawn as int8 codes with power-of-two and
  non-power-of-two scales. The f16 cache holds `f16(code * scale)` of the same rows, computed on the host in f32 and then rounded to nearest
  even, as the kernel does.
- Every fused variant (`attn_fused` hd64 and hd128, `attn_fused_bm` 32x64, 32x32 and 128 at both head dims) on the int8 cache must equal
  its `_kv16` twin on the f16 cache **bit for bit**. The test uses a ragged tail tile and at least two key tiles, so that the vector path
  and the scalar edge path both run.
- The f32 and f16 modules' PTX is byte-unchanged (`TestKVTwins_f32ModulesUnchanged` and the build check).
- Plants, each red: the scale ignored; the K scale used for V; the scale from the wrong head (`kvh + 1`); the vector path's unpack
  shifted one lane.

### Route (tier 0, hard)

On `llama-attnfa-tiny` at `-kv i8`, a prefill above the fast-prefill floor launches the fused tile (`fastAttnLaunches` > 0), and no
`attn_batched` launch is made for the layers the tile serves. A plant restoring the `!r.kvI8` exclusion is red.

### G4, fidelity (published; bars unchanged from `s4-registration.md`)

- **Cells:** D7 (qwen2.5-7b, K=8000) and S (qwen2.5-coder-1.5b, K=3900), with the same instrument and the same three arms. The f32 and f16
  arms take the tile already, so only the int8 arm's prefill route changes.
- **Tree:** the pinned `7545a43cf` with this stage's commits on top. They touch `backend/cuda` and tests only, so set B's references stay
  current; the gate's own currency check refuses them otherwise. If the commits do not apply there, the references are rebuilt at that
  tree first (about 4 h 40 min of CPU, a night job by the rules), and this file says so before the run.
- **Bars:** S4's criteria (a), (b) and (c), and its ambiguous band. The reading replaces S4's in the docs and the banner if it moves.
  Ambiguous is parked.

### Speed (by day, under the timing lock)

- **Instrument:** `scripts/bench_peer.py`, two arms of one serve binary built from this stage. `goinfer` is the default (`auto`, f16 on these
  families); `goinfer_old` is `-kv i8`. The order is A B B A, two runs per arm per pass, two passes in reversed order, the void rule
  applied. The prompts are S4's (`prompts-s4-added.json`). Every load is fit-checked first by serve's banner.
- **Cells:** the 1.5B and the 7B, at 128, 512, 4,096 and 16,384 keys.
- **Not re-measured:** S4's int8 cells (the old route) are the before arm, cited, not re-run.

**Bars:**
- **The decision cell, the cold first request (bench_peer's warm-up: the full prefill plus 64 tokens) on the 7B at 16,384 keys,
  int8/f16:**
  - **PASS at or under 1.20.**
  - **PARKED from 1.20 to 1.50.**
  - **FAIL over 1.50.**
- **Reported with the same bands:** the 1.5B at 16,384, and both models at 4,096.
- **Control (hard):** decode at 128 and 512 keys within 0.02 of S4's int8/f16 ratios (0.871 / 0.723 on the 1.5B, 0.954 / 0.883 on the
  7B). The decode kernels are not touched. A ratio outside that stops the reading until the mechanism is found.

**Ship rule:**
- The tile becomes int8's prefill route if G1 and the route gate hold **and** the 7B's 16K cold request is at least 2x faster than S4's
  int8 (under 157 s), whatever its band. The band decides the docs' wording: "near f16" only on a PASS.
- A FAIL is still shipped if it is 2x faster than today. The record then names the remaining gap, and a follow-up needs its own
  registration.

**Predictions:**
- G1 holds (0.9).
- The 7B's 16K cold int8/f16 reads in 1.0 to 1.2 (0.6): the extra cost is one scale load a key row and the int8 unpack, beside the same
  mma work.
- D7 stays PARKED (0.6) and S stays DOES NOT PASS (0.8): the tile rounds int8's dequantized values to f16, which the f32 and f16 arms'
  tiles already do to theirs.

**Cost:**
- G1 and the route gate: seconds.
- PTX twins: minutes.
- G4: about 17 min, S4's measured 1,014 s; the int8 arm should now be quicker.
- Speed: about 40 min, S4's measured wall. The 16K int8 cells should shrink.

**Timeline:** by day on 2026-11-08, start and expected finish stated in the status file before each timed step. One timed run on the box
(the timing lock).

## Readings, 2026-11-08 (nobara, RTX 2070 SUPER, driver 595.112.07)

### G1 and the route gate: PASS (tier 0, on `f457c8a0`)
- **G1** (`TestKV8_exactInputIdentity_fusedPrefill`): every fused variant's `_kv8` twin, on an int8 cache written by a real prefill,
  equals its `_kv16` twin on `f16(code * scale)` bit for bit. That covers `llama-attnfa-tiny` (hd128) and the 0.5B (hd64), at
  (startPos, M) = (0, 70), (0, 300), (37, 200), (100, 129).
  - Plants, each red: the scale ignored; V staged with K's scale; the next head's scale; the vector unpack shifted one lane
    (`s6-g1-plant-*.log`).
  - The f32 and f16 modules' PTX is byte-unchanged (only the two `_kv8` files are new). `TestPTX_matchesSourcesAndBindings`, the toolchain
    banner test and `TestKV16_exactInputIdentity_prefillReaders` pass.
- **Route** (`TestKV8_everyAttentionPathStaysCloseToF32`, fast-prefill cell, the 1.5B): the int8 arm launched the tile 28 times. Prefill
  cosine against f32 is 0.9963, min 0.9708 over 9 rows, against that test's 0.97 smoke bar: a thin margin, recorded. Restoring the int8
  exclusion at the tile load turns this and G1 red (`s6-route-plant.log`).

### G4, on `12eedab5` (the pinned `7545a43cf` + the S6 commits; reference source unchanged, set B current)

The gate binary passed every precondition. The run started at a 1-minute load of 2.50, a fidelity cell, not a timed one. It took 355 s:
D7 275 s, against 895 s in S4, because the int8 arm now prefills on the tile. Logs: `s6-g4-12eedab5/gate.log`.

| cell, int8 against f32 | agreement | hard flips | KL ratio | verdict | S4 (old route) |
|---|---|---|---|---|---|
| D7, qwen2.5-7b, K=8000 | 92.19 % vs 93.12 % (-0.93 pt), int8 >= f32 on 7/10 | 14 vs 11 (bar 17.6) | 1.0800 | **AMBIGUOUS, PARKED** | PARKED: -0.15 pt, 11 flips, 1.078 |
| S, qwen2.5-coder-1.5b, K=3900 | 88.28 % vs 91.09 % (-2.81 pt), 5/10 | 27 vs 9 (bar 15) | 1.6881 | **DOES NOT PASS** | DOES NOT PASS: -1.40 pt, 19 flips, 1.661 |

- D7's (b) also sits inside its last 0.2 pt (-0.93 against -1.0), so it is ambiguous on (b) as well as (c).
- The controls reproduce S4's exactly: f16 against f32 is 1.0569 (PARKED) at D7 and 1.0057 (PASSES) at S.
- int8 against f16 is inside every bar at D7 (1.0219).
- **Reading:** int8 through the tile is a little lossier than through `attn_batched` at both cells.
  - The tile rounds each dequantized K/V value to f16 before the mma. `attn_batched` reads the dequantized value in f32.
  - The f32 and f16 arms' tiles round too, which is why the controls do not move.
  - No verdict changes. The published wording stays: about f16's quality on the 7B at 8K keys, a measurable loss on the 1.5B.
- **Predictions:**
  - "D7 stays PARKED (0.6)": right.
  - "S stays DOES NOT PASS (0.8)": right.
