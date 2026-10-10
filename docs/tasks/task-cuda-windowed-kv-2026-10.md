# Task: windowed KV on the CUDA resident path (2026-10)

> **Status: BUILT 2026-10-10 (increments 0-3), day gates green, night gates G-W2/G-W3 queued; default-on is the owner's decision after them.** Origin: Mellum2.1 follow-up C (`docs/tasks/task-mellum21-2026-10.md`, "C. What is the prize for a windowed-KV plan on CUDA?"):
> fully resident decodes **1.206x** faster than `--moe-cache-experts` (C') at ctx 2048 on the same 8 GB card (87.5 against 72.6 tok/s), bar 1.15, and fully resident
> only loads up to ctx 2048 because the plan prices KV for all 28 layers although 21 are windowed to 1024. This doc is the design, the gates and the staging; section 7 records what was built.

## 1. What is wrong now

`cuda/backend.go` (`BuildResident`, the KV block) allocates every attention layer's K and V as `ctxCap * kvDim`, windowed or not, and `kvBytesForCap` (`cuda/resident.go`), `ctxForSlots` and `checkKVFits`
price the same. A sliding-window layer (Mellum2.1: 21 of 28, window 1024) only ever reads the last `window` positions, so at ctx 16384 it holds 16x what it can use (but see section 9: windowed KV does NOT make 16384 fit this card):
about 1.8 GB of KV against about 0.55 GB needed, on a card whose weights leave about 1 GB. The kernels already know the window (`winStart`, `Ly.window`, `nWin`).

## 2. The mechanism: a shifted view, no kernel change

Every attention and cache-write kernel addresses K and V by **absolute position** (`kc[pos * kvDim + ...]`) and is handed `winStart` / `nKeys`. Give a windowed layer a
buffer of only `cap = window + slack` positions, hold a per-layer **base** (the absolute position stored at physical slot 0), and hand every kernel
`kc.At(-base * kvDim * 4)`: the same device pointer minus the bytes of the positions that are gone. Every address a kernel computes for a position in
`[base, base + cap)` lands in the physical buffer; no position outside it is ever read (the window floor `winStart >= base` is kept by construction) or written.
`aikit/gpu.Buffer.At(byteOff int)` already binds a view at an offset; `uintptr(byteOff)` of a negative offset wraps, so the bound pointer is the shifted one. (To be proven
before anything else, increment 0: a one-kernel test that a view at a negative offset reads and writes the right bytes.)

When the next write would pass `base + cap` (decode: position `pos`; prefill: the last row of a chunk), **compact**: device-to-device copy the live tail
`[pos - window + 1, pos)` to physical slot 0 and set `base = pos - window + 1`. With `slack >= max(prefill chunk, verify width, 64)` this runs once per `slack` tokens
(about every 512 for the default chunk): a copy of `window * kvDim` bytes per windowed layer, noise against a token.

## 3. What has to change (the whole list, from a read of the tree)

- `cuda/backend.go`: the KV allocation (about line 1530): `cap_l` per layer; the K=V and KV-shared aliasing (`kvSrc`); `allocKVSlot` (MC1 slots, each with its own base).
- `cuda/resident.go`: the 20 `r.kc[l]` / `r.vc[l]` uses become `r.kview(l)`; `UploadKV` (the CPU-prefill bridge: write only the tail that fits, set base), `TruncateTo`
  (a no-op today because the cache is positional: a rollback of up to `slack` positions stays valid, further does not), `Reset`, `UseKVSlot`.
- `cuda/prefill.go`, `cuda/flash_decode.go`, `cuda/drafter.go` (its own `kc` for its layers): the remaining uses.
- Pricing: `kvBytesForCap` / `ctxForSlots` / `checkKVFits` price a windowed layer at `min(ctx, cap)` (`kvBytesForCap` charges every layer `cap * kvDim * 2 * 4` today). The decoder side already
  prices windowed layers: `kvBytesForCtx` in `decoder/arch.go` sums `kvPositionsAt` per layer, and its CPU cache keeps a ring per sliding layer (an f32 ring holds `2 * W` once wrapped, because decode reads a
  mirror in place), so this is the CUDA catching up, not a new idea.
- **Prefix reuse** (`residentReuseLen`, the agent loop's whole point): reuse of a prefix `P` is valid for a windowed layer only if `P - window >= base` (the oldest position still held).
  The resident reports this as a reuse **floor**; the decoder caps reuse at it and prefills the rest. Typical agent turns (the new prompt is the old prompt plus a short suffix) stay inside
  the slack; a violation costs a longer prefill, never a wrong answer.
- Declined, not built: CUDA graphs (the shifted pointer changes with `base`; graphs are opt-in and off by default, and `graphsSafe` already declines them under conditions), mixed window
  sizes across layers (Gemma 3/4 have per-layer widths: decline unless all windowed layers share one window), MLA and DeltaNet layers (no K/V cache or a latent one: unchanged), the speculative
  `flashDecode` multi-row path until its `faRowRuns` read the view (increment 3).

## 4. Gates (pre-registered; tier, instrument, stopping rule, cost basis per TE11)

| gate | tier | instrument | bar | stopping rule | cost |
|---|---|---|---|---|---|
| **G-W0** the view works | day | one kernel read/write through `At(-n)` on the card, plus the same on a fresh context after a compaction copy | byte-exact | one run | seconds |
| **G-W1** identity on a windowed fixture | day | `mistral-tiny-window` (and a tiny Mellum) resident on CUDA, windowed KV against full KV, same process: decode 1,500 tokens (past the window), batched prefill crossing the window in chunks, forced compaction at an odd boundary, a reuse inside and outside the slack | **logits byte-identical** at every step; a single differing bit is a failure | fixed | minutes |
| **G-W2** the real model | night | Mellum2.1 int4mix: ctx 2048 greedy text, windowed against full, 3 prompts x 128 tokens (follow-up C's driver, sessions ABBA); and the load claim (REVISED 2026-10-10 before any graded run, section 9: ctx 4096, not 16384): windowed loads fully resident where full-KV declines | texts identical across arms and across sessions; the 4096 windowed load succeeds and decodes, the full-KV 4096 control declines | fixed, one pass | about 15 min |
| **G-W3** no speed cost | night | the same ABBA driver, windowed against full at ctx 2048 | median ratio within 0.98-1.02; ratio below 0.97 is a FAIL (compaction or the view is costing) | fixed | about 10 min |
| **G-W4** pricing is truthful | day | a plan test: priced KV bytes equal the allocated bytes (the `allocation rounding` lesson: price what cuMemAlloc rounds to) | equal | fixed | seconds |

Ship rule: G-W0, G-W1, G-W4 by day and G-W2, G-W3 on a night, all green. Default-on is an **owner decision after** (it changes memory use at every load of a windowed model on CUDA); until then it is an
`Options` field (not an environment read: `testdata/env_reads.txt` only shrinks), off by default.

## 5. Increments

0. **G-W0** and the view helper (`kview`, `kvBase`), no behaviour change.
1. Allocation, pricing (G-W4), decode, `UploadKV`, `Reset`, compaction; everything else declines to full KV with a named reason. G-W1 for decode.
2. Batched prefill (text and image) and the reuse floor.
3. Verify rows (`ForwardN`, flash-decode multi-row), MC1 slots and MC3, the drafter's cache.
4. Real-model gates (G-W2, G-W3) at night; then the owner's default decision.

Increments 1-3 are each a day's work for someone who knows the file; the identity gate is what makes each safe to land alone.

## 6. Not in scope / open

- A Metal or WebGPU version: not looked at here; the shape (a shifted view over `Buffer.At`) is CUDA-specific.
- Follow-up C's flag is separate and **not fixed by this**: two cold C' sessions of the same build gave different greedy text on Mellum2.1 (and neither matched the resident arm). It decides whether
  the C' arm's 72 tok/s is a fair baseline, not whether this is worth building (the resident arm is deterministic).
- The 1.206x is at ctx 2048; at the 16384 a harness needs the prize is an extrapolation (attention cost differs), which is why G-W2 measures its load claim as a load-and-decode, not a speed claim (and at 4096, not 16384: section 9).

## 7. What was built (2026-10-10) and where it differs from the design

Scope: increments 0-3 in one pass. `Options.ResidentWindowedKV` / `--windowed-kv`, off by default; the option grid classifies it
(`TestOptionGrid_everyOptionClassified`). Code: `cuda/kvwindow.go` (the view, `kvEnsure`, compaction, `ReusableKV`), `decoder` pricing
(`cudaKVBytes`, `chooseCtx`, `ResidentWindowedKVCap`, `WindowedKVSlack` = 512) and the reuse floor (`ResidentReusableKV`, `residentReuseFloor`).

Differences and decisions the design left open:

- **Engagement.** Windowed KV engages in `BuildResident` only when the option is on, a layer is a sliding-window attention layer (not MLA, not DeltaNet),
  and the context exceeds window+slack; every windowed layer must share one window (`SlidingWindowResident`), so a layer with another window keeps
  full KV. A KV-shared layer takes its source's flag. Below window+slack a layer would hold the whole context anyway, so nothing changes.
- **Where the base lives.** One base per KV slot (`kvBases`), the bound slot's cached in `kvBase`; `UseKVSlot` and `stepAttnRows` swap it with the
  buffers. `kvEnsure` runs at the top of `launchToken` and inside `prefillCore`'s executor body (per row slot for an MC3 step).
- **Declines instead of errors.** A pass wider than the slack allows (an image block, a verify run wider than 513 rows) returns `errPrefillDeclined`
  (`kvRoomFor`), and the callers already fall back (sequential decode loop; CPU prefill + `UploadKV`). `prefillChunked` caps its chunk at slack+1.
  A write whose window history was compacted away is an ERROR (`kvEnsure`), never a stale read; `ReusableKV` is how a caller learns it first.
- **Reuse floor.** `residentReuseLen` and `residentAcquireSlot` call `residentReuseFloor`: a prefix shorter than base+window-1 is reused as 0. A continuation
  of the committed sequence is always reusable (its next row's window starts at or above the base); an edit that rewinds past the base prefills cold.
- **UploadKV.** The CPU-prefill bridge keeps only the last window-1 positions of a windowed layer, at physical slot 0, and sets the base.
- **Pricing.** `kvBytesForCap` is piecewise linear now; the trims use `kvCapForBytes` (a search) when any layer is windowed; the old linear arithmetic is
  untouched otherwise. The compaction scratch ((window-1) x the widest windowed kvDim floats) is a build allocation covered by the margin, not priced
  per slot; it is one buffer per resident, shared by every slot.
- **Graphs** are declined when windowed KV engages (`r.graphs` requires `!r.kvWin`).
- **Drafter** (`cuda/drafter.go`) keeps its own caches and reads no target KV: unchanged.

### Day gates (G-W0, G-W1, G-W4), 2026-10-10, `go test -tags 'cuda goinfer_testhooks' -run TestWindowedKV ./cuda/` and `go test -run TestWindowedKV ./decoder/`

Slack lowered to 24 (`kvWindowSlack`, a test seam) so the 512-position production slack does not hide compaction on fixtures with a 512 context.
Fixtures: `mistral-tiny-window` (all layers windowed, window 16), `gemma3-vl-tiny` (window 16, global layer 1), `cohere2-tiny` (window 8, 3 of 4 windowed).
Every test compares the windowed load against the same load without the option, logits bit for bit, in one process:

| test | what it drives | result |
|---|---|---|
| `TestWindowedKV_view` (G-W0) | a kernel through `Buffer.At(-n)`, and a compaction copy | PASS |
| `TestWindowedKV_decodeIdentity` | 400 single-token steps, ~16 compactions | PASS x3 fixtures |
| `TestWindowedKV_prefillIdentity` | prompts of 20, 41, 97, 250 in slack+1 chunks, then 60 decode steps | PASS x3 |
| `TestWindowedKV_verifyIdentity` | `ForwardN` of 1-9 rows with partial-accept rollback, then a 34-row pass that declines to the sequential loop | PASS x3 |
| `TestWindowedKV_reuseFloor` | `ReusableKV`; a forward below the base is an error; rollback within the slack is fine; position 0 restarts | PASS x3 |
| `TestWindowedKV_slotsIdentity` (MC1) | three slots at different paces, each its own base | PASS x3 |
| `TestWindowedKV_stepIdentity` (MC3) | `StepBatch` of 2-4 rows at depths 9/70/130/33 | PASS x3 |
| `TestWindowedKV_uploadKV` | the full resident's K/V uploaded into a fresh windowed one, then 150 decode steps | PASS x3 |
| `TestWindowedKV_agentTurns` | `Model.Generate` over 8 continuing turns, then an edited turn: same ids, reuse equal on continuations, 0 on the edit (the full load reused 20) | PASS x3 |
| `TestWindowedKV_allocation` (G-W4, cuda half) | a windowed layer holds window+slack positions; `kvBytesForCap` equals the allocated bytes | PASS |
| `TestWindowedKV_planPrices`, `_planChoosesLongerContext`, `_gatesOnLayer` (G-W4, decoder half) | `ResidentKVBytes("cuda")` per layer; a budget that fits 20k full-KV positions keeps ctx 200000 windowed | PASS |

Can the gate go red? With the compaction copy back into slot 0 removed, decode, prefill and verify identity fail at position 40 (first compaction); with the
slot-base swap removed from `UseKVSlot` or from `stepAttnRows`, the slots and the step tests fail. The full `./cuda/` package ran green except
`TestS10DeepstackPrefillCUDA_tiny` (untracked fixture absent from a fresh worktree; pre-existing) and one run of `TestResidentDenseBytes_matchesCUDADevice`
that read 8 MB under on a loaded box and passed alone (a device-memory reading, no KV in it).

Not covered by a day test, and why: flash-decode and split-KV at depth (they engage at >= 2048 keys; the tiny fixtures stop at 512, so G-W2's long prompt is
the first time they run through a view); the image-block prefill (declines by `kvRoomFor`, tested only through the generic wide-pass decline); MLA/DeltaNet layers
(never windowed).

## 8. Night gates G-W2 / G-W3, pre-registered

Script `docs/measurements/cuda-windowed-kv-2026-10/run-windowed-kv-gates.sh`, driver `windowed_kv_gate.py` (its docstring is the registration), pinned binary
`~/goinfer-bench/windowed-kv/serve-cuda` (rev beside it). Differences from the table in section 4: the speed gate has two classes (short prompts: the view's
overhead at base 0; one 1500+ token prompt decoding 384 tokens: compaction runs, and flash-decode / split-KV see a view), and the load session is at ctx 4096 (section 9)
and is a load-and-decode claim. Bars unchanged: texts identical (any difference FAILS); W/F per class in [0.98, 1.02] PASS, < 0.97 FAIL, 0.97-0.98 ambiguous -> parked,
> 1.02 reported as faster. The job's speed verdict is the worse class. VOID: a resident arm not cuda-resident, a W arm without the windowed line, a long prompt
under 1500 tokens.

## 9. The 16384 claim was wrong (found by the smoke, 2026-10-10)

Sections 1 and 4 said windowed KV would let Mellum2.1 load fully resident at ctx 16384 on the 8 GB card. It cannot. A smoke of the night driver (exploratory, not a
result: `serve --ctx 16384 --windowed-kv`) declined with `resident context 16384 positions needs 0.60 GB of KV (35.9 KB/position across 28 layers) but only 0.66 GB is
free on the device beside the weights (plus 384 MB reserved ...)`. The arithmetic was available from follow-up C's own numbers: 112 KB/position over 28 layers is
about 4 KB per layer-position (K and V, f32); windowed layers cost 21 x 1536 layer-positions, but the 7 full-attention layers still cost 7 x ctx, so at 16384 the
KV is 0.57 GB against 0.66 - 0.384 = 0.28 GB usable. The ceiling on this card is about ctx 5400 (the Plan picks it); the 21 sliding-window layers were never the
dominant term at 16384, the 7 global ones are. What the lever buys here is 2048 -> about 5400 fully resident, not 16384; reaching 16384 needs the global layers' KV
to shrink too (f16/i8 KV on CUDA is not offered: `KVPrecision` is ignored by CUDA, f32 always) or a smaller reserve.

Consequences: the registered load claim moved to ctx 4096 before any graded run (the full-KV plan declines there, follow-up C; windowed prices 244 MB against ~276 MB
usable), with the full-KV control session recorded in the same job. Follow-up C's "at 16384 a harness needs" extrapolation stands unchanged as a statement about
C', and its note that the lever is "untested" at 16384 is now answered: windowed KV alone does not fit it.

Exploratory smoke timings (single session, 2 short + 1 long request per arm, NOT a result): full KV 91.1 / 90.8 / 87.6 tok/s, windowed 87.3 / 88.2 / 84.9 tok/s (about
-3%). G-W3's bar (0.98-1.02, FAIL below 0.97) will say whether that is session drift or the view's cost.
