# Task: windowed KV on the CUDA resident path (2026-10)

> **Status: DESIGNED 2026-10-10, not started.** Origin: Mellum2.1 follow-up C (`docs/tasks/task-mellum21-2026-10.md`, "C. What is the prize for a windowed-KV plan on CUDA?"):
> fully resident decodes **1.206x** faster than `--moe-cache-experts` (C') at ctx 2048 on the same 8 GB card (87.5 against 72.6 tok/s), bar 1.15, and fully resident
> only loads up to ctx 2048 because the plan prices KV for all 28 layers although 21 are windowed to 1024. This doc is the design, the gates and the staging. No code yet.

## 1. What is wrong now

`cuda/backend.go` (`BuildResident`, the KV block) allocates every attention layer's K and V as `ctxCap * kvDim`, windowed or not, and `kvBytesForCap` (`cuda/resident.go`), `ctxForSlots` and `checkKVFits`
price the same. A sliding-window layer (Mellum2.1: 21 of 28, window 1024) only ever reads the last `window` positions, so at ctx 16384 it holds 16x what it can use:
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
| **G-W2** the real model | night | Mellum2.1 int4mix: ctx 2048 greedy text, windowed against full, 3 prompts x 128 tokens (follow-up C's driver, sessions ABBA); and ctx 16384: windowed loads fully resident where full-KV declines | texts identical across arms and across sessions; the 16384 load succeeds and decodes | fixed, one pass | about 15 min |
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
- The 1.206x is at ctx 2048; at the 16384 a harness needs the prize is an extrapolation (attention cost differs), which is why G-W2 measures 16384 as a load-and-decode, not a speed claim.
