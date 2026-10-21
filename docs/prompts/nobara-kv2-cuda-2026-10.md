# nobara, 2026-10-14: KV2's CUDA half (the copy between slots) and the CUDA arm of every KV2 gate

For a session on `nobara-pc`. The task doc is `docs/tasks/task-kv-architecture-2026-10.md`. Read these first:
- §6 KV2, with KV1's "Left for KV2" list above it;
- §5.13's paragraph on CUDA's windowed layers;
- the registration, `docs/measurements/kv2-2026-10-14.md`. Its §1 holds the choices made before the code, and its §2
  the gates whose CUDA arms are yours.

## Status, 2026-10-14 evening: the Mac half is done; the CUDA arm starts 2026-10-15 by day

- **Every Mac arm of G0 to G7 is read and passes** (`docs/measurements/kv2-2026-10-14.md` §3, "The Mac's night,
  2026-10-14"):
  - G5 T-busy DELIVERS: 0.949 of the projection, 32.5 s saved;
  - G5 T-idle PASS: 1.902;
  - G6 PASS on all four cells.
- **KV2 merges when this CUDA arm reads.** Nothing else is outstanding on the Mac.
- **When:** 2026-10-15, by day, after nobara's release queue for v0.23.0 has finished (the CUDA GPU gate, the parity
  sweep and `quick -full`, all at `695ce16c`). Nothing non-doc merges to `main` before the tag, so build on this
  branch and merge after the tag.
- `kv3-cuda-2026-10` is merged (PR #14), so the start condition below holds. Merge `main` into this branch first.

## 0. When to start, and where to work

- **Start after `kv3-cuda-2026-10` is merged to `main`** (tonight's merge). KV2's 16-row cap and the class records
  assume KV3's chunker.
- **Wait for the seam.** The Mac builds the decoder side on `kv2-2026-10`: the interface below, the lookup, the
  publish point and the slot choice. Check `origin/kv2-2026-10` for `decoder.ResidentKVCopier`. If it is not there
  yet, do the reading in §1 and stop; do not write the decoder side here.
- **Work on your own branch,** `kv2-cuda-2026-10`, made from `origin/kv2-2026-10`. Merge `origin/main` into it once
  KV3 is in. The Mac merges your branch into `kv2-2026-10`; do not push to `kv2-2026-10` and do not merge to `main`.
  KV2 merges to `main` when every arm on both boxes is read.
- **Write your readings in `docs/measurements/kv2-2026-10-14/cuda-readings.md`,** a file only you write. The Mac
  folds them into the record's §3. Then two sessions never edit one file.
- **The timing lock:**
  - every batch of commands opens with the guard, so a held lock ends it before anything runs:
    `cd ~/mycode/goinfer && python3 scripts/timing_lock.py status | grep -q '^free' || { echo HELD; exit 3; }; <work>`
  - nothing timed by day: G5 and G6 go on nobara's night queue (`night.py add`), and the owner starts it.
- **Staging:** stage explicit paths, never `git add -A`. Before pushing, run `gofmt -l`, `go vet -tags 'cuda goinfer_testhooks' ./cuda/`, and CI's
  pinned staticcheck with its canary.

## 1. The contract (decided by the owner, 2026-10-14)

```go
// in decoder (the Mac writes these):
type ResidentKVCopier interface {
	CopyKV(src, dst, start, end int) error // rows [start, end) of every layer, slot src to slot dst; start, end multiples of 64
}
type KVCopyDecline struct{ Reason string } // returned by CopyKV, before writing anything, when the backend cannot serve the range
type ResidentReusableKVSlot interface {
	ReusableKVSlot(slot, prefix int) int // ResidentReusableKV's answer for any slot, bound or not
}
```

Notes on the contract:
- The decoder calls `CopyKV` on the executor's terms, as it calls every resident method, and never with `src == dst`.
- dst then continues at position `end`. For a windowed layer, that means dst needs src's rows `[end − (window − 1), end)`.
- If src no longer holds those rows, return `KVCopyDecline` naming the layer and src's base, and the decoder prefills.
- Any other error is handled the same way, so a failed copy may cost time but must never leave dst claiming rows it
  does not hold.
- `ReusableKVSlot` answers what `cudaResident.ReusableKV` answers today, for a slot that may not be bound.

## 2. Build (CUDA module only)

- **`CopyKV` on `cudaResident`.**
  - A device-to-device copy per layer of K and V, on the executor's stream, at the slot's KV precision.
  - A full-attention layer copies the rows.
  - A windowed layer (`cuda/kvwindow.go`) reads through src's shifted view and writes through dst's, then sets dst's
    base for that layer. Copy only what dst will attend to from `end`.
  - A layer that reads another layer's KV copies nothing of its own.
  - An MLA latent layer copies its latent rows.
  - A family or layout the copy does not cover returns `KVCopyDecline` with the reason. Do not guess.
- **A busy source.** The decoder may copy the prompt rows of a slot whose generation is running (choice 2 of the
  record). Those rows are final, but the copy must be ordered after that slot's prefill on the device and must not
  race its decode writes. Say in the code comment what orders it.
- **`ReusableKVSlot`.** `ReusableKV`'s rule, reading the named slot's base.
- **Comments:** follow `CLAUDE.md`'s "Code comments": the contract and guardrails, no history.

## 3. The CUDA arm of each gate

Each gate's rule, void conditions and planted defects are in the registration. A skip is not a pass: read every
`go test -v` run for `--- PASS`.

**By day:**
- **G1:** `TestWindowedKV_*`, and every CUDA test matching `Reuse|Slot|MC3|MC4|Lead|Resident`, with their test files
  byte-identical to `main`.
- **G2:** the copy exact on CUDA. Use the tiny fixtures and the 1.5B, at f32 and f16 KV, and the three windowed
  fixtures (`mistral-tiny-window`, `gemma3-vl-tiny`, `cohere2-tiny`).
  - Add a case where a windowed slot has lost the rows dst needs: it must decline, and dst must be untouched.
  - The two planted defects.
- **G3:** schedule independence on CUDA with the registration's schedules. The preamble is 1,000 tokens, above the
  512-token floor. Use the tiny fixture and the 1.5B int4, greedy plus the sampled arm. Add one schedule on a windowed
  fixture in which a copy declines.
- **G4(a):** the recorded class of every row against KV0's launch counts, for prompts of 985, 1,030 and 4,000 tokens
  and a hit followed by a suffix.
  - Planted: the decoder records a whole call as one pass. Restore it in a test hook if the decoder no longer does.
- **G4(b), the device arm:** a hit that would leave 15 rows, the case KV0's `split@985` arm found.
  - A prompt of 975 tokens whose first 970 another sequence holds: the block-aligned hit is 960, which leaves 15 rows.
  - The cap must shrink the hit to 896, leaving 79 rows.
  - The reply must equal the prompt prefilled alone.
  - Planted: the cap at n − 1. The reply then differs from 960 on, as KV0's arm did.
- **G4(e):** on a windowed layer forced onto the fast class (KV1's G2b instrument), rows from a pass that started off a
  block boundary are never copied.
- **G7, by day:** the in-process `CopyKV` benchmark for 18,176 rows of the 1.5B and the 7B, labelled exploratory.
  The night job's reading is the one quoted.

**At night, queued on nobara, one job per item:**
- **G5:** the sharing gate's three arms and three scripts, on the 1.5B at `-ctx 24576 -kv-sessions 2`, 6 blocks.
  - Queue `JOB=share` of `docs/measurements/kv2-2026-10-14/run-kv2-night.sh`, which runs `kv2_share.py` and its grader
    beside it. It builds the CUDA serve from `REV` and `BASE` itself.
  - Estimate it from one cold 20K prefill on this box, by day, with the lock free.
  - The control and void rules apply as written.
- **G6:** KV1's four CUDA cells, unchanged: 8 blocks for decode and 12 for time to first token. Queue `JOB=speed` of
  the same script, graded by `grade_kv2_speed.py` (G6's interval rule). KV1 measured 31 minutes; queue with `--est 45`.
- `REV` is your branch's head, merged with the Mac's, and `BASE` is `main` at `kv2-2026-10`'s branch point (`1cc385ef`).

## 4. Report

In `cuda-readings.md`, and briefly to the owner, report:
- each CUDA gate's verdict, with its planted reds;
- the copy's cost, in-process and served;
- G5's median saving against its projection, and B1's median times;
- G6's four cells;
- anything that declined and why;
- the commit the readings are at.

Push `kv2-cuda-2026-10`.
