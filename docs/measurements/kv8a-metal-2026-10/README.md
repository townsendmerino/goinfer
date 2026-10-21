# KV8a on Metal — registration (2026-10-21, before any code)

KV8a's end-of-prompt recurrent checkpoint on the Metal resident, modelled on the CUDA gates (`docs/measurements/kv8a-2026-10/README.md`,
all PASS on nobara, merged in #115). The decoder and `internal/kvmgr` halves are shared and already gated; this adds Metal's
`decoder.ResidentRecurrentCheckpoint` (bytes, save, restore). Owner's go: 2026-10-21 07:50 PT, "yes to all four as recommended" (start
after #115 merges, by day, on its own branch; this registration goes to him before the first timed run). The rules below do not move
after data.

## What carries over from CUDA, and what does not

- **The save point and the bar carry over:** `c = len(prompt)`, no cut; gate 2's bar is bit-identity. Metal promises the property the
  bar rests on: `metal/backend.go:metalResident.PrefillTailExact` is true, proven for the Gated-DeltaNet pass by
  `TestDB01_chunkedPrefillMatchesWhole` and `TestMC5_prefillChunkInvariance`. The routing reads the whole prompt's length, so a restore
  at `c` followed by the suffix runs the cold pass's kernels.
- **The trap to test against** (the 2026-09-27 and 2026-10-04 Metal identity incidents): a warm arm and a cold arm that reuse different
  amounts look like a defect. Every comparison checks `prefill_reused_tokens` on both arms first.
- **The fast-prefill floor is 16 rows on Metal** (`metalFastPrefillFloor`), not CUDA's 512, so gate 2(a)'s two sizes move.
- **What the checkpoint holds:** each Gated-DeltaNet layer's conv ring and recurrent state (`metal/deltanet.go:deltaNetLayer`, `win` and
  `state`), and each Mamba-2 layer's conv window and SSM state (`metal/mamba.go`), copied by one blit into a second buffer set.
- **Where it lives:** a private Metal buffer set allocated after KV and scratch and priced by Metal's fit guard. When the guard cannot
  admit it, KV8a declines and the slot behaves as today, as on CUDA. Unified memory: the bytes count against the Mac's 16 GB.
- **No KV0-H on Metal.** Gate 5's do-nothing arm on CUDA was KV0-H's own readings; nothing equivalent exists on the Mac, so gate M5
  is redesigned below.
- **The Mac's run budget:** by day, building and tier 0 only; every timed or 9B gate runs on the Mac's night queue (`CLAUDE.md`,
  "Run budget"), after a day fit check of each load it makes.

## Gates

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| **M1. Bit-exact save and restore** (tier 0, Metal) | On `qwen3next-tiny` (DeltaNet) and `nemotron-tiny` (Mamba): prefill; save; 8 decode steps; restore. Every recurrent layer's state and conv window, read back after the restore, equals the read-back right after the save, byte for byte. Plant: a restore that skips the conv window must differ. Then the 9B's checkpoint bytes from one load's own line (by day: a load only, no timing). | A fixture is missing (a skip is a fail); the plant does not differ. | Every layer of both fixtures once. | Seconds; the 9B load about 20 s. |
| **M2(a). Agreement with a cold prefill, tier 0** | As CUDA's 2(a), on the Metal resident: turn 1 prefilled and saved at `c`; 8 decode steps; turn 2 (turn 1's prompt, a reply differing 15 tokens in, a new message) restored at `c`. Last-row logits, every recurrent state and conv window, and the attention K/V rows equal a cold prefill of turn 2 on a fresh resident, bit for bit. Two sizes: both turns under Metal's 16-row floor, and both over it. Plants: no conv window; restore at `c - 1`. | A fixture is missing; a plant does not differ. | Each fixture once. | Seconds. |
| **M2(b). Agreement on real traffic** (the 9B, Mac night queue) | CUDA's gate 2(b) as it passed: the 24 pilot pairs, turn 1 greedy with the recorded reply's first token forbidden (`logit_bias` -100), turn 2 greedy, against a cold request on a fresh server: served tokens and every logprob's text, digit for digit; the restore used (`ckpt_restore` = turn 1's length) on every tested pair, 0 reuse on every cold one. **Any difference fails.** Harness: `kv8a-2026-10/gate2b.py` with a `--backend` argument (it hardcodes cuda today), on `--backend metal`. | A tested pair reports no restore (void that pair); under 15 tested; turn 1 writes its forbidden token (void the run). | The 24 pairs once. | On CUDA 65 min; on the Mac, longer, estimated from a day one-pair probe before queueing. Claude Code's 28k pairs need the 32k context: fit-checked by day first, and dropped with a dated note if the Mac cannot load it, which leaves 16 testable pairs. |
| **M3. The cache-state grid** (tier 0) | The grid's checkpoint cells (`decoder/cachestate.go`) hold for the Metal resident: a declared cell for each kind of state the Metal recurrent families carry. Plant: drop one, and the test goes red. | | Every cell. | Seconds. |
| **M4. The cost on every turn** | (a) Tier 0: the save is encoded before the first chained decode step's command buffer, checked through the caller with a recording fake; plant: the save after the first step goes red. (b) In M5's run: the save's own time per request, **p95 ≤ 12 ms**, CUDA's bar. | The save time is missing from a request line. | Every request of M5. | Inside M5. |
| **M5. The measured win on Metal** (the 9B, Mac night queue) | **The owner's decision, 2026-10-21 (below).** No KV0-H readings exist on the Mac, so the do-nothing arm is measured here: the 16 opencode and Codex pilot pairs (B-shaped by the forbidden token, as M2(b)), each turn 2 timed on two builds in one session, interleaved per pair and order-reversed on a second pass: main before KV8a (`a8b1c806`, cold by construction) and the KV8a Metal build. **Bar: the median of the paired turn-2 time-to-first-token ratios (KV8a ÷ before) ≤ 0.5**, an ambiguous band 0.5–0.7 parked for the owner, above 0.7 FAIL. Context, not graded: the same ratio on CUDA from 2(b)'s third run's serve logs. | The timing lock shared; the Mac on battery or swapping (the swap guard's void); a pair whose warm arm reports no restore. | The 16 pairs × 2 builds × 2 orders once. | About 1 h; estimated from M2(b)'s probe. |

## Open for the owner before M2(b) and M5 run: decided 2026-10-21, before any M-gate ran

Francis, 2026-10-21 10:11 PT, verbatim: "yes to the B35 test fix with the full CUDA package run, and yes to M5's design and bar as
recommended".

1. **M5's design: the paired two-build replay**, as proposed above: the 16 opencode and Codex pilot pairs, turn 2 timed on main before
   KV8a (`a8b1c806`) and on the KV8a Metal build in one session, interleaved per pair and order-reversed on a second pass.
2. **M5's bar: the median paired turn-2 ratio (KV8a ÷ before) ≤ 0.5 PASS; 0.5–0.7 parked for Francis; above 0.7 FAIL.**
