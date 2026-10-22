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

**Corrected 2026-10-21, at M2(a)'s first run (tier 0):** M2(a)'s wording, "a reply differing 15 tokens in", is CUDA's and cannot hold
under Metal's 16-row floor, where both turns must be under 16 rows. The under-the-floor case uses turn 1 of 6 tokens, a 4-token reply
and a 4-token message (turn 2: 14); the over-the-floor case keeps CUDA's 300 + 15 + 45. The bar (bit-identity) and the plants are
unchanged. The case ran before this note was written; it is recorded here rather than changed silently.

## Open for the owner before M2(b) and M5 run: decided 2026-10-21, before any M-gate ran

Francis, 2026-10-21 10:11 PT, verbatim: "yes to the B35 test fix with the full CUDA package run, and yes to M5's design and bar as
recommended".

1. **M5's design: the paired two-build replay**, as proposed above: the 16 opencode and Codex pilot pairs, turn 2 timed on main before
   KV8a (`a8b1c806`) and on the KV8a Metal build in one session, interleaved per pair and order-reversed on a second pass.
2. **M5's bar: the median paired turn-2 ratio (KV8a ÷ before) ≤ 0.5 PASS; 0.5–0.7 parked for Francis; above 0.7 FAIL.**

## Runners (written 2026-10-21, before any 9B run)

- `run-daycheck.sh` (by day, Francis 2026-10-21 "lets do 2 by day soon", once he has closed his apps): the heavy-load void rule at the
  start (`common.sh:start_clean`: load average ≤ 4.0, free memory ≥ the 9B's fit need 6.7 GB + 2 GB, both recorded), the fit check at
  `-ctx 20480` and `32768` (M1's 9B bytes are read from the same loads), then the one-pair probe: opencode pair #1, about 7.3k tokens,
  through `../kv8a-2026-10/gate2b.py --backend metal --start 1 --limit 1`, which prints no verdict. A refused fit stops the job (exit 3).
- `run-gate2b-metal.sh` (M2(b)) and `run-m5.sh` with `m5.py` (M5 and M4(b)), on the night queue; each re-queues itself (exit 75) on a
  start that is not clean or a refused fit. M5 reads turn 2's time to first token from serve's own request log (`ttft`).
- Pinned inputs: KV8a serve `serve-metal-9488f592`, the before-KV8a serve `serve-metal-a8b1c806` (main before #115), and the 9B's Metal
  `.giw`; the pilot's bodies, ids and key, copied from nobara, stay outside the repository.

## The day checks (2026-10-21, this Mac, Francis's apps closed)

- **Fit: the 9B is resident on Metal** (`metal-resident (int4)`) at `-ctx 20480` (KV 0.6 GB, weights 5.5 GB, budget 6.1 GB) and at
  `-ctx 32768` (KV 1.0 GB). **M1's 9B checkpoint: 50.2 MiB in 48 buffers**, as on CUDA.
- **The one-pair probe** (opencode pair #1; job `kv8a-metal-probe`, 11:03:57–11:09:39; start clean at load 2.10 and 11.38 GB free,
  after a `sudo purge` that Francis ran, since the 9B's own cached file pages counted against free memory): turn 1 (7,314 tokens) cold
  in 121.5 s; turn 2 (7,403) restored at turn 1's end in 2.57 s to first token; the cold turn 2 in 119.9 s; warm equal to cold; the save
  1.8–2.2 ms. A first attempt (10:53) got a 404, the runner asking for the model by a name the server did not serve; fixed, and the
  harnesses now stop their servers on any exit (that attempt left one running).
- **Estimates from it** (about 61 tokens/s of cold prefill at 7–10k, taken as 45–50 at Claude Code's 17–19k): M2(b) about 2 h 40 min to
  2 h 50 min; M5 about 3 h 30 min, over the night queue's 3-hour cap per job, so M5 runs as two jobs, one per pass, back to back the same
  night under one record directory (`m5.py --pass`); pass 2's job grades both. The design and the bar are unchanged.

**A purge before each 9B job's start check (2026-10-21, before any night run; the owner's change).** After a 9B load exits, macOS files the
model's cached `.giw` pages as active, and the void rule's reading (free + inactive + speculative + purgeable, the fit guard's own) counts
them as used: this morning 8.13 GB ten seconds after the last 9B server exited, against the 8.7 GB need, and 11.38 GB after a purge. With
nothing to apply memory pressure at night the cache would stay, so every job after the first would wait until 06:00. Francis installed
`/etc/sudoers.d/purge` (that one command, passwordless). Each runner now calls `common.sh:purge_then_check`: the reading before, `sudo -n
/usr/sbin/purge` (a failed purge voids that start and the job re-queues), the reading after, then the unchanged start check. The void rule
and the 8.7 GB need are unchanged. The jobs are queued without `night.py`'s own `--fit-need-gb` gate, which reads memory before the job
starts and so before its purge; the job's start check applies the same rule after the purge.

## The grade (2026-10-22)

Graded on the Mac's records, kept outside the repository as the CUDA ones were: M2(b) in `~/goinfer-bench/kv8a-metal/gate2b-2026-10-21-2201`
(22:01–23:48, start check load 3.05, 10.04 GB free); M5 and M4(b) in `~/goinfer-bench/kv8a-metal/m5-2026-10-21` (pass 1 from 23:49, pass 2
01:41–03:31; start checks load 0.73 and 1.26, 11.51 and 11.07 GB free). No start was void. Builds: KV8a serve `9488f592`, before-KV8a
serve `a8b1c806`.

| gate | reading | result |
|---|---|---|
| M1 | byte-exact save and restore on `qwen3next-tiny` and `nemotron-tiny`, the plant red; the 9B's checkpoint 50.2 MiB in 48 buffers | **PASS** |
| M2(a) | restore then suffix bit-identical to a cold prefill (logits, recurrent state, attention K/V), under and over the 16-row floor; both plants red | **PASS** |
| M2(b) | **The 20K cells (owner, 2026-10-22: graded on them):** 17 pairs, 16 tested, **0 differ**. Each tested pair restored at turn 1's end (`warm_reused` = turn 1's length) and matched its cold arm on the served tokens and the prompt hash, and every cold arm reused 0; opencode's pair 0 (turn 1 of 613 tokens) cannot test the restore and voids itself, as on CUDA. **The 28K Claude Code chain:** its first pair's turn 1 (18,052 tokens) answered 503, the swap guard having tripped at +0.64 GB of system swap over its 512 MB bar (1.82 GB in use), and the run ended there. **Owner, 2026-10-22:** that is the 28K limit of a 16 GB Mac with the 9B, recorded here; the 28K chain moves to nobara as a CUDA KV8a cell (another session runs it); the swap guard's threshold is unchanged | **PASS** on the 20K cells |
| M3 | the cache-state grid passes as is | **PASS** |
| M4(a) | the save is encoded before the first chained decode step; the plant red | **PASS** |
| M4(b) | the save's own time, p95 **11.929 ms** over 64 saves (median 4.477, max 14.908); bar ≤ 12 ms | **PASS** |
| M5 | the median paired turn-2 time-to-first-token ratio (KV8a ÷ before) **0.069 over 32** (pass 1 0.070, pass 2 0.069; per pair 0.021–0.090, e.g. opencode #8 at 194.8 s before against 12.2 s restored); bar ≤ 0.5 | **PASS** |

**A finding for the owner, not a FAIL:** gate 2(b)'s logprob comparison never had anything to compare, on CUDA's PASS
(`~/goinfer-bench/kv8a/gate2b-2026-10-20-2248` on nobara) or here. `gate2b.py` sends `"logprobs": true` on `/v1/chat/completions`,
and every reply's logprob list came back empty on both arms, so that equality held trivially. Both verdicts rest on the served tokens
and the prompt hash, which were compared on every tested pair. Whether serve should return logprobs for a bare `"logprobs": true`, or
the harness should also send `top_logprobs`, is open.

Merged with `origin/main` merged into the branch (not rebased: the branch was already pushed); `scripts/touched.py` green on the merged
tree (the whole `metal` package, 93.8 s) and the four KV8a tests (`-tags goinfer_testhooks`) PASS.

