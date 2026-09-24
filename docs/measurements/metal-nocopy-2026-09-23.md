# S6 Step 0 — Metal resident footprint, today's (copied) path, 2026-09-23

**Provenance.** MacBook (darwin/arm64) · goinfer `8f34a177` · `metal/cmd/serve` built locally
(`go build ./metal/cmd/serve`, `go.work` active — NOT `GOWORK=off`, which resolves the root module
from the last published proxy tag and fails to build against this session's unpublished changes)
· `--backend metal --quant int4` · real checkpoints already on disk (`qwen2.5-coder-1.5b-instruct-
q4_k_m.gguf`, `qwen2.5-7b-instruct-q4_k_m.gguf`) · `footprint <pid>` (macOS) captured once per load,
process idle (no request sent) right after the "goinfer serving on ..." banner · `sysctl
vm.swapusage` alongside each capture.

**Scope, deliberately narrower than the brief's own Step 0.** The brief's rule names three loads:
1.5B, 7B, and M26 (N=8). This pass covers **1.5B and 7B only** — no M26-class (~20GB+) checkpoint
fits this machine's free disk today (the same constraint that blocked S5's real measurement), and
R11(c)'s own history (three swap-spiral/kernel-panic incidents on this exact machine, 2026-09-20/22)
makes an M26 Metal run something to run only with the external kill script armed and full
attention, not as an incidental Step-0 data point. This is a deliberate, user-confirmed scope cut —
see this section's own status note in the task doc.

## Results

| model | quant | phys_footprint | phys_footprint_peak | IOAccelerator (graphics) dirty | untagged (VM_ALLOCATE) dirty / reclaimable | mapped file (clean) | swap-used at capture |
|---|---|---|---|---|---|---|---|
| qwen2.5-coder-1.5b-instruct-q4_k_m | int4 | 1242 MB | 1339 MB | 927 MB | 305 MB / 741 MB | 1229 MB | 1.74 GB (baseline, pre-existing on this shared dev box) |
| qwen2.5-7b-instruct-q4_k_m | int4 | 4855 MB | 5733 MB | 4023 MB | 818 MB / 557 MB | 2102 MB | 2.30 GB (baseline, pre-existing) |

Full raw `footprint` output for both runs is below (not archived as separate log files — short
enough to keep inline).

<details><summary>1.5B raw footprint + load banner</summary>

```
loaded "qwen2.5-coder-1.5b-instruct-q4_k_m": 28-layer model (vocab 151936) in 741ms [chat: chatml]
  decode path: metal-resident (int4)
  fit: 32768-token cap needs 1.8 GB KV (weights 1.2 GB, budget 3.7 GB, 1.9 GB left for a request's own prefill)
swap guard: armed, threshold +512 MB over baseline
swap guard: baseline 1.83 GB swap-used

======================================================================
goinfer-metal-serve [70831]: 64-bit    Footprint: 1242 MB (16384 bytes per page)
======================================================================
  Dirty      Clean  Reclaimable    Regions    Category
 927 MB        0 B          0 B        298    IOAccelerator (graphics)
 305 MB        0 B       741 MB        218    untagged (VM_ALLOCATE)
   ... (small categories omitted, <5 MB dirty each) ...
    0 B    1229 MB          0 B          8    mapped file
    0 B    7280 KB          0 B        350    __TEXT
1242 MB    1237 MB       741 MB       3209    TOTAL
phys_footprint: 1242 MB
phys_footprint_peak: 1339 MB
```

</details>

<details><summary>7B raw footprint + load banner</summary>

```
loaded "qwen2.5-7b-instruct-q4_k_m": 28-layer model (vocab 152064) in 14.517s [chat: chatml]
  decode path: metal-resident (int4)
  fit: 32768-token cap needs 3.5 GB KV (weights 4.8 GB, budget 2.0 GB, -1.5 GB left for a request's own prefill)
swap guard: armed, threshold +512 MB over baseline

======================================================================
goinfer-metal-serve [71188]: 64-bit    Footprint: 4855 MB (16384 bytes per page)
======================================================================
  Dirty      Clean  Reclaimable    Regions    Category
4023 MB        0 B          0 B        330    IOAccelerator (graphics)
 818 MB        0 B       557 MB        255    untagged (VM_ALLOCATE)
   ... (small categories omitted, <7 MB dirty each) ...
    0 B    2102 MB          0 B          8    mapped file
    0 B    6672 KB          0 B        350    __TEXT
4855 MB    2109 MB       557 MB       3280    TOTAL
phys_footprint: 4855 MB
phys_footprint_peak: 5733 MB
```

</details>

## Reading

**Confirms S0's qualitative claim, at a scale S0 itself never measured (S0 only read the code).**
The dominant DIRTY (anonymous, swap-eligible) category at both sizes is `IOAccelerator (graphics)`
— the resident Metal weight buffers `int4Buf`/`int4Concat` build today — not KV or scratch:

- 1.5B: 927 MB of 1242 MB total footprint (**75%**) is the IOAccelerator dirty term. The decoder's
  own "fit:" banner reports 1.2 GB of resident weights at this quant — close to the 927 MB
  IOAccelerator figure (int4-packed weights plus f16 scales land a bit smaller than the estimator's
  1.2 GB, which pins to a slightly coarser bytes/element figure — expected, not a discrepancy worth
  chasing here).
- 7B: 4023 MB of 4855 MB total footprint (**83%**) is the IOAccelerator dirty term, against a
  reported 4.8 GB weight estimate — same pattern, the dominant share grows slightly with model
  size (75% → 83%), consistent with KV/scratch staying roughly fixed while the weight term scales.

**`mapped file` (clean, file-backed, reclaimable for free) already holds the `.giw` sidecar** —
1229 MB / 2102 MB respectively — confirming the sidecar mapping itself costs nothing anonymous.
What S6 removes is the SECOND copy: today, `int4Buf` reads those same bytes back out of the
mapping, copies them to a heap `[]uint32`, and `NewBufferUint32s` copies THAT into the
`StorageModeShared` MTLBuffer that becomes the IOAccelerator-dirty term above — the mapped file is
already there and already cheap; the dirty IOAccelerator term is the redundant copy `S6`'s
`NewBufferNoCopy` aliasing is meant to eliminate.

**Directly informs the registered rule's band.** S6 ships only if "anonymous footprint after load
≤ KV + scratch + 15%" once the dense term is gone. At 1.5B, KV+scratch is a small fraction of the
927 MB IOAccelerator term (the "fit:" line's own KV estimate at the ACTIVE 4096-token context is
well under 1 GB); removing the dense copy should cut this process's anonymous footprint by roughly
the 75-83% measured here — a large, real target, not a marginal one. This is the "before" the rule
needs; the "after" requires the writer + reader S6's own Build steps 1-3, not attempted this pass.

## M26 (N=8) cell — measured 2026-09-24, after a reboot

The cell the 2026-09-23 pass deliberately did not run. It is the R11(c) configuration (three
swap-spiral / kernel-panic incidents on this machine), so it ran only after a **reboot** and with the
external kill-watch armed on a per-tick rate rule, not just a total: `scripts/swap_killwatch.sh`
(kill on 2 consecutive samples of >80 MB swap growth, or one jump >320 MB, or +600 MB total; 1 s poll)
plus the harness's own +400 MB kill and S3's in-process tripwire.

**Provenance.** MacBook (darwin/arm64, 16 GB), booted ~10 min earlier · goinfer `972a77a7` +
harness edits (`scripts/moe_pager_ab.py`, `swap_killwatch.sh`, committed with this record) ·
`metal/cmd/serve`, `-backend metal -moe-cache-experts -moe-cache-slots 8 -ctx 512` (no
`-stream-weights`: that flag builds the CPU pager, not Metal's) · `gemma4-26b-int4.giw` (15.7 GB,
local `~/models`) · `footprint <pid>` sampled every 6 s during the load, right after it, and at
tokens 1/16/32 of a 32-token greedy request. Machine at start: swap-used **0**, ~7.3 GB available.

| | attempt 1 (cold page cache) | attempt 2 (warm page cache) |
|---|---|---|
| starting swap-used / available | 0 MB / ~7.3 GB | 557 MB (left by attempt 1) / ~9.4 GB |
| load | banner at 50.6 s (`decode path: metal-resident (int4)`) | 12.8 s |
| peak RSS | ~8.0 GB | 7.6 GB |
| swap | **0 → 53 → 628 MB within 2 s of load completing → kill-watch fired** | +6.5 MB max; never approached a rule |
| outcome | killed by the rate rule; no panic, machine stayed responsive | served the request end to end |

Attempt 1 reproduces R11(c)'s signature on a **clean, freshly booted** machine with ~7 GB free: the
load itself completes, RSS peaks near 8 GB during the build, and swap jumps ~0.6 GB in two seconds as
it finishes. Attempt 2 did not spike — the one variable that differed was a warm page cache (and a
larger available figure), so this is **one spike in two attempts, not a rate**; it says the failure
is reachable on a clean box, not how often.

**Footprint, attempt 2 (`footprint`, MB; dirty = anonymous, clean = file-backed):**

| point | phys footprint | untagged VM_ALLOCATE (dirty) | IOAccelerator (dirty) | mapped file (clean) |
|---|---|---|---|---|
| load t = 6 s | 2,770 | 2,774 | — | 2,730 |
| after load (12.5 s) | **6,940** | 4,513 | 2,407 | 2,257 |
| token 1 | 7,359 | 4,820 | 2,519 | 0 |
| token 16 | 7,406 | 4,868 | 2,519 | 0 |
| token 32 | **7,452** | 4,912 | 2,519 | 0 |

Decode 7.0 tok/s (paged, N=8), first token 14.9 s (a 96-token prompt through the per-token path — the
banner notes batched prefill is declined for this arch's FFN shape). 155k faults, 151 page-ins.

**Against S6's registered rule** ("anonymous footprint after load ≤ KV + slots + scratch + 15%"): today's
path holds **~7.4 GB anonymous** (7.45 GB phys, ~0 clean) of a 16 GB machine. KV at ctx 512 is small and
8 slots × 30 layers is on the order of 1 GB, so the dense/other term is roughly **6 GB above the target**.
Two findings follow, one measured and one inferred:

- *Measured:* the 2.4–2.5 GB `IOAccelerator` term is the resident MTLBuffer weights S6 aliases away.
- *Inferred, not profiled:* the **4.5–4.9 GB `untagged (VM_ALLOCATE)`** term is the larger one, and is
  probably the same thing found on the CPU M35 path — the reader **copies every int4 group-scale array
  to the heap because the format does not align them** (`moe-pager-mode-darwin-2026-09-23.md`,
  "Finding"). gemma4-26b's experts' scales at 25% of 15.7 GB of nibbles is ~3 GB, plausibly most of it.
  If so, **S6's `NewBufferNoCopy` alone would not reach the rule's bar**: it fixes the MTLBuffer copy but
  leaves the heap copy, and the writer-side alignment change needs to land with (or before) it. This
  needs a heap profile of a Metal load to confirm; it was not taken.

**Two side observations.**
- The generated text of the 32-token request was incoherent (`" complexity is a- (\n면- ( ) )…"`), and
  a short raw `/v1/completions` prompt gave garbage too, but the **chat endpoint answered correctly**
  ("The capital of France is Paris.") on the same Metal server — so the path works, and raw completions
  on gemma4 (no chat template; possibly no BOS) are a separate, unexamined issue. Memory numbers do not
  depend on the prompt.
- Each attempt leaves swap-used higher (557 → 515 → 531 MB here, not returning to 0), the same
  ratchet R11(c) recorded.

Raw data: `metal-nocopy-m26-2026-09-24/` (both attempts' JSON, per-second samples, kill-watch logs).

## What this does NOT show

- **M26 (N=8): measured 2026-09-24 — see the section above.** The 1.5B/7B pass by itself was not a
  substitute for that cell.
- **No decode-time footprint** (after 32 tokens, per the brief's own Measure section) — this is a
  post-load snapshot only, process idle.
- **No memory-hog arm** (the 6 GB anonymous hog process the registered prediction is about) — not
  run this pass.
- **No depth-128/2048 tok/s bench** — this is a memory measurement only, not a performance one.

## Status

**Step 0 (measurement only) done for 1.5B/7B.** Build steps 1-4 (the new `metal` `.giw` kind
writer, the `NewBufferNoCopy` reader wiring, expert-slot handling, the banner) are **NOT started**
— user-scoped this session to Step 0 only, given the size and risk of rewriting the shipping Metal
resident weight-loading path without the M26-scale validation the brief's own Gates section
requires (byte-identical logits on every resident-parity fixture, a page-boundary fixture, `Close`
ordering under the leak checker — none of that exists yet). M26 Step 0 remains owed, same
disk-space constraint as S5's real measurement.
