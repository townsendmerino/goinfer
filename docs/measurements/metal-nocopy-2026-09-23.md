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

## What this does NOT show

- **No M26-class measurement** — see Scope above. The 75%/83% pattern at 1.5B/7B is suggestive but
  not a substitute for the actual M26 (N=8) cell the registered rule names, and R11(c)'s own
  incidents happened at THAT scale specifically, not at 1.5B/7B.
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
