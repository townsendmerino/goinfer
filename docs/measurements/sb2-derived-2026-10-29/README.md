# SB2 memory reads, step 3, PR 2: the derived model on real checkpoints (2026-10-29, nobara-sb01)

The model, the bound and the gates are registered in `docs/tasks/task-subsystems-2026-10.md`, "SB2 memory reads, step 3, PR
2", at `352681be8`, before the code. Box: nobara, RTX 2070 SUPER, driver 595.91.07, the checkpoints from `~/models`. The pass
held the timing lock exclusively and started with only the desktop compositor on the card (`gdm2-card.txt`).

## G-DM1

`TestDeriveMemory` passes. Derive mode (`cudaBackend.deriveMemory`) is committed.

## G-DM2: the first pass is void

`void1-*` (22:36). The probe left `Options.Quant` unset, so the checkpoints loaded at f32, not int4: the quant serve's fit plans
at and the cuda gates load at. At f32:
- the host fit guard refused four of the seven loads;
- the 7B was refused at the weights check;
- only the 0.5B built, and its KV error was -1,403 MiB.

That figure is the f32 dense pricing against a device that does not hold f32. It is not counted. The probe was fixed to int4
and re-run.

## G-DM2: the pass (22:37–22:39, `gdm2.txt`, probe `sb2_dmprobe_test.go.txt`)

Error = derived − real, at each site the build read. P is the registered memory probe's reading before each load:
7,545,815,040 B on every cell.

| cell | built | sites read: error (MiB) | decisions, real vs derived |
|---|---|---|---|
| 0.5b default | yes | weights 0, kv +5.1, prefill +3.1 | equal |
| 0.5b 4-slots | yes | weights 0, kv +5.1, prefill +5.1 | context 32768 vs 16384 (see below) |
| 7b default | yes | weights 0, kv +19.4, prefill +17.4 | equal |
| 7b 2-slots | yes | weights 0, kv +17.4, prefill +19.4 | context 21120 vs 10560 (see below) |
| 26b expert-cache | refused at the KV fit (an unplanned 146,434-position context: 66 GB of KV) | weights 0, kv +82.0 | both refuse at kv |
| 26b expert-cache windowed | yes | weights 0, kv +74.0, grantPre +72.0, slots +47.0 | expert slots 28 vs 29 |
| 9b Qwen3.5 | yes | weights 0, kv +14.8, ckpt +12.8, prefill +14.6 | equal (checkpoint 52,690,944 B kept on both) |

## The rule

The largest **counted** |error| is 74.0 MiB, at the windowed 26B's KV site. The registered rule does not count the sites of a
cell the build declines, so the unwindowed 26B's 82.0 MiB is not counted. This read first said 82.0; the correction followed
Gemini's review of #495. 74.0 MiB is between 64 and 128 MiB: **ambiguous, parked**. No census (PR 3) is built, and the
decision goes to Francis.

Every error is positive: the model always thinks more is free than the card says, which is the dangerous direction. The dense
and 9B errors are 3–20 MiB. The 26B's are 47–82 MiB. On the windowed 26B, the 47 MiB at the slots site is enough to turn the
build's 28 expert slots into 29.

## Two findings beyond the bound

1. **The plan does not model `kvRaise`.** With more than one KV slot and no context asked for, the build lets one
   conversation's cap grow to the whole KV pool (`kvCapFor`): 16384 → 32768 on the 0.5B's 4 slots, and 10560 → 21120 on the
   7B's 2. `memPlan.CtxCap` reports the per-slot plan. Step 1's G-MP3 and PR 1's G-PA3 did not see it: every census cell asks
   for one slot.
2. **The 26B's larger error is where the build allocates most.** Its KV-site error (74–82 MiB) is about four times the dense
   models'. Candidates, not yet attributed:
   - the expert cache's pinned host mapping;
   - the float-activation buffers a Gemma 4 MoE build prices in `postPoolBytes` but allocates before the KV pools;
   - the decode scratch the plan does not price.

**Decision (Francis, 2026-10-29 22:52, option (a)):** attribute the 82 MiB with a diagnostic on the card, add the missing term
and `kvRaise` to the plan red-first, and re-run G-DM2 as a new pass under the same bound, registered first.

## After Gemini's review of #495 (2026-10-29, before the attribution diagnostic)

- **The windowed grant was taken off at its raw cost** (`extra × perPos`), not at what the card charges for the window pools'
  growth, which is what the registered model says. It now uses `windowedPoolCharged`, the pools sized as `kvReallocWindowed`
  sizes them. Red first (`review495-red.log`, the raw subtraction restored for the red run): on a 26B-like windowed resident,
  41,943,040 B charged against 37,748,736 B raw. On the 26B cell this moves the error between the grant site (+72.0) and the
  slots site (+47.0) by the grant's charged/raw difference. The re-run will show it.
- **Derive mode marks a reading read only when the plan derives it.** It no longer starts every reading as read with 0. The
  slots site is derived only for a cached load, and the prefill site only for a load without windowed KV, as the build reads
  them.
- **New tests:**
  - the KV pools and post-pool deductions, exactly;
  - the degenerate expert cache;
  - a `grantPre` plant;
  - derive mode on a windowed fixture.
- **Two grants on one load.** A degenerate expert cache with windowed KV can take both the grant before the cache and the last
  grant, because `allocSlots` turns the cache off between them. The build does the same, so the plan mirrors it.


## The attribution (2026-10-30 05:53, `attrib.txt`, probe `sb2_attrib_test.go.txt`)

At the KV site, the build's device ledger by line against the plan's terms (driver 595.91.07, card clear):

| cell | drop P → KV reading | ledger VRAM | ledger dense vs plan dense + slack | scratch | not on any ledger line |
|---|---|---|---|---|---|
| 7B | 4,661,968,896 | 4,644,818,548 | +1.3 MiB | 1.8 MiB | 16.4 MiB (context, modules) |
| 26B windowed | 2,111,832,064 | 1,925,088,900 | **−105.7 MiB** (the plan over-prices) | 1.7 MiB | **178.1 MiB**: the 132.0 MiB pool plus ~46 MiB of context and modules |

- **The missing term:** with the expert cache, the context is not planned, so `moe_route`'s pool is paid in the setup job
  (`payRoutePool`, before `checkKVFits`). The model took it off only for a planned context. It is added, red first
  (`pool-red.log`).
- **Two findings, not changed here:**
  - **The 26B's planned slack** exceeds what the card charges for its dense weights by 105.7 MiB. It appears to round expert
    stacks the cache keeps in pinned host memory. The real build's context plan reads the same slack, so the 26B's plan is
    conservative by that much.
  - **The context and module overhead** (16 MiB on the 7B, ~46 MiB on the 26B) is on no plan line. It is the part of the error
    that remains.

## G-DM2 pass 2 (registered at `a96cf7a2a` before it ran; 05:56–05:57; again 05:58–05:59, the stopping rule's second pass)

The same cells at int4, card clear (`pass2-card.txt`, `pass2b-card.txt`). The model now has the pool term, `kvRaise`, and the
grant at its charged cost.

| cell | largest |error|, pass 2 / 2b (MiB) | decisions, real vs derived |
|---|---|---|
| 0.5b default | 5.1 / 5.1 | equal |
| 0.5b 4-slots | 5.1 / 5.1 | equal (context 32768: `kvRaise`) |
| 7b default | 19.4 / 19.4 | equal |
| 7b 2-slots | 19.4 / 19.4 | equal (context 21504) |
| 26b expert-cache | not counted: declined at the KV fit on both sides (−52.0) | equal |
| 26b expert-cache windowed | 60.0 / 60.0 (all negative: −58, −60, −60) | pass 2 equal (29 slots); pass 2b 33 real vs 32 derived |
| 9b Qwen3.5 | 14.8 / 14.8 | equal (checkpoint kept, 52,690,944 B) |

**Read: holds.** The largest counted |error| is 60.0 MiB in both passes, inside 64 MiB. The second pass ran because 60.0 is
within 8 MiB of 64, as the stopping rule says. As stated before the run, the 26B's −60 MiB is two terms partly cancelling: the
slack over-pricing (−106) and the unledgered overhead (+46). Its sign is the safe one: the model thinks less is free than the
card says, as in pass 2b's one-slot difference. The dense and 9B errors are positive, 3 to 20 MiB, all of it context, modules
and scratch.

The census (PR 3) may go ahead. Its lines should say the post-allocation figures are derived, and give this record's error
band.

## G-DM2 after the driver change (registered at `0e6eac7d8` before its counted run; 06:47–06:48; again 06:49–06:50)

nobara was updated and rebooted on the morning of 2026-10-30, up again since 06:37:49 PDT:

| | before (passes 1–2b) | after (pass 3, 3b) |
|---|---|---|
| NVIDIA driver | 595.91.07 | 595.112.07 |
| kernel | 7.2.0-202.nobara.fc44.x86_64 | 7.3.0-202.nobara.fc44.x86_64 |
| card at the start | kwin_wayland only (20 MiB compute) | kwin_wayland only (16 MiB compute); the logged-in desktop's graphics clients 525 MiB used in all |
| P, the probe's free VRAM | 7,589,986,304 | 7,502,495,744 (−83.4 MiB) |

- **The pool (`../sb2-memcensus-2026-10-29/gmc3-postreboot*`, 06:40):** G-MC3's probe re-run on the new driver. `moe_route`'s
  pool is 138,412,032 B, unchanged: plan − (probe − pool) = 0 on all four qwen3next-tiny cells that plan, and all 412 cells
  pass. `cuda_moe_route_pool_bytes` stays in both nobara profiles.
- **The first try is void (`void-pass3-postreboot*`, 06:43):** the windowed 26B was not reached. The host fit guard refused
  its load at 42.4 GB available against 61 GB at the start, most likely the unwindowed 26B's heap not yet returned to the OS.
  The probe now calls `debug.FreeOSMemory()` after each cell, which changes host memory only.

| cell | largest |error|, pass 3 / 3b (MiB) | decisions, real vs derived |
|---|---|---|
| 0.5b default | 5.1 / 5.1 | equal |
| 0.5b 4-slots | 5.1 / 5.1 | equal (context 32768) |
| 7b default | 19.4 / 19.4 | equal |
| 7b 2-slots | 19.4 / 19.4 | equal (context 20736, was 21504: P fell 83 MiB) |
| 26b expert-cache | 62.0 / 62.0 (kv −60, slots −62, prefill **+58**) | context 4096; slots 28 real vs 27 derived; **prefill width 1024 real vs 2048 derived** |
| 26b expert-cache windowed | 62.0 / 62.0 (kv −60, grant −62, slots −62) | context 4096; slots 37 real vs 35 derived |
| 9b Qwen3.5 | 14.8 / 14.8 | equal (checkpoint kept) |

**Read: holds.** The largest counted |error| is 62.0 MiB in both passes, inside 64 MiB. The second pass ran because 62.0 is
within 8 MiB of 64, and it matched the first to the byte. Two things differ from pass 2, and both bear on the census:

- **The 26B cells ran in another regime.** Before the reboot the host had less memory available. The host fit guard
  (`mem.HostCheck.Decide`) then pinned the context, to 121,898 and 17,709. The CUDA resident takes a pinned context as asked, so
  it declined the unwindowed cell and built the windowed one at 17,709. Today the host fits, nothing is pinned, and the resident
  trims both cells to 4096. So the unwindowed 26B is now a counted cell, and its expert cache has more room (28 and 37 slots,
  against 29 to 33).
- **One decision now goes the unsafe way.** On the unwindowed 26B, one slot fewer in the derived model (−62 MiB at the slots
  site) leaves +58 MiB more free at the prefill site than the card has. That moves the prefill width over a step: 2048 derived,
  1024 built. The error is inside the bound, but the decision flips in the direction the measure calls dangerous. A
  post-allocation decision read off a derived figure within 64 MiB of its threshold can differ from the build either way. The
  census marks each such decision.

The error band across passes 2 to 3b is −62 to +58 MiB at the 26B and +3 to +20 MiB elsewhere.

**The mechanism, and why it matters beyond the bound (added after Gemini's post-merge review).** On the unwindowed 26B in passes 3
and 3b:
1. The model under-predicts the free VRAM at the slots site by 62 MiB: the safe direction, within the bound.
2. That is enough to budget one expert slot fewer than the build, 27 against 28. One slot is about 120 MiB of VRAM.
3. Every later site then sees that slot's bytes as free. The prefill site is over-predicted by +58 MiB, and the width becomes
   2048 where the build takes 1024: the unsafe direction.

So a small error on the safe side, where it crosses a step in a quantized decision (a slot, a rung of the prefill ladder), flips
a later decision by a whole step the other way. The bound limits the size of each figure's error. It does not limit the
direction of the decisions downstream of a step. That is why the census marks (`~`) every decision that moves within ±64 MiB, and
why it marks a refusal 64 MiB away. The census golden carries no `~` today (`../sb2-dcensus-2026-10-30/README.md`): its cells are
tiny fixtures with gigabytes to spare, none of them windowed or caching experts. The 26B-shaped case appears only on the card.
