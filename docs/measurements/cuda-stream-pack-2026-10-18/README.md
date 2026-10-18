# CUDA resident build: stream the packing into the device buffer, and correct the fit guard's host term (pre-registration, 2026-10-18)

**Registered before any code of the change is written** (owner, 2026-10-18; question and measurements in
`docs/measurements/cuda-build-heap-2026-10-18/README.md`). Branch `cuda-stream-pack`. The results are appended below the line when they are read, and no bar
above is moved afterwards without a mechanism.

## The change

1. **Streaming (cuda/).** A dense projection is no longer packed in `BuildResident`'s first loop and uploaded in a second job. The first loop *plans* it
   (`planWeight`: the accept-or-decline decision `packWeight` makes today, with no packing and no allocation) and keeps only the weight handle and the sizes.
   The upload allocates the device buffers, then packs row blocks into one reused staging slice and uploads each at its byte offset (`Buffer.At`, `gpu.Upload`).
   The bytes are the bytes `packWeight` produces. The expert stacks (`packWeightStack`, the C′ pinned host copy), the shared experts, Mamba, and the drafter keep
   the eager `packWeight`.
2. **Fit guard (decoder/).** For a load that will be CUDA-resident: KV is priced as VRAM, so it adds nothing to the host need (`kvPricingFor`, as Metal's sizing
   is already a special case there; the VRAM plan is the CUDA KV's guard); and the host term is `max(load phase, build phase)`: load phase = the canonical weights
   plus the source file (the `.gguf` direct path only), build phase = the canonical weights plus one packed layer plus the staging block. The C′ expert-cache
   term (`cudaBuildBytes`) is unchanged.

**A deviation from the instruction, stated before the run.** The instruction was to price the packed copy *instead of* the source file. The measured timeline
(`cuda-build-heap-2026-10-18/README.md`, and its raw samples) shows two peaks of about the same height on the direct `.gguf` path: the load phase (canonical
weights growing on the heap while the source file is mapped: 8.4 GiB of RSS at t = 3.3 s, before any packing) and the pack phase (the packed copy beside the
canonical weights: 8.7 GiB at t = 14.1 s). Streaming removes only the second. Dropping the source-file term would price the direct path at 4.2 GiB against an 8.4 GiB
process, so the term stays for the direct path, in the `max`. G-F2 below is what decides whether that is right.

## Gates (tier 0 unless stated; each names its plant)

| id | gate | plant (must go red) | void if |
|---|---|---|---|
| G-S1 | **Device bytes identical to today's.** For each kind (`int4`, `int8`, `q4k`, an `f32` weight quantized to int8) on tiny weights, at one block and at several (staging forced to a few KiB), every device buffer after the streamed upload equals the buffer `packWeight`+`upW` produced, read back byte for byte | skip `permuteFast` on the second block of the int4 stream | no CUDA device (a skip counts as a failure on nobara) |
| G-S2a | **Host bytes allocated per tensor are the staging block, not the tensor.** Streaming M weights totalling B bytes allocates at most one staging block plus 64 KiB per weight on the host (`runtime.MemStats.TotalAlloc` delta), against at least B for the eager path | allocate the staging slice per tensor | the eager path allocates less than B (fixture too small) |
| G-S2b | **Nothing is packed while planning.** After the first loop, a planned `hostW` holds no packed words (`len(wpk) == 0`), and `packedAllocSlack` over planned layers equals the slack over the same layers packed eagerly | make `planWeight` pack | — |
| G-S3 | **A decline allocates nothing.** `planWeight` returns exactly `packWeight`'s error for every shape it declines (K not a multiple of 4; int4 K not a multiple of 32; an int4 tensor with no canonical bytes; q4k size mismatch; an unsupported kind), and when it does, the device-buffer allocation counter is 0; a model whose last layer declines uploads nothing from the earlier layers | drop the int4 `K % 32` check from `planWeight` (the decline moves to stream time) | — |
| G-F1 | **The fit guard prices a CUDA-resident load's KV as VRAM.** With a CUDA backend registered: `kvBytes` is 0 for a load that will be CUDA-resident (a pin or not), unchanged for `cpu` and for a CUDA load whose architecture is outside the CUDA gate; `guardGIWFit` adds no KV for it; a 7B-shaped load at 32,768 with 14 GiB available is admitted, and refused by the tree before the change | price KV as host again | — |
| G-F2 | **The guard's number matches the measured peak.** `FitDescribe`'s total for the 7B (`int4`, `-direct-load`) and for phi3 (`q4k`), each over the measured peak VmRSS of that cell **after** the change (`build_peak.py`, sampled from spawn), lies in **[0.90, 1.15]**. Under 0.90 the guard under-prices; over 1.15 it over-prices. Reported either way; the band is not moved to fit | — | the measured run saw swap growth in the server (`VmSwap` over 8 MiB) |
| G-T1 | **Tier 1, `build_peak.py` on nobara (about 45 s).** Peak anonymous memory: 7B through the sidecar ≤ 0.6 GiB (was 4.0), 7B `-direct-load` ≤ 5.0 GiB (was 8.5), phi3 q4k ≤ 3.3 GiB (was 5.2); and the 7B at `-ctx 32768` starts. Load time (`ready_s`) is read beside, with no bar: more than 1.10× the before is reported | — | another timed run held the box |

**Stopping rule.** Fixed: each gate is read once. A red gate is fixed and re-read as a new attempt, recorded as such, never re-rolled until green.
**Cost basis.** Tier 0 is seconds (the GPU tests need the card, under the timing lock); G-T1 is the 43 s the before-run took.
**Not gated here.** Decode and prefill speed are untouched (no kernel changes); the C′ expert cache, Mamba, MLA and DeltaNet projections: MLA and DeltaNet projections go through the same dense loop and are covered by G-S1's kinds; no real MLA or DeltaNet checkpoint is built in this gate.

## MEM6 (registered with this change, owner 2026-10-18)

MEM0's nobara half measured the host embeddings of a 20K prompt as **held after the request**: +218 MiB (1.5B) and +476 MiB (7B) of anonymous memory still present 10 s later.
MEM6's fix (`docs/tasks/task-memory-campaign-2026-10.md`) therefore has two requirements in its gate, beside the allocation bound it already has: it embeds **per chunk**
into one reused slab, **and the memory is returned afterwards**: the server's `RssAnon` 10 s after a 20K cold prompt is within 64 MiB of its value after load plus the
session's retained state, on the 1.5B and the 7B (`docs/measurements/mem0-2026-10/mem0_linux.py`, the same cells).

---

## Results

(appended when read)
