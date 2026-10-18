# CUDA resident build: stream the packing into the device buffer, and correct the fit guard's host term (pre-registration, 2026-10-18)

**Registered before any code of the change is written** (owner, 2026-10-18; question and measurements in
`docs/measurements/cuda-build-heap-2026-10-18/README.md`). Branch `cuda-stream-pack`. The results are appended below the line when they are read, and no bar
above is moved afterwards without a mechanism.

## Provenance of the decision

The memory work is the owner's, started from `docs/tasks/task-memory-campaign-2026-10.md`. The order to stream (rather than only price the packed copy) reached the nobara session as a
relayed note on 2026-10-18, and the note did not name the document or whose decision it was. The owner confirmed the same day that the memory work is his and that this branch continues
as planned: branch cells when the timing lock frees, then a PR for his yes, no merge. That campaign document has no section that poses stream-versus-price: it names the CUDA host copy only
in MEM10 (the expert-cache registration, which this change leaves alone) and registers MEM6's gate with this change. The stream-versus-price question was posed by the
scoping note that preceded the build (the Mac's swap work), and the choice of "stream" is the owner's confirmed decision as relayed.

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

### Tier 0 (branch `cuda-stream-pack`, 2026-10-18, on the RTX 2070 SUPER)

G-S1, G-S2a, G-S2b, G-S3 and the decline-allocates-nothing test pass (`cuda/pack_stream_test.go`, run with `-v` and read for `--- PASS`). The fit-guard tests in
`decoder/fitguard_cuda_test.go` pass, `gofmt -l` is empty for `cuda/` and `decoder/`, and `go vet` is clean for `./cuda/` (tags `cuda goinfer_testhooks`) and `./decoder/`.
`scripts/touched.py` is RED on one test that is not this change's: `TestKernelLocalMemoryCensus` fails with
`PINNED: mamba_conv declares 32 B/thread of local memory and is not in the pinned set`, and it fails identically on a clean `origin/main` (`1ba40e335`) worktree. The branch changes no kernel.

### Tier 1: `build_peak.py`, one run per cell, 09:14–09:16 PDT, binary built from the branch (`CGO_ENABLED=0 go build -tags cuda cuda/cmd/serve`), raw in `raw/`

| cell | peak anon, before | peak anon, now | bar | peak VmRSS, now (when) | final anon / file | `ready_s` before → now |
|---|---|---|---|---|---|---|
| 7B sidecar, ctx 4096 | 4121 MiB | **113 MiB** | ≤ 0.6 GiB: pass | 4019 MiB (t = 16.6 s) | 88 / 3873 MiB | 10.89 → 17.58 |
| 7B `-direct-load`, ctx 4096 | 8677 MiB | **4202 MiB** | ≤ 5.0 GiB: pass | 8627 MiB (t = 3.2 s) | 4184 / 78 MiB | 15.44 → 20.14 |
| phi3 q4k, ctx 4096 | 5314 MiB | **2737 MiB** | ≤ 3.3 GiB: pass | 4970 MiB (t = 0.8 s) | 2724 / 78 MiB | 2.53 → 3.54 |
| 7B `-direct-load`, ctx 32768 | — | 4209 MiB | starts: pass | 8524 MiB (t = 1.3 s) | 4186 / 78 MiB | 17.62 |

No cell grew swap (`peak_swap_mib` 0). The "before" column is `cuda-build-heap-2026-10-18/raw/` from main, in another session.

**The direct path's peak VmRSS did not fall, and the record predicted that.** The peak moved from t = 14 s (the pack phase) to t = 1–3 s (the load phase: canonical weights growing on
the heap while the source is mapped). Streaming removed the second peak, which was the higher by 0.3 GiB; the first is untouched and is why the guard keeps the source-file term on the
direct path (the deviation stated above).

**G-F2** (`testdata/fitdescribe.go`, the CUDA backend compiled in; `fitGB` is 1 GiB; the guard prints one decimal):

| cell | guard prices | measured peak VmRSS | ratio | band [0.90, 1.15] |
|---|---|---|---|---|
| 7B `-direct-load`, ctx 4096 | 8.4 GiB (4.0 weights + 4.4 reading the checkpoint; KV in VRAM) | 8627 MiB = 8.42 GiB | 0.997 | pass |
| 7B `-direct-load`, ctx 32768 | 8.4 GiB (the same: KV is no longer host) | 8524 MiB = 8.32 GiB | 1.009 | pass |
| phi3 q4k, ctx 4096 | 4.9 GiB (2.6 + 2.2) | 4970 MiB = 4.85 GiB | 1.010 | pass |

On main the same loads were priced 8.8, 11.9 (the 7B at 32K) and 7.9 GiB; the 7B at 32K was over-priced against the 8.5 GiB it measures, and phi3 by 1.5x. The sidecar load is priced at
nothing, as before; its peak VmRSS is now 4.0 GiB, almost all the mapped sidecar (final file 3873 MiB), not heap.

**Load time (`ready_s`) is up in every cell, 1.3–1.6x the before, above the 1.10x mark that is reported.** The before and now are different sessions, so this is not a ratio
(`docs/measurements/noise-registry.md`, TE3): all it says is that streaming is not shown to be free. The same-session pair is below.

### Deviations from the instruction and the registration

1. **The source-file term stays on the direct `.gguf` path** (stated before the run; G-F2 above is the evidence it is right: dropping it would price 4.2 GiB against an 8.4 GiB process).
2. **"The 7B at 32K is admitted on nobara" is tested with injected availability** (`TestFitCheck_cudaResident7BAt32KIsAdmitted`): this 64 GB box already admitted it under the old
   pricing. The measured 32K start above is the end-to-end half.
3. `testdata/fitdescribe.go` here is a copy of the earlier one that compiles the CUDA backend in; the earlier copy prices the CPU path, because the root module registers no backend.

### Load time, same session, interleaved A B B A (main's binary, the branch's, the branch's, main's), 09:34-09:42 PDT, `raw/ab/`

`ready_s`, two runs per arm per cell (the two main runs are runs 1 and 4, the two branch runs 2 and 3):

| cell | main | branch | branch / main (ratio of means) |
|---|---|---|---|
| 7B sidecar | 11.03, 11.03 | 16.78, 16.96 | **1.53** |
| 7B `-direct-load` | 13.39, 13.15 | 18.47, 18.14 | **1.38** |
| phi3 q4k | 2.68, 2.27 | 3.49, 3.34 | **1.38** |

The order effect is small beside the difference (the two main runs agree within 0.4 s on the 7B and the two branch runs within 0.4 s): the branch is slower in every cell, by 5-6 s on the 7B. Peak anonymous memory
reproduced the first reading (sidecar 104, 108 MiB against 4179, 4128; direct 4217, 4202 against 8627, 8635; phi3 2737, 2736 against 5314, 5314).

**Read.** Streaming saves 0.4-8.5 GiB of host memory per load and costs 1.4-1.5x the load time, above the 1.10x mark, so by the rule set for this change the streaming is held until it is no slower and the fit-guard
correction (KV priced as VRAM) goes on its own. The cause is not measured. Where the cost may sit: the eager path packed on the first loop and uploaded in a second job, and the streamed path packs one block, uploads it
(`gpu.Upload`, synchronous), then packs the next, so pack and upload no longer overlap and each pays its own latency; a pipeline (pack block n+1 while block n uploads) is the first thing to try.
