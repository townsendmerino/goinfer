# MEM13-copies-CUDA: a Mamba-2 layer's projections stream to the device (2026-11, nobara-sb01)

The ledger row (`docs/tasks/task-memory-campaign-2026-10.md`, MEM13-copies-CUDA; P36 in `docs/queue-performance.md`): the dense projections already stream to the device in row blocks through one staging block
(`backend/cuda/pack_stream.go:planWeight`, `cudaResident.streamW`), and what was left of the whole-tensor host packing is `backend/cuda/resident.go:packWeight`, "about 4 GB of int8 host copies on Nemotron-H 9B
by the survey's arithmetic". **Read against the code at `6ab62e5b2`, the only caller left is `packMambaLayer`** (`backend/cuda/mamba_resident.go`): every Mamba-2 layer's `in_proj` and `out_proj` are packed whole into
host memory while the build walks the layers, and held until `upW` uploads them. The shared experts and the drafter's weights the row names are already planned (`planWeight`, `planRows`); `packWeight` has no other caller outside tests.

## The change

`packMambaLayer` plans the two projections (`planWeight`) instead of packing them. A planned weight names its source and the sizes of the device buffers, and `upW` streams it (`h.src != nil`), exactly as it does every
attention and MLP projection. The kind check (`int8`, gemv_w8a8) reads the plan's kind, which `planWeight` gives as `packWeight` does for the same weight.

## Gates

Registered after the tier 0 tests were written red first (`red.log`) and before the tier 1 run.

| gate | tier | instrument | PASS iff |
|---|---|---|---|
| G-1 plans, packs nothing | 0 | `TestMambaHostCopy_planningHoldsNoPackedWords`: `packMambaLayer` on every Mamba-2 layer of `nemotron-tiny` | `in` and `out` hold 0 packed words, name a source, and name the buffer sizes `packWeight` makes. **Red first:** 2,856 and 1,088 words held per layer today |
| G-2 same device bytes | 0, the card | `TestMambaHostCopy_deviceBytesIdenticalToPackWeight`: each layer's projections uploaded planned against `packWeight`+`upW`, at one staging block and at several (64 words, 1 word) | every device buffer byte-equal |
| G-2b neighbours | 0, the card | `TestNemotronResidentCUDA_{tinyAgainstControl,slots,guards,pricing,int8Declines}`, `TestPackStream_*` | pass as on main |
| G-3 host peak | 1, the card | `TestMambaHostCopy_tier1PeakAnon`: Nemotron-Nano-9B-v2 Q8_0 whole (`~/models/nemotron/nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q8_0.gguf`, local NVMe), `Backend: "cuda"`, int4, token table at int4, ctx 4096 (the tier 1 test's "9b" cell): the load's **peak `RssAnon` and peak `VmSwap`** of the test process, sampled every 50 ms, and 16 greedy tokens after the first four of the cell's seed. The test binary is built from main plus only this test file, and from the branch; one run each, main first | the branch's peak `RssAnon` is at least **2 GiB** below main's (half the survey's 4 GB, which was arithmetic, not a measurement); the 16 tokens are equal; each run's peak `VmSwap` at most 512 MB; both resident (not declined) |
| G-4 identity | 1, the card | `go run ./cmd/gate identity -backend cuda <main> <branch>` | every family that ran IDENTICAL |

**PARKED:** a fall between 0.5 and 2 GiB, the others holding: reported, not shipped as the arithmetic's number. **FAIL:** a fall under 0.5 GiB, different tokens, or a swap peak over 512 MB.

- **Void:** a run with the one-minute load above 8.0 at its start, or the exclusive timing lock not held. Re-run once in order and listed.
- **Not the question:** speed (the plan is the same; the upload streams the same bytes), the first-token time, other families.
- **Cost and times (from `date`):** tier 0 seconds; G-3 two loads of a 9.4 GB checkpoint at about 3 to 6 minutes each; G-4 about 6 minutes. Stated again with the clock when each runs.
- **Void if:** a card run outside the lock, a NOT RUN counted as a pass, a bar moved after a reading.

## G-3 read once, 2026-11-05 16:04 PST: FAIL by the rule as registered

`g3-first/` (`g3-before.log`, `g3-after.log`, `summary.txt`). Main's test binary, then the branch's, one run each, the exclusive timing lock (free), one-minute load 0.54 and 1.72, 59 GB available, no void run. Both resident, **the same 16 greedy tokens**, peak `VmSwap` 0 MiB.

| | main | branch |
|---|---|---|
| peak `RssAnon` of the whole test process | 16,345 MiB | 16,577 MiB |

**No fall (+232 MiB), against a bar of at least 2 GiB: FAIL.** It stays on record as it ran. The load took 16 s and 12 s, so the checkpoint came from a cached sidecar, and the whole-process peak was set elsewhere than where the Mamba projections sit: `decoder.Load` builds the host model before `BuildResident` plans the layers, and a peak over the whole process reads that phase. The instrument could not see the packed copies, whatever they weigh. The survey's 4 GB was arithmetic, and this reading neither confirms nor refutes it.

## G-3b: the build's own window, registered 2026-11-05 16:10 PST before the reading

**What it adds, and why:** the same test, with a window opened where `BuildResident` begins planning its layers (`planStartForTest`) and closed when the load returns. It logs `RssAnon` at the window's start (the host model, loaded), the peak inside it, and the growth between them. The packed copies, if they exist, sit on top of the loaded model inside this window. **This is a new reading with a corrected instrument, after a FAIL; it does not replace G-3 above.**

- **Instrument:** `TestMambaHostCopy_tier1PeakAnon` with the window, built from main plus only that file (`cuda-mem13-before2.test`) and from the branch (`cuda-mem13-after2.test`); the same checkpoint, options and 16 tokens; one run each, main first; the exclusive timing lock.
- **Decision, on the window's growth (peak in window minus `RssAnon` at its start):**

| | |
|---|---|
| **PASS** | the branch's growth is at least 2 GiB below main's; the 16 tokens equal; each run's peak `VmSwap` at most 512 MB; both resident |
| **PARKED** | a fall of at least 0.5 GiB and under 2 GiB, the others holding: reported with the figure, not the arithmetic's |
| **FAIL** | a fall under 0.5 GiB, different tokens, or a swap peak over 512 MB |

If main's own growth is under 2 GiB the bar cannot be met by any change, and the reading says so: that is a finding about the ledger's arithmetic, not a pass.
- **Void and stopping rule:** as above; one run each, no repeat unless a run is void.
- **Cost:** about 20 s a run. Start from `date` at the run.
