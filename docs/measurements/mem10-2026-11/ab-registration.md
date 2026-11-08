# MEM10 tier 1: expert slots uploaded from pageable pages, against today's pinned copy (registered before any code, nobara-sb01, 2026-11-08 12:05 PST)

**Why.**
- Francis, 2026-11-08 10:04 (via the coordinator): measure the unpinned path.
- MEM10's tier 0 (README, "MEM10 tier 0") found that CUDA cannot pin a mapping of the `.giw` in place on this box without copying it. So the
  ~12 GiB saving a pre-permuted `cuda` target was meant to give is only reachable by not pinning at all: the expert slots upload from the
  mapping's pageable pages, and the driver stages each copy.
- This A/B measures what that costs the 26B's decode, and what it saves in host memory.

## Arms (one binary, one process, a test-only switch)

- **A: today.** `packStackPinned` packs each expert stack into freshly allocated pinned host memory (`NewMappedHostBuffer`, the `RssShmem`
  region), and the slots upload from it.
- **B: pageable.** A test-only switch (`moePageableDirForTest`, no env read, no production path) makes `packStackPinned` write the same
  packed bytes to a file on the NVMe disk. It then maps the file read-only (`PROT_READ`, `MAP_SHARED`), pins nothing, and the slots upload
  from those pages.
  - The async DMA path (the overlap, default on) issues `cuMemcpyHtoDAsync` from the pageable pointer. The batched path issues
    `UploadBatch` from the same bytes.
  - The file stands in for the `.giw` region a pre-permuted target would hold. Its pages are page cache (`RssFile`), as the `.giw`'s are.
  - The canonical expert ranges of the real `.giw` stay mapped as in A. Packing reads them once, and decode does not touch them again.
- **Correctness check (hard, before any timing):** for every expert of the first MoE layer's two stacks, B's file bytes equal A's pinned
  bytes. On the first B load, the first 64 greedy tokens of the short-context decode equal A's.

## Instrument

- `TestMEM10_pageableExpertSourceAB` (`backend/cuda`, `-tags 'cuda goinfer_testhooks'`, `GOINFER_HEAVY_TESTS=1`).
- **Model:** `~/models/gemma4-26b-q4_k_m.gguf` (local NVMe) through its sidecar `gemma4-26b-q4_k_m.int4.e4h.cuda.v17.giw`, the load
  MEM10-G0 read. Options are MoE paged experts (`MoECacheExperts`), int4, `ResidentContext` 4,352. Two 26B residents do not fit the card, so
  each arm is its own load, closed (with `debug.FreeOSMemory`) before the next.
- **Order:** A B B A A B B A, eight loads, four pairs per metric. The test binary runs directly under the exclusive timing lock.
- **Per load:**
  - **Short context:** a 32-token pseudo-random teacher-forced prompt, then 256 greedy tokens. Decode tok/s is 256 over their wall time.
  - **Long context:** a 4,096-token prompt (the same seed) through `PrefillLast`, then 256 greedy tokens. Decode tok/s as above.
  - **Memory**, read from `/proc/self/status` after load and after the long decode: `RssAnon`, `RssShmem`, `RssFile`, `VmSwap` and `VmHWM`;
    from `/proc/self/smaps`, the Rss of the `.giw` mapping and of B's file. System `SwapTotal - SwapFree` is reported as context only.
- **Void (a cell's load is re-run in place, and listed):** a 1-minute load average above 8.0 at the load's start, or `MemAvailable` under the
  26B's need plus 2 GB (30 GB here: about 14 GB of sidecar plus 12 GiB of copy, plus margin).

## Rule (fixed now)

The statistic is the paired B/A decode tok/s ratio, each B load with its adjacent A load: four pairs per context, and their median.

- **SHIP-WORTHY:** the median ratio is at least **0.97** at **both** contexts, **and** B's `RssShmem` is at least **11 GiB** under A's (the
  saving is real), **and** B's process `VmSwap` stays 0.
- **KILL:** the median ratio is under **0.90** at **either** context.
- **AMBIGUOUS, PARKED** (to Francis with the numbers): anything between, or a ship-worthy speed without the memory saving.
- A ship-worthy reading does not ship code by itself. It makes the pre-permuted target worth building: a `.giw` format variant whose
  expert region is the packed layout, so B's file is not needed. That build gets its own registration.

**Prediction:** short context 0.85 to 0.97 (0.5): an 8 GiB card keeps only a few dozen slots, so a 26B decode token misses about a hundred
expert slots, each a 2 to 3 MB pageable copy staged through the driver's bounce buffer. Long context is closer to 1.0 than short (0.6):
attention takes a larger share of each token there.

**Cost:** about 30 min. A 26B load is about 1 min (the sidecar is mapped). B's first pack writes about 12 GiB to NVMe, about 1 min. A 4,096
prefill is about 30 s, plus 2 x 256 decode tokens at about 20 to 30 tok/s. That is about 3 to 4 min a load, eight loads. By day; start and
finish go in the status file first.

## Addendum, 2026-11-08 12:11 PST: the first attempt measured the wrong load; the instrument fixed, no rule changed

- **What happened.** The first attempt (12:09) loaded `gemma4-26b-q4_k_m.gguf` with `Quant: "int4"` but without `EmbedInt4`.
  - Serve sets `EmbedInt4` on CUDA, and that is what makes it open `….int4.e4h.cuda.v17.giw`. Without it, no sidecar matched, so the
    load re-packed the weights in-process: arm A's `RssAnon` was 16,101 MiB, and the `.giw`'s Rss was 0.
  - Its `RssShmem` (12,345 MiB) did match MEM10-G0's region.
- **How it stopped.** Load 2 (arm B) was refused by the host-memory guard: "needs ~38.4 GB … (13.2 GB weights + 13.2 GB packed copy + 12.0
  GB pinned copy); 46.0 GB available". The refusal was not bypassed.
- **The fix.** The test now sets `EmbedInt4: true`, serve's own CUDA default, so it measures the registered load through the sidecar.
- **The readings.** The first attempt's single load is void and is not used. Its log is kept as `ab-attempt1/`.
- **Unchanged:** the arms, order, metrics and rule.
