# MEM11-giw-bf16: LFM2's bf16 islands mapped from the .giw, 2026-10-22

Gates: `docs/tasks/task-memory-campaign-2026-10.md`, "MEM11-giw-bf16 gates" (GB-G), registered before the code (8cd7b2eb), on
D13 (b) (owner, 2026-10-22). Machine: MacBook Pro, M1 Pro, 16 GB, macOS 26.6.2, Go 1.27.2. Nothing here is timed.

**The build.** A new weight-matrix kind, 9 (bf16): one 16-aligned binary16 array of rows×cols values, aliased from the mapping
through `linalg.WrapBF16`. `giwWriter.island` writes an island matrix the load holds bf16 as a kind-9 record, and an f32 one as the
standalone f32 array it always was. `giwReader.island` tells them apart by the encoding byte, so a bundle written before this still
loads. Under `docs/giw-bundles.md` § "From v17" it is a new encoding and not a layout change, so the version stays 17. An older
build refuses a kind-9 bundle by name (`unknownEncoding`) and rebuilds its sidecar. `mmapAliasedBytes` counts an aliased bf16
matrix, and the short-conv islands go through the same count as every other matrix.

| gate | reading | result |
|---|---|---|
| GB-G1 identity | `TestMEM11_giwBF16Islands`, lfm2-tiny with its islands rounded to bf16, written as a `.giw` and loaded mapped: the islands come back bf16 with all 98,304 B aliased, and 0 of the logits over 16 tokens differ from the direct load. The committed fixture (f32 islands, not bf16-exact) comes back f32 with 0 aliased bytes and 0 logits differing | **PASS** |
| GB-G2 heap, tier 0 | the same bundles: `ResidentWeightBytes` 608,512 B for both forms; `ResidentHostCopyBytes` 608,512 → 510,208 B, lower by exactly the islands' 98,304 B | **PASS** |
| GB-G2 heap, tier 1 | `TestMEM11_giwBF16HeapLFM2`, LFM2.5-2.6B at int4 written as a `.giw` in each form and loaded mapped, one at a time: Go heap in use after GC **0.748 → 0.014 GB, a fall of 0.734 GB, 99.4% of the islands' 0.738 GB** (bar: within 5%); 0 logits over 8 tokens differ between the two. Log: `~/goinfer-logs/mem11-gb-g2-heap.log`, 23 s | **PASS** |
| GB-G3 an older bundle | the tiny bundle in the f32 form (the writer seam) and one written by `origin/main`'s build (135974e9 plus docs): both load, their islands bf16 on the heap (0 aliased), 0 logits differing | **PASS** |
| GB-G4 unchanged | the serialize tests and every family's round trip pass; `scripts/refresh_parity_hashes.sh` green | **PASS** |

**Plant** (the writer takes kind 9 for an f32 island, rounding it): red. The committed fixture's islands came back bf16 with 4,053
logits differing, and `TestLFM2_serializeRoundTripsTheConvMixer` read `-0.008605957` for `-0.008598174`.

**What the first tier-1 run found.** The first run faulted while writing the 2.6B's `.giw` (SIGSEGV in the writer's CRC). That was
not this change. It was a MEM11-bf16 defect on main: a direct load of an all-BF16 checkpoint closed the mapping its islands alias.
It is fixed separately (#165, `docs/measurements/mem11-2026-10/README.md`). This branch carries that fix, and the reading above
was taken with it.

`TestB27_unknownEncodingRefusedByName` used 9 as its unknown encoding. It now uses 10, since 9 is read.
