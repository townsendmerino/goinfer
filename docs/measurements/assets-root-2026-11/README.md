# The asset registry from a test two levels down (nobara-sb01, 2026-11-06)

`decoder/assets.go:repoRoot` was the constant `".."`. The GPU backends moved to `backend/<name>/` on 2026-11-03, so from `backend/cuda` and `backend/metal` the registry opened
`../testdata/assets.json`, failed, and every `decoder.AssetPathForTest` call skipped (`asset registry unreadable`). A skipped test prints `ok`; nothing went red. Found 2026-11-06 while reading the MEM11 gate (`../mem11-cuda-2026-11/README.md`).

- **Red:** `red.log`: `TestAssetRegistry_resolvesFromEveryCallerDir` (decoder) chdirs into every directory that has a test calling the registry and loads it; `backend/cuda` and `backend/metal` FAIL, `decoder` passes.
- **Change:** `assetRoot()` walks up from the working directory to the nearest ancestor holding `testdata/assets.json` and returns it as `..`, `../..`; `repoRoot` stays `..` for decoder's own golden paths. `loadAssetRegistry` and the `$REPO/` expansion use it. `green.log`: the four registry tests pass, three caller directories.
- **Parity manifest:** not stale (`TestParityManifest*`, `TestParityKey*` pass); `assets.go` is not a parity input, so there is no refresh. `touched.log`: 12 checks green.
- **`unskipped.txt`:** every test function that asks the registry directly, by package, with its build tags and asset env. Two more reach it through `requireCUDAResidentMRoPE`
  (`TestRopeKVMRoPEBatched_degenerateMatchesScalarKernel`, `TestRopeKVMRoPEBatched_matchesCPUReference`). Most also need `GOINFER_HEAVY_TESTS=1`; without it they still skip, by design.
  Not changed by this PR: what each test then does on its first real run. CUDA ones are read in `reading/` (a later commit); the Metal ones are `backend/metal`'s nine, for the Mac.
