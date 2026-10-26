## AX2d on nobara: AX2d-ID and AX2d-T (also AX2b-ID's CUDA and Linux CPU half)

**Verdict: PASS, identical before and after.** The tower records match hash for hash, and every CUDA package's results match by test name. The only failures are four that fail identically at origin/main (below), plus the known `TestCUDA_launchDirect` crash on both sides.

**Setup.** nobara: NVIDIA GeForce RTX 2070 SUPER (8 GB), driver 595.91.07, Go 1.27.3 linux/amd64. Before = origin/main `fb3156262`, which differs from this PR's head only by the AX2d import switch (85 files: imports, `cuda/go.mod`, docs). After = head `528ced858`. I used two detached worktrees, each with a `go.work` of `.` and `./cuda`, and ran under the timing lock, exclusive so nothing else shared the card, 16:23 to 16:41 PDT. Load average 0.9 to 3.5 and 58 GB available at every start; nothing else was queued. `kwin_wayland` held a CUDA context (17 MiB), so a desktop session was logged in.

### Tower identity (`ax2_compare.py`, no tolerance)

| pair | CPU linux/amd64 | CUDA linux/amd64 |
|---|---|---|
| B `7d0727c0` vs H2 `67bcf5d4` (AX2b-ID) | 22 cases, 0 differ | 17 cases, 0 differ |
| H2 vs D `528ced85` (AX2d-ID) | 22, 0 differ | 17, 0 differ |
| origin/main vs D | 22, 0 differ | 17, 0 differ |

All 17 CUDA cases record `device: true` at B, H2, main and D, and all four records are equal:

| family | input | device (main / D) | sha256 (first 12) | equal at B, H2, main, D |
|---|---|---|---|---|
| siglip | normal-9 | True / True | `3d0f9b628136` | yes |
| qwen2.5-vl | [[1 8 8]] | True / True | `269467a96da4` | yes |
| qwen2.5-vl | [[1 12 16]] | True / True | `332cee8f8b75` | yes |
| qwen3 | [[1 4 4]] | True / True | `70292239bb92` | yes |
| qwen3 | [[1 6 8]] | True / True | `a33b7eb6d8d4` | yes |
| qwen3 | [[1 10 6]] | True / True | `b3197ddce09b` | yes |
| qwen3 | [[1 4 4] [1 6 4]] | True / True | `19f140ecc9a4` | yes |
| qwen3-deepstack | [[1 4 6]] | True / True | `5803987f13d9` | yes |
| qwen3-deepstack | [[1 6 4] [1 4 4]] | True / True | `20b3da6edd5f` | yes |
| glm-ocr | [[1 4 4]] | True / True | `445e1731e5d2` | yes |
| glm-ocr | [[1 6 8]] | True / True | `8aff8319db1c` | yes |
| glm-ocr | [[1 10 6]] | True / True | `8914bddf032e` | yes |
| glm-ocr | [[1 4 4] [1 6 4]] | True / True | `5e2bc50df847` | yes |
| gemma4-vision | 2x3 pooled | True / True | `3141f5150054` | yes |
| gemma4-vision | 3x2 pooled | True / True | `d05256cc58fa` | yes |
| whisper | normal-80 | True / True | `f6023fd08d45` | yes |
| voxtral | normal-81 | True / True | `4d6eb4426d10` | yes |

### Device tests: `go test -v -count=1 -timeout 60m -tags 'cuda goinfer_testhooks' -skip '^TestCUDA_launchDirect$'`, one package per call, before then after

| package | before PASS / FAIL / SKIP | after PASS / FAIL / SKIP | by name |
|---|---|---|---|
| `cuda` | 792 / 4 / 338 (8 m 26 s) | 792 / 4 / 338 (8 m 23 s) | 0 lost passes; the same 4 failures and the same 338 skips |
| `cuda/internal/gpudev` | 65 / 0 / 0 | 65 / 0 / 0 | identical |
| `cuda/internal/qwencuda` | 4 / 0 / 0 | 4 / 0 / 0 | identical |
| `cuda/internal/visioncuda` | 4 / 0 / 0 | 4 / 0 / 0 | identical |
| `cuda/s4siglip` | 0 / 0 / 1 (`TestVisionCUDA_real`) | 0 / 0 / 1 | identical |
| `-run '^TestCUDA_launchDirect$' ./internal/gpudev/` | segfault | segfault | the known crash, which aikit has at `gpu/v0.35.0` too (#279); not investigated |

`ax1_compare.py` reports `cuda` as "regressions 4" because it counts any failure in the after log. All four also fail before, with the same messages and the same figures, so this PR did not cause them. **They are red on origin/main on this card**, which is worth a separate look (the most recent commit to `cuda/kv_slots_test.go` on main is `b0aa5d367`, #264):

- `TestCUDAKVSlots_pricedAgainstWhatIsLeft`: `kv_slots_test.go:322: the free-VRAM probe ran 2 times during the build, want once (at checkKVFits)`
- `TestResidentDenseBytes_matchesCUDADevice`, with its subtests `qwen2.5-coder-0.5b-instruct-q4_k_m.gguf` and `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf`: "the build put 940 MB on the device before its KV, 479 MB more than Plan's 461 MB: more than the 214 MB margin reserved for scratch" (0.5B), and 2046 MB against 1048 MB with a 248 MB margin (1.5B)

**Logs** are on nobara in `~/goinfer-bench/ax2d-logs/`: `run.log` (phases, plus load, memory and GPU at each start), `id-<rev>-{cpu,cuda}.log`, the records `<rev>-{cpu,cuda}-linux-amd64.json`, `t-{base,head}-<pkg>.log`, `t-{base,head}-launchDirect.log`, and `cmp-<pkg>.txt`.

Not pushed to this branch; not merged. (nobara-ax1-cuda)
