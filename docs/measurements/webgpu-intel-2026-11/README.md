# WG4-intel: WebGPU on an Intel integrated GPU (mbp2, 2026-11-20)

WebGPU campaign row WG4-intel (goinfer-notes, `tasks/task-webgpu-2026-11.md`, § "WG4-intel"): the first run of goinfer's
WebGPU backend on a GPU vendor other than NVIDIA or Apple. **Correctness only.** Nothing timed here is published; the tok/s
lines in the logs come from tests that print them anyway and say nothing about this card's speed.

**Verdict.** G-WG4i-1 (adapter), G-WG4i-2 (correctness) and G-WG4i-3 (plants) are met. The WGSL kernels compile and run
on the Intel driver and agree with the CPU reference at the existing bars. Two tests went red, and neither is an Intel
fault: `TestTowerGolden_webgpu` refuses a golden pinned to the RTX 2070 SUPER, as it is built to; and `TestE2E_webgpu`'s
0.5B int8int8 cell starts from a near-tied first token (§ Findings). No fix was made, so there is nothing to check on
NVIDIA or Metal for bit-identity.

## The machine and the build

- **mbp2:** MacBook Pro 13-inch 2020 running Kubuntu (Ubuntu 26.04.1, kernel 7.3.3-2-t2-resolute), Core i5-8257U
  (8 logical CPUs, AVX2), 15.5 GiB host RAM shared with the GPU.
- **GPU:** Intel Iris Plus Graphics 645 (Coffee Lake GT3, integrated), Mesa 26.0.8 (Ubuntu's 26.0.8-1ubuntu0.3), the
  Vulkan backend of wgpu-native (`github.com/oliverbestmann/webgpu` v1.38.0).
- **goinfer** `018cbe74c` (main, 2026-11-20), aikit v1.76.0, go1.27.2. Clean worktree; the gate record says `dirty: false`.
- **Device access:** the session's login predates `francist` joining the `render` group, so every GPU command ran under
  `sg render -c '…'`. A plain command cannot open the render node, and wgpu then falls back to llvmpipe without
  an error (MM5's WebGPU half of 2026-11-15 was measured that way). Step 1 below is what shows the hardware adapter was picked.
- **Models:** qwen2.5-coder 0.5B and 1.5B instruct (Q4_K_M GGUF) from `~/models` on mbp2. The 1.5B GGUF and the
  gemma-3-270m checkpoint were copied from nobara's local disk that morning (nobara's lock was held shared, not
  exclusively), and gemma-3-270m was linked into the worktree's gitignored testdata. The int4 bundles were transcoded
  on mbp2 with `go run ./cmd/prequant -quant int4 -target webgpu`. All builds and tests held the timing lock shared.

## Step 1: the adapter (G-WG4i-1)

```
sg render -c 'go test -tags gpu ./backend/webgpu/internal/kernels/ -run TestAdapterProbe -v -count=1'
ADAPTER_PROBE: backend=vulkan software=false
```

## Step 2: the hardware report (G-WG4i-1)

`go build -tags gpu -o goinfer-serve github.com/townsendmerino/goinfer/backend/webgpu/cmd/serve`, then
`sg render -c 'goinfer-serve check --hardware'`. (The root `cmd/serve` refuses `-tags gpu`, by design.) The whole block:

```
goinfer: v0.24.2-0.20261120185135-018cbe74c0f2 (018cbe74), go1.27.2, linux/amd64
os: Ubuntu 26.04.1 LTS, kernel 7.3.3-2-t2-resolute
cpu: Intel(R) Core(TM) i5-8257U CPU @ 1.40GHz; 8 logical CPUs
cpu kernels (amd64): detected [avx2 f16c popcnt]; in use [avx2 f16c popcnt]; forced by build tags none
memory: 15.5 GiB host RAM, 11.4 GiB available
backends linked into this binary: cpu, webgpu
  webgpu: adapter: Intel(R) Iris(R) Plus Graphics 645 (CFL GT3)
  webgpu: vendor: Intel open-source Mesa driver, architecture: unknown
  webgpu: graphics API: vulkan; adapter type: integrated-gpu
  webgpu: limits: max buffer 2.0 GiB, max storage binding 2.0 GiB, workgroup memory 48 KiB, 1024 invocations per workgroup
  webgpu: integer dot-product (dot4I8Packed): yes
  webgpu: memory and driver version: not reported by WebGPU
self-tests (each runs the kernels this process would use against a reference on a small fixed input):
  cpu pass (2.0 ms)
  webgpu pass (2332.1 ms), worst cosine 0.99988 (floor 0.995), worst relative L2 0.0155 (ceiling 0.10) over 4 fixtures
```

**`dot4I8Packed`: yes.** This GPU has the integer dot product, so the run exercised the same dp4a kernels as the NVIDIA
and Apple adapters. WebGPU's non-dp4a fallbacks are still unrun on hardware.

## Step 3: the gate

`sg render -c 'go run ./cmd/gate gpu -record <dir> -logdir <dir>/raw'`, 10:54 to 10:57 PST. The backend is `none`
(no nvidia-smi, not darwin), so the gate ran the seam, group W and repo hygiene. The heavy tier is CUDA's and did not
run. With no `-heavy-group` flag, nothing was narrowed. The verdict is `FAIL`, from one test:

| group | result |
|---|---|
| 1. seam | PASS |
| W. webgpu-suite (`-tags gpu -short`, both WebGPU packages) | 109 pass, 38 skip, **1 fail**: `TestTowerGolden_webgpu` |
| W. webgpu-parity | not run by the gate: it runs only after the suite passes |
| 5. repo hygiene | PASS (citation lint; 23 darwin-only steps skipped) |

**The suite's one failure is an anchor refusal, not numerics.** `TestTowerGolden_webgpu` compares the vision towers'
output bytes against `internal/diag/towerid/testdata/towers.webgpu-linux-amd64.json`, which is anchored to
`webgpu vulkan NVIDIA GeForce RTX 2070 SUPER (595.112.07)`. On another adapter it fails by design ("re-anchor it
deliberately, never carry it forward"). All 15 of the Iris's tower hashes differ from the NVIDIA ones, which is what two
drivers' float reduction orders give. The towers' agreement with the CPU, `TestTowerIdentity_webgpu`, passed in the same
run. Re-anchoring the golden to this adapter would swap NVIDIA's anchor for Intel's, so it was not done. A golden per
adapter is a design question for the towerid owner, not this row.

**The webgpu-parity cell, by hand**, with the gate's own arguments (`-tags 'gpu goinfer_testhooks'`,
`-run 'Parity|_parity|matchesCPU|matchesSequential'`, `GOINFER_DNET_PARITY=1 GOINFER_SSM_PARITY=1`, `-p 1`, 10 m):
**74 pass, 12 skip, 0 fail.** The skips are missing fixtures (three Gemma 4 tiny fixtures, the Granite and Mamba
weights, a Qwen3 1.7B GGUF, a phi3-shaped checkpoint) or the heavy opt-in, and step 4 ran the heavy ones that matter here.

**Classification of every red cell (G-WG4i-2):** none is a driver, WGSL-portability or kernel-limit fault. Every
non-skipped cell of group W passes apart from the one anchor refusal above.

## Step 4: real checkpoints

**The tests the row names**, `-tags 'gpu goinfer_testhooks'`:

| test | result |
|---|---|
| `TestWebGPU_forwardParity` (gemma-3-270m, greedy token against the f32 oracle) | PASS |
| `TestResidentSelfTest_passesOnHealthyWebGPU` int4 | pass: worst cosine **0.99988**, relative L2 **0.0155** (bars 0.995 / 0.10) |
| `TestResidentSelfTest_passesOnHealthyWebGPU` int8int8 | pass: worst cosine **0.99929**, relative L2 **0.0378** |
| `TestResidentSelfTest_goesRedOnWrongResidentInputsWebGPU`, rms epsilon | **declined**: cosine 0.67057, relative L2 0.9830 |
| `TestResidentSelfTest_goesRedOnWrongResidentInputsWebGPU`, rope base | **declined**: cosine 0.92499, relative L2 0.3875 |

The worst-case self-test figures equal NVIDIA's (CHANGELOG, the WebGPU self-test entry) and the Mac's
(census `webgpu-apple-metal`) to five digits. The worst fixture is set by quantization, not by the adapter's arithmetic.

**The 0.5B and 1.5B.** The named tests use gemma-3-270m and the four tiny fixtures; none of them takes a qwen GGUF. So
the qwen half ran the tree's real-checkpoint WebGPU tests: `GOINFER_HEAVY_TESTS=1`, with `GOINFER_RESIDENT_GGUF` and
`GOINFER_E2E_MODEL` set to the GGUF and `GOINFER_W4A8_GIW` to the int4 webgpu bundle.

| test | what it compares | 0.5B | 1.5B |
|---|---|---|---|
| `TestResidency_matchesCPU` | int4 resident against the CPU's int4 greedy, 16 tokens, a chat prompt | PASS, 16/16 | PASS, 16/16 |
| `TestE2E_webgpu` | int8int8 WebGPU against CPU greedy, 16 tokens from ids 1..8 | **FAIL, 0/16** (§ Findings) | PASS, 16/16 |
| `TestResidentForwardN_parity` | int8int8 batched `ForwardN` against sequential `Forward`, 5 rows | PASS, cosine 1.000000, max diff 0 | PASS, same |
| `TestResidentPrefillLast_parity` | int8int8 prefill-last against the full pass | PASS, cosine 1.000000, max diff 0 | PASS, same |
| `TestGenerate_batchedPrefillMatchesSequential` | int8int8 batched against sequential prefill, 24 tokens | PASS, identical | PASS, identical |

## Step 5: the memory gate on hardware (G-WG4i-3's companion)

`TestMemGate_webgpu` and `TestMemGate_webgpuPlantsThroughTheTable`, `-tags 'gpu goinfer_testhooks'`:
**244 builds held, 39 options declined by WebGPU's own build; the plan's decode scratch covered 244**. These are the
same counts as MM5's llvmpipe reading and the RTX 2070 SUPER's (`docs/measurements/mm5b-scratch-read-2026-11-15/README.md`,
§ "WebGPU on the same card"). The only over-plan lines are the known stacked-experts gap on `qwen3vlmoe-tiny` (MM8
WebGPU), which the gate forgives by name. All four plants (dense, experts, kv, state) stay red through the table.

## Findings

1. **`TestE2E_webgpu`'s 0.5B int8int8 cell is a near-tie on a degenerate prompt, not a kernel fault.** The test asserts
   that a first-token divergence means a wiring bug. Its prompt is token ids 1..8, a run of punctuation. A diagnostic
   (`logs/first-token-margins-diag.go.txt`, a test file that was run and not committed) read the first-token logits from
   the CPU (`PromptLogitsMany`) and from the resident (`ResidentPrefillSeedForTest`) on both models and quants:

   | model, quant, prompt | cosine (full vocab) | CPU top-1 / top-2 (margin) | WebGPU top-1 / top-2 (margin) |
   |---|---|---|---|
   | 0.5B int8int8, 1..8 | 0.995815 | 31 / 1294 (**0.037**) | 1294 / 31 (0.092) |
   | 0.5B int8int8, 1..4 | 0.996767 | 3 / 5 (0.688) | 3 / 5 (0.632) |
   | 0.5B int4, 1..8 | 0.997585 | 31 / 21122 (0.634) | 31 / 21122 (0.453) |
   | 0.5B int4, 1..4 | 0.996430 | 3 / 2 (1.626) | 3 / 5 (1.677) |
   | 1.5B int8int8, 1..8 | 0.994538 | 353 / 3764 (0.711) | 353 / 3764 (0.410) |
   | 1.5B int8int8, 1..4 | 0.996387 | 5 / 3 (2.341) | 5 / 2 (2.615) |
   | 1.5B int4, 1..8 | 0.994107 | 353 / 3764 (1.554) | 353 / 3764 (0.944) |
   | 1.5B int4, 1..4 | 0.994133 | 5 / 2 (2.267) | 5 / 3 (1.868) |

   On the failing cell the CPU's top two tokens are 0.037 logits apart and WebGPU swaps them. In the other seven cells
   the argmax agrees. With the first token swapped, the GPU's 16 tokens contain the CPU's own continuation six positions
   later (CPU 31 48077 45899 5 9 368 62; GPU positions 6 to 12: the same seven). The test's bar is wrong for this prompt
   on any adapter whose rounding differs from the CPU's. It has never run on nobara: its one record there is a skip with
   no `GOINFER_E2E_MODEL`. Whether the RTX 2070 SUPER flips the same tie is unmeasured. The check to book on nobara is in
   the PR that carries this record. A changed prompt or a margin-aware bar is a test fix for the backend's owner.
2. **The relative L2 of a whole-model prefill against the CPU is 0.07 to 0.12 here,** above the self-test's 0.10
   ceiling on the 1.5B. It is not a gate: the self-test's ceiling is for its tiny fixtures, and no recorded NVIDIA or
   Metal figure exists for this measurement to compare against. It is written down as the first such reading.
3. **`TestTowerGolden_webgpu` cannot pass on a second Linux WebGPU adapter.** A golden keyed by GOOS/GOARCH holds one
   anchor, so every vendor but the anchored one fails it (§ Step 3).

## Logs

`logs/`: each step's output (`step1-adapter.log` to `step5-memgate.log`), the gate's record (`gate-gpu-record.json`)
and the `go test -json` streams of its suite cell and of the hand-run parity cell, the diagnostic's source and output
(`diag-margins-*.log`).
