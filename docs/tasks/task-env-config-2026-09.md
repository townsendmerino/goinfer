# Task: configuration out of the process environment (2026-09)

> **Status 2026-09-24: phase 1 (the ratchet) DONE; phases 2a/2b (21 decoder knobs) and 3 (11 CUDA knobs) DONE; 4–6 open.** Owner asked for this after a review found
> configuration passed through the process environment. The four concrete defects that review named are already fixed
> (`a50815ed`: duplicate doc rows, the darwin pager mode set via env, Metal's prefill flag re-read per call, five campaign
> switches retired); this doc is the general program.

## Why

Production code reads **103** distinct `GOINFER_*` variables. The environment is process-global and mutable, so:

- **A knob read per call changes a loaded model mid-flight.** 55 of the
  61 operator knobs are read on every generation or prefill, not once.
- **Two models in one process cannot differ.** serve loads several models; a library caller may too. A knob that is really a
  per-model choice cannot be set per model when its only channel is `os.Getenv`.
- **The app edge and the library disagree.** serve translated its flags into env vars (`--exact-prefill`, `--moe-pager` until
  `a50815ed`), so a library caller silently got different defaults — measured twice in 2026-09 (`ExactPrefill` sticky across
  loads; darwin pager mmap-vs-pool).
- **Diagnostics masquerade as configuration.** 41 of the reads are campaign
  switches; a knob with no owner never gets removed.

`Options.ResidentContext` → `Model.ResidentContextRequest()` is the shape that already works: set once at `Load`, carried
by the model, read by the backend from the model.

## Rules

1. **No new production `os.Getenv` / `os.LookupEnv` of a `GOINFER_*` variable.** A new operator choice is a field on
   `decoder.Options` (or the app's own config) with a per-model accessor. A new diagnostic is a test hook
   (`goinfer_testhooks`) or a package variable a test sets. Enforced by the ratchet (phase 1).
2. **Migrating a knob keeps its env var working as an override**, read once at the edge (serve/chat flag parsing, or `Load`
   for library callers that set nothing), never per call. Removing an env var is a separate, announced step.
3. **Default behaviour is unchanged by a migration** — each phase's gate is bit-identical output on the default path.
4. `docs/env-vars.md` stays the single list; a migrated knob's row names its `Options` field.

## Phases

| phase | scope | gate |
|---|---|---|
| **1. Ratchet** | `testdata/env_reads.txt`: the committed list of every production `GOINFER_*` read. `TestEnvVars_docAndCodeAgree` fails on a production read not in the list (rule 1) and on a listed variable no longer read (so the list only shrinks). | the test is red on a planted new read and on a stale entry |
| **2. decoder operator knobs** | a `decoder.Knobs` struct snapshotted ONCE per model at `Load` (from `Options.Knobs` when set, else from the environment); every per-call read in `decoder/` reads `m.knobs` | bit-identical greedy + logits on the tiny goldens with default knobs; each knob's env override still works (one test per knob group); list shrinks by the decoder operator set |
| **3. CUDA operator knobs** | read once at resident build from the model (as `ExactPrefill` already is) | CUDA hardware test: default path bit-identical; overrides honoured |
| **4. Metal operator knobs** | same, on the Metal resident | metal-darwin CI + a Mac run |
| **5. serve / cmd / gpu / internal** | serve and chat parse flags into `Options`; no `os.Setenv` in app code (`applyExactPrefillEnv` is the last one) | serveapp suite; `serve check` |
| **6. diagnostics** | retire when the campaign's question is answered, else move to test hooks — case by case, with the owner | per retirement, the record it answered |

### Phase 2a design (decided 2026-09-24, before the code)

The decoder's 14 per-call operator knobs (`ATTN_GROUPED`, `ATTN_ROW_TILE`, `PREFILL_ATTN_WORKERS`, `FUSED_ATTENTION`,
`MLA_NAIVE`, `MOE_EXPERT_MAJOR`, `BATCHED_PREFILL`, `NO_KVONLY_PREFILL`, `NO_GREEDY_FASTPATH`, `NO_OPTFWD`,
`NO_SAMPLE_FASTPATH`, `NO_TOPK_FASTPATH`, `OPTFWD_MAX_TEMP`, `CPU_FAST_ATTENTION`) go first: they are the ones that let a
process-environment change alter an already-loaded model.

- **Snapshot, raw values.** At `Load`, each model captures those 14 variables once (value + whether set) into a knob set
  hung off `Model` and its per-model `Architecture` (every deep helper already receives one of the two).
  `Options.Knobs` (a name→value map) overrides the environment per model — the library's way to configure two models
  differently. Each helper keeps its exact parsing and reads the snapshot instead of `os.Getenv`, so defaults and
  overrides mean what they meant. Typed `Options` fields are a later, separate step.
- **Drift tripwire.** ~50 test files (decoder, cuda, gpu, metal) A/B two paths by `t.Setenv`-ing one of these AFTER
  loading a model. With a snapshot, such a test would silently compare a path with itself — an identity test would pass
  vacuously. So in `goinfer_testhooks` builds every snapshot read also checks the live environment and panics, naming the
  variable and the fix, when it differs from the snapshot for a knob not set through `Options.Knobs` or the per-model
  test setter (`SetKnobForTest`). Production builds read only the snapshot. A stale A/B therefore fails loudly wherever it
  runs, including the Mac, instead of passing for the wrong reason.
- **Gate.** Tiny-golden forward parity and the decoder/cuda suites green with default knobs; every test the tripwire
  catches migrated to the setter; the 14 names leave `testdata/env_reads.txt` (their only reader becomes the snapshot).

### Phase 2a result (2026-09-24)

Built as designed: `decoder/knobs.go` (the snapshot and each knob's parser), `knobs_drift_testhooks.go` (tripwire
+ `SetKnobForTest` / `SetKnobEnvForTest`), `knobs_drift_prod.go` (no-op). What differed from the design, or was
learned doing it:

- **13 names left `testdata/env_reads.txt`, not 14.** `GOINFER_MOE_EXPERT_MAJOR` is also read by Metal's own prefill
  (`metal/prefill.go`), so it stays listed until phase 4. `GOINFER_BATCHED_PREFILL` had a second reader the design
  missed (`Model.PrefillPath`, `decoder/residency.go`); it reads the snapshot too.
- **A nil snapshot reads the live environment.** Unit tests that build an `Architecture`, scratch or worker pool by
  hand (no `Model`, no `Load`) keep working unchanged: `attn_grouped_test.go`, the pool tests in
  `prefillattnpool_test.go`, `spec_optfwd_test.go`'s hand-made model.
- **Untagged decoder tests cannot see the exported testhooks setter**, and they are exactly where the tripwire does not
  run. So in-package tests use `setKnob` / `unsetKnob` (`decoder/knob_helpers_test.go`, untagged): same effect,
  environment plus pin. Cross-module tests use `decoder.SetKnobEnvForTest`. `metal/optfwd_test.go` (a real-model
  local test CI never ran) moved to `darwin && goinfer_testhooks` to reach it.
- **Migrated:** 24 post-Load sites in 14 decoder files, 13 in 6 cuda files, 4 in 2 gpu files, 7 in 4 metal files.
  Metal's `moe_expert_major_prefill_test.go` was left alone: it drives the Metal resident directly, which still reads
  the environment.
- **Gate:** `TestKnobs_*` (snapshot at Load, `Options.Knobs` per model, nil snapshot) and
  `TestKnobDrift_firesOnPostLoadSetenv` (the tripwire goes red on a post-Load `t.Setenv` and stays quiet for both
  sanctioned routes). Forward goldens 62/62 green on amd64 via `scripts/refresh_parity_hashes.sh`.

### Phase 2b result (2026-09-24)

Seven more decoder operator knobs join the snapshot: `MOE_CACHE_EXPERTS`, `MOE_CACHE_SLOTS`, `NO_FIT_DEFAULT`,
`NO_FIT_GUARD`, `NO_RESIDENCY`, `NO_RESIDENT_REUSE`, `SSM_RESIDENT`. All seven leave `testdata/env_reads.txt` (no other
module reads them directly; CUDA reaches them through the model's accessors).

- **The fit guards run inside Load before the model exists**, so they read through `loadKnob(opts, name)`:
  `Options.Knobs`, else the environment — the snapshot's own precedence, at the same moment.
  `TestFitGuard_optionsKnobLoads` pins that an override reaches the guard (with a control arm that must refuse).
- **`NewModel` (the public wrap of pre-built weights, used by `internal/chatapp`) never bound a snapshot**, so in phase 2a
  its models still read all fourteen knobs live per call. It binds one now.
- **`Options.Knobs` became `*Knobs`** (`type Knobs map[string]string`) in `7c9be450`: the map field made `Options`
  non-comparable, a hard-tier API break the apidiff gate caught on `b772cf4d`.
- **Test migration:** two post-Load sites — `cuda/pager_determinism_test.go` (`NO_RESIDENT_REUSE` per arm on one loaded
  35B) and `gpu/matrix_bench_test.go` (the warm-TTFT column), the latter retagged `gpu && goinfer_testhooks`, which is
  how CI already builds it. Every other set of these seven precedes the Load it configures.
- **Left for later:** `CPU_FUSED_GATEUP`, `W4A8_BATCH`, `W4A8_SPLITHALF` are read once at package init (process-wide,
  so they cannot change a loaded model, but two models cannot differ); `MODELS` is the test-asset search path, not a
  model property; `P13_OFF` and `INT4_F16_SCALES` are load-time diagnostics by their own documentation — phase 6.

### Phase 3 result (2026-09-24)

All eleven CUDA operator knobs (`CUDA_FAST_PREFILL`, `CUDA_FAST_PREFILL_FLOOR`, `CUDA_FLASH_DECODE`,
`CUDA_FLASH_DECODE_MIN_KEYS`, `CUDA_FLASH_DECODE_VERIFY`, `CUDA_NO_FUSE`, `NO_LORA_CACHE`, `PREFILL_CHUNK`,
`PREFILL_IMAGE_CHUNK`, `SPLITKV_ATTN`, `SPLITKV_MIN_KEYS`) leave `testdata/env_reads.txt`.

- **One mechanism, not a second one in `cuda/`.** The names are registered in decoder's snapshot (`cudaKnobs`,
  `decoder/knobs.go`) and the resident reads them through a new exported `Model.Knob(name)`, so `Options.Knobs`, the
  Load-time read and the testhooks drift check all apply unchanged. `Model.Knob` panics on a name not on the list —
  a typo would otherwise read as "unset".
- **Build-time reads** (flash-decode lane, split-KV, `CUDA_NO_FUSE`, the fast-prefill levers) read `m.Knob` in
  `BuildResident`; **per-call reads** (prefill chunk widths, the fast-prefill floor, the LoRA cache switch) go through
  `cudaResident.knob`, the same snapshot, so a test that pins a knob on a live model still works. A resident built by
  hand in a test (no model) reads the live environment — decoder's nil-snapshot rule. The pure parse helpers now take
  the raw value; their unit tests pass it from `os.LookupEnv`.
- **Gate:** `TestFlashDecode_defaultOnRealResident/Options.Knobs_is_per_model` — two CUDA models in one process, one
  with `Options.Knobs` turning the lane off; each builds its own (lane off / default S=16). Test migration: five
  post-build sites (`lora_cache_test`, `lora_cost_measure_test`, `prefill_cancel_test`, `prefill_chunk_fast_test`,
  `prefill_chunked_test`).
- Not a phase-3 finding but met on the way: `TestFlashDecodeKernelLadder` fails with the lane at its default
  (`faSplit=0` on the 1.5B at the ladder's context) on the pre-phase-3 tree too; it passes with
  `GOINFER_CUDA_FLASH_DECODE` set, as its own message asks. And `go vet -tags cuda ./cuda/` (without
  `goinfer_testhooks`) fails on `ptx_modules_cover_test.go` since `a55841f4` — CI always adds the tag.

Phases 2–5 each shrink `testdata/env_reads.txt`; the task is done when the list holds only diagnostics with a named owner
and a reason, and the rule-1 guard stays.

## Inventory (2026-09-24, from the code; tier from the section of `docs/env-vars.md` that documents it)

Every variable read in production is documented. Names below drop the `GOINFER_` prefix.

**operator — `decoder/` (27; 24 read per call)**: `ATTN_GROUPED`, `ATTN_ROW_TILE`, `BATCHED_PREFILL`, `CPU_FAST_ATTENTION`, `CPU_FUSED_GATEUP`, `FUSED_ATTENTION`, `INT4_F16_SCALES`, `MLA_NAIVE`, `MODELS`, `MOE_CACHE_EXPERTS`, `MOE_CACHE_SLOTS`, `MOE_EXPERT_MAJOR`, `NO_FIT_DEFAULT`, `NO_FIT_GUARD`, `NO_GREEDY_FASTPATH`, `NO_KVONLY_PREFILL`, `NO_OPTFWD`, `NO_RESIDENCY`, `NO_RESIDENT_REUSE`, `NO_SAMPLE_FASTPATH`, `NO_TOPK_FASTPATH`, `OPTFWD_MAX_TEMP`, `P13_OFF`, `PREFILL_ATTN_WORKERS`, `SSM_RESIDENT`, `W4A8_BATCH`, `W4A8_SPLITHALF`
**operator — `cuda/` (11; 11 read per call)**: `CUDA_FAST_PREFILL`, `CUDA_FAST_PREFILL_FLOOR`, `CUDA_FLASH_DECODE`, `CUDA_FLASH_DECODE_MIN_KEYS`, `CUDA_FLASH_DECODE_VERIFY`, `CUDA_NO_FUSE`, `NO_LORA_CACHE`, `PREFILL_CHUNK`, `PREFILL_IMAGE_CHUNK`, `SPLITKV_ATTN`, `SPLITKV_MIN_KEYS`
**operator — `metal/` (14; 14 read per call)**: `METAL_ALIAS`, `METAL_ATTN_FA`, `METAL_BATCHED_PREFILL`, `METAL_DECODE_LANE`, `METAL_FAST_PREFILL`, `METAL_FAST_PREFILL_FLOOR`, `METAL_FUSED_ATTENTION`, `METAL_MOE_SLOTS`, `MOE_NOCACHE`, `MOE_PREAD`, `MOE_RESIDENCY`, `MOE_RESIDENCY_SCOPE`, `NO_RESIDENT_MEM_GUARD`, `PRECISE_MATH`
**operator — `internal/` (5; 4 read per call)**: `API_KEY`, `GGUF_DIRECT`, `MODEL_TMP`, `SWAP_GUARD`, `TOOL_UNION`
**operator — `gpu/` (3; 1 read per call)**: `ATTN_KEYS`, `INT4_SLOWPATH`, `WEBGPU_GEMM`
**operator — `cmd/` (1; 1 read per call)**: `NVRTC_DIRS`
**diagnostic — `decoder/` (13; 5 read per call)**: `DECODE_TIMING`, `DELTANET_TIMING`, `FAKEQUANT`, `FAKEQUANT_ACT`, `FAKEQUANT_EXPERTS`, `FAKEQUANT_PERROW`, `MOE_PREAD_CPU`, `NORM_ULP_NOISE`, `ROUTER_CAPTURE`, `SSM_NOMUL`, `SSM_Q8CPU`, `SSM_SKIPFFN`, `TEST_NOTHINK`
**diagnostic — `cuda/` (14; 13 read per call)**: `A10_PROBE`, `CUDA_ATTN_FUSED_TILE`, `CUDA_GRAPHS`, `CUDA_GRAPHS_ONLY`, `CUDA_GRAPHS_SYNC`, `CUDA_GRAPHS_UNSAFE`, `CUDA_L01_CPU_OFFLOAD`, `CUDA_MOE_EXPERT_MAJOR`, `CUDA_VISION_ATTN`, `G4_CAPTURE`, `MOE_CACHE_PROF`, `MOE_DMA_OVERLAP`, `MOE_PIN_REGISTER`, `SPLITKV_VSUM_SPLIT`
**diagnostic — `metal/` (2; 2 read per call)**: `ATTNFA_DEBUG`, `MOE_PROF_SPLIT`
**diagnostic — `gpu/` (3; 3 read per call)**: `GPU_CAPTURE`, `SSM_F16MAMBA`, `SSM_W8A16`
**diagnostic — `cmd/` (9; 9 read per call)**: `GATE_BACKEND`, `GATE_HEARTBEAT`, `GATE_MODELS`, `GATE_SKIP_HEAVY`, `GATE_SKIP_WEBGPU`, `HEAVY_PKGS`, `HEAVY_RUN`, `HEAVY_TIMEOUT`, `REQUIRE_FIXTURES`
**ci-gate — `decoder/` (1; 1 read per call)**: `PREFILL_GATE_PROMPTS`

## Not in scope

- Non-`GOINFER_` variables the platform defines (`CUDA_MPS_PIPE_DIRECTORY`). A scan on 2026-09-24 found no other unprefixed
  reads after `G4DEBUG`'s retirement.
- Test-only env reads (`_test.go`): those are the test's own interface and stay.

<!-- doc-reviewed: 2026-09-24 -->
