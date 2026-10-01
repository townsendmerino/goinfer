# Task: hardware we don't own — find the paths it runs, reach them, guard them in the field (H0–H6) — 2026-10

> **Status: SCOPED 2026-10-01, unstarted.** H0 is an inventory. H1 and H2 are the work that matters most,
> and both can start the same day. H5 is an owner decision that waits on two release sweeps.
>
> **The concern (Francis, 2026-10-01).** goinfer is built and measured on an M1 Pro (16 GB) and an
> RTX 2070 SUPER (8 GB) with a Ryzen 7 3700X. People with newer or bigger hardware will run code paths
> those machines never execute.
>
> **The evidence that it is real.** aikit v1.47.1 (goinfer `0616cdc0`, 2026-09-24) fixed an AVX-512 VNNI
> W4A8 kernel whose two f32 accumulators did not cancel: up to 3.2e-3 per logit away from the AVX2 path.
> None of our machines has AVX-512. It surfaced only because GitHub's ubuntu-24.04 pool mixes CPU models,
> and it was confirmed under Intel SDE on nobara. Both routes worked once by luck. This doc makes them
> deliberate.
>
> **Folds in** `task-metal-runtime-selftest.md` (specced 2026-07, never built). Its spec is carried into H2
> intact and extended to every backend. The original is archived in `docs/completed/` with a pointer here.
>
> **Siblings.** [`../hardware-matrix.md`](../hardware-matrix.md) (H6 adds a verified-on column) ·
> [`../gpu-vendor-coverage.md`](../gpu-vendor-coverage.md) (WebGPU's cross-vendor claim) ·
> [`task-first-hour.md`](task-first-hour.md) (the cold-user protocol H4 reuses) ·
> [`task-peer-claim-2026-09.md`](task-peer-claim-2026-09.md) (the public claim H4 must precede) ·
> [`task-fit-to-hardware.md`](task-fit-to-hardware.md) (owns the size-selected decisions H1 drives).

---

## 1. Where the risk lives (inventory at goinfer `90b7190b`, aikit `4a5a7b2`)

Three kinds of path. H0 turns this list into a checked census.

**A. Selected by a hardware feature our machines lack.**

| path | selected when | our machines | where |
|---|---|---|---|
| AVX-512 VNNI / VNNI+VL kernels | Zen 4/5, Intel Ice Lake and later | **never** (3700X is Zen 2) | aikit `linalg/`: `dot_w4a8_avx512vnni_amd64.go`, `dot_i8_avx512vnni_amd64.go`, `quant_i8_amd64.go`, `quant_w4a8_tile_amd64.go`, `weightmat_splithalf_amd64.go`, `matmul_w4a8_splithalf_tile_amd64.go` |
| amd64 without AVX2 (pure-Go fallback) | pre-2013 x86, some VMs | never on hardware | aikit `*_amd64.go` `hasAVX2` branches |
| arm64 without DotProd | Raspberry Pi 4 class (Cortex-A72), older ARM | never (M1 has DotProd) | aikit `*_arm64.go` `hasDotProd` branches |
| CUDA on any non-Turing card | every NVIDIA card newer than the 2070 | never | all PTX is `.target sm_75`, JIT-compiled forward by the driver |
| CUDA launch sizing | cards with more SMs or shared memory | only 40 SMs here | `cuda/fused_qkv_rows.go` reads SM count, max threads/SM, max shared memory/SM |
| CUDA PTX version 8.8 | drivers too old for ISA 8.8 | never (driver 595 here) | the opposite risk: an older driver may fail to load the kernels |
| Metal on M2/M3/M4 GPUs | every newer Mac | never (M1 Pro) | MSL compiled at load by the OS; no family query in `metal/`, so same kernels, different codegen |
| WebGPU on AMD, Intel, Windows DX12 | non-NVIDIA Linux GPUs, all Windows | never | `gpu/` through wgpu-native; measured only on nobara's NVIDIA |
| Windows binaries | every Windows user | **never run** | release assets ship windows-amd64 and windows-arm64; CI has no Windows job |
| linux/arm64 binaries | Graviton, Ampere, Pi | not in CI | release asset ships; check what has run (README's Pi section) |

Not on the list, checked: aikit has no i8mm or SVE paths, so an M2+ CPU runs the same arm64 kernels as the M1.

**B. Selected by size.** More memory makes these choose configurations never exercised here:
- the fit guard and `Plan` (`decoder/fitplan.go`);
- `resolveWeightCacheBudget` (`decoder/fitguard.go`);
- `metalMemoryCeiling` (`metal/backend.go`);
- the host-RAM probes (`decoder/hostram_{darwin,linux,other}.go`);
- MC1's slot count and unpinned-context trimming (`947e06ce`);
- the pager's budget.

Examples never run: the 35B MoE fully resident on Metal (no pager), a 7B at 32k context with four slots on
a 24 GB card, any model that fits resident only above 16 GB.

**C. Selected by driver or OS version.**
- Metal shaders compile on each macOS version. `positioning.md` already says bit-identity holds within a
  machine and OS, not across them.
- PTX is JIT-compiled to the card's own SASS by the driver, so contraction and ordering can differ from the
  2070's.
- wgpu-native picks Vulkan, DX12 or Metal per platform.

## 2. Items

### H0 — the hardware census (a checked file, not a list in prose)

- `docs/hardware-coverage.json`: one entry per gate in §1. Each entry records:
  - the predicate (e.g. `hasAVX512VNNIVL`, `SM count`, `unified memory ≥ 32 GiB`) and the files it selects;
  - **last executed**: machine, date, how (`native` / `emulated` / `ci-pool` / `forced`), and the goinfer
    and aikit commits;
  - the gate that ran (which tests, which goldens).
- A census test, in the shape of the existing dispatch census tests. It greps for feature predicates
  (`hasAVX*`, `hasDotProd`, `hasF16C`, CUDA `DeviceAttribute*`, the memory probes) and fails when a new one
  appears without an entry. A new hardware-gated branch cannot land unseen.
- Generate a "never executed" list from the JSON for H6 and for the release checklist.

### H1 — reach the paths without owning the hardware (cheap, mostly CI)

1. **Free CI runners we don't use yet.** A `windows-latest` job (build plus CPU tests). A Linux arm64 job on
   GitHub's arm64 runners, which are free for public repositories. Both run the CPU suite and the forward
   goldens. The Windows binaries go from never run to run on every push.
2. **Intel SDE as a scheduled job** (weekly, like `race-weekly`, or on nobara). Run the CPU parity suite and
   tiny goldens under `sde64 -icx` (AVX-512 VNNI) and `-spr` (Sapphire Rapids). SDE is slow, so goldens only.
   Precedent: `0616cdc0`'s confirmation run.
3. **Force the fallbacks.** A test-only hook (build tag, in aikit) that makes `hasAVX2` / `hasDotProd` /
   `hasAVX512VNNI*` report false. CI then runs every kernel suite twice, so the pure-Go and narrower-ISA
   paths execute on every push.
4. **QEMU user mode for no-DotProd arm64** (`-cpu cortex-a72`). Run the CPU tiny goldens for the Pi 4 class.
5. **Drive the size-selected paths with small models.** A test hook that overrides the memory probes and the
   CUDA SM / shared-memory attributes: a 0.5B told it has 64 GiB unified memory, or a card with 128 SMs.
   That reaches the residency decisions, slot counts, context defaults and launch grids on our own hardware.
   A grid sized for 128 SMs runs correctly on 40, so the launch math gets real execution, not just
   compilation. Each forced configuration runs the existing correctness gates; nothing new is asserted.

Every H1 job records its result into H0's census as `emulated`, `ci-pool` or `forced`.

### H2 — the runtime self-test, every backend (folds in `task-metal-runtime-selftest.md`)

**The design, carried over intact from the Metal spec.** A stored byte-golden cannot ship to the field: it
is machine-pinned by construction. So the probe is **relative, at init**: the backend's kernels against the
pure-Go CPU reference already in the binary, on a tiny committed fixed vector (seeded `pin_*.py`), judged by
that backend's parity contract. Specifically:
- **Kernels:** run the numerically riskiest ones, the resident kernels themselves, not reimplementations.
  Softmax past both reduction widths (≥ 256 keys), the FP16-scale MMA seam, the norm-to-int8-quant seam, the
  GELU-tanh clamp.
- **Budget:** under 5 ms per backend, once per process at `BuildResident`, never per token. If it cannot be
  met, shrink the vector, not the tolerance.
- **On mismatch: decline, never crash.**
  - Fall back through the existing decline path.
  - Name the kernel and the observed versus allowed numbers in the message.
  - `serve` logs at WARN, continues, and reports it on `/health`; `chat` prints one stderr line.
- **Break-it-first:** a test perturbs a shader or kernel, asserts the probe declines, restores it, and
  asserts it passes (`parity-coverage-policy.md`, "prove the gate red before trusting it green").
- **The Metal spec's four open questions stand:**
  - whole-backend versus per-family decline (default: whole backend);
  - a seeded vector;
  - sharing the vector with `TestMetalSnapshotGolden`;
  - the CPU reference's own init cost.

**Extended here:**
- **Metal first, then CUDA** (JIT to a new arch, SM-sized grids), **then WebGPU** (vendor drivers).
- **CPU ISA paths too, with their own bound.** At startup, the ISA kernel the dispatcher selected runs against
  the pure-Go reference on the fixed vector. **The tolerance is the CPU contract, not the GPU one.** The GPU
  probe's cosine ≥ 0.999 would have passed the 2026-09-24 AVX-512 bug (3.2e-3 per logit). The CPU bound must
  be the one aikit documents for cross-ISA agreement (v1.47.1 brought that path to 1.2e-7). The break-it-first
  test for this probe is that exact bug, reintroduced: the probe must decline v1.47.0's arithmetic.
- **Optional cache:** record a pass per (binary version, device, OS build) in the user cache directory, so
  later starts can skip the probe. Any change in that tuple re-runs it.

### H3 — a hardware report people can paste

- `goinfer-serve check --hardware` (and the same on `goinfer-chat`) prints one block:
  - OS and build;
  - CPU model, detected features, and **which kernel variant the dispatcher chose**;
  - GPU name, architecture, SM count, driver, and whether PTX was JIT-compiled;
  - Mac GPU and macOS build;
  - RAM and VRAM;
  - the fit decisions for a named model (residency, slots, context);
  - every backend's H2 self-test result.
- `.github/ISSUE_TEMPLATE/bug.yml` asks for that block. None exists today.
- **No telemetry.** Nothing is sent anywhere; the user pastes it or doesn't. State this on the site.
- Reports Francis accepts go into H0's census as `native` on that hardware, and feed a "tested on" table on
  the site.

### H4 — a rented sweep before each release that carries a public claim

- A script that a fresh rented machine runs from a clean OS (`scripts/hardware_sweep.sh`). It downloads the
  release assets, pulls a fixed small model set, runs the self-tests, `check --hardware`, the forward goldens
  and the cold-user protocol, and writes `docs/measurements/hardware-sweep-<tag>.md`.
- **Matrix:** one NVIDIA card per generation we can rent (Ampere, Ada, Blackwell consumer), one current Apple
  Silicon Mac, one Zen 4/5 or Ice Lake+ x86 box (native AVX-512), and one Windows machine.
- **Judged against the CPU reference with each backend's parity tolerance, not byte-goldens:** JIT-compiled
  SASS on another arch is not expected to match the 2070 bit for bit.
- **Correctness only.** Speeds measured on rented boxes go into a claim only under the peer-claim protocol
  (same session, interleaved, pre-registered), never as a by-product of this sweep.
- **Budget:** Francis sets a per-release cap. Consumer GPUs rent for well under a dollar an hour. Cloud Macs
  are the awkward line (AWS bills them in 24-hour minimums; check hourly providers first).
- **First run:** before the public claim that `task-peer-claim-2026-09.md`'s re-run on v0.20.0 may earn.

### H5 — buy one machine? (owner decision, after two H4 sweeps)

The one purchase worth weighing is a current Apple Silicon Mac mini with 32 GB or more. It covers three
gaps at once:
- a newer Metal GPU family and macOS compiler on a second machine;
- enough memory to run the 35B MoE fully resident;
- a second quiet box for the reproduce-across-days rule.

The Mac is also where the promotion claim leans, and where renting is clumsiest. **Trigger:** two H4 sweeps
have run, and either a Mac-only finding appeared or cloud-Mac access cost more in time than it saved. NVIDIA
stays rented.

### H6 — say what is verified, plainly

- `hardware-matrix.md` gains a **verified on** column generated from H0, distinct from capability.
- The site's Download page and the README name the machines goinfer is built and verified on, say that the
  self-test guards everything else, and invite reports. Plain, humble copy; no claims beyond the census.
- The release checklist (`RELEASING.md`) gains one line: the never-executed list from H0, read before
  tagging.

## 3. Order

H0 → (H1.1–H1.3 and H2-Metal, in parallel) → H1.4–H1.5 and H2-CUDA → H3 → H4 before the next public claim →
H2-CPU-ISA and H2-WebGPU → H6. H5 waits on its trigger.

**Enough to make a public speed claim:** H0 complete; H2 on Metal and CUDA; no never-executed entry among the
paths the claim names; one H4 sweep on the claim's hardware classes.

## 4. Not in scope, stated

- **Telemetry of any kind.**
- **Certifying every GPU.** The self-test is the field guard; the census and sweep say where we have looked.
- **New native backends** (AMD HIP, Intel); `gpu-vendor-coverage.md` owns that question.
- **Performance tuning for rented hardware.** Correctness first. A kernel tuned on a card we do not own
  becomes a path nobody can re-measure.

<!-- doc-reviewed: 2026-10-01 -->
