# Task: the aikit boundary — three safe steps (2026-09)

> **Status 2026-09-24: written up, not started.** Owner asked for the three low-risk steps below to be recorded
> for later. The larger question behind them (where aikit's GPU device layer should live) is **undecided** and
> is not part of this task; see "Not decided" at the end.

## Why

The cost of the aikit/goinfer split is mostly process, not code. Since 2026-08-01, 101 goinfer commits mention an
aikit version bump or a GPU pin update. The incidents that motivated this write-up: fresh clones were broken for
about seven hours during R13 because the tree only built through a local `go.work`; aikit v1.47.0 broke linux/amd64
CI; and goinfer work was blocked because aikit's `Queue` and `Buffer` fields are not exported. None of the three
steps below fixes those alone. Together they remove dead surface and cut the release churn without moving code.

All three are **aikit** changes. Each needs an aikit release that goinfer then pins, except step 3, which is a policy.

## 1. Delete `aikit/gpu/webgpu`

- A separate module (`gpu/webgpu/go.mod`), 730 lines tracked. Verified 2026-09-24: nothing in aikit or goinfer
  imports it; its only importer is its own `backend_test.go`.
- Deleting it takes aikit from 17 modules to 16. goinfer's WebGPU backend lives in goinfer's own `gpu/` module and is
  unaffected.
- Check before deleting: a fresh grep for `aikit/gpu/webgpu` across both repos, and aikit's CI workflow and release
  script for a hard-coded module list that would need the entry removed.

## 2. Make `vision.RegisterResident` refuse a second registration

- `aikit/vision/resident.go`: `func RegisterResident(f …) { residentFactory = f }`. That is one global slot, and the
  last registration silently wins.
- Four registrants exist: goinfer's `cuda/vision_register.go` and `gpu/vision_register.go`, and aikit's
  `gpu/visioncuda` and `gpu/visionmetal`. Two CUDA SigLIP towers (goinfer's and aikit's `visioncuda`) can therefore
  both register, and init order decides which one runs.
- **Latent, not live:** verified 2026-09-24, goinfer imports neither `visioncuda` nor `visionmetal`, so no shipped
  goinfer binary links two registrants. A binary that did would get whichever tower init order picked, with no
  error.
- Fix: a second registration panics at init, naming both registrants (or returns an error, if a caller needs to
  choose). It is a programming error, and failing at init surfaces it the first time the binary runs. Add a test that
  two registrations fail.

## 3. Release policy: pin aikit by commit, tag aikit only when goinfer tags

- goinfer's `go.mod` currently pins aikit by version tag, which is why every aikit change goinfer needs becomes an
  aikit release plus a goinfer bump commit (plus the GPU-pin refresh for `gpu/`, `cuda/`, `metal/`).
- Proposed: between goinfer releases, goinfer pins aikit by **pseudo-version (commit)**, and aikit is tagged only in
  step with a goinfer release. This is the note's recommendation instead of splitting `linalg` piecemeal.
- What to settle before adopting it:
  - `RELEASING.md` (the authority on releases) must say it, including the order: tag aikit, then pin goinfer's five
    modules to that tag, before goinfer's own two-step tag.
  - The citation lint resolves aikit citations through the module cache (`CLAUDE.md` § citations). Check it handles a
    pseudo-version the way it handles a tag.
  - `go.work` must stay optional: a fresh clone must build from the pins alone. That is the R13 failure, and a
    commit pin does not cause it, but the check belongs here.
  - Any other aikit consumer that relies on frequent aikit tags loses them. Name one or confirm there is none.

## Not decided (owner, separately)

Whether aikit's GPU device layer (~3.0k lines: `gpu/cuda*.go`, `gpu/metal*.go`, the upload/copy/residency helpers,
which goinfer's `cuda/device.go` already aliases) moves into goinfer. The obstacle: all eight of aikit's GPU backend
modules (`anncuda`, `annmetal`, `enccuda`, `encmetal`, `qwencuda`, `qwenmetal`, `visioncuda`, `visionmetal`) import
it. Moving it means either **two copies** (goinfer's and aikit's, free to drift, with driver fixes landed twice) or
**aikit leaving GPU** (its GPU backends deleted or moved too, a decision about aikit as a product). The related
recommendation to fold those eight modules into aikit's `gpu` module (17 → ~9 modules) depends on that answer, so it
waits too.

<!-- doc-reviewed: 2026-09-24 -->
