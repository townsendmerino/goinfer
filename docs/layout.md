# Repository layout

A map of the tree. Paths are relative to the repository root; every row was checked against the tree.

## Moved

- `cuda/` -> `backend/cuda/` (`module github.com/townsendmerino/goinfer/cuda` -> `github.com/townsendmerino/goinfer/backend/cuda`)
- `metal/` -> `backend/metal/` (`module github.com/townsendmerino/goinfer/metal` -> `github.com/townsendmerino/goinfer/backend/metal`)
- `gpu/` -> `backend/webgpu/` (`module github.com/townsendmerino/goinfer/gpu` -> `github.com/townsendmerino/goinfer/backend/webgpu`, `package gpu` -> `package webgpu`)
- `internal/{serveapp,chatapp,cliutil,fitcmd,decidecmd,pullcmd,inproc,connect,loadflags}` -> `internal/app/<pkg>` (package clause unchanged)
- `internal/{swapguard,prezerolint,paritykey,movedpaths,servecheck,memgate}` -> `internal/guard/<pkg>` (package clause unchanged)
- `internal/{hwcensus,hwreport,kernelprint,sb2census,towerid}` -> `internal/diag/<pkg>` (package clause unchanged)
- `internal/{giw,mem,kvmgr,f16,mamba2,whisper,scheduler,adapter,hybridattn,program,modelload,prequant,diskfree,sampling}` -> `internal/runtime/<pkg>` (`internal/mem/probe` -> `internal/runtime/mem/probe`; package clause unchanged)

## Modules and the workspace

Seven modules are developed together; an eighth `go.mod`, `.github/actions/staticcheck/`, only pins the CI linter build
(`go run ./cmd/lint` uses it) and is not part of the workspace.

| Path | Module path | Published |
|---|---|---|
| `.` | `github.com/townsendmerino/goinfer` | tagged `vX.Y.Z` |
| `backend/webgpu/` | `.../goinfer/backend/webgpu` | tagged `backend/webgpu/vX.Y.Z` (formerly `gpu/vX.Y.Z`) |
| `backend/cuda/` | `.../goinfer/backend/cuda` | tagged `backend/cuda/vX.Y.Z` (formerly `cuda/vX.Y.Z`) |
| `backend/metal/` | `.../goinfer/backend/metal` | tagged `backend/metal/vX.Y.Z` (formerly `metal/vX.Y.Z`) |
| `demo/agent/` | `.../goinfer/demo/agent` | tagged `demo/agent/vX.Y.Z` |
| `apps/sdktest/` | `.../goinfer/sdktest` | never tagged; tests only |
| `apps/site/` | `.../goinfer/site` | never tagged; the website generator |

**`go.work` is gitignored and required for cross-module work.** The submodule `go.mod` files carry no `replace`, so a
submodule built without a workspace resolves the root from the proxy at its last published tag (`RELEASING.md`). Create it
with `scripts/make_work.sh` (copies `go.work.example`, which lists all seven modules, to `go.work` when absent) and it builds
from a clean clone. Without a `go.work` in the repo, Go walks up and uses one in a parent directory, and a `GOWORK` environment variable overrides both; `make_work.sh` warns about the second and fixes the first. A local aikit checkout is an optional extra `use` line, commented in the example.

Untagged, `backend/cuda/` does not build on its own (`backend/cuda/internal/gpudev` needs the `cuda` build tag); see CLAUDE.md for the tags.

## Library packages (root module)

| Path | What it is |
|---|---|
| `decoder/` | the decoder-only transformer runtime: loaders, forward passes, KV cache, fit guards, the `.giw` sidecar |
| `decoder/fit/` | the types the fit guard and the placement planner hand their callers (`Plan`, `PlanRequest`, `Placement`, `FitAnswer`, `FitDeclineError`); a leaf over `internal/runtime/mem` |
| `engine/` | what the serving shell asks of a model: a generation from a prepared request |
| `server/` | the HTTP API (OpenAI and Anthropic routes, web UI) as a library handler |
| `chat/` | chat templates and tool-call rendering and parsing, no Jinja engine |
| `constrain/` | constrained decoding: JSON Schema and grammar logit masks |
| `tokenizer/` | the BPE tokenizers the models ship |
| `multimodal/` | vision and audio towers and preprocessing |
| `embeddinggemma2/` | the EmbeddingGemma 2 text encoder |
| `pull/` | resumable checkpoint download from HuggingFace |
| `internal/` | packages private to the root module, grouped under `internal/app/` (CLI/server apps and wiring), `internal/guard/` (lint/CI/memory/parity guards), `internal/diag/` (hardware census and reports), and `internal/runtime/` (execution, memory, KV, scheduler, model loading, sampling, and runtime kernels), plus top-level domain packages (`audio`, `batchio`, `clef`, `codec`, `confidence`, `config`, `cost`, `decide`, `fidelity`, `layout`, `vision`, `wrappedupload`); `internal/layout` holds repository-shape checks as tests |

## Tier 3: subpackages of `decoder/`

`decoder/` is one package of ~800 files. Tier 3 splits it one subsystem at a time, `decoder/fit` first (Francis, 2026-11-05;
`docs/releases/v0.25.0-plan.md` §2, ledger row T3-fit in `docs/tasks/task-subsystems-2026-10.md`). A subpackage lives
**under** `decoder/`, named for the subsystem.

**The rule for what may import what.** A subpackage never imports `decoder` (that is a cycle: the loader calls it).
`decoder` imports the subpackage and re-exports its public types as aliases (`type Plan = fit.Plan`), so a move changes no
caller until the caller chooses to name the subpackage, and a backend that does so stops depending on the loader (the
`decoderRatchet` in `internal/layout`). Each subpackage has a rule in `internal/layout/imports.json` naming the goinfer
packages it may import; `TestLayout_arrows` enforces it. Aliases are deleted when the last caller has moved.

**Why `fit` cannot move whole.** The three files (`fitguard.go`, `fitplan.go`, `fitquery.go`) are the loader's front
end, not a leaf. Moving them and reading the compiler's errors names 43 identifiers they need from `decoder`:
`Config`, `Options`, `Model` and `quantMode` (43 uses); `resolveArchitecture`, `ResolveKV`, `kvBytesForCtx`, `buildDemandFor`,
`Admit`, `loadKnob`; the loader's own quantizer (`quantizeWM`, `wmBytes`, `parseQuant`, `mlaProjInt8`, `mambaProjInt8`);
and the checkpoint openers (`openCheckpointMmap`, `OpenGGUFMmap`, `ggufConfig`, `loadConfig`, `mapGIW`, `readGIWHeadReader`).
In the other direction 13 non-test `decoder` files call it (`model.go`, `prefill_budget.go`, `sidecar.go`, `weights.go`,
`metal_fitctx.go`, `residency.go`, ...), and 40 files outside `decoder/` (17 non-test) name its exported API. The decisions are already in `internal/runtime/mem`, which imports no goinfer package.

**Order.**
1. **Types** (done): `Placement`, `PlanRequest`, `Plan`, `FitAnswer`, `FitDeclineError`, `ErrWontFitResident` in `decoder/fit`, aliased
   in `decoder/fitalias.go`. No behavior change; the files outside `decoder/` that name them compile unchanged.
2. **Callers' imports** (done): backends, `internal/app/*`, `modelload` and `memgate` name `fit.X` instead of `decoder.X`.
3. **Price from neutral inputs.** Done for the checkpoint-byte estimators and the flat KV rate: `fit.GGUFWeightBytes`,
   `fit.SafetensorsWeightBytes` take tensor names and shapes plus `fit.Rates` (the per-element prices, which `decoder` measures
   through the loader's quantizer in `fitRates`), and `fit.KVBytesPerPosition` takes the geometry; the tensor-name classifiers
   moved with them. Still in `decoder`, each because it holds a `Config`, `Options`, `Model` or the quantizer: `fitCheck` and
   `guardFit`, `guardGIWFit`/`decideGIWFit`, `fitCheckFor`, `quantBytesPerElem`, the per-layer `estimateKVBytes`, `Model.Plan`
   and `FitQuery`. Next: give `fitCheck` and the GIW decision a neutral input struct (the machine reading, the weight and KV
   figures, the pins) so they move; the `fit*_test.go` files follow their subjects.
4. `Model.Plan` and `FitQuery` stay in `decoder` as thin adapters (they hold a `*Model`/`Options`); delete the aliases once no caller names them.

## GPU backend modules

| Path | What it is |
|---|---|
| `backend/webgpu/` | the WebGPU backend (opt-in cgo build, tag `gpu`), plus `backend/webgpu/cmd/{chat,serve}` |
| `backend/cuda/` | the CUDA backend (cgo-free, tag `cuda`), `.cu` kernels and `backend/cuda/cmd/{chat,serve}` |
| `backend/metal/` | the Metal backend (tag `metal`), `backend/metal/cmd/{chat,serve}` |

## Binaries

| Path | What it is | Module |
|---|---|---|
| `cmd/serve/` | `goinfer-serve`, the server | root |
| `cmd/gate/` | the gate runner (`go run ./cmd/gate`) | root |
| `cmd/lint/` | CI's staticcheck, pinned and verified | root |
| `cmd/hostmem/`, `cmd/prequant/` | prints the memory figure the fit guard uses; builds a `.giw` bundle from a GGUF | root |
| `cmd/claude-guard/` | the Claude Code PreToolUse hook that refuses commands the repo rules forbid | root |
| `demo/chat/` | `goinfer-chat`, the single-shot chat binary (`build-embed.sh` bakes a model in) | root |
| `backend/{webgpu,cuda,metal}/cmd/{chat,serve}/` | the same two programs built with a backend | the backend module |
| `demo/agent/cmd/` | `agent-web`, `stdlib-agent`, `ken-stub` | `demo/agent` |
| `apps/site/cmd/build/` | the website generator (`site.yml` builds and deploys it) | `site` |
| `apps/examples/{confidence,embed,invoice,structured}/` | runnable library examples, each a `main` package with a test | root |

**Where each binary lives, and why** (decided in PR 3 of the Tier 2 layout move; a move would change the build and install
paths it names):

- The backend `cmd/` directories (`backend/{webgpu,cuda,metal}/cmd/{chat,serve}`) stay inside their backend module. They
  import the backend's packages, and the root cannot require its own submodules without a cycle, so they cannot move to the
  root `cmd/`. The root `cmd/serve` and `demo/chat` are the no-backend builds.
- `demo/agent/cmd/` stays inside `demo/agent`. That module is published (tags `demo/agent/vX.Y.Z`) and keeps its MCP SDK
  out of the root `go.mod`; moving its binaries to the root would pull that dependency in.
- `apps/site/cmd/build/` stays inside `apps/site`. The site is never tagged, and its generator imports its own packages.
- `demo/chat/` stays. It is a root-module `main` package, and `go install github.com/townsendmerino/goinfer/demo/chat@latest`
  is a documented install path (README, the release template, and the readme-smoke check). Moving it changes that path for
  every installer; if it moves later, the install line changes with it.
- `apps/examples/*` stays. Each example is linked from the docs and tested in place.

## Satellites

Not part of the library; nothing in the root imports them. `sdktest/`, `site/` and `examples/` live under `apps/`. `demo/` stays at the root on purpose: `demo/agent` is published (tags `demo/agent/vX.Y.Z`), and a subdirectory module's tag prefix must equal its directory, so moving it would break its tags; `demo/chat` stays beside it.

| Path | What it is | Module |
|---|---|---|
| `demo/agent/` | MCP-SDK agent demo (kept separate so the SDK stays out of the root `go.mod`) | `demo/agent` |
| `demo/demo.tape`, `demo/mellum2-gpu.tape` | VHS recordings for the README demo | none |
| `apps/sdktest/` | the official OpenAI and Anthropic SDKs against the in-process server | `sdktest` |
| `apps/site/` | goinfer.dev source and generator | `site` |
| `apps/examples/{confidence,embed,invoice,structured}/` | runnable library examples, each with a test | root |

## Test assets

| Path | What it is |
|---|---|
| `testdata/` | tiny deterministic checkpoints (committed, see `.gitignore` exceptions), goldens (`*_golden.json[.gz]`), `assets.json` (heavy-test assets), `env_reads.txt`, `gate/`, `decisions/` |
| `decoder/selftestdata/` | embedded copies of three tiny checkpoints for the startup GPU self-test |
| `<pkg>/testdata/` | per-package fixtures (for example `chat/testdata/`) |
| `docs/measurements/`, `docs/completed/` | dated measurement records and archived task docs |

## Documentation

`docs/` holds design docs, `docs/book/` the inference primer, `docs/code-notes/` per-package notes, `docs/measurements/`
records, and `docs/citation-index.md` plus `docs/QUEUE.md`. `changelog.d/` holds unreleased changelog fragments (folded into
`CHANGELOG.md` at release).

## scripts/

About 316 files, referenced by path from CI, skills, the night runner and docs: do not rename or move them. By prefix:

| Prefix or name | What it is |
|---|---|
| `pin_*` (about 160) | regenerate a tiny fixture or a golden from the HuggingFace reference; the file names the fixture |
| `bench_*` | benchmark drivers; `bench_peer.py` is the same-session peer comparison, `bench_compare.sh` goinfer-versus-goinfer |
| `test_*` | tests of the scripts themselves (Python) |
| `touched.py`, `touched_scans.py` | the day loop: build, vet and test what the diff touches |
| `timing_lock.py`, `night.py`, `install-night-agent.sh` | the timing lock and the night queue |
| `queue_citation_lint.py`, `book_link_lint.py`, `comment_census.py`, `readme_counts_check.py`, `doc_id_collisions.py`, `ci_checks.py` | lints (some run from the pre-push hook, `pre-push.sh`) |
| `task_docs.py`, `doc_review_staleness.py` | task-doc ledgers and archiving |
| `webui*` | browser gates for the web UI |
| `s14*`, `s896*`, `g31*`, `q3vlmoe*`, `gemma4*` and other task-named files | one-off graders and probes from a named task |
| `make_work.sh` | creates `go.work` from `go.work.example` |

## Release files

| Path | What it is |
|---|---|
| `RELEASING.md` | the release procedure and the module-tag rules |
| `CHANGELOG.md`, `changelog.d/` | the changelog and its pending fragments |
| `.github/workflows/` | CI (`ci.yml`), site, coverage, vulnerability, release-asset and weekly workflows |
| `LICENSE`, `NOTICE`, `THIRD_PARTY_LICENSES.md` | licensing |
