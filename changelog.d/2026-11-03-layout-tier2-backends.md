- **BREAKING (import-path change) — GPU backend modules moved under `backend/`.** The three GPU backend modules have moved under
  `backend/` (Tier 2 repository layout reorganization), and the WebGPU package clause is now `package webgpu` instead of `package gpu`.
  Any downstream caller that imports a GPU backend or installs its `cmd/serve` or `cmd/chat` binary must change its `go.mod` and import
  paths from the old module path to the new one:
  - `github.com/townsendmerino/goinfer/cuda` (and subpaths such as `cuda/cmd/serve`, `cuda/cmd/chat`) -> `github.com/townsendmerino/goinfer/backend/cuda` (`backend/cuda/cmd/serve`, `backend/cuda/cmd/chat`)
  - `github.com/townsendmerino/goinfer/metal` (and subpaths such as `metal/cmd/serve`, `metal/cmd/chat`) -> `github.com/townsendmerino/goinfer/backend/metal` (`backend/metal/cmd/serve`, `backend/metal/cmd/chat`)
  - `github.com/townsendmerino/goinfer/gpu` (and subpaths such as `gpu/cmd/serve`, `gpu/cmd/chat`) -> `github.com/townsendmerino/goinfer/backend/webgpu` (`backend/webgpu/cmd/serve`, `backend/webgpu/cmd/chat`; `package gpu` -> `package webgpu`)
  - Until the backend modules are tagged at v0.24.0, `demo/agent` still requires `github.com/townsendmerino/goinfer/gpu v0.23.0` (the last published tag); builds inside `go.work` use `backend/webgpu`, and the two-step release tag bumps `demo/agent` to `backend/webgpu v0.24.0`.
