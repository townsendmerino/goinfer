# goinfer ⇄ aikit split — execution plan

> **Audience:** Claude Code (VSCode), driven from the **goinfer** window.
> **Canonical copy.** (A context copy lives at `aikit/docs/aikit-module-split-plan.md`
> alongside the original 1.0 critique; this one is the execution source of truth.)
>
> **Goal:** finish carving the codebase into a stable retrieval toolkit (`aikit`)
> and a separately-paced LLM runtime (`goinfer`), so a consumer can `go get`
> either without dragging in the other's heavy deps (cgo `webgpu`, pre-1.0
> `gotreesitter`).
>
> **We are free to break things** — the user is the only consumer (plus `ken`,
> which they own and which is unaffected; see §3). No import-path-preservation
> constraint, no green-at-every-commit staging, no release choreography.

---

## 0. Workspace setup (do this first)

The reorg edits **both** repos. From the goinfer VSCode window, add aikit to the
workspace so Claude Code can edit both trees:

```
~/tmcode/goinfer   (this repo)
~/tmcode/aikit     (add via File ▸ Add Folder to Workspace)
```

Create a top-level dev workspace so goinfer resolves aikit locally without a
`replace` directive (this `go.work` is a **local dev artifact — never committed**;
`.gitignore` already excludes it):

```bash
cd ~/tmcode/goinfer
go work init . ../aikit
```

---

## 1. Current state (already done)

- **Repos exist.** `github.com/townsendmerino/aikit` (populated) and
  `github.com/townsendmerino/goinfer` (scaffold pushed).
- **goinfer scaffold committed:** `README.md`, `LICENSE` (MIT), `NOTICE`,
  `CHANGELOG.md`, `go.mod` (bare: `module github.com/townsendmerino/goinfer`,
  `go 1.26.3`, no `require` lines yet), `.gitignore`.
- **Name locked:** `goinfer` (deliberately not `gollm` — that namespace is
  crowded and means the opposite thing: teilomillet/gollm orchestrates *remote*
  providers; goinfer runs weights *locally, in-process*).

Everything below is the remaining work.

---

## 2. Target end state: 2 repos, 4 modules

### Repo A — `aikit`: stable retrieval toolkit (stdlib + `x/text` only)

| Module (import path) | Dir | Packages | Deps beyond stdlib |
|---|---|---|---|
| `…/aikit` | repo root | `topk ann bm25 fuse embed encoder chunk (regex/markdown/line) linalg` | `x/text` |
| `…/aikit/chunk/treesitter` | `chunk/treesitter/` | treesitter chunker | `gotreesitter`, `…/aikit` |

`linalg` is promoted from `internal/linalg` to public (shared across the repo
boundary, independently useful). `encoder` gains a pluggable `Backend` (§4) so it
carries **no** `webgpu` dependency.

### Repo B — `goinfer`: the LLM runtime (depends inward on aikit)

| Module (import path) | Dir | Packages | Deps beyond stdlib |
|---|---|---|---|
| `…/goinfer` | repo root | `decoder tokenizer constrain demo/` | `…/aikit` (embed, linalg), `x/text` |
| `…/goinfer/gpu` | `gpu/` | WebGPU backends (`//go:build gpu`) | `cogentcore/webgpu`, `…/aikit`, `…/goinfer` |

`webgpu` (cgo) is confined to the opt-in `goinfer/gpu` submodule and serves both
`encoder` (aikit) and `decoder` (goinfer) via the `Backend` registry, so neither
core module ever imports it.

---

## 3. Consumer impact (`ken`) — verified

`ken` imports only `chunk`, `chunk/{regex,markdown,treesitter}`, `embed`,
`encoder`, `bm25`, `ann`, `fuse`. It imports **none** of `decoder`/`tokenizer`/
`constrain`. Therefore:

- Every aikit path ken uses **stays** `github.com/townsendmerino/aikit/...`.
- ken's only edits: add `require …/aikit/chunk/treesitter vX.Y.Z` (now its own
  module) and bump the main `aikit` require. **Zero code changes.**
- Re-confirm after the split: `grep -r townsendmerino/aikit --include=*.go ~/tmcode/ken`.

---

## 4. The one non-mechanical refactor: pluggable `Backend`

Today `encoder/gpu` and `decoder/backend_gpu.go` import `webgpu` directly under
the `gpu` tag, which keeps `webgpu` in those modules' graphs (`go mod tidy`
considers all build tags). Invert the dependency:

1. In `encoder`, define a `Backend` interface (the compute/matmul seam already
   implied by `encoder/gpu`) + a `RegisterBackend` hook; default-register the
   pure-Go CPU backend. Same for `decoder`.
2. Move the WebGPU implementations (`encoder/gpu/*`, `decoder/backend_gpu.go`)
   into the new `goinfer/gpu` module, all under `//go:build gpu`
   (`backend_nogpu.go` collapses into the default CPU registration).
3. `goinfer/gpu` imports `encoder` + `decoder` and registers the WebGPU backends
   on init. A GPU user does `import _ ".../goinfer/gpu"` and builds `-tags gpu`.

Dependency now points `gpu → {encoder, decoder}`, so `webgpu` never enters the
core graphs. Keep the existing gpu tests (moved to `goinfer/gpu`, still tagged).
This is the only piece touching real code rather than just `go.mod` files.

---

## 5. Execution order

**In `aikit`:**
1. `git mv internal/linalg linalg`; rewrite `…/internal/linalg` → `…/linalg` in
   `encoder` (and in `decoder` before it moves). Delete empty `internal/`.
2. `Backend` inversion in `encoder` (§4): interface + default CPU registration.
   Remove `encoder/gpu/` (it moves to goinfer).
3. `chunk/treesitter`: `cd chunk/treesitter && go mod init …/aikit/chunk/treesitter && go mod tidy`.
   Confirm the main module's go.mod no longer lists `gotreesitter`.
4. `go mod tidy` the aikit main module. **Assert clean:**
   `go list -deps ./... | grep -Ei 'webgpu|gotreesitter'` → empty. Commit a
   `go.work` in aikit listing `./` and `./chunk/treesitter` (intra-repo dev only).

**In `goinfer`:**
5. Move `decoder/`, `tokenizer/`, `constrain/`, `demo/` from aikit into goinfer
   (plain move — solo repo, history preservation optional; use `git filter-repo`
   if you want it). Rewrite their internal imports:
   `…/aikit/decoder` → `…/goinfer/decoder`, `…/aikit/tokenizer` →
   `…/goinfer/tokenizer`, etc. Keep `…/aikit/embed` and `…/aikit/linalg` imports.
6. `go mod tidy` goinfer's main module (the dev `go.work` from §0 resolves aikit
   locally). Confirm goinfer's main graph has **no** `webgpu`.
7. Create the `goinfer/gpu` submodule (§4): `//go:build gpu`; `go mod init`;
   `go mod tidy`; requires aikit + goinfer + webgpu.

**Wire `ken`:**
8. `go get …/aikit@<newtag>`; add `require …/aikit/chunk/treesitter@<tag>`.
   `go build ./... && go test ./...` in ken — expect green, no code edits.

---

## 6. Versioning

- `aikit` main module: ordinary `vX.Y.Z` tags.
- `aikit/chunk/treesitter`: submodule tags `chunk/treesitter/vX.Y.Z`.
- `goinfer`: its own `vX.Y.Z` tags in this repo.
- `goinfer/gpu`: submodule tags `gpu/vX.Y.Z`.

Cross-repo `goinfer → aikit` is a normal versioned dependency; bump it when aikit
cuts a release. No intra-repo release choreography for the aikit core (one primary
module + one rarely-changing submodule).

---

## 7. CI

- **aikit:** one workflow covering the root module + `chunk/treesitter`. Add the
  **core-cleanliness guard**: in the root module,
  `go list -deps ./... | grep -Ei 'webgpu|gotreesitter'` must be empty (fail
  otherwise). Keep race detector + model-asset-skip behavior.
- **goinfer:** new workflow; matrix the root module + `gpu` submodule (build the
  gpu module with `-tags gpu`). Same model-asset skip pattern. Add the same
  cleanliness guard to goinfer's root module (no `webgpu`).

---

## 8. Docs

- **aikit README:** retitle "pure-Go retrieval toolkit" (drop the "parts of ken"
  framing); new dependency table per §2; point at goinfer for generation.
- **goinfer README:** already in place. Update the package table once the move
  lands if paths/deps shift.
- **Per-package `doc.go`** for the aikit hard tier (`topk ann bm25 fuse embed
  encoder chunk linalg`), each with a copy-pasteable snippet.
- **One end-to-end RAG example** in aikit (`examples/rag/`):
  `chunk → embed → ann + bm25 → fuse → encoder-rerank → topk`. Highest-value doc.
  A second example in goinfer (`demo/gemma`, already moving here) extends it with
  `decoder` for generation.
- **goinfer CHANGELOG:** replace the scaffold `[Unreleased]` note with the real
  first feature entry once the packages land.

---

## 9. Definition of done

- [ ] `internal/linalg` → public `linalg`; `internal/` gone.
- [ ] aikit root module: `go list -deps ./...` has **no** `webgpu` / `gotreesitter`.
- [ ] `…/aikit/chunk/treesitter` is the only aikit module pulling `gotreesitter`.
- [ ] `encoder` has a pluggable `Backend`; default build pure-Go CPU; no gpu code
      left in aikit.
- [ ] `decoder tokenizer constrain demo` live in goinfer; goinfer requires aikit;
      goinfer root module has no `webgpu`.
- [ ] `goinfer/gpu` is the only place `webgpu` appears; builds under `-tags gpu`.
- [ ] ken builds + tests green with only go.mod edits (no code changes).
- [ ] Both repos: `gofmt`/`go vet` clean, `go test ./...` green (model-asset tests
      skip as before); CI cleanliness guard in place in both.
- [ ] aikit `examples/rag/` runs end-to-end; per-package `doc.go` added.

---

## 10. Gotchas

- **Stale `.git/index.lock`:** the Cowork sandbox can't delete it; run git
  commits from your own terminal (this is why the scaffold was committed by hand).
- **Cross-repo `go.work`:** keep it for local dev (`go work init . ../aikit`) so
  goinfer resolves aikit locally without `replace`; **never commit it** to either
  published repo (`.gitignore` already excludes `go.work`).
- **`go mod tidy` + build tags:** after the `Backend` inversion, re-run tidy and
  re-check `go list -deps` for all four modules — tidy is the source of truth for
  whether the inversion actually removed `webgpu`.
- **tokenizer moves with the runtime:** `decoder` imports `tokenizer`; both go to
  goinfer. `embed` has its *own* (WordPiece) tokenizer that **stays in aikit** —
  don't conflate them.
- **Cross-module `internal/`:** don't keep `linalg` internal and share it across
  the repo boundary — promote it (step 1).
- **Model-asset tests skip on fresh checkout:** expected; populate fixtures per
  each repo's README / `testdata` notes.
