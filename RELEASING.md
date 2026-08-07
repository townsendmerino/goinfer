# Releasing goinfer (tri-module: root + gpu + cuda + metal)

goinfer is **four Go modules** in one repo — the root (`github.com/townsendmerino/goinfer`),
plus `gpu/`, `cuda/`, `metal/`, and the ancillary `demo/agent/`. They form a **requirement
cycle** (root → gpu/cuda/metal → root via `replace => ../`), so there is no single-shot tag
that produces a consistent, go-gettable set. This file is the ordering that does, plus the
module-graph fixes that must land with it. It exists because four release blockers in the
2026-08-05 audit (§1a/§1b: B-01…B-07) were "the tagged submodules don't build for anyone" —
the gitignored `go.work` masks it locally, so it survives to a tag unless caught here.

## Invariants (why the steps are shaped this way)

- **`replace => ../` is honored only when a submodule is the *main* module** (its own
  build/tidy). A *consumer* of `goinfer/metal@vX` gets the `require` line, and Go **ignores
  the replace** — so a `require goinfer v0.0.0-00010101…` placeholder (B-01) or an Aug-2
  pseudo-version (B-05) publishes a broken/ stale dependency permanently. Every submodule must
  `require` a **real, published** root tag before *it* is tagged.
- **`go.work` is gitignored** (`.gitignore`) and must stay dev-only. Nothing in the shipped
  tag may depend on it. CI's standalone-build step (below) is what proves this.
- **Version alignment (B-07):** all four modules must agree on `aikit` and `aikit/gpu`.
  As of the audit: root/cuda want `aikit v1.16.0`, gpu wants `v1.11.0` (skew); metal wants
  `aikit/gpu v0.25.2`, root records `v0.25.0`. Align everything on **`aikit v1.16.0` /
  `aikit/gpu v0.25.2`**, tidy the root last.
- **metal is missing an `aikit` require (B-02):** it imports `aikit/linalg` + `aikit/mmap`
  but only requires `aikit/gpu` (a different module). Add `require github.com/townsendmerino/aikit v1.16.0`.

## Pre-flight (before touching versions)

1. All §1 API blockers are in (B-08…B-14 — see `docs/completed/audit-2026-08-05.md`; B-09/B-10/B-11/B-13
   landed 2555edc; B-08/B-12 must be done first — they are breaking-to-fix-after-tag).
2. `docs/completed/audit-2026-08-05.md` §9 "before the tag" set (C-18/C-19 in; confirm nothing regressed).
3. **§C1 parity re-validation** (below, real T3 on the box) is scheduled — it is the one ⛔ gate.
4. Working tree clean; on `main`; `gh auth switch --user townsendmerino`.

## The three-step tag (B-06)

The cycle is broken by tagging the root **twice**: once so the submodules have a real version
to require, once so the root points at the tagged submodules.

**Step 1 — tag the root.**
```
# align the aikit versions the root records (B-07)
go get github.com/townsendmerino/aikit@v1.16.0 github.com/townsendmerino/aikit/gpu@v0.25.2
go mod tidy
git commit -am "release: root v0.9.0 (aikit aligned)"
git tag v0.9.0
```

**Step 2 — point each submodule at the tagged root, align deps, tidy, tag.**
For each of `gpu/`, `cuda/`, `metal/` (and `demo/agent/` if kept):
```
cd <mod>
go get github.com/townsendmerino/goinfer@v0.9.0          # B-01/B-05: real version, not placeholder/pseudo
go get github.com/townsendmerino/aikit@v1.16.0            # B-07 (metal ALSO fixes B-02: adds the aikit require)
go get github.com/townsendmerino/aikit/gpu@v0.25.2        # where applicable
# TEMPORARILY remove the `replace github.com/townsendmerino/goinfer => ../` line, then:
GOWORK=off go mod tidy                                    # B-02/B-03/B-04: complete go.sum for the SELECTED versions
GOWORK=off go build -tags <cuda|gpu|metal> ./...          # PROVE it builds with no workspace
# restore the `replace => ../` line (dev convenience; ignored by consumers)
```
Then tag them: `git tag gpu/v0.9.0 cuda/v0.9.0 metal/v0.9.0`.

**Step 3 — point the root at the tagged submodules; tag the root again.**
```
go get github.com/townsendmerino/goinfer/gpu@v0.9.0 \
       github.com/townsendmerino/goinfer/cuda@v0.9.0 \
       github.com/townsendmerino/goinfer/metal@v0.9.0
go mod tidy
git commit -am "release: root v0.9.1 (require tagged submodules)"
git tag v0.9.1
git push origin main v0.9.0 v0.9.1 gpu/v0.9.0 cuda/v0.9.0 metal/v0.9.0
```

## The standalone-build gate (make B-01…B-04 catchable next time)

Add to CI, one job per submodule — **no workspace**, so `replace` and the borrowed root
`go.sum` cannot mask a missing require/hash:
```
cd <mod> && GOWORK=off go build -tags <tag> ./... && GOWORK=off go vet -tags <tag> ./...
```
`cuda` runs on the Linux/NVIDIA runner; `metal` on macOS; `gpu` anywhere with the `gpu` tag.
Also add `./cuda` and `./demo/agent` to the dev `go.work` (B-05) so local backend work resolves
from the tree, and document that a workspace is mandatory for cross-module development.

## §C1 — parity re-validation (the ⛔ numeric gate)

The pre-tag campaigns (spec tree-attention in `forwardn.go`, the Cohere `lmHeadN` logit-scale
path, gemma4_text merges) and the audit fixes touched hashed-core files. Before the tag:
- `go test ./decoder -run ParityManifest` must be green. Mac non-numeric refreshes used
  `scripts/refresh_parity_hashes.sh` (goldens-gated, `validated_at` preserved) — those prove
  only the paths the *committed* fixtures exercise.
- **On the box:** run the real T3 suite (`scripts/parity_sweep.sh` / the `-tags cuda` real
  parity) on real checkpoints, then `-update` the manifest (bump `validated_at` + metrics) at
  the freeze commit. That is the true validation; the Mac refresh is not a substitute.

## Test hooks build tag (`goinfer_testhooks`)

Cross-module test-only hooks (the `*ForTest` / `Set*ForTest` seams a backend's tests use to
poke the decoder — audit B-08) live in `testhooks_gen.go` files under
`//go:build goinfer_testhooks`, so they are **not** part of the public API. Any test file that
calls one carries the tag too (`//go:build … && goinfer_testhooks`). Consequences:

- **Every test invocation that must exercise those tests passes `-tags goinfer_testhooks`**
  (combined with `gpu`/`cuda`/`metal` as needed). CI does this; `scripts/skip_census.py`
  defaults to it. A plain `go test ./...` without the tag still compiles — it just *skips* the
  tagged tests (they do not run, and are not a silent failure because CI runs them with the tag).
- **Shared test helpers must NOT live in a tagged file** — put them in an untagged
  `*_test.go` (e.g. `metal/testshared_test.go`), or an untagged sibling test can't see them.
- Production `go build ./...` (no tag) must stay clean — the hooks are absent, proving they
  are not referenced by non-test code.

## Freeze rule

Once §C1 passes: **no edits** to `serialize.go` / `weights.go` / `gguf.go` / core (`model.go`,
`mlp.go`, `arch.go`, `kvcache.go`, `forwardn.go`) / any `forward_*.go` until the tag lands.
Each such edit resets §C1. Land the tag; resume on the next patch.
