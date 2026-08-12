# Releasing goinfer (tri-module: root + gpu + cuda + metal)

goinfer is **five Go modules** in one repo — the root (`github.com/townsendmerino/goinfer`),
plus `gpu/`, `cuda/`, `metal/`, and the ancillary `demo/agent/`. This file is the tag ordering
that produces a consistent, go-gettable set, plus the module-graph fixes that must land with it.
It exists because four release blockers in the 2026-08-05 audit (§1a/§1b: B-01…B-07) were "the
tagged submodules don't build for anyone" — the gitignored `go.work` masks it locally, so it
survives to a tag unless caught here.

> **M-19 broke the requirement cycle — this is now a TWO-step tag, not three.** The audit-era
> flow tagged the root *twice* because the graph was a cycle (root → gpu/cuda/metal → root). M-19
> moved the backend imports out of the root (`cmd/serve`/`demo` build tags became no-op guards),
> so the **root no longer requires `gpu`/`cuda`/`metal`** — the graph is one-way: submodules
> require the root, the root requires none of them. Tag the root **once**, then the submodules;
> there is no second root tag. Confirm before starting: `grep -E "goinfer/(cuda|gpu|metal)" go.mod`
> in the root prints nothing. (git history holds the obsolete three-step flow.)

## Invariants (why the steps are shaped this way)

- **`replace => ../` is honored only when a submodule is the *main* module** (its own
  build/tidy). A *consumer* of `goinfer/metal@vX` gets the `require` line, and Go **ignores
  the replace** — so a `require goinfer v0.0.0-00010101…` placeholder (B-01) or an Aug-2
  pseudo-version (B-05) publishes a broken/ stale dependency permanently. Every submodule must
  `require` a **real, published** root tag before *it* is tagged.
- **`go.work` is gitignored** (`.gitignore`) and must stay dev-only. Nothing in the shipped
  tag may depend on it. CI's standalone-build step (below) is what proves this.
- **Version alignment (B-07):** all five modules must agree on `aikit`, and `cuda`/`metal` must
  agree on `aikit/gpu`. **Read the versions, do not read them here.**

  ```sh
  for m in . gpu cuda metal demo/agent; do printf '%-12s ' "$m"; grep -h aikit $m/go.mod | tr '\n' ' '; echo; done
  ```

  This used to name the versions, and it has gone stale **twice** — first telling you to align on
  `aikit/gpu v0.25.2` when that had become a downgrade, then naming `v1.16.0`/`v0.27.0` after the
  v1.17.0 bump. A release checklist that restates a value maintained somewhere else is a second copy
  that drifts, which is the defect this repo keeps finding; the command above has no such failure
  mode. Act only if a module actually disagrees, **never downgrade**, and tidy the root last.

  Two things the command will not tell you, so they are stated rather than restated:
  `aikit` and `aikit/gpu` are **separate modules with separate tag series that do not track each
  other** — equal-looking version numbers mean nothing across them, and a nested module must be
  diffed across its own tags (see `docs/QUEUE.md` E6). And `metal` needs an explicit `aikit`
  require: it imports `aikit/linalg` and `aikit/mmap` while only `aikit/gpu` is implied. (That was
  B-02; it is present now, and the command above is what confirms it still is.)

## Pre-flight (before touching versions)

1. All §1 API blockers are in (B-08…B-14 — see `docs/completed/audit-2026-08-05.md`; B-09/B-10/B-11/B-13
   landed 2555edc; B-08/B-12 must be done first — they are breaking-to-fix-after-tag).
2. `docs/completed/audit-2026-08-05.md` §9 "before the tag" set (C-18/C-19 in; confirm nothing regressed).
3. **§C1 parity re-validation** (below, real T3 on the box) is scheduled — it is the one ⛔ gate.
4. Working tree clean; on `main`; `gh auth switch --user townsendmerino`.

## The two-step tag (post-M-19)

The root no longer requires the submodules, so tag the root **once**, push it, then bump and tag
the submodules against that published root. The example uses `v0.10.1`.

**Step 1 — tag the root, push FIRST.** The submodules `go get` the root from the proxy, so it must
be published before Step 2 runs.
```
# align aikit ONLY if a module actually disagrees (B-07 above) — as of v0.10.3 they all agree.
go mod tidy
git commit -am "release: v0.10.1 (<one-line summary>)"
git tag v0.10.1
git push origin main v0.10.1        # published BEFORE any submodule go-gets it
```

> **If `git commit -am` says "nothing to commit", that is expected — tag HEAD instead.** This step
> assumes the `[Unreleased]` → `[vX]` CHANGELOG move happens *here*. When that move already landed
> in an earlier commit (it usually has, if the release was staged over more than one sitting) and
> `go mod tidy` is a no-op, there is nothing left for a release commit. Do **not** force an empty
> one: `git tag vX` on the current HEAD and push. Observed on the v0.10.3 run.

> **Do not use `git commit -am` with unrelated work in the tree.** Both this step and Step 2 stage
> every tracked modification. Stash anything not part of the release first (`git stash push -u --
> <paths>`) — Step 2 in particular runs `-am` four times across four modules.

**Step 2 — point each submodule at the tagged root, prove the standalone build, tag.**
Do `gpu/`, `cuda/`, `metal/` first, then `demo/agent/` LAST (it also requires `goinfer/gpu`, so
`gpu/vX` must be tagged and pushed before demo/agent can `go get` it).
```
cd <mod>
# TEMPORARILY remove the `replace github.com/townsendmerino/goinfer => ..` line, then:
GOWORK=off go get github.com/townsendmerino/goinfer@v0.10.1        # B-01/B-05: real version, not placeholder/pseudo
GOWORK=off go get github.com/townsendmerino/goinfer/gpu@v0.10.1    # demo/agent ONLY (it requires goinfer/gpu too)
GOWORK=off go mod tidy                                             # B-02/B-03/B-04: complete go.sum for the SELECTED versions
GOWORK=off go build -tags <cuda|gpu|metal> ./...                  # PROVE it builds with no workspace
# restore the `replace => ..` line (dev convenience; ignored by consumers)
```
`metal` on a Linux box prints `matched no packages` (its files are darwin-gated) — that is
EXPECTED; the `go mod tidy` go.sum bump is the deliverable, and macOS/CI runs the real build gate.
Then commit the go.mod/go.sum bumps and tag. **`git tag` takes ONE name per call** (unlike
`git push`, which takes many):
```
git commit -am "release: gpu/cuda/metal require goinfer v0.10.1"
git tag cuda/v0.10.1 ; git tag gpu/v0.10.1 ; git tag metal/v0.10.1
# a SEPARATE commit for demo/agent (requires goinfer + goinfer/gpu):
git commit -am "release: demo/agent requires goinfer + goinfer/gpu v0.10.1"
git tag demo/agent/v0.10.1
git push origin main cuda/v0.10.1 gpu/v0.10.1 metal/v0.10.1 demo/agent/v0.10.1
```

## GitHub Release (non-optional final step)

Every version also gets a **GitHub Release on the ROOT tag only** (13/13 prior versions; never
per-submodule). After all five tags are pushed:
```
gh release create v0.10.1 \
  --title "goinfer v0.10.1 — <one-line descriptor>" \
  --notes-file <notes-from-changelog> \
  --latest
```
Notes come from the CHANGELOG section for that version (Added/Changed/Fixed); keep any BREAKING
marker honest even on a patch bump. `go get` resolves from the git tag via the proxy and needs no
Release object — this step is for the rendered notes + the watcher notification.

## Queue-gated follow-ups — consult QUEUE.md at each tag (B5)

A tag is the natural checkpoint to review what is outstanding, and the release process is read at
exactly the moment those triggers fire. **After pushing the tags, open `docs/QUEUE.md` and action any
item whose trigger is a release or an aikit bump.** The queue is the list; this line is what makes a
tagger look at it — a file nothing reads is inert the first week nobody opens it.

First concrete customer:

- **C3 · Metal consumer window.** If THIS release carries an **aikit bump** (`aikit` and/or `aikit/gpu`
  increased vs the previous tag — the B-07 version-alignment step above is where you'd have seen it),
  then **C3 runs on `macbook-arm64` against this tag**: an out-of-tree consumer evaluation of the
  cgo-free Metal backend (build with no Xcode, decode tok/s vs the 73.6 claim, bit-identity, and
  whether the tautological-gate shape is live on Metal). See `docs/QUEUE.md` → "C3 · Metal consumer
  window" for the full scope, trigger, and bound. **This line is the durable carrier** — read by
  whoever cuts the tag, dependent on no session or cron surviving.

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
- `go test ./decoder -run ParityManifest` must be green. Non-numeric refreshes use
  `scripts/refresh_parity_hashes.sh` (goldens-gated, `validated_at` preserved) — those prove
  only the paths the *committed* fixtures exercise. **It runs on EITHER machine** (it ran on
  `linux-62gb` repeatedly on 2026-08-12), not the mac — earlier text here calling it "the Mac
  tool" was wrong. **Its proof trailer records the goldens count but NOT the execution arch**
  (now `arch=` — see the script), so a *past* refresh with only `goldens=N` cannot tell you whether
  its f32 goldens ran on arm64 or amd64.
- **Arch exception for a float expression-rewrite.** When a bump carries an expression-rewrite to a
  **float** path (e.g. a reworked matmul inner loop), the argmax+cosine goldens can pass on one arch
  and breach on the other — arm64 fuses FMA where amd64's baseline does not (`parity-coverage-policy.md`,
  "arch-scoped"). So the f32 goldens must be run **on arm64 explicitly**, and the run must **say so**;
  an amd64-only refresh does not discharge it. (Concretely: `2e8dfb6`'s 19 f32 rows carry no arch, and
  the aikit v1.17.0 f32 blocked-matmul rework is exactly such a rewrite — the arm64 read is owed. See
  `docs/QUEUE.md` mac batch.)
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
