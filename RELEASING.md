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

- **Submodule go.mods carry NO `replace` (2026-08-15).** The dev-convenience
  `replace github.com/townsendmerino/goinfer => ../` was **removed** from `gpu`/`cuda`/`metal`/
  `demo-agent` go.mods, because a committed replace makes **`go install …/metal/cmd/serve@vX`
  fail** — `go install pkg@version` treats the target as main and rejects replace directives (C3
  finding, `docs/measurements/c3-metal-consumer-window.md`). The replace was only ever "ignored by
  consumers" on the *require* path; the *binary* path could not `go install`. Every submodule still
  must `require` a **real, published** root tag (`v0.0.0-00010101…` placeholder = B-01, a pseudo-
  version = B-05 — both publish a broken dependency permanently). **Do NOT re-add the replace.**
- **`go.work` is gitignored, per-machine, and now MANDATORY for cross-module dev.** With the replaces
  gone, a submodule build outside a workspace resolves the root from the proxy; local cross-module
  work needs a `go.work` covering **all five modules** (`.`, `./gpu`, `./cuda`, `./metal`,
  `./demo/agent`). It stays dev-only — nothing in the shipped tag may depend on it, which CI's
  standalone-build step (below) is what proves.
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
  diffed across its own tags (see `docs/queue-release.md` E6). And `metal` needs an explicit `aikit`
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
# The go.mods are replace-free (2026-08-15) — nothing to remove/restore. Just point at the tag:
GOWORK=off go get github.com/townsendmerino/goinfer@v0.10.1        # B-01/B-05: real version, not placeholder/pseudo
GOWORK=off go get github.com/townsendmerino/goinfer/gpu@v0.10.1    # demo/agent ONLY (it requires goinfer/gpu too)
GOWORK=off go mod tidy                                             # B-02/B-03/B-04: complete go.sum for the SELECTED versions
GOWORK=off go build -tags <cuda|gpu|metal> ./...                  # PROVE it builds with no workspace
# NO replace to restore — do not re-add it (it breaks `go install …@vX`; see the Invariants).
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

**The GitHub Release body is CANONICAL for release notes.** `docs/internal/RELEASE-*.md` is a
**working draft**: it is gitignored, so it exists on one machine and in no clone, and it **stops
being authoritative the moment the Release is published**. Edit the Release, not the draft, once it
is out — otherwise the only copy anyone can read and the only copy anyone is updating are different
documents, and the divergence is invisible to everyone but the person holding the draft.

## Release assets embed somebody else's weights — check the pins (M-33)

`.github/workflows/release-assets.yml` attaches 12 `goinfer-chat-{0.5b,1.5b}-*` binaries with a
Qwen2.5-Coder-Instruct GGUF **compiled in**. Those assets are a redistribution of Apache-2.0
weights, which is why the workflow also attaches `QWEN2.5-CODER-LICENSE.txt` (§4(a)) and `NOTICE.txt`
(§4(c)), and why `NOTICE` carries an "Embedded model" section. Until 2026-09-02 `NOTICE` said the
project "does not embed or distribute any model weights itself" while exactly that shipped, no
license travelled with the asset, and the GGUF was fetched from `resolve/main` with **no digest** —
so a re-upload upstream, or a `workflow_dispatch` re-run under the same tag, silently changed the
model inside an already-published release.

Nothing to do per-release while the pins hold: the workflow verifies each weight file's sha256 and
the license's, and **fails the build on a mismatch**. What that failure means is that upstream
MOVED — go read the model repo's history and update the matrix deliberately. It is not a flake, and
deleting the check is not the fix. If the tier list or the model ever changes, `NOTICE` and
`THIRD_PARTY_LICENSES.md` ("Embedded model weights", hand-maintained — `go-licenses` will not
regenerate it) both have to move with it.

## Queue-gated follow-ups — consult QUEUE.md at each tag (B5)

A tag is the natural checkpoint to review what is outstanding, and the release process is read at
exactly the moment those triggers fire. **After pushing the tags, open the four queues — `docs/queue-release.md` first, then
performance/correctness/engineering (`docs/QUEUE.md` indexes them) — and action any
item whose trigger is a release or an aikit bump.** The queue is the list; this line is what makes a
tagger look at it — a file nothing reads is inert the first week nobody opens it.

First concrete customer:

- **C3 · Metal consumer window.** If THIS release carries an **aikit bump** (`aikit` and/or `aikit/gpu`
  increased vs the previous tag — the B-07 version-alignment step above is where you'd have seen it),
  then **C3 runs on `macbook-arm64` against this tag**. *(The condition is the bump, deliberately —
  no version floor. A number standing in for the real condition is a literal that drifts; QUEUE.md's
  copy carried "≥ v0.13.0" and was corrected 2026-08-12 for that reason.)*: an out-of-tree consumer evaluation of the
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
  only the paths the *committed* fixtures exercise.

  **Quote the script's COUNTS, not the fact that it passed, and record the machine.** A refresh's
  green is exactly as wide as the fixtures that machine has, and skips are silent — they are
  missing assets, not failures, and the run still reads PASS. Measured on the MacBook 2026-08-31:
  **28 ran / 20 skipped / 0 failed**, and among the skips were **all eleven GGUF quant gates**
  (`no GGUF at ../testdata/tinyllama-gguf/…`) plus two of the three int8×int8 goldens. int4 and one
  int8×int8 golden did run. The same script on the box proves a different set. See
  `docs/parity-coverage-policy.md` §"Scoped: a goldens green names the quantizations that actually
  RAN". **It runs on EITHER machine** (it ran on
  `linux-62gb` repeatedly on 2026-08-12), not the mac — earlier text here calling it "the Mac
  tool" was wrong. **Its proof trailer records the goldens count but NOT the execution arch**
  (now `arch=` — see the script), so a *past* refresh with only `goldens=N` cannot tell you whether
  its f32 goldens ran on arm64 or amd64.
- **Arch exception for a float expression-rewrite.** When a bump carries an expression-rewrite to a
  **float** path (e.g. a reworked matmul inner loop), the argmax+cosine goldens can pass on one arch
  and breach on the other — arm64 fuses FMA where amd64's baseline does not (`parity-coverage-policy.md`,
  "arch-scoped"). So the f32 goldens must be run **on arm64 explicitly**, and the run must **say so**;
  an amd64-only refresh does not discharge it. And a green proves the **argmax margin survives the
  reordered summation** on the FMA-fusing arch — **not** cross-arch byte-identity, which f32 goldens
  (argmax+cosine) never show. (Concretely: `2e8dfb6`'s 19 f32 rows carry no arch, and the aikit v1.17.0
  f32 blocked-matmul rework is exactly such a rewrite — so this is a **new gate created 2026-08-12**,
  open, not a pre-existing one that was skipped. See `docs/QUEUE.md` mac batch.)
- **On the box:** run the real T3 suite (`go run ./cmd/gate parity` / the `-tags cuda` real
  parity) on real checkpoints, then `-update` the manifest (bump `validated_at` + metrics) at
  the freeze commit. That is the true validation; the Mac refresh is not a substitute.

    **§C1 HAS TWO HALVES, AND THE SECOND CAN FAIL SEPARATELY.** The wording above assumed an
    emitter that stamps *truthfully*, so that "run the sweep" and "`-update` the manifest" read as
    one step described twice. **B15 disproves that assumption.** They are separate obligations:

    | half | what discharges it |
    |---|---|
    | **evidence** | the sweep runs green on real checkpoints |
    | **stamp** | the manifest records what the sweep actually proved |

    **The stamp half can be UNSATISFIABLE while the evidence half is met.** At v0.13.0 the emitter
    would have promoted `glm4_moe`, `mixtral`, `qwen2_5_vl` and `qwen2_moe` from `experimental` to
    **`validated`** while leaving `method: tiny-golden`, and rewritten `mellum`'s
    `real-model-oracle` to `real-oracle` — not a T3 method at all. `TestParityManifest_methodTier`
    catches it, which is why it did not ship.

    **When the stamp cannot be made truthfully, TAG WITHOUT IT and record the exception** — here,
    in the tag's own record, and in the queue entry for the emitter defect.
    **Committing a known-false promotion to satisfy a checklist is worse than an openly unmet
    item**: the checklist is a proxy for the claim, and satisfying the proxy by falsifying the claim
    inverts the point of having it. An unmet item is visible and fixable; a false `validated` row is
    neither.

    The decision is the release owner's — waiting for the emitter fix is equally legitimate. What
    is **not** legitimate is proceeding quietly: either way it gets **written down as a decision**
    rather than skipped as a step.

    **§C1 HAS A THIRD OBLIGATION: PROMOTE WHAT THE SWEEP CONFIRMED.** A gate with no entry in
    `testdata/gate_ledger.json` is FIRST-RUN, so its failure is reported as an ITEM and **cannot
    block the next tag**. The sweep never writes that record itself — auto-promotion would turn
    "never checked" into "expected" in one silent step — and `gate_ledger.py reconcile` only
    *prints* the first-run list. So after a green sweep, promote each gate the sweep confirmed:

    ```
    python3 scripts/gate_ledger.py promote --gate <G> --value PASS --by <you> \
        --commit <sweep SHA> --note "<the sweep log this came from>"
    ```

    Skipping this is not inert: the ledger was seeded once on 2026-08-14 and left alone, and five
    required gates — including `TestInt4_forwardParity`, the broadest quant check in the checkset —
    were still non-blocking two and a half weeks later **while a PASS for each sat in the v0.15.0
    sweep log** (audit-2026-09-02 G-04). `TestParity_everyRequiredGateIsConfirmed` now fails CI on
    a required gate that is neither in the ledger nor in `neverConfirmed` with a written reason.

## §C1-M — the Metal device gate (manual, and what "green" means)

**CI cannot run it, and that is a property of the runner, not a choice.** `macos-latest`'s
paravirtual Metal/objc layer SIGSEGVs inside purego's `objc_msgSend` the moment a test touches the
device (measured: a near-null crash in `objc.ID.Send`), and the abort takes the whole test binary
down — so the package's device tests cannot even attempt-and-skip. `.github/workflows/ci.yml`'s
`metal-darwin` job therefore runs **build + vet + one device-free test**
(`TestParity_NaNCosineFailsTheGate`) and nothing else. Everything device-bound rests on a human
running it on a real Metal box. Writing that down is the point of this section: at 1.0 the run is a
ritual with a stated meaning, not a habit someone remembers.

**The exact command**, on a Mac with the checkpoints in `$HOME/models`:

```
go run ./cmd/gate gpu                       # auto-detects darwin ⇒ metal
GOINFER_GATE_BACKEND=metal go run ./cmd/gate gpu   # explicit
GOINFER_GATE_MODELS=/path/to/models go run ./cmd/gate gpu
```

**What green means — all four, or it is not green:**

1. **All 7 declared check groups report a verdict.** Metal declares
   `cleangpu seam suite cgofree lifecycle prefill repo`. The script reconciles declared against
   emitted, because a block that died mid-run used to vanish and still report PASS (audit G-01). A
   group that emits nothing is itself a FAIL.
2. **Zero FAIL**, and the verdict line names the commit. A dirty tree is reported as a
   PROVENANCE failure even when every check passed — the verdict names a commit, and an
   uncommitted edit means it does not describe what that commit contains.
3. **A SKIP IS NOT A PASS.** Go prints `ok` for a package whose tests all skipped, so the script
   counts real runs and lists every skip under "this gate does NOT cover". Read that list; it is
   the honest scope of the run. `RAN == 0` prints `NO GATE` and exits non-zero.
4. **The log is archived, not `mktemp`'d.** C1a's lesson: a verdict nobody can re-check is not
   evidence. Paste or commit the output with the tag record.

**What the Metal run vouches for has WIDENED, so re-read it rather than assuming the old scope:**
gpt-oss Metal residency (G10) and the Mellum2 real-weight resident-parity gate (G11) both landed
after the last ritual was written.

**Known-red at the time of writing, with a safety condition — do not "fix" it by re-baking blind.**
`TestMetalSnapshotGolden` is EXPECTED to fail on the Mac until its goldens are re-baked
(`GOINFER_UPDATE_GOLDENS=1 go test -run TestMetalSnapshotGolden ./metal/`). The cause is audit
G-02, fixed on Linux where the suite cannot run: the checkpoint call now drives `ForwardEmb` with
the production-scaled embedding row and `Forward`/`ForwardArgmax` apply the arch embed scale. The
re-bake is only legitimate if **`gemma4-dense-scaled` entries move** (it has `EmbedScale = √hidden`)
**and `mixtral-tiny` entries do NOT** (it has no embed scale). If mixtral-tiny moves, something
other than G-02 changed and the re-bake must be refused pending investigation.

## Test hooks build tag (`goinfer_testhooks`)

Cross-module test-only hooks (the `*ForTest` / `Set*ForTest` seams a backend's tests use to
poke the decoder — audit B-08) live in `testhooks_gen.go` files under
`//go:build goinfer_testhooks`, so they are **not** part of the public API. Any test file that
calls one carries the tag too (`//go:build … && goinfer_testhooks`). Consequences:

- **Every test invocation that must exercise those tests passes `-tags goinfer_testhooks`**
  (combined with `gpu`/`cuda`/`metal` as needed). CI does this; `go run ./cmd/gate census`
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
