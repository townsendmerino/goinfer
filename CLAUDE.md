# goinfer — agent notes

## Models: stored in the archive, benchmarked from local disk

**Every timed run reads its checkpoint from `~/models` on the machine doing the timing.
The archive is storage, never a read path for a measurement — on either machine.**

| | bench set — the ONLY place a row may be measured from | the archive, which is NOT a bench surface |
|---|---|---|
| `nobara-pc` (amd64/CUDA) | `~/models` on NVMe | **`/srv/models`** — local, but a 5400 rpm SMR disk |
| MacBook (arm64/Metal) | `~/models` on internal SSD | **`/Volumes/…`** — the SMB mount of the same disk |

Both forbidden roots are named above because **neither prohibition catches the other**.
`/srv/models` is *local storage on the box that measures CUDA*, so "benchmark from local
disk" reads as permission for it; `/Volumes/` is the only path the older phrasing named,
and it does not exist on Linux. A run off either one measures a 5400 rpm SMR disk — over
the LAN as well, in the `/Volumes/` case — instead of the engine. **It does not error.** It
returns a plausible, wrong number. Any row whose model path starts with `/srv/models` or
`/Volumes/` is void and must be re-measured after a `models-pull`.

The full storage table, the `models-pull` / `models-push` usage, and the reason the share is
deliberately not automounted live in `docs/benchmarks.md` § "Model storage" — **that section
is the authority; do not restate its details here.** Enough to work from: `models-pull <name>`
copies archive → `~/models` (resumable, rsync over SSH, must be on the LAN — Tailscale SSH
intercepts port 22 and hangs), `models-push <name>` goes the other way and refuses to claim
success unless byte counts match both sides.

## Benchmarking

`docs/benchmarks.md` is provenance-gated: a number enters a table only with machine,
checkpoint+quant, greedy/seed, pinned versions, date, thermal note, and local-disk path.
Read its Methodology section before adding or changing any measurement, and reproduce via
`scripts/bench_peer.py` — **not** `scripts/bench_compare.sh`, which runs in-process Go benchmarks and
by its own design note drives no peer. Putting its output beside a peer number divides a kernel
throughput by an end-to-end one; that is what produced the retired "0.5B 1.78×" claim.
`bench_compare.sh` is still the right tool for goinfer-vs-goinfer work, and only that. Peer
comparisons must be same-session interleaved — drift between sessions is ~3.5% on this box and
silently corrupts ratios.

CUDA rows are anchored to a specific NVIDIA driver version. Changing the driver invalidates
comparability and requires a deliberate re-anchor, not a silent carry-forward.

## Working in the tree

**Five Go modules**, not one: the root, `gpu/`, `cuda/`, `metal/`, `demo/agent/`. `go.work` is
gitignored and **mandatory** for cross-module work; a `GOWORK=off` build of a submodule resolves
the root from the proxy at its last published tag, so it can fail with "method not found" on
perfectly good code. That failure is expected between releases, not a bug.

**Build tags gate real code.** `realckpt` (heavy, real checkpoints), `goinfer_testhooks`
(cross-module test seams), `cuda` / `gpu` / `metal`. A change that compiles untagged can be
broken under a tag — this has shipped to `main` at least twice. `go vet -tags realckpt ./...`
and `go vet -tags 'cuda goinfer_testhooks' ./cuda/` are cheap; run the ones your change touches.

**`gofmt -l .` is a CI gate and nothing here auto-formats.** Run it across every module you
touched before committing.

**Never `git add -A` / `git add .`.** It sweeps generated fixture metadata into git, which makes
dir-only skip-guards think a fixture exists and flips skips into failures. Stage explicit paths.

**`main` moves under you.** Other sessions and the other machine push to it. `git fetch` +
rebase before pushing, and if a file you are editing has changes you did not make, they are
someone else's in-flight work — commit *your* hunks, leave theirs in the tree.

## Tests

**A SKIP IS NOT A PASS.** `go test` prints `ok` for a package whose tests all skipped, so a
green line in 0.02s usually means "no assets, nothing ran". Confirm with `-v` and read for
`--- PASS`, not `ok`.

**`go test -v ./a/ ./b/` prints nothing until `./a/` finishes.** Output is buffered per package.
Run one package per invocation and `tee` if you need to watch progress.

Heavy tests need `GOINFER_HEAVY_TESTS=1` and their assets; `testdata/assets.json` is the
registry and `go run ./cmd/gate` is the runner (`census`, `heavy`, `parity`, `composition`,
`selector`, `gpu`, `mutation`).

## Long-running work

Anything over a few minutes must be **detached** — `setsid nohup … </dev/null &` — because a
plain background shell dies at session boundaries. Verify it took: `ps -o pid,ppid,sid` should
show **PPID 1**.

**Archive the log; do not leave it in `/tmp`.** `/tmp` gets cleared, and a verdict nobody can
re-read is not evidence. Write it somewhere durable and reference the path in the writeup.

Prefer committed increments that survive interruption over one big commit at the end.

## Citations and the pre-push hook

`scripts/queue_citation_lint.py` runs **as a pre-push hook and REFUSES the push** on a red. Two
traps worth knowing before you hit them:

- A backtick-quoted concrete path under a **gitignored** directory (`docs/internal/…`) is a
  FORBIDDEN DESTINATION — *including in another repo*, since nobody else can resolve it.
  Describe the record in prose instead; that is what `c494c62` did.
- A stale `path:line` index is fixed with `--update`, not by deleting the citation.

`git push --no-verify` exists and is almost never the right answer — the lint is usually right.

## Measurement discipline

This repo's measurements are the product, so the standards are load-bearing:

- **Difference matched observations; do not pool them.** When variants can be interleaved on the
  same input, pooled means carry between-input variance that swamps the effect. Measured here:
  the same data read pooled gave sd 10–35 tok/s against an ~8% effect, and read paired gave sd
  5.5–8.7 with 11/12 pairs. They disagreed about whether an effect existed. Recorded as rule 7
  of aikit's internal measuring-performance notes (described, not cited as a path — a cross-repo
  path does not resolve from any other clone, which is the trap two bullets down. Note the file is
  TRACKED in aikit, not gitignored as this line used to claim; the advice is unchanged, the reason
  was wrong).
- **Include the do-nothing arm.** "Beats every configuration" means nothing if *off* wins. A
  speculation suite was found where no verify width beat running no drafter at all — only
  visible because `off` was a competitor.
- **Pre-register the decision rule** for anything that will be argued about, and include an
  explicit *ambiguous → parked* band. The zone just below the threshold is where motivated
  reasoning lives.
- **Negative results get committed with the same care as wins**, and re-baselining a floor
  because a number moved is how a regression gets blessed — move a bar only with a mechanism.
- **Prior art is read before re-deriving.** A doc may say something narrower than its headline:
  `01-grammar-fused.md`'s α ≈ 0.20 is a verdict on the grammar-automaton drafter, in its own
  words, *not* on grammar speculation in principle. Treating it as settled would have killed a
  live question with the wrong evidence.

## Releases

`RELEASING.md` is the authority — five modules, a two-step tag, and a GitHub Release on the
**root tag only** as a non-optional final step. Do not restate its version numbers anywhere;
read them with the command it gives. `gh` operations use the **townsendmerino** account.
