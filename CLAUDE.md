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

**`staticcheck` is also a CI gate, and a LOCAL RUN CAN SILENTLY CHECK NOTHING.** Install CI's pinned
version once — `go install honnef.co/go/tools/cmd/staticcheck@v0.8.0` — then run the tagged variants
(`-tags 'cuda goinfer_testhooks' ./cuda/...`, `-tags 'gpu goinfer_testhooks' ./gpu/...`).

**Do NOT combine `go run …@v0.8.0` with `GOOS=linux`.** `go run` then builds a LINUX staticcheck and
fails to exec it on darwin — `exec format error`, **and the shell still reports exit 0**. Use an
INSTALLED native binary and set `GOOS` only for the analysis target:
`GOOS=linux GOARCH=amd64 CGO_ENABLED=0 ~/go/bin/staticcheck -tags cuda ./...`.

**An empty staticcheck result is indistinguishable from one that never ran, so prove the gate can go
red.** A four-line throwaway package with an unused struct field must print
`field deadField is unused (U1000)`. That is the exact defect that held CI red across three pushes on
2026-08-28, and the same class `cmd/gate/mutation.go` was built to prevent.
A `staticcheck` on PATH can be years older than the toolchain and then fails to analyse ANYTHING,
emitting only `internal error in importing "internal/cpu" (unsupported version: 4)` — which looks
like a broken tool rather than an unrun gate, and exits without checking your code. Measured
2026-08-28: a dead struct field (U1000) shipped in `decoder/mtp.go` and held CI red across **three
pushes**, because `gofmt` and `go vet` were run and staticcheck was not.

**Check CI after pushing.** `gh run list --limit 5`. Red does not announce itself, and a break rides
along under every subsequent push until someone looks.

**Never `git add -A` / `git add .`.** It sweeps generated fixture metadata into git, which makes
dir-only skip-guards think a fixture exists and flips skips into failures. Stage explicit paths.

**`main` moves under you.** Other sessions and the other machine push to it. `git fetch` +
rebase before pushing, and if a file you are editing has changes you did not make, they are
someone else's in-flight work — commit *your* hunks, leave theirs in the tree.

## Tests

**A SKIP IS NOT A PASS.** `go test` prints `ok` for a package whose tests all skipped, so a
green line in 0.02s usually means "no assets, nothing ran". Confirm with `-v` and read for
`--- PASS`, not `ok`.

**A UNIT TEST THAT SUPPLIES ITS OWN CALLING CONVENTION PROVES THE UNIT WORKS WHEN CALLED THAT WAY —
NOT THAT ANYTHING CALLS IT THAT WAY.** It is the microbenchmark trap one level up: same failure, in
composition rather than in cost. Measured here (G27): `optFwdGate` documents a two-way hysteresis
band, and `TestOptFwdGate_hysteresis` confirms it by driving `Observe` in an unconditional loop. But
production calls `Observe` only from inside the branch that `Should()` guards, so once the gate turns
off nothing is observed again, the estimate freezes, and the re-enable half of the band is
unreachable. The component test passes, is correct, and vouches for behaviour the system cannot
produce. **When a component's contract depends on HOW OFTEN or UNDER WHAT CONDITION it is called,
test it through its caller, or the test is asserting your assumption back at you.**

**A DOC COMMENT CLAIMING COVERAGE IS NOT COVERAGE, AND IT IS WORSE THAN SILENCE.** This is
NOT the rule above — that one is about a calling convention the test supplies for itself.
This one is about a comment asserting something the assertions never touch, which turns the
test into a trap for whoever audits the claim later: they read the promise, match it to a
test name that fits, and stop. Measured 2026-08-28: `a3_divergence_test.go`'s
`TestA3FastAttentionDivergence` opens by saying it pins *"MoE excluded: the flag cannot turn
f32 attention on for a MoE arch at all"*; the body loads the DENSE bench checkpoint and
asserts nothing about MoE anywhere. The `--cpu-fast-attention` MoE exclusion it advertised as
pinned had in fact never been measured on a MoE — no MoE appears in its kernel-ratio record
either — while the excluded term was 97.1% of an 8k MoE prefill (a 4-layer-slice figure;
the full model is lower, and the win it was blocking measured 1.52x, not the slice's 3.11x —
which is its own lesson about quoting a slice as a model). **The check: for any doc
comment that asserts coverage, the body must contain an assertion naming the thing.** Both
this and the rule above defeat the same defence — reading the test NAME instead of the test
BODY — which is why neither is caught by running the suite.

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
- **A marginal stopping rule encodes an unstated assumption about CURVE SHAPE.** "Stop when a
  doubling buys < X%" is valid only where the curve is known to be monotone-diminishing. Measured
  2026-08-28 (Metal MoE slot sweep): 8→16 bought +4.1%, under a pre-registered 5% bar, so the rule
  as written stops at N=16 — and the only resolvable win was **+14.8% at N=64, two doublings
  later**. Plateau-then-step is the normal shape wherever a resource threshold gets crossed, which
  is exactly what N=64 was. If you do not know the shape, run the full ladder; that is what buys
  the right to stop early *next* time. **Corollary: pre-register two things that can disagree.**
  The ladder and the stop rule answer different questions, so the wrong one got caught by the right
  one. After a rule fails the instinct is to write a better rule; the more reliable fix is a second,
  independent pre-registration.
- **A guard that INVERTS under the condition it exists for is worse than no guard** — it actively
  reassures. Same sweep: an RSS-based memory ceiling, written specifically to catch the N=128
  slot-pressure cliff, reported **263–426 MB at N=128 against 1154 MB at N=8** — *less* memory at
  the failure point than at the baseline. Darwin's UBC reclaims under pressure, so RSS reports what
  survived, not what was asked for. Key a budget guard on a quantity you compute yourself
  (allocated slot bytes = `N × layers × per-expert`, known at build), never on the OS's account of
  what remains.
- **A RETRACTION IS NOT DONE UNTIL IT REACHES EVERY PAGE QUOTING THE FIGURE.** When you strike a
  number, `grep` for *the figure with its unit* right then and fix every instance, at the retraction
  site, while you still have the context. Recording the correction where it was found is not enough:
  measured 2026-08-28, `~1.2-1.4 tok/s` was withdrawn in `task-zeno-compare.md` and went on
  disqualifying `Qwen3.5-35B-A3B` from a qualifying agent-loop run in `queue-engineering.md` for
  months, on a figure that direct measurement then put at 1.52-1.73 (CPU) / 1.97-2.02 (Metal).
  Grep the figure WITH its unit: the bare digits matched six unrelated quantities (ms, kernel
  ratios, KV-quant multipliers), the number plus `tok/s` matched only the real ones. This is a
  convention on purpose, not a lint — the check the lint already documents as out of scope (see its
  own `aikit v1.16.0 (go.mod:6)` example) was scoped out deliberately, and one incident is not
  grounds to overturn that. Note the reach limit either way: figures also propagate into published
  artifacts outside the repo, which no repo-side tooling can ever see.

## Releases

`RELEASING.md` is the authority — five modules, a two-step tag, and a GitHub Release on the
**root tag only** as a non-optional final step. Do not restate its version numbers anywhere;
read them with the command it gives. `gh` operations use the **townsendmerino** account.
