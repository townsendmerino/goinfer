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
comparisons must be same-session interleaved. Same-build cross-session drift is ~0.7% RMS for peers on nobara (tail
to ~5%); goinfer's own is measured once, 3.6%, in a sampled cell. A cross-session ratio is not a ratio
(`docs/measurements/noise-registry.md`, TE3).

CUDA rows are anchored to a specific NVIDIA driver version. Changing the driver invalidates
comparability and requires a deliberate re-anchor, not a silent carry-forward.

## Working in the tree

**Seven Go modules**, not one: the root, `gpu/`, `cuda/`, `metal/`, `demo/agent/`, `sdktest/` (tests only: the
official OpenAI and Anthropic SDKs against the in-process server; never tagged), and `site/` (the website generator,
built by `site.yml`). `go.work` is
gitignored and **mandatory** for cross-module work; a `GOWORK=off` build of a submodule resolves
the root from the proxy at its last published tag, so it can fail with "method not found" on
perfectly good code. That failure is expected between releases, not a bug.

**Build tags gate real code.** `realckpt` (heavy, real checkpoints), `goinfer_testhooks`
(cross-module test seams), `cuda` / `gpu` / `metal`. A change that compiles untagged can be
broken under a tag — this has shipped to `main` at least twice. `go vet -tags realckpt ./...`
and `go vet -tags 'cuda goinfer_testhooks' ./cuda/` are cheap; run the ones your change touches.

**`gofmt -l .` is a CI gate and nothing here auto-formats.** Run it across every module you
touched before committing.

**`staticcheck` is also a CI gate, and a LOCAL RUN CAN SILENTLY CHECK NOTHING.** Build CI's pinned
version once — `GOWORK=off go build -C .github/actions/staticcheck -o "$(go env GOPATH)/bin/staticcheck" honnef.co/go/tools/cmd/staticcheck`
(v0.8.1 against golang.org/x/tools v0.51.0: since the move to Go 1.27.2 on 2026-10-08 no staticcheck RELEASE reads the
toolchain's export data, version 5, so `go install …@v0.8.x` analyses nothing) — then run the tagged variants
(`-tags 'cuda goinfer_testhooks' ./cuda/...`, `-tags 'gpu goinfer_testhooks' ./gpu/...`).

**Do NOT combine `go run …staticcheck` with `GOOS=linux`.** `go run` then builds a LINUX staticcheck and
fails to exec it on darwin — `exec format error`, **and the shell still reports exit 0**. Use an
INSTALLED native binary and set `GOOS` only for the analysis target:
`GOOS=linux GOARCH=amd64 CGO_ENABLED=0 ~/go/bin/staticcheck -tags cuda ./...`.

**An empty staticcheck result is indistinguishable from one that never ran, so prove the gate can go
red.** A throwaway package with an unused struct field must print
`field deadField is unused (U1000)`, and it must IMPORT A STANDARD-LIBRARY PACKAGE: a canary that imports
nothing never reads export data, so it passed on a staticcheck that failed on every real package (2026-10-08). That is the exact defect that held CI red across three pushes on
2026-08-28, and the same class `cmd/gate/mutation.go` was built to prevent.
A `staticcheck` on PATH can be years older than the toolchain and then fails to analyse ANYTHING,
emitting only `internal error in importing "internal/cpu" (unsupported version: 4)` — which looks
like a broken tool rather than an unrun gate, and exits without checking your code. Measured
2026-08-28: a dead struct field (U1000) shipped in `decoder/mtp.go` and held CI red across **three
pushes**, because `gofmt` and `go vet` were run and staticcheck was not.

**Product code reaches `main` through a pull request that merges itself; do not watch CI.** Owner rule, 2026-10-26 (SD4,
`docs/tasks/task-subsystems-2026-10.md` §12 PW1), replacing the push rules of 2026-10-13 and 2026-10-17. Commit locally, push
your branch (seconds), open a PR, and right away, while its checks are still pending, set it to merge itself with
`gh pr merge <n> --repo <owner/repo> --auto --merge` as its own bare command (no `cd`, no `&&`): a merge commit, as the merge
train used, made by GitHub once the 19 checks `main` requires pass on the PR (Francis, 2026-10-26: he does not run merges).
Then print `BRANCHES READY: #<n> <head sha>`. A session never runs a plain `gh pr merge` without `--auto`: the permission
classifier denies that as a merge without review. If `--auto` is denied too, print `BRANCHES READY` and say so, and the
coordinator takes it to Francis; **a session never routes around a denial** (no other merge command, no API call, no asking
another session to run it); it reports the denial and moves on. The rule is not strict, so the PR need
not be rebased onto the newest `main`; a conflict leaves it open. Nobody holds a turn open on its checks or on main's CI,
polls them, or schedules wake-ups for them. Read the result at the next natural point (`gh pr view <n> --json state` and
`gh run list --limit 5`: your next push, the start of the next task, and always before a tag) and report a red the moment
you see it: red does not announce itself, and a break rides along under every later merge until someone looks. Waiting on
CI was 16% of both boxes' session time over 30 days (`docs/measurements/test-wait-2026-10.md`).

**Branch protection does not enforce this; the rule does.** Admins are exempt and every session pushes as the owner's account,
so a direct push of product code to `main` goes through. Two direct pushes stay: a docs-only change
(`scripts/ci_docs_only.sh`: `docs/**`, `RELEASING.md` and `.claude/**`, none read by Go code or tests) goes straight to
`main` and runs only CI's light lint job, and a release pushes its commits and tags as `RELEASING.md`'s two-step tag says.
No hook waits on main's CI: the pre-push guard that did was removed on 2026-10-26 (owner, PW1). A main run that has started
always finishes: main's workflows never cancel a running run for the next push. Merges that land while one runs collapse
into one pending run on the tip (`.github/workflows/ci.yml`, the concurrency note), so a red main run can cover several PRs,
each green on its own. **If a hook holds a push you believe is wrong, stop and report
it to Francis. Never `--no-verify`.** A background push job pushes a named commit (`git push origin <sha>:<branch>`), never
`HEAD`, and runs in a worktree no one else edits while it waits.

**No new production env reads.** `testdata/env_reads.txt` lists every `GOINFER_*` variable production code reads
from the environment, and `TestEnvVars_docAndCodeAgree` fails on a read that is not on it. A new operator choice is a
`decoder.Options` field with a per-model accessor; a new diagnostic is a test hook. The list only shrinks — see
`docs/env-vars.md` § "Policy — where a new setting goes" for the rules and why (per-call reads change a loaded model mid-flight; two
models in one process cannot differ); the migration record is `docs/completed/task-env-config-2026-09.md`.

**A change to any guard threshold (swap, fit, memory) needs Francis's word, the same as a bypass flag.** The swap guard's default is 512 MB. Nobara's run plans set `swap_guard_mb` 4096 only as Francis's decision of 2026-10-17, made after the fact (an agent first set it on 2026-10-13 without approval), and only until a build carries the per-process guard (the server's own `VmSwap` over its pid and descendants at 512 MB, system-wide swap at a 4096 MB backstop: Francis's decision, 2026-10-17); a plan then drops `swap_guard_mb` and takes the default. No other plan, script or box takes 4096 on its own. `night.py add` refuses a run plan that pins a build carrying the guard (dba04134 or later) and sets `swap_guard_mb` unless the plan records `"swap_guard_approved_by": "Francis"` and `"swap_guard_approved_on"` with a date; plans pinning older builds are unaffected. Only Francis adds `swap_guard_approved_by`. An agent writing it is the same slip as typing a bypass flag. The commit that adds it cites where Francis gave the word.

**Never `git add -A` / `git add .`.** It sweeps generated fixture metadata into git, which makes
dir-only skip-guards think a fixture exists and flips skips into failures. Stage explicit paths.

**`main` moves under you.** Other sessions and the other machine push to it. `git fetch` +
rebase before pushing, and if a file you are editing has changes you did not make, they are
someone else's in-flight work — commit *your* hunks, leave theirs in the tree.

## Code comments

**A Go comment says what is true now.** It states what the code does, its contract (ownership, units,
concurrency, what a caller must check), and, where the shape is not obvious, the reason in a sentence or
two. It keeps every guardrail — "`fastAttn` is a parameter so spec-decode verify cannot turn it on",
"the floor keys on K deliberately; re-keying it needs a measurement at that shape first" — with a pointer
to where the evidence lives.

**It does not carry the history.** How a bug was found, dates, commit ids, which session or reviewer,
audit or tracker ids used as content, measured figures and tables, retractions and open work go
elsewhere: the commit message, the measurement or task doc, or the queue. A comment points at those by
doc path and heading, and at code by declaration name — never `file.go:NNN`, which goes stale on the
next edit above the line.

**The test for a sentence:** would someone changing this code next week act differently because of it?
Keep it. Does it explain how we came to know? Move it.

**Length is part of the rule.** Say each thing once, at the declaration it binds, in as many lines as the
contract needs and no more: an optional-interface or accessor comment is about six lines. Do not restate the
signature or a sibling's comment, do not list which backends, families or callers currently take a path (the
code is that list, and the comment is wrong the day one changes), and do not narrate the body. A comment that
runs past a short paragraph usually holds a design explanation (move it to the doc the comment cites, or
`docs/code-notes/<package>.md`) or a guardrail (keep it as its own sentence with a pointer to the evidence).

**The comments around you do not set the style; this section does.** Many files still carry long
history comments; clearing them is `docs/tasks/task-code-comments-2026-10.md`'s work, done package by
package with a comments-only gate. Do not rewrite comments in a file you are only passing through, and
do not add to the history in one. Directive comments (`//go:build`, `//go:embed`, `//nolint`, an
Example's `// Output:`) are code, not comments, for every purpose here.
`scripts/comment_census.py diff` (run by the pre-push hook, a warning only) lists added comments that carry a history marker.

## Tests

**A SKIP IS NOT A PASS.** `go test` prints `ok` for a package whose tests all skipped, so a
green line in 0.02s usually means "no assets, nothing ran". Confirm with `-v` and read for
`--- PASS`, not `ok`.

**A NEW FIXED-RESOLUTION REAL-CHECKPOINT GOLDEN GETS WRITTEN GZIP-COMPRESSED.** A family whose
vision input has no small-test-image option (SigLIP: 896×896 for every image, unlike Qwen2.5-VL's
dynamic resolution) produces a golden dominated by one giant `pixel_values` float array — measured
2026-09-08, `gemma3_real_golden.json` was 52.72 MB uncompressed, over GitHub's 50 MB
recommendation; gzip took it to 2.58 MB (~20×, JSON float-array text compresses hard). Convention:
write it `.json.gz` (Python: `gzip.open(OUT, "wt")` instead of `open(OUT, "w")`), read it via
`decoder.ReadGoldenJSONForTest` (gunzips transparently on a `.gz` suffix, reads a bare `.json`
unchanged) rather than a raw `os.ReadFile`+`json.Unmarshal` (in-package tests that want the bytes use
the `readGolden` test helper, same rule). Every golden that was over 1 MB was migrated 2026-09-25
(24 files, 56.5 → 20.3 MB); small `.json` goldens stay as they are.

**THE SAME GOES FOR LARGE MEASUREMENT RAW DATA (ncu CSVs, sweep dumps): commit it `.gz`.** The eleven
`docs/measurements/` CSVs over 1 MB were compressed 2026-09-25 (64.2 → 0.8 MB; ncu output is extremely
repetitive). The originals are still in history, so that shrank a checkout, not a clone — which is why
anything over ~1 MB of text goes in as `.gz` from the start, and the record names the `.gz` path.

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

**A MINIMAL REPRO CAN BE MINIMAL IN EXACTLY THE DIMENSION THAT HIDES THE BUG.** Shrinking an
input to isolate a defect is right, but each thing you remove can also remove a defect's only
symptom — and the repro then reports GREEN on broken code, which is worse than not having run
it. Measured 2026-08-31 bringing up LFM2: a single-token prompt was used deliberately to
eliminate conv-window history, and it worked — it isolated a zeroed `NormEps`. But **softmax
over one element is 1.0 at any scale**, so that same repro certified an attention path whose
`AttnScale` was 0 and which returned a uniform average of the whole context. It took ≥2 tokens
to see at all. Before trusting a shrunk case, ask which effects are IDENTICALLY ZERO at that
size — n=1 kills every interaction term, any softmax/normalisation over the reduced axis
becomes the identity, and any per-position table (RoPE at pos 0) becomes trivial.

**PREFER DIFFERENCING PER LAYER OVER REASONING FROM FINAL LOGITS.** HF's
`output_hidden_states=True` gives a per-layer reference; comparing against it names the exact
first divergent layer in ONE run. In the same bring-up, guessing from the final logits had
already burned several hypotheses — the conv split order and tap indexing were both suspected
and both were correct, measuring cosine 1.00000000 / max|diff| 0.000000 the first time they
were tested directly. Two cautions: HF appends the last hidden state AFTER the final norm, so
the last row compares normed against un-normed unless you account for it (that artifact read as
a catastrophic 0.50 cosine and was NOT a bug); and a matching argmax means nothing — both LFM2
bugs held argmax while the logit cosine was 0.897.

**A STRUCT LITERAL BUILT BY HAND PER FAMILY WILL EVENTUALLY OMIT A FIELD, AND ZERO USUALLY
LOOKS LEGAL.** Both LFM2 bugs were that one shape (`NormEps` read from the wrong JSON key,
`AttnScale` simply absent). The fix that generalises is a check at the single chokepoint every
family passes through — `resolveArchitecture` → `validateResolved()` — not a per-adapter fix,
because the point is to cover the families nobody has written yet. Note the family validator may
fire first for any given family, so test the chokepoint guard DIRECTLY or its branch is never
exercised.

**A TIGHT REAL-CHECKPOINT BAR CAN CATCH A BUG IN THE REFERENCE, NOT JUST IN GOINFER — DO
NOT ASSUME THE DIVERGENCE IS YOURS.** internlm2's real-checkpoint promotion
(`internlm/internlm2_5-1_8b-chat`) first failed at cosine 0.87 against HF. The bug was in
`transformers`, not goinfer: `InternLM2RotaryEmbedding.__init__` computes `inv_freq` as a
`persistent=False` buffer, and the installed version's `from_pretrained` fast-init path
never re-runs that formula for buffers absent from the checkpoint's state dict — so
`inv_freq` came back as uninitialized memory (denormals, sometimes literal NaN) instead of
the real frequency table. Found by bisecting the REAL loaded `nn.Module`'s forward line by
line (same discipline as "prefer differencing per layer," above) until the NaN's first
appearance pinpointed `rotary_emb`; confirmed by patching the buffer post-load and matching
goinfer's own output to 7 significant figures. goinfer was correct the entire time this
gate reported red. **The lesson generalises beyond this one bug**: any
`trust_remote_code=True` model with a `persistent=False` buffer computed in `__init__`
(RoPE tables are the common case, not the only one) is exposed to this class of
`from_pretrained` fast-init defect — sanity-check such buffers (non-NaN, roughly the
expected magnitude) before trusting a reference disagreement as goinfer's bug. This is also
why the T3 real-checkpoint tier exists at all: no amount of code review or a tiny golden
would have found this, because the defect isn't in goinfer's code — the tight bar has to
actually run against real, independently-produced numbers before it can catch something it
doesn't control.

Heavy tests need `GOINFER_HEAVY_TESTS=1` and their assets; `testdata/assets.json` is the
registry and `go run ./cmd/gate` is the runner (`census`, `heavy`, `parity`, `composition`,
`selector`, `gpu`, `mutation`).

## Testing tiers: the smallest test that answers (owner rule, 2026-10-15)

**Every gate is registered at the lowest tier that can answer it.** Pre-registration applies at every tier.

- **Tier 0, seconds, the default for every change.** Test the mechanism with deterministic counters
  (`testing.AllocsPerRun`, bytes reserved or allocated, pass counts, KV rows written) and bit-identity on tiny
  fixtures.
- **Tier 1, by day.** One small real model at the smallest size that exercises the mechanism. State the expected wall time
  before starting.
- **Tier 2, by day by default.** Only for a number that will be published, or for a decision that hinges on a small effect.
  The registration says why tiers 0 and 1 cannot answer it, and gives a wall-clock budget. It goes to the night queue only
  under "Run budget"'s two night cases.

**Three rules apply at every tier:**

- **Never re-measure an unchanged arm.** Reuse its last readings and cite them (record and commit).
- **Stop as soon as the answer is clear.** Run a larger size only for a stated reason why the mechanism changes with
  scale.
- **Before any timed run or night job, fit-check every load it will make.** That takes seconds. Do not start or queue a
  job whose load would be refused.

**Never-swap gates read the test process, not only the system** (owner, 2026-10-15). The rule is on the process's own
peak resident memory and swap: `VmHWM` and the peak of `VmSwap` from `/proc/<pid>/status`, sampled until it exits. Run the
test binary directly (`go test -c`), because `go test` runs it as a child. System-wide swap in use is reported as context
only. A box already gigabytes into swap moves by hundreds of MB for reasons no test owns: KV2's CUDA heavy job read +511 MB system
swap on 2026-10-15, which only a per-process reading could attribute.

**On the CUDA box (nobara, 8 GB):**
- **A fit check is the server's own plan, not arithmetic.** Start the binary the job will run with its flags, and read the
  `fit:` and `KV plan:` banner, or the refusal. One `gv_plan.py` start is seconds to a minute; the 26B's expert cache and KV
  are sized against free VRAM at load.
- **Count a desktop compositor's CUDA context.** While anyone is logged in to the desktop, `kwin_wayland` holds one (about
  16 MiB). Plans move by that much, and the drain-group tests skip. Note it in the record; a GPU gate that needs the drain group
  runs with the desktop stopped.
- **Tier 1 GPU runs hold the timing lock** (`timing_lock.py run`), so two sessions' GPU checks do not share the card, and
  nothing else runs beside a timed job.
- **Count allocations as the driver does:** `cuMemAlloc` rounds a buffer of 2 MiB or more up to 2 MiB, so a byte counter
  checks the requested bytes and the plan prices the rounded ones.

Measured cost of not having this rule: on 2026-10-14/15 the G2-R night job was refused by the fit guard on two nights
running, and G40's memory and speed gates spent a night on sizes a tier 0 counter could have answered.

## Day loop: seconds, not minutes

**Owner rule, 2026-10-13: the day loop is seconds. A 10-minute gate after every edit is the development process failing, not diligence.**
Measured cost: on 2026-10-12/13 a CUDA-only feature ran `gate quick` twice (9–10 min each), the parity refresh three times (4 min each) and
the CUDA suite twice, and most of it re-checked what the previous run had already shown.

- **Check a change with `python3 scripts/touched.py`.** It gofmts the touched files, builds and vets the touched modules (the tagged variants they
  need) and tests the touched packages; in a package whose cold run is over a minute (`decoder`: 310 s) it runs the tests of the test file beside each
  touched file, or the `-run` you name. It also runs every source-scan test that reads a changed file, from any package or module and for any kind of
  file (`scripts/touched_scans.py`: a census, a switch's case labels, every route's guard; a `.cu` or a doc counts). A new test that reads the repo's
  own source goes in that table, or `scripts/test_touched.py` fails. It takes seconds. It is not a gate.
- **Do not run `go run ./cmd/gate quick`, a whole-module or whole-package `go test`, or the CUDA suite by day.** CI runs them on every push, and the heavy tier is
  its own run with a stated wall time (see "Run budget"). `gate quick` is the pre-tag tier (RELEASING.md): run it when cutting a release or when the owner asks.
- **One run of each thing a push requires, once, immediately before the push, after the last merge of `main`.** The parity refresh
  (`scripts/refresh_parity_hashes.sh`, only when the code of a non-test `decoder/*.go` file changed: the hash reads Go files as tokens, so a
  comment or formatting edit does not stale it) and the citation lint are the two; `main` moving under you is a reason
  to merge and re-run those two, not to re-run everything.
- **A red CI run is fixed by the one failing test**, run by name, not by re-running the gate that was green.
- **A change that forces a long run on the day loop is a modularity bug.** Say so in the report, with the edge that caused it (a CUDA edit that lands in
  `decoder/`, a one-line edit that stales the parity manifest), and fix the edge when it is cheap.

## Run budget: timed runs by day, night only for what cannot run by day

**Owner rule, 2026-10-19, replacing the 2026-09-28 "quick by day, long by night" rule.**
The old rule sent every timed run to the night queue, and the cost was days of waiting for answers that took minutes:
fit-refused night jobs, a queue nobody started, and results graded a day late.

- **Timed runs go by day by default, on both boxes.** They still go through the timing lock (a harness that takes it, or
  `python3 scripts/timing_lock.py run --label <name> -- <cmd>`) and, when queued, through `night.py`, as before.
- **State the expected wall time before starting** (cells × runs × per-cell time, plus idle-gate waits), as a start time
  and an expected finish time. This replaces the old ~10-minute day cap.
- **Night is only for two cases:**
  - (a) a Mac job the fit guard refuses while the owner's apps are open;
  - (b) a job over about 2 hours.
- **Comparisons are interleaved:** A B B A, in one session, on one box.
- **Every cell records, at its start, the load average and the free memory** of the box doing the timed work.
- **A cell started under heavy load is VOID, not averaged.** Heavy load means either of these, on the box doing the timed
  work, at the cell's start:
  - the 1-minute load average is above **half its core count**: 4.0 on the Mac (8 cores), 8.0 on nobara (16 cores);
  - its free memory is **below what the fit guard needs for the cell's load, plus 2 GB**. The free-memory term is goinfer's reclaimable reading (`cmd/hostmem -reclaimable`; macOS free + file-backed + purgeable pages, speculative counted once; Linux `MemAvailable`): what a cell can take without macOS compressing or swapping other memory under it. **Changed 2026-10-21 (Francis, B41, registered before any data):** that reading, for jobs queued after B41 merged that day; before that, macOS read free + inactive + speculative + purgeable pages. The fit guard decides loads by the separate available reading, `memory_pressure`'s free percentage × RAM − 500 MiB (Francis, 2026-10-24), which counts compressible app memory and so is never the void rule's term. The 2 GB margin is unchanged.

  A void cell is re-run in the same interleaved order. It is never averaged in, and it is never quietly dropped: the record
  lists it. A box that only drives a harness against the other box (tack's runs from the Mac against nobara, say)
  records its own load but never voids a cell. The same definition is in `docs/measurements/noise-registry.md` §8.
- **Night jobs:**

  ```sh
  python3 scripts/night.py add <name> --est <minutes> --by "<who>, <campaign>" --doc <task doc> -- bash docs/measurements/<campaign>/run-<x>.sh
  ```

  A night case adds `--not-before 22:00`. A job whose own fit check can refuse exits 75 on that refusal and is queued
  with `--retry-until HH:MM`; `--fit-need-gb` gives the memory the heavy-load gate checks. Tell the owner what you queued
  and the total (`add` prints it). The queue is per machine: queue a job on the box that has to measure it.
  - **A box running `night.py serve` runs its queue itself** (B38; `scripts/install-night-agent.sh` installs it as a
    launchd agent or a systemd user service). Each job starts when its gates allow: its `--not-before` hour, the
    heavy-load rule above, and the timing lock free through a settle that holds the lock. A job that exits 75 is
    re-queued until its `--retry-until`; a night job whose night ends without a start moves to the next night. **Do not
    start a second runner, wait on it, or poll it.** `night.py status` says `running <job>`, `settling for <job>` or `idle`;
    `pause` / `resume` are the owner's.
  - **A box without it:** do not start the queue; the owner does (`night.py start`).
- **One copy of each run.** `night.py list` shows every queued command; check it before adding. If a queued job
  already answers your question (the same gate or suite on tonight's tree, or a superset of it), do not queue another
  — name that job in your report to the owner. `add` refuses an identical command from the same directory outright.
  By day the same goes for a check that already passed at this commit: cite its log, do not re-run it. A repeat
  that is part of a pre-registered design (the order-reversed second pass, an A/A control, a reproduce-on-another-day
  run) is not a duplicate: queue it with `--allow-duplicate`, and pre-register a skip condition for it where one
  fits, as the L1 served gate's 2026-09-28 amendment did (pass 2 skipped when pass 1 clears the bar by more than the
  two passes' own measured spread).
- **A night job must run with nobody watching.** Use the `docs/measurements/<campaign>/run-*.sh` shape: pinned revs
  or pre-built binaries (the tree may move before tonight), durable log/record paths, no prompts, nothing that
  needs a Claude session alive. Its output also lands in `~/goinfer-logs/night/runs/<date>/<name>.log`. `night.py` marks a job FAILED when
  its own output carries a failure marker (`--- FAIL`, a `FAIL` line, `FAILED`, `VOID`) even if it exits 0; a line a job prints on
  purpose, such as a plant's red run, is tagged `[expected]`. Still exit non-zero on any failure.
- **One timed run per box** (TE9, `docs/completed/task-test-efficiency-2026-09.md`).
  - The `bench_peer*.py` harnesses and every `night.py` job hold `~/.goinfer-timing.lock` for their whole run
    (`scripts/timing_lock.py`). A second timed run refuses, naming the holder, and the holder's children inherit it.
  - A `run-*.sh` whose timed step is not one of those harnesses (a `go test` measurement gate, a sweep loop) wraps it:
    `python3 scripts/timing_lock.py run --label <name> -- <cmd>`.
  - A refusal means another run is measuring on this box. Wait for it (`timing_lock.py status` names it); never delete
    the lock file to get past it: its holder keeps its flock on the deleted file, and the next run takes a new one beside it.
  - **Work that only adds load holds it shared:** `timing_lock.py run --shared -- <cmd>` (`acquire(shared=True)` in Python) for
    a gate, a test run, or a screen that times another box. Loaders run together. A timed run (exclusive, the default) waits
    for every loader and refuses naming them, and no loader starts under a timed run.
  - **The day tools hold it shared.** `touched.py`, `refresh_parity_hashes.sh` and `gate quick` hold the lock shared while
    they run, so a timed run waits for them, and they refuse while a timed run holds it, naming the holder (exit 75): a
    build or test run beside it contaminates its timings. `bench_compare.sh` takes the lock
    itself. `--wait SECONDS` waits for the holder, and `--ignore-timing-lock` overrides, for emergencies only, and the
    overlap then goes in that run's record. Any other build or test you run by hand holds it shared the same way:
    `python3 scripts/timing_lock.py run --shared --label <what> -- <cmd>`.
  - **A command that touches the other box checks that box's lock in the same command, and ends if it is held:**
    `ssh <box> 'cd <repo> && python3 scripts/timing_lock.py status' && ssh <box> '<the work>'` — one invocation, not a
    check in one call and the work in the next. An ssh session takes no lock of its own, so nothing else stops it: on
    2026-10-13 the Mac's session read a 2.78 GB checkpoint on nobara in the middle of a timed I3 cell
    (`docs/measurements/integration-harness-2026-10/i3-night-2026-10-13.md`).
- **Two arms when the decision is new ÷ old** (TE5(a)). A gate that grades a goinfer-vs-goinfer ratio runs
  `BENCH_ENGINES=goinfer,goinfer_old`, with no peer arm. The peer ratio is read once, after the lever ships, as its own
  same-session interleaved run. In the 2026-09-28 L1 gates Ollama took a quarter to a third of the cell time and
  decided nothing.
- **A kernel speed question starts in-process, by day** (TE5(b), owner decision 2026-09-28, the resolved-only reading).
  - Run a whole-token in-process A/B, interleaved in one process, first. When it resolves a direction (its interval or
    pair signs clear 1), that direction may be acted on by day. Over 142 examined changes, every resolved kernel-local
    pair agreed in sign with its served A/B (37/37 same-session, 45/45 overall).
  - An unresolved in-process result goes to the served gate.
  - The served gate stays the instrument for any end-to-end or peer claim.
  - A kernel-only microbenchmark gives direction, not size: its served effect ranged 0.05–1.72× of it.
  - `docs/measurements/test-efficiency-2026-09/te5b-concordance-2026-09-28.md` has the table.
- **Timed runs use the harness defaults** (on the Mac the instant idle gate since 2026-10-02, TE1; on Linux
  `BENCH_MAX_LOADAVG=1.0`), by day as at night. The void rule above applies on top of them, never instead of them.
- **Jobs of 3 h or less; never 6.** `add` refuses an estimate over 360 min. Shrink the sample or fix the slow path
  first, as D6a did. The runner does not START a night job whose estimate would end after 06:30, so an over-full queue
  carries to the next night instead of running into the owner's morning.
- **Pre-registration is unchanged.** The rule and its bands are written and committed before the run, even when
  the run is twelve hours away.
- **Morning:** `python3 scripts/night.py morning` prints the latest `SUMMARY.md` (who queued each job, its doc,
  outcome, and the tail of anything that failed). Grade against the pre-registration and commit the records. A job
  that failed or timed out gets fixed, then re-run by day, unless it is one of the two night cases.
- **The owner can override.** "Run it now" means run it now.

## Gate tiers (TE11)

Every gate or check names its tier, its instrument, its stopping rule and where its cost estimate comes from. The pre-registration template (`docs/measurements/noise-registry.md` §8) asks for the same
five. The release tier (`RELEASING.md`: the full parity sweep and the peer matrix) is unchanged.

| check | tier | instrument | stopping rule | cost basis |
|---|---|---|---|---|
| `python3 scripts/touched.py` (gofmt, build, vet, the touched packages' tests) | day | the script: what the diff touches, a slow package by the test file beside the change | runs to completion | seconds |
| `go vet` (tagged), `staticcheck` across a module, `go run ./cmd/gate quick` | pre-push / pre-tag: CI runs it on every push; locally only when cutting a release or when asked (see "Day loop") | `gate quick` (TE7): the tests that can observe the diff, one `go test` per package in parallel, cache-aware. It defers, and prints as `DEFERRED`, each package's real-checkpoint tests and, when no changed file is under `metal/`, `gpu/` or `cuda/`, that module's long kernel tests (`quickModule.Slow`) | runs to completion; no sampling | about 2 min for a decoder change on the Mac (2026-10-13: the `metal` and `gpu` device cells, run one at a time, are the floor; the `decoder` cell is about 30 s since the sampler moved to `internal/sampling`), about 4 for one under `metal/`; an unchanged `decoder` re-run replays from the test cache in about 2 s |
| `go run ./cmd/gate quick -full -base <previous tag>` | day / pre-tag | the same runner with nothing deferred; writes a record under `docs/measurements/quick-full/` that `gate release` requires from each box (`RELEASING.md` step 8) | complete | about 10 min on the Mac (measured 2026-10-13: the Metal suite 4m12s, the real-checkpoint cells 3m44s) |
| `go test -run '<yours>'` on one package; tiny-fixture goldens | quick | the test itself | fixed | seconds to minutes |
| `scripts/refresh_parity_hashes.sh` | day | the goldens-gated refresh | completes | about 4 min |
| `go run ./cmd/gate identity <old> <new>` | day | byte identity of full logits at the last validated build and this one, per family (TE6(b)): validation is inherited by identity | all families, all steps | families x steps x per-prompt time; state it first |
| a kernel speed question | day | a whole-token in-process A/B, interleaved in one process; act on it only when its interval or pair signs resolve 1 (TE5(b)) | stops at a resolved direction; an unresolved one goes to the served gate | tens of seconds to minutes |
| a served or peer gate (`bench_peer*.py`) | day (night only for the two night cases) | same-session interleaved cells, A B B A, with a cell started under heavy load void (see "Run budget"); two arms for a new/old ratio (TE5(a)); one timed run per box (the timing lock, TE9) | fixed N, pre-registered, plus the registered skip of the order-reversed pass 2 only when pass 1 clears the bar by more than the two passes' own spread. TE4's sequential stopping rules did not survive their own tests (SEQ-v1 killed on order sensitivity, SEQ-v2 failed its screen), so no ad hoc early stop | cells x runs x per-cell time plus idle-gate waits (about 45% of cell wall is decode being timed; see the census) |
| a prefill/decode fidelity gate | day (night only for the two night cases) | `internal/fidelity` margin-required non-inferiority, N from `scripts/power.py fidelity` (TE12) | a fixed, hashed prompt set; no "wins in half" rule | from the previous run's positions log |
| the two env-gated correctness gates (draft≠target speculative parity, expert-paging bit-exactness): `docs/measurements/b13-dark-gates-2026-10-06/run-dark-gates.sh` | day | each `go test -run` in a detached worktree of `origin/main`, a SKIP counted as a failure (B13) | both run; each has its own `-timeout` | 62 s warm, about 6 min cold (the 35B giw loads twice), measured 2026-10-06 |
| a `go test` measurement gate | day (night only for the two night cases) | resumable and content-keyed where it is a reference (`TestPrefillGateReference`, TE6(a)); a timed A/B is deliberately not resumed | resume skips finished cells only | the gate's own prior wall, from the record |
| `go run ./cmd/gate parity`, `gate gpu`, the heavy tier | day / release (night when over about 2 hours) | the named-gate checksets | complete | from the census and the task doc; state the wall time before starting |

The measurement standards in this file did not change; these designs make them cost less. The before-and-after re-count of where session time goes is in `docs/measurements/test-efficiency-2026-09.md` §6.

## Long-running work

A run over about 2 hours goes on the night queue (above), and `night.py` detaches its jobs itself.
Anything over a few minutes that you do launch must be **detached** — `setsid nohup … </dev/null &` — because a
plain background shell dies at session boundaries. Verify it took: `ps -o pid,ppid,sid` should
show **PPID 1**.

**macOS has no `setsid`.** Use `launchctl submit -l <label> -- /bin/bash -lc "cd <repo> && … >
<log> 2>&1"` instead (macOS `ps` also lacks the `sid` column — check `ppid` = 1 instead).
**A `launchctl submit` job restarts itself on exit** — undocumented, found the hard way 2026-09-05:
a completed run's log was silently overwritten by a fresh, immediately-failing re-invocation of
the same command within moments of the original process exiting, before it could be read, and the
real run's per-prompt trace was gone for good (`docs/measurements/prefill-gate-l1-2026-09-05.md`
records what that cost). Fix: append `; launchctl remove <label>` to the submitted command so the
job un-registers itself as its own last action, closing the race before launchd can restart it.

**Archive the log; do not leave it in `/tmp`.** `/tmp` gets cleared, and a verdict nobody can
re-read is not evidence. Write it somewhere durable and reference the path in the writeup.

Prefer committed increments that survive interruption over one big commit at the end.

## Citations and the pre-push hook

`scripts/queue_citation_lint.py` runs **as a pre-push hook and REFUSES the push** on a red. Its generated index
of commit and bare-file citations lives in `docs/citation-index.md` (moved out of `docs/QUEUE.md` 2026-09-24). Four
traps worth knowing before you hit them:

- A backtick-quoted concrete path under a **gitignored** directory (`docs/internal/…`) is a
  FORBIDDEN DESTINATION — *including in another repo*, since nobody else can resolve it.
  Describe the record in prose instead; that is what `c494c62` did.
- **Code is cited by declaration, never by line.** A live doc writes `decoder/arch_voxtral.go:voxtralArchitecture` or
  `chat/chat.go:Template.Render` (`Type.Method` for a method; a const or var in a block by its own name), and the lint checks the
  file declares it today. A rename or a deletion is red, with the message to name what it became; a function that merely moved
  within its file is green, and nothing has to be renumbered. A **dated record** (an audit, a review, a measurement write-up) is
  instead **pinned**: `<!-- citations-at: <commit> -->` in its first lines, and its `path:line` numbers are checked against that
  commit only (the file, the line, a range's end) — never against HEAD, never rewritten. `path:line` in an unpinned doc is red:
  "pin this doc (`citations-at`) or name the declaration". Pin to the commit the numbers were TRUE at, which is not always the one
  the header names (a baseline); `docs/measurements/code-comments-2026-10/pins.tsv` shows how that was measured for the first 35.
  A shell script has no declaration form: cite it by name, or by line in a pinned record.
- **A missing MODULE reads as a broken lint but is a missing download.** The lint validates
  citations that point into `aikit`, and treats an unresolvable module as a hard failure on
  purpose — calling it a skip "would make the green cover nothing". So when `go.mod`'s pinned
  aikit is not in `~/go/pkg/mod` it refuses the push with `CANNOT SEARCH — no module cache and
  no checkout for: github.com/townsendmerino/aikit`. Fix with `go mod download`, **not**
  `--no-verify` and not `--update`. A record pinned to an older commit needs the aikit version THAT
  commit required: `python3 scripts/queue_citation_lint.py --pinned-modules | xargs go mod download`
  fetches them all (CI runs the same list). It fires after every aikit bump you have not built since,
  and on every fresh `git worktree` (which has no `go.work`). Measured 2026-08-31: a docs-only
  push refused because `origin/main` had moved v1.30.0 → v1.31.0.
- **Read its exit code DIRECTLY, never through a pipe.** `python3 scripts/queue_citation_lint.py
  | tail -6; echo $?` reports **tail's** status, so a red lint looks green — the script prints a
  long green-scope epilogue either way, which is exactly what makes the misread easy. Redirect to
  a file and check `$?`, or the hook will be the first thing that tells you.

`git push --no-verify` is never the answer (owner rule, 2026-10-18): a lint or hook you believe is wrong is reported to
Francis, not skipped.

<!-- citation-lint: allow-path file.go the placeholder in this repo's own descriptions of the citation forms (file.go:NNN, file.go:NN), not a citation -->
<!-- citation-lint: allow-path path.go the placeholder in the symbol-citation examples (path.go:Name, path.go:Type.Method), not a citation -->

## Task docs: the ledger, and closing as you go

**Owner rule, 2026-10-21.** The close-out sweep of the eight most recently worked task docs found every status line days or
weeks behind the tree, each item's state scattered through prose, and items moved to a queue entry or another doc with no
link back. So:

- **A live task doc carries one ledger,** under its status block: the line `<!-- ledger -->`, then a table
  `| id | state | next | evidence |`, one row per item. States: open, active, blocked, decision (waiting on the owner's
  word), done, killed, moved (the row names where). It is the current state; prose notes below it are history.
- **Change the row in the same commit as the work.** A commit that ships, kills, blocks or moves an item updates its row. A
  move also leaves a link back in the destination, naming the doc it came from.
- **The commit that closes the last row archives the doc, in the same push:** `python3 scripts/task_docs.py archive <doc>
  --status <file>` prints the plan; read it, then add `--apply --stage`. It copies the ARCHIVED header verbatim, moves the
  doc to `docs/completed/`, re-points the references in docs, keeps a pointer stub when code, help text or a generated page
  cites the old path (so the push stays docs-only and no `_fresh` test goes red), recounts `docs/README.md` and runs the
  citation lint. The status block is yours to write: what closed it, with commits; what the tree says now; what the doc did
  not own and where it went.
- **A new task doc starts with a ledger,** and commit subjects name its items with an ID the doc owns (`KV4`, `MEM12`, or
  the doc's slug with a short ID). A bare `S4` or `G1` matches several campaigns.
- `python3 scripts/task_docs.py rank --top 8` lists the most recently worked docs on any ref; `check` lists closable docs,
  uncommitted docs in any worktree on the box, archives without the header, and drifted README counts. Several docs at
  once is a close-out sweep (the owner's `close-out` skill), which runs `.claude/skills/doc-review/SKILL.md` on each doc in
  parallel.

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
- **A bit-identical speed change ships even under its registered bar** when all four hold: (1) its outputs are bit-identical;
  (2) its scratch and allocated memory are not higher; (3) it is faster on at least 9 of 10 timed pairs; (4) it replaces code
  rather than adding a second path (a test-only switch back to the old form is not a second path). The registered bar still
  decides any change that alters outputs. Owner's rule, 2026-10-23.
- **A lossless opt-in feature merges when it is consistently faster:** median ≥ 1.05× with at least 9 of 10 timed pairs faster.
  Lossless means bit-identical, or inside a registered fidelity gate. Its PR states the memory cost: device and host bytes, and any
  effect on the longest context that fits. **Changing a default needs more:** the gain judged against its memory trade-off, and
  Francis's call. The rule above still covers a bit-identical replacement that costs no memory. **An investment bar is not a merge
  bar:** a projection that decides whether a track is worth building (SP0's ≥ 1.3×) does not carry over to the merge gate of what
  the track produces. Owner's rule, 2026-10-25 ("even 1.05 should be a merge … unless there's some other issue").
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
  measured 2026-08-28, `~1.2-1.4 tok/s` was withdrawn in `docs/completed/task-zeno-compare.md` and went on
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
