# Task: the first hour — what a cold user found, what was fixed, and how to run it again

> **Status: R1–R4 SHIPPED 2026-09-06** against the run recorded in
> [`measurements/cold-user-2026-09-06.md`](measurements/cold-user-2026-09-06.md) (M1 Pro / 16 GB,
> v0.16.0, five scenarios, 13.5 min). Every fix below carries a gate that goes red on v0.16.0;
> the mutation used to prove each one is named. R5 is the protocol, which is this document's §1
> and does not "ship" — it is run.
>
> **Batch 2 — R6–R12 IMPLEMENTED 2026-09-07, uncommitted this pass**, against the run recorded
> in [`measurements/cold-user-2026-09-06-nobara-pc.md`](measurements/cold-user-2026-09-06-nobara-pc.md)
> (nobara-pc, Ryzen 3700X + RTX 2070 SUPER 8 GB, v0.17.0, five scenarios, ~32 min — this is run 2
> of the protocol §1 called for). Every fix below carries a gate that goes red on v0.17.0 and was
> mutation-checked against the actual pre-fix behavior (not merely reasoned about); several were
> additionally verified end to end against real hardware and real checkpoints, not only unit
> fixtures — noted per finding. R7's registry addition substitutes a real, resolvable checkpoint
> (gpt-oss-20b) for the one the run's own author specified (Gemma-4-26B-A4B), because no
> verifiable download exists anywhere in this tree for that model — see R7 for why fabricating
> one was refused rather than worked around.
>
> Sibling docs, neither superseded: [`task-embed-and-harness-ux.md`](task-embed-and-harness-ux.md)
> owns the facade and the harness recipes (§4 below scores its predictions), and
> [`task-fit-to-hardware.md`](task-fit-to-hardware.md) owns "will it fit" — R3 implements that
> doc's **Phase 0 and nothing else**.

**The premise.** Everything this repo measures well, it measures from the inside. The first hour is
the one interval no gate covers, because every gate is run by someone who already knows where
things are. So it is measured the only way it can be: by handing the published tag to somebody
with no access to the tree and reading what they hit.

The run cost 13.5 minutes and found three defects that had survived an audit, a release, and a
CI suite — including a released Mac binary that could not use the GPU, which had been silently
converted into an *engine* comparison against Ollama.

---

## 1. The protocol — how to run this again

**Who may run it.** A window with no access to the source tree, `docs/`, or git history, working in
an empty directory outside the repo, installing the **published tag** the way a stranger would.
Prior contact with the project is not disqualifying but **must be declared in the report**: the
2026-09-06 tester had edited the README in earlier sessions and said so in the first paragraph,
and named the three findings they were most confident were uncontaminated (D1, B1, E2 — each
discovered by failing, not by knowing). The next run uses a tester who has not.

**Allowed sources.** The GitHub README, the release API and its assets, `--help`, error text, and
pkg.go.dev. Nothing else. Note that pkg.go.dev renders doc comments lifted from source, so it *is*
the source's comments — the 2026-09-06 run flagged this itself rather than letting it pass.

**The five scenarios**, each time-boxed at **25 minutes**:

| | scenario | the question |
|---|---|---|
| A | Try it | from nothing to an answer |
| B | Point my tools at it | an OpenAI-speaking client, and an agent CLI |
| C | Embed it | a ≤40-line Go program that prints a completion |
| D | Run bigger than my hardware | a model that does not fit |
| E | Control | the same question through a peer tool, recorded to the same standard |

**Rules.**

1. Two attempts per obstacle, then stop and record a dead end. A dead end is a finding, not a
   failure of the run.
2. **The swap-safety rule: stop at the first pageouts.** Scenario D obeyed it and the run survived;
   without it the machine does, and there is no report.
3. Record a friction log with **timestamps**, tagged `Guessed` / `Wanted and absent` / `Error`,
   plus a numbers table per scenario. The timestamps are what make "half the time if…" checkable.
4. Record `--version` output for every binary used. (This exists *because* of R2; the 2026-09-06
   run could not, which is how the Metal gap survived to the release.)
5. The time box is a **stop rule, not a target**. Nobody came near it in 2026-09-06 — the longest
   scenario was D at 3.5 minutes — and the box stays at 25 anyway.
6. Deliverable: one Markdown file into `docs/measurements/cold-user-<date>.md`, **verbatim**,
   including its contamination note. It is a measurement record; it is not edited to look better
   afterwards, and a later fix does not rewrite it.

**Per-scenario assertions the next run must record**, so a repeat measures the fixes rather than
re-discovering the findings:

- **Scenario D — "the tool told me before the machine did."** The run passes this leg only if a
  refusal or a budget warning appeared BEFORE any swap growth. On 2026-09-06 the machine told the
  tester and the tool never did; R3's guard exists precisely to invert that, and a cold run is the
  only place it gets tested the way a user meets it.
- **Every scenario — record `--version` for each binary used** (R2). The 2026-09-06 run could not,
  which is how a Mac asset with no Metal in it reached a release.
- **Scenario A — the cold-start total**, which is the number goinfer wins and worth tracking.

**Scenario D stays one class above the brief.** The 2026-09-06 run substituted a 35B-A3B on 16 GB
for the brief's "20–30B", because the README named no MoE and the tester would not download 15–20
GB onto a disk with 23 GB free. That is a *harder* test, and it is the one that found the real
thing. Keep it there.

**Scenario B needs an agent CLI installed.** The 2026-09-06 run could not test one: aider,
opencode, cline, `llm` and codex were all absent, and `continue` matched a shell builtin — a false
positive worth knowing about.

**`nobara-pc`: `opencode` 1.18.29**, installed into a contained prefix
(`npm install --prefix ~/.local/opt/opencode opencode-ai`; the binary is at
`~/.local/opt/opencode/node_modules/.bin/opencode`). A contained prefix rather than a global
install because npm's global prefix here is `/usr/local` and needs root; nothing on `PATH` and
nothing in `npm config` was changed, so the next tester must use the full path or add it
themselves — which is a step to note in their report, not a hidden one.

**`aider` was tried first and does not install on this box:** the system Python is 3.14.7, and
pip only offers `aider-chat` up to **0.16.0** there — every current version declares an upper
Python bound that excludes 3.14, and pip backtracks to a 2023 release whose pinned `multidict`
6.0.4 then fails to build. This is a fact about aider and 3.14, not about goinfer; if the next
run wants aider specifically it needs an older interpreter.

**The MacBook is still unprovisioned** — it cannot be reached from `nobara-pc`, so whoever runs
scenario B there installs one first and names it.

**`macbook-arm64`: `opencode` 1.18.29 too**, installed 2026-09-06 the same way
(`npm install --prefix ~/.local/opt/opencode opencode-ai`; binary at
`~/.local/opt/opencode/node_modules/.bin/opencode`), so both machines now have scenario B
provisioned identically — same tool, same version, same contained-prefix path. Same caveat
applies: nothing on `PATH`, use the full path or add it in the run's own report.

**Next run:** the **Linux box**, after the R1–R3 tags land. The CUDA path makes "bigger than my
hardware" a different story (expert streaming vs `-stream-weights`), so it is not a repeat. Its
results go in as run 2 and §4's table gains a column.

### Protocol amendments, after run 2

Run 2 found gaps in the protocol itself, not only in the product. Recorded here as amendments
rather than silently fixed, per this document's own rule that R5 (the protocol) is run, not
shipped — a protocol that changes without a record of why is not reproducible either.

- **The tester window is created with NO working directories granted.** Run 2's session had
  `goinfer/{chat,decoder,multimodal,cmd/serve,cuda,metal,internal/chatapp,internal/gemmaapp,
  scripts,docs/measurements}` pre-granted as "additional working directories" — package-level
  structural knowledge (backends, internal app layout, a docs/measurements directory even
  existing) that a genuinely first-contact window would not carry. The tester declared this
  before starting and the run proceeded anyway, on explicit instruction, with none of those
  directories opened — but a window that finds repo paths pre-granted at all is evidence of
  prior contact by construction, and the correct response is to declare it and count the run
  contaminated, not to proceed carefully around it. Whoever provisions the next tester window
  must confirm zero working-directory grants into this tree before the run starts.
- **"Install the latest release" — the tag is recorded from the release page, never given in the
  prompt.** Run 2 was told to install v0.16.1; `GET
  /repos/townsendmerino/goinfer/releases/tags/v0.16.1` returned 404 — that release never existed.
  A prompt-supplied tag can be stale, mistyped, or (as here) simply wrong, and the tester has no
  way to distinguish "this tag doesn't exist" from "something about my install is broken" without
  checking the release page directly — which is what the README's own install instructions tell a
  real user to do anyway. The tag a run installs should always be discovered from the release
  page at run time, never dictated in advance.
- **Scenario A records download and run as two legs**, because they have different bottlenecks
  and a single combined number hides which one dominates. Run 2's clean-path total (56.5 s) was
  ~98% download time (55.3 s for 1.71 GiB at this run's ~31.5 MB/s) and ~2% launch-to-first-token
  (1.22 s) — a number that would read as "goinfer is slow to start" is actually "this network was
  slow," and only splitting the legs makes that visible. **Scenario E starts Ollama with an empty
  `OLLAMA_MODELS`** for the same reason from the peer side: run 2's Ollama already had 8 models
  cached from unrelated prior work on the same machine, including the exact model class needed,
  so its "cold start" leg measured a warm cache and had to be flagged as non-comparable rather
  than fixed — recording its download leg the same way scenario A does is what makes the two
  sides actually comparable next time, rather than caveated after the fact.
- **Provision the `openai` Python package alongside opencode on both boxes.** Scenario B's
  protocol asks for three clients in order — `curl`, the `openai` Python package, then an agent
  CLI — and run 2's box had the CLI provisioned (per the existing note below) but not the Python
  package, so that leg was skipped rather than run. Symmetric with opencode's own contained-prefix
  provisioning note: install it ahead of the run, not discovered as missing during it.
- **Rule 5 (write only in this directory and `~/models`) permits the tool's own scratchpad.** Run
  2's tester kept working notes in the harness-provided session scratchpad (outside both the run
  directory and `~/models`) before assembling the final report, and flagged it as a rule breach
  because the rule as written does not carve out that space. It should: a scratchpad the tooling
  itself provides and manages is not writing into the project tree or leaving artifacts behind,
  which is what the rule exists to prevent, and forbidding it only pushes a tester toward keeping
  notes in-memory (worse for a long run) or writing them into the one directory the rule is
  actually protecting.

---

## 2. Findings, fixes and gates

Ordered by user impact, which is the order they were fixed in.

### R1 — the README described a binary the release did not ship

**Found** (finding #2, Critical). The README ran `goinfer-serve` three times, including the
headline browser-UI example. v0.16.0 shipped **21 assets and none contained the string "serve"**.
The only documented install — `go get github.com/townsendmerino/goinfer` — is a library fetch of a
module with no root package and produced four `missing go.sum entry` errors when built against.
The tester reached a server only by probing pkg.go.dev for HTTP 200 (`cmd/goinfer-serve` → 404,
`serve` → 404, `cmd/serve` → **200**).

**Fixed.** The release workflow now cross-compiles `goinfer-serve` per platform from the backend's
own entrypoint. The README's Install section moved to the top and gives commands that work from an
empty directory, with the library case spelled out — `go get <module>` is *not* enough, and the
per-package `go get` that is, is shown. Asset sizes corrected (8.3 MB / 652 MB / 1.81 GB against
the claimed ~5 MB / ~615 MB / ~1.7 GB).

**And the question the brief asked, answered by running it rather than reasoning: the tag's
module graph was CLEAN. The command was wrong.** Reproduced against the unmodified `v0.16.0` tag
in an empty module:

```
$ go get github.com/townsendmerino/goinfer@v0.16.0     # the README's command
go: added github.com/townsendmerino/goinfer v0.16.0
$ go build ./...
missing go.sum entry for module providing package github.com/townsendmerino/aikit/embed …
    to add: go get github.com/townsendmerino/goinfer/decoder@v0.16.0

$ go get github.com/townsendmerino/goinfer/decoder@v0.16.0   # the PACKAGE, not the module
$ go build ./...                                              # rc=0
```

The mechanism is Go 1.17+ **module graph pruning**: `go get <module>` records the requirement and
only the sums needed to resolve the graph — not the sums needed to *build* packages you have not
named. goinfer's root module has no root package, so that command yields a `require` line and
nothing buildable. Go's own error even prints the fix.

So `RELEASING.md`'s B-01…B-05 module-graph invariants are **not implicated**: nothing was wrong
with what was published, and re-tagging would have fixed nothing. The defect was one line of
README, which is the cheaper thing to have been wrong — but only checkable by running it, which is
why the brief asked.

**Gate — `readme-smoke` (CI).** From an **empty temp directory outside the checkout** with
`GOWORK=off` and no clone, it runs every fenced command the README marks `<!-- smoke -->` and
fails on non-zero, then **builds a trivial importer against what was installed**.

> The build step is the gate, and the first version of this job was **theatre without it**:
> `go get github.com/townsendmerino/goinfer` **exits 0**. The broken install the tester hit only
> fails at build time, so a job that checked exit codes alone passed on the exact README that
> produced the finding. Mutation-checked afterwards: restoring v0.16.0's install line turns the
> job red with the tester's own error text.

**Commit:** `e57fef11`.

### R2 — the shipped Mac binary ran on CPU and said it was on Metal

**Found** (finding #3). Two consecutive lines: `decoder: metal backend not built in … using cpu`,
then `loaded 28-layer model … [backend=metal quant=int4]`. The warning scrolls; the status line is
what gets screenshotted.

**Verified before fixing**, as the brief required. The darwin-arm64 asset was built from the
**root** `cmd/serve`, which since v0.10.0 (audit M-19) imports no backend at all. So this was not
only a banner defect: **the release shipped a Mac binary that could not use the GPU**, and scenario
E's "Ollama is 2.2× faster on decode and 3.5× on load" is a measurement of that packaging gap, not
of the engine. Those numbers stand as recorded and are **not** carried into `benchmarks.md`.

**Fixed, in three parts.**

1. *Packaging.* darwin assets build from `metal/cmd/serve`, linux from `cuda/cmd/serve` (both
   cgo-free, so both belong in a static asset); `-tags gpu` stays out because WebGPU needs cgo.
2. *The banner.* `decoder.BackendReport()` renders the **effective** backend, and when it differs
   from the request it reads `requested metal → running on cpu: <reason>` on one line.
   `[backend=…]` can no longer name a backend that is not executing. All four `Model` construction
   sites record the requested/effective split, so a load path added later cannot skip it.
3. *`--version`.* `goinfer-serve --version` prints the version, the **backends compiled into this
   binary**, and the Go toolchain — answerable without a model, which is the question the cold run
   had no way to ask.

**Gates — three, because the three parts fail separately.**

- `decoder.TestBackendReport_namesTheEffectiveBackend` — constructs a declined backend and asserts
  the banner names what is executing plus the reason.
- `decoder.TestBackendBanner_usesTheReport` — walks every `.go` file for a site formatting
  `[backend=%s` and requires `BackendReport()` in its arguments. **This is the half that goes red
  on v0.16.0**: a correct report that no banner calls is precisely the state the cold run found.
  Mutation-checked by restoring the request-printing form in each of the three banners; all three
  are flagged, individually.
- `serveapp.TestVersionReport_backendsLineIsDerived` proves the `backends:` line comes from the
  registry (registering a backend must change it — a hard-coded list would satisfy a text check
  while being the same lie), and `TestServeVersionFlag_reportsOnlyCPUForTheRootBinary` builds and
  runs the real root binary. Mutation-checked by deleting the `--version` handling: the binary
  test goes red with v0.16.0's `flag provided but not defined: -version`.
- Release workflow: each asset's **main module** is asserted from `go version -m` — the darwin
  assets must report `mod …/goinfer/metal`, the linux ones `…/goinfer/cuda`. The main module, not
  a `-tags` grep: **nothing in `metal/` is gated on a `metal` build tag** (every file is
  darwin-gated), so a tags check there would pass on a binary with no Metal in it. The one asset
  the runner can execute (linux-amd64) additionally has its `--version` output grepped for `cuda`.

**Commits:** `e57fef11` (packaging, R2a) · `6b731976` (banner, `--version`, asset assertions).

### R2-follow-on — every release's Mac and Linux binaries were the PREVIOUS release's engine

**Not a cold-run finding. R2's own gate found it**, hours later, which is the argument for the
gate.

Replaying the release workflow's build locally produced a `goinfer-serve` **without the
`--version` flag that had just been added to `internal/serveapp`** — so the new assertion went
red. The cause is structural, and it is not new:

- `release-assets.yml` fires on the **root tag**, and builds the GPU assets from
  `cuda/cmd/serve` and `metal/cmd/serve`.
- At that instant `cuda/go.mod` and `metal/go.mod` still require the **previous** root release —
  `RELEASING.md`'s two-step tag bumps them *afterwards*, deliberately.
- `go.work` is gitignored, so a bare submodule build in CI resolves
  `github.com/townsendmerino/goinfer` from the **proxy**, at that stale version.

**Measured, not inferred.** A `GOWORK=off` submodule build of exactly the workflow's shape
produced a binary reporting:

```
mod  github.com/townsendmerino/goinfer/cuda  v0.16.1-0.20260906162554-d3e7237d1cde
dep  github.com/townsendmerino/goinfer       v0.16.0  h1:IDFfr1l9bbi5WnGflG91suNh2MYbKVjaZedAZmW3rEs=
```

The submodule is current; **the engine inside it is the last release**. Every `decoder`,
`serveapp`, tokenizer and kernel fix between two releases was missing from the Mac and Linux
binaries of the later one, for as long as the split entrypoints have existed. Only the Windows
asset — built from the root itself — was ever current.

This compounds the cold run's finding #3 rather than duplicating it: that one was "the Mac binary
has no GPU backend", this one is "the Mac binary is also a release behind".

**Fixed** by `go mod edit -replace ...=$root` on the ephemeral checkout before each submodule
build, so the asset carries the tagged tree. Nothing is committed, and `standalone-build.yml`
still proves a real consumer can resolve the submodule from the proxy — a different question from
what *we* ship.

**Gate.** A root resolved from the proxy carries an `h1:` module hash on its `dep` line; one
supplied by the replace does not. The workflow now fails any GPU asset whose `dep` line carries
that hash. Exercised both ways against real binaries: red on all four slots when filled with the
proxy-built asset, green when filled with the replace-built one. The darwin half of the
main-module check could not be exercised on this box (no darwin binary to build here); the linux
half was, and the code is symmetric.

> Two smaller things this shook out, both in the same block: `-tags=metal` proves nothing (no
> file in `metal/` is gated on that tag), and a `cmd && { ...; fail=1; }` guard **exits non-zero
> on the PASSING case** under `set -e` and would abort the job. Both are now `if` blocks and
> main-module checks.

### R3 — a model bigger than RAM swapped with no warning, and the flag that fixes it was invisible

**Found** (finding #1 + scenario D's dead end). A 21 GB 35B-A3B on a 16 GB Mac: **+7,819 MB of swap
in five seconds**, no message, watchdog kill. `serve --help` names `-stream-weights` for that exact
model and that exact RAM — 13,583 bytes in. With the flag, RSS peaked at 8.95 GB and *fell*, with
zero swapouts for 115 s. The engine already did the right thing; the product hid it.

**Fixed, smallest first.**

1. *README* — a "Running a model bigger than your RAM" section with the flag, a rule of thumb
   (checkpoint larger than about half your RAM → use it), a real model in the example, and the
   measured before/after with its provenance. It also says plainly that this is `goinfer-serve`'s
   job and that `goinfer-chat` has no such flag by design.
2. *The load-time guard* (`decoder/fitguard.go`) — **`task-fit-to-hardware.md` Phase 0 only**.
   Before `loadWeights` allocates a byte, it prices the checkpoint from GGUF metadata at the
   requested quant, adds KV at a pinned context if one was pinned, and compares against **70% of
   physical RAM** — the same fraction and the same single-measurement provenance as
   `metal/backend.go`'s `residentMemFraction`. Over budget refuses, with the arithmetic and the
   remedy. It does **not** plan a configuration and does **not** flip `-stream-weights` on; those
   are that doc's later phases.
3. *The banner* — at ≥75% of budget the same arithmetic prints unasked, so a user sees the cliff on
   the run **before** the one that steps off it.

Everything unknown proceeds: an unreadable RAM figure (Windows and the BSDs have no probe, and a
container's `MemTotal` is the host's), a non-GGUF source, a zero estimate. **A safetensors
directory is deliberately not estimated from its file size** — those are f32/bf16 on disk and
shrink loading at int4, so file bytes would refuse models that fit comfortably. An estimate wrong
in the refusing direction is worse than none.

```
decoder: Qwen3.5-35B-A3B-Q4_K_M.gguf needs ~21.0 GB resident at quant int4; this machine has
16.0 GB RAM (budget 11.2 GB = 70%).
  Loading it would page to swap rather than run, so it was NOT loaded.
  Re-run goinfer-serve with -stream-weights: it caches the model as a sidecar .giw once, then
  pages weights out of it on demand instead of holding them all resident.
  Or run a smaller model.
  Set GOINFER_NO_FIT_GUARD=1 to load anyway if this machine really fits it
```

**Gate — `decoder.TestFitGuard_refusesBeforeAllocating`.** RAM is injected, so the 16 GB machine's
arithmetic runs on any box. It asserts the refusal, that the message names `-stream-weights` and
the numbers, **and that `loadWeights` was never entered** — observed through a counter, not
inferred from the error text.

> That last assertion is the one that matters, and it needs two mutations to prove:
> deleting the guard makes the load succeed (red), and **moving the guard to after `loadWeights`
> produces the identical error message and the identical swap storm** — caught only by the
> counter. A guard that fires one line too late looks exactly like a guard that works.

Three more pin the directions it must not fail in: the env override loads, a 64 GB machine is
silent, and an unknown RAM figure proceeds. `TestFitEstimate_agreesWithResidentWeightBytes`
pins the pre-load estimator against M-01's post-load accountant (measured 0.96 and 1.07 on the
tiny GGUF at int4 / int8int8) so the two cannot drift apart.

**Commit:** `db61c833`.

### R3-follow-on — on Apple Silicon, `--quant int4` uses MORE RAM than `--quant int8int8`

**Not a cold-run finding either. R3's gate found it**, and it contradicts what `serve --help` and
this repo have been telling users.

Measured on darwin/arm64 in CI, 2026-09-06, by pushing a probe matrix through the loader's own
quantization and asking `wmBytes` what it cost:

| quant | bytes/element, arm64 | bytes/element, amd64+VNNI |
|---|---|---|
| `int8` / `int8int8` | **1.0156** | 1.0156 |
| `int4` / `int4mix` | **1.2500** | 0.6250 |

The cause is `RepackInt4Row4` (`linalg/weightmat_row4_arm64.go:21`): on arm64 with dotprod it
populates `q4Row4` and `q4Row4Scales` **in addition to** the canonical `q4`/`q4s`, clearing
neither — so an int4 weight carries two full layouts, 0.625 + 0.625. int8 has no such repack. The
same shape applies on AVX2-without-VNNI amd64 via the split-half repack.

So on an M-series Mac, `--quant int4` costs about **23% more resident RAM than `--quant
int8int8`**, at lower accuracy. Two shipped claims are wrong there:

- `serve --help`, `-quant`: *"int4 … smallest"* — it is not, on arm64.
- `serve --help`, `-quant`: *"int8int8 … ~2x the RAM of int4"* — it is **0.81x**, on arm64.

**This is not a small correction.** The cold run that started this whole pass was a 16 GB M1 Pro
running out of memory, and the advice the product gave that user — reach for int4, it is the
smallest — is inverted on their machine. It also means R3's own decline message, which offers
"`--quant int4` (the smallest…)" as a remedy, was offering the wrong one on arm64; that line is
now conditioned.

**What is NOT claimed here.** This is a RESIDENT-MEMORY measurement only. int4 remains the faster
option on Apple Silicon CPU — that is a separate, separately-measured claim
(`docs/benchmarks.md`, `docs/task-w4a8-neon-bandwidth.md`), and the repack is precisely what buys
that speed. The trade on arm64 is "int4 is faster and larger", not "int4 is worse". Nothing about
the speed rows is touched, and no benchmark number is restated.

**Gate.** `TestQuantBytesPerElem_everyModeIsPlausible` pins int4 to one of exactly **two**
legitimate costs — ~0.625 (encoding only) or ~1.250 (repacked) — rather than to a range spanning
both, because a wrong measurement lands *between* them and a range wide enough to hold both would
accept it. Mutation-checked: scaling the measurement by 1.5 lands at 0.9375 and the gate names
both expected values. It replaces an assertion of mine that int4 must be cheaper than int8, which
CI proved false on the first arm64 run — the assertion was the bug, and the fact it was hiding was
worth more than the assertion.

### R4 — scenarios B and C, the friction entries

**C (embed).** The tester found the entry point by probing 14 package URLs against pkg.go.dev, and
`decoder`'s index has 346 entries with no "start here". Fixed with a compilable
[`examples/embed/main.go`](../examples/embed/main.go) — 40 lines, the scenario's own bar — linked
from the README and **built by CI**, which is not decoration: the first draft of that example did
not compile, against three separate signatures. The facade this really wants stays scoped in
`task-embed-and-harness-ux.md` §2; this is the stopgap that exists today.

**B (harness).** The friction was entirely in *getting* a server (R1) — once running, protocol
compatibility was flawless with zero goinfer-specific knowledge. The one testable gap left was the
agent round-trip, and `serve check` gained a **tools row**: a two-turn OpenAI round-trip, call →
result → answer. Both turns are checked because they fail separately — a server can emit a
well-formed `tool_call` and then choke on the `role:"tool"` message coming back, which a harness
experiences as a conversation that dies on turn two. This extends
`task-embed-and-harness-ux.md` §3.4's doctor rather than adding a second one.

**Gate.** Five `servecheck` tests drive the checker against servers each wrong in one specific way:
a missing `tool_call.id` (a harness has nothing to pair the result with), `arguments` emitted as an
object rather than the OpenAI JSON *string*, a server that 400s the tool result, the correct
round-trip, and a model that simply does not call the tool — which must **skip**, not fail, because
that is a property of the checkpoint and a red row there trains an operator to ignore the row.

**Commit:** `f1b59235`.

### Not fixed in this pass, deliberately

- **The binary installs as `serve`.** `go install …/cmd/serve@latest` drops a generically-named
  binary on `$PATH`. The README says so and says to rename it; renaming the directory would move a
  Hard-tier import path and break the M-19 build guards' own error text.
- **`--help` is 13,583 bytes across 39 flags**, with commit SHAs and self-critique in the prose,
  and the tester could not skim it for the flag they needed. A short/long help split is a real
  item; it is not this pass's.
- **No one-shot prompt flag.** Every scenario-A user guessed stdin. Ollama takes the prompt as an
  argument, and the tester noticed.
- **ANSI escapes leak into non-TTY output** (`^[[1myou>^[[0m`). Both tools do it; goinfer's is a
  one-line fix and is queued, not done.
- **The Ollama comparison was not re-measured.** It is a v0.16.0 release-build artefact by R2's own
  verification, and re-running it belongs with the next cold run on the fixed assets. The
  `benchmarks.md` figures are untouched.

### R6 — version was unanswerable on `goinfer-chat`, and wrong on the release `goinfer-serve`

**Found** (run 2, scenario A + E). `goinfer-chat --version` → `flag provided but not defined:
-version`; a bare `version` positional was silently swallowed and started an interactive chat
session with the embedded model instead of erroring. Separately, `goinfer-serve --version` on
the actual v0.17.0 **release asset** printed `v0.0.0-20260907045005-f36b095ac9a1+dirty` — not
`v0.17.0`. Root-caused, not guessed: `internal/serveapp/version.go`'s `buildIdent()` reads
`runtime/debug.ReadBuildInfo()`, and the release workflow's GPU assets are built from an
ephemeral submodule checkout with a `go mod edit -replace` applied (R2-follow-on's own fix for a
worse bug) — that uncommitted go.mod edit alone is enough for the VCS stamp to read "modified"
even though the tree is exactly the tagged release. A `go install .../cmd/serve@v0.17.0` build
(no replace, no ephemeral checkout) was never affected. Also found: the embedded-tier binaries
(`goinfer-chat-0.5b`/`-1.5b`) are baked at a FIXED quant chosen at build time — `internal/
chatapp/prequant.go`'s `loadEmbedded` never reads `opts.Quant` at all — while `--help`'s shared
`-quant` text says "Default int4" regardless of build; the release assets actually ship at
`int8int8` (`cmd/prequant`'s own default), a real gap between documented and actual default.

**Fixed, four parts.**

1. *`goinfer-chat --version`.* New `internal/chatapp/version.go` (`isVersionArg`,
   `versionReport`), dispatched in `Main()` before `flag.Parse` and registered as a `-version`
   flag, mirroring `internal/serveapp`'s existing pattern deliberately kept as a separate,
   unshared implementation (`internal/chatapp/version.go:87-93`) rather than shared, so a change
   to one binary's dispatch cannot silently reach the other.
2. *Unrecognized positionals now error.* Both `internal/chatapp/main.go` and `internal/serveapp/
   main.go` check `flag.Args()` after `flag.Parse()` and exit 2 naming the real subcommands —
   the exact `version` typo that fell through silently on v0.17.0 now errors immediately.
3. *`-ldflags -X` version injection*, so a release asset's `--version` states the TAG rather than
   a VCS pseudo-version: `injectedVersion` vars in both `internal/chatapp/version.go` and
   `internal/serveapp/version.go`, preferred over `debug.ReadBuildInfo()` when set. Wired into
   every release build site — `.github/workflows/release-assets.yml`'s plain runtime loop, the
   `goinfer-serve` `build()` function (all six targets, including the two GPU submodule builds),
   and `demo/chat/build-embed.sh` (via new `GOINFER_RELEASE_TAG`/`GOINFER_TIER` env vars the
   embedded-tier job now sets).
4. *The embed-quant discrepancy.* `build-embed.sh` gained an explicit `QUANT` variable
   (`int8int8`, matching `cmd/prequant`'s own default but now a single named value instead of an
   implicit one) threaded through to BOTH the actual bake (`go run ./cmd/prequant -quant
   "$QUANT" ...`) and a new `-X .../chatapp.embeddedQuant=$QUANT` ldflag — so the two cannot
   drift apart the way the flag help text and the shipped binary just had. `versionReport` prints
   `embedded: tier=<T> quant=<Q> (baked at build time; --quant has no effect on this binary)` for
   a `-tags prequant` build, gated on a new `quantIsFixedAtBuildTime` const (`true` in
   `prequant.go`, `false` in `embed.go`'s `--gguf` mode, which genuinely does read `--quant` at
   launch).

**Verified end to end, not only unit-tested.** Built the real `-tags prequant` binary against the
local prequant bundle staged in `internal/chatapp` (build-embed.sh's own build input, gitignored
and not committed — a real one already sat on this box) with the actual
`-X` flags `build-embed.sh` now emits: `--version` printed exactly `goinfer-chat-1.5b-smoketest
v0.17.0-test (…)` / `embedded: tier=1.5b quant=int8int8 (baked at build time; --quant has no
effect on this binary)`. Then ran the unmodified, edited `build-embed.sh` script itself
end-to-end against a real cached GGUF (not a hand-typed equivalent command) and got the same
result from the real output binary.

**Gates.**

- `internal/chatapp/version_test.go`: `TestIsVersionArg_recognizesAllForms` (unit);
  `TestChatVersionFlag_answersWithoutAModel` and `TestChatUnknownPositional_namesTheSubcommands`
  build and run the real `demo/chat` binary and assert on its actual stdout/exit code.
- `internal/serveapp/version_test.go` gained `TestServeUnknownPositional_namesTheSubcommands`,
  same shape.
- `internal/chatapp/version_prequant_test.go` (`//go:build prequant`): asserts unset
  `embeddedTier`/`embeddedQuant` name themselves as unset rather than silently reading as the
  unrelated `--quant` flag default, and that injected values are reported verbatim. Requires the
  build tag and a staged `.giw` — deliberately not part of the default `go test ./...`, so a
  600+ MB local asset is never required to pass.
- Release workflow: three new `assert --version prints the exact tag` steps (plain runtime,
  `goinfer-serve`, embedded tier) — linux-amd64 assets run `--version` for real; darwin/windows
  assets are grepped for the tag string's byte-presence in the binary (the `-X` value is a
  rodata literal), the same shape R2's `go version -m` check already used for what it cannot
  execute. The embedded-tier assertion also greps for `tier=<matrix.tier> quant=\S+ (baked at
  build time`.
- **Mutation-checked.** Before this fix, `goinfer-chat --version` on v0.17.0 produced exactly
  `flag provided but not defined: -version` (reproduced directly during the run, not inferred) —
  the new tests assert this string is absent from a passing run's output.

### R7 — the README's own "bigger than my hardware" examples named a checkpoint nobody can get

**Found** (run 2, scenario D). The README's `-stream-weights` section named
`~/models/qwen3.5-35b-a3b-q4_k_m.gguf` as its example — a local path, never claimed as
`pull`-able, but the only size-class example the README gave for the exact scenario it was
illustrating. `goinfer-chat models`'s curated list topped out at `granite-4.0-h-tiny` (7.4 GB) —
nowhere near 20–35B. A cold user following the README's own "run bigger than your hardware"
story had no way to obtain a checkpoint actually large enough to need the flag.

**The specified fix could not be built as specified, and that refusal is itself part of the
finding.** The instruction for this item named **Gemma-4-26B-A4B** as the checkpoint to add.
No verifiable download exists for it anywhere in this tree: every reference in `decoder/`'s own
tests (`gemma4_26b_real_test.go`, `gemma4_coherence_probe_test.go`, seven others) points at a
LOCAL, unpinned safetensors directory (`GOINFER_GEMMA4_26B=~/models/gemma-4-26b-a4b-it`), never
a GGUF with a repo, file and digest. `pull/registry_test.go`'s own
`TestRegistry_digestsMatchLocalFiles` comment records that this project shipped exactly this
mistake before: *"two of the three entries shipped with a FABRICATED digest and a wrong byte
count... a fabricated digest is still well-formed lowercase hex of the correct length."*
Inventing a sha256 here would reproduce that defect deliberately, and would make
`pull gpt-oss-20b`-shaped commands fail sha256 verification on every attempt — a worse outcome
than the dead end this item exists to close.

**Substituted `gpt-oss-20b`** (OpenAI, real, already in `docs/capability-matrix.json` at
`real-oracle` parity — a T3 method, `pull/registry_test.go`'s
`TestRegistry_noEntryOutrunsItsParity` bar) instead. Its own family description already claimed
*"resident on an 8 GB card via `-moe-cache-experts`, validated on the real 20B"* — this is that
real 20B, and it is the checkpoint this project can actually recommend for the exact scenario
(bigger than an 8 GB GPU) the README's example was trying to illustrate. sha256 (`27cd6c43…`)
and byte count (12,109,566,624) came from Hugging Face's own git-LFS-recorded digest
(`ggml-org/gpt-oss-20b-GGUF`, `?blobs=true`, 2026-09-07) — then independently cross-verified
against a REAL local copy of the file at `~/models/gpt-oss-20b-MXFP4.gguf` via the existing
`TestRegistry_digestsMatchLocalFiles` gate, which reported "verified 4 of 4 entries" including
this one.

**Learned mid-fix: `docs/capability-matrix.json` is a GENERATED artifact.** The first attempt
hand-edited the JSON directly; `decoder/capability_matrix_test.go`'s own comment on the
`Checkpoint` field says exactly why not to — *"Hand-editing the JSON is silently undone by the
next `-update`, which is how the first version of this shipped and went red in CI."* Reverted,
added the entry to the real source (`recommendedCheckpoints["gpt-oss"]` in
`decoder/capability_matrix_test.go`), and regenerated with `go test ./decoder -run
CapabilityMatrix -update` + `cp docs/capability-matrix.json pull/` — the documented,
already-established workflow.

**Follow-up, same day, on review: Gemma-4-26B-A4B does have a real download — the "no verifiable
download exists" conclusion above was too narrow.** It was drawn from grepping this tree's OWN
tests, which reference only a local, unpinned safetensors directory
(`GOINFER_GEMMA4_26B=~/models/gemma-4-26b-a4b-it`) — but that is a fact about this tree's test
fixtures, not about whether Google ever published GGUF weights, and the right check is Hugging
Face directly, not this repo's grep results. It searched: `google/gemma-4-26B-A4B-it-qat-q4_0-gguf`
is real, official (Google's own QAT q4_0 GGUF — the exact artifact `docs/benchmarks.md` §B4/§B4.1
already measures, and the one Ollama's own retraction note in that section confirms it loads
too), and matches the naming convention the README's own "bake any model" example already used at
a smaller Gemma-4 tier. sha256/bytes came from HF's git-LFS digest (2026-09-07), then — unlike
gpt-oss-20b, where a 12 GB local re-hash was skipped as impractical — actually cross-verified
against a real local copy already on this box
(`~/models/gemma4-26b-gguf/gemma-4-26B_q4_0-it.gguf`, exact byte match) via
`TestRegistry_digestsMatchLocalFiles`. Added as `recommendedCheckpoints["gemma4"]` alongside
`gpt-oss`, not instead of it — both are real, both are 20-35B-class, and Gemma-4-26B-A4B is the
one this project has by far the most measurements on and the one that specifically exercises the
8 GB card's host↔VRAM expert-streaming design (`-moe-cache-experts`, §B4's "C′" cache) that
`-moe-cache-experts`'s own README example now demonstrates with a real number (16.12 tok/s at 30
slots) instead of a generic claim. The lesson, stated plainly since it very nearly went out
wrong: **"I could not find a download in this repo's own tests" and "no download exists" are not
the same claim**, and only the second one licenses refusing to add an entry.

**README fixed** to use real checkpoints: the "Running a model bigger than your RAM — or your
GPU" section now states the per-backend rule plainly (cuda/metal: fully resident or CPU, no
partial path — see R9) before naming `goinfer-chat pull gemma-4-26b-a4b` as the primary
`-moe-cache-experts` example (real, measured numbers) and `goinfer-chat pull gpt-oss-20b` as a
second option, alongside the `-stream-weights` example scenario D's own dead end needed and never
had a runnable one for.

**Gates.**

- `pull/registry_test.go`'s existing `TestRegistry_noEntryOutrunsItsParity`,
  `TestRegistry_everyEntryIsVerifiable` and `TestRegistry_digestsMatchLocalFiles` all cover both
  new entries with no changes needed — the registry's existing gates were already the right
  shape. `TestRegistry_digestsMatchLocalFiles` now reports "verified 5 of 5 entries against local
  files" (up from 4), real hashing, ~28-30s for the full registry including the 14.4 GB
  Gemma-4-26B-A4B file.
- New `scripts/readme_smoke.sh` marker, `<!-- smoke-model -->`: resolves every README-named
  `pull`/`--model` reference against the registry (`goinfer-chat models`'s own output, no
  download), `demo:` tiers (`pull/curated.json`), or an HF `owner/repo` (the models API,
  metadata only) — marked on 6 README lines. **Verified against the actual published v0.17.0
  module**, not a local build: 4 of 6 resolve; `gpt-oss-20b` and `gemma-4-26b-a4b` correctly
  FAIL, because neither registry addition is in the published release yet — exactly the
  red-on-v0.17.0, green-once-tagged shape this pass's gates are supposed to have. Caught and
  fixed a self-introduced bug while building this: wrapping a `<!-- smoke-help -->` marker's
  comment across two lines broke the existing one-marker-next-line extraction convention, found
  only by actually running the script.
- New `scripts/readme_smoke.sh` step: every `[...](docs/...)` link the README cites must exist
  in the checkout (R12's gate; described there, exercised here too since this item added two new
  citations). Mutation-checked: an appended dead link is caught by name; reverted, clean.

### R8 — a registry-recommended checkpoint loaded with a tokenizer decline

**Found** (run 2, scenario D). `granite-4.0-h-tiny` — already in the registry, recommended to
first-time users — printed `GGUF tokenizer.ggml.pre="dbrx" is not a known pre-tokenizer; falling
back to cl100k with a 1-digit cap, so this model's token ids may differ from HF and llama.cpp` on
every load.

**Measured, same discipline as C-10** — walked the real shape rather than guessing a name→shape
mapping (the mistake C-10 was). Fetched `ibm-granite/granite-4.0-h-tiny`'s real HF
`tokenizer.json` (2026-09-07, repo sha `791e0d3d…`): its Split regex is byte-identical to the
cl100k pattern `tokenizer/bytelevel.go:258` already documents, `\p{N}{1,3}` digit runs
(Llama-3's cap), `normalizer: null`, `model.ignore_merges: false` — the one knob that makes it
its own case rather than an alias for `llama-bpe` (which has `ignoreMerges: true`).

**Fixed.** `tokenizer/gguf.go`'s `byteLevelKnobs` gained a measured `"dbrx"` case
(`return 3, norm.NFC, false, false, false, shapeCl100k, true`), with the regex and its source
quoted in the comment.

**Gates.**

- New `pull/registry_tokenizer_test.go`: `TestRegistry_noEntryHasATokenizerDecline` reads a
  COMMITTED, real GGUF-header fixture per registry entry and asserts
  `PreTokenizerDecline() == ""`. The `granite-4.0-h-tiny` fixture
  (`tokenizer/testdata/granite-dbrx-meta.gguf`, 3.6 MB) is a REAL extraction — via
  `cmd/prequant` against the actual local 6.9 GB checkpoint, then `giw.Read`'s tokenizer half —
  not synthesized. A `qwen2.5-coder-0.5b` fixture (already committed at
  `testdata/qwen2-gguf/...`) runs as a sanity control; `phi3-mini-4k`/`gpt-oss-20b` skip cleanly
  (no fixture yet), logged as a skip, not a pass.
- **Mutation-checked**: `git stash`-reverted `gguf.go`'s fix and reran — the test failed with the
  EXACT original decline message; unstashed, green again.
- The capability-matrix line for the family states the tokenizer tier: `granitemoehybrid`'s
  registry `Needs` field now reads *"Tokenizer: pre=\"dbrx\", measured cl100k-shaped
  (tokenizer/gguf.go) — no PreTokenizerDecline"* — scoped to the checkpoint entry rather than a
  new capability-matrix schema column, which would have needed generator changes beyond this
  item's reasonable size; noted as a scope cut, not silently dropped.

### R9 — CUDA and Metal have no staged GPU path at all; "cuda-staged" was a label for the CPU

**Found** (run 2, scenario D). `granite-4.0-h-tiny --backend cuda` (no `--require-backend`)
loaded, banner said `decode path: cuda-staged (int4)`, and `nvidia-smi` sampled at 1 Hz through a
full completion request stayed at the idle baseline (464 MiB) for all 15 samples — the label
said `cuda`, nothing on the device moved.

**First pass root-caused the int4 half and wrongly generalized the rest — caught on review, before
it shipped.** `decoder/weightmat.go`'s `matmul()` dispatches int4-stored tensors to
`w.MatmulBTW4A8Into` — a CPU/host integer kernel — unconditionally, with no backend parameter at
all; `decoder.QuantBackend` declares only `MatmulW8A8`, no int4 counterpart anywhere in this
tree. That part was right. The first pass then wrote "native f32 and int8/int8int8 genuinely
reach the backend" as if that were true on every backend, which is what a review pass (comparing
against `docs/benchmarks.md` §B10's "GPU staged (int8)" row, 485 MiB VRAM, 20-22 tok/s — real
device use) asked to be reconciled. It reconciles because that row is **WebGPU**, not CUDA
(`gpu/matrix_bench_test.go:142`, `Backend: "webgpu"`), and re-reading `cuda/backend.go` and
`metal/backend.go` finds the real shape: **neither implements `decoder.QuantBackend` at all**,
and each one's own `Backend.MatmulBT` is a bare CPU call (`linalg.MatmulBT`, no device dispatch)
— so `matmul()`'s `be.(QuantBackend)` assertion fails for int8 on cuda/metal too, falling to the
same CPU kernel int4 uses. **CUDA's and Metal's staged path has no device dispatch at ANY
quant** — it was never a real path, only WebGPU's is (for f32/int8/int8int8; WebGPU's own real
int4 GPU kernel, `gpu/gemv_w4a8.go`, is wired only into the separate fully-resident runner and is
unreachable from this staged dispatch either). This is a **static fact of backend ×
implementation**, not a live-device measurement — which is also why the fix needed no new
runtime tracking.

**Fixed by retiring the label rather than annotating it.** `decoder/residency.go`'s
`DecodePath()` no longer emits `"<backend>-staged"` for cuda/metal at all: when a cuda/metal
model is not resident-eligible, it now reports through the same shape `BackendSummary()` (R2)
already uses for a backend that failed to build in the first place — `requested <backend> →
running on cpu: <reason>, and <backend> has no staged decode path` — because it is the same kind
of claim (what the user asked for vs. what is actually executing), and
`docs/hardware-matrix.md`'s generated table already only ever says `✅ resident` or `CPU` per
cell; this makes the banner agree with that page instead of inventing a third state ("staged")
that page never claimed. WebGPU keeps a real `"webgpu-staged"` path, still annotated per-quant
(`stagedDeviceNote`, now WebGPU-only and simplified back to a pure function of quant alone) for
its own int4/int4mix gap. Also touched, to state the same rule where a user reads it before
ever loading a model: `--backend`'s help text on both binaries, `docs/benchmarks.md` §B10's row
labels (all three GPU-prefixed rows are WebGPU-only measurements — relabeled from a bare "GPU"),
and `docs/task-fit-to-hardware.md` §8, which now says plainly that its planner plans slots and
context, not layer placement, and that real hybrid CPU/GPU layer placement (the gap the peer
matrix's llama.cpp `--fit` comparison exposed at this same size class) is a separate, larger,
not-yet-started item.

**Confirmed unchanged and already correct:** `--require-backend` already refuses every staged/
declined case unconditionally (`internal/serveapp/main.go`'s `requireFastPaths` checks
`!ResidentActive()`, independent of quant or backend), which is exactly what the run's own
scenario D observed firing before any GPU/swap touch.

**Verified end to end on real hardware, twice** — the original int4 finding and the corrected
backend-wide one. Built `cuda/cmd/serve` from the already-present local `go.work` (no manual `go
mod replace` needed) against the real `granite-4.0-h-tiny-Q8_0.gguf` on the actual RTX 2070
SUPER, before and after the fix. After: `decode path: cpu (int4) — requested cuda → running on
cpu: arch is not eligible for the resident decode runner, and cuda has no staged decode path`.
Also confirmed no regression on a resident-eligible checkpoint (`qwen2.5-coder-0.5b`, same box):
`decode path: cuda-resident (int4)`, unchanged.

**Gates.** `decoder/staged_device_note_test.go`: `TestDeclinedToCPUReason` (the cuda/metal
reason-string builder, pure function, no live device needed) and `TestStagedDeviceNote`
(WebGPU's per-quant note, also pure). Both are static-fact tests by design — which (backend,
quant) pairs reach a device is determined by which interfaces `cuda/backend.go`, `metal/
backend.go` and `gpu/backend.go` implement, not by anything measured at runtime.

### R10 — the embed example compiled first try; testing it against a real model found a real bug

**Found** (run 2, scenario C). A README-and-pkg.go.dev-only reader's own ≤40-line program
compiled and ran on the first try — genuinely clean — but printed "Hello. Hello. Hello. ..."
instead of a coherent reply, because it encoded the raw prompt with no chat template.
`examples/embed/main.go` (R4's own stopgap from batch 1) already existed and already applied a
template — but ACTUALLY RUNNING it against a real fixture (not just `go vet`, which is all CI
did) surfaced a second, real bug: it hardcoded `chat.ChatML()` regardless of the checkpoint's
own template, producing garbage on a model trained on a different one.

**Fixed.** Switched to `chat.Detect(chat.Meta{ChatTemplate: tok.ChatTemplate(), HasToken:
tok.Has})` with a raw-completion fallback on `ErrUnknownTemplate` — the same detection
`internal/chatapp` and `internal/serveapp` already use. Stayed at exactly 40 non-comment lines.
One sentence each added to `decoder.Generate`'s doc comment and the README's library section:
`Generate` is a raw completion primitive, chat formatting is the caller's job, here is the
example.

**Verified against two real fixtures**, which is how the second bug was found: `gemma-3-270m`
(degenerate either way — likely a base/non-chat checkpoint, not this item's bug) and
`tinyllama-1.1b-chat` (real chat-tuned model): the OLD example produced a real, if
harder-to-spot, degeneracy — a `User:.../Assistant:...` loop appearing only in the LATTER half of
a 256-token generation — and the NEW example produced varied, if oddly-spaced (a separate,
accepted limitation of no UTF-8/space holdback in a ≤40-line program), numbered-list output.

**Gates.**

- `examples/embed/main_test.go`: builds and runs the REAL binary against the real, already-
  committed `testdata/tinyllama-gguf/` fixture, asserting no substring repeats 3+ times
  immediately back-to-back — searched over every period 8–64 characters (character n-grams, not
  word-split, because this example's naive per-token decode does not reliably preserve
  whitespace). A first version of this test checked only 4 guessed period widths and MISSED the
  old example's real loop (period 34) — caught only by mutation-testing it, not by inspection.
  **Mutation-checked properly after the fix**: old example fails with its real degeneracy
  pattern, new example passes.
- `decoder/doc_fields_test.go`: `go doc . Options`/`SamplingParams` already list every field
  completely — confirmed locally; the truncation the run hit was the tester's own web-fetch
  tool's rendering of the live pkg.go.dev page, not a defect in the source doc comments or in
  `go doc` itself, so no comment restructuring was needed. The gate reflects the REAL struct
  fields (self-updating, no hardcoded list to drift) and asserts `go doc`'s text names every one
  — a regression gate for a defect that, on inspection, was not actually in this tree.

### R11 — the doctor's minimal tool schema passed; a real agent's schema did not

**Found** (run 2, scenario B). `goinfer-serve check`'s `tools, OpenAI` row — a one-function
schema — passed. opencode (a real agent CLI), driving the SAME server, printed a fake JSON tool
call as assistant prose, twice, instead of a real `tool_calls` response, burning ~350 s. Isolated
before assuming a goinfer bug: a raw curl with a minimal one-tool schema against the same server
DID get a proper `tool_calls` response — the gap is schema SIZE, invisible to a doctor that only
ever sends one tool.

**Fixed.** `internal/servecheck/check.go` gained `ToolsHarness` — a second tools row using a
dozen tools with nested object parameters (file read/write/edit, shell, search, plus
`get_weather` among them), shaped like opencode's own "build" agent. Reuses the existing
`toolCall` helper unchanged (schema-agnostic), so "a model did not call the tool" is reported as
a SKIP naming the schema size, not a failure — the same rule `Tools` already applies to its
minimal case, so a red row here does not train an operator to ignore it.

**Verified end to end on real hardware, against two real checkpoints** — the exact real
qwen2.5-coder-1.5b GGUF from scenario A, and separately the actual registry entry
`qwen2.5-coder-0.5b`: both show `tools, OpenAI ... ok` and `tools, harness-scale ... skip`, a
direct, automated reproduction of the opencode finding — the doctor now predicts this before an
operator hits it, rather than vouching past it with a green minimal-schema row.

**Registry gains a measured `tools` column** (`pull.Checkpoint.Tools` /
`recommendedCheckpoint.Tools`), shown in `goinfer-chat models`. Populated from real runs where
run 2 had time to make them, honestly marked `"not yet measured"` where it did not — a
`granite-4.0-h-tiny` attempt was started and killed after it did not finish in a reasonable time
(R9 explains why: CPU-only staged int4 plus this arch's sequential prefill), and was recorded as
unmeasured rather than extrapolated from a different checkpoint's result.

**README's harness recipe** (`docs/integrations/claude-code.md`, previously unlinked from the
README — fixed in passing) gained a section citing the measured evidence for "which model
tool-calls under a real agent": that page's own 2026-09-02 run already answered this for a
different checkpoint — **Qwen2.5-7B-Instruct** held a real 25-tool-schema agent loop, and that
page's own text already says a 1.5B "re-calls the same tool forever." Cited rather than
re-measured, plus this pass's fresh registry-entry results.

**Gates.**

- `internal/servecheck/check_test.go`: `TestToolsHarness_roundTripOK` and
  `TestToolsHarness_noCallIsSkipWithReason` (fake-HTTP-server pattern, matching the five existing
  `TestTools_*` tests exactly).
- `pull/registry_test.go`'s new `TestRegistry_toolsColumnIsNonEmpty`: every entry must have SOME
  value in `Tools`, even an honest `"not yet measured"` — it cannot distinguish a real
  measurement from a placeholder (that is a human's job when filling one in), only catch the
  entry that forgot the field.

### R12 — README numbers lacked provenance context, and a citation can silently rot

**Found** (run 2). The "25 seconds" cold-start claim carried no download size or network
context, so a different tester's very different number (56.5 s, dominated by a 1.71 GiB download
at this run's ~31.5 MB/s) reads as a regression rather than a network fact. The scenario-E hedge
("Ollama led it on the release build that run tested") did not say WHICH defect made that true
(R2's Metal-less Mac asset), so a reader cannot tell a packaging artifact from an engine result.
Run 2's own new CUDA decode measurement (192.8 vs 183.6 tok/s) initially read as contradicting
`docs/benchmarks.md` §B8's formally-provenanced anchor table — it does not: §B8's shallow-KV-
depth cells already show goinfer ahead of Ollama on the same quant class, and run 2's short
completion sits in exactly that regime.

**Fixed.** README's cold-start section now states the download size/network speed beside the
56.5 s number, names R2's Metal gap explicitly next to the old hedge, adds the run-2 CUDA result
with its §B8-consistency note, and points to `scripts/bench_peer.py` — the already-committed,
real harness both the official anchor table and (informally) this run's own method used
underneath — as the "measure it yourself" recipe, so the next tester does not reconstruct
client-side-tok/s-from-first-token by hand the way this run did.

**Gate.** New `scripts/readme_smoke.sh` step: every `[...](docs/...)` markdown link the README
cites must resolve to a file that exists in the checkout. **Mutation-checked**: appended a dead
citation to the real README, confirmed it is the only one flagged among 20; reverted.

---

## 3. What the cold run confirmed was good

Worth recording, because a protocol that only produces defects gets read as noise:

- **Cold start is the number goinfer wins**: 25 s from nothing to an answer against Ollama's 33 s,
  from an 8 MB binary with no daemon.
- The bare-run error names the missing flag *and* prints usage.
- `pull` resumes, verifies sha256, and prints the resolved cache path.
- The `missing go.sum entry` message was good enough that the tester fixed R1's install failure by
  following its own advice.
- The startup banner — "the best thing in the product" — lists every route, names the decode and
  prefill paths, and warns when no `-api-key` is set.
- Protocol compatibility: `curl` streaming and non-streaming, and the unmodified `openai` Python
  client (`models.list`, non-stream, 30 stream deltas) all worked with no goinfer-specific
  knowledge.

---

## 4. Predicted vs found — scoring `task-embed-and-harness-ux.md`

That doc was written 2026-09-02 from the inside, four days before the run. It is the closest thing
to a prediction of the first hour that existed, so it is worth scoring honestly.

| run | what the cold run found | did the doc predict it? | where |
|---|---|---|---|
| 1 | the banner is what a harness user reads, and must carry the facts | **yes, exactly** — "the banner is the UI", with a per-line design | §3.3 |
| 1 | `serve check` needs to drive the routes a harness uses | **yes** — the doctor, with the tools rows sketched | §3.4 |
| 1 | per-harness recipes with an expectation line | **yes** — one per harness, ≤40 lines | §3.5 |
| 1 | the embedder has no "start here"; `decoder` is 346 entries | **yes** — the whole premise of the facade | §1, §2 |
| 1 | finding a checkpoint is a first-hour problem | **yes, and already closed** by `pull` + `hf:`/`demo:` refs | §1.1 |
| 1 | **the release does not ship the binary the README names** | **no** | — |
| 1 | **the darwin asset links no GPU backend** | **no** | — |
| 1 | **the banner prints the requested backend, not the effective one** | **no** — the doc says what the banner should *add*, never that what it prints could be false | §3.3 |
| 1 | **a model bigger than RAM swaps silently** | **partly** — `task-fit-to-hardware.md` owns it and had the guard scoped as Phase 0; nothing said the *flag was undiscoverable* | fit §7.0 |
| 1 | the install command in the README does not build | **no** | — |
| 1 | **the GPU assets are a release behind the tag** | **no** — and no doc anywhere named it; it was found by R2's gate, not by the run | — |
| 1 | `--help` is unusable as a quick reference | **no** | — |
| 1 | no one-shot prompt flag | **no** | — |
| 2 | **`--version` unanswerable on `goinfer-chat`; wrong on the released `goinfer-serve`** | **no** — a build/release-hygiene defect the facade/harness doc has no occasion to name | — |
| 2 | **the README's own "bigger than your GPU/RAM" examples name a checkpoint nobody can get** | **partly** — §1.1's "finding a checkpoint is closed" was scored true for the SMALL end (`pull`/`hf:`/`demo:` work); run 2 found the LARGE end is not: the registry has nothing near 20–35B, so the doc's own claim does not extend to the scenario it is illustrating | §1.1 |
| 2 | **a registry-recommended checkpoint (`granite-4.0-h-tiny`) loads with a tokenizer decline** | **no** | — |
| 2 | **`cuda-staged (int4)` never reaches the GPU, for any architecture, always** | **partly** — R2 (run 1) already made the banner name the EFFECTIVE backend by NAME; nothing said a nominally-selected backend could still dispatch zero device bytes for a whole quant mode | §3.3 |
| 2 | **the (already-fixed, batch-1) embed example still degenerates on a real model, from a hardcoded template** | **yes, specifically** — §1 step 3 names `chat.Detect` as one of the six steps a caller must not skip; the batch-1 stopgap example skipped it anyway | §1 step 3 |
| 2 | **the doctor's minimal one-tool schema does not predict a real agent's larger-schema failure** | **no** — §3.4 sketched more tool-call PROTOCOL variants (Anthropic, Responses) to check, never schema SIZE as its own dimension | §3.4 |
| 2 | **README numbers carry no provenance context; a citation can silently rot** | **no** | — |

**The pattern.** The doc was right about **everything that needed designing** and blind to
**everything that needed checking**. Every miss is a place where a claim the project makes about
itself — the README's commands, the release's assets, the banner's `[backend=…]`, the install line
— had no gate asserting it was true. None of the misses are hard problems; all of them are things
nobody was looking at, because the people looking already knew the answers.

That is the argument for the ritual in §1, and it is why every fix in §2 ships with a gate that
would have gone red on v0.16.0 (run 1) or v0.17.0 (run 2) rather than with a doc that says what
should be true. **Run 2's pattern is the same shape at a different layer**: every miss is again a
claim nothing gated — a version string, a registry entry's own download, a tokenizer, a banner's
device-use claim, a stopgap example's own template choice, a doctor's schema size, a README
citation — and the one hit (R10) landed exactly where the design doc had already named the right
step and a later stopgap skipped it anyway, which is its own small lesson: a correct design does
not enforce itself.

---

## Sources

[`measurements/cold-user-2026-09-06.md`](measurements/cold-user-2026-09-06.md) (run 1) ·
[`measurements/cold-user-2026-09-06-nobara-pc.md`](measurements/cold-user-2026-09-06-nobara-pc.md)
(run 2) ·
[`task-embed-and-harness-ux.md`](task-embed-and-harness-ux.md) (§4 scores it) ·
[`task-fit-to-hardware.md`](task-fit-to-hardware.md) (R3 is its Phase 0) ·
[`api-tiers.md`](api-tiers.md) (what R1's install line may promise) ·
`RELEASING.md` (the ritual this doc is now part of) ·
`metal/backend.go` (the 70% fraction R3 reuses, and its single-measurement provenance) ·
`docs/audit-2026-09-02.md` M-01/M-02 (the accounting R3's guard reads), M-19 (why the root
`cmd/serve` links no backend, which is what R2 found shipped) ·
`docs/benchmarks.md` §B8 (R12's peer-decode consistency check) ·
[`docs/integrations/claude-code.md`](integrations/claude-code.md) (R11's "which model tool-calls
under a real agent" evidence) ·
`pull/registry_test.go` (R7's `TestRegistry_digestsMatchLocalFiles`, whose own comment records
the fabricated-digest mistake R7 refused to repeat)
