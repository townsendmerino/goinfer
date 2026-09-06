# Task: the first hour — what a cold user found, what was fixed, and how to run it again

> **Status: R1–R4 SHIPPED 2026-09-06** against the run recorded in
> [`measurements/cold-user-2026-09-06.md`](measurements/cold-user-2026-09-06.md) (M1 Pro / 16 GB,
> v0.16.0, five scenarios, 13.5 min). Every fix below carries a gate that goes red on v0.16.0;
> the mutation used to prove each one is named. R5 is the protocol, which is this document's §1
> and does not "ship" — it is run.
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

**Next run:** the **Linux box**, after the R1–R3 tags land. The CUDA path makes "bigger than my
hardware" a different story (expert streaming vs `-stream-weights`), so it is not a repeat. Its
results go in as run 2 and §4's table gains a column.

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

| what the cold run found | did the doc predict it? | where |
|---|---|---|
| the banner is what a harness user reads, and must carry the facts | **yes, exactly** — "the banner is the UI", with a per-line design | §3.3 |
| `serve check` needs to drive the routes a harness uses | **yes** — the doctor, with the tools rows sketched | §3.4 |
| per-harness recipes with an expectation line | **yes** — one per harness, ≤40 lines | §3.5 |
| the embedder has no "start here"; `decoder` is 346 entries | **yes** — the whole premise of the facade | §1, §2 |
| finding a checkpoint is a first-hour problem | **yes, and already closed** by `pull` + `hf:`/`demo:` refs | §1.1 |
| **the release does not ship the binary the README names** | **no** | — |
| **the darwin asset links no GPU backend** | **no** | — |
| **the banner prints the requested backend, not the effective one** | **no** — the doc says what the banner should *add*, never that what it prints could be false | §3.3 |
| **a model bigger than RAM swaps silently** | **partly** — `task-fit-to-hardware.md` owns it and had the guard scoped as Phase 0; nothing said the *flag was undiscoverable* | fit §7.0 |
| the install command in the README does not build | **no** | — |
| **the GPU assets are a release behind the tag** | **no** — and no doc anywhere named it; it was found by R2's gate, not by the run | — |
| `--help` is unusable as a quick reference | **no** | — |
| no one-shot prompt flag | **no** | — |

**The pattern.** The doc was right about **everything that needed designing** and blind to
**everything that needed checking**. Every miss is a place where a claim the project makes about
itself — the README's commands, the release's assets, the banner's `[backend=…]`, the install line
— had no gate asserting it was true. None of the misses are hard problems; all of them are things
nobody was looking at, because the people looking already knew the answers.

That is the argument for the ritual in §1, and it is why every fix in §2 ships with a gate that
would have gone red on v0.16.0 rather than with a doc that says what should be true.

---

## Sources

[`measurements/cold-user-2026-09-06.md`](measurements/cold-user-2026-09-06.md) (the run) ·
[`task-embed-and-harness-ux.md`](task-embed-and-harness-ux.md) (§4 scores it) ·
[`task-fit-to-hardware.md`](task-fit-to-hardware.md) (R3 is its Phase 0) ·
[`api-tiers.md`](api-tiers.md) (what R1's install line may promise) ·
`RELEASING.md` (the ritual this doc is now part of) ·
`metal/backend.go` (the 70% fraction R3 reuses, and its single-measurement provenance) ·
`docs/audit-2026-09-02.md` M-01/M-02 (the accounting R3's guard reads), M-19 (why the root
`cmd/serve` links no backend, which is what R2 found shipped)
