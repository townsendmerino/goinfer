# Policy: parity & e2e coverage — what "supported" means

> **Audience:** durable contract (not a task — the task doc is
> `task-parity-coverage.md`). Defines the testing tiers, the gates a model family
> must clear, and the rule binding README support *claims* to a current
> validation record. The motivating problem: real-checkpoint parity is
> necessarily asset-gated and `t.Skip`s when assets are absent, so **a green CI
> run does not, by itself, prove any model's numerics** — and nothing today binds
> "we claim family X" to "X has a passing parity record at this commit." This
> policy closes both gaps without pretending we can run multi-GB checkpoints in
> CI.

## Principle: factor the matrix, don't test the cross product

Families × {safetensors, GGUF, GPTQ, AWQ} × {f32, int8, int4} × {CPU SIMD,
WebGPU} × KV precisions is untestable as a cross product, even on the big box.
The axes are largely independent, so each is validated **once** against a
reference and composition is trusted (with a few full-stack spot-checks):

| Axis | Gate | Reference | Needs a big checkpoint? |
|---|---|---|---|
| **Family correctness** | tiny-synthetic golden (CI) + one real-checkpoint parity (gated) | HF bf16 | tiny: no · real: yes |
| **Loader correctness** | `weightDiff` GGUF-vs-safetensors to Q8_0 tol | the other loader | no oracle |
| **Quant** | quant-vs-f32 argmax + cosine within a family | the f32 path | no |
| **Backend** | WebGPU output == CPU output on 1–2 models | the CPU path | no |

The contract a family's *numerics* must hold is per-(family × loader × quant);
backends and quants are proven by their own equivalence gates, so they don't
multiply the family work.

## The three tiers

**T1 — always-on, in CI, no external asset (green must mean something).**
Model-free property/equivalence tests (chunked-scan == sequential, serialize
round-trip, KV-paging byte-identity, `weightDiff`) **and** a per-family
**tiny-synthetic checkpoint + committed golden** (a deterministically generated
small model; its golden logits pinned once offline via the HF reference and
committed, a few KB). Every claimed family has a T1 golden. CI is red if any T1
gate fails — and because T1 needs no assets, red means a real regression.

**T2 — scheduled (nightly/weekly) on a normal box, small *real* models.**
The smallest published checkpoint per family (e.g. `gemma-3-270m`, Qwen2.5-0.5B,
a small Llama/Mistral/Mixtral, Mellum): load → generate → assert a recorded
continuation. Not full parity — the broad, cheap net that catches "this family's
loader segfaults / emits garbage" across the whole claimed matrix. Drives
`parity_sweep.sh`.

**T3 — release-gate, asset-gated, on the big box (or `weightDiff` when the
oracle won't fit).** Full argmax-exact + logit-cosine vs the HF bf16 reference on
the real checkpoint; when model + reference won't co-reside (the 35B-A3B case),
`weightDiff` GGUF-vs-safetensors is the substitute proof. Results are **recorded
in the validation manifest** (below). Run per release and whenever a family's
numerics surface changes.

**T3 valid `method` values:** `full-forward-oracle`, `real-model-oracle`
(int8-resident vs bf16 reference), `weightDiff` (+ optional layer-slice), and
**`shared-path (via <family>)`** — see next.

**Shared-path validation (alias families).** When a family's adapter *aliases*
another's (e.g. `kimi_k2 → deepseekArchitecture`) so that it shares the same
forward file(s) **and the same `deps_hash`** as an already-validated family, its
numerics for the shared surface are *already proven* by that family's T3 run.
Such a family clears T3 with `method: shared-path (via <family>)`, `validated_at`
= the shared commit, and a reference noting that its **config-delta** (the few
scalars/flags that differ — e.g. head/expert counts, routing default) is covered
by its descriptor + tiny-golden tests. No separate oracle run is required. This
is sound precisely because the staleness gate keys on `deps_hash`: if the shared
forward path changes, *both* families go stale together, so the proxy can never
silently drift from its source. (If an alias family ever gains its own forward
file or a distinct `deps_hash`, it loses proxy status and needs its own T3.)

## A gate must be able to run, and able to fail

The tiers above say *what* to test; this says when a gate actually counts. Two ways a
gate reads green while proving nothing — both hit this repo inside one session (the CUDA
MoE bring-up; the Metal Gemma hunt found the mirror image from the other side), so they
are policy, not folklore.

### Runnable: a committed golden over an uncommitted checkpoint is not T1

T1's contract is *no external asset*. A test that reads a committed golden but `os.Stat`s
a `.gitignore`d `model.safetensors` is a T1 in name and a T3 in practice: it `t.Skip`s in
CI and runs only where someone regenerated the fixture. The Mixtral MoE gate was exactly
this — `mixtral_forward_golden.json` committed, its checkpoint gitignored — so **neither the
CPU nor the CUDA MoE parity ever executed in CI**, and the skip was invisible.

Rule: a family's T1 checkpoint is **committed**. These are KB–few-MB deterministic
random-weight models (no license, and no size argument outweighs a family's correctness
gate running on every push — `mixtral-tiny` is 3.6 MB). If a fixture genuinely cannot be
committed, the row is T2/T3, not T1, and the family is **not** "CI-covered" — label it so.

One distinction the fixture-commit does *not* erase: a **GPU-kernel** gate still needs a
real device. Committing its fixture makes it *runnable on a GPU runner*, not run in CI.
The CPU path and the device path are **separate coverage**; only the CPU one runs without
hardware. "The fixture is committed now" must never be read as "the kernel is tested in
CI" — that conflation is how a device-only bug (the Metal MMA NaN) hides behind a green
CPU gate. A device gate is CI-covered only when a GPU runner executes it.

#### The committed set (chosen, not accidental)

To keep the runnable/skipped split a deliberate policy rather than an artifact of which
fixtures someone happened to regenerate, the committed set is enumerated here. **These
tiny checkpoints are tracked** (via targeted `!` exceptions in `.gitignore`, each carrying
this rationale), so their families' **CPU forward-parity is a real T1 that runs on every
push**:

| Committed fixture | Family gate | Size | Generator |
|---|---|---|---|
| `testdata/mixtral-tiny/` | Mixtral / MoE dispatch (`decoder`, CUDA) | 3.6 MB | `scripts/pin_mixtral_tiny.py` |
| `testdata/cohere-tiny/` | Cohere / Command-R v1 (`cohere_test.go`) | 656 KB | `scripts/pin_cohere_tiny.py` |
| `testdata/cohere2-tiny/` | Cohere2 / Command-R v2 (`cohere2_test.go`) | 656 KB | `scripts/pin_cohere2_tiny.py` |

All three are deterministic random-weight models (no license, no real data), committed
because each is the *only* CI-runnable proof of its family's numerics.

**Everything else is generator-reproducible per-machine, and that is also a choice.** The
other tiny/scaled fixtures (`gemma4-dense-scaled-*`, `tiny-qwen2-moe`, the `*-vl-tiny`
vision checkpoints, `qwen35-tiny`, the `*-tiny` families listed in `.gitignore`) keep their
**`*_golden.json` tracked** but their **checkpoint gitignored** — regenerated locally by the
matching `scripts/pin_*.py`. Those families' CPU parity is therefore T3-in-practice (an
asset-gated `t.Skip` in CI) until someone promotes the fixture into the committed set above.

**The rule for promoting a fixture into the committed set:** it is a deterministic
random-weight model of KB–few-MB, *and* we want that family's CPU forward-parity to fail CI
on every push (not just when an asset is present). When both hold, commit it with a targeted
`.gitignore` exception (never a blanket un-ignore — `38e5cd7` shows a broad sweep re-tracks
stray metadata and breaks unrelated gates). Otherwise it stays generator-reproducible and
its family is CI-covered only at the tiny-golden level its committed assets allow.

### Falsifiable: prove the gate red before trusting it green

A parity gate is **vacuous until seen to fail on a real defect**. Before a new gate counts:

1. **Break-it-first.** Perturb the thing under test (route to the wrong expert, swap
   gate/up, mis-stride the stacked experts) and confirm the gate goes red. Three gates in
   one session were vacuous on first write and only a deliberate break-table exposed it. A
   gate never seen red is an assumption, not a test.
2. **The metric must be able to fail.** `if c < minCos` never updates on a NaN (`NaN < x`
   is false), so a NaN cosine — the signature of the *worst* bugs, degenerate output —
   sails through the floor untouched. Guard NaN explicitly. Any reduction tracking a
   min/max/threshold has this hole; a mis-strided-stack control measured "min cosine
   1.000000" next to a 79% argmax gap because of it.
3. **Calibrate the floor to a measured break-table, not to taste.** On tiny random-weight
   fixtures, argmax + the 3% near-tie rule is *necessary but not sufficient*: experts are
   near-interchangeable, so a wrong one contributes a similar-magnitude vector argmax
   cannot see (measured: two real dispatch bugs passed the 3% rule at cosine 0.9887 and
   0.9977). Set the cosine floor from the gap between the correct run and the *tightest
   surviving* broken control, and record that table in the test. An invented floor is its
   own bug — an early CUDA gate asserted cosine ≥ 0.999 and failed the *shipped* dense path
   at 0.9936.

### Relevant: the gate must exercise the path the product runs (audit G-01, the class)

Runnable and falsifiable are both about the *gate's* machinery. This third one is about what
the gate is pointed at, and it is the one that keeps recurring. The class:

> **A test, benchmark or gate exercises something *adjacent* to what ships, and reports its
> result as if it had exercised the real thing.**

The result is not wrong the way a failing test is wrong. It is green, plausible, sometimes
published — and about a different computation than the one users run. Nothing in the tooling
distinguishes the two, because the gate's own machinery is working perfectly.

**The recognition test** — apply to any new test, benchmark, or gate, and to any old one whose
number you are about to publish:

> **Does it call the same code path the product calls, and does it assert on the result?**

Both halves are load-bearing. Same path without an assertion measures a corrupted computation
and reports its speed. An assertion on a parallel path pins a number the product never produces.
A "no" to either means the artifact is about something adjacent — say so, or fix it.

Instances to date. They are listed together deliberately: each was found and fixed as a one-off,
and the cost of that is that the next one was also found as a one-off.

*In code — a parallel implementation drifts from the shipped one:*

- **The three CUDA e2e tests bound `argmax_reduce` from the glue module** — the pre-C-14 kernel,
  without the index tie-break — while production loads `argmax.ptx`. They gated the kernel the
  product had stopped using (`7f03230`).
- **`TestRealE2EDecodeThroughput` hand-rolled its own launch sequence** and omitted `rope_kv`'s
  `rhalf` argument (added when partial rotary landed). The kernel read garbage for the rotary
  half-width and corrupted K/V, and the test reported a throughput number for a broken forward
  for as long as partial rotary has existed. The CUDA launch API does not arity-check, so nothing
  failed (`f5ec7a2`; see the production-side note below).
- **`bench_compare.sh` measured goinfer with in-process Go benchmarks while the peer ran over
  HTTP** — a published ratio dividing a kernel throughput by an end-to-end one. Same class,
  across a process boundary rather than a code path.
- **`glue.ptx` kept a committed symbol its source no longer generates**, so the shipped artifact
  could not be regenerated from the code it claimed to come from (`e2913dc`). The artifact-level
  form: the *thing* is adjacent to its source, not the test to the thing.

*One level up — the mechanism that should have caught it is itself adjacent:*

- **A gate that can only FAIL is ignored exactly like one that can only pass.** `gpu_gate.sh`
  could not reach a green verdict across the whole of v0.10.x and v0.11.0 (it built an
  entrypoint that is a deliberate compile error). Once a gate's red is unremarkable, its red
  stops carrying information — the same end state as G-01's original tautological green, reached
  from the opposite direction (`d2c4858`).
- **A check emitting neither pass nor fail vanished from the tally**, and the gate printed PASS
  having tested nothing. The tally was computed from what emitted, so silence was
  indistinguishable from success. Fixed by reconciling emitted verdicts against a **declared**
  set (`4e293f9`).

*And once as process:*

- **`docs/completed/task-cuda-cgofree-spike.md:798` already recorded** that the reale2e harness
  was not authoritative. The fact was known, written down, and filed where nobody reads it.
  Knowledge parked under `completed/` is indistinguishable from knowledge nobody had — the same
  decay as the stale version pin in `RELEASING.md`. **This is why the class lives here and not
  in an audit file**: a policy doc is read before the next gate is written; an audit is read
  once, at most.

**Production-side status (checked, `f5ec7a2`).** The CUDA launch path hand-rolls its arguments
too: `launch(f Pipeline, cfg LaunchConfig, args ...KernelArg)` (`cuda/resident.go:910`), called
variadically from 48 sites (`resident.go` 36, `prefill.go` 11, `testhooks_gen.go` 1). There are
no typed per-kernel wrappers, so **the compiler enforces neither arity nor argument order** — a
missing trailing argument is silent, and so is a transposition between same-typed parameters,
which counting cannot catch either. `rope_kv`'s tail is five consecutive `int`s
(`nH, nKV, hd, pos, rhalf`); swapping `nH`/`nKV` is invisible on every MHA model and surfaces
only under GQA, and `rhalf == hd/2` under full rotary hides a whole family of confusions there
too. What protects production today is **behavioral, not structural**: the end-to-end parity
gates assert on the shipped path, so a corrupted launch shows up as a parity failure. That is a
real defense — it is the reason the same bug was live in a test and not in production — but it
is coverage-shaped, not compiler-shaped, and it is only as good as the geometries the gates run.
Structural options exist (typed per-kernel wrappers mirroring each `.cu` signature; or
`cuKernelGetParamInfo`, CUDA 12.4+, to check arity against the module at load). **Not funded —
recorded so the next signature change knows what is and isn't holding the line.**

### Representative: a fixture must be real in the dimensions the failure lives in

Two properties have to survive a fixture's shrink, and only one of them is a dimension. Both were
learned by paying for them.

**Geometry.** A′ zero-copy was bit-exact on a 32-expert / hidden-256 fixture and 255/256 logits
wrong at hidden 2048 / `moe_inter` 768. The post-mortem excludes the alternatives by measurement —
"not offset, K, occupancy, **expert count**, or the allocation". So a Gemma-4 MoE fixture shrinks
expert count and depth and keeps `hidden`/`moe_intermediate` real. The reduction that *looks*
right — keep 128 experts, cut hidden — rebuilds the configuration that already passed while the
code was broken. **Shrink the axis the post-mortem excluded, never the one it indicted.**

**Distribution.** v0.9.0's fused multiply-add caused an 84% token-stream divergence on real models
and was invisible on random-weight fixtures, because uniform data rounds the same way in both
orders. Perfect geometry with random weights reproduces that false negative exactly — one axis
over from the first mistake.

Measured on the real Gemma-4 26B (per-group absmax, groups of 32 along K):

| tensor class | log2 std | dynamic range |
|---|---|---|
| experts (`gate_up`, `down`) | 0.31 | **13× within a layer, 20–24× across layers** |
| `q_proj` / `embed_tokens` | 0.49 | 35–39× |
| HF random init `normal(0, 0.02)` | 0.27 | 5.0× |

**Quote 24×, not "orders of magnitude."** The v0.9.0 release note says the latter; it is an
explanatory aside and stays as history, but anyone calibrating a new fixture against it will build
the wrong thing. The gap that matters is real and modest: 24× against 5×.

The cheap way to get it right is not to fit a distribution but to **transplant** one — impose real
per-row group scales on random weights, sampling whole rows so within-row correlation survives.
That only works if the geometry rule above kept K real, which is the second reason to follow it.
See `scripts/pin_gemma4_moe_scaled.py`, which refuses to write a fixture with any untransplanted
tensor, and pins its own output hash because gates asserting bit-identity cannot afford a fixture
that drifts between machines.

State what a fixture does **not** cover, in the fixture's own docs — host-buffer ratio, depth,
whether routing is trained. A gate's edges should be legible to whoever trusts it next.

## Claim discipline: rules for whoever drafts an announcement

These were written as release-checklist items and archived with the release, which is the wrong
place for them — they apply *when a claim is drafted*, and a checklist under `docs/completed/`
runs never. Their absence is traceable: the retracted 476/268 headline and the peer-multiple
framing the README spent a week retiring are both what these rules exist to prevent. Recovered
here as live policy.

1. **Name the regime the number came from.** "dense-model GPU decode, cgo-free / driver-only
   (no CUDA toolkit, no Xcode)" is the claim. A number measured on one model at one context depth
   on one card is not a property of the runtime. If the regime does not fit in the sentence, the
   sentence is too short.

2. **Lead with the property, not a multiple.** cgo-free / no-toolchain plus correctness parity is
   the distinction; a raw-speed multiple over a peer is not, and it inverts the moment the peer
   ships a kernel. Peer comparisons are a row in a table with a date and a version, never a
   headline.

3. **A comparison names the measurement method for BOTH sides, and they must be the same method.**
   This is the rule the 476/268 headline broke, and it is not a special case of rule 1. That number
   was not stale, and it was not a sampling-config difference: **goinfer's side came from an
   in-process Go benchmark and Ollama's from an HTTP server.** Both sides could have been described
   in full — model, quant, context depth, card, driver, version — and the comparison would still
   have been meaningless, because one side was not paying for a socket, a scheduler, or a
   serialize/deserialize round trip and the other was.

   So: measure both sides through the same door. If the peer is only reachable over HTTP, put your
   own side over HTTP too. In-process numbers are legitimate and useful — for tracking your own
   regressions across commits, where the harness is constant — and they are **not** comparable to
   anything measured a different way.

   Where the protocols genuinely cannot match, say so at the point of comparison, say why, and say
   which direction it biases. `docs/benchmarks.md` §B7 is the shape: *"One protocol difference,
   deliberate: a 32k prefill costs orders of magnitude more than the decode being measured, so deep
   cells use fewer requests with more decode tokens each"* — followed by the run-to-run spread
   showing the smaller sample did not cost precision. A named, justified, bounded difference is
   honest. An unnamed one is the 476/268.

4. **State opt-in-ness in the same breath as the result.** If reproducing a number needs a flag,
   an environment variable, or a non-default build, say so where the number appears — not in a
   later section. Two statements, each true in its own place, compose into a false picture for a
   reader who does not read one section at a time. That is the documentation-adjacency class
   above, and the 26B-A4B result is what it cost to learn.

5. **Do not imply generality a gate does not cover.** "Lossless" means the gate asserts losslessness
   on the path being described; "bit-identical" means some specific pair is bit-identical — say
   which pair. The v0.9.0 "every GPU path is byte-reproducible against the CPU reference" was
   assembled from a true GPU-vs-GPU property and an untested GPU-vs-CPU one.

6. **Quote the figure with its basis.** Same measurement, different denominators are not a
   contradiction, but three unlabelled hit rates read as one. Whole-run vs steady-state,
   argmax-exact vs byte-identical, floor vs median — the qualifier is part of the number.

7. **A claim nobody can reproduce from the public documents is not shipped.** Before publishing,
   read the user-facing docs as a stranger with a default build and check that the claim survives.

## Rule: a gate lands with a mutation check

**A new or changed gate is not landed until something demonstrates it can fail.** The commit (or
the test's own comment) names *what was mutated* and *what the failure looked like*. "I reasoned
about it" does not count; nor does "it passed", which is the state a gate that cannot fail is
permanently in.

This is the base rate, not a recurring surprise. The "verification artifact that is not itself
verified" class has surfaced about a dozen times across this program — and three of those were
inside a single gate group written in one week:

- the skip census read `0` because `go test` prints no `--- SKIP` without `-v`, so a suite known to
  skip six reported none;
- the heavy tier's failure filter was copied from a non-`-v` group, so under `-v` it matched every
  `t.Logf` line and `head` truncated the real `--- FAIL` away — a 28-minute group that lost its own
  failure;
- and the fix for that one piped through `tee`, which would have made the group take **tee's** exit
  status and be structurally incapable of reporting red. That is the sharp case: a defect that
  arrived *as the remedy for another*, in the code whose job is catching exactly this.

What a mutation check looks like, in ascending cost: assert the negative case directly
(`(exit 1) | tee | grep | sed` → `PIPESTATUS[0]=1`, and without `pipefail` → `0`, the bug
confirmed); or perturb the thing under test and confirm red (route to the wrong expert, drop a
launch argument, point the pattern at a renamed test); or run the gate against a known-bad commit.
Cheapest sufficient one wins — the point is evidence, not ceremony.

Related and already policy: **Falsifiable** (break-it-first, above) says the same thing about
numeric gates. This generalises it to the tooling: scripts, censuses, lints and filters are gates
too, and they have been the likelier place for this defect precisely because nobody thinks of them
as tests.

## Rule: reverting a commit includes reverting what it claimed

`git revert` undoes code mechanically and leaves prose behind. A commit that changed a default and
told users about it in the README has *two* effects, and reverting one of them ships documentation
that describes software that no longer exists.

Concretely: a change that made the MoE expert cache self-sizing reduced the README's documented
configuration from two environment variables to one. The revert restored the code and left the
README saying "needs one environment variable", with a copyable command that produced a third of
the published rate. That is claim-discipline rule 7 — a claim nobody can reproduce from the public
documents is not shipped — broken by an operation that never looked at the claim.

So a revert's checklist is the original commit's diff, not just its code hunks: every doc, comment,
CHANGELOG entry and published number the commit touched gets re-examined in the same commit.

## Rule: archiving a doc strips its imperatives

When a task doc or checklist moves to `docs/completed/`, its imperative content either **moves to
live policy or is struck**. Nothing that tells a future reader what to do survives archival
unchanged.

This is the sibling of the campaign-closeout checklist, and it exists for the same reason: several
instances proved the step does not happen on its own. `release-v0.9.0-checklist.md` sat in
`completed/` instructing whoever wrote the next announcement to lead with a claim that was wrong —
a stale instruction, still live, that would have regenerated the wrong text into the next promo
cycle no matter how carefully the drafts were edited. The rules above are what was recovered from that
sweep, plus what this year's retractions cost to learn.

Every file in `docs/completed/` now carries an archival header saying its checkboxes are a record
rather than a task list, so an unticked box no longer reads as outstanding work. That handles the
backlog; this rule is what keeps it from re-accumulating.

## The validation manifest is the source of truth

`testdata/parity_manifest.json` records, per family: the commit it was last
T3-validated at, the measured argmax % / cosine, the loader+forward file set that
defines its numerics surface, the reference, machine, and date. A **model-free CI
test** (T1) hashes each family's declared source set (+ a shared-core hash); if it
differs from the manifest's recorded hash, CI fails with *"parity stale for
<family> — re-run T3 and update the manifest."* This is what turns a silent
`t.Skip` into tracked, visible debt: you always know which families' recorded
numerics predate their current code.

### A `deps_hash` refresh is not a re-validation

The staleness gate goes green two ways, and only one is honest. **Re-validation:** the
family's T3 gate was re-run at the new commit and passed — bump `deps_hash` **and**
`validated_at` (and metrics) together. **Re-hash:** rewrite `deps_hash` to match HEAD
while leaving `validated_at` at a pre-change commit — this silences the gate *without*
re-running anything, converting stale→green by editing the answer key. That is the exact
failure the gate exists to prevent.

**Rule:** a bare `deps_hash` refresh for a `validated` family whose changed files include a
**forward (`forward_*.go`) or core** file is **forbidden**. It is permitted only when
either (a) the gate was re-run at that commit and `validated_at` is bumped with it, or (b)
the changed files are provably non-numeric for that family — the `serialize` and `gpu/`
residency sets, which the deps-split deliberately keeps out of every forward family's
hashed set (so a `.giw`/upload-only edit legitimately doesn't touch their numerics). When
in doubt, re-run the gate; the tool must never edit the answer key.

## Claim discipline (the answer to "models we claim to support")

A family appears as **supported** in the README / capability matrix **only if** it
has *both* a T1 committed golden **and** a current (non-stale) T3 manifest row —
where a `shared-path (via <family>)` row counts as T3, since the staleness gate
binds it to its source family's validation by shared `deps_hash`. A family with
neither, or with a stale row, is labeled **experimental / unverified** until
re-validated. The "every cell with provenance" rule from `benchmarks.md` extends
to the support matrix: no claim without a pointer to its gate.

## Definition of done for a new family (the recurring muscle, completed)

"Descriptor + loader + parity golden" gains an explicit, enforced checklist —
applies to the GLM / Granite work (`completed/task-model-families-glm-granite.md`) and
every family after:

1. Arch adapter + tensor schema (`registry.go`) and loader(s).
2. **T1:** a tiny-synthetic checkpoint test + committed golden (CI, no asset).
3. **T3:** a real-checkpoint parity run (or `weightDiff` if the oracle OOMs, or a
   `shared-path` row if the family aliases an already-validated adapter and shares
   its `deps_hash`), recorded in `parity_manifest.json` at the landing commit.
4. **T2:** the family's smallest real model added to the sweep list.
5. README support-matrix row added **only after** 2–4 exist.

Until 2–4 exist, the family ships as experimental.

## Non-goals

- Running multi-GB checkpoints in CI (impossible and unnecessary — T1 + the
  staleness check cover CI; T2/T3 cover real numerics off-CI).
- Testing the full family × format × quant × backend cross product (the factoring
  above replaces it).
- Bit-identical parity as the universal bar — the gate is argmax-exact + cosine
  (+ the rare-token set where a precision claim is made); lossy paths (quant, KV
  precision, paging) gate against their own opt-in bars, default stays bit-exact.

## The pure-Go CPU reference is bit-identical WITHIN an architecture, not across

The same kind of scope statement as Metal's bit-identity (pinned to machine + OS): the pure-Go CPU
reference — the thing every GPU backend is gated against — is bit-identical **within an architecture,
not across arm64 and amd64**. The divergence is real but decision-irrelevant, and the margin is
measured, not assumed.

**Decision headroom — stated as a threshold.** Cross-box, identical weights + fixed prompt + f32,
qwen2.5-coder-0.5b, 64 greedy positions: **no argmax flip at any position.** A flip requires a **top-2
margin below ~2×10⁻⁵** — twice the maximum absolute divergence measured on the decision-relevant
(winner/runner-up) logits (7.6×10⁻⁶). The **tightest top-2 margin observed across all 64 positions was
0.65** — more than four orders of magnitude above that threshold. The margin would have to be ~10⁴×
tighter than anything measured before the question even becomes live; a challenger now knows exactly
what to look for. And the tightest position was **position 0**, which is the expected shape rather than
a lucky draw: a greedy continuation settles and the model grows more confident, so the first generated
token has the narrowest margin by construction — the run started at the hardest position. (For
calibration against something this codebase has actually hit: MoE top-k routing margins run ~10⁻³, the
tightest margin regime seen here; even there the headroom clears the threshold by ~66×.)

**Mechanism.** Go's spec permits fusing `x*y + z` into a single-rounding FMA "across statements." gc
does this on **arm64** and not on **amd64's default GOAMD64 baseline** (FMA is not in v1): `-gcflags=-S`
shows **85** fused sites in `decoder` and **47** in `aikit/linalg` on arm64 (e.g. `FMADDS` at
`dot.go:25`, the SIMD dot inner loop) — **0** on amd64. A fusion-sensitive `x*y+z` reproducer diverges
on arm64, agrees on amd64. One source, two f32 results: 93% of the 151,936 logits differ bit-for-bit.

**On the tail figure, and why ULP is the secondary metric.** The largest single divergence looks like
91,136 ULP — but that logit has |value| = 2.6e-4 (near zero) and an absolute error of 2.6e-6. ULP
spacing collapses toward zero, so a near-zero logit posts a huge ULP count for a negligible absolute
error. **The absolute error is magnitude-independent** (~1.4e-6 median, ~1.0e-5 max over all logits);
ULP divergence is strongly inversely correlated with |logit| (median 91,136 ULP in the |logit|<1e-3
bucket vs **4** ULP in the 1–10 range where every decision lives). The tail is an artifact of measuring
values near zero, not depth-driven growth — lead with absolute error, carry ULP as secondary.

**The 6.6% that agree bit-for-bit are coincidental, not structural.** 10,057 / 151,936 logits match
exactly, but **none are zeros** — they are ordinary non-zero logits that happened to round identically
under fuse-vs-no-fuse. This corrects an expectation: the resident-vs-CPU comparison found its matches
were all `logit[0] == 0.0` (structural, ≈0.02% agreement), so the same structural explanation was
expected here — it does not hold. Nor are the two agreement rates the same kind of number: fusion
perturbs **one** rounding per multiply-accumulate (6.6% survive), whereas two different reduction
implementations perturb **many** (≈0.02% survive). The cross-architecture effect is the smaller of the
two; do not flatten them together.

**The cross-architecture contract is argmax + cosine** — the gate's existing bar — not bit-for-bit.
Both arches pass every argmax-exact + cosine golden; the divergence lives entirely inside the tolerance,
10⁵× below the decision boundary. The parity manifest's `deps_hash` is a hash of **source bytes +
aikit_version** (architecture-independent), so it needs **no architecture field** — it is valid on both
boxes unchanged. Only a *float-valued* golden would be arch-sensitive, and those are argmax+cosine.

**Remediation declined.** Forcing bit-agreement means explicit `math.FMA` everywhere — a software
fallback on amd64 that would cost the SIMD performance the CPU backend exists for. Not worth it for a
divergence that clears every decision by 10⁵×. Recorded here so nobody re-derives the worry.
