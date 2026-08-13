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

*And once as process, with the cost now measured rather than argued:*

**The `completed/` folder buried ten fixed findings, and every one of them stayed listed as open.**
The F group — five §4 gates (G-01, G-02, G-04, G-05, G-06) and five §2/§3 criticals (C-05, C-06,
C-08, C-14, C-30, plus C-21/C-22 and C-31) — came from an audit filed under `docs/completed/`. Swept
against the tree on 2026-08-12: **all fixed, none propagated back.** The fixes were real, landed, and
often carried their own named gate; the register that people read still said open.

This was an argument before ("knowledge parked under `completed/` is indistinguishable from knowledge
nobody had") and it is evidence now. The cost is not that the work was lost — it was done — but that
**a decision register accumulated ten false entries**, and correctness and security items are where a
false entry costs most in *both* directions: an open item listed fixed hides a hole, and a fixed item
listed open spends attention on nothing. The remedy shipped with the measurement: every row now
carries a content-keyed citation and CI fails on the commit that makes one stale.

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

### Sibling drift: the fix lands on one member of a pair

Adjacent to G-01 and distinct from it. G-01 is a gate pointed at the wrong path. This one is a
*fix* pointed at the wrong path — or rather, at only half of it.

> **Two code paths that are siblings by construction — the same operation across quantizations,
> backends, or dense versus sparse. A fix or invariant lands on one and not the other, and nothing
> fails when they diverge.**

The pair is usually written in one sitting by one person, which is exactly why it drifts: at the
time of writing the two bodies are obviously the same shape, so neither carries a note saying the
other exists. Months later a fix arrives via one call site. The sibling is not forgotten — it is
never brought to mind at all, and no test is looking, because both members were already covered
by tests that pass just as happily before and after the divergence.

**Two locations, and they are not the same defect.** Everything above is the *check* shape — a gate,
lint or golden that names one member. There is a **dispatch** shape too, in production code:

> **A dispatch names one member of the set instead of dispatching on the property.**

`matmulInto` is the instance (P7): it special-cases `isW8A8(w)` and delegates everything else to
`matmul`, so W4A8 never reaches the per-stream `Workspace` its six call sites already hand in. Nothing
is broken and nothing fails — one quantization silently gets a different implementation.

The two shapes have opposite roles and it is worth keeping them straight: **the dispatch shape
CREATES divergences; the check shape FAILS TO CATCH them.** The same recognition test finds both, and
the remedies differ:

**A QUERY THAT SUCCEEDS AGAINST A HANDLE DOES NOT PROVE THE HANDLE IS LIVE.** Many APIs answer from
a cached descriptor that outlives the thing it describes, so the probe returns a confident, valid,
*wrong* answer.

*Recognition test:* **does this probe have to touch the thing to answer?** If it can be satisfied
from a cache, a descriptor, or metadata, it is not a liveness check — whatever it is named.

*Instance (A13, 2026-08-13).* `cuFuncGetAttribute` on a cached CUDA function handle returned
**byte-identical valid values** in a poisoned run and a clean one — `maxThreadsPerBlock=1024`,
`numRegs=62`, `ptxVersion=75` — because those live in metadata that survives the device code being
evicted. The handle answered while what it named was gone; the launch then reported success and
executed nothing.

**The probe was cheap, it was mine, and ALONE IT WAS WORSE THAN NOTHING** — it returns a confident
negative ("the handle is fine, look elsewhere") and would have sent the investigation away from the
right answer. What settled it was the *pair*: forcing a module reload before the launch fixed the
result, 3/3 against a 2/2 control. **Neither probe alone was sufficient, and the cheap one alone was
actively misleading.** A probe that cannot discriminate should be labelled as such when it is
proposed, not after it has been believed.

**A DOCUMENTED MECHANISM WITH NO ENUMERATION is the same shape, and it is the one that feels
safest.** When a defect is found, understood, written up, and fixed *at the site where it was found*,
the write-up creates a strong impression that the class is handled. It is not: the fix landed on one
member and nothing enumerated the others.

*Instance (A12, 2026-08-13).* `scripts/gpu_gate.sh`'s header documents this exact defect from a
previous incident, in detail — *"the tests DROPPED those errors, and the resulting zero-filled
buffers surfaced as 'cosine 0.000000 — layout/unpack mismatch'. An OOM wore a parity bug's clothes
for long enough that two people independently concluded 'the tests just interfere; they pass
individually' and moved on."* The mechanism was known, the remedy was applied where it was found, and
**nobody enumerated**. `errcheck -blank` then found **268 production sites** across four modules,
including `cuda/`'s decode forward path — and the defect recurred in
`cuda/e2e_decode_test.go`, a file the header does not cover, reproducing the *"they pass
individually"* conclusion a second time, in the same repository, against a document describing it.

Same family as W8A8-fixed/W4A8-missed and `capSlots`' inline copy: **a fix at the instance, with no
check for the other members.** The distinguishing feature here is that the documentation makes it
*less* likely anyone looks — a written-up mechanism reads as a closed one.

*Recognition test:* **when a defect is written up, was the class ENUMERATED or only the instance
fixed?** If the write-up names a mechanism and no list exists, the list is the missing work.

**And a third shape, found 2026-08-12: a CONSTANT restating a value maintained elsewhere.**

> **A hand-maintained literal duplicates a value that is computed or declared somewhere else.
> Sibling drift with a literal as one of the siblings.**

Two instances landed the same day, which is what made it a shape rather than a chore. The manifest's
`aikit_version` field is mixed into `deps_hash` — so changing it re-stales every family, which is the
right design — but nothing *computes* it, and it read `v1.12.0` against a `go.mod` saying `v1.16.0`
(QUEUE B7). And `RELEASING.md`'s version-alignment step named the versions to align on, going stale
**twice** — once telling you to align on what had become a downgrade — before being rewritten to read
them instead (`0898295`).

Being **data rather than code buys it nothing, and in practice buys it less**: no compiler, `vet`,
`staticcheck` or lint reads a literal in a JSON field or a Markdown checklist, so the copy drifts in
total silence. The failure is also quieter than the other two shapes — a stale constant feeding a
computed gate produces a **green that means "nothing asked"** rather than "nothing changed", which is
the absence-of-signal shape one level down.

| shape | remedy |
|---|---|
| check names one member | **enumerate** the members and assert the invariant on all of them |
| dispatch names one member | **dispatch on the property**, not the member — here, "does this path have a reusable Workspace", not "is this W8A8" |
| constant restates a value | **derive it** from the source of truth at build or test time, and fail on disagreement — never restate it |
| **a literal names a CHOICE where the rule means a PROPERTY** | **state the property**, and let the choice follow from it per case |

**The fourth row is the same shape one level up, and it caught a rule written the SAME DAY this class
was named** — which is the strongest argument available that naming a class does not immunise you
against it.

The instance (2026-08-12, P10's pre-registration). The dilution bound is computed by dividing a
benchmark-level figure by the changed path's share of runtime, and the two arms had different shares.
The rule was written as:

> *use the **smaller** share*

...reasoned from the **flat** branch, where a smaller share means a larger admissible hidden effect —
correctly conservative. The result landed on the **win** branch, where dividing by the smaller share
yields a **larger** claimed speedup, i.e. the **least** conservative reading. The literal "smaller"
had been standing in for the property "**conservative**", and the two point at opposite shares
depending on which way the result falls.

**Restated as the property: divide by whichever share makes the resulting claim WEAKER.** On a flat
result that is the smaller share (bigger admissible hidden effect); on a win or a loss it is the
larger (smaller claimed effect). Following the literal would have added two points to a claim about
someone else's work.

Exactly the shape of a version literal standing in for a condition (`≥ v0.13.0` for "carries an aikit
bump"; "the aikit v1.17.0 bump" for "any aikit bump"). A rule that names the *answer* instead of the
*question* is correct only for the case its author had in mind.

The recognition test is the same for all four, and for the constant shape it is one question: **is
this value maintained anywhere else?** For the fourth: **does this rule name a choice, or the reason
for the choice?**

**Tractability, assessed before building a lint rather than after.** The two shapes automate very
differently, and a lint doing only the tractable half would ship as though it covered the class —
reproducing, inside the gate built for the class, exactly the thing the class is about.

- **The check shape is enumerable.** Given a declared sibling set, a test can assert the invariant
  over all members and fail when one is missing. That is a real gate and B6 builds it.
- **The dispatch shape is NOT tractable as a verdict.** Measured on this tree: `decoder/` production
  code contains **one** identity predicate of this form (`isW8A8`, 5 if-sites) and 3 type switches.
  The surface is small enough to enumerate — but `if isW8A8(w)` is **syntactically indistinguishable**
  from a legitimate special case, and 4 of those 5 sites *are* legitimate (W8A8 genuinely has its own
  `QuantBackend` kernel). Only the 5th was the defect. Nothing in the syntax separates them; it takes
  knowing the intended set.

So B6 ships the check half as a gate and the dispatch half as a **census** — `TestDispatchCensus`,
which declares every identity-predicate dispatch site and type switch with a one-line reason and goes
red when the set changes in either direction.

**The census detects CHANGE, not correctness, and that has to be said where the green is read.** A
pass means "the dispatch surface is what it was when a person last reviewed it". It does **not** mean
"no dispatch drift exists" — 4 of the 5 declared sites are legitimate and one was P7's defect, and
nothing in the census could tell them apart. Reading a green as the second thing is the class's own
mistake, one level up, which is why the test logs `GREEN MEANS UNCHANGED, NOT CORRECT` rather than a
bare pass.

The predicate set is **derived from signatures** (`func isX(... *linalg.WeightMat ...) bool`), not
listed, so a predicate that does not exist yet still enters the census — the property that matters,
since the next instance will be named by something nobody has written. Mutation-checked both
directions: a new site goes red, and removing a declared one goes red for the other reason.

**Recognition test:**

> **When a fix lands on one member of a pair, what checks the other?**
> **And when a dispatch names a member, what does the rest of the set get?**

*Remedy shape:* a test that **enumerates** the members and asserts the invariant on all of them.
A test that names one member reproduces the class — it is what the passing sibling already had.
Where enumeration is not mechanical, the invariant's own comment names the full set, so the next
fix is written by someone who has been told the set exists.

**The class is broader than any census of it.** `TestDispatchCensus` covers *dispatch* sites in
`decoder/`, and that is one shape in one language in one package. The sixth instance found was in
**Python, in a gate** — `path_repos()` and `sibling_repos()` were two implementations of "the sibling
set" that disagreed the moment an override was exercised. Neither the dispatch census nor any Go lint
would ever have seen it. So the census's green is qualified twice over: **unchanged, not correct —
and unchanged *within its scope*, which is narrower than the class.**

Instances at time of writing, six, none of them found by a failing test:

| the pair | what drifted |
|---|---|
| W8A8 / W4A8 projection | W8A8 was fixed to reuse a `Workspace`; W4A8 still allocates a fresh one per projection per token |
| dense `mlp` / `moeMLP` | the dense sibling honours the `decodeScratch` invariant; `moeMLP` skips it and allocates ~7–8 MB/token |
| batched GEMV int8 / int4 | fix applied to one quantization |
| `capSlots` / its inline copy in `allocSlots` | production runs the inline copy; the gate tests `capSlots`, so a change to either is uncontradicted by the other |
| SIMD / scalar widen | a SIMD `int8→f32` widen sits in the same package as the scalar one still used by the LM head |
| `path_repos` / `sibling_repos` (Python, in a gate) | two notions of "the sibling repo set"; only one honoured the env override, so a foreign commit read as fabricated while its repo counted as present |

The `capSlots` row is the sharpest, because there the drift is between the shipped path and *the
thing written to check it* — sibling drift and G-01 in the same object. It is also the reason this
class is stated as enumeration rather than diligence: a documented claim that the gate
"corroborates production sizing" survived precisely because both halves passed.

### The measurement's shape

The number is real and the code path is real, but where the instrument sits, or what the comparison
differences away, determines what could have been visible. The reading is then interpreted as though
it described the system rather than the instrument's view of it.

Three shapes:

- **Environment — WHERE it ran.** The check is correct and the context is not, so the result is about
  the context. Two instances, both this campaign, both a *false* answer from a *working* check:
  `gpu_gate.sh`'s module-boundary guard reported `cuda`, `gpu` and `webgpu` leaking into the root
  module graph — an artifact of a committed `go.work` that CI's root job does not have; and the
  goldens refresh reported 7 passed / 35 skipped in a fresh worktree against 33 / 9 in the checkout,
  an artifact of gitignored fixtures. *Recognition test:* **if this ran one environment over, would
  the answer change?** Same test as Position, applied to the surroundings rather than the placement.
  A third instance (2026-08-12) moves the wrong context from *where* to *across what*: `git diff
  v1.16.0..v1.17.0 -- gpu/` reported aikit's quantized GEMV PTX changed by 72 lines, when across
  `gpu/v0.27.0..gpu/v0.28.0` — the tags goinfer actually consumes — it is byte-identical. `gpu/` is a
  nested module with its **own tag series**, and the two series do not track: the new tags are the
  same commit, the old ones are not. Diffing a nested module across the *parent's* tags spans commits
  the consumer already had, re-reporting a weeks-old shipped change as new. **Diff a nested module
  across its own tags.** *Recognition test for the variant:* is the boundary I measured across the
  same boundary the consumer sees? Nothing gates this; it was caught only because the number
  contradicted a claim already written down.
- **Position** — the probe sits on one side of the event and reports the other side's state.
  *Recognition test:* if the probe were one line earlier or later, would the number change? If yes,
  the position is part of the claim and has to be stated with it.
- **Differencing** — a delta between two configurations cannot see a cost that does not scale with
  the configuration. *Recognition test:* what cancels? Where a sweep compares configurations, **at
  least one absolute measurement is required before any mechanism is proposed.**

*Remedy shape:* the measured-quantities rule below already requires machine, method and date. Add
**probe position** for any figure whose value depends on it — which call it was taken before or
after.

Instances at time of writing, all 2026-08-12, three readings and one shape:

- the **ladder ceilings** — contiguity reported as a rising fraction of free (36% → 50% → 60%) when
  the ladder stopped at first success against a falling denominator, so the trend was the
  instrument's, not the heap's;
- the **cross-run deltas** — per-slot cost derived from between-configuration differences, which
  cancel any fixed cost exactly, however many configurations are sampled;
- **free-at-failing-launch** read from `describeLaunchErr`, which is reached only *after*
  `Launch` has returned non-nil and therefore reports the **post-failure** state
  (265,945,088 = 198,836,224 + exactly 64 MiB, the first-launch figure — an exact 2^26 that reads as
  a driver block unwinding rather than as application scratch released).

Each was stated as a fact about the system before the instrument's position was checked.

**A check has a measurement's shape too, and a false RED only by luck.** `scripts/gpu_gate.sh`'s
repo-hygiene group was rewritten to run CI's checks derived from `ci.yml` rather than a hand-written
copy (B0). On its first run the module-boundary guard reported a **leak of `cuda`, `gpu` and `webgpu`
into the root module graph** — a serious-looking red that was entirely an artifact of *where the
check ran*: CI's root job has no `go.work`, this box has one committed, and a workspace unions every
submodule into the graph the guard inspects. Same command, same repo, opposite verdict.

Two things make it worth recording here rather than in the gate's own notes. The recognition test is
the position one, applied to a check instead of a probe: *if this ran one environment over, would the
answer change?* And the failure direction was **luck** — a workspace makes the module-boundary guard
report a false red, but a check whose environment hides a condition would report a false green just as
silently, and nothing about the setup chose which. **Reproducing a check's command without
reproducing its environment is not reproducing the check**, and the environment therefore belongs in
the derivation, not in a developer's habits.

**The third one was then settled by moving the probe**, which is the remedy the class prescribes and
worth recording because it came out on the artifact side. A reading taken immediately *before* each
`cuLaunchKernel` returned 198,836,224 B at every one of the 20 launches of the token, including
immediately before the failing `fRoute` — identical to free after `allocSlots`. Nothing was released
between launches; the 64 MiB appears only after the failure, so "64 MiB was released" was a fact
about where the probe sat and not about the system. The claim never reached a document.

**And the instrument written to settle it had the same defect, one level down.** Its trigger was
specified as two decrements — free after `allocSlots` → free at first launch → free at failing
launch — on the reasoning that a deferred module load would show up as a gap between them. Every
gap came back **0**, which by that wording reads as "module loading cost nothing". It does not: under
the driver's default lazy module loading the module under suspicion materialises *during* the launch
that fails, which is after the last pre-launch reading and before the post-failure one. **No
difference of those three readings can contain it.** The zero was the instrument's blind spot
presented as a measurement — the differencing shape, in a probe written by someone who had just
finished writing this section. Recognition tests do not fire on their own; they have to be applied to
the instrument being built, not only to the one being criticised.

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

**A gate whose value depends on an axis must PRINT its composition along that axis.** The
Representative rule above is a design-time property — the gate varies over the axis it protects.
Nothing made a gate *say* that it had, and the gap is not cosmetic: the forward goldens reported
"19 passed" through **nine** deps_hash refreshes while every one of those 19 was f32, on a runtime
whose documented default quantization is int4. The count was accurate. It was also the only thing
anyone read, and it cannot distinguish 19 f32 goldens from 19 that span three quantizations.

Same shape as the skip census, one level along: a skip census exists because "0 failures" cannot
distinguish "nothing failed" from "nothing ran", and this exists because "19 passed" cannot
distinguish "the axis is covered" from "the axis collapsed to one value".

*Recognition test:* **name the axis the gate is supposed to vary over, then read its output — can you
tell from the output alone what values it actually covered?**

Audited 2026-08-12. Reporting the composition: the goldens refresh (quantization), `gpu_gate.sh`'s
skip census (run-vs-skipped) and its derived hygiene group (platform), `TestInt4_forwardParity`
(fixture), `TestKernelLocalMemoryCensus` (kernel). **Not reporting it:** `parity_sweep.sh` (family ×
quant × loader — pass/fail per gate, no quant column), `TestSlotAllocation_matchesGranularityForm`
(slot count — one per run, absent from the verdict), `TestApplySoftcap_bitIdentical` (size ×
GOMAXPROCS — loops both, prints neither), `TestMoERouteDemandThreshold` (balloon shape — an env var
absent from the verdict line). The capability matrix (family × backend) is unread. Listed rather than
fixed: the audit is the deliverable, and `parity_sweep.sh` is the one that matters most, because it
is the release gate.

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

   **SCOPE — this governs ANY arithmetic over two readings, not only comparisons that get
   published.** The rule was written for headline claims and therefore did not fire on an internal
   delta inside an experiment, which is the same defect one layer down and harder to see, because
   nobody reviews a subtraction the way they review a number in a README.

   *Instance.* A10's per-context cost was computed as `pre − post` with **`pre` from `nvidia-smi` and
   `post` from `cuMemGetInfo`** — two instruments that disagree by up to a MiB of granularity. It
   gave **107,806,720 B**. The same instrument on both sides gives **106,954,752 B**, and the
   decomposition only closes with the latter: 44,236,800 + 106,954,752 = 151,191,552 exactly, where
   the mixed figure left 43,384,832 and no clean split. The discrepancy was *visible* in the log the
   whole time — 851,968 B between the two readings — and was written off as "does not affect the
   delta, since that is computed within one process". It was computed within one process and **across
   two instruments**, which is the thing the rule is about.

   So: **name the instrument on both sides of every subtraction**, including the ones that never
   leave the test output. A delta between two instruments measures their disagreement as well as the
   quantity.

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

**"Could what I ran have found it?" is a question for prose sweeps, not only for tooling.** The
citation lint was taught, this same day, that a repository which cannot be searched is a hard error
rather than a negative result — after `git -C` against a non-repo path reported a valid commit as
fabricated. Hours later a manual sweep of the F group recorded C-30 as "unverifiable, names a paging
path that is not a file", having globbed `decoder/paging*.go` when the files are `layerpaging.go` and
`moepaging.go`. Same error, no tool involved.

*Recognition test:* not "did I look" but **"could the thing I ran have matched the thing I was looking
for?"** A glob, a grep pattern, or a directory choice is a search scope, and a scope that excludes the
target produces a confident absence.

**A claim that arrives with its own corroborating detail has not been corroborated.** The supporting
fact came from the same source as the claim, so it adds *texture* rather than evidence — and texture
is precisely what a reader uses to judge whether something was researched. A bare assertion invites a
check; an assertion with a specific-sounding detail beside it does not.

*Recognition test:* **does the corroborating detail come from a DIFFERENT source than the claim, or
from the same one?**

*Instance.* `9e5f8fa` was cited across two documents as the goldens-refresh precedent, alongside "a
metadata field addition re-staled `decoder/weights.go` and the refresh ran 19 goldens". Neither was
true: that commit is `fix(quant): reject --quant that conflicts with a prequant .giw` and touches the
manifest not at all, and **none of the nine real refreshes touches `decoder/weights.go`**. The pair
read as researched and survived weeks of citation. Either half alone would have read as a guess and
been checked.

*A SECOND INSTANCE, and a bigger one (A12, 2026-08-13) — SHARED PRECONDITION IS NOT SHARED
MECHANISM.* Four CUDA tests failed in one run. `scripts/gpu_gate.sh`'s header supplies a mechanism
for one of the symptoms — *"parallel packages contend for VRAM and the failures come back as bogus
numerics (cosine 0.000000)"* — and that sentence, written from a real past incident, was **inherited
rather than measured**. It framed every candidate for a day: leak, per-test isolation, partitioning,
async teardown, per-context reservations. **Three explanations died inside the frame** before anyone
grepped the run for a CUDA error and found **none**.

The four shared only a *precondition* — the same long suite on the same card. Their mechanisms were
four different things: an unchecked type assertion panicking on a **correct, designed decline**; a
test that prints *"this run says nothing about the probe"* and reports FAIL; a test requiring it be
the **first to launch a kernel in the process**, with nothing enforcing that; and one genuine
unknown, which turned out to be **every CUDA call's error discarded with `_ =`**, leaving an
unwritten buffer to read as `cosine 0.000000`.

This is the same shape as the rule above, one level up: the header's detail came from the *same
source* as the framing, so it read as researched. A bare "these fail together" would have invited the
grep on day one.

*Recognition test for the variant:* **do these observations share a mechanism, or only a
precondition?** Co-occurrence in one run is a precondition. And when a document hands you a
mechanism for a symptom, check whether that document *measured this instance* or is describing a
past one.

*Where the check goes:* **into the description trigger, not into a new sweep.** When an entry is
re-read against its source at pickup, the specific details inside it — counts, file names,
measurements that no lint covers — get the same read as the description. That is where D3's detail
already was, and it is the only moment at which someone has the source open anyway.

**A read-only question gets a throwaway worktree by default, not by judgment.** Asking "what would
this tool change?" with the tool that changes things is a category error, and the standing form is
cheap enough that no judgment call is warranted: `git worktree add`, run it there, read the diff,
remove the worktree.

**And the boundary, which is not optional: a fresh worktree lacks everything gitignored** — fixtures,
checkpoints, caches, build artifacts. That makes it safe for **reading code** and unsafe for **running
anything that consumes untracked assets**. A measurement taken in one is about *the worktree*, unless
every asset it needs is tracked. Measured instance: the same commit's goldens refresh reported
**33 passed / 9 skipped** in the main checkout and **7 / 35** in a fresh worktree, because the fixture
checkpoints are gitignored — and `goldens=7` would have gone into a commit body reading exactly like
`goldens=33`. The skip-ratio warning in `scripts/refresh_parity_hashes.sh` catches that one instance;
this rule is for the next one.

The instance: `go fix ./decoder/` was run against the real tree to answer exactly that question — a
tool the standing constraints forbid, reached for because it was the obvious way to see the answer.
Three checks then established the tree was unmodified, and **only one of them actually established
it**: `git status` and `git diff HEAD` both only show agreement with HEAD, which a modify-and-restore
also satisfies. The **byte comparison against a snapshot taken before the command** is the one that
carried the claim. Worth naming, because two of the three checks were reassurance rather than
evidence, and a reader counting three would have over-weighted the result.

**G-01 has a third variant: CORRECTLY SCOPED, SILENTLY NARROW.** Distinct from the tautological
gate (cannot fail) and from exercised-but-never-triggered (runs, never reaches the condition), and it
is **the one that looks most like a working gate** — because it is one. It runs, it can fail, it
tests a real property, and its scope is a fraction of the claim readers take it to support.

`TestResidentCloseFreesVRAM` is the instance: three Load+Forward+Close cycles asserting used ≤
baseline+128 MiB. Well built and green. It covers the **0.5B** model with a **single one-token
forward**, while the claim it gets read as supporting is "`Close()` frees the model's VRAM" —
across every model and workload, three orders of magnitude wider.

*Recognition test:* **name the axes the gate actually ran on, then ask what it is being cited for.**
Model size, workload shape, quantization, backend — a gate that fixes all four and a claim that fixes
none are not the same statement.

*Remedy:* **print the scope with the verdict**, exactly as a gate whose value depends on an axis must
print its composition along that axis. This is the axis-composition rule applied to a *resource* gate
rather than to a parity matrix, and it costs one `Logf`.

*Honesty note attached to this instance (2026-08-13):* the narrowness was real, but the defect it was
first cited for was not — the wider gate at 7B came back green, and the "leak" it was supposed to
have missed turned out to be a probe-position artifact. A gate being narrow does not imply something
is hiding in the gap. Both halves of that belong in the record.

**When a pre-registration compares a FRESH measurement against a RECORDED one, the branch set must
include "THE RECORDED VALUE IS THE ERROR."** A recorded number is a measurement too — taken once,
possibly under conditions nobody wrote down — and treating it as the fixed point smuggles in an
assumption that the machine is what changed.

The instance (A11, 2026-08-12). `TestMoERouteDemandThreshold` failed with both bounds moved by
exactly 589,824 B. Three branches were pre-registered — *the floor moved*, *the residual moved*, or
*the floor+residual identity is wrong* — and **the actual outcome was outside all three**, because
all three assumed the machine moved. What had happened:

    floor     unchanged
    residual  unchanged
    demand    now equals floor + residual, to the byte

The **old pin** was the outlier, recorded from the one measurement that did not close, with the
589,824 shortfall misattributed to "baseline drift" (A9-RESID) rather than read as *a failure to
close*. The new measurement was not a regression; it was the identity finally holding.

*Recognition test:* **is the recorded value load-bearing, and was it ever re-measured?** If it was
recorded once and never reproduced, it is a hypothesis with a number attached.

*Remedy, and it generalises past this case:* **where a closed form relates the quantities, check the
IDENTITY before treating a mismatch as a change.** If the components still satisfy it, the mismatch
is in the record rather than in the machine. Prefer a relation over a constant when pinning at all —
a pinned identity survives conditions a pinned scalar does not.

**A correctness argument with PER-ARCHITECTURE BRANCHES is verified per-branch.** Verifying it on
one architecture verifies **one branch**, and the branches may not be equally strong — so *"the
argument was verified"* carries a scope exactly the way a gate's axis composition does, and must be
stated with it.

The instance (2026-08-12, aikit's f32 blocked-matmul rework, `linalg/matmul_blocked.go`). The comment
justifies bit-identity **per architecture**:

- **amd64** — the removed round trip was "32 adds of which 24 added literal `0.0`", and adding `0.0`
  is exact in IEEE-754. **Structural**: it cannot move a bit whatever the inputs.
- **arm64** — "the four lanes per column are **real partial sums**" folded "in this same
  left-to-right order". An **ordering claim about the new implementation**. f32 addition is not
  associative, so it holds only as long as that order actually holds.

goinfer's goldens went green on amd64 — which exercised the **structural** branch, the one that could
not have failed. **The weaker branch is the one nothing tested**, and a green that reads as "the
bit-identity argument was verified" silently generalises from the strong branch to the weak one.

*Recognition test:* **does the argument say "on X… on Y…"? Then a pass on X is a pass about X.** Same
question as the axis-composition rule — a result whose value depends on an axis must state its
position on that axis — applied to the reasoning rather than to the test matrix. The remedy is the
same too: run the other branch, or state the scope.

**An instruction that arrives GARBLED or TRUNCATED is not evidence of intent — quote it back and
stop.** Relayed messages can arrive mangled. When one does, reproduce **exactly what arrived**, say
what it was probably meant to be, and **wait** — do not infer through it and act.

The instance: two messages reached this session as `"geton top"` and `"get on tip"`, and both were
acted on by inference. It happened to be harmless; the reasoning was not. A garbled instruction acted
on confidently is **worse than a delay**, because the delay is visible and the misreading is not —
and unlike every other failure mode in this document, **no gate covers it.** Confidence in a reading
is not evidence about the text, and a short message carries fewer bits with which to be wrong.

**When a citation check goes red, the fix may be the CITATION or the PROSE — and the citation is
always cheaper.** Re-pointing a reference is one `sed`; correcting the sentence around it means
rereading what was claimed and deciding whether it still holds. Both turn the gate green, and only
one of them is honest when the prose is what went stale. **The pressure toward the cheap fix is
strongest exactly when CI is red and everything else is green**, which is also the moment the choice
matters most — a wrong citation is a broken pointer, but a re-pointed citation under an unchanged
sentence is a false statement with a working link, and the lint will never flag it again.

The instance: `c8b65ba` failed in CI, and its rebased successor `bacc04c` sat on `main` carrying the
**same subject and the same five files**. One `sed` from green. Their **patch-IDs differ**, so they
are not the same change, and the passage cites the branch as it stood *before* the rebase — a state
`bacc04c` no longer illustrates. Re-pointing would have gone green while making the sentence false.
Allowlisted with the reason instead. *Ask which one actually went stale, and prefer the answer that
costs more.*

**Read a recorded figure; do not retype it — and note WHY that works: the early write-down is what
catches you.** Two retyped-number errors landed the same day (2026-08-12): an extraction that
silently compared two **empty** files and printed a false match, and a conversion that used one A/B
run's samples for a different run, producing +0.916% where the recorded value was +0.43%.

**Both were caught by comparison against something already written down** — the recorded raw sample
table, and P9's recorded delta. Neither was caught by noticing at the time, and neither would have
been catchable at all had the earlier result not been committed with its raw numbers.

So the remedy is **not only** "read rather than retype". It is an argument for **writing the raw data
down early**, because a recorded figure is what makes a later mistyping *detectable* rather than
merely *avoidable*. A measurement recorded with its samples is a check on every future statement
about it; a measurement recorded only as a conclusion is not.

**A claim that a check passed names the COMMITTED check that produced it.** Gates police committed
files; a command typed in a session is outside every gate, and no gate can be added that fixes that.
So the rule is about the claim rather than the tooling: "staticcheck is clean" is only reportable if
a committed check ran it — a script, a CI step, a Makefile target. An **ad-hoc command is evidence
for the person who ran it and for nobody else**, because nobody else can re-run it, and because its
failure modes are invisible in the transcript.

The instance: `command -v staticcheck >/dev/null && staticcheck ... | head`, typed at a prompt. The
binary was not on `PATH`, the `&&` short-circuited, the whole check evaluated to nothing, and it was
reported as clean. Re-running it as CI invokes it was clean too — so the conclusion survived and the
*claim* had been unfounded when it was made. That is the failure this rule addresses: not a wrong
answer, an unearned one.

**Every number stated in a check's documentation must be derived, not asserted.** The mutation
requirement above makes a gate demonstrate a red; this makes its *description* checkable too,
because that description is what the next person reads to decide whether the check still means
what it says. A slot-cap gate shipped with a comment claiming "removing the margin yields 35 not
34". The assertion beside it was sound — it only tested that the value differed — but 35 was a
figure nobody had computed. The real answer is 38: one slot costs 30 layers x 3.49 MB = 104.7 MB,
so a 384 MB margin buys 3.67 of them and the cap moves by 4. An unverified number inside the check
whose job is being checkable is the same defect one layer out, and the assertion and the prose fail
INDEPENDENTLY — a green test does not vet its own comment.

**Keep a table of measured quantities; every new model must reproduce all of them.** Seven
mechanism claims were made in one day on a single defect. Six were caught by a reader. The seventh
was caught by arithmetic: a proposed accounting model predicted 276.8 MB per slot where an earlier
sweep had measured ~106 MB, and a 2.5x disagreement with an existing number cannot be argued with.

That is the cheap detector, and it is the one that scales — it does not require a second person.
Record what has actually been measured, with its units and the conditions, and make reproducing
every entry a precondition for any new account. A model that explains the failure but contradicts
a measurement is not a candidate.

**But record what each measurement can and cannot support** — see "the measurement's shape" above,
of which the sweep that could not derive a per-slot cost is the differencing instance.

**An absence of signal is not a positive state.** Distinct from premature mechanism, which names
a cause too early; this one assigns a benign meaning to a silence that is consistent with several
states. Five instances, all the same shape:

| the absence | read as | also consistent with |
|---|---|---|
| `no tokens generated` | "generation produced nothing" | a swallowed error, EOS on token 1, a routing result selecting nothing, an unlogged exit condition |
| a skip census of `0` | "nothing skipped" | `go test` printing no `--- SKIP` without `-v` — nobody looked |
| a process "still running" | "progressing" | a wait condition that can never become false; it never started |
| a probe recording nothing | "no launches happened" | the probe's env var read at package-init, before `t.Setenv` ran |
| **an experiment's null under a forcing flag** | "the forced thing is excluded" | **the flag never engaged, so nothing was forced** |
| a guard's tool not on `PATH` | "the check passed" | `command -v tool && tool ...` short-circuiting the whole `&&` chain into silence |

The last one is the sharpest, because the null was the *designed output* of the experiment. A 26B run
under `CUDA_MODULE_LOADING=EAGER` came back byte-identical to the default run and failed identically,
which reads as "module load excluded" — one of the branches pre-registered for it. A control taking
under a second showed the flag changes no reading at all on this driver and path: it never fired, so
its null was about the flag, not about module loading. **A forcing mechanism has to be shown to fire
before a null from it means anything**, and that check belongs in the experiment rather than after
its result has been written down.

That control also relocated the answer. Measuring each step directly on a fresh context — no model,
no cache, under a second — found the deferred cost the experiment was looking for, and it was not the
one the experiment named: `moe_route`'s first launch retains 138,412,032 B of **local memory**, while
module memory is a true zero on both instruments. The five-minute 26B run had been the wrong
instrument for a question that was never model-dependent.

**And that answer was still not the cause.** Its own recognition test, alongside the ones sibling
drift and the measurement's shape carry:

> **Put the measured figure into the observed arithmetic — does it reproduce the observation?**

Draft-checkable, and it does not need the answer known first: it is subtraction against numbers
already written down. Here it fails by 60,424,192 B. This is the failure mode of a *satisfying*
result. 138,412,032 B was a real, twice-confirmed measurement, and it
was adopted as the explanation without checking that it explained the observation. It does not: free
before the failing launch was 198,836,224, which exceeds it by 60,424,192. The refuting arithmetic was
sitting in the numbers already written down, including a post-failure reading 67,108,864 B *above* the
pre-attempt level — which an unwind cannot produce. Measuring the demand directly put it at
289,013,760 B, 2.09× the residual. **A measurement that is real, reproducible and about the right
object can still not be the cause; the check is whether it reproduces the observation, and it is a
different check from whether the measurement is sound.**

The last-mile control there is worth naming too. Three identical repeats look like capacity rather
than contiguity, but a deterministic balloon produces a deterministic layout, so identical repeats
only exclude run-to-run noise. The discriminating variable is the balloon's *shape*: same free bytes,
many small blocks instead of a few large ones. **Vary what the hypothesis says should not matter, not
just the seed.**

The third cost 27 minutes and produced no measurement, while being reported on twice as progress —
including a confident account of what its *duration implied*. The discriminating check (GPU at 0%,
370 MiB) was one command away and stopped seeming necessary once a plausible story existed.

**The rule: any run longer than a few minutes emits periodic progress**, so that silence is
unambiguous. A heartbeat, or the underlying signal — GPU utilisation, tokens so far, elapsed loads
— at an interval. Not a final result with nothing before it. Then "still running" is a reading
rather than an inference.

This was already fixed once, as an instance rather than a property: the heavy-tier gate group
buffered 28 minutes of silence so that a hung run and a working run were byte-identical from
outside, and it was corrected there. The chaining mechanism had the same defect and it went
unnoticed four days later, because the fix had been applied to one runner instead of made a rule.

**One signal is never enough, and the rule applies to itself.** "GPU at 0%" was the check that
would have caught the dead wrapper, and on its own it is still ambiguous — three states share it:

| GPU util | device memory | host CPU | state |
|---|---|---|---|
| 0% | climbing | busy | LOADING (pinned-host allocation, PCIe-bound — the 26B spends ~7 min here) |
| 0% | static | busy | HOST-SIDE WORK — a CPU-path fallback or any long host phase. Not a hang. |
| 0% | static | flat | STALLED or never started |

So the hang signature is **GPU 0% AND device memory static AND host CPU flat**, past the
threshold. `/proc/<pid>/stat` fields 14+15 (utime+stime) give the third in one line. Each signal
alone assigns a benign or alarming meaning to an absence; together they discriminate — which is
this entry's own argument applied one level further than the entry first went.

Same argument as the unconditional decline reason: **a state that cannot report itself will be
reported as whatever the observer finds plausible.** Prefer a bound that yields evidence at the
boundary — `go test -timeout` dumps goroutine stacks — over a wall-clock rule that yields only
"too long".

**If a summarising noun appears before an interval, the noun is the claim and the interval is
missing.** "Fragmentation", "capacity shortfall", "different arithmetic by construction" — each of
those was written when there was a *suggestive* number and no *discriminating* one, and each was
overturned by the next measurement. The word arrives before the evidence, and once written it
reads as concluded: a named mechanism in prose is indistinguishable from a diagnosed one.

The rule: **state the measurement and its bounds; let the name wait for the control.** "Largest
contiguous block is in [96, 128) MiB with 189.8 MiB free, against a 133.5 MiB demand" survives
scrutiny; "fragmentation" does not, and in this case was refuted by a control that produced the
reverse of its prediction — a fresh heap filled by ten large allocations had *worse* contiguity
(32–64 MiB) than the one filled by two thousand small ones.

This is catchable in a draft, which "be more careful" is not: scan for the noun, and check whether
an interval precedes it.

**A shared precondition is not a shared mechanism.** Two failures that print the same line are
not thereby one bug. `TestGemma4_26B_1bBound` and `TestGemma4_26B_cache_B` both logged
`capping to 34` and were grouped as a single defect for a day — but the cap runs UPSTREAM of both,
so that line is a precondition they inherit, not evidence they share a cause. They fail in
different places (`cuLaunchKernel: CUDA_ERROR_OUT_OF_MEMORY` versus a generation loop returning
nothing) on different shapes (one token at position 0 versus a 27-token prefill plus 64 decode
steps with the expert cache cycling), and an out-of-memory condition announces itself where a
routing or eviction defect does not.

This is a distinct variant of premature mechanism: not inferring a cause with no evidence, but
inferring ONE cause from two symptoms whose only common element is something both were configured
into. The check is to ask what the shared line actually is — an output of the code under
suspicion, or an input both received.

It surfaced only because a reader happened to redo the arithmetic. That is not a detection
mechanism, so it becomes a rule: show the derivation, or assert the number in the test.

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
`linalg/dot.go:25`, the SIMD dot inner loop) — **0** on amd64. A fusion-sensitive `x*y+z` reproducer diverges
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

### The contracting-arch fixture gap — a standing structural cost every float-rewrite hits

A corollary of the above, filed so the next instance starts from a known price. A change that
**rewrites a float expression** (a reworked matmul inner loop, a `minmax`, a reassociated sum) can
pass the argmax+cosine goldens on amd64 and breach on **arm64** — the arch that fuses `x*y+z` (the
arch exception, `RELEASING.md` §C1). So the goldens for such a change **must run on arm64**. But the
tiny fixtures the goldens need are **gitignored**, and the arm64 machine typically starts with none.

**Name the trap precisely: it is not that the fixtures were absent — it is that their absence rendered
as 31 silent skips instead of a message.** A skip is a *legitimate* outcome for a machine that
**cannot** run something (no GPU, no real checkpoint). It is the *wrong* outcome for a machine that
**could** run it after one command. First seen 2026-08-13 (aikit v1.17.0 f32 rework): the initial
arm64 run rendered 8 f32 green and 31 skips — and read as "partly covered" when the true state was
"one `rsync`/regen away from full." The defect is the silent skip, not the missing bytes.

**Remedies, cheapest-and-best first — the price is measured, not estimated:**

1. **Regenerate on demand — the best permanent fix, zero repo cost.** `pin_*_tiny.py` is **seeded**
   (`torch.manual_seed(0)`), deterministic, and runs in **seconds** with no network. So the right
   behaviour is for the gate to **generate what is missing, or fail naming the exact command**, rather
   than skip. A fixture that is one deterministic command away should never render as a silent skip.
   This costs nothing in the repo and removes the trap at its root (the skip), not just its symptom.
2. **`rsync` ~38M between machines — works today, minutes.** Measured 2026-08-13 (box→mac): the 14
   synthetic dirs the arm64 machine lacked, ~38M, a few minutes. Needs both machines up. This is the
   *today* fix; regeneration (1) is the *standing* one.
3. **Commit the fixtures — the MOST EXPENSIVE, despite reading as "permanent."** `testdata/` ships in
   the module, so ~38M reaches **every `go get`, every proxy copy, and every clone, forever** — for a
   runtime whose whole pitch is "deploy by copying one static binary." Do **not** pick this because it
   is labelled permanent: regeneration (1) is permanent *and* free. Committing is justified only for
   the rare fixture that is both CI-load-bearing and cannot be regenerated (the `mixtral-tiny` case —
   see "The committed set (chosen, not accidental)"), never as the default answer to a skip.

**Not part of this gap — a separate, arch-INDEPENDENT tier:** the *real* checkpoints
(`qwen2.5-0.5b`, `qwen3-1.7b`, `tinymistral-248m`, `gpt2`, `llama-3.2-1b`, `gemma-3-270m`) are absent
on **both** machines (they need HF downloads and are not kept), so those families skip on amd64 too.
Their skip *is* the legitimate kind (no local checkpoint, not one command away). Do not conflate them
with the arch trap: transferring or regenerating synthetic fixtures does not fetch these, and their
absence is not an arm64-specific coverage hole.
