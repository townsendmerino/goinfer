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
