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

## The validation manifest is the source of truth

`testdata/parity_manifest.json` records, per family: the commit it was last
T3-validated at, the measured argmax % / cosine, the loader+forward file set that
defines its numerics surface, the reference, machine, and date. A **model-free CI
test** (T1) hashes each family's declared source set (+ a shared-core hash); if it
differs from the manifest's recorded hash, CI fails with *"parity stale for
<family> — re-run T3 and update the manifest."* This is what turns a silent
`t.Skip` into tracked, visible debt: you always know which families' recorded
numerics predate their current code.

## Claim discipline (the answer to "models we claim to support")

A family appears as **supported** in the README / capability matrix **only if** it
has *both* a T1 committed golden **and** a current (non-stale) T3 manifest row.
A family with neither, or with a stale row, is labeled **experimental /
unverified** until re-validated. The "every cell with provenance" rule from
`benchmarks.md` extends to the support matrix: no claim without a pointer to its
gate.

## Definition of done for a new family (the recurring muscle, completed)

"Descriptor + loader + parity golden" gains an explicit, enforced checklist —
applies to the GLM / Granite work (`completed/task-model-families-glm-granite.md`) and
every family after:

1. Arch adapter + tensor schema (`registry.go`) and loader(s).
2. **T1:** a tiny-synthetic checkpoint test + committed golden (CI, no asset).
3. **T3:** a real-checkpoint parity run (or `weightDiff` if the oracle OOMs),
   recorded in `parity_manifest.json` at the landing commit.
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
