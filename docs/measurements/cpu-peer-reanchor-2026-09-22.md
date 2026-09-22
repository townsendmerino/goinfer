# Linux CPU decode re-anchor, 2026-09-22 — the §B8 CPU column, before/after R9's fixes, against Ollama v0.32.5

R9's "Measure" item (`docs/tasks/red-october.md`): the standing 2026-08-26 CPU column (goinfer 23.5 / 17.6 / 4.9
tok/s on 0.5B / 1.5B / 7B) was shown stale in both directions by the Linux attribution
(`cpu-decode-attribution-2026-09-22-linux.md`). This is the served-harness re-anchor, same protocol as every
§B8 row: `scripts/bench_peer.py`, phase A only (depth 128, greedy, 64 generated tokens, n=2 runs per cell,
spread reported, the harness's loadavg gate between cells), both engines over their own HTTP servers.

**Provenance.** `nobara-pc`, Ryzen 7 3700X (8c/16t), Nobara 44, weights from local NVMe under `~/models`
(the qwen2.5-coder 0.5B/1.5B instruct and qwen2.5 7B instruct q4_K_M GGUFs; Ollama's `q05`/`q15`/`q7b` are the
same tensors per `scripts/gguf_same_weights.py`'s earlier verification). Ollama **v0.32.5** at `~/ollama-0325`,
CPU-forced (`num_gpu 0`). goinfer at int4, three binaries built from this tree: `goinfer` = HEAD (R9's fixes),
`goinfer_old` = `3ea2f93d` (the commit before them), interleaved with Ollama in ONE session so drift cannot
pose as an effect (the harness's before/after design). GPU idle throughout (CPU lane). Date 2026-09-22.
Logs and JSON: `cpu-peer-reanchor-2026-09-22.log` / `.json` (final), `…-pass1.log` / `.json` (the first
full sweep, superseded — see below), `…-05b-*.log` (the 0.5B diagnosis passes).

## The 0.5B: a real, small, unexplained regression — investigated, not resolved

The first full sweep read the 1.5B and 7B as expected (HEAD 17.8 / 4.9 vs old 13.2 / 4.5) but the **0.5B
2.6–3.6% SLOWER on HEAD** across every repeat (37.0–37.5 vs 38.2–38.5) — a model neither R9 fix targets: its
intermediate (4864) is below the activation fan-out threshold either way, and its 7-heads-per-KV geometry
never takes the grouped path regardless of the gate. Reproduced 5 times across independent full-process runs,
never inside one run's own spread (≤ 0.7 tok/s).

**First hypothesis, built and retracted.** A bisection (reverting `decoder/{model,attention,mlp}.go` to
`3ea2f93d` in a worktree, re-adding one file/line at a time) pointed at a single line — the runtime `var`
read in `parallelElementwise`'s entry condition (`!activationFanoutEnabled || n < ...`) — reproducing ~3% loss
in isolation and ~0% with every other file reverted. Folding it into a per-architecture `const` seemed to
fix it. **It did not**: re-tested with a controlled single-line A/B on the *actual* current tree (both arms
built from the identical full R9 codebase, differing only in that one line, same session, same load) —
const 37.5 vs var 37.2, both within their own run-to-run spread of each other. The apparent 3% effect in the
bisection was an artifact of the bisection method, not of that line: reverting three files to `3ea2f93d` and
re-adding one changes far more than the named line (every other diag/gate in those files, and whatever the
compiler does differently with a smaller total diff) — an "isolated" test that touches three files' worth of
surrounding code is not actually isolating the one line it names. The const change was reverted; it fixed
nothing and would have been an unexplained diff kept for the wrong reason.

**What is established:** the 0.5B is reproducibly ~1 tok/s slower on HEAD than on `3ea2f93d`, entirely inside
`decoder/{model,attention,mlp}.go`'s changes (a binary with those three files reverted and everything else at
HEAD matches `3ea2f93d`'s speed). **What is not established:** which specific change, or whether it is a
semantic cost at all rather than a code-layout/inlining side effect of the diff's shape — the one clean,
same-tree, single-variable test that was actually run (const vs var) found no effect, and no other
single-variable test was completed to the same standard before this record was written up. Not chased
further: the effect is small (~1 tok/s, ≤ the box's own run-to-run drift budget noted elsewhere in this repo
as ~3.5%), touches only a model neither shipped fix claims to help, and every further bisection attempt was
itself shown to risk manufacturing exactly this kind of false positive.

## Result

Same-session, interleaved (`goinfer`, `goinfer_old` = `3ea2f93d`, `ollama` v0.32.5), n=2 runs/cell, greedy,
depth 128, 64 generated tokens (log `cpu-peer-reanchor-2026-09-22.log` / `.json`):

| model | goinfer (HEAD) | goinfer (`3ea2f93d`) | Ollama v0.32.5 | HEAD/old | HEAD/Ollama |
|---|---:|---:|---:|---:|---:|
| 0.5B | 37.5 | 38.5 | 57.6 | 0.974 (**unexplained regression, see above**) | 0.651 |
| 1.5B | 17.8 | 13.2 | 24.1 | **1.348** | 0.739 |
| 7B | 4.9 | 4.5 | 6.0 | **1.089** | 0.817 |

The 1.5B and 7B numbers match the attribution record's own paired A/B (1.37×/1.10× measured in-process) to
within run-to-run noise — the served-harness protocol confirms the in-process result, on the models the
fixes target. **This supersedes the stale 2026-08-26 §B8 CPU row** (23.5/17.6/4.9), which was itself
already known-stale before either R9 fix (per the attribution record's own pre-fix baseline: 0.5B ~41,
1.5B ~13.5). None of today's rows are directly comparable to 08-26's fit — different code, and (now
confirmed) the 08-26 numbers were never re-verified same-session against a fresh Ollama pull.

**HEAD still trails Ollama on all three sizes** (0.651×/0.739×/0.817×) — closer than before (the 1.5B moved
from ~0.55× to 0.74×) but not parity; CPU decode remains behind on this box, MLP-bound per the attribution
record, with S-05 (Mac, in progress) the next lever and the isolated-vs-in-token matmul gap on this box
still unexplained (see `cpu-worker-pool-2026-09-22-linux.md` — dispatch is not the cause).
