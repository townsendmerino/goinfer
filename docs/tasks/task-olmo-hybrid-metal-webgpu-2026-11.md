# Olmo Hybrid on Metal and WebGPU: NegEigval and ONormEps in their DeltaNet (2026-11)

**Registered 2026-11-02, before any code (mac-sb11m; the commit's time is the registration's).** Francis's direction of 09:52 PST,
relayed by the coordinator: do on Metal, then on WebGPU, what #604 does on CUDA (`docs/tasks/task-olmo-hybrid-cuda-2026-11.md`).
Implement `linear_allow_neg_eigval` (NegEigval, the write gate doubled to [0, 2)) and the output norm's own epsilon (ONormEps) in the
backend's Gated-DeltaNet, gate it against the CPU, then claim both in the backend's table, so Olmo Hybrid runs resident again.
One PR per backend.

<!-- ledger -->
| id | state | next | evidence |
|---|---|---|---|
| OM | active | Metal: review | this doc, § Metal, readings |
| OW | decision | Francis: Olmo Hybrid was never resident on WebGPU, so this is a five-feature port, not a restore; not started | this doc, § WebGPU |

## Metal (OM)

**Why.** OP4m-c1 made `program.Metal` the build's admission table. Metal's DeltaNet reads neither setting, so Olmo Hybrid declines to
the CPU (`TestOlmoHybridDeclinesOnMetal`). The CPU computes `beta = 2·sigmoid(b)` and the gated output RMSNorm at 1e-5, beside a
`rms_norm_eps` of 1e-6.

**What changes.**
- `delta_gates` and `delta_gates_rows` take a `betaScale`, 1 or 2, and multiply the sigmoid by it. Both multiplications are exact, so
  every other DeltaNet family is unchanged bit for bit.
- The `delta_gnorm` and `delta_gnorm_rows` dispatches (decode, the batched prefill, and the step's per-row DeltaNet, which is decode's)
  read the DeltaNet's own epsilon instead of the model's.
- Both come from the program's `OpDeltaNet` parameters at build. A model whose DeltaNet layers disagree on them declines.
- `program.Metal` sets `DeltaNegEigval` and `DeltaONormEps`. The step tables do not: Olmo Hybrid stays off the multi-sequence step.
- **Named outcome change:** `olmo_hybrid` goes resident on Metal, at int4. Metal runs no recurrent family at int8 (G44).

**Gates.**
- **OM-1, whole model, tiny (Metal GPU, seconds).** `olmo_hybrid-tiny`, Metal resident against the CPU at int4 on both sides, 128
  tokens, at the bar of `TestQwen35ResidentParityMetal` and #604's O-2: worst cosine at least 0.95, drift (token 1 minus worst) at most
  0.02, and the replay after `Reset` at a self-cosine of at least 0.9999999. The build must be resident (non-vacuity).
  - **Plants, each setting dropped at the build:** both must be red at this bar. #604's O-2 read both red on this fixture on CUDA
    (NegEigval 0.889, ONormEps 0.620). If the ONormEps plant is not red here, OM-2 must carry it before the claim.
  - **Red first:** before the change, the test fails on the decline.
- **OM-2, the prompt route.** ~~A prompt at or above the pass's floor equals the sequential decode's last-row logits within the bar
  the DeltaNet prefill test holds, on `olmo_hybrid-tiny`, with both plants red. If no such test admits this fixture today, it does after.~~
  **Amended before any code or reading:** Metal's batched DeltaNet prefill does not take a post-only model, so Olmo Hybrid's prompts
  run the sequential loop on decode's kernels, which OM-1 measures. Extending the pass to post-only layers is not this item.
  OM-2 asserts that `olmo_hybrid-tiny`'s prefill is not batched, so OM-1 covers its prompts. The rows kernels take the new arguments
  for every family, and OM-4 holds the other DeltaNet families identical through them.
- **OM-3, a real checkpoint.** None is on the Mac (`~/models` has no Olmo Hybrid). **NOT RUN, named.** That is not a pass.
- **OM-4, identity.** `gate identity -backend metal <main base> <head>`: every family that ran stays IDENTICAL, with `qwen3_5`,
  `qwen3_5_moe` and `qwen3_next` among them. `olmo_hybrid`'s cells may change, since their backend changes. They are listed.
- **OM-5, the censuses.** The admission and decision censuses are regenerated, and their diffs touch only `olmo_hybrid` lines.
  Then `touched.py`, `go run ./cmd/lint` and the citation lint.

**Void if:** a GPU run outside the timing lock; a NOT RUN counted as a pass; a bar moved after a reading.
**Stopping rule:** one run each. **GPU time:** a few minutes, under the shared lock.

### Readings, 2026-11-02 (mac-sb11m, branch `olmo-metal-dn-2026-11-02`; shared Mac lock, 13:50 to 14:11 PST)

Tests 0939fa47, code 92fabc7b. Logs are in `docs/measurements/olmo-hybrid-metal-2026-11/`.

- **OM-1: PASS** (`om1-om2-tiny.log`). `olmo_hybrid-tiny`, Metal resident against the CPU at int4 on both sides, 128 tokens:
  - worst cosine 0.999527 (token 1 0.999848, drift 0.0003), argmax 123/128, replay 1.000000000.
  - **Red first:** at the tests' commit, all four subtests fail on the decline naming NegEigval.
  - **Plants, both outside the bar:** NegEigval dropped, worst 0.923217 with drift 0.0756. ONormEps dropped, worst 0.540932 with drift 0.4428. So OM-2 did not need to carry ONormEps.
- **OM-2: PASS.** `olmo_hybrid-tiny`'s prefill is not batched on Metal, so OM-1 covers its prompts.
- **OM-3: NOT RUN.** No Olmo Hybrid checkpoint is on the Mac. So the family has no real-checkpoint reading on Metal.
- **OM-4: PASS** (`om4-identity-metal.log`). `gate identity -backend metal 69fead12 92fabc7b`: 28 IDENTICAL, 1 DIFFERENT, 14 NOT RUN.
  - The DIFFERENT family is `olmo_hybrid`, as named: its int4 cell moves from the CPU to Metal resident.
  - Its int8int8 cell is NOT RUN, since Metal declines a recurrent family at int8.
  - `qwen3_5`, `qwen3_5_moe` and `qwen3_next` are IDENTICAL, and so is every other family that ran.
  - The neighbours pass: the DeltaNet parity, KS-D3, MC3 recurrent, slot, KV8a and kernel tests.
- **OM-5: PASS** (`om5-censuses.log`).
  - **Admission, regenerated:** `sb2_metal_census.arm64` gains 18 lines, and `sb2_admission_census.arm64` and `.amd64` change 7 cells each. Every changed line names `olmo_hybrid`. The amd64 golden was regenerated on this Mac under Rosetta, where every other line already matched it.
  - **Decision (`testdata/mm1_census.json.gz`), regenerated on Linux amd64.** It holds one class, amd64, and its total hashes every group's full text, so it could not be patched from the Mac. Under Rosetta, 4,555 groups differ from it without this change, which is that box and not this change, so regenerating there would have written those in.
    - **The expected diff, measured on the Mac before the regeneration, in one process with the table flipped: exactly 86 groups, all `olmo_hybrid-tiny`** (D1 12, D2 6, D3 56, D4 12).
    - **The reading.** The coordinator regenerated it on nobara (linux/amd64, go1.27.3) at fc9375d1 with `go test ./decoder -run '^TestDecisionCensus$' -census.update`. Against the previous golden, of 29,637 groups: 86 changed, 0 gone, 0 added, and the changed set equals the expected list exactly. CI's `TestDecisionCensus` is the check that it holds on Linux.
    - On the Mac the test is a capture-only SKIP, which is not a pass: the golden has no arm64 class.
    - On this worktree the census also needs `testdata/gpt2` and `testdata/tiny-qwen2-moe` moved aside: gitignored, not in the golden, and holding subdirectories the census cannot read. A local condition, not this change.
  - **`touched.py` 17/17, the lint 7/7 and the citation lint pass** (`om5-touched-lint.log`). The lint and touched ran before the hardware-matrix regeneration below; it changes one generated doc.
- **Found on the way, and fixed:**
  - `docs/hardware-matrix.md` still listed the decline, and `touched.py` did not run its freshness test. It is regenerated: Olmo Hybrid now reads resident on Metal.
  - `TestSupport_eachParameterCheckDeclines` and `TestSupport_gapsOnOneOperationAllNamed` used `program.Metal` as their table without these settings. They now use a copy with both off, and `TestSupport_metalClaimsOlmoHybridsDeltaNet` holds the claims, and the step tables' lack of them.

## WebGPU (OW)

**Not started; the decision is with Francis.** The relayed direction was to restore Olmo Hybrid in WebGPU's table, but it has never been resident there. The admission census reads `webgpu does not implement [nope post-only-norm qk-norm-whole]`, and NegEigval and ONormEps would come on top, so it is a five-feature port, not sized here. If Francis wants it, its gates are registered here before its code.
