# Olmo Hybrid on Metal and WebGPU: NegEigval and ONormEps in their DeltaNet (2026-11)

**Registered 2026-11-02, before any code (mac-sb11m; the commit's time is the registration's).** Francis's direction of 09:52 PST,
relayed by the coordinator: do on Metal, then on WebGPU, what #604 does on CUDA (`docs/tasks/task-olmo-hybrid-cuda-2026-11.md`).
Implement `linear_allow_neg_eigval` (NegEigval, the write gate doubled to [0, 2)) and the output norm's own epsilon (ONormEps) in the
backend's Gated-DeltaNet, gate it against the CPU, then claim both in the backend's table, so Olmo Hybrid runs resident again.
One PR per backend.

<!-- ledger -->
| id | state | next | evidence |
|---|---|---|---|
| OM | active | Metal: code, then gates OM-1 to OM-5 | this doc, § Metal |
| OW | open | WebGPU: register its gates here after OM, then the same | this doc, § WebGPU |

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
- **OM-2, the batched prefill.** A prompt at or above the pass's floor equals the sequential decode's last-row logits within the bar
  the DeltaNet prefill test holds, on `olmo_hybrid-tiny`, with both plants red. If no such test admits this fixture today, it does after.
- **OM-3, a real checkpoint.** None is on the Mac (`~/models` has no Olmo Hybrid). **NOT RUN, named.** That is not a pass.
- **OM-4, identity.** `gate identity -backend metal <main base> <head>`: every family that ran stays IDENTICAL, with `qwen3_5`,
  `qwen3_5_moe` and `qwen3_next` among them. `olmo_hybrid`'s cells may change, since their backend changes. They are listed.
- **OM-5, the censuses.** The admission and decision censuses are regenerated, and their diffs touch only `olmo_hybrid` lines.
  Then `touched.py`, `go run ./cmd/lint` and the citation lint.

**Void if:** a GPU run outside the timing lock; a NOT RUN counted as a pass; a bar moved after a reading.
**Stopping rule:** one run each. **GPU time:** a few minutes, under the shared lock.

## WebGPU (OW)

Registered here after OM's PR, before OW's code.
