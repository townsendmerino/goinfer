# R1 — served tok/s: W4F16 is faithful but not fast. KILLED on speed at the decision cell.

**Result: R1's own gate (3) fidelity pass** ([`w4f16-decode-fidelity-PREREGISTERED.md`](w4f16-decode-fidelity-PREREGISTERED.md))
**does not save the lane on speed. At the decision cell (1.5B, depth 128, served), W4F16 measures
75.2 tok/s against W4A8's 73.1 — a 1.03× ratio, below even the registered 1.10× "killed" floor
(81 tok/s), let alone the 1.25× ship bar (92 tok/s). The depth curve confirms this is not
depth-concentrated: W4F16 is 2–4% faster than W4A8 at every depth tested (128/512/2048/4000), never
close to the threshold anywhere. R1's Build phase is now fully decided: KILLED on speed, despite a
clean — and slightly favorable — fidelity result.**

## Why this measurement, now

R1 was UN-PARKED on 2026-09-20 ([`r1-layer26-rootcause-2026-09-20.md`](r1-layer26-rootcause-2026-09-20.md)):
the 2026-09-19 investigation's "catastrophic bug" was an oracle error (the shipped W4A8 lane used
as ground truth at the attention-sink position), not a real kernel defect. That correction reopened
the two gates the original Build attempt never reached: gate (3), the pooled fidelity gate against
a real CPU reference, and the served-tok/s measurement against R1's registered band. Gate (3) ran
and passed on 2026-09-21 ([`w4f16-decode-fidelity-PREREGISTERED.md`](w4f16-decode-fidelity-PREREGISTERED.md),
pre-registered before any cell ran): pooled over 10 prompts × 64 teacher-forced positions against
the S (1.5B) CPU f32-weight/f32-activation reference, all three criteria hold — hard flips equal
(4 vs 4), agreement slightly higher for f16 (91.25% vs 90.78%), mean KL lower for f16 (0.0460 vs
0.0474, lower on 7/10 prompts). **This record is the other half: does it ship on speed.**

## Method

Box `apple-m1pro` (M1 Pro, 8 cores, 16 GB), Darwin 25.6.0, goinfer `1e05f181` (the R1 root-cause
correction commit). Metal serve binary built fresh from the `metal/` submodule
(`go build ./cmd/serve`, per `SERVE`'s own documented pattern in `scripts/bench_peer.py`).

**Served sweep** (`scripts/bench_peer.py`, `BENCH_MODELS=1.5B BENCH_BACKENDS=metal
BENCH_ENGINES=goinfer,ollama BENCH_DEPTHS=none`, greedy, depth 128, default run count (NRUNS=2,
NCOMP=8, NGEN=64), `BENCH_MAX_LOADAVG=2.0` as the brief itself registers — both runs waited for the
box to clear that threshold rather than raising it, since this is a throughput measurement where
background contention would genuinely corrupt the result, unlike a TTFT/fit-guard question).
**W4A8 and W4F16 are the same binary; `GOINFER_METAL_DECODE_LANE` cannot toggle mid-process
(`decodeLaneW4F16` is read once at `buildResident`), so — same limitation this session's own R-06
follow-up already disclosed for the identical reason — this is two separate process invocations,
not one interleaved session.** Ollama was run inside both invocations as the peer control, so its
own number is reported twice as a drift check.

**Depth curve** (`metal/depth_bench_test.go`'s `TestZZ_metalDepthBench`, `GOINFER_METAL_DEPTH_BENCH=1`,
qwen2.5-coder-1.5b, steady greedy decode via `ForwardArgmax` at depths {128, 512, 2048, 4000}, min-of-5
batches), run once per lane setting (same env-at-process-start limitation).

**Fit guard note:** both the served sweep and the depth bench loaded at 88–93% of the fit guard's
budget ("tight" but not declined) — this machine's real headroom is limited today (see this
session's own `mac-16gb-model-size-limits` memory). An external swap monitor ran throughout with an
auto-kill trigger; it never fired. No near-incident.

## Data — served, decision cell (1.5B, depth 128)

| arm | tok/s (mean of 2 runs) | spread |
|---|---:|---:|
| W4A8 (shipped) | **73.1** | 0.4 |
| W4F16 | **75.2** | 0.2 |
| Ollama (run 1, alongside W4A8) | 85.1 | 0.2 |
| Ollama (run 2, alongside W4F16) | 83.3 | 2.1 |

W4F16/W4A8 ratio: **1.029×**. Ollama's two independent measurements (85.1 vs 83.3, ~2% apart) are
consistent with this session's own established ~3.5% ordinary-session-drift baseline — not evidence
of a confound in the W4A8-vs-W4F16 comparison itself, which used the identical peer harness and
protocol both times.

R1's registered band (`docs/tasks/red-october.md`): **ships at ≥92 tok/s (1.25×); parked at
81–92; killed below 81 (1.10×)** — the ratios keyed to the brief's own cited 73.9 tok/s standing
W4A8 figure (73.9 × 1.25 = 92.4 ≈ 92; 73.9 × 1.10 = 81.3 ≈ 81; today's freshly measured 73.1 baseline
confirms that standing figure, within ordinary drift). **75.2 tok/s is below the 81 floor — killed,
not parked, not close to shipping.**

## Data — depth curve (qwen2.5-coder-1.5b, `TestZZ_metalDepthBench`)

| depth | W4A8 tok/s | W4F16 tok/s | ratio |
|---:|---:|---:|---:|
| 128 | 69.9 | 72.4 | 1.036× |
| 512 | 66.7 | 68.7 | 1.030× |
| 2048 | 48.4 | 50.5 | 1.043× |
| 4000 | 37.6 | 38.3 | 1.019× |

A consistent, real, small win (~2–4%) at every depth — **not depth-concentrated, and not a case of
"a shallow win bought by a deep regression"** (the brief's own stated concern for this check): there
is no regime where W4F16 loses to W4A8, but there is also no regime where it comes anywhere near
the registered thresholds. The depth-bench cell's own served-sweep number (69.9/72.4 at depth 128)
differs slightly from the `bench_peer.py` served numbers (73.1/75.2) because it measures the raw
`ForwardArgmax` path directly rather than through the served HTTP/streaming harness — expected, and
both instruments agree on the same ~3% ratio.

**Minor, independent finding, not fixed here:** `metal/depth_bench_test.go`'s own `t.Logf` header
is hardcoded `"...qwen2.5-coder-1.5b (W4A8)..."` regardless of which lane `GOINFER_METAL_DECODE_LANE`
actually selects — the W4F16 run's log line still says "(W4A8)". Confirmed this is a label bug, not
a real toggle failure, by the numbers themselves (would be bit-identical to the W4A8 run if the
lane weren't actually engaging, and it is not). Worth a one-line fix if anyone else is confused by
this log in the future; out of scope for this record.

## Gate (4): `go test ./metal/...`, lane off and on

Lane off (default): full non-heavy `./metal/...` package suite (no `GOINFER_HEAVY_TESTS`, the
default CI-equivalent suite, `-v`, 137s): **PASS, 0 failures**. Includes `TestMetalSnapshotGolden`
(the W4A8 path's own byte-identity pin) and every resident-parity test — confirms the lane is a
true no-op when off, as R1's original Build attempt already established.

Lane on (`GOINFER_METAL_DECODE_LANE=w4f16`, same suite, 133s): **1 failure** —
`TestMoE_assemblyVsDense`, min cosine 0.998953 against its own 0.9999 floor. **Diagnosed, not a
regression the lane introduces into the shipped MoE path.** That test builds TWO separate Metal
residents from two separate checkpoints — one MoE (identical experts, so its output should reduce
to a plain dense FFN), one genuinely dense — and compares their per-step decode output at a tight
0.9999 cosine floor. `GOINFER_METAL_DECODE_LANE` is read once per resident at `buildResident` from
the process environment, with no architecture check at that point (`r.decodeLaneW4F16 =
os.Getenv(...) == "w4f16"`, unconditional); the real gating is per-layer, inside `canUseF16Lane`
(`L.moe == nil && ...`), evaluated at dispatch time, not at resident-build time. Confirmed directly
with a scratch diagnostic (written, run, and deleted — not committed): `moeR.decodeLaneW4F16=true`
and `denseR.decodeLaneW4F16=true` (both residents pick up the ambient env var identically), but
`moeR.canUseF16Lane(l)` is `false` on every layer (the MoE exclusion holds, confirmed directly, not
assumed) while `denseR.canUseF16Lane(l)` is `true` on every layer (a plain dense checkpoint, fully
eligible). So with the lane on, the test's **dense** comparison arm silently starts using
higher-fidelity f16 activations while its **MoE** arm, correctly, does not move at all — the two
arms are no longer on the same numerics, which the test's 0.9999 threshold was calibrated assuming.
Forcing `denseR.decodeLaneW4F16` back to `false` (matching `moeR`'s actual, unchanged behavior)
restored the comparison to **cosine 1.000000** exactly, over the same 16 decode steps — conclusive.
**The shipped MoE dispatch path is provably untouched by this lane** (`canUseF16Lane`'s MoE
exclusion holds on every layer, directly verified); the failure is a test-isolation artifact of one
test building two residents under one ambient env var, not a defect. Worth a one-line follow-up
(pin `t.Setenv("GOINFER_METAL_DECODE_LANE", "")` explicitly in `TestMoE_assemblyVsDense`, the same
discipline `metal/r1_gate3_test.go` and the other R1 tests already use) so this test doesn't again
read as a lane regression under some future full-suite run with the env var set ambiently; out of
scope to fix here since R1 itself is already killed on speed regardless.

## Decision

**R1's Build phase is fully decided: KILLED, on speed, at the registered decision cell.** Per the
brief's own decision rule ("a ship on speed with (c) [fidelity] in the parked band is *parked*, not
shipped — the same rule that parked the CUDA spike"), the inverse also governs here: a lane that
*passes* fidelity is still killed if it fails the speed band, with no exception. `GOINFER_METAL_DECODE_LANE`
stays off by default, undocumented as an operator knob (already the case). The kernel, wiring, and
the three new test files (`metal/r1_gu_reference_test.go`, `metal/r1_lane_vs_cpu_test.go`,
`metal/r1_gate3_test.go`) are kept — real, reusable, correctness-proven groundwork, exactly the
disposition R1's own 2026-09-19 record asked for a "parked, not killed" kernel to receive, now
formally closed out with a real decision rather than left open.

**Why the speed didn't materialize, briefly:** the brief's own premise was that f16-FMA activation
staging reaches the ≥130–140 GB/s regime MLX/llama.cpp use, versus W4A8's observed 84–105 GB/s. The
~3% wall-clock gain measured here is far short of that — consistent with the weight-stream
bandwidth no longer being the bottleneck at this shape (K=1536/8960, N up to 17920), with something
else (dispatch count, threadgroup occupancy, the M1's per-dispatch launch floor — the same floor
`docs/completed/metal-verdict.md`'s Stage-A/Stage-B history already names as the recurring limiter
on this GPU) now dominating. Not independently diagnosed here; the speed band's own verdict does not
need the mechanism to be decided.

## What this does not establish

One model size (S/1.5B, the decision cell) — 0.5B, 7B and phi3-mini from the brief's full
measurement plan were not run (no Ollama peer pulled for those sizes on this machine today, and 7B
carries this session's own established memory risk at real headroom this tight). MLX was not run
(the `mlx_lm` tooling used for R12's MLX row on 2026-09-18 is not currently available in this
session's environment). Two sequential process invocations, not one truly interleaved session
(disclosed above) — the ~2% Ollama drift between them is consistent with ordinary noise, not a
sign this matters for the decision, since the ratio that decides everything (W4F16/W4A8) comes from
the *same*-process-pair comparison methodology used throughout, not from the Ollama figure.
