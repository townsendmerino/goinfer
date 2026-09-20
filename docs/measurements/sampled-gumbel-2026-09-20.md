# R7b — temperature-only sampling by Gumbel-max on every backend; CUDA and WebGPU draw on-device (2026-09-20)

**Verdict.** Plain-`temperature` sampling (temperature > 0, no `top_k` / `top_p` / `min_p` — the OpenAI default, the cell
R7 could not serve) is now drawn as `argmax(logit/T + Gumbel noise)` on every backend, and CUDA and WebGPU draw it
on-device and return only the token id. Paired against greedy on the 0.5B, same session:

| backend | device draw off | device draw on |
|---|---:|---:|
| CUDA, T=1.0 (n=15) | 0.744 | **1.008** (paired SD 0.016) |
| CUDA, T=1.3 (n=15) | — | 1.010 |
| WebGPU, T=1.0 (n=12) | 0.803 | **1.036** |

**This is a disclosed break: the distribution is unchanged, the seeded stream is not** — for a given seed the tokens
differ from every earlier release, on every backend (the host draw is the reference and every backend without a device
kernel uses it). Owner decision, 2026-09-20 ("b for all backends", after the options were laid out).

Predecessors: [`sampled-topk-baseline-2026-09-20.md`](sampled-topk-baseline-2026-09-20.md) (step 0) and
[`sampled-topk-2026-09-20.md`](sampled-topk-2026-09-20.md) (the filtered configs). Raw logs:
[`sampled-gumbel-2026-09-20.log`](sampled-gumbel-2026-09-20.log).

## The algorithm (one definition, every backend)

Token = `argmax_i ( l_i * invT + G_i )`, ties to the **lowest index**; `G_i = -ln(-ln1p(-w_i))`,
`w_i = (h_i + 0.5) / 2^32`; `h_i` is a word of **Philox4x32-10** keyed by the seed with counter
`(i>>2, draw, draw>>32, 0)`, lane `i&3`, where `draw` is the sampler's own running index of temperature-only draws
(`Sampler.NextDraw`). By the Gumbel-max theorem that is an exact draw from `softmax(l/T)` with no normalisation, no
cumulative sum and no dependence between tokens, so it is an argmax every GPU can do in one parallel pass with no f64.
Reference: `decoder/sampler_gumbel.go`, `decoder/philox.go`. Devices: `cuda/gumbel.cu`, `gpu/gumbel.go` (WGSL).

**Noise range.** G is bounded to [-3.13, 22.9], so a token needs a logit gap above ~26 nats to be unreachable, where
the exact draw gives it e^-gap. The affected mass is ≲1e-6 even over a 262k vocabulary (computed, not measured — no test
can resolve a probability that small). An f32 uniform would cap G at 16.6 and lose ~5e-4; that is why the noise is
computed from `w = 1-u` with `log1p`, and on the device by the `min(h, ~h)` symmetry so f32 never subtracts near 1.

**Scope.** Plain decode at temperature > 0 with no `top_k` / `top_p` / `min_p`, including the logprobs, penalty and bias
variants (so requesting logprobs cannot change which token a seed yields). The device draw additionally requires no
logprobs, bias, penalties, logit processor, and a backend whose logits are not transformed on the host after readback.
**Speculative decoding** needs explicit probabilities for accept/reject and keeps drawing those from its own stream
(`spec_sample.go`, "in-distribution lossless"); only its seed and per-round bonus tokens use the Gumbel draw
(`Sampler.drawTarget`), which keeps its first token equal to plain decoding's under the same seed.

## Provenance

| | |
|---|---|
| machine | `nobara-pc`: Ryzen 7 3700X, RTX 2070 SUPER 8 GB, NVIDIA driver **595.91.07**, Nobara 44 |
| checkpoint | `~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf` (local NVMe, not `/srv/models`), CUDA `--quant int4`, WebGPU `int8int8` |
| code | branch `r7b-gumbel-sampling`: CUDA/decoder at `a4d66f92` (the CUDA ladder ran on that source before it was committed), WebGPU at `67d7b4b5`, parity refresh `aaa50d9d`; Go 1.27.0; `gumbel.ptx` built at the ambient NVRTC 12.9.86 (a new isolated module) |
| instrument | `TestSampledDecodeLadder` in `cuda/` and `gpu/`: interleaved rounds with a rotating start, one discarded warm-up per arm, 77-token narrative-prose prompt (in the source), ≤160 (CUDA) / 128 (WebGPU) generated tokens, decode rate excludes prefill, paired ratio to greedy within a round. **Do-nothing arm included**: the same config with `GOINFER_NO_SAMPLE_FASTPATH=1` in the same session. |
| load | CUDA ladder: run after waiting for the 1-min load average to fall below 0.6 (it started at 0.57). WebGPU ladder (archived re-run): load average **0.75** at start, i.e. good but not fully idle; the first WebGPU run read 0.796 → 1.035 at 1.27. |
| thermal | not logged |

## Results

CUDA (median decode tok/s, n=15, `sampled-gumbel-2026-09-20.log`): greedy 331.1; T=1.0 device-sample off 246.2 (**0.744**,
paired SD 0.011); T=1.0 **332.3 (1.008**, SD 0.016); T=1.3 333.7 (1.010); and, for context, the R7 filtered arms in the
same run: `top_p` 0.95 top-K off 216.7 (0.653) → 315.3 (**0.951**), `top_k` 40 0.969, `min_p` 0.05 0.972.

WebGPU (n=12): greedy 183.7; T=1.0 device-sample off 147.4 (**0.803**); T=1.0 **189.9 (1.036)**. The device path beats
*greedy* there because WebGPU's greedy still maps the whole logits row back to the host every token, and the device
sampler reads back 4 bytes.

Host draw (CPU, Metal, and anything without a device kernel): 0.73 ms vs 1.31 ms per draw at a 152k vocabulary, ~1.8x
cheaper. Those two figures come from `TestSamplingThroughputGate` output in **different runs** (Gumbel from this branch,
the legacy chunked draw from the last run before it), so treat the ratio as indicative.

The CUDA T=1.0 ratio sits slightly *above* 1.0. A hypothesis, **unmeasured**: greedy's own argmax is a single-block kernel
and the Gumbel kernel is parallel, so greedy may carry a small avoidable cost (~1%). Not investigated.

## Correctness gates

Statistical rules were **pre-registered in the test header before the first run** (`decoder/sampler_gumbel_test.go`);
none was changed afterwards.

| gate | result |
|---|---|
| Philox4x32-10 vs the Random123 known-answer vectors | all 3 match |
| goodness of fit vs exact softmax(l/T), chi-square, |z| < 4.5 two-sided | T=0.6/1.0/1.7: z = +0.63 / −0.71 / −0.33 (1M draws each) |
| two-sample equivalence to the **old** sampler | z = +0.57 (df 198, 1M draws each) |
| tail on a 32k Zipf vocabulary | mass beyond rank 256: exact 0.0248, observed 0.0247 (z = −0.04); top-10 frequencies within 4.5σ |
| noise transform vs an f64 reference | max abs error 5.4e-7 over 2M points (bound 4e-6); monotone; range as claimed |
| parallel = sequential | identical on every row and temperature tried (argmax is exact) |
| invariances | same seed ⇒ same stream; different seeds differ; **logprobs on/off ⇒ same tokens**; -Inf never drawn; flat logits uniform |
| device kernel vs host reference, CUDA | **0 mismatches in 15,840 draws** (rule: ≥99.99% and every mismatch a near-tie by the host's own keys) |
| device kernel vs host reference, WebGPU | **0 mismatches in 12,000 draws** |
| end-to-end device vs host token streams, CUDA | **9/9 identical**, 8,585 tokens (qwen2.5-coder-0.5b, gemma3-1b, llama-3.2-1b × T=1.0/0.7/1.3); the device sampled every step |
| end-to-end device vs host token streams, WebGPU | **4/4 identical**, 1,680 tokens (qwen2.5-coder-0.5b, llama-3.2-1b × T=1.0/0.7) |
| `TestNgramSampledFirstTokenMatchesPlain` | passes: speculative decoding's first token equals plain decoding's under a seed |
| FMA lint, local-memory census (CUDA) | green with `gumbel.cu` registered |

**Mutation checks** (each applied by hand, the test observed red, then restored — restorations verified byte-identical):
noise scaled by 0.8 → chi-square z = +8597; the four lanes of a Philox block sharing one word → z = +5150; the draw
counter frozen → z = +65546; logprobs drawing differently → the invariance test red; a Philox constant changed on the
CUDA kernel → mismatches with host-key gaps 1.4–5.2 ("not a near-tie"); on WebGPU, the `mulhi` carry term dropped →
gaps up to 7.6, and the `log1p` polynomial removed (always `log(1-w)`) → gaps 1.7e-4 to 21, which **confirms WGSL's weak
`log` near 1 really would have broken the noise**.

**Full suites on this code:** cuda 166 pass / 0 fail (124 skipped); gpu 125 pass / 0 fail (48 skipped); decoder 579 pass /
2 fail / 97 skipped — see the two items below. A skip is not a pass.

## Things that went wrong or need saying plainly

1. **A claim I made was wrong.** I told the owner speculative decoding "never shared a stream with plain decoding".
   `TestNgramSampledFirstTokenMatchesPlain` showed it deliberately shares the *first token*, and it failed on the first
   full run. Fixed by `drawTarget`; the claim is corrected in the handoff note and here.
2. **`TestSamplingThroughputGate` is flaky on this box, independent of this change.** Its ratio (top_p ÷ temperature-only,
   bar 5.0×) swung 3.83×–5.35× on identical code: 4.83 / 4.27 / 3.90 on unmodified `main`, 4.48 / 4.24 / 3.83 on this
   branch standalone, and it failed once (5.35×) inside a full-suite run. I re-anchored its denominator to the legacy draw
   (kept unchanged in `sampler_chunked_ref_test.go`) because the production denominator got cheaper, with the bar
   **unchanged**; that does not remove the run-to-run noise, which is the parallel denominator's dependence on how idle the
   box is. Reported, not fixed.
3. **`TestParityManifest_fresh` was red** because `decoder/model.go` is in the manifest's `core` set; refreshed by the
   sanctioned script after the forward goldens ran (61 passed / 0 failed / 0 skipped), commit `aaa50d9d`.
4. **A false alarm of my own making:** the `gpu` suite showed one failure (`TestParentSelfSkipsWhenNoSubtestRan`) that
   reproduced on clean `main` — because I had been passing `GOWORK=` as an environment variable, which that test's
   `go test` subprocess inherits. Without it the test passes and the suite is 125/0.

## Not done, and residual risks

- **Metal has no device kernel** (this is a Linux box; Metal only compiles on macOS). It already draws the new stream on
  the host. Handoff note written for the Mac session; two seeded Metal tests (`optfwd_test.go`,
  `optfwd_bench_test.go`) need checking there. Any unfiltered-sampling Mac/peer number needs re-baselining.
- **Device vs host can pick different tokens only where the two best keys are within an f32 rounding.** Observed 0 in
  15,840 + 12,000 kernel draws and 0 in 10,265 end-to-end tokens; expected ~1e-6 per token (WebGPU somewhat higher: WGSL
  guarantees `log` only to an absolute 2^-21 near 1). Not zero, so a rare seeded divergence between the device and the host
  path (`GOINFER_NO_SAMPLE_FASTPATH`) is possible.
- **WGSL gives inf/NaN no defined result**; the device path is only used when nothing masks the row (no logit processor,
  bias or penalties) and the tests use a large finite negative for masked entries.
- The **speed band was measured on one checkpoint** per backend; gemma3-1b and llama-3.2-1b were identity-tested, not
  speed-benchmarked. The **Ollama peer sweep was not re-run**, so `benchmarks.md` §B5.1's peer ratios remain stale for
  every sampled cell (noted there).
- Statistical tests cannot see a bias smaller than roughly 1e-3 in a token's probability at these sample sizes; the
  ≲1e-6 tail bound above is an argument from the noise range, not a measurement.

## Reproduce

```sh
GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' -run 'TestGumbelDeviceAgreesWithHost|TestSampledGumbelStreamIdentity|TestSampledDecodeLadder' -v ./cuda/
GOINFER_HEAVY_TESTS=1 go test -tags 'gpu goinfer_testhooks' -run 'TestGumbelDeviceAgreesWithHost|TestSampledGumbelStreamIdentity|TestSampledDecodeLadder' -v ./gpu/
go test -run 'TestGumbel|TestPhilox' -v ./decoder/
# A/B on any run: GOINFER_NO_SAMPLE_FASTPATH=1
```
(Run without a `GOWORK` environment variable set; `go.work` is discovered from the directory.)
