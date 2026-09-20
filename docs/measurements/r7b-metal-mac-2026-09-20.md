# R7b Mac session — verifying the Gumbel-max sampling change, then building the Metal device kernel (2026-09-20)

`docs/tasks/red-october.md` R7b: the Linux session's `r7b-gumbel-sampling` branch changed how
plain-`temperature` sampling is drawn (inverse-CDF → `argmax(logit/T + Gumbel noise)`, Philox4x32-10
keyed by seed/draw) on every backend, and asked this Mac session to (1) verify the change on real
darwin/arm64 hardware — the box that can't build Metal is exactly the box that made the change —
and (2) if time allowed, build the Mac-only piece the Linux session explicitly could not: a Metal
device kernel implementing `decoder.ResidentSample`. Both were done. This record covers both.

**Status: DONE 2026-09-20.** Verification found no regressions and one real, useful correction to
the branch's own claim (see §1). The Metal device kernel is built, gated, end-to-end correct, and
measured two ways: an internal harness (device draw converges on ~0.96× greedy across three runs)
and, after fixing a real gap in `scripts/bench_peer.py` (Phase C had no Metal-backend override),
the official peer instrument — where **goinfer beats Ollama on both greedy (1.15×) and
temperature-only sampling (1.10×)** on this Mac (see §4). Not full greedy parity the way CUDA/WebGPU
reach on their own sampled/greedy ratio, but ahead of the peer on both configs regardless.

## 0. A claim in the handoff was wrong — checked, not trusted

The task description said `r7b-gumbel-sampling` was "5 commits ahead of main; NOT merged, do not
merge it or push to main." On `git fetch`, `origin/r7b-gumbel-sampling` and `origin/main` were the
**same commit** (`e9fbc54c`) — the branch was already merged (or fast-forwarded) into main by the
time this session started. There was no "5 commits ahead" state to find, and nothing to avoid
merging — it was already done. This didn't change what needed verifying, only where the code lived:
this session's work is a child branch of that commit (`r7b-metal-verify-mac`), not a checkout of a
still-separate feature branch.

This session's own current `main` (from earlier, unrelated work) was 3 commits ahead of / 12 behind
`origin/main` — so verification was done on a fresh branch off `origin/r7b-gumbel-sampling` (=
`origin/main`), not on the locally-checked-out `main`, to avoid mixing this session's own unrelated
in-flight work (stashed) with R7b's code.

## 1. `metal/optfwd_test.go` / `metal/optfwd_bench_test.go` — checked first, as asked

Neither compares against a fixed token list. Both compare two **live** runs of the same code
(`TestOptFwd_bitIdenticalStream`: optFwd on vs off; `TestOptFwd_lowTempHighHitRate`: low vs high
temperature; `BenchmarkOptFwd`: a pure timing A/B) — a fixed-list comparison is exactly what the
Gumbel-max stream change would break, and neither test does that. Both also use `SamplingParams{...,
TopP: 0.9, ...}`, which is outside R7b's scope entirely (`temperature > 0` with **no** `top_k` /
`top_p` / `min_p` — the doc's own §"Scope" line): these tests' sampling path never touches the new
Gumbel code at all.

Ran anyway, to confirm rather than infer:

```
$ go test -mod=readonly -tags goinfer_testhooks ./metal/ -run 'TestOptFwd_bitIdenticalStream|TestOptFwd_lowTempHighHitRate' -v
--- PASS: TestOptFwd_bitIdenticalStream (1.90s)   stream identical (40 tokens); OptFwd hit rate 16.7%
--- PASS: TestOptFwd_lowTempHighHitRate (2.15s)   hit rate T=0.2 94.0% > T=1.0 0.0%
```

`BenchmarkOptFwd` was smoke-tested (one iteration, `-benchtime=1x`) to confirm it still compiles and
runs; not re-timed for real numbers (a benchmark run, not a correctness gate).

## 2. Full Metal + decoder suites, native darwin/arm64

**Metal, non-heavy** (`go test ./metal/... -skip TestProfileKernels`): **164 pass / 0 fail / 59
skip.** `TestProfileKernels` is excluded from the headline number and discussed on its own below —
it is unrelated to sampling and was independently confirmed not to be a regression from anything in
this branch.

**Metal, heavy (Gumbel-specific)** — see §4; folded in separately since it needed the kernel this
session built, not something that existed at the start of the session.

**Decoder, non-heavy** (`go test ./decoder/...`): **539 pass / 5 fail.** All five checked against
the commit immediately before R7b's own feature commit (`a4d66f92^` = `19d7e119`) and **found to
fail identically there** — pre-existing, unrelated to this branch:

```
--- FAIL: TestBailingHybrid_forwardParity     (numeric logit mismatches, a synthetic tiny fixture)
--- FAIL: TestGemma3VL_textParity
--- FAIL: TestGemma3VL_imageParity            (argmax got=144 want=179, cosine -0.0036)
--- FAIL: TestGenerateVL_streams
--- FAIL: TestInt4_forwardParity/gemma3-vl-tiny
```

None of the five is sampler-, Philox-, or Gumbel-related by name or by content (Bailing hybrid
forward parity, Gemma3-VL text/image parity, an Int4-forward-parity subtest for the same VL family).
`TestGumbelDraw_goodnessOfFit`, `TestSample_DrawIdentity`, and the rest of the sampler/Gumbel/Philox
tests in the decoder suite all passed.

**`TestSamplingThroughputGate`** — the doc's own item 2 (flaky, re-anchored to the legacy draw,
same bar) — passed on this run, consistent with "flaky, not broken."

### TestProfileKernels: environmentally flaky in a long single-process run, confirmed not a regression

Running the full non-heavy suite in one process, `TestProfileKernels` (a pure kernel-timing
microbenchmark, no correctness assertions, unrelated to sampling) hung indefinitely after ~800
lines of prior test output — `SIGQUIT`'d for a goroutine dump, confirming it was genuinely parked
(all threads idle, `pthread_cond_wait`/`kevent`), not spinning. Run in isolation immediately after
(`-run TestProfileKernels`), it completed in **5.76s** with normal output. This matches this repo's
own already-documented "concurrent Metal test contention" pattern (crash/hang site migrates,
reproducible even single-process, tied to resource accumulation over a long run rather than to any
specific test's own code) — not independently re-verified against the pre-R7b baseline commit
(its content has nothing to do with sampling, so this wasn't judged worth the extra real-checkpoint
load), but circumstantially it is exactly the shape of a pre-existing, environment-dependent issue,
not a regression. The rest of the suite (§ above) was re-run cleanly with `-skip TestProfileKernels`
to get a trustworthy full-suite number without this test's own flakiness in the way.

## 3. Benchmark cells needing re-measurement

Searched `docs/benchmarks.md` for every `temp > 0` sampled cell: the only ones that exist are
§B5.1's CUDA rows (`temp 1.0`, `temp 0.8 + top_p 0.95`, `temp 0.8 + top_k 40`), and they are
**already correctly flagged stale** by R7b's own 2026-09-20 addendum in that section. There is no
Mac/Metal-specific sampled cell anywhere in `benchmarks.md` to re-baseline — `docs/tasks/red-
october.md`'s own R7 row already said as much ("the Mac matrix's sampled row is still empty"),
which was true until §4 below and is updated there. Nothing else needs flagging.

## 4. The Metal device sampler (`decoder.ResidentSample`)

### What was built

- **`metal/gumbel.go`** — `gumbel_stage1` / `gumbel_stage2`, a direct port of `cuda/gumbel.cu` /
  `gpu/gumbel.go` (WGSL) checked against the actual **Metal Shading Language Specification**
  (2026-06-04 rev, fetched and grepped directly — not assumed to match WGSL's constraints):
  - **MSL has no `log1p`** — absent from Table 8.1/8.2's complete function lists. The small-`w`
    noise branch uses the same 12-term Taylor polynomial `gpu/gumbel.go`'s WGSL kernel already
    needed, for the same reason: `1.0 - w` in f32 rounds exactly to `1.0` below `w`'s representable
    ULP spacing near 1 (~6e-8), destroying the input regardless of which `log` variant runs
    afterward — a cancellation problem the polynomial avoids by never forming `1 - w`.
  - **MSL natively supports 64-bit unsigned integers** (`ulong`, Metal 2.4+; this library already
    targets MSL3.1), so Philox's `mulhi` is a direct 64-bit multiply + shift — simpler than WGSL's
    16-bit-limb reconstruction, and the same computation `decoder.philox4x32` does in Go.
  - **`precise::log`** is used for the `w >= 0.25` branch and the `h >= 2^31` branch: Table 8.1
    (fast math off, what `precise::` selects) gives `log <= 4 ulp` for any `x > 0`; Table 8.2 (fast
    math, the library's default — `metal/model.go`'s `preciseMathCompile` defaults false) gives only
    "absolute error <= 2^-21 near x=1, else <= 3 ulp" — verified against the spec's actual tables,
    not assumed to match WGSL's own (weaker, WGSL-specific) guarantee.
  - **No FMA in the key.** `#pragma METAL fp contract(off)` brackets both kernels. The spec states
    the library's default (`fast`) contraction fuses `a*b+c` **across statements**, not only within
    one expression — so writing the multiply and add as two separate statements (the way
    `decoder.gumbelKey`'s explicit `float32(...)` casts do on the host) is documented as
    **insufficient** on its own under `fast`. The pragma is scoped to just these two kernels and
    restored to `fast` immediately after, so no other kernel sharing `allKernels` is affected.
- **`metal/gumbel_sample.go`** — `(*resident).SampleAvailable`/`ForwardSample`/`GumbelForTest`.
  `SampleAvailable` declines when a host-side final-logit softcap or non-identity logit scale is
  configured (the device row wouldn't be ranking what `decoder.gumbelDraw` actually scores — the
  same condition `gpu.residentDecoder.SampleAvailable` uses) or when the model is paged MoE
  (`forwardLogitsPaged`/`forwardLogitsMoEPaged`'s multi-command-buffer flow isn't implemented here;
  declining routes those models to the always-correct host draw instead). `ForwardSample` extends
  the SAME command buffer as the trunk + LM head with the two Gumbel dispatches (one `Begin`/`End`,
  not a second round trip) and reads the drawn id directly off unified memory — no download step,
  unlike CUDA (`gpu.Download`) or WebGPU (`MapAsync`).
- **`metal/backend.go`** — `*metalResident.SampleAvailable`/`ForwardSample`, delegating to `.r`.
  **Necessary and easy to miss**: `decoder.Model.resident` holds a `*metalResident` wrapper around
  `*resident` via a *named* field (`r *resident`), not an embedded one, so Go does not promote
  `*resident`'s methods onto `*metalResident` — every other `decoder.ResidentForward` method on this
  type is already hand-delegated the same way (`Forward`, `ForwardNoLogits`, `PrefillLast`, …), and
  `ResidentSample` needed the same treatment. Missing this made the first end-to-end run pass
  **vacuously** (`DeviceSampled == 0` — see §"A real bug" below for the *second* thing this caught).

### Gates (`metal/gumbel_test.go`, `metal/sampled_gumbel_identity_test.go`)

| gate | result |
|---|---|
| `TestGumbelMSL_philoxKnownAnswers` — Random123 KAT vectors, direct MSL dispatch (no model needed) | 3/3 match |
| `TestGumbelDeviceAgreesWithHost` — port of CUDA's own gate, same rows/seeds/pre-registered bar (mismatch rate <= 1e-4, every mismatch a host-key near-tie <= 5e-5) | **0 mismatches in 15,840 draws** — matches CUDA's own figure exactly |
| `TestSampledGumbelStreamIdentity` — port of CUDA's own end-to-end gate: qwen2.5-coder 0.5B + 1.5B × T∈{1.0, 0.7, 1.3}, 1,000 tokens each, device vs `GOINFER_NO_SAMPLE_FASTPATH=1` host | **12,000/12,000 tokens identical, device sampler engaged on every token (0 fallbacks)** |
| `TestPhiloxGumbelMSL_mutationDetectsAConstantChange` — flip `gb_philox`'s `0xD2511F53u` constant, rebuild, re-check against the host | **189/300 draws mismatch, worst host-key gap 11** — confirms the gate is not vacuous |

Full run:

```
$ GOINFER_HEAVY_TESTS=1 go test -tags goinfer_testhooks ./metal/ -run \
  'TestGumbelMSL_philoxKnownAnswers|TestGumbelDeviceAgreesWithHost|TestSampledGumbelStreamIdentity|TestPhiloxGumbelMSL_mutationDetectsAConstantChange' -v
--- PASS: TestGumbelMSL_philoxKnownAnswers (0.11s)
--- PASS: TestGumbelDeviceAgreesWithHost (14.51s)         15840 draws: 0 mismatches
--- PASS: TestSampledGumbelStreamIdentity (148.25s)       6/6 configs, 12000/12000 tokens identical
--- PASS: TestPhiloxGumbelMSL_mutationDetectsAConstantChange (0.27s)   189/300 mismatched, worst gap 11
```

The full non-heavy metal suite was re-run afterward with these files in place: **164 pass / 0 fail /
59 skip** (the 3 new heavy tests correctly counted as skipped without `GOINFER_HEAVY_TESTS=1`; the
new KAT test — cheap, no real checkpoint needed — runs unconditionally and is in the 164).

### A real bug, found and fixed during development

`gb_gumbel`'s small-`w` branch called `gb_log1p_neg(w)` (which computes `log(1-w)`, a small
**negative** number) where it needed `-gb_log1p_neg(w)` (`e = -log1p(-w)`, a small **positive**
number) — a dropped negation lost while porting WGSL's own `e = -log1p_neg(w)` call site. The
symptom was NaN keys for roughly half the `uint32` range (every `h < 2^31` whose `w` landed below
0.25), invisible on any row where the true winner's logit margin dominated the (garbage) noise, but
**100% wrong on a flat-logits row**, where noise is the only differentiator — `TestGumbelDevice
AgreesWithHost`'s `"flat"` case caught it immediately (400/400 mismatches, gaps of 5-9, nothing like
a near-tie).

Found by isolating the two pieces the doc-comment claims separately, cheapest check first:
1. A throwaway MSL kernel calling `gb_philox` directly, checked against the Random123 vectors —
   **matched exactly**, ruling Philox out.
2. A throwaway MSL kernel calling `gb_gumbel(h)` directly for eight `h` values spanning the full
   `uint32` range, checked against the host's `-log1p(-w)`/`-log(v)` formula computed in Python at
   float64 — **four of eight (all in the `h < 2^31` branch) came back NaN**, isolating the bug to
   that branch specifically before ever touching the two-stage reduction or a real model.

Fixed (`e = -gb_log1p_neg(w)`), both debug probes re-confirmed the fix, then all four gates above
were re-run clean. This is the same discipline `docs/measurements/r2-attn-fa-followup-2026-09-20.md`
used earlier the same day for a different Metal bug: isolate the cheapest-to-check layer first
(known-answer vectors), then the next (a hand-computed reference), before trusting a result from the
full pipeline.

### Speed, internal harness (added in a follow-up pass, same day)

`metal/sampled_gumbel_speed_test.go`: three arms (greedy T=0; host draw T=1.0 with
`GOINFER_NO_SAMPLE_FASTPATH=1`, the do-nothing arm; device draw T=1.0), one seed, one 125-token
narrative-prose prompt, 200 generated tokens/round, 15 rounds + 1 discarded warm-up round, **arm
order rotated every round** and ratios computed **per round then summarized** (never pooled means —
CLAUDE.md's own measurement-discipline rule 7), decode-only timing (first emitted token to last,
prefill excluded). CUDA's own `TestSampledDecodeLadder` was not found committed anywhere in this
tree to port directly (checked repo-wide — no match), so this is a from-scratch harness built to
the same protocol description, not a line-for-line port. Three runs, same session (the third after
rebasing onto `origin/main`), qwen2.5-coder-0.5b-instruct-q4_k_m.gguf, int4, from `~/models`:

| | run 1 (13:58) | run 2 (13:59, cosmetic date-fmt fix) | run 3 (14:12, post-rebase) |
|---|---:|---:|---:|
| greedy | 161.5 tok/s | 159.3 tok/s | 160.4 tok/s |
| host (do-nothing arm) | 137.4 (**0.842**×) | 129.5 (**0.822**×) | 127.0 (**0.799**×) |
| device | 154.8 (**0.958**×) | 154.6 (**0.964**×) | 154.8 (**0.964**×) |
| device ÷ host | **1.132**× | **1.186**× | **1.212**× |

Device sampler engaged every round in all three (`device-sampled-ever=true`); the ratios above are
paired medians per round, not ratios of the pooled medians shown for readability. **A real,
repeatable ~13-21% win over the do-nothing arm**, device draw converging tightly around **0.96×
greedy** across three independent runs.

### Speed, official peer harness (`scripts/bench_peer.py`, same day)

The internal harness above is a from-scratch Go test, not this repo's committed peer-measurement
tool — the number that actually belongs beside CUDA's and WebGPU's in `benchmarks.md` §B5.1 has to
come from `scripts/bench_peer.py`, driven through the real HTTP server, the same instrument every
other backend's R7/R7b row used. Running it surfaced a real, reusable gap: **Phase C (the sampling
axis) was hard-coded to `backend="cuda"`**, with no override — unlike Phase A/B, which both respect
`BENCH_BACKENDS`/`BENCH_DEPTH_BACKEND`. A `BENCH_CONFIGS=temp1.0_notrunc BENCH_BACKENDS=metal` run
silently tried a nonexistent `/home/francis/.../serve-cuda` path and produced nothing for goinfer on
Metal at all — which is *why* no Mac/WebGPU sampled peer cell has ever existed before, not merely
that nobody had run one. Fixed in `scripts/bench_peer.py` by adding `BENCH_SAMPLED_BACKEND`,
mirroring `BENCH_DEPTH_BACKEND`'s existing override pattern exactly (same "must be in
`BENCH_BACKENDS`" guard, same reasoning) rather than a one-off hack — this is now available for any
future Mac or WebGPU sampled cell, not just this one.

**Provenance.** goinfer serve built from `561e72f6` (this branch, tree clean) vs Ollama v0.32.5 —
version-matched to the CUDA peer sweep's own anchor. Both over HTTP, `scripts/bench_peer.py`,
interleaved with a server restart between cells, depth 128, 1,024 decode tokens/cell (16 completions
× 64 tokens × 2 runs), qwen2.5-coder-0.5b, int4 / q4_K_M, **same weights verified per-tensor**
(`scripts/gguf_same_weights.py`: 291/291 tensors identical between `~/models/qwen2.5-coder-0.5b-
instruct-q4_k_m.gguf` and Ollama's `q05` blob). Raw cells `b5-mac-r7b-561e72f6.json`, log
`b5-mac-r7b-561e72f6_run.log` (in `docs/measurements/`).
**Disclosed deviation from protocol**: `BENCH_MAX_LOADAVG` raised from the script's default 1.0 to
3.0. This machine's ambient 1-minute load (this very Claude Code session, an editor, the desktop)
sits at 1.6-2.2 even with nothing else running — 1.0 is not reachable on this box without stopping
the session doing the measuring, and the script's own preflight text names raising the cap
deliberately as the sanctioned alternative to an unbounded wait. Recorded loadavg at every cell was
1.5-2.2, i.e. genuinely idle *for this machine*, not spiking under the load of a concurrent unrelated
job — but this is a looser bar than the CUDA sweep's, and is flagged here rather than left implicit.

| config | engine | tok/s | vs Ollama | own cost vs greedy |
|---|---|---:|---:|---:|
| greedy | goinfer | 163.4 | **1.154×** | — |
| greedy | Ollama | 141.6 | (ref) | — |
| temp 1.0, no truncation | goinfer | 159.1 | **1.100×** | 0.974× |
| temp 1.0, no truncation | Ollama | 144.7 | (ref) | 1.022× |

goinfer beats Ollama on **both** greedy (1.15×) and temperature-only sampling (1.10×) on this Mac's
Metal backend — and goinfer's own sampled/greedy ratio (0.974×) independently confirms the internal
harness's three runs (0.958-0.964×) via a completely different measurement path (through the real
server, restarted between cells, peer-interleaved) rather than merely repeating the same number.
Ollama's own sampled/greedy ratio (1.022×, essentially free) is the interesting asymmetry: Ollama's
sampling cost on this box is near zero while goinfer's is ~2.6%, which is consistent with §4's
"Metal draw isn't a PCIe/MapAsync-readback win, just a full-vocab-loop-replacement win" inference —
Ollama likely never paid a comparable host-side full-vocab cost to begin with on Metal, so it has
nothing large to recover.

### Not done this session

- **No `--exact-prefill`/CPU-fast-attention interaction check** beyond what `SampleAvailable`'s own
  gate already declines (paged MoE). Every other family this session's fixture set could reach
  (qwen2.5-coder dense) went through cleanly; an exotic family combining device sampling with a
  path this session didn't test is possible but unconfirmed.

## 5. Branch and commits

Work is on `r7b-metal-verify-mac`, branched from `origin/r7b-gumbel-sampling` (= `origin/main` at
`e9fbc54c`, see §0) — not on `main`, not pushed anywhere yet. This session's own unrelated
in-progress work (an R2 investigation follow-up, from earlier the same day) was stashed before
switching, untouched by anything here, and was restored after this work was committed.
