# Correctness queue

Parity, numerics, goldens, quantization, model families. Anything whose success criterion is **agreement with a reference** — a cosine, an argmax match, a golden. If the question is *does it compute the right thing*, it belongs here.

> **One of four queues.** The work list is split by *success criterion*, not by component:
> [performance](queue-performance.md) · [correctness](queue-correctness.md) ·
> [engineering](queue-engineering.md) · [release](queue-release.md).
> [`QUEUE.md`](QUEUE.md) is the index over all four and holds the cross-cutting sweeps.
>
> **Task docs are NOT queues.** `docs/task-*.md` are *design records* — why a thing is built as it
> is — and they are cited from 88 code comments. A queue entry cannot carry that, so the task docs
> stay put and the queues hold only the open work.
>
> Entries keep the section they were filed under (`In flight`, `Queued`, …) and their original IDs,
> so a citation to an ID still finds it.


## Queued

**G1 · LFM2.5-2.6B as an experimental family** — `linux`

Scoping prompt written. A fifth sequence-mixing family: interleaved gated short-convolution blocks
and GQA, `layer_types` controlling the pattern, `conv_L_cache` 3, LayerNorm QK-norm (not RMSNorm),
FFN dim computed rather than stated. The conv layers carry a rolling conv state instead of a KV
cache.

The estimate turns on two questions: whether Mamba-2's causal depthwise `conv1d` is factored out or
inlined, and whether the cache abstraction already carries mixed per-layer state types
(Granite-4.0-H and Nemotron-H suggest it may). Also unestablished: **whether LFM2.5 is
architecturally the same as LFM2** — the transformers docs cover only LFM2.

Blast radius matters: anything touching shared `decoder/` core re-stales all 19 enforced families.
Answer that before estimating.

**Q2 · The GGUF-quant cross-gate gap — CLOSED, and it was unplumbed too** — `linux`, `bd08936`→

The cross-gate check showed `scripts/parity_sweep.sh` covering the GGUF quant formats while the goldens
refresh did not. **(a) Exposure: a LAG, not a hole.** `scripts/parity_sweep.sh` is not in CI — it is
release-only, run by hand on the box (`RELEASING.md` §C1). So the formats are covered at release and
**not between releases**, which is exactly when a frozen-core edit gets only the goldens refresh.

**(b) Both routes priced before choosing, and route B turned out unnecessary:**

| route | cost |
|---|---|
| extend the goldens selector to the existing GGUF gates | **26.8 s**, 11 gates, no new fixtures |
| author GGUF-quant goldens for those 11 rows | unnecessary — the gates already exist and already pass |

Same shape as Q1(b): **unplumbed, not missing.** The gates were simply outside `GOLDEN_RE`. Adding
`^TestGGUF_.*_parity$` took the refresh from **19 passed / 0 quantized** at the start of this campaign
to **33 passed / 14 quantized**, and the cross-gate check now reports *"the two gates span the same
quantizations."*

One bug fixed in the cross-gate check itself: it compared a composite label (`int4/int8`, from a file
driving two quantizations) against atomic ones and reported a difference that was purely notational —
a permanent false positive in the check built to make real differences visible. Both sides are
atomised now.

**Q1 · The forward goldens prove f32 ONLY — no quantized path has a golden that runs** — `linux`,
**NEW. G-01 at the largest scale it has appeared.**

> **The "14 quantized" composition figure, resolved by enumeration rather than by authority
> (2026-08-12).** Two classifiers disagreed — an ad-hoc name grep said **7**, the refresh script's
> said **14** — and 14 had already propagated into commit bodies and into the proof requirement.
> Adopting it because it was the script's would have been a tiebreak by authority, so both were
> tested instead.
>
> **7 was structurally incapable of being right**, for two independent reasons. Five of the fourteen
> carry no quantization token in their NAME at all: `TestGemma4_logitParity` and
> `TestMellum2_logitParity` set it in the test body (`Options{Quant: "int8int8"}`), and
> `TestGGUF_gemma3/qwen2/qwen3_parity` set it in the **fixture filename** the test loads. No
> name-based match can see either. (The other two misses, `Q2_K` and `Q3_K_M`, were a plain gap in
> the ad-hoc pattern, which listed `q4|q5|q6|q8` — a bug rather than a structural limit, but it lands
> in the same place.)
>
> **The script's classifier cannot double-count.** `grep -c` counts matching LINES; every top-level
> result is one line; subtest lines are indented and excluded by its `^--- PASS:` anchor. Measured on
> the captured run: 33 top-level PASS lines, **0** indented ones, no duplicate names among the 14.
>
> And it does not misclassify — all fourteen drive a genuinely quantized path:
>
> | gate | quantization | set where |
> |---|---|---|
> | `TestGemma4_logitParity` | int8×int8 | test body |
> | `TestMellum2_logitParity` | int8×int8 | helper body |
> | `TestInt4_forwardParity` | int4 group-wise | test body |
> | `TestGGUF_Q2_K_parity` | Q2_K (+Q3_K/Q4_K/Q6_K mix-ins) | fixture |
> | `TestGGUF_Q3_K_M_parity` | Q3_K (+Q4_K/Q6_K) | fixture |
> | `TestGGUF_Q4_0_parity` | Q4_0 | fixture |
> | `TestGGUF_Q4_K_M_parity` | Q4_K (+Q6_K) | fixture |
> | `TestGGUF_Q4_K_S_parity` | Q4_K_S | fixture |
> | `TestGGUF_Q5_K_M_parity` | Q5_K (+Q6_K) | fixture |
> | `TestGGUF_Q6_K_parity` | Q6_K | fixture |
> | `TestGGUF_Q8_0_parity` | Q8_0 (tinyllama) | fixture |
> | `TestGGUF_gemma3_parity` | Q8_0 (gemma-3-270m) | fixture |
> | `TestGGUF_qwen2_parity` | Q8_0 (Qwen2.5-0.5B) | fixture |
> | `TestGGUF_qwen3_parity` | Q8_0 (Qwen3-1.7B) | fixture |
>
> **So 14 stands, and every commit body citing it is correct.** The reason is now recorded, which is
> the point: the figure is load-bearing in the proof requirement, and "the script said so" is not a
> reason. Note what the table also shows — **11 of the 14 take their quantization from a fixture**,
> so any future classifier that reads test names will undercount for the same structural reason.

int4 is the documented default quantization. **Zero goldens drive it.** And the hole is wider than
that: of the 19 goldens that actually RAN in the 2026-08-12 refresh, **every one is f32**.

| quantization | golden files | did any RUN? |
|---|---|---|
| f32 (explicit or default) | 24 | **19 ran** |
| `int8int8` (W8A8) | 3 — `gemma4_parity`, `gemma4_12b_parity`, `mellum2_parity` | **all 3 SKIPPED** |
| `int8` (weight-only Q8) | 1 — `gptoss_real` | not matched by the goldens regexp at all |
| **`int4` / W4A8** | **0** | — |

So `scripts/refresh_parity_hashes.sh` — the sanctioned freeze-exception path, and the thing that makes a
core edit auditable — **proves f32 numerics and nothing else**. A change that is bit-identical in f32
and wrong in int4 passes it in 6 seconds.

**Retroactive scope, and this is the part to act on.** Any claim of the form *"the parity suite
covers X"* is scoped to **the quantizations the goldens drive**, which today is f32. Every place such
a claim is written down needs that scope added — `docs/parity-coverage-policy.md`'s tier table,
`RELEASING.md`'s §C1, the README's support matrix, and the P6 commit body (which states it already).

**And the freeze protects what the goldens check.** The `6edd1ca` numerics freeze over `decoder/` is
enforced by `deps_hash` staleness, whose release valve is this goldens run. Where the goldens are
silent — every quantized path — the freeze is a *procedural* barrier with no numeric proof behind it.
That is not an argument for lifting it; it is an argument for knowing what it is.

**WHY THIS OUTRANKS THE REST OF THE QUEUE — sequencing, not enthusiasm.**

**P1 is the v1.0 headline and lives in the frozen core.** The numeric proof available when that core
unfreezes was **f32-only**. So lifting the freeze did not buy the ability to verify the work the
freeze defers — and the shortfall **would not have announced itself**, because the goldens would pass.
An f32-green refresh over an int4 regression is a passing gate, not a silent one; nothing in the
output distinguishes them.

That makes Q1(c) a **prerequisite for the v1.0 core work**, not a parallel item, and it belongs ahead
of the E-group release gate for that reason rather than because it is interesting. **Done
2026-08-12 (`1d0d1ed`)**: 23 fixtures across 16 architectures, so the prerequisite is now met for
int4 specifically.

**RUN WHAT EXISTS FIRST — and most of it was UNPLUMBED, not missing.** Done 2026-08-12, `a6c5b57`:

- **(b) the three `int8int8` goldens** skipped for one liftable reason, the same for all three:
  `GOINFER_HEAVY_TESTS` unset. **Two of the three pass here in ~70 s** (gemma4, mellum2). The refresh
  now enables heavy by default. The third (gemma4-12B) skips on a genuinely absent GGUF — an asset
  question, not a plumbing one.
- **(a) the `int8` golden did NOT turn out to be a selector bug.** `TestGptOssReal_logitParity` **does**
  match the regexp. It is invisible because `decoder/gptoss_real_test.go` is behind `//go:build realckpt`,
  which the refresh does not pass — and with the tag it still skips for a missing GGUF. **Two gates,
  either sufficient.** A one-line regexp change would have bought nothing.

**Non-f32 rows after (a) and (b): 2** (21 passed, 2 quantized). The distinction the ordering was meant
to test comes out clearly: **int8 was unplumbed** (one env var), **int4 is genuinely missing**, and the
gpt-oss int8 row is **asset-blocked behind a build tag**.

The refresh now also prints the **quantization breakdown**, because "19 passed" and "21 passed" read
identically to a human and that is precisely how this stayed invisible through nine prior refreshes.

**(c) int4 goldens — DONE `1d0d1ed`.** Scope measured *before* authoring and stated as a target: int4
has no divisibility constraint (`nGroups` is a ceiling divide), so eligibility was never the limit —
fixture availability was. **Target: 23 fixtures / 16 architectures. Delivered: 23 / 16.**

The goldens compare **int4 output against recorded int4 output**, not int4 against f32 within a
tolerance. A tolerance band against f32 measures quantizer loss — a real question with its own gate
on the policy's quant axis — and would read as "int4 is covered" while proving nothing about whether
the W4A8 path still computes what it computed yesterday. Only the self-comparison catches a
regression in the path the freeze protects and P7 will change.

Fixtures are **enumerated** from `testdata/` rather than listed by name, so a new family is picked up
without editing the gate, and a run comparing **zero** fixtures **fails** rather than passing.
Mutation-checked by perturbing the quantizer itself (`int4GroupSize` 32 → 64 → red).

Recorded **absences**, not gaps: `gpt_oss` (MXFP4-prequant, rejects a conflicting `--quant` by
design), `siglip_vision_model` (an encoder), `gpt2` / `mellum` / `qwen2` / `qwen3` (no tiny
safetensors fixture), `qwen2_moe` and `gemma4-dense-scaled-{24,48,64}` (incomplete fixture dirs).

**Refresh now reports 22 passed / 3 quantized**, against 19 passed / 0 quantized when this began.

**Also record with P6's 6.09 s price: cheap and thorough are different properties.** 6.09 s buys 19
passes and 11 skips. The skips are not free — they are the coverage this item is about.

**`TestDecodeParityInt4` diverges from its recorded golden — REAL checkpoint, NOT the synthetic
goldens above, found 2026-08-15, unclaimed.** `decoder/parity_int4_test.go`, real
qwen2.5-coder-0.5b int4 (W4A8, safetensors-loaded gguf), greedily continuing a fixed prompt: got vs
want diverge at token index 5 (`got 1438 want 11047`) and every token after — not a subtle drift,
a different continuation entirely. **Confirmed pre-existing and unrelated to two same-day changes**
via an isolated `git worktree` bisect: fails identically on the P1 pre-change tree AND at aikit
`v1.17.1` (before the day's aikit v1.19.0 bump) — same got/want arrays, byte for byte. So this
predates both P1 (`97f824a`) and the bump (`fb8e26b`); it was sitting on `main` before either.

One live lead, not yet chased: the test's own comment records a **recent asset-resolution fix**
("this site previously skipped whenever `GOINFER_PREQUANT_GGUF` was unset... under a bare `go test
./decoder`, this gate now RUNS where it used to skip") — meaning this real-checkpoint gate may have
been silently skipping for a long stretch, during which `parityWantInt4`'s golden could have gone
stale against real drift nobody was watching for. That is a hypothesis, not a finding — needs a real
bisect (not the two-point check done here) to find which commit actually broke it, or whether the
golden itself was simply never right. **Unclaimed — pick up either box.**
