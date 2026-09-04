# QUEUE — the index over four queues

The work list is split by **success criterion**, not by component. An entry lives in exactly one
queue, keeps its original ID, and keeps the section it was filed under.

| queue | holds | the question it answers |
|---|---|---|
| [**performance**](queue-performance.md) | throughput, latency, kernels, residency, memory | *how fast, how much memory* |
| [**correctness**](queue-correctness.md) | parity, numerics, goldens, quantization, families | *does it compute the right thing* |
| [**engineering**](queue-engineering.md) | gates, lints, censuses, tooling, process rules | *would we find out* |
| [**release**](queue-release.md) | release gates, tagging, v1.0 criteria, capability claims | *can we tag* |

**Task docs are not queues.** `docs/task-*.md` are design records — why a thing is built as it is —
and they are cited from **88 code comments**. A queue entry cannot carry that, so they stay where
they are and the queues hold only the open work. Finished ones are archived to `docs/completed/`.

**This file keeps** the cross-cutting material that is not any one queue's: the sweeps, the
sequencing notes, the release draft, and the generated citation indexes below.

# Work queue — the shared, claimable list

> **Why this file exists.** The queue used to live in conversation, where only the top of it gets
> restated each turn and everything below silently sinks. Three items aged out that way — the Metal
> consumer window, the out-of-tree consumer audit, the drain fix's CUDA verification — none through
> carelessness. And two boxes pulling from the same unstated queue independently built two
> mechanisms for running the heavy tier, because neither could see the other's progress.
>
> That makes the conversational queue an instance of the class this fortnight has been cataloguing:
> an artifact that exists and is not composed into any decision. A check that cannot fail.
>
> **This file is the queue.** If it is not written here, it is not queued.

## How to use it

- **Claim before starting.** Move the item to `In flight` and put your box and the date on it.
  A claim is what stops the other box duplicating it.
- **Release on finish** — move to `Done` with the commit, or back to `Queued` with what you learned.
- **Never delete an item to tidy up.** Strike it with a reason, so "we decided not to" is
  distinguishable from "it sank".
- **Add the whole item, not a title.** Enough that whoever picks it up does not have to reconstruct
  the context from a transcript they may not have.

Boxes: `linux` (nvidia-rtx2070s, CUDA) · `mac` (Apple Silicon, Metal).

## In flight

## Queued

Ordered roughly by priority within each group. Each item carries enough context to be picked up
cold. Where something is believed done but unconfirmed, it says so — **verify before striking**.

### A. Open investigation

**Two decoder tests are RED on main, and neither is from the spec-x-pager work** *(observed
2026-09-02, both A/B-confirmed pre-existing by stashing the day's changes)*. A full
`go test ./decoder/ -v -timeout 30m` on nobara-pc ran 451 tests with exactly these two failures:

- **`TestLoadSession_rejectsCorrupt`** — fails on its *"clean blob, no id check should load"* case
  with `kv snapshot: tokens: 3 ids for pos 0 — a longer token list makes rewindForReuse report an
  exact match it never performed`. So a LEGITIMATE snapshot is being refused by a reuse-safety
  guard. Worth noting the shape: that is the same family of problem as the resident prefix-reuse
  bug fixed today, one layer down in the staged cache — a reuse path and its safety check
  disagreeing about what counts as a match. Not investigated.
- **`TestParityManifest_fresh`** — 28 validated families stale, since a spread of old commits
  (`bd085de`, `e8fa53c`, `3f23c96`, `41673f7`, `25a4711`, `b067007`, `d64afe4`). Its own message
  names the fix: re-run `go run ./cmd/gate parity` then `-update`, or `scripts/refresh_parity_hashes.sh`
  for a provably non-numeric edit. Deliberately NOT refreshed as a side effect of an unrelated fix —
  re-baselining 28 families' parity hashes to make a red go green is how a real regression gets
  blessed.

**The decoder suite's runtime depends on whether the developer has `~/models`, and CI cannot see
it** *(observed 2026-09-02)*. Nine test files — `decoder/eagle_test.go`, `decoder/eagle_throughput_test.go`,
`decoder/eagle_accept_test.go`, `decoder/eagle_alpha_test.go`, `decoder/eagle_diag_test.go`, `decoder/eagle_forward_test.go`,
`decoder/mtp_head_test.go`, `decoder/spec_eagle_test.go`, `decoder/tree_verify_test.go` — read `~/models` with no
`GOINFER_HEAVY_TESTS` or `realckpt` gate. On CI there are no checkpoints so they skip and the job
lands at ~10 min; on a box with a populated `~/models` they run. The measured local figure was
**1366 s (22.8 min)** against CI's ~10 min, and `-timeout 25m` in `.github/workflows/ci.yml` is already sized for the
CI shape, not this one. Two consequences: "green on CI" and "green on my box" are not the same
statement, and a local run with the DEFAULT 10 m timeout fails as a timeout that looks like a hang.
Within that, **`TestMoEExpertMajor_bitIdentical` alone is 569 s — 42% of the whole suite.**

**Resident prefix reuse returned WRONG OUTPUT on recurrent families — `3358e6ba`** *(found
2026-09-02, A/B-confirmed; **FIXED, feature kept**)*. Prefix reuse on the resident KV corrupts generation on Gated-DeltaNet
models: repeated identical greedy prompts on qwen3.6-35B-A3B diverge at token 0, differently on
every repeat, decaying to a 1-token reply, with `gen.Err()` nil throughout. The A/B is decisive —
`TestPagerDeterminism/reuse-on` FAILS and `/reuse-off` (`GOINFER_NO_RESIDENT_REUSE=1`) PASSES 5/5
byte-identical, on the same load, same box.

`residentReuseLen` (`decoder/resident_reuse.go`) gates only on the env flag and the token-id
match — there is **no recurrent-state exclusion**, though the rest of the tree already refuses this
family for exactly this reason (`decoder/deltanet.go`: the conv ring and matrix state are "NOT
position-truncatable (why qwen3_5_moe falls back from prefix reuse / speculative)"; `decoder/forwardn.go`'s
`specRollbackSafe` refuses it too). The resident path re-zeroes that state only at `pos == 0`, which
a reused prefix never reaches, so turn N decodes from turn N−1's tail state. The commit's premise —
the resident cache is positional so truncation is free — holds for KV and fails for recurrent
state, and nothing asks which the model has.

**Blast radius was the case the commit was written for**: an agent loop, where turn N+1 is a prefix
extension of turn N by construction. Invisible to single-generation tests, which is what the
existing 26B/35B cache tests are.

**FIX (applied, not a revert).** `residentReuseLen` returns 0 when `Model.hasRecurrentState()`, so a
recurrent model reuses only from position 0 and every attention-only family — most of what an agent
loop actually runs — keeps the 21.7x win untouched. `TestPagerDeterminism/reuse-on` is the gate:
red before the guard, green after. Getting reuse BACK for DeltaNet needs a state checkpoint at the
reuse point, which is **L-05's** job and deliberately not patched in here.

**THIS IS THE EIGHTH CONSUMER OF THE FAMILY LIST TO MISS A STATE KIND IN A WEEK, AND THAT IS NOW THE
FINDING.** C-02 was LFM2's conv window missing from every site that should have named it;
`hasRecurrentState`'s own doc comment closes with "the fourth kind will be added here, once, or it
will be missed at four sites again" — and the next new consumer promptly wrote its own implicit list
by asking no question at all. A predicate that must be REMEMBERED at each new call site will keep
being forgotten at the rate new call sites appear. **The argument for deriving it once and reading
it everywhere (L-16) is now made by the tree itself, not by anyone's judgement.** The fix here takes
that shape as far as one change reasonably can: `specRollbackSafe`'s inline copy and the new
resident-reuse guard both read a single named `Model.hasRecurrentState()`, which reads the dispatch
table's `Recurrent` bit — so a fourth state kind is added in the table once.

Repro and evidence: `cuda/pager_determinism_test.go`,
`docs/measurements/spec-x-pager-2026-09-02.md`.

**Granite/Nemotron lost resident residency on WebGPU — `[nope]`** *(found 2026-09-02, unfiled
until now)*. `TestGraniteResidentSpeedup` and `TestResident_C01_pos0ResetsRecurrent` both fail in a
full `./gpu/` run with `BuildResident declined: arch needs unimplemented feature(s) [nope]`.
Verified on a CLEAN `origin/main` checkout, so it is not from the G35/G36/reuse work that was in
flight when it surfaced. `FeatNoPE` arrived with the Cohere family; something in the decoder
tranche now has granite declaring it, and the WebGPU backend does not implement it — so granite
silently dropped off the resident path onto CPU/staged. That is a **capability regression, not a
test problem**: the tests are correctly reporting it.

**`TestPrefillAttnRowTileInvariance` fails in `./decoder/`** *(found 2026-09-02, unfiled)*. Also
reproduces on a clean `origin/main` checkout. Not investigated.

**G5 leftovers** *(the harness-UX gate, `docs/task-embed-and-harness-ux.md`)*. The Claude Code half
is done and `docs/integrations/claude-code.md` is published with measured numbers. Still open:
- **dsh's 277 s run is un-re-measured** — deliberately parked 2026-09-02 (dsh is not installed and
  `server.md` documents the install as an ordeal). The figure stands unverified, not disproven.
- **N-18 and M-20**, two of G5's three stated preconditions, are still open: `tool_choice`
  `required`/`any` forces nothing with 2+ tools (`forcedTool` handles `none` and a named function,
  and `required` falls through to the single-tool case), and Gemma-4's tool rendering disagrees
  with its own template. Both are stated in the recipe so nobody debugs them as their own setup.
- **Open-WebUI and Continue recipes** are unwritten, because §3.5's rule is that a recipe with no
  number is not published and neither has one.

**Resident prefix reuse is single-conversation.** Two interleaved conversations on one model each
cold-prefill as they alternate. The fix is per-conversation KV parked in host RAM and swapped back
(~257 MiB and ~43 ms for a 2.3k-token conversation, against ~8.9 s to recompute it) — worth doing
when concurrency is real, not before.

**The facade (`task-embed-and-harness-ux.md` phase 1) is unblocked and undecided.** G4 was the
gate and it passes at 1.21×.

**Model-pull leftovers** — three small items left when `task-model-pull.md` was archived
2026-09-02 (all of its phases shipped; these were never in scope). Recorded here because
`docs/completed/` is a record, not a task list, so anything still to do had to leave it.

- **Disk-space precheck** before a multi-gigabyte fetch. Today `pull` discovers a full disk by
  failing partway; the sizes are known up front from the tree API, so this is a cheap check.
- **Split-GGUF checkpoints.** A shard is *detected* and refused by name (`Select`), so nobody
  gets one useless piece — but assembling a split checkpoint needs loader work beyond the fetch
  command, which is why it stopped at the refusal.
- **Generate `release-assets.yml`'s matrix from `internal/modelpull/curated.json`.** Optional:
  `TestCurated_matchesReleaseWorkflow` already makes the two impossible to ship out of step, so
  this is tidiness, not a correctness gap. Weigh it against editing the release path.


**G26 · phi3-mini lost 5.8% at `temperature 1.0`, and only phi3-mini** — **CLOSED 2026-08-27: the
cause is optimistic forward (`6a4e0ae`), a net loss above T ≈ 0.26; about half the headline was the
anchor's own spread. Full result at the end of this entry — the investigation below is left intact,
including two findings it had to retract.** Measured
2026-08-27 in the §B5 re-anchor (`docs/benchmarks.md` §B5.1): 116.6 → **109.8** tok/s on CUDA at
depth 128, against an Ollama side that read 125.6 both times. Well outside the old ±0.5 spread.

**What makes it worth a look rather than a shrug:** the same configuration *gained* on the other
two models in the same sweep — qwen2.5-coder-0.5b 219.2 → 237.7 (+8.4%) and gemma3-1b 131.7 → 143.7
(+9.1%). A uniform sampler change cannot produce that split, so it is model-specific.

**Do not guess the cause from the numbers.** Two anchors differ by more than one thing: goinfer
moved several releases, and the driver/distro moved on 2026-08-25. The greedy phi3-mini CUDA cell
is *unchanged* across the same pair (124.9 vs Ollama 125.9, parity), which points at the sampling
path rather than the forward — but that is a lead, not a finding.

Raw cells: `docs/measurements/b5-reanchor-61b1e03.json`. Reproduce with
`BENCH_MODELS=phi3-mini BENCH_CONFIGS=temp1.0_notrunc python3 scripts/bench_peer.py <out>`.

**2026-08-27, two findings — and the second demotes this from "regression" to "not established".**

**1. P10 is NOT the cause, and the obvious argument for it was backwards.** `4da116d` (P10) is the
only sampler change between the anchors, and it removes a per-token full-vocab allocation whose size
scales with vocab — so the tempting story is that phi3-mini's 32k vocab gains least. Its benchmark
population was only 152k and 262k, so the small vocab was genuinely never measured, which made this
look like the int8-bar-on-int4 shape: a change validated on one population, shipped for another.

Measured it instead of arguing it. `BenchmarkFilterFresh*` (added here) is the pre-P10 shape — a
fresh buffer per call — paired against the reused-scratch path, `-count 5`:

| vocab | pre-P10 | post-P10 | P10 effect |
|---|---|---|---|
| 32k | 220,801 ns | 200,779 ns | **+9.1% faster** |
| 152k | 989,363 ns | 908,777 ns | +8.1% faster |
| 262k | 2,800,565 ns | 2,687,574 ns | +4.0% faster |

P10 helps **most** at 32k. It cannot be the cause. The vocab-scaling intuition was exactly wrong.

**2. The baseline is a different protocol, and this cell is known to move under one.** §B5's
footnote ᵈ records the phi3-mini rows as **5 runs × 8 completions**; the re-anchor used
`bench_peer`'s default **2 runs × 8**. The same footnote records this cell moving **112.4 → 116.6**
(+3.7%) purely from re-measurement. So the recorded values are **112.4, 116.6, 109.8 across three
protocols** — a spread of the same magnitude as the "regression".

**2026-08-27, protocol-matched re-measure: the regression is REAL.** phi3-mini,
`temp1.0_notrunc`, 5 runs x 8 completions — the anchor's own protocol:

    goinfer  109.7   runs [108.1, 110.5, 107.8, 110.8, 111.2]
    ollama   125.2   runs [125.5, 125.3, 125.1, 125.2, 125.1]

109.7 against the anchor's 116.6 at matched protocol: **-5.9%**, and essentially identical to the
2-run 109.8, so it was never a protocol artifact. The peer is unchanged (125.2 vs 125.6), so it is
goinfer-side. Promoted from unestablished to confirmed.

**Two corrections to the analysis above, both mine.**

*The "greedy is at parity" argument was unsupported.* There is no historical greedy phi3-mini cell
— §B5 carries only the two sampled rows — so parity with Ollama today says nothing about whether
greedy moved since the anchor. It cannot be used to localise the fault.

*The P10 elimination measured the wrong function.* `benchFilter` exercises `topFilterLogits`, the
**top_p** path; temp-only with no truncation goes through `sampleChunked`/`expChunked` instead. That
matters because P10 touched both, and the two configs moved in OPPOSITE directions on this model
(top_p +2.7%, temp-only -5.9%) — exactly the split those two paths would produce. Re-measured on
the correct path (`BenchmarkExpChunked{Fresh,Reuse}*`): P10 is **+23.1% at 32k**, +20.9% at 152k,
+21.8% at 262k. Uniformly a win. **P10 is eliminated on both paths.**

**The more useful result is a bound on where the cause can be.** At 116 tok/s a token costs ~8.6 ms;
`expChunked` at 32k costs ~96 us. Even counting the whole sampler generously, sampling is a
low-single-digit percentage of per-token time — so **no sampler change can move the end-to-end
figure by 5.9% in either direction**. The cause is in the decode path or the stack beneath it, not
the sampler. That also means the top_p/temp-only split is probably not the signal it looks like.

**What is left, and what is not worth doing.** The driver/distro upgrade is a candidate but moved
the other two models the other way (+8-9%). The goinfer range between `ca29d6c` and HEAD is the
other. Bisecting a 5.9% end-to-end delta across that range costs a 40-minute peer sweep per point;
before spending that, get a phi3-mini GREEDY number at the anchor commit — if greedy regressed too,
this is a decode-path question and the sampled framing has been a distraction from the start.

**2026-08-27, running: the anchor commit BUILT AND MEASURED TODAY, which separates the two
confounded causes instead of bisecting into them.** The reason the greedy number was missing is
that it never existed historically; the reason it was expensive is that a targeted cell used to
drag a ~40-minute sweep behind it. Both are now fixed — `scripts/bench_peer.py` gained `BENCH_BACKENDS`
and `BENCH_DEPTHS=none`, so this question costs **4 cells** rather than 13, and `ca29d6c` is built
from a worktree with the aikit its own `go.mod` pins (**v1.16.0**, against HEAD's v1.28.0).

Measuring the anchor *today* holds the driver fixed at `595.91.07` and varies only goinfer, which
the original anchor-vs-re-anchor comparison could not do — those two differed by both at once.
Both goinfer builds are compiled today with the same toolchain; ollama runs in both sweeps as the
drift control.

**Reading, fixed before the numbers land.** HEAD today reads greedy **124.6** and temp1.0 **109.7**;
the anchor recorded temp1.0 **116.6**. Let *A* be the anchor build's temp1.0 measured now:

| A | reading |
|---|---|
| ≈ 116.6 (matches its own record) | The regression lives in `ca29d6c..HEAD`. **Driver exonerated**, bisect is then justified and cheap. |
| ≈ 109.7 (matches HEAD) | goinfer's commit range is **exonerated**; the 116.6 was an old-driver number and the delta is environmental. G26 closes as environment, not code. |
| between, or neither | **Ambiguous → parked**, not argued either way. The band exists because both endpoints are ~6% apart and a middling value supports whichever story the reader brought. |

And independently, on greedy — the question the queue actually asked for:

- **anchor greedy ≈ 124.6** → greedy did not move; the effect is confined to the sampled path,
  which sits awkwardly with the arithmetic bound above (no sampler change can move end-to-end by
  5.9%). That combination is a finding to explain, not to wave through.
- **anchor greedy materially > 124.6** → greedy regressed too, the sampled framing was a
  distraction from the start, and this is a decode-path question.

**2026-08-27, RESULT at n=5 — greedy is clean, and the sampled cell turns out to be UNSTABLE on the
anchor build.** Both binaries built today from the same toolchain (anchor from a worktree at its own
pinned aikit v1.16.0; HEAD at v1.28.0), same driver `595.91.07`, ollama in both sweeps as the drift
control. Raw: `docs/measurements/g26-anchor.json`, `docs/measurements/g26-head.json`, log
`g26-anchor-vs-head_run.log`.

| cell | anchor `ca29d6c` | HEAD | Δ |
|---|---|---|---|
| **greedy** | 124.61 ± 0.12 | 124.85 ± 0.04 | +0.2% |
| **temp1.0_notrunc** | 114.17 ± **2.63** | 111.05 ± 0.45 | −2.7% |
| *ollama greedy* | *125.73* | *125.74* | *0.0%* |
| *ollama temp1.0* | *125.52* | *125.63* | *0.1%* |

**The drift control is as good as this box gets** (≤0.1% on both ollama cells across two sweeps), so
the goinfer-side comparison is sound rather than session noise.

**1. Greedy did not move — the decode path is clean.** 124.61 vs 124.85 at sd ≤ 0.12. That answers
the question this item was parked on, and it answers it the *other* way: the sampled framing was not
a distraction hiding a decode regression. Nothing in `ca29d6c..HEAD` (746 commits) moved greedy
phi3-mini at depth 128.

**2. The anchor's temp1.0 cell has 6× the variance of anything else measured here** — sd 2.63,
runs spanning **110.8 → 116.9**, against HEAD's sd 0.45 and every other cell on this page at sd
≤ 0.15. **The recorded anchor value of 116.6 sits essentially at the top of that build's own range,
and 109.7/109.8 sit near the bottom of HEAD's.** A large part of the original "5.9% regression" is
therefore a high draw compared against a low one, across protocols that never repeated the cell
enough to see its spread.

**3. What survives is at most −2.7%, and at n=5 that is not established.** Welch on the two run
sets gives t ≈ 2.6, p ≈ 0.06 — inside the band this item pre-registered as **ambiguous → parked**.
Per that rule the conclusion is parked and NOT argued in either direction.

**Parking the conclusion is not the same as shelving the question, and this cell is now cheap.**
`BENCH_BACKENDS` + `BENCH_DEPTHS=none` cut it from 13 cells to 4, so the correct response to an
ambiguous result is more of it rather than a shelf: **n=15 on both builds is running**, sweep order
reversed (anchor first) so an order or thermal effect shows up as disagreement with the pair above
rather than hiding inside it. Raw will be `g26-anchor-n15.json` / `g26-head-n15.json`.

**Do not bisect yet.** A bisect over 746 commits searches for a step change; if the effect is
substantially cell variance, every step reads as noise and the search converges on whichever commit
happened to draw high. Variance first, then bisect only if a real step survives it.

## G26 RESOLVED, 2026-08-27 (n=15) — real, HALF the claimed size, and the sampler is back

Raw: `docs/measurements/g26-anchor-n15.json`, `g26-head-n15.json`, log `g26-n15_run.log`.
Sweep order reversed from the n=5 pair (anchor first) as an order/thermal control; ollama held
within 0.15% of itself across all four sweeps.

| cell | anchor `ca29d6c` | HEAD | Δ |
|---|---|---|---|
| greedy | 124.41 ± 0.20 | 124.60 ± 0.13 | **−0.15%** (nil) |
| temp1.0_notrunc | 114.40 ± **2.07** | 111.40 ± 0.55 | **+2.7%**, t = 5.42, 95% CI [+1.91, +4.08] |

**1. The regression is real but roughly HALF what was recorded.** −2.7%, not −5.8/−5.9%. The
missing half is a sampling artifact of the original comparison: at n=15 the anchor build's temp1.0
cell spans **109.6 → 116.9**, and the recorded **116.6 is the 14th of 15 sorted values — the ~90th
percentile of that build's own distribution.** The anchor figure was a high draw; 109.7/109.8 were
low-to-central draws of HEAD's. Comparing them measured the spread as much as the change.

**2. Greedy did not move at all** (−0.15% at sd ≤ 0.20, n=15), across 746 commits. The decode path
is clean and the question this item was parked on is answered negatively.

**3. THE COST IS ON THE SAMPLED PATH, and the arithmetic that eliminated the sampler was wrong.**
Because greedy and temp1.0 differ only in the sampling config and generate identical token counts
(2560 every cell, verified), the same-build gap between them isolates the sampled-path tail:

| build | greedy | temp1.0 | sampling step | share of token |
|---|---|---|---|---|
| anchor | 8.038 ms/tok | 8.741 ms/tok | **0.703 ms** | 8.0% |
| HEAD | 8.026 ms/tok | 8.976 ms/tok | **0.950 ms** | 10.6% |

That tail got **+35% more expensive**, and the +0.247 ms/token accounts for **2.8%** of end-to-end
against the **2.7%** observed — the effect is fully explained with nothing left over.

> **CORRECTION, made the same day and before the number was acted on: this tail is NOT the sampler
> alone, and calling it "the sampling step" would have sent the next reader to the wrong function.**
> On CUDA the two configs do not differ only in host-side sampling. `cuda/resident.go:2530`
> documents `ForwardArgmax` as the greedy fast path that "reduce[s] the argmax on-device and read[s]
> back 4 B instead of the whole logits vector", and `cuda/softcap.go:25` records the consequence:
> the sampled path is "the path that also does the ~1 MB readback", and pays softcap where the
> family has it. So the 0.703 → 0.950 ms gap is **the whole sampled-path tail**: the full-vocab
> device→host readback that greedy skips entirely, plus any softcap, plus `Sampler.Sample`.
>
> The arithmetic below still refutes the recorded "sampler is too small" bound — the tail *is*
> 8–10.6% of the token, not low-single-digit. What it does not do is name which component moved.
> For scale, phi3-mini's 32064 logits are ~128 KB, which is tens of microseconds of PCIe, not 700 —
> so readback bandwidth alone does not explain the tail's size either. Attributing the +0.247 ms
> requires measuring the components, which is what the next step does.

**This refutes the bound recorded above**, which read: *"Even counting the whole sampler generously,
sampling is a low-single-digit percentage of per-token time — so no sampler change can move the
end-to-end figure by 5.9%. The cause is in the decode path or the stack beneath it, not the
sampler."* **That is wrong.** It was built from `expChunked`'s ~96 µs microbenchmark, but the real
end-to-end sampling step costs **703–950 µs — seven to ten times more.** The microbenchmark measures
one function; the config's actual path costs an order of magnitude more than it. The lesson is the
one this repo already writes down elsewhere: a sub-component measurement does not bound the total,
and here it under-bounded it by ~10x and retired the correct hypothesis for two rounds.

**4. A second, unexplained effect: HEAD is 4x more STABLE and slower** (sd 0.55 vs 2.07). Whatever
changed made the sampling step both costlier and more consistent. Note P10 — the one sampler change
between the anchors — was measured *faster* on both paths, so something else in the range more than
cancelled it; P10's elimination stands, the conclusion drawn from it does not.

**Next step, now cheap and well-aimed: do NOT bisect end-to-end.** The signal is a 0.247 ms/token
step in the sampled-path tail. Split it before bisecting:

1. **Host-side `Sampler.Sample`, whole path** (not `expChunked` alone — that is the error above), at
   phi3-mini's 32064 vocab, at both commits. `decoder/g26_sampler_bench_test.go` does this, with a
   greedy arm as the control so it decomposes exactly as the end-to-end numbers did. Minutes.
2. **Whatever Sample does not account for is above it** — the readback/sync path, which greedy's
   on-device argmax never touches. That is a different search, in `cuda/`, not `decoder/`.

Note for reading step 1: those logits are SYNTHETIC. If the regression is data-dependent, a flat
result there is evidence about the cause's shape rather than its absence — this repo has the
matching rule already, that a synthetic reproduces shape and not pressure.

## G26 CAUSE FOUND, 2026-08-27 — optimistic forward is a NET LOSS at temperature 1.0

Both steps above ran. Raw: `docs/measurements/g26-sampler-bench.log` (host sampler, both commits),
`g26-head-nooptfwd-n15.json` (the A/B), plus the n=15 pair already cited.

**Step 1 exonerated the host sampler, in the opposite direction.** `Sampler.Sample` over phi3-mini's
32064 vocab, `-count 5`, at both commits:

| benchmark | `ca29d6c` | HEAD | Δ |
|---|---|---|---|
| Sample, temp 1.0, 32k | 627.6 us | 552.6 us | **−11.9% (HEAD FASTER)** |
| Sample, greedy, 32k | 25.6 us | 18.0 us | −29.9% |
| Sample, temp 1.0, 152k | 1358.1 us | 1676.1 us | **+23.4% (HEAD slower)** |

HEAD's sampler is *faster* at the vocab that regressed, so it cannot be the cause; the remainder
(tail − Sample) went 75 us → 397 us, putting the whole move above the sampler. **Recorded separately
because it is a real finding and not this item's: the sampler CROSSES OVER with vocab — HEAD is
−12% at 32k and +23% at 152k.** That is not what P10's "uniformly faster" microbenchmarks predicted,
and the 152k regression is unexamined.

**Step 2 identified it: `6a4e0ae`, optimistic next-token forward.** It is in HEAD, absent from the
anchor, and gated `useGPU && !fastGreedy && optFwdEligible(sp) && GOINFER_NO_OPTFWD == ""` — so it
runs on **sampled decode only and never on greedy**, which is precisely the observed signature. It
overlaps the CPU sampler with a speculative next forward; on a miss that forward is discarded.

A/B on the same HEAD binary, n=15, phi3-mini, temp1.0, CUDA:

| arm | mean | sd |
|---|---|---|
| HEAD, optFwd ON (the default) | 111.40 | 0.55 |
| HEAD, optFwd OFF (`GOINFER_NO_OPTFWD=1`) | **117.84** | 0.16 |
| anchor `ca29d6c` (predates the feature) | 114.40 | 2.07 |

**Turning the feature off is worth +5.8%** — `+0.491 ms/token`. And it lands HEAD **+3.0% above the
anchor**, so the accounting is: the other 745 commits made sampled decode ~3% faster, optFwd gave
back ~5.8%, and −2.6% net is exactly what the peer sweep measured. Greedy is unmoved in every arm,
as the `!fastGreedy` gate requires.

**Why it loses here.** The feature's own gates record the hit rate collapsing with temperature —
98.0% at T=0.2, **55.6% at T=1.0**. Near-greedy it nearly always hits and the overlap is close to
free; at T=1.0 roughly half the speculative forwards are discarded, and contending for the GPU costs
more than the ~0.55 ms of host sampler the overlap hides. **This is the do-nothing-arm rule in the
repo's own words: the feature was validated where it wins and shipped enabled for every sampled
config.** It is not a bug and the mechanism is sound — the default is what is wrong.

**SCOPE, so this is not over-read.** One model, one temperature, one depth, CUDA, greedy-comparable
protocol. It does NOT show optFwd is a loss generally; at T=0.2 the same evidence suggests it wins.
What is established is that **at T=1.0 on this pairing it costs 5.8%**, and that the on/off decision
is currently made without reference to temperature.

**A PREDICTION I MADE AND GOT WRONG, recorded because it was pre-registered.** I predicted that
disabling optFwd would restore the anchor's *spread* as well as its mean — the story being that
overlap hides the sampler's data-dependent cost. It does not: HEAD-no-optFwd has **sd 0.16**, tighter
than either other arm. The variance story is refuted. **The anchor build's sd 2.07 on this cell
remains unexplained and is now a separate open question**, not a side effect of this one.

**Follow-ups, none of them started:**
1. **Gate optFwd on temperature or on realized hit rate.** `OptFwdStats{Guessed, Hit}` already
   tracks it per run, so an adaptive gate has its input. Needs a sweep across T before a rule.
2. ~~**The 152k sampler crossover** (+23.4%)~~ — **RETRACTED 2026-08-27, measured end-to-end and
   the microbenchmark had the DIRECTION WRONG.** See the retraction below.
3. **The anchor's temp1.0 variance** (sd 2.07 vs 0.16-0.55) — mechanism unknown.

**Follow-up 1 RUNNING (2026-08-27): the temperature ladder, and how it will be read.** T ∈ {0.2,
0.4, 0.6, 0.8, 1.0}, temp-only/no-truncation, optFwd ON vs OFF on the same HEAD binary, n=6,
phi3-mini/CUDA, peer kept in both arms as drift control. Raw: `g26-tsweep-optfwd-{on,off}.json`.
Configs added to `scripts/bench_peer.py` as a ladder because two endpoints cannot locate a crossover.

Read Δ = (OFF − ON)/ON at each T. **Δ > 0 means the feature LOSES at that temperature** (turning it
off is faster); Δ < 0 means it wins.

| outcome | reading, fixed in advance |
|---|---|
| Δ < −1% at low T, > +1% at high T | A real crossover. Gate optFwd on temperature, with the threshold set BELOW the measured crossover, not at it. |
| Δ > +1% at every T | The feature does not pay on this pairing at all. Default should be off pending a config where it wins — its 98%-hit-rate case is T=0.2, so a null there is the strong result. |
| Δ < −1% at every T | T=1.0's 5.8% is not temperature-driven and the mechanism story above is wrong; reopen. |
| \|Δ\| < 1% throughout | No effect to gate; the T=1.0 result needs re-examining before anything is changed. |

**LADDER RESULT (2026-08-27), n=6, phi3-mini/CUDA/depth 128 — a real crossover at T ≈ 0.26.**
Raw: `g26-tsweep-optfwd-{on,off}.json`, log `g26-tsweep_run.log`.

| T | optFwd ON | optFwd OFF | Δ = (OFF−ON)/ON | verdict |
|---|---|---|---|---|
| 0.2 | 119.34 ± 1.17 | 118.03 ± 0.15 | **−1.1%** | optFwd **wins** |
| 0.4 | 114.80 ± **3.51** | 118.00 ± 0.13 | +2.8% | loses |
| 0.6 | 111.23 ± 1.26 | 118.20 ± 0.17 | +6.3% | loses |
| 0.8 | 110.34 ± 0.76 | 117.80 ± 0.17 | +6.8% | loses |
| 1.0 | 111.72 ± 0.44 | 117.89 ± 0.12 | +5.5% | loses |

**The OFF arm is FLAT** — 117.80 to 118.20, a 0.40 tok/s spread across the whole range — which is
what makes it a baseline rather than a second variable: the sampler's work does not depend on
temperature, only the drawn tokens do. Every movement in the table is the feature's.

**Reading it by the rule fixed above: a real crossover, so gate on temperature with the threshold
set BELOW the measured 0.26 — T ≤ 0.2 on this evidence.** Note the payoff is badly asymmetric: the
win is **1.1%** and the losses run to **6.8%**, so a threshold placed at the crossover risks ~6x what
it gains if the crossover moves on another model. That asymmetry, not the crossover, is the argument
for a conservative default.

**What the current default costs.** optFwd is on for ALL sampled decode. Typical chat sampling sits
at T = 0.7–1.0, which is squarely the losing regime, so goinfer's default sampled decode on this
pairing is **~5.5–6.8% slower than simply turning the feature off**. That is the practical result of
this whole item, and it is larger than the regression that started it.

**Second-order, and it is the hit-rate lottery made visible: the ON arm's VARIANCE peaks mid-ladder**
(sd 3.51 at T=0.4, against 0.12–0.17 everywhere on the OFF arm). At T=0.2 the guess nearly always
hits and at T≥0.6 it nearly always misses; in between, each run draws a different hit rate. So the
feature adds unpredictability as well as cost in the middle of its range — and this, not the anchor
build, is where an overlap-related variance story actually holds. **The anchor's sd 2.07 is still
unexplained**; my earlier attempt to attribute it to overlap was refuted (HEAD-no-optFwd is the
tightest arm measured, sd 0.16).

**SCOPE — n=1 model.** phi3-mini, depth 128, CUDA, one card. The crossover is a property of the
hit-rate curve, which is model- and prompt-dependent; a second model is needed before a threshold
ships. Shipping a gate off one model would repeat precisely the error that put this feature on by
default.

**SECOND MODEL RUNNING (2026-08-27), and it is chosen to TEST the mechanism rather than just add a
point.** qwen2.5-coder-1.5B, vocab **151936** against phi3-mini's 32064. optFwd's benefit is bounded
by how much sampler the overlap can hide, and that share differs by 3.4x between the two (both from
today's HEAD, optFwd OFF):

| model | vocab | decode step | sampler | sampler share of the sampled token |
|---|---|---|---|---|
| phi3-mini | 32064 | 8.026 ms | 0.457 ms | **5.4%** |
| 1.5B | 151936 | 4.535 ms | 1.009 ms | **18.2%** |

**Prediction, fixed before the run: the 1.5B's crossover should sit at a HIGHER temperature than
phi3-mini's 0.26.** More sampler to hide means the overlap keeps paying at hit rates that would
already be uneconomic on phi3-mini. The reasoning is falsifiable in both directions and the outcomes
decide the SHAPE of the gate, not merely its constant:

| outcome | what it means for the gate |
|---|---|
| crossover moves HIGHER, as predicted | The mechanism story holds and the crossover is **model-dependent**. A single temperature constant is then the wrong shape — it must be conservative enough for the WORST model, giving up most of the win on the best. Argues for the hit-rate-adaptive gate. |
| crossover lands at ~0.26 again | Temperature alone predicts the crossover across a 4.7x vocab range. A constant threshold is defensible and much simpler; ship T ≤ 0.2. |
| crossover moves LOWER | The sampler-share reasoning is wrong and something else drives it. Do not gate on either axis until that is understood. |

**SECOND-MODEL RESULT (2026-08-27): PREDICTION CONFIRMED — the crossover moves 0.26 -> 0.95, and a
single temperature constant is therefore the WRONG SHAPE for the gate.** Raw:
`g26-tsweep15-optfwd-{on,off}.json`, log `g26-tsweep15_run.log`. n=6, CUDA, depth 128.

| T | phi3-mini (vocab 32064) | 1.5B (vocab 151936) |
|---|---|---|
| 0.2 | −1.1% **wins** | **−7.4% wins** |
| 0.4 | +2.8% loses | **−6.0% wins** |
| 0.6 | +6.3% loses | **−5.1% wins** |
| 0.8 | +6.8% loses | −0.9% no effect |
| 1.0 | +5.5% loses | −0.9% no effect |
| **crossover** | **≈ 0.26** | **≈ 0.95** |

The sampler-share reasoning predicted both the direction and roughly the size: 18.2% of the sampled
token available to hide on the 1.5B against 5.4% on phi3-mini. **On the 1.5B optFwd is never a
significant loss anywhere in the measured range** — win or neutral throughout. The harm is
concentrated on SMALL-VOCAB models, which is the opposite of where a reader would look for it.

**A SINGLE TEMPERATURE THRESHOLD IS PROVABLY WRONG IN BOTH DIRECTIONS, and this is what the second
model was run to establish:**

- Set it safe for phi3-mini (T ≤ 0.2) and the 1.5B forfeits **6.0% at T=0.4 and 5.1% at T=0.6** —
  real wins, thrown away.
- Set it for the 1.5B (T ≤ 0.95) and phi3-mini pays **2.8 / 6.3 / 6.8 / 5.5%** across T=0.4–1.0.

There is no constant that is right for both, and they differ by only a vocab size.

**REFINEMENT ON WHAT I PRE-REGISTERED, because "gate on hit rate" is not quite sufficient either.**
The pre-registration said this outcome argues for a hit-rate-adaptive gate. That is the right
direction but an incomplete rule: the feature's value is roughly

    value  ~=  p_hit * c_sampler  -  (1 - p_hit) * c_miss

so hit rate alone does not decide it — **`c_sampler` is the other term, and it is exactly what
differs between these two models.** A hit-rate threshold tuned on phi3-mini would still be wrong on
the 1.5B, for the same reason a temperature threshold is. The gate needs both, and both are
measurable at runtime: `OptFwdStats{Guessed, Hit}` already gives the hit rate, and the sampler's
share is the greedy-vs-sampled step cost, which the decode loop can time directly.

**RECOMMENDATION (not implemented — this is a design change and wants sign-off): do not ship a
temperature constant.** The defensible interim is that the current unconditional default is wrong on
small-vocab models and should not stay as-is. Two models is enough to rule out the constant; it is
not enough to fit the two-term rule, which needs points spanning `c_sampler` rather than two of them.

**Temperature is a PROXY and the ladder should not be mistaken for the mechanism.** What determines
the feature's value is the realized hit rate, which `OptFwdStats{Guessed, Hit}` already measures per
run; temperature only predicts it. A hit-rate-adaptive gate is strictly better-targeted than a
temperature threshold and has its input available today — but it can only be *evaluated* against a
ladder like this one, which is why the ladder comes first. **One model, one depth: a gate shipped
off n=1 model would repeat exactly the mistake that put this feature on by default.**


**G25 · Does the oracle bar need a sparsity axis, or is per-precision enough?** — PARKED with a
trigger, 2026-08-27. Either box, desk work first.

**The precision half is DONE** (`c62f2b7`). `decoder/real_oracle_test.go` had one bar, 0.99, whose
own comment said it was the int8 W8A8 number — and `qwen3_next` was measured against it as the
repo's first int4 T3, missing by 0.000124. Now `int8int8`/`int8` keep 0.99, `int4` takes 0.98 (G5,
pre-registered before the checkpoint finished downloading), and an unregistered precision fails hard
rather than inheriting someone else's bar. The oracle passes at its measured 0.989876 and the
`qwen3_next` manifest row is `validated` / `real-model-oracle` (`8f003f2`).

**What is still open is the SHAPE, not the number.** A scalar bar per precision assumes precision is
the only thing that moves the cosine. In a sparse MoE it is not: quantization can change *which*
experts fire, a discrete flip no averaging smooths, so divergence should track sparsity too.

| family | routing | active | measured |
|---|---|---|---|
| `qwen3_next` | 10/512 | 1.95% | 0.98988 at int4 |
| `nemotron3nano` | 6/128 | 4.7% | 0.978 at int8 activations, 0.9977 at f32 |
| dense families | — | — | no expert flipping at all |

Three families at one precision could each want a different bar; `int4 = 0.98` may be loose for a
dense model and tight for something sparser still.

**Parked rather than solved, deliberately.** There is exactly ONE int4 T3 family today, so any
sparsity rule derived now would be fitted to n=1 — and this queue's own standard is that a bar moves
only with a mechanism, not with a number. Fitting a curve through a single point is the same error
wearing a lab coat.

**Trigger to re-open — any one of:**
1. A **second** int4 T3 family lands, giving a real second point.
2. An int4 gate fails within ~0.005 of 0.98, where the bar's shape starts deciding outcomes rather
   than merely being defensible.
3. A dense family is added at int4 — the case where 0.98 is most likely too loose, since it has no
   expert-flip term at all.

Until one fires, the per-precision split is correct and sufficient, and nothing is mis-gated.

## Struck — decided against, kept so the decision is visible

- ~~**Default `top_k`**~~ — truncating the distribution changes which tokens are reachable, which is
  a silent substitution of something other than what was asked. Document it; do not default it.
- ~~**Change the global `--quant` default**~~ — CPU inverts the CUDA quant ordering, so a single
  global default cannot be right for both, and the evidence is one model on one box, never
  reproduced at 1.5B.
- ~~**Force cross-architecture float agreement**~~ — explicit `math.FMA` everywhere is a software
  fallback on amd64 that costs the SIMD performance the CPU backend exists for. Scoped in the policy
  instead.
- ~~**Slab restructure for expert slots**~~ — the control produced the reverse of fragmentation's
  prediction: a fresh heap with ~10 large allocations had *worse* contiguity (32–64 MiB) than the
  slot-loaded heap (96–128 MiB) at the same free figure.
- ~~**aikit branch protection**~~ — required checks force PR-only merges, which is friction against
  a threat model aikit doesn't have. The gate ritual is the enforcement. Revisit at v1.0.
- ~~**Metal `ResidentGreedy` gap**~~ — measured **net-negative**. Kept here rather than under group
  P because it is not work. The 2026-08-12 audit reached the same conclusion **independently**, from
  code, without access to the measurement — recorded as a corroboration of that audit's calibration,
  which is the only reason the entry is worth keeping at all.

## Done

_(append with commit sha and date)_

**G12 · `role: "developer"` silently demoted to a user turn on the OpenAI surfaces** — `mac`,
**DONE `4ca19e9` (2026-08-25).** Claimed and finished the same day. One `case "developer":` arm in
`messagesToTurns` (`internal/serveapp/openai.go`), which all four OpenAI-side surfaces funnel
through, so chat/completions, the Responses message-item path, the tools path and the vision path
were covered at once. Precedence needed no new concept — two `system` messages already resolve
last-one-wins and the alias inherits it.

**Two corrections to the record, both worth more than the fix:**

1. The entry was first written (`9a9594c`) as "the known first blocker for the dsh Tier-0 run". It
   was not a blocker — DeepSeek Harness ships `compat.supportsDeveloperRole: false`, so the run
   could always have happened. Corrected at `0221d32`. What made it worth sequencing first was
   that the failure was **silent**, not that it was blocking.
2. **The remaining Step-0 assumption was wrong, and the wrong half was the one nobody checked.**
   The task doc said a developer-role message on `/v1/messages` was "structurally impossible"
   because Anthropic carries `system` as a top-level field, and asked only that the error be
   confirmed clean. There is no error: `anthropicMessage.Role` is a free string, `anthropicRole`
   maps everything that is not `"assistant"` to a user turn, and **there is no role validation
   anywhere in `internal/serveapp`** — so the same silent demotion was live on that surface too.
   Left demoted **deliberately** (the Anthropic API has no developer role; honoring one invents
   behavior upstream does not have, on a surface whose bar is "works for the apps that matter"),
   but now pinned by `TestAnthropicDeveloperRoleStaysUser` so the decision is visible rather than
   silent. A prediction that a thing is impossible is not a check that it is.

**Gate power, both directions:** a test pinning the old demotion was written first and shown
failing against the fix; the two new alias gates were then shown failing with the arm reverted. The
three that pin unchanged behavior (`instructions` plumbing, the non-goal guard, the Anthropic pin)
pass either way, as they should. Equality gate is byte-identical rendering against the equivalent
`system` request across all six families plus the `rawPrompt` fallback.

**Unblocks** Tier 0 of `docs/scoping-dsh-goinfer.md` — the dsh run can now start past this.

**Follow-up filed as G13** (`docs/queue-engineering.md`): demote-vs-alias was the wrong menu for
`/v1/messages`. Upstream rejects an illegal role, and `anthropicRole`'s
everything-non-assistant-becomes-user mapping means ANY typo'd or invented role silently
restructures the conversation today — `developer` was just the instance that got caught. Role
validation kills the class; the pin above is its recorded before-state. Not blocking Tier 0.

**Second exhibit for E2's lesson.** E2's four "demotion judgments" turned out to be two engineering
bugs that only a released checkpoint could reveal: a generated fixture cannot disagree with the
loader it gates. This is the same shape in different clothes — a predicted branch cannot disagree
with the predictor, so "structurally impossible" typed where a check belonged was, with mechanical
inevitability, the half that was wrong. If `docs/parity-coverage-policy.md` picks up the
fixture rule on its next edit as E2 suggested, both belong in that paragraph.

## Sequencing — release BEFORE G2

**Revised, and D3 is OUT of the release — no rebase attempted.**

> **cut the release → G2 → D3 design read → B1, B2 → mac batch**

The README change in this release is a **retraction**: the workaround language goes away because the
cap holds. D3, if it survives its design read, is an *addition to adjacent text later* — not the same
edit made twice, which was the argument for including it.

A repo-wide mechanical diff immediately before a tag costs bisectability and reasoning room and buys
the modernizers nothing. G2 is not urgent and never was; it is cleared, which is different from being
next.


## Freshness sweep — C, D, E, G (2026-08-12)

F was **fifteen for fifteen already fixed**, because it was seeded from `docs/completed/`. These
groups were seeded from conversation, and **the rate is much lower**, which is the useful result:

| entry | state | evidence |
|---|---|---|
| C1 drain fix — CUDA verification | **open** | no CUDA unload/drain test found |
| C2 out-of-tree consumer audit | **open** | needs a fresh no-repo session by design |
| C3 Metal consumer window | **RUN + CLOSED 2026-08-19 (metal/v0.14.0)** | v0.13.0 run (2026-08-15) found `go install …@v0.13.0` BROKEN by the committed `replace`; replaces removed from all 4 submodule go.mods, RELEASING.md updated. **v0.14.0 re-run (2026-08-19) verifies the fix reached consumers for real:** `go install …/metal/cmd/serve@v0.14.0` builds (13.7 MB, clean GOPATH); resident decode via `Load(Backend:"metal")`+blank-import at **65.4 tok/s out-of-tree** (was 65.2 at v0.13.0 — no regression); cgo-free otool-confirmed both paths. Full note: `docs/measurements/c3-metal-consumer-window-v0.14.0.md` (supersedes the v0.13.0 note, kept as history). |
| C4 soak testing | **open** | `internal/serveapp/fuzz_test.go` and `internal/serveapp/chaos_test.go` exist; neither is an hours-long soak |
| D1 trace tap + launch-site table | **open** | no coverage table in `docs/` |
| D2 launch-wrapper commit 1 | **open** | no `cuda/internal/gen` |
| D3 parked flag-pair | **open**, design read done above | — |
| E1 v1.0 gate as written criteria | **open** | prose item, no tree anchor |
| E2 four per-family demotions | **DONE 2026-08-15 `c3e43c8`** — and NO demotions | all four validated at T3, so the judgment the entry anticipated never had to be made: gpt2 `full-forward-oracle`, granitemoehybrid + nemotron_h `real-model-oracle`, kimi_k2 `shared-path (via deepseek_v3)`. **The tripwire now enforces 23/23, up from 19/23.** The two hybrids needed loader fixes first — neither could load its RELEASED checkpoint, and granite was roping a NoPE model (see the commit; the demotion rule's "unfinished does not qualify" is what forced fixing over demoting) |
| E3 freeze re-declaration | **DONE `cda8cfe`** | re-declared as a proof requirement, with decider and date |
| E4 `scripts/bench_compare.sh` fix or retire | **FIXED** | it now opens with *"goinfer's OWN numbers only. NOT a peer comparison"* and points at `scripts/bench_peer.py`, which drives both sides |
| E5 promo drafts | **unverifiable** | held in conversation, nothing in the tree to check |
| E6 aikit release | **CLOSED 2026-08-12** — superseded by events, not by reversal | aikit cut `v1.17.0`/`gpu/v0.28.0` (`ada417e`); goinfer is on it (`f33fcaf`). The release met E6's own "a reason a consumer can receive" test |
| G1 LFM2.5 family | **open** | no LFM2 code in the tree |
| G2 `go fix` modernizers | **DONE `3d6ae1e`** | — |

**Rate: 1 of 13 previously-open entries was silently already fixed (E4), against F's 15 of 15.**
Two more (E3, G2) were closed by this campaign and were already recorded.

**That difference is the finding.** F was seeded from a *filed audit* — work done elsewhere, reported
once, never propagated back. C/D/E/G were seeded from *conversation*, where the person who did the
work was the person holding the list. **The burial folder is what produced the 15/15, not the passage
of time.** So the sweep paid for itself once, on the group that came from a document, and should not
be assumed to pay again on groups that did not.

E5 is recorded as **unverifiable** rather than open: nothing in the tree can confirm or deny it, which
is a different state and should read as one.


## Description sweep (2026-08-12) — does each entry match its source?

The status sweep found 1 of 13. **D3 shows description can be wrong while status is right, and
description is what someone acts on.** So: for every open entry with a source outside conversation —
a branch, a commit, an audit line, a script — does the entry describe it correctly?

*(A goldens run in a fresh `git worktree` is fixture-less: the same commit proved 33 goldens in the
main checkout and 7 in the worktree, because the fixture checkpoints are gitignored. The refresh
script now says so when skips outnumber passes — `scripts/refresh_parity_hashes.sh`. Found by running
D3's refresh in the rebase worktree and getting `goldens=7`.)*

| entry | source | description matches? |
|---|---|---|
| D3 | branch `flag-pair-moe-cache` + `BRANCH-NOTE.md` | **NO — corrected `2d28358`.** Called a "parked flag-pair" on a workaround premise; it is an API-surface promotion following `KVPrecision` |
| B4 | a stash that does not exist here | **unverifiable** — the description is all that survives, and it names a file that resolves nowhere |
| C1 | `588052b` (the drain fix) | matches — Metal-verified, CUDA arm untested |
| D2 | design recorded in-entry, no branch | matches; no external source to drift from |
| E2 | `testdata/parity_manifest.json` | matched when written; **resolved 2026-08-15** — all four are now `validated`. What the description got wrong was not its status but its FRAMING: it called them "demotion judgments", inheriting the campaign doc's "validation + recording, NOT engineering". Two of the four were engineering, and only a released checkpoint could say so |
| E4 | `scripts/bench_compare.sh` | **stale** — the entry says "status unconfirmed, may still measure the two sides differently"; the script now refuses that use and points at `scripts/bench_peer.py`. Corrected in the status sweep above |
| E6 | aikit tree + `gpu/v0.27.0` | **now stale by design** — the tag it pinned is superseded by `gpu/v0.28.0`. Re-checked at the bump: `be049df` is an ancestor of `gpu/v0.27.0`, and across `gpu/v0.27.0..gpu/v0.28.0` the quantized GEMV PTX is byte-identical |
| G1 | `docs/scoping-lfm2.md` | matches |
| P1, P2, P4, P5, P8 | audit lines + the cited source | match; each carries a measured figure or an explicit ESTIMATE label |

**Split: 9 entries had an external source and were checkable; 2 of those 9 were wrong (D3, E4).
4 entries — C2, C3, E1, E5 — have no source outside conversation and are recorded as unverifiable
rather than checked.**

**THE TRIGGER, not a cadence.** An entry's description **and the specific details inside it** — counts,
file names, measurements that no lint covers — are re-read against their source **at the moment the
item is picked up for work** — nothing schedules this and nothing lints it. That is when the
description matters, when someone is already loading the context anyway, and it is exactly what caught
D3: the read happened because the item was next, not because a sweep came due. The cost falls at the
only point where the drift would have changed what someone did.

**That rate (2 of 9) is higher than the status sweep's (1 of 13), and the two are not the same
population.** A description drifts silently because nothing re-reads it against its source; a status
drifts only when work lands elsewhere. The queue's SHA and path citations are now linted, but **no
lint reads an entry's prose against the branch note or audit line it describes** — that remains a
person's job, and this sweep is its baseline.

## Sequencing

**D3 (loaded and bounded) → the mac batch as one session → B1, B2.**

Within the mac batch, **C3 goes FIRST**, not last: it is the largest completely uncovered surface and
it sank once already. Batching it behind two chores is precisely how that happened. Then
`metal-rope-merge`'s push, then B4's stash check.

**C3 DONE 2026-08-15** (auto-pickup fired on the `v0.13.0` tag). cgo-free/no-Xcode build verified via
the require path; two real gaps found — `go install …/metal/cmd/serve@v0.13.0` is broken by the
committed `replace github.com/townsendmerino/goinfer => ../` in the tagged `metal/go.mod`, and
resident-metal decode isn't drivable via the public API (needs the serve binary, which `go install`
can't build). Full findings + the resolved dep set in `docs/measurements/c3-metal-consumer-window.md`.
**Follow-up worth queuing:** tag `metal/` from a replace-free tree so `go install` works.

**C3 RE-RUN 2026-08-19, CLOSED for real** (auto-pickup fired again — aikit v1.17.1 → v1.21.0 — on
the `v0.14.0`/`metal/v0.14.0` tags). The follow-up above is done: `go install
…/metal/cmd/serve@v0.14.0` now **WORKS** from a clean GOPATH (13.7 MB binary, no checkout) — the
replace-free fix reaching consumers for the first time. Resident-metal decode via the public API
also verified out-of-tree: 65.4 tok/s decode-only, consistent with v0.13.0's 65.2 (no regression
across the aikit bump). RELEASING.md Step 2's two Mac gates (`GOWORK=off go build -tags metal ./...`
standalone build; the device gate — then a bash script, which needed Homebrew bash because macOS's
stock `/bin/bash` 3.2 cannot run `declare -A`; that dependency went away with the script when E8
replaced it with `go run ./cmd/gate gpu`) both passed clean on a committed tree; the gate's only finding was
the pre-existing, already-documented `c8b65ba` local-reflog artifact (allowlisted 2026-08-12,
CI-invisible), unrelated to this release. Full findings in
`docs/measurements/c3-metal-consumer-window-v0.14.0.md`; gate log archived at
`docs/measurements/gpu_gate_metal_v0.14.0_4d91858_FAIL-c8b65ba-only.log`.

## Draft: contents of the next release

**Not a version number** — that is a separate call. This is what has accumulated since
`demo/agent/v0.11.0` (93 commits) that a user would notice, and **none of it depends on the freeze
decision**.

### The headline: the 26B expert cache sizes itself correctly

The defect that opened this campaign was live in the product and is fixed. On an 8 GB card the
runtime auto-capped the MoE expert cache to **34 slots/layer, which allocates and then cannot
launch** — the forward produced **zero tokens**.

- **A5 (`6091e7a`)** — the cap is a **search over the granularity form**, not a division. The driver
  charges each of four buffers per layer its own whole 2 MiB quantum, so the requirement is a step
  function; at 34 all four tip at once, putting it 203,816,960 B over free. Verified through the
  shipping auto-cap path: `capping to 34` → 0 tokens becomes `capping to 33` → coherent output.
- **A9-FIX (`0103b49`)** — the deferred first-launch reservation (`moe_route` takes 138,412,032 B of
  local memory the first time it runs) is now paid **before** the free reading that sizes the cache,
  so the cap is correct by construction rather than covered by a margin. Costs two slots, and that is
  the point: 384 MiB now means 384 MiB.
- **A3 (`e42e83e`)** — a launch OOM now names the kernel and **both** the requested and effective
  slot counts, instead of a bare `cuLaunchKernel: CUDA_ERROR_OUT_OF_MEMORY`.
- **README** — the manual-workaround section is retracted and replaced with what the cap now
  accounts for, plus a version test (`capping to 33` has the fix, `34` does not).

### Performance, all bit-identical

- **P3 (`4c26a58`)** — Gemma's final-logit softcap parallelised: **1.43 ms → 640 µs** per sampled
  token at 262,144 vocab. Sampling path only; greedy never paid it.
- **P6 (`eea7f29`)** — MoE experts share one gate/up buffer pair per token instead of one per expert:
  **16 allocations → 2** at top-k 8.
- **P7 (`91f359f`)** — W4A8 reaches the per-stream `Workspace` it was silently excluded from, ending
  a fresh allocation per projection per token.

### Verification a user can check

- **int4 forward goldens** (`1d0d1ed`) — 23 fixtures across 16 architectures. int4 is the documented
  default quantization and **nothing gated it** before this.
- The goldens refresh went from **19 passed / 0 quantized** to **33 passed / 14 quantized**, and now
  prints its composition rather than a bare count.

### Known-unfixed, disclosed

- **A10** — a ~150 MiB driver allocation floor: memory `cuMemGetInfo` reports as free and
  `cuMemAlloc` will not hand out, at any request size down to 1 MiB. Measured, unattributed. It is
  why the margin cannot simply be lowered to recover the two slots.

<!-- sha-lint: allow c8b65ba UNPUSHED — the PRE-REBASE D3 flag-pair commit, on the local-only branch `flag-pair-moe-cache`; never pushed, so CI cannot resolve it (it failed there 2026-08-12). DELIBERATELY NOT re-pointed to its rebased successor `bacc04c` (on main via the D3 merge): the two have DIFFERENT patch-ids, and the passage citing this one is historical — it describes the branch as it stood BEFORE the rebase ("its branch predates the fix it completes"), which `bacc04c` no longer illustrates. Re-pointing would keep the lint green by making the surrounding sentence false. Allowlisted rather than laundered. Flagged 2026-08-12 -->
<!-- sha-lint: allow d682315 UNPUSHED — Metal branch `metal-rope-merge`, mac-local; not on origin and not in any clone here. Owner: whoever cited it. P4's "already implemented, snapshot-golden byte-exact" rests on a commit only that machine can see; push the branch or the claim stays unverifiable from anywhere else. Flagged 2026-08-12 -->

<!-- CITATION-INDEX: generated by scripts/queue_citation_lint.py --update; do not edit by hand -->

## SHA index

Generated. Every commit id cited above, with the subject it resolved to at the time
of generation. Regenerate with `scripts/queue_sha_lint.py --update`.

| sha | subject |
|---|---|
| `0103b49` | fix(cuda): pay the deferred reservation before sizing the cache (A9-FIX) |
| `0221d32` | docs: the developer-role task is NOT a blocker -- it is silent-wrong, which is worse |
| `1d0d1ed` | test(decoder): int4 forward goldens — 23 fixtures, 16 architectures (Q1c) |
| `25a4711` | refactor(cuda): re-point the device layer onto aikit/gpu v0.3.1 (native-GPU Phase 1) |
| `2d28358` | docs(branch-note): re-derive against the corrected cap (D3 design read) |
| `3358e6ba` | perf(decoder): prefix reuse on the resident KV — agent turn 3 goes 9.13s → 0.42s (21.7x) |
| `3d6ae1e` | chore: go fix modernizers, one deterministic pass (G2) |
| `3f23c96` | feat(llama4): GGUF deepseek2-style loader + real Scout-109B gate (coherent + factual) |
| `41673f7` | re-point doc citations shifted by the laguna edits |
| `4c26a58` | perf(cuda): parallelise the Gemma final-logit softcap, bit-identical (P3) |
| `4ca19e9` | fix(serve): accept `role: "developer"` as an alias for `system` (G12) |
| `4da116d` | perf(decoder): P10 — reuse Sampler's full-vocab scratch buffer across draws |
| `588052b` | serve: drain in-flight requests before freeing an unloaded model (fixes the leak safely) |
| `6091e7a` | fix(cuda): size the expert cache by SEARCH over the granularity form (A5) |
| `61b1e03` | bench: add temp1.0_notrunc, the config §B5's temp-only rows actually used |
| `6a4e0ae` | decoder: optimistic next-token forward for sampled decode (Metal-verified, CUDA untested) |
| `8f003f2` | parity: v0.15.0 sweep GREEN at bd085de; qwen3_next validated by real oracle |
| `91f359f` | fix(decoder): matmulInto dispatches on the property, not on W8A8 (P7) |
| `9a9594c` | docs(prompts): task brief for `role: "developer"` compat on the serve surface |
| `ada417e` | [aikit] scripts: ptx-repro is n/a on darwin, keyed on the PLATFORM not on NVRTC's absence |
| `b067007` | B-12 cuda: make io.Closer contract explicit + propagate teardown error |
| `bacc04c` | feat(serve): --moe-cache-experts / --moe-cache-slots — PARKED on the freeze |
| `bd085de` | test(decoder): build the ETA from the recent window, not all history |
| `be049df` | [aikit] gpu(gemv): explicit __fmaf_rn in the quantized GEMV — the bit-identity contraction rule |
| `c3e43c8` | E2: the four pending families get real oracles — and two of them were decoding released checkpoints wrong |
| `c62f2b7` | test(decoder): give the real-model oracles a bar per precision |
| `ca29d6c` | cuda: resident context cap becomes configuration-derived (-ctx), VRAM-checked at load |
| `cda8cfe` | docs: re-declare the freeze as a proof requirement; clear G2 for amd64 alone |
| `d64afe4` | chore(parity): record real-model validations for the new families + sweep gates |
| `e42e83e` | fix(cuda): name the kernel and both slot counts when a launch runs out of memory |
| `e8fa53c` | G7 follow-up: land the goldens the CUDA mscale declaration was supposed to move |
| `eea7f29` | perf(decoder): one gate/up pair per token in MoE, not one per expert (P6) |
| `f33fcaf` | chore(deps): aikit v1.16.0 -> v1.17.0, aikit/gpu v0.27.0 -> v0.28.0 |

## Path index

Generated. Every `file:line` cited above, the repo it resolved in, and the trimmed
content of that line. A line that MOVED is reported with its new number; content that
has VANISHED is red, because the citation then claims something the file no longer
supports.

| doc \| path:line | repo | line content |
|---|---|---|
| `docs/QUEUE.md|cuda/resident.go:2530` | goinfer | `// ForwardArgmax is the greedy fast path (decoder.ResidentGreedy): reduce the argmax on-` |
| `docs/QUEUE.md|cuda/softcap.go:25` | goinfer | `// This runs on the SAMPLING path only. ForwardArgmax reduces the argmax on-device and r` |
| `docs/audit-2026-09-02.md|chat/chat.go:1` | goinfer | `// Package chat renders a conversation into the exact prompt string a model's` |
| `docs/audit-2026-09-02.md|chat/chat.go:33` | goinfer | `"github.com/townsendmerino/goinfer/tokenizer"` |
| `docs/audit-2026-09-02.md|chat/gemma4_tools.go:28` | goinfer | `if s := strings.TrimSpace(system); s != "" {` |
| `docs/audit-2026-09-02.md|chat/templates.go:156` | goinfer | `func ChatML() *Template {` |
| `docs/audit-2026-09-02.md|chat/templates.go:197` | goinfer | `date := timeNow().Format("02 Jan 2006")` |
| `docs/audit-2026-09-02.md|chat/tools.go:56` | goinfer | `func (t *Template) RenderToolsSegments(system string, turns []Turn, tools []Tool) []Segm` |
| `docs/audit-2026-09-02.md|chat/tools_test.go:100` | goinfer | `// TestGemma4_declaration_byteExact pins the Gemma 4 declaration micro-language` |
| `docs/audit-2026-09-02.md|cmd/gate/configs.go:34` | goinfer | `Env: map[string]string{` |
| `docs/audit-2026-09-02.md|cmd/gate/gpu.go:1015` | goinfer | `func (g *gpuGate) metalSuite() {` |
| `docs/audit-2026-09-02.md|cmd/gate/gpu.go:1029` | goinfer | `func (g *gpuGate) metalCgoFree() {` |
| `docs/audit-2026-09-02.md|cmd/gate/gpu.go:1111` | goinfer | `func (g *gpuGate) metalModel() string {` |
| `docs/audit-2026-09-02.md|cmd/gate/gpu.go:1127` | goinfer | `_, cr, out := g.run(cell{` |
| `docs/audit-2026-09-02.md|cmd/gate/gpu.go:1161` | goinfer | `_, cr, out := g.run(cell{` |
| `docs/audit-2026-09-02.md|cmd/gate/gpu.go:1253` | goinfer | `lint := exec.Command("python3", "scripts/queue_citation_lint.py")` |
| `docs/audit-2026-09-02.md|cmd/gate/gpu.go:219` | goinfer | `func (g *gpuGate) noteIfEmpty(c cell, res *results) {` |
| `docs/audit-2026-09-02.md|cmd/gate/gpu.go:349` | goinfer | `func detectBackend() string {` |
| `docs/audit-2026-09-02.md|cmd/gate/gpu.go:397` | goinfer | `for _, v := range []string{"GOINFER_HEAVY_TESTS", "GOINFER_DRAIN_GROUP"} {` |
| `docs/audit-2026-09-02.md|cmd/gate/gpu.go:406` | goinfer | `switch g.backend {` |
| `docs/audit-2026-09-02.md|cmd/gate/gpu.go:409` | goinfer | `case "metal":` |
| `docs/audit-2026-09-02.md|cmd/gate/gpu.go:456` | goinfer | `switch g.backend {` |
| `docs/audit-2026-09-02.md|cmd/gate/gpu.go:543` | goinfer | `// The header used to read "CUDA kernels + parity" while running NEITHER the resident pa` |
| `docs/audit-2026-09-02.md|cmd/gate/gpu.go:544` | goinfer | `// NOR anything that asserts a forward. Every resident parity gate is behind `goinfer_te` |
| `docs/audit-2026-09-02.md|cmd/gate/gpu.go:578` | goinfer | `// ---- 2b. resident PARITY gates — the forward is asserted here ----` |
| `docs/audit-2026-09-02.md|cmd/gate/gpu.go:692` | goinfer | `Env: map[string]string{"CGO_ENABLED": "0", "GOINFER_HEAVY_TESTS": "1"},` |
| `docs/audit-2026-09-02.md|cmd/gate/gpu.go:793` | goinfer | `bin := filepath.Join(os.TempDir(), "gpu_gate_serve")` |
| `docs/audit-2026-09-02.md|cmd/gate/gpu.go:864` | goinfer | `if v := os.Getenv("GOINFER_NVRTC_DIRS"); v != "" {` |
| `docs/audit-2026-09-02.md|cmd/gate/gpu.go:938` | goinfer | `ptxFiles, _ := filepath.Glob(filepath.Join("cuda", "testdata", "*.ptx"))` |
| `docs/audit-2026-09-02.md|cmd/gate/parity.go:158` | goinfer | `// DERIVED FROM THE TAGGED FILES, not hand-written. The hand-written pattern could` |
| `docs/audit-2026-09-02.md|cmd/gate/parity.go:19` | goinfer | `// THIS IS A CHECKSET, NOT A TALLY, and that is the whole reason it needs its own decisi` |
| `docs/audit-2026-09-02.md|cmd/gate/parity.go:283` | goinfer | `cfg := &gateConfig{Name: "parity", Decision: "checkset", TopLevelOnly: true, RCIsFailure` |
| `docs/audit-2026-09-02.md|cmd/gate/parity.go:292` | goinfer | `for _, c := range cfg.Cells {` |
| `docs/audit-2026-09-02.md|cmd/gate/parity.go:314` | goinfer | `// Safety net: any OTHER parity/gate-shaped test that skipped — a family the lists forgo` |
| `docs/audit-2026-09-02.md|cmd/gate/parity.go:382` | goinfer | `// THE FOURTH OUTCOME (B14). A gate failing with no confirmed prior result is asserting ` |
| `docs/audit-2026-09-02.md|cmd/gate/parity.go:536` | goinfer | `// This exists because the distinction cost five weeks. TestQwen3NextReal_oracle was rep` |
| `docs/audit-2026-09-02.md|cmd/gate/parity.go:545` | goinfer | `func whyNoResult(test string, cells []cell) string {` |
| `docs/audit-2026-09-02.md|cmd/gate/parity_test.go:283` | goinfer | `func TestRealckptCellCanReachEveryGate(t *testing.T) {` |
| `docs/audit-2026-09-02.md|cmd/gate/parity_test.go:319` | goinfer | `func TestParity_missingGateSaysWhichCause(t *testing.T) {` |
| `docs/audit-2026-09-02.md|constrain/constrain.go:153` | goinfer | `logits[id] = neg` |
| `docs/audit-2026-09-02.md|constrain/constrain.go:198` | goinfer | `if len(m.eosIDs) > 0 {` |
| `docs/audit-2026-09-02.md|constrain/constrain.go:50` | goinfer | `isEOS  []bool` |
| `docs/audit-2026-09-02.md|constrain/constrain.go:52` | goinfer | `// plainOK marks ids that are unconditionally legal inside a JSON string and leave the` |
| `docs/audit-2026-09-02.md|constrain/json.go:100` | goinfer | `func (g *jsonGrammar) TryBytes(bs []byte) bool {` |
| `docs/audit-2026-09-02.md|constrain/json.go:84` | goinfer | `func (g *jsonGrammar) CanEnd() bool {` |
| `docs/audit-2026-09-02.md|constrain/reflect.go:12` | goinfer | `// GrammarFromStruct derives a JSON Schema from a Go struct (via its json tags)` |
| `docs/audit-2026-09-02.md|constrain/reflect.go:195` | goinfer | `case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:` |
| `docs/audit-2026-09-02.md|constrain/reflect.go:77` | goinfer | `for f := range t.Fields() {` |
| `docs/audit-2026-09-02.md|constrain/schema.go:196` | goinfer | `propsRaw, _ := s["properties"].(map[string]any)` |
| `docs/audit-2026-09-02.md|constrain/schema.go:200` | goinfer | `// An object with no declared properties and no `additionalProperties:false` is the` |
| `docs/audit-2026-09-02.md|constrain/schema.go:318` | goinfer | `// encodeLiteral renders an enum/const value to the compact JSON bytes the model must` |
| `docs/audit-2026-09-02.md|constrain/schema_grammar.go:112` | goinfer | `func (g *schemaGrammar) TryBytes(bs []byte) bool {` |
| `docs/audit-2026-09-02.md|constrain/schema_grammar.go:78` | goinfer | `// CanEnd reports whether the committed output is a complete document: the root` |
| `docs/audit-2026-09-02.md|constrain/schema_grammar.go:88` | goinfer | `func (g *schemaGrammar) CanEnd() bool {` |
| `docs/audit-2026-09-02.md|constrain/schema_test.go:183` | goinfer | `out := genConstrained(t, g, int64(i)+1, 15) // cap digits so ints fit int64` |
| `docs/audit-2026-09-02.md|constrain/tool_grammar.go:31` | goinfer | `if len(paramSchema) == 0 {` |
| `docs/audit-2026-09-02.md|constrain/tool_grammar.go:37` | goinfer | `name, err := encodeLiteral(toolName)` |
| `docs/audit-2026-09-02.md|cuda/backend.go:1014` | goinfer | `L.hd, L.nKV, L.rhalf, L.qDim, L.kvDim = 0, 0, 0, 0, 0` |
| `docs/audit-2026-09-02.md|cuda/backend.go:204` | goinfer | `hls := make([]hlayer, nLayers)` |
| `docs/audit-2026-09-02.md|cuda/backend.go:225` | goinfer | `hl.isDeltaNet = true` |
| `docs/audit-2026-09-02.md|cuda/backend.go:455` | goinfer | `anchor: func (b *cudaBackend) BuildResident(m *decoder.Model) (rf decoder.ResidentForwar` |
| `docs/audit-2026-09-02.md|cuda/backend.go:499` | goinfer | `ctxCap:      resolveCtxCap(m.ResidentContextRequest(), m.Config().MaxPositions),` |
| `docs/audit-2026-09-02.md|cuda/backend.go:53` | goinfer | `func layerFusable(qkvInt4, moe, guInt4 bool) bool {` |
| `docs/audit-2026-09-02.md|cuda/backend.go:559` | goinfer | `r.cacheSlots = topK` |
| `docs/audit-2026-09-02.md|cuda/backend.go:617` | goinfer | `// THESE MODULE AND PIPELINE HANDLES DO NOT SURVIVE DEVICE EXHAUSTION. Read this before` |
| `docs/audit-2026-09-02.md|cuda/backend.go:703` | goinfer | `// fix didn't force a glue.ptx regen. (glue.ptx and all production PTX except moe.ptx ar` |
| `docs/audit-2026-09-02.md|cuda/backend.go:719` | goinfer | `if pbmod, e3 := r.dev.CompileLibrary(prefillBatchedPTX); e3 == nil {` |
| `docs/audit-2026-09-02.md|cuda/backend.go:805` | goinfer | `r.splitkvAttn = os.Getenv("GOINFER_SPLITKV_ATTN") != "0"` |
| `docs/audit-2026-09-02.md|cuda/blockspec_test.go:63` | goinfer | `maxNew := 96` |
| `docs/audit-2026-09-02.md|cuda/blockspec_test.go:91` | goinfer | `ch, gen := mc.Generate(context.Background(), prompt, len(got), decoder.SamplingParams{})` |
| `docs/audit-2026-09-02.md|cuda/doc.go:10` | goinfer | `//     dlopen libcuda.so.1 at runtime, so `CGO_ENABLED=0` and the single-static-` |
| `docs/audit-2026-09-02.md|cuda/doc.go:19` | goinfer | `// Build with `-tags cuda`. BuildResident is live: it builds a resident forward when the` |
| `docs/audit-2026-09-02.md|cuda/drafter.go:112` | goinfer | `d.fc = r.upW(fcw)` |
| `docs/audit-2026-09-02.md|cuda/drafter.go:187` | goinfer | `if n > d.ctxCap {` |
| `docs/audit-2026-09-02.md|cuda/drafter.go:265` | goinfer | `need := d.ctxLen + n` |
| `docs/audit-2026-09-02.md|cuda/drafter.go:298` | goinfer | `if n > d.extCap {` |
| `docs/audit-2026-09-02.md|cuda/drafter.go:427` | goinfer | `if d.ctxLen+M > d.kvCap {` |
| `docs/audit-2026-09-02.md|cuda/drafter.go:495` | goinfer | `if e := d.r.launch(d.attnBlock, LaunchConfig{GridX: uint32(nH), GridY: uint32(M), GridZ:` |
| `docs/audit-2026-09-02.md|cuda/drafter.go:506` | goinfer | `gpu.ArgValue(int32(d.ctxLen)), gpu.ArgValue(d.attnScale()),` |
| `docs/audit-2026-09-02.md|cuda/drafter.go:618` | goinfer | `if M > d.headCap {` |
| `docs/audit-2026-09-02.md|cuda/drafter_loop_test.go:267` | goinfer | `const maxNew = 96` |
| `docs/audit-2026-09-02.md|cuda/drafter_vs_off_test.go:145` | goinfer | `maxNew := 96` |
| `docs/audit-2026-09-02.md|cuda/foreign_context_test.go:52` | goinfer | `func foreignCUDAContexts() (out []foreignCtx, ok bool) {` |
| `docs/audit-2026-09-02.md|cuda/gptoss_cache_ab_test.go:52` | goinfer | `opts.MoECacheSlots = wantSlots` |
| `docs/audit-2026-09-02.md|cuda/gptoss_real20b_test.go:36` | goinfer | `// Skips until CUDA declares the two features, exactly as metal/gptoss_real_test.go does` |
| `docs/audit-2026-09-02.md|cuda/gptoss_real20b_test.go:44` | goinfer | `// modelPath, NOT a direct environment read: the asset registry owns GOINFER_GPTOSS_GGUF` |
| `docs/audit-2026-09-02.md|cuda/graphs_safe.go:109` | goinfer | `// admitGraphs applies the safe-gate: it is the ONLY place r.graphs is promoted from "re` |
| `docs/audit-2026-09-02.md|cuda/kernel_fma_lint_test.go:15` | goinfer | `// moe.cu is exempt because the shipped moe.ptx was a FROZEN artifact, audited at NVRTC ` |
| `docs/audit-2026-09-02.md|cuda/kernels.go:107` | goinfer | `// this box's NVRTC 12.9.86, not 12.6. moe.ptx was the audited 12.6.85 artifact (R-26) a` |
| `docs/audit-2026-09-02.md|cuda/kernels.go:179` | goinfer | `func f32tof16(f float32) uint16 {` |
| `docs/audit-2026-09-02.md|cuda/prefill.go:157` | goinfer | `func (r *cudaResident) prefillStaticDecline() error {` |
| `docs/audit-2026-09-02.md|cuda/prefill.go:216` | goinfer | `func nonBatchableKind(Ly *cudaLayer) string {` |
| `docs/audit-2026-09-02.md|cuda/prefill.go:264` | goinfer | `if e := r.checkCap(startPos, M); e != nil {` |
| `docs/audit-2026-09-02.md|cuda/prefill.go:289` | goinfer | `var scratch []Buffer` |
| `docs/audit-2026-09-02.md|cuda/prefill.go:382` | goinfer | `if e := r.launch(r.bAttn, LaunchConfig{GridX: uint32(r.nH), GridY: uint32(M), GridZ: 1, ` |
| `docs/audit-2026-09-02.md|cuda/prefill.go:391` | goinfer | `gpu.ArgValue(Ly.window), gpu.ArgValue(int32(M)), Arg(cctxB), r.sinkArg(l)); e != nil {` |
| `docs/audit-2026-09-02.md|cuda/prefill.go:499` | goinfer | `for m := first; m < M; m++ {` |
| `docs/audit-2026-09-02.md|cuda/prefill.go:552` | goinfer | `func (r *cudaResident) batchedHeadArgmax(xB, aqB, aScB Buffer, M int, out *[]int) error ` |
| `docs/audit-2026-09-02.md|cuda/prefill.go:567` | goinfer | `// ONE head GEMV for all M rows: the weights are read once instead of M times.` |
| `docs/audit-2026-09-02.md|cuda/resident.go:1014` | goinfer | `for j := 0; j < r.topK; j++ {` |
| `docs/audit-2026-09-02.md|cuda/resident.go:1340` | goinfer | `if r.prefillReady && r.dnet == nil {` |
| `docs/audit-2026-09-02.md|cuda/resident.go:1347` | goinfer | `if startPos == 0 {` |
| `docs/audit-2026-09-02.md|cuda/resident.go:1371` | goinfer | `if e := r.checkCap(0, len(keys)/kvDim); e != nil {` |
| `docs/audit-2026-09-02.md|cuda/resident.go:1499` | goinfer | `r.reqCh = nil` |
| `docs/audit-2026-09-02.md|cuda/resident.go:152` | goinfer | `func splitkvThreshold(nH, hd int) int {` |
| `docs/audit-2026-09-02.md|cuda/resident.go:1636` | goinfer | `func (r *cudaResident) capVec(src Buffer, dst [][]float32, l, n int) {` |
| `docs/audit-2026-09-02.md|cuda/resident.go:2355` | goinfer | `func (r *cudaResident) launchToken(emb []float32, pos int, head bool) error {` |
| `docs/audit-2026-09-02.md|cuda/resident.go:2425` | goinfer | `nWin := nKeys` |
| `docs/audit-2026-09-02.md|cuda/resident.go:2443` | goinfer | `if r.splitkvAttn && r.skScores != (Pipeline{}) && (mustSplit \|\| nWin >= r.splitkvMin(Ly.` |
| `docs/audit-2026-09-02.md|cuda/resident.go:2462` | goinfer | `if err := r.launch(r.bAttn, LaunchConfig{GridX: uint32(r.nH), GridY: 1, GridZ: 1, BlockX` |
| `docs/audit-2026-09-02.md|cuda/resident.go:2467` | goinfer | `if err := r.launch(r.fAttn, LaunchConfig{GridX: uint32(r.nH), GridY: 1, GridZ: 1, BlockX` |
| `docs/audit-2026-09-02.md|cuda/resident.go:2778` | goinfer | `func (r *cudaResident) layerTail(Ly *cudaLayer, l int, gC bool) error {` |
| `docs/audit-2026-09-02.md|cuda/resident.go:343` | goinfer | `// hidCap is the PRODUCTION hidden-state seam (P10 / docs/spec/08): the resident` |
| `docs/audit-2026-09-02.md|cuda/resident.go:580` | goinfer | `func (r *cudaResident) cacheWQ(h hostW) cudaWQ {` |
| `docs/audit-2026-09-02.md|cuda/resident.go:733` | goinfer | `if decline {` |
| `docs/audit-2026-09-02.md|cuda/resident.go:951` | goinfer | `// Synchronize — right for per-request uploads, wrong here: a MoE decode token loads ~12` |
| `docs/audit-2026-09-02.md|cuda/resident.go:979` | goinfer | `func (r *cudaResident) loadRoutedExperts(L *cudaLayer) error {` |
| `docs/audit-2026-09-02.md|cuda/resident.go:998` | goinfer | `if e := r.stream.Sync(); e != nil {` |
| `docs/audit-2026-09-02.md|cuda/theta_probe_test.go:49` | goinfer | `for _, mdl := range []string{` |
| `docs/audit-2026-09-02.md|decoder/a3_fanout_test.go:190` | goinfer | `func TestAttendF32Fanout_bitIdentical(t *testing.T) {` |
| `docs/audit-2026-09-02.md|decoder/a3_moe_exclusion_test.go:15` | goinfer | `// forwardn.go excludes MoE from --cpu-fast-attention unconditionally, on a stated` |
| `docs/audit-2026-09-02.md|decoder/api_tiers_test.go:67` | goinfer | `bare := name` |
| `docs/audit-2026-09-02.md|decoder/arch.go:559` | goinfer | `func (a *Architecture) finalizeRoPE() {` |
| `docs/audit-2026-09-02.md|decoder/assets.go:49` | goinfer | `if m := os.Getenv("GOINFER_MODELS"); m != "" {` |
| `docs/audit-2026-09-02.md|decoder/attention.go:11` | goinfer | `func addBias(x, b []float32) {` |
| `docs/audit-2026-09-02.md|decoder/attention.go:124` | goinfer | `// 4. Append this position's K/V, then attend over the stored history. Route` |
| `docs/audit-2026-09-02.md|decoder/attention.go:139` | goinfer | `ctx := cache.scr.ctx[:nH*hd]` |
| `docs/audit-2026-09-02.md|decoder/attention.go:148` | goinfer | `pool := scr.headWorkerPool(nH, 1, nKeys, hd, !acc64 && cache.treeMask == nil)` |
| `docs/audit-2026-09-02.md|decoder/blockspec.go:179` | goinfer | `// LOSSLESS BY CONSTRUCTION: every emitted token is one the TARGET's own argmax produced` |
| `docs/audit-2026-09-02.md|decoder/blockspec.go:185` | goinfer | `anchor: func (m *Model) NewBlockSpec(dw BlockDrafterWeights, taps []int) (*BlockSpec, er` |
| `docs/audit-2026-09-02.md|decoder/blockspec.go:230` | goinfer | `rd.TruncateContext(0) // fresh sequence: the previous generation's context must not leak` |
| `docs/audit-2026-09-02.md|decoder/blockspec.go:238` | goinfer | `for i := range n {` |
| `docs/audit-2026-09-02.md|decoder/blockspec.go:286` | goinfer | `for opt.MaxTokens <= 0 \|\| len(out) < opt.MaxTokens {` |
| `docs/audit-2026-09-02.md|decoder/blockspec.go:349` | goinfer | `blockIn := make([][]float32, width)` |
| `docs/audit-2026-09-02.md|decoder/blockspec.go:354` | goinfer | `trunk, e := rd.DraftBlock(blockIn)` |
| `docs/audit-2026-09-02.md|decoder/blockspec.go:385` | goinfer | `// TRUNCATE BEFORE EOS INSIDE THE BURST. A round commits several tokens at once, so a` |
| `docs/audit-2026-09-02.md|decoder/blockspec.go:482` | goinfer | `func (s *BlockSpec) GenerateStream(ctx context.Context, prompt []int, maxTokens int,` |
| `docs/audit-2026-09-02.md|decoder/blockspec.go:530` | goinfer | `stats.Emitted = len(toks)` |
| `docs/audit-2026-09-02.md|decoder/blockspec_cpu_test.go:87` | goinfer | `n := min(len(base), len(got))` |
| `docs/audit-2026-09-02.md|decoder/config.go:1048` | goinfer | `// resolveEOSIDs returns the ids that end generation: config.json's` |
| `docs/audit-2026-09-02.md|decoder/config.go:1090` | goinfer | `func loadConfig(fsys fs.FS, name string) (*Config, error) {` |
| `docs/audit-2026-09-02.md|decoder/config.go:143` | goinfer | `AttentionChunkSize     int     `json:"attention_chunk_size"`` |
| `docs/audit-2026-09-02.md|decoder/config.go:491` | goinfer | `func (c *Config) validateGptOss() error {` |
| `docs/audit-2026-09-02.md|decoder/config.go:645` | goinfer | `func (c *Config) normalizeQwen3NextLayerTypes() error {` |
| `docs/audit-2026-09-02.md|decoder/deepseek_real_test.go:210` | goinfer | `prompt := "The capital of France is"` |
| `docs/audit-2026-09-02.md|decoder/deltanet.go:21` | goinfer | `var deltaNetTiming = os.Getenv("GOINFER_DELTANET_TIMING") != ""` |
| `docs/audit-2026-09-02.md|decoder/deltanet.go:226` | goinfer | `S := st.s[headV*hk*hv : (headV+1)*hk*hv] // [hk, hv]` |
| `docs/audit-2026-09-02.md|decoder/deltanet.go:77` | goinfer | `// THE THREE DOMINANT PROJECTIONS ARE QUANTIZABLE (2026-08-19). They were []float32 —` |
| `docs/audit-2026-09-02.md|decoder/deltanet_chunked.go:10` | goinfer | `// matmuls). It is NOT yet wired into the forward — the qwen3_5_moe path is still` |
| `docs/audit-2026-09-02.md|decoder/deltanet_chunked.go:137` | goinfer | `// decay of ~0.2 gives 0.2^64 ≈ 1e-45, below f32 min-normal. c[m] then rounds to ZERO an` |
| `docs/audit-2026-09-02.md|decoder/deltanet_chunked_test.go:62` | goinfer | `aLog[i] = -2 * rng.Float32() // A_log in [-2,0] → gt = exp(g) in (0,1), stable` |
| `docs/audit-2026-09-02.md|decoder/dflash.go:651` | goinfer | `scale := 1 / math.Sqrt(float64(hd))` |
| `docs/audit-2026-09-02.md|decoder/embed.go:101` | goinfer | `for _, id := range ids {` |
| `docs/audit-2026-09-02.md|decoder/embed.go:36` | goinfer | `func (m *Model) HiddenLast(ids []int) ([]float32, error) {` |
| `docs/audit-2026-09-02.md|decoder/embed.go:43` | goinfer | `if _, own := a.ownForward(); own {` |
| `docs/audit-2026-09-02.md|decoder/embed.go:85` | goinfer | `cache := m.NewCache(len(ids))` |
| `docs/audit-2026-09-02.md|decoder/features.go:261` | goinfer | `// residentBackendMoECap is the router-kernel capacity of each backend whose MoE scorebo` |
| `docs/audit-2026-09-02.md|decoder/features.go:273` | goinfer | `"webgpu": {experts: 512, groups: 32}, // gpu/moe.go: MAXE 512, array<f32,512> score/sel ` |
| `docs/audit-2026-09-02.md|decoder/features.go:433` | goinfer | `//   FeatAttnSink  the learned per-head softmax sink, the clamped interleaved-SwiGLU` |
| `docs/audit-2026-09-02.md|decoder/features_test.go:423` | goinfer | `// The cap was raised 256 -> 512 (MOE_MAX_E / MAXE) so Kimi-K2's 384 is now ADMITTED on ` |
| `docs/audit-2026-09-02.md|decoder/forward_gemma4.go:216` | goinfer | `if lw.LayerScalar != 0 {` |
| `docs/audit-2026-09-02.md|decoder/forward_gemma4.go:301` | goinfer | `for kvh := range nKV {` |
| `docs/audit-2026-09-02.md|decoder/forward_gemma4_moe.go:67` | goinfer | `if routerCapture {` |
| `docs/audit-2026-09-02.md|decoder/forward_gptoss.go:130` | goinfer | `sinkLogit := float64(lw.AttnSinks[qh])` |
| `docs/audit-2026-09-02.md|decoder/forward_lfm2.go:111` | goinfer | `pos := cache.Pos()` |
| `docs/audit-2026-09-02.md|decoder/forward_lfm2.go:126` | goinfer | `mix = shortConvStep(n, lw.shortConv, *g, hidden, cache.conv[l])` |
| `docs/audit-2026-09-02.md|decoder/forward_lfm2.go:32` | goinfer | `bcx := matvec(w.inProj, 3*cd, hidden, n)` |
| `docs/audit-2026-09-02.md|decoder/forward_llama4.go:16` | goinfer | `// Chunked (local) attention on the RoPE layers: a query at position p attends only with` |
| `docs/audit-2026-09-02.md|decoder/forward_llama4.go:93` | goinfer | `attendQuery(q, ctx, cache.scr.scoresBuf(nKeys), cache, layer, pos, true /*no sliding win` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:106` | goinfer | `if _, own := a.ownForward(); own {` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:1104` | goinfer | `// position ([K][VocabSize]) — used by the speculative verifier. Bit-identical to` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:1146` | goinfer | `h, err := m.forwardLayersN(reqCtx, ids, cache, fastAttn)` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:1185` | goinfer | `h, err := m.forwardLayersN(ctx, prompt, cache, cpuFastAttention())` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:125` | goinfer | `// It is the arch-side view of KVCache.hasRecurrentState(), read from the dispatch table` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:176` | goinfer | `// reused across the K rows (aikit's column-blocked W8A8 kernel); attention stays` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:222` | goinfer | `norm := make([]float32, K*hidden)` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:233` | goinfer | `maxKeys := startPos + K` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:312` | goinfer | `if fastAttn && K < fastAttnMinPrompt {` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:32` | goinfer | `// WHAT IT GIVES UP IS BIGGER THAN "prefill != decode", and the help text understated it` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:337` | goinfer | `attnPool := newHeadWorkerPool(prefillAttnWorkers(K, maxKeys, hd, arch.maxHeads()), K, ma` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:355` | goinfer | `var ws linalg.Workspace` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:393` | goinfer | `if err := reqCtx.Err(); err != nil {` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:418` | goinfer | `matmul(be, &lw.QProj, norm, q, K)` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:424` | goinfer | `addBias(row(q, i, qDim), lw.QBias)` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:439` | goinfer | `if arch.QKNorm {` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:44` | goinfer | `// TestSessionFastAttnDivergence pins the new behaviour; the equality is still gated, un` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:461` | goinfer | `if isLocal {` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:462` | goinfer | `base, nRows := cache.batchReadLocal(l, startPos, K, k, v, alk, alv)` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:50` | goinfer | `// MoE IS *NOT* EXCLUDED, and that is deliberate: 66d0a05 removed the exclusion after me` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:555` | goinfer | `if moeExpertMajor() {` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:556` | goinfer | `emOut = make([]float32, K*hidden)` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:559` | goinfer | `ok, err := moeMLPBatch(norm[c0*hidden:c1*hidden], c1-c0, lw, arch, be, m.pager,` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:59` | goinfer | `func cpuFastAttention() bool { return os.Getenv("GOINFER_CPU_FAST_ATTENTION") != "0" }` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:661` | goinfer | `cache.advanceTo(startPos + K)` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:733` | goinfer | `nKeys := len(keys) / kvDim` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:760` | goinfer | `fusedOK := !useAcc64 && cache.treeMask == nil` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:784` | goinfer | `tile := attendTileFor(ws, K, nKeys, hd)` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:789` | goinfer | `copy(qh[i*hd:i*hd+hd], q[b:b+hd])` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:952` | goinfer | `gatherKV := func(ws *headWorkerScratch, kvh int) {` |
| `docs/audit-2026-09-02.md|decoder/forwardn.go:99` | goinfer | `func (m *Model) canBatchN(K int) bool {` |
| `docs/audit-2026-09-02.md|decoder/forwardn_test.go:160` | goinfer | `for i := range K {` |
| `docs/audit-2026-09-02.md|decoder/forwardn_test.go:97` | goinfer | `// TestForwardN_matchesSequential checks the batched multi-position forward` |
| `docs/audit-2026-09-02.md|decoder/fp8.go:124` | goinfer | `// Shape is checked against the ARCHITECTURE (in/out from the config), not just against` |
| `docs/audit-2026-09-02.md|decoder/fusedattn.go:114` | goinfer | `a0, a1 := max(lo[i], k0), min(hi[i], k1-1)` |
| `docs/audit-2026-09-02.md|decoder/fusedattn.go:45` | goinfer | `func fusedAttention() bool { return os.Getenv("GOINFER_FUSED_ATTENTION") != "0" }` |
| `docs/audit-2026-09-02.md|decoder/fusedattn.go:85` | goinfer | `anchor: func attendTileFused(` |
| `docs/audit-2026-09-02.md|decoder/gguf.go:1251` | goinfer | `// canSerialize once refused MLA / Mamba-2 / Gemma-4 PLE / Llama-4 here; since v6 the wr` |
| `docs/audit-2026-09-02.md|decoder/gguf.go:1561` | goinfer | `if arch.gemma4 != nil {` |
| `docs/audit-2026-09-02.md|decoder/gguf.go:1564` | goinfer | `if err := sink.writeHeadGlobals(w, id); err != nil {` |
| `docs/audit-2026-09-02.md|decoder/gguf.go:1789` | goinfer | `// expert weights are MXFP4 in the real checkpoint — stackedExperts routes them` |
| `docs/audit-2026-09-02.md|decoder/gguf.go:1885` | goinfer | `if err := parallelLayers(arch.NumLayers, loadGptOss); err != nil {` |
| `docs/audit-2026-09-02.md|decoder/gguf.go:27` | goinfer | `// Architectures: whatever decoder/registry.go's `registry` map carries (34 as of 2026-0` |
| `docs/audit-2026-09-02.md|decoder/gguf.go:46` | goinfer | `defer func() {` |
| `docs/audit-2026-09-02.md|decoder/gguf.go:488` | goinfer | `numLayers := u("block_count")` |
| `docs/audit-2026-09-02.md|decoder/gguf.go:569` | goinfer | `if k := u("leading_dense_block_count"); k > 0 {` |
| `docs/audit-2026-09-02.md|decoder/gptoss_real_test.go:61` | goinfer | `prompt := "The capital of France is"` |
| `docs/audit-2026-09-02.md|decoder/gptoss_safetensors.go:112` | goinfer | `{&l.QProj, &l.QBias, "q_proj", qDim, hidden},` |
| `docs/audit-2026-09-02.md|decoder/gptoss_safetensors.go:97` | goinfer | `for i := range arch.NumLayers {` |
| `docs/audit-2026-09-02.md|decoder/gptq.go:41` | goinfer | `func parseQuantConfig(raw json.RawMessage) (*quantConfig, error) {` |
| `docs/audit-2026-09-02.md|decoder/int4f16scales.go:10` | goinfer | `// GOINFER_INT4_F16_SCALES=1 is a DIAGNOSTIC (default-off, one env read per weight at LO` |
| `docs/audit-2026-09-02.md|decoder/int4f16scales.go:41` | goinfer | `func f32ToF16bits(f float32) uint16 {` |
| `docs/audit-2026-09-02.md|decoder/kvcache.go:235` | goinfer | `func (r *ring) truncate(p int) bool {` |
| `docs/audit-2026-09-02.md|decoder/kvcache.go:239` | goinfer | `exact := r.count <= r.w \|\| p >= r.count-1` |
| `docs/audit-2026-09-02.md|decoder/kvcache.go:408` | goinfer | `func (c *KVCache) resetRecurrent() {` |
| `docs/audit-2026-09-02.md|decoder/kvcache.go:458` | goinfer | `if pos == 0 {` |
| `docs/audit-2026-09-02.md|decoder/kvcache.go:470` | goinfer | `for l := range c.numLayers {` |
| `docs/audit-2026-09-02.md|decoder/kvcache.go:515` | goinfer | `func (c *KVCache) batchReadLocal(layer, startPos, K int, newK, newV, dstK, dstV []float3` |
| `docs/audit-2026-09-02.md|decoder/kvcache.go:521` | goinfer | `base = max(startPos-r.w+1, 0)` |
| `docs/audit-2026-09-02.md|decoder/kvcache_recurrent_test.go:13` | goinfer | `c.mamba = []*mamba2State{{ssm: []float32{1, 2, 3}, convWin: [][]float32{{9}}}}` |
| `docs/audit-2026-09-02.md|decoder/kvsnapshot.go:214` | goinfer | `// M-04: numLayers == 0 and kvDim == 0 USED TO PASS — only negatives and over-maxes were` |
| `docs/audit-2026-09-02.md|decoder/kvsnapshot.go:241` | goinfer | `// Each of the pos positions stores at least one byte across the numLayers·kvDim KV, so ` |
| `docs/audit-2026-09-02.md|decoder/kvsnapshot.go:279` | goinfer | `if st != kvDim \|\| rr.count < 0 \|\| nLive > rr.w \|\| rr.count < nLive {` |
| `docs/audit-2026-09-02.md|decoder/kvsnapshot.go:312` | goinfer | `} else if quant == kvI8 {` |
| `docs/audit-2026-09-02.md|decoder/kvsnapshot.go:71` | goinfer | `if c.hasRecurrentState() \|\| len(c.mlaLatent) > 0 {` |
| `docs/audit-2026-09-02.md|decoder/layerpaging.go:107` | goinfer | `const ahead = 1` |
| `docs/audit-2026-09-02.md|decoder/layerpaging.go:64` | goinfer | `if _, own := w.arch.ownForward(); own {` |
| `docs/audit-2026-09-02.md|decoder/lfm2_test.go:178` | goinfer | `for _, id := range g.PromptIDs {` |
| `docs/audit-2026-09-02.md|decoder/llama4_real_test.go:52` | goinfer | `prompt := "The capital of France is"` |
| `docs/audit-2026-09-02.md|decoder/longprompt_golden_test.go:84` | goinfer | `os.Unsetenv("GOINFER_CPU_FAST_ATTENTION")` |
| `docs/audit-2026-09-02.md|decoder/mamba2.go:44` | goinfer | `inProj  []float32 // [projDim, hidden]` |
| `docs/audit-2026-09-02.md|decoder/mamba2.go:76` | goinfer | `if ssmQ8CPU { // confirmation seam: match the resident W8A8 — int8 weights AND int8 acti` |
| `docs/audit-2026-09-02.md|decoder/mamba2_chunked.go:91` | goinfer | `// checkpoint rates — A_log = log(U(1,16)) with dt in [0.001, 0.1] gives a per-step` |
| `docs/audit-2026-09-02.md|decoder/mlp.go:471` | goinfer | `func moeExpertMajor() bool { return os.Getenv("GOINFER_MOE_EXPERT_MAJOR") != "0" }` |
| `docs/audit-2026-09-02.md|decoder/mlp.go:491` | goinfer | `func moeMLPBatch(rows []float32, n int, lw *LayerWeights, arch *Architecture, be Backend` |
| `docs/audit-2026-09-02.md|decoder/mlp.go:493` | goinfer | `if moe == nil \|\| len(lw.Experts) == 0 \|\| moeSelOverride != nil \|\| moeSelTrace != nil \|\| ` |
| `docs/audit-2026-09-02.md|decoder/mlp.go:511` | goinfer | `logits := make([]float32, n*nE)` |
| `docs/audit-2026-09-02.md|decoder/mlp.go:583` | goinfer | `// moePrefillScratch enables the P18 ATTRIBUTION arm: reuse one scratch across the` |
| `docs/audit-2026-09-02.md|decoder/model.go:1101` | goinfer | `if sp.Logprobs {` |
| `docs/audit-2026-09-02.md|decoder/model.go:1142` | goinfer | `fastNext, err = greedyRF.ForwardArgmax(emb, gpuPos)` |
| `docs/audit-2026-09-02.md|decoder/model.go:1177` | goinfer | `func (m *Model) isStop(id int, sp SamplingParams) bool {` |
| `docs/audit-2026-09-02.md|decoder/model.go:220` | goinfer | `weightsBlob, _, gerr := giw.Read(data)` |
| `docs/audit-2026-09-02.md|decoder/model.go:702` | goinfer | `// mixers whose "residual after layer l" needs deciding rather than assuming, mla and ll` |
| `docs/audit-2026-09-02.md|decoder/model.go:707` | goinfer | `// Derived from the dispatch table's Captures bit rather than re-listed: the families wh` |
| `docs/audit-2026-09-02.md|decoder/model.go:737` | goinfer | `// EVERY own-forward family, derived: this seam needs runLayersFromEmbed's uniform block` |
| `docs/audit-2026-09-02.md|decoder/model.go:806` | goinfer | `func (m *Model) Generate(ctx context.Context, prompt []int, maxTokens int, sp SamplingPa` |
| `docs/audit-2026-09-02.md|decoder/model.go:849` | goinfer | `func (m *Model) residentPrefillSeed(ctx context.Context, prompt []int, from int) ([]floa` |
| `docs/audit-2026-09-02.md|decoder/model.go:986` | goinfer | `greedyRF, hasGreedy := m.resident.(ResidentGreedy)` |
| `docs/audit-2026-09-02.md|decoder/model.go:997` | goinfer | `optFwd := useGPU && !fastGreedy && m.optFwdEligible(sp) && os.Getenv("GOINFER_NO_OPTFWD"` |
| `docs/audit-2026-09-02.md|decoder/moe_expert_batch_test.go:220` | goinfer | `func loadMoEBitIdentModel(t *testing.T) (*Model, error) {` |
| `docs/audit-2026-09-02.md|decoder/moecap_kernel_pin_test.go:38` | goinfer | `// webgpu: array<f32, 256> score / array<f32, 32> gscore` |
| `docs/audit-2026-09-02.md|decoder/moepaging.go:61` | goinfer | `anchor: func newExpertPager(w *Weights, mapping []byte, budget int64) *expertPager {` |
| `docs/audit-2026-09-02.md|decoder/mtp.go:35` | goinfer | `anchor: type MTPHead struct {` |
| `docs/audit-2026-09-02.md|decoder/prefillattnpool_test.go:83` | goinfer | `os.Unsetenv("GOINFER_PREFILL_ATTN_WORKERS")` |
| `docs/audit-2026-09-02.md|decoder/registry.go:109` | goinfer | `func (a *Architecture) validateResolved() error {` |
| `docs/audit-2026-09-02.md|decoder/registry.go:1285` | goinfer | `QKNorm:          true, // per-head RMSNorm over head_dim — see the note above` |
| `docs/audit-2026-09-02.md|decoder/registry.go:1533` | goinfer | `// DeepSeek's YaRN attention_factor (the cos/sin mscale) is NOT the generic` |
| `docs/audit-2026-09-02.md|decoder/registry.go:1613` | goinfer | `// Plain qk_head_dim^-0.5. ⚠️ Phase 3: the real V2-Lite/V3 fold YaRN's` |
| `docs/audit-2026-09-02.md|decoder/registry.go:1728` | goinfer | `scaling, err := parseRopeScaling(cfg.RopeScaling)` |
| `docs/audit-2026-09-02.md|decoder/registry.go:1784` | goinfer | `var base float64` |
| `docs/audit-2026-09-02.md|decoder/registry.go:1814` | goinfer | `useRope := make([]bool, cfg.NumLayers)` |
| `docs/audit-2026-09-02.md|decoder/registry.go:1879` | goinfer | `llama4: &llama4Params{` |
| `docs/audit-2026-09-02.md|decoder/registry.go:270` | goinfer | `AttnScale:         math.Pow(cfg.QueryPreAttnScalar, -0.5),` |
| `docs/audit-2026-09-02.md|decoder/registry.go:407` | goinfer | `// backfillFlatRope fills the flat rope_theta / rope_scaling fields from transformers >=` |
| `docs/audit-2026-09-02.md|decoder/residency.go:129` | goinfer | `if m.w.arch.nemotron != nil && os.Getenv("GOINFER_SSM_RESIDENT") == "" {` |
| `docs/audit-2026-09-02.md|decoder/residency.go:225` | goinfer | `// Nemotron 3 Nano adds a FOURTH block kind (MoE FFN, arch.MoE != nil — plain Nemotron-H` |
| `docs/audit-2026-09-02.md|decoder/residency.go:60` | goinfer | `type ResidentGreedy interface {` |
| `docs/audit-2026-09-02.md|decoder/residency.go:726` | goinfer | `// staged/CPU path reports canBatchN, which excludes the families with their own sequent` |
| `docs/audit-2026-09-02.md|decoder/residency.go:730` | goinfer | `if !m.canBatchN(2) {` |
| `docs/audit-2026-09-02.md|decoder/residency.go:831` | goinfer | `out := make([]float32, len(lw.Experts)*2*inter)` |
| `docs/audit-2026-09-02.md|decoder/residency.go:854` | goinfer | `out := make([]float32, len(lw.Experts)*hidden)` |
| `docs/audit-2026-09-02.md|decoder/rmsnorm.go:32` | goinfer | `row[i] = (v * inv) * weight[i]` |
| `docs/audit-2026-09-02.md|decoder/rope.go:168` | goinfer | `if img >= len(grids) {` |
| `docs/audit-2026-09-02.md|decoder/rope.go:29` | goinfer | `half := len(invFreq) // == rotaryDim/2` |
| `docs/audit-2026-09-02.md|decoder/rope.go:31` | goinfer | `for d := range half {` |
| `docs/audit-2026-09-02.md|decoder/routercapture.go:27` | goinfer | `var routerCaptureBuf [][]int` |
| `docs/audit-2026-09-02.md|decoder/sampler.go:147` | goinfer | `func (s *Sampler) recordHistory(id int) {` |
| `docs/audit-2026-09-02.md|decoder/sampler.go:158` | goinfer | `//   - Temperature > 0: softmax at that temperature, optionally restricted to` |
| `docs/audit-2026-09-02.md|decoder/sampler.go:191` | goinfer | `// `top_p` / `min_p` at any value are safe alongside it: both cuts clamp at ≥1 retained ` |
| `docs/audit-2026-09-02.md|decoder/sampler.go:328` | goinfer | `func (s *Sampler) applyPenalties(logits []float32) {` |
| `docs/audit-2026-09-02.md|decoder/sampler.go:340` | goinfer | `func (s *Sampler) applyPenaltiesOver(logits []float32, window []int) {` |
| `docs/audit-2026-09-02.md|decoder/sampler.go:376` | goinfer | `func computeLogprobs(logits []float32, chosen int, temperature float64, topN int) (float` |
| `docs/audit-2026-09-02.md|decoder/sampler.go:441` | goinfer | `return lastWithMass(probs, 0, len(probs))` |
| `docs/audit-2026-09-02.md|decoder/sampler.go:540` | goinfer | `if topPActive {` |
| `docs/audit-2026-09-02.md|decoder/sampler.go:553` | goinfer | `case minP > 0:` |
| `docs/audit-2026-09-02.md|decoder/sampler.go:655` | goinfer | `func topKByLogit(logits []float32, k int) []int {` |
| `docs/audit-2026-09-02.md|decoder/sampler_chunked.go:110` | goinfer | `func forEachChunk(n int, fn func(c, lo, hi int)) {` |
| `docs/audit-2026-09-02.md|decoder/sampler_chunked.go:159` | goinfer | `return lastWithMass(e, lo, hi) // N-02: not hi-1, which is often a masked token` |
| `docs/audit-2026-09-02.md|decoder/sampler_selection_test.go:296` | goinfer | `// RE-BOUNDED after P2b (2026-08-09). This gate compares temp+top_p against temp-only, a` |
| `docs/audit-2026-09-02.md|decoder/scratch.go:194` | goinfer | `func attnRowTile(K, nKeys int) int {` |
| `docs/audit-2026-09-02.md|decoder/scratch.go:224` | goinfer | `if v := os.Getenv("GOINFER_PREFILL_ATTN_WORKERS"); v != "" {` |
| `docs/audit-2026-09-02.md|decoder/scratch.go:232` | goinfer | `// Per slot, in bytes, at the TILED size (G20): scores (tile*nKeys) + kh + vt` |
| `docs/audit-2026-09-02.md|decoder/scratch.go:236` | goinfer | `t := attnRowTile(K, nKeys)` |
| `docs/audit-2026-09-02.md|decoder/scratch.go:275` | goinfer | `for i := range s.headPool[:n] {` |
| `docs/audit-2026-09-02.md|decoder/scratch.go:328` | goinfer | `func newHeadWorkerPool(n, K, nKeys, hd int) []headWorkerScratch {` |
| `docs/audit-2026-09-02.md|decoder/scratch.go:345` | goinfer | `kh:     make([]float32, nKeys*hd),` |
| `docs/audit-2026-09-02.md|decoder/scratch.go:86` | goinfer | `ws := &linalg.Workspace{}` |
| `docs/audit-2026-09-02.md|decoder/serialize.go:105` | goinfer | `func canSerialize(a *Architecture) *SerializeError {` |
| `docs/audit-2026-09-02.md|decoder/serialize.go:119` | goinfer | `anchor: func canSerialize(a *Architecture) *SerializeError {` |
| `docs/audit-2026-09-02.md|decoder/serialize.go:1257` | goinfer | `return unsafe.Slice((*int8)(unsafe.Pointer(&b[0])), n)` |
| `docs/audit-2026-09-02.md|decoder/serialize.go:1352` | goinfer | `func (r *giwReader) layer(l *LayerWeights) {` |
| `docs/audit-2026-09-02.md|decoder/serialize.go:24` | goinfer | `// Discipline mirrors ken's index_serialize.go: magic + version + a config/quant` |
| `docs/audit-2026-09-02.md|decoder/serialize.go:271` | goinfer | `// LoadSerializedWeights reconstructs a *Weights from a SerializeWeights blob` |
| `docs/audit-2026-09-02.md|decoder/serialize.go:305` | goinfer | `// CRC: verify the whole payload (everything before the trailing crc word)` |
| `docs/audit-2026-09-02.md|decoder/serialize.go:315` | goinfer | `var cfg Config` |
| `docs/audit-2026-09-02.md|decoder/serialize.go:352` | goinfer | `n := int(r.u32())` |
| `docs/audit-2026-09-02.md|decoder/serialize.go:395` | goinfer | `func validateShapes(w *Weights, arch *Architecture) *SerializeError {` |
| `docs/audit-2026-09-02.md|decoder/serialize.go:985` | goinfer | `func (w *giwWriter) layer(l *LayerWeights) {` |
| `docs/audit-2026-09-02.md|decoder/serialize_census_test.go:135` | goinfer | `// "passed" while the new code went unexercised.` |
| `docs/audit-2026-09-02.md|decoder/session.go:176` | goinfer | `func (s *Session) GenerateNgramSpeculativeAdaptive(ctx context.Context, prompt []int, ma` |
| `docs/audit-2026-09-02.md|decoder/session.go:73` | goinfer | `func (s *Session) rewindForReuse(prompt []int) int {` |
| `docs/audit-2026-09-02.md|decoder/session.go:86` | goinfer | `func (s *Session) reconcile(seq []int) {` |
| `docs/audit-2026-09-02.md|decoder/session.go:98` | goinfer | `if rolledBack && s.cache.hasRecurrentState() {` |
| `docs/audit-2026-09-02.md|decoder/session_test.go:217` | goinfer | `bad := append([]byte(nil), blob...)` |
| `docs/audit-2026-09-02.md|decoder/spec_adaptive.go:124` | goinfer | `func (target *Model) GenerateNgramSpeculativeAdaptive(ctx context.Context, prompt []int,` |
| `docs/audit-2026-09-02.md|decoder/spec_adaptive.go:177` | goinfer | `case "cuda":` |
| `docs/audit-2026-09-02.md|decoder/spec_adaptive.go:82` | goinfer | `if a.Theta >= 1 {` |
| `docs/audit-2026-09-02.md|decoder/spec_eagle.go:260` | goinfer | `feats = make([][]float32, len(ids))` |
| `docs/audit-2026-09-02.md|decoder/spec_eagle.go:28` | goinfer | `if sp.Temperature != 0 \|\| sp.LogitProcessor != nil {` |
| `docs/audit-2026-09-02.md|decoder/spec_eagle.go:85` | goinfer | `feats[i] = fuseAt(i)` |
| `docs/audit-2026-09-02.md|decoder/spec_hitrate_probe_test.go:40` | goinfer | `giw := os.Getenv("GOINFER_SPEC_PROBE_GIW")` |
| `docs/audit-2026-09-02.md|decoder/spec_ngram.go:358` | goinfer | `lookupCtx := append(slices.Clone(hist), cur)` |
| `docs/audit-2026-09-02.md|decoder/spec_ngram.go:405` | goinfer | `p := dist(logitsN[i], ph)` |
| `docs/audit-2026-09-02.md|decoder/spec_ngram_test.go:83` | goinfer | `recurrent := map[string]*Model{` |
| `docs/audit-2026-09-02.md|decoder/spec_optfwd.go:30` | goinfer | `// GOINFER_OPTFWD_MAX_TEMP overrides it, for MEASUREMENT rather than tuning: moving this` |
| `docs/audit-2026-09-02.md|decoder/spec_sample.go:117` | goinfer | `for i, v := range slices.Backward(p) { // float-rounding guard: last token with mass` |
| `docs/audit-2026-09-02.md|decoder/spec_sample.go:37` | goinfer | `return softmaxStable(logits, s.p.Temperature) // drawFull draws from this directly` |
| `docs/audit-2026-09-02.md|decoder/spec_sample.go:82` | goinfer | `func (s *Sampler) specStep(p []float64, x int) (int, bool) {` |
| `docs/audit-2026-09-02.md|decoder/weightbytes.go:71` | goinfer | `func (m *Model) residentWeightBytes(slots int) int64 {` |
| `docs/audit-2026-09-02.md|decoder/weightmat.go:305` | goinfer | `var w4a8SplitHalfRepackEnabled = os.Getenv("GOINFER_W4A8_SPLITHALF") != ""` |
| `docs/audit-2026-09-02.md|decoder/weights.go:1081` | goinfer | `func loadFusedExperts(st *embed.SafetensorsFile, gateUpName, downName string, nExpert, i` |
| `docs/audit-2026-09-02.md|decoder/weights.go:114` | goinfer | `gemma4moe *gemma4MoEWeights` |
| `docs/audit-2026-09-02.md|decoder/weights.go:360` | goinfer | `// P13: release the SOURCE mapping now when nothing can alias it, instead of holding it ` |
| `docs/audit-2026-09-02.md|decoder/weights.go:694` | goinfer | `if arch.lfm2 != nil && arch.isConvLayer(i) {` |
| `docs/audit-2026-09-02.md|decoder/weights.go:90` | goinfer | `delta *deltaNetWeights` |
| `docs/audit-2026-09-02.md|decoder/weights.go:96` | goinfer | `shortConv *shortConvWeights` |
| `docs/audit-2026-09-02.md|demo/agent/agent/agent.go:335` | goinfer | `ids, err := s.tk.EncodeSegments(segs, s.tmpl == nil)` |
| `docs/audit-2026-09-02.md|demo/agent/agent/agent.go:445` | goinfer | `flush := func(final bool) {` |
| `docs/audit-2026-09-02.md|demo/agent/agent/agent.go:537` | goinfer | `return constrain.NewMasker(g, constrain.TokenBytes(s.vocab, s.tk.TokenText), eos).StopWh` |
| `docs/audit-2026-09-02.md|gpu/attention.go:961` | goinfer | `func (c *Context) ensureAttnWide() error {` |
| `docs/audit-2026-09-02.md|gpu/decode_staged_prize_test.go:23` | goinfer | `// G-10: a Benchmark, not a Test. It reports numbers and asserts nothing, so as a Test*` |
| `docs/audit-2026-09-02.md|gpu/decoderunner.go:1153` | goinfer | `fq, fs := rmsQuant(r.xd, m.finalNorm, hidden)` |
| `docs/audit-2026-09-02.md|gpu/decoderunner.go:1207` | goinfer | `func (r *DecodeRunner) ReadMambaCap(projN, convN, dInner int) (proj, conv, y, gated []fl` |
| `docs/audit-2026-09-02.md|gpu/decoderunner.go:1274` | goinfer | `enc.CopyBufferToBuffer(r.lastLogits, 0, r.stag, 0, uint64(r.vocab*4))` |
| `docs/audit-2026-09-02.md|gpu/decoderunner.go:294` | goinfer | `ssmStopLayer := -1 // GOINFER_SSM_STOP_LAYER debug (resident SSM bring-up): truncate the` |
| `docs/audit-2026-09-02.md|gpu/decoderunner.go:303` | goinfer | `w8a16 := os.Getenv("GOINFER_SSM_W8A16") != ""` |
| `docs/audit-2026-09-02.md|gpu/decoderunner.go:488` | goinfer | `gemv := func(aq, as *wgpu.Buffer, w decodeWeight) *wgpu.Buffer {` |
| `docs/audit-2026-09-02.md|gpu/decoderunner.go:745` | goinfer | `r.posUnis = append(r.posUnis, posUni{buf: b, gen: func(pos int) []uint32 {` |
| `docs/audit-2026-09-02.md|gpu/decodetoken_batched.go:11` | goinfer | `// DecodeTokenFusedBatched is the Stage-B (docs/spec/07) batched verify forward: it` |
| `docs/audit-2026-09-02.md|gpu/deltanet.go:13` | goinfer | `// This is the mixer that makes every DeltaNet hybrid CPU-only on every backend today: 4` |
| `docs/audit-2026-09-02.md|gpu/doc.go:19` | goinfer | `// Status: FOUNDATION cut. A single `dst = a·bᵀ` GEMM offloaded to the GPU` |
| `docs/audit-2026-09-02.md|gpu/doc.go:6` | goinfer | `// wgpu-native Rust library) is allowed to appear. Every file except this doc` |
| `docs/audit-2026-09-02.md|gpu/gemv_w4a8.go:121` | goinfer | `// load-bearing one — every W4A8 group-scale upload and NewKVCacheF16 go through it. It ` |
| `docs/audit-2026-09-02.md|gpu/gemv_w4a8.go:25` | goinfer | `@group(0) @binding(0) var<storage, read>       aq:      array<vec4<u32>>;  // [kp/16] in` |
| `docs/audit-2026-09-02.md|gpu/gemv_w8a16.go:21` | goinfer | `@group(0) @binding(0) var<storage, read>       act:     array<f32>;        // [kp] f32 a` |
| `docs/audit-2026-09-02.md|gpu/gpu.go:51` | goinfer | `// not. Not safe for concurrent use by multiple goroutines; wrap in your` |
| `docs/audit-2026-09-02.md|gpu/kv_longctx_test.go:24` | goinfer | `UNKEYABLE` |
| `docs/audit-2026-09-02.md|gpu/mamba_f16.go:34` | goinfer | `func f32ToF16(f float32) uint16 {` |
| `docs/audit-2026-09-02.md|gpu/moe.go:25` | goinfer | `const MAXE: u32 = 512u;` |
| `docs/audit-2026-09-02.md|gpu/moe_w4a8.go:146` | goinfer | `func (c *Context) UploadStackedExpertsInt4Packed(q4 [][]byte, scales [][]float32, nE, N,` |
| `docs/audit-2026-09-02.md|gpu/moe_w4a8_expert_test.go:37` | goinfer | `stack, err := ctx.UploadStackedExpertsInt4(nib, sc, nE, N, K)` |
| `docs/audit-2026-09-02.md|gpu/qwen35_resident_parity_test.go:30` | goinfer | `if os.Getenv("GOINFER_DNET_PARITY") == "" {` |
| `docs/audit-2026-09-02.md|gpu/residency.go:1069` | goinfer | `func (rd *residentDecoder) Reset() {` |
| `docs/audit-2026-09-02.md|gpu/residency.go:112` | goinfer | `_, _, _, _, _, _, _, _, _, granOK := m.GraniteResidentParams()` |
| `docs/audit-2026-09-02.md|gpu/residency.go:134` | goinfer | `// M-31: the cap is READ from decoder's declaration, not restated here. This site had it` |
| `docs/audit-2026-09-02.md|gpu/residency.go:155` | goinfer | `kvF16 := m.KVCacheF16()` |
| `docs/audit-2026-09-02.md|gpu/residency.go:185` | goinfer | `ctxCap := 16384` |
| `docs/audit-2026-09-02.md|gpu/residency.go:44` | goinfer | `if K%w4a8GroupSize == 0 && !int4SlowPath {` |
| `docs/audit-2026-09-02.md|gpu/residency.go:690` | goinfer | `if os.Getenv("GOINFER_SSM_F16MAMBA") != "" { // f16 (no quality gain; kept for experimen` |
| `docs/audit-2026-09-02.md|gpu/residency.go:770` | goinfer | `// Slice into W_UKᵀ [nH, kvLoRA, qkNope] (transposed for the absorb GEMV) and` |
| `docs/audit-2026-09-02.md|gpu/residency.go:954` | goinfer | `func (rd *residentDecoder) Forward(embedding []float32, pos int) ([]float32, error) {` |
| `docs/audit-2026-09-02.md|gpu/residency.go:975` | goinfer | `func (rd *residentDecoder) ForwardN(embeddings [][]float32, startPos int) ([][]float32, ` |
| `docs/audit-2026-09-02.md|gpu/residency_c01_reset_test.go:24` | goinfer | `requireHeavyModel(t)` |
| `docs/audit-2026-09-02.md|gpu/resident_pack_bench_test.go:21` | goinfer | `func BenchmarkResidentPackCost(b *testing.B) {` |
| `docs/audit-2026-09-02.md|gpu/testhooks_gen.go:1` | goinfer | `//go:build goinfer_testhooks` |
| `docs/audit-2026-09-02.md|gpu/vision.go:52` | goinfer | `let mean = smean[0] / f32(p.h);` |
| `docs/audit-2026-09-02.md|gpu/vision_encoder.go:202` | goinfer | `for head := range nH {` |
| `docs/audit-2026-09-02.md|internal/chatapp/main.go:200` | goinfer | `// bundle, or raw tokenizer.json for a safetensors-sourced one).` |
| `docs/audit-2026-09-02.md|internal/chatapp/main.go:203` | goinfer | `var tk *tokenizer.Tokenizer` |
| `docs/audit-2026-09-02.md|internal/chatapp/main.go:349` | goinfer | `// print only newly-completed bytes (a byte-fallback token may be a partial` |
| `docs/audit-2026-09-02.md|internal/chatapp/main.go:502` | goinfer | `g := constrain.JSON()` |
| `docs/audit-2026-09-02.md|internal/chatapp/prequant.go:45` | goinfer | `// N-25: a .giw's tok half is GGUF metadata for a GGUF-sourced bundle and RAW tokenizer.` |
| `docs/audit-2026-09-02.md|internal/gemmaapp/main.go:113` | goinfer | `// 4) Generate + stream. Decode the whole running sequence (prompt +` |
| `docs/audit-2026-09-02.md|internal/gemmaapp/main.go:131` | goinfer | `flush := func(final bool) error {` |
| `docs/audit-2026-09-02.md|internal/gemmaapp/main.go:175` | goinfer | `m := constrain.NewMasker(constrain.JSON(), constrain.TokenBytes(vocab, tk.TokenText), eo` |
| `docs/audit-2026-09-02.md|internal/giw/bundle.go:44` | goinfer | `func WriteStream(f *os.File, tok []byte, writeWeights func(io.Writer) (int64, error)) er` |
| `docs/audit-2026-09-02.md|internal/prequant/prequant.go:128` | goinfer | `func transcodeDir(ctx context.Context, dir, out, quant string, embedInt4, row4 bool) err` |
| `docs/audit-2026-09-02.md|internal/prequant/prequant.go:150` | goinfer | `f, err := os.Create(out)` |
| `docs/audit-2026-09-02.md|internal/prequant/prequant.go:181` | goinfer | `func EnsureCachedGIW(ctx context.Context, ggufPath, quant string) (string, error) {` |
| `docs/audit-2026-09-02.md|internal/prequant/prequant.go:216` | goinfer | `func cacheFresh(cache, src string) bool {` |
| `docs/audit-2026-09-02.md|internal/prequant/prequant.go:244` | goinfer | `// selfCheck verifies a freshly written bundle loads through the real mmap path` |
| `docs/audit-2026-09-02.md|internal/serveapp/admin.go:64` | goinfer | `anchor: func (s *server) handleAdminLoad(w http.ResponseWriter, r *http.Request) {` |
| `docs/audit-2026-09-02.md|internal/serveapp/admin.go:85` | goinfer | `lm, err := loadDecoder(r.Context(), modelSpec{name: name, path: req.Path}, c)` |
| `docs/audit-2026-09-02.md|internal/serveapp/admin_test.go:80` | goinfer | `// old contract was "busy → 409"; unload now DRAINS instead of refusing — the in-flight-` |
| `docs/audit-2026-09-02.md|internal/serveapp/anthropic.go:127` | goinfer | `writeAnthropicErr(w, http.StatusRequestEntityTooLarge, "invalid_request_error", msg)` |
| `docs/audit-2026-09-02.md|internal/serveapp/anthropic.go:357` | goinfer | `func anthropicForcedTool(mode, name string, tools []chat.Tool) *chat.Tool {` |
| `docs/audit-2026-09-02.md|internal/serveapp/anthropic.go:420` | goinfer | `func (s *server) handleMessages(w http.ResponseWriter, r *http.Request) {` |
| `docs/audit-2026-09-02.md|internal/serveapp/anthropic.go:443` | goinfer | `// O(n) tokenize. The OpenAI routes got this guard; /v1/messages did not, so a body unde` |
| `docs/audit-2026-09-02.md|internal/serveapp/anthropic.go:448` | goinfer | `if err := lm.promptTooLargeForContext(anthropicInputBytes(&req)); err != nil {` |
| `docs/audit-2026-09-02.md|internal/serveapp/anthropic.go:571` | goinfer | `func (s *server) serveCountTokensWith(w http.ResponseWriter, req anthropicReq, lm *loade` |
| `docs/audit-2026-09-02.md|internal/serveapp/anthropic_stream.go:94` | goinfer | `// THE THIRD BUFFER-THEN-STREAM SITE. G19 gave the OpenAI tool path and /v1/responses a` |
| `docs/audit-2026-09-02.md|internal/serveapp/cpufastattn_test.go:12` | goinfer | `//  1. It is OFF unless asked for. A speed flag that turns itself on is how a user` |
| `docs/audit-2026-09-02.md|internal/serveapp/decoder_embedder.go:211` | goinfer | `func (e *decoderEmbedder) encodeLocked(text string, isQuery bool) ([]float32, []int, err` |
| `docs/audit-2026-09-02.md|internal/serveapp/decoder_embedder.go:226` | goinfer | `func (e *decoderEmbedder) tokenize(text string, isQuery bool) ([]int, error) {` |
| `docs/audit-2026-09-02.md|internal/serveapp/decoder_embedder.go:84` | goinfer | `// "nothing shorter" is not "unbounded": HiddenLast preallocates KV for len(ids) positio` |
| `docs/audit-2026-09-02.md|internal/serveapp/embeddings.go:122` | goinfer | `for _, c := range counts {` |
| `docs/audit-2026-09-02.md|internal/serveapp/embeddings.go:31` | goinfer | `// positions. maxEmbedInputs matches OpenAI's per-request batch cap; maxEmbedInputBytes ` |
| `docs/audit-2026-09-02.md|internal/serveapp/embeddings.go:34` | goinfer | `maxEmbedInputs     = 2048` |
| `docs/audit-2026-09-02.md|internal/serveapp/heartbeat_test.go:161` | goinfer | `// G21 end-to-end. What this asserts is bounded by the model available: the 1.5B` |
| `docs/audit-2026-09-02.md|internal/serveapp/helpers.go:131` | goinfer | `// (audit-2026-09-02 N-16).` |
| `docs/audit-2026-09-02.md|internal/serveapp/helpers.go:172` | goinfer | `// --- SSE ---` |
| `docs/audit-2026-09-02.md|internal/serveapp/helpers.go:208` | goinfer | `// writes to a streaming ResponseWriter.` |
| `docs/audit-2026-09-02.md|internal/serveapp/helpers.go:217` | goinfer | `// support — so the error is deliberately dropped rather than made sticky.` |
| `docs/audit-2026-09-02.md|internal/serveapp/helpers.go:504` | goinfer | `func reqID() string {` |
| `docs/audit-2026-09-02.md|internal/serveapp/liveness.go:10` | goinfer | `// Model liveness + the drain-based unload path. See docs/completed/task-admin-unload-dr` |
| `docs/audit-2026-09-02.md|internal/serveapp/liveness.go:159` | goinfer | `if s.cfg.sessionDir != "" && s.cfg.kvSessions > 0 {` |
| `docs/audit-2026-09-02.md|internal/serveapp/main.go:158` | goinfer | `// default must never conflict with an already-baked bundle.` |
| `docs/audit-2026-09-02.md|internal/serveapp/main.go:376` | goinfer | `flag.StringVar(&cfg.kvPrec, "kv", "f32", "GPU residency KV cache precision: f32 (bit-exa` |
| `docs/audit-2026-09-02.md|internal/serveapp/main.go:385` | goinfer | `flag.IntVar(&cfg.kvSessions, "kv-sessions", 4, "number of conversations to keep prefille` |
| `docs/audit-2026-09-02.md|internal/serveapp/main.go:410` | goinfer | `// and a divergence a user opts into should be spelled out in --help rather than` |
| `docs/audit-2026-09-02.md|internal/serveapp/main.go:421` | goinfer | `// contradiction the user did not realise they had expressed.` |
| `docs/audit-2026-09-02.md|internal/serveapp/main.go:422` | goinfer | `if cfg.cpuExactPrefill \|\| !cfg.cpuFastAttention {` |
| `docs/audit-2026-09-02.md|internal/serveapp/main.go:483` | goinfer | `if cfg.sessionDir != "" && cfg.kvSessions > 0 {` |
| `docs/audit-2026-09-02.md|internal/serveapp/main.go:516` | goinfer | `// extension, on a payload with no OpenAI-schema contract to break. See handleHealth.` |
| `docs/audit-2026-09-02.md|internal/serveapp/main.go:554` | goinfer | `// ReadHeaderTimeout + ReadTimeout + IdleTimeout bound slow-header (slowloris), slow-bod` |
| `docs/audit-2026-09-02.md|internal/serveapp/main.go:602` | goinfer | `close(stopDemote) // stop demoting before we checkpoint` |
| `docs/audit-2026-09-02.md|internal/serveapp/main.go:770` | goinfer | `// dir is -vision if set, else the sole --model's own dir when it carries a vision` |
| `docs/audit-2026-09-02.md|internal/serveapp/main.go:888` | goinfer | `return nil, fmt.Errorf("--model %q: %w", spec.path, err)` |
| `docs/audit-2026-09-02.md|internal/serveapp/openai.go:1069` | goinfer | `ids = append(ids, id)` |
| `docs/audit-2026-09-02.md|internal/serveapp/openai.go:1073` | goinfer | `piece, _ := lm.tk.DecodePiece(id)` |
| `docs/audit-2026-09-02.md|internal/serveapp/openai.go:1154` | goinfer | `func (lm *loadedModel) logprobs(lps []decoder.SampleInfo) map[string]any {` |
| `docs/audit-2026-09-02.md|internal/serveapp/openai.go:424` | goinfer | `type completionReq struct {` |
| `docs/audit-2026-09-02.md|internal/serveapp/openai.go:508` | goinfer | `if err := lm.promptTooLargeForContext(chatInputBytes(req.Messages)); err != nil {` |
| `docs/audit-2026-09-02.md|internal/serveapp/openai.go:526` | goinfer | `if !lm.enter(w) {` |
| `docs/audit-2026-09-02.md|internal/serveapp/openai.go:549` | goinfer | `sendUsage(ss, req.StreamOptions, id, created, lm.name,` |
| `docs/audit-2026-09-02.md|internal/serveapp/openai.go:668` | goinfer | `StopIDs:     lm.stopIDs,` |
| `docs/audit-2026-09-02.md|internal/serveapp/openai.go:670` | goinfer | `TopLogprobs: deref(sm.TopLogprobs, 0),` |
| `docs/audit-2026-09-02.md|internal/serveapp/openai.go:760` | goinfer | `gr.maxTokens = clampMaxTokens(gr.maxTokens, len(promptIDs), ctx)` |
| `docs/audit-2026-09-02.md|internal/serveapp/openai.go:762` | goinfer | `g, err := grammarFor(sm.ResponseFormat)` |
| `docs/audit-2026-09-02.md|internal/serveapp/openai.go:768` | goinfer | `m := constrain.NewMasker(g, lm.cachedTokenBytes(), eos).StopWhenComplete()` |
| `docs/audit-2026-09-02.md|internal/serveapp/openai.go:899` | goinfer | `func (lm *loadedModel) promptFor(system string, turns []chat.Turn) ([]int, error) {` |
| `docs/audit-2026-09-02.md|internal/serveapp/openai.go:913` | goinfer | `func genErr(err error) error {` |
| `docs/audit-2026-09-02.md|internal/serveapp/openai.go:932` | goinfer | `// GPU-resident models take the STATELESS path. decoder.Generate only engages the reside` |
| `docs/audit-2026-09-02.md|internal/serveapp/responses.go:139` | goinfer | `ids, err := lm.chatPrompt(messages)` |
| `docs/audit-2026-09-02.md|internal/serveapp/responses.go:236` | goinfer | `var stopBeat func()` |
| `docs/audit-2026-09-02.md|internal/serveapp/responses.go:282` | goinfer | `// Tool-call continuations round-trip via the next request's input; store the` |
| `docs/audit-2026-09-02.md|internal/serveapp/responses.go:299` | goinfer | `func responseInputToMessages(raw json.RawMessage) ([]chatMessage, error) {` |
| `docs/audit-2026-09-02.md|internal/serveapp/responses.go:94` | goinfer | `if req.PreviousResponseID != "" {` |
| `docs/audit-2026-09-02.md|internal/serveapp/responses_test.go:133` | goinfer | `// 4. Tool round-trip: forced function call → a function_call output item.` |
| `docs/audit-2026-09-02.md|internal/serveapp/sessions.go:136` | goinfer | `func (l *sessionLRU) acquire(prompt []int) *decoder.Session {` |
| `docs/audit-2026-09-02.md|internal/serveapp/sessions.go:165` | goinfer | `func bestExtend(sessions [][]int, prompt []int) int {` |
| `docs/audit-2026-09-02.md|internal/serveapp/sessions.go:20` | goinfer | `UNKEYABLE` |
| `docs/audit-2026-09-02.md|internal/serveapp/sessions.go:236` | goinfer | `// the ordinary way.` |
| `docs/audit-2026-09-02.md|internal/serveapp/sessions.go:385` | goinfer | `// Snapshot refuses RECURRENT state (Mamba-2 / DeltaNet / LFM2 conv / MLA latent),` |
| `docs/audit-2026-09-02.md|internal/serveapp/tools.go:112` | goinfer | `if ss != nil {` |
| `docs/audit-2026-09-02.md|internal/serveapp/tools.go:116` | goinfer | `stopBeat = sseHeartbeat(ss)` |
| `docs/audit-2026-09-02.md|internal/serveapp/tools.go:125` | goinfer | `sseSend(ss, chatChunk(id, created, lm.name, delta{Content: out}, nil))` |
| `docs/audit-2026-09-02.md|internal/serveapp/tools.go:129` | goinfer | `stopBeat() // joins the ticker goroutine before anything else writes to w` |
| `docs/audit-2026-09-02.md|internal/serveapp/tools.go:154` | goinfer | `usagev := usage{len(gr.promptIDs), nComp, len(gr.promptIDs) + nComp}` |
| `docs/audit-2026-09-02.md|internal/serveapp/tools.go:23` | goinfer | `func (s *server) serveChatToolsWith(w http.ResponseWriter, r *http.Request, req chatReq,` |
| `docs/audit-2026-09-02.md|internal/serveapp/tools.go:240` | goinfer | `g, gerr := constrain.ToolCallGrammar(prefix, suffix, argsKey, forced.Name, array, forced` |
| `docs/audit-2026-09-02.md|internal/serveapp/tools.go:248` | goinfer | `// N-23: lm.cachedTokenBytes(), not a fresh constrain.TokenBytes. The table is ~152k ent` |
| `docs/audit-2026-09-02.md|internal/serveapp/tools.go:286` | goinfer | `// toolChoiceMode returns "auto" (default), "none", or "required"/"function" from` |
| `docs/audit-2026-09-02.md|internal/serveapp/tools.go:301` | goinfer | `func forcedTool(toolChoice json.RawMessage, tools []chat.Tool) *chat.Tool {` |
| `docs/audit-2026-09-02.md|internal/serveapp/tools.go:83` | goinfer | `if req.Stream {` |
| `docs/audit-2026-09-02.md|internal/serveapp/vision_serve.go:127` | goinfer | `pv, err := vision.Preprocess(img.data, lm.vcfg)` |
| `docs/audit-2026-09-02.md|internal/serveapp/vision_serve.go:210` | goinfer | `system, turns := messagesToTurns(req.Messages)` |
| `docs/audit-2026-09-02.md|internal/serveapp/vision_serve.go:228` | goinfer | `if req.Stream {` |
| `docs/audit-2026-09-02.md|internal/serveapp/vision_serve.go:72` | goinfer | `anchor: func encodeVisionSegments(lm *loadedModel, system string, turns []chat.Turn, blo` |
| `docs/audit-2026-09-02.md|metal/attn_shape_test.go:152` | goinfer | `enc.Dispatch(pAttn, nH*128, 128, qB, kc, vc, out, uNH, uNKV, uHd, uNKeys, uScale, uWin, ` |
| `docs/audit-2026-09-02.md|metal/backend.go:143` | goinfer | `if os.Getenv("GOINFER_NO_RESIDENT_MEM_GUARD") != "" {` |
| `docs/audit-2026-09-02.md|metal/backend.go:146` | goinfer | `need := residentNeedBytes(m)` |
| `docs/audit-2026-09-02.md|metal/backend.go:81` | goinfer | `if !residentFitsMemory(m) {` |
| `docs/audit-2026-09-02.md|metal/batched_verify_test.go:139` | goinfer | `e.Dispatch(r.pRope, r.nH*g.half, 64, qkv.At(m*qkvRows*4), L.invf, g.uHd, uPos, g.uQtotal` |
| `docs/audit-2026-09-02.md|metal/close_leak_test.go:182` | goinfer | `requireHeavyModel(t)` |
| `docs/audit-2026-09-02.md|metal/close_leak_test.go:45` | goinfer | `requireHeavyModel(t)` |
| `docs/audit-2026-09-02.md|metal/cmdbuf_status_test.go:94` | goinfer | `// This exercises the ceil-sized reduction on device for a %8 vocab (tmVocab=64) — a com` |
| `docs/audit-2026-09-02.md|metal/deltanet.go:139` | goinfer | `e.Dispatch(r.pDnNorm, dp.nk*128, 128, r.dnConvOut, r.dnQn, r.dnKn, r.uDnNk, r.uDnHk, r.u` |
| `docs/audit-2026-09-02.md|metal/deltanet.go:149` | goinfer | `e.DispatchTG(r.pSAResid, r.H*32, 256, dp.valueDim*2, D.outW, D.outS, r.dnGq, r.dnGSc, r.` |
| `docs/audit-2026-09-02.md|metal/gemma4_moe.go:333` | goinfer | `panic(fmt.Sprintf("metal gemma4 MoE pread gate\|up expert %d: %v", ei, err))` |
| `docs/audit-2026-09-02.md|metal/gemma4_moe.go:387` | goinfer | `e.Dispatch(r.pRms, 256, 256, r.x, ml.preFFN, r.mq, r.mSc, r.uH, r.uEps, r.uAddOne)` |
| `docs/audit-2026-09-02.md|metal/gemma4_moe.go:408` | goinfer | `for j := 0; j < g.topK; j++ {` |
| `docs/audit-2026-09-02.md|metal/gptoss_real20b_test.go:42` | goinfer | `// modelPath, NOT a direct environment read: the asset registry owns GOINFER_GPTOSS_GGUF` |
| `docs/audit-2026-09-02.md|metal/heavytest_test.go:20` | goinfer | `func requireHeavyModel(t *testing.T) {` |
| `docs/audit-2026-09-02.md|metal/kernels.go:511` | goinfer | `kernel void rope(device float* x[[buffer(0)]], device const float* invf[[buffer(1)]],` |
| `docs/audit-2026-09-02.md|metal/kernels.go:515` | goinfer | `float th=float(pos)*invf[dd]; float c=cos(th)*scale,s=sin(th)*scale;` |
| `docs/audit-2026-09-02.md|metal/kernels.go:606` | goinfer | `kernel void attention(device const float* q[[buffer(0)]], device const half* kc[[buffer(` |
| `docs/audit-2026-09-02.md|metal/layer_test.go:148` | goinfer | `enc.Dispatch(pAttn, nH*128, 128, qB, kc, vc, ctx, uNH, uNKV, uHd, uNKeys, uScale, uWindo` |
| `docs/audit-2026-09-02.md|metal/model.go:1482` | goinfer | `e.Dispatch(r.pRope, r.nH*g.half, 64, r.qkv, L.invf, g.uHd, r.uPos, g.uQtotal, g.uHalf, L` |
| `docs/audit-2026-09-02.md|metal/model.go:330` | goinfer | `// N-32: dnValueDim is DeltaNet's out-projection staging width. deltanet.go dispatches p` |
| `docs/audit-2026-09-02.md|metal/model.go:346` | goinfer | `if words, scales, ok := int4DirectWords(w); ok {` |
| `docs/audit-2026-09-02.md|metal/model.go:444` | goinfer | `if preciseMathCompile \|\| os.Getenv("GOINFER_PRECISE_MATH") != "" {` |
| `docs/audit-2026-09-02.md|metal/model.go:489` | goinfer | `r.kvF32 = false` |
| `docs/audit-2026-09-02.md|metal/model.go:49` | goinfer | `anchor: const (` |
| `docs/audit-2026-09-02.md|metal/model.go:631` | goinfer | `L.attnSinks, L.uHasSink = NewBufferFloats(d, []float32{0}), NewBufferU32(d, 0)` |
| `docs/audit-2026-09-02.md|metal/model.go:776` | goinfer | `// Vocab is NOT checked here (2026-08-18): it never routes through an SA-family kernel —` |
| `docs/audit-2026-09-02.md|metal/model.go:899` | goinfer | `paged := (r.g4moe != nil && r.g4moe.paged) \|\| (r.moe != nil && r.moe.paged)` |
| `docs/audit-2026-09-02.md|metal/moe.go:106` | goinfer | `for (uint j=0u;j<k;j++) {` |
| `docs/audit-2026-09-02.md|metal/moe.go:209` | goinfer | `if (lane==0) out[row] += wgt[slot]*(acc*asc[0] + bias[bidx[slot]*rowsPerExpert + row]);` |
| `docs/audit-2026-09-02.md|metal/moe.go:242` | goinfer | `uint biasOff = hasBias != 0u ? idx[slot]*2u*I : 0u;` |
| `docs/audit-2026-09-02.md|metal/moe.go:425` | goinfer | `if s := os.Getenv("GOINFER_METAL_MOE_SLOTS"); s != "" {` |
| `docs/audit-2026-09-02.md|metal/moe.go:538` | goinfer | `panic(fmt.Sprintf("metal MoE pread gate expert %d: %v", ei, err))` |
| `docs/audit-2026-09-02.md|metal/moe.go:610` | goinfer | `e.Dispatch(r.pRms, 256, 256, r.x, L.postNorm, r.mq, r.mSc, r.uH, r.uEps, r.uAddOne)` |
| `docs/audit-2026-09-02.md|metal/moe.go:634` | goinfer | `for j := 0; j < mo.k; j++ {` |
| `docs/audit-2026-09-02.md|metal/moe.go:676` | goinfer | `e.Dispatch(mo.pActGptOss, 256, 256, r.gu, r.gu.At(mo.inter*4), r.dq, r.dSc, mo.uInter,` |
| `docs/audit-2026-09-02.md|metal/moe.go:715` | goinfer | `func (r *resident) forwardLogitsMoEPaged(pos int) (logits []float32) {` |
| `docs/audit-2026-09-02.md|metal/prefill_nan_test.go:20` | goinfer | `requireHeavyModel(t)` |
| `docs/audit-2026-09-02.md|metal/prefill_parity_test.go:20` | goinfer | `requireHeavyModel(t)` |
| `docs/audit-2026-09-02.md|metal/qwen35_35b_paged_test.go:95` | goinfer | `r, err := buildResident(m)` |
| `docs/audit-2026-09-02.md|metal/snapshot_golden_test.go:40` | goinfer | `// N-29: the re-bake this note demanded HAS happened — TestMetalSnapshotGolden reports` |
| `docs/audit-2026-09-02.md|metal/snapshot_golden_test.go:57` | goinfer | `// no heavy-model dependency). Coverage: mixtral-tiny is full-causal (attention softmax ` |
| `docs/audit-2026-09-02.md|multimodal/qwen_preprocess.go:233` | goinfer | `func qwenExtractRGB(img image.Image, h, w int) []float32 {` |
| `docs/audit-2026-09-02.md|multimodal/qwen_preprocess.go:25` | goinfer | `// patch-row, patch-col). The resize is BICUBIC (qwenBicubicU8, called at the resize sit` |
| `docs/audit-2026-09-02.md|multimodal/qwen_preprocess_test.go:14` | goinfer | `// TestQwenPreprocess_exact gates the Qwen2.5-VL preprocessing (normalize +` |
| `docs/audit-2026-09-02.md|scripts/apidiff_check.sh:11` | goinfer | `#   scripts/apidiff_check.sh                 # baseline v0.13.0 (the last released tag)` |
| `docs/audit-2026-09-02.md|scripts/asset_registry.py:45` | goinfer | `def models_root():` |
| `docs/audit-2026-09-02.md|scripts/ci_checks.py:51` | goinfer | `HYGIENE = re.compile(r"gofmt\|staticcheck\|^vet\|^build\|cleanliness\|lint", re.I)` |
| `docs/audit-2026-09-02.md|scripts/gate_ledger.py:151` | goinfer | `if e is None:` |
| `docs/audit-2026-09-02.md|scripts/gptoss_tiny_golden.py:32` | goinfer | `UNKEYABLE` |
| `docs/audit-2026-09-02.md|scripts/queue_citation_lint.py:574` | goinfer | `update = "--update" in sys.argv` |
| `docs/audit-2026-09-02.md|tokenizer/bytelevel.go:196` | goinfer | `// decodeByteLevel inverts the byte-level map: render each piece (special tokens` |
| `docs/audit-2026-09-02.md|tokenizer/bytelevel.go:244` | goinfer | `// splitGPT2 reproduces the Qwen/Llama-3 pretokenizer split — the GPT-2 regex` |
| `docs/audit-2026-09-02.md|tokenizer/bytelevel.go:269` | goinfer | `for i := 0; i < n; {` |
| `docs/audit-2026-09-02.md|tokenizer/bytelevel.go:27` | goinfer | `func (t *Tokenizer) initByteLevel(tj *tokenizerJSON, dir string) error {` |
| `docs/audit-2026-09-02.md|tokenizer/bytelevel.go:388` | goinfer | `func normalizerForm(raw json.RawMessage) (norm.Form, bool) {` |
| `docs/audit-2026-09-02.md|tokenizer/bytelevel_test.go:18` | goinfer | `//	.venv/bin/python scripts/pin_qwen3_tokenizer.py` |
| `docs/audit-2026-09-02.md|tokenizer/doc.go:26` | goinfer | `// Golden parity against HF `tokenizers` is the gate for every family (M2 /` |
| `docs/audit-2026-09-02.md|tokenizer/gguf.go:119` | goinfer | `for i, tok := range tokens {` |
| `docs/audit-2026-09-02.md|tokenizer/gguf.go:276` | goinfer | `// byteLevelKnobs maps a tokenizer.ggml.pre identifier to the byte-level pipeline` |
| `docs/audit-2026-09-02.md|tokenizer/gguf.go:280` | goinfer | `// ignore_merges; Qwen takes one digit, NFC-normalizes, and honors merges; Mellum2` |
| `docs/audit-2026-09-02.md|tokenizer/gguf_test.go:19` | goinfer | `//	.venv/bin/python scripts/pin_tinyllama_tokenizer.py` |
| `docs/audit-2026-09-02.md|tokenizer/sentencepiece.go:441` | goinfer | `// Encode turns text into token ids. If addBOS, prepend the BOS token (the` |
| `docs/audit-2026-09-02.md|tokenizer/sentencepiece.go:466` | goinfer | `if t.mode == modeByteLevel {` |
| `docs/audit-2026-09-02.md|tokenizer/sentencepiece.go:752` | goinfer | `func (t *Tokenizer) Decode(ids []int) (string, error) {` |
| `docs/audit-2026-09-02.md|tokenizer/sentencepiece_test.go:51` | goinfer | `//	.venv/bin/python scripts/pin_gemma_tokenizer.py` |
| `docs/audit-2026-09-02.md|tokenizer/tokentext_test.go:35` | goinfer | `if _, err := os.Stat(c.dir); errors.Is(err, fs.ErrNotExist) {` |
| `docs/book/04-the-loop-and-the-kv-cache.md|decoder/deltanet.go:145` | goinfer | `// last K-1 conv inputs (so the causal conv has its left context at decode) and` |
| `docs/book/09-guessing-ahead.md|decoder/deltanet.go:145` | goinfer | `// last K-1 conv inputs (so the causal conv has its left context at decode) and` |
| `docs/book/09-guessing-ahead.md|decoder/speculative.go:89` | goinfer | `// rolls back the rejected tail. A recurrent (Mamba-2 / Gated DeltaNet) or staged` |
| `docs/cuda-megakernel-spec.md|gpu/attention.go:18` | goinfer | `// uses f64 accumulation; the GPU f32 — cosine ~1.0, not bit-exact).` |
| `docs/cuda-megakernel-spec.md|gpu/decoderunner.go:828` | goinfer | `// moeExpert records one indexed sparse-expert GEMV: dst[n] = expert[idx[slot]]·aq` |
| `docs/cuda-megakernel-spec.md|gpu/decoderunner.go:933` | goinfer | `// relu²→int8 → down + residual into xd. The other kinds fall through to the mixer.` |
| `docs/cuda-megakernel-spec.md|gpu/forward_parity_test.go:36` | goinfer | `func TestWebGPU_forwardParity(t *testing.T) {` |
| `docs/cuda-megakernel-spec.md|gpu/gemv.go:41` | goinfer | `@compute @workgroup_size(64)` |
| `docs/gpu-residency-coverage.md|decoder/registry.go:262` | goinfer | `IntermediateDim:   cfg.IntermediateDim,` |
| `docs/how-inference-works.md|decoder/attention.go:117` | goinfer | `if !arch.LearnedPosEmbed && !arch.isNoPELayer(layer) {` |
| `docs/how-inference-works.md|decoder/attention.go:157` | goinfer | `cache.Append(layer, k, v)` |
| `docs/how-inference-works.md|decoder/attention.go:59` | goinfer | `nH, nKV, hd := arch.headsAt(layer), arch.NumKVHeads, arch.HeadDim` |
| `docs/how-inference-works.md|decoder/kvcache.go:132` | goinfer | `subCapture bool` |
| `docs/how-inference-works.md|decoder/kvcache.go:20` | goinfer | `func quantizeHeads(src []float32, q []int8, scales []float32, nKV, headDim int) {` |
| `docs/how-inference-works.md|decoder/model.go:1027` | goinfer | `for range maxTokens {` |
| `docs/how-inference-works.md|decoder/model.go:909` | goinfer | `func (m *Model) generateInto(ctx context.Context, out chan<- int, g *Generation, cache *` |
| `docs/how-inference-works.md|decoder/registry.go:19` | goinfer | `var registry = map[string]archAdapter{` |
| `docs/how-inference-works.md|decoder/sampler.go:179` | goinfer | `// can never silently diverge. They are separate predicates, not one widened one, so tha` |
| `docs/how-inference-works.md|decoder/sampler.go:186` | goinfer | `// though a temperature is set — the `top_k=1` shape. It is TRUE at any temperature, whi` |
| `docs/how-inference-works.md|decoder/sampler.go:188` | goinfer | `// distribution restricted to ONE token is deterministic regardless of that token's prob` |
| `docs/how-inference-works.md|decoder/session.go:71` | goinfer | `// stale history. Callers must skip it (and reconcile) for an empty prompt, so a rejecte` |
| `docs/ideas-weight-memory.md|decoder/mlp.go:69` | goinfer | `anchor: func mlp(h, out []float32, lw *LayerWeights, arch *Architecture, be Backend, scr` |
| `docs/measurements/c3-metal-consumer-window-v0.14.0.md|metal/gemma_parity_test.go:84` | goinfer | `t.Fatal("metal resident DECLINED — admission says it should be admitted")` |
| `docs/measurements/c3-metal-consumer-window.md|decoder/model.go:305` | goinfer | `switch o.Backend {` |
| `docs/measurements/c3-metal-consumer-window.md|decoder/residency.go:580` | goinfer | `func (m *Model) withResidency() *Model {` |
| `docs/measurements/c3-metal-consumer-window.md|metal/gemma_parity_test.go:84` | goinfer | `t.Fatal("metal resident DECLINED — admission says it should be admitted")` |
| `docs/measurements/demo-chat-gemma4e2b-blocked-2026-08-22.md|decoder/config.go:252` | goinfer | `SharedKVLayers          int   `json:"num_kv_shared_layers"`` |
| `docs/measurements/demo-chat-gemma4e2b-blocked-2026-08-22.md|decoder/gguf.go:2322` | goinfer | `firstShared := arch.NumLayers - g4.SharedKVLayers` |
| `docs/measurements/demo-chat-gemma4e2b-blocked-2026-08-22.md|decoder/registry.go:362` | goinfer | `SharedKVLayers:          cfg.SharedKVLayers,` |
| `docs/measurements/demo-chat-tier2-gates-2026-08-22.md|decoder/config.go:1101` | goinfer | `// under "text_config" rather than at the top level. Flatten it: decode` |
| `docs/measurements/demo-chat-tier2-gates-2026-08-22.md|decoder/weights.go:549` | goinfer | `if have["model.language_model.embed_tokens.weight"] {` |
| `docs/measurements/demo-chat-tier2-gates-2026-08-22.md|decoder/weights.go:962` | goinfer | `if d.inProjQKV, err = mkQ(nm("linear_attn.in_proj_qkv.weight"), convDim, hidden); err !=` |
| `docs/measurements/moe-expert-batching-m1-vs-mn-2026-09-01.md|decoder/forwardn.go:584` | goinfer | `ff, err = moeMLP(row(norm, i, hidden), lw, arch, be, moePrefillScr, m.pager)` |
| `docs/measurements/moe-expert-batching-m1-vs-mn-2026-09-01.md|decoder/mlp.go:292` | goinfer | `matmul(be, &ex.Gate, h, gate, 1)` |
| `docs/measurements/spec-x-pager-2026-09-02.md|cuda/backend.go:96` | goinfer | `return declined(fmt.Errorf("arch needs unimplemented feature(s) %v", missing))` |
| `docs/measurements/spec-x-pager-2026-09-02.md|cuda/prefill.go:161` | goinfer | `if r.moe \|\| r.gemma4Moe {` |
| `docs/measurements/spec-x-pager-2026-09-02.md|decoder/forwardn.go:146` | goinfer | `func (m *Model) specRollbackSafe() bool {` |
| `docs/measurements/spec-x-pager-2026-09-02.md|decoder/spec_adaptive.go:177` | goinfer | `case "cuda":` |
| `docs/measurements/spec-x-pager-2026-09-02.md|internal/serveapp/blockdrafter.go:13` | goinfer | `// IT FAILS STARTUP RATHER THAN DEGRADING SILENTLY. An operator who passed --drafter wan` |
| `docs/measurements/spec-x-pager-prereg-2026-09-02.md|cuda/prefill.go:162` | goinfer | `return fmt.Errorf("cuda prefill: arch needs the sequential path (moe/gemma4moe): %w", er` |
| `docs/measurements/theta-per-backend-2026-09-01.md|metal/backend.go:280` | goinfer | `func (a *metalResident) ForwardN(embeddings [][]float32, startPos int) ([][]float32, err` |
| `docs/multimodal.md|decoder/config.go:1110` | goinfer | `if json.Unmarshal(b, &nest) == nil && len(nest.TextConfig) > 0 {` |
| `docs/multimodal.md|decoder/gguf_qwen35.go:77` | goinfer | `anchor: func ggufQwen35Config(g *embed.GGUFFile) (*Config, error) {` |
| `docs/multimodal.md|decoder/weights.go:410` | goinfer | `const shardIndexFile = "model.safetensors.index.json"` |
| `docs/ollama-chase.md|cuda/resident.go:1454` | goinfer | `// All of it runs ON the executor thread — that thread made the context current — and th` |
| `docs/ollama-chase.md|cuda/resident.go:42` | goinfer | `// resolveCtxCap turns a request into the effective resident KV capacity:` |
| `docs/ollama-chase.md|cuda/resident.go:469` | goinfer | `g4x1, g4x2, g4rn Buffer` |
| `docs/ollama-chase.md|cuda/resident.go:718` | goinfer | `// declined to the staged/CPU path upstream.` |
| `docs/ollama-chase.md|decoder/gguf.go:644` | goinfer | `numLayers := u("block_count") - u("nextn_predict_layers")` |
| `docs/ollama-chase.md|decoder/gguf_qwen35.go:33` | goinfer | `numLayers := blocks - u("nextn_predict_layers") // drop the NextN/MTP block(s)` |
| `docs/ollama-chase.md|decoder/model.go:1058` | goinfer | `// sample. Identical to the logits path — guarded by ArgmaxEquivalent/GreedyEquivalent.` |
| `docs/ollama-chase.md|decoder/model.go:917` | goinfer | `// logits. On the batched archs this runs the layers at M=len in one pass (each` |
| `docs/ollama-chase.md|decoder/registry.go:1143` | goinfer | `// num_nextn_predict_layers MTP head is dropped (only num_hidden_layers load). The` |
| `docs/ollama-chase.md|decoder/residency.go:747` | goinfer | `return false, "sequential — this backend has no batched prefill (per-token resident forw` |
| `docs/ollama-chase.md|decoder/weightmat.go:380` | goinfer | `var matmulWSPool = sync.Pool{New: func() any { return new(linalg.Workspace) }}` |
| `docs/ollama-chase.md|decoder/weights.go:541` | goinfer | `// index so one loader serves both — the vision tower (model.visual.*) and MTP` |
| `docs/parity-coverage-policy.md|cuda/resident.go:1231` | goinfer | `// always been allocated without one, and a hard failure here would regress every driver` |
| `docs/parity-coverage-policy.md|linalg/dot.go:25` | aikit | `sum += a[k] * b[k]` |
| `docs/plan-cpubrrr-steal-and-bindings.md|decoder/registry.go:58` | goinfer | `"gpt_oss":          gptOssArchitecture,      // gpt-oss (20b/120b): sparse MoE + per-hea` |
| `docs/plan-cpubrrr-steal-and-bindings.md|linalg/quant.go:551` | aikit | `func QuantizeGroupInt4Row(row []float32, cols, group int, packed []byte, scales []float3` |
| `docs/post-v1.0-models.md|decoder/registry.go:1395` | goinfer | `func nemotronhArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {` |
| `docs/post-v1.0-models.md|decoder/registry.go:1511` | goinfer | `func deepseekArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {` |
| `docs/post-v1.0-models.md|decoder/registry.go:534` | goinfer | `// cohere2Architecture expresses Cohere2 / Command-R7B (model_type "cohere2":` |
| `docs/post-v1.0-models.md|decoder/registry.go:58` | goinfer | `"gpt_oss":          gptOssArchitecture,      // gpt-oss (20b/120b): sparse MoE + per-hea` |
| `docs/post-v1.0-models.md|decoder/registry.go:844` | goinfer | `Name:            "qwen3_5_moe",` |
| `docs/prompts/attention-a1-bit-identical-restructure.md|decoder/forwardn.go:709` | goinfer | `// of the next matmul); then ctx_head[K,hd] = scores·V_head, expressed as` |
| `docs/prompts/attention-a1-bit-identical-restructure.md|decoder/forwardn.go:919` | goinfer | `// stride kvDim (vt's column index steps by a whole KV row) — skipping` |
| `docs/prompts/goinfer-w4a8-opsperbyte-citations.md|linalg/quant.go:283` | aikit | `func QuantizeActivationsInto(aq []int8, scales []float32, a []float32, M, K int) {` |
| `docs/prompts/mac-cpu-decode-vs-ollama.md|decoder/sampler_chunked.go:111` | goinfer | `workers := min(runtime.GOMAXPROCS(0), numChunks)` |
| `docs/prompts/mac-cpu-decode-vs-ollama.md|decoder/weightmat.go:409` | goinfer | `w.MatmulBTW4A8Into(ws, a, dst, M)` |
| `docs/prompts/mac-cpu-decode-vs-ollama.md|decoder/weights.go:286` | goinfer | `workers := min(runtime.GOMAXPROCS(0), n)` |
| `docs/prompts/mac-demo-finish.md|internal/chatapp/main.go:381` | goinfer | `fmt.Fprintf(os.Stderr, "\033[2m[%d tok, %.1f tok/s]\033[0m", len(out), float64(len(out))` |
| `docs/queue-engineering.md|cmd/gate/configs.go:14` | goinfer | `models := env("GOINFER_GATE_MODELS", filepath.Join(home(), "models"))` |
| `docs/queue-engineering.md|cmd/gate/gpu.go:388` | goinfer | `g.models = env("GOINFER_GATE_MODELS", filepath.Join(home(), "models"))` |
| `docs/queue-engineering.md|cuda/argmax_tiebreak_test.go:19` | goinfer | `func TestArgmaxTieBreak(t *testing.T) {` |
| `docs/queue-engineering.md|cuda/backend.go:1175` | goinfer | `// cache, so the cap is correct by construction rather than covered by a margin.` |
| `docs/queue-engineering.md|cuda/prefill.go:290` | goinfer | `defer func() {` |
| `docs/queue-engineering.md|cuda/resident.go:301` | goinfer | `// backend.go locals; the per-layer KV cache and UploadKV read r.layers[l].kvDim.` |
| `docs/queue-engineering.md|cuda/resident.go:532` | goinfer | `func (r *cudaResident) recordUpload(e error) {` |
| `docs/queue-engineering.md|decoder/forwardn.go:1086` | goinfer | `logits[j] = sc * float32(math.Tanh(float64(val/sc)))` |
| `docs/queue-engineering.md|decoder/kvsnapshot_gemma4_test.go:10` | goinfer | `func TestSnapshot_refusesNonUniformKVWidth_C05(t *testing.T) {` |
| `docs/queue-engineering.md|decoder/layerpaging.go:42` | goinfer | `// mu guards the mutable paging state below (audit C-30). The pager lives on *Model, sha` |
| `docs/queue-engineering.md|decoder/model.go:733` | goinfer | `// Diagnostic — same byte-identical-output contract as ForwardCapture. Not wired for own` |
| `docs/queue-engineering.md|decoder/serialize.go:763` | goinfer | `func (w *Weights) hasPopulatedLayers() bool {` |
| `docs/queue-engineering.md|decoder/serialize_shapecheck_test.go:15` | goinfer | `func TestValidateShapes_catchesArchMismatch(t *testing.T) {` |
| `docs/queue-engineering.md|decoder/serialize_test.go:436` | goinfer | `t.Fatalf("streamed length %d != buffered %d", n, len(want))` |
| `docs/queue-engineering.md|internal/giw/bundle.go:114` | goinfer | `if avail := fi.Size() - (tokOff + 4); tokLen > avail {` |
| `docs/queue-engineering.md|internal/serveapp/embeddings.go:26` | goinfer | `// Embedding request bounds (audit C-21). /v1/embeddings is deliberately un-queued (the ` |
| `docs/queue-engineering.md|internal/serveapp/main.go:595` | goinfer | `// A SECOND signal during the drain force-exits instead of being swallowed by the buffer` |
| `docs/queue-engineering.md|linalg/quant.go:216` | aikit | `dequantRowInt8(deq, bq, 1.0)` |
| `docs/queue-engineering.md|metal/model.go:1061` | goinfer | `r.logitsHost[j] = sc * float32(math.Tanh(float64(v/sc)))` |
| `docs/queue-engineering.md|scripts/bench_peer.py:471` | goinfer | `def gate_cell_idle():` |
| `docs/review-2026-09-04.md|cmd/gate/gpu.go:1089` | goinfer | `_, cr, out := g.run(cell{` |
| `docs/review-2026-09-04.md|cmd/gate/gpu.go:1168` | goinfer | `g.bad("prefill parity/NaN gate — the f16-MMA TTFT path is wrong on a shipped model")` |
| `docs/review-2026-09-04.md|cmd/gate/gpu.go:1206` | goinfer | `// The resident-parity gates G-09 found opt-in-by-private-env-var. Every qwen3.5 fixture` |
| `docs/review-2026-09-04.md|cmd/gate/gpu.go:373` | goinfer | `func detectWebGPU() (present bool, backend string) {` |
| `docs/review-2026-09-04.md|cmd/gate/run.go:218` | goinfer | `func (r *results) tally(cellName string, topOnly bool) cellResult {` |
| `docs/review-2026-09-04.md|constrain/reflect.go:77` | goinfer | `for f := range t.Fields() {` |
| `docs/review-2026-09-04.md|constrain/reflect.go:78` | goinfer | `// V-14 (docs/review-2026-09-04.md): an anonymous field's reflect name IS its type` |
| `docs/review-2026-09-04.md|cuda/drafter.go:471` | goinfer | `anchor: func (d *residentDrafter) DraftBlock(blockIn [][]float32) ([][]float32, error) {` |
| `docs/review-2026-09-04.md|cuda/prefill.go:377` | goinfer | `maxNWin := startPos + M` |
| `docs/review-2026-09-04.md|cuda/resident.go:2435` | goinfer | `mustSplit := splitKVRequired(nWin)` |
| `docs/review-2026-09-04.md|cuda/spec_pager_interaction_test.go:1259` | goinfer | `UNKEYABLE` |
| `docs/review-2026-09-04.md|decoder/generate_vl.go:22` | goinfer | `// forget here was copy-pasted from a resident-path template and was doubly wrong: the` |
| `docs/review-2026-09-04.md|decoder/lora.go:273` | goinfer | `case hasOwnForward:` |
| `docs/review-2026-09-04.md|decoder/model.go:1027` | goinfer | `for range maxTokens {` |
| `docs/review-2026-09-04.md|decoder/model.go:1029` | goinfer | `case <-ctx.Done():` |
| `docs/review-2026-09-04.md|decoder/model.go:214` | goinfer | `if strings.HasSuffix(dir, ".giw") {` |
| `docs/review-2026-09-04.md|decoder/model.go:862` | goinfer | `if lg, perr := pf.PrefillLast(embs, from); perr == nil {` |
| `docs/review-2026-09-04.md|decoder/residency.go:596` | goinfer | `rf, ok, err := rb.BuildResident(m)` |
| `docs/review-2026-09-04.md|decoder/spec_eagle.go:21` | goinfer | `// resident-commit fix that applied to genNgramInto does NOT apply here for the same` |
| `docs/review-2026-09-04.md|decoder/speculative.go:167` | goinfer | `l, err := draft.resident.Forward(draft.embedResident(tok), dpos)` |
| `docs/review-2026-09-04.md|demo/agent/agent/agent.go:329` | goinfer | `turns := append([]msg(nil), s.history...)` |
| `docs/review-2026-09-04.md|gpu/attention.go:272` | goinfer | `// attnKeysDisabled force-disables the key-split kernel (GOINFER_ATTN_KEYS=0), so the ol` |
| `docs/review-2026-09-04.md|gpu/bufaccount.go:20` | goinfer | `// to the test that caused it. See TestNoBufferLeak.` |
| `docs/review-2026-09-04.md|gpu/layer.go:160` | goinfer | `keep = append(keep, newDeviceBuffer(xn, H))` |
| `docs/review-2026-09-04.md|gpu/qwen35_resident_parity_test.go:29` | goinfer | `func TestQwen35ResidentParity(t *testing.T) {` |
| `docs/review-2026-09-04.md|internal/prequant/prequant.go:91` | goinfer | `tmp := strings.TrimSuffix(out, ".giw") + ".tmp.giw"` |
| `docs/review-2026-09-04.md|internal/serveapp/helpers.go:103` | goinfer | `func requireAuth(key string, h http.HandlerFunc) http.HandlerFunc {` |
| `docs/review-2026-09-04.md|internal/serveapp/limits_test.go:187` | goinfer | `func TestDecoderEmbedder_truncatesToTheContextWindow(t *testing.T) {` |
| `docs/review-2026-09-04.md|internal/serveapp/main.go:457` | goinfer | `if !addrIsLoopback(*addr) && authKey == "" {` |
| `docs/review-2026-09-04.md|internal/serveapp/main.go:538` | goinfer | `anchor: func Main() {` |
| `docs/review-2026-09-04.md|internal/serveapp/main.go:546` | goinfer | `mux.HandleFunc("GET /{$}", srv.handleWebUI)` |
| `docs/review-2026-09-04.md|internal/serveapp/openai.go:103` | goinfer | `lm.tokenBytes = constrain.TokenBytes(lm.vocab, lm.tk.TokenText)` |
| `docs/review-2026-09-04.md|internal/serveapp/responses.go:282` | goinfer | `// Tool-call continuations round-trip via the next request's input; store the` |
| `docs/review-2026-09-04.md|internal/serveapp/sse_writer_test.go:211` | goinfer | `if strings.Contains(cb, "sseSend(") \|\| strings.Contains(cb, "sseEvent(") \|\|` |
| `docs/review-2026-09-04.md|internal/serveapp/vision_serve.go:81` | goinfer | `func spliceImageBlock(segs []tokenizer.Segment, block string) ([]tokenizer.Segment, erro` |
| `docs/review-2026-09-04.md|internal/serveapp/webui.go:114` | goinfer | `func (s *server) handleWebPull(w http.ResponseWriter, r *http.Request) {` |
| `docs/review-2026-09-04.md|internal/servecheck/check.go:206` | goinfer | `// Structured checks the promise the README makes: a schema the model cannot violate. Us` |
| `docs/review-2026-09-04.md|metal/batched_verify_test.go:288` | goinfer | `func TestBatchedVerifyKernelParity(t *testing.T) {` |
| `docs/review-2026-09-04.md|metal/gemma4_dense_scaled_test.go:25` | goinfer | `func TestGemma4DenseScaled_metalParity(t *testing.T) {` |
| `docs/review-2026-09-04.md|metal/gemma4_router_parity_test.go:31` | goinfer | `func TestGemma4Router_residentIdxParity(t *testing.T) {` |
| `docs/review-2026-09-04.md|pull/resolve.go:26` | goinfer | `// An already-cached file with the right size and digest is returned without touching th` |
| `docs/review-2026-09-04.md|scripts/bench_peer.py:719` | goinfer | `# Keep whichever identifying keys the previous header actually had. A reconstructed` |
| `docs/review-2026-09-04.md|scripts/bench_peer_prefill.py:88` | goinfer | `OLLAMA_MODELS_DEFAULT = os.environ.get("OLLAMA_MODELS", "")` |
| `docs/review-2026-09-04.md|scripts/remap_gate_citations.py:28` | goinfer | `base=args[0] if args else "HEAD~1"` |
| `docs/review-2026-09-04.md|tokenizer/bytelevel.go:206` | goinfer | `// N-24: an ADDED token's surface is stored verbatim, NOT byte-level-encoded, so pushing` |
| `docs/review-2026-09-04.md|tokenizer/sentencepiece.go:456` | goinfer | `func (t *Tokenizer) encode(text string, addBOS, parseSpecial bool) ([]int, error) {` |
| `docs/review-2026-09-04.md|tokenizer/sentencepiece.go:831` | goinfer | `func (t *Tokenizer) TokenText(id int) []byte {` |
| `docs/review-2026-09-04.md|tokenizer/sentencepiece.go:841` | goinfer | `if id < len(t.isAdded) && t.isAdded[id] {` |
| `docs/review-2026-09-04.md|tokenizer/splitshape.go:42` | goinfer | `// shapeGPT2Original is GPT-2's own ` ?\p{L}+\| ?\p{N}+\| ?[^\s\p{L}\p{N}]+\|\s+(?!\S)\|\s+`` |
| `docs/scoping-lfm2.md|decoder/arch.go:186` | goinfer | `type nemotronParams struct {` |
| `docs/scoping-lfm2.md|decoder/attention.go:107` | goinfer | `if arch.QKNorm {` |
| `docs/scoping-lfm2.md|decoder/config.go:903` | goinfer | `case c.UseQKNorm:` |
| `docs/scoping-lfm2.md|decoder/deltanet.go:176` | goinfer | `// 1. Projection + depthwise causal conv (+ SiLU). Taps t-K+1..t: the last K-1` |
| `docs/scoping-lfm2.md|decoder/forward_qwen35.go:30` | goinfer | `if arch.isLinearLayer(l) {` |
| `docs/scoping-lfm2.md|decoder/kvcache.go:50` | goinfer | `type KVCache struct {` |
| `docs/scoping-lfm2.md|decoder/mamba2.go:89` | goinfer | `// 2. Depthwise causal conv over xBC (+ bias, + SiLU). Taps t-K+1..t: the last` |
| `docs/scoping-lfm2.md|decoder/mamba2_chunked.go:60` | goinfer | `// Depthwise causal conv over xBC (+bias, +SiLU), then split into x/B/C.` |
| `docs/scoping-lfm2.md|decoder/rmsnorm.go:49` | goinfer | `func layerNorm(x, weight, bias []float32, rows, dim int, eps float64) {` |
| `docs/scoping-qwen38-flash-next.md|decoder/registry.go:2066` | goinfer | `// qwen35DenseArchitecture expresses Qwen3.8 (model_type qwen3_5): the SAME Gated-DeltaN` |
| `docs/scoping-qwen38-flash-next.md|decoder/registry.go:44` | goinfer | `"qwen3_5_moe_text": qwen35Architecture,      // the text-only checkpoint's model_type` |
| `docs/spec/09-mtp-heads.md|cuda/resident.go:268` | goinfer | `// owns a contiguous row. dnWin is the causal-conv ring, [(K-1)*convDim]. Both COMPOUND,` |
| `docs/spec/09-mtp-heads.md|cuda/resident.go:275` | goinfer | `dnWin, dnState               Buffer // persistent: conv ring, recurrent matrix state` |
| `docs/spec/09-mtp-heads.md|decoder/blockspec.go:550` | goinfer | `// breakEvenTokensPerRound is the acceptance below which block drafting LOSES.` |
| `docs/spec/09-mtp-heads.md|decoder/deltanet.go:147` | goinfer | `// head). Fixed size — independent of sequence length, and NOT position-` |
| `docs/spec/09-mtp-heads.md|decoder/deltanet.go:150` | goinfer | `type deltaState struct {` |
| `docs/spec/09-mtp-heads.md|decoder/deltanet.go:184` | goinfer | `win := st.convWin` |
| `docs/spec/09-mtp-heads.md|decoder/forwardn.go:146` | goinfer | `func (m *Model) specRollbackSafe() bool {` |
| `docs/spec/09-mtp-heads.md|decoder/gguf.go:644` | goinfer | `numLayers := u("block_count") - u("nextn_predict_layers")` |
| `docs/spec/09-mtp-heads.md|decoder/gguf_qwen35.go:33` | goinfer | `numLayers := blocks - u("nextn_predict_layers") // drop the NextN/MTP block(s)` |
| `docs/spec/09-mtp-heads.md|decoder/model.go:707` | goinfer | `// Derived from the dispatch table's Captures bit rather than re-listed: the families wh` |
| `docs/spec/09-mtp-heads.md|decoder/registry.go:1143` | goinfer | `// num_nextn_predict_layers MTP head is dropped (only num_hidden_layers load). The` |
| `docs/spec/09-mtp-heads.md|decoder/speculative.go:92` | goinfer | `if !target.specRollbackSafe() {` |
| `docs/spec/09-mtp-heads.md|decoder/weights.go:541` | goinfer | `// index so one loader serves both — the vision tower (model.visual.*) and MTP` |
| `docs/spec/README.md|decoder/forwardn.go:146` | goinfer | `func (m *Model) specRollbackSafe() bool {` |
| `docs/task-attention-decode-cost.md|decoder/forwardn.go:709` | goinfer | `// of the next matmul); then ctx_head[K,hd] = scores·V_head, expressed as` |
| `docs/task-attention-decode-cost.md|decoder/forwardn.go:819` | goinfer | `// MatmulBTAcc64Strided runs the SAME sequential f64 reduction as` |
| `docs/task-attention-decode-cost.md|decoder/forwardn.go:919` | goinfer | `// stride kvDim (vt's column index steps by a whole KV row) — skipping` |
| `docs/task-attention-decode-cost.md|internal/serveapp/main.go:371` | goinfer | `flag.BoolVar(&cfg.moeCacheExperts, "moe-cache-experts", false, "run a MoE model whose ex` |
| `docs/task-attention-decode-cost.md|linalg/linalg.go:58` | aikit | `var parThreshold = 1 << 24 // 16.78M MACs` |
| `docs/task-attention-decode-cost.md|linalg/matmul_strided.go:30` | aikit | `func MatmulBTAcc64Strided(a, bMat, dst []float32, M, K, N, bOff, bRowStride, bElemStride` |
| `docs/task-embed-and-harness-ux.md|chat/chat.go:112` | goinfer | `func Detect(meta Meta) (*Template, error) {` |
| `docs/task-embed-and-harness-ux.md|decoder/model.go:150` | goinfer | `type Options struct {` |
| `docs/task-embed-and-harness-ux.md|decoder/model.go:197` | goinfer | `func Load(dir string, opts Options) (*Model, error) {` |
| `docs/task-embed-and-harness-ux.md|decoder/model.go:806` | goinfer | `func (m *Model) Generate(ctx context.Context, prompt []int, maxTokens int, sp SamplingPa` |
| `docs/task-embed-and-harness-ux.md|internal/serveapp/main.go:328` | goinfer | `os.Exit(pullcmd.Run(os.Args[2:]))` |
| `docs/task-embed-and-harness-ux.md|internal/serveapp/main.go:957` | goinfer | `for _, str := range tmpl.Stops().Strings {` |
| `docs/task-fit-to-hardware.md|decoder/model.go:127` | goinfer | `// MoECacheSlotsRequest returns the requested per-layer expert-slot count, or 0 for "as ` |
| `docs/task-fit-to-hardware.md|decoder/weightbytes.go:56` | goinfer | `func (m *Model) ResidentWeightBytes() int64 { return m.residentWeightBytes(0) }` |
| `docs/task-fit-to-hardware.md|internal/serveapp/main.go:347` | goinfer | `flag.StringVar(&cfg.visionQuant, "vision-quant", "f32", "vision encoder weight quant: f3` |
| `docs/task-fit-to-hardware.md|internal/serveapp/main.go:363` | goinfer | `"  int4mix   attn int8 + FFN int4 (GGUF only): near-int8 quality at below-int8 RAM.\n"+` |
| `docs/task-fit-to-hardware.md|internal/serveapp/main.go:381` | goinfer | `"Repeatable. Unlike --lora (merged, one base per fine-tune), N adapters of one base cost` |
| `docs/task-fit-to-hardware.md|metal/backend.go:102` | goinfer | `const residentMemFraction = 0.70` |
| `docs/task-fit-to-hardware.md|metal/backend.go:162` | goinfer | `"GOINFER_NO_RESIDENT_MEM_GUARD=1 if this machine really fits it.\n",` |
| `docs/task-fit-to-hardware.md|metal/backend.go:81` | goinfer | `if !residentFitsMemory(m) {` |
| `docs/task-fit-to-hardware.md|metal/gemma4_moe.go:207` | goinfer | `if s := os.Getenv("GOINFER_METAL_MOE_SLOTS"); s != "" {` |
| `docs/task-fit-to-hardware.md|metal/moe.go:319` | goinfer | `// Synchronous paging (GOINFER_METAL_MOE_SLOTS=N>0): generalizes gemma4_moe.go's paging ` |
| `docs/task-fit-to-hardware.md|pull/pull.go:174` | goinfer | `Size   int64` |
| `docs/task-fp4-formats.md|decoder/forward_gptoss.go:17` | goinfer | `// speed on x86, and bench numbers are deferred (docs/task-mxfp4-gptoss.md §6.6).` |
| `docs/task-fp4-formats.md|decoder/gguf.go:1755` | goinfer | `anchor: func buildWeightsFromGGUF(cfg *Config, arch *Architecture, g *embed.GGUFFile, qu` |
| `docs/task-fp4-formats.md|decoder/gptoss_safetensors.go:17` | goinfer | `//  1. MXFP4 nibbles are SEQUENTIAL here (byte j holds elements 2j and 2j+1), where GGML` |
| `docs/task-freetoken-techniques.md|decoder/model.go:167` | goinfer | `MoECacheSlots int` |
| `docs/task-freetoken-techniques.md|internal/serveapp/main.go:278` | goinfer | `moeCacheSlots    int    // per-layer expert slot REQUEST (--moe-cache-slots); an upper b` |
| `docs/task-gpu-batched-prefill.md|decoder/residency.go:54` | goinfer | `// ResidentGreedy is an optional capability on a ResidentForward: compute the token's gr` |
| `docs/task-int4-int8-exact-mma.md|metal/kernels.go:220` | goinfer | `#define W4A8_BODY \` |
| `docs/task-int4-int8-exact-mma.md|metal/kernels.go:223` | goinfer | `device const half*  srow = bsc + (uint)gid*(K/32u); \` |
| `docs/task-int4-int8-exact-mma.md|metal/kernels.go:231` | goinfer | `acc += float(gi) * float(srow[wi>>2]); \` |
| `docs/task-int4-int8-exact-mma.md|metal/kernels.go:237` | goinfer | `kernel void gemv_w4a8_coal(device const uint* bq[[buffer(0)]], device const half* bsc[[b` |
| `docs/task-int4-int8-exact-mma.md|metal/kernels.go:241` | goinfer | `if (lid == 0) out[gid] = acc * asc[0];` |
| `docs/task-int4-int8-exact-mma.md|metal/kernels.go:287` | goinfer | `#define SA_BODY \` |
| `docs/task-int4-int8-exact-mma.md|metal/kernels.go:290` | goinfer | `uint G = K>>5u; \` |
| `docs/task-int4-int8-exact-mma.md|metal/kernels.go:293` | goinfer | `device const half*  sr = sct + (uint)row*G; \` |
| `docs/task-int4-int8-exact-mma.md|metal/kernels.go:301` | goinfer | `kernel void gemv_w4a8_sa(device const uint4* wq[[buffer(0)]], device const half* sct[[bu` |
| `docs/task-int4-int8-exact-mma.md|metal/kernels.go:307` | goinfer | `if (lane==0) out[row] = acc*asc[0];` |
| `docs/task-int4-int8-exact-mma.md|metal/kernels.go:51` | goinfer | `float sc=red[0]/127.0f; if(sc==0)sc=1; if(tid==0)asc[0]=sc; float inv=1/sc;` |
| `docs/task-int4-int8-exact-mma.md|metal/model.go:1352` | goinfer | `e.DispatchTG(r.pSABias, qkvRows*32, 256, r.H*2, L.qkvW, L.qkvS, r.aq, r.aSc, r.qkv, L.qk` |
| `docs/task-int4-int8-exact-mma.md|metal/model.go:1381` | goinfer | `e.Dispatch(r.pGemv, r.H*32, 32, L.dW, L.dS, r.dq, r.dSc, r.dO, r.uI)` |
| `docs/task-int4-int8-exact-mma.md|metal/model.go:460` | goinfer | `r.pRms, r.pQv, r.pGemv = pipe("rmsnorm_quant"), pipe("quant_vec"), pipe("gemv_w4a8_coal"` |
| `docs/task-int4-int8-exact-mma.md|metal/model.go:462` | goinfer | `r.pSA, r.pSABias, r.pSAResid = pipe("gemv_w4a8_sa"), pipe("gemv_w4a8_sa_bias"), pipe("ge` |
| `docs/task-metal-batched-verify-kernel.md|metal/kernels.go:220` | goinfer | `#define W4A8_BODY \` |
| `docs/task-metal-batched-verify-kernel.md|metal/kernels.go:287` | goinfer | `#define SA_BODY \` |
| `docs/task-metal-batched-verify-kernel.md|metal/model.go:330` | goinfer | `// N-32: dnValueDim is DeltaNet's out-projection staging width. deltanet.go dispatches p` |
| `docs/task-moe-streaming.md|decoder/forwardn.go:502` | goinfer | `// Sequential: add the attention residual, then re-norm the updated stream for the MLP.` |
| `docs/task-moe-streaming.md|decoder/forwardn.go:97` | goinfer | `// MoE FFN itself stays per-row (router picks different experts per token).` |
| `docs/task-moe-streaming.md|decoder/mlp.go:83` | goinfer | `// Only the chosen experts are evaluated — the point of MoE.` |
| `docs/task-moe-streaming.md|decoder/moepaging.go:15` | goinfer | `// only K·L per token; the router's top-k selection is the demand signal. The` |
| `docs/task-moe-streaming.md|decoder/moepaging_test.go:13` | goinfer | `// it with the frequency-aware policy (TestSpanCache_evictsLeastRecentWithPolicy),` |
| `docs/task-moe-streaming.md|decoder/residency.go:130` | goinfer | `return m.residentProjsInt4()` |
| `docs/task-recompute-audit.md|cuda/resident.go:275` | goinfer | `dnWin, dnState               Buffer // persistent: conv ring, recurrent matrix state` |
| `docs/task-recompute-audit.md|decoder/attention.go:89` | goinfer | `matmulInto(scr.ws, be, &lw.QProj, h, q, 1)` |
| `docs/task-recompute-audit.md|decoder/blockspec.go:195` | goinfer | `func (s *BlockSpec) generate(prompt []int, opt BlockSpecOptions, emit func([]int) bool) ` |
| `docs/task-recompute-audit.md|decoder/forwardn.go:134` | goinfer | `func (m *Model) hasRecurrentState() bool {` |
| `docs/task-recompute-audit.md|decoder/forwardn.go:146` | goinfer | `func (m *Model) specRollbackSafe() bool {` |
| `docs/task-recompute-audit.md|decoder/kvcache.go:448` | goinfer | `func (c *KVCache) TruncateTo(pos int) (exact bool) {` |
| `docs/task-recompute-audit.md|decoder/model.go:1029` | goinfer | `case <-ctx.Done():` |
| `docs/task-recompute-audit.md|decoder/model.go:1109` | goinfer | `case <-ctx.Done():` |
| `docs/task-recompute-audit.md|decoder/model.go:1163` | goinfer | `// completion. Every other exit above left resIDs nil, so the next turn cold-prefills.` |
| `docs/task-recompute-audit.md|decoder/model.go:949` | goinfer | `reuseFrom := m.residentReuseLen(prompt)` |
| `docs/task-recompute-audit.md|decoder/moepaging.go:62` | goinfer | `// A kind-4 tensor carries TWO on-disk representations (canonical + row4,` |
| `docs/task-recompute-audit.md|decoder/resident_reuse.go:50` | goinfer | `func (m *Model) residentReuseLen(prompt []int) int {` |
| `docs/task-recompute-audit.md|decoder/session.go:73` | goinfer | `func (s *Session) rewindForReuse(prompt []int) int {` |
| `docs/task-recompute-audit.md|decoder/session.go:98` | goinfer | `if rolledBack && s.cache.hasRecurrentState() {` |
| `docs/task-recompute-audit.md|decoder/speculative.go:125` | goinfer | `if atomic.CompareAndSwapInt32(&target.resBusy, 0, 1) {` |
| `docs/task-verification-surface-audit.md|decoder/blockspec.go:550` | goinfer | `// breakEvenTokensPerRound is the acceptance below which block drafting LOSES.` |
| `docs/task-zeno-compare.md|decoder/gguf.go:1448` | goinfer | `embMat := func(name string, out, in int) (linalg.WeightMat, error) {` |
| `docs/task-zeno-compare.md|decoder/gguf.go:1551` | goinfer | `if g.Has("output.weight") {` |
| `docs/task-zeno-compare.md|decoder/gguf.go:1561` | goinfer | `if arch.gemma4 != nil {` |
| `docs/task-zeno-compare.md|decoder/serialize.go:177` | goinfer | `anchor: func SerializeWeightsToRow4(out io.Writer, w *Weights, id string) (int64, error)` |
| `docs/task-zeno-compare.md|decoder/weightmat.go:125` | goinfer | `func streamQuantized(rows, cols int, mode quantMode, rowInto func(r int, dst []float32) ` |
| `docs/task-zeno-compare.md|internal/prequant/prequant.go:66` | goinfer | `// 2) Weights half: transcode the GGUF straight into the bundle, ONE LAYER at a` |

## Bare file index

Generated. Every file referenced WITHOUT a line number, and the repo it resolves in.
Existence only — there is no line to key content against, which is recorded rather
than papered over.

| file | repo |
|---|---|
| `cuda/pager_determinism_test.go` | goinfer |
| `decoder/deltanet.go` | goinfer |
| `decoder/eagle_accept_test.go` | goinfer |
| `decoder/eagle_alpha_test.go` | goinfer |
| `decoder/eagle_diag_test.go` | goinfer |
| `decoder/eagle_forward_test.go` | goinfer |
| `decoder/eagle_test.go` | goinfer |
| `decoder/eagle_throughput_test.go` | goinfer |
| `decoder/forwardn.go` | goinfer |
| `decoder/g26_sampler_bench_test.go` | goinfer |
| `decoder/mtp_head_test.go` | goinfer |
| `decoder/real_oracle_test.go` | goinfer |
| `decoder/resident_reuse.go` | goinfer |
| `decoder/spec_eagle_test.go` | goinfer |
| `decoder/tree_verify_test.go` | goinfer |
| `internal/serveapp/chaos_test.go` | goinfer |
| `internal/serveapp/fuzz_test.go` | goinfer |
| `internal/serveapp/openai.go` | goinfer |
| `scripts/bench_compare.sh` | goinfer |
| `scripts/bench_peer.py` | goinfer |
| `scripts/refresh_parity_hashes.sh` | goinfer |

<!-- /CITATION-INDEX -->
## RETRACTION, 2026-08-27 — the "152k sampler crossover" was a microbenchmark artifact

Raw: `docs/measurements/g26-152k-{anchor,head}.json`, log `g26-152k_run.log`.

I filed a finding that HEAD's sampler regressed +23.4% at 152k vocab, from
`BenchmarkG26SampleTemp1_152k`. **It does not reproduce end-to-end, and the sign is opposite.**
Tested on qwen2.5-coder-1.5B (vocab **151936** — the standard bench model already sits at the vocab
in question), `GOINFER_NO_OPTFWD=1` in BOTH arms so the optFwd effect cannot contaminate it, n=8:

| | anchor `ca29d6c` | HEAD | |
|---|---|---|---|
| greedy (control) | 219.81 ± 0.91 | 220.51 ± 1.36 | +0.3% |
| temp1.0_notrunc | 166.22 ± 2.31 | 180.37 ± 2.21 | **+8.5% HEAD FASTER** |
| **production sampling step** | **1467 us** | **1009 us** | **−31%** |
| *microbenchmark claimed* | *1358 us* | *1676 us* | *+23%* |

**In production HEAD's sampler is 31% FASTER at 152k. P10 delivered what it claimed.** There is no
sampler regression at any vocab measured: −35% at phi3-mini's 32064 (0.703 → 0.457 ms, from the
optFwd-OFF arms) and −31% at 151936.

**THE METHODOLOGICAL POINT, which is the durable part.** A tight-loop microbenchmark of the sampler
is not representative of decode, and this is the second time in ONE investigation that a sampler
microbenchmark produced a wrong conclusion — the first was `BenchmarkExpChunked`'s ~96 us
under-bounding the real tail by 7–10x. In a tight loop the scratch buffer stays hot and the
allocator serves from a warm free-list; in decode the same buffer is cold behind an ~8 ms forward.
The loop does not merely mis-scale the number, it **inverted the comparison**.

**Use the in-situ measurement instead, which is available and costs nothing extra: the same-build
greedy-vs-temp1.0 end-to-end difference, with `GOINFER_NO_OPTFWD=1` so overlap cannot confound it.**
Greedy's on-device argmax means that difference is the whole sampled tail rather than the sampler
alone — but it is measured where the code actually runs, and both times it disagreed with a
microbenchmark it was the microbenchmark that was wrong.

**Consequently the tail decomposition recorded above is DOWNGRADED.** Its "remainder 75 → 397 us"
split subtracted microbenchmark `Sample` figures from in-situ tails, and those figures are now known
to be unrepresentative — at 32k the microbenchmark says 553 us where production says 457 us. **The
split is not reliable and should not be quoted.** What is unaffected: the optFwd conclusion, which
rests entirely on an end-to-end A/B (`GOINFER_NO_OPTFWD=1`, n=15, plus the five-point ladder) and
never depended on the microbenchmark at all. `decoder/g26_sampler_bench_test.go` is kept, but its
doc comment should be read with this retraction attached.

## G27 · optFwdGate's hysteresis is a one-way latch, and its thresholds are calibrated on one model

Found 2026-08-28 while shipping the T ≤ 0.2 cap. Two independent defects in `decoder/spec_optfwd.go`,
neither of which the shipped cap depends on — it makes both moot in the losing regime — but both of
which matter if anyone re-enables the overlap more widely.

**1. The re-enable branch is unreachable.** `Observe` is called only from `optFwdStep`; the caller
invokes `optFwdStep` only when `Should()` is true. Once α drops below DisableAt the gate turns off,
no further outcomes are observed, α freezes, and `α ≥ EnableAt` can never be reached again within
that Generate. The documented dead band is one-way in practice.
`TestOptFwdGate_hysteresis` does not catch it because it calls `Observe` in an unconditional loop —
a unit test that is correct about the component while the composition defeats it. **A test at the
wiring level would have caught what a test at the component level could not.**

**2. EnableAt/DisableAt (0.90/0.75) are model-independent constants fitted on one model.** The
comment names it: the 90.9% worst-case break-even was measured on qwen2.5-coder-0.5b, a 152k-vocab
model. Break-even rises as the sampler share falls, so on phi3-mini (5.4% share vs 18.2%) the true
break-even is above 0.90 and the whole dead band sits below it — the gate cannot turn off in the
regime where the feature loses 2.8-6.8%.

**ESCALATED 2026-08-28 — G27 is no longer latent, it is blocking.** The latch stopped a measurement,
not just a code path: `TestOptFwd_hitRateLadder` cannot estimate low hit rates because the gate turns
off and stops observing (guess counts collapse to 6-18 on all three models). Break-even α is the
input the binary load-time gate needs (spec/10), so the latch now sits on the critical path. Fixing
it also needs the test's degenerate `[]int{1, 7, 42}` prompt replaced with a realistic one at depth
128 — measured on that prompt, phi3-mini reports α=1.00 where its throughput says optFwd loses 6.8%,
and the 1.5B reports α=0.61 where its throughput says it wins 5.1%; neither reconciles.

**Not fixed, deliberately.** Fixing (2) properly means knowing whether break-even tracks a static
property like sampler share, which is the open question the third-model run addresses. Fixing (1) is
small and self-contained but pointless while the overlap is capped at T ≤ 0.2, where the gate barely
runs. Both should be revisited together if the cap is ever raised.

## G28 · every bench prompt has four unique words, and speculation numbers were measured on them

Found 2026-08-28 while re-checking optFwd. `scripts/prompts.json` is calibrated for token DEPTH —
correct and deliberate for throughput, where decode cost is content-independent — but every entry is
`"Continue this text. the the the ... the"`. Anything whose value depends on what gets GENERATED was
measured on a regime the engine never serves.

**Demonstrated cost on the one case re-run:** the optFwd A/B moved up to **9.4 points** on a single
cell between filler and a 127-token prose paragraph, converting a 5.1% win into a 4.3% loss and
moving a model's break-even temperature from 0.95 to 0.37. That artifact was the entire case for an
adaptive gate (spec/10), which it retired.

**Unaudited and in scope**, all content-dependent:
1. **optFwd's `EnableAt = 0.90`** — sourced to a "90.9% worst-case break-even on qwen2.5-coder-0.5b",
   plausibly measured on this prompt set, since it is the only one in the repo (see G27).
2. **02's n-gram drafter, "wins on copy-heavy traffic"** — a claim about content, tested on a prompt
   that is maximally copy-heavy. Direction of the bias is obvious and favourable to the drafter.
3. **05/06 EAGLE acceptance (~1.6 tok/verify)**, and anything derived from it — including 09's Gate 1
   comparison, though 09 ran its own suite prompts rather than this file.

**The rule, cheap to apply going forward: a prompt file calibrated for DEPTH may be used for
THROUGHPUT, never for ACCEPTANCE.** Now recorded in benchmarks.md Methodology.

**Not chased here.** Re-measuring (2) and (3) is real work and neither currently gates a decision;
optFwd was re-measured only because it had just shipped a behaviour change.

## G29 · gate-runner reachability — AUDITED AND CLOSED 2026-08-28, no work needed

Carried from the untracked July audit as "census, heavy, composition, selector, mutation are
unchecked for reachability; mechanical, ~1 hour, the pattern is already written
(`TestRealckptCellCanReachEveryGate`)". **Audited instead of implemented, and the premise does not
hold.**

**The specific bug — a `-run` filter that cannot match a required gate — needs a filter to exist.**
Only three non-test files in `cmd/gate` carry `Run:` at all: `parity.go`, `gpu.go`, `configs.go`.
Census, composition, selector and mutation narrow nothing, so nothing can be unreachable by filter
there. Copying the parity test to them would assert a property they cannot violate.

**The GENERAL failure — "ran nothing, reported green" — is already covered everywhere, by five
different mechanisms. That is why it reads as unaudited:**

| runner | guard |
|---|---|
| parity | static: `TestRealckptCellCanReachEveryGate`, `TestBaseCellIsUnfiltered` |
| gpu | runtime: `noteIfEmpty`, with `TestGPUGate_emptyFilteredCellIsNotAPass` |
| heavy | `ZeroPolicy: "no-pass"` — zero PASSES is red even if tests ran and skipped |
| census | `ZeroPolicy: "no-tests"` — zero test events is red |
| composition | `len(parityGates) == 0` → "composition derived from nothing" |
| selector | `len(selS) == 0`, `len(rows) == 0` |
| mutation | baseline-must-be-green, plus asserting the mutation actually CHANGED something |

**`mutation.go`'s header is the best statement of this bug class in the repo** and records two prior
instances *inside the mechanism built to prevent it*: a `command -v staticcheck && staticcheck …`
whose `&&` short-circuited so the check "evaluated to nothing, and it was reported as clean", and a
`… | head -3; echo "exit=$?"` that read `head`'s status instead of the lint's. Its own words:
**"G-01 inside the mechanism built to prevent G-01."**

**A third instance landed the same day, on me** — `~/go/bin/staticcheck` was version-skewed against
the Go 1.27 toolchain, analysed NOTHING, and exited; a U1000 then held CI red across three pushes.
Same shape as that first bullet, years apart, in a different tool. The countermeasure is in CLAUDE.md
(run CI's pinned `go run …@v0.8.0`, and check CI after pushing).

**The real gap was legibility, not coverage: nobody could tell the property was covered without
redoing this audit.** That is what this entry fixes. **Closed — no code change.**

## G30 · the g4x2 clear's sync is REDUNDANT, not expensive — negative result, change NOT shipped

`docs/ollama-chase.md` lists, as a verified small win: *"CUDA g4x2 accumulator clear: H2D per MoE
layer per token … An on-stream memset/zero kernel removes an H2D (and its implicit null-stream sync)
per MoE layer per token."* **Built the cheapest form of that and measured it. It buys nothing.**

**What was built** (kept only in a scratchpad; reverted, not committed): replace the
`r.stream.Sync()` + `gpu.Upload(r.g4x2, r.g4zero)` pair in `layerTail` with a single
`r.stream.UploadAsync` from a pinned zero buffer. Same bytes, issued ON `r.stream`, so stream order
supplies the guarantee the Sync was buying and the drain becomes unnecessary.

**Measured on the 26B (`TestGemma4_26B_cache_B`), paired same-session, load-sampled every 15s:**

| arm | slots | ms/tok | tok/s | hits/misses | continuation |
|---|---|---|---|---|---|
| baseline | 48→30 | 65 | 15.44 | 16611/5229 | — |
| g4zero | 48→30 | 64 | **15.51 (+0.45%)** | 16611/5229 | **identical** |
| baseline | 16 | 86 | 11.61 | 12524/9316 | — |
| g4zero | 16 | 86 | **11.67 (+0.52%)** | 12524/9316 | **identical** |

**WHY IT BUYS NOTHING, which is the useful part.** In the cached configuration the clear is
immediately followed by `loadRoutedExperts`, which performs its own **Sync → D2H routing indices →
H2D expert misses**. Removing the clear's drain only relocates the stall into the next one. The
audit item is factually correct — there IS an H2D and an implicit sync per MoE layer per token — and
economically empty, because a second drain follows it unconditionally.

**Not shipped.** It trades away an `r.stream.Sync()` that guards a documented data race (audit R-03:
the null-stream DMA landing mid-`segC(l-1)` and zeroing the previous layer's expert contribution) for
0.5%, which is noise. Correctness risk for no gain.

**A CAUTION ABOUT THE FIRST MEASUREMENT, because it nearly shipped this.** An earlier unpaired-ish run
showed **+6.2%** (15.27 → 16.22) and I began writing it up as a win. It did not reproduce: two clean
load-audited pairs give +0.45% and +0.52%. The outlier was the g4zero arm, not a contaminated
baseline as I first guessed. A `go test` records no machine state — unlike `bench_peer` — so the
first run was unauditable after the fact, which is why the rerun added a 15-second load sampler.
**Two paired runs and per-arm load sampling turned a 6.2% "win" into noise.**

**WHERE THE EFFORT SHOULD GO INSTEAD.** The same audit entry names the real target one bullet down:
`loadRoutedExperts`'s Sync → D2H → H2D round trip, which is the drain that absorbs everything. That
is the item the standing Metal verdict addresses — *synchronous MoE paging is dead and speculative
prefetch is the path* — and it is a substantially larger piece of work than an on-stream memset.

## G31 · the C′ round trip PRICED: the sync is 0.4%, the expert DMA is 45–60% of decode

Measured 2026-08-28 on the 26B (`TestGemma4_26B_cache_B`, `GOINFER_MOE_CACHE_PROF=1`, load-checked
starts). **`cacheProf` had existed and been read by NOTHING** — `CacheProfForTest()` had no callers,
so the instrument was built and never wired. Raw `docs/measurements/g31-cprime-roundtrip.log`.

| | 30 slots (48 req) | 16 slots | ratio |
|---|---|---|---|
| **stall** (the `Sync` at `resident.go:887`) | 15 ms (**0.4%**) | 15 ms (**0.3%**) | 1.00 |
| **host** (slot bookkeeping) | 107 ms (2.7%) | 108 ms (2.0%) | 1.01 |
| **dma** (expert transfers) | 1.808 s (**45.1%**) | 3.227 s (**59.9%**) | **1.785** |
| misses | 5229 | 9316 | **1.782** |
| per token | 30.14 ms of 62.60 | 52.35 ms of 84.17 | |

**THE SYNC IS 0.4% OF DECODE.** That is the whole explanation for G30's null result, and it retires
the standing audit item: *"an on-stream memset/zero kernel removes an H2D (and its implicit
null-stream sync) per MoE layer per token"* targets **0.4%**, so no implementation of it — memset,
`UploadAsync`, or the new `CopyDevice` from aikit v0.31.0 — can pay. **Stop trying to remove syncs
from this path.**

**THE COST IS THE DMA, AND IT OBEYS AN EXACT LAW.** DMA time scales 1.785× against a 1.782× miss
increase: **346 µs per expert miss in both configurations**, to three digits. Stall and host are
constant because calls (2730 = ~43 MoE layers × 64 tokens) do not depend on slot count.

    per-token ms  ≈  32.1 (compute)  +  1.9 (round-trip overhead)  +  0.346 × misses_per_token

**Compute is constant at 32.5 / 31.8 ms across the two configurations, as it must be — so EVERY
tok/s difference between slot counts is DMA, nothing else.** That also re-explains the ctx-vs-slots
result: more slots raise the hit rate, which buys throughput purely by removing 346 µs transfers.

### What this says about speculative prefetch

**The prize is now quantified rather than inherited.** If the expert DMA were fully hidden behind
compute, token time would fall to `max(compute, dma)`:

| config | today | perfect overlap | ceiling |
|---|---|---|---|
| 30 slots | 15.97 tok/s | 30.81 tok/s | **+93%** |
| 16 slots | 11.88 tok/s | 19.10 tok/s | **+61%** |

**Perfect overlap is unattainable** — routing for layer L+1 is not known until layer L computes,
which is the serial dependency that makes the paging synchronous in the first place, and the reason
the Metal verdict says *speculative* prefetch rather than plain async. But the gap between 30 ms of
transfer and 32 ms of compute per token is close to ideal for hiding: there is almost exactly enough
compute to cover the transfers, if the transfers can be started early enough.

**This is the first quantitative case for that work.** Everything before it was a standing verdict
carried from Metal. The scoping should now be judged against a measured ~1.9× ceiling at 30 slots
and a 346 µs-per-miss cost model, not against "misses are on the critical path".

## G32 · Phase 0: the expert DMA is BANDWIDTH-bound at line rate — batching is dead, overlap is the only lever

Measured 2026-08-28, 26B at 30 slots, `GOINFER_MOE_CACHE_PROF=1`. Raw
`docs/measurements/g31-phase0-upload-split.log`. Each miss issues **four** blocking null-stream
uploads (weights + scales, for `expGU` and `expDown`), so the aggregate rate hides two populations:

| | calls | bytes | time | rate | per call |
|---|---|---|---|---|---|
| **weights** | 10458 | 15.55 GB | 1.427 s | **10.90 GB/s** | 136 µs over 1.49 MB |
| **scales** | 10458 | 1.94 GB | 0.384 s | 5.06 GB/s | 37 µs over 186 KB |

**THE WEIGHT COPIES RUN AT LINE RATE.** 10.90 GB/s is PCIe 3.0 x16 practical throughput on this
board. They are not overhead-bound, and no batching, pinning or stream trick makes them faster.

**This refutes the Phase 0 hypothesis, which was mine.** The aggregate figure from G31 — 346 µs per
miss over 2.40 MB = 6.95 GB/s — looked like a ~40% overhead gap. It is not: it is a weighted average
of large efficient copies (89% of bytes, at line rate) and small inefficient ones (11% of bytes, at
half rate). **An aggregate rate over a mixed size distribution is not a rate.**

**Batching is therefore nearly worthless.** Bringing the scale copies up to the weight copies' rate
saves 206 ms of 1814 ms DMA — **11.3% of DMA, 5.2% of decode**, and that is the ceiling, not an
estimate. Phase 1 as proposed is dropped.

**THE PATH MOVES 273 MB PER TOKEN**, sustained 4.44 GB/s across the whole decode, at a link that
tops out near 11. The volume is the cost, and it is irreducible at this hit rate.

### What survives, and the constraint that changes the prefetch design

Two levers remain, and only one is large:

1. **Move less** — a higher hit rate. Slots are VRAM-capped at 30 of a requested 48 on an 8 GB card,
   and §B4.1's slot sweep was already saturating. Little headroom.
2. **Overlap** — hide ~30 ms of transfer behind ~32 ms of compute. This is the whole prize (G31's
   ~1.9x ceiling), and it is now the ONLY large lever, because the transfers cannot be made faster
   and the volume cannot be much reduced.

**A CONSTRAINT ON SPECULATION THAT THE METAL VERDICT DOES NOT CARRY: THE LINK IS SATURATED.** A
mispredicted prefetch consumes the exact resource that is the bottleneck. That is unlike speculative
COMPUTE, where a wrong guess wastes idle cycles — here a wrong guess displaces a transfer that real
work is waiting on. **Prediction accuracy is therefore a kill gate, not a tuning parameter**, and a
prefetch scheme that is right 60% of the time may be a net LOSS rather than a smaller win.

**And the obvious predictor is already spent.** The Metal note proposes prefetching *last token's
experts*; a 30-slot LRU cache retaining 8 routed experts per layer already IS a last-token predictor,
and its hit rate is the 76%. **The 24% that miss are precisely the experts a last-token predictor
gets wrong.** Any CUDA prefetch must predict NEW routing — e.g. running layer L's router early on an
approximate residual — which is a different and harder proposition than the verdict implies.

**Next step, before any prefetch code: measure whether routing is predictable at all.** Offline, from
captured routing indices — how often does a router run on the pre-MoE residual agree with the true
top-8? That number, against the saturated-link constraint above, decides whether Phase 2 exists.

## G33 · SCOPE — is expert routing predictable at all? (decides whether prefetch exists)

Scoped 2026-08-28 off G31/G32. **No code written.** G32 left overlap as the only large lever
(~1.9x ceiling) and named two constraints the inherited Metal verdict does not carry. This scopes
the measurement that decides whether any prefetch scheme can work on CUDA.

### The structural fact that shapes everything

**Each layer owns its own bank of 128 experts** (`expGU`/`expDown` are fields on `cudaLayer`).
Expert 37 at layer 5 is a different tensor from expert 37 at layer 6. **Therefore cross-layer
prediction is meaningless** — there is no "the same expert" to prefetch — and the only predictors
that can exist are TEMPORAL: same layer, across tokens.

**But the 30-slot LRU cache IS a temporal predictor**, and its accuracy is the measured 76.1% hit
rate. So a prefetcher built on temporal correlation is competing with the cache using the same
signal, and can only win where the cache's policy — not its signal — falls short.

### Tier 1 — trace-only, and it may end the whole item

**The trace is free.** `GOINFER_G4_CAPTURE=1` already fills `g4capIdx [][]uint32` in "APPEND order
(token-outer, layer-inner)". Only a dump-to-file is missing (~10 lines in the test). One capture run
on the box, then all analysis is offline Python.

**Step 1 — validate the simulator before trusting it.** Replay the trace through a 30-slot per-layer
LRU and check it reproduces the measured **16611 hits / 5229 misses / 76.1%**. If the simulation does
not match, the model of the cache is wrong and nothing downstream means anything. This is the
do-nothing arm for the analysis itself.

**Step 2 — decompose the 5229 misses into COLD vs EVICTED.**

- **EVICTED**: the expert WAS used at that layer within the last K tokens, and the cache pushed it
  out. A better policy (or more slots) recovers it. Prefetch is then cache-policy work.
- **COLD**: the expert has not been used at that layer recently at all. **No temporal predictor can
  see it coming**, because there is no signal to predict from.

**PRE-REGISTERED READING, fixed before the run:**

| result | conclusion |
|---|---|
| mostly EVICTED | The signal exists and the policy wastes it. Try policy first (LFU, frequency-aware admission, layer-aware slot allocation) — cheaper and safer than speculation. |
| mostly COLD | **No temporal scheme can help, and that includes every prefetcher built on the routing history.** Prefetch as the Metal verdict describes it is dead on CUDA. Only Tier 2 remains. |
| split | Report the split and size each half against the 346 µs/miss cost model before choosing. |

### Tier 2 — early-router speculation, only if Tier 1 says COLD dominates

The only way to anticipate a cold expert is to COMPUTE the routing early, not predict it: run
layer L's router on an approximation of its input (the residual before layer L-1's MoE contribution
is joined), start the transfers, and correct on mismatch. This needs new instrumentation — the
router is small (hidden x 128), so the compute is cheap; the question is purely whether the
approximate input yields the same top-8.

**THE PRECISION GATE, and it is unusually strict because the link is SATURATED (G32).** A mispredicted
prefetch spends the bottleneck resource itself. With gain ≈ p·(transfer hidden) and cost ≈
(1−p)·(transfer wasted), and both at the same 346 µs, break-even sits near **p = 0.5 before any
discount for imperfect overlap** — so the real bar is higher, call it **p > 0.6**. A 60%-accurate
scheme that would be a clear win for speculative COMPUTE can be a net LOSS here.

**Lead-time requirement, for sizing:** a layer's transfers are ~1.9 misses x 346 µs ≈ 660 µs against
~755 µs of per-layer compute (32.5 ms / 43 layers). **Roughly one full layer of lead is needed to
hide a layer's transfers** — which is exactly the horizon the early-router approach would buy, and
no more. There is no slack for a two-layer pipeline.

### Cost

Tier 1: ~10 lines of dump code, one ~5 min capture run, then offline analysis. **Tier 1 is cheap
enough that it should happen before any further CUDA work on this path**, because a COLD-dominated
answer retires the largest remaining item on the page.

## G33 RESULT · prefetch is aimed at the wrong 10% — the misses are CAPACITY, not prediction

Tier 1 ran 2026-08-28. Trace `docs/measurements/g33-routing-trace.json` (2730 decisions, 30 MoE
layers x 91 positions = 27 prompt + 64 generated), replay `scripts/g33_replay.py`.

**VALIDATION GATE PASSED EXACTLY.** The offline per-layer LRU replay reproduces **16611 hits / 5229
misses / 76.1%** — the measured hardware figures, to the token. The cache is a plain per-layer LRU
and the model of it is right, so everything below is trustworthy. (Positive-controlled first on
synthetic traces: constant routing → all-cold, no evictions; 40 experts through 16 slots → 87.5%
evicted at age 5.)

**COLD IS A STARTUP ARTIFACT.** The headline split — cold 2239 (42.8%) vs evicted 2990 (57.2%) —
inverts once read against position:

| positions | cold | evicted | cold % |
|---|---|---|---|
| 0–12 | 1232 | 34 | **97.3%** |
| 26–38 | 233 | 481 | 32.6% |
| 52–64 | 106 | 470 | 18.4% |
| 78–90 | 65 | 698 | **8.5%** |

Cold misses are FIRST TOUCH of a (layer, expert) pair. In a 91-position window they cannot amortize;
by the last bucket they are 8.5%. **In steady state ~90% of misses are EVICTIONS.**

**THE SLOT CURVE, and it saturates at the cold floor:**

| slots | hit rate | misses | DMA/token |
|---|---|---|---|
| 16 | 57.3% | 9316 | 35.4 ms |
| **30 (today, VRAM-capped)** | **76.1%** | **5229** | **19.9 ms** |
| 48 (requested) | 85.5% | 3170 | 12.1 ms |
| 64 | 88.7% | 2466 | 9.4 ms |
| 96 / 128 | 89.7% | **2239** | 8.5 ms |

**2239 is exactly the cold count.** With every expert resident the only misses left are unavoidable
first touch; everything above that floor is capacity.

### The conclusion, which retires Tier 2

**Speculation can only attack COLD misses — and those are the floor, already the small half in steady
state.** Capacity attacks the ~90% that are evictions. **Prefetch is aimed at the wrong 10%**, and
the early-router scheme scoped in Tier 2 is not worth building: even perfect prediction of new
routing cannot beat simply keeping the expert resident, and it would spend saturated bandwidth to do
worse.

**The lever is VRAM, and it is now priced exactly** at 346 µs per miss avoided:

- 30 → 48 slots: **7.8 ms/token saved, +14% throughput**
- 30 → 64 slots: **10.5 ms/token, +20%**
- beyond 96: nothing, the floor is reached

That makes the existing **ctx-vs-slots trade** (`GOINFER_26B_CTX`) the actionable lever rather than a
curiosity: it converts resident KV into expert slots, and this curve gives the exchange rate. **Cost
a context reduction against the hit-rate curve before spending any effort on prediction.**

**Tier 2 is CLOSED, not parked.** The saturated-link precision gate that worried G32 turns out not to
matter — the question it guarded was never the binding one.
## G34 RESULT · block verify pays, but the SLOT BUDGET caps the width — not acceptance

Scoped and measured 2026-08-31 off the rev-9 framing: *block verify attacks the expert-DMA term by
batching real, already-decided work across tokens rather than guessing routing ahead, so G33's kill
of routing-PREDICTION prefetch does not rule it out.* That framing is correct, and the mechanism is
genuinely different — but the answer is not the one the framing implies.

**Method.** Offline replay of `docs/measurements/g33-routing-trace.json` (91 positions × 30 MoE
layers, topK 8, 30 slots) through G33's per-layer LRU, which reproduces the measured 76.1% hit rate
— that validation gate is why these numbers mean anything. A block verify of width K runs the target
over K draft positions in ONE pass, so at layer L all K positions' routing is computed together and
the **union** is requested at once. Script: `scripts/g34_blockverify_replay.py`.

**Result 1 — batching does NOT reduce misses. It increases them.**

| K | union per layer | fits 30 slots | misses vs serial |
|---|---|---|---|
| 2 | 16 | yes | 1.007× |
| 3 | 24 | yes | 1.017× |
| 4 | **32** | **NO** | 1.026× |
| 6 | 48 | NO | 1.069× |
| 8 | 64 | NO | 1.187× |

**The reason is the same shape as G33's: the predictor is already spent.** A 30-slot LRU already
retains recent experts, so the cross-token reuse block verify would batch is reuse the cache was
*already* converting into hits. Unioning captures nothing new and perturbs eviction. Worse, a block
needs K×topK experts resident **simultaneously** where serial decode needs 8 — so past K=3 the
block's working set exceeds the slot budget and the cache thrashes.

**Result 2 — block verify still pays, via compute amortization, and the width cap is the finding.**
Under G31's model (`32.1 compute + 1.9 overhead + 0.346 × misses`), with compute amortized over α
accepted tokens and DMA scaling with *verified* positions:

| K | α=1.6 | α=2.0 | α=2.5 | α=3.0 |
|---|---|---|---|---|
| serial | | **18.56 tok/s** | | |
| 2 | 21.62 | **27.02** | — | — |
| 3 | 16.90 | 21.13 | **26.41** | 31.69 |
| 4 | 13.84 | 17.30 | 21.63 | 25.96 |
| 6 | 9.90 | 12.38 | 15.47 | 18.57 |

**K=4 at α=2.0 is 17.30 — BELOW the 18.56 serial baseline.** The cliff sits exactly where
K×topK = 32 crosses the 30-slot budget. So the verify width here is capped by **VRAM slots, not by
acceptance** — the opposite of the usual speculative-decoding constraint, and not something the
Metal-derived framing anticipated.

**Where that lands against measured α.** MTP Gate 1 gave 2.02–2.91 (`docs/spec/09-mtp-heads.md`);
EAGLE-3 gave 1.60. So the operating point is **K=2–3**, and both pay: +46% at K=2/α=2.0, +42% at
K=3/α=2.5. Note the DMA per *accepted* token is flat when α ≈ K and rises as α falls below K — the
win is entirely compute amortization, and the DMA term is a drag that grows with the gap between K
and α.

**Limits, stated.** One trace, 64 generated tokens, gemma-4-26b at 30 slots on the 2070S; G31's
constants come from the same box and configuration. Rollback cost on rejection is not modelled. And
the slot budget is the binding constraint, so **a card with more VRAM moves the cliff** — 30 slots
was itself a cap on a requested 48 (G32), which makes this a finding about an 8 GB card rather than
about block verify in general.

**Result 2's compute-amortization case is unverified against a real kernel — there isn't one yet.**
G31's 32.1 ms/token compute figure was measured on ordinary serial decode, one token at a time, and
Result 2 treats it as roughly flat per verify-pass regardless of K. But `cuda/moe.cu`'s kernels are
GEMV, architecturally: `moe_route` is one thread per token by design ("runs ONE thread once per
layer per token"), `gemv_w4a8_moe` takes a single `slot`, not an array, and `prefill_batched.cu`
says outright that moe.ptx is untouched by the batched-prefill work. There is no fused, batched MoE
kernel anywhere in this codebase to run a K-wide verify pass through — and G34 itself is an offline
trace replay, not a CUDA measurement — so the flat-compute assumption behind +46%/+42% has not been
tested against real hardware. External data point, same day: [llama.cpp#27621](https://github.com/ggml-org/llama.cpp/pull/27621)
fused exactly this — MoE router and GLU/up-projection fusion, from nrows==1 to nrows-up-to-8 — for
the identical reason, speculative-decode verify batches on MoE, and it took dedicated kernel work to
get there: 2–22% E2E, gains tapering past width 4. Real, but bounded, not free. If block verify is
ever built, `mmvq.cu` / `topk-moe.cu` in that PR is a working reference for the fusion shape; until
then, Result 2 is a projection, not a measurement.

## G35 · the WGSL megakernel is unreachable — and the kernel actually pinning decode was a serial reduce

Scoped 2026-09-02 as two experiments: (A) execute the CUDA cgo-free megakernel spike's "Phase 2",
(B) test whether that spike's K1/K2/K3 stage-grouping can be built as 3 WGSL dispatches instead of
13, to close the WebGPU-vs-native retention gap. **Experiment A was already done. Experiment B's
target is not expressible in WGSL. And the ablation profile run to decide B named a different
kernel entirely — a 1-line-of-reasoning defect worth +9.9% decode, bit-identical.**

**Box:** RTX 2070 SUPER, driver 595.91.07, Qwen2.5-Coder-1.5B (`~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf`,
loaded `int8int8`, GPU-resident), greedy/temp-0, warm, best-of-6 × 48 tokens. Local disk, not the archive.

### (A) Already executed — the spike doc is archived with the answer in it

`docs/completed/task-cuda-cgofree-spike.md` carries the whole Phase-2 log through 2026-08-31, well past
"the box fills": production backend, real-checkpoint parity, W4A8 coalescing 43%→80% of peak, the
launch diet (18→13→8 launches/layer), **and the §5.2 three-super-kernel fusion itself** — K1
(rmsnorm folded into the QKV GEMV) and K3a shipped as `cuda/fused_qkv.cu` behind `fuseQKV` /
`GOINFER_CUDA_NO_FUSE`; **K2 built, measured at ~0%, and reverted.** `cuda/megakernel.cu` is a dead
July scaffold referenced only from tests; the work landed elsewhere. Nothing in Experiment A was
open. Recorded here because the spec (`docs/cuda-megakernel-spec.md`) still reads as a live
prep artifact and does not say the spike it belongs to has closed.

### (B) 3 WGSL dispatches/layer is NOT reachable — and 8 is already the floor CUDA found

The stage-grouping does not survive translation, for the reason the CUDA spec routes around:

| stage-group | expressible in WGSL? | why |
|---|---|---|
| K1 = rmsnorm+quant ⊕ QKV GEMV | **yes**, mechanically | redundant recompute in `var<workgroup>` + `workgroupBarrier()` |
| K2 = attention ⊕ quant ⊕ O-proj | **NO** | attention is one workgroup/head; O-proj needs every head. Cross-workgroup dependency ⇒ grid-wide sync. WGSL has none — `storageBarrier()` orders memory *within* a workgroup, it is not an execution barrier across them |
| K2′ = quant ⊕ O-proj (what CUDA actually shipped) | yes | but CUDA **measured it at ~0% and reverted it** |
| K3 = ...⊕ SwiGLU ⊕ down-proj | expressible, **bandwidth-fatal** | down-GEMV blocks must redundantly read gO+uO — CUDA measured ~6.9 MB/layer against a 4 MB L2 |

The escape hatch for K2 — have every O-proj block redundantly recompute all heads' attention — is
not viable: it multiplies KV-cache traffic by the O-proj block count (~24× at this shape), on the
term that already dominates the token. So the reachable target is **8 dispatches/layer, exactly
where CUDA landed, by the same three concessions.** "13 → 3" was never a WGSL-side possibility.

**And the premise that WebGPU sits at 13/layer is stale.** Measured: **366 dispatches/token = 12 per
layer** (28 layers) + 30, across 7 pipeline classes. `docs/decode-fusion-next.md`'s "~535
dispatches/token" predates Increments 1–2. The gap B was scoped to close is roughly half the size it
was described as, before any of the above.

### The measurement that mattered: ablate, don't attribute

`TestDecode_dispatchProfile` (`gpu/decode_dispatch_profile_test.go`) re-records the whole token plan
R×8 into one pass with one pipeline class **omitted**, one Submit, one blocking Poll, and differences
against the unablated plan. Ablation, not per-kernel attribution, because the question is "what would
deleting this buy" — a dispatch that fully overlaps its neighbours costs wall-clock nothing to remove,
and attribution would still bill it. Pipeline classes are named by reflecting over `Context`'s
pipeline fields, so it cannot drift as kernels are added.

**Baseline, before any change (ms/token, ablation delta):**

| class | n/token | pos=64 | % | pos=512 | % |
|---|---|---|---|---|---|
| gemv | 113 | 4.226 | 49.1% | 4.220 | 20.5% |
| **attn** | 28 | 1.473 | 17.1% | **13.440** | **65.3%** |
| **quantize** | 28 | **1.039** | **12.1%** | 1.054 | 5.1% |
| swigluQuant | 28 | 0.848 | 9.9% | 0.840 | 4.1% |
| gemvBias | 84 | 0.455 | 5.3% | 0.473 | 2.3% |
| rmsQuant | 57 | 0.452 | 5.2% | 0.457 | 2.2% |
| qkvFinalize | 28 | 0.049 | 0.6% | 0.068 | 0.3% |
| **token** | 366 | **8.605** | | **20.570** | |

Two things fall out, and neither is the megakernel:

1. **Attention is the bottleneck at any real context**, and it is O(pos): 1.47 ms at pos 64 → 13.44 ms
   at pos 512, 65% of the token. It is dispatched as `nH`=12 workgroups on a 40-SM card. This is the
   same occupancy wall `docs/ollama-chase.md` names on the CUDA side (12 blocks / 40 SMs, 11.9%
   occupancy) and answers there with split-KV. **Not attempted here** — flagged as the next lever,
   not a result.

   > **CORRECTED by G36 (same day), on both counts.** The split-KV figure quoted here as "1.30×" was
   > a GEMV `RN·MT` tuning result, not split-KV; split-KV measured **1.20×**. And the occupancy
   > *diagnosis* does not transfer: it is correct for the CUDA kernel, but WebGPU's was running a
   > different algorithm — reducing once per KEY rather than twice per layer — so the lever was the
   > barrier count, not the workgroup count. Acting on the analogy would have shipped the wrong fix;
   > CUDA's own P6a data says split-KV *loses* on this exact geometry at this depth. G36 measured
   > **9.2× on the kernel and 2.39× on the token at 1k context** instead.
2. **The entire glue budget K1/K3a would attack is 5.2% of the token** (rmsQuant), falling to 2.2% at
   pos 512. Even a free, perfect K1 could not clear a 1.3× bar; CUDA's own K1+K3a bought **+1.5% on
   the 1.5B** (its +21% was the 0.5B, which is glue-dominated). Experiment B is **NO-GO on its own
   arithmetic**, before the expressiveness wall above is even reached.

### The defect the profile exposed, fixed and measured

`quantize` cost **37 µs/dispatch against rmsQuant's 7.9 µs for strictly more work** — an inversion
with no bandwidth explanation. Cause, in the kernel's own comment: it computed the row max-abs as a
**serial scan on lane 0**, justified as "trivial at decode; the rows run in parallel". That is exactly
inverted — **decode is M=1, so there is only ever one row**; one lane scanned all 1536 elements while
63 idled at the barrier, in a single-workgroup dispatch. The parallel-reduce idiom was already in the
adjacent file (`rmsnormQuantWGSL`), which is why the two kernels' costs diverged.

Replaced with a 64-lane tree reduce. **Bit-identical, provably: f32 `max` is exact and
order-independent** (associative, commutative, no rounding), so every reduction order yields the same
scale and the same packed int8 — this is not a tolerance argument.

| | before | after | |
|---|---|---|---|
| `quantize` class, pos=64 | 1.039 ms | **0.104 ms** | **−90%** |
| token, pos=64 (plan) | 8.605 ms | **7.694 ms** | −10.6% |
| token, pos=512 (plan) | 20.570 ms | 19.709 ms | −4.2% |
| **real-model decode** (interleaved A/B) | **104.8 / 104.8** | **118.4 / 118.4** | **+13.0%** |

**And it holds through the server, which is the claim that matters.** The change is one shared
WGSL pipeline (`c.quantizePipeline`) bound by the resident decode path, the batched/fused paths and
the staged path alike — no flag, no build tag — so `serve` gets it by construction. Measured anyway,
because this repo has a retired claim built on exactly that inference (an in-process kernel number
published beside a peer's HTTP number). Real `gpu/cmd/serve`, `-backend webgpu -quant int8int8`,
streaming `/v1/chat/completions`, greedy, 128 max tokens, inter-token rate excluding TTFT:

| server-to-server | best of 4 |
|---|---|
| old kernel | **95.9 tok/s** |
| new kernel | **104.3 / 104.5 tok/s** (two separate server starts, interleaved around the old) |
| | **+8.9%** |

Note the level shift: **118.4 in-process vs 104.5 through the server** on the same kernel. The
server number is the smaller and the honest one for any user-facing claim; the in-process number is
correct only for what it measures. TTFT was unchanged (~132–151 ms both arms) — this is a decode
lever, not a prefill one.

The decode row is a **same-session interleaved A/B** — control, treatment, control, treatment,
rebuilding between each — not two numbers from different sessions, because this box drifts ~3.5%
between sessions and the effect had to be separated from that. Both arms reproduced exactly
(the harness reports best-of-6 × 48 tokens). A cross-session pair taken earlier the same day read
103.7 → 114.0 (+9.9%); the interleaved pair supersedes it and is the number to quote.

Parity: `TestResidentForwardN_parity` gives cosine=1.000000, maxAbsDiff=0 on the resident path — and
**was run on the old kernel too, giving the same maxAbsDiff=0**, so the two agree exactly with the
CPU reference and therefore with each other. The A/B is measured, not inferred from the max argument.

**This is larger than everything Experiment B was scoped to win**, and it needed no fusion, no new
kernel, and no dispatch-count change — which is the finding. The lever was inside a kernel the
dispatch-count framing treated as a fixed unit.

### Limits, stated

- One box, one card, one dense checkpoint. `swigluQuant` (0.81 ms, 10.6% at pos 64) has the *same*
  single-workgroup shape and was **not** examined; it is the obvious next check, not a claim.
- The **89.7 tok/s** baseline in `docs/completed/gpu-assessment.md` §0.0 is superseded twice over:
  fresh-and-unmodified is **104.8**, and **118.4** after this change. Both are goinfer-vs-goinfer on this
  box. **No peer was run in this session**, so nothing here licenses restating any "% of Ollama"
  figure — that needs a same-session interleaved `scripts/bench_peer.py` run.
- The profiler's values go garbage after the first repetition (residual epilogues accumulate R times).
  Deliberate: both arms are equally garbage and NVIDIA f32 has no denormal/NaN timing cliff to bias
  the comparison. It is a timing harness; the parity tests are the correctness gate.
- Attention's split-KV lever is **named, not measured**. The CUDA 1.30× does not transfer on its own.

## G36 · decode attention was reducing once per KEY — 9.2× on the kernel, 2.4× on the token at 1k

Executed 2026-09-02, directly off G35's "attention is the next lever" and its ablation profile.
G35 named the term; this identifies the mechanism and removes it. **The lever was not occupancy,
which is what the CUDA-side analogy predicted, and acting on that analogy first would have been
wrong.**

**Box:** RTX 2070 SUPER, driver 595.91.07, Qwen2.5-Coder-1.5B int8 GPU-resident (`nH=12 nKV=2
headDim=128`, 28 layers), greedy, local disk. A/B by `GOINFER_ATTN_KEYS=0/1` — one binary, no
rebuild between arms.

### The mechanism: a barrier per key, not too few workgroups

`attnShaderWGSL` put one lane on each of the `hd` dimensions. That makes the q·k dot for **every
key** a cross-lane reduction: `red[d]=prod`, barrier, 7 barrier'd tree levels, trailing barrier =
**9 `workgroupBarrier()` per key**, or **4,617 per layer** at nKeys=513.

The roofline says how far off that is. 28 layers × 513 keys × 2048 B = **29.4 MB of KV per token**,
which at 448 GB/s is a **0.066 ms** job. Measured: **13.68 ms**.

| | measured | roofline | efficiency |
|---|---|---|---|
| GEMV (weights) | 4.14 ms | ~1.55 GB | ~83% |
| attention, before | 13.68 ms | 29.4 MB | **0.48%** |
| attention, after | 1.49 ms | 29.4 MB | 4.4% |

Two halves of one token differing by 175× in bandwidth efficiency is not a tuning gap, and the
implied 106 ns/barrier (13.68 ms ÷ 4,617 ÷ 28 layers) says what the time was.

### Why the occupancy reading was wrong, and what it would have cost

G35 flagged "nH=12 workgroups on a 40-SM card" and pointed at CUDA's split-KV, which `ollama-chase.md`
records at 11.9% achieved occupancy, Waves/SM 0.04. That diagnosis is correct **for the CUDA kernel**,
and following it here would have been a mistake twice over:

1. **CUDA's own P6a amendment says split-KV LOSES on this exact geometry at this depth** — the 1.5B
   measures 0.941 at 256 keys and 0.939 at 512; its crossover is in (512, 1024]. phi3-mini (MHA,
   nH=32) never crosses at any depth, declining monotonically to 0.754. The shipped gate had to
   become a measured per-geometry lookup because the *form* "ON iff nWin ≥ f(geometry)" is falsified.
2. **CUDA and WebGPU were not running the same algorithm.** `cuda/attn_block.cu` splits lanes over
   **keys** and reduces twice per layer; WebGPU split over **dims** and reduced once per key —
   a ~250× difference in barrier count. Split-KV is an occupancy fix for a kernel already
   structurally right. Ours was not, and adding workgroups underneath a per-key barrier storm
   would have bought little and shipped a lookup table to maintain.

**The general form: an analogy that transfers a DIAGNOSIS also transfers the assumption that the
two things are the same underneath.** The occupancy numbers were real, measured, and about a kernel
this one only resembled. Reading both kernels side by side cost ten minutes and changed the whole plan.

### The port, and the one thing WGSL genuinely cannot do

Lanes own disjoint keys; only the softmax max and denominator reduce. `workgroupBarrier()` maps 1:1
to `__syncthreads()`, and **no subgroup ops are needed** — which matters, because the cogentcore
binding exposes none.

The real constraint is **dynamic workgroup storage**: CUDA sizes `sc[nWin]` per launch via
`extern __shared__`; a WGSL `var<workgroup>` is fixed at compile time. So this **tiles** — TILE=2048
keys, online-softmax state (m, l, acc) carried across tiles. Storage is 2048·4 + 128·4 = **8.5 KB,
inside WebGPU's guaranteed 16 KB**: no limit raise, no portability cost, any context length. Barriers
per layer go 9/key → ~19/tile.

`vec4` K/q loads are load-bearing, not a micro-optimisation. Splitting over keys strides the K read
by kvDim across the warp; ncu measured that pattern using **~22% of each 32-byte L1TEX sector** on
the CUDA twin, which is why `attn_block.cu` reads `float4`. Hence the `hd%4==0 && kvDim%4==0`
eligibility guard.

### Result — and the shape matters more than any row

| server-to-server (best of 2) | dim-split | key-split | |
|---|---|---|---|
| 128 tokens | 106.2 | **131.0** | 1.23× |
| 512 tokens | 69.0 / 67.6 | **122.8** | **1.80×** |
| 1024 tokens | 48.5 / 47.9 | **115.2** | **2.39×** |

Real `gpu/cmd/serve`, streaming `/v1/chat/completions`, greedy, inter-token rate excluding TTFT;
the dim-split arm was re-measured after the key-split arm to close the interleave and reproduced
(69.0→67.6, 48.5→47.9).

### Combined with G35 — the end-to-end number for the two changes together

Measured separately afterwards, because the table above is the attention A/B *on top of* G35's
`quantize` fix and multiplying two deltas is not a measurement. Both-off is HEAD with `gpu/device.go`
reverted to its pre-G35 state (verified: no other commit has touched that file since) and
`GOINFER_ATTN_KEYS=0`; both-on is the shipped binary. Same server, same prompt, interleaved.

| server-to-server | before (both off) | now | |
|---|---|---|---|
| 128 tokens | 93.5 / 97.0 | **131.8** | **1.38×** |
| 512 tokens | 63.7 | **125.3** | **1.97×** |
| 1024 tokens | 46.2 / 46.8 | **116.2** | **2.50×** |

Two bit-level-cheap changes — one serial reduction made parallel, one workgroup decomposition
transposed — for **2.5× at 1k context**, on a path a whole-token decomposition had already declared
to be within ~10% of its ceiling. Neither was a fusion, a new kernel, or a dispatch-count change.

**The old kernel loses 54% of its rate from 128 to 1024 tokens; the new one loses 12%.** Decode no
longer falls off a cliff as context grows — which is the regime agent loops, RAG and code editing
actually run in, and which every short-prompt benchmark in this repo is blind to.

Ablation profile, same session: attention **13.676 → 1.491 ms** at pos 512 (**9.2×**), whole token
**19.609 → 7.431 ms** (**2.64×**). Attention falls from 70% of the token to 20%; GEMV is the
majority term again at 55.6%, which is the healthy shape.

### Correctness

**Not bit-identical** — the denominator sums in a different order and the tiled rescale reassociates.
Attention never was: it runs f32 against the CPU oracle's f64. `TestAttnKeys_parity` gates it against
that same f64 reference at **11 depths chosen around the TILE boundary** — 2047 / 2048 / 2049 /
4133 / 5000 — because the cross-tile rescale is the only genuinely new arithmetic and a test staying
under TILE would exercise the single-tile path, pass, and vouch for a kernel broken at every real
long context. cosine=1.00000000, maxAbs ≤ 1.5e-7, plus agreement with the dim-split kernel it
replaces. nKeys=1 is in the list but is explicitly **not** sufficient alone: softmax over one element
is 1.0 at any scale. `TestWebGPU_forwardParity` and `TestResidentForwardN_parity` (cosine=1.000000,
maxAbsDiff=0) both pass with it on.

### Limits, stated

- **Scoped to f32 KV, hd ≤ 128, hd%4==0, kvDim%4==0.** f16/int8 KV and the wide (hd>128) kernels are
  untouched and still take the dim-split path — so gemma3/gemma4's 256/512-wide layers get **nothing**
  from this. Porting the shape to those is the obvious follow-up and is NOT done.
- Attention is now at 4.4% of roofline, up from 0.48%. Still ~20× off. It is no longer the dominant
  term, so the next profile should be re-read before assuming it is the next target.
- One box, one card, one checkpoint. No peer was run; nothing here licenses a "% of Ollama" figure.
- **The full `./gpu/` suite cannot confirm this and did not.** After ~110 tests the run exhausts
  WebGPU devices (`failed to request device`), and from there `TestWebGPU_forwardParity` and
  `TestResidentForwardN_parity` **SKIP** — the exact gates that matter, reported as `ok`-adjacent
  noise rather than as failures. `TestKVCacheF16_fit` and `TestVisionEnableResident_parity` turn red
  from the same cause; both pass standalone, under both kernels, and a control run on unmodified code
  reproduces the exhaustion. Pre-existing and not caused by this change, but it means **a green full
  suite here is not evidence** — the gates above were run standalone on purpose. Worth its own fix.

## G37 · P-20 measured, then fixed: constrained decoding 1.9× → **1.21×** — and the ratio is model-dependent by construction

Measured 2026-09-02, before designing anything on top of it, per the ordering revision in
`docs/task-embed-and-harness-ux.md` §6. P-20 was explicit that its numbers were an **estimate**
("Estimate 40–120 ns/token → 6–30 ms per step… plausibly 3–10× slower per token on GPU") and
named the measurement to run; the audit itself lists P-20 among the items whose measurement is
cheap. It cost about an hour and it changes the answer.

**Why it was worth doing first.** Constrained generation is the README's headline promise —
"a Go struct the model cannot violate" — and `Into[T]` is the proposed facade's flagship call.
Whether that costs 1.2× or 10× decides whether it is a feature or a documented caveat, and the
facade would otherwise have been designed around a guess.

**Method.** `MaskAt` is `Process`'s hot loop without the commit, so driving a grammar to a chosen
state by committing BYTES and timing `MaskAt` isolates the per-step masking cost at that state —
no tokenizer round-trip, no decode, nothing else in the sample. Real vocab (V=151,936 Qwen2.5),
min-of-7 per run. `constrain/maskcost_test.go`, `GOINFER_HEAVY_TESTS=1`.

| grammar state | ms/step | ns/token |
|---|---|---|
| `fsObjKeyOrClose` (after `{`) | 3.17 | 20.9 |
| **`fsStr` (inside a string — the dominant JSON state)** | **6.03** | **39.7** |
| `fsNum` | 4.11 | 27.1 |
| complete document | 2.10 | 13.8 |

**Against P-20's estimate: the per-token cost lands at the BOTTOM of the predicted band** (39.7 ns
vs 40–120), and the per-step cost likewise (6.0 ms vs 6–30). The estimate's arithmetic was sound;
its inputs were pessimistic.

### The lever P-20 called cheapest was real and unspent

`isEOS` was a `map[int]bool`, **probed once per vocab id per step — 151,936 map lookups**. Replaced
with an indexed `[]bool` (bounds-checked, since logits can be the model's padded vocab length —
the same over-long-logits case M26 guarded in `tokenBytes`). Behaviour-identical; the `constrain`
suite is green.

Same-session **interleaved A/B**, stash/restore between arms, `fsStr` state:

| | run 1 | run 2 | run 3 | mean |
|---|---|---|---|---|
| `map[int]bool` | 7.121 | 7.091 | 7.096 | **7.103** |
| `[]bool` | 6.235 | 5.623 | 6.247 | **6.035** |

**−15%**, 3/3 pairs the same sign. Interleaved because a first single pair read −21% and the next
two runs of the same build read 6.14/6.18 — the variance is ~11% and a lone pair would have
overstated it. (Note the arms differ in stability: the map arm is tight at 7.09–7.12, the slice arm
spreads 5.6–6.25.)

### The verdict, against the pre-registered bar

G4 in the task doc pre-registered: **≤1.5× ships, >2× the facade documents the cost instead of
hiding it**, ambiguous in between. Against this box's post-G36 resident-GPU 1.5B token
(6.2 ms at pos 64, 7.4 ms at pos 512):

- before the lever: **1.96–2.15×**
- after the lever: **1.81–1.97×**

That is **inside the ambiguous band and at the top of it** — which is exactly the zone the
pre-registration exists to stop being argued away. So: not "fine", not "blocking". `Into[T]` ships
only with the cost stated, or after the remaining L-07 levers move it under 1.5×.

**The structural finding, which the estimate did not make: the mask cost is CONSTANT in the model.**
It is O(V) and independent of model size, so the RATIO worsens as decode gets faster — the same
6.0 ms against a ~2 ms step is ~4×, not ~1.9×. A single number cannot answer G4; the bar has to be
stated per model class, and the fast paths are where constrained decoding hurts most. That also
means the levers pay for themselves twice on the small/fast models the demo tiers ship.

### The second lever: an EXACT bitmap for the string state — 17×, and G4 now passes

Proposed in review as "a Bloom filter?", and the measurement said use an exact structure instead.
Both grammars answer identically inside a JSON string:

    '"' → closes it (legal, moves state) · '\' → escape (legal, moves state)
    b < 0x20 → illegal · anything else → legal, AND THE STATE DOES NOT MOVE

So for a token containing none of those three classes, legality inside a string is a property of
the TOKEN ALONE — precomputable once per vocabulary, answerable with one bit test. Measured share
of Qwen2.5's vocab:

| class | share | verdict in a string |
|---|---|---|
| no control, no `"`, no `\` | **96.88%** | always legal, state unchanged ⇒ **one bit test** |
| contains a control byte | 2.32% | usually illegal — but NOT fast-pathed (see below) |
| contains `"` or `\` | 0.81% | legal but transitions ⇒ real walk |

**Why exact and not Bloom.** A Bloom "maybe" forces the very walk being avoided, so it only saves
work on definite-nos — while here the definite answers are 97% of the vocabulary. And at ~19 KB
for the whole vocab there is no space pressure to trade accuracy for. A bit test is also cheaper
than k hash rounds.

**Conservative by construction.** Control-byte tokens are NOT classified illegal, even though most
are: a token like `"`+0x0A closes the string *before* the control byte, which is then judged in a
different state. Only "provably legal and state-invariant" is fast-pathed; everything else walks.

Interleaved A/B, `fsStr`: **5.628 / 6.160 ms OFF → 0.343 / 0.343 / 0.347 ms ON — ~17×**, and the
ON arm is far more stable than the OFF arm.

| state | before both levers | after |
|---|---|---|
| `fsStr` (dominant) | 7.10 | **0.35** |
| `fsObjKeyOrClose` | 3.93 | 3.30 |
| `fsNum` | 5.17 | 4.46 |
| complete document | 2.99 | 2.12 |

**And per-state costs are not what a caller pays.** A real document spends most of its steps inside
strings. Timing the mask at *every step* of a representative 17-token document gives **1.299 ms/step
weighted**, i.e. **1.21× (pos 64) / 1.18× (pos 512)** — **under G4's 1.5× bar, so G4 PASSES on this
model class.** The worst single state is now a ≤1.72× upper bound rather than a typical step, and
quoting it would overstate the real cost several-fold.

**Correctness is proved, not sampled.** `TestPlainString_exact` compares the fast path against the
full walk for EVERY id of an adversarial vocabulary (exhaustive over 1- and 2-byte tokens across
control/quote/backslash/UTF-8 classes, plus shapes like `"`+0x0A where the first byte changes which
state later bytes are judged in), at every string state a nested document passes through, for both
grammar implementations. Proven able to go red: mis-classifying control bytes as safe fails it with
`id 2 ("\x00") — fast path says masked=false, full walk says masked=true`.

### Limits, stated

- **G4 passes for THIS model class, not universally.** The mask is still constant in model size, so
  1.299 ms against a ~2 ms decode step is ~1.65× and would miss the bar. The per-class statement
  from the first measurement stands.
- The weighted figure is one 17-token document. A document of many short keys and numbers sits
  higher (those states are untouched by the bitmap); one with long string content sits lower.
  `fsObjKeyOrClose` and `fsNum` are the remaining targets, and the same cache-an-exact-bitmap idea
  generalises to them per `(node, state)`.
- The snapshot/restore of the frame stack — the mechanism that makes the untouched states cost what
  they do, and which tracks stack DEPTH — is still there. Not attempted.
- One box, one vocab (V=151,936), one grammar shape (`struct{string; int; []string}`). A larger
  schema has more states but the same O(V) per step; a smaller vocab is proportionally cheaper.
- The mask is CPU work that runs *after* the logits come back, so it is additive to the decode
  step rather than overlapped. Not separately verified against the resident path's own timing.
- **Two costs are excluded and neither is measured here.** A non-nil `LogitProcessor` disables
  the speculative-decode paths (`decoder/spec_eagle.go:21`) and, on the resident backends, the
  on-device greedy argmax fast path. So the real cost of constrained decoding on a spec-enabled
  or greedy-fast-path configuration is HIGHER than the ratios above. Sizing that is open.
- The remaining L-07 levers (per-state first-byte bitmap, string-state cache, vocab byte-trie)
  are untouched. This measured the floor and took only the cheapest one.
