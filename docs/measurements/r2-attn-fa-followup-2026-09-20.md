# R2 follow-up — a real (but irrelevant) race found and fixed, root cause still open

Continuation of `docs/measurements/r2-attn-fa-2026-09-19.md` (read that first — this record
assumes its background, kernel description, and the original `TestAttentionFA_endToEndReproduction`
table). That record parked R2 with two named-but-untested hypotheses: (1) whether Metal's automatic
hazard tracking correctly serializes `attention_fa` → `attention_fa_combine` under this kernel
pair's specific buffer-reuse pattern, and (2) whether the third-and-later command buffer
specifically behaves differently from the first two. This session picked R2 back up, tested both,
and closes neither — but narrows the search space in a way the next session shouldn't have to
redo.

**Status: STILL PARKED 2026-09-20. A genuine race was found, reproduced in isolation, and fixed —
but proven, empirically, to have zero effect on the actual reproduction test. §2's own
"position-independence, call-count/state mechanism" conclusion is RETRACTED — see §2b — by a
sharper experiment that found it was drawing a real conclusion from a confounded test. The
corrected picture is that the divergence IS position-linked, but not to a fixed absolute position
or residue class either; a full line-by-line audit of the kernel source and every Go-side
dispatch/uniform/buffer site still found no defect. The root cause remains unknown.**

## 1. The encode-ahead race: real, reproduced, fixed — then ruled out as R2's cause

### Hypothesis

`metal/model.go`'s production decode path (`execLoop`) pipelines command buffers: it encodes
buffer N+1 while buffer N is still executing on the GPU (`BeginNP`/`Commit`/`FinishEncoding`/
`WaitDone`, distinct from the synchronous `Begin`/`End` pattern used elsewhere). The
`attention_fa` dispatch site was, at the time, calling `r.uAttnFAG.SetU32(...)` /
`r.uAttnFANSplit.SetU32(...)` inline, once per layer, during encoding. Those are raw CPU writes to
Metal shared memory — not a tracked GPU command — so if that write happens while buffer N's
`attention_fa` dispatch is still reading the same buffer on the GPU, Metal's automatic hazard
tracking (which only orders GPU-encoded commands against each other) does not protect it. A classic
"safe on every machine until it isn't" pattern.

### Isolated repro

`metal/attn_fa_pipeline_race_test.go` (new): a minimal harness using the exact production
pipelining pattern (`cq.BeginNP()` / `enc.FinishEncoding()` / `cur.Commit()` / `cur.WaitDone()`),
with shared `uG`/`uNSplit` uniform buffers whose value is deliberately varied every iteration
(`splitVariants := []int{14, 8, 4, 28, 14, 8, 4, 28}`) so a race becomes observable instead of
silently landing on the same value every time. Re-run today for exact numbers:

```
$ go test -mod=readonly -tags goinfer_testhooks ./metal/ -run TestAttentionFA_pipelinedEncodeRace -v
iter 0: wantG=6 wantNSplit=14 cosine=1.0000000 maxAbs=0.0000
iter 1: wantG=6 wantNSplit=8  cosine=1.0000000 maxAbs=0.0000
iter 2: wantG=6 wantNSplit=4  cosine=NaN        maxAbs=1.3436   -- FAIL
iter 3: wantG=6 wantNSplit=28 cosine=1.0000000 maxAbs=0.0000
iter 4: wantG=6 wantNSplit=14 cosine=1.0000000 maxAbs=0.0000
iter 5: wantG=6 wantNSplit=8  cosine=1.0000000 maxAbs=0.0000
iter 6: wantG=6 wantNSplit=4  cosine=-0.0010009 maxAbs=1.5352   -- FAIL
iter 7: wantG=6 wantNSplit=28 cosine=0.1645985  maxAbs=0.8199   -- FAIL
--- FAIL: TestAttentionFA_pipelinedEncodeRace (0.10s)
```

This is a genuine race: iterations 0-1 correct, iteration 2 wrong (echoing R2's own "2 correct,
then wrong" shape closely enough to be the reason this was tried first) — but then iterations 3-5
correct again and 6-7 wrong, an irregular pattern with no fixed period. That irregularity is itself
diagnostic: a deterministic bug would fail the same iterations every run; this is what an actual
timing race looks like when it's reproduced at all.

### Fix applied

`metal/model.go`: moved the `SetU32` calls for `uAttnFAG`/`uAttnFANSplit` out of the per-layer
dispatch site (`encodeAttentionResidualWith`) and into `setPos`, which runs once per forward call,
before any layer's dispatch is encoded and before the command buffer that will use those values is
committed — the same timing guarantee `uPos`/`uNKeys`/`uRopePos` already relied on. This is safe
specifically because `nSplit`/`G` are provably depth-independent constants in the only regime
`attention_fa` ever engages in (`attnFASplitFor`'s `nKeys/32` cap never binds above
`attnFADepthFloor`), so a single write per forward call — instead of one that could race the
*next* call's encode-ahead — is correct, not just less racy. Added `r.attnFANKV` (cached at
`BuildResident`) so `setPos` can compute `G` without a `*decoder.Model` reference. Full comment
trail is at the `uAttnFAG, uAttnFANSplit Buffer` field declaration and at `setPos` itself.

### Result: zero effect on the actual bug

Re-running `TestAttentionFA_endToEndReproduction` after the fix produced **bit-for-bit identical
(still wrong) output** — same failure at step 2, same magnitude. This is because
`TestAttentionFA_endToEndReproduction` calls `forwardLogits`, which uses the synchronous
`Begin()`/`End()` pattern: one command buffer per forward call, encoded and then fully waited-on
before the function returns. There is no encode-ahead, no pipelining, no second command buffer
in flight while the first is executing — the exact condition the race requires never occurs on
this code path. The fix is real hardening (it closes a genuine defect that *would* bite the
pipelined `execLoop` production decode path under the right timing), but it does not touch, and
was never going to touch, this reproduction test.

**This was reported to the user directly as a walk-back rather than presented as a fix**, and the
user chose to keep investigating for the real, stateful cause rather than stop here.

## 2. Position-independence claimed here — RETRACTED in §2b, read that first

### Hypothesis under test

Decode step 2 (absolute position 1602 in the original repro) is where the prefill's own content
first gets attended to *and* is one step further from the prefill boundary than steps 0-1. Is the
divergence caused by something specific to the data at that particular position (an accumulator
overflow, a NaN-producing edge case in a specific K or V value), or does it depend only on how many
decode calls have happened, independent of what the data at that call actually is?

### Test

Added `metal/attn_fa_e2e_depth_test.go` (`TestAttentionFA_endToEndReproductionDepth2200`), a copy
of `attn_fa_e2e_test.go` with `prefillLen` changed from 1600 to 2200. Because the harness's
Gaussian inputs are drawn from `rand.NewSource(7)` and prefill consumes `prefillLen` rows before
decode starts, this also changes every actual float value the decode steps see — not just the
position label.

```
$ GOINFER_HEAVY_TESTS=1 GOINFER_TEST_MODEL=~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
  go test -mod=readonly -tags goinfer_testhooks ./metal/ -run TestAttentionFA_endToEndReproductionDepth2200 -v -timeout=5m
decode step 0 (pos=2200): cosine=1.0000000 maxAbs=1.9073e-06
decode step 1 (pos=2201): cosine=1.0000000 maxAbs=1.9073e-06
decode step 2 (pos=2202): cosine=0.9994620 maxAbs=6.1551e-01   -- FAIL
decode step 3 (pos=2203): cosine=0.9992243 maxAbs=8.9019e-01   -- FAIL
decode step 4 (pos=2204): cosine=0.9997135 maxAbs=4.9738e-01   -- FAIL
--- FAIL (80.04s)
```

Side by side with the original depth-1600 result:

| step | depth-1600 pos | depth-1600 cosine | depth-1600 maxAbs | depth-2200 pos | depth-2200 cosine | depth-2200 maxAbs |
|---|---|---|---|---|---|---|
| 0 | 1600 | 1.0000000 | 2.86e-06 | 2200 | 1.0000000 | 1.91e-06 |
| 1 | 1601 | 1.0000000 | 2.86e-06 | 2201 | 1.0000000 | 1.91e-06 |
| 2 | 1602 | 0.9995727 | 0.588 | 2202 | 0.9994620 | 0.616 |
| 3 | 1603 | 0.9995052 | 0.623 | 2203 | 0.9992243 | 0.890 |
| 4 | 1604 | 0.9995974 | 0.569 | 2204 | 0.9997135 | 0.497 |

### Conclusion — RETRACTED, see §2b

**This section's conclusion is wrong. Left in place, struck through in spirit not in markdown, so
the reasoning error is visible rather than quietly edited away — see §2b for why and for the
corrected picture.**

Changing the prefill depth by 600 tokens — which changes every K/V value the decode steps
attend to, via a different RNG consumption offset — does **not** move the failure. It still starts
at relative decode step 2 (the third `ForwardEmb` call), at a comparable divergence magnitude
(cosine 0.9992-0.9997, maxAbs 0.5-0.9 in both runs). It would be an extraordinary coincidence for a
data-dependent numerical edge case (an overflow triggered by a specific K or V value at a specific
position) to land on the same *relative* call twice with completely different underlying data. This
rules the data-dependent-content hypothesis out and confirms the divergence is a **call-count or
carried-state mechanism**: something about being the third call to use `attention_fa` since the
resident was built, independent of what position that call is at or what the surrounding data is.

One structural fact this also establishes, worth recording since it wasn't explicit before: `Forward
Batch` (the batched prefill call) cannot itself be dispatching `attention_fa` at any point during
prefill, regardless of prefill length — `attention_fa`'s kernel addresses `q` as exactly one
query position per kv head (`qr := q + kvh*G*hd`), which cannot represent a multi-position batch.
If prefill silently invoked it per-token internally, the two prefill lengths (1600 vs 2200) would
put a very different number of prior `attention_fa` calls behind decode step 0 (roughly 64 vs 664,
counting positions past the depth floor within each prefill), and the failure would not land on the
same relative decode step. It does, which is only possible if `attention_fa` activity starts fresh
at decode step 0 in both cases — i.e., the "third call" is the third `forwardLogits`/`ForwardEmb`
invocation, full stop, not the third time `attention_fa` fires including some prefill-internal
usage.

## 2b. RETRACTION, 2026-09-20 (later the same day): §2's conclusion was drawn from a confounded test

An outside review (relayed by the user, from a separate Claude session that read this record cold)
caught the error directly: **§2's two prefill depths, 1600 and 2200, are both multiples of 8.**
Every decode step in both runs therefore lands on a position congruent to the same residue mod 4
(and mod 8) as the equivalent step in the other run — so "the third `ForwardEmb` call" and "key
count ≡ 3 (mod 4)" were **the same event in both runs tested**, and §2's experiment could not have
distinguished a call-count trigger from a position-residue trigger even in principle. `attention_fa`
groups its cooperative key load by 4 (`metal/kernels.go`'s own doc comment: "32 lanes x half4"), so
a boundary tied to `nKeys mod 4` is not a stretch — it is the single most structurally-motivated
alternative the original test happened to be blind to.

The same review offered a second, independent hypothesis worth checking alongside the first: that
"two clean steps, then a stable wrong plateau" is simply the generic *shape* of crossing a rounding
boundary in ANY non-bit-identical kernel (attention_fa is deliberately not bit-identical, by its own
gate's design), with no state-carrying mechanism at all — in which case R2 should stop hunting a
mechanism and go straight to its registered fidelity gate.

**Both were tested directly, same session, same hardware (`metal/attn_fa_confound_test.go`).**

### Test 1 — break the mod-4 confound: sweep prefillLen over 1601, 1602, 1603

Three depths chosen specifically NOT to share 1600/2200's residue. Each run as its own process
(subtests invoked one at a time) after two back-to-back loads inside one process both hit this
machine's memory-fit guard — a real, incidental finding of its own, not the mechanism under test.

| prefillLen | first bad step | first bad absolute position | curNKeys at first bad step |
|---|---|---|---|
| 1600 (§2, for reference) | 2 | 1602 | 1603 |
| 1601 | **1** | **1602** | 1603 |
| 1602 | **0** | **1602** | 1603 |
| 1603 | 0 (only step tested; 1602 never visited as a decode step — it's already in the prefill) | 1603 | 1604 |
| 2200 (§2, for reference) | 2 | 2202 | 2203 |

**This falsifies the call-count conclusion outright.** If the trigger were "the third `ForwardEmb`
call, independent of position," the first-bad STEP would stay at 2 across 1601/1602/1603 — instead
it moves to step 1, then step 0, tracking the absolute position exactly: every one of 1600, 1601,
and 1602 first goes bad at **curNKeys = 1603**, regardless of whether that curNKeys value is reached
on the first, second, or third decode call. Position 1602's *entry into the attended set* — not a
count of calls — is what the four 1600-neighborhood runs have in common.

But it is not simply "absolute position 1602/1603 is poisoned forever once attended," either: the
2200 run's own prefill already has position 1603 in its KV cache from the start (it's deep inside a
2200-token prefill), and that run's first two decode steps (positions 2200, 2201) are still clean —
so whatever is special about "curNKeys reaching 1603" for the 1600-neighborhood is NOT simply about
that fixed absolute position's content being bad; the 2200 run has its own, different threshold
(curNKeys ≈ 2203) that happens to sit at the same **+3 offset from 1600** that 2200 itself sits at
relative to 1600 (2203 − 1603 = 600 = 2200 − 1600). That arithmetic coincidence is recorded, not
explained — it is exactly the kind of detail a real mechanism should eventually account for, and
right now nothing here does.

### Test 2 — control arm: perturb the REFERENCE kernel's scale by 1 float32 ULP

Both arms use the shipped (non-`attention_fa`) kernel only; `r.uScale`'s bit pattern differs by
exactly one ULP (`math.Nextafter32`, 0.0883883461 → 0.0883883536, Δ7.45e-9) between them. If
"clean-then-jump" is the generic shape of any tiny perturbation crossing a rounding boundary
somewhere downstream, this control should show it.

**It does not.** All 5 steps (positions 1600-1604) diverge immediately and uniformly (cosine
0.9990-0.9994, maxAbs 0.72-0.94) — no clean steps at all, let alone two. This is evidence against
the "any perturbation looks like this" hypothesis: a genuine ULP-scale perturbation on this same
model, same depth, produces a qualitatively different signature (immediate, not delayed) from
`attention_fa`'s own divergence. One control, one perturbation site and direction — not proof the
hypothesis is false everywhere, but the direct test it invited came back negative.

### What this changes

**Established, corrected:** the divergence is real and position-linked, but neither to a fixed
absolute position nor to a simple residue class — it tracks something that shifts with prefillLen
in a way only checked at two prefillLen "families" (1600-neighborhood, 2200) so far. **Retracted:**
the "call-count/carried-state, independent of position" conclusion — falsified directly. **Still
standing, unaffected by the retraction:** the ForwardBatch-cannot-dispatch-attention_fa structural
argument two paragraphs above (a separate, sound argument from the kernel's own addressing scheme,
not from the call-count interpretation), the race fix in §1, and the clean kernel/dispatch audit in
§3 below (re-read now with the corrected question: not "why does the third call fail" but "what
does curNKeys crossing ~1603, or ~2203, actually change").

## 3. Full kernel and dispatch source audit — clean

Per the user's explicit choice of next step (cheaper than building a new multi-layer harness): read
`attention_fa` and `attention_fa_combine`'s MSL source (`metal/kernels.go:953-1058`) and every
Go-side site that builds, sizes, or writes into the buffers they use (`metal/model.go`), looking
specifically for anything keyed on a value that could legitimately differ between the first two
calls and the third. Nothing was found. Specifically checked and ruled out:

- **`nSplit`/`G` uniform timing.** Written exactly once per `setPos` call (see §1's fix), before
  any layer's dispatch is encoded, under a fully synchronous `Begin()`/`End()` command buffer with
  no pipelining — there is no window in which a later write could land before an earlier read.
- **`nSplit`'s value itself.** `attnFASplitFor(nKeys, nKV) = clamp((2*14+nKV-1)/nKV, ..., nKeys/32,
  attnFAMaxSplit)`. For this model (nKV=2), `want=14`; the `nKeys/32` cap only binds below
  `nKeys≈448`, far below `attnFADepthFloor=1536` where the kernel can ever engage — so `nSplit=14`
  is provably identical across steps 0, 1, and 2 at both prefill depths tested. Confirmed by direct
  computation, not just by re-trusting the original investigation's debug-print claim.
- **`attnFAPartial` buffer sizing and addressing.** `pStride = G*(hd+2)` is computed identically on
  the Go allocation side (`maxAttnFAPartialElems = max over layers of nKV*G*(hd+2)`) and inside
  both kernels. The buffer is allocated for `attnFAMaxSplit=32` splits but only `nSplit=14` are
  ever written or read in a given call — same 14 on every call, so there is no leftover-from-a-
  different-nSplit staleness possible. `attention_fa`'s write region (`(kvh*nSplit+split)*pStride`)
  and `attention_fa_combine`'s read region use the same uniform-derived `nSplit`, so they cannot
  disagree with each other.
- **`uAttnFAG`/`uAttnFANSplit` aliasing.** Independently allocated (`NewBufferU32(d, 0)` twice at
  `BuildResident`, metal/model.go:1221) — no shared storage with any other uniform buffer that a
  copy-paste field-omission bug could explain.
- **Threadgroup memory (`shm`) staleness.** All 128 threads unconditionally write their own row
  (`shm + tid*stride`) before the barrier and before any thread reads a different row back — the
  combine step never depends on threadgroup memory being pre-zeroed or retained from a prior
  dispatch.
- **Online-softmax identity handling for empty key chunks.** Computed the actual chunk boundaries
  at prefillLen=1600, steps 0 and 2 by hand: `chunkLen = ceil(nWin/14)` is 115 in both cases, and
  the last split's chunk (`chunkStart=1495, chunkEnd=min(1610,nKeys)`) is non-empty at both steps —
  the `m == -INFINITY` empty-chunk path this kernel relies on for correctness is not newly
  triggered at step 2.
- **`hd` consistency.** Hardcoded `128u` inside `attention_fa`, passed as a uniform (`g.uHd`) into
  `attention_fa_combine` — both are 128 for this model at every layer, confirmed via the same `g :=
  L.geom` binding used earlier in the same function for the QK-norm dispatch.

No defect was found anywhere in this pass. Combined with §1 and §2, the three most concrete,
checkable hypotheses this investigation has produced are now all closed without finding the cause:
it is not the encode-ahead race (wrong code path entirely), not data-dependent K/V content
(position-independence proven), and not a dispatch-parameter, buffer-sizing, buffer-aliasing, or
uniform-timing bug anywhere in the code this session could read (full audit, clean).

## 4. What is and isn't established now

**Established:**
- The `execLoop` pipelining race is real, reproduced in isolation, and fixed — genuine hardening,
  independently worth keeping, but not R2's cause.
- The divergence is **position-linked** (§2b): four separately-tested prefillLen values in the
  1600-neighborhood (1600, 1601, 1602, 1603) all first go bad at the same absolute `curNKeys =
  1603`, regardless of which decode call reaches it — not at a fixed call count. **Retracted:**
  the earlier "call-count/state, independent of position" conclusion (§2/§2b) — directly falsified.
- The threshold is NOT a single fixed absolute position either: the 2200 run's own threshold sits
  at `curNKeys ≈ 2203`, a different absolute value, offset from the 1600-neighborhood's 1603 by
  exactly the same 600 that separates the two prefill lengths. Unexplained arithmetic, recorded
  honestly rather than papered over.
- A direct control (perturbing the REFERENCE kernel's own scale by 1 float32 ULP, `attention_fa`
  not involved at all) does NOT reproduce the "clean steps then stable jump" shape — it diverges
  immediately and uniformly instead. Evidence against "any tiny perturbation looks like this,"
  though only one perturbation site/direction was tried.
- `ForwardBatch` cannot itself be invoking `attention_fa` mid-prefill (a structural fact from the
  kernel's own single-query-position addressing scheme, unaffected by the retraction above).
- Every dispatch parameter, buffer size, buffer address computation, and uniform-write timing site
  reachable by reading the code is self-consistent — §3's audit stands, but its own framing
  ("why does the third call fail") was aimed at the wrong question; re-reading it against "what
  does curNKeys crossing ~1603/~2203 change" is the next pass, not yet done.

**Still not established:** the actual mechanism. The corrected empirical picture (position-linked,
but not fixed-absolute-position, with a same-offset relationship between the two tested prefill
families) is a real, narrower target than "third call" was, but nothing here yet explains WHY
curNKeys crossing a specific, prefillLen-relative threshold flips the result. Two prefill "families"
is not enough data to fit the relationship with confidence — a third, unrelated prefillLen (not
1600±3 and not 2200) would test whether the "+3 from 1600, matching the family's own offset from
1600" pattern generalizes or was itself a two-point coincidence.

## 5. File status and recommendation

All from this session, uncommitted as of this record:
- `metal/model.go` — the `setPos`-relocation fix (§1). Real hardening; recommend keeping and
  committing regardless of R2's outcome, with its comment trail already framed honestly (explains
  what it fixes and, via this record, what it doesn't).
- `metal/attn_fa_pipeline_race_test.go` — the isolated race repro. Recommend keeping as a permanent
  regression test for the `execLoop` pipelining pattern; its own doc comment should be checked
  against this record before commit so it doesn't imply (as an earlier draft of the reasoning did)
  that this race explains the real-generation bug.
- `metal/attn_fa_e2e_depth_test.go` — the depth-2200 variant. Recommend keeping as a second,
  permanent repro alongside the original.
- `metal/attn_fa_confound_test.go` (§2b) — the position sweep (subtests, one process per depth —
  necessary on this machine: two back-to-back real-checkpoint loads inside one process hit the
  memory-fit guard even though either alone loads fine) and the ULP control arm. Recommend keeping
  both as permanent diagnostics; their own doc comments already state what each result does and
  does not establish.
- `docs/tasks/red-october.md` — R2's row there needs its own correction pass to match §2b before
  being treated as current; not done as part of this edit.

## 6. Next step, for whoever picks this up

**Do not re-open the call-count hypothesis — §2b closed it directly, with data, not by inference.**
The corrected target is narrower: what does `curNKeys` crossing a threshold near `prefillLen`'s own
"+3 from 1600" offset actually change. Concretely:
- **A third, unrelated prefillLen family** (not near 1600, not near 2200) would test whether the
  "+3 offset scales with prefillLen" pattern §2b found is real or a two-point coincidence — the
  single highest-value next experiment, cheaper than anything below.
- **`GOINFER_ATTNFA_DEBUG=1`'s per-dispatch print, re-read now for curNKeys specifically crossing
  1603/2203** rather than for cross-layer consistency (which is what the original investigation
  used it for) — the debug print already exists and was never read with this question in mind.
- A Metal GPU frame capture / `MTLCaptureManager` trace at the specific curNKeys value that flips,
  to see what the GPU actually does differently on either side of the threshold.
- The kernel-source audit in §3, re-read against "what changes about `attnFASplitFor`'s inputs, or
  any other per-call-recomputed quantity, specifically as `nKeys` crosses ~1603 or ~2203" — not
  re-read for the "third call" framing it was originally checked against.

`TestAttentionFA_endToEndReproduction`, `TestAttentionFA_endToEndReproductionDepth2200`, and now
`TestAttentionFA_positionSweep`/`TestAttentionFA_ulpPerturbationControl` are the known-good, fully
deterministic repros — start from them, not from rebuilding new ones.
