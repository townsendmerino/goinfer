# Release queue

Release gates, tagging, versioning, the v1.0 criteria, capability claims, and the consumer surface. Anything whose success criterion is **a decision recorded before shipping**. If the question is *can we tag*, it belongs here.

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

**R3 · `.giw` claims coverage it does not have — CLOSED 2026-08-19 by v6 (complete coverage) — one SILENT wrong-answer path, one broken artifact**
— `linux`, **refusals landed 2026-08-19; format extension (v6) still open**

Asked while sizing the v1.0 gate's `.giw` promise ("how safe is it?"). The frame is healthier than
the gate doc implies — but the CLAIM was wrong, and two families proved it. **Measured, not read:**

| family | `canSerialize` | round-trip result |
|---|---|---|
| `gpt_oss` | **accepted** | `AttnSinks 8 -> 0`, **bundle LOADED CLEAN** — silent |
| `laguna` | **accepted** | writer emitted; **reader rejected**: `layer 1 QProj: 128 rows, arch expects 64` |

`gpt-oss` is the dangerous one: per-head attention sinks are per-layer state the writer has no
field for, so `prequant` (or `serve --stream-weights`) yields a CRC-valid, sink-free bundle that
generates confidently wrong text with no error anywhere. Laguna fails loud at load — per-layer
query heads and the `GProj` output gate are likewise unstored — but a bundle that cannot be loaded
is still a broken artifact produced without a warning at write time.

**`TestCanSerialize_refusesUnrepresentable` was ENFORCING the false claim.** Its `refused` map
listed only MLA / Mamba-2 / Llama-4, and its second branch errors when `canSerialize` refuses
anything *not* in the map — so the gate would have blocked the fix. Both the function and the map
are now updated together.

**Landed:** both families refused, with the measurements in the code comments so the next reader
does not have to re-derive them.

**DONE — .giw v6 (2026-08-19): every registered family is representable, `canSerialize` is EMPTY.**
The tail writes `GProj` (Laguna), `AttnSinks` + per-expert biases (gpt-oss), the `*mlaWeights`
sub-struct (DeepSeek/Kimi) and the `*mamba2Weights` sub-struct (Granite/Nemotron); Llama-4 needed
nothing, its refusal was caution. The reader's last uniform-geometry assumption — `qDim` from the
model-level `NumHeads` — now uses `headsAt(i)`, which is what rejected a correctly-written Laguna
bundle. `GProj` is validated at either legal width (per-head or per-element), since the tensor and
not the config decides which ships.

**The tail is UNCONDITIONAL, not arch-gated like the gemma4 one.** Arch-gating is precisely how
gpt-oss's sinks went missing — it is one more place to remember — and the cost of always writing is
a few zero lengths per layer on families that do not use the fields.

**Verified**: the census now covers **21 fixtures** across both testdata roots, field-by-field AND
by decode identity (greedy 6 tokens through each family's forward, requiring identical tokens).
Mutation-verified twice over: perturbing ONE float of the MLA output projection did NOT fire (too
small to flip an argmax), zeroing the whole projection failed `deepseek-tiny` and `kimi-tiny` with
`CHANGED THE DECODE`. That is the honest sensitivity — structural damage, not precision drift — and
it was confirmed by printing the actual decoded tokens rather than trusting a pass.

**Superseded (kept for the record):** representing them properly — `AttnSinks` per layer, `GProj`, and per-layer query
heads (which also means the reader's uniform `NumHeads*HeadDim` validation has to learn `headsAt`).
That is a format bump, not a patch. **Nothing here is hard**: gpt-oss needed ONE `[]float32` field
written. The gap was never difficulty, it was that adding a field to `LayerWeights` and forgetting
`serialize.go` produces no error anywhere.

**So the recurrence guard landed first** (`TestSerializeCensus_noSilentFieldDrop`): it reflects over
EVERY `LayerWeights` field, round-trips each committed tiny fixture, and fails when a field that was
populated comes back empty — whatever family introduced it and whoever forgot it. Refused families
skip (refusing is a correct answer; dropping is not). **Mutation-verified**: making the reader
discard `PreAttnNorm` produces `SILENT DROP: layer 0 field PreAttnNorm was populated (64) and is
empty after a .giw round-trip`. It also fails a family that `canSerialize` accepts but the reader
cannot load — the Laguna shape.

Order matters here: adding the two fields without this test would fix today's two families and
leave the next one to be found by a user. `canSerialize` is a blocklist maintained by memory, and
memory is what failed.

**What this means for the v1.0 §3 promise**, which was the question:

- The **frame already reads back**: outer `GINFB` v1+v2, inner `GINFW` `giwMinReadV=3 .. giwVersion=5`.
  Every bump so far has been strictly ADDITIVE (v2 hybrid tail, v3 RouterBias, v4 gemma4 tail, v5
  quant label), and older bundles keep loading.
- **Bump rate: 4 bumps, none since v0.10.2** — through v0.11-v0.14, while six families landed. But
  that stability is partly BECAUSE unrepresentable families are refused rather than represented, so
  it overstates the format's reach.
- Therefore the honest promise is **documented rebuild-on-minor with best-effort read-back to N-2**,
  NOT forward-compat. A stale or unsupported bundle must fail LOUD (it does: magic/version/CRC
  checks plus the arch-shape validation that caught Laguna) and be rebuildable from source.
- **State the caveat with it:** rebuild requires the SOURCE checkpoint. A `.giw` is a derived cache,
  and anyone keeping only the bundle (they are 8-26 GB; it is tempting) cannot regenerate it. That
  is the sentence the README and the code comment should share.


**R2 · The v0.14.0 sweep: 47/48 gates PASS, one BLOCKER — a required gate DID NOT RUN, caused by my
own diagnostics** — `linux`, **fixed 2026-08-19, re-run owed**

`scripts/parity_sweep.sh` at `94c3ffd`: every gate green except

```
qwen3.6-weightdiff   TestQwen35GGUF_weightDiff   ⛔ DID NOT RUN (blocker)
```

Not a numerics failure. Phase 2 (`-tags realckpt -run 'Qwen35|Real_gate'`, **120m timeout**) ran out
of time *inside* another test:

```
panic: test timed out after 2h0m0s
        TestQwen35GGUF_routeFlipAtOutlier (28m47s)
```

**The budget was eaten by the two B13 DIAGNOSTICS added earlier the same day** —
`locateDivergence` (710 s) + `routeFlipAtOutlier` (1727 s) ≈ **41 minutes** — both of which match
the sweep's `Qwen35` pattern by name. So instruments that deliberately assert almost nothing
displaced `weightDiff`, a required 40-SECOND gate, off the end of the run. By the sweep's own rule a
gate that did not run is a blocker, which is the correct call: "it would have passed" is not
evidence.

**Fixed**: both diagnostics now skip unless `GOINFER_DIAG=1`. Gated by ENV rather than renamed
deliberately — Go's `-run` matches substrings, so excluding them by name would mean stripping
`Qwen35` from a qwen3.5 test, trading discoverability for a scheduling workaround. The sweep now
reports them as honest SKIPs and gets its 41 minutes back.

**Owed**: re-run phase 2 (~50 min without the diagnostics) so `weightDiff` actually executes. The
full phase 1 (340 PASS / 0 FAIL / 12 SKIP, 87.6 min) does not need repeating — nothing in this fix
touches it.

**The lesson generalises past this test**: a diagnostic and a gate look identical to a `-run`
pattern. If a probe is added to a family whose name the release sweep matches, it joins the release
ritual silently — and the first symptom is not "the sweep is slower", it is "a different, required
gate stopped running".

**AND THE MARGIN IS STILL THIN, so reclaiming the 41 minutes is not the whole fix.** Measured on the
re-run: even with both diagnostics gated, phase 2 needs **~95 of its 120-minute budget**
(`gate2FullModel` ~17 min + `GGUF_gate` ~14 min + `vsSafetensors` ~31 min + the rest). Its single
longest gate is 31 minutes, so ~25 minutes of headroom is about one slow checkpoint — and the
failure mode is not a clean timeout error against the slow test, it is a DIFFERENT, faster gate
never running. Raise phase 2's timeout (or split it per-gate so one long test cannot starve the
others) before the next family lands a real-model gate.


**R1 · The v0.14.0 pre-tag CUDA gate is RED — two failures, classified, neither caused by this
release** — `linux`, **decision owed before the tag**

`scripts/gpu_gate.sh` at `f7d8dd9`: **9 groups declared / 9 reported, 9 pass, 1 skip, 2 fail** →
`FAIL — cuda on Linux @ f7d8dd9. Do not tag.` Both failures were run down rather than retried, and
they are different animals.

**(a) `TestMoERouteDemandThreshold` — DETERMINISTIC and PRE-EXISTING. It fails identically at the
`v0.13.0` TAG**, same bytes: FAIL at 139,460,608 / PASS at 141,557,760 against a pin of
287,506,432 / 289,603,584. So it is not a regression from anything in this release.

Following the pin's OWN instruction ("check the identity first: if floor + residual still equals
the demand, the components are what moved and this pin is downstream of them"):

| component | pinned | measured now |
|---|---|---|
| `moe_route` residual | 138,412,032 | **138,412,032 — held exactly** |
| device-wide floor | 151,191,552 | **absent from the measurement** |
| demand (= floor + residual) | 289,603,584 | 141,557,760 (= residual + 3 MiB) |

The residual is bit-for-bit what it was; the FLOOR is not being charged to the launch. It is not a
test-ordering artifact — it reproduces with the test run alone in its own process — and it is not
the parent's device handle, which the pin commit (`ab5d6e9`, 2026-08-13) already created. That
leaves an environmental change on this box since the pin was derived.

**The move is in the SAFE direction and the margin still clears**: `slotMarginBytes 402,653,184 >=
peak demand 141,557,760, clear by 261,095,424 B (249.0 MiB)`. Lower measured demand makes the
33-slot cap *more* conservative, not less.

**What it must NOT get is a nudged pin.** The test says so itself — "if it has moved, A1/A5/A7/A9
need re-deriving, not editing" — and it is right: the pin is downstream of a derivation, so editing
it to match would erase the only signal that the derivation's basis moved. Re-deriving A9 is real
work and does not belong in a release window.

**(b) `TestRealE2EDecodeThroughput` — INTERMITTENT, and that is the more interesting one.** In the
gate's heavy tier it failed the teacher-forced argmax guard: 2/5 exact, hard fails at **28.835%**
and **7.109%** margins — *not* near-ties, so not float noise. It then passed:

- run alone (4/5 exact, 0 hard fails);
- run after its immediate predecessors (`TestPrefillTTFT`, every `TestPrefillPath_*`);
- **and on a full re-run of the identical heavy tier — 2241 s, ZERO failures.**

So one occurrence in ~200 tests, not reproduced. **"Passed on re-run" does not clear it**: an
argmax divergence at a 28.8% margin is a real anomaly whichever way it arrived — cross-test device
state, or an intermittent fault in the resident decode path. It needs a characterization run
(repeat the heavy tier N times and count), not a shrug.

**A correction belongs in the record**: this was first attributed to the aikit v1.19.0→v1.21.0 bump
landed the same day. That was wrong — `HEAD` with the bump passes in isolation, and the comparison
that produced the claim set a sweep-run failure against isolated runs, which is not a comparison at
all.

**(b) RESOLVED 2026-08-19 — RETIRED TO A BENCHMARK, not fixed and not deleted.**
`TestRealE2EDecodeThroughput` is now `BenchmarkRealE2EDecode`. The reasoning, in order:

1. **It could not indict the product.** In the very run where its hand-rolled sequence diverged at
   28.835%, `TestBackendResidentWired` — the same argmax comparison against the same CPU reference,
   on the PRODUCTION resident path — passed at 7/8 exact, worst near-tie 0.087%, zero hard fails.
   The anomaly was in the test's duplicate forward.
2. **A test whose failure cannot indict the product does not belong in a correctness gate.** It
   spends the gate's credibility, and a red meaning "the harness wobbled" teaches people to re-run
   reds — which is how a real red gets re-run too.
3. **Deleting it would have thrown away the instrument.** The throughput headline and the launch
   decomposition (366 launches/token, CPU issue vs GPU exec, GEMV vs glue bandwidth) are the reason
   the file exists, and nothing else measures them. A benchmark keeps them: it runs only under
   `-bench`, so it is out of `go test ./...` and out of `gpu_gate.sh`.
4. **Its correctness check is KEPT and still fails the run.** A throughput number for a forward that
   does not reproduce production is worse than no number — that check is what caught this file's
   last real drift (`rope_kv` missing `rhalf`). Its failures now cost a benchmark run, not a tag.

Verified after the move: `-bench` reports 200.2 tok/s / 366 launches per token, and
`go test -run TestRealE2E` reports "no tests to run". Coverage lost: none — token identity was
already `TestBackendResidentWired`'s, throughput on the production path is `TestProdThroughput`'s.

**(a) RESOLVED 2026-08-19 — RE-DERIVED, not edited.** The old assertion pinned a SUM whose value
depends on a precondition the test does not control: whether another CUDA context is alive on the
device while the child launches. Measured both ways on one box, one commit, minutes apart:

| child driven by | `leave` | freeBefore | launch |
|---|---|---|---|
| this test (parent holds a context) | 141,819,904 | 141,557,760 | **ok** |
| a bare shell (nothing else on the card) | 141,819,904 | 140,050,432 | **fails** |
| a bare shell | 289,603,584 | 288,948,224 | **ok** |

So the requirement differs by exactly the device-wide reserve the FIRST context pays:

- **WARM** (another context alive) → demand = residual ≈ 138.4 MiB
- **COLD** (this is the first) → demand = floor + residual = **289,603,584** — the old pin

The residual held **bit-for-bit** (138,412,032) in every measurement and the floor is the same
151,191,552 `TestAllocFloor` reports. Nothing about the kernel or the machine moved: the pin
recorded the COLD sum, and the test now runs WARM because `e39c0fb` moved it into the drain group's
own process, where the preceding A10 tests leave a context on the card. **A pin that flips with its
neighbours is measuring the neighbourhood.**

The fix pins the COMPONENTS — each already gated by its own test — and asserts the IDENTITY against
the regime actually observed, to a 6 MiB window (the bisection quantum plus the launch's transient;
the measured figure moved 141,557,760 → 142,147,584 between two runs, which is why byte-exactness
was the wrong instrument). **Mutation-verified**: a wrong residual constant makes it report
`demand identity BROKEN` and fail.

The safety check is now **regime-independent**: it asserts `slotMarginBytes` against
`max(measured, cold)` rather than the measured figure alone, because the margin must cover a box
where goinfer's is the first context on the card. It clears by 107.8 MiB against the cold demand.

**Both gate failures are now resolved.** Re-run the CUDA gate before tagging.


**C1 · Drain fix — CUDA verification** — `linux`

Prompt already written. The admin-unload drain (`588052b`) is verified on Metal: unload freed
325 MB and reported `freed:true`, against 95 MB before. CUDA is the arm that can't run there, and
it's the backend where `Close` does the most — pinned host memory, mapped expert stacks, CUDA
graphs, then `ReleaseObjects` routed through the pinned executor via `reqCh`/`ackCh`. That teardown
meeting the drain is the untested interaction.

Four parts: VRAM reclaim across load/generate/unload/reload; the `preamblePark` regression test
under `-tags goinfer_testhooks` against CUDA; the straggler case if adapters are loadable; and
`--unload-drain-wait`'s 5s default under a real generation.

**C2 · Out-of-tree consumer audit against v0.11.0** — fresh session, **no repo access**

Prompt already written. More valuable now than when drafted, because the README has since acquired
many specific provenanced claims — deployment size, JIT timings, the depth curve, the configuration
sweep, request-body caps, unload semantics. The audit's Tier 1 is a claim-by-claim check from
outside, which is the only instrument that tests claim-discipline rule 7 ("a claim nobody can
reproduce from the public documents is not shipped") **from the position the rule is about**.

Must run blind: no clone, no source, no test suite. Carry the known-findings list so nothing is
rediscovered.

**C3 · Metal consumer window** — `mac`

The largest completely uncovered surface. Nothing has tested Metal from outside. Claims attached:
cgo-free with no Xcode, 73.6 tok/s, 0.96×/0.74× against Ollama-Metal, bit-identity within machine
and OS.

Sharpened by two things found since: `TestMetalSnapshotGolden` is a §4 gate that **cannot fail** (it
drives `Forward`/`ForwardArgmax`, which apply no embed scale, where production applies √hidden), and
the Metal device suite **doesn't run in GitHub CI at all** — the runner's paravirtual objc layer
SIGSEGVs inside purego. So Metal's entire device coverage is one manual box run, behind an
unfalsifiable gate.

The tautological-gate shape was found on CUDA today (four graph tests comparing graphs-on against
graphs-off without asserting graphs were admitted). **The same shape is plausibly live on Metal and
nothing would say so.**

**DEFERRED BY CHOICE (2026-08-12) — auto-pickup, trigger pinned. Not sunk: deferral is the decision.**

- **Trigger = the next goinfer RELEASE TAG THAT CARRIES AN AIKIT BUMP.** No version floor: the
  condition is the *bump*, and a version number standing in for it is a literal that drifts. **Same
  shape corrected twice in one day** — "the aikit **v1.17.0** bump" was loosened to *any* aikit bump
  this morning when v1.17.1 shipped hours later, and "**≥ v0.13.0**" is the same substitution one
  level up (it happens to fire correctly for v0.13.0, so this is hygiene, not a fix). B7's constant
  class. `aikit` and/or
  `aikit/gpu` increased against the previous release tag, whatever the version. **Was written as "the
  aikit v1.17.0 bump" and that literal drifted within hours**: aikit shipped v1.17.1 the same day, so
  the trigger named a version `main` was no longer on. RELEASING.md's copy was already generic, which
  is the only reason the trigger still fires — the two carriers disagreed and only the durable one was
  right. This is B7's constant shape (a literal restating a value maintained in five `go.mod` files),
  and it is now stated the same way in both places. It is *not*
  the bump commit on `main`. C3 is a public-view consumer evaluation: an external consumer `go get`s a
  released tag, so evaluating main-HEAD-with-the-bump would certify a state **nobody installs**. This
  is the (b) reading, chosen deliberately over the faster (a): "post-bump tag" here means a **release**
  tag, not the bump commit. (v0.12.0 shipped 2026-08-12, so v0.13.0 may be days out — hence the bound.)
- **Bound = 2026-08-26 (14 days).** If no qualifying release tag by then, run C3 anyway against the
  **latest published goinfer tag** and **record the exact dependency set it evaluated** (resolved
  `aikit` + `aikit/gpu` versions), flagged as the bounded fallback. A consumer window against a
  slightly stale set beats one that never runs — an auto-pickup with no bound is indistinguishable
  from forgetting. **The bound is a date with no in-repo reader, so it is carried EXTERNALLY:** Francis
  is arranging a persistent 2026-08-26 reminder from the Cowork side. Do **not** assume the cron or a
  session covers the bound — the cron expires at 7 days, well before it.
- **Why deferred, not dropped:** the attached claims (73.6 tok/s, cgo-free/no-Xcode, 0.96×/0.74× vs
  Ollama-Metal, bit-identity) are version-sensitive and all originate in `aikit/gpu`; running mid-bump
  documents a set superseded within hours and forces a re-run. This surface **sank once already** and
  was first in its batch precisely to prevent that.
- **Carriers, in order of durability:** (1) **`RELEASING.md`** carries the trigger as a release-process
  line — *"if this release carries an aikit bump, C3 runs on macbook-arm64 against this tag; see
  QUEUE.md C3"* — read by whoever cuts the tag, **at the moment it fires**, surviving every session
  ending. This is the actual carrier (and B5's first concrete customer, landed with it). (2) a
  **session-scoped** cron (daily, 7-day cap) runs C3 the moment a qualifying tag appears *while this
  session lives* — a bonus accelerator, not the guarantee. (3) **this entry** is the record. The
  fragile half — needing a session to outlive the wait — is retired: the release process carries it.

**C4 · Soak testing** — either box

Nothing has run longer than minutes. The G1 memory-plateau finding rested on **75 seconds**. Memory
growth, KV cache reuse, session accumulation, fd leaks, thermal behaviour over hours are all
unobserved.

### D. Structural work, sequenced

**D3 · Promote the expert-cache env vars to CLI flags** — `linux`, **re-derived `2d28358`; ready to
rebase**

**This entry's own description was wrong at the source, not stale.** It read as a "parked flag-pair"
with a workaround premise. `BRANCH-NOTE.md` says what it is: **an API-surface promotion** of
env-var-only controls to CLI flags, wired `decoder.Options` → `Model` accessor → CUDA backend,
following the `KVPrecision` pattern rather than adding more `os.Getenv` to the backend. The entry was
mine and it mischaracterised the branch from the beginning — a status sweep would never have caught
it, because the status was right.

**Does it complete the MoE-cache story the release headlines? YES.** `c8b65ba` adds
`--moe-cache-experts` / `--moe-cache-slots` to `serve`, and the README instructs the **env vars in
three places** — including the very section this release rewrites around the cap fix. Shipping the
release without D3 means rewriting that section again next version, for the same feature.

**But its branch predates the fix it completes.** `flag-pair-moe-cache` is based on `7ccec1e` — the
slot-default commit that was **reverted** — and it touches `cuda/backend.go` and `decoder/model.go`,
both of which A5 (`6091e7a`) and A9-FIX (`0103b49`) changed substantially. So this is a **rebase and
re-verify**, not a merge.

**DESIGN READ DONE 2026-08-12 — the reason SURVIVES, but the entry was resting on a stale premise.**

Read from `BRANCH-NOTE.md`, not the diff. The stated intent is *"the env-var-only expert-cache
controls promoted to real CLI flags, wired `decoder.Options` → `Model` accessor → CUDA backend,
following the `KVPrecision` pattern rather than adding more `os.Getenv` to the backend"*. That is an
**API-surface change**, not a workaround.

**And the branch note itself draws the line the question asks about**: *"the user-visible half of this
work — the slot default, which was costing ~3× decode rate — was CUDA-only and landed on `main` at
`7ccec1e`"*. The workaround-shaped part was always a **separate change**. It landed, was reverted, and
was redone correctly as A5. **A5 changed what the default computes; it did not remove a user's reason
to override it.** So the flag pair is the second branch: legitimate explicit override, and it stays.

**What needs re-deriving, per that branch:**

1. **The flag's documented meaning.** Written when the cap could not be trusted, so it read as *how
   you get a working cache*. With A5 it means *request no more than N*, and the cap may still lower
   it — the log line says which.
2. **`BRANCH-NOTE.md`'s own rebase guidance is stale.** It says to expect one conflict in
   `cuda/backend.go` and to *"keep `main`'s comment"* — but that hunk has changed three times since
   (`7ccec1e` reverted at `97ee663`, then A5 `6091e7a`, then A9-FIX `0103b49`). The instruction now
   points at text that no longer exists.
3. **Its freeze paragraph is superseded** — it waits on a lift; the freeze is now a proof requirement
   and the goldens run is the authorisation.

`decoder/model.go` and `decoder/gguf.go` were untouched by A5/A9-FIX/P6/P7, so only `cuda/backend.go`
should conflict.

**REBASED 2026-08-12 (`2d28358`), not yet merged.** One conflict, in `cuda/backend.go`, and it was
not the plumbing conflict the old note predicted:

**the branch defaults the slot request to `nE` — "ask for all, auto-cap" — which is `7ccec1e`'s
reverted behaviour.** Resolved by keeping the accessor (the flag promotion) and **preserving main's
`topK` default**: unset changes nothing. Raising the default is a **separate decision** and main's own
comment states its precondition — *"fixing the margin FIRST and proving it on the 26B"* — which A5
(`6091e7a`) and A7 have now met. It belongs in a change about defaults, not one promoting env vars to
flags. **Queued below as D3b.**

Flag help rewritten for the corrected cap: `--moe-cache-slots` now documents itself as *request at
most this many*, notes that the runtime lowers it and logs what it chose, and no longer describes a
"deliberately greedy" default that is not taken.

**The goldens run is still owed, on a fixture-bearing checkout** — see the finding below. Merging is
the next action.

**D3b · Should the expert-cache default rise above `topK`?** — `linux`, **unblocked, now a real question**

`topK` degenerates to re-fetching every routed expert every token (~5 tok/s against ~17 at 38 slots on
the 26B). It was set there because the cap could not be trusted; A5 fixed the cap and A7 proved it on
the 26B, which is exactly the precondition `cuda/backend.go`'s own comment names. The candidate
default is "request all, let the corrected cap decide" — which on this card lands at 31.

Separate from D3 on purpose: D3 changes the *surface*, this changes *what happens to someone who sets
nothing*. Needs the 26B run and a hit-rate figure at the chosen cap before it lands.

**PRECONDITION READ, 2026-08-12 — and it is NOT met. D3b waits on A10.**

`cuda/backend.go`'s comment says *"raising it again requires fixing the margin FIRST and proving it on
the 26B"*. Two readings are available and the comment's own complaint chooses between them: it faults
the margin for being *"a flat `marginBytes = 384 MB` **described as** covering the greedy-argmax
readback + driver overhead — per-token costs — **while what it must actually leave room for** is
everything the forward allocates AFTER it runs"*.

That is an objection to the margin **not being derived from what it must cover**. So "fixing the
margin" means **deriving** it, not checking it — and the second reading is the intended one:

> **margin derived rather than asserted → A10 blocks D3b, and it waits on the floor being attributed.**

**What we have instead**, and it is genuinely better than when the comment was written, just not the
thing it asks for:

- `slotMarginBytes` is still **402,653,184 by assertion** — the same flat constant.
- **A9-FIX removed the largest unaccounted consumer from the margin's job**, by paying the deferred
  first-launch reservation *before* sizing rather than expecting the margin to absorb it. The
  concrete failure the comment names — capped to 34, then `CUDA_ERROR_OUT_OF_MEMORY` — cannot recur
  that way.
- A gate asserts `slotMarginBytes ≥ measured peak demand` (402,653,184 ≥ 289,013,760, clear by
  113,639,424). **That is a check, not a derivation**: it confirms the constant is big enough today
  on this card, and would confirm it just as happily if the constant had been picked by coin flip.
- **A10's 151,191,552 B floor is unexplained and sits inside that margin** — 37% of it. A derivation
  cannot be written while more than a third of what the margin covers is unattributed.

**THE BLOCKER NOW HAS A ROUTE — read this before treating D3b as indefinitely stalled.** A candidate
derivation exists:

> **margin ≥ reporting gap + peak transient** = 151,191,552 + 137,822,208 = **289,013,760**

against the shipped 402,653,184, which clears it by 113,639,424.

**BASIS — read this rather than the number alone.** The derivation rests on **single-context scoping**,
not on a per-device result. The reserve was measured as **per-context** (a second context costs
107,806,720 B), so the margin is not a device constant in general. It is a constant *for goinfer*
because goinfer creates exactly one context and **cannot create a second** — `cuCtxCreate` is not
bound by gocudrv, and aikit uses only the refcounted primary context.

**Stated precondition: revisit if goinfer ever creates a second CUDA context**, or if a dependency
gains `cuCtxCreate` and something uses it. Until then D3b's blocker is **resolved on that basis**, and
and A10 is now **fully decomposed**, so there is no longer a residue to worry about: the gap is
44,236,800 B once per device plus 106,954,752 B per context, summing exactly. The derivation used the
*measured gap* and did not depend on the decomposition — it is simply better founded for having it.

**So: D3b is unblocked as a question and blocked as a change.** The precondition's second half
("proving it on the 26B") is met — A7 did that. The first half is not. Recorded here so the next
person does not re-decide it; **reopen when A10 is attributed**, not before.

**Historical framing: out of the release, and the first question was not a merge question.**

D3 was designed **while the cap computed the wrong value**. A5 fixed the cap. So before anything:
**does the flag pair still have a reason?**

- **If the flags exist to work around a cap that could not size the cache correctly** — that reason
  is **gone**, and shipping them would document a control whose justification was removed. The item
  **closes** rather than rebases.
- **If they exist for legitimate explicit override** — a smaller cache than the correct cap, chosen
  deliberately — they stay. But the **defaults and the docs were written against the old behaviour**
  and both need re-deriving against the corrected cap.

**A clean rebase would not distinguish those two.** Read the design, not the diff. Scheduled after G2.

**D3 (original) · blocked on the freeze** — superseded

`flag-pair-moe-cache` (`bacc04c`) carries `--moe-cache-experts` and `--moe-cache-slots` as CLI
flags. The `Options` fields and accessors touch `decoder/model.go` and `decoder/gguf.go`, which re-stales 19
families' `deps_hash`. `BRANCH-NOTE.md` records the pickup steps and the instruction that matters:
**run the goldens, do not refresh `deps_hash` to quiet the gate**.

Precedent exists for a goldens-gated refresh on exactly this shape: **`ca29d6c`**, where making the
resident context cap configuration-derived added `Options` plumbing to **`decoder/model.go` and
`decoder/gguf.go` — the same two files this branch touches** — and refreshed behind 19 goldens.
(This line previously cited `9e5f8fa`, which touches the manifest not at all; the "re-staled
`decoder/weights.go`" detail was fabricated with it — none of the nine real refreshes touches that
file.) It was deliberately not spent on ergonomics.

### E. Release and claims

**E1 · v1.0 gate as written criteria** — `linux`

**The parity backfill lands as `v0.14.0`** — moved from `v0.13.0`, **decided 2026-08-12 by
Francis**, which is the **second** move of this reservation and the history is kept deliberately:

| target | moved because |
|---|---|
| `v0.12.0` | that number was taken by the CUDA expert-cache campaign |
| `v0.13.0` | *(superseded, 2026-08-12)* |
| **`v0.14.0`** | **current** — v0.13.0 is being cut for the aikit bump + D3's flag promotion |

**The reason, recorded because it is the useful part.** `v0.13.0` is the honest number for what it
carries: **D3's `--moe-cache-experts` / `--moe-cache-slots` promotion is new user-visible CLI
surface**, and a minor is what that content warrants. The backfill reservation had been attached to
a **number** rather than to a **plan** — and reservations attach to plans. Moving the reservation is
a **smaller correction than numbering a release to satisfy a bookkeeping artifact**, which is what
holding v0.13.0 for the backfill would have been.

*(Same shape as B7's constant class, one level up: a plan pinned to a literal drifts when the
literal is needed for something else. The remedy there was to derive the value; here it is to attach
the reservation to the work rather than to the number.)*

**E2's obligation is unchanged in substance — only its target release moves.** The four families
still carry `validated_at: null`; see E2.

v1.0 gets its own gate
requiring parity coverage complete, the verification machinery sound, the loader and
architecture-descriptor surface **actually frozen** (the docs still say it may change), and a clean
out-of-tree audit against the release candidate.

Write that as a checklist so 1.0 is a decision against criteria rather than a feeling.

**The v1.0 gate checklist (draft — the point of E1):**
- [x] Parity coverage complete (E2's four `validated_at: null` families resolved: T3 or demoted to experimental).
  **DONE 2026-08-15 `c3e43c8`** — all four cleared at T3, none demoted; the staleness tripwire now
  enforces 23/23 families. Five rows remain `experimental` (`glm4_moe`, `mixtral`, `qwen2_5_vl`,
  `qwen2_moe`, `llama4_text` — all `tiny-golden`), which is a separate line: they are honestly
  labelled and excluded from the supported count, not `pending`.
- [ ] Verification machinery sound (the gates run and can fail; skip census clean at the freeze).
- [ ] Loader + architecture-descriptor surface **actually frozen** (docs stop saying it may change).
- [ ] Clean out-of-tree audit against the release candidate (C-group consumer window).
- [ ] **The repo contains no Python** — all analysis in Go tests; shell minimized to process
  orchestration. **Decided 2026-08-12 by Francis.** Inventory, ranking, acceptance criteria and the
  reference-tensor carve-out are in **E7**. (The reference-tensor / `pin_*` generation is *excluded*
  from this line — blocked on Francis's torch-replacement research; see E7 item 7.)

**E2 · The four per-family demotion judgments** — `linux` — **DONE 2026-08-15 (`c3e43c8` code,
manifest in the follow-up), and not one demotion among them**

`gpt2`, `granitemoehybrid`, `kimi_k2`, `nemotron_h` carried `validated_at: null` and were the same four
the `deps_hash` tripwire did not enforce — so 19/23 tracked both the backfill's progress and the
tripwire's coverage, and clearing it closed both. **The tripwire now enforces 23/23.**

| family | method | measured (linux-62gb) |
|---|---|---|
| `gpt2` | `full-forward-oracle` | HF f32 GPT-2 small: argmax exact, cosine 0.99999999999994 |
| `granitemoehybrid` | `real-model-oracle` | HF bf16 Granite-4.0-H-Tiny, int8 resident: argmax exact, cosine 0.995662, 6/6 continuation exact |
| `nemotron_h` | `real-model-oracle` | HF bf16 Nemotron-Nano-9B-v2, int8 resident: argmax exact, cosine 0.995737, 6/6 continuation exact |
| `kimi_k2` | `shared-path (via deepseek_v3)` | no run — same `forward_deepseek.go`, same `deps_hash`; config-delta covered by `TestKimi_*` |

**The finding, which is bigger than the entry.** The campaign doc promised "validation + recording,
NOT engineering". That was true for `gpt2` (run the existing gate) and `kimi_k2` (a one-line row).
It was **false for both hybrids, and false in the same way**: neither could LOAD its released
checkpoint. Granite demanded transformers ≥5.10's `rope_parameters` where IBM ships 4.56-era flat
`rope_theta`, then roped a model whose config says `position_embedding_type: "nope"` (measured:
roped ⇒ f32 cosine 0.9936 + wrong continuation; NoPE ⇒ 0.9995 + exact — and the GGUF path had it
too). Nemotron-H reads `layers_block_type` where NVIDIA ships `hybrid_override_pattern`, and
`backbone.embedding.weight` where the release says `backbone.embeddings.weight`.

**Both fixtures were built by instantiating a config on the INSTALLED transformers**, so each
encoded that version's spelling and neither could disagree with the loader. A tiny golden cannot
catch a released-checkpoint schema — which is the argument for T3 stated more sharply than the
policy currently states it, and it generalizes to every family whose T1 fixture is generated
rather than downloaded.

**The demotion rule did the work it was written for.** "Unfinished does not qualify" is exactly
what a two-line loader gap is, so the honest reading forced fixing over demoting — the cheaper
path (two `experimental` rows) would have permanently hollowed the tier to save an afternoon.

**Retargeted to `v0.14.0`** (2026-08-12, with E1's reservation — substance unchanged, target only).

Rule: every family claimed as supported at v1.0 has a current T3 row; families that can't get one go
experimental. **Honesty test per family — would you move it to experimental if no release were
pending?** Structural reasons qualify (no reference, fixture size, licence). "Unfinished" does not;
demoting unfinished work to clear a release hollows out the tier permanently.

**E3 · Freeze re-declaration** — `linux`, **inventory taken and the condition drafted; see below**

**THE FREEZE-BLOCKED INVENTORY, read rather than grepped** (21 frozen paths, from
`testdata/parity_manifest.json`'s `shared_sets`):

| column | item | what blocks it |
|---|---|---|
| **freeze-only** | **D3** the parked flag-pair | `Options` fields touch `decoder/model.go` + `decoder/gguf.go`; re-stales 19 families |
| **freeze-only** | **G2** `go fix` modernizers | re-stales the manifest wholesale |
| freeze **plus other** | **P1** KV re-gather / V re-transpose | freeze **+** a new aikit row-pitch API **+** E6's deferred aikit release |

Everything else touching a frozen path has **landed** (P6 `eea7f29`, P7 `91f359f`) or only references
those paths as instances (B6, P8 — `decoder/sampler_chunked.go` is not in the manifest).

**So the freeze-only column is TWO items — and that is the answer to what an unfreeze buys.** It is
smaller than it looks, because both are landable *today* under the goldens exception, exactly as P6
and P7 were: the cost is a ~33-golden run, not a blocked queue.

**THE UNFREEZE CONDITION, drafted as a capability rather than a version number:**

> The core unfreezes when a change to a frozen path receives numeric proof across the **loader** and
> **quantization** axes it can affect, demonstrated by a gate that **prints its composition**.

**What remains unmet: nothing.** Checked against the axes, not against a summary:

| axis | release gate | goldens refresh (the freeze-exception path) |
|---|---|---|
| quantization | f32, int4, int8, int8int8 | f32, int4, int8, int8int8 |
| loader | safetensors, gguf | safetensors, gguf |

Both print their composition (`scripts/sweep_composition.py`, and the refresh's own
"33 passed / 14 quantized" line). **The loader axis was the open question and it is covered** — but
only since this turn, and only because the GGUF parity gates entered the selector: before `f9d5d07`
the refresh was safetensors-only on loader as well as f32-only on quant.

**THE FREEZE, RE-DECLARED AS A PROOF REQUIREMENT.** It has functioned as one all day — every
frozen-path change that landed ran the goldens, and none was refused:

> **Changes to paths covered by `testdata/parity_manifest.json` require a goldens run whose axis
> composition is printed with the result. No version gate, no per-change exception.**

**Decider: Francis. Declared 2026-08-12.** Recorded with an author because a rule with none drifts
back into habit.

**Justifying inventory:** the freeze-only column is **D3** and **G2**, both landable under this rule
today; **P1** is blocked on the aikit row-pitch API and E6 independently. **Lifting the freeze as a
freeze buys nothing the rule does not.**

**THE AXES, and why ARCHITECTURE is excluded — stated, not left silent.** The condition names
**loader** and **quantization**. It does not name architecture, and the reason is measured rather
than assumed: arm64 contracts `x*y+z` at **85 decoder sites** where amd64's baseline contracts none,
and the FMA campaign measured **114,431× minimum headroom with no argmax flip**. A separate arm64
run is therefore very likely unnecessary, and 18 of 23 manifest rows are `linux-62gb` anyway.

**The exception, in the same breath:** that headroom was measured **for the code as written**. A
change that **rewrites expressions** rather than allocations puts it back in scope, because the
measurement does not survive the expressions changing. **G2 is exactly that change class** — which is
why it gets the check below rather than a wave-through.

The `6edd1ca` freeze remains in force; tagging on top of it touches no core numerics and does not
lift it. But it needs re-declaring in a **live document** with scope, an explicit lift condition,
and who decides — rather than being reconstructed from a commit several tags back.

Enforced scope, now quantified: **19 of 23 families, `decoder/` surface only, zero GPU coverage.**
No `cuda/` file appears in the manifest at all.

And answer, rather than leave as an absence: **should `cuda/` files be in the parity manifest**, or
are the resident parity gates the right home for that guarantee with the manifest deliberately
CPU-only? Note that until B2/`scripts/gpu_gate.sh` ran the parity gates, GPU forward numerics had no
enforced signal in the release gate — not a staleness tripwire, not a parity assertion.

**E4 · `scripts/bench_compare.sh` — fix or retire** — `linux`, **status unconfirmed**

It measures goinfer with in-process Go benchmarks and never drives the peer, which is what made the
476/268 headline divide a kernel throughput by an end-to-end one. The README's false-rigor sentence
is gone, but **if the committed artifact still measures the two sides differently the gap reopens
the next time it runs.** Either make it produce a defensible server-to-server comparison, or remove
it and record that peer figures are measured manually with the procedure written down.

**E5 · Promo drafts** — Francis / Claude

Blocked on nothing now. They need **rebuilding, not editing**: written for v0.9.0, quoting withdrawn
476/268-era figures and the pre-fix peer table, carrying the 26B claim without its configuration,
and predating the `top_k` guidance and the §5 bit-identity correction. Claude holds them and will
rebuild against current numbers on request.

**E6 · aikit release** — `linux` or `mac` — **SUPERSEDED BY EVENTS 2026-08-12, and the deferral was
right on its own terms**

aikit cut `v1.17.0` / `gpu/v0.28.0` (`ada417e`), and goinfer is on it (`f33fcaf`). E6 is closed by the
release happening, not by the argument below being withdrawn — and the release **satisfied E6's own
criterion** rather than overriding it. The reason a consumer can receive is `linalg.MatmulBTW8A8Pre`
gaining 8-column blocking: goinfer never calls it directly, but `MatmulBTW8A8Into` now delegates to
it, so every W8A8 decode matmul goes through changed code. That is a consumer-visible change. The
gates and CI that E6 declined to release *for* rode along as line items, which is exactly the shape
E6 predicted for them.

**The FMA fix was never the pending part, and the bump re-confirmed it.** `be049df` is an ancestor of
`gpu/v0.27.0`, so goinfer had already been running that PTX; across `gpu/v0.27.0..gpu/v0.28.0` the
quantized GEMV PTX is **byte-identical** and `gpu/gemv_quant.cu` changes by three lines, all comment.
E6's read of its own diff was accurate.

**A measurement trap found while checking this, worth more than the closure.** `git diff
v1.16.0..v1.17.0 -- gpu/` reports the quantized GEMV PTX changed by 72 lines. It is a **misleading
comparison**, and it looks authoritative. `gpu/` is a nested module with its **own tag series**, and
the two series do not track: `v1.17.0` and `gpu/v0.28.0` are the same commit, but `v1.16.0`
(`a79303e`) and `gpu/v0.27.0` (`4642b7c`) are not. Diffing a nested module across the *parent's* tags
therefore spans commits the consumer already had — here it re-reports `be049df`, weeks-old and
already shipped, as if the pending bump introduced it. **Diff a nested module across ITS OWN tags.**
This is the Environment class from the measurement-shape list (docs/parity-coverage-policy.md) with a
new instance: not where it ran, but which boundary it was measured across. It was caught only because
72 changed lines contradicted a claim already written down — an expectation, not an instrument. There
is no gate for this; the countermeasure is the rule in bold.

The original reasoning, kept because the rule in it outlives the decision:

**Deliberately not cut.** `be049df` (aikit: *gpu(gemv): explicit `__fmaf_rn` in the quantized GEMV —
the bit-identity contraction rule*, 2026-08-04, in six tags `gpu/v0.25.0`…`gpu/v0.27.0`) — its FMA fix
is already released (contained in `gpu/v0.25.0`
onward, and goinfer requires `gpu/v0.27.0`), and the unreleased diff is two test files plus
comment-only edits with byte-identical PTX. The rule recorded: **a release needs a reason a consumer
can receive**; test coverage, lint rules and CI are properties of the repository, not of the
artifact. The three gates and the first-ever GPU CI job ride with v1.0, where they are a line item
rather than the whole changelog.

Also open there, deliberately: branch protection is not enabled and `gpu-kernels` is advisory.
`scripts/gpu_gate.sh` plus a `RELEASING.md` gate ritual is the enforcement instead. Revisit at v1.0.

**G2 · Items from the Go-for-AI tooling inventory** — either box

- **PGO** — absent from both repos. goinfer's default build is the pure-Go CPU path and this is a
  performance project; 2–7% is typical. Gate it on the parity goldens, since PGO changes inlining
  and inlining could shift Go's permitted FMA fusion.
- **govulncheck** — VERIFY FIRST: goinfer already runs it in CI and it is green (confirmed 2026-08-12), so this is stale for goinfer. aikit may still lack it; the entry should end up saying which rather than being struck entire. Originally: absent from both. For a project whose pitch is one static binary you `scp` and
  run offline, a reachability-filtered vulnerability statement is part of the deployment claim.
- **Fuzz corpora** — sixteen fuzz targets across the two repos, three committed corpus directories.
  A crasher found once and not committed is found again next year. The audit's hostile-input
  findings should each be seeds.
- ~~**Execution tracing**~~ — **DONE (2026-08-11).** `go tool trace` on `BenchmarkDecode` (0.5B,
  M1 Pro) resolved it: the "~8 ms host cost" / "71% `pthread_cond`" is an **idle-M sampling artifact**,
  not a recoverable cost — serial (zero fork/join) ties parallel in tok/s, the trace's real
  scheduler-wake tax is ~1%/token, and pprof's `pthread_cond` samples are parked idle workers between
  dispatches (a CPU profiler counts them, a wall-clock trace shows them idle). The right tool
  dissolved the question. Confirms the Phase-3b pool-null-result. Writeup: perf-campaign.md
  "Profiling coda". (Lesson: for park/wake questions use `-pprof=sync`/`-pprof=sched` from `trace`,
  not pprof CPU, which can't tell critical-path stall from an idle parked M.)
- **`go fix` modernizers** — one deterministic pass, reviewed as a diff. **CLEARED FOR THE amd64
  GOLDENS RUN ALONE; no arm64 read needed.** Checked before running, in an isolated `git worktree` so
  the real tree could not be touched:

  **21 of the 22 registered analyzers are numerics-inert by construction** — API/idiom migration
  (`any`, `fmtappendf`, `mapsloop`, `newexpr`, `omitzero`, `reflecttypefor`, `slices*`, `stditerators`,
  `strings*`, `testingcontext`, `waitgroup`, `inline`), loop/scope forms (`forvar`, `rangeint`), build
  directives (`buildtag`, `plusbuild`), and one diagnostic-only (`hostport`). None rewrites an
  arithmetic expression.

  **`minmax` is the one that could**, and it is the reason to check rather than assume: it replaces
  `if a > b { m = a } else { m = b }` with `max(a, b)`, and Go's builtins **propagate NaN** where the
  if/else form does not — a real behaviour change in a float path. Its candidates in `decoder/` are
  **7, and every one is integer** dimension or index arithmetic:

      ge := min(gs+group, cols)                                  end := min(g+int4GroupSize, len(row))
      sc := max(moe.SharedIntermediateDim, moe.IntermediateDim)  b := min(32, n)          (x2)
      window := max(len(access)/8, 1)                            workers := min(GOMAXPROCS(0), numChunks)

  **Censused across G2's ACTUAL scope**, not just `decoder/` — 9 candidates, **all integer, zero
  float**:

  | package | candidates | float | integer |
  |---|---|---|---|
  | `decoder` | 7 | 0 | 7 |
  | `cuda` | 2 | 0 | 2 — `cuda/softcap.go`, worker count and chunk bounds |
  | `gpu` | 0 | — | — |
  | `metal` | 0 | — | — |
  | aikit | 0 | — | — |

  **No float `min`/`max` anywhere, and none of the 85 contraction sites is touched.** The headroom
  measurement survives and G2 needs no scope narrowing.

  **`slicessort`'s NaN axis, answered rather than left unasked.** Tie-order was the first question and
  it is not the only one: `slices.Sort` uses `cmp.Less`, which *defines* NaN placement, where a bare
  `<` does not — the same shape as `minmax`, one analyzer over, and the tie-order answer does not
  cover it. Its single site sorts `[]ResidentFeature`, and `ResidentFeature` is a **`string`** type.
  **Strings cannot carry NaN, so the question is moot** — recorded so it is answered.

  **WHAT CLEARED G2 WAS SOURCE ANALYSIS, NOT THE GOLDENS RUN — and the distinction is load-bearing.**
  A float `minmax` rewrite differs from the if/else form only on **NaN**, and NaN paths trigger on
  degenerate inputs while goldens run normal ones. Such a rewrite would have landed **green** and sat
  dormant until a real NaN arrived. That is exercised-but-never-triggered, in the one change class the
  proof requirement above does **not** cover — the requirement proves numerics for the inputs the
  goldens carry, and this class changes behaviour only outside them.

  So do not let "goldens green" later read as the authorization for G2. **The authorization is the
  census**, and it must be re-run if the analyzer set changes.

- **D3** has no expression-rewriting exposure at all (it adds `Options` fields and accessors), so it
  proceeds on the goldens run alone.

### P. Audit findings, 2026-08-12 — nine survived adversarial verification

Eight are below. The **ninth is the Metal `ResidentGreedy` gap**, filed under Struck rather than here
because it is measured net-negative and therefore not work — the count is stated so its absence from
this list reads as a decision rather than as a dropped item.

**Every figure below is a verifier's ESTIMATE, not a measurement.** Written with that word attached
deliberately: these came from reading code, not from running it. Any figure later measured **moves
to the measured-quantities table** in `parity-coverage-policy.md` with machine, method and date, and
stops being an estimate here.


## Draft: contents of the next release

## C1a's discharge is UNVERIFIABLE, not wrong — retraction 2026-08-13

**This entry previously asserted that C1a's discharge record was WRONG. That assertion is
RETRACTED: I compared it against the wrong log.**

C1a is recorded as *"49 pass / 0 skip / 0 fail, exit 0, ALL REQUIRED GATES GREEN"*, **41 min**, in
`679960d`. I read a 2026-08-12 sweep log as that run, found a required-gate skip in it, and concluded
the summary contradicted its own classifier.

**That log contains 142.8 minutes of summed test time.** It cannot be a 41-minute run. It is a
different, fuller sweep from the same day — almost certainly one with the `realckpt` 35B gates
enabled, since a single test in it runs 1735s. **C1a's own log was a `mktemp` file and is gone**, so
its summary can be neither confirmed nor falsified from anything now available.

### What the mechanism question resolves to

Two hypotheses were offered — a **parser gap** (the skip never reached `classify()`) or **sibling
drift** (counting and classification on separate paths). **Both require a demonstrated disagreement,
and there is none**: the disagreement was manufactured by comparing against the wrong run.

On inspection the summary is a **rendering of the verdict, not an independent assertion**:

```
echo "== ${SHA}: $([ "$blockers" -eq 0 ] && echo 'ALL REQUIRED GATES GREEN' \
                                        || echo "${blockers} BLOCKER(S)") =="
```

`blockers` is incremented only inside `check_all`'s `case`, from the same `classify()` result that
prints each row. **One path.** A required-gate skip cannot coexist with a green summary. So previous
green summaries from this gate mean what they say — for the gates their list covered.

### One real discrepancy survives, and it is smaller

**The sweep never prints `"N pass / N skip / N fail"` in any form** — that phrasing does not occur in
the script. C1a's discharge note therefore **quotes two different tools as one result**: the sweep's
`ALL REQUIRED GATES GREEN`, and a pass/skip/fail count from elsewhere (the skip census, or a manual
tally). Neither is wrong alone; the note reads as one verdict from one run, and that is precisely
what became uncheckable when the log disappeared. **The lesson is about the note's construction, not
the gate's.**

**Standing, and unaffected by the retraction:** `TestQwen35GGUF_vsSafetensors` is **not in the
required gate list**, failed on 2026-08-12 (cosine 0.987835 against a 0.998 floor, self-described as
a loader bug) and fails today; C1a's verdict never covered it either way. `TestW4A8DecodeParity` is
in the required list and the int8 `.giw` half it needs has never been built — a classification
problem in the list, not evidence about C1a.

## C1b RESULT (2026-08-13, `611ed28`) — evidence half GREEN, stamp half declined

**The sweep, counted from its own gate table rather than from its summary line:**

| | |
|---|---|
| required gates (`GATES` 39 + `REALCKPT_GATES` 9) | **48** |
| **pass** | **47** |
| coverage gap (`w4a8-int4`, asset never built) | 1 |
| **failures INSIDE the required set** | **0** |

**Standing failures OUTSIDE the required set — the denominator nobody had counted.** 341 tests ran,
4 failed, and **all four are outside the required list**:

| test | status |
|---|---|
| `TestDecodeParityInt4` | ~~standing — red on 2026-08-12 and today~~ **RESOLVED 2026-08-15**: stale golden, not a defect. Bisected to `7deb368` (aikit 1.7.3→1.8.1's W4A8 scale-fold kernel); re-captured on evidence — vs an f32 forward the new path matches 11 leading ids, the pinned one 5. Red since 2026-06-14, not 2026-08-12: the gate was skipping until the asset registry gave it a fallback |
| `TestSerializeWeightsTo_matchesBuffer` | standing — red on 2026-08-12 and today ([[B11]]) |
| `TestQwen35GGUF_vsSafetensors` | standing — red on 2026-08-12 and today; cosine 0.987835 vs a 0.998 floor |
| `TestParityManifest_methodTier` | **not standing** — red only because of the emission; passes on the committed manifest ([[B15]]) |

So **three** genuine standing failures, not one. This is a **denominator, not a campaign** — the
campaign is [[B13]] and it is budgeted for after the release. Nothing here is to be fixed now.

**Update 2026-08-18 (v1.0 gate work): ONE, not two — and the survivor has its diagnosis.**

`TestSerializeWeightsTo_matchesBuffer` is **GREEN** (`streamed 632821831 bytes, byte-identical to
buffered`). [[B11]]'s 2026-08-14 fix resolved it; the entry below and `docs/release-1.0-gate.md`
both still listed it as standing because they were read from queue text rather than re-run. Struck
by re-running it, not by reasoning about it.

`TestQwen35GGUF_vsSafetensors` **reproduces exactly** — min cosine 0.987835 at step 63, mean
0.998114 over 80 teacher-forced steps, the same figure as 2026-08-12, so it is deterministic rather
than flaky. **Three pieces of evidence now say the label "loader bug" is wrong**, and the decision
of what to do about it is Francis's (below):

1. **`TestQwen35GGUF_weightDiff` is clean.** Every transform-bearing tensor in layers 0-3 — the V
   un-tile, the fused q‖gate, `−exp(A_log)`, the norm un-bake, conv1d, q/k_norm — is either
   BIT-EXACT (`cos=1.000000, maxAbs=0`) or sits at a **uniform** `relL2 ≈ 0.0057`, the Q8_0-vs-bf16
   dequant floor. Worst tensor 0.999980. A transform bug craters ONE tensor; nothing stands out.
2. **The divergence curve has no step** (`TestQwen35GGUF_locateDivergence`, new). Residuals
   captured after all 40 layers from both containers on identical input: divergence is already
   present at layer 0 (0.99978) and decays smoothly and NON-MONOTONICALLY to 0.99835 by layer 39,
   final logits 0.99792. The per-layer delta oscillates in BOTH directions (−0.00087 at 16 then
   +0.00030/+0.00031; −0.00192 at 37 then +0.00097/+0.00037). **A localized defect cannot recover** —
   a wrong tensor keeps being wrong at every subsequent layer. Repeated recovery is what a small
   random perturbation looks like as downstream normalization re-aligns it. The biggest single
   drop is ~2.5× the median wobble and is gone two layers later.
3. **The floor is a statistical impossibility as written.** It requires `min` over 80 steps ≥ 0.998
   while the MEAN is 0.998114 — i.e. every step must be at least as good as the average. For a
   MoE, the spread has an obvious mechanism: a ~0.5% weight delta flips borderline top-k router
   choices at a few positions, and a flipped expert moves one step's logits far more than the
   noise floor. That is the same near-tie story the sibling bf16 gate already recorded (argmax
   68/80, divergences rank-2 near-ties).

**MECHANISM MEASURED 2026-08-18 — and it corrected the inference above.** The prediction written
before the run was "flips are rare near-ties at a few positions". **Two of its three clauses are
refuted**, which is why it was measured rather than asserted:

| prediction | result |
|---|---|
| most steps have ZERO flipped layers | **refuted** — only **1 of 80**; 779 of 3200 (step,layer) pairs differ |
| step 63 near the top of the flip ranking | **no** — 16/40 flips, outside the top ten (26…20) |
| worst cosine ⇒ most flips | **weak** — step 74 has 26 flips at cosine 0.998497; step 58 has 4 at 0.996661 |

The one directional signal held: the single flip-free step scores **0.999778** against **0.998093**
mean for flipped steps (n=1, so suggestive only).

**That result exposed a blind spot this probe set had left open: `weightDiff` never compared the
ROUTER.** Pervasive routing divergence with an undiffed router is exactly where a real defect could
hide, so the router was added to it — and **the router is BIT-IDENTICAL** across the two containers
(`cos=1.000000, maxAbs=0, relL2=0` at every sliced layer; llama.cpp keeps `ffn_gate_inp` at f32, so
it takes no Q8_0 rounding at all).

**The corrected, coherent account:** the router is byte-for-byte the same, so the flips come from
its INPUT differing by quant noise. A top-8-of-128 selection over near-tied scores is very
sensitive to a ~0.5% perturbation, so flips are PERVASIVE rather than rare — and each is a
legitimate alternative expert, not a wrong one. It also explains why flip COUNT does not predict
cosine (the flipped expert's weight is what matters, not how many flipped) and why a min-over-80
statistic is the wrong quantity to put a floor on.

**DECISION TAKEN 2026-08-18 (decider: Francis, who chose "prove the mechanism first, then
re-express").** The gate now floors the MEAN (0.997, measured 0.998114) and keeps a measured min
floor (0.980, measured 0.987835) for catastrophic single steps, with the three measurements cited
in the test body. Both bars come from two independent reproductions of the same numbers
(2026-08-12 and 2026-08-18). **The forbidden move — nudging the min floor to 0.987 so the red goes
away — was not made**; the min bar sits ~0.008 BELOW the observed value precisely so it still has
room to catch a real regression, and the mean bar is the one that now carries the systematic-drift
duty. The original wording is preserved below.

**DECISION OWED (superseded by the above; kept for the record) — decider: Francis.** The gate's own doc-comment
says "loader bug (not Q8_0 quant)"; the evidence says the opposite. The honest options are (a)
re-express the gate on the statistic that is stable — a MEAN floor plus an argmax/near-tie check —
with the min floor set from measured evidence, or (b) leave it red and carry it. **What must NOT
happen is the min floor being nudged to 0.987 so the red goes away**, which is the move the v1.0
gate names as forbidden. One further experiment would settle mechanism rather than infer it:
instrument the two containers' router top-k choices at step 63 and show the flip. ~35 min, not yet
run.

**Update 2026-08-15: two, not three.** `TestDecodeParityInt4` was carved out of B13 and closed
(above) because it turned out to have a *decidable* answer — a stale golden pinning a superseded
kernel — where the other two are still open questions. **The carve-out rule that produced that
call:** a standing red leaves the denominator early when triaging it is cheaper than carrying it,
which is a property of the individual failure, not of the campaign's schedule. `TestQwen35GGUF_vsSafetensors`
(cosine 0.987835 vs a 0.998 floor) and `TestSerializeWeightsTo_matchesBuffer` stay in B13; neither
has a reference to decide against the way the int4 gate had an f32 forward.

**Stamp half DECLINED, per the §C1 exception in `RELEASING.md`.** `EMIT_MANIFEST=1` reproduced the
[[B15]] defect exactly: 4× `status: experimental → validated` with `method` left at `tiny-golden`,
and `mellum`'s `real-model-oracle → real-oracle`. The emission was dropped; `TestParityManifest`
passes on the committed manifest. **Commit Y therefore contains no manifest change**, recorded as a
decision rather than a skipped step.


## REQUIRED-LIST CLASSIFICATION SPLIT (2026-08-13)

## REQUIRED-LIST CLASSIFICATION SPLIT (2026-08-13)

`scripts/parity_sweep.sh` now distinguishes **required-and-available** from **required, asset never
created**. `ASSET_NEVER_BUILT` holds the second kind; such a gate reports as a **COVERAGE GAP on its
own line, counted separately, and does not increment `blockers`**.

**Why this is a classification fix and not an excuse.** `TestW4A8DecodeParity` needs a matched
int4+int8 `.giw` pair and **only int4 bundles have ever been produced** — so no invocation, on any
machine, could have made it green. Counting it as a blocker calls the release broken for a coverage
gap at every tag, forever. **A permanent blocker is not a gate; it is a thing people learn to
override, and an override habit is worse than an honest gap.**

**The only correct way off the list is to build the asset.** Mutation-checked both directions: a
listed gate becomes a gap, an unlisted one still blocks.
