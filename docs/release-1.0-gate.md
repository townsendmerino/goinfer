# The v1.0 gate — criteria, evidence, and the run plan

> ## ⏸ DEFERRED 2026-08-19 (decider: Francis): **v1.0 comes after v0.16.0**, not next.
>
> The reason is the honest one — there is too much real work still in front of it. This is not a
> retraction of anything below: the criteria stand, the evidence gathered so far stands (§1 and §2
> are green, §3's tier declaration is signed and machine-checked by apidiff, §4's replace-free line
> was discharged by the real v0.14.0 tag), and the remaining lines stay the definition of done.
>
> **What changes is what the document DOES.** Until this deferral, §7's "explicitly NOT blocking
> v1.0" list also functioned as a scope fence for near-term work — DeltaNet residency, continuous
> batching, further families were all "later" partly because 1.0 was imminent. With 1.0 moved past
> v0.16.0, that fence is gone: those items are ordinary queue work again, prioritised on their own
> merits rather than deferred by a release date.
>
> The gate is picked back up when v0.16.0 is out. Nothing here needs re-deriving at that point
> except the lines that say **VERIFY** and anything whose evidence names a stale commit.


> **This is E1's deliverable** (`queue-release.md` §E1: "write that as a checklist so 1.0 is a
> decision against criteria rather than a feeling"). Close E1 by pointing it at this file.
>
> **How to use it.** v1.0 is tagged when every **REQUIRED** line is checked, and each check names
> its evidence — a test that ran, a sha, a recorded decision. A line is checked against its
> pointer, never from memory. Waiving a REQUIRED line is a strike with a reason and a decider,
> exactly as in the queues — never a silent skip. **Decider: Francis.**
>
> Drafted 2026-08-18 from the queues at HEAD `1c0a9ed` (post-v0.13.0: Qwen3.8, InternLM2/3,
> gpt-oss-Metal all landed). Statuses marked **VERIFY** were read from queue entries, not re-run —
> verify before striking. Queue IDs (C1, C2, B13, …) refer to the four queues under `docs/`.

## 0. The identity criterion — met, recorded here so the question stays closed

The roadmap's rolling v1.0 criterion (2026-06-07) was: *a second hybrid family lands on the
existing deltanet/hybrid-cache shapes without breaking them.* **Granite-4.0-H met it** (roadmap
§"Model freshness" calls it the v1.0 trigger), and the bar has since been over-met from four
directions: `qwen3_next`, `qwen3_5` (Qwen3.8 — the dense member), `gpt_oss`, and `laguna` all
landed on unchanged shapes. The loader/descriptor contract has settled in practice; what remains
below is declaring it (§3), not proving it.

- [x] Second-hybrid criterion met — Granite (v0.8.0), corroborated by four later families.

## 1. Correctness and parity

- [x] **Parity backfill complete** — zero `pending` rows; staleness tripwire enforces every
  family (23/23 at `c3e43c8`, 2026-08-15; the count grows with the registry). Nothing was
  demoted; two real loader bugs were fixed instead (granite NoPE roping, nemotron_h schema).
- [x] **REQUIRED · B13's standing reds resolved or formally reclassified** (2026-08-18; full
  evidence in `docs/queue-release.md`). **Both are closed: one was already green, one was
  reclassified on measured evidence with the decision recorded.**
  - `TestSerializeWeightsTo_matchesBuffer` — **GREEN, struck by re-running it.** B11's 2026-08-14
    fix resolved it; this line listed it as red because it was read from queue text rather than
    re-run. Exactly what this document's own **VERIFY** caveat is for.
  - `TestQwen35GGUF_vsSafetensors` — reproduces deterministically (min 0.987835 @ step 63, mean
    0.998114 over 80 steps). **The evidence contradicts its "loader bug" label**: `weightDiff` is
    clean (every transform bit-exact or at a uniform `relL2 ≈ 0.0057` Q8_0 floor, worst 0.999980),
    and the new `TestQwen35GGUF_locateDivergence` shows divergence entering at layer 0 and decaying
    smoothly and NON-MONOTONICALLY with no step — the per-layer delta recovers repeatedly, which a
    localized defect cannot do. The floor also demanded `min` ≥ the mean over 80 steps, which no
    spread can satisfy.
    **The mechanism was then MEASURED, not inferred** (Francis chose "prove it first"), and it
    corrected the guess: routing flips are not rare near-ties but PERVASIVE — 779 of 3200
    (step,layer) pairs, 79 of 80 steps. That result exposed a blind spot the probes had left:
    `weightDiff` never compared the ROUTER. It does now, and **the router is BIT-IDENTICAL**
    (`maxAbs=0`; llama.cpp keeps `ffn_gate_inp` at f32, so it takes no Q8_0 rounding). With an
    identical router, the flips come from its INPUT differing by quant noise at a top-8-of-128
    decision boundary — each flip a legitimate alternative expert, which is also why flip COUNT
    does not predict cosine.
    **DECIDED (Francis):** the gate floors the MEAN (0.997, measured 0.998114) plus a measured min
    (0.980, measured 0.987835) for catastrophic single steps, with the three measurements cited in
    the test body. **The forbidden move — nudging the min to 0.987 so the red goes away — was not
    made**; the min bar sits ~0.008 below the observed value so it retains room to catch a real
    regression, and the mean now carries the systematic-drift duty. Green on the re-run, with
    numbers identical to six decimals across three independent reproductions.
- [x] **REQUIRED · The w4a8 coverage gap closed by building the asset** (2026-08-18). Both halves
  built from the SAME source GGUF — the 1.5B the test's own docstring names — so "matched" is by
  construction: `prequant -quant int4` and `-quant int8int8` over
  `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf`. **`TestW4A8DecodeParity` passed on first invocation:
  16/16 greedy-token agreement, identical decoded text on both halves.** `ASSET_NEVER_BUILT` is now
  EMPTY and got there the only correct way; the classification machinery stays for the next such
  gate. Both paths are registered in `testdata/assets.json` and the gate reads them through
  `assetPath`, so the sweep preflight and the gate apply one predicate (the registry's own
  `noDirectReads` gate required that, and was right to).
- [x] **REQUIRED · B15 manifest-emission defect fixed** (2026-08-18). Both halves, at the source
  rather than only in the reader: (1) the merge **derives status from the method** instead of
  asserting `validated` — T3 ⇒ validated, anything else ⇒ experimental, which also lets a row
  DEMOTE a family whose evidence was downgraded; (2) `method` is a **closed vocabulary** checked by
  `emitParityRow` itself, and mellum's call site now says `real-model-oracle` rather than the
  one-word-short `real-oracle` that reached the manifest verbatim. The merge loop was extracted to
  `applyParityRows` so the regression test drives the REAL code path, not a copy of it. Two new
  gates, both running in plain CI where the emitter never does:
  `TestParityEmit_methodVocabulary` (a source census over all 18 call sites — this is the one that
  would have caught mellum's typo without a heavy run) and
  `TestParityMerge_noPromotionWithoutMethod` (tiny-golden stays experimental; T3 validates;
  downgrade demotes; `real-oracle` and unknown families are rejected).
- [ ] **REQUIRED · Every registered family has a manifest row at its honest tier at the RC
  commit**, tripwire green at N/N. New since the backfill closed and to confirm on the RC:
  `qwen3_5` (real-checkpoint gate `4d1a7f2`, token-id parity `e3674aa`), the InternLM2/3
  llama-dialect rows, `laguna`. **VERIFY** the regenerated matrix carries them at the RC sha.
  **Pre-verified at HEAD 2026-08-18** (the RE-check at the RC sha is what remains): tripwire green
  at **27/27**, `qwen3_5` = experimental/tiny-golden+coherent, `laguna` and `internlm2` =
  experimental/tiny-golden, `gpt-oss` = validated/real-model-oracle. **`internlm3` has no row of
  its own ON PURPOSE and must not be given one** — it is a registry ALIAS of `llama` (same
  descriptor, dynamic-NTK rope being in-window identity), so it rides llama's row and the
  coverage gate agrees; adding a separate row would claim independent evidence that does not
  exist.
- [ ] **RECOMMENDED · Upgrade four of the five `experimental` rows.** The backfill doc already
  assigned each its method; this is run-and-record: `glm4_moe` → weightDiff + layer-slice
  (`glm4moe_gguf_test.go` *is* the test), `qwen2_5_vl` → full-forward-oracle (Bucket A),
  `mixtral` + `qwen2_moe` → int8-resident-vs-bf16 real-model-oracle (Bucket B, linux-62gb).
  `llama4_text` may honestly remain experimental if no small checkpoint materializes — labelled,
  excluded from the supported count, per the demotion rule.

## 2. Verification machinery — "the gates run and CAN FAIL"

- [x] Freeze re-declared as a proof requirement — goldens run with printed axis composition for
  any frozen-path change (`cda8cfe`, decider Francis, 2026-08-12). Loader axis covered since
  `f9d5d07`.
- [x] Manifest method-tier validation enforced (`TestParityManifest_methodTier`) — `tiny-golden`
  can no longer sit silently in a T3 slot.
- [x] **REQUIRED · Tautological-gate hunt, run as a census** (2026-08-18). Run over `cuda/`,
  `metal/`, `gpu/` and `decoder/` in three passes — every A-on/A-off **env toggle** (all 14 that
  any test flips), every `ResidentForwardForTest()` acquisition, and every
  `residentParity`/`BuildResident` call. **Two real holes found, both on CUDA, both now guarded
  and mutation-verified on the hardware:**

  - `loadG4MoECache` (4 gates: `cacheExpertsBitExact_tiny`/`_scaled`, `cacheReuse_tiny`/`_scaled`)
    asserted residency but never that the **expert cache engaged**. That is not hypothetical:
    `allocSlots` CLEARS `r.cacheExperts` when the first routed layer reports a zero per-expert
    stride (audit R-25), and both arms would then run fully-resident and report
    "DMA'd-into-slots == fully-resident, BIT-IDENTICAL" having compared nothing. Guard added
    (reads the flag AFTER the build); mutating the env to simulate the decline makes it fail with
    that exact message, and the unmutated gates stay green.
  - `TestGreedyFastPathIdentical` asserted residency but not `ResidentGreedy`, which is the
    interface assertion — not the env var — that actually routes decode to the on-device argmax.
    Guarded; green on the 1.5B (token-identical over 24 tokens).

  **Everything else was already sound, including both named suspects.** The four CUDA graph tests
  carry an explicit tautology guard that SKIPS when `admitGraphs` declines under DEFAULT compute
  mode (declining is correct production behaviour; reporting it as a pass is not) — the **VERIFY**
  is discharged. `TestKVOnlyPrefill_byteIdentical_tiny` asserts its `ResidentPrefillKV`
  precondition. Metal's `residentParity` helper carries the assertion itself
  (`gemma_parity_test.go`: "without this, a silent CPU fallback would pass every assertion
  trivially"), which covers every gate that goes through it. The Metal batched-prefill toggles are
  measurement-only or negative tests (`moe_model_test.go` asserts prefill is REFUSED), not
  comparisons that could pass vacuously.

  **`TestMetalSnapshotGolden`'s embed-scale defect is already FIXED in code** (audit G-02, fixed on
  Linux where the suite cannot run): the checkpoint call drives `ForwardEmb` with the
  production-scaled row, and `Forward`/`ForwardArgmax` apply the arch embed scale. **A re-bake is
  owed on the Mac and the gate is EXPECTED RED there until it happens** —
  `GOINFER_UPDATE_GOLDENS=1`. The safety condition is in the test's own header and must be honoured
  at the RC: `gemma4-dense-scaled` entries SHOULD move (√hidden now applied), `mixtral-tiny`
  entries must NOT — if they do, something other than G-02 changed and the re-bake is refused
  pending investigation.
- [x] **REQUIRED · The Metal manual gate written into `RELEASING.md`** (2026-08-18, new section
  **§C1-M**). Names the exact command (`go run ./cmd/gate gpu`, `GOINFER_GATE_BACKEND=metal`), the
  box (a real Mac — CI's `macos-latest` SIGSEGVs inside purego's `objc_msgSend` on first device
  touch, so its job is build + vet + the one device-free falsifiability test), and **what green
  means in four parts**: all 7 declared check groups emit a verdict (declared-vs-emitted
  reconciliation, audit G-01), zero FAIL with a clean-tree provenance line, **a skip is not a
  pass** (the run's honest scope is the skip list), and the log archived rather than `mktemp`'d
  (C1a). It also records that G10/G11 widened what the run vouches for, and carries
  `TestMetalSnapshotGolden`'s known-red + re-bake safety condition (`gemma4-dense-scaled` entries
  MUST move, `mixtral-tiny` MUST NOT) so the re-bake cannot be done blind.
- [ ] **REQUIRED · Sweep discipline holds at the RC**: `go run ./cmd/gate parity` zero blockers,
  coverage gaps on their own line (none silently absorbed), composition printed
  (`go run ./cmd/gate composition`), and the heavy tier (`go run ./cmd/gate heavy`, E8's successor
  to `scripts/heavy_gate.sh`) run on the RC with its log **committed or archived** — C1a's lesson: a `mktemp` log is a verdict nobody can
  re-check.

- [ ] **REQUIRED · v1.0 gets its OWN benchmark sweep. It cannot inherit v0.11.0's.** Added
  2026-08-31 because the plan it replaces was silently void. `benchmarks.md`'s v0.11.0
  qualification claimed to "double as v1.0's sweep **iff** the code delta between the two tags
  stays zero" — a reasonable plan when written, and false since: v0.11.0 → HEAD is **802 commits,
  467 Go files, +48,477 / −2,066 lines**. Reusing it would qualify v1.0 against a codebase that no
  longer exists. That clause is now struck at its own site, and this line is what replaces it, so
  the obligation is recorded rather than merely the old plan removed. Scope: the same legs the
  2026-08-27 re-anchor covered (§B4.1/§B4.2, §B5.1, §B6.3, §B7.1, §B8) re-run at the RC commit, on
  `scripts/bench_peer.py` — **not** `bench_compare.sh`, which drives no peer.

## 3. API freeze and compatibility — the declaration §0 says is owed

- [x] **REQUIRED · The tier declaration — `docs/api-tiers.md`, SIGNED OFF 2026-08-18 (decider:
  Francis).** Written from the ACTUAL exported surface enumerated with `go doc`,
  not from memory. Hard = `Load`/`LoadGGUFBytes`/`Options`/`Model.{Close,Config,Dims,Quant,NewCache,
  NewSession,Generate}`/`Session`/`SamplingParams`/`Generation`, all four `tokenizer` loaders + the
  `Tokenizer` methods, `chat`'s `Detect`/`Template`/constructors, `constrain`'s `Masker` surface,
  `cmd/serve`'s HTTP routes + operator flags, and the env-var registry's OPERATOR rows.
  Experimental is named in categories rather than by omission: the backend/residency seam and every
  `*Resident*` method, family descriptors/`Weights`, all drafters + speculative entry points,
  multimodal, compute-time adapters, capture/diagnostics, serialization plumbing, and the
  `Options`/flag fields that reach them. It deliberately does NOT decide the three items that are
  their own lines (submodule posture, the `.giw` promise, the apidiff baseline). The
  sign-off unblocked the §3 docs rewrite (done, same day) and gives the apidiff baseline something
  to check against. **The split takes effect AT the v1.0 tag, not at sign-off** — until then the
  project is still pre-1.0 and the docs say so.

  Original wording: A `docs/` page (aikit's Hard/Experimental pattern)
  naming what v1.0 semver-binds: the `decoder` load/generate surface (`Load`, `Options`,
  `Generate`, `Session`, sampler params), `tokenizer`, `constrain`, `chat`, `cmd/serve`'s flag +
  HTTP surface, and the env-var registry (`docs/env-vars.md` becomes contract, not
  documentation). Everything else — including the residency/backend seams and `qwen35Params`-class
  internals — is named Experimental explicitly, not by omission.
- [~] **REQUIRED · apidiff baseline — MACHINERY LANDED AND GREEN 2026-08-18; the "one minor"
  clock starts at the v0.14.0 tag.** `scripts/apidiff_check.sh` compares a baseline TAG against
  HEAD for `decoder`/`tokenizer`/`chat`/`constrain` and **fails only on incompatible changes to
  HARD-TIER names**, read from `testdata/apidiff/hard_tier.txt` — the machine-checkable half of
  `docs/api-tiers.md`. Experimental breaks are reported, not fatal, because the residency seam and
  family descriptors move by design and a gate that failed on those would be switched off within a
  minor.
  - **v0.13.0 → HEAD is CLEAN: zero incompatible changes in all four packages.**
  - **Mutation-verified**: reordering `constrain.TokenBytes`'s parameters makes it report
    `constrain: HARD-TIER BREAK` and exit non-zero; reverting returns it to PASS.
  - Wired into CI on the `root` job (the one that already needs `fetch-depth: 0` — a baseline tag
    a shallow clone cannot see exits non-zero rather than passing, so "could not compare" never
    reads as "nothing changed").
  - `TestAPITiers_hardListMatchesDoc` keeps the list and the document from drifting. **It caught a
    real disagreement on its first run** — the list promised `Options.Quant`/`Options.LoRA` while
    the document named them only under Experimental — which is precisely the failure mode of
    having a prose promise and a machine gate that nobody reconciles.
  - **What remains is time, not work**: the baseline must be clean across a released minor, so the
    v0.14.0 tag starts that clock and the check at the RC discharges the line.
- [ ] **REQUIRED · The serialized-format promises written**: `.giw` (and the `.giw-kv` snapshot
  fingerprint) currently carry "rebuild per minor" + a version guard that fails loud. 1.0 states
  the actual promise — read-N−1, or reserved-header forward-compat, or documented
  rebuild-on-minor — and says which, in one place both the code comment and README cite.
- [ ] **REQUIRED · The docs stop saying "may change."** CHANGELOG header, README status section,
  ARCHITECTURE's "The contract" — all still describe the loader/descriptor surface as pre-1.0 and
  moving. Rewrite to the frozen statement at RC; C2's blind audit reads these from outside.
- [ ] **REQUIRED · Submodule posture decided and recorded**: do `gpu/`, `cuda/`, `metal/`,
  `demo/agent` tag v1.0.0 alongside the core or stay on their own 0.x series (aikit kept `gpu/`
  at 0.x)? Either is fine; undecided is not — the tag-day script needs to know.
- [ ] `RELEASING.md` gains the 1.0 addendum: this gate, the tag order, and the C3 aikit-bump
  trigger it already carries.

## 4. Consumer trust — the C group, run against the RC

- [ ] **REQUIRED, and first, independent of 1.0 · A replace-free tag consumers can
  `go install`.** v0.13.0's tagged submodule `go.mod`s carry committed `replace` directives —
  `go install …@v0.13.0` is broken from outside. The removal landed on main 2026-08-15
  (RELEASING.md updated, cuda verified replace-free on the box); **the fix reaches nobody until
  it is tagged** (v0.13.1 or v0.14.0). Verify `go install` from a clean machine, not a checkout.
  **Measured 2026-08-18 from OUTSIDE the checkout** (isolated GOPATH, module proxy, no local
  source):
  - The `v0.13.0` breakage is confirmed and precisely scoped: the `gpu/`, `cuda/` and `metal/`
    tags each carry `replace github.com/townsendmerino/goinfer => ../`; the ROOT `v0.13.0` go.mod
    is clean. (A `replace` in a dependency is ignored, but `go install pkg@version` makes that
    module the MAIN module, so the directive applies and `../` does not exist.)
  - main is replace-free in **all five** go.mods.
  - `go install github.com/townsendmerino/goinfer/cmd/serve@main` **succeeds from a clean GOPATH**
    — resolves through the proxy, pulls aikit v1.21.0, produces a 13.9 MB binary.
  - All four submodules resolve at `@main` (`go list -m` returns the pseudo-version for `gpu`,
    `cuda`, `metal`, `demo/agent`).

  The fix is therefore real and proven on the tree; **what remains is the tagging**, plus re-running
  the same probe against the TAG — a pseudo-version proves the tree, not the tag.
- [ ] **REQUIRED · C2, the blind out-of-tree audit, against the RC tag.** Fresh session, no repo
  access, prompt already written, known-findings list carried. Tier 1 is the claim-by-claim README
  check — claim-discipline rule 7 tested from the position the rule is about. A clean C2 is the
  last line of this gate to go green before the tag.
- [ ] **REQUIRED · C1, CUDA drain/unload verification.** The admin-unload drain (`588052b`) is
  Metal-verified only; CUDA is the backend where `Close` does the most (pinned host memory,
  mapped expert stacks, CUDA graphs, executor-routed release). Four parts as queued: VRAM reclaim
  across load/generate/unload/reload; the `preamblePark` regression under `-tags
  goinfer_testhooks`; the adapter-straggler case; `--unload-drain-wait` under a real generation.
- [ ] **REQUIRED · C4, soak.** Nothing has run longer than minutes (the G1 memory-plateau
  finding rested on 75 seconds). One overnight `cmd/serve` soak per backend under a request loop:
  RSS/VRAM growth, KV-session accumulation, fd count, thermal throttle — recorded numbers, not
  impressions.
- [ ] **RECOMMENDED · Sign/notarize the darwin release assets.** The Gatekeeper `xattr` step
  contradicts the one-file pitch at the exact moment of first contact. If it can't land by 1.0,
  keep the README caveat honest and lead the launch with the gif, not the download.
- [ ] **RECOMMENDED · Supply-chain lines**: govulncheck green in goinfer CI (confirmed
  2026-08-12) — **VERIFY aikit** has the same; commit fuzz-corpus seeds for the audit's
  hostile-input findings (16 targets, 3 committed corpora today — a crasher found once and not
  committed is found again next year).

## 5. Decisions v1.0 makes permanent — decide, don't drift into them

- [ ] **REQUIRED · The name.** Two other `goinfer` repos exist (synw/goinfer, LM4eu/goinfer — an
  LLM proxy). The module path is the one thing 1.0 freezes culturally forever; renaming later is
  a v2-scale event. Decide keep-or-rename **at RC cut**, recorded with reasons, so the launch
  thread answers it in one line instead of relitigating it.
- [ ] **REQUIRED · The supported-vs-experimental line in the README**, stated once and generated
  where possible: supported = current-T3 manifest row; experimental = named list. The number in
  the launch copy comes from this line and nowhere else.
- [x] **D3b — the expert-cache default. LANDED 2026-08-20 in `8f3c5e7`**, before this line was
  written and while it still read as pending: the default is a bounded `8 × topK`, floored at
  `topK` and still VRAM-capped by `allocSlots`. The two things this line asked for both exist —
  the 26B run (`benchmarks.md` §B4.1: 30 slots, 76.1% hit, 16.12 tok/s; 40 slots, 82.2%, 17.62)
  and the hit-rate figure. Derivation basis unchanged and still scoped to a single CUDA context
  (289,013,760 ≤ 402,653,184); revisit if goinfer ever creates a second.
  **It did not ride in silently** — `8f3c5e7` is a standalone default change with its sweep in the
  commit subject, which is what this line was guarding against. What went wrong was the record,
  not the landing.
- [ ] **aikit branch protection** — the struck entry says "revisit at v1.0." Revisit, decide,
  record (the gate ritual may remain the enforcement; then say so).
- [ ] **E7 · no-Python** (decided 2026-08-12): the sweep done, with the `pin_*` reference-tensor
  carve-out recorded as *the 1.0 state* (blocked on the torch-replacement research), not as a
  leak. The gate line is: no Python outside the carve-out, and the carve-out named in one place.

## 6. Claims and docs at the RC

- [ ] **benchmarks.md held to its own maintenance rules at the RC sha**: every number ≤ one
  minor stale or struck, peer cells re-verified against peer docs, no floating figures. The
  16.98-tok/s-class numbers with heavy provenance caveats either get re-measured on the RC or
  keep their caveats verbatim.
- [ ] **Matrices regenerated at the RC sha** (`CapabilityMatrix`/`HardwareMatrix -update`) — the
  freshness gates enforce this; the line exists so tag-day doesn't discover it.
- [ ] **README sweep**: no env-var instructions D3's flags replaced, no stale coverage phrasing.
  C2 checks the same thing from outside; this is the inside pass so C2 finds nothing cheap.
- [ ] **E5 · launch materials rebuilt, not edited.** The July drafts quote withdrawn figures
  (476/268-era) and the pre-fix peer table. Rebuild against RC numbers after C2 comes back clean.

## 7. Explicitly NOT blocking v1.0 — scope, held on both sides

Continuous batching · vision prefill performance · GPU-residency breadth (including DeltaNet
residency — every hybrid family is CPU-only on all backends and honestly labelled) · G1/LFM2.5
and further new families · the wasm/browser demo · PGO · D1/D2 CUDA launch-table tooling ·
hybrid state-checkpoint reuse/spec. Each stays queued where it is. Listed so nothing here creeps
into the gate, and so nothing in the gate is deferred *into* this list without a strike.

## 8. The run, in order

1. **v0.14.0 — RC minus one.** Replace-free tags across all five modules (the `go install` fix
   reaches consumers); D3 merged behind its goldens run on a fixture-bearing checkout; the
   backfill record rides as reserved (E1's 2026-08-12 decision); B13's two reds resolved; the
   w4a8 asset built; B15 fixed. If this tag carries an aikit bump, C3's auto-pickup fires — let it.
2. **The freeze declaration** (§3 wholesale): tier doc, apidiff baseline, format promises, docs
   rewrite, E7 sweep. If any of it moves user-visible surface, that's v0.15.0; otherwise it rides
   the RC.
3. **Cut `v1.0.0-rc.1`** (a real semver pre-release tag, installable from outside). Against it:
   C1, C2 (blind), C4, the tautology census results, the §6 docs/claims sweep. The name decision
   (§5) is recorded **before** this tag.
4. **Findings loop.** C2/C1/C4 findings fixed → `rc.2` if anything was a blocker. An RC that
   needed no `rc.2` is a nice surprise, not the plan.
5. **`v1.0.0`.** Tag order per RELEASING.md; matrices + CHANGELOG + release notes at the tag sha;
   `go install` verified from a clean machine on both boxes; **then** E5's rebuilt launch
   materials go out. Promotion follows verification — same rule as July, still the house rule.

## 9. Tag-day mechanical checklist

- Goldens at the tag sha, composition printed; sweep: zero blockers, gaps listed by name.
- Heavy gate run on the RC, log archived where a later reader can find it.
- Capability + hardware matrices regenerated at the tag sha; CHANGELOG finalized; release notes
  in the release-notes path `task-release-path-restructure.md` settled on.
- All module tags cut from the replace-free tree; `go install` of `cmd/serve` and the demo
  verified from outside the repo, both platforms.
- README's supported count equals the manifest's; the launch copy quotes only §6-verified numbers.
