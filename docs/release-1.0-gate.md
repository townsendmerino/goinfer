# The v1.0 gate — criteria, evidence, and the run plan

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
- [ ] **REQUIRED · B13's standing reds resolved or formally reclassified.** Two remain:
  `TestQwen35GGUF_vsSafetensors` (cosine 0.987835 vs a 0.998 floor — self-described as a loader
  bug) and `TestSerializeWeightsTo_matchesBuffer` (B11). A 1.0 does not tag over standing red
  parity-adjacent tests. "Reclassified" means a recorded decision with a decider, not a floor
  adjustment to make red green.
- [ ] **REQUIRED · The w4a8 coverage gap closed by building the asset.** `TestW4A8DecodeParity`
  has been `ASSET_NEVER_BUILT` (the matched int8 `.giw` was never produced) since the sweep
  learned to say so. The sweep's own rule applies: the only correct way off that list is to build
  the asset. int4 is the documented default quant; its decode-parity gate must actually run at 1.0.
- [ ] **REQUIRED · B15 manifest-emission defect fixed** (`EMIT_MANIFEST=1` flips
  `experimental → validated` while leaving `method: tiny-golden`, and mangles mellum's method).
  Until fixed, every stamp is manual — fine for a minor, not for the 1.0 stamp.
- [ ] **REQUIRED · Every registered family has a manifest row at its honest tier at the RC
  commit**, tripwire green at N/N. New since the backfill closed and to confirm on the RC:
  `qwen3_5` (real-checkpoint gate `4d1a7f2`, token-id parity `e3674aa`), the InternLM2/3
  llama-dialect rows, `laguna`. **VERIFY** the regenerated matrix carries them at the RC sha.
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
- [ ] **REQUIRED · Tautological-gate hunt, run as a census.** The shape: a gate that compares
  A-on vs A-off **without asserting A was admitted** cannot fail when A silently declines. Found
  on CUDA (four graph tests, 2026-08-12 — **VERIFY** fixed) and suspected on Metal
  (`TestMetalSnapshotGolden` drives `Forward`/`ForwardArgmax`, which apply no embed scale where
  production applies √hidden — a gate that cannot fail). Census every `*_test.go` that guards a
  resident/optimized path; each must assert admission before comparing.
- [ ] **REQUIRED · The Metal manual gate written into `RELEASING.md`.** The Metal device suite
  does not run in GitHub CI (the runner's paravirtual objc layer SIGSEGVs inside purego), so
  Metal's device coverage is a manual box run. At 1.0 that run must be a written ritual — exact
  suite, box, and what green means — not a habit. gpt-oss Metal residency (`1c0a9ed`) just
  widened what that run vouches for.
- [ ] **REQUIRED · Sweep discipline holds at the RC**: `scripts/parity_sweep.sh` zero blockers,
  coverage gaps on their own line (none silently absorbed), composition printed
  (`scripts/sweep_composition.py`), and the heavy tier (`scripts/heavy_gate.sh`) run on the RC
  with its log **committed or archived** — C1a's lesson: a `mktemp` log is a verdict nobody can
  re-check.

## 3. API freeze and compatibility — the declaration §0 says is owed

- [ ] **REQUIRED · The tier declaration.** A `docs/` page (aikit's Hard/Experimental pattern)
  naming what v1.0 semver-binds: the `decoder` load/generate surface (`Load`, `Options`,
  `Generate`, `Session`, sampler params), `tokenizer`, `constrain`, `chat`, `cmd/serve`'s flag +
  HTTP surface, and the env-var registry (`docs/env-vars.md` becomes contract, not
  documentation). Everything else — including the residency/backend seams and `qwen35Params`-class
  internals — is named Experimental explicitly, not by omission.
- [ ] **REQUIRED · apidiff baseline, clean across at least one minor before the tag.** aikit's
  precedent was two consecutive minors verified before freezing; goinfer should meet at least
  one, in CI, against the declared Hard tier.
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
- [ ] **D3b — the expert-cache default.** The blocker's derivation basis is recorded
  (single-context margin: 289,013,760 ≤ 402,653,184, revisit if goinfer ever creates a second
  CUDA context); landing still needs the 26B run + a hit-rate figure. Decide: land before 1.0 or
  explicitly defer post-1.0. Either is fine; it must not ride in silently with the RC.
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
