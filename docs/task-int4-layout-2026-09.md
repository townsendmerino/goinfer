# Task: one int4 layout per tensor, chosen for its consumer — 2026-09 (L1–L5)

> **Status: OPEN, drafted 2026-09-11** on branch `aikit-v1.41.0-bump` (off `main` at `46c9dfa`),
> against aikit v1.41.0 (its `docs/audit-2026-09-10.md` M-22 entry is the aikit-side contract).
> Companion to `docs/task-gpu-paths-2026-09.md`; this is the CPU/format half of the same question —
> which representation of an int4 tensor exists in memory and on disk, and who decided.
>
> Suggested order: L1 (in progress, finish first) → L3 (ten minutes, do with L1) → L2 → L5 on
> nobara once L1 lands (its park condition is met — see L5). L4 is filed, not scheduled.

## The decision this doc records

An int4 tensor has three CPU representations in aikit: canonical nibbles (what a GPU backend
uploads, what `Row()` read until v1.41.0, what every kernel accepts), arm64 row4 (the fast NEON
decode/prefill layout, a fixed permutation of the same nibbles local to a quad of 4 rows), and
amd64 split-half (same idea, per row). Until now goinfer kept **canonical + row4 both**, always —
in RAM (`repackW4A8Row4IfEligible`, a heap copy next to the canonical heap copy: ~0.95 GB on a
1.5B int4 model) and, when opted in, on disk (`.giw` kind 4, 2× the int4 bytes in the file).

The "both" policy exists for portability: one `*decoder.Model` — or one `.giw` — usable by any
consumer. But every consumer is known at the moment the representation is chosen:

- **At `decoder.Load`**, the caller names the backend. A GPU backend (metal/cuda/webgpu) uploads
  canonical once (resident) or hands canonical bytes to a kernel per token (staged consult). A
  CPU backend never needs canonical once row4 exists — `Row()` is layout-independent since
  aikit v1.41.0, and `MatmulBT*`/`MatmulBTW4A8Batch` prefer row4 when present.
- **At prequant time**, the `.giw` is built either on the box that will read it
  (`internal/prequant.EnsureCachedGIW`, `serve --stream-weights`) or in CI for a per-os/arch
  asset whose backend tag is already chosen (`demo/chat/build-embed.sh`). Nobody carries a
  `.giw` from one kind of machine to another.

So: **one representation per tensor, chosen from the consumer, stated explicitly, never inferred
from an omission.** "Both" survives only as the legacy read path for existing kind-4 files.

## Ground rules

- Additive on aikit: nothing here needs an aikit change. `WrapInt4Row4Only`,
  `RepackInt4Row4InPlace`, `RepackInt4Row4Quad`, `Int4Row4Usable`, `IsInt4`, `Int4Layout` all
  exist at v1.41.0. `Int4()` keeps meaning "canonical bytes present" — never repurpose it.
- Bit-identity contract holds: this is a storage choice, not a numerics one. No golden may
  depend on which layout a tensor took (the existing kind-3/kind-4 rule, `serialize.go` format
  comment).
- A wrong-representation model or file fails **loudly and at load** — a named error saying which
  representation was present, which was needed, and what to do — never a silent fallback to a
  slower path. The failure mode we are designing against is a Metal box quietly decoding on the
  CPU.
- Paged tensors (MoE experts, layer paging) stay canonical on every target. Paging has no
  load-time repack step and preads canonical spans off the mapping (`metal/moe.go:446`,
  `metal/gemma4_moe.go:225`, `decoder/moepaging.go`).
- One doc. Findings from doing the work go into the per-item status line here.

---

## L1 — Load-time policy: `Backend: "cpu"` is a promise, and it unlocks repacked-only (DONE 2026-09-11)

**Where.** `decoder/weightmat.go:414 wantsCanonicalInt4(backendName, be)`,
`:440 repackedOnlyOrCanonical`, `:524 isBatchedProjTensor`; `decoder/model.go:382` (computed once
at Load); the `needCanonical bool` threaded through `loadWeights` → `loadGGUFWeights` /
`buildWeightsFromSafetensors` → `quantizeEmbedWM` / `streamQuantizedEmbed` /
`quantizeBatchedProjWM` / `streamQuantizedBatchedProj`. Dispatch prerequisite already done:
`isW4A8`, `matmul`'s and `matmulInto`'s int4 branches route on `IsInt4()`; the
`QuantBackend4.MatmulW4A8` consult stays under `Int4()`'s ok (mutation-tested: reverting is a
nil-slice fault in `linalg.MatmulBT`, not a slowdown).

**The gap it hit.** `TestMetalSnapshotGolden` (`metal/snapshot_golden_test.go:124`) loads with no
`Backend` set, then calls Metal's unexported `buildResident` directly — an idiom used at 87 sites
in `metal/*_test.go` (6 in `gpu/`, `cuda/` unchecked). With the empty backend resolving to CPU
and therefore repacked-only, `int4Concat` (`metal/model.go:465`) declined via its
panic-and-recover. Not a crash — but it shows the gate had turned `*decoder.Model` from
backend-agnostic data into something with a hidden property and a silent failure mode.

**Fix.**
1. Repacked-only is chosen **only when `Options.Backend` is the literal `"cpu"`**. Empty /
   unspecified keeps canonical(+row4), today's default. The interface checks
   (`QuantBackend4` / `QuantBatchBackend4` / `ResidencyBackend`) stay as a second guard when a
   GPU backend is named. This fixes all 87 sites as a class; do not audit them one by one.
2. Make the property visible: the per-class int4 layout (`WeightMat.Int4Layout()`) in
   `BackendReport()` and the parity manifest, so `serve check` shows it.
3. Make the decline loud and correct: `int4Concat`'s message reads `weight kind "int4" not int8
   or int4` (it checks `Int4()`'s ok and reports `Kind()`). Replace with the real condition —
   "int4 tensor has no canonical bytes (layout row4-only): model was loaded with Backend "cpu";
   load with Backend set to this backend". `buildResident`'s recover carries that text into the
   error; `withResidency` logs it. Same by inspection in `cuda/` and `gpu/`.
4. Root (no-tags) `goinfer-serve` / `demo/chat` / the embedded variant must pass `"cpu"`
   explicitly if they currently leave `Backend` empty — otherwise the CPU release binaries never
   get the saving. `metal/cmd` and `cuda/cmd` already name theirs.

**Gate — done, all green.**
- `metal.TestBuildResident_declinesWithLoudReasonForRepackedOnlyInt4`
  (`metal/int4_repacked_only_decline_test.go`): loads `gemma4-dense-scaled` with `Backend:"cpu"`,
  calls the unexported `buildResident` directly (the same idiom `TestMetalSnapshotGolden` uses),
  asserts the decline error contains "no canonical bytes", "row4-only", "Options.Backend", and
  `"metal"`. Mutation-checked: reverting the message fix makes it fail on all four assertions
  with the old text. `TestMetalSnapshotGolden` itself now passes again (no `Backend` set → stays
  canonical(+row4), the gap this whole item exists to close).
- Three suites, same run: `decoder` — 2 pre-registered failures only
  (`TestOlmo3_forwardParity`, `TestParityManifest_fresh` — the latter is the deps_hash staleness
  this edit itself causes, refreshed separately before commit, not a real failure). `-tags metal`
  — 1 pre-registered failure only (`TestMoE_declinesPrefill`, unrelated, confirmed pre-existing
  earlier this session in an isolated worktree). `-tags gpu` — fully green, `ok`, no exclusions
  needed.
- RSS, real checkpoint (`~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf`, int4, `syscall.
  Getrusage(RUSAGE_SELF).Maxrss` immediately after `decoder.Load` returns, this Mac):
  `Backend:"cpu"` **2365.1 MB** vs unspecified **3254.7 MB** — **889.6 MB saved**, matching
  aikit's own "~0.95 GB on a 1.5B" estimate closely. (No `-embed-int4`, so Embed/LMHead stayed
  int8I8 in both arms — this delta is qkv+gate/up alone; down-proj stayed `canonical+row4` in
  both arms, as designed.)

**Findings / deviations from this doc, reported rather than worked around:**
- **Item 2 ("and the parity manifest"): interpreted as `serve`'s actual diagnostic surfaces, not
  `testdata/parity_manifest.json`.** That file tracks per-family NUMERIC validation
  (`deps_hash`/`cosine`/`validated_at`), not a per-load runtime layout choice — Backend:"cpu" is
  a request-time decision, and the same family can load either way in different processes, so
  there is no single per-family "layout" fact to record there. Implemented instead on the two
  surfaces that are actually consulted: `Model.BackendReport()` (what `internal/chatapp`,
  `internal/gemmaapp`, `demo/gemma-web` print at load) and `Model.DecodePath()` (what
  `internal/serveapp` — the actual `goinfer-serve` binary — logs and returns as `decode_path` in
  the API; also `servecheck`'s wire format shows it via `decode_path` if a route echoes it,
  though servecheck itself is an HTTP client, not a direct `*Model` inspector, and has no route
  today that surfaces this specifically). Format: `int4 layout: row4-only (embed, qkv, gate/up);
  canonical+row4 (down)` — grouped by tensor CLASS (embed, lm_head, qkv, gate/up, down), one
  representative tensor per class (every tensor in a class shares one Load's `needCanonical`
  decision). New: `Model.int4LayoutSummary()` (`decoder/residency.go`), tested by
  `TestBackendReport_int4LayoutVisible` (both surfaces, both arms — `Backend:"cpu"` shows
  `row4-only`, unspecified does not).
- **Item 3 ("Same by inspection in `cuda/` and `gpu/`") found a REAL latent bug in `cuda/`, worse
  than Metal's.** `cuda/resident.go:3064`'s `packWeight` switches on `w.Kind()` (stays `"int4"`
  for a repacked-only tensor — `Kind()` is precision, not layout) and used to discard `Int4()`'s
  `ok` entirely (`q4, sc, _, _ := w.Int4()`), so a repacked-only tensor's nil `q4` would panic on
  an out-of-range slice index (`q4[i*4:i*4+4]`) rather than decline through the function's own
  `(hostW, error)` contract — not merely a worse message, an actual crash path with no recover
  visible at this level. Fixed the same way as Metal: check `ok`, return a matching named error
  ("no canonical bytes (layout %s-only) … Options.Backend set to \"cuda\""). **Unverified on real
  hardware — this Mac has no CUDA device or working local `cuda` build at all** (pre-existing,
  confirmed unrelated to this work earlier this session: `undefined: gpu.MappedHostBuffer` etc.,
  reproduces identically on a pre-this-branch commit in an isolated worktree). `gofmt`/build
  (module-graph-only, since `cuda` cannot compile here) confirm the edit is syntactically sound;
  nothing more.
  <br>`gpu/` (webgpu) has no direct `int4Concat`/`packWeight` equivalent — its closest analog,
  `uploadProj`'s int4 case (`gpu/residency.go`), already returns a clean `error` rather than
  panicking (`"gpu: residency int4 group %d != %d"` when `Int4()`'s `ok` is false and `group`
  comes back zero), so no code change was needed there. Under this gate it is now provably
  unreachable in practice too: `webgpu` is never the literal string `"cpu"`, so
  `wantsCanonicalInt4` always keeps canonical for it regardless.
- **Item 4: no code change needed.** `internal/serveapp/main.go:418`,
  `internal/chatapp/main.go:177`, and `internal/gemmaapp/main.go:47` all already register `--backend` with
  `flag.String(..., "cpu", ...)` — the literal default is already `"cpu"`, not empty. The root
  (no-tags) CPU release binaries already got this saving the moment L1 landed; nothing to wire up.
  `demo/chat`'s embedded variant shares `internal/chatapp`'s flag registration, so the same holds
  for it. Did not check `internal/gemmaapp`'s actual callers beyond confirming its own flag
  default, since gemma-web was out of this item's named list.

**Out of scope, unchanged:** down-proj / router / experts; amd64 split-half (L5); the
`.giw` path (L2, and its scope note in `decoder/gguf.go:1388` is corrected under L3).

**Size.** Small — the branch already had the mechanism; this was the gate's final shape plus
reporting and the error.

## L2 — `.giw` kind per target: emit the one layout the reader will use (format v11) (DONE 2026-09-11)

**Where (as implemented).** `decoder/serialize.go` — format comment (`:49–72`), `giwWriter.target`
(`:933–970`), `weightMat`/`weightMatKind3Only`/`weightMatKind` (`:1009–1103`), `readWeightMat`
kinds 3/4/5 (`:1466–1525`), `giwVersion = 11` (`:87`), `SerializeWeightsForTarget`/
`SerializeWeightsToForTarget` (`:207–233`); `decoder/weightmat.go:542–614` (`GIWTarget`,
`GIWTargetForBackend`, `ParseGIWTarget` — new, not anticipated by the "Where" list above);
`decoder/gguf.go` (`StreamTranscodeGGUF`'s `target GIWTarget` param); `decoder/weights.go`
(`repackedOnlyInt4Count`); `decoder/model.go` (the `.giw` branch's post-load backend check);
`decoder/w4a8_row4_emit_arm64.go` (`repackRow4ForEmit`, reused unchanged);
`internal/prequant/prequant.go` (`Transcode`'s `target GIWTarget` param, `EnsureCachedGIW`'s new
`backend string` param, `streamCachePath`, `selfCheck` — see Findings for why the last one
needed a fix); `cmd/prequant/main.go` (`-target` flag replaces `-row4`); `demo/chat/build-embed.sh`
around its `go run ./cmd/prequant` call (the hyphen in this filename defeats
queue_citation_lint.py's path regex for a `:line` suffix, so cited by name only, matching this
repo's other docs).

**What is wrong today.** Kind 4 stores canonical **and** row4 — 2× disk per int4 tensor — so that
one file loads on any core (`WrapInt4Row4` gates on `row4Usable()` and falls back). That
portability is unused (see the decision above). Worse, the serve-side auto-cache always emits
kind 3, so on the Mac `serve --stream-weights` at int4 runs the **canonical** W4A8 kernel — no
row4 at all, since the in-RAM repack is deliberately not applied to an mmap'd `.giw`. The fast
layout is reachable only by running `cmd/prequant -row4` by hand.

Note what this is *not*: it is not an RSS saving. `.giw` weights alias the mmap, and only touched
pages become resident; a kind-4 file's canonical pages stay cold on the CPU decode path. The wins
are half the disk and page cache per int4 tensor, and row4-by-default for CPU caches.

**Fix.**
1. **A target, not a flag.** Replace `Transcode`'s `row4 bool` with a target
   (`cpu-arm64` | `cpu-amd64` | `metal` | `cuda` | `webgpu`), derived: in `EnsureCachedGIW` from
   the backend `serve` is about to load with; in `cmd/prequant` from a `-target` flag defaulting
   to the build box's CPU; in `build-embed.sh` from the tag it already picks.
2. **Per-tensor kind from the target.** Kind 3 (canonical) for metal/cuda/webgpu targets (the
   resident build uploads canonical once; `int4Concat` needs no change) and for every paged
   tensor on every target. **New kind 5, row4-only** (`q4Row4Scales`, `q4Row4`, no canonical
   arrays) for dense projections and the embedding on a `cpu-arm64` target where
   `Int4Row4Usable` passes; loaded with `WrapInt4Row4Only`, zero-copy off the mapping — the
   canonical-never-exists form. Kind 4 is no longer emitted; it stays readable as legacy.
   (Kind 6, split-half-only, is L5.)
3. **Cache key carries the target.** `streamCachePath(ggufPath, quant)` → includes the target,
   so a CPU-built cache is never reused by a Metal load. The v1-blob version guard already does
   "wrong file → rebuild from GGUF"; a target mismatch takes the same path.
4. **Loader mismatch is an error, not a fallback.** A kind-5 tensor on a core where
   `WrapInt4Row4Only` returns ok=false, or under a GPU backend that needs canonical, fails
   `readWeightMat` with: which kind, which core/backend, and "rebuild with `prequant -target …`
   (or delete the stream-weights cache)". `giwVersion = 11`.
5. Emission of kind 5 reuses `repackRow4ForEmit` (already out-of-place from canonical during
   the transcode; the transcode holds canonical transiently anyway) — just skip writing the
   canonical arrays. arm64 build box required for a `cpu-arm64` target, as today.

**Gate — done, all green (pre-registered/environmental failures named below).**
- `decoder.TestSerializedInt4Weights_kind5RepackedOnly_matchesCanonical`
  (`decoder/w4a8_row4_giwkind5_test.go`): on the real int4 fixture, 168/168 row4-eligible int4
  tensors round-trip through a `SerializeWeightsForTarget(GIWTargetCPUArm64)` bundle with
  `Int4()`'s ok FALSE and `Int4Row4()`'s ok TRUE (repacked-only); greedy first-token is
  bit-identical across GGUF (in-RAM row4) / kind-3 `.giw` / kind-5 `.giw`; bundle size delta is
  **0 bytes** vs kind 3 (497.4 MB both) — row4 swaps in for canonical, doesn't add, confirmed on
  the real 1.5B checkpoint too (1293375051 bytes, identical to the byte, canonical vs cpu-arm64).
- `decoder.TestLoad_kind5UnderBackendNeedingCanonical_declinesLoudlyAtLoad`: a kind-5 `.giw`
  loads fine under `Backend:"cpu"`, then fails `decoder.Load` under `Backend:"metal"` with:
  `"decoder: <path>: 168 int4 tensor(s) are stored row4-only (kind 5, a cpu-arm64 prequant
  target) but Backend \"metal\" needs canonical bytes — rebuild with \`go run ./cmd/prequant
  -target <matching this backend>\` (or delete the stream-weights cache so it rebuilds
  automatically)"`.
- `decoder.TestGiwReaderWeightMat_kind5DeclinesWhenThisCoreCannotUseRow4Only`: hand-built kind-5
  bytes with `rows=3` (not a multiple of 4 — `Int4Row4Usable` must reject it) fail
  `giwReader.weightMat` with `"...is stored row4-only (kind 5, a cpu-arm64 prequant target) but
  this core cannot use that layout — rebuild with \`go run ./cmd/prequant -target <this
  core>\`..."`. Mutation-checked: the message names "kind 5" / "row4-only" / "this core" as
  asserted, and the test fails (as intended) if any of those strings changes.
- `decoder.TestSerializedInt4Weights_row4Kind_matchesCanonical` (existing, unmodified): kind 4
  still round-trips correctly — the legacy path is unaffected.
- Three suites, real checkpoints, GOINFER_HEAVY_TESTS=1, run sequentially so they don't contend
  with each other or with another session's concurrent `linalg -race` run (a first, contended
  attempt produced spurious OOM/fit-guard failures that vanished on isolated re-run — this box
  was at load average 5–6.7 on 17 GB RAM at the time):
  - `decoder`: 4 failures — `TestOlmo3_forwardParity` and `TestParityManifest_fresh`
    (pre-registered, L1's own baseline, the latter refreshed separately before commit) are real;
    `TestRequantBar_DoubleQuantCost` and `TestSamplingThroughputGate` are NOT — both are
    unrelated subsystems (int4 requant timing; sampler top_p throughput ratio) that fit-guard
    themselves off *currently available* RAM/CPU headroom, both failed with
    "this machine currently has N GB of memory available" / a throughput-ratio gate a busy box
    pushes over, and neither touches any file this item edited. Confirmed load-sensitive, not
    code-sensitive, by re-observation: identical fit-guard-shaped failures appeared on a
    DIFFERENT, unrelated set of tests during the first (contended) run and disappeared for those
    once contention cleared.
  - `-tags metal`: 4 failures — `TestMoE_declinesPrefill` is L1's own pre-registered baseline;
    `TestEncodeAhead`, `TestPrefillNoNaN`, `TestPrefillParity` are NOT — all three fail with
    "resident context N positions exceeds this backend's hard ceiling of 4096", where N is
    whatever the fit guard auto-sized FROM currently-available RAM (20907, 28015, 5986 across the
    three) — a pre-existing fit-guard-vs-Metal-ceiling interaction, unrelated to int4 layout
    (none of these tests load an int4 model or touch a `weightMat`/`GIWTarget` code path), and
    reproduced under the same elevated-load conditions as the decoder suite's two.
  - `-tags gpu`: fully green, `ok`, no exclusions needed.
- On-disk size, real 1.5B int4 checkpoint (`~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf`,
  via `cmd/prequant`): `-target canonical` 1293375051 bytes, `-target cpu-arm64` 1293375051
  bytes — **identical to the byte**.
- CPU decode tok/s, same checkpoint, `Backend:"cpu"`, greedy, interleaved (canonical then kind-5,
  ×3, each after its own untimed warm-up generate, 64 timed tokens/run — difference matched, not
  pooled, per this repo's own measurement discipline): canonical 36.79 / 39.66 / 39.19 tok/s vs
  cpu-arm64 (kind 5) 47.71 / 49.37 / 46.75 tok/s — **+19% to +30% per pair** (best-of-3: 39.66 →
  49.37, **+24.5%**). This is the row4-vs-canonical CPU kernel gap the doc predicted as "the
  visible number" — informational (not a pass/fail gate), same discipline as the existing
  `TestW4A8Row4GiwKind_loadTimeAndMemoryDelta`.

**Findings / deviations from this doc, reported rather than worked around:**
- **The ground rule's paging exclusion, checked against the actual pager code, contradicts
  itself if read broadly.** `decoder/moepaging.go`'s `newExpertPager` and
  `decoder/layerpaging.go`'s `newLayerPager` BOTH already call `WeightMat.MappedSpanRow4` before
  falling back to `MappedSpan` — i.e. paging today already prefers a tensor's row4 span over its
  canonical one whenever a row4 span exists (a kind-4 file), and neither pager requires canonical
  bytes to be present at all for this to work. So: (a) `weightMatKind3Only`'s MoE-expert exclusion
  (`l.Experts[*]`, `mo.expertsGateUp/expertsDown`) is IMPLEMENTED PER THE DOC'S LITERAL TEXT, but
  is more conservative than the inspected code strictly requires — kept as written rather than
  silently loosened, since Metal's OWN expert paging (`metal/moe.go:446`,
  `metal/gemma4_moe.go:225`, cited by the ground rule, NOT inspected this round) may have a real
  canonical-only requirement the CPU pager does not. (b) The doc's worked example for kind 5 —
  "dense projections" — is EXACTLY what `layerpaging.go` pages for a big dense model that doesn't
  fit resident; reading the ground rule to also exclude THOSE would gut L2's own stated purpose
  for the one case it names. They were NOT excluded (plain `weightMat`, kind-5-eligible), on the
  strength of the same `MappedSpanRow4`-first inspection. Flagging both rather than deciding
  either silently.
- **Item 4's "fails `readWeightMat`" is split across two functions, not one.** The arch/shape
  mismatch (`WrapInt4Row4Only`'s `ok=false`) DOES fail inside `readWeightMat`, named. The BACKEND
  mismatch (a kind-5 file loaded under `Backend:"metal"`) cannot — a `.giw`'s target is baked in
  at WRITE time, and `readWeightMat`/`LoadSerializedWeights` have no visibility into the
  CALLER's `Options.Backend` (`LoadSerializedWeights(data []byte)`'s existing public signature,
  ~25 call sites across the tree). Implemented instead as a post-`LoadSerializedWeights` check in
  `decoder.Load`'s own `.giw` branch (`repackedOnlyInt4Count`, `decoder/weights.go`), reusing the
  same `wantsCanonicalInt4` the GGUF path already computes — still "fails loudly and at load"
  (before `Load` returns a `*Model`), without widening `LoadSerializedWeights`'s signature. This
  check is NOT redundant with L1's Metal `buildResident` decline: `withResidency`'s decline is
  soft (logged, falls back to CPU/staged, `Load` still succeeds) by design, so it alone would NOT
  satisfy "fails at load" for this case — verified by writing the test first with only L1's
  protection in place and watching it fail to fail (i.e. `Load` returned a `*Model` for a
  Metal-backend kind-5 load with no error) before adding this check.
- **Kind-5 eligibility scoped to `Weights.matmulWeights()`'s existing enumeration**
  (dense Q/K/V/O/gate/up/down, embed/lm_head, router, shared-expert(+gate), gemma4 PLE,
  per-layer-embed) so `decoder.Load`'s post-load census stays complete by construction rather
  than needing its own separate tensor list. KDA/DeltaNet/qattn mixer projections and gemma4's
  fused-MoE router+experts (absent from `matmulWeights()`) stay `weightMatKind3Only` — kind 3
  always — for now; extending kind 5 to them is a follow-on, not started, not blocking.
- **Found a real bug fixing this: `internal/prequant.selfCheck` used `Options{}` (empty
  `Backend`).** Empty means "needs canonical" (L1's own literal-"cpu"-is-a-promise design), so
  EVERY `-target cpu-arm64` transcode failed its own self-check the first time this was run for
  real: `prequant: self-check: decoder: ...: 196 int4 tensor(s) are stored row4-only ... needs
  canonical bytes`. Fixed to `Options{Backend: "cpu"}` (accepts both kind 3 and kind 5; matches
  what the file is actually loaded with in production either way).
- **`demo/chat/build-embed.sh` builds ONE `.giw` before its cross-compile loop, then embeds the
  same bytes into several os/arch/backend binaries** (darwin+metal, linux+cuda, windows/plain
  CPU — potentially cross-arch relative to the build host). No single non-canonical target is
  safe for all of them; defaulting `-target` to the host's own arch would have silently baked an
  arm64-only kind-5 layout into e.g. a cross-compiled linux/amd64 binary, failing loudly only on
  an end user's machine instead of here at build time. Fixed by passing `-target canonical`
  explicitly — a new value (`ParseGIWTarget`, not one of the doc's 5 named targets) added for
  exactly this "more than one consumer" case.
- **Kind 4 (`SerializeWeightsRow4`/`SerializeWeightsToRow4`) was NOT retired**, despite "Kind 4 is
  no longer emitted" in the Fix list above. Read that as describing the NEW target-driven pathway
  (`Transcode`/`EnsureCachedGIW`/`cmd/prequant`, which indeed never chooses kind 4 anymore after
  this change), not as an instruction to delete the pre-existing legacy API and its three
  dedicated tests, which still exercise a real, working, independently useful format ("usable on
  any core" portability kind 5 does not have). `giwWriter` carries both `row4 bool` (legacy) and
  `target GIWTarget` (new) as independent fields; kind 5 wins if a caller somehow set both, though
  none does.

**Out of scope, unchanged:** L4 (Metal-resident triple-residency, filed); L5 (amd64 split-half —
blocked on an aikit prerequisite, see that section).

**Size.** Medium, as scoped. Touched `decoder/serialize.go`, `decoder/weightmat.go`,
`decoder/gguf.go`, `decoder/weights.go`, `decoder/model.go`, `internal/prequant/prequant.go` (+
its test), `cmd/prequant/main.go`, `internal/serveapp/main.go`, `demo/chat/build-embed.sh`,
`metal/moe_pread_test.go`, plus one new test file
(`decoder/w4a8_row4_giwkind5_test.go`).

## L3 — Doc corrections (do with L1) (DONE 2026-09-11)

- `decoder/gguf.go:1388` and the branch's scope notes say the `.giw` path is out of scope
  "mirroring `repackW4A8Row4IfEligible`'s 'deliberately NOT wired into the .giw loader'
  precedent" and that "the existing canonical+row4 both policy isn't wired into `.giw` loading".
  The second claim is false — kind 4 *is* the both policy on disk, loaded at
  `decoder/serialize.go:1499`. The true precedent is that the **in-RAM** repack is not applied to a
  mmap'd `.giw`. Replace both with a pointer to L2.
- `internal/prequant/prequant.go:36–47` comment: "always emits kind 3" becomes the L2 target
  rule when L2 lands; until then add one line saying the CPU cache is on the canonical kernel.

**Findings.** The false "isn't wired into .giw loading" claim was mine — introduced this same
session while writing L1 (`quantizeEmbedWM`'s original doc comment, since rewritten). It existed
in exactly one place by the time this item ran (`decoder/gguf.go`'s two `buildWeightsFromGGUF`
call-site comments, `needsResidentSerialize` branch and the `giwWriter` branch just below it) —
both now point at L2 instead of repeating the false generalization. The ORIGINAL, pre-existing
`repackW4A8Row4IfEligible` comment (`decoder/weightmat.go:~234`, "Deliberately NOT wired into the
.giw loader") was re-read against this item's claim and found accurate as written — it is
specifically about the in-RAM repack function, gives the real reason (paging/`madvise` pageability
would be pinned by a heap-resident row4 copy sitting next to a pageable mmap alias), and was not
touched. `internal/prequant/prequant.go`'s comment got the one line requested, pointing at L2 by
name rather than restating the rule speculatively.

**Size.** Ten minutes — accurate; this took about that.

## L4 — Metal-resident int4 exists three times in the same RAM (FILED, not scheduled)

On the M1 Pro, unified memory means the Metal buffer and the host copies share physical RAM. A
GGUF-loaded int4 tensor on a Metal-resident box is canonical (heap) + row4 (heap,
`repackW4A8Row4IfEligible`) + the Metal upload. The CPU copies exist only for the staged
fallback (LoRA/session paths, `task-gpu-paths` G3/G4). Whether to skip the row4 repack on a
resident box, or release the CPU copies after a successful upload, is a residency-policy
decision with its own measurement — file here, do not do under L1/L2.

## L5 — amd64 split-half-only (DEFERRED until split-half is un-parked)

Same shape as L1/L2 for `cpu-amd64`: `RepackInt4SplitHalfInPlace` / `WrapInt4SplitHalfOnly` at
load; kind 6 on disk (split-half nibbles + canonical scales, which split-half shares). Nothing
to do until the parked amd64 split-half feature is back; when it is, the nobara box is the
measuring box.
