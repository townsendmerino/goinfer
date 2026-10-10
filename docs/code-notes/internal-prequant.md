# internal/prequant: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `internal/prequant`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## Transcode: the weights half

Moved from `internal/prequant/prequant.go` (the comment in `Transcode`) on 2026-10-09.

```text
2) Weights half: transcode the GGUF straight into the bundle, ONE LAYER at a
time (decoder.StreamTranscodeGGUF), so peak RAM is ~one layer rather than the
whole resident model — this is what lets a model larger than RAM be prequant'd
(e.g. a 106B-A12B int4 on a 62 GB box). Every family streams now (S2): the resident
build some families used to fall back to is gone.
```

## Transcode: the temp file and its name

Moved from `internal/prequant/prequant.go` (the comment in `Transcode`) on 2026-10-09.

```text
TEMP + RENAME, not os.Create(out) directly (M-12). giw.WriteStream patches the body
length placeholder at the END, so a bundle whose write was interrupted has a ZERO length
in its header — and the error paths below cannot help, because the interruptions that
matter are the ones that run no cleanup: SIGKILL, the OOM killer, power loss. Written in
place, such a file exists, has an mtime NEWER than the source, and is therefore judged
"fresh" forever — so every subsequent `serve --stream-weights` fails at boot with
"truncated bundle", naming the .giw rather than the cause, until a human deletes it.

With a temp file, an interrupted run leaves out.tmp and no `out` at all, so the next run
simply rebuilds. The rename is atomic within a directory, so `out` only ever appears
once the bytes are complete AND selfCheck has passed.

The temp name MUST still end in ".giw" (V-01, docs/review-2026-09-04.md): selfCheck below
calls decoder.Load(tmp, ...), and Load's only entry to the bundle loader is
strings.HasSuffix(dir, ".giw") -- anything else falls to loadWeights, which wants a .gguf
file or a safetensors directory and finds neither. A plain `out + ".tmp"` (e.g.
"model.int4.giw.tmp") does not end in ".giw", so selfCheck failed for every GGUF Transcode
unconditionally, deleted the temp file, and returned "self-check: ..." -- the rename was
never reached. Boxes that already had a sidecar from before this bug never called
Transcode again and so never saw it, which is how it stayed green.
```

## transcodeDir: the temp file

Moved from `internal/prequant/prequant.go` (the comment in `transcodeDir`) on 2026-10-09.

```text
TEMP + RENAME, same reason as Transcode's GGUF branch above (M-12/M-33): a write to
`out` directly leaves a placeholder-length bundle behind on SIGKILL/OOM-kill/power
loss — exactly the interruption class this whole-model-resident path is most exposed
to, since it holds the entire quantized model in RAM while writing. Written in place,
that half-written file is newer than the source and "fresh" forever, so every later
`serve` fails at boot with "truncated bundle" until a human deletes it. The temp name
must still end in ".giw" (V-01) for selfCheck's decoder.Load to route to the bundle
loader at all.
```

## residentDirBody: ResidentContext

Moved from `internal/prequant/prequant.go` (the comment in `residentDirBody`) on 2026-10-09.

```text
ResidentContext 1: a transcode runs no request, so it allocates no KV (selfCheck does the same). Unpinned, the fit
guard priced the model's full context: on 2026-10-07 it refused Gemma 4 E4B for 8.8 GB of weights "+ 7.1 GB KV".
```

## EnsureCachedGIW: the free-disk check

Moved from `internal/prequant/prequant.go` (the comment in `EnsureCachedGIW`) on 2026-10-09.

```text
S1 (task-never-swap-2026-09.md): a half-written sidecar on a full disk is a failure this
repo has already had once (M-12's own history), so refuse up front when the sidecar will not
fit. The size is projected from the source's tensor shapes at this quant
(projectedSidecarBytes); it used to be the source's own size, on the argument that a sidecar is
never bigger than an f32 source — but sources are quantized, and real int4 sidecars measured
1.02–1.16× their q4_k_m source, int8int8 ~1.6×, so the check passed and the disk could still
fill. freeDiskBytes returning !ok (no portable probe, or the statfs itself failed) proceeds
unguarded, same as every other unknown quantity in this codebase (fitguard.go's own rule) — a
missing probe must never be the reason a load that would have worked gets refused.
```

## SidecarPathIfFresh

Moved from `internal/prequant/prequant.go` (the comment above `SidecarPathIfFresh`) on 2026-10-09.

```text
SidecarPathIfFresh returns the sidecar .giw for ggufPath at quant/backend's target and
embedInt4 setting, and true, ONLY when a fresh one already exists — it never transcodes. For a
caller like `fit` (task-never-swap-2026-09.md S1 item 5) where measuring is supposed to stay
cheap; forcing a transcode just to check fit would trade a 32 s / 256 CPU-s resident build for
an equally expensive one-time transcode, not for "nearly free" as the brief asks.
```

## DefaultToSidecar

Moved from `internal/prequant/prequant.go` (the comment above `DefaultToSidecar`) on 2026-10-09.

```text
DefaultToSidecar reports whether a .gguf source should resolve to its sidecar .giw by default
— S1's own registered rule: "the .gguf direct heap load becomes the opt-out, not the default".
darwin since S1 (2026-09-22), where the historical swap incidents (gpt-oss-20b, M35/M26)
happened; linux since 2026-09-24, owner decision, after docs/measurements/cpu-giw-vs-direct-2026-09-24.md
found no CPU decode cost (1.0007x / 1.0016x on 1.5B / 7B, inside the A/A arm) and a load that
maps instead of re-quantizing (0.01 s and ~0 heap vs 5.6-16.5 s and 1.3-5 GB). Other platforms
keep the direct load. directLoad is the caller's already-resolved -direct-load flag /
GOINFER_GGUF_DIRECT env var; it can only turn the sidecar OFF.
```

## cacheFresh

Moved from `internal/prequant/prequant.go` (the comment above `cacheFresh`) on 2026-10-09.

```text
cacheFresh reports whether cache exists, is newer than src, AND actually loads.

mtime alone is not freshness (M-12). It cannot see a bundle that is truncated, written by an
older writer, or missing a tensor a newer reader requires — all of which are newer than the
source and all of which fail at load. That is also M-11's trigger: a pre-v6 gpt-oss sidecar
is "fresh" by mtime, passes validateShapes, and panics at the first forward.

So freshness ends with the question that actually matters — does it load? — using the same
mmap load selfCheck uses. That load runs the bundle's whole-payload CRC, which reads every byte
of the file; it used to be described here as lazy and cheap, and was not. It is now done once
per (size, mtime) (decoder/giwverify.go), so this check costs a full pass only the first time a
sidecar is seen. A bundle that does not load is not fresh, and the caller rebuilds it instead of
failing at boot with an error that names the .giw rather than the cause.

Loading is not the whole answer either: an int4 bundle written before minInt4CacheGIWVersion
DOES load, by converting, and would be kept forever (cacheNewer + selfCheck both pass). See
cacheLayoutCurrent — quant is the cache's own quant, which decides whether that applies.
```

## minInt4CacheGIWVersion

Moved from `internal/prequant/prequant.go` (the comment above `minInt4CacheGIWVersion`) on 2026-10-09.

```text
minInt4CacheGIWVersion is the oldest weights-blob version whose int4 group scales an mmap load
can ALIAS: v15 stores them as binary16, aikit v1.50.0's in-RAM form (decoder/serialize.go's
giwVF16Scales). An older int4 bundle still loads — the reader converts its f32 scales to a heap
f16 copy — but that copy sits beside the file's f32 pages on every load, and a sidecar that loads
was judged fresh, so an upgraded box kept paying it forever. Measured 2026-09-28 (1.5B CPU, peak
RSS): old build on v12 1,123-1,127 MB, new build on the same v12 1,179-1,184 MB, new build on
its own v15 1,045-1,053 MB; the Mac's +478-589 MB on the 7B was the middle row.

Raise this only when an older file loads at a real cost, never merely because the format gained
a kind: every bump here costs each user a one-time re-transcode.
```

## selfCheck

Moved from `internal/prequant/prequant.go` (the comment above `selfCheck`) on 2026-10-09.

```text
selfCheck verifies a freshly written bundle loads through the real mmap path — the streamed
weights deserialize, and the whole-payload CRC passes (a full read of the file; the pass is
recorded so it is not repeated for an unchanged file — decoder/giwverify.go).

Backend:"cpu", not Options{} (found writing L2, docs/tasks/task-int4-layout-2026-09.md):
an EMPTY Backend means "needs canonical" (wantsCanonicalInt4's own literal-"cpu"-
is-a-promise rule, L1), so Options{} declined every kind-5 (row4-only) bundle
this function itself just wrote for a cpu-arm64 target — self-check would have
failed every -target cpu-arm64 transcode. "cpu" accepts both kind 3 and kind 5
(the plain CPU backend implements none of wantsCanonicalInt4's interfaces, so it
never needs canonical either way) and matches what a real cpu-arm64-target
bundle is actually loaded with in production.

ResidentContext 1: the check runs no request, so it allocates no KV. Unpinned, the host fit guard priced it at the
CPU's per-request ceiling (f32 over the model's whole window) and, on every sidecar check of a big-window model,
printed a "context capped" line belonging to no real load; under tight memory it could refuse the check outright
(A3, docs/completed/task-audit-followups-2026-10-06.md).
```

## projectedSidecarBytes

Moved from `internal/prequant/projected.go` (the comment above `projectedSidecarBytes`) on 2026-10-09.

```text
projectedSidecarBytes is the size a sidecar built at quant will have, projected from the source
GGUF's tensor shapes (a header read — no weights are paged in). The pre-transcode disk check used
the SOURCE's size as its proxy, on the argument that a sidecar is never bigger than an f32 source;
but the sources are already quantized, and real int4 sidecars measured 1.02–1.16× their q4_k_m
source, int8int8 about 1.6× — so the check passed and the transcode could still run out of disk.
```

## TestGIWRoundTripPreservesRouterBias

Moved from `internal/prequant/stream_test.go` (the comment above `TestGIWRoundTripPreservesRouterBias`) on 2026-10-09.

```text
TestGIWRoundTripPreservesRouterBias guards the .giw serialize format against
silently dropping per-layer fields. The byte-identity test above can't catch a
field BOTH the streamed and resident paths skip (they share the serializer); only
a transcode → load → inspect round-trip does. RouterBias (GLM/DeepSeek
e_score_correction_bias) was added to LayerWeights but initially not to the giw
format, so a stream-weights GLM lost its routing bias — caught only by the real
106B gate. This pins it on the tiny GLM model.
```

## residentLabelFor

Moved from `internal/prequant/stream_test.go` (the comment above `residentLabelFor`) on 2026-10-09.

```text
residentLabelFor is what decoder.(*Weights).quantLabel() actually resolves to for
testdata/glm-tiny.gguf's RESIDENT (buffer-path) bundle at a given requested quant — which is
NOT always what decoder.Model.Quant() reports for a live load, and callers here must not
assume it is. liveQuant is m.Quant()'s own answer, the correct value for every case except the
one M-27 changed.

M-27 (docs/audit-2026-09-10.md): before that fix, glm-tiny's router was (incorrectly)
quantized right alongside the rest of the body, so an "int4" load was uniformly int4 and
quantLabel() collapsed to "int4". Now the router correctly stays f32 regardless of the
ambient quant — a real, intentional precision difference from the int4 body — and
quantLabel()'s own documented contract (decoder/serialize.go: "int4mix... when int4 coexists
with a higher-precision BODY weight," which explicitly classifies the router as a body
weight) correctly reports that as "int4mix", not "int4". This was the bug being fixed, not a
fact worth re-pinning: decoder.Model.Quant() (used for a LIVE, non-.giw model) is unaffected
and still reports back exactly what was requested for "int4" (Model.quant short-circuits
before ever calling quantLabel()) — only the .giw bundle's own baked, inferred label changes,
honestly, because the bundle genuinely is no longer uniform-precision. Every other quant
mode (including "", where Model.Quant() already falls through to the real quantLabel()
inference rather than echoing the request) is unaffected and liveQuant is already correct.
```

## TestGiwQuantLabel_headerAsymmetry

Moved from `internal/prequant/stream_test.go` (the comment above `TestGiwQuantLabel_headerAsymmetry`) on 2026-10-09.

```text
TestGiwQuantLabel_headerAsymmetry pins a v5 behaviour that until now lived only as a comment in
decoder/serialize.go: the BUFFER path (full weights in hand) records the resolved quant label,
while the STREAMING transcode records "" — it writes the header BEFORE its layers load and cannot
yet know the resolved quant, so a reader of a streamed bundle falls back to inference (the pre-v5
behaviour).

This is a gate, not a nicety. That asymmetry is exactly what made the old byte-identity assertion
wrong the day ac6977f landed, and nothing caught it: testdata/glm-tiny.gguf was untracked, so the
test self-skipped in CI and had never once run there.
```

## TestStreamTranscodeMatchesResident.fields

Moved from `internal/prequant/stream_test.go` (the comment inside the byte-identity test comment) on 2026-10-09.

```text
Byte-identity is asserted over everything outside the two fields that MUST differ, and each of
those is pinned exactly rather than skipped: the v5 quant label (see the asymmetry test above;
giwSplit re-asserts it here) and the trailing CRC32 covering it (giwSplit verifies each bundle's
CRC independently, so a wrong CRC fails). Measured on this fixture, every remaining byte is equal
across all three quant modes — a size delta alone would not have shown that, and did not: the
original diagnosis missed the CRC entirely.
```

## TestTranscode_embedInt4Threaded

Moved from `internal/prequant/stream_test.go` (the comment above `TestTranscode_embedInt4Threaded`) on 2026-10-09.

```text
TestTranscode_embedInt4Threaded gates M-31: Transcode's GGUF branch hardcoded `false` for
StreamTranscodeGGUF's embedInt4 parameter instead of threading the one it was given, so
`cmd/prequant -embed-int4` silently produced an int8-pinned embed/head table for every GGUF
input — identical to omitting the flag, with no error. The safetensors-directory branch
(transcodeDir) already threaded it correctly; only the GGUF branch was broken.
```

## TestTranscode_realGGUFSucceedsAndPublishedBundleLoads

Moved from `internal/prequant/stream_test.go` (the comment above `TestTranscode_realGGUFSucceedsAndPublishedBundleLoads`) on 2026-10-09.

```text
V-01 (docs/review-2026-09-04.md): no existing test ever drove a SUCCESSFUL GGUF Transcode —
the M-12 tests above assert failure on a non-GGUF source and an AST shape, neither of which
exercises selfCheck on a real bundle. That is exactly why the temp name `out + ".tmp"` (not
ending in ".giw", so decoder.Load's suffix dispatch in selfCheck could never route to the
bundle loader) went unnoticed: every real Transcode failed its own self-check unconditionally.
This drives the whole function end to end on a real tokenizer-bearing GGUF (glm-tiny.gguf has
no tokenizer -- see giwFixture's own comment elsewhere -- so this needs a different, real small
checkpoint) and loads the PUBLISHED bundle afterward, proving both that Transcode succeeds and
that the file it left behind is loadable.
```

## TestTranscodeDir_writesViaTempThenRenames

Moved from `internal/prequant/stream_test.go` (the comment above `TestTranscodeDir_writesViaTempThenRenames`) on 2026-10-09.

```text
M-33 (audit-2026-09-10): transcodeDir — the safetensors-directory sibling of Transcode's GGUF
branch, used for safetensors-only families like Mellum2 — wrote straight to `out` via
os.Create(out) and removed `out` (not a temp file) on failure, reintroducing exactly the M-12
class of bug the GGUF branch above was already fixed for: an OOM-kill during the write (this
path holds the WHOLE resident model in RAM, exactly where a killer fires) leaves a
placeholder-length bundle at the final path, newer than its source, "fresh" forever, and
`serve` then fails at boot with "truncated bundle" until a human deletes it. Same structural
guard as TestTranscode_writesViaTempThenRenames, targeting transcodeDir instead.
```

## TestStreamTranscode_perFamilyBodiesCarryTheirLayers

Moved from `internal/prequant/stream_families_test.go` (the comment above `TestStreamTranscode_perFamilyBodiesCarryTheirLayers`) on 2026-10-09.

```text
M-09: StreamTranscodeGGUF wrote a HEADER-ONLY bundle for five GGUF families, and the comment
above canSerialize claimed they were refused before the load.

canSerialize has returned nil unconditionally since v6, so nothing was refused. The gpt-oss,
laguna, granitehybrid, nemotron_h/_moe and llama4 branches each build every layer and
`return w, nil` WITHOUT calling sink.layer — so the writer emitted a header declaring N
layers followed by zero layers. cmd/prequant and `serve --stream-weights` then load the whole
model resident (defeating the one-layer-peak-RAM contract this path exists for) and fail
minutes later with "truncated body: unexpected end of data": a broken supported path whose
error names the symptom and not the cause.

decoder/testdata/gptoss_tiny.gguf is COMMITTED and nothing drove the stream path on it —
stream_test.go uses glm-tiny, whose generic loader does stream. That gap is why this shipped,
so closing it is the test. Through StreamTranscodeGGUF directly, as the neighbouring tests
do: the tiny fixtures carry no tokenizer, so the full Transcode refuses before the weights.
```

## TestStreamTranscode_perFamilyBodiesCarryTheirLayers.gptoss

Moved from `internal/prequant/stream_families_test.go` (the comment inside the per-family table) on 2026-10-09.

```text
The regression, historically: a family routed through the resident-build
fallback. S2 (task-never-swap-2026-09.md, 2026-09-23) moved gpt-oss OFF that
fallback — its own loadGptOss closure already builds one layer independently of
every other, so it now streams natively too, same as glm below. `streams` is
this test's own record of that; TestGptOss_streamedMatchesResident is the byte-
identity gate that actually proves it (this test only proves the bundle isn't
header-only, not which path produced it).
```

## TestGptOss_streamedMatchesResident

Moved from `internal/prequant/stream_families_test.go` (the comment above `TestGptOss_streamedMatchesResident`) on 2026-10-09.

```text
TestGptOss_streamedMatchesResident is S2's own registered gate for the first family moved off
the resident-serialize fallback (needsResidentSerialize, deleted 2026-09-24 when gemma4 — the last
family on it — began streaming too): the streamed bundle must be byte-identical to the resident-build-then-
serialize path's output, same shape as stream_test.go's TestStreamTranscodeMatchesResident (glm)
— extended here per family rather than widening that one, since a failure on one fixture should
name which family broke, not force a reader to guess from a shared table's row count.
```

## TestProjectedSidecarBytes_neverUnderCounts

Moved from `internal/prequant/projected_test.go` (the comment above `TestProjectedSidecarBytes_neverUnderCounts`) on 2026-10-09.

```text
TestProjectedSidecarBytes_neverUnderCounts: the pre-transcode disk check must not pass a sidecar that
will not fit, so the projection has to be at least what the writer actually produces, at every quant.
(It used to be the source's size, which a real int4 sidecar exceeds by up to 16% and an int8int8 one by
~60%.) Measured against the writer on the tiny fixture here; against real sidecars when written:
0.5B/1.5B/7B/Llama-1B/Gemma-4-26B int4 and 0.5B int8int8 projected 1.03–1.14× their actual size.
```

## TestDirSidecar_keepsMRopeSection

Moved from `internal/prequant/dir_sidecar_test.go` (the comment above `TestDirSidecar_keepsMRopeSection`) on 2026-10-09.

```text
TestDirSidecar_keepsMRopeSection: a Qwen-VL's m-RoPE section survives the sidecar. Real checkpoints carry it in
rope_scaling, which the adapters read and clear, and Config.MRopeSection was `json:"-"`, so a .giw dropped it and the
model loaded from one ran plain RoPE on image positions (found 2026-10-09 by S16's real gate). The tiny fixtures carry
it in rope_parameters, a raw field that survives, which is why TestDirSidecar_matchesDirectLoad (text-only, where the
three axes coincide anyway) passed: each fixture here is copied with the section moved into rope_scaling, as the
released checkpoints have it. A config with no section at all is refused, which is what makes a stale sidecar fail its
self-check and rebuild.
```

## dirStreamDiff

Moved from `internal/prequant/dir_stream_test.go` (the comment above `dirStreamDiff`) on 2026-10-09.

```text
dirStreamDiff compares a resident and a streamed bundle as G-DS1 is amended (2026-10-09, the task doc): byte-identical
before the v5 quant label and after its alignment pad, both CRCs valid. The streamed bundle records the label as ""
because the header is written before any layer exists (B11, exactly as StreamTranscodeGGUF's bundles); a reader infers
it from the identical tensors. giwSplit checks the pad is zeros and the stored CRC matches each body.
```

## sidecar_default_test.header

Moved from `internal/prequant/sidecar_default_test.go` (the comment at the top of the file) on 2026-10-09.

```text
task-never-swap-2026-09.md S1: sidecar .giw by default on darwin. These gates cover the three
pieces this session added — SidecarPathIfFresh (fit's reuse-only-if-fresh path),
DefaultToSidecar (the platform policy), and the disk-space guard in EnsureCachedGIW — separate
from stream_test.go's existing Transcode/cacheFresh coverage, which these build on rather than
duplicate.
```
