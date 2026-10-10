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
