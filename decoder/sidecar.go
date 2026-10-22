// The .giw transcode and the sidecar cache: a GGUF or a safetensors directory loaded at a fixed quant, its quantized
// weights streamed to a bundle with the source's tokenizer half beside them. cmd/prequant runs it by hand, and Load runs
// it by default for a .gguf or a safetensors directory (sidecarRoute), as serve and chat do through internal/modelload.

package decoder

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// metaPrefixCap bounds how much of the source GGUF we read to extract the tokenizer
// half. The metadata + tensor directory live at the front of the file (KVs + tensor
// infos, no weight bytes), a few MB even for a 256-expert MoE — so reading the whole
// multi-GB model just to slice its header wastes that much heap (and OOMs on a 35B).
// 64 MiB is comfortably more than any real header.
const metaPrefixCap = 64 << 20

// freeDiskBytes is indirected (fitguard.go's own hostRAM/hostRAMAvailable convention)
// so a test can inject a disk's worth of free space instead of needing one — statfsFreeBytes
// (diskspace_unix.go / diskspace_other.go) is the real platform probe.
var freeDiskBytes = statfsFreeBytes

// TranscodeGIW writes a .giw bundle at out from the model at in, quantized to quant
// ("int8int8" | "int8" | "int4" | "" for f32). `in` may be a GGUF file (streamed one layer at a
// time, peak RAM about one layer: fits a 35B on a modest box) OR a safetensors model DIRECTORY (the
// path for safetensors-only models like Mellum2; see transcodeDir). A failed write removes the
// partial output. A cancelled ctx aborts a long streaming transcode at the next layer boundary and
// removes the partial output.
//
// target names the ONE consumer this bundle is promised to (docs/tasks/task-int4-layout-2026-09.md):
// on a cpu-arm64 target, every eligible int4 tensor writes kind 5 (row4-only: the on-disk arm64
// split-half, 4-row-interleaved layout, docs/completed/task-w4a8-neon-bandwidth.md) instead of kind 3
// (decoder/serialize.go's weightMat vs weightMatKind3Only decides which tensors are eligible); every
// other target, including GIWTargetNone, writes kind 3 for every int4 tensor.
// GIWTargetForBackend derives a target from a backend name (EnsureCachedGIW);
// ParseGIWTarget parses cmd/prequant's own -target flag. A cpu-arm64 target needs an arm64
// box (the repack functions are NEON-only in aikit); a shape the repack rejects, or a non-arm64
// build, falls back to kind 3 for that tensor, so any target is always safe to pass.
func TranscodeGIW(ctx context.Context, in, out, quant string, embedInt4 bool, target GIWTarget) error {
	_, err := TranscodeGIWTimed(ctx, in, out, quant, embedInt4, target)
	return err
}

// TranscodePhases is where a transcode's wall time went. Stream is building the bundle and writing it to disk; SelfCheck is
// loading it back through the mmap path, whole-file CRC included.
type TranscodePhases struct {
	Stream, SelfCheck time.Duration
}

// String is the phase line's tail: "stream 20.1s, self-check 0.9s".
func (p TranscodePhases) String() string {
	return fmt.Sprintf("stream %.1fs, self-check %.1fs", p.Stream.Seconds(), p.SelfCheck.Seconds())
}

// TranscodeGIWTimed is TranscodeGIW, and reports where the time went. A phase that did not finish reads zero.
func TranscodeGIWTimed(ctx context.Context, in, out, quant string, embedInt4 bool, target GIWTarget) (TranscodePhases, error) {
	if fi, err := os.Stat(in); err == nil && fi.IsDir() {
		return transcodeDir(ctx, in, "", out, quant, embedInt4, target)
	}
	var ph TranscodePhases
	// 1) Tokenizer half: the source GGUF truncated at the tensor-data boundary —
	// metadata + tensor infos, no weight bytes. Only the file's head is read.
	head, err := readHead(in, metaPrefixCap)
	if err != nil {
		return ph, fmt.Errorf("read gguf head: %w", err)
	}
	prefix, err := metadataPrefixLen(head)
	if err != nil {
		return ph, fmt.Errorf("locate gguf metadata (header > %d MB?): %w", metaPrefixCap>>20, err)
	}
	tokBytes := head[:prefix]
	if _, err := tokenizer.LoadGGUFBytes(tokBytes); err != nil {
		return ph, fmt.Errorf("metadata GGUF does not load a tokenizer: %w", err)
	}

	// 2) Weights half: transcode the GGUF straight into the bundle, ONE LAYER at a time
	// (StreamTranscodeGGUF), so peak RAM is about one layer rather than the whole resident
	// model: this is what lets a model larger than RAM be prequant'd.
	// TEMP + RENAME, not os.Create(out) directly. giw.WriteStream patches the body length placeholder at
	// the END, so a bundle whose write was interrupted has a ZERO length in its header, and the
	// interruptions that matter (SIGKILL, the OOM killer, power loss) run no cleanup. Written in place,
	// such a file has an mtime NEWER than the source and is judged "fresh" forever, so every later
	// `serve --stream-weights` fails at boot with "truncated bundle", naming the .giw rather than the
	// cause, until a human deletes it. With a temp file an interrupted run leaves out.tmp.giw and no
	// `out`, so the next run rebuilds; the rename is atomic within a directory, so `out` appears only once
	// the bytes are complete AND selfCheck has passed.
	//
	// The temp name MUST still end in ".giw": selfCheck calls Load(tmp, ...), whose only entry to
	// the bundle loader is strings.HasSuffix(dir, ".giw"). A plain `out + ".tmp"` fails selfCheck for every
	// GGUF TranscodeGIW.
	tmp := strings.TrimSuffix(out, ".giw") + ".tmp.giw"
	t0 := time.Now()
	f, err := os.Create(tmp)
	if err != nil {
		return ph, fmt.Errorf("create %s: %w", tmp, err)
	}
	werr := writeSidecarBundle(f, tokBytes, func(w io.Writer) (int64, error) {
		return StreamTranscodeGGUF(ctx, in, w, quant, embedInt4, target, filepath.Base(in))
	})
	runtime.GC()
	if cerr := f.Close(); werr == nil {
		werr = cerr
	}
	if werr != nil {
		removeTempGIW(tmp)
		return ph, fmt.Errorf("write bundle: %w", werr)
	}
	ph.Stream = time.Since(t0)

	// 3) Verify the bundle loads through the real (mmap) load path: it deserializes and its shapes validate. Whether
	// that load recomputes the whole-file CRC is markWritten's rule. On the TEMP file, so a bundle that fails it never
	// becomes the sidecar even for an instant.
	markWritten(tmp, in)
	if afterWriteForTest != nil {
		afterWriteForTest(tmp)
	}
	t1 := time.Now()
	if err := transcodeSelfCheck(tmp); err != nil {
		removeTempGIW(tmp)
		return ph, fmt.Errorf("self-check: %w", err)
	}
	ph.SelfCheck = time.Since(t1)
	if err := os.Rename(tmp, out); err != nil {
		removeTempGIW(tmp)
		return ph, fmt.Errorf("publish %s: %w", out, err)
	}
	carryVerifiedMarker(tmp, out)
	_ = os.Remove(loraSidecar(out)) // a plain bundle carries no adapter, whatever an earlier build at this path did
	return ph, nil
}

// transcodeDir builds a .giw from a safetensors model DIRECTORY (no GGUF available — the
// path for Mellum2 and other safetensors-only models). It streams the weights into the
// bundle one layer at a time (StreamTranscodeDir; a LoRA merge or a family whose loader
// does not stream loads the whole model instead), and carries the dir's tokenizer.json
// verbatim as the tok half (the serve side loads it via tokenizer.LoadJSONBytes when the
// blob isn't GGUF metadata).
func transcodeDir(ctx context.Context, dir, lora, out, quant string, embedInt4 bool, target GIWTarget) (TranscodePhases, error) {
	var ph TranscodePhases
	if err := ctx.Err(); err != nil {
		return ph, err
	}
	// Tokenizer half is best-effort: the resident/decode path never reads it (the Model
	// is built from the serialized weights alone), so a missing or non-goinfer-loadable
	// tokenizer.json must NOT block the weights bundle — it only affects serve. Carry the
	// bytes when present; warn (don't fail) otherwise.
	var tokBytes []byte
	if b, rerr := os.ReadFile(filepath.Join(dir, "tokenizer.json")); rerr == nil {
		if _, lerr := tokenizer.LoadJSONBytes(b); lerr != nil {
			fmt.Fprintf(os.Stderr, "prequant: note: %s/tokenizer.json present but not goinfer-loadable (%v); bundle carries it, but serve may need the original tokenizer\n", dir, lerr)
		}
		tokBytes = b
	} else {
		fmt.Fprintf(os.Stderr, "prequant: note: no tokenizer.json in %s — weights-only bundle (serve needs a separate tokenizer)\n", dir)
	}
	id := filepath.Base(dir)
	var side []byte
	if lora != "" {
		var err error
		if side, err = loraSidecarBytes(lora); err != nil {
			return ph, err
		}
		id += " + LoRA " + filepath.Base(filepath.Dir(lora)) + "/" + filepath.Base(lora)
	}
	// TEMP + RENAME, for the reason given in TranscodeGIW (the temp name must end in ".giw" too). This
	// path is the most exposed to an interrupted write, since the resident fallback holds the whole
	// quantized model in RAM while writing.
	tmp := strings.TrimSuffix(out, ".giw") + ".tmp.giw"
	t0 := time.Now()
	f, err := os.Create(tmp)
	if err != nil {
		return ph, fmt.Errorf("create %s: %w", tmp, err)
	}
	// Streamed one layer at a time (docs/tasks/task-prequant-dir-streaming-2026-10.md): peak memory about the globals plus
	// one layer, byte-identical to the resident transcode. A LoRA merge, and a family whose loader does not stream yet, take
	// the resident transcode instead.
	body := func(w io.Writer) (int64, error) { return residentDirBody(w, dir, lora, quant, embedInt4, target, id) }
	if lora == "" {
		body = func(w io.Writer) (int64, error) {
			n, serr := StreamTranscodeDir(ctx, dir, w, quant, embedInt4, target, id)
			if IsDirNoStream(serr) && n == 0 {
				fmt.Fprintf(os.Stderr, "prequant: streams: no (%v); building it resident\n", serr)
				return residentDirBody(w, dir, lora, quant, embedInt4, target, id)
			}
			return n, serr
		}
	}
	werr := writeSidecarBundle(f, tokBytes, body)
	runtime.GC()
	if cerr := f.Close(); werr == nil {
		werr = cerr
	}
	if werr != nil {
		removeTempGIW(tmp)
		return ph, fmt.Errorf("write bundle: %w", werr)
	}
	ph.Stream = time.Since(t0)
	markWritten(tmp, dir)
	if afterWriteForTest != nil {
		afterWriteForTest(tmp)
	}
	t1 := time.Now()
	if err := transcodeSelfCheck(tmp); err != nil {
		removeTempGIW(tmp)
		return ph, fmt.Errorf("self-check: %w", err)
	}
	ph.SelfCheck = time.Since(t1)
	if err := os.Rename(tmp, out); err != nil {
		removeTempGIW(tmp)
		return ph, fmt.Errorf("publish %s: %w", out, err)
	}
	carryVerifiedMarker(tmp, out)
	// The sidecar is published after the bundle, so a bundle never claims an adapter it was not built with; a crash in
	// between leaves a bundle without its sidecar, which a head refuses (GIWAdapterLoRA) rather than trusts.
	if side == nil {
		_ = os.Remove(loraSidecar(out))
	} else if err := os.WriteFile(loraSidecar(out), side, 0o644); err != nil {
		return ph, fmt.Errorf("write %s: %w", loraSidecar(out), err)
	}
	return ph, nil
}

// residentDirBody is the resident directory transcode: Load the whole model at quant, then serialize it. A LoRA merge
// needs it (the merge works on the f32 projections), and so does a family whose safetensors loader does not stream; it is
// also G-DS1's reference for the streamed bytes.
func residentDirBody(w io.Writer, dir, lora, quant string, embedInt4 bool, target GIWTarget, id string) (int64, error) {
	// ResidentContext 1: a transcode runs no request, so it allocates no KV (selfCheck does the same).
	// Unpinned, the fit guard prices the model's full context and can refuse a model whose weights fit.
	m, err := Load(dir, Options{Quant: quant, EmbedInt4: embedInt4, LoRA: lora, ResidentContext: 1, DirectLoad: true})
	if err != nil {
		return 0, fmt.Errorf("load %s (%s): %w", dir, quant, err)
	}
	defer m.Close()
	return SerializeWeightsToForTarget(w, m.Weights(), id, target)
}

// EnsureCachedGIW returns a .giw for the GGUF at ggufPath quantized to quant, promised to backend,
// transcoding once into a sidecar cache (alongside the GGUF, "<base>.<quant>.<target>.v<N>.giw", or
// "<base>.<quant>.e4h.<target>.v<N>.giw" when embedInt4 is set; streamCachePath) when no fresh cache exists. The cache is
// fresh when it is newer than the source AND was built for the same target (the cache key carries the
// target, so a CPU-built cache is never reused by a Metal load) AND the same embedInt4 setting (a
// plain-head sidecar and an embed-int4 one are different bundles, never interchangeable; folding
// embedInt4 into the filename, not just the TranscodeGIW call, is what stops a stale plain-head cache
// being silently served). The one-time transcode is logged to stderr (it can take minutes and write
// tens of GB) so a slow first start isn't mistaken for a hang. Returns the .giw path to load.
func EnsureCachedGIW(ctx context.Context, ggufPath, quant, backend string, embedInt4 bool) (string, error) {
	target := GIWTargetForBackend(backend)
	cache := streamCachePath(ggufPath, quant, embedInt4, target)
	if freshCache(cache, legacyCachePath(ggufPath, quant, embedInt4, target), ggufPath) {
		return cache, nil
	}
	// Refuse up front when the sidecar will not fit: a half-written sidecar on a full disk is worse than
	// refusing. The size is projected from the source's tensor shapes at this quant
	// (projectedSidecarBytes), not taken from the source's own size, because sources are already quantized
	// and a sidecar can be larger than its source. freeDiskBytes returning !ok (no portable probe, or the
	// statfs itself failed) proceeds unguarded, as fitguard.go's rule says: a missing probe must never be
	// the reason a load that would have worked gets refused.
	if fi, serr := os.Stat(ggufPath); serr == nil {
		need, ok := projectedSidecarBytes(ggufPath, quant)
		if fi.IsDir() {
			need, ok = projectedDirSidecarBytes(ggufPath, quant)
		}
		if !ok {
			need = fi.Size() // header unreadable: the old proxy, better than none
			if n, ok := GGUFFileBytes(ggufPath); ok {
				need = n // a split set's whole size, not its first shard's
			}
		}
		if free, ok := freeDiskBytes(filepath.Dir(cache)); ok && free < need {
			return "", fmt.Errorf("stream-weights: refusing to transcode %s — projected sidecar size ~%.1f GB exceeds %.1f GB free on this disk (a half-written sidecar on a full disk is worse than refusing up front); free some space and retry, or pass -direct-load to skip the sidecar entirely",
				filepath.Base(ggufPath), float64(need)/1e9, float64(free)/1e9)
		}
	}
	fmt.Fprintf(os.Stderr, "stream-weights: transcoding %s → %s (%s, one-time — minutes + ~model-size on disk)…\n",
		filepath.Base(ggufPath), filepath.Base(cache), sidecarQuantLabel(quant))
	t0 := time.Now()
	ph, err := TranscodeGIWTimed(ctx, ggufPath, cache, quant, embedInt4, target)
	if err != nil {
		return "", err
	}
	if fi, e := os.Stat(cache); e == nil {
		fmt.Fprintf(os.Stderr, "stream-weights: cache ready (%.0f MB) in %s: %s\n",
			float64(fi.Size())/1048576, time.Since(t0).Round(time.Second), ph)
	}
	if lf, e := os.Stat(legacyCachePath(ggufPath, quant, embedInt4, target)); e == nil {
		fmt.Fprintf(os.Stderr, "stream-weights: note: %s (%.0f MB) is from a build that wrote unversioned sidecar names; this build leaves it alone. Delete it once no such build is in use.\n",
			filepath.Base(legacyCachePath(ggufPath, quant, embedInt4, target)), float64(lf.Size())/1048576)
	}
	return cache, nil
}

// SidecarPathIfFresh returns the sidecar .giw for ggufPath at quant/backend's target and embedInt4
// setting, and true, ONLY when a fresh one already exists: it never transcodes. For a caller like `fit`
// (docs/tasks/task-never-swap-2026-09.md) where measuring is supposed to stay cheap: forcing a
// transcode just to check fit would cost as much as the resident build it avoids.
func SidecarPathIfFresh(ggufPath, quant, backend string, embedInt4 bool) (string, bool) {
	target := GIWTargetForBackend(backend)
	cache := streamCachePath(ggufPath, quant, embedInt4, target)
	if freshCache(cache, legacyCachePath(ggufPath, quant, embedInt4, target), ggufPath) {
		return cache, true
	}
	return "", false
}

// SidecarState is what SidecarPathIfCurrent found.
type SidecarState int

const (
	SidecarMissing SidecarState = iota // no sidecar, a stale one, or one in another format: the load builds it
	SidecarCurrent                     // the sidecar the load opens, as far as its header can tell
	SidecarLegacy                      // only a legacy-named one, which the load adopts if it loads, else rebuilds
)

// SidecarPathIfCurrent is SidecarPathIfFresh without the load: the sidecar exists, is newer than the source, and its
// header is this build's format (GIWHeaderCheck). It never loads the bundle, so it does not prove the bundle loads:
// a truncated payload or a missing tensor is found by the load's own freshness check, which then rebuilds it. For a
// caller that must not load (POST /admin/models/fit).
func SidecarPathIfCurrent(src, quant, backend string, embedInt4 bool) (string, SidecarState) {
	target := GIWTargetForBackend(backend)
	cache := streamCachePath(src, quant, embedInt4, target)
	if cacheNewer(cache, src) && GIWHeaderCheck(cache) == nil {
		return cache, SidecarCurrent
	}
	if _, err := os.Stat(cache); err != nil && cacheNewer(legacyCachePath(src, quant, embedInt4, target), src) {
		return cache, SidecarLegacy
	}
	return cache, SidecarMissing
}

// SidecarPath is the sidecar .giw a load of ggufPath (a .gguf or a safetensors directory) at quant/backend's target and
// embedInt4 setting opens, or builds first when it is not there or not fresh (EnsureCachedGIW). It reads nothing.
func SidecarPath(ggufPath, quant, backend string, embedInt4 bool) string {
	return streamCachePath(ggufPath, quant, embedInt4, GIWTargetForBackend(backend))
}

// DefaultToSidecar reports whether a .gguf source should resolve to its sidecar .giw by default: the
// .gguf direct heap load is the opt-out, not the default. True on darwin (where the swap incidents
// happened) and on linux (the sidecar showed no CPU decode cost and a load that maps instead of
// re-quantizing; docs/measurements/cpu-giw-vs-direct-2026-09-24.md); other platforms keep the direct
// load. directLoad is the caller's already-resolved -direct-load flag / GOINFER_GGUF_DIRECT env var;
// it can only turn the sidecar OFF.
func DefaultToSidecar(directLoad bool) bool {
	if directLoad {
		return false
	}
	return runtime.GOOS == "darwin" || runtime.GOOS == "linux"
}

// sidecarInTests is whether Load takes the sidecar default inside a test binary. Off: a test that loads a .gguf or a
// safetensors directory keeps testing that load, and writes nothing beside its fixture or checkpoint. A test of the
// default itself turns it on.
var sidecarInTests = false

// sidecarRoute is what Load opens for src: src itself, or the sidecar .giw the apps open for it (internal/modelload
// routes by the same rules). A .gguf whose sidecar cannot be built is the load's error, naming Options.DirectLoad as the
// apps' refusal names -direct-load; a safetensors directory falls back to the direct load with a note. q4k has no .giw
// form, so a q4k .gguf loads directly.
func sidecarRoute(src string, opts Options) (string, error) {
	if opts.DirectLoad || opts.noSelfTest || opts.planResidency || testing.Testing() && !sidecarInTests {
		return src, nil
	}
	backend := opts.Backend
	if backend == "" {
		backend = "cpu" // Options.Backend's default; GIWTargetForBackend reads "" as no target
	}
	ensure := func() (string, error) {
		return EnsureCachedGIW(context.Background(), src, opts.Quant, backend, opts.EmbedInt4)
	}
	gguf := strings.HasSuffix(src, ".gguf")
	switch {
	case gguf && opts.Quant == "q4k":
		return src, nil
	case gguf && (opts.StreamWeights || DefaultToSidecar(false)):
		p, err := ensure()
		if err != nil {
			return "", fmt.Errorf("decoder: sidecar .giw for %s: %w — set Options.DirectLoad to load this .gguf straight into the heap instead", src, err)
		}
		return p, nil
	case DirSidecarApplies(src, opts) && DefaultToSidecar(false):
		p, err := ensure()
		if err != nil {
			fmt.Fprintf(os.Stderr, "decoder: loading %s directly: its sidecar .giw could not be built (%v)\n", src, err)
			return src, nil
		}
		return p, nil
	}
	return src, nil
}

// DirSidecarApplies reports whether a safetensors directory source takes the sidecar default: not with a LoRA to merge
// (the merge needs the safetensors base, and a plain sidecar would drop the adapter), not at q4k, bf16 or f16 (no .giw
// form), and not under StreamWeights, which streams only a .gguf's sidecar.
func DirSidecarApplies(src string, opts Options) bool {
	if opts.LoRA != "" || opts.Quant == "q4k" || opts.Quant == "bf16" || opts.Quant == "f16" || opts.StreamWeights || !isDir(src) {
		return false
	}
	m, _ := filepath.Glob(filepath.Join(src, "*.safetensors"))
	return len(m) > 0
}

// streamCachePath is the sidecar cache for a GGUF at a quant, embedInt4 setting and target:
// "<base>.<quant>.<target>.v<N>.giw", or "<base>.<quant>.e4h.<target>.v<N>.giw" when embedInt4 is set, N being the
// weights format version this build reads (GIWFormatVersion). The version is in the name so two builds at
// different formats on one box keep a sidecar each instead of rewriting one file in turn (a build that reads only vN
// refuses a v(N+1) file as too new and replaces it, and the other then does the same). GIWTargetNone spells as
// "canonical" rather than an empty segment, so the path stays unambiguous. The "e4h" segment (embed-int4-head)
// exists so a plain-head sidecar built before EmbedInt4 defaulted on is never mistaken for, or overwritten by, an
// embed-int4 one built after: they are different bundles at the same source and quant, and need different cache keys.
func streamCachePath(ggufPath, quant string, embedInt4 bool, target GIWTarget) string {
	return strings.TrimSuffix(legacyCachePath(ggufPath, quant, embedInt4, target), ".giw") + fmt.Sprintf(".v%d.giw", GIWFormatVersion)
}

// legacyCachePath is the sidecar name builds before the version was in it wrote: "<base>.<quant>[.e4h].<target>.giw".
// A build from before the change still reads and rewrites that name, so this build never writes or deletes it; it only
// adopts a file there that already loads as this build's format (adoptLegacy).
func legacyCachePath(ggufPath, quant string, embedInt4 bool, target GIWTarget) string {
	base := ggufPath[:len(ggufPath)-len(filepath.Ext(ggufPath))]
	if isDir(ggufPath) {
		// A safetensors directory: the sidecar sits beside it, named after the whole directory. Its name is not a
		// file name with an extension, and "qwen2.5-0.5b-instruct" would otherwise lose ".5-0.5b-instruct".
		base = filepath.Clean(ggufPath)
	}
	tgt := string(target)
	if tgt == "" {
		tgt = "canonical"
	}
	label := sidecarQuantLabel(quant)
	if embedInt4 {
		label += ".e4h"
	}
	return base + "." + label + "." + tgt + ".giw"
}

// freshCache reports whether the sidecar at cache is fresh for src, adopting the legacy-named file first when this
// build has none of its own: a legacy file that is newer than src and loads under this build's format is already
// this build's sidecar, so it is hard-linked to the versioned name (no copy, no transcode). The old name stays, for
// the builds that still use it, and an older build rewriting it by temp-and-rename leaves this link untouched. A
// legacy file in another format, or stale, is left alone, silently: the one that matters is the versioned name.
func freshCache(cache, legacy, src string) bool {
	if cacheFresh(cache, src) {
		return true
	}
	return adoptLegacy(cache, legacy, src) && cacheFresh(cache, src)
}

func adoptLegacy(cache, legacy, src string) bool {
	if _, err := os.Stat(cache); err == nil {
		return false // a versioned file exists and was refused: it is rebuilt, not replaced by an older name
	}
	if !cacheNewer(legacy, src) || selfCheck(legacy) != nil {
		return false
	}
	if err := os.Link(legacy, cache); err != nil {
		return false
	}
	_ = os.Link(GIWVerifiedMarkerPath(legacy), GIWVerifiedMarkerPath(cache)) // best effort: the marker keeps the first load off a whole-file CRC
	fmt.Fprintf(os.Stderr, "stream-weights: using %s as %s (the same file: already this build's format)\n", filepath.Base(legacy), filepath.Base(cache))
	return true
}

// cacheFresh reports whether cache exists, is newer than src, AND actually loads.
//
// mtime alone is not freshness: it cannot see a bundle that is truncated, written by an older writer,
// or missing a tensor a newer reader requires, all of which are newer than the source and all of which
// fail at load. So freshness ends with the question that matters, does it load?, using the same mmap
// load selfCheck uses. For a bundle with no verified-marker that load runs the whole-payload CRC, which
// reads every byte of the file; it is done once per (size, mtime) (decoder/giwverify.go), and a bundle
// this box transcoded and read back is marked when it is written (markWritten), so a full pass is paid only
// the first time a sidecar from elsewhere is seen. A bundle that does not load is not fresh, and the caller rebuilds it instead
// of failing at boot with an error that names the .giw rather than the cause.
//
// A sidecar in an older format is one of those: the reader reads exactly one version (decoder/serialize.go), so
// it fails here on its header, before any of the file is read, and is rebuilt once.
func cacheFresh(cache, src string) bool {
	if !cacheNewer(cache, src) {
		return false
	}
	if err := selfCheck(cache); err != nil {
		fmt.Fprintf(os.Stderr, "stream-weights: cache %s is newer than the source but does not "+
			"load (%v) — rebuilding\n", filepath.Base(cache), err)
		return false
	}
	return true
}

// cacheNewer is the mtime half of freshness, kept separate so each half can be tested for what
// it actually decides: this one answers "has the source changed since the cache was built",
// which is all an mtime can answer.
func cacheNewer(cache, src string) bool {
	cs, err := os.Stat(cache)
	if err != nil {
		return false
	}
	// A split GGUF is as new as its newest shard: a re-pulled shard 2 makes the sidecar stale as surely as a new
	// shard 1 does. A single file is its own one-element set.
	srcs, err := sourceFiles(src)
	if err != nil {
		return false
	}
	for _, p := range srcs {
		ss, err := os.Stat(p)
		if err != nil || !cs.ModTime().After(ss.ModTime()) {
			return false
		}
	}
	return true
}

// sourceFiles is what a sidecar's freshness is judged against: every shard of a split GGUF, a single file itself, or
// every regular file at the top of a safetensors directory. A directory's own mtime moves only when an entry is
// added, removed or renamed, so a safetensors file rewritten in place would leave a stale sidecar judged fresh.
func sourceFiles(src string) ([]string, error) {
	if !isDir(src) {
		return GGUFShards(src)
	}
	ents, err := os.ReadDir(src)
	if err != nil {
		return nil, err
	}
	var out []string
	for _, e := range ents {
		if e.Type().IsRegular() && !strings.HasPrefix(e.Name(), ".") {
			out = append(out, filepath.Join(src, e.Name()))
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("%s: no files", src)
	}
	return out, nil
}

func isDir(p string) bool {
	fi, err := os.Stat(p)
	return err == nil && fi.IsDir()
}

func sidecarQuantLabel(q string) string {
	if q == "" {
		return "f32"
	}
	return q
}

// selfCheck verifies a bundle loads through the real mmap path: the weights deserialize and their shapes validate
// against the architecture. Whether the whole-payload CRC is recomputed is the load's own rule (decoder/giwverify.go):
// it is for a file with no marker, such as a cache found on disk or a bundle markWritten left unmarked, and it is not
// for a bundle the transcode has marked.
//
// Backend:"cpu", not Options{}: an EMPTY Backend means "needs canonical" (wantsCanonicalInt4's
// literal-"cpu"-is-a-promise rule), so Options{} would decline every kind-5 (row4-only) bundle this
// function itself wrote for a cpu-arm64 target. "cpu" accepts both kind 3 and kind 5 and matches how a
// cpu-arm64-target bundle is loaded in production
// (docs/tasks/task-int4-layout-2026-09.md).
//
// ResidentContext 1: the check runs no request, so it allocates no KV. Unpinned, the host fit guard
// prices the CPU's per-request ceiling over the model's whole window, prints a "context capped" line
// belonging to no real load, and under tight memory can refuse the check outright.
func selfCheck(path string) error {
	m, err := Load(path, Options{Backend: "cpu", ResidentContext: 1})
	if err != nil {
		return err
	}
	return m.Close()
}

// transcodeSelfCheck is the self-check a transcode runs on the bundle it has just written. A variable so a test can
// make the span it is timed over longer than the clock's resolution. Set in init: selfCheck reaches Load, which reaches the
// transcode, so a package-level initializer would be a cycle.
var transcodeSelfCheck func(path string) error

func init() { transcodeSelfCheck = selfCheck }

// afterWriteForTest, when set, is called with the temp bundle once it is written and markWritten has run, before its
// self-check.
var afterWriteForTest func(tmp string)

// removeTempGIW deletes a temp bundle and its verified-marker.
func removeTempGIW(tmp string) {
	_ = os.Remove(tmp)
	_ = os.Remove(GIWVerifiedMarkerPath(tmp))
}

// carryVerifiedMarker moves a temp bundle's verified-marker to the bundle's final name, after the caller has os.Renamed
// tmp onto out (the rename stays at the call site because TestTranscode_writesViaTempThenRenames asserts it
// structurally). The transcode marked tmp at its (size, mtime) and a rename preserves both, so the marker is still true
// of out: the first real load of a fresh sidecar does not read the whole file. Best-effort: a marker that fails to move
// only costs that load one CRC pass, and none is left behind.
func carryVerifiedMarker(tmp, out string) {
	if err := os.Rename(GIWVerifiedMarkerPath(tmp), GIWVerifiedMarkerPath(out)); err != nil {
		_ = os.Remove(GIWVerifiedMarkerPath(tmp))
	}
}

// readHead reads up to capBytes from the front of path (a GGUF's metadata + tensor
// directory live at the front). Returns fewer bytes for a smaller file.
func readHead(path string, capBytes int) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}
	n := capBytes
	if sz := int(fi.Size()); sz < n {
		n = sz
	}
	buf := make([]byte, n)
	if _, err := io.ReadFull(f, buf); err != nil {
		return nil, err
	}
	return buf, nil
}
