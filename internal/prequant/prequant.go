// Package prequant builds a goinfer prequant bundle (.giw) from a GGUF model: it
// loads the model at a fixed quant, streams the already-quantized resident weights
// to disk, and pairs them with a metadata-only GGUF (the source truncated at the
// tensor-data boundary) that carries the tokenizer. Shared by cmd/prequant (the
// explicit CLI) and the serve-side transparent cache (EnsureCachedGIW), so a user
// who points --stream-weights at a plain .gguf never has to run prequant by hand.
package prequant

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/giw"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// metaPrefixCap bounds how much of the source GGUF we read to extract the tokenizer
// half. The metadata + tensor directory live at the front of the file (KVs + tensor
// infos, no weight bytes), a few MB even for a 256-expert MoE — so reading the whole
// multi-GB model just to slice its header wastes that much heap (and OOMs on a 35B).
// 64 MiB is comfortably more than any real header.
const metaPrefixCap = 64 << 20

// freeDiskBytes is indirected (decoder/fitguard.go's own hostRAM/hostRAMAvailable convention)
// so a test can inject a disk's worth of free space instead of needing one — statfsFreeBytes
// (diskspace_unix.go / diskspace_other.go) is the real platform probe.
var freeDiskBytes = statfsFreeBytes

// Transcode writes a .giw bundle at out from the model at in, quantized to quant
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
// other target, including decoder.GIWTargetNone, writes kind 3 for every int4 tensor.
// decoder.GIWTargetForBackend derives a target from a backend name (EnsureCachedGIW);
// decoder.ParseGIWTarget parses cmd/prequant's own -target flag. A cpu-arm64 target needs an arm64
// box (the repack functions are NEON-only in aikit); a shape the repack rejects, or a non-arm64
// build, falls back to kind 3 for that tensor, so any target is always safe to pass.
func Transcode(ctx context.Context, in, out, quant string, embedInt4 bool, target decoder.GIWTarget) error {
	if fi, err := os.Stat(in); err == nil && fi.IsDir() {
		return transcodeDir(ctx, in, "", out, quant, embedInt4, target)
	}
	// 1) Tokenizer half: the source GGUF truncated at the tensor-data boundary —
	// metadata + tensor infos, no weight bytes. Only the file's head is read.
	head, err := readHead(in, metaPrefixCap)
	if err != nil {
		return fmt.Errorf("read gguf head: %w", err)
	}
	prefix, err := metadataPrefixLen(head)
	if err != nil {
		return fmt.Errorf("locate gguf metadata (header > %d MB?): %w", metaPrefixCap>>20, err)
	}
	tokBytes := head[:prefix]
	if _, err := tokenizer.LoadGGUFBytes(tokBytes); err != nil {
		return fmt.Errorf("metadata GGUF does not load a tokenizer: %w", err)
	}

	// 2) Weights half: transcode the GGUF straight into the bundle, ONE LAYER at a time
	// (decoder.StreamTranscodeGGUF), so peak RAM is about one layer rather than the whole resident
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
	// The temp name MUST still end in ".giw": selfCheck calls decoder.Load(tmp, ...), whose only entry to
	// the bundle loader is strings.HasSuffix(dir, ".giw"). A plain `out + ".tmp"` fails selfCheck for every
	// GGUF Transcode.
	tmp := strings.TrimSuffix(out, ".giw") + ".tmp.giw"
	f, err := os.Create(tmp)
	if err != nil {
		return fmt.Errorf("create %s: %w", tmp, err)
	}
	werr := giw.WriteStream(f, tokBytes, func(w io.Writer) (int64, error) {
		return decoder.StreamTranscodeGGUF(ctx, in, w, quant, embedInt4, target, filepath.Base(in))
	})
	runtime.GC()
	if cerr := f.Close(); werr == nil {
		werr = cerr
	}
	if werr != nil {
		removeTempGIW(tmp)
		return fmt.Errorf("write bundle: %w", werr)
	}

	// 3) Verify the bundle round-trips through the real (mmap) load path — low RAM (file-backed
	// pages) but a full read of the file for its CRC, which is what makes it a real integrity
	// check and records the verified-marker carryVerifiedMarker carries over. On the TEMP file, so a
	// bundle that fails it never becomes the sidecar even for an instant.
	if err := selfCheck(tmp); err != nil {
		removeTempGIW(tmp)
		return fmt.Errorf("self-check: %w", err)
	}
	if err := os.Rename(tmp, out); err != nil {
		removeTempGIW(tmp)
		return fmt.Errorf("publish %s: %w", out, err)
	}
	carryVerifiedMarker(tmp, out)
	_ = os.Remove(loraSidecar(out)) // a plain bundle carries no adapter, whatever an earlier build at this path did
	return nil
}

// transcodeDir builds a .giw from a safetensors model DIRECTORY (no GGUF available — the
// path for Mellum2 and other safetensors-only models). It streams the weights into the
// bundle one layer at a time (decoder.StreamTranscodeDir; a LoRA merge or a family whose loader
// does not stream loads the whole model instead), and carries the dir's tokenizer.json
// verbatim as the tok half (the serve side loads it via tokenizer.LoadJSONBytes when the
// blob isn't GGUF metadata).
func transcodeDir(ctx context.Context, dir, lora, out, quant string, embedInt4 bool, target decoder.GIWTarget) error {
	if err := ctx.Err(); err != nil {
		return err
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
			return err
		}
		id += " + LoRA " + filepath.Base(filepath.Dir(lora)) + "/" + filepath.Base(lora)
	}
	// TEMP + RENAME, for the reason given in Transcode (the temp name must end in ".giw" too). This
	// path is the most exposed to an interrupted write, since the resident fallback holds the whole
	// quantized model in RAM while writing.
	tmp := strings.TrimSuffix(out, ".giw") + ".tmp.giw"
	f, err := os.Create(tmp)
	if err != nil {
		return fmt.Errorf("create %s: %w", tmp, err)
	}
	// Streamed one layer at a time (docs/tasks/task-prequant-dir-streaming-2026-10.md): peak memory about the globals plus
	// one layer, byte-identical to the resident transcode. A LoRA merge, and a family whose loader does not stream yet, take
	// the resident transcode instead.
	body := func(w io.Writer) (int64, error) { return residentDirBody(w, dir, lora, quant, embedInt4, target, id) }
	if lora == "" {
		body = func(w io.Writer) (int64, error) {
			n, serr := decoder.StreamTranscodeDir(ctx, dir, w, quant, embedInt4, target, id)
			if decoder.IsDirNoStream(serr) && n == 0 {
				fmt.Fprintf(os.Stderr, "prequant: streams: no (%v); building it resident\n", serr)
				return residentDirBody(w, dir, lora, quant, embedInt4, target, id)
			}
			return n, serr
		}
	}
	werr := giw.WriteStream(f, tokBytes, body)
	runtime.GC()
	if cerr := f.Close(); werr == nil {
		werr = cerr
	}
	if werr != nil {
		removeTempGIW(tmp)
		return fmt.Errorf("write bundle: %w", werr)
	}
	if err := selfCheck(tmp); err != nil {
		removeTempGIW(tmp)
		return fmt.Errorf("self-check: %w", err)
	}
	if err := os.Rename(tmp, out); err != nil {
		removeTempGIW(tmp)
		return fmt.Errorf("publish %s: %w", out, err)
	}
	carryVerifiedMarker(tmp, out)
	// The sidecar is published after the bundle, so a bundle never claims an adapter it was not built with; a crash in
	// between leaves a bundle without its sidecar, which a head refuses (AdapterLoRA) rather than trusts.
	if side == nil {
		_ = os.Remove(loraSidecar(out))
	} else if err := os.WriteFile(loraSidecar(out), side, 0o644); err != nil {
		return fmt.Errorf("write %s: %w", loraSidecar(out), err)
	}
	return nil
}

// residentDirBody is the resident directory transcode: Load the whole model at quant, then serialize it. A LoRA merge
// needs it (the merge works on the f32 projections), and so does a family whose safetensors loader does not stream; it is
// also G-DS1's reference for the streamed bytes.
func residentDirBody(w io.Writer, dir, lora, quant string, embedInt4 bool, target decoder.GIWTarget, id string) (int64, error) {
	// ResidentContext 1: a transcode runs no request, so it allocates no KV (selfCheck does the same).
	// Unpinned, the fit guard prices the model's full context and can refuse a model whose weights fit.
	m, err := decoder.Load(dir, decoder.Options{Quant: quant, EmbedInt4: embedInt4, LoRA: lora, ResidentContext: 1})
	if err != nil {
		return 0, fmt.Errorf("load %s (%s): %w", dir, quant, err)
	}
	defer m.Close()
	return decoder.SerializeWeightsToForTarget(w, m.Weights(), id, target)
}

// EnsureCachedGIW returns a .giw for the GGUF at ggufPath quantized to quant, promised to backend,
// transcoding once into a sidecar cache (alongside the GGUF, "<base>.<quant>.<target>.giw", or
// "<base>.<quant>.e4h.<target>.giw" when embedInt4 is set) when no fresh cache exists. The cache is
// fresh when it is newer than the source AND was built for the same target (the cache key carries the
// target, so a CPU-built cache is never reused by a Metal load) AND the same embedInt4 setting (a
// plain-head sidecar and an embed-int4 one are different bundles, never interchangeable; folding
// embedInt4 into the filename, not just the Transcode call, is what stops a stale plain-head cache
// being silently served). The one-time transcode is logged to stderr (it can take minutes and write
// tens of GB) so a slow first start isn't mistaken for a hang. Returns the .giw path to load.
func EnsureCachedGIW(ctx context.Context, ggufPath, quant, backend string, embedInt4 bool) (string, error) {
	target := decoder.GIWTargetForBackend(backend)
	cache := streamCachePath(ggufPath, quant, embedInt4, target)
	if cacheFresh(cache, ggufPath, quant) {
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
			if n, ok := decoder.GGUFFileBytes(ggufPath); ok {
				need = n // a split set's whole size, not its first shard's
			}
		}
		if free, ok := freeDiskBytes(filepath.Dir(cache)); ok && free < need {
			return "", fmt.Errorf("stream-weights: refusing to transcode %s — projected sidecar size ~%.1f GB exceeds %.1f GB free on this disk (a half-written sidecar on a full disk is worse than refusing up front); free some space and retry, or pass -direct-load to skip the sidecar entirely",
				filepath.Base(ggufPath), float64(need)/1e9, float64(free)/1e9)
		}
	}
	fmt.Fprintf(os.Stderr, "stream-weights: transcoding %s → %s (%s, one-time — minutes + ~model-size on disk)…\n",
		filepath.Base(ggufPath), filepath.Base(cache), quantLabel(quant))
	t0 := time.Now()
	if err := Transcode(ctx, ggufPath, cache, quant, embedInt4, target); err != nil {
		return "", err
	}
	if fi, e := os.Stat(cache); e == nil {
		fmt.Fprintf(os.Stderr, "stream-weights: cache ready (%.0f MB) in %s\n",
			float64(fi.Size())/1048576, time.Since(t0).Round(time.Second))
	}
	return cache, nil
}

// SidecarPathIfFresh returns the sidecar .giw for ggufPath at quant/backend's target and embedInt4
// setting, and true, ONLY when a fresh one already exists: it never transcodes. For a caller like `fit`
// (docs/tasks/task-never-swap-2026-09.md) where measuring is supposed to stay cheap: forcing a
// transcode just to check fit would cost as much as the resident build it avoids.
func SidecarPathIfFresh(ggufPath, quant, backend string, embedInt4 bool) (string, bool) {
	cache := streamCachePath(ggufPath, quant, embedInt4, decoder.GIWTargetForBackend(backend))
	if cacheFresh(cache, ggufPath, quant) {
		return cache, true
	}
	return "", false
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

// streamCachePath is the sidecar cache for a GGUF at a quant, embedInt4 setting and target:
// "<base>.<quant>.<target>.giw", or "<base>.<quant>.e4h.<target>.giw" when embedInt4 is set —
// GIWTargetNone spells as "canonical" rather than an empty segment, so the path stays
// unambiguous. The "e4h" segment (embed-int4-head) exists so a plain-head sidecar built before
// EmbedInt4 defaulted on is never mistaken for, or overwritten by, an embed-int4 one built after:
// they are different bundles at the same source and quant, and need different cache keys.
func streamCachePath(ggufPath, quant string, embedInt4 bool, target decoder.GIWTarget) string {
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
	label := quantLabel(quant)
	if embedInt4 {
		label += ".e4h"
	}
	return base + "." + label + "." + tgt + ".giw"
}

// cacheFresh reports whether cache exists, is newer than src, AND actually loads.
//
// mtime alone is not freshness: it cannot see a bundle that is truncated, written by an older writer,
// or missing a tensor a newer reader requires, all of which are newer than the source and all of which
// fail at load. So freshness ends with the question that matters, does it load?, using the same mmap
// load selfCheck uses. That load runs the bundle's whole-payload CRC, which reads every byte of the
// file; it is done once per (size, mtime) (decoder/giwverify.go), so a full pass is paid only the first
// time a sidecar is seen. A bundle that does not load is not fresh, and the caller rebuilds it instead
// of failing at boot with an error that names the .giw rather than the cause.
//
// Loading is not the whole answer either: an int4 bundle written before minInt4CacheGIWVersion DOES
// load, by converting, and would be kept forever (cacheNewer and selfCheck both pass). See
// cacheLayoutCurrent; quant is the cache's own quant, which decides whether that applies.
func cacheFresh(cache, src, quant string) bool {
	if !cacheNewer(cache, src) {
		return false
	}
	if v, ok := cacheLayoutCurrent(cache, quant); !ok {
		fmt.Fprintf(os.Stderr, "stream-weights: cache %s is format v%d; this build keeps int4 group scales "+
			"as binary16 (v%d) and would convert them on every load, holding both copies — rebuilding once\n",
			filepath.Base(cache), v, minInt4CacheGIWVersion)
		return false
	}
	if err := selfCheck(cache); err != nil {
		fmt.Fprintf(os.Stderr, "stream-weights: cache %s is newer than the source but does not "+
			"load (%v) — rebuilding\n", filepath.Base(cache), err)
		return false
	}
	return true
}

// minInt4CacheGIWVersion is the oldest weights-blob version whose int4 group scales an mmap load can
// ALIAS: v15 stores them as binary16, aikit's in-RAM form (decoder/serialize.go's giwVF16Scales). An
// older int4 bundle still loads, but the reader converts its f32 scales to a heap f16 copy that sits
// beside the file's pages on every load, and a sidecar that loads is judged fresh, so an upgraded box
// would pay it forever.
//
// Raise this only when an older file loads at a real cost, never merely because the format gained a
// kind: every bump here costs each user a one-time re-transcode.
const minInt4CacheGIWVersion = 15

// cacheLayoutCurrent reports whether cache's weights layout is one this build loads without
// converting, and the version it read. Only an int4-bearing quant ("int4", "int4mix") is held to
// minInt4CacheGIWVersion — v15 changed nothing else, so rebuilding an int8/f32 sidecar would cost
// a transcode for nothing. An unreadable header reports ok, leaving the verdict to selfCheck,
// which names the actual failure.
func cacheLayoutCurrent(cache, quant string) (uint32, bool) {
	if !strings.HasPrefix(quant, "int4") {
		return 0, true
	}
	v, err := giw.WeightsVersionFile(cache)
	if err != nil {
		return 0, true
	}
	return v, v >= minInt4CacheGIWVersion
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
		return decoder.GGUFShards(src)
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

func quantLabel(q string) string {
	if q == "" {
		return "f32"
	}
	return q
}

// selfCheck verifies a freshly written bundle loads through the real mmap path: the streamed weights
// deserialize, and the whole-payload CRC passes (a full read of the file; the pass is recorded so it is
// not repeated for an unchanged file, decoder/giwverify.go).
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
	m, err := decoder.Load(path, decoder.Options{Backend: "cpu", ResidentContext: 1})
	if err != nil {
		return err
	}
	return m.Close()
}

// removeTempGIW deletes a temp bundle and the verified-marker its self-check may have written.
func removeTempGIW(tmp string) {
	_ = os.Remove(tmp)
	_ = os.Remove(decoder.GIWVerifiedMarkerPath(tmp))
}

// carryVerifiedMarker moves a temp bundle's verified-marker to the bundle's final name, after the
// caller has os.Renamed tmp onto out (the rename stays at the call site because
// TestTranscode_writesViaTempThenRenames asserts it structurally). selfCheck ran the full CRC on
// tmp and recorded (size, mtime); a rename preserves both, so the marker is still true of out —
// moving it means the first real load of a fresh sidecar does not read the whole file a second
// time. Best-effort: a marker that fails to move only costs one pass, and none is left behind.
func carryVerifiedMarker(tmp, out string) {
	if err := os.Rename(decoder.GIWVerifiedMarkerPath(tmp), decoder.GIWVerifiedMarkerPath(out)); err != nil {
		_ = os.Remove(decoder.GIWVerifiedMarkerPath(tmp))
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
