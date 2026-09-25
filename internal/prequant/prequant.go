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
// ("int8int8" | "int8" | "int4" | "" for f32). `in` may be a GGUF file (streamed one
// layer at a time, peak RAM ≈ one layer — fits a 35B on a modest box) OR a safetensors
// model DIRECTORY (loaded whole then serialized, peak RAM ≈ the resident weight size —
// the path for safetensors-only models like Mellum2). A failed write removes the partial
// output. A cancelled ctx aborts a long streaming transcode at the next layer boundary
// (audit M-21) and removes the partial output.
//
// target names the ONE consumer this bundle is promised to (docs/task-int4-layout-
// 2026-09.md's L2): on a cpu-arm64 target, every eligible int4 tensor writes kind
// 5 (row4-only — the on-disk arm64 split-half + 4-row-interleaved layout,
// docs/completed/task-w4a8-neon-bandwidth.md's "Format follow-on") instead of kind 3
// (decoder/serialize.go's weightMat vs weightMatKind3Only decides which tensors
// are eligible, and why); every other target, including decoder.GIWTargetNone,
// writes kind 3 for every int4 tensor. decoder.GIWTargetForBackend derives a
// target from a backend name (EnsureCachedGIW, below); decoder.ParseGIWTarget
// parses cmd/prequant's own -target flag. Requires running on an arm64 box for a
// cpu-arm64 target (the repack functions are NEON-only in aikit); a shape the
// repack rejects, or a non-arm64 build, falls back to kind 3 automatically for
// that tensor — always safe to pass any target.
func Transcode(ctx context.Context, in, out, quant string, embedInt4 bool, target decoder.GIWTarget) error {
	if fi, err := os.Stat(in); err == nil && fi.IsDir() {
		return transcodeDir(ctx, in, out, quant, embedInt4, target)
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

	// 2) Weights half: transcode the GGUF straight into the bundle, ONE LAYER at a
	// time (decoder.StreamTranscodeGGUF), so peak RAM is ~one layer rather than the
	// whole resident model — this is what lets a model larger than RAM be prequant'd
	// (e.g. a 106B-A12B int4 on a 62 GB box). Every family streams now (S2): the resident
	// build some families used to fall back to is gone.
	// TEMP + RENAME, not os.Create(out) directly (M-12). giw.WriteStream patches the body
	// length placeholder at the END, so a bundle whose write was interrupted has a ZERO length
	// in its header — and the error paths below cannot help, because the interruptions that
	// matter are the ones that run no cleanup: SIGKILL, the OOM killer, power loss. Written in
	// place, such a file exists, has an mtime NEWER than the source, and is therefore judged
	// "fresh" forever — so every subsequent `serve --stream-weights` fails at boot with
	// "truncated bundle", naming the .giw rather than the cause, until a human deletes it.
	//
	// With a temp file, an interrupted run leaves out.tmp and no `out` at all, so the next run
	// simply rebuilds. The rename is atomic within a directory, so `out` only ever appears
	// once the bytes are complete AND selfCheck has passed.
	//
	// The temp name MUST still end in ".giw" (V-01, docs/review-2026-09-04.md): selfCheck below
	// calls decoder.Load(tmp, ...), and Load's only entry to the bundle loader is
	// strings.HasSuffix(dir, ".giw") -- anything else falls to loadWeights, which wants a .gguf
	// file or a safetensors directory and finds neither. A plain `out + ".tmp"` (e.g.
	// "model.int4.giw.tmp") does not end in ".giw", so selfCheck failed for every GGUF Transcode
	// unconditionally, deleted the temp file, and returned "self-check: ..." -- the rename was
	// never reached. Boxes that already had a sidecar from before this bug never called
	// Transcode again and so never saw it, which is how it stayed green.
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
	return nil
}

// transcodeDir builds a .giw from a safetensors model DIRECTORY (no GGUF available — the
// path for Mellum2 and other safetensors-only models). It loads the model at `quant`,
// serializes the resident weights into the bundle, and carries the dir's tokenizer.json
// verbatim as the tok half (the serve side loads it via tokenizer.LoadJSONBytes when the
// blob isn't GGUF metadata). Peak RAM ≈ the resident weight size, since the whole model
// is loaded rather than layer-streamed — acceptable for the models this targets.
func transcodeDir(ctx context.Context, dir, out, quant string, embedInt4 bool, target decoder.GIWTarget) error {
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
	m, err := decoder.Load(dir, decoder.Options{Quant: quant, EmbedInt4: embedInt4})
	if err != nil {
		return fmt.Errorf("load %s (%s): %w", dir, quant, err)
	}
	defer m.Close()
	// TEMP + RENAME, same reason as Transcode's GGUF branch above (M-12/M-33): a write to
	// `out` directly leaves a placeholder-length bundle behind on SIGKILL/OOM-kill/power
	// loss — exactly the interruption class this whole-model-resident path is most exposed
	// to, since it holds the entire quantized model in RAM while writing. Written in place,
	// that half-written file is newer than the source and "fresh" forever, so every later
	// `serve` fails at boot with "truncated bundle" until a human deletes it. The temp name
	// must still end in ".giw" (V-01) for selfCheck's decoder.Load to route to the bundle
	// loader at all.
	tmp := strings.TrimSuffix(out, ".giw") + ".tmp.giw"
	f, err := os.Create(tmp)
	if err != nil {
		return fmt.Errorf("create %s: %w", tmp, err)
	}
	werr := giw.WriteStream(f, tokBytes, func(w io.Writer) (int64, error) {
		return decoder.SerializeWeightsToForTarget(w, m.Weights(), filepath.Base(dir), target)
	})
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
	return nil
}

// EnsureCachedGIW returns a .giw for the GGUF at ggufPath quantized to quant,
// promised to backend — transcoding once into a sidecar cache (alongside the
// GGUF, "<base>.<quant>.<target>.giw") when no fresh cache exists. The cache is
// fresh when it's newer than the source AND was built for the same target (L2:
// the cache key carries the target, so a CPU-built cache is never reused by a
// Metal load — the same "wrong file → rebuild" path the version guard already
// takes for a stale writer). The one-time transcode is logged to stderr (it can
// take minutes and write tens of GB) so a slow first start isn't mistaken for a
// hang. Returns the .giw path to load.
func EnsureCachedGIW(ctx context.Context, ggufPath, quant, backend string) (string, error) {
	target := decoder.GIWTargetForBackend(backend)
	cache := streamCachePath(ggufPath, quant, target)
	if cacheFresh(cache, ggufPath) {
		return cache, nil
	}
	// S1 (task-never-swap-2026-09.md): a half-written sidecar on a full disk is a failure this
	// repo has already had once (M-12's own history), so refuse up front when the sidecar will not
	// fit. The size is projected from the source's tensor shapes at this quant
	// (projectedSidecarBytes); it used to be the source's own size, on the argument that a sidecar is
	// never bigger than an f32 source — but sources are quantized, and real int4 sidecars measured
	// 1.02–1.16× their q4_k_m source, int8int8 ~1.6×, so the check passed and the disk could still
	// fill. freeDiskBytes returning !ok (no portable probe, or the statfs itself failed) proceeds
	// unguarded, same as every other unknown quantity in this codebase (fitguard.go's own rule) — a
	// missing probe must never be the reason a load that would have worked gets refused.
	if fi, serr := os.Stat(ggufPath); serr == nil {
		need, ok := projectedSidecarBytes(ggufPath, quant)
		if !ok {
			need = fi.Size() // header unreadable: the old proxy, better than none
		}
		if free, ok := freeDiskBytes(filepath.Dir(cache)); ok && free < need {
			return "", fmt.Errorf("stream-weights: refusing to transcode %s — projected sidecar size ~%.1f GB exceeds %.1f GB free on this disk (a half-written sidecar on a full disk is worse than refusing up front); free some space and retry, or pass -direct-load to skip the sidecar entirely",
				filepath.Base(ggufPath), float64(need)/1e9, float64(free)/1e9)
		}
	}
	fmt.Fprintf(os.Stderr, "stream-weights: transcoding %s → %s (%s, one-time — minutes + ~model-size on disk)…\n",
		filepath.Base(ggufPath), filepath.Base(cache), quantLabel(quant))
	t0 := time.Now()
	if err := Transcode(ctx, ggufPath, cache, quant, false, target); err != nil {
		return "", err
	}
	if fi, e := os.Stat(cache); e == nil {
		fmt.Fprintf(os.Stderr, "stream-weights: cache ready (%.0f MB) in %s\n",
			float64(fi.Size())/1048576, time.Since(t0).Round(time.Second))
	}
	return cache, nil
}

// SidecarPathIfFresh returns the sidecar .giw for ggufPath at quant/backend's target, and true,
// ONLY when a fresh one already exists — it never transcodes. For a caller like `fit`
// (task-never-swap-2026-09.md S1 item 5) where measuring is supposed to stay cheap; forcing a
// transcode just to check fit would trade a 32 s / 256 CPU-s resident build for an equally
// expensive one-time transcode, not for "nearly free" as the brief asks.
func SidecarPathIfFresh(ggufPath, quant, backend string) (string, bool) {
	cache := streamCachePath(ggufPath, quant, decoder.GIWTargetForBackend(backend))
	if cacheFresh(cache, ggufPath) {
		return cache, true
	}
	return "", false
}

// DefaultToSidecar reports whether a .gguf source should resolve to its sidecar .giw by default
// — S1's own registered rule: "the .gguf direct heap load becomes the opt-out, not the default".
// darwin since S1 (2026-09-22), where the historical swap incidents (gpt-oss-20b, M35/M26)
// happened; linux since 2026-09-24, owner decision, after docs/measurements/cpu-giw-vs-direct-2026-09-24.md
// found no CPU decode cost (1.0007x / 1.0016x on 1.5B / 7B, inside the A/A arm) and a load that
// maps instead of re-quantizing (0.01 s and ~0 heap vs 5.6-16.5 s and 1.3-5 GB). Other platforms
// keep the direct load. directLoad is the caller's already-resolved -direct-load flag /
// GOINFER_GGUF_DIRECT env var; it can only turn the sidecar OFF.
func DefaultToSidecar(directLoad bool) bool {
	if directLoad {
		return false
	}
	return runtime.GOOS == "darwin" || runtime.GOOS == "linux"
}

// streamCachePath is the sidecar cache for a GGUF at a quant and target:
// "<base>.<quant>.<target>.giw" — GIWTargetNone spells as "canonical" rather than
// an empty segment, so the path stays unambiguous.
func streamCachePath(ggufPath, quant string, target decoder.GIWTarget) string {
	base := ggufPath[:len(ggufPath)-len(filepath.Ext(ggufPath))]
	tgt := string(target)
	if tgt == "" {
		tgt = "canonical"
	}
	return base + "." + quantLabel(quant) + "." + tgt + ".giw"
}

// cacheFresh reports whether cache exists, is newer than src, AND actually loads.
//
// mtime alone is not freshness (M-12). It cannot see a bundle that is truncated, written by an
// older writer, or missing a tensor a newer reader requires — all of which are newer than the
// source and all of which fail at load. That is also M-11's trigger: a pre-v6 gpt-oss sidecar
// is "fresh" by mtime, passes validateShapes, and panics at the first forward.
//
// So freshness ends with the question that actually matters — does it load? — using the same
// mmap load selfCheck uses. That load runs the bundle's whole-payload CRC, which reads every byte
// of the file; it used to be described here as lazy and cheap, and was not. It is now done once
// per (size, mtime) (decoder/giwverify.go), so this check costs a full pass only the first time a
// sidecar is seen. A bundle that does not load is not fresh, and the caller rebuilds it instead of
// failing at boot with an error that names the .giw rather than the cause.
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
	cs, err1 := os.Stat(cache)
	ss, err2 := os.Stat(src)
	return err1 == nil && err2 == nil && cs.ModTime().After(ss.ModTime())
}

func quantLabel(q string) string {
	if q == "" {
		return "f32"
	}
	return q
}

// selfCheck verifies a freshly written bundle loads through the real mmap path — the streamed
// weights deserialize, and the whole-payload CRC passes (a full read of the file; the pass is
// recorded so it is not repeated for an unchanged file — decoder/giwverify.go).
//
// Backend:"cpu", not Options{} (found writing L2, docs/tasks/task-int4-layout-2026-09.md):
// an EMPTY Backend means "needs canonical" (wantsCanonicalInt4's own literal-"cpu"-
// is-a-promise rule, L1), so Options{} declined every kind-5 (row4-only) bundle
// this function itself just wrote for a cpu-arm64 target — self-check would have
// failed every -target cpu-arm64 transcode. "cpu" accepts both kind 3 and kind 5
// (the plain CPU backend implements none of wantsCanonicalInt4's interfaces, so it
// never needs canonical either way) and matches what a real cpu-arm64-target
// bundle is actually loaded with in production.
func selfCheck(path string) error {
	m, err := decoder.Load(path, decoder.Options{Backend: "cpu"})
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
