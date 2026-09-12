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
// docs/task-w4a8-neon-bandwidth.md's "Format follow-on") instead of kind 3
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
	// (e.g. a 106B-A12B int4 on a 62 GB box). The dedicated qwen35/gemma4 loaders fall
	// back to a resident build inside StreamTranscodeGGUF (those models fit).
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
		return decoder.StreamTranscodeGGUF(ctx, in, w, quant, false, target, filepath.Base(in))
	})
	runtime.GC()
	if cerr := f.Close(); werr == nil {
		werr = cerr
	}
	if werr != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("write bundle: %w", werr)
	}

	// 3) Verify the bundle round-trips through the real (mmap) load path — cheap
	// (lazy faults), confirms the streamed weights deserialize. On the TEMP file, so a
	// bundle that fails it never becomes the sidecar even for an instant.
	if err := selfCheck(tmp); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("self-check: %w", err)
	}
	if err := os.Rename(tmp, out); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("publish %s: %w", out, err)
	}
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
	f, err := os.Create(out)
	if err != nil {
		return fmt.Errorf("create %s: %w", out, err)
	}
	werr := giw.WriteStream(f, tokBytes, func(w io.Writer) (int64, error) {
		return decoder.SerializeWeightsToForTarget(w, m.Weights(), filepath.Base(dir), target)
	})
	runtime.GC()
	if cerr := f.Close(); werr == nil {
		werr = cerr
	}
	if werr != nil {
		_ = os.Remove(out)
		return fmt.Errorf("write bundle: %w", werr)
	}
	if err := selfCheck(out); err != nil {
		_ = os.Remove(out)
		return fmt.Errorf("self-check: %w", err)
	}
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
// mmap probe selfCheck uses, which is lazy and cheap (it faults pages it never reads). A
// bundle that does not load is not fresh, and the caller rebuilds it instead of failing at
// boot with an error that names the .giw rather than the cause.
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

// selfCheck verifies a freshly written bundle loads through the real mmap path
// (lazy, low RAM) — the streamed weights deserialize.
//
// Backend:"cpu", not Options{} (found writing L2, docs/task-int4-layout-2026-09.md):
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
