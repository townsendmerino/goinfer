// The .giw transcode and the sidecar cache: a GGUF or a safetensors directory loaded at a fixed quant, its quantized
// weights streamed to a bundle with the source's tokenizer half beside them. cmd/prequant runs it by hand, and Load runs
// it by default for a .gguf or a safetensors directory (sidecarRoute), as serve and chat do through internal/modelload.

package sidecar

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/internal/runtime/giw"
	"github.com/townsendmerino/goinfer/internal/runtime/giwcodec"
	"github.com/townsendmerino/goinfer/internal/runtime/logs"
	"github.com/townsendmerino/goinfer/internal/runtime/wbuild"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// metaPrefixCap bounds how much of the source GGUF we read to extract the tokenizer
// half. The metadata + tensor directory live at the front of the file (KVs + tensor
// infos, no weight bytes), a few MB even for a 256-expert MoE — so reading the whole
// multi-GB model just to slice its header wastes that much heap (and OOMs on a 35B).
// 64 MiB is comfortably more than any real header.
const metaPrefixCap = 64 << 20

// TranscodeGIW writes a .giw bundle at out from the model at in, quantized to quant
// ("int8int8" | "int8" | "int4" | "" for f32). `in` may be a GGUF file (streamed one layer at a
// time, peak RAM about one layer: fits a 35B on a modest box) OR a safetensors model DIRECTORY (the
// path for safetensors-only models like Mellum2; see transcodeDir). A failed write removes the
// partial output. A cancelled ctx aborts a long streaming transcode at the next layer boundary and
// removes the partial output.
//
// target names the ONE consumer this bundle is promised to:
// on a cpu-arm64 target, every eligible int4 tensor writes kind 5 (row4-only: the on-disk arm64
// split-half, 4-row-interleaved layout) instead of kind 3
// (decoder/serialize.go's weightMat vs weightMatKind3Only decides which tensors are eligible); every
// other target, including giwcodec.TargetNone, writes kind 3 for every int4 tensor.
// giwcodec.TargetForBackend derives a target from a backend name (EnsureCachedGIW);
// Parsegiwcodec.Target parses cmd/prequant's own -target flag. A cpu-arm64 target needs an arm64
// box (the repack functions are NEON-only in aikit); a shape the repack rejects, or a non-arm64
// build, falls back to kind 3 for that tensor, so any target is always safe to pass.
func Transcode(ops Ops, ctx context.Context, in, out, quant string, embedInt4 bool, target giwcodec.Target) error {
	_, err := TranscodeTimed(ops, ctx, in, out, quant, embedInt4, target)
	return err
}

// TranscodePhases is where a transcode's wall time went. Stream is building the bundle and writing it to disk; SelfCheck is
// loading it back through the mmap path, whole-file CRC included.

// Ops supplies the decoder-level callbacks sidecar transcoding invokes.
type Ops struct {
	StreamGGUF       func(ctx context.Context, in string, w io.Writer, quant string, embedInt4 bool, target giwcodec.Target, id string) (int64, error)
	StreamDir        func(ctx context.Context, dir string, w io.Writer, quant string, embedInt4 bool, target giwcodec.Target, id string) (int64, error)
	ResidentDirBody  func(w io.Writer, dir, lora, quant string, embedInt4 bool, target giwcodec.Target, id string) (int64, error)
	SelfCheck        func(path string) error
	HostRAM          func() int64
	VerifyMarkerPath func(path string) string
	IsDirNoStream    func(error) bool
	MarkGIWWritten   func(path string)
	OpenGGUFMmap     func(path string) (*embed.GGUFFile, error)
	GGUFFileBytes    func(path string) (int64, bool)
	TargetForBackend func(backend string) giwcodec.Target
	HeaderCheck      func(path string) error
	MapGIW           func(path string) ([]byte, error)
	UnmapGIW         func(data []byte) error
	ScanGIWWeights   func(blob []byte) (string, error)
	Rename           func(oldpath, newpath string) error
	FreeDisk         func(path string) (int64, bool)
	GGUFShards       func(path string) ([]string, error)
	ParseQuant       func(q string) (wbuild.QuantMode, error)
}

type TranscodePhases struct {
	Stream, SelfCheck time.Duration
}

// String is the phase line's tail: "stream 20.1s, self-check 0.9s".
func (p TranscodePhases) String() string {
	return fmt.Sprintf("stream %.1fs, self-check %.1fs", p.Stream.Seconds(), p.SelfCheck.Seconds())
}

// TranscodeGIWTimed is TranscodeGIW, and reports where the time went. A phase that did not finish reads zero.
func TranscodeTimed(ops Ops, ctx context.Context, in, out, quant string, embedInt4 bool, target giwcodec.Target) (TranscodePhases, error) {
	if fi, err := os.Stat(in); err == nil && fi.IsDir() {
		return TranscodeDir(ops, ctx, in, "", out, quant, embedInt4, target)
	}
	var ph TranscodePhases
	// 1) Tokenizer half: the source GGUF truncated at the tensor-data boundary —
	// metadata + tensor infos, no weight bytes. Only the file's head is read.
	head, err := ReadHead(in, metaPrefixCap)
	if err != nil {
		return ph, fmt.Errorf("read gguf head: %w", err)
	}
	prefix, err := MetadataPrefixLen(head)
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
	werr := writeSidecarBundle(ctx, f, tokBytes, func(w io.Writer) (int64, error) {
		return ops.StreamGGUF(ctx, in, w, quant, embedInt4, target, filepath.Base(in))
	})
	runtime.GC()
	if cerr := f.Close(); werr == nil {
		werr = cerr
	}
	if werr != nil {
		RemoveTempGIW(ops, tmp)
		return ph, fmt.Errorf("write bundle: %w", werr)
	}
	ph.Stream = time.Since(t0)

	// 3) Verify the bundle loads through the real (mmap) load path: it deserializes and its shapes validate. Whether
	// that load recomputes the whole-file CRC is markWritten's rule. On the TEMP file, so a bundle that fails it never
	// becomes the sidecar even for an instant.
	MarkWritten(tmp, in, ops.HostRAM, ops.MarkGIWWritten)
	if hooks := TestHooksOf(ctx); hooks.AfterWrite != nil {
		hooks.AfterWrite(tmp)
	}
	t1 := time.Now()
	if err := ops.SelfCheck(tmp); err != nil {
		RemoveTempGIW(ops, tmp)
		return ph, fmt.Errorf("self-check: %w", err)
	}
	ph.SelfCheck = time.Since(t1)
	if err := PublishGIW(ops, tmp, out); err != nil {
		RemoveTempGIW(ops, tmp)
		return ph, fmt.Errorf("publish %s: %w", out, err)
	}
	CarryVerifiedMarker(ops, tmp, out)
	_ = os.Remove(LoraSidecar(out)) // a plain bundle carries no adapter, whatever an earlier build at this path did
	return ph, nil
}

// transcodeDir builds a .giw from a safetensors model DIRECTORY (no GGUF available — the
// path for Mellum2 and other safetensors-only models). It streams the weights into the
// bundle one layer at a time (StreamTranscodeDir; a LoRA merge or a family whose loader
// does not stream loads the whole model instead), and carries the dir's tokenizer.json
// verbatim as the tok half (the serve side loads it via tokenizer.LoadJSONBytes when the
// blob isn't GGUF metadata).
func TranscodeDir(ops Ops, ctx context.Context, dir, lora, out, quant string, embedInt4 bool, target giwcodec.Target) (TranscodePhases, error) {
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
			logs.Infof("prequant: note: %s/tokenizer.json present but not goinfer-loadable (%v); bundle carries it, but serve may need the original tokenizer", dir, lerr)
		}
		tokBytes = b
	} else {
		logs.Infof("prequant: note: no tokenizer.json in %s — weights-only bundle (serve needs a separate tokenizer)", dir)
	}
	id := filepath.Base(dir)
	var side []byte
	if lora != "" {
		var err error
		if side, err = LoraSidecarBytes(lora); err != nil {
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
	// Streamed one layer at a time: peak memory about the globals plus
	// one layer, byte-identical to the resident transcode. A LoRA merge, and a family whose loader does not stream yet, take
	// the resident transcode instead.
	body := func(w io.Writer) (int64, error) {
		return ops.ResidentDirBody(w, dir, lora, quant, embedInt4, target, id)
	}
	if lora == "" {
		body = func(w io.Writer) (int64, error) {
			n, serr := ops.StreamDir(ctx, dir, w, quant, embedInt4, target, id)
			if ops.IsDirNoStream(serr) && n == 0 {
				logs.Infof("prequant: streams: no (%v); building it resident", serr)
				return ops.ResidentDirBody(w, dir, lora, quant, embedInt4, target, id)
			}
			return n, serr
		}
	}
	werr := writeSidecarBundle(ctx, f, tokBytes, body)
	runtime.GC()
	if cerr := f.Close(); werr == nil {
		werr = cerr
	}
	if werr != nil {
		RemoveTempGIW(ops, tmp)
		return ph, fmt.Errorf("write bundle: %w", werr)
	}
	ph.Stream = time.Since(t0)
	MarkWritten(tmp, dir, ops.HostRAM, ops.MarkGIWWritten)
	if hooks := TestHooksOf(ctx); hooks.AfterWrite != nil {
		hooks.AfterWrite(tmp)
	}
	t1 := time.Now()
	if err := ops.SelfCheck(tmp); err != nil {
		RemoveTempGIW(ops, tmp)
		return ph, fmt.Errorf("self-check: %w", err)
	}
	ph.SelfCheck = time.Since(t1)
	if err := PublishGIW(ops, tmp, out); err != nil {
		RemoveTempGIW(ops, tmp)
		return ph, fmt.Errorf("publish %s: %w", out, err)
	}
	CarryVerifiedMarker(ops, tmp, out)
	// The sidecar is published after the bundle, so a bundle never claims an adapter it was not built with; a crash in
	// between leaves a bundle without its sidecar, which a head refuses (GIWAdapterLoRA) rather than trusts.
	if side == nil {
		_ = os.Remove(LoraSidecar(out))
	} else if err := os.WriteFile(LoraSidecar(out), side, 0o644); err != nil {
		return ph, fmt.Errorf("write %s: %w", LoraSidecar(out), err)
	}
	return ph, nil
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
func EnsureCached(ops Ops, ctx context.Context, ggufPath, quant, backend string, embedInt4 bool) (string, error) {
	target := ops.TargetForBackend(backend)
	cache := StreamCachePath(ops, ggufPath, quant, embedInt4, target)
	if FreshCache(ops, cache, LegacyCachePath(ops, ggufPath, quant, embedInt4, target), ggufPath) {
		return cache, nil
	}
	// Refuse up front when the sidecar will not fit: a half-written sidecar on a full disk is worse than
	// refusing. The size is projected from the source's tensor shapes at this quant
	// (projectedSidecarBytes), not taken from the source's own size, because sources are already quantized
	// and a sidecar can be larger than its source. freeDiskBytes returning !ok (no portable probe, or the
	// statfs itself failed) proceeds unguarded, as fitguard.go's rule says: a missing probe must never be
	// the reason a load that would have worked gets refused.
	if fi, serr := os.Stat(ggufPath); serr == nil {
		need, ok := ProjectedSidecarBytes(ops, ggufPath, quant)
		if fi.IsDir() {
			need, ok = ProjectedDirSidecarBytes(ggufPath, quant)
		}
		if !ok {
			need = fi.Size() // header unreadable: the old proxy, better than none
			if n, ok := ops.GGUFFileBytes(ggufPath); ok {
				need = n // a split set's whole size, not its first shard's
			}
		}
		if free, ok := ops.freeDisk(filepath.Dir(cache)); ok && free < need {
			return "", fmt.Errorf("stream-weights: refusing to transcode %s — projected sidecar size ~%.1f GB exceeds %.1f GB free on this disk (a half-written sidecar on a full disk is worse than refusing up front); free some space and retry, or pass -direct-load to skip the sidecar entirely",
				filepath.Base(ggufPath), float64(need)/1e9, float64(free)/1e9)
		}
	}
	logs.Infof("stream-weights: transcoding %s → %s (%s, one-time — minutes + ~model-size on disk)…",
		filepath.Base(ggufPath), filepath.Base(cache), sidecarQuantLabel(quant))
	t0 := time.Now()
	ph, err := TranscodeTimed(ops, ctx, ggufPath, cache, quant, embedInt4, target)
	if err != nil {
		return "", err
	}
	if fi, e := os.Stat(cache); e == nil {
		logs.Infof("stream-weights: cache ready (%.0f MB) in %s: %s",
			float64(fi.Size())/1048576, time.Since(t0).Round(time.Second), ph)
	}
	if lf, e := os.Stat(LegacyCachePath(ops, ggufPath, quant, embedInt4, target)); e == nil {
		logs.Infof("stream-weights: note: %s (%.0f MB) is from a build that wrote unversioned sidecar names; this build leaves it alone. Delete it once no such build is in use.",
			filepath.Base(LegacyCachePath(ops, ggufPath, quant, embedInt4, target)), float64(lf.Size())/1048576)
	}
	return cache, nil
}

// SidecarPathIfFresh returns the sidecar .giw for ggufPath at quant/backend's target and embedInt4
// setting, and true, ONLY when a fresh one already exists: it never transcodes. For a caller like `fit`
// where measuring is supposed to stay cheap: forcing a
// transcode just to check fit would cost as much as the resident build it avoids.
func PathIfFresh(ops Ops, ggufPath, quant, backend string, embedInt4 bool) (string, bool) {
	target := ops.TargetForBackend(backend)
	cache := StreamCachePath(ops, ggufPath, quant, embedInt4, target)
	if FreshCache(ops, cache, LegacyCachePath(ops, ggufPath, quant, embedInt4, target), ggufPath) {
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
func PathIfCurrent(ops Ops, src, quant, backend string, embedInt4 bool) (string, SidecarState) {
	target := ops.TargetForBackend(backend)
	cache := StreamCachePath(ops, src, quant, embedInt4, target)
	if CacheNewer(ops, cache, src) && ops.HeaderCheck(cache) == nil {
		return cache, SidecarCurrent
	}
	if _, err := os.Stat(cache); err != nil && CacheNewer(ops, LegacyCachePath(ops, src, quant, embedInt4, target), src) {
		return cache, SidecarLegacy
	}
	return cache, SidecarMissing
}

// SidecarPath is the sidecar .giw a load of ggufPath (a .gguf or a safetensors directory) at quant/backend's target and
// embedInt4 setting opens, or builds first when it is not there or not fresh (EnsureCachedGIW). It reads nothing.
func Path(ops Ops, ggufPath, quant, backend string, embedInt4 bool) string {
	return StreamCachePath(ops, ggufPath, quant, embedInt4, ops.TargetForBackend(backend))
}

// DefaultToSidecar reports whether a .gguf source should resolve to its sidecar .giw by default: the.
// gguf direct heap load is the opt-out, not the default. True on darwin (where the swap incidents
// happened) and on linux (the sidecar showed no CPU decode cost and a load that maps instead of
// re-quantizing; docs/measurements/cpu-giw-vs-direct-2026-09-24.md); other platforms keep the direct
// load. directLoad is the caller's already-resolved -direct-load flag / GOINFER_GGUF_DIRECT env var;
// it can only turn the sidecar OFF.
func DefaultEnabled(directLoad bool) bool {
	if directLoad {
		return false
	}
	return runtime.GOOS == "darwin" || runtime.GOOS == "linux"
}

// sidecarInTests is whether Load takes the sidecar default inside a test binary. Off: a test that loads a .gguf or a
// safetensors directory keeps testing that load, and writes nothing beside its fixture or checkpoint. A test of the
// default itself turns it on.
var sidecarInTests = false

// streamCachePath is the sidecar cache for a GGUF at a quant, embedInt4 setting and target:
// "<base>.<quant>.<target>.v<N>.giw", or "<base>.<quant>.e4h.<target>.v<N>.giw" when embedInt4 is set, N being the
// weights format version this build reads (giwcodec.FormatVersion). The version is in the name so two builds at
// different formats on one box keep a sidecar each instead of rewriting one file in turn (a build that reads only vN
// refuses a v(N+1) file as too new and replaces it, and the other then does the same). giwcodec.TargetNone spells as
// "canonical" rather than an empty segment, so the path stays unambiguous. The "e4h" segment (embed-int4-head)
// exists so a plain-head sidecar built before EmbedInt4 defaulted on is never mistaken for, or overwritten by, an
// embed-int4 one built after: they are different bundles at the same source and quant, and need different cache keys.
func StreamCachePath(ops Ops, ggufPath, quant string, embedInt4 bool, target giwcodec.Target) string {
	return strings.TrimSuffix(LegacyCachePath(ops, ggufPath, quant, embedInt4, target), ".giw") + fmt.Sprintf(".v%d.giw", giwcodec.FormatVersion)
}

// legacyCachePath is the sidecar name builds before the version was in it wrote: "<base>.<quant>[.e4h].<target>.giw".
// A build from before the change still reads and rewrites that name, so this build never writes or deletes it; it only
// adopts a file there that already loads as this build's format (adoptLegacy).
func LegacyCachePath(ops Ops, ggufPath, quant string, embedInt4 bool, target giwcodec.Target) string {
	base := ggufPath[:len(ggufPath)-len(filepath.Ext(ggufPath))]
	if IsDir(ggufPath) {
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
	label += GatherTableTag(ops, ggufPath, quant, embedInt4) + KQuantTag(quant, target)
	return base + "." + label + "." + tgt + ".giw"
}

// kquantTag is the name segment of a q4k sidecar that keeps the GGUF's Q5_K and Q6_K tensors natively (kinds 11 and 12,
// q4kNative56), ".k56"; "" otherwise. A q4k sidecar written before those kinds widened them under the name without the
// segment, so the two are never mistaken for each other, and no legacy (unversioned) name carries it.
func KQuantTag(quant string, target giwcodec.Target) string {
	if quant != "q4k" || !giwcodec.CarriesKQuant56(target) || !linalg.KQuantNative() {
		return ""
	}
	return ".k56"
}

// gatherTableTag is the name segment of a sidecar whose per-layer token table (Gemma 4's E-models) is int4 while the head
// keeps the int8 pin, ".ple4" (quantMode.gatherTable); "" otherwise. A sidecar written before the table left the pin holds it
// at int8 under the name without the segment, so the two are never mistaken for each other, and no legacy (unversioned) name
// carries it, so none is adopted.
func GatherTableTag(ops Ops, src, quant string, embedInt4 bool) string {
	if q, err := ops.ParseQuant(quant); err != nil || q != wbuild.QuantInt4 || embedInt4 || !HasPerLayerTable(ops, src) {
		return ""
	}
	return ".ple4"
}

// hasPerLayerTable reports whether src (a .gguf or a safetensors directory) carries Gemma 4's per-layer token table, from
// the GGUF's tensor directory or the directory's config.json. An unreadable source reads as false.
func HasPerLayerTable(ops Ops, src string) bool {
	if IsDir(src) {
		raw, err := os.ReadFile(filepath.Join(src, "config.json"))
		if err != nil {
			return false
		}
		var c struct {
			PLE  int `json:"hidden_size_per_layer_input"`
			Text *struct {
				PLE int `json:"hidden_size_per_layer_input"`
			} `json:"text_config"`
		}
		if json.Unmarshal(raw, &c) != nil {
			return false
		}
		return c.PLE > 0 || (c.Text != nil && c.Text.PLE > 0)
	}
	g, err := ops.OpenGGUFMmap(src)
	if err != nil {
		return false
	}
	defer g.Close()
	_, ok := g.Dims("per_layer_token_embd.weight")
	return ok
}

// freshCache reports whether the sidecar at cache is fresh for src, adopting the legacy-named file first when this
// build has none of its own: a legacy file that is newer than src and loads under this build's format is already
// this build's sidecar, so it is hard-linked to the versioned name (no copy, no transcode). The old name stays, for
// the builds that still use it, and an older build rewriting it by temp-and-rename leaves this link untouched. A
// legacy file in another format, or stale, is left alone, silently: the one that matters is the versioned name.
func FreshCache(ops Ops, cache, legacy, src string) bool {
	if CacheFresh(ops, cache, src) {
		return true
	}
	return adoptLegacy(ops, cache, legacy, src) && CacheFresh(ops, cache, src)
}

func adoptLegacy(ops Ops, cache, legacy, src string) bool {
	if _, err := os.Stat(cache); err == nil {
		return false // a versioned file exists and was refused: it is rebuilt, not replaced by an older name
	}
	if !CacheNewer(ops, legacy, src) || ops.SelfCheck(legacy) != nil {
		return false
	}
	if err := os.Link(legacy, cache); err != nil {
		return false
	}
	_ = os.Link(ops.VerifyMarkerPath(legacy), ops.VerifyMarkerPath(cache)) // best effort: the marker keeps the first load off a whole-file CRC
	logs.Infof("stream-weights: using %s as %s (the same file: already this build's format)", filepath.Base(legacy), filepath.Base(cache))
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
func CacheFresh(ops Ops, cache, src string) bool {
	if !CacheNewer(ops, cache, src) {
		return false
	}
	if err := ops.SelfCheck(cache); err != nil {
		logs.Warnf("stream-weights: cache %s is newer than the source but does not "+
			"load (%v) — rebuilding", filepath.Base(cache), err)
		return false
	}
	return true
}

// GIWWeightsQuant reads the quant of the .giw at path as a load would label it, "native" for f32, without copying any weights: what
// a caller decides on before the load.
func WeightsQuant(ops Ops, path string) (string, error) {
	data, err := ops.MapGIW(path)
	if err != nil {
		return "", err
	}
	defer ops.UnmapGIW(data)
	blob, _, err := giw.Read(data)
	if err != nil {
		return "", err
	}
	return ops.ScanGIWWeights(blob)
}

// ReplacedInfix names a bundle publishGIW moved aside. Its name does not end in ".giw", so nothing lists or loads it.
const ReplacedInfix = ".replaced-"

// publishGIW moves a finished bundle to its name. Where the name is taken by a bundle a model holds mapped, and the platform refuses to
// replace it (Windows can), the old file moves aside first: renaming a mapped file is allowed where replacing one is not, the mapping
// keeps reading it, and it is removed here or by a later publish once nothing holds it. A publish that still cannot complete puts the
// old file back.
func PublishGIW(ops Ops, tmp, out string) error {
	SweepReplacedGIW(out)
	err := ops.rename(tmp, out)
	if err == nil {
		return nil
	}
	aside := fmt.Sprintf("%s%s%d-%d", out, ReplacedInfix, os.Getpid(), time.Now().UnixNano())
	if merr := ops.rename(out, aside); merr != nil {
		return err // the name holds nothing to move, or nothing can: the first error is the one that says why
	}
	if err := ops.rename(tmp, out); err != nil {
		_ = ops.rename(aside, out)
		return err
	}
	_ = os.Remove(aside)
	return nil
}

// sweepReplacedGIW removes what earlier publishes of out moved aside, where nothing holds it now.
func SweepReplacedGIW(out string) {
	entries, err := os.ReadDir(filepath.Dir(out))
	if err != nil {
		return
	}
	prefix := filepath.Base(out) + ReplacedInfix
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), prefix) {
			_ = os.Remove(filepath.Join(filepath.Dir(out), e.Name()))
		}
	}
}

// cacheNewer is the mtime half of freshness, kept separate so each half can be tested for what
// it actually decides: this one answers "has the source changed since the cache was built",
// which is all an mtime can answer.
func CacheNewer(ops Ops, cache, src string) bool {
	cs, err := os.Stat(cache)
	if err != nil {
		return false
	}
	// A split GGUF is as new as its newest shard: a re-pulled shard 2 makes the sidecar stale as surely as a new
	// shard 1 does. A single file is its own one-element set.
	srcs, err := sourceFiles(ops, src)
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
func sourceFiles(ops Ops, src string) ([]string, error) {
	if !IsDir(src) {
		return ops.GGUFShards(src)
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

func IsDir(p string) bool {
	fi, err := os.Stat(p)
	return err == nil && fi.IsDir()
}

func sidecarQuantLabel(q string) string {
	if q == "" {
		return "f32"
	}
	return q
}

// removeTempGIW deletes a temp bundle and its verified-marker.
func RemoveTempGIW(ops Ops, tmp string) {
	_ = os.Remove(tmp)
	_ = os.Remove(ops.VerifyMarkerPath(tmp))
}

// carryVerifiedMarker moves a temp bundle's verified-marker to the bundle's final name, after the caller has published
// tmp onto out (the publish stays at the call site because TestTranscode_writesViaTempThenRenames asserts it
// structurally). The transcode marked tmp at its (size, mtime) and a rename preserves both, so the marker is still true
// of out: the first real load of a fresh sidecar does not read the whole file. Best-effort: a marker that fails to move
// only costs that load one CRC pass, and none is left behind.
func CarryVerifiedMarker(ops Ops, tmp, out string) {
	if err := os.Rename(ops.VerifyMarkerPath(tmp), ops.VerifyMarkerPath(out)); err != nil {
		_ = os.Remove(ops.VerifyMarkerPath(tmp))
	}
}

// readHead reads up to capBytes from the front of path (a GGUF's metadata + tensor
// directory live at the front). Returns fewer bytes for a smaller file.
func ReadHead(path string, capBytes int) ([]byte, error) {
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

func (o Ops) rename(oldpath, newpath string) error {
	if o.Rename != nil {
		return o.Rename(oldpath, newpath)
	}
	return os.Rename(oldpath, newpath)
}

func (o Ops) freeDisk(path string) (int64, bool) {
	if o.FreeDisk != nil {
		return o.FreeDisk(path)
	}
	return 0, false
}
