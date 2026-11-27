package decoder

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"path/filepath"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/goinfer/internal/runtime/giwcodec"

	"github.com/townsendmerino/goinfer/internal/runtime/ggufconfig"
	"github.com/townsendmerino/goinfer/internal/runtime/logs"

	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/decoder/config"
	"github.com/townsendmerino/goinfer/internal/runtime/wbuild"
)

// GGUF loading: read a quantized llama.cpp checkpoint and run it through the generic forward. The GGUF file carries both
// the architecture config (metadata) and the weights (dequantized from the mmap, then optionally re-quantized to resident
// int8/int4 per the quant mode), so no separate config.json or safetensors is needed. Layout quirks against the HF
// safetensors path are normalized at load: tensors use llama.cpp's blk.N.* names; the q/k projections of NORM-rope archs
// (llama, granite) are in llama.cpp's interleaved-RoPE permutation, inverted here to rotate_half order (NEOX-rope archs,
// qwen, gemma and mellum, are left as is; see ggufQKPermuted); and Gemma's (1+w) norm offset, baked into the stored weights by
// llama.cpp, is subtracted back out.
//
// Architectures: whatever registry.go's `registry` map carries, not restated here (a short list reads as a limit rather
// than an example). Quant types: F32/F16, Q8_0/Q4_0/Q5_0, the K-quants Q2_K/Q3_K/Q4_K/Q5_K/Q6_K, and IQ4_NL/IQ4_XS.

// ggufConfig reads a Config from a GGUF file's metadata (internal/runtime/ggufconfig), reporting a file no reader handles
// as ErrUnsupportedModel with the reader's text. The resolved Config feeds resolveArchitecture, so a GGUF reuses the same
// per-family adapter as the safetensors path.
func ggufConfig(g *embed.GGUFFile) (*Config, error) {
	cfg, err := ggufconfig.Read(g)
	if errors.Is(err, ggufconfig.ErrUnsupported) {
		return nil, errorOfKind(ErrUnsupportedModel, "%s", err)
	}
	return cfg, err
}

// The metadata ceilings, ggufconfig's, which validateConfigBounds also holds a config.json to.
const (
	maxGGUFLayers       = ggufconfig.MaxLayers
	maxGGUFHidden       = ggufconfig.MaxHidden
	maxGGUFHeads        = ggufconfig.MaxHeads
	maxGGUFVocabSize    = ggufconfig.MaxVocabSize
	maxGGUFExperts      = ggufconfig.MaxExperts
	maxGGUFIntermediate = ggufconfig.MaxIntermediate
)

// loadGGUFWeights parses a .gguf file and builds the weight bundle, mapping
// llama.cpp tensor names to the descriptor and un-permuting q/k.
func loadGGUFWeights(path string, quant quantMode, embedInt4, needCanonical, skipRow4 bool, abort <-chan struct{}) (*darch.Weights, error) {
	// mmap, not heap-read: the raw quantized bytes stay in reclaimable page
	// cache while we dequantize tensor-by-tensor. The weights end up as fresh
	// (f32 or int8) copies, so the mapping is unneeded once the build returns.
	prof := &darch.LoadProfile{}
	if n, ok := GGUFFileBytes(path); ok {
		prof.SetBytes(n)
	}
	var g *embed.GGUFFile
	// `map` is the storage-bound phase: open + mmap + header/metadata parse. Separated from
	// `build` because a slow disk and a slow repack want different remedies.
	if err := prof.Timed("map", func() (e error) { g, e = OpenGGUFMmap(path); return e }); err != nil {
		return nil, err
	}
	keep := false // a q4k load with Q5_K/Q6_K tensors hands the mapping to Weights for the resident build (KQuantReader)
	defer func() {
		if !keep {
			g.Close()
		}
	}()
	var w *darch.Weights
	// `build` is the CPU-bound phase: dequantize every tensor and re-quantize/repack it into the
	// resident precision. On a large model this dominates, which is the fact the banner surfaces.
	if err := prof.Timed("build", func() (e error) {
		w, e = buildGGUFWeights(g, quant, embedInt4, needCanonical, skipRow4, abort)
		return e
	}); err != nil {
		return nil, err
	}
	w.Prof = prof
	if len(w.KQuant) > 0 {
		w.KQuantFile, w.GGUFPath, keep = g, path, true
	}
	return w, nil
}

// buildGGUFWeights builds the weight bundle from an already-open GGUF — whether
// memory-mapped (loadGGUFWeights) or backed by an in-memory slice
// (LoadGGUFBytes). It resolves the config + architecture from the GGUF's own
// metadata, so both entry points produce an identical model.
func buildGGUFWeights(g *embed.GGUFFile, quant quantMode, embedInt4, needCanonical, skipRow4 bool, abort <-chan struct{}) (*darch.Weights, error) {
	cfg, err := ggufConfig(g)
	if err != nil {
		return nil, err
	}
	if err := ggufconfig.ValidateDims(cfg); err != nil {
		return nil, err
	}
	arch, _, err := resolveArchitecture(cfg) // llama descriptor + finalizeRoPE
	if err != nil {
		return nil, err
	}
	return wbuild.BuildWeightsFromGGUF(cfg, arch, g, quant, embedInt4, needCanonical, skipRow4, nil, "", abort, wbuildHooks())
}

// StreamTranscodeGGUF transcodes the GGUF at path into a .giw weights body written to out (the GINFW serialization plus
// trailing CRC), loading one layer at a time so peak RAM is about one layer rather than the whole model: the streaming
// analogue of Load + SerializeWeightsTo, for models too large to hold resident. Every family streams, dedicated loaders
// included. Returns the bytes written. Typically invoked inside giw.WriteStream as the weights half of a .giw bundle.
func StreamTranscodeGGUF(ctx context.Context, path string, out io.Writer, quant string, embedInt4 bool, target GIWTarget, id string) (int64, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	// Cancellation: a transcode of a >RAM model can run for minutes writing tens of GB. Wrapping the sink makes every
	// per-layer write observe ctx, so a cancelled context aborts at the next layer boundary rather than after the whole pass.
	out = &ctxWriter{ctx: ctx, w: out}
	q, err := parseQuant(quant)
	if err != nil {
		return 0, err
	}
	if q.Half16() {
		return 0, fmt.Errorf("decoder: quant %q has no bundle form (a CPU-path load mode)", quant)
	}
	g, err := OpenGGUFMmap(path)
	if err != nil {
		return 0, err
	}
	defer g.Close()
	cfg, err := ggufConfig(g)
	if err != nil {
		return 0, err
	}
	// This streaming path never goes through decoder.Load, so it must repeat Load's resolveEOSIDs write-back: resolve against
	// the GGUF's own parent directory (os.DirFS(path) cannot open anything inside a FILE), then write the resolved set back
	// into cfg.EOSTokenID so it round-trips through the bundle. Without it a .giw would carry only the raw GGUF-metadata EOS,
	// dropping any extra stop id from a sibling generation_config.json, and the stream would diverge from the resident bundle
	// (TestStreamTranscodeMatchesResident byte-diffs the two).
	if raw, jerr := json.Marshal(config.ResolveEOSIDs(filepath.Dir(path), cfg)); jerr == nil {
		cfg.EOSTokenID = raw
	}
	if err := ggufconfig.ValidateDims(cfg); err != nil {
		return 0, err
	}
	arch, _, err := resolveArchitecture(cfg)
	if err != nil {
		return 0, err
	}
	// canSerialize refuses a family the writer cannot express; it returns nil for every registered family today.
	if serr := giwcodec.CanSerialize(arch); serr != nil {
		return 0, serr
	}
	// Every family streams: buildWeightsFromGGUF's sink != nil branch builds, writes and releases one layer at a time, so peak
	// RSS is about one layer rather than the whole model.
	//
	// needCanonical=true unconditionally, regardless of target: the writer needs canonical bytes in RAM to choose what to
	// write (repackRow4ForEmit computes row4 from canonical), even on a cpu-arm64 target that writes kind 5
	// (canonical-absent) to disk; per layer, that is one layer's canonical bytes. With a sink the build keeps nothing
	// else in RAM (repackForDecode). abort=nil: this is the cmd/prequant
	// transcode path, which observes ctx per layer through ctxWriter; the swap tripwire's load-time consumer is scoped to
	// decoder.Load's resident build only (see Options.LoadAbort).
	wr := newGIWWriter(out, target, streamLayerBudget(), wbuild.StreamTestHooksOf(ctx))
	if _, err := wbuild.BuildWeightsFromGGUF(cfg, arch, g, q, embedInt4, true, false, &wr.StreamWriter, id, nil, wbuildHooks()); err != nil {
		return wr.N, err
	}
	var crc [4]byte
	binary.LittleEndian.PutUint32(crc[:], wr.CRC) // running CRC over the streamed body
	if _, err := out.Write(crc[:]); err != nil {
		return wr.N, err
	}
	return wr.N + 4, nil
}

// ctxWriter aborts an in-flight streamed transcode: it returns ctx.Err() before each underlying Write once the context is
// cancelled, so StreamTranscodeGGUF stops writing (and the caller removes the partial .giw) instead of running the whole
// multi-GB pass to completion after a shutdown or disconnect.
type ctxWriter struct {
	ctx context.Context
	w   io.Writer
}

func (c *ctxWriter) Write(p []byte) (int, error) {
	if err := c.ctx.Err(); err != nil {
		return 0, err
	}
	return c.w.Write(p)
}

// LoadGGUFBytes loads a GGUF model from an in-memory slice (e.g. an inflated
// //go:embed asset) — same result as Load on the equivalent .gguf file, but
// nothing touches the filesystem (no temp file, no writable disk needed). raw
// is parsed in place and only needs to live until this returns; the built
// weights are fresh copies.
//
// EOS ids come from the GGUF's own tokenizer metadata (there is no directory to
// read a generation_config.json from), matching the file path's resolution for
// a bare .gguf.
func LoadGGUFBytes(raw []byte, opts Options) (*Model, error) {
	opts = opts.withAutoBackend()
	be, beErr := NewBackend(opts.Backend)
	quant, err := parseQuant(opts.Quant)
	if err != nil {
		closeBackend(be)
		return nil, err
	}
	if quant.Half16() {
		closeBackend(be)
		return nil, fmt.Errorf("decoder: quant bf16/f16 applies to safetensors checkpoints, not a GGUF")
	}
	g, err := embed.OpenGGUFBytes(raw)
	if err != nil {
		closeBackend(be)
		return nil, err
	}
	w, err := buildGGUFWeights(g, quant, opts.EmbedInt4, wantsCanonicalInt4(opts.Backend, be), !wbuild.WantsRow4Fallback(opts.layoutBackend()), opts.LoadAbort)
	g.Close()
	if err != nil {
		closeBackend(be)
		return nil, err
	}
	if beErr != nil {
		logs.Warnf("%v", beErr) // webgpu requested but fell back — not fatal (stderr, not stdout — never contaminate a piped token stream)
	}
	m := modelFromOptions(w, be, opts)
	m.quant = opts.Quant
	return m.withBackendNames(opts.Backend, beErr).bindKnobs(opts.Knobs.values()).withResidency(), nil
}

// ggufRowFilter, when non-nil, rewrites each dequantized layer-matmul row (streamMat,
// streamMatBatched and Phi-3's fusedSplit) before it is quantized. A test seam only: set by SetGGUFRowFilterForTest under
// goinfer_testhooks, nil in every production build. It is used
// to fake-quantize rows by their source GGUF type.
var ggufRowFilter func(name string, row []float32)

// q4kWiden56ForTest restores the int8 widening of Q5_K and Q6_K on every q4k load, for the test that compares the two.
var q4kWiden56ForTest = false
