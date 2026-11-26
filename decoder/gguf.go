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
	"github.com/townsendmerino/aikit/linalg"

	"github.com/townsendmerino/goinfer/internal/runtime/ggufconfig"
	"github.com/townsendmerino/goinfer/internal/runtime/logs"

	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/decoder/config"
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
)

// GGUF loading: read a quantized llama.cpp checkpoint and run it through the generic forward. The GGUF file carries both
// the architecture config (metadata) and the weights (dequantized from the mmap, then optionally re-quantized to resident
// int8/int4 per the quant mode), so no separate config.json or safetensors is needed. Layout quirks against the HF
// safetensors path are normalized at load: tensors use llama.cpp's blk.N.* names; the q/k projections of NORM-rope archs
// (llama, mellum) are in llama.cpp's interleaved-RoPE permutation, inverted here to rotate_half order (NEOX-rope archs,
// qwen and gemma, are left as is; see ggufQKPermuted); and Gemma's (1+w) norm offset, baked into the stored weights by
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

// loadDeepseekAttnGGUF loads one DeepSeek MLA layer's attention tensors as f32 (the
// parity-first forward uses plain matvec, like the qwen35/granite mixer weights), from
// llama.cpp's deepseek2 names: attn_q_a/attn_q_a_norm/attn_q_b (q-LoRA) or attn_q
// (direct), attn_kv_a_mqa, attn_kv_a_norm, attn_kv_b, attn_output. GGUF stores each matrix
// as dims [in, out] with the data row-major [out, in] — exactly the layout mlaWeights
// expects (and what g.Tensor dequantizes to).
func loadDeepseekAttnGGUF(g *embed.GGUFFile, p string, l *darch.LayerWeights, arch *darch.Architecture) error {
	pa := arch.MLA
	qkHeadDim := pa.QKHeadDim()
	f32 := func(name string, out, in int) ([]float32, error) {
		dims, data, err := g.Tensor(name)
		if err != nil {
			return nil, err
		}
		if len(dims) != 2 || dims[0] != in || dims[1] != out {
			return nil, fmt.Errorf("decoder(gguf-deepseek): %q dims %v, want [in=%d, out=%d]", name, dims, in, out)
		}
		return data, nil
	}
	flat := func(name string, n int) ([]float32, error) {
		dims, data, err := g.Tensor(name)
		if err != nil {
			return nil, err
		}
		if len(data) != n {
			return nil, fmt.Errorf("decoder(gguf-deepseek): %q has %d elems, want %d (dims %v)", name, len(data), n, dims)
		}
		return data, nil
	}
	w := &archparts.MLAWeights{}
	var err error
	if pa.QLoRARank > 0 { // q-LoRA bottleneck (V3 671B); V2-Lite/Moonlight are direct-q
		if w.QAProj, err = f32(p+"attn_q_a.weight", pa.QLoRARank, arch.HiddenDim); err != nil {
			return err
		}
		if w.QALayernorm, err = flat(p+"attn_q_a_norm.weight", pa.QLoRARank); err != nil {
			return err
		}
		if w.QBProj, err = f32(p+"attn_q_b.weight", arch.NumHeads*qkHeadDim, pa.QLoRARank); err != nil {
			return err
		}
	} else {
		if w.QProj, err = f32(p+"attn_q.weight", arch.NumHeads*qkHeadDim, arch.HiddenDim); err != nil {
			return err
		}
	}
	if w.KVAProj, err = f32(p+"attn_kv_a_mqa.weight", pa.KVLoRARank+pa.QKRopeHeadDim, arch.HiddenDim); err != nil {
		return err
	}
	if w.KVALayernorm, err = flat(p+"attn_kv_a_norm.weight", pa.KVLoRARank); err != nil {
		return err
	}
	if w.KVBProj, err = f32(p+"attn_kv_b.weight", arch.NumHeads*(pa.QKNopeHeadDim+pa.VHeadDim), pa.KVLoRARank); err != nil {
		return err
	}
	if w.OProj, err = f32(p+"attn_output.weight", arch.HiddenDim, arch.NumHeads*pa.VHeadDim); err != nil {
		return err
	}
	l.MLA = w
	return nil
}

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
	return buildWeightsFromGGUF(cfg, arch, g, quant, embedInt4, needCanonical, skipRow4, nil, "", abort)
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
	if q.half16() {
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
	if serr := canSerialize(arch); serr != nil {
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
	wr := &giwWriter{sink: out, target: target, layerBudget: streamLayerBudget(), testHooks: streamTestHooksOf(ctx)}
	if _, err := buildWeightsFromGGUF(cfg, arch, g, q, embedInt4, true, false, wr, id, nil); err != nil {
		return wr.n, err
	}
	var crc [4]byte
	binary.LittleEndian.PutUint32(crc[:], wr.crc) // running CRC over the streamed body
	if _, err := out.Write(crc[:]); err != nil {
		return wr.n, err
	}
	return wr.n + 4, nil
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
	if quant.half16() {
		closeBackend(be)
		return nil, fmt.Errorf("decoder: quant bf16/f16 applies to safetensors checkpoints, not a GGUF")
	}
	g, err := embed.OpenGGUFBytes(raw)
	if err != nil {
		closeBackend(be)
		return nil, err
	}
	w, err := buildGGUFWeights(g, quant, opts.EmbedInt4, wantsCanonicalInt4(opts.Backend, be), !wantsRow4Fallback(opts.layoutBackend()), opts.LoadAbort)
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

// ggufRowFiltered reads one row through the dequantizer and applies ggufRowFilter.
func ggufRowFiltered(name string, into func(start int, dst []float32) error, start int, dst []float32) error {
	if err := into(start, dst); err != nil {
		return err
	}
	if f := ggufRowFilter; f != nil {
		f(name, dst)
	}
	return nil
}

// q4kMat builds the native K-quant WeightMat for quantQ4K: output rows [0, out) of tensor name, where
// rowSrc(r) is destination row r's element offset in the tensor (a permutation for RoPE-permuted
// q/k, an offset for a fused split). A Q4_K tensor is always kept as Q4_K; a Q5_K or Q6_K one only
// when native56 is set (q4kNative56). ok is false, with a nil error, for any other tensor; the
// caller then loads it at int8 W8A8. The super-blocks are COPIED: loadGGUFWeights closes the
// mapping once the load returns.
func q4kMat(g *embed.GGUFFile, name string, out, in int, rowSrc func(r int) int, native56 bool) (linalg.WeightMat, bool, error) {
	wrap, rowBytes := linalg.WrapQ4K, linalg.Q4KRowBytes
	dims, raw, ok, err := g.Q4KRaw(name)
	if err == nil && !ok && native56 {
		if dims, raw, ok, err = g.Q5KRaw(name); err == nil && ok {
			wrap, rowBytes = linalg.WrapQ5K, linalg.Q5KRowBytes
		} else if err == nil {
			dims, raw, ok, err = g.Q6KRaw(name)
			wrap, rowBytes = linalg.WrapQ6K, linalg.Q6KRowBytes
		}
	}
	if err != nil || !ok {
		return linalg.WeightMat{}, false, err
	}
	if len(dims) != 2 || dims[0] != in || in%256 != 0 {
		return linalg.WeightMat{}, false, fmt.Errorf("decoder(gguf): %q dims %v for a q4k load of [%d, %d]", name, dims, out, in)
	}
	rb := rowBytes(in)
	buf := make([]byte, out*rb)
	for r := range out {
		off := rowSrc(r)
		if off%in != 0 || off/in >= dims[1] {
			return linalg.WeightMat{}, false, fmt.Errorf("decoder(gguf): %q row offset %d outside [%d, %d]", name, off, in, dims[1])
		}
		src := off / in
		copy(buf[r*rb:(r+1)*rb], raw[src*rb:(src+1)*rb])
	}
	wm, err := wrap(buf, out, in)
	return wm, err == nil, err
}

// q4kNative56 reports whether a q4k load keeps the file's Q5_K and Q6_K projections natively instead of widening them to
// int8: only where linalg has an arch kernel for them (linalg.KQuantNative; the portable dot is far slower than int8), and
// only where the reader takes the blocks from the weight itself: a heap load with needCanonical false (no GPU lane; a GPU lane
// on a heap load takes them from the int8 copy through KQuantReader), or a .giw written for the cpu-amd64 or cuda target (kinds
// 11 and 12). A cuda .giw holds no GGUF mapping for KQuantReader, so CUDA's build plans its native weights from the blocks the
// bundle carries (cuda's buildState.planProj); every other target widens.
func q4kNative56(needCanonical bool, sink *giwWriter) bool {
	if !linalg.KQuantNative() || q4kWiden56ForTest {
		return false
	}
	if sink != nil {
		return giwCarriesKQuant56(sink.target)
	}
	return !needCanonical
}

// giwCarriesKQuant56 reports whether a q4k .giw written for target holds the file's Q5_K and Q6_K tensors natively (kinds 11
// and 12) rather than widened to int8: the cpu-amd64 target, whose CPU kernels read them, and cuda, whose resident uploads them.
// q4kNative56 also asks for linalg.KQuantNative on the writing host, for the cuda target too: a resident that declines falls back
// to the CPU on the same weights, and without that kernel a native weight is far slower there than int8.
func giwCarriesKQuant56(target GIWTarget) bool {
	return target == GIWTargetCPUAmd64 || target == GIWTargetCUDA
}

// q4kWiden56ForTest restores the int8 widening of Q5_K and Q6_K on every q4k load, for the test that compares the two.
var q4kWiden56ForTest = false

// buildWeightsFromGGUF dequantizes the GGUF tensors into the weight bundle. When quant is set, each matmul tensor is
// re-quantized (per-row int8 or group-wise int4) right after it is dequantized (and un-permuted) and its f32 is freed, so
// a Q4/Q8 GGUF lands resident as int8/int4 (about 1/4 and 1/8 of f32) without ever materializing the whole model in f32
// (see loadWeights). The GGUF's own quant is lossy and so is the re-quant, but it captures nearly all of what a Q4_K_M
// file carries.
//
// When sink is non-nil, the weights are streamed to it (a .giw body) instead of retained: the header and globals are
// written, then each layer is loaded, written and freed in turn, so peak RAM is about one layer, which is what lets a
// model larger than RAM be transcoded. The returned *Weights then holds no layer tensors; the caller writes the trailing
// CRC. Every family's branch streams. A branch that returns without driving the sink would write a header declaring N
// layers and then none, so any new family branch must stream too; decoder/gguf_streaming_shape_test.go checks its
// closure reads only per-layer tensors, and TestGemma4GGUF_streamedMatchesResident-style byte identity is the gate.
func buildWeightsFromGGUF(cfg *Config, arch *darch.Architecture, g *embed.GGUFFile, quant quantMode, embedInt4, needCanonical, skipRow4 bool, sink *giwWriter, id string, abort <-chan struct{}) (*darch.Weights, error) {
	hidden, hd := arch.HiddenDim, arch.HeadDim
	w := &darch.Weights{Cfg: *cfg, Arch: arch, Layers: make([]darch.LayerWeights, arch.NumLayers)}
	skipRow4 = skipRow4 || !repackForDecode(sink) // every closure and family branch below takes it from here
	native56 := q4kNative56(needCanonical, sink)

	// streamMat builds a [out, in] linalg.WeightMat by streaming the tensor's rows through
	// the GGUF dequantizer and quantizing each row directly into the resident
	// arrays — no whole-tensor f32 intermediate (see streamQuantized). rowSrc maps
	// a destination row index to its source element offset in the tensor (identity
	// for a plain load; a permutation for the RoPE-permuted q/k projections).
	streamMat := func(name string, out, in int, mode quantMode, rowSrc func(r int) int) (linalg.WeightMat, error) {
		dims, into, err := ggufScaledRowDequantizer(g, name)
		if err != nil {
			return linalg.WeightMat{}, err
		}
		if len(dims) != 2 || dims[0] != in || dims[1] != out {
			return linalg.WeightMat{}, fmt.Errorf("decoder(gguf): %q dims %v, want [in=%d, out=%d]", name, dims, in, out)
		}
		q4k := mode == quantQ4K && !g.Has(name+".scale")
		if q4k {
			if wm, ok, err := q4kMat(g, name, out, in, rowSrc, native56); ok || err != nil {
				return wm, err
			}
			mode = quantInt8I8
		}
		rowInto := func(r int, dst []float32) error { return ggufRowFiltered(name, into, rowSrc(r), dst) }
		// streamMat covers everything mat/permMat do not route to streamMatBatched (o_proj, down_proj, router, any other
		// non-batched, non-embedding layer tensor), so skipRow4 applies here as in quantizeWMSkipRow4.
		wm, err := streamQuantizedLayer(out, in, mode, needCanonical, skipRow4, rowInto)
		if q4k && err == nil && sink == nil {
			w.NoteKQuant(g, name, wm, in, rowSrc)
		}
		return wm, err
	}
	// streamMatBatched is streamMat's twin for attention Q/K/V and MLP gate/up (isBatchedProjTensor):
	// streamQuantizedBatchedProj instead of streamQuantized, so it may build repacked-only int4 when this load's backend
	// allows it. See quantizeBatchedProjWM for the safety argument.
	streamMatBatched := func(name string, out, in int, mode quantMode, rowSrc func(r int) int) (linalg.WeightMat, error) {
		dims, into, err := ggufScaledRowDequantizer(g, name)
		if err != nil {
			return linalg.WeightMat{}, err
		}
		if len(dims) != 2 || dims[0] != in || dims[1] != out {
			return linalg.WeightMat{}, fmt.Errorf("decoder(gguf): %q dims %v, want [in=%d, out=%d]", name, dims, in, out)
		}
		q4k := mode == quantQ4K && !g.Has(name+".scale")
		if q4k {
			if wm, ok, err := q4kMat(g, name, out, in, rowSrc, native56); ok || err != nil {
				return wm, err
			}
			mode = quantInt8I8
		}
		wm, err := streamQuantizedBatchedProj(out, in, mode, needCanonical, skipRow4, func(r int, dst []float32) error {
			return ggufRowFiltered(name, into, rowSrc(r), dst)
		})
		if q4k && err == nil && sink == nil {
			w.NoteKQuant(g, name, wm, in, rowSrc)
		}
		return wm, err
	}
	// mat loads a tensor as a [out, in] linalg.WeightMat, shape-checked, quantized when
	// requested. Routes attn_q/attn_k/attn_v/ffn_gate/ffn_up through streamMatBatched instead
	// of the plain streamMat (isBatchedProjTensor) — see that function's own doc comment.
	mat := func(name string, out, in int) (linalg.WeightMat, error) {
		if isBatchedProjTensor(name) {
			return streamMatBatched(name, out, in, matmulQuant(quant, name), func(r int) int { return r * in })
		}
		return streamMat(name, out, in, matmulQuant(quant, name), func(r int) int { return r * in })
	}
	// embMatAt streams an embedding-shaped table at q through streamQuantizedEmbed, which may build repacked-only when
	// needCanonical is false and int4 mode applies; see that function.
	embMatAt := func(name string, out, in int, q quantMode) (linalg.WeightMat, error) {
		dims, into, err := ggufScaledRowDequantizer(g, name)
		if err != nil {
			return linalg.WeightMat{}, err
		}
		if len(dims) != 2 || dims[0] != in || dims[1] != out {
			return linalg.WeightMat{}, fmt.Errorf("decoder(gguf): %q dims %v, want [in=%d, out=%d]", name, dims, in, out)
		}
		// into reads the mapping and writes dst alone, so every row range shares it.
		read := func(r int, dst []float32) error { return into(r*in, dst) }
		return streamQuantizedEmbed(out, in, q, needCanonical, !repackForDecode(sink), 1, streamEmbedWorkers(sink), func() rowReader { return read })
	}
	// embInto loads an embedding-shaped table into *dst through embMatAt. A streaming build (sink, streamTablesOn) builds nothing:
	// it registers the table's source with the writer, which writes it from the source rows (streamedTable).
	embInto := func(dst *linalg.WeightMat, name string, out, in int, q quantMode) error {
		if sink == nil || !streamTablesOn {
			m, err := embMatAt(name, out, in, q)
			*dst = m
			return err
		}
		dims, into, err := ggufScaledRowDequantizer(g, name)
		if err != nil {
			return err
		}
		if len(dims) != 2 || dims[0] != in || dims[1] != out {
			return fmt.Errorf("decoder(gguf): %q dims %v, want [in=%d, out=%d]", name, dims, in, out)
		}
		read := func(r int, dst []float32) error { return into(r*in, dst) }
		if sink.tables == nil {
			sink.tables = map[*linalg.WeightMat]*streamedTable{}
		}
		sink.tables[dst] = &streamedTable{rows: out, cols: in, mode: q, newReader: func() rowReader { return read }}
		*dst = linalg.WeightMat{}
		return nil
	}
	// embMat loads the embedding / LM head, which is logit-critical: quantize it with the embedding policy (int8 even in int4
	// mode unless EmbedInt4), not the projection mode.
	embMat := func(dst *linalg.WeightMat, name string, out, in int) error {
		return embInto(dst, name, out, in, quant.embeddingWith(embedInt4))
	}
	// vec loads a 1-D tensor (norm or bias).
	vec := func(name string, n int) ([]float32, error) {
		dims, data, err := g.Tensor(name)
		if err != nil {
			return nil, err
		}
		if len(dims) != 1 || dims[0] != n {
			return nil, fmt.Errorf("decoder(gguf): %q dims %v, want [%d]", name, dims, n)
		}
		return data, nil
	}
	// vnorm loads a norm weight. For the Gemma family (RMSAddOne), llama.cpp's GGUF
	// conversion bakes the (1+w) offset into every stored *norm.weight, so this
	// package's RMSAddOne forward would apply it twice — subtract the 1 back out at
	// load to restore the HF convention the shared descriptor expects. No-op for the
	// other architectures (plain RMSNorm), so all norm loads route through here.
	subOneNorm := arch.RMSAddOne
	vnorm := func(name string, n int) ([]float32, error) {
		v, err := vec(name, n)
		if err != nil {
			return nil, err
		}
		if subOneNorm {
			for i := range v {
				v[i] -= 1
			}
		}
		return v, nil
	}
	// permMat loads a q/k projection and inverts llama.cpp's RoPE row permutation.
	// Because the permutation is a pure reorder of whole rows (output features), it
	// commutes with per-row quantization: rather than permute an f32 buffer then
	// quantize, we dequant the rows straight into HF order — destination row hfRow
	// pulls from GGUF source row (h*hd + 2*j + s) — and quantize in place. See
	// ggufInvPermute for the index derivation.
	permMat := func(name string, out, in, nHead int) (linalg.WeightMat, error) {
		// nHead is a validated head count (>0), but a hostile head_dim can still make out/nHead == 1 (half == 0, a divide-by-zero
		// in the row map below) or odd (the RoPE row permutation is only defined for an even head_dim).
		if nHead <= 0 || out%nHead != 0 || (out/nHead)%2 != 0 {
			return linalg.WeightMat{}, fmt.Errorf("decoder(gguf): %q needs an even head_dim (out=%d, heads=%d)", name, out, nHead)
		}
		hd := out / nHead
		half := hd / 2
		rowSrc := func(hfRow int) int {
			h, rem := hfRow/hd, hfRow%hd
			ggufRow := h*hd + 2*(rem%half) + rem/half
			return ggufRow * in
		}
		if isBatchedProjTensor(name) { // attn_q/attn_k — see mat's own comment
			return streamMatBatched(name, out, in, matmulQuant(quant, name), rowSrc)
		}
		return streamMat(name, out, in, matmulQuant(quant, name), rowSrc)
	}
	// stackedExperts loads a GGUF MoE expert tensor — a 3-D [in, out, nExpert]
	// (fastest-first) blob where each expert occupies a contiguous [out, in]
	// row-major slice — and returns one quantized linalg.WeightMat per expert.
	stackedExperts := func(name string, out, in, nExpert int) ([]linalg.WeightMat, error) {
		dims, into, err := ggufScaledRowDequantizer(g, name)
		if err != nil {
			return nil, err
		}
		if len(dims) != 3 || dims[0] != in || dims[1] != out || dims[2] != nExpert {
			return nil, fmt.Errorf("decoder(gguf): %q dims %v, want [in=%d, out=%d, experts=%d]", name, dims, in, out, nExpert)
		}
		// Each expert occupies a contiguous [out, in] row-major slice; stream its
		// rows directly into a per-expert quantized linalg.WeightMat (no whole-tensor f32).
		res := make([]linalg.WeightMat, nExpert)
		one := func(e int) error {
			rowInto := func(r int, dst []float32) error { return into((e*out+r)*in, dst) }
			var m linalg.WeightMat
			var err error
			// MoE experts: the same skip-row4-only scope as streamMat.
			m, err = streamQuantizedLayer(out, in, matmulQuant(quant, name), needCanonical, skipRow4, rowInto)
			if err != nil {
				return err
			}
			res[e] = m
			return nil
		}
		// The dequantizer is a pure function of the mapped tensor bytes and streamQuantized allocates its own scratch, so
		// concurrent experts share nothing and each lands in its own res[e] (forExperts).
		if err := forExperts(nExpert, sink, one); err != nil {
			return nil, err
		}
		return res, nil
	}
	// fusedSplit streams the output-row range [rowStart, rowStart+rows) of a fused
	// [outTotal, in] tensor into its own quantized linalg.WeightMat — Phi-3's fused
	// attn_qkv (q‖k‖v) and ffn_up (gate‖up), split without a whole-tensor f32 buffer.
	fusedSplit := func(name string, outTotal, in, rowStart, rows int) (linalg.WeightMat, error) {
		dims, into, err := g.RowDequantizer(name)
		if err != nil {
			return linalg.WeightMat{}, err
		}
		if len(dims) != 2 || dims[0] != in || dims[1] != outTotal {
			return linalg.WeightMat{}, fmt.Errorf("decoder(gguf-phi3): %q dims %v, want [in=%d, out=%d]", name, dims, in, outTotal)
		}
		mode := matmulQuant(quant, name)
		rowSrc := func(r int) int { return (rowStart + r) * in }
		q4k := mode == quantQ4K
		if q4k {
			if wm, ok, err := q4kMat(g, name, rows, in, rowSrc, native56); ok || err != nil {
				return wm, err
			}
			mode = quantInt8I8
		}
		rowInto := func(r int, dst []float32) error { return ggufRowFiltered(name, into, rowSrc(r), dst) }
		// Phi-3's fused qkv/gate_up split: the same skip-row4-only scope as streamMat. It is not routed through the
		// isBatchedProjTensor naming convention (GGUF's own fused-tensor special case), so there is no needCanonical or
		// repacked-only branch here.
		var wm linalg.WeightMat
		if skipRow4 {
			wm, err = streamQuantizedSkipRow4(rows, in, mode, rowInto)
		} else {
			wm, err = streamQuantized(rows, in, mode, rowInto)
		}
		if q4k && err == nil && sink == nil {
			w.NoteKQuant(g, name, wm, in, rowSrc)
		}
		return wm, err
	}

	var err error
	if err = embMat(&w.Embed, "token_embd.weight", cfg.VocabSize, hidden); err != nil {
		return nil, err
	}
	if w.FinalNorm, err = vnorm("output_norm.weight", hidden); err != nil {
		return nil, err
	}
	// Separate output head when present; else tied to the embedding.
	arch.TiedLMHead = true
	if g.Has("output.weight") {
		if err = embMat(&w.LMHead, "output.weight", cfg.VocabSize, hidden); err != nil {
			return nil, err
		}
		if quant == quantQ4K && sink == nil {
			w.NoteKQuant(g, "output.weight", w.LMHead, hidden, func(r int) int { return r * hidden })
		}
		arch.TiedLMHead = false
	}

	// Streaming transcode: emit the header + globals now; each layer is written and
	// freed as it loads (below), so the whole model never sits resident.
	if sink != nil {
		// The writer gates gemma4's head extras and per-layer tail on arch, exactly as writeBundle does
		// for the resident path (it sets wr.arch = w.arch); a streaming writer built without it would
		// silently drop both.
		sink.arch = arch
		// gemma4's head carries its model-level PLE inputs, which load in its own branch below — so
		// it writes the head there, once they exist. Every other family's head is complete here.
		if arch.Gemma4 == nil {
			if err := sink.writeHeadGlobals(w, id); err != nil {
				return nil, err
			}
		}
	}

	b := &ggufBuild{arch: arch, g: g, w: w, quant: quant, needCanonical: needCanonical, skipRow4: skipRow4, sink: sink, abort: abort,
		hidden: hidden, streamMat: streamMat, mat: mat, vec: vec, vnorm: vnorm, stackedExperts: stackedExperts, hd: hd, permMat: permMat, cfg: cfg, embInto: embInto, id: id, fusedSplit: fusedSplit}

	if q := arch.Qwen35; q != nil {
		return b.loadQwen35(q)
	}

	if arch.GPTOSS != nil {
		return b.loadGptOss()
	}

	if arch.Laguna != nil {
		return b.loadLaguna()
	}

	if gp := arch.Granite; gp != nil {
		return b.loadGranite(gp)
	}

	if npar := arch.Nemotron; npar != nil {
		return b.loadNemotron(npar)
	}

	if g4 := arch.Gemma4; g4 != nil {
		return b.loadGemma4(g4)
	}

	if lp := arch.Llama4; lp != nil {
		return b.loadLlama4(lp)
	}

	return b.loadLayers()
}

// ggufScaledRowDequantizer wraps g.RowDequantizer(name) and, when a companion `<name>.scale` tensor is present (llama.cpp
// PR #21445's GGUF NVFP4 per-tensor F32 [1] multiplier), scales each dequantized row in place before quantization.
func ggufScaledRowDequantizer(g *embed.GGUFFile, name string) ([]int, func(offset int, dst []float32) error, error) {
	dims, into, err := g.RowDequantizer(name)
	if err != nil || !g.Has(name+".scale") {
		return dims, into, err
	}
	sDims, sData, sErr := g.Tensor(name + ".scale")
	if sErr != nil {
		return nil, nil, fmt.Errorf("decoder(gguf): %q: %w", name+".scale", sErr)
	}
	if len(sDims) != 1 || sDims[0] != 1 || len(sData) != 1 {
		return nil, nil, fmt.Errorf("decoder(gguf): %q dims %v, want [1]", name+".scale", sDims)
	}
	scale := sData[0]
	if scale == 1 {
		return dims, into, nil
	}
	return dims, func(offset int, dst []float32) error {
		if err := into(offset, dst); err != nil {
			return err
		}
		for i := range dst {
			dst[i] *= scale
		}
		return nil
	}, nil
}

// ggufQKPermuted reports whether llama.cpp's GGUF conversion permuted this architecture's q/k weights (and their per-row
// biases and head-dim norms). It permutes only for the NORM rope type: llama and its derivatives, including mellum and
// dense granite (llama.cpp's GraniteModel inherits LlamaModel's undo_permute). The NEOX rope type (qwen2/qwen3/gemma and
// the other modern families) leaves q/k in HF rotate_half order, so no un-permutation is needed. Unknown archs default to
// NEOX (no permute), the common modern case.
func ggufQKPermuted(archName string) bool {
	switch archName {
	case "llama", "mellum", "granite":
		return true
	default:
		return false
	}
}

// ggufInvPermuteVec inverts llama.cpp's q/k RoPE permutation for a single
// head_dim-wide vector (a QK-norm weight, shared across heads): HF position
// (s*half + j) comes from GGUF position (2*j + s). Mirrors ggufInvPermute with
// nHead=1 and no input dimension.
func ggufInvPermuteVec(v []float32) []float32 {
	hd := len(v)
	half := hd / 2
	res := make([]float32, hd)
	for s := range 2 {
		for j := range half {
			res[s*half+j] = v[2*j+s]
		}
	}
	return res
}

// ggufInvPermute inverts llama.cpp's q/k weight permutation. llama.cpp stores
// the projection rows in interleaved-pair order for its RoPE; this package uses
// HF's rotate_half (first-half / second-half) order. For head h, the HF row
// (s*half + j) — s∈{0,1} selecting the half, j the position within it — comes
// from the GGUF row (2*j + s). w is [out, in] row-major; out = nHead*headDim.
func ggufInvPermute(w []float32, out, in, nHead int) []float32 {
	hd := out / nHead
	half := hd / 2
	res := make([]float32, len(w))
	for h := range nHead {
		for s := range 2 {
			for j := range half {
				hfRow := h*hd + s*half + j
				ggufRow := h*hd + 2*j + s
				copy(res[hfRow*in:hfRow*in+in], w[ggufRow*in:ggufRow*in+in])
			}
		}
	}
	return res
}
