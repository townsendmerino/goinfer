package decoder

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"

	"github.com/townsendmerino/aikit/linalg"
	"hash/crc32"
	"io"
	"math"
	"unsafe"
)

// This file defines a versioned binary format for an already-quantized *Weights
// bundle (a ".giw" — goinfer weights), so the resident weights can be produced
// once at build time and embedded, skipping the GGUF dequant+requant on every
// launch. The big int8/int4 weight arrays are ALIASED directly over the input
// slice at load (zero-copy — this is the speed/RAM win); the small per-row scale
// floats and the norm/bias vectors are COPIED (the input isn't guaranteed
// 4-byte aligned, and unaligned float reads are UB).
//
// Discipline mirrors ken's index_serialize.go: magic + version + a config/quant
// guard + CRC, and a LAZY FALLBACK — any mismatch returns a typed error and never
// panics, so the caller can rebuild from the GGUF.
//
// Format (little-endian throughout):
//
//	magic   [5]byte = "GINFW"
//	version uint32
//	quant   uint32   (quantMode enum: first-weight kind — the legacy tag, validated on read)
//	id      str      (model identity — source name/hash, for tooling)
//	config  str      (Config as JSON; arch is re-derived from it on load)
//	quantLabel str   (v5+: the resolved quant label — int4|int4mix|int8int8|int8|native — or "" to
//	                  fall back to inference; the reader PREFERS this over re-deriving from kinds)
//	Embed, LMHead, PosEmbed     weightMat
//	FinalNorm, FinalNormBias    f32
//	numLayers uint32
//	  per layer: the LayerWeights fields, in declaration order, then a v2 hybrid
//	  tail (uint8 kind: 0 none | 1 DeltaNet | 2 gated-softmax) with the
//	  qwen3_5_moe per-layer delta / qattn f32 tensors when set.
//	crc     uint32   (CRC32-IEEE over every preceding byte)
//
// str  = uint32 len + len bytes
// f32  = uint32 len + len*4 LE-float32 bytes   (len 0 ⇒ nil on load)
// i8   = uint32 len + len bytes                (aliased on load)
// raw  = uint32 len + len bytes                (aliased on load)
// weightMat = uint8 kind (0 empty|1 f32|2 q8|3 q4|4 q4-row4); if non-empty:
//             int32 rows, cols, group; uint8 w8a8; then the kind's arrays.
//             kind 4 (v7+) is kind 3's arrays (q4s, q4 — canonical, always present
//             and always authoritative) followed by q4Row4Scales, q4Row4 (the arm64
//             split-half + 4-row-interleaved layout, docs/task-w4a8-neon-bandwidth.md
//             "Format follow-on") — the on-disk form of the row4 layout the load-time
//             repack (decoder/weightmat.go's repackW4A8Row4IfEligible) otherwise
//             builds in RAM. Opt-in at prequant time (SerializeWeightsRow4/
//             SerializeWeightsToRow4/StreamTranscodeGGUF's row4 param) for shapes
//             RepackW4A8Row4/RepackW4A8Row4Scales accept; every other int4 tensor
//             still writes kind 3. Bit-identical dispatch either way
//             (TestDotW4A8SplitHalf4Row_bitIdenticalToCanonical) — this is a storage
//             choice, not a numerics one, so no golden depends on which kind a
//             tensor took.
//
// v2 added the per-layer hybrid tail so the qwen3_5_moe (DeltaNet + gated-softmax)
// family round-trips through .giw; v1 blobs (no tail) are rejected by the version
// guard and rebuilt from the source GGUF.
//
// v8 added the per-layer LFM2 short-conv mixer (presence byte + inProj/convW/outProj), the same
// shape as the v6 Mamba-2 block. Before it, `grep shortConv decoder/serialize.go` returned nothing:
// cmd/prequant wrote a CRC-valid bundle with every conv layer's mixer missing, selfCheck passed
// because it only Loads, and the first forward nil-dereferenced in the decode goroutine
// (audit-2026-09-02 C-03, a regression of R3).

const (
	giwMagic       = "GINFW"
	giwVersion     = 8 // v8: the per-layer LFM2 short-conv mixer — see the format comment above
	giwMinReadV    = 3 // read v3/v4 too (each version only ADDS: v4 the gemma4-gated tail, v5 the quant-label field, v7 kind 4, v8 shortConv; older bundles stay valid and fall back to inference)
	giwV4Gemma4    = 4 // the version at/after which the gemma4 tail is present
	giwV6Tail      = 6 // the version at/after which the completeness tail is present (GProj / AttnSinks / expert biases / MLA / Mamba-2)
	giwV8ShortConv = 8 // the version at/after which the LFM2 short-conv tail is present
	// v3: per-layer RouterBias (DeepSeek/GLM e_score_correction_bias); v2: qwen3_5_moe hybrid tail
	// Sanity ceilings on the count fields, generous vs any real checkpoint
	// (largest models: ~120 layers, a few hundred experts) but low enough that a
	// corrupt/hostile blob can't drive a multi-GB make() before the body reader
	// hits its first short read. A LayerWeights is a large struct, so an unbounded
	// layer count is the worst offender.
	maxSerializedLayers  = 4096
	maxSerializedExperts = 4096
)

// SerializeError is returned by LoadSerializedWeights on any magic/version/
// quant/CRC mismatch. It is distinct so callers can fall back to building from
// the source GGUF rather than treating a stale blob as fatal.
type SerializeError struct{ Reason string }

func (e *SerializeError) Error() string { return "decoder: serialized weights: " + e.Reason }

// canSerialize reports why a model's per-layer state cannot round-trip through the .giw format,
// or nil if it can. The writer expresses the standard attention+MLP(+MoE) block plus
// qwen3_5_moe's DeltaNet/gated-softmax extras and (v4) the gemma4 PLE / layer_scalar /
// KV-share / MoE tail; it does NOT write MLA latent projections (DeepSeek/Kimi) or Mamba-2
// SSM weights (Granite/Nemotron) — so serializing those yields a CRC-valid bundle that
// nil-derefs at the first forward. Refuse those families up front rather than emit silent
// garbage (C2).
func canSerialize(a *Architecture) *SerializeError {
	// EMPTY AS OF v6 (2026-08-19), and that is the point: every registered family is representable.
	//
	// This used to be a hand-maintained blocklist of families the writer could not express — and it
	// DRIFTED, twice, silently: gpt-oss rode it while dropping its attention sinks (bundles loaded
	// clean and generated wrong text) and Laguna rode it while producing bundles the reader refused.
	// The v6 completeness tail writes GProj, AttnSinks, per-expert biases, and the MLA / Mamba-2
	// sub-structs, so there is nothing left to list.
	//
	// KEEP THE FUNCTION. A future family may genuinely be unrepresentable (a new per-layer state
	// with no field here), and refusing is the correct answer for it — an empty list is today's
	// truth, not a reason to delete the mechanism. What guards against the drift returning is
	// TestSerializeCensus_noSilentFieldDrop, which asks the STRUCT whether a round-trip lost
	// anything rather than asking a human whether they remembered to update this.
	return nil
}

// SerializeWeights writes the resident weight bundle (already quantized to its
// current precision) to a flat little-endian blob suitable for embedding. id is
// an opaque model-identity string (e.g. the source filename) stored for tooling.
func SerializeWeights(w *Weights, id string) ([]byte, error) {
	wr := &giwWriter{}
	if err := wr.writeBundle(w, id); err != nil {
		return nil, err
	}
	wr.u32(crc32.ChecksumIEEE(wr.buf)) // CRC over the body; appended last
	return wr.buf, nil
}

// SerializeWeightsTo streams the same bundle directly to out, never materializing
// the whole blob in memory — so prequantizing a large model peaks at ~the resident
// weight size, not 2× (resident + blob). It returns the number of bytes written
// (body + trailing CRC). The big int8/int4 arrays are written straight from the
// resident slices (one write each); only the small per-tensor scale/norm vectors
// are buffered transiently. out should be a regular file for the prequant path.
func SerializeWeightsTo(out io.Writer, w *Weights, id string) (int64, error) {
	wr := &giwWriter{sink: out}
	if err := wr.writeBundle(w, id); err != nil {
		return wr.n, err
	}
	var crc [4]byte
	binary.LittleEndian.PutUint32(crc[:], wr.crc) // running CRC over the body
	if _, err := out.Write(crc[:]); err != nil {
		return wr.n, err
	}
	return wr.n + 4, nil
}

// SerializeWeightsRow4 is SerializeWeights, but ALSO opts every eligible int4
// tensor into weightMat kind 4 — the on-disk arm64 split-half + 4-row-
// interleaved layout (docs/task-w4a8-neon-bandwidth.md's "Format follow-on"),
// so the paged-MoE path can use the faster kernel without an in-RAM repack.
// Never the default: SerializeWeights (kind 3 only) is what every existing
// caller gets and stays unaffected by this function's existence. A tensor
// whose shape RepackW4A8Row4/RepackW4A8Row4Scales reject (the router, or any
// int4 tensor not a multiple of 4 rows / group cols), or a run on a non-arm64
// build, falls back to kind 3 automatically — this is always safe to call.
func SerializeWeightsRow4(w *Weights, id string) ([]byte, error) {
	wr := &giwWriter{row4: true}
	if err := wr.writeBundle(w, id); err != nil {
		return nil, err
	}
	wr.u32(crc32.ChecksumIEEE(wr.buf))
	return wr.buf, nil
}

// SerializeWeightsToRow4 is SerializeWeightsTo with the same kind-4 opt-in as
// SerializeWeightsRow4 — see that function's doc for the fallback contract.
func SerializeWeightsToRow4(out io.Writer, w *Weights, id string) (int64, error) {
	wr := &giwWriter{sink: out, row4: true}
	if err := wr.writeBundle(w, id); err != nil {
		return wr.n, err
	}
	var crc [4]byte
	binary.LittleEndian.PutUint32(crc[:], wr.crc)
	if _, err := out.Write(crc[:]); err != nil {
		return wr.n, err
	}
	return wr.n + 4, nil
}

// writeBundle writes the bundle body (everything but the trailing CRC) via the
// writer's current sink (buffer or stream). Shared by SerializeWeights and
// SerializeWeightsTo so the field order can't drift between them or from the reader.
func (wr *giwWriter) writeBundle(w *Weights, id string) error {
	if err := canSerialize(w.arch); err != nil {
		return err
	}
	wr.arch = w.arch // gates the gemma4 model-level PLE + per-layer tail
	if err := wr.writeHeadGlobals(w, id); err != nil {
		return err
	}
	for i := range w.Layers {
		wr.layer(&w.Layers[i])
	}
	return wr.err
}

// writeHeadGlobals writes everything up to and including the layer count: the
// header (magic/version/quant/id/config), the global tensors, and u32(len(Layers)).
// Split out so a streaming transcode can emit the head, then produce-write-free each
// layer one at a time (peak RAM ~one layer, not the whole model) before the loop in
// writeBundle. The layer count is len(w.Layers), so the streamer must allocate
// w.Layers to NumLayers up front even though it never fills them all at once.
func (wr *giwWriter) writeHeadGlobals(w *Weights, id string) error {
	cfgJSON, err := json.Marshal(w.Cfg)
	if err != nil {
		return fmt.Errorf("decoder: marshal config: %w", err)
	}
	wr.raw([]byte(giwMagic))
	wr.u32(giwVersion)
	wr.u32(uint32(w.quantMode()))
	wr.str(id)
	wr.bytesField(cfgJSON)

	// v5: the resolved quant label, so the reader need not re-infer it (the source of truth is
	// recorded, not reconstructed).
	//
	// GATED ON DATA AVAILABILITY, NOT ON WHICH WRITER IS IN USE (B11). The condition used to be
	// `wr.sink == nil` — "are we the buffered writer" — on the theory that only the buffered path
	// has full weights in hand. That conflated two different questions: which io.Writer the bytes
	// go to, and whether w.Layers is actually populated yet. They agree for the true incremental
	// GGUF transcode (gguf.go's per-family streaming path calls writeHeadGlobals on a freshly
	// make()'d, all-zero Layers slice BEFORE streaming any layer in — quantLabel() truly cannot see
	// real data there, and its default case returns "native", a REAL quant mode, not an empty
	// string, so calling it unconditionally would have baked a FALSE "native" label into every
	// genuinely-streamed bundle). But they disagree for a caller that already has a fully-loaded
	// *Weights and simply chooses the streaming API for its I/O shape — internal/prequant and the
	// qwen35 GGUF branch (a dedicated loader that fully materializes w, THEN calls
	// SerializeWeightsTo) both do exactly this — and there the label WAS resolvable, just skipped
	// because the wrong signal was being tested. That mismatch is B11: a buffered and a streamed
	// call on the SAME fully-loaded model produced non-identical bytes for no reason tied to the
	// data itself, differing by exactly len("int8int8") = 8 bytes in the length-prefixed field.
	label := ""
	if w.hasPopulatedLayers() {
		label = w.quantLabel()
	}
	wr.str(label)

	wr.weightMat(&w.Embed)
	wr.weightMat(&w.LMHead)
	wr.weightMat(&w.PosEmbed)
	wr.f32(w.FinalNorm)
	wr.f32(w.FinalNormBias)

	// v4: Gemma 4 model-level Per-Layer-Embedding inputs (empty on the PLE-free
	// E-model/26B variants, but present as empty WeightMats so the layout is stable).
	// Gated on gemma4 so every other family's bundle is byte-identical to v3.
	if wr.arch != nil && wr.arch.gemma4 != nil {
		wr.weightMat(&w.PerLayerTokenEmbed)
		wr.weightMat(&w.PerLayerModelProj)
		wr.f32(w.PerLayerProjNorm)
		// FFNPerLayer (the E-models' variable per-layer FFN widths) is json:"-" on
		// Config, so it does NOT survive the config-JSON round-trip — without it ffnAt()
		// falls back to IntermediateDim and mis-sizes the MLP matmuls. Carry it here.
		ffn := wr.arch.gemma4.FFNPerLayer
		wr.u32(uint32(len(ffn)))
		for _, v := range ffn {
			wr.u32(uint32(v))
		}
	}

	wr.u32(uint32(len(w.Layers)))
	return wr.err
}

// LoadSerializedWeights reconstructs a *Weights from a SerializeWeights blob
// WITHOUT any dequant/requant. Big int8/int4 arrays are aliased into data
// (zero-copy); float arrays are copied. data MUST stay alive for the returned
// model's lifetime (the aliased slices point into it). On any magic/version/
// quant/arch/CRC mismatch it returns a *SerializeError so the caller can fall
// back to the GGUF.
func LoadSerializedWeights(data []byte) (*Weights, error) {
	r := &giwReader{data: data}
	if got := r.rawN(len(giwMagic)); string(got) != giwMagic {
		return nil, &SerializeError{fmt.Sprintf("bad magic %q (want %q)", got, giwMagic)}
	}
	v := r.u32()
	if v < giwMinReadV || v > giwVersion {
		return nil, &SerializeError{fmt.Sprintf("format version %d, this build reads %d..%d", v, giwMinReadV, giwVersion)}
	}
	r.version = v
	quant := quantMode(r.u32())
	_ = r.str() // id — stored for tooling, not validated here
	cfgJSON := r.bytesField()
	// v5+: the recorded resolved quant label (may be "" for a streamed bundle → infer). Absent
	// entirely in v3/v4 bundles, which keep working unchanged.
	bakedQuant := ""
	if v >= 5 {
		bakedQuant = r.str()
	}
	if r.err != nil {
		return nil, &SerializeError{"truncated header"}
	}

	// CRC: verify the whole payload (everything before the trailing crc word)
	// before trusting any offsets/lengths.
	if len(data) < 4 {
		return nil, &SerializeError{"too short"}
	}
	body, want := data[:len(data)-4], binary.LittleEndian.Uint32(data[len(data)-4:])
	if got := crc32.ChecksumIEEE(body); got != want {
		return nil, &SerializeError{fmt.Sprintf("CRC mismatch (got %08x want %08x) — corrupt or truncated", got, want)}
	}

	var cfg Config
	if err := json.Unmarshal(cfgJSON, &cfg); err != nil {
		return nil, &SerializeError{"config json: " + err.Error()}
	}
	arch, _, err := resolveArchitecture(&cfg)
	if err != nil {
		return nil, &SerializeError{"arch: " + err.Error()}
	}
	r.arch = arch // gates the v4 gemma4 model-level + per-layer tail

	w := &Weights{Cfg: cfg, arch: arch, backing: data, bakedQuant: bakedQuant}
	w.Embed = r.weightMat()
	w.LMHead = r.weightMat()
	// A tied checkpoint (Qwen3/Qwen2.5-0.5B/Llama-3.2 with no output.weight) round-trips with an
	// empty LMHead; every other loader sets TiedLMHead from lm_head presence, so mirror that here.
	// Without it the head reads as untied+empty and the forward emits ALL-ZERO logits — greedy
	// loops on token 0, sampling is uniform noise, with no error (C2).
	arch.TiedLMHead = w.LMHead.Rows() == 0
	w.PosEmbed = r.weightMat()
	w.FinalNorm = r.f32()
	w.FinalNormBias = r.f32()
	if r.version >= giwV4Gemma4 && arch.gemma4 != nil { // v4 gemma4 model-level PLE inputs
		w.PerLayerTokenEmbed = r.weightMat()
		w.PerLayerModelProj = r.weightMat()
		w.PerLayerProjNorm = r.f32()
		nf := int(r.u32())
		if nf < 0 || nf > maxSerializedLayers {
			return nil, &SerializeError{"implausible ffn-per-layer count"}
		}
		if nf > 0 {
			ffn := make([]int, nf)
			for i := range ffn {
				ffn[i] = int(r.u32())
			}
			arch.gemma4.FFNPerLayer = ffn
		}
	}
	n := int(r.u32())
	if n < 0 || n > maxSerializedLayers {
		return nil, &SerializeError{"implausible layer count"}
	}
	w.Layers = make([]LayerWeights, n)
	for i := range w.Layers {
		r.layer(&w.Layers[i])
	}
	if r.err != nil {
		return nil, &SerializeError{"truncated body: " + r.err.Error()}
	}
	if quant != w.quantMode() {
		// the serialized quant tag must match what the tensors actually are
		return nil, &SerializeError{fmt.Sprintf("quant tag %d disagrees with tensor kinds", quant)}
	}
	if err := validateShapes(w, arch); err != nil {
		return nil, err
	}
	return w, nil
}

// validateShapes cross-checks the deserialized tensors against the architecture's
// expected dims (audit C-06). The .giw reader validates only internal consistency
// (array length vs the blob's own rows/cols), so a bundle whose Router declares
// rows = NumExperts+K, or an Embed/LMHead with the wrong vocab, passes every reader
// check and then writes past a config-sized scratch slice at decode
// (moeMLP's `make([]float32, NumExperts)`, the qDim/kvDim/vocab decodeScratch
// buffers) — a heap corruption from caller-supplied bytes (LoadSerializedWeights is
// exported). The GGUF/safetensors loaders already do this cross-check; only .giw skipped.
//
// Universal invariants (vocab + expert count are uniform in every serializable
// family) are always checked. The attention/FFN projection dims are checked only for
// uniform-geometry families: gemma-4 carries per-layer geometry (FFNPerLayer / two-geom
// head dims), so a model-level dim would false-reject it — its own descriptor + tiny
// goldens cover that path.
func validateShapes(w *Weights, arch *Architecture) *SerializeError {
	eq := func(name string, got, want int) *SerializeError {
		if got != want {
			return &SerializeError{fmt.Sprintf("%s: %d rows, arch expects %d", name, got, want)}
		}
		return nil
	}
	// vec checks a per-layer f32 vector (bias / norm weight) whose length the blob controls but the
	// forward indexes at an arch-derived width — addBias iterates over the projection output and
	// rmsNorm indexes weight[0:dim], so a short vector slice-panics in the decode goroutine and a long
	// one is silently mis-consumed (audit R-07). 0 = absent (a family that omits it), allowed.
	vec := func(name string, got, want int) *SerializeError {
		if got != 0 && got != want {
			return &SerializeError{fmt.Sprintf("%s: len %d, arch expects %d", name, got, want)}
		}
		return nil
	}
	// req is vec for a vector the family's forward dereferences UNCONDITIONALLY, where
	// "absent" is not a family that omits it but a bundle that is missing it. vec's `got == 0
	// ⇒ allowed` is what let a pre-v6 gpt-oss sidecar through: it is still within
	// giwMinReadV, still "fresh" by mtime, reads AttnSinks as nil, passes validateShapes, and
	// panics at forward_gptoss.go's `lw.AttnSinks[qh]` on the first request (M-11). Same
	// shape as the LFM2 conv presence check below (C-03).
	req := func(name string, got, want int) *SerializeError {
		if got == 0 {
			return &SerializeError{fmt.Sprintf("%s: absent, arch requires len %d — the bundle "+
				"predates this field (rewrite the .giw sidecar; mtime freshness cannot see a "+
				"missing tensor)", name, want)}
		}
		return vec(name, got, want)
	}
	if w.Embed.Rows() > 0 {
		if e := eq("Embed", w.Embed.Rows(), arch.VocabSize); e != nil {
			return e
		}
	}
	if w.LMHead.Rows() > 0 { // 0 = tied (validated via Embed above)
		if e := eq("LMHead", w.LMHead.Rows(), arch.VocabSize); e != nil {
			return e
		}
	}
	// The blob controls len(w.Layers), but the forward indexes arch.NumLayers — a short blob is an
	// out-of-bounds layer read the per-layer checks below never reach (C-06).
	if len(w.Layers) != arch.NumLayers {
		return &SerializeError{fmt.Sprintf("layer count: blob has %d, arch expects %d", len(w.Layers), arch.NumLayers)}
	}
	// The per-layer accessors (headDimAt/kvHeadsAt/ffnAt) collapse to the uniform Architecture fields
	// for every non-gemma-4 family, so ONE per-layer check set covers all families — gemma-4 included,
	// closing the old `uniform := arch.gemma4 == nil` skip that left its projections unvalidated. Each
	// check is guarded by Rows()>0, so a family that legitimately omits a projection (a routed layer's
	// empty dense FFN, gemma-4's MLP living in the MoE sub-block) is not false-rejected.
	for i := range w.Layers {
		lw := &w.Layers[i]
		hd := arch.headDimAt(i)
		// headsAt(i), not NumHeads: Laguna varies the QUERY head count per layer (48 on its
		// full-attention layers, 64 on the sliding ones), and this line was the last uniform-geometry
		// assumption in the reader — it rejected a correctly-written Laguna bundle with
		// "layer 1 QProj: 128 rows, arch expects 64". headDimAt/kvHeadsAt/ffnAt were already
		// per-layer here; this one was missed because no serializable family had needed it yet.
		qDim, kvDim, ffn := arch.headsAt(i)*hd, arch.kvHeadsAt(i)*hd, arch.ffnAt(i)
		for _, c := range []struct {
			name string
			got  int
			want int
		}{
			{"QProj", lw.QProj.Rows(), qDim},
			{"KProj", lw.KProj.Rows(), kvDim},
			{"VProj", lw.VProj.Rows(), kvDim},
			{"OProj", lw.OProj.Rows(), arch.HiddenDim},
			{"GateProj", lw.GateProj.Rows(), ffn},
			{"UpProj", lw.UpProj.Rows(), ffn},
			{"DownProj", lw.DownProj.Rows(), arch.HiddenDim},
		} {
			if c.got > 0 {
				if e := eq(fmt.Sprintf("layer %d %s", i, c.name), c.got, c.want); e != nil {
					return e
				}
			}
		}
		// GProj (Laguna's attention output gate) is legal at EITHER width — per-head (one scalar per
		// query head) or per-element (the full qDim) — and which one ships is decided by the tensor,
		// not the config (XS.2 declares per-element and ships per-head). So both are accepted and
		// anything else is rejected, rather than leaving the field unchecked.
		if g := lw.GProj.Rows(); g > 0 && g != arch.headsAt(i) && g != qDim {
			return &SerializeError{fmt.Sprintf("layer %d GProj: %d rows, arch expects %d (per-head) or %d (per-element)",
				i, g, arch.headsAt(i), qDim)}
		}
		// Per-layer f32 vectors: biases feed addBias over the projection output, norm weights feed
		// rmsNorm at their consumed width (QK-norm is per-head-dim hd; the block norms are hidden).
		// The blob controls their length; the forward indexes an arch width with no check (R-07).
		for _, c := range []struct {
			name string
			got  int
			want int
		}{
			{"QBias", len(lw.QBias), qDim}, {"KBias", len(lw.KBias), kvDim},
			{"VBias", len(lw.VBias), kvDim}, {"OBias", len(lw.OBias), arch.HiddenDim},
			{"QNorm", len(lw.QNorm), hd}, {"KNorm", len(lw.KNorm), hd},
			{"PreAttnNorm", len(lw.PreAttnNorm), arch.HiddenDim}, {"PostAttnNorm", len(lw.PostAttnNorm), arch.HiddenDim},
			{"PreMLPNorm", len(lw.PreMLPNorm), arch.HiddenDim}, {"PostMLPNorm", len(lw.PostMLPNorm), arch.HiddenDim},
			{"UpBias", len(lw.UpBias), ffn}, {"DownBias", len(lw.DownBias), arch.HiddenDim},
		} {
			if e := vec(fmt.Sprintf("layer %d %s", i, c.name), c.got, c.want); e != nil {
				return e
			}
		}
		// LFM2's short-conv mixer. Its three tensors are flat f32 slices the forward indexes at
		// arch-derived widths, so a short one slice-panics in the decode goroutine exactly as R-07
		// describes for the bias vectors. And the PRESENCE check is the load-bearing half here: a
		// conv layer whose mixer is absent is what a pre-v8 .giw hands back, and the panic it
		// produces at the first forward is the defect this validation exists to convert into a
		// refusal at load (audit-2026-09-02 C-03).
		if arch.lfm2 != nil {
			cd, k := arch.lfm2.ConvDim, arch.lfm2.ConvLCache
			isConv := lw.QProj.Rows() == 0 // a conv layer loads no attention projections
			switch {
			case isConv && lw.shortConv == nil:
				return &SerializeError{fmt.Sprintf("layer %d: lfm2 conv layer has no short-conv "+
					"mixer — the bundle was written before the v8 tail and would nil-deref at the "+
					"first forward", i)}
			case lw.shortConv == nil:
			default:
				c := lw.shortConv
				for _, ck := range []struct {
					name string
					got  int
					want int
				}{
					{"shortConv.inProj", len(c.inProj), 3 * cd * arch.HiddenDim},
					{"shortConv.convW", len(c.convW), cd * k},
					{"shortConv.outProj", len(c.outProj), arch.HiddenDim * cd},
				} {
					if e := eq(fmt.Sprintf("layer %d %s", i, ck.name), ck.got, ck.want); e != nil {
						return e
					}
				}
			}
		}
		// gemma-4 per-layer-embedding (PLE) branch: PLEGate/PLEProj are matmul'd and PostPLENorm
		// rmsNorm'd at fixed widths; a wrong-rows blob OOB-panics or silently truncates the PLE
		// activation (audit F-02, R-07 remainder).
		if arch.gemma4 != nil {
			if pleDim := arch.gemma4.HiddenSizePerLayerInput; pleDim > 0 {
				if lw.PLEGate.Rows() > 0 {
					if e := eq(fmt.Sprintf("layer %d PLEGate", i), lw.PLEGate.Rows(), pleDim); e != nil {
						return e
					}
				}
				if lw.PLEProj.Rows() > 0 {
					if e := eq(fmt.Sprintf("layer %d PLEProj", i), lw.PLEProj.Rows(), arch.HiddenDim); e != nil {
						return e
					}
				}
			}
			if e := vec(fmt.Sprintf("layer %d PostPLENorm", i), len(lw.PostPLENorm), arch.HiddenDim); e != nil {
				return e
			}
		}
		// gemma-4 dense‖MoE sub-block (l.gemma4moe): the standard Router/Experts are empty for gemma-4,
		// so the arch.MoE block below never validates it. The router feeds make([]float32, nE) and each
		// expert index runs to nE — a short router mis-routes silently, a long one or ne != NumExperts
		// panics in the decode goroutine (R-07). Cross-check against the arch.
		if mo := lw.gemma4moe; mo != nil && arch.MoE != nil {
			nE, moeInter := arch.MoE.NumExperts, arch.MoE.IntermediateDim
			if e := eq(fmt.Sprintf("layer %d gemma4moe router", i), mo.routerProj.Rows(), nE); e != nil {
				return e
			}
			if len(mo.expertsGateUp) != nE || len(mo.expertsDown) != nE {
				return &SerializeError{fmt.Sprintf("layer %d gemma4moe expert count: gate|up %d / down %d, arch expects %d", i, len(mo.expertsGateUp), len(mo.expertsDown), nE)}
			}
			if e := vec(fmt.Sprintf("layer %d gemma4moe perExpertScale", i), len(mo.perExpertScale), nE); e != nil {
				return e
			}
			// routerScale scales the [hidden] router input; the three branch norms rmsNorm at hidden —
			// a short slice OOB-panics in the decode goroutine (audit F-02, R-07 remainder).
			for _, c := range []struct {
				name string
				got  int
			}{
				{"gemma4moe routerScale", len(mo.routerScale)},
				{"gemma4moe postFFNNorm1", len(mo.postFFNNorm1)},
				{"gemma4moe preFFNNorm2", len(mo.preFFNNorm2)},
				{"gemma4moe postFFNNorm2", len(mo.postFFNNorm2)},
			} {
				if e := vec(fmt.Sprintf("layer %d %s", i, c.name), c.got, arch.HiddenDim); e != nil {
					return e
				}
			}
			for xe := range mo.expertsGateUp {
				if e := eq(fmt.Sprintf("layer %d gemma4moe expert %d gate|up", i, xe), mo.expertsGateUp[xe].Rows(), 2*moeInter); e != nil {
					return e
				}
				if e := eq(fmt.Sprintf("layer %d gemma4moe expert %d down", i, xe), mo.expertsDown[xe].Rows(), arch.HiddenDim); e != nil {
					return e
				}
			}
		}
		if arch.MoE != nil {
			// RouterBias is addBias'd over the [NumExperts] router logits; the shared expert's
			// Gate/Up write SharedIntermediateDim scratch, Down writes hidden, and SharedGate is the
			// scalar sigmoid gate (1 row). A short blob panics; validate them (audit F-02).
			if e := vec(fmt.Sprintf("layer %d RouterBias", i), len(lw.RouterBias), arch.MoE.NumExperts); e != nil {
				return e
			}
			for _, c := range []struct {
				name string
				got  int
				want int
			}{
				{"SharedExpert Gate", lw.SharedExpert.Gate.Rows(), arch.MoE.SharedIntermediateDim},
				{"SharedExpert Up", lw.SharedExpert.Up.Rows(), arch.MoE.SharedIntermediateDim},
				{"SharedExpert Down", lw.SharedExpert.Down.Rows(), arch.HiddenDim},
				{"SharedGate", lw.SharedGate.Rows(), 1},
			} {
				if c.got > 0 {
					if e := eq(fmt.Sprintf("layer %d %s", i, c.name), c.got, c.want); e != nil {
						return e
					}
				}
			}
			// Router feeds moeMLP's make([]float32, NumExperts) — the exploit the audit names.
			if lw.Router.Rows() > 0 {
				if e := eq(fmt.Sprintf("layer %d Router", i), lw.Router.Rows(), arch.MoE.NumExperts); e != nil {
					return e
				}
			}
			// Per-expert Gate/Up (moe intermediate width) and Down (hidden) — same config-sized-scratch
			// write class as the dense projections. Empty for families that stack experts elsewhere.
			for xe := range lw.Experts {
				ex := &lw.Experts[xe]
				for _, c := range []struct {
					name string
					got  int
					want int
				}{
					{"Gate", ex.Gate.Rows(), arch.MoE.IntermediateDim},
					{"Up", ex.Up.Rows(), arch.MoE.IntermediateDim},
					{"Down", ex.Down.Rows(), arch.HiddenDim},
				} {
					if c.got > 0 {
						if e := eq(fmt.Sprintf("layer %d expert %d %s", i, xe, c.name), c.got, c.want); e != nil {
							return e
						}
					}
				}
			}
		}

		// M-11: the v6 completeness tail. forward_gptoss.go indexes AttnSinks[qh] for every
		// head and addBias iterates every expert bias with no nil check, so for THIS family
		// these are required, not optional — and the trigger is real rather than theoretical:
		// a gpt-oss sidecar written before v6 is sink-free, within giwMinReadV, and judged
		// fresh by mtime.
		if arch.gptoss != nil {
			if e := req(fmt.Sprintf("layer %d AttnSinks", i), len(lw.AttnSinks), arch.NumHeads); e != nil {
				return e
			}
			for xe := range lw.Experts {
				ex := &lw.Experts[xe]
				if ex.Gate.Rows() == 0 {
					continue // an expert the bundle stacks elsewhere; its biases live there too
				}
				for _, c := range []struct {
					name string
					got  int
					want int
				}{
					{"GateBias", len(ex.GateBias), arch.MoE.IntermediateDim},
					{"UpBias", len(ex.UpBias), arch.MoE.IntermediateDim},
					{"DownBias", len(ex.DownBias), arch.HiddenDim},
				} {
					if e := req(fmt.Sprintf("layer %d expert %d %s", i, xe, c.name), c.got, c.want); e != nil {
						return e
					}
				}
			}
		}
	}

	// M-11: the model-level PLE tail. gemma4's forward reads all three unconditionally when
	// the arch declares PLE, and a bundle from before v4 has none of them.
	if arch.gemma4 != nil && w.PerLayerTokenEmbed.Rows() > 0 {
		if e := eq("PerLayerModelProj", w.PerLayerModelProj.Rows(), arch.HiddenDim); e != nil {
			return e
		}
		if e := req("PerLayerProjNorm", len(w.PerLayerProjNorm), arch.HiddenDim); e != nil {
			return e
		}
	}
	return nil
}

// Weights exposes the loaded weight bundle, e.g. so a build-time tool can
// SerializeWeights(m.Weights(), …). It returns the LIVE bundle (a defensive copy would
// duplicate gigabytes and the device buffers), so the forward pass's correctness depends on it
// being treated as IMMUTABLE — the derived *Architecture, RoPE tables and resident buffers are
// built from it at load and are not rebuilt, so any mutation silently desyncs them (audit M-23).
// Read-only.
func (m *Model) Weights() *Weights { return m.w }

// Quant names the precision the model's matmul weights are resident in
// ("int8int8" | "int8" | "int4" | "native"). For a prequant model the runtime
// --quant flag is moot — this reports what was actually baked in.
func (m *Model) Quant() string {
	// A direct load records the requested quant — accurate for the KV-snapshot
	// fingerprint (so int4 / int4mix / int8 don't collide on the same file). A
	// prequant .giw leaves it empty and derives from the resident weight kinds.
	switch m.quant {
	case "int8", "int8int8", "int4", "int4mix":
		return m.quant
	}
	// A v5 .giw records the resolved label at bake time — prefer it over re-inferring. Empty for
	// a pre-v5 or streamed bundle, which fall back to the (corrected) tensor-kind inference.
	if m.w.bakedQuant != "" {
		return m.w.bakedQuant
	}
	return m.w.quantLabel()
}

// CheckGiwQuantMatch returns a startup error when an EXPLICIT weight-quant request cannot take
// effect because the model is an already-baked prequant .giw whose quant differs. A .giw is
// serialized at a fixed precision, so --quant is inert for it (Load ignores it); today it is
// silently dropped, which the T1-7 report flagged. This surfaces the mismatch instead.
//
// `requested` is the quant the user EXPLICITLY asked for — pass "" when they did not (relied on the
// default). A bare default must NOT conflict: a .giw carries its own quant, and running it with
// process defaults is the normal cross-format case (the caller, which alone knows whether the flag
// was set, is responsible for passing "" then). For a non-.giw model (GGUF/safetensors), where
// --quant IS honored at load, this is a no-op.
//
// The comparison uses the corrected Quant() label (commit 9020160) — the resident weight kinds —
// not the raw .giw header field. Message shape mirrors the safetensors int4mix decline
// (weights.go): the constraint, the requested value, the baked value, and the file.
func (m *Model) CheckGiwQuantMatch(requested string) error {
	path := m.GiwPath()
	if requested == "" || path == "" {
		return nil
	}
	if baked := m.Quant(); requested != baked {
		return fmt.Errorf("decoder: --quant %q cannot apply to the prequantized .giw bundle %s — it is baked at %q, and a .giw carries its own quant; pass --quant %s or omit --quant", requested, path, baked, baked)
	}
	return nil
}

// hasPopulatedLayers reports whether w's body matmul weights actually hold data, as opposed to a
// freshly make()'d Layers slice whose elements are still zero-valued (the true streaming GGUF
// transcode's state at header-write time, before any layer has streamed in). MUST be checked
// before calling quantLabel() to decide whether to trust its answer: quantLabel()'s own "nothing
// matched" case returns "native", a real quant mode, not an empty string — so an unpopulated w
// would silently produce a FALSE "native" label rather than a distinguishable absence (B11).
func (w *Weights) hasPopulatedLayers() bool {
	for _, m := range w.bodyMatmulWeights() {
		if m.Rows() > 0 {
			return true
		}
	}
	return false
}

// quantLabel names the precision of the resident matmul weights for display + the
// KV-snapshot fingerprint, accounting for MIXED bundles. It scans the BODY matmuls — the
// per-layer attention/FFN projections, experts, and routers, i.e. exactly what
// `-quant int4|int4mix|int8int8` selects and what the batched-prefill gate inspects — and
// returns "int4mix" only when int4 coexists with a higher-precision BODY weight. Pure
// bundles collapse to int4 / int8int8 / int8 / native.
//
// The token embedding and LM head are EXCLUDED. int4 mode pins them to int8 by DEFAULT
// (logit-critical; the EmbedInt4 knob relaxes them), so their precision is orthogonal to
// the int4-vs-int4mix distinction — a plain `-quant int4` bundle keeps an int8 head.
// Including them made such a bundle mislabel as "int4mix" (audit/T1-6) even though every
// projection is int4 and the prefill gate correctly batched it: the label contradicted the
// path. Excluding them also sidesteps the older quantMode failure this comment used to cite
// (that .giw header field derives from the FIRST weight — the int8 embed — so it reports
// plain "int8" for the same all-int4-body bundle). Both mislabels have the same root: the
// int8-pinned logit tables are not part of the quant the user chose.
func (w *Weights) quantLabel() string {
	var hasInt4, hasInt8I8, hasInt8, hasOther bool
	// bodyMatmulWeights excludes the int8-pinned logit tables — the single definition of that
	// exclusion (weights.go), so the label and the .giw resolved-quant field can never disagree.
	for _, m := range w.bodyMatmulWeights() {
		if m.Rows() == 0 {
			continue
		}
		switch m.Kind() {
		case "int4":
			hasInt4 = true
		case "int8":
			if _, _, w8a8, _ := m.Int8(); w8a8 {
				hasInt8I8 = true
			} else {
				hasInt8 = true
			}
		default:
			hasOther = true
		}
	}
	switch {
	case hasInt4 && (hasInt8I8 || hasInt8 || hasOther):
		return "int4mix"
	case hasInt4:
		return "int4"
	case hasInt8I8:
		return "int8int8"
	case hasInt8:
		return "int8"
	default:
		return "native"
	}
}

// NewModel wraps an already-built *Weights (e.g. from LoadSerializedWeights)
// into a runnable *Model: it attaches a compute backend and resolves the EOS
// ids from the config. The weights are used as-is — their quantization is fixed.
func NewModel(w *Weights, backend string) (*Model, error) {
	be, beErr := NewBackend(backend)
	if beErr != nil {
		fmt.Fprintln(os.Stderr, beErr) // webgpu/cuda requested but fell back — not fatal.
	}
	return (&Model{w: w, be: be, eosIDs: w.Cfg.EOSIDs()}).withResidency(), nil
}

// quantMode reports the precision the bundle's matmul weights are in, derived
// from the first non-empty weightMat (they are all quantized uniformly at load).
func (w *Weights) quantMode() quantMode {
	for _, m := range w.matmulWeights() {
		if m.Rows() == 0 {
			continue
		}
		switch m.Kind() {
		case "int4":
			return quantInt4
		case "int8":
			if _, _, w8a8, _ := m.Int8(); w8a8 {
				return quantInt8I8
			}
			return quantInt8
		default:
			return quantNone
		}
	}
	return quantNone
}

// --- writer ---

// giwWriter serializes the .giw / .giw-kv formats in one of two modes: buffer mode
// (sink == nil) accumulates into buf — the caller reads buf and appends its own CRC
// (SerializeWeights, kvsnapshot); stream mode (sink != nil) writes straight to sink
// while maintaining a running CRC and byte count, so a large bundle never lives in
// memory (SerializeWeightsTo). Both modes route every byte through raw, so the two
// produce identical bytes.
type giwWriter struct {
	buf  []byte        // buffer mode
	sink io.Writer     // stream mode (nil ⇒ buffer mode)
	crc  uint32        // running CRC32-IEEE over bytes written (stream mode)
	n    int64         // bytes written (stream mode)
	err  error         // first sink error (stream mode)
	arch *Architecture // set in writeBundle; gates the v4 gemma4 model-level + per-layer tail

	// row4 opts weightMat into emitting kind 4 (the on-disk split-half + 4-row
	// layout) for every eligible int4 tensor, instead of always kind 3. Never the
	// default: set only by SerializeWeightsRow4/SerializeWeightsToRow4/
	// StreamTranscodeGGUF's row4 parameter — docs/task-w4a8-neon-bandwidth.md's
	// "Format follow-on" requires this be opt-in, since a tensor's WeightMat may
	// ALREADY carry an in-RAM row4 repack (repackW4A8Row4IfEligible, wired into
	// the GGUF/safetensors streaming loaders unconditionally) that has nothing to
	// do with whether THIS serialize call should bake it onto disk.
	row4 bool
}

func (w *giwWriter) raw(b []byte) {
	if w.sink == nil {
		w.buf = append(w.buf, b...)
		return
	}
	if w.err == nil {
		if _, err := w.sink.Write(b); err != nil {
			w.err = err
		}
	}
	w.crc = crc32.Update(w.crc, crc32.IEEETable, b)
	w.n += int64(len(b))
}

func (w *giwWriter) u32(v uint32) {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], v)
	w.raw(b[:])
}
func (w *giwWriter) str(s string) { w.u32(uint32(len(s))); w.raw([]byte(s)) }

func (w *giwWriter) bytesField(b []byte) { w.u32(uint32(len(b))); w.raw(b) }

// f32 batches into one raw write (a per-tensor temp), not 4 bytes at a time — the
// stream path would otherwise do millions of tiny writes.
func (w *giwWriter) f32(s []float32) {
	w.u32(uint32(len(s)))
	if len(s) == 0 {
		return
	}
	b := make([]byte, 4*len(s))
	for i, v := range s {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(v))
	}
	w.raw(b)
}

func (w *giwWriter) i8(s []int8) {
	w.u32(uint32(len(s)))
	if len(s) > 0 {
		w.raw(unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), len(s)))
	}
}

func (w *giwWriter) weightMat(m *linalg.WeightMat) {
	if m.Rows() == 0 {
		w.raw([]byte{0}) // empty
		return
	}
	q4, q4s, group, isQ4 := m.Int4()
	q8, scales, w8a8, isQ8 := m.Int8()
	f32, _ := m.F32()
	var kind byte
	var q4Row4 []byte
	var q4Row4Scales []float32
	switch {
	case isQ4:
		kind = 3
		// row4 emission is purely a function of the opt-in flag + this tensor's
		// shape — NEVER of whatever repack state already happens to sit in RAM
		// (repackW4A8Row4IfEligible populates q4Row4 unconditionally for every
		// GGUF/safetensors-streamed int4 tensor on an arm64 box, regardless of
		// whether THIS serialize call is the opt-in prequant path). Recomputing
		// from canonical q4/q4s here, rather than reading m.Int4Row4(), keeps
		// kind 3 the default for every existing caller unless w.row4 is set.
		if w.row4 {
			if r4, r4s, ok := repackRow4ForEmit(q4, q4s, m.Rows(), m.Cols(), group); ok {
				kind = 4
				q4Row4, q4Row4Scales = r4, r4s
			}
		}
	case isQ8:
		kind = 2
	default:
		kind = 1 // f32
	}
	w.raw([]byte{kind})
	w.u32(uint32(m.Rows()))
	w.u32(uint32(m.Cols()))
	w.u32(uint32(group)) // 0 unless int4
	if w8a8 {
		w.raw([]byte{1})
	} else {
		w.raw([]byte{0})
	}
	switch kind {
	case 1:
		w.f32(f32)
	case 2:
		w.f32(scales)
		w.i8(q8)
	case 3:
		w.f32(q4s)
		w.bytesField(q4)
	case 4:
		w.f32(q4s)
		w.bytesField(q4)
		w.f32(q4Row4Scales)
		w.bytesField(q4Row4)
	}
}

func (w *giwWriter) layer(l *LayerWeights) {
	w.weightMat(&l.QProj)
	w.weightMat(&l.KProj)
	w.weightMat(&l.VProj)
	w.weightMat(&l.OProj)
	w.f32(l.QBias)
	w.f32(l.KBias)
	w.f32(l.VBias)
	w.f32(l.OBias)
	w.f32(l.QNorm)
	w.f32(l.KNorm)
	w.f32(l.PreAttnNorm)
	w.f32(l.PreAttnNormBias)
	w.f32(l.PostAttnNorm)
	w.weightMat(&l.GateProj)
	w.weightMat(&l.UpProj)
	w.weightMat(&l.DownProj)
	w.f32(l.UpBias)
	w.f32(l.DownBias)
	w.f32(l.PreMLPNorm)
	w.f32(l.PreMLPNormBias)
	w.f32(l.PostMLPNorm)
	w.weightMat(&l.Router)
	w.f32(l.RouterBias) // v3: DeepSeek/GLM e_score_correction_bias
	w.u32(uint32(len(l.Experts)))
	for e := range l.Experts {
		w.weightMat(&l.Experts[e].Gate)
		w.weightMat(&l.Experts[e].Up)
		w.weightMat(&l.Experts[e].Down)
	}
	w.weightMat(&l.SharedExpert.Gate)
	w.weightMat(&l.SharedExpert.Up)
	w.weightMat(&l.SharedExpert.Down)
	w.weightMat(&l.SharedGate)
	w.hybridLayer(l)
	if w.arch != nil && w.arch.gemma4 != nil { // v4 gemma4 per-layer tail
		w.gemma4Layer(l)
	}
	w.v6Layer(l) // v6 completeness tail — see below
	w.v8Layer(l) // v8 LFM2 short-conv tail — see below
}

// v6Layer writes the state that made five families unrepresentable, in one unconditional tail.
//
// WHY UNCONDITIONAL RATHER THAN ARCH-GATED like the gemma4 tail: every field here is empty on the
// families that do not use it, so the cost is a handful of zero lengths per layer, and an
// arch-gated tail is precisely how gpt-oss's sinks went missing — the gate is another place to
// remember. A tail that always writes what the struct holds cannot be forgotten for the next
// family, and TestSerializeCensus_noSilentFieldDrop checks that claim against the struct itself.
func (w *giwWriter) v6Layer(l *LayerWeights) {
	w.weightMat(&l.GProj) // Laguna's attention output gate (per-head or per-element)
	w.f32(l.AttnSinks)    // gpt-oss per-head attention sinks
	// gpt-oss per-expert biases. The expert loop above writes only the three matrices; these are
	// nil for every other family, so this is three zero lengths per expert elsewhere.
	for e := range l.Experts {
		w.f32(l.Experts[e].GateBias)
		w.f32(l.Experts[e].UpBias)
		w.f32(l.Experts[e].DownBias)
	}
	w.f32(l.SharedExpert.GateBias)
	w.f32(l.SharedExpert.UpBias)
	w.f32(l.SharedExpert.DownBias)
	// MLA (DeepSeek / Kimi): presence byte then the eight tensors.
	if l.mla == nil {
		w.raw([]byte{0})
	} else {
		w.raw([]byte{1})
		m := l.mla
		w.f32(m.qAProj)
		w.f32(m.qALayernorm)
		w.f32(m.qBProj)
		w.f32(m.qProj)
		w.f32(m.kvAProj)
		w.f32(m.kvALayernorm)
		w.f32(m.kvBProj)
		w.f32(m.oProj)
	}
	// Mamba-2 (Granite / Nemotron): presence byte then the eight tensors. The recurrent STATE is
	// not written and must not be — it is per-sequence, rebuilt at load; only the weights are here.
	if l.mamba == nil {
		w.raw([]byte{0})
	} else {
		w.raw([]byte{1})
		m := l.mamba
		w.f32(m.inProj)
		w.f32(m.convW)
		w.f32(m.convB)
		w.f32(m.aLog)
		w.f32(m.d)
		w.f32(m.dtBias)
		w.f32(m.normW)
		w.f32(m.outProj)
	}
}

// v8Layer writes the LFM2 gated short-convolution mixer: presence byte then the three tensors.
//
// UNCONDITIONAL, LIKE THE v6 TAIL AND FOR THE SAME REASON. An arch-gated tail is how gpt-oss's
// attention sinks went missing, and the cost here is one zero byte per layer on every other family.
//
// This field existed for a whole family and serialize.go did not mention it once — `grep shortConv
// decoder/serialize.go` returned zero matches. cmd/prequant loaded an LFM2 checkpoint, wrote every
// field EXCEPT this one, appended a valid CRC, and selfCheck passed because selfCheck only Loads.
// Serving the bundle, the first token reached conv layer 0 with lw.shortConv == nil and
// nil-dereferenced in the decode goroutine (audit-2026-09-02 C-03, a regression of R3).
//
// As with mamba, only the WEIGHTS are here — the rolling conv window is per-sequence state that
// lives in the KVCache and is rebuilt at load.
func (w *giwWriter) v8Layer(l *LayerWeights) {
	if l.shortConv == nil {
		w.raw([]byte{0})
		return
	}
	w.raw([]byte{1})
	c := l.shortConv
	w.f32(c.inProj)
	w.f32(c.convW)
	w.f32(c.outProj)
}

// gemma4Layer writes the v4 Gemma 4 per-layer tail: the PLE branch, the per-layer
// output scalar, the KV-share flags, and (when enable_moe_block) the gemma4moe
// sub-block's OWN tensors — the ones NOT already covered by the standard block. The
// dense-branch MLP + its pre/post norms are l.GateProj/UpProj/DownProj +
// l.PreMLPNorm/PostMLPNorm (already written), which the reader re-aliases into
// gemma4moe; only the three parallel-branch norms, the router (l.Router is empty for
// gemma4), the per-expert scale, and the fused experts are new here.
func (w *giwWriter) gemma4Layer(l *LayerWeights) {
	w.weightMat(&l.PLEGate)
	w.weightMat(&l.PLEProj)
	w.f32(l.PostPLENorm)
	w.u32(math.Float32bits(l.LayerScalar))
	w.raw([]byte{b2u8(l.KVShared), b2u8(l.VFromK)})
	if l.gemma4moe == nil {
		w.raw([]byte{0})
		return
	}
	w.raw([]byte{1})
	mo := l.gemma4moe
	w.f32(mo.postFFNNorm1)
	w.f32(mo.preFFNNorm2)
	w.f32(mo.postFFNNorm2)
	w.weightMat(&mo.routerProj)
	w.f32(mo.routerScale)
	w.f32(mo.perExpertScale)
	w.u32(uint32(len(mo.expertsGateUp)))
	for e := range mo.expertsGateUp {
		w.weightMat(&mo.expertsGateUp[e])
		w.weightMat(&mo.expertsDown[e])
	}
}

func b2u8(b bool) byte {
	if b {
		return 1
	}
	return 0
}

// hybridLayer writes the qwen3_5_moe per-layer extras (v2): a kind byte then the
// DeltaNet (linear-layer) or gated-softmax (qattn) f32 tensor set. Every other
// family writes kind 0 and nothing more, so the field stays one byte per layer.
func (w *giwWriter) hybridLayer(l *LayerWeights) {
	switch {
	case l.delta != nil:
		w.raw([]byte{1})
		d := l.delta
		w.weightMat(&d.inProjQKV)
		w.weightMat(&d.inProjZ)
		w.f32(d.inProjB)
		w.f32(d.inProjA)
		w.f32(d.convW)
		w.f32(d.dtBias)
		w.f32(d.negExpA) // the precomputed −exp(A_log); stored as-is, no recompute on load
		w.f32(d.normW)
		w.weightMat(&d.outProj)
	case l.qattn != nil:
		w.raw([]byte{2})
		q := l.qattn
		w.weightMat(&q.qProj)
		w.weightMat(&q.kProj)
		w.weightMat(&q.vProj)
		w.weightMat(&q.oProj)
		w.f32(q.qNorm)
		w.f32(q.kNorm)
	default:
		// MLA / Mamba-2 used to be refused here. They are written by v6Layer below (the
		// completeness tail), so this stays the "no hybrid extras" marker it was named for.
		w.raw([]byte{0})
	}
}

// --- reader (cursor over data; big arrays aliased, floats copied) ---

type giwReader struct {
	data    []byte
	off     int
	err     error
	version uint32        // parsed file version; gemma4 v4 fields are read only when ≥4
	arch    *Architecture // resolved from the serialized config; gates the gemma4 tail
}

func (r *giwReader) fail(msg string) {
	if r.err == nil {
		r.err = &SerializeError{msg}
	}
}

func (r *giwReader) need(n int) bool {
	if r.err != nil {
		return false
	}
	if n < 0 || r.off+n > len(r.data) {
		r.fail("unexpected end of data")
		return false
	}
	return true
}

func (r *giwReader) u32() uint32 {
	if !r.need(4) {
		return 0
	}
	v := binary.LittleEndian.Uint32(r.data[r.off:])
	r.off += 4
	return v
}

func (r *giwReader) rawN(n int) []byte {
	if !r.need(n) {
		return nil
	}
	b := r.data[r.off : r.off+n]
	r.off += n
	return b
}

func (r *giwReader) str() string        { return string(r.rawN(int(r.u32()))) }
func (r *giwReader) bytesField() []byte { return r.rawN(int(r.u32())) }

func (r *giwReader) u8() byte {
	if !r.need(1) {
		return 0
	}
	b := r.data[r.off]
	r.off++
	return b
}

// f32 copies (the input isn't guaranteed aligned). len 0 ⇒ nil, preserving the
// "absent ⇒ nil" convention the forward pass checks for biases/norms.
func (r *giwReader) f32() []float32 {
	n := int(r.u32())
	if n == 0 || !r.need(n*4) {
		return nil
	}
	out := make([]float32, n)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(r.data[r.off:]))
		r.off += 4
	}
	return out
}

// i8 ALIASES the int8 weight bytes over data (zero-copy). Read-only at inference.
func (r *giwReader) i8() []int8 {
	n := int(r.u32())
	if n == 0 || !r.need(n) {
		return nil
	}
	b := r.data[r.off : r.off+n]
	r.off += n
	return unsafe.Slice((*int8)(unsafe.Pointer(&b[0])), n)
}

// rawAlias ALIASES a []byte (int4 packed nibbles) — a plain subslice of data.
func (r *giwReader) rawAlias() []byte {
	n := int(r.u32())
	if n == 0 || !r.need(n) {
		return nil
	}
	b := r.data[r.off : r.off+n]
	r.off += n
	return b
}

func (r *giwReader) weightMat() linalg.WeightMat {
	if !r.need(1) {
		return linalg.WeightMat{}
	}
	kind := r.data[r.off]
	r.off++
	if kind == 0 {
		return linalg.WeightMat{}
	}
	rows, cols, group := int(r.u32()), int(r.u32()), int(r.u32())
	if !r.need(1) {
		return linalg.WeightMat{}
	}
	w8a8 := r.data[r.off] == 1
	r.off++
	// M17: rows/cols/group are blob-controlled. linalg.Wrap{Int8,Int4} PANIC on a length
	// mismatch or group<=0, and a wrong-length WrapF32 defers the panic to first use — a one-byte
	// flip with a recomputed CRC would crash the loader. Validate dims + array lengths here (the
	// arrays were already bounded by r.need); a mismatch is a *SerializeError via r.fail, not a
	// panic. The maxWeightDim cap keeps rows*cols from overflowing int before the equality check.
	const maxWeightDim = 1 << 26
	if rows <= 0 || cols <= 0 || rows > maxWeightDim || cols > maxWeightDim {
		r.fail(fmt.Sprintf("weightMat implausible dims %d×%d", rows, cols))
		return linalg.WeightMat{}
	}
	switch kind {
	case 1:
		f := r.f32()
		if len(f) != rows*cols {
			r.fail(fmt.Sprintf("f32 weightMat %d×%d has %d values", rows, cols, len(f)))
			return linalg.WeightMat{}
		}
		return linalg.WrapF32(f, rows, cols)
	case 2:
		scales := r.f32() // read order matches the writer (scales, then codes)
		q8 := r.i8()
		if len(q8) != rows*cols || len(scales) != rows {
			r.fail(fmt.Sprintf("int8 weightMat %d×%d: q8=%d (want %d) scales=%d (want %d)", rows, cols, len(q8), rows*cols, len(scales), rows))
			return linalg.WeightMat{}
		}
		return linalg.WrapInt8(q8, scales, rows, cols, w8a8)
	case 3:
		q4s := r.f32()
		q4 := r.rawAlias() // zero-copy alias into the mmap'd blob (WrapInt4 keeps it)
		if group <= 0 {
			r.fail(fmt.Sprintf("int4 weightMat group %d ≤ 0", group))
			return linalg.WeightMat{}
		}
		wantQ4, wantScales := rows*((cols+1)/2), rows*((cols+group-1)/group)
		if len(q4) != wantQ4 || len(q4s) != wantScales {
			r.fail(fmt.Sprintf("int4 weightMat %d×%d group=%d: q4=%d (want %d) q4s=%d (want %d)", rows, cols, group, len(q4), wantQ4, len(q4s), wantScales))
			return linalg.WeightMat{}
		}
		return linalg.WrapInt4(q4, q4s, rows, cols, group)
	case 4:
		q4s := r.f32()
		q4 := r.rawAlias() // canonical bytes stay authoritative — same as kind 3
		if group <= 0 {
			r.fail(fmt.Sprintf("int4-row4 weightMat group %d ≤ 0", group))
			return linalg.WeightMat{}
		}
		wantQ4, wantScales := rows*((cols+1)/2), rows*((cols+group-1)/group)
		if len(q4) != wantQ4 || len(q4s) != wantScales {
			r.fail(fmt.Sprintf("int4-row4 weightMat %d×%d group=%d: q4=%d (want %d) q4s=%d (want %d)", rows, cols, group, len(q4), wantQ4, len(q4s), wantScales))
			return linalg.WeightMat{}
		}
		q4Row4Scales := r.f32()
		q4Row4 := r.rawAlias() // zero-copy — the whole point of kind 4 (WrapInt4Row4 gates on row4Usable() before aliasing it in)
		// RepackW4A8Row4/RepackW4A8Row4Scales preserve length exactly (a repack,
		// not a requant), so the row4 arrays share kind 3's own want* values.
		if len(q4Row4) != wantQ4 || len(q4Row4Scales) != wantScales {
			r.fail(fmt.Sprintf("int4-row4 weightMat %d×%d group=%d: q4Row4=%d (want %d) q4Row4Scales=%d (want %d)", rows, cols, group, len(q4Row4), wantQ4, len(q4Row4Scales), wantScales))
			return linalg.WeightMat{}
		}
		return linalg.WrapInt4Row4(q4, q4s, rows, cols, group, q4Row4, q4Row4Scales)
	default:
		r.fail(fmt.Sprintf("unknown weightMat kind %d", kind))
		return linalg.WeightMat{}
	}
}

func (r *giwReader) layer(l *LayerWeights) {
	l.QProj = r.weightMat()
	l.KProj = r.weightMat()
	l.VProj = r.weightMat()
	l.OProj = r.weightMat()
	l.QBias = r.f32()
	l.KBias = r.f32()
	l.VBias = r.f32()
	l.OBias = r.f32()
	l.QNorm = r.f32()
	l.KNorm = r.f32()
	l.PreAttnNorm = r.f32()
	l.PreAttnNormBias = r.f32()
	l.PostAttnNorm = r.f32()
	l.GateProj = r.weightMat()
	l.UpProj = r.weightMat()
	l.DownProj = r.weightMat()
	l.UpBias = r.f32()
	l.DownBias = r.f32()
	l.PreMLPNorm = r.f32()
	l.PreMLPNormBias = r.f32()
	l.PostMLPNorm = r.f32()
	l.Router = r.weightMat()
	l.RouterBias = r.f32() // v3: DeepSeek/GLM e_score_correction_bias
	ne := int(r.u32())
	if ne < 0 || ne > maxSerializedExperts {
		r.fail("implausible expert count")
		return
	}
	if ne > 0 {
		l.Experts = make([]expertWeights, ne)
		for e := range l.Experts {
			l.Experts[e].Gate = r.weightMat()
			l.Experts[e].Up = r.weightMat()
			l.Experts[e].Down = r.weightMat()
		}
	}
	l.SharedExpert.Gate = r.weightMat()
	l.SharedExpert.Up = r.weightMat()
	l.SharedExpert.Down = r.weightMat()
	l.SharedGate = r.weightMat()
	r.hybridLayer(l)
	if r.version >= giwV4Gemma4 && r.arch != nil && r.arch.gemma4 != nil { // v4 gemma4 tail
		r.gemma4Layer(l)
	}
	if r.version >= giwV6Tail { // v6 completeness tail
		r.v6Layer(l)
	}
	if r.version >= giwV8ShortConv { // v8 LFM2 short-conv tail
		r.v8Layer(l)
	}
}

// v6Layer mirrors giwWriter.v6Layer: the state that made five families unrepresentable before v6.
// Bundles written at v3-v5 simply do not carry it, which is why it is version-gated rather than
// probed — an older bundle's layer block ends where it always did.
func (r *giwReader) v6Layer(l *LayerWeights) {
	l.GProj = r.weightMat()
	l.AttnSinks = r.f32()
	for e := range l.Experts {
		l.Experts[e].GateBias = r.f32()
		l.Experts[e].UpBias = r.f32()
		l.Experts[e].DownBias = r.f32()
	}
	l.SharedExpert.GateBias = r.f32()
	l.SharedExpert.UpBias = r.f32()
	l.SharedExpert.DownBias = r.f32()
	if r.u8() != 0 {
		m := &mlaWeights{}
		m.qAProj = r.f32()
		m.qALayernorm = r.f32()
		m.qBProj = r.f32()
		m.qProj = r.f32()
		m.kvAProj = r.f32()
		m.kvALayernorm = r.f32()
		m.kvBProj = r.f32()
		m.oProj = r.f32()
		l.mla = m
	}
	if r.u8() != 0 {
		m := &mamba2Weights{}
		m.inProj = r.f32()
		m.convW = r.f32()
		m.convB = r.f32()
		m.aLog = r.f32()
		m.d = r.f32()
		m.dtBias = r.f32()
		m.normW = r.f32()
		m.outProj = r.f32()
		l.mamba = m
	}
}

// v8Layer mirrors giwWriter.v8Layer. Version-gated, so a v3-v7 bundle's layer block ends where it
// always did — and an LFM2 bundle written at v7 or earlier is one whose conv weights were never in
// the file at all, which validateShapes rejects rather than loading into a nil-deref.
func (r *giwReader) v8Layer(l *LayerWeights) {
	if r.u8() == 0 {
		return
	}
	c := &shortConvWeights{}
	c.inProj = r.f32()
	c.convW = r.f32()
	c.outProj = r.f32()
	l.shortConv = c
}

// gemma4Layer reads the v4 Gemma 4 per-layer tail and, when the MoE sub-block is
// present, reconstructs gemma4moe — re-aliasing the dense-branch MLP + pre/post norms
// already read into the standard block, and taking the fixed dims from the resolved
// arch (denseInter = IntermediateDim, moe/experts/topK from arch.MoE).
func (r *giwReader) gemma4Layer(l *LayerWeights) {
	l.PLEGate = r.weightMat()
	l.PLEProj = r.weightMat()
	l.PostPLENorm = r.f32()
	l.LayerScalar = math.Float32frombits(r.u32())
	l.KVShared = r.u8() != 0
	l.VFromK = r.u8() != 0
	if r.u8() == 0 { // no gemma4moe sub-block (dense E-model layer)
		return
	}
	mo := &gemma4MoEWeights{
		preFFNNorm:  l.PreMLPNorm,  // dense pre-norm (already read)
		postFFNNorm: l.PostMLPNorm, // joint post-norm (already read)
		mlpGate:     l.GateProj,
		mlpUp:       l.UpProj,
		mlpDown:     l.DownProj,
		layerScalar: l.LayerScalar,
		denseInter:  r.arch.IntermediateDim,
	}
	if m := r.arch.MoE; m != nil {
		mo.moeInter, mo.nE, mo.topK = m.IntermediateDim, m.NumExperts, m.TopK
	}
	mo.postFFNNorm1 = r.f32()
	mo.preFFNNorm2 = r.f32()
	mo.postFFNNorm2 = r.f32()
	mo.routerProj = r.weightMat()
	mo.routerScale = r.f32()
	mo.perExpertScale = r.f32()
	ne := int(r.u32())
	if ne < 0 || ne > maxSerializedExperts {
		r.fail("implausible gemma4moe expert count")
		return
	}
	mo.expertsGateUp = make([]linalg.WeightMat, ne)
	mo.expertsDown = make([]linalg.WeightMat, ne)
	for e := range ne {
		mo.expertsGateUp[e] = r.weightMat()
		mo.expertsDown[e] = r.weightMat()
	}
	l.gemma4moe = mo
}

// hybridLayer reconstructs the qwen3_5_moe per-layer extras written by
// giwWriter.hybridLayer. negExpA was stored precomputed, so it loads straight in.
func (r *giwReader) hybridLayer(l *LayerWeights) {
	switch r.u8() {
	case 0:
		// no hybrid extras (every non-qwen3_5_moe family)
	case 1:
		// FIELD ORDER IS THE WIRE ORDER: the three projections are WeightMats as of the
		// quantization change, and a struct literal evaluates its fields in source order, so these
		// must be read in exactly the order hybridLayer writes them.
		d := &deltaNetWeights{}
		d.inProjQKV = r.weightMat()
		d.inProjZ = r.weightMat()
		d.inProjB = r.f32()
		d.inProjA = r.f32()
		d.convW = r.f32()
		d.dtBias = r.f32()
		d.negExpA = r.f32()
		d.normW = r.f32()
		d.outProj = r.weightMat()
		l.delta = d
	case 2:
		a := &qwenAttnWeights{}
		a.qProj = r.weightMat()
		a.kProj = r.weightMat()
		a.vProj = r.weightMat()
		a.oProj = r.weightMat()
		a.qNorm = r.f32()
		a.kNorm = r.f32()
		l.qattn = a
	default:
		r.fail("unknown per-layer hybrid kind")
	}
}
