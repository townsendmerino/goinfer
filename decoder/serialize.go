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
//	quant   uint32   (quantMode the weights were serialized at)
//	id      str      (model identity — source name/hash, for tooling)
//	config  str      (Config as JSON; arch is re-derived from it on load)
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
// weightMat = uint8 kind (0 empty|1 f32|2 q8|3 q4); if non-empty:
//             int32 rows, cols, group; uint8 w8a8; then the kind's arrays.
//
// v2 added the per-layer hybrid tail so the qwen3_5_moe (DeltaNet + gated-softmax)
// family round-trips through .giw; v1 blobs (no tail) are rejected by the version
// guard and rebuilt from the source GGUF.

const (
	giwMagic   = "GINFW"
	giwVersion = 3 // v3: per-layer RouterBias (DeepSeek/GLM e_score_correction_bias); v2: qwen3_5_moe hybrid tail
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
// qwen3_5_moe's DeltaNet/gated-softmax extras; it does NOT write MLA latent projections
// (DeepSeek/Kimi), Mamba-2 SSM weights (Granite/Nemotron), or the Gemma-4 per-layer-embedding
// stack — so serializing those yields a CRC-valid bundle that nil-derefs at the first forward.
// Refuse those families up front rather than emit silent garbage (C2).
func canSerialize(a *Architecture) *SerializeError {
	switch {
	case a == nil:
		return nil
	case a.mla != nil:
		return &SerializeError{"MLA latent attention (DeepSeek/Kimi) is not representable in .giw"}
	case a.granite != nil || a.nemotron != nil:
		return &SerializeError{"Mamba-2 SSM state (Granite/Nemotron) is not representable in .giw"}
	case a.gemma4 != nil:
		return &SerializeError{"Gemma-4 per-layer-embedding stack is not representable in .giw"}
	case a.llama4 != nil:
		return &SerializeError{"Llama-4 is not yet supported by .giw serialization"}
	}
	return nil
}

// fail records the first serialization refusal (the writer's belt-and-suspenders to canSerialize:
// a per-layer field the arch-level check missed still stops the bundle rather than dropping state).
func (w *giwWriter) fail(reason string) {
	if w.err == nil {
		w.err = &SerializeError{reason}
	}
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

// writeBundle writes the bundle body (everything but the trailing CRC) via the
// writer's current sink (buffer or stream). Shared by SerializeWeights and
// SerializeWeightsTo so the field order can't drift between them or from the reader.
func (wr *giwWriter) writeBundle(w *Weights, id string) error {
	if err := canSerialize(w.arch); err != nil {
		return err
	}
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

	wr.weightMat(&w.Embed)
	wr.weightMat(&w.LMHead)
	wr.weightMat(&w.PosEmbed)
	wr.f32(w.FinalNorm)
	wr.f32(w.FinalNormBias)

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
	if v := r.u32(); v != giwVersion {
		return nil, &SerializeError{fmt.Sprintf("format version %d, this build reads %d", v, giwVersion)}
	}
	quant := quantMode(r.u32())
	_ = r.str() // id — stored for tooling, not validated here
	cfgJSON := r.bytesField()
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

	w := &Weights{Cfg: cfg, arch: arch, backing: data}
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
	return w, nil
}

// Weights exposes the loaded weight bundle, e.g. so a build-time tool can
// SerializeWeights(m.Weights(), …). The forward pass treats it as immutable.
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
	return m.w.quantLabel()
}

// quantLabel names the precision of the resident matmul weights for display + the
// KV-snapshot fingerprint, accounting for MIXED bundles. quantMode (the .giw format tag)
// reports only the FIRST weight's kind, which mislabels a mixed bundle — e.g. Mellum2's
// int4 experts with an int8-kept embedding/LM head report as plain "int8" — so this scans
// every matmul weight and returns "int4mix" when int4 coexists with higher-precision
// tensors. Pure bundles collapse to int4 / int8int8 / int8 / native as before.
func (w *Weights) quantLabel() string {
	var hasInt4, hasInt8I8, hasInt8, hasOther bool
	for _, m := range w.matmulWeights() {
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
	buf  []byte    // buffer mode
	sink io.Writer // stream mode (nil ⇒ buffer mode)
	crc  uint32    // running CRC32-IEEE over bytes written (stream mode)
	n    int64     // bytes written (stream mode)
	err  error     // first sink error (stream mode)
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
	switch {
	case isQ4:
		kind = 3
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
	}
}

func (w *giwWriter) layer(l *LayerWeights) {
	// Gemma-4 per-layer-embedding / KV-sharing state has no slot in this format — refuse rather
	// than drop it (the stream path reaches here without the arch-level canSerialize gate). C2.
	if l.PLEGate.Rows() > 0 || l.KVShared || l.VFromK {
		w.fail("Gemma-4 per-layer-embedding / KV-sharing layer is not representable in .giw")
		return
	}
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
}

// hybridLayer writes the qwen3_5_moe per-layer extras (v2): a kind byte then the
// DeltaNet (linear-layer) or gated-softmax (qattn) f32 tensor set. Every other
// family writes kind 0 and nothing more, so the field stays one byte per layer.
func (w *giwWriter) hybridLayer(l *LayerWeights) {
	switch {
	case l.delta != nil:
		w.raw([]byte{1})
		d := l.delta
		w.f32(d.inProjQKV)
		w.f32(d.inProjZ)
		w.f32(d.inProjB)
		w.f32(d.inProjA)
		w.f32(d.convW)
		w.f32(d.dtBias)
		w.f32(d.negExpA) // the precomputed −exp(A_log); stored as-is, no recompute on load
		w.f32(d.normW)
		w.f32(d.outProj)
	case l.qattn != nil:
		w.raw([]byte{2})
		q := l.qattn
		w.f32(q.qProj)
		w.f32(q.kProj)
		w.f32(q.vProj)
		w.f32(q.oProj)
		w.f32(q.qNorm)
		w.f32(q.kNorm)
	default:
		if l.mla != nil || l.mamba != nil {
			w.fail("layer has MLA / Mamba-2 state not representable in .giw")
			return
		}
		w.raw([]byte{0})
	}
}

// --- reader (cursor over data; big arrays aliased, floats copied) ---

type giwReader struct {
	data []byte
	off  int
	err  error
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
}

// hybridLayer reconstructs the qwen3_5_moe per-layer extras written by
// giwWriter.hybridLayer. negExpA was stored precomputed, so it loads straight in.
func (r *giwReader) hybridLayer(l *LayerWeights) {
	switch r.u8() {
	case 0:
		// no hybrid extras (every non-qwen3_5_moe family)
	case 1:
		l.delta = &deltaNetWeights{
			inProjQKV: r.f32(),
			inProjZ:   r.f32(),
			inProjB:   r.f32(),
			inProjA:   r.f32(),
			convW:     r.f32(),
			dtBias:    r.f32(),
			negExpA:   r.f32(),
			normW:     r.f32(),
			outProj:   r.f32(),
		}
	case 2:
		l.qattn = &qwenAttnWeights{
			qProj: r.f32(),
			kProj: r.f32(),
			vProj: r.f32(),
			oProj: r.f32(),
			qNorm: r.f32(),
			kNorm: r.f32(),
		}
	default:
		r.fail("unknown per-layer hybrid kind")
	}
}
