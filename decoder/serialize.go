package decoder

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"github.com/townsendmerino/aikit/linalg"
	"hash/crc32"
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
//	  per layer: the LayerWeights fields, in declaration order
//	crc     uint32   (CRC32-IEEE over every preceding byte)
//
// str  = uint32 len + len bytes
// f32  = uint32 len + len*4 LE-float32 bytes   (len 0 ⇒ nil on load)
// i8   = uint32 len + len bytes                (aliased on load)
// raw  = uint32 len + len bytes                (aliased on load)
// weightMat = uint8 kind (0 empty|1 f32|2 q8|3 q4); if non-empty:
//             int32 rows, cols, group; uint8 w8a8; then the kind's arrays.

const (
	giwMagic   = "GINFW"
	giwVersion = 1
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

// SerializeWeights writes the resident weight bundle (already quantized to its
// current precision) to a flat little-endian blob suitable for embedding. id is
// an opaque model-identity string (e.g. the source filename) stored for tooling.
func SerializeWeights(w *Weights, id string) ([]byte, error) {
	cfgJSON, err := json.Marshal(w.Cfg)
	if err != nil {
		return nil, fmt.Errorf("decoder: marshal config: %w", err)
	}
	wr := &giwWriter{}
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
	for i := range w.Layers {
		wr.layer(&w.Layers[i])
	}

	crc := crc32.ChecksumIEEE(wr.buf)
	wr.u32(crc)
	return wr.buf, nil
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
	switch m.w.quantMode() {
	case quantInt8I8:
		return "int8int8"
	case quantInt8:
		return "int8"
	case quantInt4:
		return "int4"
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
		fmt.Println(beErr) // webgpu requested but fell back — not fatal.
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

type giwWriter struct{ buf []byte }

func (w *giwWriter) u32(v uint32) { w.buf = binary.LittleEndian.AppendUint32(w.buf, v) }
func (w *giwWriter) raw(b []byte) { w.buf = append(w.buf, b...) }
func (w *giwWriter) str(s string) { w.u32(uint32(len(s))); w.raw([]byte(s)) }

func (w *giwWriter) bytesField(b []byte) { w.u32(uint32(len(b))); w.raw(b) }

func (w *giwWriter) f32(s []float32) {
	w.u32(uint32(len(s)))
	for _, v := range s {
		w.u32(math.Float32bits(v))
	}
}

func (w *giwWriter) i8(s []int8) {
	w.u32(uint32(len(s)))
	if len(s) > 0 {
		w.raw(unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), len(s)))
	}
}

func (w *giwWriter) weightMat(m *linalg.WeightMat) {
	if m.Rows() == 0 {
		w.buf = append(w.buf, 0) // empty
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
	w.buf = append(w.buf, kind)
	w.u32(uint32(m.Rows()))
	w.u32(uint32(m.Cols()))
	w.u32(uint32(group)) // 0 unless int4
	if w8a8 {
		w.buf = append(w.buf, 1)
	} else {
		w.buf = append(w.buf, 0)
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
	switch kind {
	case 1:
		return linalg.WrapF32(r.f32(), rows, cols)
	case 2:
		scales := r.f32() // read order matches the writer (scales, then codes)
		q8 := r.i8()
		return linalg.WrapInt8(q8, scales, rows, cols, w8a8)
	case 3:
		q4s := r.f32()
		q4 := r.rawAlias() // zero-copy alias into the mmap'd blob (WrapInt4 keeps it)
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
}
