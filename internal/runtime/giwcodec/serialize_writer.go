package giwcodec

import (
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"io"
	"math"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
	darch "github.com/townsendmerino/goinfer/decoder/arch"
)

// --- writer ---

// Writer serializes the .giw / .giw-kv formats in one of two modes: buffer mode
// (sink == nil) accumulates into buf — the caller reads buf and appends its own CRC
// (SerializeWeights, kvsnapshot); stream mode (sink != nil) writes straight to sink
// while maintaining a running CRC and byte count, so a large bundle never lives in
// memory (SerializeWeightsTo). Both modes route every byte through raw, so the two
// produce identical bytes.
type Writer struct {
	Buf         []byte
	Sink        io.Writer
	CRC         uint32
	N           int64
	Err         error
	Arch        *darch.Architecture
	Target      Target
	F32Kind1    bool
	IslandsF32  bool
	HeadTableFn func(w *Writer, dst *linalg.WeightMat) bool
}

func (w *Writer) Raw(b []byte) {
	if w.Sink == nil {
		w.Buf = append(w.Buf, b...)
		return
	}
	if w.Err == nil {
		if _, err := w.Sink.Write(b); err != nil {
			w.Err = err
		}
	}
	w.CRC = crc32.Update(w.CRC, crc32.IEEETable, b)
	w.N += int64(len(b))
}

// Pos is how many bytes have been written since the blob start, in either mode — the quantity the
// alignment padding is defined over.
func (w *Writer) Pos() int64 {
	if w.Sink == nil {
		return int64(len(w.Buf))
	}
	return w.N
}

// ArrayPad is the number of zero bytes to insert at blob offset off so that an array written as
// `u32 count | payload` has its PAYLOAD (off+pad+4) on a 16-byte boundary. A pure function of the
// offset, so writer and reader compute the same answer without recording it.
func ArrayPad(off int64) int { return int((-(off + 4)) & 15) }

// AlignHead ends the variable-length header on a 16-byte boundary. The header carries a length-prefixed quant label
// that the resident writer emits and the streaming writer deliberately does not, so the two paths' offsets differ by the
// label's length. Every later array's padding is a function of its absolute offset, so without this the two would lay out
// the same weights with different padding and stop being byte-identical
// (TestStreamTranscodeMatchesResident). With it, everything after the header is at the same offsets in both.
func (w *Writer) AlignHead() { w.Align16() }

// Align16 pads to a 16-byte boundary: before a block that carries no count (a fused group's nibbles and scales).
func (w *Writer) Align16() {
	if pad := int((-w.Pos()) & 15); pad > 0 {
		var z [16]byte
		w.Raw(z[:pad])
	}
}

// ErrCopiedF32Bundle refuses a bundle whose f32 weights are in kind 1, which a load would copy onto the heap, where the fit guard treats
// a bundle's weights as file-backed. A sidecar that hits this is rebuilt (cacheFresh).
var ErrCopiedF32Bundle = &SerializeError{"this bundle predates the mapped f32 encoding, and would load its f32 weights onto the heap, where the fit guard does not price them; rebuild it from its source: go run ./cmd/prequant -quant f32 -o <out.giw> <source>"}

// AlignArray pads before the next array's count.
func (w *Writer) AlignArray() {
	if pad := ArrayPad(w.Pos()); pad > 0 {
		var z [16]byte
		w.Raw(z[:pad])
	}
}

func (w *Writer) U32(v uint32) {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], v)
	w.Raw(b[:])
}
func (w *Writer) Str(s string) { w.U32(uint32(len(s))); w.Raw([]byte(s)) }

func (w *Writer) BytesField(b []byte) { w.U32(uint32(len(b))); w.Raw(b) }

// u16s writes s as little-endian uint16s, with no count.
func (w *Writer) u16s(s []uint16) {
	if len(s) == 0 {
		return
	}
	if nativeLittleEndian {
		w.Raw(unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), 2*len(s)))
		return
	}
	b := make([]byte, 2*len(s))
	for i, v := range s {
		binary.LittleEndian.PutUint16(b[2*i:], v)
	}
	w.Raw(b)
}

// Scales16 writes binary16 int4 scales, the WeightMat's own storage, as a u32 count + little-endian uint16 payload.
func (w *Writer) Scales16(s []uint16) {
	w.U32(uint32(len(s)))
	w.u16s(s)
}

// F32 writes a standalone f32 array: its encoding byte (EncF32), then the array (f32Body).
func (w *Writer) F32(s []float32) {
	w.Raw([]byte{EncF32})
	w.F32Body(s)
}

// F32Body batches into one raw write (a per-tensor temp), not 4 bytes at a time — the
// stream path would otherwise do millions of tiny writes. A weight matrix's own arrays use it directly: the matrix's
// kind byte is their encoding.
func (w *Writer) F32Body(s []float32) {
	w.U32(uint32(len(s)))
	if len(s) == 0 {
		return
	}
	b := make([]byte, 4*len(s))
	for i, v := range s {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(v))
	}
	w.Raw(b)
}

func (w *Writer) I8(s []int8) {
	w.U32(uint32(len(s)))
	if len(s) > 0 {
		w.Raw(unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), len(s)))
	}
}

// WeightMat writes m, eligible for kind 5 (row4-only) under a cpu-arm64 target —
// see weightMatKind3Only for the tensors that must never take kind 5.
func (w *Writer) WeightMat(m *linalg.WeightMat) { w.WeightMatKind(m, true) }

// WeightMatKind3Only is weightMat for a tensor that must never take kind 5 regardless of target: it always writes kind 3
// for an int4 tensor. Two reasons put a call site here:
//
//   - MoE-paged experts (l.Experts[*], gemma4's mo.expertsGateUp/expertsDown): moepaging.go reads these off the mmap with
//     no load-time repack step, so the file must carry whatever layout the pager needs, chosen once at write time, not
//     per reader. (layerpaging.go's dense per-layer pager prefers WeightMat.MappedSpanRow4 over MappedSpan, so the dense
//     QProj..DownProj stay kind-5-eligible through plain weightMat.)
//   - Not yet scoped: KDA/DeltaNet/qattn mixer projections and gemma4's fused-MoE router (mo.routerProj). These are absent
//     from Weights.matmulWeights(), which decoder.Load's post-load kind-5-vs-backend check walks (see
//     repackedOnlyInt4Count), so a kind-5 instance would slip past that check and rely solely on the soft downstream
//     residency decline. They stay kind-3-only until they get their own entry in that census.
func (w *Writer) WeightMatKind3Only(m *linalg.WeightMat) { w.WeightMatKind(m, false) }

func (w *Writer) WeightMatKind(m *linalg.WeightMat, eligible bool) {
	if raw, ok := m.Q4K(); ok {
		w.Raw([]byte{8})
		w.U32(uint32(m.Rows()))
		w.U32(uint32(m.Cols()))
		w.U32(0)
		w.Raw([]byte{0})
		w.AlignArray()
		w.BytesField(raw)
		return
	}
	if raw, kind, ok := kquant56Raw(m); ok {
		// Kinds 11 and 12 are written for the targets q4kNative56 builds them for (CarriesKQuant56); a bundle for any other
		// reader must not carry them.
		if !CarriesKQuant56(w.Target) {
			if w.Err == nil {
				w.Err = fmt.Errorf("decoder: a %dx%d tensor is held as native %s (a q4k load keeps the file's blocks); only a cpu-amd64 or cuda .giw carries it — write for one of those targets, or load with another Backend", m.Rows(), m.Cols(), m.Kind())
			}
			return
		}
		w.Raw([]byte{kind})
		w.U32(uint32(m.Rows()))
		w.U32(uint32(m.Cols()))
		w.U32(0)
		w.Raw([]byte{0})
		w.AlignArray()
		w.BytesField(raw)
		return
	}
	if m.Rows() == 0 {
		w.Raw([]byte{0}) // empty
		return
	}
	q4, q4s, group, isQ4 := m.Int4F16()
	if m.IsInt4() && !isQ4 {
		// Repacked-only (one row4 copy, quantizeLayerWM under Backend "cpu"): every kind is written from canonical bytes.
		if w.Err == nil {
			w.Err = fmt.Errorf("decoder: a %dx%d int4 tensor holds only its row4 layout (a Backend \"cpu\" load keeps no canonical copy); load with another Backend to serialize it", m.Rows(), m.Cols())
		}
		return
	}
	q8, scales, w8a8, isQ8 := m.Int8()
	f32, _ := m.F32()
	var kind byte
	switch {
	case isQ4:
		kind = 3
		// Kind 5 is a function of the writer's target and this tensor's shape, never of whatever repack state already
		// sits in RAM (repackW4A8Row4IfEligible repacks every heap-built int4 tensor on arm64, whether or not this call
		// is a prequant path). So the row4 arrays are recomputed here from the canonical ones.
		if eligible && w.Target == TargetCPUArm64 {
			if r4, r4s, ok := RepackRow4ForEmit(q4, q4s, m.Rows(), m.Cols(), group); ok {
				kind, q4, q4s = 5, r4, r4s
			}
		}
	case isQ8:
		kind = 2
	case w.F32Kind1:
		kind = 1
	default:
		kind = KindF32
	}
	w.Raw([]byte{kind})
	w.U32(uint32(m.Rows()))
	w.U32(uint32(m.Cols()))
	w.U32(uint32(group)) // 0 unless int4
	if w8a8 {
		w.Raw([]byte{1})
	} else {
		w.Raw([]byte{0})
	}
	// Every array below but kind 1's is aliased by the reader and is padded so the alias is legal (see ArrayPad).
	switch kind {
	case 1:
		w.F32Body(f32)
	case KindF32:
		w.AlignArray()
		w.F32Body(f32)
	case 2:
		w.AlignArray()
		w.F32Body(scales)
		w.AlignArray()
		w.I8(q8)
	case 3, 5: // binary16 group scales, then the nibbles: canonical for kind 3, the row4 layout for kind 5
		w.AlignArray()
		w.Scales16(q4s)
		w.AlignArray()
		w.BytesField(q4)
	}
}

// fusedEligible reports whether ms can be written as one fused group (kind 6): a Metal-target writer, at least
// two members, every member a canonical group-32 int4 with the same K (K%32==0, so every member's nibble bytes are a
// multiple of 16 and the members stay 16-aligned back to back). Anything else (an absent V on a K=V layer, an int8
// tensor, a mixed K) is written the ordinary way, one record at a time.
func (w *Writer) fusedEligible(ms []*linalg.WeightMat) bool {
	if w.Target != TargetMetal || len(ms) < 2 {
		return false
	}
	k := ms[0].Cols()
	for _, m := range ms {
		if m.Rows() == 0 || m.Cols() != k || k%32 != 0 {
			return false
		}
		if _, _, group, ok := m.Int4F16(); !ok || group != 32 {
			return false
		}
	}
	return true
}

// FusedGroup writes ms, in the order a resident backend fuses them (q,k,v / gate,up), as ONE group whose nibbles are
// contiguous, and whose binary16 scales are too: a Metal fused GEMV wants its rows in one buffer, and an mmap'd file can
// only be aliased into one if the rows are adjacent there. Layout:
//
//	per member:  kind 6 | rows | cols | group | w8a8=0                                      (no arrays)
//	then:        pad to 16 | nibbles of member 0 | nibbles of member 1 | ...                (no length prefixes)
//	then:        pad to 16 | scales of member 0 | scales of member 1 | ...                  (little-endian uint16)
//
// Both blocks' lengths are a pure function of each member's shape (rows*cols/2 nibble bytes, rows*cols/group scales),
// so they need no prefixes: a prefix between two members would be exactly the gap this exists to remove. A group that
// is not eligible is written as plain consecutive records, byte-for-byte what weightMat writes.
func (w *Writer) FusedGroup(ms ...*linalg.WeightMat) {
	if !w.fusedEligible(ms) {
		for _, m := range ms {
			w.WeightMat(m)
		}
		return
	}
	for _, m := range ms {
		_, _, group, _ := m.Int4F16()
		w.Raw([]byte{6})
		w.U32(uint32(m.Rows()))
		w.U32(uint32(m.Cols()))
		w.U32(uint32(group))
		w.Raw([]byte{0})
	}
	w.Align16()
	for _, m := range ms {
		q4, _, _, _ := m.Int4F16()
		w.Raw(q4)
	}
	w.Align16()
	for _, m := range ms {
		w.u16s(m.Int4ScalesF16())
	}
}

// v6Layer writes the state that made five families unrepresentable, in one unconditional tail. Unconditional rather than
// arch-gated like the gemma4 tail: every field here is empty on the families that do not use it, so the cost is a handful
// of zero lengths per layer, and an arch-gated tail is how a family's state goes missing (the gate is another place to
// remember). TestSerializeCensus_noSilentFieldDrop checks that claim against the struct itself.
func (w *Writer) v6Layer(l *darch.LayerWeights) {
	w.WeightMat(&l.GProj) // Laguna's attention output gate (per-head or per-element)
	w.F32(l.AttnSinks)    // gpt-oss per-head attention sinks
	// gpt-oss per-expert biases. The expert loop above writes only the three matrices; these are
	// nil for every other family, so this is three zero lengths per expert elsewhere.
	for e := range l.Experts {
		w.F32(l.Experts[e].GateBias)
		w.F32(l.Experts[e].UpBias)
		w.F32(l.Experts[e].DownBias)
	}
	w.F32(l.SharedExpert.GateBias)
	w.F32(l.SharedExpert.UpBias)
	w.F32(l.SharedExpert.DownBias)
	// MLA (DeepSeek / Kimi): presence byte then the eight tensors.
	if l.MLA == nil {
		w.Raw([]byte{0})
	} else {
		w.Raw([]byte{1})
		m := l.MLA
		w.mambaProj(m.QAProj, &m.QAQ) // a quantized load's int8 projections as weight records, else f32 arrays
		w.F32(m.QALayernorm)
		w.mambaProj(m.QBProj, &m.QBQ)
		w.mambaProj(m.QProj, &m.QQ)
		w.mambaProj(m.KVAProj, &m.KVAQ)
		w.F32(m.KVALayernorm)
		w.F32(m.KVBProj)
		w.mambaProj(m.OProj, &m.OQ)
	}
	// Mamba-2 (Granite / Nemotron): presence byte then the eight tensors. The recurrent STATE is
	// not written and must not be — it is per-sequence, rebuilt at load; only the weights are here.
	if l.Mamba == nil {
		w.Raw([]byte{0})
	} else {
		w.Raw([]byte{1})
		m := l.Mamba
		w.mambaProj(m.InProj, &m.InQ)
		w.F32(m.ConvW)
		w.F32(m.ConvB)
		w.F32(m.ALog)
		w.F32(m.D)
		w.F32(m.DtBias)
		w.F32(m.NormW)
		w.mambaProj(m.OutProj, &m.OutQ)
	}
}

// mambaProj writes a Mamba projection in the encoding it is held in: an int8 weight matrix (kind 2) when the load quantized
// it, else a standalone f32 array (EncF32). Both start with their encoding byte, so the reader tells them apart.
func (w *Writer) mambaProj(f []float32, q *linalg.WeightMat) {
	if q.Rows() > 0 {
		w.WeightMatKind3Only(q)
		return
	}
	w.F32(f)
}

// v8Layer writes the LFM2 gated short-convolution mixer: presence byte then the three tensors. Unconditional like the v6
// tail and for the same reason; the cost is one zero byte per layer on every other family. A writer that omits this field
// produces a CRC-valid bundle whose first forward nil-dereferences on conv layer 0 (lw.shortConv == nil), and prequant's
// selfCheck cannot see it because it only Loads.
//
// As with mamba, only the weights are here: the rolling conv window is per-sequence state that lives in the KVCache and
// is rebuilt at load.
func (w *Writer) v8Layer(l *darch.LayerWeights) {
	if l.ShortConv == nil {
		w.Raw([]byte{0})
		return
	}
	w.Raw([]byte{1})
	c := l.ShortConv
	w.island(&c.InProj)
	w.F32(c.ConvW)
	w.island(&c.OutProj)
}

// island writes an island matrix in the kind it is held in: a bf16 one as a kind-9 record, its values aliased on load; an f32
// one as a standalone f32 array (EncF32). The reader tells the two apart by the encoding byte (giwReader.island).
func (w *Writer) island(m *linalg.WeightMat) {
	u, bf16, ok := m.Half16()
	if !ok || !bf16 || w.IslandsF32 {
		w.F32(IslandF32(m))
		return
	}
	w.Raw([]byte{giwKindBF16})
	w.U32(uint32(m.Rows()))
	w.U32(uint32(m.Cols()))
	w.U32(0)
	w.Raw([]byte{0})
	w.AlignArray()
	w.Scales16(u)
}

// v9Layer writes Bailing Hybrid's (Ling 3.0) per-layer v9 tail: MLA's optional attention-output gate (l.mla.gProj, added to
// mlaWeights after v6Layer's MLA block shipped, so it rides a new version rather than retrofitting v6Layer's fixed byte
// layout, which would corrupt the read of every existing v6/v7/v8 file), then the KDA mixer (presence byte + the thirteen
// tensors, kdaWeights' own field count). Without it a round-tripped bailing_hybrid bundle nil-dereferences in
// kdaMixerStep on the first KDA layer; TestSerializeCensus_noSilentFieldDrop catches a missing field.
func (w *Writer) v9Layer(l *darch.LayerWeights) {
	var gProj []float32
	if l.MLA != nil {
		gProj = l.MLA.GProj // nil for deepseek_v2/v3/kimi_k2 (no gate); set for Bailing Hybrid
	}
	w.F32(gProj)
	if l.KDA == nil {
		w.Raw([]byte{0})
		return
	}
	w.Raw([]byte{1})
	k := l.KDA
	w.WeightMatKind3Only(&k.QProj)
	w.WeightMatKind3Only(&k.KProj)
	w.WeightMatKind3Only(&k.VProj)
	w.F32(k.QConvW)
	w.F32(k.KConvW)
	w.F32(k.VConvW)
	w.WeightMatKind3Only(&k.FProj)
	w.F32(k.DtBias)
	w.F32(k.ALog)
	w.F32(k.BProj)
	w.WeightMatKind3Only(&k.GProj)
	w.F32(k.ONormW)
	w.WeightMatKind3Only(&k.OProj)
}

// gemma4Layer writes the v4 Gemma 4 per-layer tail: the PLE branch, the per-layer
// output scalar, the KV-share flags, and (when enable_moe_block) the gemma4moe
// sub-block's OWN tensors — the ones NOT already covered by the standard block. The
// dense-branch MLP + its pre/post norms are l.GateProj/UpProj/DownProj +
// l.PreMLPNorm/PostMLPNorm (already written), which the reader re-aliases into
// gemma4moe; only the three parallel-branch norms, the router (l.Router is empty for
// gemma4), the per-expert scale, and the fused experts are new here.
func (w *Writer) gemma4Layer(l *darch.LayerWeights) {
	w.WeightMat(&l.PLEGate)
	w.WeightMat(&l.PLEProj)
	w.F32(l.PostPLENorm)
	w.U32(math.Float32bits(l.LayerScalar))
	w.Raw([]byte{b2u8(l.KVShared), b2u8(l.VFromK)})
	if l.Gemma4MoE == nil {
		w.Raw([]byte{0})
		return
	}
	w.Raw([]byte{1})
	mo := l.Gemma4MoE
	w.F32(mo.PostFFNNorm1)
	w.F32(mo.PreFFNNorm2)
	w.F32(mo.PostFFNNorm2)
	w.WeightMatKind3Only(&mo.RouterProj)
	w.F32(mo.RouterScale)
	w.F32(mo.PerExpertScale)
	w.U32(uint32(len(mo.ExpertsGateUp)))
	for e := range mo.ExpertsGateUp {
		w.WeightMatKind3Only(&mo.ExpertsGateUp[e])
		w.WeightMatKind3Only(&mo.ExpertsDown[e])
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
func (w *Writer) hybridLayer(l *darch.LayerWeights) {
	switch {
	case l.Delta != nil:
		w.Raw([]byte{1})
		d := l.Delta
		w.WeightMatKind3Only(&d.InProjQKV)
		w.WeightMatKind3Only(&d.InProjZ)
		w.F32(d.InProjB)
		w.F32(d.InProjA)
		w.F32(d.ConvW)
		w.F32(d.DtBias)
		w.F32(d.NegExpA) // the precomputed −exp(A_log); stored as-is, no recompute on load
		w.F32(d.NormW)
		w.WeightMatKind3Only(&d.OutProj)
	case l.QAttn != nil:
		w.Raw([]byte{2})
		q := l.QAttn
		w.WeightMatKind3Only(&q.QProj)
		w.WeightMatKind3Only(&q.KProj)
		w.WeightMatKind3Only(&q.VProj)
		w.WeightMatKind3Only(&q.OProj)
		w.F32(q.QNorm)
		w.F32(q.KNorm)
	default:
		// The "no hybrid extras" marker. MLA and Mamba-2 are written by v6Layer.
		w.Raw([]byte{0})
	}
}

// HeadTable writes the table that stands for dst, delegating to HeadTableFn if configured.
func (w *Writer) HeadTable(dst *linalg.WeightMat) {
	if w.HeadTableFn != nil && w.HeadTableFn(w, dst) {
		return
	}
	w.WeightMat(dst)
}

// SetErr records err as the writer first error.
func (w *Writer) SetErr(err error) {
	if w.Err == nil {
		w.Err = err
	}
}
