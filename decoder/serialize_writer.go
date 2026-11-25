package decoder

import (
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"io"
	"math"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
)

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
	arch *Architecture // set in writeBundle; gates the gemma4 model-level inputs and per-layer tail

	// target opts weightMat into emitting kind 5 (row4-only, no canonical arrays) for every eligible int4 tensor on a
	// cpu-arm64 target, instead of kind 3: the one-representation-per-target policy, .giw's counterpart to
	// wantsCanonicalInt4's in-RAM decision. GIWTargetNone (the zero value) keeps
	// kind 3 for every int4 tensor.
	target GIWTarget

	// tables are the embedding-shaped tables a streaming GGUF build registered instead of building (streamedTable), keyed by
	// the Weights field each stands for; writeHeadGlobals writes them from their sources.
	tables map[*linalg.WeightMat]*streamedTable

	// layerBudget is the memory streamLayers may hold in built, unwritten layers (stream mode). Zero builds one layer at
	// a time.
	layerBudget int64

	// testHooks carries per-stream test overrides populated from context by StreamTranscodeGGUF/StreamTranscodeDir.
	testHooks streamTestHooks
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

// pos is how many bytes have been written since the blob start, in either mode — the quantity the
// alignment padding is defined over.
func (w *giwWriter) pos() int64 {
	if w.sink == nil {
		return int64(len(w.buf))
	}
	return w.n
}

// giwArrayPad is the number of zero bytes to insert at blob offset off so that an array written as
// `u32 count | payload` has its PAYLOAD (off+pad+4) on a 16-byte boundary. A pure function of the
// offset, so writer and reader compute the same answer without recording it.
func giwArrayPad(off int64) int { return int((-(off + 4)) & 15) }

// alignHead ends the variable-length header on a 16-byte boundary. The header carries a length-prefixed quant label
// that the resident writer emits and the streaming writer deliberately does not, so the two paths' offsets differ by the
// label's length. Every later array's padding is a function of its absolute offset, so without this the two would lay out
// the same weights with different padding and stop being byte-identical
// (TestStreamTranscodeMatchesResident). With it, everything after the header is at the same offsets in both.
func (w *giwWriter) alignHead() { w.align16() }

// align16 pads to a 16-byte boundary: before a block that carries no count (a fused group's nibbles and scales).
func (w *giwWriter) align16() {
	if pad := int((-w.pos()) & 15); pad > 0 {
		var z [16]byte
		w.raw(z[:pad])
	}
}

// errCopiedF32Bundle refuses a bundle whose f32 weights are in kind 1, which a load would copy onto the heap, where the fit guard treats
// a bundle's weights as file-backed. A sidecar that hits this is rebuilt (cacheFresh).
var errCopiedF32Bundle = &SerializeError{"this bundle predates the mapped f32 encoding, and would load its f32 weights onto the heap, where the fit guard does not price them; rebuild it from its source: go run ./cmd/prequant -quant f32 -o <out.giw> <source>"}

// giwF32Kind1ForTest writes an f32 matrix as kind 1, the encoding bundles from before kind 10 carry, and lets a load read an f32 body
// in it, so a test can build one and compare it. Set through the test helper forceGIWF32Kind1; a package global, so not safe under
// t.Parallel.
var giwF32Kind1ForTest bool

// alignArray pads before the next array's count.
func (w *giwWriter) alignArray() {
	if pad := giwArrayPad(w.pos()); pad > 0 {
		var z [16]byte
		w.raw(z[:pad])
	}
}

func (w *giwWriter) u32(v uint32) {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], v)
	w.raw(b[:])
}
func (w *giwWriter) str(s string) { w.u32(uint32(len(s))); w.raw([]byte(s)) }

func (w *giwWriter) bytesField(b []byte) { w.u32(uint32(len(b))); w.raw(b) }

// u16s writes s as little-endian uint16s, with no count.
func (w *giwWriter) u16s(s []uint16) {
	if len(s) == 0 {
		return
	}
	if nativeLittleEndian {
		w.raw(unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), 2*len(s)))
		return
	}
	b := make([]byte, 2*len(s))
	for i, v := range s {
		binary.LittleEndian.PutUint16(b[2*i:], v)
	}
	w.raw(b)
}

// scales16 writes binary16 int4 scales, the WeightMat's own storage, as a u32 count + little-endian uint16 payload.
func (w *giwWriter) scales16(s []uint16) {
	w.u32(uint32(len(s)))
	w.u16s(s)
}

// f32 writes a standalone f32 array: its encoding byte (giwEncF32), then the array (f32Body).
func (w *giwWriter) f32(s []float32) {
	w.raw([]byte{giwEncF32})
	w.f32Body(s)
}

// f32Body batches into one raw write (a per-tensor temp), not 4 bytes at a time — the
// stream path would otherwise do millions of tiny writes. A weight matrix's own arrays use it directly: the matrix's
// kind byte is their encoding.
func (w *giwWriter) f32Body(s []float32) {
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

// weightMat writes m, eligible for kind 5 (row4-only) under a cpu-arm64 target —
// see weightMatKind3Only for the tensors that must never take kind 5.
func (w *giwWriter) weightMat(m *linalg.WeightMat) { w.weightMatKind(m, true) }

// weightMatKind3Only is weightMat for a tensor that must never take kind 5 regardless of target: it always writes kind 3
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
func (w *giwWriter) weightMatKind3Only(m *linalg.WeightMat) { w.weightMatKind(m, false) }

func (w *giwWriter) weightMatKind(m *linalg.WeightMat, eligible bool) {
	if raw, ok := m.Q4K(); ok {
		w.raw([]byte{8})
		w.u32(uint32(m.Rows()))
		w.u32(uint32(m.Cols()))
		w.u32(0)
		w.raw([]byte{0})
		w.alignArray()
		w.bytesField(raw)
		return
	}
	if raw, kind, ok := kquant56Raw(m); ok {
		// Kinds 11 and 12 are written for the targets q4kNative56 builds them for (giwCarriesKQuant56); a bundle for any other
		// reader must not carry them.
		if !giwCarriesKQuant56(w.target) {
			if w.err == nil {
				w.err = fmt.Errorf("decoder: a %dx%d tensor is held as native %s (a q4k load keeps the file's blocks); only a cpu-amd64 or cuda .giw carries it — write for one of those targets, or load with another Backend", m.Rows(), m.Cols(), m.Kind())
			}
			return
		}
		w.raw([]byte{kind})
		w.u32(uint32(m.Rows()))
		w.u32(uint32(m.Cols()))
		w.u32(0)
		w.raw([]byte{0})
		w.alignArray()
		w.bytesField(raw)
		return
	}
	if m.Rows() == 0 {
		w.raw([]byte{0}) // empty
		return
	}
	q4, q4s, group, isQ4 := m.Int4F16()
	if m.IsInt4() && !isQ4 {
		// Repacked-only (one row4 copy, quantizeLayerWM under Backend "cpu"): every kind is written from canonical bytes.
		if w.err == nil {
			w.err = fmt.Errorf("decoder: a %dx%d int4 tensor holds only its row4 layout (a Backend \"cpu\" load keeps no canonical copy); load with another Backend to serialize it", m.Rows(), m.Cols())
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
		if eligible && w.target == GIWTargetCPUArm64 {
			if r4, r4s, ok := repackRow4ForEmit(q4, q4s, m.Rows(), m.Cols(), group); ok {
				kind, q4, q4s = 5, r4, r4s
			}
		}
	case isQ8:
		kind = 2
	case giwF32Kind1ForTest:
		kind = 1
	default:
		kind = giwKindF32
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
	// Every array below but kind 1's is aliased by the reader and is padded so the alias is legal (see giwArrayPad).
	switch kind {
	case 1:
		w.f32Body(f32)
	case giwKindF32:
		w.alignArray()
		w.f32Body(f32)
	case 2:
		w.alignArray()
		w.f32Body(scales)
		w.alignArray()
		w.i8(q8)
	case 3, 5: // binary16 group scales, then the nibbles: canonical for kind 3, the row4 layout for kind 5
		w.alignArray()
		w.scales16(q4s)
		w.alignArray()
		w.bytesField(q4)
	}
}

// fusedEligible reports whether ms can be written as one fused group (kind 6): a Metal-target writer, at least
// two members, every member a canonical group-32 int4 with the same K (K%32==0, so every member's nibble bytes are a
// multiple of 16 and the members stay 16-aligned back to back). Anything else (an absent V on a K=V layer, an int8
// tensor, a mixed K) is written the ordinary way, one record at a time.
func (w *giwWriter) fusedEligible(ms []*linalg.WeightMat) bool {
	if w.target != GIWTargetMetal || len(ms) < 2 {
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

// fusedGroup writes ms, in the order a resident backend fuses them (q,k,v / gate,up), as ONE group whose nibbles are
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
func (w *giwWriter) fusedGroup(ms ...*linalg.WeightMat) {
	if !w.fusedEligible(ms) {
		for _, m := range ms {
			w.weightMat(m)
		}
		return
	}
	for _, m := range ms {
		_, _, group, _ := m.Int4F16()
		w.raw([]byte{6})
		w.u32(uint32(m.Rows()))
		w.u32(uint32(m.Cols()))
		w.u32(uint32(group))
		w.raw([]byte{0})
	}
	w.align16()
	for _, m := range ms {
		q4, _, _, _ := m.Int4F16()
		w.raw(q4)
	}
	w.align16()
	for _, m := range ms {
		w.u16s(m.Int4ScalesF16())
	}
}

// v6Layer writes the state that made five families unrepresentable, in one unconditional tail. Unconditional rather than
// arch-gated like the gemma4 tail: every field here is empty on the families that do not use it, so the cost is a handful
// of zero lengths per layer, and an arch-gated tail is how a family's state goes missing (the gate is another place to
// remember). TestSerializeCensus_noSilentFieldDrop checks that claim against the struct itself.
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
		w.mambaProj(m.qAProj, &m.qAQ) // a quantized load's int8 projections as weight records, else f32 arrays
		w.f32(m.qALayernorm)
		w.mambaProj(m.qBProj, &m.qBQ)
		w.mambaProj(m.qProj, &m.qQ)
		w.mambaProj(m.kvAProj, &m.kvAQ)
		w.f32(m.kvALayernorm)
		w.f32(m.kvBProj)
		w.mambaProj(m.oProj, &m.oQ)
	}
	// Mamba-2 (Granite / Nemotron): presence byte then the eight tensors. The recurrent STATE is
	// not written and must not be — it is per-sequence, rebuilt at load; only the weights are here.
	if l.mamba == nil {
		w.raw([]byte{0})
	} else {
		w.raw([]byte{1})
		m := l.mamba
		w.mambaProj(m.inProj, &m.inQ)
		w.f32(m.convW)
		w.f32(m.convB)
		w.f32(m.aLog)
		w.f32(m.d)
		w.f32(m.dtBias)
		w.f32(m.normW)
		w.mambaProj(m.outProj, &m.outQ)
	}
}

// mambaProj writes a Mamba projection in the encoding it is held in: an int8 weight matrix (kind 2) when the load quantized
// it, else a standalone f32 array (giwEncF32). Both start with their encoding byte, so the reader tells them apart.
func (w *giwWriter) mambaProj(f []float32, q *linalg.WeightMat) {
	if q.Rows() > 0 {
		w.weightMatKind3Only(q)
		return
	}
	w.f32(f)
}

// v8Layer writes the LFM2 gated short-convolution mixer: presence byte then the three tensors. Unconditional like the v6
// tail and for the same reason; the cost is one zero byte per layer on every other family. A writer that omits this field
// produces a CRC-valid bundle whose first forward nil-dereferences on conv layer 0 (lw.shortConv == nil), and prequant's
// selfCheck cannot see it because it only Loads.
//
// As with mamba, only the weights are here: the rolling conv window is per-sequence state that lives in the KVCache and
// is rebuilt at load.
func (w *giwWriter) v8Layer(l *LayerWeights) {
	if l.shortConv == nil {
		w.raw([]byte{0})
		return
	}
	w.raw([]byte{1})
	c := l.shortConv
	w.island(&c.inProj)
	w.f32(c.convW)
	w.island(&c.outProj)
}

// giwIslandsF32ForTest writes islands as f32 arrays whatever kind the load holds: the form a bundle had before kind 9.
var giwIslandsF32ForTest bool

// island writes an island matrix in the kind it is held in: a bf16 one as a kind-9 record, its values aliased on load; an f32
// one as a standalone f32 array (giwEncF32). The reader tells the two apart by the encoding byte (giwReader.island).
func (w *giwWriter) island(m *linalg.WeightMat) {
	u, bf16, ok := m.Half16()
	if !ok || !bf16 || giwIslandsF32ForTest {
		w.f32(islandF32(m))
		return
	}
	w.raw([]byte{giwKindBF16})
	w.u32(uint32(m.Rows()))
	w.u32(uint32(m.Cols()))
	w.u32(0)
	w.raw([]byte{0})
	w.alignArray()
	w.scales16(u)
}

// v9Layer writes Bailing Hybrid's (Ling 3.0) per-layer v9 tail: MLA's optional attention-output gate (l.mla.gProj, added to
// mlaWeights after v6Layer's MLA block shipped, so it rides a new version rather than retrofitting v6Layer's fixed byte
// layout, which would corrupt the read of every existing v6/v7/v8 file), then the KDA mixer (presence byte + the thirteen
// tensors, kdaWeights' own field count). Without it a round-tripped bailing_hybrid bundle nil-dereferences in
// kdaMixerStep on the first KDA layer; TestSerializeCensus_noSilentFieldDrop catches a missing field.
func (w *giwWriter) v9Layer(l *LayerWeights) {
	var gProj []float32
	if l.mla != nil {
		gProj = l.mla.gProj // nil for deepseek_v2/v3/kimi_k2 (no gate); set for Bailing Hybrid
	}
	w.f32(gProj)
	if l.kda == nil {
		w.raw([]byte{0})
		return
	}
	w.raw([]byte{1})
	k := l.kda
	w.weightMatKind3Only(&k.qProj)
	w.weightMatKind3Only(&k.kProj)
	w.weightMatKind3Only(&k.vProj)
	w.f32(k.qConvW)
	w.f32(k.kConvW)
	w.f32(k.vConvW)
	w.weightMatKind3Only(&k.fProj)
	w.f32(k.dtBias)
	w.f32(k.aLog)
	w.f32(k.bProj)
	w.weightMatKind3Only(&k.gProj)
	w.f32(k.oNormW)
	w.weightMatKind3Only(&k.oProj)
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
	w.weightMatKind3Only(&mo.routerProj)
	w.f32(mo.routerScale)
	w.f32(mo.perExpertScale)
	w.u32(uint32(len(mo.expertsGateUp)))
	for e := range mo.expertsGateUp {
		w.weightMatKind3Only(&mo.expertsGateUp[e])
		w.weightMatKind3Only(&mo.expertsDown[e])
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
		w.weightMatKind3Only(&d.inProjQKV)
		w.weightMatKind3Only(&d.inProjZ)
		w.f32(d.inProjB)
		w.f32(d.inProjA)
		w.f32(d.convW)
		w.f32(d.dtBias)
		w.f32(d.negExpA) // the precomputed −exp(A_log); stored as-is, no recompute on load
		w.f32(d.normW)
		w.weightMatKind3Only(&d.outProj)
	case l.qattn != nil:
		w.raw([]byte{2})
		q := l.qattn
		w.weightMatKind3Only(&q.qProj)
		w.weightMatKind3Only(&q.kProj)
		w.weightMatKind3Only(&q.vProj)
		w.weightMatKind3Only(&q.oProj)
		w.f32(q.qNorm)
		w.f32(q.kNorm)
	default:
		// The "no hybrid extras" marker. MLA and Mamba-2 are written by v6Layer.
		w.raw([]byte{0})
	}
}
