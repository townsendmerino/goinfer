package decoder

import (
	"encoding/binary"
	"fmt"
	"math"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
)

// --- reader (cursor over data; big arrays aliased, floats copied) ---

type giwReader struct {
	data []byte
	off  int
	err  error
	arch *Architecture // resolved from the serialized config; gates the gemma4 tail
	scan bool          // skip kind-1 matrices (scanGIWWeights)
}

// opensCopiedF32 reports whether a bundle's first weight record, the embedding, is a kind-1 matrix. A quant bundle's embedding is
// quantized (quantMode.embedding) and an f32 bundle's is f32, so this one byte tells a copied f32 body from an int8 or int4 bundle
// whose routers alone are kind 1, on every family (TestGIWCopiedF32_everyFamily), with nothing copied. Call it at the first record.
func (r *giwReader) opensCopiedF32() bool {
	return !r.scan && !giwF32Kind1ForTest && r.off < len(r.data) && r.data[r.off] == 1
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

// align16 skips the padding the writer put before a 16-aligned position (giwWriter.align16): the end of the header and
// the two blocks of a fused group.
func (r *giwReader) align16() {
	if r.err != nil {
		return
	}
	if pad := int((-int64(r.off)) & 15); r.need(pad) {
		r.off += pad
	}
}

func (r *giwReader) alignHead() { r.align16() }

// alignArray skips the zero padding the writer put before an aligned array's count (see giwArrayPad).
func (r *giwReader) alignArray() {
	if r.err != nil {
		return
	}
	if pad := giwArrayPad(int64(r.off)); r.need(pad) {
		r.off += pad
	}
}

// nativeLittleEndian reports whether float32s can be read straight out of the little-endian blob.
var nativeLittleEndian = func() bool { x := uint16(1); return *(*byte)(unsafe.Pointer(&x)) == 1 }()

// f32Alias reads a padded f32 array as an ALIAS of the mapping when its payload is 4-byte aligned
// in memory (always true for a blob inside a v3-bundle mapping), and copies otherwise, so an
// oddly-placed blob still loads, onto the heap. For weight-matrix scale arrays only: the mapping is read-only, and a
// scale array is read-only by construction (its sibling nibble/int8 arrays are already aliased).
// The caller must have called alignArray first.
func (r *giwReader) f32Alias() []float32 {
	n := int(r.u32())
	if n == 0 || !r.need(n*4) {
		return nil
	}
	if p := unsafe.Pointer(&r.data[r.off]); nativeLittleEndian && uintptr(p)%4 == 0 {
		r.off += n * 4
		return unsafe.Slice((*float32)(p), n)
	}
	out := make([]float32, n)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(r.data[r.off:]))
		r.off += 4
	}
	return out
}

// f32 reads a standalone f32 array (giwWriter.f32): its encoding byte, which must be giwEncF32, then the array.
func (r *giwReader) f32() []float32 {
	if enc := r.u8(); enc != giwEncF32 {
		r.unknownEncoding(enc)
		return nil
	}
	return r.f32Body()
}

// unknownEncoding refuses a tensor whose encoding this build does not read. Only that encoding is refused: a newer writer
// adds an encoding without a version bump, and a bundle that never uses it still loads.
func (r *giwReader) unknownEncoding(enc byte) {
	r.fail(fmt.Sprintf("a tensor's encoding %d is not one this build reads (written by a newer goinfer; the sidecar is rebuilt)", enc))
}

// f32Body copies (the input isn't guaranteed aligned). len 0 ⇒ nil, preserving the
// "absent ⇒ nil" convention the forward pass checks for biases/norms.
func (r *giwReader) f32Body() []float32 {
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

// u16View returns n little-endian uint16s at the cursor as an ALIAS of the blob when 2-byte aligned on a little-endian
// host, else a copy: f32Alias's rule for f16 scales.
func (r *giwReader) u16View(n int) []uint16 {
	if n == 0 || !r.need(2*n) {
		return nil
	}
	if p := unsafe.Pointer(&r.data[r.off]); nativeLittleEndian && uintptr(p)%2 == 0 {
		r.off += 2 * n
		return unsafe.Slice((*uint16)(p), n)
	}
	out := make([]uint16, n)
	for i := range out {
		out[i] = binary.LittleEndian.Uint16(r.data[r.off:])
		r.off += 2
	}
	return out
}

// int4Scales reads an int4 scale array (kinds 3 and 5) as binary16, aliased from the mapping. The caller must have
// called alignArray first.
func (r *giwReader) int4Scales() []uint16 { return r.u16View(int(r.u32())) }

// rawAlias ALIASES a []byte (int4 packed nibbles): a plain subslice of data.
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
	if kind != 1 && kind != giwKindF32 && kind != 2 && kind != 3 && kind != 5 && kind != 8 && kind != giwKindBF16 && kind != giwKindQ5K && kind != giwKindQ6K { // kind 6 is read as a fused group, never here
		r.unknownEncoding(kind)
		return linalg.WeightMat{}
	}
	rows, cols, group := int(r.u32()), int(r.u32()), int(r.u32())
	if !r.need(1) {
		return linalg.WeightMat{}
	}
	w8a8 := r.data[r.off] == 1
	r.off++
	// rows/cols/group are blob-controlled. linalg.Wrap{Int8,Int4} PANIC on a length mismatch or group<=0, and a wrong-length
	// WrapF32 defers the panic to first use, so a one-byte flip with a recomputed CRC would crash the loader. Validate dims
	// and array lengths here (the arrays were already bounded by r.need); a mismatch is a *SerializeError via r.fail, not a
	// panic. The maxWeightDim cap keeps rows*cols from overflowing int before the equality check.
	const maxWeightDim = 1 << 26
	if rows <= 0 || cols <= 0 || rows > maxWeightDim || cols > maxWeightDim {
		r.fail(fmt.Sprintf("weightMat implausible dims %d×%d", rows, cols))
		return linalg.WeightMat{}
	}
	switch kind {
	case 1, giwKindF32:
		var f []float32
		if kind == 1 {
			if r.scan {
				if n := int(r.u32()); n != rows*cols || !r.need(n*4) {
					r.fail(fmt.Sprintf("f32 weightMat %d×%d has %d values", rows, cols, n))
				} else {
					r.off += n * 4
				}
				return linalg.WeightMat{}
			}
			f = r.f32Body()
		} else {
			r.alignArray()
			f = r.f32Alias()
		}
		if len(f) != rows*cols {
			r.fail(fmt.Sprintf("f32 weightMat %d×%d has %d values", rows, cols, len(f)))
			return linalg.WeightMat{}
		}
		return linalg.WrapF32(f, rows, cols)
	case 2:
		r.alignArray()
		scales := r.f32Alias() // read order matches the writer (scales, then codes)
		r.alignArray()
		q8 := r.i8()
		if len(q8) != rows*cols || len(scales) != rows {
			r.fail(fmt.Sprintf("int8 weightMat %d×%d: q8=%d (want %d) scales=%d (want %d)", rows, cols, len(q8), rows*cols, len(scales), rows))
			return linalg.WeightMat{}
		}
		return linalg.WrapInt8(q8, scales, rows, cols, w8a8)
	case 3:
		r.alignArray()
		q4s := r.int4Scales()
		r.alignArray()
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
		wm4, _ := linalg.WrapInt4(q4, q4s, rows, cols, group, linalg.LayoutCanonical)
		return wm4
	case 5:
		if group <= 0 {
			r.fail(fmt.Sprintf("int4-row4-only weightMat group %d ≤ 0", group))
			return linalg.WeightMat{}
		}
		// The row4 arrays have kind 3's lengths: a repack, not a requant.
		wantQ4, wantScales := rows*((cols+1)/2), rows*((cols+group-1)/group)
		r.alignArray()
		q4Row4Scales := r.int4Scales()
		r.alignArray()
		q4Row4 := r.rawAlias() // zero-copy — no canonical bytes exist in this file at all
		if len(q4Row4) != wantQ4 || len(q4Row4Scales) != wantScales {
			r.fail(fmt.Sprintf("int4-row4-only weightMat %d×%d group=%d: q4Row4=%d (want %d) q4Row4Scales=%d (want %d)", rows, cols, group, len(q4Row4), wantQ4, len(q4Row4Scales), wantScales))
			return linalg.WeightMat{}
		}
		wm, ok := linalg.WrapInt4(q4Row4, q4Row4Scales, rows, cols, group, linalg.LayoutRow4)
		if !ok {
			// Named and actionable: a kind-5 file is a promise to one target. ok=false here means Int4Row4Usable rejected
			// this core: wrong arch (the file was written for arm64), or a shape this core's build cannot run the row4
			// kernel for.
			r.fail(fmt.Sprintf("int4 weightMat %d×%d group=%d is stored row4-only (kind 5, a cpu-arm64 prequant target) but this core cannot use that layout — rebuild with `go run ./cmd/prequant -target <this core>` (or delete the stream-weights cache so it rebuilds automatically)", rows, cols, group))
			return linalg.WeightMat{}
		}
		return wm
	case giwKindBF16:
		r.alignArray()
		u := r.u16View(int(r.u32())) // zero-copy: aliased from the mapping
		if len(u) != rows*cols {
			r.fail(fmt.Sprintf("bf16 weightMat %d×%d has %d values", rows, cols, len(u)))
			return linalg.WeightMat{}
		}
		return linalg.WrapBF16(u, rows, cols)
	case 8:
		r.alignArray()
		raw := r.rawAlias() // zero-copy: the super-blocks are aliased from the mapping
		if cols%256 != 0 || len(raw) != rows*linalg.Q4KRowBytes(cols) {
			r.fail(fmt.Sprintf("q4k weightMat %d×%d: %d bytes, want %d whole 256-value super-blocks per row", rows, cols, len(raw), cols/256))
			return linalg.WeightMat{}
		}
		wm, err := linalg.WrapQ4K(raw, rows, cols)
		if err != nil {
			r.fail(fmt.Sprintf("q4k weightMat %d×%d: %v", rows, cols, err))
			return linalg.WeightMat{}
		}
		return wm
	case giwKindQ5K, giwKindQ6K:
		r.alignArray()
		raw := r.rawAlias() // zero-copy: the super-blocks are aliased from the mapping
		wrap, name := linalg.WrapQ5K, "q5k"
		if kind == giwKindQ6K {
			wrap, name = linalg.WrapQ6K, "q6k"
		}
		wm, err := wrap(raw, rows, cols)
		if err != nil {
			r.fail(fmt.Sprintf("%s weightMat %d×%d: %v", name, rows, cols, err))
			return linalg.WeightMat{}
		}
		return wm
	default:
		r.unknownEncoding(kind)
		return linalg.WeightMat{}
	}
}

// mambaProj reads giwWriter.mambaProj's record: an f32 array (giwEncF32) or a weight matrix, by its encoding byte.
func (r *giwReader) mambaProj() ([]float32, linalg.WeightMat) {
	if r.need(1) && r.data[r.off] == giwEncF32 {
		return r.f32(), linalg.WeightMat{}
	}
	return nil, r.weightMat()
}

// fusedPending is a kind-6 member whose header has been read; its nibbles and scales are in the group's two
// blocks, which follow the group's last record.
type fusedPending struct {
	dst               *linalg.WeightMat
	rows, cols, group int
}

// fusedGroup reads what giwWriter.fusedGroup wrote: for each member either an ordinary weightMat record or a
// kind-6 header, then, if any member was kind 6, the shared nibble block and the shared binary16 scale block, each
// member's part aliased out of them in member order.
func (r *giwReader) fusedGroup(dst ...*linalg.WeightMat) {
	var pend []fusedPending
	for _, d := range dst {
		if r.err == nil && r.need(1) && r.data[r.off] == 6 {
			r.off++
			rows, cols, group := int(r.u32()), int(r.u32()), int(r.u32())
			if !r.need(1) {
				return
			}
			r.off++ // w8a8 flag: unused for int4
			const maxWeightDim = 1 << 26
			if rows <= 0 || cols <= 0 || rows > maxWeightDim || cols > maxWeightDim || group <= 0 {
				r.fail(fmt.Sprintf("fused int4 member implausible dims %d×%d group=%d", rows, cols, group))
				return
			}
			pend = append(pend, fusedPending{dst: d, rows: rows, cols: cols, group: group})
			continue
		}
		*d = r.weightMat()
	}
	if len(pend) == 0 || r.err != nil {
		return
	}
	// Each block starts 16-aligned (the writer pads with the position, not with a length prefix). A member's nibbles
	// are a multiple of 16 bytes (K%32==0), so every member's nibbles stay 16-aligned and adjacent.
	r.align16()
	nibbles := make([][]byte, len(pend))
	for i, p := range pend {
		if nibbles[i] = r.rawN(p.rows * ((p.cols + 1) / 2)); r.err != nil {
			return
		}
	}
	r.align16()
	for i, p := range pend {
		scales := r.u16View(p.rows * ((p.cols + p.group - 1) / p.group))
		if r.err != nil {
			return
		}
		*p.dst, _ = linalg.WrapInt4(nibbles[i], scales, p.rows, p.cols, p.group, linalg.LayoutCanonical)
	}
}

func (r *giwReader) layer(l *LayerWeights) {
	r.fusedGroup(&l.QProj, &l.KProj, &l.VProj)
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
	r.fusedGroup(&l.GateProj, &l.UpProj)
	l.DownProj = r.weightMat()
	l.UpBias = r.f32()
	l.DownBias = r.f32()
	l.PreMLPNorm = r.f32()
	l.PreMLPNormBias = r.f32()
	l.PostMLPNorm = r.f32()
	l.Router = r.weightMat()
	l.RouterBias = r.f32() // DeepSeek/GLM e_score_correction_bias
	ne := int(r.u32())
	if ne < 0 || ne > maxSerializedExperts {
		r.fail("implausible expert count")
		return
	}
	// The per-layer half of the same bound, multiplied by the layer count: the reader already holds the resolved arch, so a
	// blob claiming more experts than the family has is refused before the structs exist rather than after.
	if r.arch != nil && r.arch.MoE != nil && r.arch.MoE.NumExperts > 0 && ne > r.arch.MoE.NumExperts {
		r.fail(fmt.Sprintf("expert count: blob has %d, arch expects at most %d",
			ne, r.arch.MoE.NumExperts))
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
	if r.arch != nil && r.arch.gemma4 != nil {
		r.gemma4Layer(l)
	}
	r.v6Layer(l) // the completeness tail
	r.v8Layer(l) // the LFM2 short-conv tail
	r.v9Layer(l) // the KDA and MLA-gate tail
}

// v6Layer mirrors giwWriter.v6Layer.
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
		m.qAProj, m.qAQ = r.mambaProj()
		m.qALayernorm = r.f32()
		m.qBProj, m.qBQ = r.mambaProj()
		m.qProj, m.qQ = r.mambaProj()
		m.kvAProj, m.kvAQ = r.mambaProj()
		m.kvALayernorm = r.f32()
		m.kvBProj = r.f32()
		m.oProj, m.oQ = r.mambaProj()
		l.mla = m
	}
	if r.u8() != 0 {
		m := &mamba2Weights{}
		m.inProj, m.inQ = r.mambaProj()
		m.convW = r.f32()
		m.convB = r.f32()
		m.aLog = r.f32()
		m.d = r.f32()
		m.dtBias = r.f32()
		m.normW = r.f32()
		m.outProj, m.outQ = r.mambaProj()
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
	hidden, cd := 0, 0
	if r.arch != nil && r.arch.lfm2 != nil {
		hidden, cd = r.arch.HiddenDim, r.arch.lfm2.ConvDim
	}
	c.inProj = r.island(3*cd, hidden)
	c.convW = r.f32()
	c.outProj = r.island(hidden, cd)
	l.shortConv = c
}

// island reads giwWriter.island's record by its encoding byte: a kind-9 bf16 matrix aliased from the mapping, or an f32 array,
// which becomes a WeightMat of the config's shape [rows, cols] through islandMat (bf16 on the heap when bf16-exact). An f32
// array whose length does not fit that shape is kept as one row, for validateShapes to report.
func (r *giwReader) island(rows, cols int) linalg.WeightMat {
	if r.need(1) && r.data[r.off] != giwEncF32 {
		wm := r.weightMat()
		if r.err == nil && rows > 0 && (wm.Rows() != rows || wm.Cols() != cols) {
			r.fail(fmt.Sprintf("an island matrix is %d×%d, want %d×%d", wm.Rows(), wm.Cols(), rows, cols))
		}
		return wm
	}
	f := r.f32()
	if rows <= 0 || cols <= 0 || rows*cols != len(f) {
		return linalg.WrapF32(f, 1, len(f))
	}
	return islandMat(f, rows, cols)
}

// v9Layer mirrors giwWriter.v9Layer. gProj is always present in the byte stream (possibly
// zero-length) regardless of l.mla, since the writer runs unconditionally — read it first and
// only attach it when this layer actually has MLA weights (v6Layer, read earlier, already set
// l.mla by the time this runs). Version-gated the same way v8Layer is, so a pre-v9 bundle's layer
// block ends where it always did.
func (r *giwReader) v9Layer(l *LayerWeights) {
	gProj := r.f32()
	if l.mla != nil {
		l.mla.gProj = gProj
	}
	if r.u8() == 0 {
		return
	}
	k := &kdaWeights{}
	k.qProj = r.weightMat()
	k.kProj = r.weightMat()
	k.vProj = r.weightMat()
	k.qConvW = r.f32()
	k.kConvW = r.f32()
	k.vConvW = r.f32()
	k.fProj = r.weightMat()
	k.dtBias = r.f32()
	k.aLog = r.f32()
	k.bProj = r.f32()
	k.gProj = r.weightMat()
	k.oNormW = r.f32()
	k.oProj = r.weightMat()
	l.kda = k
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
		// Field order is the wire order: a struct literal evaluates its fields in source order, so these must be read in exactly
		// the order hybridLayer writes them.
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

// kquant56Raw returns m's super-blocks and .giw kind when m holds native Q5_K or Q6_K blocks.
func kquant56Raw(m *linalg.WeightMat) ([]byte, byte, bool) {
	if raw, ok := m.Q5K(); ok {
		return raw, giwKindQ5K, true
	}
	if raw, ok := m.Q6K(); ok {
		return raw, giwKindQ6K, true
	}
	return nil, 0, false
}
