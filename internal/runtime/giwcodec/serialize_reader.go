package giwcodec

import (
	"encoding/binary"
	"fmt"
	"math"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
)

// --- reader (cursor over data; big arrays aliased, floats copied) ---

type Reader struct {
	Data     []byte
	Off      int
	Err      error
	Arch     *darch.Architecture
	Scan     bool
	F32Kind1 bool
}

// OpensCopiedF32 reports whether a bundle's first weight record, the embedding, is a kind-1 matrix. A quant bundle's embedding is
// quantized (quantMode.embedding) and an f32 bundle's is f32, so this one byte tells a copied f32 body from an int8 or int4 bundle
// whose routers alone are kind 1, on every family (TestGIWCopiedF32_everyFamily), with nothing copied. Call it at the first record.
func (r *Reader) OpensCopiedF32() bool {
	return !r.Scan && !r.F32Kind1 && r.Off < len(r.Data) && r.Data[r.Off] == 1
}

func (r *Reader) Fail(msg string) {
	if r.Err == nil {
		r.Err = &SerializeError{msg}
	}
}

func (r *Reader) Need(n int) bool {
	if r.Err != nil {
		return false
	}
	if n < 0 || r.Off+n > len(r.Data) {
		r.Fail("unexpected end of data")
		return false
	}
	return true
}

func (r *Reader) U32() uint32 {
	if !r.Need(4) {
		return 0
	}
	v := binary.LittleEndian.Uint32(r.Data[r.Off:])
	r.Off += 4
	return v
}

func (r *Reader) RawN(n int) []byte {
	if !r.Need(n) {
		return nil
	}
	b := r.Data[r.Off : r.Off+n]
	r.Off += n
	return b
}

func (r *Reader) Str() string        { return string(r.RawN(int(r.U32()))) }
func (r *Reader) BytesField() []byte { return r.RawN(int(r.U32())) }

func (r *Reader) U8() byte {
	if !r.Need(1) {
		return 0
	}
	b := r.Data[r.Off]
	r.Off++
	return b
}

// Align16 skips the padding the writer put before a 16-aligned position (giwWriter.align16): the end of the header and
// the two blocks of a fused group.
func (r *Reader) Align16() {
	if r.Err != nil {
		return
	}
	if pad := int((-int64(r.Off)) & 15); r.Need(pad) {
		r.Off += pad
	}
}

func (r *Reader) AlignHead() { r.Align16() }

// AlignArray skips the zero padding the writer put before an aligned array's count (see ArrayPad).
func (r *Reader) AlignArray() {
	if r.Err != nil {
		return
	}
	if pad := ArrayPad(int64(r.Off)); r.Need(pad) {
		r.Off += pad
	}
}

// nativeLittleEndian reports whether float32s can be read straight out of the little-endian blob.
var nativeLittleEndian = func() bool { x := uint16(1); return *(*byte)(unsafe.Pointer(&x)) == 1 }()

// f32Alias reads a padded f32 array as an ALIAS of the mapping when its payload is 4-byte aligned
// in memory (always true for a blob inside a v3-bundle mapping), and copies otherwise, so an
// oddly-placed blob still loads, onto the heap. For weight-matrix scale arrays only: the mapping is read-only, and a
// scale array is read-only by construction (its sibling nibble/int8 arrays are already aliased).
// The caller must have called alignArray first.
func (r *Reader) f32Alias() []float32 {
	n := int(r.U32())
	if n == 0 || !r.Need(n*4) {
		return nil
	}
	if p := unsafe.Pointer(&r.Data[r.Off]); nativeLittleEndian && uintptr(p)%4 == 0 {
		r.Off += n * 4
		return unsafe.Slice((*float32)(p), n)
	}
	out := make([]float32, n)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(r.Data[r.Off:]))
		r.Off += 4
	}
	return out
}

// F32 reads a standalone f32 array (giwWriter.f32): its encoding byte, which must be EncF32, then the array.
func (r *Reader) F32() []float32 {
	if enc := r.U8(); enc != EncF32 {
		r.unknownEncoding(enc)
		return nil
	}
	return r.F32Body()
}

// unknownEncoding refuses a tensor whose encoding this build does not read. Only that encoding is refused: a newer writer
// adds an encoding without a version bump, and a bundle that never uses it still loads.
func (r *Reader) unknownEncoding(enc byte) {
	r.Fail(fmt.Sprintf("a tensor's encoding %d is not one this build reads (written by a newer goinfer; the sidecar is rebuilt)", enc))
}

// F32Body copies (the input isn't guaranteed aligned). len 0 ⇒ nil, preserving the
// "absent ⇒ nil" convention the forward pass checks for biases/norms.
func (r *Reader) F32Body() []float32 {
	n := int(r.U32())
	if n == 0 || !r.Need(n*4) {
		return nil
	}
	out := make([]float32, n)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(r.Data[r.Off:]))
		r.Off += 4
	}
	return out
}

// I8 ALIASES the int8 weight bytes over data (zero-copy). Read-only at inference.
func (r *Reader) I8() []int8 {
	n := int(r.U32())
	if n == 0 || !r.Need(n) {
		return nil
	}
	b := r.Data[r.Off : r.Off+n]
	r.Off += n
	return unsafe.Slice((*int8)(unsafe.Pointer(&b[0])), n)
}

// U16View returns n little-endian uint16s at the cursor as an ALIAS of the blob when 2-byte aligned on a little-endian
// host, else a copy: f32Alias's rule for f16 scales.
func (r *Reader) U16View(n int) []uint16 {
	if n == 0 || !r.Need(2*n) {
		return nil
	}
	if p := unsafe.Pointer(&r.Data[r.Off]); nativeLittleEndian && uintptr(p)%2 == 0 {
		r.Off += 2 * n
		return unsafe.Slice((*uint16)(p), n)
	}
	out := make([]uint16, n)
	for i := range out {
		out[i] = binary.LittleEndian.Uint16(r.Data[r.Off:])
		r.Off += 2
	}
	return out
}

// int4Scales reads an int4 scale array (kinds 3 and 5) as binary16, aliased from the mapping. The caller must have
// called alignArray first.
func (r *Reader) int4Scales() []uint16 { return r.U16View(int(r.U32())) }

// rawAlias ALIASES a []byte (int4 packed nibbles): a plain subslice of data.
func (r *Reader) rawAlias() []byte {
	n := int(r.U32())
	if n == 0 || !r.Need(n) {
		return nil
	}
	b := r.Data[r.Off : r.Off+n]
	r.Off += n
	return b
}

func (r *Reader) WeightMat() linalg.WeightMat {
	if !r.Need(1) {
		return linalg.WeightMat{}
	}
	kind := r.Data[r.Off]
	r.Off++
	if kind == 0 {
		return linalg.WeightMat{}
	}
	if kind != 1 && kind != KindF32 && kind != 2 && kind != 3 && kind != 5 && kind != 8 && kind != giwKindBF16 && kind != giwKindQ5K && kind != giwKindQ6K { // kind 6 is read as a fused group, never here
		r.unknownEncoding(kind)
		return linalg.WeightMat{}
	}
	rows, cols, group := int(r.U32()), int(r.U32()), int(r.U32())
	if !r.Need(1) {
		return linalg.WeightMat{}
	}
	w8a8 := r.Data[r.Off] == 1
	r.Off++
	// rows/cols/group are blob-controlled. linalg.Wrap{Int8,Int4} PANIC on a length mismatch or group<=0, and a wrong-length
	// WrapF32 defers the panic to first use, so a one-byte flip with a recomputed CRC would crash the loader. Validate dims
	// and array lengths here (the arrays were already bounded by r.need); a mismatch is a *SerializeError via r.fail, not a
	// panic. The maxWeightDim cap keeps rows*cols from overflowing int before the equality check.
	const maxWeightDim = 1 << 26
	if rows <= 0 || cols <= 0 || rows > maxWeightDim || cols > maxWeightDim {
		r.Fail(fmt.Sprintf("weightMat implausible dims %d×%d", rows, cols))
		return linalg.WeightMat{}
	}
	switch kind {
	case 1, KindF32:
		var f []float32
		if kind == 1 {
			if r.Scan {
				if n := int(r.U32()); n != rows*cols || !r.Need(n*4) {
					r.Fail(fmt.Sprintf("f32 weightMat %d×%d has %d values", rows, cols, n))
				} else {
					r.Off += n * 4
				}
				return linalg.WeightMat{}
			}
			f = r.F32Body()
		} else {
			r.AlignArray()
			f = r.f32Alias()
		}
		if len(f) != rows*cols {
			r.Fail(fmt.Sprintf("f32 weightMat %d×%d has %d values", rows, cols, len(f)))
			return linalg.WeightMat{}
		}
		return linalg.WrapF32(f, rows, cols)
	case 2:
		r.AlignArray()
		scales := r.f32Alias() // read order matches the writer (scales, then codes)
		r.AlignArray()
		q8 := r.I8()
		if len(q8) != rows*cols || len(scales) != rows {
			r.Fail(fmt.Sprintf("int8 weightMat %d×%d: q8=%d (want %d) scales=%d (want %d)", rows, cols, len(q8), rows*cols, len(scales), rows))
			return linalg.WeightMat{}
		}
		return linalg.WrapInt8(q8, scales, rows, cols, w8a8)
	case 3:
		r.AlignArray()
		q4s := r.int4Scales()
		r.AlignArray()
		q4 := r.rawAlias() // zero-copy alias into the mmap'd blob (WrapInt4 keeps it)
		if group <= 0 {
			r.Fail(fmt.Sprintf("int4 weightMat group %d ≤ 0", group))
			return linalg.WeightMat{}
		}
		wantQ4, wantScales := rows*((cols+1)/2), rows*((cols+group-1)/group)
		if len(q4) != wantQ4 || len(q4s) != wantScales {
			r.Fail(fmt.Sprintf("int4 weightMat %d×%d group=%d: q4=%d (want %d) q4s=%d (want %d)", rows, cols, group, len(q4), wantQ4, len(q4s), wantScales))
			return linalg.WeightMat{}
		}
		wm4, _ := linalg.WrapInt4(q4, q4s, rows, cols, group, linalg.LayoutCanonical)
		return wm4
	case 5:
		if group <= 0 {
			r.Fail(fmt.Sprintf("int4-row4-only weightMat group %d ≤ 0", group))
			return linalg.WeightMat{}
		}
		// The row4 arrays have kind 3's lengths: a repack, not a requant.
		wantQ4, wantScales := rows*((cols+1)/2), rows*((cols+group-1)/group)
		r.AlignArray()
		q4Row4Scales := r.int4Scales()
		r.AlignArray()
		q4Row4 := r.rawAlias() // zero-copy — no canonical bytes exist in this file at all
		if len(q4Row4) != wantQ4 || len(q4Row4Scales) != wantScales {
			r.Fail(fmt.Sprintf("int4-row4-only weightMat %d×%d group=%d: q4Row4=%d (want %d) q4Row4Scales=%d (want %d)", rows, cols, group, len(q4Row4), wantQ4, len(q4Row4Scales), wantScales))
			return linalg.WeightMat{}
		}
		wm, ok := linalg.WrapInt4(q4Row4, q4Row4Scales, rows, cols, group, linalg.LayoutRow4)
		if !ok {
			// Named and actionable: a kind-5 file is a promise to one target. ok=false here means Int4Row4Usable rejected
			// this core: wrong arch (the file was written for arm64), or a shape this core's build cannot run the row4
			// kernel for.
			r.Fail(fmt.Sprintf("int4 weightMat %d×%d group=%d is stored row4-only (kind 5, a cpu-arm64 prequant target) but this core cannot use that layout — rebuild with `go run ./cmd/prequant -target <this core>` (or delete the stream-weights cache so it rebuilds automatically)", rows, cols, group))
			return linalg.WeightMat{}
		}
		return wm
	case giwKindBF16:
		r.AlignArray()
		u := r.U16View(int(r.U32())) // zero-copy: aliased from the mapping
		if len(u) != rows*cols {
			r.Fail(fmt.Sprintf("bf16 weightMat %d×%d has %d values", rows, cols, len(u)))
			return linalg.WeightMat{}
		}
		return linalg.WrapBF16(u, rows, cols)
	case 8:
		r.AlignArray()
		raw := r.rawAlias() // zero-copy: the super-blocks are aliased from the mapping
		if cols%256 != 0 || len(raw) != rows*linalg.Q4KRowBytes(cols) {
			r.Fail(fmt.Sprintf("q4k weightMat %d×%d: %d bytes, want %d whole 256-value super-blocks per row", rows, cols, len(raw), cols/256))
			return linalg.WeightMat{}
		}
		wm, err := linalg.WrapQ4K(raw, rows, cols)
		if err != nil {
			r.Fail(fmt.Sprintf("q4k weightMat %d×%d: %v", rows, cols, err))
			return linalg.WeightMat{}
		}
		return wm
	case giwKindQ5K, giwKindQ6K:
		r.AlignArray()
		raw := r.rawAlias() // zero-copy: the super-blocks are aliased from the mapping
		wrap, name := linalg.WrapQ5K, "q5k"
		if kind == giwKindQ6K {
			wrap, name = linalg.WrapQ6K, "q6k"
		}
		wm, err := wrap(raw, rows, cols)
		if err != nil {
			r.Fail(fmt.Sprintf("%s weightMat %d×%d: %v", name, rows, cols, err))
			return linalg.WeightMat{}
		}
		return wm
	default:
		r.unknownEncoding(kind)
		return linalg.WeightMat{}
	}
}

// mambaProj reads giwWriter.mambaProj's record: an f32 array (EncF32) or a weight matrix, by its encoding byte.
func (r *Reader) mambaProj() ([]float32, linalg.WeightMat) {
	if r.Need(1) && r.Data[r.Off] == EncF32 {
		return r.F32(), linalg.WeightMat{}
	}
	return nil, r.WeightMat()
}

// fusedPending is a kind-6 member whose header has been read; its nibbles and scales are in the group's two
// blocks, which follow the group's last record.
type fusedPending struct {
	dst               *linalg.WeightMat
	rows, cols, group int
}

// FusedGroup reads what giwWriter.fusedGroup wrote: for each member either an ordinary weightMat record or a
// kind-6 header, then, if any member was kind 6, the shared nibble block and the shared binary16 scale block, each
// member's part aliased out of them in member order.
func (r *Reader) FusedGroup(dst ...*linalg.WeightMat) {
	var pend []fusedPending
	for _, d := range dst {
		if r.Err == nil && r.Need(1) && r.Data[r.Off] == 6 {
			r.Off++
			rows, cols, group := int(r.U32()), int(r.U32()), int(r.U32())
			if !r.Need(1) {
				return
			}
			r.Off++ // w8a8 flag: unused for int4
			const maxWeightDim = 1 << 26
			if rows <= 0 || cols <= 0 || rows > maxWeightDim || cols > maxWeightDim || group <= 0 {
				r.Fail(fmt.Sprintf("fused int4 member implausible dims %d×%d group=%d", rows, cols, group))
				return
			}
			pend = append(pend, fusedPending{dst: d, rows: rows, cols: cols, group: group})
			continue
		}
		*d = r.WeightMat()
	}
	if len(pend) == 0 || r.Err != nil {
		return
	}
	// Each block starts 16-aligned (the writer pads with the position, not with a length prefix). A member's nibbles
	// are a multiple of 16 bytes (K%32==0), so every member's nibbles stay 16-aligned and adjacent.
	r.Align16()
	nibbles := make([][]byte, len(pend))
	for i, p := range pend {
		if nibbles[i] = r.RawN(p.rows * ((p.cols + 1) / 2)); r.Err != nil {
			return
		}
	}
	r.Align16()
	for i, p := range pend {
		scales := r.U16View(p.rows * ((p.cols + p.group - 1) / p.group))
		if r.Err != nil {
			return
		}
		*p.dst, _ = linalg.WrapInt4(nibbles[i], scales, p.rows, p.cols, p.group, linalg.LayoutCanonical)
	}
}

func (r *Reader) Layer(l *darch.LayerWeights) {
	r.FusedGroup(&l.QProj, &l.KProj, &l.VProj)
	l.OProj = r.WeightMat()
	l.QBias = r.F32()
	l.KBias = r.F32()
	l.VBias = r.F32()
	l.OBias = r.F32()
	l.QNorm = r.F32()
	l.KNorm = r.F32()
	l.PreAttnNorm = r.F32()
	l.PreAttnNormBias = r.F32()
	l.PostAttnNorm = r.F32()
	r.FusedGroup(&l.GateProj, &l.UpProj)
	l.DownProj = r.WeightMat()
	l.UpBias = r.F32()
	l.DownBias = r.F32()
	l.PreMLPNorm = r.F32()
	l.PreMLPNormBias = r.F32()
	l.PostMLPNorm = r.F32()
	l.Router = r.WeightMat()
	l.RouterBias = r.F32() // DeepSeek/GLM e_score_correction_bias
	ne := int(r.U32())
	if ne < 0 || ne > maxSerializedExperts {
		r.Fail("implausible expert count")
		return
	}
	// The per-layer half of the same bound, multiplied by the layer count: the reader already holds the resolved arch, so a
	// blob claiming more experts than the family has is refused before the structs exist rather than after.
	if r.Arch != nil && r.Arch.MoE != nil && r.Arch.MoE.NumExperts > 0 && ne > r.Arch.MoE.NumExperts {
		r.Fail(fmt.Sprintf("expert count: blob has %d, arch expects at most %d",
			ne, r.Arch.MoE.NumExperts))
		return
	}
	if ne > 0 {
		l.Experts = make([]archparts.ExpertWeights, ne)
		for e := range l.Experts {
			l.Experts[e].Gate = r.WeightMat()
			l.Experts[e].Up = r.WeightMat()
			l.Experts[e].Down = r.WeightMat()
		}
	}
	l.SharedExpert.Gate = r.WeightMat()
	l.SharedExpert.Up = r.WeightMat()
	l.SharedExpert.Down = r.WeightMat()
	l.SharedGate = r.WeightMat()
	r.hybridLayer(l)
	if r.Arch != nil && r.Arch.Gemma4 != nil {
		r.gemma4Layer(l)
	}
	r.v6Layer(l) // the completeness tail
	r.v8Layer(l) // the LFM2 short-conv tail
	r.v9Layer(l) // the KDA and MLA-gate tail
}

// v6Layer mirrors giwWriter.v6Layer.
func (r *Reader) v6Layer(l *darch.LayerWeights) {
	l.GProj = r.WeightMat()
	l.AttnSinks = r.F32()
	for e := range l.Experts {
		l.Experts[e].GateBias = r.F32()
		l.Experts[e].UpBias = r.F32()
		l.Experts[e].DownBias = r.F32()
	}
	l.SharedExpert.GateBias = r.F32()
	l.SharedExpert.UpBias = r.F32()
	l.SharedExpert.DownBias = r.F32()
	if r.U8() != 0 {
		m := &archparts.MLAWeights{}
		m.QAProj, m.QAQ = r.mambaProj()
		m.QALayernorm = r.F32()
		m.QBProj, m.QBQ = r.mambaProj()
		m.QProj, m.QQ = r.mambaProj()
		m.KVAProj, m.KVAQ = r.mambaProj()
		m.KVALayernorm = r.F32()
		m.KVBProj = r.F32()
		m.OProj, m.OQ = r.mambaProj()
		l.MLA = m
	}
	if r.U8() != 0 {
		m := &archparts.Mamba2Weights{}
		m.InProj, m.InQ = r.mambaProj()
		m.ConvW = r.F32()
		m.ConvB = r.F32()
		m.ALog = r.F32()
		m.D = r.F32()
		m.DtBias = r.F32()
		m.NormW = r.F32()
		m.OutProj, m.OutQ = r.mambaProj()
		l.Mamba = m
	}
}

// v8Layer mirrors giwWriter.v8Layer. Version-gated, so a v3-v7 bundle's layer block ends where it
// always did — and an LFM2 bundle written at v7 or earlier is one whose conv weights were never in
// the file at all, which validateShapes rejects rather than loading into a nil-deref.
func (r *Reader) v8Layer(l *darch.LayerWeights) {
	if r.U8() == 0 {
		return
	}
	c := &archparts.ShortConvWeights{}
	hidden, cd := 0, 0
	if r.Arch != nil && r.Arch.LFM2 != nil {
		hidden, cd = r.Arch.HiddenDim, r.Arch.LFM2.ConvDim
	}
	c.InProj = r.island(3*cd, hidden)
	c.ConvW = r.F32()
	c.OutProj = r.island(hidden, cd)
	l.ShortConv = c
}

// island reads giwWriter.island's record by its encoding byte: a kind-9 bf16 matrix aliased from the mapping, or an f32 array,
// which becomes a WeightMat of the config's shape [rows, cols] through IslandMat (bf16 on the heap when bf16-exact). An f32
// array whose length does not fit that shape is kept as one row, for validateShapes to report.
func (r *Reader) island(rows, cols int) linalg.WeightMat {
	if r.Need(1) && r.Data[r.Off] != EncF32 {
		wm := r.WeightMat()
		if r.Err == nil && rows > 0 && (wm.Rows() != rows || wm.Cols() != cols) {
			r.Fail(fmt.Sprintf("an island matrix is %d×%d, want %d×%d", wm.Rows(), wm.Cols(), rows, cols))
		}
		return wm
	}
	f := r.F32()
	if rows <= 0 || cols <= 0 || rows*cols != len(f) {
		return linalg.WrapF32(f, 1, len(f))
	}
	return IslandMat(f, rows, cols)
}

// v9Layer mirrors giwWriter.v9Layer. gProj is always present in the byte stream (possibly
// zero-length) regardless of l.mla, since the writer runs unconditionally — read it first and
// only attach it when this layer actually has MLA weights (v6Layer, read earlier, already set
// l.mla by the time this runs). Version-gated the same way v8Layer is, so a pre-v9 bundle's layer
// block ends where it always did.
func (r *Reader) v9Layer(l *darch.LayerWeights) {
	gProj := r.F32()
	if l.MLA != nil {
		l.MLA.GProj = gProj
	}
	if r.U8() == 0 {
		return
	}
	k := &archparts.KDAWeights{}
	k.QProj = r.WeightMat()
	k.KProj = r.WeightMat()
	k.VProj = r.WeightMat()
	k.QConvW = r.F32()
	k.KConvW = r.F32()
	k.VConvW = r.F32()
	k.FProj = r.WeightMat()
	k.DtBias = r.F32()
	k.ALog = r.F32()
	k.BProj = r.F32()
	k.GProj = r.WeightMat()
	k.ONormW = r.F32()
	k.OProj = r.WeightMat()
	l.KDA = k
}

// gemma4Layer reads the v4 Gemma 4 per-layer tail and, when the MoE sub-block is
// present, reconstructs gemma4moe — re-aliasing the dense-branch MLP + pre/post norms
// already read into the standard block, and taking the fixed dims from the resolved
// arch (denseInter = IntermediateDim, moe/experts/topK from arch.MoE).
func (r *Reader) gemma4Layer(l *darch.LayerWeights) {
	l.PLEGate = r.WeightMat()
	l.PLEProj = r.WeightMat()
	l.PostPLENorm = r.F32()
	l.LayerScalar = math.Float32frombits(r.U32())
	l.KVShared = r.U8() != 0
	l.VFromK = r.U8() != 0
	if r.U8() == 0 { // no gemma4moe sub-block (dense E-model layer)
		return
	}
	mo := &archparts.Gemma4MoEWeights{
		PreFFNNorm:  l.PreMLPNorm,  // dense pre-norm (already read)
		PostFFNNorm: l.PostMLPNorm, // joint post-norm (already read)
		MlpGate:     l.GateProj,
		MlpUp:       l.UpProj,
		MlpDown:     l.DownProj,
		LayerScalar: l.LayerScalar,
		DenseInter:  r.Arch.IntermediateDim,
	}
	if m := r.Arch.MoE; m != nil {
		mo.MoeInter, mo.NE, mo.TopK = m.IntermediateDim, m.NumExperts, m.TopK
	}
	mo.PostFFNNorm1 = r.F32()
	mo.PreFFNNorm2 = r.F32()
	mo.PostFFNNorm2 = r.F32()
	mo.RouterProj = r.WeightMat()
	mo.RouterScale = r.F32()
	mo.PerExpertScale = r.F32()
	ne := int(r.U32())
	if ne < 0 || ne > maxSerializedExperts {
		r.Fail("implausible gemma4moe expert count")
		return
	}
	mo.ExpertsGateUp = make([]linalg.WeightMat, ne)
	mo.ExpertsDown = make([]linalg.WeightMat, ne)
	for e := range ne {
		mo.ExpertsGateUp[e] = r.WeightMat()
		mo.ExpertsDown[e] = r.WeightMat()
	}
	l.Gemma4MoE = mo
}

// hybridLayer reconstructs the qwen3_5_moe per-layer extras written by
// giwWriter.hybridLayer. negExpA was stored precomputed, so it loads straight in.
func (r *Reader) hybridLayer(l *darch.LayerWeights) {
	switch r.U8() {
	case 0:
		// no hybrid extras (every non-qwen3_5_moe family)
	case 1:
		// Field order is the wire order: a struct literal evaluates its fields in source order, so these must be read in exactly
		// the order hybridLayer writes them.
		d := &archparts.DeltaNetWeights{}
		d.InProjQKV = r.WeightMat()
		d.InProjZ = r.WeightMat()
		d.InProjB = r.F32()
		d.InProjA = r.F32()
		d.ConvW = r.F32()
		d.DtBias = r.F32()
		d.NegExpA = r.F32()
		d.NormW = r.F32()
		d.OutProj = r.WeightMat()
		l.Delta = d
	case 2:
		a := &archparts.QwenAttnWeights{}
		a.QProj = r.WeightMat()
		a.KProj = r.WeightMat()
		a.VProj = r.WeightMat()
		a.OProj = r.WeightMat()
		a.QNorm = r.F32()
		a.KNorm = r.F32()
		l.QAttn = a
	default:
		r.Fail("unknown per-layer hybrid kind")
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
