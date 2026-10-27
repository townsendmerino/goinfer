// Package wpack packs host weights into the layouts the CUDA kernels read: int8 rows as little-endian words, int4 nibbles
// permuted for the forward GEMV with f16 group scales, Q4_K blocks as words, and f32 or bf16 quantized to int8 per row.
// It imports aikit's linalg and nothing from goinfer, so kernel tests can pack weights without reaching decoder.
//
// It does not decide whether a weight is admitted: package cuda's planWeight does, ahead of every Pack. Pack itself
// refuses a kind it has no layout for. The bytes are pinned by cuda's TestWeightPackers_bitIdentity.
package wpack

import (
	"fmt"
	"math"

	"github.com/townsendmerino/aikit/linalg"
)

// Packed is one packed weight: Kind is the device layout ("int4", "int8" or "q4k"; f32 and bf16 pack as "int8"), Wpk
// its words, Ws the int8 row scales, Ws16 the int4 group scales as f16, and N x K the weight's shape.
type Packed struct {
	Kind string
	Wpk  []uint32
	Ws   []float32
	Ws16 []uint16
	N, K int
}

// Pack packs w by its kind. The caller has admitted w (cuda's planWeight checks K and the int4 and q4k layouts).
func Pack(w *linalg.WeightMat) (Packed, error) {
	N, K := w.Rows(), w.Cols()
	switch w.Kind() {
	case "int4":
		q4, s16, _, _ := w.Int4F16()
		var sc []float32
		if s16 != nil {
			sc = make([]float32, len(s16))
			linalg.F16ToF32Slice(sc, s16)
		}
		wpk := make([]uint32, N*(K/8))
		for i := range wpk {
			b := q4[i*4 : i*4+4]
			wpk[i] = Permute(uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24)
		}
		gs := make([]uint16, len(sc))
		for i, v := range sc {
			gs[i] = F32ToF16(v)
		}
		return Packed{Kind: "int4", Wpk: wpk, Ws16: gs, N: N, K: K}, nil
	case "int8":
		q8, sc, _, _ := w.Int8()
		return Packed{Kind: "int8", Wpk: PackI8(q8, N, K), Ws: sc, N: N, K: K}, nil
	case "q4k":
		raw, _ := w.Q4K()
		wpk := make([]uint32, len(raw)/4)
		for i := range wpk {
			wpk[i] = uint32(raw[4*i]) | uint32(raw[4*i+1])<<8 | uint32(raw[4*i+2])<<16 | uint32(raw[4*i+3])<<24
		}
		return Packed{Kind: "q4k", Wpk: wpk, N: N, K: K}, nil
	case "f32", "bf16":
		f32, ok := w.F32()
		if !ok { // bf16: widened row by row, so the bytes are the f32 weight's
			f32 = make([]float32, N*K)
			for i := range N {
				w.Row(i, f32[i*K:(i+1)*K])
			}
		}
		q8, sc := linalg.QuantizeRowsInt8(f32, N, K)
		return Packed{Kind: "int8", Wpk: PackI8(q8, N, K), Ws: sc, N: N, K: K}, nil
	default:
		return Packed{}, fmt.Errorf("cuda: unsupported projection kind %q", w.Kind())
	}
}

// PackStack packs ws as one stack, rows concatenated in order, for kernels that index experts by row range. admit, when
// not nil, is called on each member before it is packed, and its error ends the stack. Every member must share the
// first's kind and K: the kernel strides by one Kwords and unpacks one way.
func PackStack(admit func(*linalg.WeightMat) error, ws ...*linalg.WeightMat) (Packed, error) {
	if len(ws) == 0 {
		return Packed{}, fmt.Errorf("cuda: packWeightStack needs at least one weight")
	}
	var out Packed
	for i, w := range ws {
		var h Packed
		err := error(nil)
		if admit != nil {
			err = admit(w)
		}
		if err == nil {
			h, err = Pack(w)
		}
		if err != nil {
			return Packed{}, fmt.Errorf("cuda: packWeightStack[%d]: %w", i, err)
		}
		if i == 0 {
			out.Kind, out.K = h.Kind, h.K
			// Reserve the whole stack up front: without it the appends below regrow the slice geometrically and leave every
			// outgrown copy as garbage (docs/code-notes/cuda.md#packWeightStack.reserve).
			if h.N > 0 {
				totalN := 0
				for _, x := range ws {
					totalN += x.Rows()
				}
				if n := len(h.Wpk); n > 0 {
					out.Wpk = make([]uint32, 0, n/h.N*totalN)
				}
				if n := len(h.Ws); n > 0 {
					out.Ws = make([]float32, 0, n/h.N*totalN)
				}
				if n := len(h.Ws16); n > 0 {
					out.Ws16 = make([]uint16, 0, n/h.N*totalN)
				}
			}
		}
		if h.Kind != out.Kind {
			return Packed{}, fmt.Errorf("cuda: packWeightStack[%d]: kind %q != %q — a mixed-precision "+
				"stack cannot share one kernel's unpack path", i, h.Kind, out.Kind)
		}
		if h.K != out.K {
			return Packed{}, fmt.Errorf("cuda: packWeightStack[%d]: K=%d != %d — the kernel strides by a "+
				"single Kwords, so a ragged stack reads across row boundaries", i, h.K, out.K)
		}
		out.Wpk = append(out.Wpk, h.Wpk...)
		out.Ws = append(out.Ws, h.Ws...)
		out.Ws16 = append(out.Ws16, h.Ws16...)
		out.N += h.N
	}
	return out, nil
}

// PackI8 packs N x K int8 values four to a little-endian word.
func PackI8(q8 []int8, N, K int) []uint32 {
	p := make([]uint32, N*(K/4))
	for i := range p {
		p[i] = uint32(uint8(q8[i*4])) | uint32(uint8(q8[i*4+1]))<<8 | uint32(uint8(q8[i*4+2]))<<16 | uint32(uint8(q8[i*4+3]))<<24
	}
	return p
}

// PackI8Into is PackI8 into caller-owned storage: dst[i] takes q8[4i:4i+4]. It allocates nothing.
func PackI8Into(dst []uint32, q8 []int8) {
	for i := range dst {
		dst[i] = uint32(uint8(q8[i*4])) | uint32(uint8(q8[i*4+1]))<<8 | uint32(uint8(q8[i*4+2]))<<16 | uint32(uint8(q8[i*4+3]))<<24
	}
}

// PackInt4Into is Pack's int4 layout written into caller-owned storage: the nibble-permuted words, and the group scales
// through the same f16 -> f32 -> f16 trip. It allocates nothing.
func PackInt4Into(dstW []uint32, dstS []uint16, q4 []byte, s16 []uint16) {
	for i := range dstW {
		b := q4[i*4 : i*4+4]
		dstW[i] = Permute(uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24)
	}
	for i, h := range s16 {
		dstS[i] = F32ToF16(linalg.F16ToF32(h))
	}
}

// NibblePos maps a weight's index within an 8-weight word (0..7) to its nibble slot, so the coalesced GEMV's even/odd
// byte split (word&0x0F0F0F0F / (word>>4)&0x0F0F0F0F) lands weights 0..3 in the low-nibble bytes and 4..7 in the high.
func NibblePos(i int) int {
	if i < 4 {
		return 2 * i
	}
	return 2*(i-4) + 1
}

// Permute converts a natural-order packed word (element i at nibble i, the straight byte copy of the decoder's int4) into
// the nibble-permuted layout the coalesced forward GEMV expects (element i at nibble NibblePos(i)).
func Permute(w uint32) uint32 {
	var o uint32
	for i := range 8 {
		nv := (w >> (4 * i)) & 0xf
		o |= nv << (4 * NibblePos(i))
	}
	return o
}

// F32ToF16 encodes a float32 as an f16 bit pattern, byte for byte identical to decoder's f32ToF16bits, the canonical
// resident-backend f16 scale representation that metal/pack.go, aikit/linalg and the CPU's f16-scale diagnostic replicate.
// The int4 group scales it encodes must match every other backend's f16 scales bit for bit. It rounds half up with
// gradual underflow to subnormals; it is not RNE or saturating, which would reintroduce a divergence.
func F32ToF16(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16((b >> 16) & 0x8000)
	e := int32((b>>23)&0xFF) - 112 // f16-biased exponent = (exp-127) + 15
	m := b & 0x7FFFFF
	switch {
	case (b>>23)&0xFF == 0xFF: // Inf / NaN
		if m != 0 {
			return sign | 0x7E00
		}
		return sign | 0x7C00
	case e >= 0x1F: // overflow → Inf
		return sign | 0x7C00
	case e <= 0: // subnormal or underflow to zero
		if e < -10 {
			return sign
		}
		m |= 0x800000 // restore the implicit leading 1
		sh := uint32(14 - e)
		return sign | uint16((m+(1<<(sh-1)))>>sh) // round-half-up
	default: // normal f16
		half := sign | uint16(e<<10) | uint16(m>>13)
		if m&0x1000 != 0 { // guard bit set → round up
			half++
		}
		return half
	}
}
