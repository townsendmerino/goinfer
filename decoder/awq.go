package decoder

import (
	"fmt"

	"github.com/townsendmerino/aikit/embed"
)

// AWQ (safetensors-resident int4) — the AutoAWQ "GEMM" packing. Like GPTQ it
// ships a quantized linear as qweight/qzeros/scales, but with three differences:
//
//   - qweight is [in, out/8] int32 — 8 codes per word packed along the OUTPUT
//     dim (GPTQ packs along the input dim, [in/8, out]).
//   - there is no g_idx (AWQ does not use activation-order); groups run along the
//     input dim (scales/qzeros are [in/group, …]).
//   - within each int32 the 8 codes are interleaved by awqOrder, not sequential.
//
// Reconstruct: w[i,j] = (code - zero) * scale (asymmetric, the stored zero used
// directly — no GPTQ-style +1), then transpose to [out, in].

// awqOrder is AutoAWQ's GEMM nibble de-interleave (AWQ_REVERSE_ORDER): logical
// output channel c (within a packed group of 8) lives at nibble awqOrder[c] of
// the int32 — i.e. shifts [0,16,4,20,8,24,12,28] for c = 0..7.
var awqOrder = [8]uint{0, 4, 1, 5, 2, 6, 3, 7}

// awqReconstruct dequantizes one AWQ linear (named base) to a [out, in]
// row-major f32 matrix.
func awqReconstruct(st *embed.SafetensorsFile, base string, in, out int) ([]float32, error) {
	// 8 4-bit codes pack into each int32 along the output dim, so a
	// non-multiple-of-8 out would make the out/8 shape check pass via integer
	// division with an undersized qweight/qzeros, then the dequant loop would
	// index out of bounds. Reject up front.
	if in <= 0 || out <= 0 || out%8 != 0 {
		return nil, fmt.Errorf("awq %q: in=%d out=%d invalid (out must be a positive multiple of 8, in positive)", base, in, out)
	}
	qw, err := st.TensorI32(base + ".qweight") // [in, out/8]
	if err != nil {
		return nil, err
	}
	qz, err := st.TensorI32(base + ".qzeros") // [groups, out/8]
	if err != nil {
		return nil, err
	}
	sc, err := st.TensorF32(base + ".scales") // [groups, out]
	if err != nil {
		return nil, err
	}
	outP := out / 8
	if len(qw) != in*outP || len(sc)%out != 0 {
		return nil, fmt.Errorf("awq %q: bad shapes qweight=%d scales=%d (in=%d out=%d)", base, len(qw), len(sc), in, out)
	}
	nGroups := len(sc) / out
	if len(qz) != nGroups*outP {
		return nil, fmt.Errorf("awq %q: qzeros len %d != groups*out/8 %d", base, len(qz), nGroups*outP)
	}
	if nGroups == 0 || in%nGroups != 0 {
		return nil, fmt.Errorf("awq %q: in=%d not divisible by groups=%d", base, in, nGroups)
	}
	groupSize := in / nGroups

	// Precompute the per-output-channel nibble shift (depends only on j%8).
	var shift [8]uint
	for c := range 8 {
		shift[c] = 4 * awqOrder[c]
	}

	res := make([]float32, out*in)
	for i := range in {
		g := i / groupSize
		qwRow := i * outP
		qzRow := g * outP
		scRow := g * out
		for j := range out {
			s := shift[j&7]
			jw := j >> 3
			code := (qw[qwRow+jw] >> s) & 0xF
			zero := (qz[qzRow+jw] >> s) & 0xF
			res[j*in+i] = float32(code-zero) * sc[scRow+j]
		}
	}
	return res, nil
}
