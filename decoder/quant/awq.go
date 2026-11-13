package quant

import (
	"fmt"
	"strings"

	"github.com/townsendmerino/aikit/embed"
)

var awqOrder = [8]uint{0, 4, 1, 5, 2, 6, 3, 7}

// AWQReconstruct dequantizes one AWQ linear (named base) to a [out, in]
// row-major float32 matrix.
func AWQReconstruct(st *embed.SafetensorsFile, base string, in, out int) ([]float32, error) {
	if in <= 0 || out <= 0 || out%8 != 0 {
		return nil, fmt.Errorf("awq %q: in=%d out=%d invalid (out must be a positive multiple of 8, in positive)", base, in, out)
	}
	qw, err := st.TensorI32(base + ".qweight")
	if err != nil {
		return nil, err
	}
	qz, err := st.TensorI32(base + ".qzeros")
	if err != nil {
		return nil, err
	}
	sc, err := st.TensorF32(base + ".scales")
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

// AWQReconstructor implements Reconstructor for 4-bit AWQ GEMM tensors.
type AWQReconstructor struct{}

func (AWQReconstructor) Method() string { return "awq" }

func (AWQReconstructor) ReconstructMat2D(st *embed.SafetensorsFile, name string, out, in int) ([]float32, error) {
	return AWQReconstruct(st, strings.TrimSuffix(name, ".weight"), in, out)
}
