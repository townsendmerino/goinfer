package quant

import (
	"fmt"
	"strings"

	"github.com/townsendmerino/aikit/embed"
)

// GPTQReconstruct dequantizes one GPTQ linear (named base, e.g.
// "model.layers.0.self_attn.q_proj") to a [out, in] row-major float32 matrix.
func GPTQReconstruct(st *embed.SafetensorsFile, base string, in, out int) ([]float32, error) {
	if in <= 0 || out <= 0 || in%8 != 0 || out%8 != 0 {
		return nil, fmt.Errorf("gptq %q: in=%d out=%d must be positive multiples of 8 (4-bit packing)", base, in, out)
	}
	qw, err := st.TensorI32(base + ".qweight")
	if err != nil {
		return nil, err
	}
	qz, err := st.TensorI32(base + ".qzeros")
	if err != nil {
		return nil, err
	}
	gidx, err := st.TensorI32(base + ".g_idx")
	if err != nil {
		return nil, err
	}
	sc, err := st.TensorF32(base + ".scales")
	if err != nil {
		return nil, err
	}
	if len(qw) != (in/8)*out || len(gidx) != in || len(sc)%out != 0 {
		return nil, fmt.Errorf("gptq %q: bad shapes qweight=%d gidx=%d scales=%d (in=%d out=%d)", base, len(qw), len(gidx), len(sc), in, out)
	}
	nGroups := len(sc) / out
	outP := out / 8
	if len(qz) != nGroups*outP {
		return nil, fmt.Errorf("gptq %q: qzeros len %d != groups*out/8 %d", base, len(qz), nGroups*outP)
	}

	res := make([]float32, out*in)
	for i := range in {
		g := int(gidx[i])
		if g < 0 || g >= nGroups {
			return nil, fmt.Errorf("gptq %q: g_idx[%d]=%d out of range [0,%d)", base, i, g, nGroups)
		}
		qwRow := (i / 8) * out
		shift := uint(4 * (i % 8))
		scRow := g * out
		qzRow := g * outP
		for j := range out {
			code := (qw[qwRow+j] >> shift) & 0xF
			zero := ((qz[qzRow+j/8] >> uint(4*(j%8))) & 0xF) + 1
			res[j*in+i] = float32(code-zero) * sc[scRow+j]
		}
	}
	return res, nil
}

// GPTQReconstructor implements Reconstructor for 4-bit GPTQ tensors.
type GPTQReconstructor struct{}

func (GPTQReconstructor) Method() string { return "gptq" }

func (GPTQReconstructor) ReconstructMat2D(st *embed.SafetensorsFile, name string, out, in int) ([]float32, error) {
	return GPTQReconstruct(st, strings.TrimSuffix(name, ".weight"), in, out)
}
