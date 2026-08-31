package decoder

import (
	"encoding/json"
	"fmt"

	"github.com/townsendmerino/aikit/embed"
)

// GPTQ (safetensors-resident int4) — the HF/AutoGPTQ packing that ships a
// quantized linear as four tensors instead of one f32 .weight: qweight (packed
// 4-bit, [in/8, out] int32, 8 values per word along the input dim), qzeros
// (packed 4-bit zero-points, [groups, out/8]), scales ([groups, out] f16), and
// g_idx ([in] int32, the per-input group index — a permutation under act-order).
// We reconstruct each linear to f32 once at load: w[i,j] = (code - (zero+1)) *
// scale, picking the group via g_idx[i], then transpose to the [out, in] layout
// the rest of the decoder uses. The reconstructed weight then streams through
// the same int8/int4 re-quant path as any other (so a GPTQ checkpoint can run
// resident-int4 too). Only the projections are GPTQ; embeddings, norms, and the
// LM head ship in bf16/f16 and load unchanged.

// quantConfig is the resolved quantization_config for a pre-quantized
// safetensors checkpoint (GPTQ or AWQ — both 4-bit group-quant int4, differing
// only in how the codes are packed; see gptqReconstruct / awqReconstruct).
type quantConfig struct {
	method    string // "gptq" | "awq" | "fp8"
	bits      int
	groupSize int
	descAct   bool // gptq act-order
	sym       bool // gptq symmetric
	// fp8 only: the 2-D weight block a single scale covers (config.json's
	// weight_block_size, [128,128] for DeepSeek V3/V4 and Qwen3-FP8). Unlike gptq/awq's
	// 1-D groupSize this needs both dims, because the scale grid is
	// [ceil(rows/blockR), ceil(cols/blockC)] rather than a per-row run.
	blockR, blockC int
}

// parseQuantConfig reads config.json's quantization_config. Returns nil for a
// full-precision checkpoint (absent/null). Only 4-bit GPTQ/AWQ are supported;
// other methods/bit-widths error.
func parseQuantConfig(raw json.RawMessage) (*quantConfig, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var obj struct {
		QuantMethod string `json:"quant_method"`
		Bits        int    `json:"bits"`
		GroupSize   int    `json:"group_size"`
		DescAct     bool   `json:"desc_act"`
		Sym         *bool  `json:"sym"`
		Fmt         string `json:"fmt"`               // fp8: "e4m3" | "e5m2"
		WeightBlock []int  `json:"weight_block_size"` // fp8: [blockR, blockC]
	}
	if err := json.Unmarshal(raw, &obj); err != nil {
		return nil, fmt.Errorf("quantization_config: %w", err)
	}
	// fp8 returns EARLY: it shares none of the gptq/awq shape below. It carries no `bits`
	// and no 1-D `group_size` (the checks beneath would reject it on both), and its format
	// is a dtype in the safetensors header rather than a packing this code unpacks.
	if obj.QuantMethod == "fp8" {
		if obj.Fmt != "" && obj.Fmt != "e4m3" {
			// e5m2 is a different exponent/mantissa split, so the decode table in fp8.go
			// is simply wrong for it. Refuse rather than produce plausible numbers.
			return nil, fmt.Errorf("quantization_config(fp8): fmt %q unsupported (have: e4m3)", obj.Fmt)
		}
		if len(obj.WeightBlock) != 2 || obj.WeightBlock[0] <= 0 || obj.WeightBlock[1] <= 0 {
			// A per-tensor or per-channel fp8 checkpoint (compressed-tensors style) has no
			// weight_block_size. It is a REAL format, just not this one, and loading it with
			// block arithmetic would misread every scale — so say which one is missing.
			return nil, fmt.Errorf("quantization_config(fp8): weight_block_size %v unsupported "+
				"(need a 2-element block, e.g. [128,128]); per-tensor/per-channel fp8 is a different layout", obj.WeightBlock)
		}
		return &quantConfig{method: "fp8", blockR: obj.WeightBlock[0], blockC: obj.WeightBlock[1]}, nil
	}
	if obj.QuantMethod != "gptq" && obj.QuantMethod != "awq" {
		return nil, fmt.Errorf("quantization_config: method %q unsupported (have: gptq, awq, fp8)", obj.QuantMethod)
	}
	if obj.Bits != 4 {
		return nil, fmt.Errorf("quantization_config(%s): %d-bit unsupported (have: 4-bit)", obj.QuantMethod, obj.Bits)
	}
	if obj.GroupSize <= 0 {
		return nil, fmt.Errorf("quantization_config(%s): group_size %d unsupported (need a positive group)", obj.QuantMethod, obj.GroupSize)
	}
	sym := true
	if obj.Sym != nil {
		sym = *obj.Sym
	}
	return &quantConfig{method: obj.QuantMethod, bits: obj.Bits, groupSize: obj.GroupSize, descAct: obj.DescAct, sym: sym}, nil
}

// gptqReconstruct dequantizes one GPTQ linear (named base, e.g.
// "model.layers.0.self_attn.q_proj") to a [out, in] row-major f32 matrix — the
// transpose of the [in, out] reconstruction, matching nn.Linear's [out, in].
func gptqReconstruct(st *embed.SafetensorsFile, base string, in, out int) ([]float32, error) {
	// 8 4-bit codes pack into each int32 along BOTH dims (qweight packs the input
	// dim, qzeros the output dim), so non-multiple-of-8 in/out would make the
	// (in/8)*out and out/8 shape checks pass via integer division with undersized
	// tensors, then the dequant loop would index out of bounds. Reject up front.
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
	// Shape checks (8 4-bit codes per int32 word; one group per group_size rows).
	if len(qw) != (in/8)*out || len(gidx) != in || len(sc)%out != 0 {
		return nil, fmt.Errorf("gptq %q: bad shapes qweight=%d gidx=%d scales=%d (in=%d out=%d)", base, len(qw), len(gidx), len(sc), in, out)
	}
	nGroups := len(sc) / out
	outP := out / 8 // packed-output width of qzeros
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
