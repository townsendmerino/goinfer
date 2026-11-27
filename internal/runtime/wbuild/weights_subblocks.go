package wbuild

import (
	"fmt"
	"github.com/townsendmerino/goinfer/decoder/config"
	"slices"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
	"github.com/townsendmerino/goinfer/internal/runtime/hybridattn"
)

// loadQwen35Attn loads one qwen3_5_moe layer's attention tensors as f32 (the parity-first forward uses plain matvec): the
// Gated DeltaNet set on linear layers (linear_attn.*), the gated-softmax set on the rest (self_attn.*, with a
// double-width q_proj: query || gate per head). The MoE FFN is loaded by the shared path. See docs/qwen3_5_moe.md.
//
// mkQ builds a quantized-if-requested WeightMat, so this loader honours Options.Quant like every other family; it is
// passed in because the quant resolution lives in buildWeights. mkQ and f32 are the loader's merge-aware readers
// (loadMatMerged, loadF32Merged): every tensor here is read through one of them, so a LoRA delta on any projection is
// merged on the on-disk tensor, before qwen3_next's fused split or Olmo's q/k/v concatenation rearranges it.
func loadQwen35Attn(i int, l *darch.LayerWeights, arch *darch.Architecture, hidden int,
	tn func(int, string) string, mkQ func(string, int, int) (linalg.WeightMat, error),
	f32 func(string, ...int) ([]float32, error), quant QuantMode, needCanonical, skipRow4 bool, h *Hooks) error {
	g := arch.Qwen35
	var err error
	nm := func(suf string) string { return tn(i, suf) }
	// qw is quantizeWM, or quantizeWMSkipRow4 when skipRow4, for the fused-tensor split cases below, which build their
	// WeightMat directly rather than through mkQ.
	qw := func(m linalg.WeightMat) linalg.WeightMat {
		return h.QuantizeLayerWM(m, quant, needCanonical, skipRow4)
	}
	if arch.IsLinearLayer(i) {
		keyDim, valueDim := g.KeyHeadDim*g.NumKeyHeads, g.ValueHeadDim*g.NumValueHeads
		convDim := 2*keyDim + valueDim
		d := &archparts.DeltaNetWeights{}
		if g.FusedDeltaNetProj {
			qkvzRows := 2*keyDim + 2*valueDim
			qkvz, e := f32(nm("linear_attn.in_proj_qkvz.weight"), qkvzRows, hidden)
			if e != nil {
				return e
			}
			ba, e := f32(nm("linear_attn.in_proj_ba.weight"), 2*g.NumValueHeads, hidden)
			if e != nil {
				return e
			}
			qkvF, zF := splitQwen3NextQKVZ(qkvz, g, hidden)
			d.InProjQKV = qw(linalg.WrapF32(qkvF, convDim, hidden))
			d.InProjZ = qw(linalg.WrapF32(zF, valueDim, hidden))
			d.InProjB, d.InProjA = splitQwen3NextBA(ba, g, hidden)
		} else if g.SeparateQKVProj {
			// Olmo Hybrid: q_proj/k_proj/v_proj are three fully independent tensors —
			// more unfused than qwen3.5's own pre-concatenated in_proj_qkv. Each
			// nn.Linear's weight is already [out_features, hidden] with out_features
			// in head-major order (view(...,-1,head_dim) is a reshape, not a
			// reorder), so vertically stacking the three f32 matrices in q,k,v order
			// reproduces the SAME flat [convDim, hidden] layout in_proj_qkv already
			// has — a plain row concat, no per-group interleaving like
			// splitQwen3NextQKVZ needs for the head-grouped fused case.
			qf, e := f32(nm("linear_attn.q_proj.weight"), keyDim, hidden)
			if e != nil {
				return e
			}
			kf, e := f32(nm("linear_attn.k_proj.weight"), keyDim, hidden)
			if e != nil {
				return e
			}
			vf, e := f32(nm("linear_attn.v_proj.weight"), valueDim, hidden)
			if e != nil {
				return e
			}
			qkv := make([]float32, 0, convDim*hidden)
			qkv = append(qkv, qf...)
			qkv = append(qkv, kf...)
			qkv = append(qkv, vf...)
			d.InProjQKV = qw(linalg.WrapF32(qkv, convDim, hidden))
			if d.InProjZ, err = mkQ(nm("linear_attn.g_proj.weight"), valueDim, hidden); err != nil {
				return err
			}
			if d.InProjB, err = f32(nm("linear_attn.b_proj.weight"), g.NumValueHeads, hidden); err != nil {
				return err
			}
			if d.InProjA, err = f32(nm("linear_attn.a_proj.weight"), g.NumValueHeads, hidden); err != nil {
				return err
			}
		} else {
			if d.InProjQKV, err = mkQ(nm("linear_attn.in_proj_qkv.weight"), convDim, hidden); err != nil {
				return err
			}
			if d.InProjZ, err = mkQ(nm("linear_attn.in_proj_z.weight"), valueDim, hidden); err != nil {
				return err
			}
			if d.InProjB, err = f32(nm("linear_attn.in_proj_b.weight"), g.NumValueHeads, hidden); err != nil {
				return err
			}
			if d.InProjA, err = f32(nm("linear_attn.in_proj_a.weight"), g.NumValueHeads, hidden); err != nil {
				return err
			}
		}
		if g.SeparateConv {
			// Olmo Hybrid: q_conv1d/k_conv1d/v_conv1d, split at the SAME q/k/v
			// channel boundaries as q_proj/k_proj/v_proj (keyDim, keyDim,
			// valueDim rows) — see SeparateConv's own comment.
			qc, e := f32(nm("linear_attn.q_conv1d.weight"), keyDim, 1, g.ConvKernel)
			if e != nil {
				return e
			}
			kc, e := f32(nm("linear_attn.k_conv1d.weight"), keyDim, 1, g.ConvKernel)
			if e != nil {
				return e
			}
			vc, e := f32(nm("linear_attn.v_conv1d.weight"), valueDim, 1, g.ConvKernel)
			if e != nil {
				return e
			}
			conv := make([]float32, 0, convDim*g.ConvKernel)
			conv = append(conv, qc...)
			conv = append(conv, kc...)
			conv = append(conv, vc...)
			d.ConvW = conv
		} else if d.ConvW, err = f32(nm("linear_attn.conv1d.weight"), convDim, 1, g.ConvKernel); err != nil {
			return err
		}
		if d.DtBias, err = f32(nm("linear_attn.dt_bias"), g.NumValueHeads); err != nil {
			return err
		}
		aLog, aerr := f32(nm("linear_attn.A_log"), g.NumValueHeads)
		if aerr != nil {
			return aerr
		}
		d.NegExpA = hybridattn.NegExpAFromLog(aLog) // store −exp(A_log) (GGUF bakes this; here we compute it)
		// DeltaNetNormSuffix/DeltaNetOutProjSuffix (Olmo Hybrid: o_norm/o_proj instead of
		// qwen3.5's norm/out_proj) — see their own comment on qwen35Params.
		normSuffix := "linear_attn.norm.weight"
		if g.DeltaNetNormSuffix != "" {
			normSuffix = g.DeltaNetNormSuffix
		}
		outProjSuffix := "linear_attn.out_proj.weight"
		if g.DeltaNetOutProjSuffix != "" {
			outProjSuffix = g.DeltaNetOutProjSuffix
		}
		if d.NormW, err = f32(nm(normSuffix), g.ValueHeadDim); err != nil {
			return err
		}
		if d.OutProj, err = mkQ(nm(outProjSuffix), hidden, valueDim); err != nil {
			return err
		}
		l.Delta = d
		return nil
	}
	hd := arch.HeadDim
	kvd := arch.NumKVHeads * hd
	a := &archparts.QwenAttnWeights{}
	if a.QProj, err = mkQ(nm("self_attn.q_proj.weight"), arch.NumHeads*hd*2, hidden); err != nil {
		return err
	}
	if a.KProj, err = mkQ(nm("self_attn.k_proj.weight"), kvd, hidden); err != nil {
		return err
	}
	if a.VProj, err = mkQ(nm("self_attn.v_proj.weight"), kvd, hidden); err != nil {
		return err
	}
	if a.OProj, err = mkQ(nm("self_attn.o_proj.weight"), hidden, arch.NumHeads*hd); err != nil {
		return err
	}
	if a.QNorm, err = f32(nm("self_attn.q_norm.weight"), hd); err != nil {
		return err
	}
	if a.KNorm, err = f32(nm("self_attn.k_norm.weight"), hd); err != nil {
		return err
	}
	l.QAttn = a
	return nil
}

// loadBailingKDA loads one Bailing Hybrid (Ling 3.0) KDA layer's tensors. All under the
// "attention." prefix (BailingMoeV3DecoderLayer assigns BOTH its MLA and KDA mixers to
// self.attention, not self.self_attn — see mlaParams.AttnPrefix's own comment). Only the
// no_kda_lora path (a single f_proj/g_proj linear per gate) is implemented — see kdaArchitecture's
// own "what was deliberately not done" for the LoRA'd a/b-split variant, which ValidateBailingHybrid
// refuses before this is ever reached.
func loadBailingKDA(st *embed.SafetensorsFile, i int, l *darch.LayerWeights, arch *darch.Architecture, hidden int,
	tn func(int, string) string, mkQ func(string, int, int) (linalg.WeightMat, error)) error {
	g := arch.KDA
	projSize := g.NumHeads * g.HeadDim
	nm := func(suf string) string { return tn(i, "attention."+suf) }
	w := &archparts.KDAWeights{}
	var err error
	if w.QProj, err = mkQ(nm("q_proj.weight"), projSize, hidden); err != nil {
		return err
	}
	if w.KProj, err = mkQ(nm("k_proj.weight"), projSize, hidden); err != nil {
		return err
	}
	if w.VProj, err = mkQ(nm("v_proj.weight"), projSize, hidden); err != nil {
		return err
	}
	if w.QConvW, err = st.TensorF32(nm("q_conv1d.weight"), projSize, 1, g.ConvKernel); err != nil {
		return err
	}
	if w.KConvW, err = st.TensorF32(nm("k_conv1d.weight"), projSize, 1, g.ConvKernel); err != nil {
		return err
	}
	if w.VConvW, err = st.TensorF32(nm("v_conv1d.weight"), projSize, 1, g.ConvKernel); err != nil {
		return err
	}
	if w.FProj, err = mkQ(nm("f_proj.weight"), projSize, hidden); err != nil {
		return err
	}
	if w.DtBias, err = st.TensorF32(nm("dt_bias"), projSize); err != nil {
		return err
	}
	if w.ALog, err = st.TensorF32(nm("A_log"), g.NumHeads); err != nil {
		return err
	}
	if w.BProj, err = st.TensorF32(nm("b_proj.weight"), g.NumHeads, hidden); err != nil {
		return err
	}
	if w.GProj, err = mkQ(nm("g_proj.weight"), projSize, hidden); err != nil {
		return err
	}
	if w.ONormW, err = st.TensorF32(nm("o_norm.weight"), g.HeadDim); err != nil {
		return err
	}
	if w.OProj, err = mkQ(nm("o_proj.weight"), hidden, projSize); err != nil {
		return err
	}
	l.KDA = w
	return nil
}

// loadDeepseekAttn loads one DeepSeek MLA layer's attention tensors as f32 (the
// parity-first forward uses plain matvec). q-LoRA (q_a_proj→norm→q_b_proj) when
// arch.mla.QLoRARank > 0, else a direct q_proj (V2-Lite). The KV down-proj
// (kv_a_proj_with_mqa) emits the latent ‖ the shared rope key; kv_b_proj reconstructs
// per-head k_nope ‖ v from the normalized latent.
func loadDeepseekAttn(st *embed.SafetensorsFile, i int, l *darch.LayerWeights, arch *darch.Architecture, hidden int, tn func(int, string) string) error {
	p := arch.MLA
	qkHeadDim := p.QKHeadDim()
	qOut := arch.NumHeads * qkHeadDim
	kvUp := arch.NumHeads * (p.QKNopeHeadDim + p.VHeadDim)
	// AttnPrefix/DenseSuffix (Bailing Hybrid: "attention"/"dense.weight") override the
	// deepseek_v2/v3 defaults — see mlaParams' own comment.
	prefix := p.AttnPrefix
	if prefix == "" {
		prefix = "self_attn"
	}
	oProjSuffix := p.DenseSuffix
	if oProjSuffix == "" {
		oProjSuffix = "o_proj.weight"
	}
	nm := func(suf string) string { return tn(i, prefix+"."+suf) }
	w := &archparts.MLAWeights{}
	var err error
	if p.QLoRARank > 0 {
		if w.QAProj, err = st.TensorF32(nm("q_a_proj.weight"), p.QLoRARank, hidden); err != nil {
			return err
		}
		if w.QALayernorm, err = st.TensorF32(nm("q_a_layernorm.weight"), p.QLoRARank); err != nil {
			return err
		}
		if w.QBProj, err = st.TensorF32(nm("q_b_proj.weight"), qOut, p.QLoRARank); err != nil {
			return err
		}
	} else {
		if w.QProj, err = st.TensorF32(nm("q_proj.weight"), qOut, hidden); err != nil {
			return err
		}
	}
	if w.KVAProj, err = st.TensorF32(nm("kv_a_proj_with_mqa.weight"), p.KVLoRARank+p.QKRopeHeadDim, hidden); err != nil {
		return err
	}
	if w.KVALayernorm, err = st.TensorF32(nm("kv_a_layernorm.weight"), p.KVLoRARank); err != nil {
		return err
	}
	if w.KVBProj, err = st.TensorF32(nm("kv_b_proj.weight"), kvUp, p.KVLoRARank); err != nil {
		return err
	}
	if w.OProj, err = st.TensorF32(nm(oProjSuffix), hidden, arch.NumHeads*p.VHeadDim); err != nil {
		return err
	}
	switch p.GateGranularity {
	case "head_wise":
		if w.GProj, err = st.TensorF32(nm("g_proj.weight"), arch.NumHeads, hidden); err != nil {
			return err
		}
	case "element_wise":
		if w.GProj, err = st.TensorF32(nm("g_proj.weight"), arch.NumHeads*p.VHeadDim, hidden); err != nil {
			return err
		}
	}
	l.MLA = w
	return nil
}

// FusedLayout decides how a fused expert tensor is laid out from its shape, as transformers' own conversion does
// (Transpose(1, 2, check_dims=True) for qwen3_vl_moe: transpose only when the stored shape is not the module's): rows is
// the layout read as [E, a, b] (a rows of b), cols the transposed [E, b, a]. A shape matching rows (the square case
// included, where HF does not transpose either) reads as rows; one matching only cols reads transposed; anything else
// is refused. Element counts alone cannot tell the two apart, so a check on them loads the transposed layout as garbage
// with no error.
func FusedLayout(name string, shape []int, nExpert, a, b int, h *Hooks) (transposed bool, err error) {
	rows := slices.Equal(shape, []int{nExpert, a, b})
	cols := slices.Equal(shape, []int{nExpert, b, a})
	switch {
	case rows:
		return false, nil
	case cols:
		return true, nil
	}
	return false, fmt.Errorf("experts %q: shape %v is neither [%d %d %d] nor [%d %d %d]", name, shape, nExpert, a, b, nExpert, b, a)
}

// LoadFusedExperts reads a fused+stacked MoE expert set: gate_up_proj and down_proj, each [experts, ...], in either the
// Qwen3.5-MoE row layout ([E, 2I, H] and [E, H, I]) or transformers 4.57's transposed one ([E, H, 2I] and [E, I, H],
// Qwen3-VL-MoE's release), decided by shape as HF decides it (FusedLayout).
//
// It streams one expert at a time, via Tensor.SubF32 or, for the transposed layout, Tensor.SubF32Transposed, as
// streamExperts does for gemma4's fused experts: whole-tensor TensorF32 reads would materialize both stacks as f32, a
// multi-GB transient per layer times every layer parallelLayers has in flight. The experts are built through
// forExperts.
func LoadFusedExperts(st *embed.SafetensorsFile, gateUpName, downName string, nExpert, inter, hidden int, quant QuantMode, needCanonical, skipRow4 bool, sink *StreamWriter, h *Hooks) ([]archparts.ExpertWeights, error) {
	guT, err := st.Tensor(gateUpName)
	if err != nil {
		return nil, err
	}
	dnT, err := st.Tensor(downName)
	if err != nil {
		return nil, err
	}
	guStride, downStride, half := 2*inter*hidden, hidden*inter, inter*hidden
	if guT.Elements() != nExpert*guStride {
		return nil, fmt.Errorf("experts %q: %d elements, want %d (=%d×%d×%d)", gateUpName, guT.Elements(), nExpert*guStride, nExpert, 2*inter, hidden)
	}
	if dnT.Elements() != nExpert*downStride {
		return nil, fmt.Errorf("experts %q: %d elements, want %d (=%d×%d×%d)", downName, dnT.Elements(), nExpert*downStride, nExpert, hidden, inter)
	}
	guTrans, err := FusedLayout(gateUpName, guT.Shape, nExpert, 2*inter, hidden, h)
	if err != nil {
		return nil, err
	}
	dnTrans, err := FusedLayout(downName, dnT.Shape, nExpert, hidden, inter, h)
	if err != nil {
		return nil, err
	}
	if h.fusedExpertsTransposed() {
		guTrans, dnTrans = false, false
	}
	// A slice read in place from an F32 tensor is a view of the mapping; every other read is a slice of its own.
	guView, dnView := guT.DType == "F32" && !guTrans, dnT.DType == "F32" && !dnTrans
	// q quantizes one expert matrix. A matrix that stays f32 and was a view of the mapping gets a slice of its own.
	q := func(data []float32, rows, cols int, view bool) linalg.WeightMat {
		var m linalg.WeightMat
		m = h.QuantizeLayerWM(linalg.WrapF32(data, rows, cols), quant, needCanonical, skipRow4)
		if f, ok := m.F32(); ok && view {
			m = linalg.WrapF32(slices.Clone(f), rows, cols)
		}
		return m
	}
	experts := make([]archparts.ExpertWeights, nExpert)
	err = ForExperts(nExpert, sink, func(e int) error {
		var guE, dnE []float32
		var err error
		if guTrans { // [H, 2I] -> [2I, H]: gate rows first, then up
			guE, err = guT.SubF32Transposed(e*guStride, hidden, 2*inter)
		} else {
			guE, err = guT.SubF32(e*guStride, guStride)
		}
		if err != nil {
			return err
		}
		if dnTrans { // [I, H] -> [H, I]
			dnE, err = dnT.SubF32Transposed(e*downStride, inter, hidden)
		} else {
			dnE, err = dnT.SubF32(e*downStride, downStride)
		}
		if err != nil {
			return err
		}
		experts[e] = archparts.ExpertWeights{
			Gate: q(guE[:half:half], inter, hidden, guView),
			Up:   q(guE[half:], inter, hidden, guView),
			Down: q(dnE, hidden, inter, dnView),
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return experts, nil
}

// loadGemma4MoE loads one gemma4 layer's parallel dense+MoE FFN sub-block (enable_moe_block) into a gemma4MoEWeights,
// consumed by gemma4MoEFFN. The dense branch MLP (l.GateProj/UpProj/DownProj) and its sandwich norms (l.PreMLPNorm =
// pre_feedforward_layernorm, l.PostMLPNorm = the joint post_feedforward_layernorm) are already loaded by the caller; this
// aliases them and loads the MoE-specific tensors: the three parallel-branch norms, the weightless-norm/learned-scale
// router with its per-expert scale, and the fused gelu-tanh experts (gate_up || down).
//
// Router weights stay f32 (loadMat, no quant: the router is logit-critical); the experts quantize at load through the
// layer's quant mode. The experts stream one at a time via Tensor.SubF32, so a bf16 checkpoint never materializes the
// whole [nExperts, 2*inter, hidden] gate_up, only one expert's f32 at a time.
func loadGemma4MoE(st *embed.SafetensorsFile, i int, cfg *config.Config, arch *darch.Architecture, hidden int, l *darch.LayerWeights, quant QuantMode, tn func(int, string) string, needCanonical, skipRow4 bool, sink *StreamWriter, h *Hooks) (*archparts.Gemma4MoEWeights, error) {
	m := arch.MoE
	nm := func(s string) string { return tn(i, s) }
	w := &archparts.Gemma4MoEWeights{
		PreFFNNorm:  l.PreMLPNorm,  // pre_feedforward_layernorm   (dense pre-norm)
		PostFFNNorm: l.PostMLPNorm, // post_feedforward_layernorm  (joint post-norm on x1+x2)
		MlpGate:     l.GateProj,
		MlpUp:       l.UpProj,
		MlpDown:     l.DownProj,
		LayerScalar: l.LayerScalar,
		DenseInter:  cfg.IntermediateDim,
		MoeInter:    m.IntermediateDim,
		NE:          m.NumExperts,
		TopK:        m.TopK,
	}
	var err error
	if w.PostFFNNorm1, err = st.TensorF32(nm("post_feedforward_layernorm_1.weight"), hidden); err != nil {
		return nil, err
	}
	if w.PreFFNNorm2, err = st.TensorF32(nm("pre_feedforward_layernorm_2.weight"), hidden); err != nil {
		return nil, err
	}
	if w.PostFFNNorm2, err = st.TensorF32(nm("post_feedforward_layernorm_2.weight"), hidden); err != nil {
		return nil, err
	}
	if w.RouterProj, err = LoadMat(st, nm("router.proj.weight"), m.NumExperts, hidden); err != nil {
		return nil, err
	}
	if w.RouterScale, err = st.TensorF32(nm("router.scale"), hidden); err != nil {
		return nil, err
	}
	if w.PerExpertScale, err = st.TensorF32(nm("router.per_expert_scale"), m.NumExperts); err != nil {
		return nil, err
	}
	// Fused experts: gate_up [E, 2*moeInter, hidden] (gate ‖ up on the row axis),
	// down [E, hidden, moeInter]. Streamed + quantized per expert.
	guT, err := st.Tensor(nm("experts.gate_up_proj"))
	if err != nil {
		return nil, err
	}
	dnT, err := st.Tensor(nm("experts.down_proj"))
	if err != nil {
		return nil, err
	}
	if w.ExpertsGateUp, err = streamExperts(guT, m.NumExperts, 2*m.IntermediateDim, hidden, quant, needCanonical, skipRow4, sink, h); err != nil {
		return nil, err
	}
	if w.ExpertsDown, err = streamExperts(dnT, m.NumExperts, hidden, m.IntermediateDim, quant, needCanonical, skipRow4, sink, h); err != nil {
		return nil, err
	}
	return w, nil
}

// streamExperts slices a fused [nExpert, rows, cols] safetensors tensor into per-
// expert WeightMats, widening + quantizing ONE expert at a time via Tensor.SubF32
// so the whole 3-D f32 is never materialized (the bf16-26B transient win). Each
// expert passes through quantizeWM: int8/int4 produces owned quantized data and the
// f32 slice is dropped; f32/QuantNone leaves an f32 WeightMat aliasing the mapping
// (pageable, valid while w.st is retained) — no heap copy. The experts are built through forExperts.
func streamExperts(t embed.Tensor, nExpert, rows, cols int, quant QuantMode, needCanonical, skipRow4 bool, sink *StreamWriter, h *Hooks) ([]linalg.WeightMat, error) {
	stride := rows * cols
	if t.Elements() != nExpert*stride {
		return nil, fmt.Errorf("experts %q: %d elements, want %d (=%d×%d×%d)", t.Name, t.Elements(), nExpert*stride, nExpert, rows, cols)
	}
	out := make([]linalg.WeightMat, nExpert)
	err := ForExperts(nExpert, sink, func(e int) error {
		f32, err := t.SubF32(e*stride, stride)
		if err != nil {
			return err
		}
		if h.fakeQuantScheme() != "" && h.fakeQuantExpertsOnly() { // DIAGNOSTIC (default-off): fake-4-bit the EXPERTS only; see fakequant.go
			out[e] = h.FakeInt4WM(f32, rows, cols, h.fakeQuantScheme())
			return nil
		}
		out[e] = h.QuantizeLayerWM(linalg.WrapF32(f32, rows, cols), quant, needCanonical, skipRow4)
		return nil
	})
	if err != nil {
		return nil, err
	}
	return out, nil
}
