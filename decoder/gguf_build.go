package decoder

import (
	"errors"
	"fmt"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
)

// ggufBuild is the state buildWeightsFromGGUF hands a family builder: the load's inputs, the weight bundle it fills, and the
// tensor readers every family shares. A builder reads them under the names the shared body uses and returns the finished
// bundle; it drives the streaming sink itself when one is set.
type ggufBuild struct {
	arch                    *Architecture
	g                       *embed.GGUFFile
	w                       *Weights
	quant                   quantMode
	needCanonical, skipRow4 bool
	sink                    *giwWriter
	abort                   <-chan struct{}
	hidden                  int
	streamMat               func(name string, out, in int, mode quantMode, rowSrc func(r int) int) (linalg.WeightMat, error)
	mat                     func(name string, out, in int) (linalg.WeightMat, error)
	vec, vnorm              func(name string, n int) ([]float32, error)
	stackedExperts          func(name string, out, in, nExpert int) ([]linalg.WeightMat, error)
}

// qwen3_5_moe (qwen35moe): the Gated DeltaNet / gated-softmax hybrid. The GGUF
// converter bakes layout transforms the HF safetensors path never sees — V
// heads tiled (num_v>num_k), A_log stored as −exp(A_log), the standard norms
// (1+w)'d (ssm_norm exempt) — all reversed here so the shared forward runs
// unchanged. Attention stays f32 (parity-first, like the safetensors path);
// experts/embeddings quantize. Only blk.{0..NumLayers-1} load — the trailing
// NextN/MTP block is excluded by ggufQwen35Config's NumLayers.
func (b *ggufBuild) loadQwen35(q *qwen35Params) (*Weights, error) {
	arch, g, w, quant, needCanonical, skipRow4, sink, abort, hidden := b.arch, b.g, b.w, b.quant, b.needCanonical, b.skipRow4, b.sink, b.abort, b.hidden
	streamMat, mat, vec, vnorm, stackedExperts := b.streamMat, b.mat, b.vec, b.vnorm, b.stackedExperts
	numK, numV := q.NumKeyHeads, q.NumValueHeads
	vpk := numV / numK
	hkd, hvd := q.KeyHeadDim, q.ValueHeadDim
	keyDim, valueDim := hkd*numK, hvd*numV
	convDim := 2*keyDim + valueDim
	// f32mat dequantizes a [out, in] tensor to a row-major f32 slice. Some of these need a
	// layout transform (untileVHeads) before they can be used, so the f32 form is still the
	// working intermediate; wmQ below wraps the result and applies Options.Quant.
	f32mat := func(name string, out, in int) ([]float32, error) {
		dims, data, derr := g.Tensor(name)
		if derr != nil {
			return nil, derr
		}
		if len(dims) != 2 || dims[0] != in || dims[1] != out {
			return nil, fmt.Errorf("decoder(gguf-qwen35): %q dims %v, want [in=%d, out=%d]", name, dims, in, out)
		}
		return data, nil
	}
	// wmQ is f32mat + Options.Quant, for the projections that dominate decode bandwidth. Transform-then-quantize: the untile
	// below has to see f32. These are the projections for which the row4 side-copy is waste, so both route through the
	// skip-aware wrapper like every other family's layer projections.
	wmQ := func(name string, out, in int) (linalg.WeightMat, error) {
		f, e := f32mat(name, out, in)
		if e != nil {
			return linalg.WeightMat{}, e
		}
		return quantizeLayerWM(linalg.WrapF32(f, out, in), quant, needCanonical, skipRow4), nil
	}
	wrapQ := func(f []float32, out, in int) linalg.WeightMat {
		return quantizeLayerWM(linalg.WrapF32(f, out, in), quant, needCanonical, skipRow4)
	}
	loadQ35 := func(i int) error {
		l := &w.Layers[i]
		p := fmt.Sprintf("blk.%d.", i)
		var e error
		// attn_norm = input_layernorm (pre-attn); post_attention_norm =
		// post_attention_layernorm (pre-MLP, Pre2). Both (1+w)'d → vnorm subtracts.
		if l.PreAttnNorm, e = vnorm(p+"attn_norm.weight", hidden); e != nil {
			return e
		}
		if l.PreMLPNorm, e = vnorm(p+"post_attention_norm.weight", hidden); e != nil {
			return e
		}
		if arch.isLinearLayer(i) {
			d := &deltaNetWeights{}
			// in_proj_qkv (attn_qkv): only the V output rows are tiled; q/k untouched.
			qkv, err := f32mat(p+"attn_qkv.weight", convDim, hidden)
			if err != nil {
				return err
			}
			vOff := 2 * keyDim * hidden
			copy(qkv[vOff:], untileVHeads(qkv[vOff:], 1, numK, vpk, hvd, hidden))
			d.inProjQKV = wrapQ(qkv, convDim, hidden)
			// in_proj_z (attn_gate): all rows tiled.
			z, err := f32mat(p+"attn_gate.weight", valueDim, hidden)
			if err != nil {
				return err
			}
			d.inProjZ = wrapQ(untileVHeads(z, 1, numK, vpk, hvd, hidden), valueDim, hidden)
			// in_proj_a / in_proj_b (ssm_alpha/ssm_beta): one row per value head (hd=1).
			a, err := f32mat(p+"ssm_alpha.weight", numV, hidden)
			if err != nil {
				return err
			}
			d.inProjA = untileVHeads(a, 1, numK, vpk, 1, hidden)
			b, err := f32mat(p+"ssm_beta.weight", numV, hidden)
			if err != nil {
				return err
			}
			d.inProjB = untileVHeads(b, 1, numK, vpk, 1, hidden)
			// conv1d (ssm_conv1d): only the V channel block tiled; trail = kernel width.
			cw, err := f32mat(p+"ssm_conv1d.weight", convDim, q.ConvKernel)
			if err != nil {
				return err
			}
			vOffC := 2 * keyDim * q.ConvKernel
			copy(cw[vOffC:], untileVHeads(cw[vOffC:], 1, numK, vpk, hvd, q.ConvKernel))
			d.convW = cw
			// ssm_a is already −exp(A_log); reorder by value head (1-D).
			av, err := vec(p+"ssm_a", numV)
			if err != nil {
				return err
			}
			d.negExpA = untileVHeads(av, 1, numK, vpk, 1, 1)
			// dt_bias (ssm_dt.bias): reorder by value head (1-D).
			dt, err := vec(p+"ssm_dt.bias", numV)
			if err != nil {
				return err
			}
			d.dtBias = untileVHeads(dt, 1, numK, vpk, 1, 1)
			// ssm_norm: per-head_v_dim gated-RMSNorm weight — NOT tiled and NOT
			// (1+w)'d by the converter, so load raw (vec, not vnorm).
			if d.normW, err = vec(p+"ssm_norm.weight", hvd); err != nil {
				return err
			}
			// out_proj (ssm_out): the value/input columns are tiled.
			op, err := f32mat(p+"ssm_out.weight", hidden, valueDim)
			if err != nil {
				return err
			}
			d.outProj = wrapQ(untileVHeads(op, hidden, numK, vpk, hvd, 1), hidden, valueDim)
			l.delta = d
		} else {
			// Gated softmax layer: double-width q_proj (query ‖ gate per head, kept
			// fused in attn_q), per-head QK-norm. NEOX rope ⇒ no q/k permute.
			a := &qwenAttnWeights{}
			shd := arch.HeadDim
			kvDim := arch.NumKVHeads * shd
			if a.qProj, e = wmQ(p+"attn_q.weight", arch.NumHeads*shd*2, hidden); e != nil {
				return e
			}
			if a.kProj, e = wmQ(p+"attn_k.weight", kvDim, hidden); e != nil {
				return e
			}
			if a.vProj, e = wmQ(p+"attn_v.weight", kvDim, hidden); e != nil {
				return e
			}
			if a.oProj, e = wmQ(p+"attn_output.weight", hidden, arch.NumHeads*shd); e != nil {
				return e
			}
			if a.qNorm, e = vnorm(p+"attn_q_norm.weight", shd); e != nil {
				return e
			}
			if a.kNorm, e = vnorm(p+"attn_k_norm.weight", shd); e != nil {
				return e
			}
			l.qattn = a
		}
		// DENSE (Qwen3.8, llama.cpp arch "qwen35"): a plain SwiGLU where the MoE sibling has a router. The rest of this loader (the
		// V un-tile, the -exp(A_log) un-bake, the (1+w) norms) is shared.
		if arch.MoE == nil {
			if l.GateProj, e = mat(p+"ffn_gate.weight", arch.IntermediateDim, hidden); e != nil {
				return e
			}
			if l.UpProj, e = mat(p+"ffn_up.weight", arch.IntermediateDim, hidden); e != nil {
				return e
			}
			if l.DownProj, e = mat(p+"ffn_down.weight", hidden, arch.IntermediateDim); e != nil {
				return e
			}
			return nil
		}
		// MoE FFN (every layer): router f32 (matching safetensors), 256 stacked
		// experts quantized, shared expert quantized + f32 sigmoid gate.
		if l.Router, e = streamMat(p+"ffn_gate_inp.weight", arch.MoE.NumExperts, hidden, quantNone, func(r int) int { return r * hidden }); e != nil {
			return e
		}
		expInter := arch.MoE.IntermediateDim
		gate, gerr := stackedExperts(p+"ffn_gate_exps.weight", expInter, hidden, arch.MoE.NumExperts)
		up, uerr := stackedExperts(p+"ffn_up_exps.weight", expInter, hidden, arch.MoE.NumExperts)
		down, derr := stackedExperts(p+"ffn_down_exps.weight", hidden, expInter, arch.MoE.NumExperts)
		if gerr != nil || uerr != nil || derr != nil {
			return fmt.Errorf("decoder(gguf-qwen35): experts layer %d: %w", i, errors.Join(gerr, uerr, derr))
		}
		l.Experts = make([]expertWeights, arch.MoE.NumExperts)
		for ei := range arch.MoE.NumExperts {
			l.Experts[ei] = expertWeights{Gate: gate[ei], Up: up[ei], Down: down[ei]}
		}
		sInter := arch.MoE.SharedIntermediateDim
		if l.SharedExpert.Gate, e = mat(p+"ffn_gate_shexp.weight", sInter, hidden); e != nil {
			return e
		}
		if l.SharedExpert.Up, e = mat(p+"ffn_up_shexp.weight", sInter, hidden); e != nil {
			return e
		}
		if l.SharedExpert.Down, e = mat(p+"ffn_down_shexp.weight", hidden, sInter); e != nil {
			return e
		}
		// shared_expert_gate (ffn_gate_inp_shexp): 1-D [hidden] → [1, hidden] f32.
		sg, err := vec(p+"ffn_gate_inp_shexp.weight", hidden)
		if err != nil {
			return err
		}
		l.SharedGate = linalg.WrapF32(sg, 1, hidden)
		return nil
	}
	// Streaming (sink != nil, e.g. cmd/prequant): loadQ35 builds one layer's data independently of every other layer
	// (parallelLayers's concurrency is a speed choice, not a correctness dependency), so streamLayers' build, write and
	// release, a window of layers at a time, writes bit-identical per-layer output while bounding peak RSS to about one
	// window. Non-streaming (sink == nil): all layers in parallel.
	if sink != nil {
		if err := streamLayers(w, arch, sink, loadQ35); err != nil {
			return nil, err
		}
		return w, nil
	}
	if err := parallelLayers(arch.NumLayers, abort, loadQ35); err != nil {
		return nil, err
	}
	return w, nil
}
