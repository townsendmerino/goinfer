package decoder

import (
	"errors"
	"fmt"
	"math"

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
	hd                      int
	permMat                 func(name string, out, in, nHead int) (linalg.WeightMat, error)
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

// gpt-oss: sparse MoE on every layer with per-head attention sinks, per-expert biases and a router-logit bias. q/k/v/o all
// carry biases; no QK-norm. Attention/embeddings/router quantize per policy; the (F32) sinks and biases load raw. The
// expert weights are MXFP4 (ggml type 39) in the real checkpoint; stackedExperts routes them through aikit's
// RowDequantizer.
func (b *ggufBuild) loadGptOss() (*Weights, error) {
	arch, g, w, sink, abort, hidden, hd, streamMat, mat, vec, vnorm, stackedExperts := b.arch, b.g, b.w, b.sink, b.abort, b.hidden, b.hd, b.streamMat, b.mat, b.vec, b.vnorm, b.stackedExperts
	nH, nKV := arch.NumHeads, arch.NumKVHeads
	qDim, kvDim := nH*hd, nKV*hd
	nExp := arch.MoE.NumExperts
	expInter := arch.MoE.IntermediateDim
	// stackedExpertBias slices a 2-D [out, nExpert] (fastest-first) f32 bias
	// tensor into one [out] slice per expert (expert e occupies [e*out, e*out+out)).
	stackedExpertBias := func(name string, out int) ([][]float32, error) {
		dims, data, derr := g.Tensor(name)
		if derr != nil {
			return nil, derr
		}
		if len(dims) != 2 || dims[0] != out || dims[1] != nExp {
			return nil, fmt.Errorf("decoder(gguf-gptoss): %q dims %v, want [out=%d, experts=%d]", name, dims, out, nExp)
		}
		res := make([][]float32, nExp)
		for e := range nExp {
			res[e] = append([]float32(nil), data[e*out:(e+1)*out]...)
		}
		return res, nil
	}
	loadGptOss := func(i int) error {
		l := &w.Layers[i]
		p := fmt.Sprintf("blk.%d.", i)
		var e error
		if l.PreAttnNorm, e = vnorm(p+"attn_norm.weight", hidden); e != nil {
			return e
		}
		if l.PreMLPNorm, e = vnorm(p+"post_attention_norm.weight", hidden); e != nil {
			return e
		}
		// Attention: q/k/v/o projections + biases. gpt-oss GGUF stores q/k in HF
		// order (no undo_permute), so a plain mat load — no permMat.
		if l.QProj, e = mat(p+"attn_q.weight", qDim, hidden); e != nil {
			return e
		}
		if l.KProj, e = mat(p+"attn_k.weight", kvDim, hidden); e != nil {
			return e
		}
		if l.VProj, e = mat(p+"attn_v.weight", kvDim, hidden); e != nil {
			return e
		}
		if l.OProj, e = mat(p+"attn_output.weight", hidden, qDim); e != nil {
			return e
		}
		if l.QBias, e = vec(p+"attn_q.bias", qDim); e != nil {
			return e
		}
		if l.KBias, e = vec(p+"attn_k.bias", kvDim); e != nil {
			return e
		}
		if l.VBias, e = vec(p+"attn_v.bias", kvDim); e != nil {
			return e
		}
		if l.OBias, e = vec(p+"attn_output.bias", hidden); e != nil {
			return e
		}
		if l.AttnSinks, e = vec(p+"attn_sinks.weight", nH); e != nil {
			return e
		}
		// MoE: router (+ logit bias) + stacked routed experts (+ per-expert biases). Router stays f32 regardless of the ambient
		// quant mode (matching gptoss_safetensors.go and qwen35's streamMat(..., quantNone, ...) below): top-k selection is
		// discrete, so quantizing it flips which experts win rather than adding rounding noise. Plain mat() would quantize it.
		if l.Router, e = streamMat(p+"ffn_gate_inp.weight", nExp, hidden, quantNone, func(r int) int { return r * hidden }); e != nil {
			return e
		}
		if l.RouterBias, e = vec(p+"ffn_gate_inp.bias", nExp); e != nil {
			return e
		}
		gate, ge := stackedExperts(p+"ffn_gate_exps.weight", expInter, hidden, nExp)
		up, ue := stackedExperts(p+"ffn_up_exps.weight", expInter, hidden, nExp)
		down, de := stackedExperts(p+"ffn_down_exps.weight", hidden, expInter, nExp)
		if ge != nil || ue != nil || de != nil {
			return fmt.Errorf("decoder(gguf-gptoss): experts layer %d: %w", i, errors.Join(ge, ue, de))
		}
		gb, gbe := stackedExpertBias(p+"ffn_gate_exps.bias", expInter)
		ub, ube := stackedExpertBias(p+"ffn_up_exps.bias", expInter)
		db, dbe := stackedExpertBias(p+"ffn_down_exps.bias", hidden)
		if gbe != nil || ube != nil || dbe != nil {
			return fmt.Errorf("decoder(gguf-gptoss): expert biases layer %d: %w", i, errors.Join(gbe, ube, dbe))
		}
		l.Experts = make([]expertWeights, nExp)
		for ei := range nExp {
			l.Experts[ei] = expertWeights{
				Gate: gate[ei], Up: up[ei], Down: down[ei],
				GateBias: gb[ei], UpBias: ub[ei], DownBias: db[ei],
			}
		}
		return nil
	}
	// Streaming (sink != nil): loadGptOss builds one layer's data independently of every other layer (every tensor it reads is
	// named blk.{i}.*, including the per-layer AttnSinks and RouterBias; stackedExperts and stackedExpertBias are per-tensor
	// RowDequantizer reads keyed by the same name), so the loadQ35 shape applies: streamLayers' build, write and release
	// bounds peak RSS to about one window. Non-streaming (sink == nil): all layers in parallel.
	if sink != nil {
		if err := streamLayers(w, arch, sink, loadGptOss); err != nil {
			return nil, err
		}
		return w, nil
	}
	if err := parallelLayers(arch.NumLayers, abort, loadGptOss); err != nil {
		return nil, err
	}
	return w, nil
}

// Laguna (poolside). llama.cpp names its tensors the way every other MoE family here is named, so the only genuinely new
// one is blk.N.attn_gate.weight, the softplus output gate. Two shape details drive the rest:
//
//   - qDim is per layer (arch.headsAt(i)): full-attention and sliding layers project different head counts, so
//     attn_q/attn_output differ by layer.
//   - the gate's granularity is read from attn_gate's row count, exactly as the safetensors loader does, because neither
//     format carries a trustworthy declaration of it (XS.2's config says per-element and ships per-head).
//
// Experts are fused and stacked per projection (ffn_*_exps), the shared expert is *_shexp, and the router bias is
// exp_probs_b: all shapes goinfer already reads for GLM/DeepSeek/Granite. Layers below FirstKDense are plain dense FFNs.
func (b *ggufBuild) loadLaguna() (*Weights, error) {
	arch, g, w, sink, abort, hidden, streamMat, mat, vnorm, stackedExperts := b.arch, b.g, b.w, b.sink, b.abort, b.hidden, b.streamMat, b.mat, b.vnorm, b.stackedExperts
	kvDim := arch.NumKVHeads * arch.HeadDim
	// exp_probs_b is 1-D; the shared `mat` helper wants 2-D, so read it as a flat
	// vector the same way the other MoE GGUF loaders do.
	flat := func(name string, n int) ([]float32, error) { // 1-D or [1,n]/[n,1] → n floats
		_, data, derr := g.Tensor(name)
		if derr != nil {
			return nil, derr
		}
		if len(data) != n {
			return nil, fmt.Errorf("decoder(gguf-laguna): %q has %d elems, want %d", name, len(data), n)
		}
		return data, nil
	}
	loadLaguna := func(i int) error {
		l := &w.Layers[i]
		p := fmt.Sprintf("blk.%d.", i)
		lqDim := arch.headsAt(i) * arch.HeadDim
		var e error
		if l.PreAttnNorm, e = vnorm(p+"attn_norm.weight", hidden); e != nil {
			return e
		}
		if l.PreMLPNorm, e = vnorm(p+"ffn_norm.weight", hidden); e != nil {
			return e
		}
		if l.QProj, e = mat(p+"attn_q.weight", lqDim, hidden); e != nil {
			return e
		}
		if l.KProj, e = mat(p+"attn_k.weight", kvDim, hidden); e != nil {
			return e
		}
		if l.VProj, e = mat(p+"attn_v.weight", kvDim, hidden); e != nil {
			return e
		}
		if l.OProj, e = mat(p+"attn_output.weight", hidden, lqDim); e != nil {
			return e
		}
		if l.QNorm, e = vnorm(p+"attn_q_norm.weight", arch.HeadDim); e != nil {
			return e
		}
		if l.KNorm, e = vnorm(p+"attn_k_norm.weight", arch.HeadDim); e != nil {
			return e
		}
		// The gate: per-head (headsAt rows) or per-element (headsAt*headDim). Try the
		// per-head shape first and fall back, so both are accepted and anything else
		// is a hard error rather than a silently mis-shaped gate.
		if l.GProj, e = mat(p+"attn_gate.weight", arch.headsAt(i), hidden); e != nil {
			if l.GProj, e = mat(p+"attn_gate.weight", lqDim, hidden); e != nil {
				return fmt.Errorf("decoder(gguf-laguna): layer %d attn_gate is neither per-head [%d,%d] nor per-element [%d,%d]: %w",
					i, arch.headsAt(i), hidden, lqDim, hidden, e)
			}
		}
		if i < arch.FirstKDense {
			// Dense prefix layer: a plain SwiGLU at the model's feed_forward_length.
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
		moe := arch.MoE
		// Router stays f32 regardless of the ambient quant mode (matching qwen35/gptoss above): top-k selection is discrete, so
		// quantizing it flips which experts win rather than adding rounding noise.
		if l.Router, e = streamMat(p+"ffn_gate_inp.weight", moe.NumExperts, hidden, quantNone, func(r int) int { return r * hidden }); e != nil {
			return e
		}
		// exp_probs_b is llama.cpp's name for e_score_correction_bias.
		if l.RouterBias, e = flat(p+"exp_probs_b.bias", moe.NumExperts); e != nil {
			return e
		}
		expInter := moe.IntermediateDim
		gate, ge := stackedExperts(p+"ffn_gate_exps.weight", expInter, hidden, moe.NumExperts)
		up, ue := stackedExperts(p+"ffn_up_exps.weight", expInter, hidden, moe.NumExperts)
		down, de := stackedExperts(p+"ffn_down_exps.weight", hidden, expInter, moe.NumExperts)
		if ge != nil || ue != nil || de != nil {
			return fmt.Errorf("decoder(gguf-laguna): experts layer %d: %w", i, errors.Join(ge, ue, de))
		}
		l.Experts = make([]expertWeights, moe.NumExperts)
		for ei := range moe.NumExperts {
			l.Experts[ei] = expertWeights{Gate: gate[ei], Up: up[ei], Down: down[ei]}
		}
		if sInter := moe.SharedIntermediateDim; sInter > 0 {
			if l.SharedExpert.Gate, e = mat(p+"ffn_gate_shexp.weight", sInter, hidden); e != nil {
				return e
			}
			if l.SharedExpert.Up, e = mat(p+"ffn_up_shexp.weight", sInter, hidden); e != nil {
				return e
			}
			if l.SharedExpert.Down, e = mat(p+"ffn_down_shexp.weight", hidden, sInter); e != nil {
				return e
			}
		}
		return nil
	}
	// Streaming (sink != nil): loadLaguna is self-contained per layer (every tensor it reads is blk.{i}.*-named;
	// decoder/gguf_streaming_shape_test.go proves this from source and keeps it true), so the loadQ35/loadGptOss streaming
	// shape applies unchanged.
	if sink != nil {
		if err := streamLayers(w, arch, sink, loadLaguna); err != nil {
			return nil, err
		}
		return w, nil
	}
	if err := parallelLayers(arch.NumLayers, abort, loadLaguna); err != nil {
		return nil, err
	}
	return w, nil
}

// Granite-4.0-H (granitehybrid): per-layer Mamba-2 mixer or GQA attention, MoE on every layer. The Mamba-2 ssm_*
// tensors use llama.cpp's conventions (shared with qwen35): ssm_a stores -exp(A_log) directly (reversed to A_log via
// log(-a) so the shared mamba2Step works), conv1d is [convDim, K], ssm_norm raw. Mixer stays f32 (parity-first);
// experts/attention/embeddings quantize. NEOX rope, so no q/k permute.
func (b *ggufBuild) loadGranite(gp *graniteParams) (*Weights, error) {
	arch, g, w, sink, abort, hidden, streamMat, mat, vnorm, stackedExperts, permMat := b.arch, b.g, b.w, b.sink, b.abort, b.hidden, b.streamMat, b.mat, b.vnorm, b.stackedExperts, b.permMat
	dInner := gp.NHeads * gp.HeadDim
	convDim := dInner + 2*gp.NGroups*gp.DState
	projDim := 2*dInner + 2*gp.NGroups*gp.DState + gp.NHeads
	hd := arch.HeadDim
	qDim, kvDim := arch.NumHeads*hd, arch.NumKVHeads*hd
	f32mat := func(name string, out, in int) ([]float32, error) {
		dims, data, derr := g.Tensor(name)
		if derr != nil {
			return nil, derr
		}
		if len(dims) != 2 || dims[0] != in || dims[1] != out {
			return nil, fmt.Errorf("decoder(gguf-granite): %q dims %v, want [in=%d, out=%d]", name, dims, in, out)
		}
		return data, nil
	}
	flat := func(name string, n int) ([]float32, error) { // 1-D or [1,n]/[n,1] tensor → n floats
		_, data, derr := g.Tensor(name)
		if derr != nil {
			return nil, derr
		}
		if len(data) != n {
			return nil, fmt.Errorf("decoder(gguf-granite): %q has %d elems, want %d", name, len(data), n)
		}
		return data, nil
	}
	loadGranite := func(i int) error {
		l := &w.Layers[i]
		p := fmt.Sprintf("blk.%d.", i)
		var e error
		if l.PreAttnNorm, e = vnorm(p+"attn_norm.weight", hidden); e != nil {
			return e
		}
		if l.PreMLPNorm, e = vnorm(p+"ffn_norm.weight", hidden); e != nil {
			return e
		}
		if arch.isMambaLayer(i) {
			mw := &mamba2Weights{}
			if mw.inProj, e = f32mat(p+"ssm_in.weight", projDim, hidden); e != nil {
				return e
			}
			if mw.convW, e = f32mat(p+"ssm_conv1d.weight", convDim, gp.DConv); e != nil {
				return e
			}
			if mw.convB, e = flat(p+"ssm_conv1d.bias", convDim); e != nil {
				return e
			}
			if mw.d, e = flat(p+"ssm_d", gp.NHeads); e != nil {
				return e
			}
			if mw.dtBias, e = flat(p+"ssm_dt.bias", gp.NHeads); e != nil {
				return e
			}
			if mw.normW, e = flat(p+"ssm_norm.weight", dInner); e != nil {
				return e
			}
			if mw.outProj, e = f32mat(p+"ssm_out.weight", hidden, dInner); e != nil {
				return e
			}
			// ssm_a is −exp(A_log); reverse to A_log so mamba2Step's −exp(aLog) recovers it.
			a, aerr := flat(p+"ssm_a", gp.NHeads)
			if aerr != nil {
				return aerr
			}
			mw.aLog = make([]float32, gp.NHeads)
			for h := range a {
				mw.aLog[h] = float32(math.Log(-float64(a[h])))
			}
			l.mamba = mw
		} else {
			// llama.cpp converts GraniteHybrid attention with undo_permute=True, so
			// q/k are RoPE-permuted in the GGUF (like Llama) and must be un-permuted.
			if l.QProj, e = permMat(p+"attn_q.weight", qDim, hidden, arch.NumHeads); e != nil {
				return e
			}
			if l.KProj, e = permMat(p+"attn_k.weight", kvDim, hidden, arch.NumKVHeads); e != nil {
				return e
			}
			if l.VProj, e = mat(p+"attn_v.weight", kvDim, hidden); e != nil {
				return e
			}
			if l.OProj, e = mat(p+"attn_output.weight", hidden, qDim); e != nil {
				return e
			}
		}
		// MoE on every layer: router + stacked routed experts + ungated shared.
		expInter := arch.MoE.IntermediateDim
		// Router stays f32; see the Laguna site above.
		if l.Router, e = streamMat(p+"ffn_gate_inp.weight", arch.MoE.NumExperts, hidden, quantNone, func(r int) int { return r * hidden }); e != nil {
			return e
		}
		gate, ge := stackedExperts(p+"ffn_gate_exps.weight", expInter, hidden, arch.MoE.NumExperts)
		up, ue := stackedExperts(p+"ffn_up_exps.weight", expInter, hidden, arch.MoE.NumExperts)
		down, de := stackedExperts(p+"ffn_down_exps.weight", hidden, expInter, arch.MoE.NumExperts)
		if ge != nil || ue != nil || de != nil {
			return fmt.Errorf("decoder(gguf-granite): experts layer %d: %w", i, errors.Join(ge, ue, de))
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
		return nil
	}
	// Streaming (sink != nil): loadGranite is self-contained per layer (every tensor it reads is blk.{i}.*-named;
	// decoder/gguf_streaming_shape_test.go proves this from source and keeps it true), so the loadQ35/loadGptOss streaming
	// shape applies unchanged.
	if sink != nil {
		if err := streamLayers(w, arch, sink, loadGranite); err != nil {
			return nil, err
		}
		return w, nil
	}
	if err := parallelLayers(arch.NumLayers, abort, loadGranite); err != nil {
		return nil, err
	}
	return w, nil
}

// Nemotron-H (nemotron_h): single-op-per-block hybrid. Each layer is one of
// mamba / attention (NoPE — no q/k permute) / non-gated mlp, under a single
// attn_norm. The ssm_* tensors reuse llama.cpp's conventions (ssm_a = −exp(A_log),
// conv [convDim,K], grouped ssm_norm). Mixer f32; attention/mlp/embeddings quantize.
func (b *ggufBuild) loadNemotron(npar *nemotronParams) (*Weights, error) {
	arch, g, w, quant, sink, abort, hidden, streamMat, mat, vnorm, stackedExperts := b.arch, b.g, b.w, b.quant, b.sink, b.abort, b.hidden, b.streamMat, b.mat, b.vnorm, b.stackedExperts
	dInner := npar.NHeads * npar.HeadDim
	convDim := dInner + 2*npar.NGroups*npar.DState
	projDim := 2*dInner + 2*npar.NGroups*npar.DState + npar.NHeads
	hd := arch.HeadDim
	qDim, kvDim := arch.NumHeads*hd, arch.NumKVHeads*hd
	inter := arch.IntermediateDim
	f32mat := func(name string, out, in int) ([]float32, error) {
		dims, data, derr := g.Tensor(name)
		if derr != nil {
			return nil, derr
		}
		if len(dims) != 2 || dims[0] != in || dims[1] != out {
			return nil, fmt.Errorf("decoder(gguf-nemotron): %q dims %v, want [in=%d, out=%d]", name, dims, in, out)
		}
		return data, nil
	}
	flat := func(name string, n int) ([]float32, error) {
		_, data, derr := g.Tensor(name)
		if derr != nil {
			return nil, derr
		}
		if len(data) != n {
			return nil, fmt.Errorf("decoder(gguf-nemotron): %q has %d elems, want %d", name, len(data), n)
		}
		return data, nil
	}
	loadNemo := func(i int) error {
		l := &w.Layers[i]
		p := fmt.Sprintf("blk.%d.", i)
		var e error
		if l.PreAttnNorm, e = vnorm(p+"attn_norm.weight", hidden); e != nil {
			return e
		}
		switch npar.blockKind[i] {
		case nemoMamba:
			mw := &mamba2Weights{}
			if mw.inProj, e = f32mat(p+"ssm_in.weight", projDim, hidden); e != nil {
				return e
			}
			if mw.convW, e = f32mat(p+"ssm_conv1d.weight", convDim, npar.DConv); e != nil {
				return e
			}
			if mw.convB, e = flat(p+"ssm_conv1d.bias", convDim); e != nil {
				return e
			}
			if mw.d, e = flat(p+"ssm_d", npar.NHeads); e != nil {
				return e
			}
			if mw.dtBias, e = flat(p+"ssm_dt.bias", npar.NHeads); e != nil {
				return e
			}
			if mw.normW, e = flat(p+"ssm_norm.weight", dInner); e != nil {
				return e
			}
			if mw.outProj, e = f32mat(p+"ssm_out.weight", hidden, dInner); e != nil {
				return e
			}
			a, aerr := flat(p+"ssm_a", npar.NHeads)
			if aerr != nil {
				return aerr
			}
			mw.aLog = make([]float32, npar.NHeads)
			for h := range a {
				mw.aLog[h] = float32(math.Log(-float64(a[h])))
			}
			if mambaProjInt8(quant) {
				mw.quantizeProjections(projDim, hidden, dInner)
			}
			l.mamba = mw
		case nemoAttn:
			// NoPE → no q/k permute.
			if l.QProj, e = mat(p+"attn_q.weight", qDim, hidden); e != nil {
				return e
			}
			if l.KProj, e = mat(p+"attn_k.weight", kvDim, hidden); e != nil {
				return e
			}
			if l.VProj, e = mat(p+"attn_v.weight", kvDim, hidden); e != nil {
				return e
			}
			if l.OProj, e = mat(p+"attn_output.weight", hidden, qDim); e != nil {
				return e
			}
		case nemoMLP:
			if l.UpProj, e = mat(p+"ffn_up.weight", inter, hidden); e != nil {
				return e
			}
			if l.DownProj, e = mat(p+"ffn_down.weight", hidden, inter); e != nil {
				return e
			}
		case nemoMoE:
			// Nemotron 3 Nano's MoE FFN. The tensor names are the GGUF's own, not llama.cpp's conversion-script source mapping (which
			// names the source safetensors tensor, not the output GGUF tensor). Experts are fused per projection (one 3-D
			// [in,out,nExpert] tensor each), unlike the safetensors path's one tensor per expert, so stackedExperts, the helper other
			// GGUF MoE families use for this shape, applies. exp_probs_b.bias is llama.cpp's name for e_score_correction_bias.
			moe := arch.MoE
			// Router stays f32; see the Laguna site above.
			if l.Router, e = streamMat(p+"ffn_gate_inp.weight", moe.NumExperts, hidden, quantNone, func(r int) int { return r * hidden }); e != nil {
				return e
			}
			if l.RouterBias, e = flat(p+"exp_probs_b.bias", moe.NumExperts); e != nil {
				return e
			}
			upExp, uerr := stackedExperts(p+"ffn_up_exps.weight", moe.IntermediateDim, hidden, moe.NumExperts)
			if uerr != nil {
				return uerr
			}
			downExp, derr := stackedExperts(p+"ffn_down_exps.weight", hidden, moe.IntermediateDim, moe.NumExperts)
			if derr != nil {
				return derr
			}
			l.Experts = make([]expertWeights, moe.NumExperts)
			for ei := range l.Experts {
				l.Experts[ei].Up, l.Experts[ei].Down = upExp[ei], downExp[ei]
			}
			if moe.SharedIntermediateDim > 0 {
				if l.SharedExpert.Up, e = mat(p+"ffn_up_shexp.weight", moe.SharedIntermediateDim, hidden); e != nil {
					return e
				}
				if l.SharedExpert.Down, e = mat(p+"ffn_down_shexp.weight", hidden, moe.SharedIntermediateDim); e != nil {
					return e
				}
			}
		}
		return nil
	}
	// Streaming (sink != nil): loadNemo is self-contained per layer (every tensor it reads is blk.{i}.*-named;
	// decoder/gguf_streaming_shape_test.go proves this from source and keeps it true), so the loadQ35/loadGptOss streaming
	// shape applies unchanged.
	if sink != nil {
		if err := streamLayers(w, arch, sink, loadNemo); err != nil {
			return nil, err
		}
		return w, nil
	}
	if err := parallelLayers(arch.NumLayers, abort, loadNemo); err != nil {
		return nil, err
	}
	return w, nil
}
