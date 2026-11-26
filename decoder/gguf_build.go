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
	hd                      int
	id                      string
	permMat                 func(name string, out, in, nHead int) (linalg.WeightMat, error)
	cfg                     *Config
	embInto                 func(dst *linalg.WeightMat, name string, out, in int, q quantMode) error
	fusedSplit              func(name string, outTotal, in, rowStart, rows int) (linalg.WeightMat, error)
}

// loadLayers is the loader every family without its own branch takes: the shared per-layer tensor set (attention, MLA or the
// standard q/k/v/o; the dense or MoE feed-forward), loaded in parallel and streamed to the sink when one is set.
func (b *ggufBuild) loadLayers() (*Weights, error) {
	arch, g, w, quant, sink, abort, hidden, hd, streamMat, mat, vec, vnorm, stackedExperts, permMat, fusedSplit := b.arch, b.g, b.w, b.quant, b.sink, b.abort, b.hidden, b.hd, b.streamMat, b.mat, b.vec, b.vnorm, b.stackedExperts, b.permMat, b.fusedSplit
	qDim, kvDim := arch.NumHeads*hd, arch.NumKVHeads*hd
	// llama.cpp permutes the q/k weights (and their per-row biases / head-dim
	// norms) only for the NORM rope type (ggufQKPermuted); the NEOX rope type
	// leaves q/k in HF order, so no un-permutation is needed there.
	permuteQK := ggufQKPermuted(arch.Name)
	loadQK := func(name string, out, nHead int) (linalg.WeightMat, error) {
		if permuteQK {
			return permMat(name, out, hidden, nHead)
		}
		return mat(name, out, hidden)
	}
	// Load the layers in parallel: each is independent (its own linalg.WeightMat slots over the read-only mmap), and the
	// per-tensor dequant and re-quant is the load's cost.
	loadLayer := func(i int) error {
		l := &w.Layers[i]
		p := fmt.Sprintf("blk.%d.", i)
		var err error
		if l.PreAttnNorm, err = vnorm(p+"attn_norm.weight", hidden); err != nil {
			return err
		}
		if arch.mla != nil {
			// DeepSeek MLA: the latent-attention tensor set (q-LoRA / kv down+up + the
			// two latent norms), loaded f32 — not the standard q/k/v/o. The FFN block
			// below is shared with the other DeepSeek-style MoE families.
			if err = loadDeepseekAttnGGUF(g, p, l, arch); err != nil {
				return err
			}
			if mlaProjInt8(quant) {
				l.mla.quantizeProjections(arch)
			}
		} else if arch.Name == "phi3" {
			// Phi-3/Phi-4: fused attn_qkv [qDim+2*kvDim, hidden] → Q ‖ K ‖ V by output
			// rows (NEOX rope ⇒ no permute). o_proj is unfused.
			if l.QProj, err = fusedSplit(p+"attn_qkv.weight", qDim+2*kvDim, hidden, 0, qDim); err != nil {
				return err
			}
			if l.KProj, err = fusedSplit(p+"attn_qkv.weight", qDim+2*kvDim, hidden, qDim, kvDim); err != nil {
				return err
			}
			if l.VProj, err = fusedSplit(p+"attn_qkv.weight", qDim+2*kvDim, hidden, qDim+kvDim, kvDim); err != nil {
				return err
			}
			if l.OProj, err = mat(p+"attn_output.weight", hidden, qDim); err != nil {
				return err
			}
		} else {
			if l.QProj, err = loadQK(p+"attn_q.weight", qDim, arch.NumHeads); err != nil {
				return err
			}
			if l.KProj, err = loadQK(p+"attn_k.weight", kvDim, arch.NumKVHeads); err != nil {
				return err
			}
			if l.VProj, err = mat(p+"attn_v.weight", kvDim, hidden); err != nil {
				return err
			}
			if l.OProj, err = mat(p+"attn_output.weight", hidden, qDim); err != nil {
				return err
			}
			// q/k/v projection bias (Qwen2): the q and k biases are per-output-row, so
			// llama.cpp's RoPE row permutation applies to them exactly as to the q/k
			// weight rows (ggufInvPermute with in=1); the v bias is not permuted.
			if arch.QKVBias {
				qb, qe := vec(p+"attn_q.bias", qDim)
				kb, ke := vec(p+"attn_k.bias", kvDim)
				vb, ve := vec(p+"attn_v.bias", kvDim)
				if qe != nil || ke != nil || ve != nil {
					return fmt.Errorf("decoder(gguf): qkv bias layer %d: %w", i, errors.Join(qe, ke, ve))
				}
				if permuteQK {
					qb = ggufInvPermute(qb, qDim, 1, arch.NumHeads)
					kb = ggufInvPermute(kb, kvDim, 1, arch.NumKVHeads)
				}
				l.QBias, l.KBias, l.VBias = qb, kb, vb
			}
			// QK-norm (Mellum, Qwen3): per-head RMSNorm over head_dim, before RoPE.
			// llama.cpp permutes the q/k weights for its RoPE, so the matching
			// per-head-dim norm weights are un-permuted the same way.
			if arch.QKNorm {
				qn, kerr := vnorm(p+"attn_q_norm.weight", hd)
				if kerr != nil {
					return kerr
				}
				kn, kerr := vnorm(p+"attn_k_norm.weight", hd)
				if kerr != nil {
					return kerr
				}
				if permuteQK {
					qn, kn = ggufInvPermuteVec(qn), ggufInvPermuteVec(kn)
				}
				l.QNorm, l.KNorm = qn, kn
			}
			if arch.hasAttnSinks() {
				if l.AttnSinks, err = vec(p+"attn_sinks.weight", arch.NumHeads); err != nil {
					return err
				}
			}
		}
		// Pre-MLP norm. GLM4_MOE has no ffn_norm — llama.cpp stores its (positionally
		// pre-MLP, Pre2) post_attention_layernorm under the ATTN_POST_NORM name; every
		// other family here uses ffn_norm.
		preMLPNorm := "ffn_norm.weight"
		if arch.Name == "glm4_moe" {
			preMLPNorm = "post_attention_norm.weight"
		}
		if l.PreMLPNorm, err = vnorm(p+preMLPNorm, hidden); err != nil {
			return err
		}
		// Sandwich norms (Gemma 3): a post-attention and a post-FFN RMSNorm in
		// addition to the pre-norms. GGUF names them post_attention_norm /
		// post_ffw_norm.
		if arch.NormPlacement == NormSandwich4 {
			if l.PostAttnNorm, err = vnorm(p+"post_attention_norm.weight", hidden); err != nil {
				return err
			}
			if l.PostMLPNorm, err = vnorm(p+"post_ffw_norm.weight", hidden); err != nil {
				return err
			}
		}
		if arch.MoE != nil && i >= arch.FirstKDense {
			// Sparse MoE: router + stacked per-expert SwiGLU at the narrower
			// moe_intermediate_size. Mellum stops there; GLM/DeepSeek add an
			// e_score_correction_bias (exp_probs_b) and an always-on shared expert
			// (ffn_*_shexp, ungated). GLM's first_k_dense_replace prefix (i < FirstKDense)
			// has no router and falls through to the dense FFN below.
			expInter := arch.MoE.IntermediateDim
			// Router stays f32; see the Laguna site above.
			if l.Router, err = streamMat(p+"ffn_gate_inp.weight", arch.MoE.NumExperts, hidden, quantNone, func(r int) int { return r * hidden }); err != nil {
				return err
			}
			if arch.MoE.RouterSigmoid && arch.Name != "flint" { // DeepSeek/GLM e_score_correction_bias (a bias term; Flint has none)
				if l.RouterBias, err = vec(p+"exp_probs_b.bias", arch.MoE.NumExperts); err != nil {
					return err
				}
			}
			gate, gerr := stackedExperts(p+"ffn_gate_exps.weight", expInter, hidden, arch.MoE.NumExperts)
			up, uerr := stackedExperts(p+"ffn_up_exps.weight", expInter, hidden, arch.MoE.NumExperts)
			down, derr := stackedExperts(p+"ffn_down_exps.weight", hidden, expInter, arch.MoE.NumExperts)
			if gerr != nil || uerr != nil || derr != nil {
				return fmt.Errorf("decoder(gguf): experts layer %d: %w", i, errors.Join(gerr, uerr, derr))
			}
			l.Experts = make([]expertWeights, arch.MoE.NumExperts)
			for e := range arch.MoE.NumExperts {
				l.Experts[e] = expertWeights{Gate: gate[e], Up: up[e], Down: down[e]}
			}
			// Shared always-on expert (GLM: ffn_*_shexp). Ungated for GLM/DeepSeek;
			// Qwen2-MoE-style families would also carry a sigmoid gate (not in GGUF here).
			if arch.MoE.SharedIntermediateDim > 0 {
				sInter := arch.MoE.SharedIntermediateDim
				if l.SharedExpert.Gate, err = mat(p+"ffn_gate_shexp.weight", sInter, hidden); err != nil {
					return err
				}
				if l.SharedExpert.Up, err = mat(p+"ffn_up_shexp.weight", sInter, hidden); err != nil {
					return err
				}
				if l.SharedExpert.Down, err = mat(p+"ffn_down_shexp.weight", hidden, sInter); err != nil {
					return err
				}
				if !arch.MoE.SharedUngated {
					sg, serr := vec(p+"ffn_gate_inp_shexp.weight", hidden)
					if serr != nil {
						return serr
					}
					l.SharedGate = linalg.WrapF32(sg, 1, hidden)
				}
			}
			return nil
		}
		if arch.Name == "phi3" {
			// Phi-3/Phi-4: fused ffn_up [2*inter, hidden] is gate‖up (no separate
			// ffn_gate); ffn_down is unfused.
			inter := arch.IntermediateDim
			if l.GateProj, err = fusedSplit(p+"ffn_up.weight", 2*inter, hidden, 0, inter); err != nil {
				return err
			}
			if l.UpProj, err = fusedSplit(p+"ffn_up.weight", 2*inter, hidden, inter, inter); err != nil {
				return err
			}
			if l.DownProj, err = mat(p+"ffn_down.weight", hidden, inter); err != nil {
				return err
			}
			return nil
		}
		if l.GateProj, err = mat(p+"ffn_gate.weight", arch.IntermediateDim, hidden); err != nil {
			return err
		}
		if l.UpProj, err = mat(p+"ffn_up.weight", arch.IntermediateDim, hidden); err != nil {
			return err
		}
		if l.DownProj, err = mat(p+"ffn_down.weight", hidden, arch.IntermediateDim); err != nil {
			return err
		}
		return nil
	}
	if sink != nil {
		// Build, write and free a window of layers at a time (streamLayers): peak RAM is one window, not the model.
		if err := streamLayers(w, arch, sink, loadLayer); err != nil {
			return nil, err
		}
		return w, nil
	}
	if err := parallelLayers(arch.NumLayers, abort, loadLayer); err != nil {
		return nil, err
	}
	return w, nil
}
