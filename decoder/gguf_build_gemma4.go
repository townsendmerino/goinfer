package decoder

import (
	"fmt"

	"github.com/townsendmerino/aikit/linalg"
)

// Gemma 4 (E-models): a dedicated layer loader. Per-layer head_dim/FFN, the
// global/sliding split, the cross-layer KV-shared tail (layers ≥ first-shared
// carry no k/v), and the PLE branch (inp_gate/proj/post_norm + layer scalar).
// The model-level PLE inputs (per_layer_*) load here. NEOX rope ⇒ no q/k
// permute, so plain mat() throughout.
func (b *ggufBuild) loadGemma4(g4 *gemma4Params) (*Weights, error) {
	arch, g, w, quant, sink, abort, hidden, cfg, streamMat, mat, vec, vnorm, stackedExperts, embInto, id := b.arch, b.g, b.w, b.quant, b.sink, b.abort, b.hidden, b.cfg, b.streamMat, b.mat, b.vec, b.vnorm, b.stackedExperts, b.embInto, b.id
	var err error
	if g4.HiddenSizePerLayerInput > 0 {
		pleTotal := arch.NumLayers * g4.HiddenSizePerLayerInput
		if err = embInto(&w.PerLayerTokenEmbed, "per_layer_token_embd.weight", cfg.VocabSize, pleTotal, quant.gatherTable()); err != nil {
			return nil, err
		}
		if w.PerLayerModelProj, err = mat("per_layer_model_proj.weight", pleTotal, hidden); err != nil {
			return nil, err
		}
		if w.PerLayerProjNorm, err = vnorm("per_layer_proj_norm.weight", g4.HiddenSizePerLayerInput); err != nil {
			return nil, err
		}
	}
	firstShared := arch.NumLayers - g4.SharedKVLayers
	loadG4 := func(i int) error {
		l := &w.Layers[i]
		p := fmt.Sprintf("blk.%d.", i)
		hdL := arch.headDimAt(i)
		qDimL, kvDimL, ffn := arch.NumHeads*hdL, arch.kvHeadsAt(i)*hdL, arch.ffnAt(i)
		var e error
		if l.PreAttnNorm, e = vnorm(p+"attn_norm.weight", hidden); e != nil {
			return e
		}
		if l.PostAttnNorm, e = vnorm(p+"post_attention_norm.weight", hidden); e != nil {
			return e
		}
		if l.PreMLPNorm, e = vnorm(p+"ffn_norm.weight", hidden); e != nil {
			return e
		}
		if l.PostMLPNorm, e = vnorm(p+"post_ffw_norm.weight", hidden); e != nil {
			return e
		}
		if l.QProj, e = mat(p+"attn_q.weight", qDimL, hidden); e != nil {
			return e
		}
		if l.QNorm, e = vnorm(p+"attn_q_norm.weight", hdL); e != nil {
			return e
		}
		l.KVShared = i >= firstShared
		if !l.KVShared {
			if l.KProj, e = mat(p+"attn_k.weight", kvDimL, hidden); e != nil {
				return e
			}
			if l.KNorm, e = vnorm(p+"attn_k_norm.weight", hdL); e != nil {
				return e
			}
			// attention_k_eq_v (12B global layers): no v_proj — V reuses K's
			// projection. Detect by the tensor's absence (robust to whether the
			// converter omits or duplicates it).
			if g.Has(p + "attn_v.weight") {
				if l.VProj, e = mat(p+"attn_v.weight", kvDimL, hidden); e != nil {
					return e
				}
			} else {
				l.VFromK = true
			}
		}
		if l.OProj, e = mat(p+"attn_output.weight", hidden, qDimL); e != nil {
			return e
		}
		if l.GateProj, e = mat(p+"ffn_gate.weight", ffn, hidden); e != nil {
			return e
		}
		if l.UpProj, e = mat(p+"ffn_up.weight", ffn, hidden); e != nil {
			return e
		}
		if l.DownProj, e = mat(p+"ffn_down.weight", hidden, ffn); e != nil {
			return e
		}
		// The per-layer output scalar is read BEFORE the MoE branch below, which copies it into gemma4MoEWeights.layerScalar, the
		// value the MoE forward multiplies the whole layer's output by (forward_gemma4_moe.go). Read after that copy, every MoE
		// layer of a directly loaded GGUF would be scaled by 0 and the model would emit only <pad>; the .giw reader and the
		// safetensors loader also read it first (TestGemma4GGUF_moeLayerScalarMatchesLayer).
		l.LayerScalar = 1
		if sc, se := vec(p+"layer_output_scale.weight", 1); se == nil {
			l.LayerScalar = sc[0]
		}
		// 26B-A4B's parallel MoE branch (enable_moe_block; nil arch.MoE ⇒ the dense
		// E2B/E4B/12B GGUFs, byte-unchanged). llama.cpp's converter fuses gate‖up per
		// expert exactly like the safetensors path (ffn_gate_up_exps.weight is a 3-D
		// [hidden, 2*moeInter, nE] blob, structurally identical to stackedExperts'
		// existing [in,out,nExpert] contract — no new tensor-reading helper needed),
		// and stores the router's weightless-RMSNorm learned scale + the per-expert
		// scale as ffn_gate_inp.scale/ffn_down_exps.scale rather than a dequant
		// artifact (both real model parameters — routerScale/perExpertScale below).
		if arch.MoE != nil {
			nE, moeInter := arch.MoE.NumExperts, arch.MoE.IntermediateDim
			gm := &gemma4MoEWeights{
				preFFNNorm:  l.PreMLPNorm,
				postFFNNorm: l.PostMLPNorm,
				mlpGate:     l.GateProj,
				mlpUp:       l.UpProj,
				mlpDown:     l.DownProj,
				layerScalar: l.LayerScalar,
				denseInter:  ffn,
				moeInter:    moeInter,
				nE:          nE,
				topK:        arch.MoE.TopK,
			}
			if gm.postFFNNorm1, e = vnorm(p+"post_ffw_norm_1.weight", hidden); e != nil {
				return e
			}
			if gm.preFFNNorm2, e = vnorm(p+"pre_ffw_norm_2.weight", hidden); e != nil {
				return e
			}
			if gm.postFFNNorm2, e = vnorm(p+"post_ffw_norm_2.weight", hidden); e != nil {
				return e
			}
			// Router stays f32 (see the Laguna site above); the safetensors Gemma4 loader does the same.
			if gm.routerProj, e = streamMat(p+"ffn_gate_inp.weight", nE, hidden, quantNone, func(r int) int { return r * hidden }); e != nil {
				return e
			}
			if gm.routerScale, e = vec(p+"ffn_gate_inp.scale", hidden); e != nil {
				return e
			}
			if gm.perExpertScale, e = vec(p+"ffn_down_exps.scale", nE); e != nil {
				return e
			}
			if gm.expertsGateUp, e = stackedExperts(p+"ffn_gate_up_exps.weight", 2*moeInter, hidden, nE); e != nil {
				return e
			}
			if gm.expertsDown, e = stackedExperts(p+"ffn_down_exps.weight", hidden, moeInter, nE); e != nil {
				return e
			}
			l.gemma4moe = gm
		}
		if g4.HiddenSizePerLayerInput > 0 {
			if l.PLEGate, e = mat(p+"inp_gate.weight", g4.HiddenSizePerLayerInput, hidden); e != nil {
				return e
			}
			if l.PLEProj, e = mat(p+"proj.weight", hidden, g4.HiddenSizePerLayerInput); e != nil {
				return e
			}
			if l.PostPLENorm, e = vnorm(p+"post_norm.weight", hidden); e != nil {
				return e
			}
		}
		return nil
	}
	// Streaming (sink != nil): the model-level PLE inputs (per_layer_token_embd / per_layer_model_proj / per_layer_proj_norm,
	// plus FFNPerLayer) are written in the head (writeHeadGlobals) before any layer and loaded above before loadG4 runs.
	// Everything else gemma4-specific (PLE gate/proj/norm, the layer scalar, the KV-shared and K=V flags, the whole 26B-A4B
	// MoE branch) lives in each layer's own record (giwWriter.gemma4Layer) and is read from blk.{i}.* only
	// (decoder/gguf_streaming_shape_test.go keeps that true). So: head once the PLE globals exist, then streamLayers'
	// build, write and release. Non-streaming (sink == nil): all layers in parallel.
	if sink != nil {
		if err := sink.writeHeadGlobals(w, id); err != nil {
			return nil, err
		}
		// The head is on disk and nothing below reads these again (the returned *Weights of a streaming build is discarded; see
		// this function's doc), so release them before the layers: holding them is a flat floor under the whole stream.
		w.Embed, w.LMHead, w.PosEmbed = linalg.WeightMat{}, linalg.WeightMat{}, linalg.WeightMat{}
		w.PerLayerTokenEmbed, w.PerLayerModelProj = linalg.WeightMat{}, linalg.WeightMat{}
		if err := streamLayers(w, arch, sink, loadG4); err != nil {
			return nil, err
		}
		return w, nil
	}
	if err = parallelLayers(arch.NumLayers, abort, loadG4); err != nil {
		return nil, err
	}
	// K=V (attention_k_eq_v) on the 12B global layers: V reuses K's projection (v_norm(k_proj); see loadG4/the Gemma 4 program walk).
	// Parity-gated against the HF bf16 oracle by TestGemma4_12B_logitParity.
	return w, nil
}
