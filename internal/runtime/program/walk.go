package program

import "fmt"

// SublayerOutMode describes how an attention or FFN sublayer joins the residual stream.
type SublayerOutMode uint8

const (
	// OutFusedResidual adds the sublayer's output projection directly into the residual stream.
	OutFusedResidual SublayerOutMode = iota
	// OutPostNormResidual writes the output projection to scratch, applies a post-sublayer norm, and adds into the residual.
	OutPostNormResidual
	// OutParallelScratch writes the attention half of a parallel block into scratch without adding to the residual yet.
	OutParallelScratch
	// OutTwoResidual writes the FFN half of a parallel block to scratch and adds both parallel outputs into the residual.
	OutTwoResidual
)

// FFNInputMode describes how an FFN sublayer prepares its input activation.
type FFNInputMode uint8

const (
	// FFNInNorm applies the sublayer's input norm (RMSNorm or LayerNorm) to the residual stream.
	FFNInNorm FFNInputMode = iota
	// FFNInRaw reads or quantizes the raw residual stream without a norm weight (post-only placement).
	FFNInRaw
	// FFNInShared reuses the shared normed activation computed by a parallel block's attention half.
	FFNInShared
)

// AttentionWalk is the structured parameter and topology view of one SublayerAttention span, decoded from its []Op.
type AttentionWalk struct {
	InputForm    NormForm
	InputNorm    NormParams
	CoreKind     OpKind
	QKNorm       bool
	QKNormParams QKNormParams
	G4VNorm      bool
	Gated        bool
	Rope         bool
	AttnTemp     bool
	OutBias      bool
	OutMode      SublayerOutMode
	PostNorm     NormParams
	Residual     ResidualParams
}

// FFNWalk is the structured parameter and topology view of one FFN sublayer span (SublayerDenseFFN,
// SublayerNonGatedFFN, SublayerMoE, or SublayerGemma4MoE), decoded from its []Op.
type FFNWalk struct {
	Kind      Sublayer
	InputMode FFNInputMode
	InputNorm NormParams
	FFN       FFNParams
	Norms     NormEpsParams
	OutMode   SublayerOutMode
	PostNorm  NormParams
	Residual  ResidualParams
}

// LayerWalk is the decoded sublayer walk of a complete decoder layer's []Op.
type LayerWalk struct {
	Spans          []SublayerSpan
	Mixer          Mixer
	HasAttn        bool
	Attn           AttentionWalk
	MixerInputNorm NormParams
	MixerResidual  ResidualParams
	HasFFN         bool
	FFN            FFNWalk
	HasPLE         bool
	PLENorms       NormEpsParams
	HasLayerScale  bool
}

// DecodeAttentionSublayer decodes ops, the slice opsAll[span.From:span.To] of a SublayerAttention span, into an
// AttentionWalk.
func DecodeAttentionSublayer(ops []Op, parallel bool) (AttentionWalk, error) {
	var w AttentionWalk
	if len(ops) == 0 || ops[0].Kind != OpNorm {
		return w, fmt.Errorf("the attention step does not start with its input norm")
	}
	w.InputNorm = ops[0].Norm
	w.InputForm = ops[0].Norm.Form
	i := 1
	for ; i < len(ops) && ops[i].Kind != OpProjO && w.CoreKind == 0; i++ {
		switch o := &ops[i]; o.Kind {
		case OpProjQKV:
		case OpQKNorm:
			w.QKNorm = true
			w.QKNormParams = o.QKNorm
		case OpRope:
			w.Rope = true
		case OpAttnTemp:
			w.AttnTemp = true
		case OpAttention:
			w.CoreKind = OpAttention
		case OpGatedAttn:
			w.CoreKind = OpGatedAttn
			w.Gated = true
			w.QKNorm = o.Gated.QKNorm.Form != 0
			w.QKNormParams = o.Gated.QKNorm
			w.Rope = len(o.Gated.Rope.InvFreq) > 0
		case OpGemma4Attn:
			w.CoreKind = OpGemma4Attn
			w.QKNorm = true
			w.QKNormParams = QKNormParams{Form: QKRMS, Eps: o.G4.Eps, AddOne: o.G4.AddOne}
			w.G4VNorm = o.G4.OwnsKV
			w.Rope = len(o.G4.InvFreq) > 0
		default:
			return w, fmt.Errorf("the attention step's operation %d is %v, which the walk does not emit there", i, o.Kind)
		}
	}
	if w.CoreKind == 0 {
		return w, fmt.Errorf("the attention step holds no attention operation")
	}
	if w.CoreKind == OpAttention {
		if i >= len(ops) || ops[i].Kind != OpProjO {
			return w, fmt.Errorf("the attention step's attention is not followed by its output projection")
		}
		w.OutBias = ops[i].Proj.Bias
		i++
	}
	if parallel {
		if i != len(ops) {
			return w, fmt.Errorf("the parallel attention step has %d operations past its output projection", len(ops)-i)
		}
		w.OutMode = OutParallelScratch
		return w, nil
	}
	if i < len(ops) && ops[i].Kind == OpNorm {
		w.OutMode = OutPostNormResidual
		w.PostNorm = ops[i].Norm
		i++
	} else {
		w.OutMode = OutFusedResidual
	}
	if i >= len(ops) || ops[i].Kind != OpResidual {
		return w, fmt.Errorf("the attention step's output does not end in its residual add")
	}
	w.Residual = ops[i].Res
	i++
	if i != len(ops) {
		return w, fmt.Errorf("the attention step has %d operations past its residual add", len(ops)-i)
	}
	return w, nil
}

// DecodeFFNSublayer decodes ops, the slice opsAll[span.From:span.To] of an FFN sublayer span, into an FFNWalk.
func DecodeFFNSublayer(kind Sublayer, ops []Op, parallel bool) (FFNWalk, error) {
	w := FFNWalk{Kind: kind}
	if kind == SublayerGemma4MoE {
		if len(ops) != 1 || ops[0].Kind != OpGemma4MoE {
			return w, fmt.Errorf("the %v step has %d operations, want 1 OpGemma4MoE", kind, len(ops))
		}
		w.Norms = ops[0].Norms
		w.OutMode = OutPostNormResidual
		return w, nil
	}
	if len(ops) == 0 {
		return w, fmt.Errorf("the %v step has no operations", kind)
	}
	i := 0
	if parallel {
		w.InputMode = FFNInShared
	} else {
		if ops[0].Kind != OpNorm {
			return w, fmt.Errorf("the %v step does not start with its input norm", kind)
		}
		w.InputNorm = ops[0].Norm
		if ops[0].Norm.Form == NormCopy {
			w.InputMode = FFNInRaw
		} else {
			w.InputMode = FFNInNorm
		}
		i++
	}
	if i >= len(ops) || ops[i].Kind != OpFFN {
		return w, fmt.Errorf("the %v step holds no FFN operation after its input", kind)
	}
	w.FFN = ops[i].FFN
	i++
	if i < len(ops) && ops[i].Kind == OpNorm {
		if parallel {
			return w, fmt.Errorf("the parallel %v step carries a post-MLP norm", kind)
		}
		w.OutMode = OutPostNormResidual
		w.PostNorm = ops[i].Norm
		i++
	} else if parallel {
		w.OutMode = OutTwoResidual
	} else {
		w.OutMode = OutFusedResidual
	}
	if i >= len(ops) || ops[i].Kind != OpResidual {
		return w, fmt.Errorf("the %v step does not end in its residual add", kind)
	}
	w.Residual = ops[i].Res
	if w.Residual.Two != parallel {
		return w, fmt.Errorf("the %v step's residual add has Two=%v, want %v", kind, w.Residual.Two, parallel)
	}
	i++
	if i != len(ops) {
		return w, fmt.Errorf("the %v step has %d operations past its residual add", kind, len(ops)-i)
	}
	return w, nil
}

// DecodeLayerWalk partitions ops into sublayers for backend, checks sublayer claims, and decodes every sublayer into a
// structured LayerWalk.
func DecodeLayerWalk(ops []Op, backend string) (LayerWalk, error) {
	spans, err := SublayersOf(ops, backend)
	if err == nil {
		err = CheckSublayerClaims(ops, spans)
	}
	if err != nil {
		return LayerWalk{}, err
	}
	lw := LayerWalk{Spans: spans, Mixer: MixerNone}
	for _, sp := range spans {
		subOps := ops[sp.From:sp.To]
		switch sp.Kind {
		case SublayerAttention:
			aw, err := DecodeAttentionSublayer(subOps, sp.Parallel)
			if err != nil {
				return LayerWalk{}, err
			}
			lw.Mixer = MixerAttention
			lw.HasAttn = true
			lw.Attn = aw
		case SublayerDeltaNet, SublayerMamba, SublayerMLA:
			if len(subOps) != 3 || subOps[0].Kind != OpNorm || subOps[2].Kind != OpResidual {
				return LayerWalk{}, fmt.Errorf("the %v step has %d operations, want input norm, mixer, residual", sp.Kind, len(subOps))
			}
			switch sp.Kind {
			case SublayerDeltaNet:
				lw.Mixer = MixerDeltaNet
			case SublayerMamba:
				lw.Mixer = MixerMamba
			case SublayerMLA:
				lw.Mixer = MixerMLA
			}
			lw.MixerInputNorm = subOps[0].Norm
			lw.MixerResidual = subOps[2].Res
		case SublayerDenseFFN, SublayerNonGatedFFN, SublayerMoE, SublayerGemma4MoE:
			fw, err := DecodeFFNSublayer(sp.Kind, subOps, sp.Parallel)
			if err != nil {
				return LayerWalk{}, err
			}
			lw.HasFFN = true
			lw.FFN = fw
		case SublayerPLE:
			lw.HasPLE = true
			lw.PLENorms = subOps[0].Norms
		case SublayerLayerScale:
			lw.HasLayerScale = true
		}
	}
	return lw, nil
}
