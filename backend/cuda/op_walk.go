//go:build cuda

package cuda

import (
	"fmt"

	gpu "github.com/townsendmerino/goinfer/backend/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// attnOutUnfused reports whether layer L's attention output projection writes to scratch rather than accumulating directly into
// the residual stream (a post-attention norm or a parallel block's attention half).
func (L *cudaLayer) attnOutUnfused() bool {
	return L.walk.HasAttn && L.walk.Attn.OutMode != program.OutFusedResidual
}

// routedFFN reports whether layer L's FFN sublayer dispatches routed experts across a segB/segC gap (generic MoE or Gemma 4's
// parallel dense and MoE block).
func (L *cudaLayer) routedFFN() bool {
	return L.walk.HasFFN && (L.walk.FFN.Kind == program.SublayerMoE || L.walk.FFN.Kind == program.SublayerGemma4MoE)
}

// emitInputNorm launches the single-row input normalisation form on src with weight nrm into quantized activation (qOut, sOut).
func (r *cudaResident) emitInputNorm(form program.NormForm, src, nrm, qOut, sOut Buffer) error {
	switch form {
	case program.NormCopy:
		return r.launch(r.fQ, onecfg(256, 256*4), Arg(src), gpu.ArgValue(int32(r.hidden)), Arg(qOut), Arg(sOut))
	case program.NormLayer:
		return r.layerNormQuant(src, nrm, qOut, sOut)
	default:
		return r.rms(src, nrm, qOut, sOut)
	}
}

// emitInputNormB launches the M-row batched input normalisation form on x with weight w into quantized activation (qOut, sOut).
func (r *cudaResident) emitInputNormB(form program.NormForm, x, w Buffer, N int, qOut, sOut Buffer, M int) error {
	switch form {
	case program.NormCopy:
		return r.launch(r.bQuant, LaunchConfig{GridX: uint32(M), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
			Arg(x), gpu.ArgValue(int32(N)), Arg(qOut), Arg(sOut), gpu.ArgValue(int32(M)))
	case program.NormLayer:
		return r.bLayerNormQuantB(x, w, N, M, qOut, sOut)
	default:
		return r.bRmsB(x, w, N, qOut, sOut, M)
	}
}

// checkCUDALayerWalk decodes ops into Ly.walk and verifies that every sublayer's decoded norm forms, epsilons, (1 + w) offsets,
// QK-norm, output mode, FFN form, PLE branch, and layer scalar match the buffers and kernel configuration uploaded for Ly.
func (r *cudaResident) checkCUDALayerWalk(Ly *cudaLayer, ops []program.Op) error {
	lw, err := program.DecodeLayerWalk(ops, "CUDA")
	if err != nil {
		return err
	}
	Ly.walk = lw
	checkNorm := func(sub string, n *program.NormParams, slot program.NormSlot) error {
		if n.Slot != slot {
			return fmt.Errorf("the %s norm reads slot %v, want %v", sub, n.Slot, slot)
		}
		if n.Form == program.NormCopy {
			return nil
		}
		if (n.Form == program.NormLayer) != r.layerNorm {
			return fmt.Errorf("the %s norm has form %v, and the resident's LayerNorm is %v", sub, n.Form, r.layerNorm)
		}
		if float32(n.Eps) != r.eps || n.AddOne != r.rmsAddOne {
			return fmt.Errorf("the %s norm has epsilon %g and (1 + w) %v, and the resident holds (%g, %v)",
				sub, n.Eps, n.AddOne, r.eps, r.rmsAddOne)
		}
		return nil
	}
	checkRes := func(sub string, res *program.ResidualParams, wantTwo bool) error {
		if res.Two != wantTwo || res.Scale != 0 {
			return fmt.Errorf("the %s residual has Two=%v (want %v) and scale %g", sub, res.Two, wantTwo, res.Scale)
		}
		return nil
	}
	if lw.HasAttn {
		aw := &lw.Attn
		if err := checkNorm("attention input", &aw.InputNorm, program.SlotPreAttn); err != nil {
			return err
		}
		if (aw.InputForm == program.NormCopy) != r.postOnly {
			return fmt.Errorf("the attention input norm is %v, and postOnly is %v", aw.InputForm, r.postOnly)
		}
		if aw.InputForm != program.NormCopy && Ly.preNorm == (Buffer{}) {
			return fmt.Errorf("the attention input norm requires preNorm, which is empty")
		}
		if aw.Gated != Ly.qGate {
			return fmt.Errorf("the attention step is gated %v, and the layer's qGate is %v", aw.Gated, Ly.qGate)
		}
		if aw.QKNorm != r.qkNorm {
			return fmt.Errorf("the attention step has QK-norm %v, and the resident's qkNorm is %v", aw.QKNorm, r.qkNorm)
		}
		if aw.QKNorm {
			if aw.QKNormParams.Whole != r.qkNormWhole {
				return fmt.Errorf("the attention step's QK-norm is whole %v, and the resident's is %v", aw.QKNormParams.Whole, r.qkNormWhole)
			}
			if float32(aw.QKNormParams.Eps) != r.eps || aw.QKNormParams.AddOne != r.rmsAddOne {
				return fmt.Errorf("the attention step's QK-norm has (%g, %v), and the resident holds (%g, %v)",
					aw.QKNormParams.Eps, aw.QKNormParams.AddOne, r.eps, r.rmsAddOne)
			}
		}
		if aw.CoreKind == program.OpAttention && aw.OutBias != Ly.hasOBias {
			return fmt.Errorf("the attention output projection has bias %v, and the layer uploaded %v", aw.OutBias, Ly.hasOBias)
		}
		switch aw.OutMode {
		case program.OutParallelScratch:
			if !r.parallelBlock {
				return fmt.Errorf("the attention step ends in parallel scratch on a non-parallel build")
			}
		case program.OutPostNormResidual:
			if err := checkNorm("post-attention", &aw.PostNorm, program.SlotPostAttn); err != nil {
				return err
			}
			if Ly.postAttnNorm == (Buffer{}) {
				return fmt.Errorf("the post-attention norm buffer is empty")
			}
			if err := checkRes("attention", &aw.Residual, false); err != nil {
				return err
			}
		case program.OutFusedResidual:
			if err := checkRes("attention", &aw.Residual, false); err != nil {
				return err
			}
		}
	} else if lw.Mixer != program.MixerNone {
		if err := checkNorm("mixer input", &lw.MixerInputNorm, program.SlotPreAttn); err != nil {
			return err
		}
		if lw.MixerInputNorm.Form == program.NormCopy || Ly.preNorm == (Buffer{}) {
			return fmt.Errorf("the %v mixer requires a pre-attention norm buffer", lw.Mixer)
		}
		if err := checkRes("mixer", &lw.MixerResidual, false); err != nil {
			return err
		}
	}
	if lw.HasFFN {
		fw := &lw.FFN
		if fw.Kind == program.SublayerGemma4MoE {
			if !Ly.g4moe || Ly.g4preFFN == (Buffer{}) || Ly.g4postFFN == (Buffer{}) {
				return fmt.Errorf("the Gemma 4 MoE step's layer holds incomplete Gemma 4 MoE buffers")
			}
			if float32(fw.Norms.Eps) != r.eps || fw.Norms.AddOne != r.rmsAddOne {
				return fmt.Errorf("the Gemma 4 MoE step's norms have (%g, %v), and the resident holds (%g, %v)",
					fw.Norms.Eps, fw.Norms.AddOne, r.eps, r.rmsAddOne)
			}
		} else {
			if (fw.InputMode == program.FFNInShared) != r.parallelBlock {
				return fmt.Errorf("the %v step's input mode is %d, and parallelBlock is %v", fw.Kind, fw.InputMode, r.parallelBlock)
			}
			if fw.InputMode != program.FFNInShared {
				slot := program.SlotPreMLP
				if lw.Mixer == program.MixerNone {
					slot = program.SlotPreAttn
				}
				if err := checkNorm("FFN input", &fw.InputNorm, slot); err != nil {
					return err
				}
				normBuf := Ly.postNorm
				if slot == program.SlotPreAttn {
					normBuf = Ly.preNorm
				}
				if fw.InputMode == program.FFNInNorm && normBuf == (Buffer{}) {
					return fmt.Errorf("the %v step's input norm buffer for slot %v is empty", fw.Kind, slot)
				}
			}
			switch fw.Kind {
			case program.SublayerMoE:
				if !Ly.isMoE {
					return fmt.Errorf("the MoE step's layer has isMoE=false")
				}
				if (fw.FFN.Form == program.FfnGptOss) != (r.gptOssRoute != (Pipeline{})) {
					return fmt.Errorf("the MoE step's form is %v, and gptOssRoute loaded is %v", fw.FFN.Form, r.gptOssRoute != (Pipeline{}))
				}
				if (fw.FFN.MoE.SharedIntermediateDim > 0) != Ly.hasShared {
					return fmt.Errorf("the MoE step's shared expert width is %d, and hasShared is %v", fw.FFN.MoE.SharedIntermediateDim, Ly.hasShared)
				}
				if Ly.hasShared && fw.FFN.MoE.SharedUngated != (Ly.shGateW.W == (Buffer{})) {
					return fmt.Errorf("the MoE step's SharedUngated is %v, and shGateW empty is %v", fw.FFN.MoE.SharedUngated, Ly.shGateW.W == (Buffer{}))
				}
				if fw.OutMode != program.OutFusedResidual {
					return fmt.Errorf("the MoE step's output mode is %d, want fused residual", fw.OutMode)
				}
				if err := checkRes("MoE", &fw.Residual, false); err != nil {
					return err
				}
			case program.SublayerNonGatedFFN:
				if fw.FFN.Form != program.FfnNonGated || fw.FFN.Act != program.ActReLU2 {
					return fmt.Errorf("the non-gated FFN step has (%v, %v), want (non-gated, ReLU2)", fw.FFN.Form, fw.FFN.Act)
				}
				if fw.OutMode != program.OutFusedResidual {
					return fmt.Errorf("the non-gated FFN step's output mode is %d, want fused residual", fw.OutMode)
				}
				if err := checkRes("non-gated FFN", &fw.Residual, false); err != nil {
					return err
				}
			case program.SublayerDenseFFN:
				if int32(fw.FFN.Act) != r.act {
					return fmt.Errorf("the dense FFN step's activation is %v (%d), and the resident holds %d", fw.FFN.Act, int32(fw.FFN.Act), r.act)
				}
				switch fw.OutMode {
				case program.OutTwoResidual:
					if err := checkRes("parallel FFN", &fw.Residual, true); err != nil {
						return err
					}
				case program.OutPostNormResidual:
					if err := checkNorm("post-MLP", &fw.PostNorm, program.SlotPostMLP); err != nil {
						return err
					}
					if Ly.postMLPNorm == (Buffer{}) {
						return fmt.Errorf("the post-MLP norm buffer is empty")
					}
					if err := checkRes("dense FFN", &fw.Residual, false); err != nil {
						return err
					}
				case program.OutFusedResidual:
					if err := checkRes("dense FFN", &fw.Residual, false); err != nil {
						return err
					}
				}
			}
		}
	}
	if lw.HasPLE {
		if r.pleP <= 0 || Ly.pleGate.N == 0 {
			return fmt.Errorf("the PLE step's layer holds no PLE weights (pleP=%d, gate.N=%d)", r.pleP, Ly.pleGate.N)
		}
		if float32(lw.PLENorms.Eps) != r.eps || lw.PLENorms.AddOne != r.rmsAddOne {
			return fmt.Errorf("the PLE step's norms have (%g, %v), and the resident holds (%g, %v)",
				lw.PLENorms.Eps, lw.PLENorms.AddOne, r.eps, r.rmsAddOne)
		}
	}
	return nil
}
