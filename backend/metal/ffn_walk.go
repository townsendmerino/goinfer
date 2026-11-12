package metal

import (
	"fmt"

	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// ffnWalkPlantForTest plants test-only defects in the FFN and tail step walk: ffnWalkPlantResidual emits an extra residual add
// after an FFN projection that already accumulates into the residual; ffnWalkPlantDropPostNorm skips the post-MLP norm on a
// sandwich or post-only layer; ffnWalkPlantFreshParallelNorm computes a fresh norm of the post-attention residual instead of
// reusing the parallel block's shared input norm.
var ffnWalkPlantForTest int

const (
	ffnWalkPlantResidual = iota + 1
	ffnWalkPlantDropPostNorm
	ffnWalkPlantFreshParallelNorm
)

// ffnWalks reports whether encodeFFNWalk emits step st from its operations.
func ffnWalks(st layerStep) bool {
	switch st {
	case stepDenseFFN, stepNonGatedFFN, stepMoE, stepGemma4MoE, stepPLE, stepLayerScale:
		return true
	}
	return false
}

// stepWalks reports whether the decode walk emits step st from its operations.
func stepWalks(st layerStep) bool {
	return mixerWalks(st) || ffnWalks(st)
}

// ffnNormWeights returns layer L's norm weight and LayerNorm bias buffers for slot.
func (r *resident) ffnNormWeights(L *residLayer, slot program.NormSlot) (Buffer, Buffer) {
	if slot == program.SlotPreAttn {
		return L.preNorm, L.preNormBias
	}
	return L.postNorm, L.postNormBias
}

// checkFFNWalk checks that layer L's step sp and its operations ops match the weights and uniforms the resident built, or nil
// when st is not an FFN or tail step.
func (r *resident) checkFFNWalk(L *residLayer, sp stepSpan, ops []program.Op) error {
	st := sp.Kind
	if !ffnWalks(st) {
		return nil
	}
	switch st {
	case stepPLE:
		if len(ops) != 1 || ops[0].Kind != program.OpPLE || sp.Parallel {
			return fmt.Errorf("the %v step has %d operations (parallel %v), want 1 OpPLE", st, len(ops), sp.Parallel)
		}
		if L.pleGW == (Buffer{}) {
			return fmt.Errorf("the %v step's layer holds no PLE gate weights", st)
		}
		return r.checkWalkNormEps(st, ops[0].Norms)
	case stepLayerScale:
		if len(ops) != 1 || ops[0].Kind != program.OpLayerScale || sp.Parallel {
			return fmt.Errorf("the %v step has %d operations (parallel %v), want 1 OpLayerScale", st, len(ops), sp.Parallel)
		}
		if !L.layerScaleRead {
			return fmt.Errorf("the %v step's layer did not read an output scalar at build", st)
		}
		return nil
	}
	fw, err := program.DecodeFFNSublayer(st, ops, sp.Parallel)
	if err != nil {
		return err
	}
	if sp.Parallel != r.parallelBlock {
		return fmt.Errorf("the %v step is parallel %v, and the build's layers are %v", st, sp.Parallel, r.parallelBlock)
	}
	if st == stepGemma4MoE {
		if L.g4moe == nil {
			return fmt.Errorf("the %v step's layer holds no Gemma 4 MoE weights", st)
		}
		return r.checkWalkNormEps(st, fw.Norms)
	}
	if !sp.Parallel {
		slot := program.SlotPreMLP
		if sp.From == 0 {
			slot = program.SlotPreAttn
		}
		if err := r.checkWalkNorm(st, &fw.InputNorm, slot); err != nil {
			return err
		}
		if w, _ := r.ffnNormWeights(L, slot); fw.InputNorm.Form != program.NormCopy && w == (Buffer{}) {
			return fmt.Errorf("the %v step's input norm binds slot %v, whose weight buffer is empty", st, slot)
		}
	}
	switch st {
	case stepMoE:
		if L.moe == nil || r.moe == nil {
			return fmt.Errorf("the %v step's layer holds no MoE weights", st)
		}
		if fw.InputMode != program.FFNInNorm || fw.InputNorm.Form != program.NormRMS {
			return fmt.Errorf("the %v step's input norm has form %v, want RMSNorm", st, fw.InputNorm.Form)
		}
		if fw.FFN.Form != program.FfnMoE && fw.FFN.Form != program.FfnGptOss {
			return fmt.Errorf("the %v step's FFN form is %v, want MoE or gpt-oss MoE", st, fw.FFN.Form)
		}
		if (fw.FFN.Form == program.FfnGptOss) != r.moe.isGptOss {
			return fmt.Errorf("the %v step's FFN form is %v, and the build's gpt-oss router is %v", st, fw.FFN.Form, r.moe.isGptOss)
		}
		if fw.OutMode != program.OutFusedResidual {
			return fmt.Errorf("the %v step's output mode is %d, want fused residual", st, fw.OutMode)
		}
		return checkWalkResidual(st, &fw.Residual)
	case stepNonGatedFFN:
		if !r.nonGatedMLP {
			return fmt.Errorf("the %v step's resident was not built for a non-gated MLP", st)
		}
		if fw.InputMode != program.FFNInNorm || (fw.InputNorm.Form == program.NormLayer) != r.layerNorm {
			return fmt.Errorf("the %v step's input norm has form %v, and the build's LayerNorm is %v", st, fw.InputNorm.Form, r.layerNorm)
		}
		if fw.FFN.Form != program.FfnNonGated || uint32(fw.FFN.Act) != r.decode.act {
			return fmt.Errorf("the %v step's FFN is (%v, %v), and the build's activation is %d", st, fw.FFN.Form, fw.FFN.Act, r.decode.act)
		}
		if fw.OutMode != program.OutFusedResidual {
			return fmt.Errorf("the %v step's output mode is %d, want fused residual", st, fw.OutMode)
		}
		return checkWalkResidual(st, &fw.Residual)
	case stepDenseFFN:
		if fw.FFN.Form != program.FfnGated && fw.FFN.Form != program.FfnGemma4Dense {
			return fmt.Errorf("the %v step's FFN form is %v, want a gated dense FFN", st, fw.FFN.Form)
		}
		if uint32(fw.FFN.Act) != r.decode.act {
			return fmt.Errorf("the %v step's activation is %v (%d), and the resident's uniform holds %d", st, fw.FFN.Act, uint32(fw.FFN.Act), r.decode.act)
		}
		postOnlyHere := r.postOnly && L.geom != nil
		if !sp.Parallel {
			wantIn := program.NormRMS
			switch {
			case postOnlyHere:
				wantIn = program.NormCopy
			case r.layerNorm:
				wantIn = program.NormLayer
			}
			if fw.InputNorm.Form != wantIn {
				return fmt.Errorf("the %v step's input norm has form %v, and the build's layer takes %v", st, fw.InputNorm.Form, wantIn)
			}
		}
		if r.q4kLane && (fw.InputMode != program.FFNInNorm || fw.InputNorm.Form != program.NormRMS || fw.OutMode != program.OutFusedResidual) {
			return fmt.Errorf("the %v step on the q4k lane requires pre-RMSNorm and fused residual, got input %d (%v) and output %d",
				st, fw.InputMode, fw.InputNorm.Form, fw.OutMode)
		}
		switch {
		case sp.Parallel:
			if fw.OutMode != program.OutTwoResidual || fw.Residual.Scale != 0 {
				return fmt.Errorf("the parallel %v step's output is mode %d at scale %g", st, fw.OutMode, fw.Residual.Scale)
			}
			return nil
		case r.sandwich || postOnlyHere:
			if fw.OutMode != program.OutPostNormResidual {
				return fmt.Errorf("the %v step's output mode is %d, and the build's layer takes a post-MLP norm", st, fw.OutMode)
			}
			if L.postMLPNorm == (Buffer{}) {
				return fmt.Errorf("the %v step's post-MLP norm buffer is empty", st)
			}
			if err := r.checkWalkNorm(st, &fw.PostNorm, program.SlotPostMLP); err != nil {
				return err
			}
			return checkWalkResidual(st, &fw.Residual)
		default:
			if fw.OutMode != program.OutFusedResidual {
				return fmt.Errorf("the %v step's output mode is %d, want fused residual", st, fw.OutMode)
			}
			return checkWalkResidual(st, &fw.Residual)
		}
	}
	return nil
}

// checkWalkNormEps checks the epsilon and (1 + w) of a composite operation's internal norms against the resident's uniforms.
func (r *resident) checkWalkNormEps(st layerStep, n program.NormEpsParams) error {
	if float32(n.Eps) != r.decode.normEps || n.AddOne != (r.decode.addOne == 1) {
		return fmt.Errorf("the %v step's norms have epsilon %g and (1 + w) %v, which the resident's norm uniforms (%g, %v) do not hold",
			st, n.Eps, n.AddOne, r.decode.normEps, r.decode.addOne == 1)
	}
	return nil
}

// emitFFNNorm emits norm n of x into the quantized activation (mq, mSc): a raw-residual quantize for NormCopy, a LayerNorm, or
// an RMSNorm.
func (r *resident) emitFFNNorm(e *Encoder, n *program.NormParams, x, w, bias, mq, mSc Buffer) {
	switch n.Form {
	case program.NormCopy:
		e.Dispatch(r.pQv, 256, 256, x, mq, mSc, r.uH)
	case program.NormLayer:
		e.Dispatch(r.pLayerNorm, tgReduceNorm, tgReduceNorm, x, w, bias, mq, mSc, r.uH, r.uEps, r.uLNHasBias)
	default:
		e.Dispatch(r.pRms, tgReduceNorm, tgReduceNorm, x, w, mq, mSc, r.uH, r.uEps, r.uAddOne)
	}
}

// encodeFFNWalk emits layer l's FFN or tail step st from its operations ops, which checkFFNWalk admitted, on residual x.
func (r *resident) encodeFFNWalk(e *Encoder, st layerStep, l int, L *residLayer, ops []program.Op, x Buffer) {
	if st == stepMoE && L.moe.pool != nil {
		e.FinishEncoding()
		panic("metal: encodeMoEFFNWithX reached a paged MoE layer — route through forwardLogitsMoEPaged instead (C-02)")
	}
	f16Lane := st == stepDenseFFN && r.canUseF16Lane(l)
	gq, gSc := r.mq, r.mSc
	if st == stepDenseFFN && ops[0].Kind != program.OpNorm {
		if ffnWalkPlantForTest == ffnWalkPlantFreshParallelNorm {
			r.emitFFNNorm(e, &program.NormParams{Form: r.attnInputForm()}, x, L.preNorm, L.preNormBias, r.mq, r.mSc)
		} else {
			gq, gSc = r.aq, r.aSc
		}
	}
	for i := 0; i < len(ops); i++ {
		switch o := &ops[i]; o.Kind {
		case program.OpNorm:
			if i == 0 {
				w, bias := r.ffnNormWeights(L, o.Norm.Slot)
				switch {
				case st == stepDenseFFN && r.q4kLane:
					e.Dispatch(r.pRmsF32Out, tgReduceNorm, tgReduceNorm, x, w, r.axF32, r.uH, r.uEps, r.uAddOne)
				case f16Lane && o.Norm.Form == program.NormRMS:
					e.Dispatch(r.pRmsF16, tgReduceNorm, tgReduceNorm, x, w, r.mxF16, r.uH, r.uEps, r.uAddOne)
				default:
					r.emitFFNNorm(e, &o.Norm, x, w, bias, r.mq, r.mSc)
				}
				continue
			}
			if ffnWalkPlantForTest != ffnWalkPlantDropPostNorm {
				e.Dispatch(r.pRmsF32, tgReduceNorm, tgReduceNorm, r.dO, L.postMLPNorm, r.uH, r.uEps, r.uAddOne)
			}
		case program.OpFFN:
			switch o.FFN.Form {
			case program.FfnMoE, program.FfnGptOss:
				r.encodeMoERouteForForm(e, L, o.FFN.Form)
				r.encodeMoEExperts(e, L, x)
				if r.moeCap.idx != (Buffer{}) {
					r.encodeMoECapture(e, l)
				}
				if ffnWalkPlantForTest == ffnWalkPlantResidual {
					e.Dispatch(r.pRes, r.H, 256, x, r.dO)
				}
				i++
			case program.FfnNonGated:
				e.DispatchTG(r.pSABias, r.I*32, 256, r.H*2, L.upW, L.upS, r.mq, r.mSc, r.gu, L.upBias, r.uH)
				e.Dispatch(r.pActQuant, 256, 256, r.gu, r.dq, r.dSc, r.uI, r.uAct)
				e.Dispatch(r.pCoalBiasResid, r.H*32, 32, L.dW, L.dS, r.dq, r.dSc, x, L.downBias, r.uI)
				if ffnWalkPlantForTest == ffnWalkPlantResidual {
					e.Dispatch(r.pRes, r.H, 256, x, r.dO)
				}
				i++
			default: // FfnGated, FfnGemma4Dense
				fusedOut := i+1 < len(ops) && ops[i+1].Kind == program.OpResidual && !ops[i+1].Res.Two
				if r.q4kLane {
					r.q4kGemv(e, L.guSegs, false, Buffer{}, r.axF32, r.gu, r.uH)
					e.Dispatch(r.pSwF32, L.ffnI, 256, r.gu, r.gu.At(L.ffnI*4), r.swF32, L.uFFNI, r.uAct)
					r.q4kGemv(e, L.dSegs, true, Buffer{}, r.swF32, x, L.uFFNI)
					if ffnWalkPlantForTest == ffnWalkPlantResidual {
						e.Dispatch(r.pRes, r.H, 256, x, r.dO)
					}
					i++
					continue
				}
				if f16Lane {
					e.DispatchTG(r.pSAf16, (2*L.ffnI)*32, 256, r.H*2, L.guW, L.guS, r.mxF16, r.gu, r.uH)
				} else {
					p, n := saRowsPick(r.pSA, r.pSARows, 2*L.ffnI, r.gemvRows.gu)
					e.DispatchTG(p, n, 256, r.H*2, L.guW, L.guS, gq, gSc, r.gu, r.uH)
				}
				if r.loraLayers != nil {
					LA := &r.loraLayers[l]
					r.applyResidentLoRA(e, LA.gate, gq, gSc, r.gu)
					r.applyResidentLoRA(e, LA.up, gq, gSc, r.gu.At(L.ffnI*4))
				}
				e.Dispatch(r.pSw, 256, 256, r.gu, r.gu.At(L.ffnI*4), r.dq, r.dSc, L.uFFNI, r.uAct)
				if fusedOut {
					if R := r.gemvRows.down; R > 0 {
						e.DispatchTG(r.pGemvResidStaged, r.H*32/R, 256, L.ffnI, L.dW, L.dS, r.dq, r.dSc, x, L.uFFNI)
					} else {
						e.Dispatch(r.pGemvResid, r.H*32, 32, L.dW, L.dS, r.dq, r.dSc, x, L.uFFNI)
					}
					if r.loraLayers != nil {
						r.applyResidentLoRA(e, r.loraLayers[l].down, r.dq, r.dSc, x)
					}
					if ffnWalkPlantForTest == ffnWalkPlantResidual {
						e.Dispatch(r.pRes, r.H, 256, x, r.dO)
					}
					i++
				} else {
					e.Dispatch(r.pGemv, r.H*32, 32, L.dW, L.dS, r.dq, r.dSc, r.dO, L.uFFNI)
					if r.loraLayers != nil {
						r.applyResidentLoRA(e, r.loraLayers[l].down, r.dq, r.dSc, r.dO)
					}
				}
			}
		case program.OpResidual:
			e.Dispatch(r.pRes, r.H, 256, x, r.dO)
		case program.OpGemma4MoE:
			r.encodeGemma4MoEFFN(e, L)
		case program.OpPLE:
			if L.pleGW != (Buffer{}) && !g4SkipPLEForTest {
				r.encodePLE(e, l, x)
			}
		case program.OpLayerScale:
			if L.uLayerScalar != (Buffer{}) && !g4DropLayerScalarForTest {
				e.Dispatch(r.pLayerScale, r.H, 256, x, L.uLayerScalar)
			}
		}
	}
}
