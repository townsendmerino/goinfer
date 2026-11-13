package metal

import (
	"fmt"

	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// The mixer steps' operation walk (OP4m-pod; docs/tasks/task-forward-and-memory-2026-11.md, "OP4m-pod: registration"): the
// per-token decode emits a recurrent or latent mixer step, DeltaNet, Mamba-2 or MLA, from its operations, each through its
// emitter: the input norm, the mixer, the residual add. Where one dispatch serves two adjacent operations, the pair is emitted
// once: DeltaNet's out_proj adds into the residual. checkMixerWalk holds every step's operations to what the walk emits when the
// resident builds, so the walk meets no operation it cannot emit.

// mixerWalkPlantForTest plants P-1's defects in the walk: mixerWalkPlantResidual adds a residual after an output projection that
// already adds it (DeltaNet's out_proj, a fused attention o-proj), and mixerWalkPlantAddOne has the input norm ignore (1 + w).
var mixerWalkPlantForTest int

const (
	mixerWalkPlantResidual = iota + 1
	mixerWalkPlantAddOne
)

// mixerWalks reports whether the decode walk emits step st from its operations.
func mixerWalks(st layerStep) bool {
	return st == stepAttention || st == stepDeltaNet || st == stepMamba || st == stepMLA
}

// mixerCores is each walked step's mixer operation.
var mixerCores = map[layerStep]program.OpKind{stepDeltaNet: program.OpDeltaNet, stepMamba: program.OpMamba2, stepMLA: program.OpMLA}

// checkMixerWalk is why the walk cannot emit ops, step st of a layer, or nil, which it also is for a step the walk does not cover.
// The walk emits an input norm on the layer's pre-attention weights, an RMSNorm or a LayerNorm (with the weights' bias, where they
// carry one) at the epsilon and (1 + w) form the resident's norm uniforms hold, then the mixer, then a residual add of one output
// with no scale.
func (r *resident) checkMixerWalk(L *residLayer, st layerStep, ops []program.Op) error {
	if st == stepAttention {
		return r.checkAttentionWalk(L, ops)
	}
	if !mixerWalks(st) {
		return nil
	}
	want := []program.OpKind{program.OpNorm, mixerCores[st], program.OpResidual}
	if len(ops) != len(want) {
		return fmt.Errorf("the %v step has %d operations, and the walk emits %v", st, len(ops), want)
	}
	for i, k := range want {
		if ops[i].Kind != k {
			return fmt.Errorf("the %v step's operation %d is %v, and the walk emits %v there", st, i, ops[i].Kind, k)
		}
	}
	if err := r.checkWalkNorm(st, &ops[0].Norm, program.SlotPreAttn); err != nil {
		return err
	}
	if n := &ops[0].Norm; n.Form != program.NormRMS && n.Form != program.NormLayer {
		return fmt.Errorf("the %v step's input norm has form %v, which the walk has no emitter for", st, n.Form)
	}
	return checkWalkResidual(st, &ops[2].Res)
}

// checkWalkNorm is why norm n of step st, which the walk binds to slot, is not the norm its emitter computes, or nil: the slot,
// and for an RMSNorm or a LayerNorm the epsilon and (1 + w) the resident's norm uniforms hold (decodeShape's values), and a
// LayerNorm's bias where the weights carry one.
func (r *resident) checkWalkNorm(st layerStep, n *program.NormParams, slot program.NormSlot) error {
	switch {
	case n.Slot != slot:
		return fmt.Errorf("the %v step's norm reads norm slot %v, and the walk binds slot %v there", st, n.Slot, slot)
	case n.Form == program.NormCopy:
	case n.Form == program.NormLayer && !n.Bias && r.decode.lnHasBias == 1:
		return fmt.Errorf("the %v step's LayerNorm takes no bias, and the layer's weights carry one, which the emitter adds", st)
	case float32(n.Eps) != r.decode.normEps:
		return fmt.Errorf("the %v step's norm has epsilon %g, and the resident's norm uniform holds %g", st, n.Eps, r.decode.normEps)
	case n.AddOne != (r.decode.addOne == 1):
		return fmt.Errorf("the %v step's norm has (1 + w) %v, and the resident's norm uniform says %v", st, n.AddOne, !n.AddOne)
	}
	return nil
}

// checkWalkResidual is why residual add res of step st is not one the walk emits, or nil: one output, unscaled.
func checkWalkResidual(st layerStep, res *program.ResidualParams) error {
	if res.Two || res.Scale != 0 {
		return fmt.Errorf("the %v step's residual add is of two outputs (%v) or scaled (%g), which the walk does not emit", st, res.Two, res.Scale)
	}
	return nil
}

// encodeMixerWalk emits layer l's mixer step from its operations ops, which checkMixerWalk admitted, on residual x. A recurrent
// step runs on rec's conv window and state, or the layer's own when rec is the zero value.
func (r *resident) encodeMixerWalk(e *Encoder, l int, L *residLayer, ops []program.Op, x, uPos, uNKeys, uQTemp, rp Buffer, rec stepState) {
	if L.geom != nil && L.delta == nil && L.mamba == nil && L.mla == nil {
		r.encodeAttentionWalk(e, l, ops, x, uPos, uNKeys, uQTemp, rp)
		return
	}
	for i := 0; i < len(ops); i++ {
		switch o := &ops[i]; o.Kind {
		case program.OpNorm:
			r.emitNorm(e, &o.Norm, x, L.preNorm, L.preNormBias, r.aq, r.aSc)
		case program.OpDeltaNet: // and the residual add after it, which its out_proj does
			win, state := rec.orPair(L.delta.win, L.delta.state)
			r.encodeDeltaNetCoreOn(e, L, x, win, state)
			if mixerWalkPlantForTest == mixerWalkPlantResidual {
				e.Dispatch(r.pRes, r.H, 256, x, r.dO)
			}
			i++
		case program.OpMamba2:
			win, ssm := rec.orPair(L.mamba.win, L.mamba.ssm)
			r.encodeMambaCoreOn(e, L, win, ssm)
		case program.OpMLA:
			r.encodeMLACore(e, l, L, uPos, rp, uNKeys)
		case program.OpResidual:
			e.Dispatch(r.pRes, r.H, 256, x, r.dO)
		}
	}
}

// emitNorm emits norm n of x with weights w (and bias, for a LayerNorm with one) into the quantized activation aq and its scales
// aSc. Its epsilon and (1 + w) form are the resident's norm uniforms, which checkMixerWalk holds n to.
func (r *resident) emitNorm(e *Encoder, n *program.NormParams, x, w, bias, aq, aSc Buffer) {
	if n.Form == program.NormLayer {
		e.Dispatch(r.pLayerNorm, tgReduceNorm, tgReduceNorm, x, w, bias, aq, aSc, r.uH, r.uEps, r.uLNHasBias)
		return
	}
	addOne := r.uAddOne
	if mixerWalkPlantForTest == mixerWalkPlantAddOne {
		addOne = r.uZero
	}
	e.Dispatch(r.pRms, tgReduceNorm, tgReduceNorm, x, w, aq, aSc, r.uH, r.uEps, addOne)
}

// checkAttentionWalk is why the walk cannot emit ops, layer L's attention step, or nil. The walk emits, in order:
//   - the input norm on the pre-attention weights: a copy (post-only placement), an RMSNorm or a LayerNorm;
//   - the q/k/v projection, the QK-norm (per head, or whole), the query's temperature and the rotation, then the store and
//     attention; or the gated or Gemma 4 attention operation, which runs all of those itself;
//   - the output projection, then a post-attention norm and the residual add, or the residual add alone, or neither (a parallel
//     block's attention half, whose residual the FFN step's span holds).
//
// Each choice an operation makes must be one the build made the buffers for: the norm form, a QK-norm and its form, the rotation,
// the gated projection, the post-attention norm, the output bias and the parallel block.
func (r *resident) checkAttentionWalk(L *residLayer, ops []program.Op) error {
	st := stepAttention
	if L.geom == nil {
		return fmt.Errorf("the attention step's layer was built with no attention")
	}
	if len(ops) == 0 || ops[0].Kind != program.OpNorm {
		return fmt.Errorf("the attention step does not start with its input norm")
	}
	if err := r.checkWalkNorm(st, &ops[0].Norm, program.SlotPreAttn); err != nil {
		return err
	}
	in := ops[0].Norm.Form
	if want := r.attnInputForm(); in != want {
		return fmt.Errorf("the attention step's input norm has form %v, and the build's layers take %v", in, want)
	}
	i := 1
	var qkNorm, rope, temp, core, gated, g4 bool
	for ; i < len(ops) && ops[i].Kind != program.OpProjO && !core; i++ {
		switch o := &ops[i]; o.Kind {
		case program.OpProjQKV:
		case program.OpQKNorm:
			qkNorm = true
			if o.QKNorm.Whole != r.qkNormWhole {
				return fmt.Errorf("the attention step's QK-norm is whole %v, and the build's is %v", o.QKNorm.Whole, r.qkNormWhole)
			}
			if float32(o.QKNorm.Eps) != r.decode.normEps || o.QKNorm.AddOne != (r.decode.addOne == 1) {
				return fmt.Errorf("the attention step's QK-norm has epsilon %g and (1 + w) %v, which the norm uniforms do not hold", o.QKNorm.Eps, o.QKNorm.AddOne)
			}
		case program.OpRope:
			rope = true
		case program.OpAttnTemp:
			temp = true
		case program.OpAttention:
			core = true
		case program.OpGatedAttn:
			core, gated = true, true
		case program.OpGemma4Attn:
			core, g4 = true, true
		default:
			return fmt.Errorf("the attention step's operation %d is %v, which the walk does not emit there", i, o.Kind)
		}
	}
	switch {
	case !core:
		return fmt.Errorf("the attention step holds no attention operation")
	case gated != L.qGate:
		return fmt.Errorf("the attention step's attention is gated %v, and the build's layer is %v", gated, L.qGate)
	case !gated && !g4 && qkNorm != r.qkNorm:
		return fmt.Errorf("the attention step has a QK-norm %v, and the build's layers %v", qkNorm, r.qkNorm)
	case !gated && !g4 && rope && r.learnedPos:
		return fmt.Errorf("the attention step rotates, and the build's positions are learned")
	case !gated && !g4 && !rope && !r.learnedPos && !rotatesByNothing(L):
		return fmt.Errorf("the attention step does not rotate, and its layer's table does (a NoPE layer's is all zeros at scale 1)")
	case temp && !rope && !gated && !g4:
		return fmt.Errorf("the attention step scales the query's temperature with no rotation, whose dispatch carries the scale")
	case !gated && !g4 && r.attnVNormNeededAt(L):
		return fmt.Errorf("the attention step's layer takes Gemma 4's v_norm, which only its attention operation emits")
	}
	if !gated && !g4 {
		if i == len(ops) || ops[i].Kind != program.OpProjO {
			return fmt.Errorf("the attention step's attention is not followed by its output projection")
		}
		if ops[i].Proj.Bias != r.outBias {
			return fmt.Errorf("the attention step's output projection has a bias %v, and the build's %v", ops[i].Proj.Bias, r.outBias)
		}
		i++
	}
	form := attnOutFormOf(r, ops, i, !gated && !g4 && ops[i-1].Proj.Bias)
	if want := r.attnOutFormFlags(); form != want {
		return fmt.Errorf("the attention step's output ends as form %d, and the build's layers as %d", form, want)
	}
	if form == attnOutScratchNorm {
		if err := r.checkWalkNorm(st, &ops[i].Norm, program.SlotPostAttn); err != nil {
			return err
		}
		i++
	}
	if form != attnOutScratch {
		if i >= len(ops) || ops[i].Kind != program.OpResidual {
			return fmt.Errorf("the attention step's output does not end in its residual add")
		}
		if err := checkWalkResidual(st, &ops[i].Res); err != nil {
			return err
		}
		i++
	}
	if i != len(ops) {
		return fmt.Errorf("the attention step has %d operations past its residual add", len(ops)-i)
	}
	return nil
}

// attnInputForm is the input norm the build's attention layers take: a copy under post-only placement, else a LayerNorm or an
// RMSNorm.
func (r *resident) attnInputForm() program.NormForm {
	switch {
	case r.postOnly:
		return program.NormCopy
	case r.layerNorm:
		return program.NormLayer
	}
	return program.NormRMS
}

// attnOutFormFlags is the output tail the build's attention layers take.
func (r *resident) attnOutFormFlags() attnOutForm {
	switch {
	case r.sandwich || r.postOnly:
		return attnOutScratchNorm
	case r.parallelBlock:
		return attnOutScratch
	case r.outBias:
		return attnOutFusedBias
	}
	return attnOutFused
}

// attnOutFormOf is the output tail the operations after ops[i] ask for: none (a parallel block's attention half) is the scratch,
// a post-attention norm the scratch and the norm, a residual add the fused projection, with its bias where bias.
func attnOutFormOf(r *resident, ops []program.Op, i int, bias bool) attnOutForm {
	switch {
	case i == len(ops):
		return attnOutScratch
	case ops[i].Kind == program.OpNorm:
		return attnOutScratchNorm
	case bias:
		return attnOutFusedBias
	}
	return attnOutFused
}

// rotatesByNothing reports whether layer L's rotation is the identity: no rotated pairs, or an all-zero table at scale 1 (a NoPE
// layer's, decoder.Model.RopeInvFreqLayer).
func rotatesByNothing(L *residLayer) bool {
	if L.geom.half == 0 {
		return true
	}
	for _, f := range L.invf.Floats()[:L.geom.half] {
		if f != 0 {
			return false
		}
	}
	return L.mscale.Floats()[0] == 1
}

// attnVNormNeededAt is attnVNormNeeded for layer L.
func (r *resident) attnVNormNeededAt(L *residLayer) bool {
	return L.geom != nil && !L.kvShared && (L.geom.kEqV || (r.g4VNorm && !g4DropVNormForTest))
}

// encodeAttentionWalk emits layer l's attention step from its operations ops, which checkAttentionWalk admitted, on residual x.
func (r *resident) encodeAttentionWalk(e *Encoder, l int, ops []program.Op, x, uPos, uNKeys, uQTemp, rp Buffer) {
	a := r.attnEncOf(l, x, uPos, uNKeys, uQTemp, rp)
	rotated := false
	whole := func(i int) {
		r.attnOut(e, &a, attnOutFormOf(r, ops, i, false))
	}
	for i := 0; i < len(ops); i++ {
		switch o := &ops[i]; o.Kind {
		case program.OpNorm: // the input norm; a post-attention norm is the output tail's
			if mixerWalkPlantForTest == mixerWalkPlantAddOne {
				addOne := r.uAddOne
				r.uAddOne = r.uZero
				r.attnInputNorm(e, &a, o.Norm.Form)
				r.uAddOne = addOne
				continue
			}
			r.attnInputNorm(e, &a, o.Norm.Form)
		case program.OpProjQKV:
			r.attnProjQKV(e, &a)
		case program.OpQKNorm:
			r.attnQKNorm(e, &a, o.QKNorm.Whole)
		case program.OpAttnTemp: // the rotation's dispatch carries the query's temperature
		case program.OpRope:
			r.attnRope(e, &a)
			rotated = true
		case program.OpAttention:
			if !rotated && !r.learnedPos {
				r.attnRope(e, &a) // a NoPE layer: the block rotates it by its all-zero table, an identity, and so does the walk
			}
			r.attnCore(e, &a)
		case program.OpGatedAttn, program.OpGemma4Attn: // the projection through the attention, then the output tail
			if o.Kind == program.OpGatedAttn {
				r.attnGatedQKV(e, &a)
			} else {
				r.attnProjQKV(e, &a)
			}
			if r.qkNorm {
				r.attnQKNorm(e, &a, r.qkNormWhole)
			}
			if r.attnVNormNeeded(&a) {
				r.attnVNorm(e, &a)
			}
			if !r.learnedPos {
				r.attnRope(e, &a)
			}
			r.attnCore(e, &a)
			if o.Kind == program.OpGatedAttn {
				r.attnOutGate(e, &a)
			}
			whole(i + 1)
			return
		case program.OpProjO:
			form := attnOutFormOf(r, ops, i+1, o.Proj.Bias)
			r.attnOut(e, &a, form)
			if mixerWalkPlantForTest == mixerWalkPlantResidual && form == attnOutFused {
				e.Dispatch(r.pRes, r.H, 256, x, r.oO)
			}
			return
		}
	}
}
