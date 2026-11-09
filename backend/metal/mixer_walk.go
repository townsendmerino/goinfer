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

// mixerWalkBlocksForTest runs the mixer steps through their blocks instead: the walk's do-nothing arm, which P-1 and P-3 compare.
var mixerWalkBlocksForTest bool

// mixerWalkPlantForTest plants P-1's defects in the walk: mixerWalkPlantResidual adds a residual after DeltaNet's out_proj, which
// already adds it, and mixerWalkPlantAddOne has the norm emitter ignore (1 + w).
var mixerWalkPlantForTest int

const (
	mixerWalkPlantResidual = iota + 1
	mixerWalkPlantAddOne
)

// mixerWalks reports whether the decode walk emits step st from its operations.
func mixerWalks(st layerStep) bool { return st == stepDeltaNet || st == stepMamba || st == stepMLA }

// mixerCores is each walked step's mixer operation.
var mixerCores = map[layerStep]program.OpKind{stepDeltaNet: program.OpDeltaNet, stepMamba: program.OpMamba2, stepMLA: program.OpMLA}

// checkMixerWalk is why the walk cannot emit ops, step st of a layer, or nil, which it also is for a step the walk does not cover.
// The walk emits an input norm on the layer's pre-attention weights, an RMSNorm or a LayerNorm (with the weights' bias, where they
// carry one) at the epsilon and (1 + w) form the resident's norm uniforms hold, then the mixer, then a residual add of one output
// with no scale.
func (r *resident) checkMixerWalk(st layerStep, ops []program.Op) error {
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
	n := &ops[0].Norm
	switch {
	case n.Slot != program.SlotPreAttn:
		return fmt.Errorf("the %v step's input norm reads norm slot %v, and the walk binds the pre-attention weights", st, n.Slot)
	case n.Form != program.NormRMS && n.Form != program.NormLayer:
		return fmt.Errorf("the %v step's input norm has form %v, which the walk has no emitter for", st, n.Form)
	case n.Form == program.NormLayer && !n.Bias && r.uLNHasBias.U32s()[0] == 1:
		return fmt.Errorf("the %v step's LayerNorm takes no bias, and the layer's weights carry one, which the emitter adds", st)
	case float32(n.Eps) != r.uEps.Floats()[0]:
		return fmt.Errorf("the %v step's input norm has epsilon %g, and the resident's norm uniform holds %g", st, n.Eps, r.uEps.Floats()[0])
	case n.AddOne != (r.uAddOne.U32s()[0] == 1):
		return fmt.Errorf("the %v step's input norm has (1 + w) %v, and the resident's norm uniform says %v", st, n.AddOne, !n.AddOne)
	}
	if res := &ops[2].Res; res.Two || res.Scale != 0 {
		return fmt.Errorf("the %v step's residual add is of two outputs (%v) or scaled (%g), which the walk does not emit", st, res.Two, res.Scale)
	}
	return nil
}

// encodeMixerWalk emits layer l's mixer step from its operations ops, which checkMixerWalk admitted, on residual x. A DeltaNet step
// reads and adds into the resident's own residual, as its block does.
func (r *resident) encodeMixerWalk(e *Encoder, l int, L *residLayer, ops []program.Op, x, uPos, uNKeys, rp Buffer) {
	if L.delta != nil {
		x = r.x
	}
	for i := 0; i < len(ops); i++ {
		switch o := &ops[i]; o.Kind {
		case program.OpNorm:
			r.emitNorm(e, &o.Norm, x, L.preNorm, L.preNormBias, r.aq, r.aSc)
		case program.OpDeltaNet: // and the residual add after it, which its out_proj does
			r.encodeDeltaNetCoreOn(e, L, x, L.delta.win, L.delta.state)
			if mixerWalkPlantForTest == mixerWalkPlantResidual {
				e.Dispatch(r.pRes, r.H, 256, x, r.dO)
			}
			i++
		case program.OpMamba2:
			r.encodeMambaCoreOn(e, L, L.mamba.win, L.mamba.ssm)
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
