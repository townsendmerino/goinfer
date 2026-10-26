//go:build cuda

package cuda

import (
	"fmt"

	gpu "github.com/townsendmerino/goinfer/cuda/internal/gpudev"
)

// Float-activation decode for Gemma 4 MoE (floatAct, gemv_w4f32.cu): every int4 projection of a decode step reads its
// activation as f32 instead of rounding it to int8. Under per-vector int8 activations the 26B-A4B's decode is chaotic at
// some tokens, on the CPU's W4A8 path as on this one, and can land on a garbage digit; over f32 activations the same int4
// weights are stable (docs/tasks/task-multimodal-support-2026-10.md, "S19"). Prefill and the LM head keep W4A8.

// floatActOffForTest builds the next resident on the W4A8 decode: the other arm of an A/B. Set by tests only.
var floatActOffForTest bool

// floatActDecline is why this resident cannot take the float-activation decode, or "" when it can: a Gemma 4 MoE
// (not an E-model) on per-vector activation scales, with every projection a decode step runs stored int4, since
// gemv_w4f32 reads int4 only.
func (r *cudaResident) floatActDecline() string {
	switch {
	case floatActOffForTest:
		return "turned off by a test"
	case !r.gemma4Moe:
		return "not a Gemma 4 MoE model"
	case r.eModel:
		return "a Gemma 4 E-model"
	case r.actG32:
		return "per-32 activations"
	}
	for l := range r.layers {
		Ly := &r.layers[l]
		ws := []struct {
			name string
			w    cudaWQ
		}{{"q_proj", Ly.q}, {"o_proj", Ly.o}}
		if !Ly.kvShared {
			ws = append(ws, struct {
				name string
				w    cudaWQ
			}{"k_proj", Ly.k})
			if !Ly.kEqV {
				ws = append(ws, struct {
					name string
					w    cudaWQ
				}{"v_proj", Ly.v})
			}
		}
		if Ly.ffn {
			ws = append(ws, []struct {
				name string
				w    cudaWQ
			}{{"gate_proj", Ly.g}, {"up_proj", Ly.u}, {"down_proj", Ly.d}}...)
		}
		if Ly.g4moe {
			ws = append(ws, []struct {
				name string
				w    cudaWQ
			}{{"experts gate_up", Ly.expGU}, {"experts down", Ly.expDown}}...)
		}
		for _, w := range ws {
			if w.w.kind != "int4" {
				return fmt.Sprintf("layer %d %s is %q (gemv_w4f32 reads int4 only)", l, w.name, w.w.kind)
			}
		}
	}
	return ""
}

// gemvF is doG over an f32 activation: dst (+)= wt · a + bias, wt int4.
func (r *cudaResident) gemvF(wt cudaWQ, a Buffer, bias KernelArg, dst Buffer, accum int32) error {
	return r.launch(r.fW4F32, LaunchConfig{GridX: uint32((wt.N + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
		Arg(wt.W), Arg(a), Arg(wt.ws16), bias, gpu.ArgValue(int32(wt.N)), gpu.ArgValue(int32(wt.K/8)), gpu.ArgValue(int32(wt.K/32)),
		Arg(dst), gpu.ArgValue(accum))
}

// normInto is rms's normalization without the int8 rounding: xF = norm(src) with weight w, as f32.
func (r *cudaResident) normInto(src, w Buffer) error {
	if e := r.copyF32(src, r.xF, r.hidden); e != nil {
		return e
	}
	return r.normF32(r.xF, w)
}

// floatActQKV is segA's projection under floatAct: the pre-norm into xF, then q, k and v (K=V: v is a copy of the raw k,
// as the W4A8 path makes it).
func (r *cudaResident) floatActQKV(Ly *cudaLayer, qb, kb, vb KernelArg) error {
	if e := r.normInto(r.x, Ly.preNorm); e != nil {
		return e
	}
	if e := r.gemvF(Ly.q, r.xF, qb, r.qB, 0); e != nil {
		return e
	}
	if Ly.kvShared {
		return nil
	}
	if e := r.gemvF(Ly.k, r.xF, kb, r.kB, 0); e != nil {
		return e
	}
	if Ly.kEqV {
		return r.copyF32(r.kB, r.vB, Ly.kvDim)
	}
	return r.gemvF(Ly.v, r.xF, vb, r.vB, 0)
}

// floatActDense is gemma4MoeMLPPre's dense branch under floatAct, on h = x: xd = preFFNNorm(h) into xF, gate and up, the
// GLU as f32 into dF, down into g4x1.
func (r *cudaResident) floatActDense(Ly *cudaLayer, x Buffer) error {
	nullBias := ArgNull()
	if e := r.normInto(x, Ly.g4preFFN); e != nil {
		return e
	}
	if e := r.gemvF(Ly.g, r.xF, nullBias, r.gO, 0); e != nil {
		return e
	}
	if e := r.gemvF(Ly.u, r.xF, nullBias, r.uO, 0); e != nil {
		return e
	}
	if e := r.launch(r.fGluF32, g1cfg(r.inter, 256), Arg(r.gO), Arg(r.uO), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(0)),
		gpu.ArgValue(int32(r.inter)), gpu.ArgValue(r.act), Arg(r.dF)); e != nil {
		return e
	}
	return r.gemvF(Ly.d, r.dF, nullBias, r.g4x1, 0)
}

// floatActExpert is one routed expert (slot j) of gemma4MoeMLPPost under floatAct: gate‖up from xe (in xF), the GLU as
// f32 into moeGLUF, down weight-accumulated into g4x2.
func (r *cudaResident) floatActExpert(Ly *cudaLayer, j int) error {
	gu := 2 * r.moeInter
	if e := r.launch(r.fMoEGemvF32, LaunchConfig{GridX: uint32((gu + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
		Arg(Ly.expGU.W), Arg(r.xF), Arg(Ly.expGU.ws16), Arg(r.expIdx()), gpu.ArgValue(int32(j)), gpu.ArgValue(int32(gu)),
		gpu.ArgValue(int32(gu)), gpu.ArgValue(int32(r.hidden/8)), gpu.ArgValue(int32(r.hidden/32)), Arg(r.moeGU)); e != nil {
		return e
	}
	if e := r.launch(r.fGluF32, g1cfg(r.moeInter, 256), Arg(r.moeGU), Arg(r.moeGU), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(r.moeInter)),
		gpu.ArgValue(int32(r.moeInter)), gpu.ArgValue(r.act), Arg(r.moeGLUF)); e != nil {
		return e
	}
	return r.launch(r.fMoEWaccF32, LaunchConfig{GridX: uint32((r.hidden + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
		Arg(Ly.expDown.W), Arg(r.moeGLUF), Arg(Ly.expDown.ws16), Arg(r.expIdx()), Arg(r.rWgt), gpu.ArgValue(int32(j)),
		gpu.ArgValue(int32(r.hidden)), gpu.ArgValue(int32(r.hidden)), gpu.ArgValue(int32(r.moeInter/8)), gpu.ArgValue(int32(r.moeInter/32)),
		Arg(r.g4x2))
}
