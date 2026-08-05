//go:build goinfer_testhooks

// Code relocated by the B-08 build-tag pass: these are test-only hooks, compiled
// only under -tags goinfer_testhooks so they are NOT part of the public API
// (audit B-08). See RELEASING.md. Imports are added to satisfy the moved bodies.

package decoder

import "github.com/townsendmerino/aikit/linalg"

// FinalNormForTest applies the model's final normalization to a copy of h — the CPU reference
// for the last step of a resident bisect (the seam between the trunk and the LM head, where the
// hidden state is normed then projected). Does NOT project to logits; that is the head.
func (m *Model) FinalNormForTest(h []float32) []float32 {
	out := append([]float32(nil), h...)
	normalize(m.w.arch, out, m.w.FinalNorm, m.w.FinalNormBias, m.w.arch.HiddenDim)
	return out
}

func (m *Model) EmbedResidentForTest(id int) []float32 { return m.embedResident(id) }

// ForwardForTest / EmbedResidentForTest are CPU-reference seams for the gpu-package
// resident parity gates (which can't be in package decoder — import cycle via gpu).
// ForwardForTest is the CPU per-token logits; EmbedResidentForTest is the resident
// input embedding (incl. Granite EmbMul). Test-only.
func (m *Model) ForwardForTest(id int, cache *KVCache) ([]float32, error) {
	return m.forward(id, cache)
}

// ResidentForwardForTest exposes the resident forward (nil when not resident) for the
// GPU ForwardN-vs-Forward parity gate. Test-only seam; production code routes through
// Generate / GenerateSpeculative, not this.
func (m *Model) ResidentForwardForTest() ResidentForward { return m.resident }

// Gemma4MoEExpertForTest computes ONE gemma4 MoE expert's output on a caller-supplied input xe
// ([hidden]) — the gelu-tanh GeGLU expert function edown = Down · (geluTanh(gate)·up), gate‖up =
// GateUp·xe — and returns it alongside the expert's fused gate‖up and down weight matrices. The
// cuda single-expert gate (task 2c) packs those weights, runs the resident chain
// (gemv_w4a8_moe → glu_quant act=GELU_TANH → down) on the same xe, and compares: the gelu-tanh MoE
// epilogue × the indexed-expert GEMV is a combination that ships in neither Gemma-3 (dense, not
// indexed) nor Mixtral/GLM (indexed, but SiLU), so it is verified directly, not argued by
// composition. ok=false for a non-gemma4 model, a dense layer, or e out of range.
func (m *Model) Gemma4MoEExpertForTest(layer, e int, xe []float32) (edown []float32, gateUp, down *linalg.WeightMat, moeInter, hidden int, ok bool) {
	if m.w.arch.gemma4 == nil || layer < 0 || layer >= len(m.w.Layers) {
		return nil, nil, nil, 0, 0, false
	}
	gm := m.w.Layers[layer].gemma4moe
	if gm == nil || e < 0 || e >= len(gm.expertsGateUp) || len(xe) != m.w.arch.HiddenDim {
		return nil, nil, nil, 0, 0, false
	}
	be := &cpuBackend{}
	gu := make([]float32, 2*gm.moeInter)
	matmul(be, &gm.expertsGateUp[e], xe, gu, 1)
	mid := make([]float32, gm.moeInter)
	for i := 0; i < gm.moeInter; i++ {
		mid[i] = geluTanh(gu[i]) * gu[gm.moeInter+i]
	}
	out := make([]float32, m.w.arch.HiddenDim)
	matmul(be, &gm.expertsDown[e], mid, out, 1)
	return out, &gm.expertsGateUp[e], &gm.expertsDown[e], gm.moeInter, m.w.arch.HiddenDim, true
}

// Gemma4MoERouterForTest exposes a gemma4 MoE layer's f32 router projection (row-major [nE, hidden])
// and its selection bias (zeros — gemma4 has no router bias) for the resident-router idx-equality
// unit test (cuda/). That test replays captured router inputs (routerRnBuf) through the CUDA
// selection kernels and gates resident idx[] against the CPU idx[], isolating a routing FLIP from
// any expert-GEMV numeric difference — the "router first" discipline. ok=false for a non-gemma4
// model or a dense (non-MoE) layer.
func (m *Model) Gemma4MoERouterForTest(layer int) (proj, bias []float32, nE, topK, hidden int, ok bool) {
	if m.w.arch.gemma4 == nil || layer < 0 || layer >= len(m.w.Layers) {
		return nil, nil, 0, 0, 0, false
	}
	gm := m.w.Layers[layer].gemma4moe
	if gm == nil {
		return nil, nil, 0, 0, 0, false
	}
	f, has := gm.routerProj.F32()
	if !has {
		return nil, nil, 0, 0, 0, false
	}
	return f, make([]float32, gm.nE), gm.nE, gm.topK, m.w.arch.HiddenDim, true
}

// Gemma4HiddenCaptureForTest returns the captured residual stream per layer, in call order:
// index 0 = post-embedding, index i+1 = the stream after layer i (pre-final-norm, matching what a
// resident runner's trunk-to-N-layers produces).
func Gemma4HiddenCaptureForTest() [][]float32 { return gemma4HiddenBuf }

// SetGemma4HiddenCaptureForTest toggles per-layer hidden capture, clearing the buffer when enabled.
// Exported so a resident test in another package (metal/cuda) can drive a CPU gemma4 forward and
// read the per-layer residual stream — it cannot touch the unexported g4traceHidden directly.
// Capture only a SINGLE token's forward (drive one ForwardForTest between on and off): the buffer
// accumulates across calls, so a multi-token pass concatenates every token's per-layer trace.
func SetGemma4HiddenCaptureForTest(on bool) {
	if on {
		gemma4HiddenBuf = nil
		g4traceHidden = func(_ int, h []float32) {
			gemma4HiddenBuf = append(gemma4HiddenBuf, append([]float32(nil), h...))
		}
	} else {
		g4traceHidden = nil
		gemma4HiddenBuf = nil
	}
}

// RouterMarginForTest returns the per-decision top-k boundary margin (smallest selected expert's
// softmax prob minus the largest rejected expert's), same order/index as RouterCaptureForTest. It
// is the MoE-specific robustness signal a noise-floor check reads: a fixture whose margin sits below
// the int4-vs-f32 routing perturbation can flip top-k under quant and cannot gate a resident router,
// however correct the port (the reason the CUDA MoE fixture was rebuilt at 9275f94).
func RouterMarginForTest() []float32 { return routerMarginBuf }

// Gemma4MoECaptureForTest returns the CPU per-decision wts / x1 / x2 (same order as RouterCaptureForTest).
func Gemma4MoECaptureForTest() (wts, x1, x2 [][]float32) {
	return routerWtsBuf, routerX1Buf, routerX2Buf
}

// RouterCaptureForTest returns the captured per-decision selected experts and router inputs (same
// order/index). Sibling to SetRouterCaptureForTest for the cross-package (cuda) router-first gate.
func RouterCaptureForTest() (idx [][]int, rn [][]float32) { return routerCaptureBuf, routerRnBuf }

// SetRouterCaptureForTest toggles router capture and (when enabling) clears the buffers. Exported so
// the CUDA resident-router unit test (package cuda) can drive a CPU forward, capture idx/rn, and
// replay rn through the device kernels — it cannot touch these unexported vars directly.
func SetRouterCaptureForTest(on bool) {
	routerCapture = on
	if on {
		routerCaptureBuf, routerRnBuf, routerMarginBuf = nil, nil, nil
		routerWtsBuf, routerX1Buf, routerX2Buf = nil, nil, nil
	}
}

// LayerKVForTest returns a layer's stored K and V as f32 in absolute position order — the live
// sliding-window for a ring (local) layer, the full history for an append-forever global —
// dequantized when the cache is int8. base is the absolute position of row 0. Keys/Vals above
// only see the global append-forever store, so they read empty for a windowed layer (Gemma's
// local layers); this is the ring-aware read the cross-backend attention confirmer needs to
// inject goinfer's exact K/V into another engine.
func (c *KVCache) LayerKVForTest(layer int) (k, v []float32, base int) {
	if r := c.rings[layer]; r != nil {
		lo := max(0, r.count-r.w)
		nLive, st := r.count-lo, r.stride
		if st == 0 || nLive == 0 {
			return nil, nil, lo
		}
		k, v = make([]float32, nLive*st), make([]float32, nLive*st)
		nKV := st / r.headDim
		for i, p := 0, lo; p < r.count; i, p = i+1, p+1 {
			so, do := (p%r.w)*st, i*st
			if r.quant == kvI8 {
				sso := (p % r.w) * nKV
				dequantHeads(r.kq[so:so+st], r.ksc[sso:sso+nKV], nKV, r.headDim, k[do:do+st])
				dequantHeads(r.vq[so:so+st], r.vsc[sso:sso+nKV], nKV, r.headDim, v[do:do+st])
			} else {
				copy(k[do:do+st], r.k[so:so+st])
				copy(v[do:do+st], r.v[so:so+st])
			}
		}
		return k, v, lo
	}
	if c.quant == kvI8 {
		st := c.kvDim
		nKV := st / c.headDim
		n := len(c.keysQ[layer]) / st
		k, v = make([]float32, n*st), make([]float32, n*st)
		for p := 0; p < n; p++ {
			o, so := p*st, p*nKV
			dequantHeads(c.keysQ[layer][o:o+st], c.keyScale[layer][so:so+nKV], nKV, c.headDim, k[o:o+st])
			dequantHeads(c.valsQ[layer][o:o+st], c.valScale[layer][so:so+nKV], nKV, c.headDim, v[o:o+st])
		}
		return k, v, 0
	}
	return c.keys[layer], c.vals[layer], 0
}

func SetSSMQ8CPU(v bool) { ssmQ8CPU = v }

// SetSSMForceF32 / SetSSMQ8CPU toggle the CPU-reference precision-localization seams
// at runtime (gpu/ssm_kernel_control_test.go needs the staged webgpu backend, which
// lives in a package the decoder can't import — so it drives these via the registry).
func SetSSMForceF32(v bool) { ssmForceF32 = v }

// SetMambaCapHook installs/clears the capture hook (the gpu package can't import decoder-internal
// state directly, so it drives this via the export, like SetSSMQ8CPU).
func SetMambaCapHook(f func(proj, gated []float32)) { mambaCapHook = f }
