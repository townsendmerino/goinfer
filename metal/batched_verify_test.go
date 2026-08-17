//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// bvkPipes holds the five new batched-M kernels (metal/batched_verify_kernels.go), compiled once
// per test and reused across layers/positions. plainResid/coalResid are used for the non-sandwich
// O-proj/down-proj (decode fuses the residual add there); plain/coal (no fused epilogue) are used
// for the sandwich path's O-proj/down-proj, which decode itself never fuses (a sandwich norm sits
// between the projection and the residual add) — see bvkForwardM.
type bvkPipes struct {
	bias, plain, coal, plainResid, coalResid Pipeline
}

func newBVKPipes(d *Device) (bvkPipes, error) {
	lib, err := d.CompileLibrary(bvkKernels, MSL3_1)
	if err != nil {
		return bvkPipes{}, fmt.Errorf("compile bvkKernels: %w", err)
	}
	get := func(name string) (Pipeline, error) { return d.NewComputePipeline(lib, name) }
	bias, err := get("gemv_w4a8_bvk_bias")
	if err != nil {
		return bvkPipes{}, err
	}
	plain, err := get("gemv_w4a8_bvk_plain")
	if err != nil {
		return bvkPipes{}, err
	}
	coal, err := get("gemv_w4a8_bvk_coal")
	if err != nil {
		return bvkPipes{}, err
	}
	plainResid, err := get("gemv_w4a8_bvk_plain_resid")
	if err != nil {
		return bvkPipes{}, err
	}
	coalResid, err := get("gemv_w4a8_bvk_coal_resid")
	if err != nil {
		return bvkPipes{}, err
	}
	return bvkPipes{bias: bias, plain: plain, coal: coal, plainResid: plainResid, coalResid: coalResid}, nil
}

// bvkResult holds, per token position, the batched-kernel path's logits and post-trunk hidden
// state — for direct comparison against the sequential ground truth from the SAME resident's real
// per-token Forward path (encodeTrunkInto, metal/model.go).
type bvkResult struct {
	logits [][]float32 // [M][V]
	hidden [][]float32 // [M][H] — residual stream after all layers, before the final norm
}

// bvkForwardM runs M embeddings, at consecutive positions starting at startPos, through r's real
// per-layer weights and KV cache using the batched-M W4A8 kernels (bvkKernels) for the four dense
// projections (QKV in-proj, O-proj, gate/up-proj, down-proj) — and decode's OWN compiled
// pipelines (r.pRms/pRope/pKv/pAttn/pQv/pSw/pRes/pRmsF32/pQKNorm/pGemvW8), unmodified, the exact
// same Pipeline objects BuildResident created — for every other step (norm, RoPE, KV store,
// attention, activation quant, SwiGLU, residual add, LM head).
//
// Mirrors encodeAttention/encodeLayer's exact per-layer dispatch order (metal/model.go) dispatch
// for dispatch, batching only the four W4A8 GEMVs across all M tokens (one dispatch instead of M)
// and looping every other step per-token M times, in position order (attention and KV state are
// inherently sequential/causal across the M draft positions — each token's own K/V must be
// written before the NEXT token's attention can see it, exactly as sequential decode does).
//
// Out of scope, and this refuses rather than silently mismeasuring: MoE / Gemma-4 dense‖MoE
// layers (dense SwiGLU/GeGLU only — matches what decode's OWN batched prefill kernels already
// restrict to, metal/prefill.go's prefillOK), and M outside [1, 16].
func bvkForwardM(r *resident, pipes bvkPipes, embs [][]float32, startPos int) (bvkResult, error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	M := len(embs)
	if M < 1 || M > 16 {
		return bvkResult{}, fmt.Errorf("bvkForwardM: M=%d out of [1,16]", M)
	}
	if r.moe != nil || r.g4moe != nil {
		return bvkResult{}, fmt.Errorf("bvkForwardM: MoE / Gemma-4 dense-MoE layers unsupported (dense SwiGLU/GeGLU only)")
	}
	d, H, I := r.d, r.H, r.I
	limit := d.MaxThreadgroupMemoryLength()

	bx := d.NewBufferLen(M * H)
	bxf := bx.Floats()
	for m, e := range embs {
		if len(e) != H {
			return bvkResult{}, fmt.Errorf("bvkForwardM: embedding %d has len %d, want %d", m, len(e), H)
		}
		copy(bxf[m*H:(m+1)*H], e)
	}

	uH, uI, u2I := d.NewBufferU32(uint32(H)), d.NewBufferU32(uint32(I)), d.NewBufferU32(uint32(2*I))
	uKK := d.NewBufferU32(uint32(M))

	e := r.q.Begin()
	for l := 0; l < r.nL; l++ {
		L := &r.layers[l]
		g := L.geom
		nHhd := r.nH * g.hd
		qkvRows := nHhd + 2*g.kvDim
		kOff, vOff := nHhd*4, (nHhd+g.kvDim)*4
		uNHhdOut, uQkvRows := d.NewBufferU32(uint32(nHhd)), d.NewBufferU32(uint32(qkvRows))

		// --- attention block ---
		aq, asc := byteBuf(d, M*H), d.NewBufferFloats(make([]float32, M))
		for m := 0; m < M; m++ {
			e.Dispatch(r.pRms, tgReduceNorm, tgReduceNorm, bx.At(m*H*4), L.preNorm, aq.At(m*H), asc.At(m*4), uH, r.uEps, r.uAddOne)
		}
		if tgb := bvkThreadgroupBytes(H, M); tgb > limit {
			return bvkResult{}, fmt.Errorf("bvkForwardM: layer %d QKV proj needs %d B threadgroup > device limit %d B at M=%d (K=%d)", l, tgb, limit, M, H)
		}
		qkv := d.NewBufferLen(M * qkvRows)
		e.DispatchTG(pipes.bias, qkvRows*32, 256, bvkThreadgroupBytes(H, M), L.qkvW, L.qkvS, aq, asc, qkv, L.qkvBias, uH, uQkvRows, uKK)

		if r.qkNorm {
			for m := 0; m < M; m++ {
				e.Dispatch(r.pQKNorm, (r.nH+g.nKV)*tgReduceAttn, tgReduceAttn, qkv.At(m*qkvRows*4), L.qNorm, L.kNorm, r.uNH, g.uNKV, g.uHd, g.uNHhd, r.uEps, r.uAddOne)
			}
		}
		if g.kEqV {
			for m := 0; m < M; m++ {
				e.Dispatch(r.pQKNorm, g.nKV*tgReduceAttn, tgReduceAttn, qkv.At(m*qkvRows*4+vOff), r.vNormUnit, r.vNormUnit, r.uZero, g.uNKV, g.uHd, r.uZero, r.uEps, r.uZero)
			}
		}

		cq, cSc := byteBuf(d, M*nHhd), d.NewBufferFloats(make([]float32, M))
		for m := 0; m < M; m++ {
			pos := startPos + m
			uPos, uNKeys := d.NewBufferU32(uint32(pos)), d.NewBufferU32(uint32(pos+1))
			e.Dispatch(r.pRope, r.nH*g.half, 64, qkv.At(m*qkvRows*4), L.invf, g.uHd, uPos, g.uQtotal, g.uHalf)
			e.Dispatch(r.pRope, g.nKV*g.half, 64, qkv.At(m*qkvRows*4+kOff), L.invf, g.uHd, uPos, g.uKtotal, g.uHalf)
			e.Dispatch(r.pKv, g.kvDim, 64, qkv.At(m*qkvRows*4+kOff), qkv.At(m*qkvRows*4+vOff), r.kc[l], r.vc[l], g.uKvDim, uPos)
			ctx := d.NewBufferLen(nHhd)
			e.Dispatch(r.pAttn, r.nH*tgReduceAttn, tgReduceAttn, qkv.At(m*qkvRows*4), r.kc[l], r.vc[l], ctx, r.uNH, g.uNKV, g.uHd, uNKeys, r.uScale, L.uWindow)
			e.Dispatch(r.pQv, 256, 256, ctx, cq.At(m*nHhd), cSc.At(m*4), g.uNHhd)
		}

		if tgb := bvkThreadgroupBytes(nHhd, M); tgb > limit {
			return bvkResult{}, fmt.Errorf("bvkForwardM: layer %d O-proj needs %d B threadgroup > device limit %d B at M=%d (K=%d)", l, tgb, limit, M, nHhd)
		}
		if r.sandwich {
			// Decode's OWN sandwich dispatch never fuses this epilogue either (a norm sits between
			// the projection and the residual add), so the plain-then-separate-add split IS the
			// bit-identical match here — mirrors metal/model.go's r.sandwich branch exactly.
			oOut := d.NewBufferLen(M * H)
			e.DispatchTG(pipes.plain, H*32, 256, bvkThreadgroupBytes(nHhd, M), L.oW, L.oS, cq, cSc, oOut, uNHhdOut, uH, uKK)
			for m := 0; m < M; m++ {
				e.Dispatch(r.pRmsF32, tgReduceNorm, tgReduceNorm, oOut.At(m*H*4), L.postAttnNorm, uH, r.uEps, r.uAddOne)
			}
			for m := 0; m < M; m++ {
				e.Dispatch(r.pRes, H, 256, bx.At(m*H*4), oOut.At(m*H*4))
			}
		} else {
			// Decode's non-sandwich O-proj (gemv_w4a8_sa_resid) fuses "+= acc*asc[0]" in one kernel —
			// under fast-math that's licensed to become an fma with a full-precision intermediate
			// product, which a split write-then-add (two independent roundings) does not reproduce.
			// bx IS the M x H output buffer here (N=H, so out[j*N+row] aliases bx[j*H+row] exactly).
			e.DispatchTG(pipes.plainResid, H*32, 256, bvkThreadgroupBytes(nHhd, M), L.oW, L.oS, cq, cSc, bx, uNHhdOut, uH, uKK)
		}

		// --- ffn block (dense SwiGLU/GeGLU only) ---
		mq, mSc := byteBuf(d, M*H), d.NewBufferFloats(make([]float32, M))
		for m := 0; m < M; m++ {
			e.Dispatch(r.pRms, tgReduceNorm, tgReduceNorm, bx.At(m*H*4), L.postNorm, mq.At(m*H), mSc.At(m*4), uH, r.uEps, r.uAddOne)
		}
		if tgb := bvkThreadgroupBytes(H, M); tgb > limit {
			return bvkResult{}, fmt.Errorf("bvkForwardM: layer %d gate/up proj needs %d B threadgroup > device limit %d B at M=%d (K=%d)", l, tgb, limit, M, H)
		}
		guOut := d.NewBufferLen(M * 2 * I)
		e.DispatchTG(pipes.plain, (2*I)*32, 256, bvkThreadgroupBytes(H, M), L.guW, L.guS, mq, mSc, guOut, uH, u2I, uKK)

		dq, dSc := byteBuf(d, M*I), d.NewBufferFloats(make([]float32, M))
		for m := 0; m < M; m++ {
			e.Dispatch(r.pSw, 256, 256, guOut.At(m*2*I*4), guOut.At(m*2*I*4+I*4), dq.At(m*I), dSc.At(m*4), uI, r.uAct)
		}

		// down-proj: unstaged (bvk_coal/_resid), so no threadgroup-memory check needed regardless of M.
		if r.sandwich {
			downOut := d.NewBufferLen(M * H)
			e.Dispatch(pipes.coal, H*32, 32, L.dW, L.dS, dq, dSc, downOut, uI, uH, uKK)
			for m := 0; m < M; m++ {
				e.Dispatch(r.pRmsF32, tgReduceNorm, tgReduceNorm, downOut.At(m*H*4), L.postMLPNorm, uH, r.uEps, r.uAddOne)
			}
			for m := 0; m < M; m++ {
				e.Dispatch(r.pRes, H, 256, bx.At(m*H*4), downOut.At(m*H*4))
			}
		} else {
			// Decode's non-sandwich down-proj (gemv_w4a8_resid) also fuses the residual add — same
			// fma-vs-split-rounding reasoning as the O-proj branch above.
			e.Dispatch(pipes.coalResid, H*32, 32, L.dW, L.dS, dq, dSc, bx, uI, uH, uKK)
		}
	}

	// --- final norm + LM head (per-token: the int8 LM head is out of scope for this W4A8
	// investigation, and is decode's own unmodified pipeline either way) ---
	finalAq, finalAsc := byteBuf(d, M*H), d.NewBufferFloats(make([]float32, M))
	for m := 0; m < M; m++ {
		e.Dispatch(r.pRms, tgReduceNorm, tgReduceNorm, bx.At(m*H*4), r.finalNorm, finalAq.At(m*H), finalAsc.At(m*4), uH, r.uEps, r.uAddOne)
	}
	logitsOut := d.NewBufferLen(M * r.V)
	for m := 0; m < M; m++ {
		e.Dispatch(r.pGemvW8, r.V*32, 32, finalAq.At(m*H), finalAsc.At(m*4), r.lmW, r.lmS, logitsOut.At(m*r.V*4), uH)
	}
	e.End()

	res := bvkResult{}
	lf, hf := logitsOut.Floats(), bxf
	for m := 0; m < M; m++ {
		res.logits = append(res.logits, append([]float32(nil), lf[m*r.V:(m+1)*r.V]...))
		res.hidden = append(res.hidden, append([]float32(nil), hf[m*H:(m+1)*H]...))
	}
	return res, nil
}

// bvkMaxM returns the largest M in [1,16] whose per-layer QKV/O-proj/gate-up threadgroup staging
// (2*K*M bytes, the widest K being max(H, nH*hd)) fits under the device's threadgroup-memory
// limit. Down-proj is unstaged (bvk_coal) so it never constrains M. Honest capping, not a silent
// one: callers report the returned M instead of assuming 16 always fits (it does not — see
// docs/task-metal-batched-verify-kernel.md Phase 1).
func bvkMaxM(d *Device, wideK int) int {
	limit := d.MaxThreadgroupMemoryLength()
	for m := 16; m >= 1; m-- {
		if bvkThreadgroupBytes(wideK, m) <= limit {
			return m
		}
	}
	return 1
}

// loadBVKPair loads the SAME checkpoint into two independent Metal residents — one to serve as
// the sequential ground truth (decode's real, unmodified per-token Forward), one to drive
// bvkForwardM — so the batched path never shares KV-cache or scratch-buffer state with the
// sequential path it's being checked against.
func loadBVKPair(t *testing.T, path string) (seq decoder.ResidentForward, embed func(int) []float32, batched *resident, pipes bvkPipes) {
	t.Helper()
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s", path)
	}
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	mSeq, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load (seq): %v", err)
	}
	rfSeq := mSeq.ResidentForwardForTest()
	if rfSeq == nil {
		t.Fatal("metal resident DECLINED for the sequential side — admission says it should be admitted")
	}
	mBatch, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load (batched): %v", err)
	}
	rfBatch := mBatch.ResidentForwardForTest()
	if rfBatch == nil {
		t.Fatal("metal resident DECLINED for the batched side — admission says it should be admitted")
	}
	mr, ok := rfBatch.(*metalResident)
	if !ok {
		t.Fatalf("batched resident is %T, want *metalResident", rfBatch)
	}
	pipes, err = newBVKPipes(mr.r.d)
	if err != nil {
		t.Fatalf("newBVKPipes: %v", err)
	}
	return rfSeq, mSeq.EmbedResidentForTest, mr.r, pipes
}

// TestBatchedVerifyKernelParity is Phase 2: bit-identity, not tolerance. It runs the SAME M draft
// tokens, at the SAME KV positions, through (a) decode's real sequential Forward, one token at a
// time, and (b) bvkForwardM's batched-M path, on two INDEPENDENT residents loaded from the same
// checkpoint (so KV state can never leak between them) — and asserts EXACT float equality
// (math.Float32bits, not a tolerance) of every logit and every hidden-state element, at every
// position, for M in {2,4,8,16} (capped per model by bvkMaxM — see docs/task-*.md Phase 1).
//
// A tolerance failure here would mean Phase 0's numeric-contract analysis missed something in the
// actual per-layer dispatch (this test exercises the REAL weights/RoPE/attention/KV of a real
// checkpoint, not a synthetic single-kernel comparison) — go back and find it, per the task.
func TestBatchedVerifyKernelParity(t *testing.T) {
	requireHeavyModel(t)
	for _, tc := range []struct {
		name string
		path string
	}{
		{"qwen2.5-coder-0.5b (dense, non-sandwich)", os.ExpandEnv("$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")},
		{"gemma-3-4b (sandwich)", os.ExpandEnv("$HOME/models/gemma-3-4b-it-Q4_K_M.gguf")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if p := os.Getenv("GOINFER_METAL_MODEL"); p != "" {
				tc.path = p
			}
			seq, embed, r, pipes := loadBVKPair(t, tc.path)
			wideK := r.H
			for l := range r.layers {
				if nHhd := r.nH * r.layers[l].geom.hd; nHhd > wideK {
					wideK = nHhd
				}
			}
			maxM := bvkMaxM(r.d, wideK)
			t.Logf("%s: H=%d I=%d nL=%d widest staged K=%d -> bvkMaxM=%d", tc.name, r.H, r.I, r.nL, wideK, maxM)
			for _, M := range []int{2, 4, 8, 16} {
				t.Run(fmt.Sprintf("M=%d", M), func(t *testing.T) {
					if M > maxM {
						t.Skipf("M=%d exceeds threadgroup-memory budget for this model's widest K=%d (bvkMaxM=%d) — see Phase 1", M, wideK, maxM)
					}
					// Ground truth: decode's own sequential Forward, M tokens at consecutive positions,
					// starting fresh (pos 0) on the "seq" resident.
					embs := make([][]float32, M)
					wantLogits := make([][]float32, M)
					wantHidden := make([][]float32, M)
					for m := 0; m < M; m++ {
						tok := (m*997 + 13) % 5000 // arbitrary but deterministic and reproducible token ids
						embs[m] = embed(tok)
						l, err := seq.Forward(embs[m], m)
						if err != nil {
							t.Fatalf("sequential Forward pos %d: %v", m, err)
						}
						wantLogits[m] = append([]float32(nil), l...)
					}
					// The sequential ground truth's post-trunk hidden state isn't directly observable
					// through the public Forward — capture it via the existing forwardTrunkForTest hook
					// on a THIRD independent resident driven identically, so hidden-state parity is
					// checked against the real trunk output, not inferred from logits alone.
					mHidden, err := decoder.Load(tc.path, decoder.Options{Backend: "metal", Quant: "int8int8"})
					if err != nil {
						t.Fatalf("load (hidden-capture): %v", err)
					}
					rfHidden, ok := mHidden.ResidentForwardForTest().(*metalResident)
					if !ok || rfHidden == nil {
						t.Fatal("hidden-capture resident DECLINED")
					}
					for m := 0; m < M; m++ {
						wantHidden[m] = rfHidden.r.forwardTrunkForTest(embs[m], m, rfHidden.r.nL)
					}

					got, err := bvkForwardM(r, pipes, embs, 0)
					if err != nil {
						t.Fatalf("bvkForwardM: %v", err)
					}

					mismatchedLogits, mismatchedHidden := 0, 0
					for m := 0; m < M; m++ {
						if !exactEqual(wantLogits[m], got.logits[m]) {
							mismatchedLogits++
							if mismatchedLogits <= 3 {
								t.Errorf("pos %d: logits NOT bit-identical (first diff %s)", m, firstDiff(wantLogits[m], got.logits[m]))
							}
						}
						if !exactEqual(wantHidden[m], got.hidden[m]) {
							mismatchedHidden++
							if mismatchedHidden <= 3 {
								t.Errorf("pos %d: hidden state NOT bit-identical (first diff %s)", m, firstDiff(wantHidden[m], got.hidden[m]))
							}
						}
					}
					if mismatchedLogits == 0 && mismatchedHidden == 0 {
						t.Logf("M=%d: %d/%d positions bit-identical (logits and hidden state)", M, M, M)
					}
				})
			}
		})
	}
}

// exactEqual compares two float32 slices by RAW BITS (math.Float32bits), not by value — so a NaN
// in both at the same position counts as equal (matching the "bit-identical" bar this test is
// gating on) and +0/-0 count as DIFFERENT (also matching "bit-identical", not "numerically equal").
func exactEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if f32bits(a[i]) != f32bits(b[i]) {
			return false
		}
	}
	return true
}

func f32bits(f float32) uint32 {
	return math.Float32bits(f)
}

// firstDiff reports the index and values of the first bit-differing element, for a useful failure
// message without dumping the whole vector.
func firstDiff(a, b []float32) string {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if f32bits(a[i]) != f32bits(b[i]) {
			return fmt.Sprintf("[%d] want=%v got=%v (bits %08x vs %08x)", i, a[i], b[i], f32bits(a[i]), f32bits(b[i]))
		}
	}
	if len(a) != len(b) {
		return fmt.Sprintf("length mismatch %d vs %d", len(a), len(b))
	}
	return "(no diff found — race?)"
}

// TestBatchedVerifyE2ECurve is Phase 3(a)/Phase 4's real input: the full multi-layer,
// real-weights T(M) = W + C*M curve for bvkForwardM, mirroring
// metal/spec_verify_curve_test.go's exact methodology (same model, same depth=1024, same
// least-squares fit, same separately-measured real decode_ms) so the two are directly
// comparable — this curve is this task's replacement for that one, using a kernel that is
// actually bit-identical (TestBatchedVerifyKernelParity) instead of PrefillLast's f16-MMA path,
// which is not.
//
// CAVEAT stated up front, not buried: bvkForwardM allocates fresh scratch buffers every call (it
// was written for correctness clarity in the parity test, not tuned for repeated dispatch) —
// unlike PrefillLast, which reuses the resident's own persistent buffers across calls. These
// numbers therefore include Go/Metal buffer (re)allocation overhead a tuned implementation would
// eliminate by reusing scratch buffers across tokens, the same way decode's own resident already
// does for its single-token buffers. TestBatchedVerifyKernelBench's pure-kernel numbers (which DO
// reuse buffers via prof()) bound how much of this curve's fixed cost is allocation versus real
// kernel dispatch — read the two together, not this one in isolation.
func TestBatchedVerifyE2ECurve(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a 1.5B model)")
	}
	path := os.Getenv("GOINFER_METAL_MODEL")
	if path == "" {
		path = os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf") // matches spec_verify_curve_test.go, for direct comparability
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s (set GOINFER_METAL_MODEL)", path)
	}
	seq, embed, r, pipes := loadBVKPair(t, path)

	wideK := r.H
	for l := range r.layers {
		if nHhd := r.nH * r.layers[l].geom.hd; nHhd > wideK {
			wideK = nHhd
		}
	}
	maxM := bvkMaxM(r.d, wideK)
	t.Logf("H=%d widest staged K=%d -> bvkMaxM=%d (this model's real cap, not assumed 16)", r.H, wideK, maxM)

	const depth = 1024 // matches spec_verify_curve_test.go's depth for direct comparability
	// Warm both paths up to `depth` real KV positions first, so the timed calls measure
	// steady-state cost at a realistic KV depth, not cold-cache/empty-context cost. A naive
	// one-token-at-a-time loop to depth 1024 is O(depth^2) (each Forward's attention cost grows
	// with depth) and was measured elsewhere in this repo to take minutes — so, exactly like
	// spec_verify_curve_test.go itself, warmup uses the (non-bit-identical, but irrelevant for
	// warmup — only the TIMED calls below need to be the real thing) f16-MMA PrefillLast to reach
	// depth fast, forced on via the same env var that test uses.
	t.Setenv("GOINFER_METAL_BATCHED_PREFILL", "1")
	seqPF, ok := seq.(decoder.Prefiller)
	if !ok {
		t.Fatal("sequential resident does not implement decoder.Prefiller — cannot fast-warm")
	}
	warmSeq := make([][]float32, depth)
	for i := range warmSeq {
		warmSeq[i] = embed(i % 4999)
	}
	if _, err := seqPF.PrefillLast(warmSeq, 0); err != nil {
		t.Fatalf("warm seq (forced batched prefill): %v", err)
	}
	warmBatch := make([][]float32, depth)
	for i := range warmBatch {
		warmBatch[i] = embed(i % 4999)
	}
	rPF := &metalResident{r: r, hidden: r.H}
	if _, err := rPF.PrefillLast(warmBatch, 0); err != nil {
		t.Fatalf("warm batch (forced batched prefill): %v", err)
	}
	timeIt := func(f func()) time.Duration {
		best := time.Hour
		for range 5 {
			t0 := time.Now()
			f()
			if d := time.Since(t0); d < best {
				best = d
			}
		}
		return best
	}

	realDecode := timeIt(func() {
		if _, err := seq.Forward(embed(depth+5), depth); err != nil {
			t.Fatal(err)
		}
	})
	t.Logf("REAL decode_ms (sequential Forward, shipped int8 path) @depth %d: %.3f ms", depth, float64(realDecode.Microseconds())/1000)

	type point struct {
		m int
		t time.Duration
	}
	var points []point
	for _, k := range []int{1, 2, 4, 6, 8, 10, 12, 16} {
		if k > maxM {
			t.Logf("M=%-3d SKIPPED — exceeds bvkMaxM=%d for this model", k, maxM)
			continue
		}
		ek := make([][]float32, k)
		for i := range ek {
			ek[i] = embed(depth + 100 + i)
		}
		d := timeIt(func() {
			if _, err := bvkForwardM(r, pipes, ek, depth); err != nil {
				t.Fatal(err)
			}
		})
		points = append(points, point{k, d})
		t.Logf("M=%-3d T(M)=%.3f ms", k, float64(d.Microseconds())/1000)
	}

	var n, sumM, sumT, sumMM, sumMT float64
	for _, p := range points {
		mF, tF := float64(p.m), float64(p.t.Microseconds())/1000
		n++
		sumM += mF
		sumT += tF
		sumMM += mF * mF
		sumMT += mF * tF
	}
	c := (n*sumMT - sumM*sumT) / (n*sumMM - sumM*sumM)
	w := (sumT - c*sumM) / n
	t.Logf("fit: T(M) = %.3f + %.3f*M  (ms)", w, c)
	realDecodeMs := float64(realDecode.Microseconds()) / 1000
	t.Logf("ratio T(1)_fit / real_decode_ms = %.2fx", (w+c)/realDecodeMs)

	fmt.Printf("\nMETAL BATCHED-VERIFY E2E CURVE (bit-identical) — %s, depth %d\nW=%.4f C=%.4f  real_decode_ms=%.4f  bvkMaxM=%d\n",
		path, depth, w, c, realDecodeMs, maxM)
}

// TestBatchedVerifyKernelBench is Phase 3(a)/(c): the amortization curve — batched-M kernel time
// vs M x single-token kernel time — for the three W4A8 GEMV shapes (SA-style QKV/O/gate-up, and
// the unstaged down-proj), at REAL shipped-model dims, across M in {2,4,8,16} (skipping M values
// that exceed the threadgroup budget for a given shape, reported not hidden). Extends the existing
// metal/batchk_test.go methodology (same prof() pattern) from one shape (gate/up @ 1.5b) to all
// four dense projections across three real model sizes.
func TestBatchedVerifyKernelBench(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 to run (device-only, no checkpoint needed)")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Fatalf("device: %v", err)
	}
	pipes, err := newBVKPipes(d)
	if err != nil {
		t.Fatalf("newBVKPipes: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile allKernels: %v", err)
	}
	pipe := func(n string) Pipeline {
		p, e := d.NewComputePipeline(lib, n)
		if e != nil {
			t.Fatalf("%s: %v", n, e)
		}
		return p
	}
	pSA, pCoal := pipe("gemv_w4a8_sa"), pipe("gemv_w4a8_coal")

	prof := func(reps int, run func(int)) time.Duration {
		for range 4 {
			run(reps)
		}
		best := time.Hour
		for range 20 {
			t0 := time.Now()
			run(reps)
			if dt := time.Since(t0); dt < best && dt > 0 {
				best = dt
			}
		}
		return best / time.Duration(reps)
	}

	// kind: "bias" = QKV in-proj (SA-style, staged, has a bias epilogue), "plain" = O-proj /
	// gate-up-proj (SA-style, staged, no epilogue), "coal" = down-proj (unstaged, no epilogue).
	type shape struct {
		name string
		N, K int
		kind string
	}
	type modelShapes struct {
		name              string
		H, I, nH, nKV, hd int
	}
	models := []modelShapes{
		{"qwen2.5-coder-0.5b", 896, 4864, 14, 2, 64},
		{"qwen2.5-coder-1.5b", 1536, 8960, 12, 2, 128},
		{"gemma-3-4b", 2560, 10240, 8, 4, 320},
	}

	for _, mo := range models {
		shapes := []shape{
			{"qkv-proj", mo.nH*mo.hd + 2*mo.nKV*mo.hd, mo.H, "bias"},
			{"o-proj", mo.H, mo.nH * mo.hd, "plain"},
			{"gate/up-proj", 2 * mo.I, mo.H, "plain"},
			{"down-proj", mo.H, mo.I, "coal"},
		}
		for _, sh := range shapes {
			t.Run(fmt.Sprintf("%s/%s(N=%d,K=%d)", mo.name, sh.name, sh.N, sh.K), func(t *testing.T) {
				// t.Run spawns a NEW goroutine per subtest — the outer LockOSThread does not cover
				// it. Every real dispatch function in this package (bvkForwardM included) pins the
				// OS thread at its own top for exactly this reason; omitting it here caused a
				// deterministic SIGSEGV in objc.Send (a goroutine migrating OS threads mid-dispatch),
				// not a bug in the kernels themselves.
				runtime.LockOSThread()
				defer runtime.UnlockOSThread()
				wq := d.NewBufferUint32s(make([]uint32, sh.N*(sh.K/8)))
				sc := d.NewBufferU16s(make([]uint16, sh.N*(sh.K/32)))
				aq := byteBuf(d, 16*sh.K)
				asc := d.NewBufferFloats(make([]float32, 16))
				out := d.NewBufferLen(16 * sh.N)
				bias := d.NewBufferFloats(make([]float32, sh.N))
				uK, uN := d.NewBufferU32(uint32(sh.K)), d.NewBufferU32(uint32(sh.N))
				q := d.NewCommandQueue()

				var single time.Duration
				switch sh.kind {
				case "bias":
					single = prof(200, func(r int) { q.Run1DBatch(pSA, sh.N*32, 256, r, wq, sc, aq, asc, out, uK) }) // gemv_w4a8_sa_bias ~= gemv_w4a8_sa's GEMV cost
				case "plain":
					single = prof(200, func(r int) { q.Run1DBatch(pSA, sh.N*32, 256, r, wq, sc, aq, asc, out, uK) })
				default:
					single = prof(200, func(r int) { q.Run1DBatch(pCoal, sh.N*32, 32, r, wq, sc, aq, asc, out, uK) })
				}
				t.Logf("  single-token: %.1f us", float64(single.Microseconds()))

				limit := d.MaxThreadgroupMemoryLength()
				for _, M := range []int{2, 4, 8, 16} {
					uMM := d.NewBufferU32(uint32(M))
					var bk time.Duration
					switch sh.kind {
					case "bias":
						tgb := bvkThreadgroupBytes(sh.K, M)
						if tgb > limit {
							t.Logf("  M=%2d: SKIPPED — needs %d B threadgroup > device limit %d B", M, tgb, limit)
							continue
						}
						bk = prof(200, func(r int) { q.Run1DBatchTG(pipes.bias, sh.N*32, 256, r, tgb, wq, sc, aq, asc, out, bias, uK, uN, uMM) })
					case "plain":
						tgb := bvkThreadgroupBytes(sh.K, M)
						if tgb > limit {
							t.Logf("  M=%2d: SKIPPED — needs %d B threadgroup > device limit %d B", M, tgb, limit)
							continue
						}
						bk = prof(200, func(r int) { q.Run1DBatchTG(pipes.plain, sh.N*32, 256, r, tgb, wq, sc, aq, asc, out, uK, uN, uMM) })
					default:
						bk = prof(200, func(r int) { q.Run1DBatch(pipes.coal, sh.N*32, 32, r, wq, sc, aq, asc, out, uK, uN, uMM) })
					}
					loop := time.Duration(M) * single
					t.Logf("  M=%2d: batched %7.1f us  vs  %d x single %7.1f us  =>  %.2fx, %.2f us/token (%.1f%% of single)",
						M, float64(bk.Microseconds()), M, float64(loop.Microseconds()),
						float64(loop)/float64(bk), float64(bk.Microseconds())/float64(M), 100*float64(bk)/float64(M)/float64(single))
				}
			})
		}
	}
}
