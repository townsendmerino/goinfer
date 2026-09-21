//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"math"
	"os"
	"sort"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestR1_guReference is experiment X1 of the R1 (W4F16 decode lane) re-investigation. The
// parked record (docs/measurements/w4f16-decode-investigation-2026-09-19.md) scored the f16
// gate/up GEMV output against the SHIPPED W4A8 lane's output and read the disagreement as an
// f16 defect. That instrument cannot say which arm is off: W4A8 quantizes the normed FFN input
// to int8 with ONE per-tensor scale (amax/127), which at layer 26 (amax ~146) zeroes most of the
// non-massive channels. This test scores BOTH arms against an f64 reference built from the
// SAME resident int4 weight bits and the SAME post-attention residual, per layer, and also
// checks each arm's kernel against the f64 dot of the exact activation the kernel received
// (the half-rounded activation for f16, mq*mSc for W4A8) — the kernel-on-real-data check.
// Finally it re-runs the known layer-26/27 repro and decomposes the residual difference by
// channel. Read-only on production code; one model load; ~14 MB of read-back per layer.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags "darwin goinfer_testhooks" ./metal/ -run 'TestR1_guReference$' -v
func TestR1_guReference(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a real checkpoint)")
	}
	path := os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	// Lane OFF at load: the lane is toggled at runtime via r.decodeLaneW4F16 on this one model.
	t.Setenv("GOINFER_METAL_DECODE_LANE", "")

	logf := func(format string, args ...any) {
		t.Helper()
		t.Logf(format, args...)
		fmt.Fprintf(os.Stderr, "[r1-x1] "+format+"\n", args...)
	}

	m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	rf := m.ResidentForwardForTest().(*metalResident)
	r := rf.r
	const tok = 785
	H, I, K := r.H, r.I, r.H
	N := 2 * I
	G := K / 32
	eps := float64(r.uEps.Floats()[0])
	addOne := r.uAddOne.U32s()[0]
	logf("model: H=%d I=%d nL=%d N(gate|up)=%d G=%d eps=%g addOne=%d NormEps()=%g lane=%v",
		H, I, r.nL, N, G, eps, addOne, m.NormEps(), r.decodeLaneW4F16)
	if addOne != 0 {
		t.Fatalf("addOne=%d: reference assumes plain RMSNorm weight (Qwen)", addOne)
	}
	if !r.canUseF16Lane(26) {
		// canUseF16Lane reads decodeLaneW4F16 first; check eligibility with the lane on.
		r.decodeLaneW4F16 = true
		ok := r.canUseF16Lane(26)
		r.decodeLaneW4F16 = false
		if !ok {
			t.Fatalf("layer 26 not eligible for the f16 lane — premise changed")
		}
	}

	embed := append([]float32(nil), m.EmbedResidentForTest(tok)...)
	reset := func() {
		copy(r.x.Floats(), embed)
		r.setPos(0)
		r.decodeLaneW4F16 = false
	}
	runLayers := func(from, to int) { // [from, to)
		for l := from; l < to; l++ {
			e := r.q.Begin()
			r.encodeLayer(e, l)
			e.End()
		}
	}
	cosine := func(a, b []float64) float64 {
		var dot, na, nb float64
		for i := range a {
			dot += a[i] * b[i]
			na += a[i] * a[i]
			nb += b[i] * b[i]
		}
		return dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-300)
	}
	amaxOf := func(a []float64) (float64, int) {
		var mx float64
		idx := -1
		for i, v := range a {
			if math.Abs(v) > mx {
				mx = math.Abs(v)
				idx = i
			}
		}
		return mx, idx
	}
	// maxRel: max over rows of |got-ref| / (|ref| + 1e-3*amax(ref)).
	maxRel := func(got, ref []float64) (float64, int) {
		am, _ := amaxOf(ref)
		var mx float64
		idx := -1
		for i := range got {
			d := math.Abs(got[i]-ref[i]) / (math.Abs(ref[i]) + 1e-3*am)
			if d > mx {
				mx = d
				idx = i
			}
		}
		return mx, idx
	}
	silu := func(x float64) float64 { return x / (1 + math.Exp(-x)) }
	toF64 := func(a []float32) []float64 {
		out := make([]float64, len(a))
		for i, v := range a {
			out[i] = float64(v)
		}
		return out
	}

	type rowErr struct {
		row  int
		diff float64 // |got-ref|
		rel  float64
		got  float64
		ref  float64
	}
	worstRows := func(got, ref []float64, n int) []rowErr {
		am, _ := amaxOf(ref)
		rs := make([]rowErr, len(got))
		for i := range got {
			d := math.Abs(got[i] - ref[i])
			rs[i] = rowErr{i, d, d / (math.Abs(ref[i]) + 1e-3*am), got[i], ref[i]}
		}
		sort.Slice(rs, func(a, b int) bool { return rs[a].rel > rs[b].rel })
		return rs[:n]
	}

	type layerResult struct {
		layer                                                       int
		actAmax                                                     float64
		zeroFrac, le1Frac                                           float64
		cosF16, cosW4A8, cosF16Half, cosW4A8Q8                      float64
		relF16, relW4A8, relF16Half, relW4A8Q8                      float64
		producerMaxRel                                              float64
		hAmaxF16, hAmaxW4A8, dScF16, dScW4A8                        float64
		hArgF16, hArgW4A8                                           int
		fracRowsF16Closer                                           float64
		row2908Ref, row2908F16, row2908W4A8, row2908Half, row2908Q8 float64
	}
	var results []layerResult

	targets := []int{0, 2, 8, 16, 24, 25, 26, 27}
	for _, l := range targets {
		if l >= r.nL {
			continue
		}
		L := &r.layers[l]
		logf("=== layer %d: replaying W4A8 lane through layers 0..%d, then attention(%d) ===", l, l-1, l)
		reset()
		runLayers(0, l)
		e := r.q.Begin()
		r.encodeAttention(e, l)
		e.End()
		xMid := append([]float32(nil), r.x.Floats()[:H]...) // post-attention residual = FFN input
		xAmax, xArg := amaxOf(toF64(xMid))
		gw := L.postNorm.Floats()[:H]

		// ---- f16 arm (verbatim production dispatches, model.go L2186 + L2191), then pSw ----
		e = r.q.Begin()
		e.Dispatch(r.pRmsF16, tgReduceNorm, tgReduceNorm, r.x, L.postNorm, r.mxF16, r.uH, r.uEps, r.uAddOne)
		e.DispatchTG(r.pSAf16, (2*r.I)*32, 256, r.H*2, L.guW, L.guS, r.mxF16, r.gu, r.uH)
		e.Dispatch(r.pSw, 256, 256, r.gu, r.gu.At(r.I*4), r.dq, r.dSc, r.uI, r.uAct)
		e.End()
		guF16 := toF64(append([]float32(nil), r.gu.Floats()[:N]...))
		axHalf := append([]uint16(nil), r.mxF16.U16s()[:H]...)
		dScF16 := float64(r.dSc.Floats()[0])

		// ---- W4A8 arm (verbatim production dispatches, model.go L2188 + L2193), then pSw ----
		e = r.q.Begin()
		r.encodeNorm(e, r.x, L.postNorm, L.postNormBias, r.mq, r.mSc)
		e.DispatchTG(r.pSA, (2*r.I)*32, 256, r.H*2, L.guW, L.guS, r.mq, r.mSc, r.gu, r.uH)
		e.Dispatch(r.pSw, 256, 256, r.gu, r.gu.At(r.I*4), r.dq, r.dSc, r.uI, r.uAct)
		e.End()
		guW4A8 := toF64(append([]float32(nil), r.gu.Floats()[:N]...))
		mq := append([]int8(nil), r.mq.Int8s()[:H]...)
		mSc := float64(r.mSc.Floats()[0])
		dScW4A8 := float64(r.dSc.Floats()[0])

		// Sanity: r.x must be untouched by the arm dispatches.
		for i := 0; i < H; i++ {
			if r.x.Floats()[i] != xMid[i] {
				t.Fatalf("layer %d: r.x changed during arm dispatches at channel %d", l, i)
			}
		}

		// ---- CPU f64 activations ----
		var ss float64
		for _, v := range xMid {
			ss += float64(v) * float64(v)
		}
		rms := 1 / math.Sqrt(ss/float64(H)+eps)
		aRef := make([]float64, H)
		aHalfRB := make([]float64, H) // the half activation the f16 kernel actually received
		aHalfCPU := make([]float64, H)
		aQ8 := make([]float64, H) // what the W4A8 kernel actually received (dequantized)
		var producerMaxRel float64
		var nZero, nLe1 int
		for k := 0; k < H; k++ {
			aRef[k] = float64(xMid[k]) * rms * float64(gw[k])
			aHalfRB[k] = float64(f16ToF32(axHalf[k]))
			aHalfCPU[k] = float64(f16ToF32(f32ToF16(float32(aRef[k]))))
			aQ8[k] = float64(mq[k]) * mSc
			if mq[k] == 0 {
				nZero++
			}
			if mq[k] >= -1 && mq[k] <= 1 {
				nLe1++
			}
			if d := math.Abs(aHalfRB[k]-aRef[k]) / (math.Abs(aRef[k]) + 1e-6); d > producerMaxRel {
				producerMaxRel = d
			}
		}
		actAmax, actArg := amaxOf(aRef)
		// The W4A8 quant step the producer used vs amax/127.
		logf("layer %d: xMid amax=%.4g at ch %d | normed aRef amax=%.4f at ch %d | mSc=%.5g (amax/127=%.5g) | mq==0: %d/%d (%.1f%%) |mq|<=1: %d/%d (%.1f%%) | producer(f16) max rel err vs aRef=%.3g",
			l, xAmax, xArg, actAmax, actArg, mSc, actAmax/127, nZero, H, 100*float64(nZero)/float64(H),
			nLe1, H, 100*float64(nLe1)/float64(H), producerMaxRel)

		// ---- CPU f64 reference dots, one weight row at a time ----
		words := L.guW.U32s()
		scales := L.guS.U16s()
		if len(words) < N*K/8 || len(scales) < N*G {
			t.Fatalf("layer %d: weight buffers too short: words=%d need %d, scales=%d need %d", l, len(words), N*K/8, len(scales), N*G)
		}
		guRef := make([]float64, N)
		guHalfRef := make([]float64, N)
		guQ8Ref := make([]float64, N)
		nibAt := func(n, k int) uint32 {
			return (words[n*(K/8)+k/8] >> (4 * uint(k%8))) & 0xF
		}
		wrow := make([]float64, K)
		for n := 0; n < N; n++ {
			for g := 0; g < G; g++ {
				sc := float64(f16ToF32(scales[n*G+g]))
				for kk := 0; kk < 32; kk++ {
					k := g*32 + kk
					wrow[k] = float64(int(nibAt(n, k))-8) * sc
				}
			}
			var d0, d1, d2 float64
			for k := 0; k < K; k++ {
				d0 += wrow[k] * aRef[k]
				d1 += wrow[k] * aHalfRB[k]
				d2 += wrow[k] * aQ8[k]
			}
			guRef[n], guHalfRef[n], guQ8Ref[n] = d0, d1, d2
		}

		res := layerResult{layer: l, actAmax: actAmax,
			zeroFrac: float64(nZero) / float64(H), le1Frac: float64(nLe1) / float64(H),
			producerMaxRel: producerMaxRel, dScF16: dScF16, dScW4A8: dScW4A8}
		res.cosF16 = cosine(guF16, guRef)
		res.cosW4A8 = cosine(guW4A8, guRef)
		res.cosF16Half = cosine(guF16, guHalfRef)
		res.cosW4A8Q8 = cosine(guW4A8, guQ8Ref)
		var i1, i2, i3, i4 int
		res.relF16, i1 = maxRel(guF16, guRef)
		res.relW4A8, i2 = maxRel(guW4A8, guRef)
		res.relF16Half, i3 = maxRel(guF16, guHalfRef)
		res.relW4A8Q8, i4 = maxRel(guW4A8, guQ8Ref)
		// per-row: which arm is closer to guRef
		var nF16Closer int
		for n := 0; n < N; n++ {
			if math.Abs(guF16[n]-guRef[n]) < math.Abs(guW4A8[n]-guRef[n]) {
				nF16Closer++
			}
		}
		res.fracRowsF16Closer = float64(nF16Closer) / float64(N)
		logf("layer %d FIDELITY vs f64 ref (W·aRef):   gu_f16  cos=%.9f maxRel=%.4g (row %d) | gu_w4a8 cos=%.9f maxRel=%.4g (row %d) | f16 closer on %.1f%% of rows",
			l, res.cosF16, res.relF16, i1, res.cosW4A8, res.relW4A8, i2, 100*res.fracRowsF16Closer)
		logf("layer %d KERNEL  vs own-input f64 dot:   gu_f16 vs W·aHalf cos=%.9f maxRel=%.4g (row %d) | gu_w4a8 vs W·(mq*mSc) cos=%.9f maxRel=%.4g (row %d)",
			l, res.cosF16Half, res.relF16Half, i3, res.cosW4A8Q8, res.relW4A8Q8, i4)
		// maxAbs for the kernel checks too (the record's own unit)
		var mabsF16, mabsQ8, gAm float64
		for n := 0; n < N; n++ {
			mabsF16 = math.Max(mabsF16, math.Abs(guF16[n]-guHalfRef[n]))
			mabsQ8 = math.Max(mabsQ8, math.Abs(guW4A8[n]-guQ8Ref[n]))
			gAm = math.Max(gAm, math.Abs(guRef[n]))
		}
		logf("layer %d KERNEL  maxAbs: f16 %.4g, w4a8 %.4g (gu amax %.4g)", l, mabsF16, mabsQ8, gAm)

		for _, arm := range []struct {
			name string
			got  []float64
		}{{"f16", guF16}, {"w4a8", guW4A8}} {
			for _, w := range worstRows(arm.got, guRef, 5) {
				nib := nibAt(w.row, actArg)
				logf("layer %d worst[%s] row %6d (%s): got=%.4f ref=%.4f |diff|=%.4f rel=%.4f nibble@ch%d=%d (zero=%v) | other arm=%.4f",
					l, arm.name, w.row, map[bool]string{true: "gate", false: "up"}[w.row < I], w.got, w.ref, w.diff, w.rel, actArg, nib, nib == 8,
					map[string]float64{"f16": guW4A8[w.row], "w4a8": guF16[w.row]}[arm.name])
			}
		}
		if 2908 < N {
			res.row2908Ref, res.row2908F16, res.row2908W4A8 = guRef[2908], guF16[2908], guW4A8[2908]
			res.row2908Half, res.row2908Q8 = guHalfRef[2908], guQ8Ref[2908]
			logf("layer %d row 2908: ref(f64,aRef)=%.4f f16=%.4f w4a8=%.4f | W·aHalf=%.4f W·aQ8=%.4f | nibble@ch%d=%d nibble@ch408=%d",
				l, guRef[2908], guF16[2908], guW4A8[2908], guHalfRef[2908], guQ8Ref[2908], actArg, nibAt(2908, actArg), nibAt(2908, 408))
		}

		// ---- downstream: h = silu(gate)*up from each arm's gu, and the int8 step pSw chose ----
		hF16 := make([]float64, I)
		hW4A8 := make([]float64, I)
		for j := 0; j < I; j++ {
			hF16[j] = silu(guF16[j]) * guF16[I+j]
			hW4A8[j] = silu(guW4A8[j]) * guW4A8[I+j]
		}
		res.hAmaxF16, res.hArgF16 = amaxOf(hF16)
		res.hAmaxW4A8, res.hArgW4A8 = amaxOf(hW4A8)
		hRef := make([]float64, I)
		for j := 0; j < I; j++ {
			hRef[j] = silu(guRef[j]) * guRef[I+j]
		}
		hRefAmax, hRefArg := amaxOf(hRef)
		logf("layer %d DOWNSTREAM h=silu(g)*u: amax f16=%.4g@%d w4a8=%.4g@%d ref=%.4g@%d | dSc(int8 step) f16=%.5g w4a8=%.5g (ratio %.3f) | 127*dSc: f16=%.4g w4a8=%.4g | cos(h_f16,h_ref)=%.6f cos(h_w4a8,h_ref)=%.6f",
			l, res.hAmaxF16, res.hArgF16, res.hAmaxW4A8, res.hArgW4A8, hRefAmax, hRefArg,
			dScF16, dScW4A8, dScF16/dScW4A8, 127*dScF16, 127*dScW4A8, cosine(hF16, hRef), cosine(hW4A8, hRef))
		results = append(results, res)
	}

	logf("=== PER-LAYER SUMMARY ===")
	logf("%5s %9s %8s %8s | %12s %10s | %12s %10s | %12s %10s | %8s | %9s %9s",
		"layer", "actAmax", "mq==0", "|mq|<=1", "f16 cos", "f16 maxRel", "w4a8 cos", "w4a8 maxRel", "f16-vs-half cos", "maxRel", "f16closer", "dSc f16", "dSc w4a8")
	for _, x := range results {
		logf("%5d %9.3f %7.1f%% %7.1f%% | %.10f %10.3g | %.10f %10.3g | %.10f %10.3g | %7.1f%% | %9.4g %9.4g",
			x.layer, x.actAmax, 100*x.zeroFrac, 100*x.le1Frac, x.cosF16, x.relF16, x.cosW4A8, x.relW4A8,
			x.cosF16Half, x.relF16Half, 100*x.fracRowsF16Closer, x.dScF16, x.dScW4A8)
	}

	// ---------------- repro decomposition (same model) ----------------
	logf("=== REPRO DECOMPOSITION: W4A8 through 0..25 -> x1; then 26..27 on each lane ===")
	reset()
	runLayers(0, 26)
	x1 := append([]float32(nil), r.x.Floats()[:H]...)
	runLayers(26, 27)
	base26 := append([]float32(nil), r.x.Floats()[:H]...)
	runLayers(27, r.nL)
	base27 := append([]float32(nil), r.x.Floats()[:H]...)

	copy(r.x.Floats(), x1)
	r.setPos(0)
	r.decodeLaneW4F16 = true
	if !r.canUseF16Lane(26) || !r.canUseF16Lane(27) {
		t.Fatalf("f16 lane not eligible at 26/27 with the flag on")
	}
	runLayers(26, 27)
	f26 := append([]float32(nil), r.x.Floats()[:H]...)
	runLayers(27, r.nL)
	f27 := append([]float32(nil), r.x.Floats()[:H]...)
	r.decodeLaneW4F16 = false

	x1Amax, x1Arg := amaxOf(toF64(x1))
	logf("x1 (entering layer 26): amax=%.4g at ch %d; x1[408]=%.4g", x1Amax, x1Arg, x1[408])

	decompose := func(name string, a, b []float32) {
		A, B := toF64(a), toF64(b)
		cos := cosine(A, B)
		diff := make([]float64, H)
		var maxAbs, sumSq float64
		for i := range A {
			diff[i] = A[i] - B[i]
			maxAbs = math.Max(maxAbs, math.Abs(diff[i]))
			sumSq += diff[i] * diff[i]
		}
		idx := make([]int, H)
		for i := range idx {
			idx[i] = i
		}
		sort.Slice(idx, func(p, q int) bool { return math.Abs(diff[idx[p]]) > math.Abs(diff[idx[q]]) })
		logf("%s: cosine=%.8f maxAbs=%.6g ||diff||=%.6g ||w4a8||=%.6g ||f16||=%.6g", name, cos, maxAbs, math.Sqrt(sumSq),
			math.Sqrt(func() float64 {
				s := 0.0
				for _, v := range A {
					s += v * v
				}
				return s
			}()),
			math.Sqrt(func() float64 {
				s := 0.0
				for _, v := range B {
					s += v * v
				}
				return s
			}()))
		for rank := 0; rank < 10; rank++ {
			i := idx[rank]
			logf("%s: top-%2d ch %4d: w4a8=%.4f f16=%.4f diff=%.4f share=%.4f", name, rank+1, i, A[i], B[i], diff[i], diff[i]*diff[i]/sumSq)
		}
		top := idx[0]
		logf("%s: share of ||diff||^2: ch408=%.5f top-1(ch %d)=%.5f top-3=%.5f", name,
			diff[408]*diff[408]/sumSq, top, diff[top]*diff[top]/sumSq,
			(diff[idx[0]]*diff[idx[0]]+diff[idx[1]]*diff[idx[1]]+diff[idx[2]]*diff[idx[2]])/sumSq)
		// exclude top-1 channel
		A2 := append([]float64(nil), A...)
		B2 := append([]float64(nil), B...)
		A2[top], B2[top] = 0, 0
		var maxAbs2 float64
		for i := range A2 {
			if i != top {
				maxAbs2 = math.Max(maxAbs2, math.Abs(A2[i]-B2[i]))
			}
		}
		logf("%s: with top-1 channel (%d) removed: cosine=%.8f maxAbs=%.6g", name, top, cosine(A2, B2), maxAbs2)
		// exclude channel 408 and any |x|>200 in either arm
		A3 := append([]float64(nil), A...)
		B3 := append([]float64(nil), B...)
		var nEx int
		for i := range A3 {
			if i == 408 || math.Abs(A3[i]) > 200 || math.Abs(B3[i]) > 200 {
				A3[i], B3[i] = 0, 0
				nEx++
			}
		}
		logf("%s: with ch408 and any |x|>200 removed (%d channels): cosine=%.8f", name, nEx, cosine(A3, B3))
		logf("%s: ch408 values: w4a8=%.4f f16=%.4f", name, A[408], B[408])
	}
	decompose("after layer 26", base26, f26)
	decompose("after layer 27", base27, f27)
	cos27 := cosine(toF64(base27), toF64(f27))
	if cos27 > 0.9 {
		logf("NOTE: the layers-26-27 repro did NOT reproduce a negative cosine (got %.6f) — the premise of the parked record has changed", cos27)
	} else {
		logf("repro reproduced: layers-26-27 cosine %.6f (record: -0.585 from clean x1)", cos27)
	}
}
