//go:build gpu

package gpu

import (
	"math"
	"testing"
)

// W4A8 decode-runner gate. The full-token GPU W4A8 forward must match a CPU
// oracle that runs the SAME math — rmsnorm → int8-quantize activation → grouped
// int8×int4 dot (f16 group scales) → rope/attn/swiglu/residual. Both sides share
// the int4-quantized weights AND the int8-quantized activations, so this is a
// kernel/wiring gate (cosine ~1.0), not an int4-accuracy test. Mirrors
// TestDecodeToken_parity (W8A8) with the projections swapped to int4.

const w4Group = w4a8GroupSize

// refQuantInt8 mirrors the GPU activation quantize: per-row max-abs → scale
// maxabs/127, q = round(x/scale) clamped ±127.
func refQuantInt8(x []float32) ([]int8, float32) {
	var mx float32
	for _, v := range x {
		if a := float32(math.Abs(float64(v))); a > mx {
			mx = a
		}
	}
	s := mx / 127
	if s == 0 {
		s = 1
	}
	inv := 1 / s
	q := make([]int8, len(x))
	for i, v := range x {
		r := int(math.RoundToEven(float64(v * inv)))
		if r > 127 {
			r = 127
		} else if r < -127 {
			r = -127
		}
		q[i] = int8(r)
	}
	return q, s
}

// quantInt4Group quantizes f32 weights [N,K] to symmetric group-wise int4
// (nibbles 0..15 = value−8) + per-group f32 scales, group=32. Self-contained for
// the gate — both the GPU upload and the oracle use these exact nibbles/scales.
func quantInt4Group(w []float32, N, K, group int) ([]uint8, []float32) {
	nGroups := (K + group - 1) / group
	nib := make([]uint8, N*K)
	scales := make([]float32, N*nGroups)
	for n := range N {
		for g := range nGroups {
			ks, ke := g*group, min((g+1)*group, K)
			var mx float32
			for k := ks; k < ke; k++ {
				if a := float32(math.Abs(float64(w[n*K+k]))); a > mx {
					mx = a
				}
			}
			s := mx / 7 // value range −7..7 (nibble 1..15); −8 unused for symmetry
			if s == 0 {
				s = 1
			}
			scales[n*nGroups+g] = s
			for k := ks; k < ke; k++ {
				q := int(math.RoundToEven(float64(w[n*K+k]/s))) + 8
				if q < 0 {
					q = 0
				} else if q > 15 {
					q = 15
				}
				nib[n*K+k] = uint8(q)
			}
		}
	}
	return nib, scales
}

// refMatmulW4A8 is the CPU twin of the GPU W4A8 GEMV: dst[n] = aScale · Σ_g
// f16(scale[n,g]) · Σ_{k∈g} (nib−8)·actI8[k].
func refMatmulW4A8(actI8 []int8, aScale float32, nib []uint8, scales []float32, N, K, group int) []float32 {
	nGroups := (K + group - 1) / group
	dst := make([]float32, N)
	for n := range N {
		var total float64
		for g := range nGroups {
			ks, ke := g*group, min((g+1)*group, K)
			var idot int
			for k := ks; k < ke; k++ {
				idot += (int(nib[n*K+k]) - 8) * int(actI8[k])
			}
			total += float64(idot) * float64(f16to32(f32to16(scales[n*nGroups+g])))
		}
		dst[n] = float32(total * float64(aScale))
	}
	return dst
}

func TestDecodeRunnerW4A8_parity(t *testing.T) {
	ctx := newOrSkipHW(t)
	defer ctx.Close()

	const hidden, nH, nKV, hd, inter, pos, vocab, L = 1536, 12, 2, 128, 8960, 20, 4096, 2
	qDim, kvDim := nH*hd, nKV*hd
	half := hd / 2
	eps := float32(1e-6)
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	x0 := randMat(hidden, 100)
	invFreq := make([]float32, half)
	for d := range invFreq {
		invFreq[d] = float32(1.0 / math.Pow(1e6, float64(2*d)/float64(hd)))
	}

	// f32 weights per layer, then int4-quantized projections.
	type lw struct {
		an, mn                     []float32
		qN, kN, vN, oN, gN, uN, dN []uint8   // int4 nibbles
		qS, kS, vS, oS, gS, uS, dS []float32 // group scales
		priorK, priorV             []float32
	}
	layers := make([]lw, L)
	seed := uint64(1)
	for l := range layers {
		mk := func(N, K int) ([]uint8, []float32) { seed++; return quantInt4Group(randMat(N*K, seed), N, K, w4Group) }
		layers[l].an = randMat(hidden, uint64(200+l))
		layers[l].mn = randMat(hidden, uint64(300+l))
		layers[l].qN, layers[l].qS = mk(qDim, hidden)
		layers[l].kN, layers[l].kS = mk(kvDim, hidden)
		layers[l].vN, layers[l].vS = mk(kvDim, hidden)
		layers[l].oN, layers[l].oS = mk(hidden, qDim)
		layers[l].gN, layers[l].gS = mk(inter, hidden)
		layers[l].uN, layers[l].uS = mk(inter, hidden)
		layers[l].dN, layers[l].dS = mk(hidden, inter)
		layers[l].priorK = randMat(pos*kvDim, uint64(400+l))
		layers[l].priorV = randMat(pos*kvDim, uint64(500+l))
	}
	fnorm := randMat(hidden, 600)
	lmN, lmS := quantInt4Group(randMat(vocab*hidden, 999), vocab, hidden, w4Group)

	// --- CPU oracle (same math as the GPU W4A8 forward) ---
	silu := func(g float32) float32 { return g / (1 + float32(math.Exp(float64(-g)))) }
	x := append([]float32(nil), x0...)
	for l := range layers {
		L := &layers[l]
		xn := refRMSNorm(x, L.an, hidden, eps, false)
		aq, as := refQuantInt8(xn)
		q := refMatmulW4A8(aq, as, L.qN, L.qS, qDim, hidden, w4Group)
		k := refMatmulW4A8(aq, as, L.kN, L.kS, kvDim, hidden, w4Group)
		v := refMatmulW4A8(aq, as, L.vN, L.vS, kvDim, hidden, w4Group)
		refRoPE(q, nH, hd, pos, invFreq)
		refRoPE(k, nKV, hd, pos, invFreq)
		kFull := append(append([]float32(nil), L.priorK...), k...)
		vFull := append(append([]float32(nil), L.priorV...), v...)
		cv := refAttn(q, kFull, vFull, nH, nKV, hd, pos+1, scale)
		cq, cs := refQuantInt8(cv)
		ao := refMatmulW4A8(cq, cs, L.oN, L.oS, hidden, qDim, w4Group)
		for i := range x {
			x[i] += ao[i]
		}
		xn2 := refRMSNorm(x, L.mn, hidden, eps, false)
		mq, ms := refQuantInt8(xn2)
		gate := refMatmulW4A8(mq, ms, L.gN, L.gS, inter, hidden, w4Group)
		up := refMatmulW4A8(mq, ms, L.uN, L.uS, inter, hidden, w4Group)
		mid := make([]float32, inter)
		for i := range mid {
			mid[i] = silu(gate[i]) * up[i]
		}
		dq, ds := refQuantInt8(mid)
		down := refMatmulW4A8(dq, ds, L.dN, L.dS, hidden, inter, w4Group)
		for i := range x {
			x[i] += down[i]
		}
	}
	xnf := refRMSNorm(x, fnorm, hidden, eps, false)
	fq, fs := refQuantInt8(xnf)
	refLogits := refMatmulW4A8(fq, fs, lmN, lmS, vocab, hidden, w4Group)

	// --- GPU W4A8 DecodeRunner (build a runModel with int4 projections) ---
	up32 := func(v []float32) *DeviceBuffer { d, _ := ctx.UploadF32(v); return d }
	w4 := func(nib []uint8, sc []float32, N, K int) *ResidentW4A8 {
		rm, e := ctx.UploadW4A8(nib, sc, N, K)
		if e != nil {
			t.Fatal(e)
		}
		return rm
	}
	invD := up32(invFreq)
	rm := runModel{finalNorm: up32(fnorm).buf, lmHead: w4(lmN, lmS, vocab, hidden)}
	for l := range layers {
		L := &layers[l]
		kc, _ := ctx.NewKVCache(L.priorK, (pos+1)*kvDim)
		vc, _ := ctx.NewKVCache(L.priorV, (pos+1)*kvDim)
		rm.layers = append(rm.layers, runLayer{
			attnNorm: up32(L.an).buf, invFreq: invD.buf, kCache: kc.buf, vCache: vc.buf, mlpNorm: up32(L.mn).buf,
			q: w4(L.qN, L.qS, qDim, hidden), k: w4(L.kN, L.kS, kvDim, hidden), v: w4(L.vN, L.vS, kvDim, hidden),
			o:    w4(L.oN, L.oS, hidden, qDim),
			gate: w4(L.gN, L.gS, inter, hidden), up: w4(L.uN, L.uS, inter, hidden), down: w4(L.dN, L.dS, hidden, inter),
		})
	}
	runner, err := ctx.newDecodeRunner(rm, hidden, nH, nKV, hd, inter, 0, eps, scale, false)
	if err != nil {
		t.Fatalf("newDecodeRunner(W4A8): %v", err)
	}
	defer runner.Release()
	got, err := runner.Run(x0, pos)
	if err != nil {
		t.Fatalf("W4A8 Run: %v", err)
	}
	cos, maxAbs := cosine(got, refLogits)
	t.Logf("W4A8 DecodeRunner parity (%d layers, int4 projections): cosine=%.6f maxAbs=%.3e", L, cos, maxAbs)
	if cos < 0.9999 {
		t.Errorf("W4A8 DecodeRunner diverges: cosine=%.6f maxAbs=%.3e", cos, maxAbs)
	}
}
