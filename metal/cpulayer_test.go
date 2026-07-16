//go:build darwin

package metal

import "math"

// cpuLayer is the reference dense decode layer — it mirrors the GPU path EXACTLY,
// including the lossy quantization (rmsnorm→int8, per-row-int8 weights, int8 GEMV), so
// GPU vs CPU should match to f32 rounding. Any structural (buffer-wiring) bug in the GPU
// assembly shows up as a parity break.
func cpuLayer(x0, attnNorm, mlpNorm, Wq, Wk, Wv, Wo, Wg, Wu, Wd, invf, kcHist, vcHist []float32,
	H, nH, nKV, hd, I, pos int, eps, scale float32) []float32 {
	x := append([]float32(nil), x0...)
	kvDim := nKV * hd
	half := hd / 2

	rmsQuant := func(v, w []float32) ([]int8, float32) {
		var ss float64
		for _, e := range v {
			ss += float64(e) * float64(e)
		}
		rms := float32(1 / math.Sqrt(ss/float64(len(v))+float64(eps)))
		y := make([]float32, len(v))
		for i := range v {
			y[i] = v[i] * rms * w[i]
		}
		return q8row(y)
	}
	gemvQ := func(aq []int8, aSc float32, w []float32, out, in int) []float32 {
		o := make([]float32, out)
		for n := 0; n < out; n++ {
			wq, wSc := q8row(w[n*in : (n+1)*in])
			var acc int32
			for k := 0; k < in; k++ {
				acc += int32(aq[k]) * int32(wq[k])
			}
			o[n] = float32(acc) * aSc * wSc
		}
		return o
	}
	rope := func(vec []float32, nHeads int) {
		for hh := 0; hh < nHeads; hh++ {
			b := hh * hd
			for dd := 0; dd < half; dd++ {
				th := float64(pos) * float64(invf[dd])
				c, s := float32(math.Cos(th)), float32(math.Sin(th))
				x0, x1 := vec[b+dd], vec[b+half+dd]
				vec[b+dd] = x0*c - x1*s
				vec[b+half+dd] = x0*s + x1*c
			}
		}
	}

	// attention block
	aq, aSc := rmsQuant(x, attnNorm)
	q := gemvQ(aq, aSc, Wq, nH*hd, H)
	k := gemvQ(aq, aSc, Wk, kvDim, H)
	v := gemvQ(aq, aSc, Wv, kvDim, H)
	rope(q, nH)
	rope(k, nKV)
	kc := append(append([]float32(nil), kcHist...), k...)
	vc := append(append([]float32(nil), vcHist...), v...)
	nKeys := pos + 1
	ctx := make([]float32, nH*hd)
	for qh := 0; qh < nH; qh++ {
		kvh := qh / (nH / nKV)
		sc := make([]float64, nKeys)
		mx := math.Inf(-1)
		for s := 0; s < nKeys; s++ {
			var dot float64
			for dd := 0; dd < hd; dd++ {
				dot += float64(q[qh*hd+dd]) * float64(kc[s*kvDim+kvh*hd+dd])
			}
			sc[s] = dot * float64(scale)
			if sc[s] > mx {
				mx = sc[s]
			}
		}
		var sum float64
		for s := range sc {
			sc[s] = math.Exp(sc[s] - mx)
			sum += sc[s]
		}
		for dd := 0; dd < hd; dd++ {
			var acc float64
			for s := 0; s < nKeys; s++ {
				acc += sc[s] * float64(vc[s*kvDim+kvh*hd+dd])
			}
			ctx[qh*hd+dd] = float32(acc / sum)
		}
	}
	cq, cSc := q8row(ctx)
	oproj := gemvQ(cq, cSc, Wo, H, nH*hd)
	for h := 0; h < H; h++ {
		x[h] += oproj[h]
	}

	// mlp block
	mq, mSc := rmsQuant(x, mlpNorm)
	gate := gemvQ(mq, mSc, Wg, I, H)
	up := gemvQ(mq, mSc, Wu, I, H)
	sv := make([]float32, I)
	for i := range sv {
		sv[i] = (gate[i] / (1 + float32(math.Exp(-float64(gate[i]))))) * up[i]
	}
	dq, dSc := q8row(sv)
	down := gemvQ(dq, dSc, Wd, H, I)
	for h := 0; h < H; h++ {
		x[h] += down[h]
	}
	return x
}
