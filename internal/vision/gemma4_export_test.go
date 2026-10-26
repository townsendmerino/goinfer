package vision

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestGemma4Weights_reproduceForward: a tower written only from the export (Weights, Gemma4RopeTables, FinishHidden)
// reproduces Forward on the tiny fixture, so the export carries everything a device-resident tower needs: clip bounds
// (the fixture's are finite), both position tables, the axial RoPE, the v norm without a weight and the tail. A
// quantized load is refused by Weights.
func TestGemma4Weights_reproduceForward(t *testing.T) {
	dir := gemma4Testdata(t, "gemma4-vision-tiny")
	enc, err := LoadGemma4Encoder(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	w, err := enc.Weights()
	if err != nil {
		t.Fatal(err)
	}
	clipped := 0
	for _, l := range w.Layers {
		for _, p := range []Gemma4Proj{l.Q, l.K, l.V, l.O, l.Gate, l.Up, l.Down} {
			if p.Clipped() {
				clipped++
			}
		}
	}
	if clipped == 0 {
		t.Fatal("the fixture has no finite clip bound, so this test would not see a dropped clamp")
	}
	// The fixture's position tables are all ones, which hides a swapped or misindexed table; the export's slices alias
	// the encoder's, so random tables written through them reach Forward too.
	r := rand.New(rand.NewPCG(1, 2))
	for _, tb := range [][]float32{w.PosEmbX, w.PosEmbY} {
		for i := range tb {
			tb[i] = r.Float32()*2 - 1
		}
	}
	c := w.Cfg
	k := c.PoolingKernelSize
	gw, gh := 2*k, k // a 2x1 pooled grid
	var pos [][2]int
	for y := range gh {
		for x := range gw {
			pos = append(pos, [2]int{x, y})
		}
	}
	pd := 3 * c.PatchSize * c.PatchSize
	patches := make([]float32, len(pos)*pd)
	for i := range patches {
		patches[i] = r.Float32()
	}
	want, err := enc.Forward(patches, pos)
	if err != nil {
		t.Fatal(err)
	}
	h, err := gemma4ExportReference(w, patches, pos)
	if err != nil {
		t.Fatal(err)
	}
	got, err := enc.FinishHidden(h, pos)
	if err != nil {
		t.Fatal(err)
	}
	cs := cosineSim(got, want)
	t.Logf("tower from the export against Forward: cosine %.9f over %d soft tokens, %d clipped projections", cs, len(got)/enc.TextHiddenSize, clipped)
	if cs < 0.999999 {
		t.Fatalf("cosine %.9f", cs)
	}
	// standardize (26B-A4B's tail; the fixture has none): with random std_bias/std_scale switched on, the export path
	// (FinishHidden after the export's tower) still equals Forward, and it differs from the same tower without it, so
	// a device tower that relies on FinishHidden gets standardize, and a dropped standardize would show.
	enc.Cfg.Standardize = true
	enc.stdBias, enc.stdScale = make([]float32, c.HiddenSize), make([]float32, c.HiddenSize)
	for i := range enc.stdBias {
		enc.stdBias[i], enc.stdScale[i] = r.Float32()-0.5, 0.5+r.Float32()
	}
	wantStd, err := enc.Forward(patches, pos)
	if err != nil {
		t.Fatal(err)
	}
	gotStd, err := enc.FinishHidden(h, pos)
	if err != nil {
		t.Fatal(err)
	}
	if cs := cosineSim(gotStd, wantStd); cs < 0.999999 {
		t.Fatalf("standardize on: export path against Forward, cosine %.9f", cs)
	}
	if cs := cosineSim(got, wantStd); cs > 0.9999 {
		t.Fatalf("standardize on and off give cosine %.9f: the comparison cannot see a dropped standardize", cs)
	}
	enc.Cfg.Standardize, enc.stdBias, enc.stdScale = false, nil, nil

	q, err := LoadGemma4Encoder(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := q.Weights(); err == nil {
		t.Fatal("Weights accepted a quantized tower")
	}
}

// gemma4ExportReference is a plain tower from the export alone, float64 accumulation, up to the last layer's output.
func gemma4ExportReference(w Gemma4Weights, patches []float32, pos [][2]int) ([]float32, error) {
	c := w.Cfg
	H, nH, hd := c.HiddenSize, c.NumAttentionHeads, c.HeadDim
	np, pd := len(pos), 3*c.PatchSize*c.PatchSize
	lin := func(x []float32, rows int, p Gemma4Proj) []float32 {
		out := make([]float32, rows*p.Out)
		for r := range rows {
			xr := x[r*p.In : (r+1)*p.In]
			for o := range p.Out {
				var s float64
				for i, v := range xr {
					v = min(max(v, p.InMin), p.InMax)
					s += float64(v) * float64(p.W[o*p.In+i])
				}
				out[r*p.Out+o] = min(max(float32(s), p.OutMin), p.OutMax)
			}
		}
		return out
	}
	norm := func(x []float32, rows, dim int, wt []float32) []float32 {
		out := make([]float32, len(x))
		for r := range rows {
			var ss float64
			for _, v := range x[r*dim : (r+1)*dim] {
				ss += float64(v) * float64(v)
			}
			inv := 1 / math.Sqrt(ss/float64(dim)+c.RMSNormEps)
			for d := range dim {
				v := float32(float64(x[r*dim+d]) * inv)
				if wt != nil {
					v *= wt[d]
				}
				out[r*dim+d] = v
			}
		}
		return out
	}
	x := make([]float32, np*H)
	for i, p := range pos {
		for o := range H {
			var s float64
			for j := range pd {
				s += float64(2*(patches[i*pd+j]-0.5)) * float64(w.PatchEmbed[o*pd+j])
			}
			x[i*H+o] = float32(s) + w.PosEmbX[p[0]*H+o] + w.PosEmbY[p[1]*H+o]
		}
	}
	cos, sin := Gemma4RopeTables(pos, hd, w.RopeTheta)
	rope := func(v []float32) {
		half := hd / 2
		for i := range np {
			for h := range nH {
				for part := range 2 {
					seg := v[i*H+h*hd+part*half : i*H+h*hd+(part+1)*half]
					cs, sn := cos[i*hd+part*half:i*hd+(part+1)*half], sin[i*hd+part*half:i*hd+(part+1)*half]
					q := half / 2
					for d := range q {
						a, b := seg[d], seg[d+q]
						seg[d] = a*cs[d] - b*sn[d]
						seg[d+q] = b*cs[d+q] + a*sn[d+q]
					}
				}
			}
		}
	}
	for _, l := range w.Layers {
		xn := norm(x, np, H, l.InputNorm)
		q := norm(lin(xn, np, l.Q), np*nH, hd, l.QNorm)
		k := norm(lin(xn, np, l.K), np*nH, hd, l.KNorm)
		v := norm(lin(xn, np, l.V), np*nH, hd, nil)
		rope(q)
		rope(k)
		ctx := make([]float32, np*H)
		sc := make([]float64, np)
		for h := range nH {
			for i := range np {
				m := math.Inf(-1)
				for j := range np {
					var s float64
					for d := range hd {
						s += float64(q[i*H+h*hd+d]) * float64(k[j*H+h*hd+d])
					}
					sc[j] = s // scale 1.0
					m = math.Max(m, s)
				}
				var z float64
				for j := range np {
					sc[j] = math.Exp(sc[j] - m)
					z += sc[j]
				}
				for d := range hd {
					var s float64
					for j := range np {
						s += sc[j] * float64(v[j*H+h*hd+d])
					}
					ctx[i*H+h*hd+d] = float32(s / z)
				}
			}
		}
		att := norm(lin(ctx, np, l.O), np, H, l.PostAttnNorm)
		for i := range x {
			x[i] += att[i]
		}
		xn = norm(x, np, H, l.PreFFNNorm)
		g, u := lin(xn, np, l.Gate), lin(xn, np, l.Up)
		for i, gv := range g {
			gf := float64(gv)
			g[i] = float32(0.5*gf*(1+math.Tanh(0.7978845608028654*(gf+0.044715*gf*gf*gf)))) * u[i]
		}
		f := norm(lin(g, np, l.Down), np, H, l.PostFFNNorm)
		for i := range x {
			x[i] += f[i]
		}
	}
	return x, nil
}
