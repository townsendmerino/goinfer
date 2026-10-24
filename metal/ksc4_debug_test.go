//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

func TestKSC4Debug_residentPipes(t *testing.T) {
	path := os.ExpandEnv(fmt.Sprintf("$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.int4.metal.v%d.giw", decoder.GIWFormatVersion))
	m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 256})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	a := m.ResidentForwardForTest().(*metalResident)
	r := a.r
	d := r.d
	fmt.Fprintf(os.Stderr, "precise %v w8 %v w8Attn %v\n", r.preciseMath, r.w8, r.w8Attn)
	H, I, R := r.H, r.I, r.gemvRows.gu
	rng := rand.New(rand.NewSource(41))
	for _, sd := range []float64{0.001, 0.02} {
		halves := func(n int, sd, mean float64) Buffer {
			b := d.NewBufferBytes(2 * n)
			v := b.U16s()[:n]
			for i := range v {
				v[i] = f32ToF16(float32(rng.NormFloat64()*sd + mean))
			}
			return b
		}
		words := func(n int) Buffer {
			b := d.NewBufferBytes(4 * n)
			for i, u := 0, b.U32s()[:n]; i < n; i++ {
				u[i] = rng.Uint32()
			}
			return b
		}
		guW, guS := words(2*I*H/8), halves(2*I*H/32, sd, 2*sd)
		dW, dS := words(H*I/8), halves(H*I/32, 0.001, 0.002)
		aq := d.NewBufferBytes(H)
		for i, v := 0, aq.Int8s()[:H]; i < H; i++ {
			v[i] = int8(rng.Intn(255) - 127)
		}
		init := make([]float32, H)
		asc, uH, uI := NewBufferFloats(d, []float32{0.0123}), NewBufferU32(d, uint32(H)), NewBufferU32(d, uint32(I))
		q := r.q
		want := NewBufferFloats(d, slices.Clone(init))
		gu, dq, ds := d.NewBufferLen(2*I), d.NewBufferBytes(I), d.NewBufferLen(1)
		e := q.Begin()
		e.DispatchTG(r.pSARows, 2*I*32/R, 256, H*2, guW, guS, aq, asc, gu, uH)
		e.Dispatch(r.pSw, 256, 256, gu, gu.At(I*4), dq, ds, uI, r.uAct)
		e.DispatchTG(r.pGemvResidStaged, H*32/R, 256, I, dW, dS, dq, ds, want, uI)
		e.End()
		got := NewBufferFloats(d, slices.Clone(init))
		h, part := d.NewBufferLen(2*I), d.NewBufferBytes(I)
		e = q.Begin()
		e.DispatchTG(r.pSAGluRows, 2*I*32/R, 256, H*2, guW, guS, aq, asc, h, part, uH, uI, r.uAct)
		e.DispatchTG(r.pGemvResidHQ, H*32/R, 256, I, dW, dS, h, part, got, uI)
		e.End()
		w, g := want.Floats()[:H], got.Floats()[:H]
		diff, mx := 0, 0.0
		for i := range w {
			if math.Float32bits(w[i]) != math.Float32bits(g[i]) {
				diff++
				mx = max(mx, math.Abs(float64(w[i]-g[i])))
			}
		}
		fmt.Fprintf(os.Stderr, "sd %v: %d of %d differ, max abs %g, ds %g\n", sd, diff, H, mx, ds.Floats()[0])
		// h vs swiglu_f32
		hf := h.Floats()[:I]
		gf := gu.Floats()[:2*I]
		hd := 0
		for i := 0; i < I; i++ {
			x := gf[i]
			s := float32(float64(x)/(1+math.Exp(-float64(x)))) * gf[I+i]
			_ = s
			if math.IsNaN(float64(hf[i])) {
				hd++
			}
		}
		qd := 0
		dqv := dq.Int8s()[:I]
		sc := ds.Floats()[0]
		for i := 0; i < I; i++ {
			qq := int(math.Round(float64(hf[i] * (1 / sc))))
			qq = max(-127, min(127, qq))
			if int8(qq) != dqv[i] {
				qd++
			}
		}
		fmt.Fprintf(os.Stderr, "   h NaN %d; host-quantized h vs swiglu_quant dq: %d of %d differ\n", hd, qd, I)
	}
}

func TestKSC4Debug_firstLayer(t *testing.T) {
	path := os.ExpandEnv(fmt.Sprintf("$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.int4.metal.v%d.giw", decoder.GIWFormatVersion))
	m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 256})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	a := m.ResidentForwardForTest().(*metalResident)
	r := a.r
	emb := m.EmbedResidentForTest(13)
	glu := r.pSAGluRows
	for _, n := range []int{1, 2, 3, 6, 12, 24} {
		a.Reset()
		on := r.forwardTrunkForTest(emb, 0, n)
		r.pSAGluRows = Pipeline{}
		a.Reset()
		off := r.forwardTrunkForTest(emb, 0, n)
		r.pSAGluRows = glu
		diff, mx := 0, 0.0
		for i := range on[:r.H] {
			if math.Float32bits(on[i]) != math.Float32bits(off[i]) {
				diff++
				mx = max(mx, math.Abs(float64(on[i]-off[i])))
			}
		}
		fmt.Fprintf(os.Stderr, "layers %d: %d of %d hidden differ, max abs %g\n", n, diff, r.H, mx)
	}
}

func TestKSC4Debug_forwardSteps(t *testing.T) {
	path := os.ExpandEnv(fmt.Sprintf("$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.int4.metal.v%d.giw", decoder.GIWFormatVersion))
	m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 256})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	a := m.ResidentForwardForTest().(*metalResident)
	r := a.r
	glu := r.pSAGluRows
	for _, flush := range []bool{false} {
		on := ksc3Decode(t, m, a, 32)
		r.pSAGluRows = Pipeline{}
		if flush {
			r.stopExec()
		}
		off := ksc3Decode(t, m, a, 32)
		r.pSAGluRows = glu
		if flush {
			r.stopExec()
		}
		for s := range on {
			diff := 0
			for i := range on[s] {
				if math.Float32bits(on[s][i]) != math.Float32bits(off[s][i]) {
					diff++
				}
			}
			fmt.Fprintf(os.Stderr, "flush %v step %d: %d differ\n", flush, s, diff)
		}
	}
}

func TestKSC4Debug_step9Layers(t *testing.T) {
	path := os.ExpandEnv(fmt.Sprintf("$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.int4.metal.v%d.giw", decoder.GIWFormatVersion))
	m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 256})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	a := m.ResidentForwardForTest().(*metalResident)
	r := a.r
	glu := r.pSAGluRows
	steps := ksc3Decode(t, m, a, 10) // tokens: 13, then argmax of each step
	toks := []int{13}
	for s := 0; s < 9; s++ {
		lg := steps[s]
		tk := 0
		for i, v := range lg {
			if v > lg[tk] {
				tk = i
			}
		}
		toks = append(toks, tk)
	}
	prep := func() {
		a.Reset()
		for pos := 0; pos < 9; pos++ {
			if _, err := a.Forward(m.EmbedResidentForTest(toks[pos]), pos); err != nil {
				t.Fatal(err)
			}
		}
	}
	emb := m.EmbedResidentForTest(toks[9])
	for n := 1; n <= r.nL; n++ {
		prep()
		on := r.forwardTrunkForTest(emb, 9, n)
		prep()
		r.pSAGluRows = Pipeline{}
		off := r.forwardTrunkForTest(emb, 9, n)
		r.pSAGluRows = glu
		diff, mx := 0, 0.0
		for i := range on[:r.H] {
			if math.Float32bits(on[i]) != math.Float32bits(off[i]) {
				diff++
				mx = max(mx, math.Abs(float64(on[i]-off[i])))
			}
		}
		if diff > 0 || n == r.nL {
			fmt.Fprintf(os.Stderr, "pos 9, layers %d: %d of %d hidden differ, max abs %g\n", n, diff, r.H, mx)
		}
		if diff > 0 {
			// re-run both arms to n layers and capture the FFN internals of layer n-1
			prep()
			r.pSAGluRows = Pipeline{}
			r.forwardTrunkForTest(emb, 9, n)
			r.pSAGluRows = glu
			I := r.layers[n-1].ffnI
			offG := slices.Clone(r.gu.Floats()[:2*I])
			offDq := slices.Clone(r.dq.Int8s()[:I])
			offSc := r.dSc.Floats()[0]
			prep()
			r.forwardTrunkForTest(emb, 9, n)
			h := slices.Clone(r.gu.Floats()[:I])
			np := 2 * I / (8 * r.gemvRows.gu)
			part := slices.Clone(r.dq.Floats()[:np])
			pm := float32(0)
			for _, v := range part {
				pm = max(pm, v)
			}
			hm := float32(0)
			nan := 0
			for _, v := range h {
				if v != v {
					nan++
				}
				hm = max(hm, float32(math.Abs(float64(v))))
			}
			sc := pm / 127
			fmt.Fprintf(os.Stderr, "layer %d: off sc %g (max %g); on partial max %g (sc %g), host max|h| %g, NaN %d\n", n-1, offSc, offSc*127, pm, sc, hm, nan)
			inv := 1 / offSc
			qd := 0
			for i := 0; i < I; i++ {
				q := int(math.Round(float64(h[i] * inv)))
				q = max(-127, min(127, q))
				if int8(q) != offDq[i] {
					if qd < 8 {
						g, u := offG[i], offG[I+i]
						fmt.Fprintf(os.Stderr, "  i %d: h %g (bits %08x) g %g u %g; h*inv %g; on q %d off dq %d\n", i, h[i], math.Float32bits(h[i]), g, u, h[i]*inv, q, offDq[i])
					}
					qd++
				}
			}
			fmt.Fprintf(os.Stderr, "  %d of %d quantized values differ (on's h quantized with off's scale)\n", qd, I)
			break
		}
	}
}

func TestKSC4Debug_roundTies(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skip(err)
	}
	src := `#include <metal_stdlib>
using namespace metal;
kernel void rs(device const float* x[[buffer(0)]], device const float* inv[[buffer(1)]], device char* o[[buffer(2)]], uint i[[thread_position_in_grid]]) {
    o[i] = char(clamp(int(round(x[i]*inv[0])),-127,127));
}
kernel void rv(device const float* x[[buffer(0)]], device const float* inv[[buffer(1)]], device char* o[[buffer(2)]], uint i[[thread_position_in_grid]]) {
    float4 sv = ((device const float4*)x)[i];
    ((device char4*)o)[i] = char4(char(clamp(int(round(sv.x*inv[0])),-127,127)), char(clamp(int(round(sv.y*inv[0])),-127,127)),
        char(clamp(int(round(sv.z*inv[0])),-127,127)), char(clamp(int(round(sv.w*inv[0])),-127,127)));
}`
	xs := []float32{-10.5, -11.5, 10.5, 11.5, -0.5, 0.5, 2.5, -2.5}
	for _, precise := range []bool{false, true} {
		compile := d.CompileLibrary
		if precise {
			compile = d.CompileLibraryPrecise
		}
		lib, err := compile(src, MSL3_1)
		if err != nil {
			t.Fatal(err)
		}
		for _, k := range []string{"rs", "rv"} {
			p, _ := d.NewComputePipeline(lib, k)
			x, inv, o := NewBufferFloats(d, xs), NewBufferFloats(d, []float32{1}), d.NewBufferBytes(len(xs))
			q := d.NewCommandQueue()
			e := q.Begin()
			n := len(xs)
			if k == "rv" {
				n /= 4
			}
			e.Dispatch(p, n, 1, x, inv, o)
			e.End()
			fmt.Fprintf(os.Stderr, "precise %v %s: %v -> %v\n", precise, k, xs, o.Int8s()[:len(xs)])
		}
	}
}
