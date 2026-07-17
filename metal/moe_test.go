//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"testing"
)

// TestMoE_libraryCompiles compiles the full kernel library (dense + moeKernels concatenated)
// and instantiates every MoE pipeline by name — the runtime MSL-compile gate for the new
// kernels (a syntax error in moeKernels fails here, not deep in a model load).
func TestMoE_libraryCompiles(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Fatalf("device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile allKernels (incl. moeKernels): %v", err)
	}
	for _, name := range []string{"gemv_wf32_a8", "moe_route", "gemv_w4a8_moe", "gemv_w4a8_moe_wacc", "shared_gate_combine"} {
		if _, err := d.NewComputePipeline(lib, name); err != nil {
			t.Fatalf("pipeline %s: %v", name, err)
		}
	}
}

// refRoute is the CPU reference for moe_route — a verbatim port of decoder.routeExperts
// (softmax/sigmoid scoring, +bias selection, group-limiting, top-k, norm_topk_prob, scale).
func refRoute(logits, bias []float32, k int, sigmoid, norm bool, scale float64, nGroup, topkGroup int) ([]int, []float32) {
	nE := len(logits)
	score := make([]float32, nE)
	if sigmoid {
		for i, l := range logits {
			score[i] = float32(1.0 / (1.0 + math.Exp(-float64(l))))
		}
	} else {
		mx := logits[0]
		for _, v := range logits {
			if v > mx {
				mx = v
			}
		}
		var sum float64
		for i, v := range logits {
			e := math.Exp(float64(v - mx))
			score[i] = float32(e)
			sum += e
		}
		inv := float32(1.0 / sum)
		for i := range score {
			score[i] *= inv
		}
	}
	sel := make([]float32, nE)
	for i := range score {
		sel[i] = score[i] + bias[i]
	}
	negInf := float32(math.Inf(-1))
	if nGroup > 1 {
		gsz := nE / nGroup
		gscore := make([]float32, nGroup)
		for g := 0; g < nGroup; g++ {
			t1, t2 := negInf, negInf
			for i := g * gsz; i < (g+1)*gsz; i++ {
				v := sel[i]
				if v > t1 {
					t2, t1 = t1, v
				} else if v > t2 {
					t2 = v
				}
			}
			gscore[g] = t1 + t2
		}
		keepG := topKidx(gscore, topkGroup)
		keep := make([]bool, nGroup)
		for _, g := range keepG {
			keep[g] = true
		}
		for g := 0; g < nGroup; g++ {
			if !keep[g] {
				for i := g * gsz; i < (g+1)*gsz; i++ {
					sel[i] = negInf
				}
			}
		}
	}
	idx := topKidx(sel, k)
	wgt := make([]float32, len(idx))
	for j, e := range idx {
		wgt[j] = score[e]
	}
	if norm {
		var s float32
		for _, w := range wgt {
			s += w
		}
		if s > 0 {
			for j := range wgt {
				wgt[j] /= s
			}
		}
	}
	if scale != 0 && scale != 1 {
		for j := range wgt {
			wgt[j] *= float32(scale)
		}
	}
	return idx, wgt
}

func topKidx(xs []float32, k int) []int {
	used := make([]bool, len(xs))
	out := make([]int, 0, k)
	for ; k > 0; k-- {
		best, bi := float32(math.Inf(-1)), -1
		for i, v := range xs {
			if !used[i] && v > best {
				best, bi = v, i
			}
		}
		if bi < 0 {
			break
		}
		used[bi] = true
		out = append(out, bi)
	}
	return out
}

// TestMoE_route validates the on-GPU router against refRoute across the variant matrix:
// softmax+norm (Mixtral/Qwen-MoE), sigmoid+bias (GLM), and DeepSeek group-limiting. Logits
// are well-separated (×4) so the argmax selection has no near-ties — idx must match exactly;
// weights match within f32 tolerance (the kernel's softmax sum is f32 vs the reference f64).
func TestMoE_route(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Fatalf("device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "moe_route")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	b1 := func(b bool) uint32 {
		if b {
			return 1
		}
		return 0
	}
	cases := []struct {
		name              string
		nE, k             int
		sigmoid, norm     bool
		scale             float64
		nGroup, topkGroup int
	}{
		{"softmax_norm", 8, 2, false, true, 1, 1, 1},
		{"softmax_nonorm_scale", 16, 4, false, false, 2.5, 1, 1},
		{"sigmoid_bias", 32, 4, true, false, 1, 1, 1},
		{"group_limit", 16, 4, true, true, 1, 4, 2},
	}
	rng := rand.New(rand.NewSource(11))
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			logits := make([]float32, c.nE)
			bias := make([]float32, c.nE)
			for i := range logits {
				logits[i] = (rng.Float32()*2 - 1) * 4 // ×4 spread → unambiguous top-k
				if c.sigmoid {                        // GLM RouterBias only under sigmoid routing
					bias[i] = (rng.Float32()*2 - 1) * 0.1
				}
			}
			wantIdx, wantWgt := refRoute(logits, bias, c.k, c.sigmoid, c.norm, c.scale, c.nGroup, c.topkGroup)

			idxBuf := d.NewBufferUint32s(make([]uint32, c.k))
			wgtBuf := d.NewBufferLen(c.k)
			q := d.NewCommandQueue()
			q.Run1D(pipe, 1, 1,
				d.NewBufferFloats(logits), d.NewBufferFloats(bias), idxBuf, wgtBuf,
				d.NewBufferU32(uint32(c.nE)), d.NewBufferU32(uint32(c.k)),
				d.NewBufferU32(b1(c.sigmoid)), d.NewBufferU32(b1(c.norm)),
				d.NewBufferFloats([]float32{float32(c.scale)}),
				d.NewBufferU32(uint32(c.nGroup)), d.NewBufferU32(uint32(c.topkGroup)))
			gotIdx := idxBuf.U32s()
			gotWgt := wgtBuf.Floats()

			for j := 0; j < c.k; j++ {
				if int(gotIdx[j]) != wantIdx[j] {
					t.Fatalf("idx[%d]=%d want %d (all got=%v want=%v)", j, gotIdx[j], wantIdx[j], gotIdx, wantIdx)
				}
				if d := math.Abs(float64(gotWgt[j] - wantWgt[j])); d > 2e-4 {
					t.Fatalf("wgt[%d]=%.6f want %.6f (Δ=%.2e)", j, gotWgt[j], wantWgt[j], d)
				}
			}
		})
	}
}

// buildStackedExperts packs E experts × N rows × K cols of random f32 weights into ONE W4A8
// buffer (row = e*N+n), returning the device buffers plus a per-row f64 reference dot against
// the given int8 activation (computed from the SAME packed nibbles + f16 scales, so any
// mismatch is an assembly bug, not quant noise). Mirrors int4Concat's layout.
func buildStackedExperts(t *testing.T, d *Device, E, N, K int, aq []int8, aSc float32, rng *rand.Rand) (wq, sct Buffer, ref [][]float32) {
	words := make([]uint32, E*N*(K/8))
	scalesH := make([]uint16, E*N*(K/32))
	ref = make([][]float32, E)
	for e := 0; e < E; e++ {
		ref[e] = make([]float32, N)
		for n := 0; n < N; n++ {
			row := make([]float32, K)
			for k := range row {
				row[k] = rng.Float32()*2 - 1
			}
			rIdx := e*N + n
			w, s := packW4A8Row(row)
			copy(words[rIdx*(K/8):(rIdx+1)*(K/8)], w)
			var acc float64
			for g := 0; g < K/32; g++ {
				scalesH[rIdx*(K/32)+g] = f32ToF16(s[g])
				sc := float64(f16ToF32(scalesH[rIdx*(K/32)+g]))
				for j := 0; j < 32; j++ {
					k := g*32 + j
					nib := int((w[k/8]>>(4*uint(k%8)))&0xF) - 8
					acc += float64(nib) * float64(aq[k]) * sc
				}
			}
			ref[e][n] = float32(acc) * aSc
		}
	}
	return d.NewBufferUint32s(words), d.NewBufferU16s(scalesH), ref
}

// TestMoE_indexedExpertGEMV validates the indexed addressing (weightRow = idx[slot]*rowsPer
// Expert + n): gemv_w4a8_moe must read exactly the routed expert's rows out of the stacked
// buffer, and gemv_w4a8_moe_wacc must fold the router weight into the residual accumulation.
func TestMoE_indexedExpertGEMV(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Fatalf("device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pGU, err := d.NewComputePipeline(lib, "gemv_w4a8_moe")
	if err != nil {
		t.Fatalf("pipeline gemv_w4a8_moe: %v", err)
	}
	pWacc, err := d.NewComputePipeline(lib, "gemv_w4a8_moe_wacc")
	if err != nil {
		t.Fatalf("pipeline gemv_w4a8_moe_wacc: %v", err)
	}

	const E, N, K = 8, 64, 128 // E experts, N output rows/expert, K contraction (÷32)
	rng := rand.New(rand.NewSource(23))
	// int8 activation.
	act := make([]float32, K)
	for i := range act {
		act[i] = rng.Float32()*2 - 1
	}
	amx := float32(0)
	for _, v := range act {
		if a := float32(math.Abs(float64(v))); a > amx {
			amx = a
		}
	}
	aSc := amx / 127
	aq := make([]int8, K)
	for i, v := range act {
		aq[i] = int8(clampI(int(math.Round(float64(v/aSc))), -127, 127))
	}
	wq, sct, ref := buildStackedExperts(t, d, E, N, K, aq, aSc, rng)

	// Pick a non-zero expert to prove the idx offset is honored (slot 0 → idx[0]=chosen).
	const chosen = 5
	idxBuf := d.NewBufferUint32s([]uint32{chosen, 0, 0, 0})
	q := d.NewCommandQueue()

	// mode-0 overwrite: out[n] = expert[chosen].row(n)·act
	out := d.NewBufferLen(N)
	q.Run1DBatchTG(pGU, N*32, 256, 1, K*2,
		wq, sct, d.NewBufferInt8(aq), d.NewBufferFloats([]float32{aSc}), out,
		d.NewBufferU32(uint32(K)), idxBuf, d.NewBufferU32(0), d.NewBufferU32(uint32(N)))
	got := out.Floats()
	checkClose(t, "gemv_w4a8_moe", got, ref[chosen], 1e-3)

	// mode-1 weighted-accumulate: x[n] = x0[n] + wgt·expert[chosen].row(n)·act
	const wgt = 0.375
	x0 := make([]float32, N)
	for i := range x0 {
		x0[i] = rng.Float32()*2 - 1
	}
	xBuf := d.NewBufferFloats(x0)
	q.Run1DBatchTG(pWacc, N*32, 256, 1, K*2,
		wq, sct, d.NewBufferInt8(aq), d.NewBufferFloats([]float32{aSc}), xBuf,
		d.NewBufferU32(uint32(K)), idxBuf, d.NewBufferFloats([]float32{wgt, 0, 0, 0}),
		d.NewBufferU32(0), d.NewBufferU32(uint32(N)))
	wantWacc := make([]float32, N)
	for n := range wantWacc {
		wantWacc[n] = x0[n] + wgt*ref[chosen][n]
	}
	checkClose(t, "gemv_w4a8_moe_wacc", xBuf.Floats(), wantWacc, 1e-3)
}

func clampI(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func checkClose(t *testing.T, name string, got, want []float32, tol float64) {
	t.Helper()
	var dot, na, nb float64
	for i := range want {
		if d := math.Abs(float64(got[i] - want[i])); d > tol {
			t.Fatalf("%s: [%d]=%.5f want %.5f (Δ=%.2e)", name, i, got[i], want[i], d)
		}
		dot += float64(got[i]) * float64(want[i])
		na += float64(got[i]) * float64(got[i])
		nb += float64(want[i]) * float64(want[i])
	}
	if cos := dot / (math.Sqrt(na*nb) + 1e-12); cos < 0.9999 {
		t.Fatalf("%s: cosine %.6f < 0.9999", name, cos)
	}
}
