//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"

	"github.com/townsendmerino/goinfer/decoder"
)

// kso1cPlanes lays out Q6_K blocks (a weight of T = len(raw)/210 blocks) as q6kPlanes does, written independently of
// streamKQuant: ql, qh, scales, d, each over the whole weight.
func kso1cPlanes(raw []byte) []uint8 {
	T := len(raw) / 210
	out := make([]uint8, T*210)
	for g := range T {
		b := raw[g*210 : (g+1)*210]
		copy(out[g*128:], b[:128])
		copy(out[T*128+g*64:], b[128:192])
		copy(out[T*192+g*16:], b[192:208])
		copy(out[T*208+g*2:], b[208:210])
	}
	return out
}

// kso1cRef is the float64 reference of an N-row native GEMV over the per-32 codes and scales, with a planted defect when
// plant is non-zero: (1) Q5_K's high bit dropped; (2) Q6_K's −32 offset dropped; (3) a Q6_K group read with its neighbour's
// sub-scale; (4) every row read one super-block late. It returns the outputs and each row's Σ|w·a|.
func kso1cRef(kind string, raw []byte, N, K int, codes []int8, scales []float32, plant int) (want, bar []float64) {
	nSB := K / 256
	bs := map[string]int{"q5k": 176, "q6k": 210}[kind]
	want, bar = make([]float64, N), make([]float64, N)
	wrow := make([]float64, K)
	for n := range N {
		r := append([]byte(nil), raw[n*nSB*bs:(n+1)*nSB*bs]...)
		if plant == 4 && nSB > 1 {
			r = append(r[bs:], r[:bs]...)
		}
		switch {
		case kind == "q5k" && plant == 1:
			for b := range nSB {
				clear(r[b*176+16 : b*176+48])
			}
			ksA1Q5KRow(r, wrow)
		case kind == "q5k":
			ksA1Q5KRow(r, wrow)
		case plant == 2 || plant == 3:
			kso1cQ6KPlanted(r, wrow, plant)
		default:
			ksA1Q6KRow(r, wrow)
		}
		for k := range K {
			x := wrow[k] * float64(codes[k]) * float64(scales[k/32])
			want[n] += x
			bar[n] += math.Abs(x)
		}
	}
	return want, bar
}

// kso1cQ6KPlanted is ksA1Q6KRow with plant 2 (no −32) or plant 3 (each element takes the other 16-element half's sub-scale).
func kso1cQ6KPlanted(raw []byte, out []float64, plant int) {
	off := 32
	if plant == 2 {
		off = 0
	}
	for sb := 0; sb < len(raw)/210; sb++ {
		b := raw[sb*210 : sb*210+210]
		d := ksA1HalfVal(uint16(b[208]) | uint16(b[209])<<8)
		for n := range 2 {
			ql, qh, sc := b[64*n:], b[128+32*n:], b[192+8*n:]
			y := out[256*sb+128*n:]
			for l := range 32 {
				is := l / 16
				if plant == 3 {
					is ^= 1
				}
				q := [4]int{
					int(ql[l]&15|((qh[l]>>0)&3)<<4) - off, int(ql[l+32]&15|((qh[l]>>2)&3)<<4) - off,
					int(ql[l]>>4|((qh[l]>>4)&3)<<4) - off, int(ql[l+32]>>4|((qh[l]>>6)&3)<<4) - off,
				}
				for g := range 4 {
					y[l+32*g] = d * float64(int8(sc[is+2*g])) * float64(q[g])
				}
			}
		}
	}
}

// TestKSO1C_nativeKernels is O1C-G1's kernel half (docs/tasks/task-kernel-survey-2026-10.md, "KS-O1 on CUDA: gates"):
// gemv_q5k_g32 and gemv_q6k_g32 against kso1cRef at super-block counts 1, 2, 6, 12, 19, 32, 35, a row count that leaves a
// partial block of rows, with bias and accumulate, every row within G-Q1's bar; then each planted reference is red. CUDA's
// PTX is built offline, so the plants are carried by the reference: a kernel that made the planted mistake would match it.
func TestKSO1C_nativeKernels(t *testing.T) {
	ctx, _, launch := actGroupHarness(t)
	rng := rand.New(rand.NewSource(1024))
	const N = 8*kquantGemvRows*3 + 5
	worst := map[string]float64{}
	plantMin := map[int]float64{1: math.Inf(1), 2: math.Inf(1), 3: math.Inf(1), 4: math.Inf(1)}
	for _, kind := range []string{"q5k", "q6k"} {
		for _, nSB := range []int{1, 2, 6, 12, 19, 32, 35} {
			K := 256 * nSB
			av := outlierVec(rng, K)
			aC, aSc := quantG32Go(av)
			nG := K / 32
			aBuf := make([]float32, 2*nG)
			copy(aBuf, aSc)
			for g := range nG {
				var s int32
				for _, c := range aC[32*g : 32*g+32] {
					s += int32(c)
				}
				aBuf[nG+g] = aSc[g] * float32(s)
			}
			raw := ksA1Blocks(rng, kind, N, K)
			var dw gc.KernelArg
			if kind == "q5k" {
				dw = gc.Arg(upload(t, ctx, ksA1Words(raw)))
			} else {
				dw = gc.Arg(upload(t, ctx, kso1cPlanes(raw)))
			}
			bias, dst0 := make([]float32, N), make([]float32, N)
			for i := range bias {
				bias[i], dst0[i] = float32(rng.NormFloat64()), float32(rng.NormFloat64())
			}
			want, bar := kso1cRef(kind, raw, N, K, aC, aSc, 0)
			da, das, db, dd := upload(t, ctx, packActI8(aC)), upload(t, ctx, aBuf), upload(t, ctx, bias), upload(t, ctx, dst0)
			cfg := gc.LaunchConfig{GridX: uint32((N + 8*kquantGemvRows - 1) / (8 * kquantGemvRows)), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}
			launch("gemv_"+kind+"_g32", cfg, dw, gc.Arg(da), gc.Arg(das), gc.Arg(db),
				gc.ArgValue(int32(N)), gc.ArgValue(int32(nSB)), gc.Arg(dd), gc.ArgValue(int32(1)))
			got := download(t, dd, N)
			for n := range N {
				g := float64(got[n]) - float64(dst0[n]) - float64(bias[n])
				e := math.Abs(g-want[n]) / (bar[n] + 1e-30)
				if math.IsNaN(e) || e > 1e-5 {
					t.Fatalf("%s NB=%d row %d: |err|/Σ|w·a| %.3g > 1e-5 (got %v, want %v)", kind, nSB, n, e, g, want[n])
				}
				worst[kind] = max(worst[kind], e)
			}
			// The plants: each planted reference must be red against this kernel's output.
			plants := map[string][]int{"q5k": {1, 4}, "q6k": {2, 3, 4}}[kind]
			for _, p := range plants {
				if p == 4 && nSB == 1 {
					continue // a one-block row has no neighbour to shift into
				}
				pw, pb := kso1cRef(kind, raw, N, K, aC, aSc, p)
				red := 0.0
				for n := range N {
					g := float64(got[n]) - float64(dst0[n]) - float64(bias[n])
					red = max(red, math.Abs(g-pw[n])/(pb[n]+1e-30))
				}
				plantMin[p] = min(plantMin[p], red)
			}
		}
	}
	for p, v := range plantMin {
		if !(v > 1e-5) {
			t.Errorf("plant %d is not red at every shape: smallest worst |err|/Σ|w·a| %.3g", p, v)
		}
	}
	t.Logf("O1C-G1 kernels: worst |err|/Σ|w·a| Q5_K %.3g, Q6_K %.3g (bar 1e-5); plants' smallest reds %v", worst["q5k"], worst["q6k"], plantMin)
}

// TestKSO1C_fusedEqualsUnfused is O1C-G2's fused half: fused_rms_qkv_g32 and fused_rms_gu_g32 with the native kinds mixed into a
// group equal rmsnorm_quant_g32 followed by one standalone GEMV a projection, byte for byte, at every rows-per-warp.
func TestKSO1C_fusedEqualsUnfused(t *testing.T) {
	ctx, _, launch := actGroupHarness(t)
	rng := rand.New(rand.NewSource(77))
	const H, qDim, kvDim, I = 1536, 1536, 256, 512
	x := outlierVec(rng, H)
	for i := range x {
		if x[i] == 0 {
			x[i] = float32(rng.NormFloat64())
		}
	}
	nw := make([]float32, H)
	for i := range nw {
		nw[i] = float32(0.5 + rng.Float64())
	}
	dx, dnw := upload(t, ctx, x), upload(t, ctx, nw)
	type proj struct {
		kind string
		N    int
		W    gc.KernelArg
		bias *gc.Buffer[float32]
	}
	mk := func(kind string, N int) proj {
		raw := ksA1Blocks(rng, kind, N, H)
		p := proj{kind: kind, N: N}
		switch kind {
		case "q6k":
			p.W = gc.Arg(upload(t, ctx, kso1cPlanes(raw)))
		default:
			p.W = gc.Arg(upload(t, ctx, ksA1Words(raw)))
		}
		b := make([]float32, N)
		for i := range b {
			b[i] = float32(rng.NormFloat64())
		}
		p.bias = upload(t, ctx, b)
		return p
	}
	code := map[string]int32{"q4k": 2, "q5k": 3, "q6k": 4}
	dq, das := mustAlloc[int32](t, ctx, H/4), mustAlloc[float32](t, ctx, 2*H/32)
	one := gc.LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((H + 256) * 4)}
	launch("rmsnorm_quant_g32", one, gc.Arg(dx), gc.Arg(dnw), gc.ArgValue(int32(H)), gc.ArgValue(float32(1e-6)),
		gc.ArgValue(int32(0)), gc.Arg(dq), gc.Arg(das))
	gemv := func(p proj, bias gc.KernelArg) []float32 {
		out := mustAlloc[float32](t, ctx, p.N)
		R := kquantGemvRows
		if p.kind == "q4k" {
			R = 1
		}
		cfg := gc.LaunchConfig{GridX: uint32((p.N + 8*R - 1) / (8 * R)), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}
		launch("gemv_"+p.kind+"_g32", cfg, p.W, gc.Arg(dq), gc.Arg(das), bias,
			gc.ArgValue(int32(p.N)), gc.ArgValue(int32(H/256)), gc.Arg(out), gc.ArgValue(int32(0)))
		return download(t, out, p.N)
	}
	same := func(name string, got, want []float32) {
		t.Helper()
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("%s[%d] = %v, unfused %v: not bit-identical", name, i, got[i], want[i])
			}
		}
	}
	shmem := uint32((H + 256 + H/4 + 2*(H/32)) * 4)
	null := gc.ArgDevicePtr(0)
	// Q6_K in each slot, so its whole-weight plane offsets are taken from both qDim and kvDim (projRow's N).
	for _, kinds := range [][3]string{{"q6k", "q5k", "q4k"}, {"q5k", "q6k", "q6k"}} {
		q, k, v := mk(kinds[0], qDim), mk(kinds[1], kvDim), mk(kinds[2], kvDim)
		wq, wk, wv := gemv(q, gc.Arg(q.bias)), gemv(k, gc.Arg(k.bias)), gemv(v, gc.Arg(v.bias))
		for _, rpw := range []int{1, 2, 4} {
			nrows := qDim + 2*kvDim
			cfg := gc.LaunchConfig{GridX: uint32((nrows + 8*rpw - 1) / (8 * rpw)), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: shmem}
			oq, ok, ov := mustAlloc[float32](t, ctx, qDim), mustAlloc[float32](t, ctx, kvDim), mustAlloc[float32](t, ctx, kvDim)
			launch("fused_rms_qkv_g32", cfg, gc.Arg(dx), gc.Arg(dnw), gc.ArgValue(int32(H)), gc.ArgValue(float32(1e-6)), gc.ArgValue(int32(0)),
				q.W, null, gc.Arg(q.bias), gc.ArgValue(code[q.kind]),
				k.W, null, gc.Arg(k.bias), gc.ArgValue(code[k.kind]),
				v.W, null, gc.Arg(v.bias), gc.ArgValue(code[v.kind]),
				gc.ArgValue(int32(qDim)), gc.ArgValue(int32(kvDim)), gc.ArgValue(int32(rpw)), gc.Arg(oq), gc.Arg(ok), gc.Arg(ov))
			same("q ("+kinds[0]+")", download(t, oq, qDim), wq)
			same("k ("+kinds[1]+")", download(t, ok, kvDim), wk)
			same("v ("+kinds[2]+")", download(t, ov, kvDim), wv)
		}
	}
	g, u := mk("q5k", I), mk("q6k", I)
	wg, wu := gemv(g, null), gemv(u, null)
	for _, rpw := range []int{1, 8} {
		cfg := gc.LaunchConfig{GridX: uint32((2*I + 8*rpw - 1) / (8 * rpw)), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: shmem}
		og, ou := mustAlloc[float32](t, ctx, I), mustAlloc[float32](t, ctx, I)
		launch("fused_rms_gu_g32", cfg, gc.Arg(dx), gc.Arg(dnw), gc.ArgValue(int32(H)), gc.ArgValue(float32(1e-6)), gc.ArgValue(int32(0)),
			g.W, null, gc.ArgValue(code[g.kind]),
			u.W, null, gc.ArgValue(code[u.kind]),
			gc.ArgValue(int32(I)), gc.ArgValue(int32(rpw)), gc.Arg(og), gc.Arg(ou))
		same("gate (q5k)", download(t, og, I), wg)
		same("up (q6k)", download(t, ou, I), wu)
	}
}

// TestKSO1C_untouchedKernelsBitIdentical is O1C-G2's untouched half: every kernel actgroup.cu had before the native kinds, run
// from the old actgroup.ptx (GOINFER_KSO1C_OLD_ACTGROUP_PTX, the file at origin/main) and from the new one on the same inputs,
// gives the same bytes: the three standalone GEMVs, the quantizers, and both fused kernels with int8, int4 and Q4_K projections
// at rows-per-warp 1, 2 and 4.
func TestKSO1C_untouchedKernelsBitIdentical(t *testing.T) {
	oldPath := os.Getenv("GOINFER_KSO1C_OLD_ACTGROUP_PTX")
	if oldPath == "" {
		t.Skip("set GOINFER_KSO1C_OLD_ACTGROUP_PTX to origin/main's cuda/testdata/actgroup.ptx")
	}
	oldPTX, err := os.ReadFile(oldPath)
	if err != nil {
		t.Fatal(err)
	}
	ctx, stream, _ := actGroupHarness(t)
	mods := map[string]*gc.Module{}
	for name, src := range map[string][]byte{"old": oldPTX, "new": actGroupPTX} {
		if mods[name], err = ctx.LoadModule(src); err != nil {
			t.Fatalf("LoadModule(%s): %v", name, err)
		}
	}
	bg := context.Background()
	run := func(mod, fn string, cfg gc.LaunchConfig, args ...gc.KernelArg) {
		t.Helper()
		f, err := mods[mod].Function(fn)
		if err != nil {
			t.Fatalf("%s %s: %v", mod, fn, err)
		}
		if err := f.LaunchOn(bg, stream, cfg, args...); err != nil {
			t.Fatalf("%s %s: %v", mod, fn, err)
		}
		if err := stream.Synchronize(bg); err != nil {
			t.Fatalf("%s %s sync: %v", mod, fn, err)
		}
	}
	rng := rand.New(rand.NewSource(5))
	const H, qDim, kvDim, I = 1536, 1536, 256, 512
	x := outlierVec(rng, H)
	for i := range x {
		if x[i] == 0 {
			x[i] = float32(rng.NormFloat64())
		}
	}
	nw := make([]float32, H)
	for i := range nw {
		nw[i] = float32(0.5 + rng.Float64())
	}
	dx, dnw := upload(t, ctx, x), upload(t, ctx, nw)
	type proj struct {
		kind string
		N    int
		W    *gc.Buffer[uint32]
		sc   gc.KernelArg
		s8   *gc.Buffer[float32]
		s4   *gc.Buffer[uint16]
		bias *gc.Buffer[float32]
	}
	mk := func(kind string, N int) proj {
		p := proj{kind: kind, N: N, sc: gc.ArgDevicePtr(0)}
		switch kind {
		case "q4k":
			p.W = upload(t, ctx, ksA1Words(q4kHostRows(rng, N, H, false)))
		case "int8":
			w := make([]uint32, N*H/4)
			for i := range w {
				w[i] = rng.Uint32()
			}
			s := make([]float32, N)
			for i := range s {
				s[i] = float32(0.001 + 0.01*rng.Float64())
			}
			p.W, p.s8 = upload(t, ctx, w), upload(t, ctx, s)
			p.sc = gc.Arg(p.s8)
		case "int4":
			w := make([]uint32, N*H/8)
			for i := range w {
				w[i] = rng.Uint32()
			}
			s := make([]uint16, N*H/32)
			for i := range s {
				s[i] = f32tof16(float32(0.001 + 0.01*rng.Float64()))
			}
			p.W, p.s4 = upload(t, ctx, w), upload(t, ctx, s)
			p.sc = gc.Arg(p.s4)
		}
		b := make([]float32, N)
		for i := range b {
			b[i] = float32(rng.NormFloat64())
		}
		p.bias = upload(t, ctx, b)
		return p
	}
	same := func(name string, a, b []float32) {
		t.Helper()
		for i := range a {
			if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
				t.Fatalf("%s[%d]: old %v, new %v", name, i, a[i], b[i])
			}
		}
	}
	cells := 0
	// The quantizer: both modules' codes and scales.
	one := gc.LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((H + 256) * 4)}
	q := map[string]*gc.Buffer[int32]{}
	s := map[string]*gc.Buffer[float32]{}
	for _, m := range []string{"old", "new"} {
		q[m], s[m] = mustAlloc[int32](t, ctx, H/4), mustAlloc[float32](t, ctx, 2*H/32)
		run(m, "rmsnorm_quant_g32", one, gc.Arg(dx), gc.Arg(dnw), gc.ArgValue(int32(H)), gc.ArgValue(float32(1e-6)), gc.ArgValue(int32(0)), gc.Arg(q[m]), gc.Arg(s[m]))
	}
	qo, qn := download(t, q["old"], H/4), download(t, q["new"], H/4)
	for i := range qo {
		if qo[i] != qn[i] {
			t.Fatalf("rmsnorm_quant_g32 code word %d: old %#x, new %#x", i, qo[i], qn[i])
		}
	}
	same("rmsnorm_quant_g32 scales", download(t, s["old"], 2*H/32), download(t, s["new"], 2*H/32))
	cells++
	// quant_vec_g32 and glu_quant_g32 (SiLU and GELU-tanh), each module's codes and scales.
	{
		const I = 1536
		gv, uv := outlierVec(rng, I), outlierVec(rng, I)
		dg, du := upload(t, ctx, gv), upload(t, ctx, uv)
		plain := gc.LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}
		type quantOut struct {
			q []int32
			s []float32
		}
		runQ := func(m, fn string, act int32) quantOut {
			dq, ds := mustAlloc[int32](t, ctx, I/4), mustAlloc[float32](t, ctx, 2*I/32)
			if fn == "quant_vec_g32" {
				run(m, fn, plain, gc.Arg(dg), gc.ArgValue(int32(I)), gc.Arg(dq), gc.Arg(ds))
			} else {
				scr := mustAlloc[float32](t, ctx, I)
				run(m, fn, plain, gc.Arg(dg), gc.Arg(du), gc.ArgValue(int32(0)), gc.ArgValue(int32(0)), gc.ArgValue(int32(I)),
					gc.ArgValue(act), gc.Arg(dq), gc.Arg(ds), gc.Arg(scr))
			}
			return quantOut{download(t, dq, I/4), download(t, ds, 2*I/32)}
		}
		for _, c := range []struct {
			fn  string
			act int32
		}{{"quant_vec_g32", 0}, {"glu_quant_g32", 1}, {"glu_quant_g32", 0}} {
			o, n := runQ("old", c.fn, c.act), runQ("new", c.fn, c.act)
			for i := range o.q {
				if o.q[i] != n.q[i] {
					t.Fatalf("%s act %d code word %d: old %#x, new %#x", c.fn, c.act, i, o.q[i], n.q[i])
				}
			}
			same(fmt.Sprintf("%s act %d scales", c.fn, c.act), o.s, n.s)
			cells++
		}
	}
	dq, das := q["old"], s["old"]
	cfg8 := func(N int) gc.LaunchConfig {
		return gc.LaunchConfig{GridX: uint32((N + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}
	}
	for _, kind := range []string{"int8", "int4", "q4k"} {
		p := mk(kind, 200)
		var outs [2][]float32
		for i, m := range []string{"old", "new"} {
			out := upload(t, ctx, make([]float32, p.N))
			switch kind {
			case "q4k":
				run(m, "gemv_q4k_g32", cfg8(p.N), gc.Arg(p.W), gc.Arg(dq), gc.Arg(das), gc.Arg(p.bias), gc.ArgValue(int32(p.N)), gc.ArgValue(int32(H/256)), gc.Arg(out), gc.ArgValue(int32(1)))
			case "int8":
				run(m, "gemv_w8a8_g32", cfg8(p.N), gc.Arg(p.W), gc.Arg(dq), gc.Arg(p.s8), gc.Arg(das), gc.Arg(p.bias), gc.ArgValue(int32(p.N)), gc.ArgValue(int32(H/4)), gc.Arg(out), gc.ArgValue(int32(1)))
			case "int4":
				run(m, "gemv_w4a8_g32", cfg8(p.N), gc.Arg(p.W), gc.Arg(dq), gc.Arg(p.s4), gc.Arg(das), gc.Arg(p.bias), gc.ArgValue(int32(p.N)), gc.ArgValue(int32(H/8)), gc.ArgValue(int32(H/32)), gc.Arg(out), gc.ArgValue(int32(1)))
			}
			outs[i] = download(t, out, p.N)
		}
		same("gemv "+kind, outs[0], outs[1])
		cells++
	}
	code := map[string]int32{"int8": 0, "int4": 1, "q4k": 2}
	shmem := uint32((H + 256 + H/4 + 2*(H/32)) * 4)
	for _, kinds := range [][3]string{{"q4k", "int8", "int4"}, {"int8", "int4", "q4k"}, {"int4", "q4k", "int8"}} {
		pq, pk, pv := mk(kinds[0], qDim), mk(kinds[1], kvDim), mk(kinds[2], kvDim)
		for _, rpw := range []int{1, 2, 4} {
			var outs [2][3][]float32
			for i, m := range []string{"old", "new"} {
				oq, ok, ov := mustAlloc[float32](t, ctx, qDim), mustAlloc[float32](t, ctx, kvDim), mustAlloc[float32](t, ctx, kvDim)
				nrows := qDim + 2*kvDim
				cfg := gc.LaunchConfig{GridX: uint32((nrows + 8*rpw - 1) / (8 * rpw)), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: shmem}
				run(m, "fused_rms_qkv_g32", cfg, gc.Arg(dx), gc.Arg(dnw), gc.ArgValue(int32(H)), gc.ArgValue(float32(1e-6)), gc.ArgValue(int32(0)),
					gc.Arg(pq.W), pq.sc, gc.Arg(pq.bias), gc.ArgValue(code[pq.kind]),
					gc.Arg(pk.W), pk.sc, gc.Arg(pk.bias), gc.ArgValue(code[pk.kind]),
					gc.Arg(pv.W), pv.sc, gc.Arg(pv.bias), gc.ArgValue(code[pv.kind]),
					gc.ArgValue(int32(qDim)), gc.ArgValue(int32(kvDim)), gc.ArgValue(int32(rpw)), gc.Arg(oq), gc.Arg(ok), gc.Arg(ov))
				outs[i] = [3][]float32{download(t, oq, qDim), download(t, ok, kvDim), download(t, ov, kvDim)}
			}
			for j := range 3 {
				same(fmt.Sprintf("fused_rms_qkv_g32 %v rpw %d out %d", kinds, rpw, j), outs[0][j], outs[1][j])
			}
			cells++
		}
		pg, pu := mk(kinds[0], I), mk(kinds[1], I)
		for _, rpw := range []int{1, 2, 4} {
			var outs [2][2][]float32
			for i, m := range []string{"old", "new"} {
				og, ou := mustAlloc[float32](t, ctx, I), mustAlloc[float32](t, ctx, I)
				cfg := gc.LaunchConfig{GridX: uint32((2*I + 8*rpw - 1) / (8 * rpw)), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: shmem}
				run(m, "fused_rms_gu_g32", cfg, gc.Arg(dx), gc.Arg(dnw), gc.ArgValue(int32(H)), gc.ArgValue(float32(1e-6)), gc.ArgValue(int32(0)),
					gc.Arg(pg.W), pg.sc, gc.ArgValue(code[pg.kind]),
					gc.Arg(pu.W), pu.sc, gc.ArgValue(code[pu.kind]),
					gc.ArgValue(int32(I)), gc.ArgValue(int32(rpw)), gc.Arg(og), gc.Arg(ou))
				outs[i] = [2][]float32{download(t, og, I), download(t, ou, I)}
			}
			same(fmt.Sprintf("fused_rms_gu_g32 %v rpw %d gate", kinds[:2], rpw), outs[0][0], outs[1][0])
			same(fmt.Sprintf("fused_rms_gu_g32 %v rpw %d up", kinds[:2], rpw), outs[0][1], outs[1][1])
			cells++
		}
	}
	t.Logf("O1C-G2 untouched: %d cells byte-equal between the old and the new actgroup.ptx", cells)
}

// TestKSO1C_referenceMatchesAikit is O1C-G1's reference half: kso1cRef's dequantization (ksA1Q5KRow, ksA1Q6KRow) within 1e-6 of
// aikit's GGUF dequantizer, relative to the row's largest weight, on three rows of every Q5_K and Q6_K tensor of the 0.5B q4_k_m
// and Phi-3 mini q4.
func TestKSO1C_referenceMatchesAikit(t *testing.T) {
	for _, name := range kso1cModels {
		path := modelPath(name)
		if _, err := os.Stat(path); err != nil {
			t.Skipf("no model %s: %v", path, err) // run-o1c-day.sh counts a skip as a failure
		}
		g, err := decoder.OpenGGUFMmap(path)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		checked := map[string]int{}
		for _, tn := range g.Names() {
			for _, k := range []struct {
				kind string
				bb   int
				get  func(string) ([]int, []byte, bool, error)
				ref  func([]byte, []float64)
			}{{"q5k", 176, g.Q5KRaw, ksA1Q5KRow}, {"q6k", 210, g.Q6KRaw, ksA1Q6KRow}} {
				dims, raw, ok, err := k.get(tn)
				if err != nil {
					t.Fatalf("%s %s: %v", name, tn, err)
				}
				if !ok {
					continue
				}
				in, out := dims[0], dims[1]
				_, into, err := g.RowDequantizer(tn)
				if err != nil {
					t.Fatal(err)
				}
				want, got := make([]float32, in), make([]float64, in)
				rb := in / 256 * k.bb
				for _, r := range []int{0, out / 2, out - 1} {
					if err := into(r*in, want); err != nil {
						t.Fatal(err)
					}
					k.ref(raw[r*rb:(r+1)*rb], got)
					big := 0.0
					for _, v := range want {
						big = max(big, math.Abs(float64(v)))
					}
					for i := range want {
						if math.Abs(got[i]-float64(want[i])) > 1e-6*big {
							t.Fatalf("%s %s row %d [%d]: reference %v, aikit %v", name, tn, r, i, got[i], want[i])
						}
					}
				}
				checked[k.kind]++
			}
		}
		g.Close()
		if checked["q6k"] == 0 || (name != kso1cModels[0] && checked["q5k"] == 0) {
			t.Errorf("%s: checked %v, want both kinds on Phi-3 and Q6_K on the 0.5B", name, checked)
		}
		t.Logf("O1C-G1 reference: %s, within 1e-6 of aikit on %d Q5_K and %d Q6_K tensors", name, checked["q5k"], checked["q6k"])
	}
}
