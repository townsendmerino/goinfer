//go:build cuda

package cuda

import (
	"context"
	"fmt"
	"math"
	"math/big"
	"math/rand/v2"
	"os"
	"runtime"
	"slices"
	"testing"
	"time"

	gc "github.com/eitamring/gocudrv/cuda"
	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/aikit/linalg"
)

// moeCPUFixture is a stack of random int4 experts at one projection's geometry, an int8 activation for it, and the device
// copies the kernels read.
type moeCPUFixture struct {
	nE, n, k, kw, kg int
	h                hostW
	act              []int32
	aScale           float32
}

func newMoECPUFixture(t *testing.T, nE, n, k int, seed uint64) moeCPUFixture {
	t.Helper()
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	mats := make([]linalg.WeightMat, nE)
	ptrs := make([]*linalg.WeightMat, nE)
	for e := range mats {
		f := make([]float32, n*k)
		for i := range f {
			f[i] = float32(rng.NormFloat64() * 0.05)
		}
		mats[e] = linalg.QuantizeInt4(f, n, k, 32)
		ptrs[e] = &mats[e]
	}
	h, err := packWeightStack(ptrs...)
	if err != nil {
		t.Fatalf("packWeightStack: %v", err)
	}
	act := make([]int32, k/4)
	for i := range act {
		var p int32
		for b := range 4 {
			p |= (int32(int8(rng.IntN(255)-127)) & 0xff) << (8 * uint(b))
		}
		act[i] = p
	}
	return moeCPUFixture{nE: nE, n: n, k: k, kw: k / 8, kg: k / 32, h: h, act: act, aScale: float32(0.001 + rng.Float64()*0.05)}
}

// moeCPUDevice runs gemv_w4a8_moe (wgt == nil) or gemv_w4a8_moe_wacc over every expert of f in turn, the wacc form onto a
// copy of dst0, and returns each expert's output.
func moeCPUDevice(t *testing.T, f moeCPUFixture, wgt []float32, dst0 []float32) [][]float32 {
	t.Helper()
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	dev, err := gc.GetDevice(0)
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	cx, err := dev.Primary()
	if err != nil {
		t.Skipf("primary ctx: %v", err)
	}
	defer cx.Close()
	bg := context.Background()
	mod, err := cx.LoadModule(moePTX)
	if err != nil {
		t.Fatalf("LoadModule(moe): %v", err)
	}
	name := "gemv_w4a8_moe"
	if wgt != nil {
		name = "gemv_w4a8_moe_wacc"
	}
	fn, err := mod.Function(name)
	if err != nil {
		t.Fatal(err)
	}
	stream := mustStream(t, cx)
	dW := mustAlloc[uint32](t, cx, len(f.h.wpk))
	dGs := mustAlloc[uint16](t, cx, len(f.h.ws16))
	dA := mustAlloc[int32](t, cx, len(f.act))
	dAs := mustAlloc[float32](t, cx, 1)
	dIdx := mustAlloc[uint32](t, cx, f.nE)
	dWgt := mustAlloc[float32](t, cx, f.nE)
	dOut := mustAlloc[float32](t, cx, f.n)
	for _, b := range []interface{ Close() error }{dW, dGs, dA, dAs, dIdx, dWgt, dOut} {
		defer b.Close()
	}
	idx := make([]uint32, f.nE)
	for e := range idx {
		idx[e] = uint32(e)
	}
	w := wgt
	if w == nil {
		w = make([]float32, f.nE)
	}
	for _, e := range []error{gc.CopyHtoD(bg, dW, f.h.wpk), gc.CopyHtoD(bg, dGs, f.h.ws16), gc.CopyHtoD(bg, dA, f.act),
		gc.CopyHtoD(bg, dAs, []float32{f.aScale}), gc.CopyHtoD(bg, dIdx, idx), gc.CopyHtoD(bg, dWgt, w)} {
		if e != nil {
			t.Fatalf("H2D: %v", e)
		}
	}
	out := make([][]float32, f.nE)
	cfg := gc.LaunchConfig{GridX: uint32((f.n + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}
	for e := 0; e < f.nE; e++ {
		init := make([]float32, f.n)
		if dst0 != nil {
			copy(init, dst0)
		}
		if err := gc.CopyHtoD(bg, dOut, init); err != nil {
			t.Fatal(err)
		}
		if wgt != nil {
			err = fn.LaunchOn(bg, stream, cfg, gc.Arg(dW), gc.Arg(dA), gc.Arg(dGs), gc.Arg(dAs), gc.Arg(dIdx), gc.Arg(dWgt),
				gc.ArgValue(int32(e)), gc.ArgValue(int32(f.n)), gc.ArgValue(int32(f.n)), gc.ArgValue(int32(f.kw)),
				gc.ArgValue(int32(f.kg)), gc.Arg(dOut))
		} else {
			err = fn.LaunchOn(bg, stream, cfg, gc.Arg(dW), gc.Arg(dA), gc.Arg(dGs), gc.Arg(dAs), gc.Arg(dIdx),
				gc.ArgValue(int32(e)), gc.ArgValue(int32(f.n)), gc.ArgValue(int32(f.n)), gc.ArgValue(int32(f.kw)),
				gc.ArgValue(int32(f.kg)), gc.Arg(dOut))
		}
		if err != nil {
			t.Fatalf("launch: %v", err)
		}
		if err := stream.Synchronize(bg); err != nil {
			t.Fatal(err)
		}
		out[e] = make([]float32, f.n)
		if err := gc.CopyDtoH(bg, out[e], dOut); err != nil {
			t.Fatal(err)
		}
	}
	return out
}

// moeCPUHost is moeCPUDevice's computation on the host replica.
func moeCPUHost(f moeCPUFixture, wgt, dst0 []float32, d moeCPUDefect) [][]float32 {
	out := make([][]float32, f.nE)
	for e := 0; e < f.nE; e++ {
		o := make([]float32, f.n)
		w := f.h.wpk[e*f.n*f.kw : (e+1)*f.n*f.kw]
		s := f.h.ws16[e*f.n*f.kg : (e+1)*f.n*f.kg]
		w4a8MoEGemvRef(w, s, f.act, f.aScale, f.n, f.kw, f.kg, o, d)
		if wgt != nil {
			for r := range o {
				o[r] = fmaf32(o[r], wgt[e], dst0[r])
			}
		}
		out[e] = o
	}
	return out
}

func moeCPUMismatches(a, b [][]float32) (n int, first string) {
	for e := range a {
		for r := range a[e] {
			if math.Float32bits(a[e][r]) != math.Float32bits(b[e][r]) {
				if n == 0 {
					first = fmt.Sprintf("expert %d row %d: host %g, device %g", e, r, a[e][r], b[e][r])
				}
				n++
			}
		}
	}
	return n, first
}

// TestMoECPUReplica_bitIdentical is G-S20h: the host replica of the expert GEMVs against the device kernels, bit for bit, at
// the 26B's two geometries (gate/up: 2816 → 1408; down: 704 → 2816), every row of 8 random experts, both the plain kernel
// and the accumulating one; then each planted defect (a change of float-add order) must break the comparison somewhere.
func TestMoECPUReplica_bitIdentical(t *testing.T) {
	cases := []struct {
		name string
		n, k int
	}{{"gate-up", 1408, 2816}, {"down", 2816, 704}}
	defects := []struct {
		name string
		d    moeCPUDefect
	}{{"pair added unfolded", moeCPUDefect{unfoldedPair: true}}, {"blocks walked in reverse", moeCPUDefect{blocksReverse: true}},
		{"serial lane sum", moeCPUDefect{serialTree: true}}}
	seen := make([]int, len(defects))
	for i, c := range cases {
		f := newMoECPUFixture(t, 8, c.n, c.k, uint64(7+i))
		rng := rand.New(rand.NewPCG(uint64(100+i), 3))
		wgt := make([]float32, f.nE)
		for e := range wgt {
			wgt[e] = float32(rng.Float64())
		}
		dst0 := make([]float32, f.n)
		for r := range dst0 {
			dst0[r] = float32(rng.NormFloat64())
		}
		plainDev := moeCPUDevice(t, f, nil, nil)
		waccDev := moeCPUDevice(t, f, wgt, dst0)
		if n, first := moeCPUMismatches(moeCPUHost(f, nil, nil, moeCPUDefect{}), plainDev); n != 0 {
			t.Errorf("%s gemv_w4a8_moe: %d of %d values differ (first %s)", c.name, n, f.nE*f.n, first)
		}
		if n, first := moeCPUMismatches(moeCPUHost(f, wgt, dst0, moeCPUDefect{}), waccDev); n != 0 {
			t.Errorf("%s gemv_w4a8_moe_wacc: %d of %d values differ (first %s)", c.name, n, f.nE*f.n, first)
		}
		for di, d := range defects {
			n, _ := moeCPUMismatches(moeCPUHost(f, nil, nil, d.d), plainDev)
			seen[di] += n
			t.Logf("%s, planted %q: %d of %d values differ", c.name, d.name, n, f.nE*f.n)
		}
	}
	// A defect need only show at one geometry: the down rows hold a single 64-word block, so reversing the block order is the
	// identity there, and a lane's few exact products rarely round when summed.
	for di, d := range defects {
		if seen[di] == 0 {
			t.Errorf("the planted defect %q left the replica bit-identical at both geometries; the gate cannot see it", d.name)
		}
	}
}

// TestFmaf32_roundsOnce checks fmaf32 against an exact a*b+c rounded once to float32 (math/big, nearest-even) on random
// triples, most of them built so the exact sum sits at or within a few float64 ulps of a float32 midpoint: there, rounding
// to float64 first and then to float32 gives the wrong answer.
func TestFmaf32_roundsOnce(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	exact := func(a, b, c float32) float32 {
		x := new(big.Float).SetPrec(300).SetFloat64(float64(a))
		x.Mul(x, new(big.Float).SetPrec(300).SetFloat64(float64(b)))
		x.Add(x, new(big.Float).SetPrec(300).SetFloat64(float64(c)))
		f, _ := x.Float32()
		return f
	}
	nearMid := 0
	for i := 0; i < 2_000_000; i++ {
		var a, b, c float32
		switch i % 3 {
		case 0: // anything
			a, b, c = float32(rng.NormFloat64()), float32(rng.NormFloat64()), float32(rng.NormFloat64())
		default:
			// c near 1 and a*b = ±2^-24·(1+q·2^-23)(1-q·2^-23) = ±(2^-24 - q²·2^-70): half an ulp of c, off by far less than a
			// float64 ulp, so the float64 sum lands on the float32 midpoint and only the error term decides the rounding.
			q := float64(1 + rng.IntN(1<<11))
			c = float32(1 + float64(rng.IntN(1<<20))*math.Pow(2, -23))
			a = float32(math.Pow(2, -24) * (1 + q*math.Pow(2, -23)))
			b = float32(1 - q*math.Pow(2, -23))
			if i%3 == 2 {
				a = -a
			}
		}
		got, want := fmaf32(a, b, c), exact(a, b, c)
		if math.Float32bits(got) != math.Float32bits(want) {
			t.Fatalf("fmaf32(%g, %g, %g) = %g, want %g", a, b, c, got, want)
		}
		if s := float64(a)*float64(b) + float64(c); isF32Midpoint(s) {
			nearMid++
		}
	}
	t.Logf("2,000,000 triples single-rounded; %d had a float64 sum on a float32 midpoint", nearMid)
	if nearMid == 0 {
		t.Fatal("no triple reached a float32 midpoint: the test never exercised the double-rounding case")
	}
}

// TestMoECPUAsm_matchesReference holds the AVX2 kernel to the reference replica bit for bit: the 26B's two row lengths and
// the shapes that exercise each tail (8, 24 and 40 words past the last full 64-word block), random weights and activations.
func TestMoECPUAsm_matchesReference(t *testing.T) {
	if !moeCPUAsm {
		t.Skip("this CPU has no AVX2/FMA/F16C")
	}
	rng := rand.New(rand.NewPCG(11, 12))
	for _, kw := range []int{352, 88, 8, 24, 40, 64, 72, 128, 200} {
		const rows = 97
		w := make([]uint32, rows*kw)
		s := make([]uint16, rows*kw/4)
		a := make([]int32, 2*kw)
		for i := range w {
			w[i] = rng.Uint32()
		}
		for i := range s {
			s[i] = f32ToF16Test(float32(rng.NormFloat64() * 0.01))
		}
		for i := range a {
			a[i] = int32(rng.Uint32())
		}
		tab := newMoEActTables(a)
		got := make([]float32, rows)
		tab.rows(w, s, got)
		for r := 0; r < rows; r++ {
			want := w4a8RowRef(w[r*kw:(r+1)*kw], s[r*kw/4:(r+1)*kw/4], a, moeCPUDefect{})
			if math.Float32bits(got[r]) != math.Float32bits(want) {
				t.Fatalf("kwords %d row %d: AVX2 %g, reference %g", kw, r, got[r], want)
			}
		}
	}
}

// f32ToF16Test rounds a float32 to the nearest f16 (normal range only), for test scales.
func f32ToF16Test(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16(b>>16) & 0x8000
	exp := int32(b>>23&0xff) - 127 + 15
	man := b & 0x7fffff
	if exp <= 0 {
		return sign
	}
	if exp >= 31 {
		return sign | 0x7c00
	}
	h := uint32(exp)<<10 | man>>13
	if man&0x1fff > 0x1000 || (man&0x1fff == 0x1000 && h&1 == 1) {
		h++
	}
	return sign | uint16(h)
}

// TestMoESplitCombine_matchesWaccChain is G-S20i's first half: gemv_w4a8_moe_res for each of k experts, then moe_combine,
// leaves dst bit-identical to k gemv_w4a8_moe_wacc launches in expert order, at the 26B's down geometry (704 → 2816), and the
// host replica's products in place of some experts' device ones change nothing either.
func TestMoESplitCombine_matchesWaccChain(t *testing.T) {
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	dev, err := gc.GetDevice(0)
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	cx, err := dev.Primary()
	if err != nil {
		t.Skipf("primary ctx: %v", err)
	}
	defer cx.Close()
	bg := context.Background()
	const k = 8
	f := newMoECPUFixture(t, 16, 2816, 704, 21)
	mod, err := cx.LoadModule(moePTX)
	if err != nil {
		t.Fatal(err)
	}
	smod, err := cx.LoadModule(moeSplitPTX)
	if err != nil {
		t.Fatal(err)
	}
	wacc, err := mod.Function("gemv_w4a8_moe_wacc")
	if err != nil {
		t.Fatal(err)
	}
	resFn, err := smod.Function("gemv_w4a8_moe_res")
	if err != nil {
		t.Fatal(err)
	}
	comb, err := smod.Function("moe_combine")
	if err != nil {
		t.Fatal(err)
	}
	stream := mustStream(t, cx)
	rng := rand.New(rand.NewPCG(31, 32))
	idx := make([]uint32, k)
	wgt := make([]float32, k)
	for j := range idx {
		idx[j] = uint32(rng.IntN(f.nE))
		wgt[j] = float32(rng.Float64())
	}
	dst0 := make([]float32, f.n)
	for r := range dst0 {
		dst0[r] = float32(rng.NormFloat64())
	}
	dW := mustAlloc[uint32](t, cx, len(f.h.wpk))
	dGs := mustAlloc[uint16](t, cx, len(f.h.ws16))
	dA := mustAlloc[int32](t, cx, len(f.act))
	dAs := mustAlloc[float32](t, cx, 1)
	dIdx := mustAlloc[uint32](t, cx, k)
	dWgt := mustAlloc[float32](t, cx, k)
	dDst := mustAlloc[float32](t, cx, f.n)
	dRes := mustAlloc[float32](t, cx, k*f.n)
	for _, b := range []interface{ Close() error }{dW, dGs, dA, dAs, dIdx, dWgt, dDst, dRes} {
		defer b.Close()
	}
	for _, e := range []error{gc.CopyHtoD(bg, dW, f.h.wpk), gc.CopyHtoD(bg, dGs, f.h.ws16), gc.CopyHtoD(bg, dA, f.act),
		gc.CopyHtoD(bg, dAs, []float32{f.aScale}), gc.CopyHtoD(bg, dIdx, idx), gc.CopyHtoD(bg, dWgt, wgt),
		gc.CopyHtoD(bg, dDst, dst0)} {
		if e != nil {
			t.Fatal(e)
		}
	}
	cfg := gc.LaunchConfig{GridX: uint32((f.n + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}
	for j := 0; j < k; j++ {
		if err := wacc.LaunchOn(bg, stream, cfg, gc.Arg(dW), gc.Arg(dA), gc.Arg(dGs), gc.Arg(dAs), gc.Arg(dIdx), gc.Arg(dWgt),
			gc.ArgValue(int32(j)), gc.ArgValue(int32(f.n)), gc.ArgValue(int32(f.n)), gc.ArgValue(int32(f.kw)),
			gc.ArgValue(int32(f.kg)), gc.Arg(dDst)); err != nil {
			t.Fatal(err)
		}
	}
	if err := stream.Synchronize(bg); err != nil {
		t.Fatal(err)
	}
	want := make([]float32, f.n)
	if err := gc.CopyDtoH(bg, want, dDst); err != nil {
		t.Fatal(err)
	}
	split := func(hostExperts map[int]bool) []float32 {
		if err := gc.CopyHtoD(bg, dDst, dst0); err != nil {
			t.Fatal(err)
		}
		for j := 0; j < k; j++ {
			if hostExperts[j] {
				continue
			}
			if err := resFn.LaunchOn(bg, stream, cfg, gc.Arg(dW), gc.Arg(dA), gc.Arg(dGs), gc.Arg(dAs), gc.Arg(dIdx),
				gc.ArgValue(int32(j)), gc.ArgValue(int32(f.n)), gc.ArgValue(int32(f.n)), gc.ArgValue(int32(f.kw)),
				gc.ArgValue(int32(f.kg)), gc.Arg(dRes)); err != nil {
				t.Fatal(err)
			}
		}
		if err := stream.Synchronize(bg); err != nil {
			t.Fatal(err)
		}
		res := make([]float32, k*f.n)
		if err := gc.CopyDtoH(bg, res, dRes); err != nil {
			t.Fatal(err)
		}
		tab := newMoEActTables(f.act)
		for j := range hostExperts {
			e := int(idx[j])
			row := res[j*f.n : (j+1)*f.n]
			tab.rows(f.h.wpk[e*f.n*f.kw:(e+1)*f.n*f.kw], f.h.ws16[e*f.n*f.kg:(e+1)*f.n*f.kg], row)
			for r := range row {
				row[r] = float32(row[r] * f.aScale)
			}
		}
		if err := gc.CopyHtoD(bg, dRes, res); err != nil {
			t.Fatal(err)
		}
		if err := comb.LaunchOn(bg, stream, gc.LaunchConfig{GridX: uint32((f.n + 255) / 256), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
			gc.Arg(dRes), gc.Arg(dWgt), gc.ArgValue(int32(k)), gc.ArgValue(int32(f.n)), gc.Arg(dDst)); err != nil {
			t.Fatal(err)
		}
		if err := stream.Synchronize(bg); err != nil {
			t.Fatal(err)
		}
		got := make([]float32, f.n)
		if err := gc.CopyDtoH(bg, got, dDst); err != nil {
			t.Fatal(err)
		}
		return got
	}
	for _, pat := range []struct {
		name string
		host map[int]bool
	}{{"all device", map[int]bool{}}, {"all host", map[int]bool{0: true, 1: true, 2: true, 3: true, 4: true, 5: true, 6: true, 7: true}},
		{"alternating", map[int]bool{1: true, 3: true, 5: true, 7: true}}, {"first and last", map[int]bool{0: true, 7: true}}} {
		got := split(pat.host)
		for r := range got {
			if math.Float32bits(got[r]) != math.Float32bits(want[r]) {
				t.Fatalf("%s: row %d is %g after res+combine, %g after the wacc chain", pat.name, r, got[r], want[r])
			}
		}
		t.Logf("%s: %d rows bit-identical to the wacc chain", pat.name, f.n)
	}
}

// TestMoECPURate_measure is G-S20j's host half: the time to compute k experts' gate/up and down rows (the 26B's geometry) on
// the pool, for several worker counts, over experts drawn from a 128-expert pool (about 430 MB, so the rows come from DRAM as
// a cold expert's do). A measurement, not a pass/fail: run it alone, under the timing lock.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags cuda -run TestMoECPURate_measure -v ./cuda/
func TestMoECPURate_measure(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("GOINFER_HEAVY_TESTS unset: a measurement")
	}
	const nE, inter, hidden = 128, 704, 2816
	rng := rand.New(rand.NewPCG(41, 42))
	guW := make([]uint32, nE*2*inter*hidden/8)
	guS := make([]uint16, nE*2*inter*hidden/32)
	dnW := make([]uint32, nE*hidden*inter/8)
	dnS := make([]uint16, nE*hidden*inter/32)
	for _, w := range [][]uint32{guW, dnW} {
		for i := range w {
			w[i] = rng.Uint32()
		}
	}
	for _, s := range [][]uint16{guS, dnS} {
		for i := range s {
			s[i] = f32ToF16Test(float32(rng.NormFloat64() * 0.01))
		}
	}
	aGU := make([]int32, hidden/4)
	aDn := make([]int32, inter/4)
	for i := range aGU {
		aGU[i] = int32(rng.Uint32())
	}
	for i := range aDn {
		aDn[i] = int32(rng.Uint32())
	}
	t.Logf("AVX2 kernel: %v; GOMAXPROCS %d", moeCPUAsm, runtime.GOMAXPROCS(0))
	for _, workers := range []int{1, 2, 4, 6, 8, 12, 15} {
		p := newMoECPUPool(workers)
		for _, k := range []int{1, 2, 4} {
			const reps = 60
			var guT, dnT time.Duration
			for rep := 0; rep < reps; rep++ {
				tasks := make([]moeCPUTask, k)
				tabGU := newMoEActTables(aGU)
				for j := range tasks {
					e := rng.IntN(nE)
					tasks[j] = moeCPUTask{tab: &tabGU, w: guW[e*2*inter*hidden/8 : (e+1)*2*inter*hidden/8],
						s: guS[e*2*inter*hidden/32 : (e+1)*2*inter*hidden/32], out: make([]float32, 2*inter)}
				}
				t0 := time.Now()
				p.run(tasks)
				guT += time.Since(t0)
				tabDn := newMoEActTables(aDn)
				for j := range tasks {
					e := rng.IntN(nE)
					tasks[j] = moeCPUTask{tab: &tabDn, w: dnW[e*hidden*inter/8 : (e+1)*hidden*inter/8],
						s: dnS[e*hidden*inter/32 : (e+1)*hidden*inter/32], out: make([]float32, hidden)}
				}
				t0 = time.Now()
				p.run(tasks)
				dnT += time.Since(t0)
			}
			gu, dn := guT/reps, dnT/reps
			mb := float64(k) * 3.345
			t.Logf("workers %2d, %d experts: gate/up %6.1f µs, down %6.1f µs, total %6.1f µs (%.0f µs/expert, %.1f GB/s)",
				workers, k, float64(gu.Nanoseconds())/1e3, float64(dn.Nanoseconds())/1e3, float64((gu+dn).Nanoseconds())/1e3,
				float64((gu+dn).Nanoseconds())/1e3/float64(k), mb/1e3/((gu + dn).Seconds()))
		}
		p.close()
	}
}

// TestMoESplitRoundTrip_measure is G-S20j's device half: what one layer's host round trips cost with no host compute, two
// host experts, an idle main queue. Stage: moe_act_to_host on the main queue, record, host waits. GLU: glu_quant twice on a
// second queue reading the host's gate/up products from mapped memory and writing the int8 result back into it, record, host
// waits. Return: the two products uploaded to the device on the second queue, the main queue waits on that, moe_combine,
// host waits. A measurement: run it alone, under the timing lock.
func TestMoESplitRoundTrip_measure(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("GOINFER_HEAVY_TESTS unset: a measurement")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	defer dev.ReleaseAll()
	const hidden, inter, k = 2816, 704, 2
	main, side := dev.NewCommandQueue(), dev.NewCommandQueue()
	glmod, err := dev.CompileLibrary(gluePTX)
	if err != nil {
		t.Fatal(err)
	}
	smod, err := dev.CompileLibrary(moeSplitPTX)
	if err != nil {
		t.Fatal(err)
	}
	glu, err := dev.NewComputePipeline(glmod, "glu_quant")
	if err != nil {
		t.Fatal(err)
	}
	stage, err := dev.NewComputePipeline(smod, "moe_act_to_host")
	if err != nil {
		t.Fatal(err)
	}
	comb, err := dev.NewComputePipeline(smod, "moe_combine")
	if err != nil {
		t.Fatal(err)
	}
	mapped := func(n int) *gpu.MappedHostBuffer {
		m, err := dev.NewMappedHostBuffer(n)
		if err != nil {
			t.Fatal(err)
		}
		return m
	}
	hMq, hMs, hGU, hQ, hSc, hRes := mapped(hidden), mapped(4), mapped(k*2*inter*4), mapped(k*inter), mapped(k*4), mapped(k*hidden*4)
	for _, m := range []*gpu.MappedHostBuffer{hMq, hMs, hGU, hQ, hSc, hRes} {
		defer m.Close()
	}
	dMq := gpu.NewBufferLenOf[int32](dev, hidden/4)
	dMs := gpu.NewBufferLenOf[float32](dev, 1)
	dScr := gpu.NewBufferLenOf[float32](dev, k*inter)
	dRes := gpu.NewBufferLenOf[float32](dev, 8*hidden)
	dWgt := gpu.NewBufferLenOf[float32](dev, 8)
	dDst := gpu.NewBufferLenOf[float32](dev, hidden)
	evMq, _ := dev.NewEvent()
	evGlu, _ := dev.NewEvent()
	evRes, _ := dev.NewEvent()
	evDone, _ := dev.NewEvent()
	const iters = 2000
	var st, gl, rt []time.Duration
	for it := 0; it < iters; it++ {
		t0 := time.Now()
		if err := main.Launch(stage, LaunchConfig{GridX: uint32((hidden/4 + 255) / 256), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
			Arg(dMq), Arg(dMs), gpu.ArgValue(int32(hidden/4)), Arg(hMq.Buffer()), Arg(hMs.Buffer())); err != nil {
			t.Fatal(err)
		}
		if err := main.Record(evMq); err != nil {
			t.Fatal(err)
		}
		if err := evMq.Sync(); err != nil {
			t.Fatal(err)
		}
		t1 := time.Now()
		for j := 0; j < k; j++ {
			gu := hGU.Buffer().At(j * 2 * inter * 4)
			if err := side.Launch(glu, onecfg(glueQuantThreads, glueQuantThreads*4), Arg(gu), Arg(gu), gpu.ArgValue(int32(0)),
				gpu.ArgValue(int32(inter)), gpu.ArgValue(int32(inter)), gpu.ArgValue(int32(0)), Arg(hQ.Buffer().At(j*inter)),
				Arg(hSc.Buffer().At(j*4)), Arg(dScr.At(j*inter*4))); err != nil {
				t.Fatal(err)
			}
		}
		if err := side.Record(evGlu); err != nil {
			t.Fatal(err)
		}
		if err := evGlu.Sync(); err != nil {
			t.Fatal(err)
		}
		t2 := time.Now()
		for j := 0; j < k; j++ {
			if err := side.UploadAsyncAtFrom(dRes.At(j*hidden*4), hRes, j*hidden*4, hidden*4); err != nil {
				t.Fatal(err)
			}
		}
		if err := side.Record(evRes); err != nil {
			t.Fatal(err)
		}
		if err := main.Wait(evRes); err != nil {
			t.Fatal(err)
		}
		if err := main.Launch(comb, LaunchConfig{GridX: uint32((hidden + 255) / 256), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
			Arg(dRes), Arg(dWgt), gpu.ArgValue(int32(8)), gpu.ArgValue(int32(hidden)), Arg(dDst)); err != nil {
			t.Fatal(err)
		}
		if err := main.Record(evDone); err != nil {
			t.Fatal(err)
		}
		if err := evDone.Sync(); err != nil {
			t.Fatal(err)
		}
		t3 := time.Now()
		if it >= 100 { // the first launches JIT and warm the driver
			st, gl, rt = append(st, t1.Sub(t0)), append(gl, t2.Sub(t1)), append(rt, t3.Sub(t2))
		}
	}
	med := func(d []time.Duration) float64 {
		s := append([]time.Duration(nil), d...)
		slices.Sort(s)
		return float64(s[len(s)/2].Nanoseconds()) / 1e3
	}
	p90 := func(d []time.Duration) float64 {
		s := append([]time.Duration(nil), d...)
		slices.Sort(s)
		return float64(s[len(s)*9/10].Nanoseconds()) / 1e3
	}
	t.Logf("per layer, %d host experts, no host compute, %d iterations: stage %.1f µs (p90 %.1f), GLU round trip %.1f µs (p90 %.1f), "+
		"return and combine %.1f µs (p90 %.1f); total median %.1f µs",
		k, len(st), med(st), p90(st), med(gl), p90(gl), med(rt), p90(rt), med(st)+med(gl)+med(rt))
}
