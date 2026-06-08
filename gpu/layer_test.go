//go:build gpu

package gpu

import (
	"math"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/linalg"
)

func cpuMLP(x, rmsW []float32, gBQ, uBQ, dBQ []int8, gS, uS, dS []float32, H, I int, eps float32, addOne bool) []float32 {
	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	inv := float32(1.0 / math.Sqrt(ss/float64(H)+float64(eps)))
	xn := make([]float32, H)
	for i := range xn {
		w := rmsW[i]
		if addOne {
			w += 1
		}
		xn[i] = x[i] * inv * w
	}
	gate := make([]float32, I)
	linalg.MatmulBTW8A8(xn, gBQ, gS, gate, 1, H, I)
	up := make([]float32, I)
	linalg.MatmulBTW8A8(xn, uBQ, uS, up, 1, H, I)
	mid := make([]float32, I)
	for i := range mid {
		g := gate[i]
		mid[i] = (g / (1 + float32(math.Exp(float64(-g))))) * up[i]
	}
	down := make([]float32, H)
	linalg.MatmulBTW8A8(mid, dBQ, dS, down, 1, I, H)
	out := make([]float32, H)
	for i := range out {
		out[i] = x[i] + down[i]
	}
	return out
}

// mlpFixture builds a small MLP's weights + resident GPU handles.
type mlpFixture struct {
	x, rmsW              []float32
	gBQ, uBQ, dBQ        []int8
	gS, uS, dS           []float32
	rmsWDev              *DeviceBuffer
	gateRM, upRM, downRM *ResidentW8A8
	H, I                 int
	eps                  float32
}

func newMLPFixture(t *testing.T, ctx *Context, H, I int) *mlpFixture {
	f := &mlpFixture{H: H, I: I, eps: 1e-6}
	f.x = randMat(H, 7)
	f.rmsW = randMat(H, 8)
	gW := randMat(I*H, 1)
	uW := randMat(I*H, 2)
	dW := randMat(H*I, 3)
	f.gBQ, f.gS = linalg.QuantizeRowsInt8(gW, I, H)
	f.uBQ, f.uS = linalg.QuantizeRowsInt8(uW, I, H)
	f.dBQ, f.dS = linalg.QuantizeRowsInt8(dW, H, I)
	var err error
	if f.rmsWDev, err = ctx.UploadF32(f.rmsW); err != nil {
		t.Fatalf("UploadF32: %v", err)
	}
	if f.gateRM, err = ctx.UploadW8A8(f.gBQ, f.gS, I, H); err != nil {
		t.Fatalf("UploadW8A8 gate: %v", err)
	}
	if f.upRM, err = ctx.UploadW8A8(f.uBQ, f.uS, I, H); err != nil {
		t.Fatalf("UploadW8A8 up: %v", err)
	}
	if f.downRM, err = ctx.UploadW8A8(f.dBQ, f.dS, H, I); err != nil {
		t.Fatalf("UploadW8A8 down: %v", err)
	}
	return f
}

// TestFusedMLP_parity checks the whole on-device MLP block against the CPU MLP.
func TestFusedMLP_parity(t *testing.T) {
	ctx, err := New()
	if err != nil {
		t.Skipf("no GPU adapter: %v", err)
	}
	defer ctx.Close()
	f := newMLPFixture(t, ctx, 1536, 4096)

	got, err := ctx.FusedMLP(f.x, f.rmsWDev, f.gateRM, f.upRM, f.downRM, f.eps, false)
	if err != nil {
		t.Fatalf("FusedMLP: %v", err)
	}
	ref := cpuMLP(f.x, f.rmsW, f.gBQ, f.uBQ, f.dBQ, f.gS, f.uS, f.dS, f.H, f.I, f.eps, false)
	cos, maxAbs := cosine(got, ref)
	t.Logf("FusedMLP parity: cosine=%.6f maxAbs=%.3e", cos, maxAbs)
	if cos < 0.999 {
		t.Errorf("FusedMLP diverges from CPU MLP: cosine=%.6f", cos)
	}
}

// TestFusedMLP_microbench compares the fully-fused MLP (one sync) to the current
// decoder-style staging (gate/up batch + down = 2 GPU syncs, CPU norm/SwiGLU/
// residual) and to the full CPU MLP. Logs; run -v.
func TestFusedMLP_microbench(t *testing.T) {
	if testing.Short() {
		t.Skip("microbench")
	}
	ctx, err := New()
	if err != nil {
		t.Skipf("no GPU adapter: %v", err)
	}
	defer ctx.Close()
	const H, I, iters = 1536, 8960, 100
	f := newMLPFixture(t, ctx, H, I)
	gateUp := []*ResidentW8A8{f.gateRM, f.upRM}
	guRun, _ := ctx.NewGEMVRunner(f.gateRM) // not used directly; staged uses BatchGEMV + a down runner
	_ = guRun
	downRun, _ := ctx.NewGEMVRunner(f.downRM)
	defer downRun.Release()

	// staged: CPU rmsnorm → BatchGEMV(gate,up) [1 sync] → CPU swiglu → down runner
	// [1 sync] → CPU residual. Mirrors the post-fused-batch decoder MLP.
	staged := func() {
		var ss float64
		for _, v := range f.x {
			ss += float64(v) * float64(v)
		}
		inv := float32(1.0 / math.Sqrt(ss/float64(H)+float64(f.eps)))
		xn := make([]float32, H)
		for i := range xn {
			xn[i] = f.x[i] * inv * f.rmsW[i]
		}
		aq, as := linalg.QuantizeRowsInt8(xn, 1, H)
		gu, _ := ctx.BatchGEMV(aq, as[0], gateUp)
		mid := make([]float32, I)
		for i := range mid {
			g := gu[0][i]
			mid[i] = (g / (1 + float32(math.Exp(float64(-g))))) * gu[1][i]
		}
		maq, mas := linalg.QuantizeRowsInt8(mid, 1, I)
		down, _ := downRun.Run(maq, mas[0])
		_ = down // + CPU residual (cheap, omitted from the hot compare)
	}

	// warm
	ctx.FusedMLP(f.x, f.rmsWDev, f.gateRM, f.upRM, f.downRM, f.eps, false)
	staged()

	t0 := time.Now()
	for i := 0; i < iters; i++ {
		if _, err := ctx.FusedMLP(f.x, f.rmsWDev, f.gateRM, f.upRM, f.downRM, f.eps, false); err != nil {
			t.Fatalf("FusedMLP: %v", err)
		}
	}
	fused := time.Since(t0) / iters
	t1 := time.Now()
	for i := 0; i < iters; i++ {
		staged()
	}
	stg := time.Since(t1) / iters
	t2 := time.Now()
	for i := 0; i < iters; i++ {
		cpuMLP(f.x, f.rmsW, f.gBQ, f.uBQ, f.dBQ, f.gS, f.uS, f.dS, H, I, f.eps, false)
	}
	cpu := time.Since(t2) / iters

	t.Logf("MLP block (H=%d I=%d)  |  GPU fused (1 sync) %.3f ms  |  GPU staged (2 syncs+CPU glue) %.3f ms  |  CPU %.3f ms",
		H, I, ms(fused), ms(stg), ms(cpu))
	t.Logf("  → fused vs staged %.2f×  |  fused vs CPU %.2f×", float64(stg)/float64(fused), float64(cpu)/float64(fused))
}
