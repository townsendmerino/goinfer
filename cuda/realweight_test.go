//go:build cuda

package cuda

import (
	"context"
	"math"
	"os"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestRealWeightGemvParity is step 1 of B: load the REAL qwen2.5-coder-1.5b q4_k_m
// checkpoint, extract a real int4 projection via the same seam BuildResident will use
// (m.Weights().Layers[l].GateProj → WeightMat.Int4()), run our W4A8 GEMV on it, and
// verify it matches goinfer's canonical dequant-matmul (linalg.DequantizeRowInt4) —
// correctness on REAL weights, the foundation for the argmax-parity gate. This proves
// the extraction + packing + kernel math are right before wiring the full forward.
func TestRealWeightGemvParity(t *testing.T) {
	gguf := os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(gguf); err != nil {
		t.Skipf("no model: %v", err)
	}
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	dev, err := gc.GetDevice(0)
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	ctx, _ := dev.Primary()
	defer ctx.Close()
	bg := context.Background()

	m, err := decoder.Load(gguf, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load int4: %v", err)
	}
	defer m.Close()
	w := m.Weights()
	gate := &w.Layers[0].GateProj
	if gate.Kind() != "int4" {
		t.Skipf("layer0 gate is %q, not int4 (Q4_K may not map to group-int4 here)", gate.Kind())
	}
	N, K := gate.Rows(), gate.Cols()
	q4, scales, group, _ := gate.Int4()
	t.Logf("real GateProj: [%d×%d] int4 group=%d, %d q4 bytes, %d scales", N, K, group, len(q4), len(scales))
	if K%32 != 0 {
		t.Skipf("K=%d not mult of 32 (padding path not covered in step 1)", K)
	}

	// a real-ish int8 activation + scale (values irrelevant to weight-correctness; same on both sides)
	Kdiv4 := K / 4
	a := make([]int32, Kdiv4)
	af := make([]float32, K)
	const aS = float32(0.021)
	var s uint32 = 7
	for i := 0; i < K; i++ {
		s = s*1664525 + 1013904223
		v := int8(s >> 24)
		af[i] = float32(v) * aS
		a[i/4] |= (int32(v) & 0xff) << (8 * (i % 4))
	}

	// CPU reference: goinfer's own dequant per row, then dot with af.
	ref := make([]float32, N)
	rowBytes := (K + 1) / 2
	wf := make([]float32, K)
	for n := 0; n < N; n++ {
		linalg.DequantizeRowInt4(q4[n*rowBytes:(n+1)*rowBytes], scales[n*(K/group):(n+1)*(K/group)], group, K, wf)
		var acc float64
		for k := 0; k < K; k++ {
			acc += float64(wf[k]) * float64(af[k])
		}
		ref[n] = float32(acc)
	}

	// GPU: our W4A8 on the SAME extracted bytes; convert f32 group scales → f16 (as UploadW4A8Packed does).
	gs := make([]uint16, len(scales))
	for i, v := range scales {
		gs[i] = f32tof16(v)
	}
	wpk := make([]uint32, N*(K/8))
	for i := 0; i < len(wpk); i++ {
		b := q4[i*4 : i*4+4]
		wpk[i] = uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
	}
	dW, _ := gc.Alloc[uint32](ctx, len(wpk))
	dGs, _ := gc.Alloc[uint16](ctx, len(gs))
	dA, _ := gc.Alloc[int32](ctx, len(a))
	dOut, _ := gc.Alloc[float32](ctx, N)
	_ = gc.CopyHtoD(bg, dW, wpk)
	_ = gc.CopyHtoD(bg, dGs, gs)
	_ = gc.CopyHtoD(bg, dA, a)
	mod, _ := ctx.LoadModule(gemvW4A8PTX)
	fn, _ := mod.Function("gemv_w4a8")
	stream, _ := ctx.NewStream()
	_ = fn.LaunchOn(bg, stream, gc.LaunchConfig{GridX: uint32((N + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
		gc.Arg(dW), gc.Arg(dA), gc.Arg(dGs), gc.ArgValue(aS), gc.ArgValue(int32(N)), gc.ArgValue(int32(K/8)), gc.ArgValue(int32(K/32)), gc.Arg(dOut))
	_ = stream.Synchronize(bg)
	got := make([]float32, N)
	_ = gc.CopyDtoH(bg, got, dOut)

	var d, ng, nr float64
	for n := 0; n < N; n++ {
		d += float64(got[n]) * float64(ref[n])
		ng += float64(got[n]) * float64(got[n])
		nr += float64(ref[n]) * float64(ref[n])
	}
	cos := d / (math.Sqrt(ng)*math.Sqrt(nr) + 1e-30)
	t.Logf("REAL-weight W4A8 GEMV vs goinfer dequant-matmul: cosine %.6f", cos)
	if cos < 0.999 {
		t.Fatalf("real-weight GEMV incorrect (cosine %.6f) — extraction/packing/kernel mismatch on real data", cos)
	}
}
