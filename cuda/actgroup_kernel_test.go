//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"math"
	"math/rand"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/aikit/linalg"
)

// actGroupHarness loads actgroup.ptx and returns a launcher for its kernels.
func actGroupHarness(t *testing.T) (*gc.Context, *gc.Stream, func(name string, cfg gc.LaunchConfig, args ...gc.KernelArg)) {
	t.Helper()
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	ctx := dev.Context()
	mod, err := ctx.LoadModule(actGroupPTX)
	if err != nil {
		t.Fatalf("LoadModule(actGroupPTX): %v", err)
	}
	stream := mustStream(t, ctx)
	return ctx, stream, func(name string, cfg gc.LaunchConfig, args ...gc.KernelArg) {
		t.Helper()
		fn, err := mod.Function(name)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if err := fn.LaunchOn(context.Background(), stream, cfg, args...); err != nil {
			t.Fatalf("%s launch: %v", name, err)
		}
		if err := stream.Synchronize(context.Background()); err != nil {
			t.Fatalf("%s sync: %v", name, err)
		}
	}
}

func upload[T gc.Supported](t *testing.T, ctx *gc.Context, host []T) *gc.Buffer[T] {
	t.Helper()
	b := mustAlloc[T](t, ctx, len(host))
	if err := gc.CopyHtoD(context.Background(), b, host); err != nil {
		t.Fatalf("CopyHtoD: %v", err)
	}
	return b
}

func download[T gc.Supported](t *testing.T, b *gc.Buffer[T], n int) []T {
	t.Helper()
	h := make([]T, n)
	if err := gc.CopyDtoH(context.Background(), h, b); err != nil {
		t.Fatalf("CopyDtoH: %v", err)
	}
	return h
}

// quantG32Go is the host twin of actgroup.cu's quantG32: per-32 max/127, round-to-nearest-even of
// the f32 product (__float2int_rn), clamp to +-127. Codes returned unpacked.
func quantG32Go(v []float32) (codes []int8, scales []float32) {
	nG := len(v) / 32
	codes, scales = make([]int8, len(v)), make([]float32, nG)
	for g := range nG {
		var ma float32
		for _, x := range v[32*g : 32*g+32] {
			ma = float32(math.Max(float64(ma), math.Abs(float64(x))))
		}
		sc := ma / 127
		var inv float32
		if sc > 0 {
			inv = 1 / sc
		}
		scales[g] = sc
		for i := 32 * g; i < 32*g+32; i++ {
			q := math.RoundToEven(float64(v[i] * inv))
			codes[i] = int8(max(-127, min(127, q)))
		}
	}
	return codes, scales
}

func unpackI8(p []int32, n int) []int8 {
	out := make([]int8, n)
	for i := range out {
		out[i] = int8(uint32(p[i/4]) >> (8 * (i % 4)))
	}
	return out
}

func packActI8(c []int8) []int32 {
	p := make([]int32, len(c)/4)
	for i := range p {
		p[i] = int32(uint32(uint8(c[4*i])) | uint32(uint8(c[4*i+1]))<<8 | uint32(uint8(c[4*i+2]))<<16 | uint32(uint8(c[4*i+3]))<<24)
	}
	return p
}

func outlierVec(rng *rand.Rand, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(rng.NormFloat64())
	}
	v[n/3] = 400 // the Phi-3 regime: one massive outlier
	for g := 0; g < n/32; g += 7 {
		for i := 32 * g; i < 32*g+32; i++ {
			v[i] = 0 // all-zero groups: the zero-scale convention
		}
	}
	return v
}

// TestActGroupKernels_quantizers: quant_vec_g32 matches the host twin EXACTLY (max is exact and the
// rest is one f32 multiply and a round), and rmsnorm_quant_g32 / glu_quant_g32 dequantize to their
// f32 math within half a quantization step per element (their __expf / rsqrtf are not Go's).
func TestActGroupKernels_quantizers(t *testing.T) {
	ctx, _, launch := actGroupHarness(t)
	rng := rand.New(rand.NewSource(7))
	const H = 3072
	x := outlierVec(rng, H)
	one := gc.LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}

	dx, dq, ds := upload(t, ctx, x), mustAlloc[int32](t, ctx, H/4), mustAlloc[float32](t, ctx, H/32)
	launch("quant_vec_g32", one, gc.Arg(dx), gc.ArgValue(int32(H)), gc.Arg(dq), gc.Arg(ds))
	wantC, wantS := quantG32Go(x)
	gotC, gotS := unpackI8(download(t, dq, H/4), H), download(t, ds, H/32)
	for g := range wantS {
		if gotS[g] != wantS[g] {
			t.Fatalf("quant_vec_g32 scale[%d] = %v, want exactly %v", g, gotS[g], wantS[g])
		}
	}
	for i := range wantC {
		if gotC[i] != wantC[i] {
			t.Fatalf("quant_vec_g32 code[%d] = %d, want %d", i, gotC[i], wantC[i])
		}
	}

	// rmsnorm_quant_g32 vs the f32 normalize, dequantized.
	w := make([]float32, H)
	for i := range w {
		w[i] = float32(0.5 + rng.Float64())
	}
	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	rnorm := 1 / math.Sqrt(ss/H+1e-6)
	dw := upload(t, ctx, w)
	shared := one
	shared.SharedMemBytes = uint32((H + 256) * 4)
	launch("rmsnorm_quant_g32", shared, gc.Arg(dx), gc.Arg(dw), gc.ArgValue(int32(H)), gc.ArgValue(float32(1e-6)),
		gc.ArgValue(int32(0)), gc.Arg(dq), gc.Arg(ds))
	gotC, gotS = unpackI8(download(t, dq, H/4), H), download(t, ds, H/32)
	for i := range H {
		want := float64(x[i]) * float64(w[i]) * rnorm
		got := float64(gotC[i]) * float64(gotS[i/32])
		if math.Abs(got-want) > 0.51*float64(gotS[i/32])+1e-6*math.Abs(want) {
			t.Fatalf("rmsnorm_quant_g32 [%d]: dequant %v, want %v (scale %v)", i, got, want, gotS[i/32])
		}
	}

	// glu_quant_g32 (SiLU) vs silu(g)*u, dequantized.
	const I = 8960
	gv, uv := outlierVec(rng, I), outlierVec(rng, I)
	dg, du := upload(t, ctx, gv), upload(t, ctx, uv)
	dqI, dsI, dscr := mustAlloc[int32](t, ctx, I/4), mustAlloc[float32](t, ctx, I/32), mustAlloc[float32](t, ctx, I)
	launch("glu_quant_g32", one, gc.Arg(dg), gc.Arg(du), gc.ArgValue(int32(0)), gc.ArgValue(int32(0)), gc.ArgValue(int32(I)),
		gc.ArgValue(int32(1)), gc.Arg(dqI), gc.Arg(dsI), gc.Arg(dscr))
	gotC, gotS = unpackI8(download(t, dqI, I/4), I), download(t, dsI, I/32)
	for i := range I {
		g := float64(gv[i])
		want := g / (1 + math.Exp(-g)) * float64(uv[i])
		got := float64(gotC[i]) * float64(gotS[i/32])
		if math.Abs(got-want) > 0.51*float64(gotS[i/32])+1e-5*math.Abs(want) {
			t.Fatalf("glu_quant_g32 [%d]: dequant %v, want %v (scale %v)", i, got, want, gotS[i/32])
		}
	}
}

// TestActGroupKernels_gemv: gemv_w4a8_g32 and gemv_w8a8_g32 against a host sum over the SAME codes
// and scales (int4 weights packed as the resident path packs them, f16 group scales), with bias and
// accumulate, at a decode shape with a massive activation outlier.
func TestActGroupKernels_gemv(t *testing.T) {
	ctx, _, launch := actGroupHarness(t)
	rng := rand.New(rand.NewSource(11))
	const K, N = 3072, 200
	a := outlierVec(rng, K)
	aC, aS := quantG32Go(a)
	wf := make([]float32, N*K)
	for i := range wf {
		wf[i] = float32(rng.NormFloat64() * 0.05)
	}
	bias, dst0 := make([]float32, N), make([]float32, N)
	for i := range bias {
		bias[i], dst0[i] = float32(rng.NormFloat64()), float32(rng.NormFloat64())
	}
	da, das, db := upload(t, ctx, packActI8(aC)), upload(t, ctx, aS), upload(t, ctx, bias)
	cfg := gc.LaunchConfig{GridX: uint32((N + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}

	check := func(name string, got []float32, want []float64) {
		t.Helper()
		var num, den float64
		for i := range want {
			d := float64(got[i]) - want[i]
			num, den = num+d*d, den+want[i]*want[i]
		}
		if e := math.Sqrt(num / den); e > 1e-5 {
			t.Errorf("%s: rel err %.3g vs host sum", name, e)
		}
	}

	// int4: canonical nibbles -> permuteFast words, f16 group scales; the host sum uses the same
	// f16-rounded scales.
	q4, s4 := linalg.QuantizeGroupsInt4(wf, N, K, 32)
	words := make([]uint32, N*(K/8))
	for i := range words {
		b := q4[4*i : 4*i+4]
		words[i] = permuteFast(uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24)
	}
	gs := make([]uint16, len(s4))
	for i, v := range s4 {
		gs[i] = f32tof16(v)
	}
	want := make([]float64, N)
	for n := range N {
		acc := float64(dst0[n]) + float64(bias[n])
		for k := range K {
			nib := int(q4[(n*K+k)/2]>>(4*(k%2))) & 0xF
			ws := float64(f16tof32(gs[n*(K/32)+k/32]))
			acc += float64(nib-8) * ws * float64(aC[k]) * float64(aS[k/32])
		}
		want[n] = acc
	}
	dw4, dgs, dd := upload(t, ctx, words), upload(t, ctx, gs), upload(t, ctx, dst0)
	launch("gemv_w4a8_g32", cfg, gc.Arg(dw4), gc.Arg(da), gc.Arg(dgs), gc.Arg(das), gc.Arg(db),
		gc.ArgValue(int32(N)), gc.ArgValue(int32(K/8)), gc.ArgValue(int32(K/32)), gc.Arg(dd), gc.ArgValue(int32(1)))
	check("gemv_w4a8_g32", download(t, dd, N), want)

	// int8: per-row weight scale.
	q8, s8 := linalg.QuantizeRowsInt8(wf, N, K)
	for n := range N {
		acc := 0.0
		for k := range K {
			acc += float64(q8[n*K+k]) * float64(aC[k]) * float64(aS[k/32])
		}
		want[n] = float64(dst0[n]) + acc*float64(s8[n]) + float64(bias[n])
	}
	dw8, ds8, dd8 := upload(t, ctx, packI8(q8, N, K)), upload(t, ctx, s8), upload(t, ctx, dst0)
	launch("gemv_w8a8_g32", cfg, gc.Arg(dw8), gc.Arg(da), gc.Arg(ds8), gc.Arg(das), gc.Arg(db),
		gc.ArgValue(int32(N)), gc.ArgValue(int32(K/4)), gc.Arg(dd8), gc.ArgValue(int32(1)))
	check("gemv_w8a8_g32", download(t, dd8, N), want)
}
