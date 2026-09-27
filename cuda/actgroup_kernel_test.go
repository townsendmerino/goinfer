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

// checkSums: a per-32 quantizer's second half of the scale buffer is aS[g]·Σ codes of group g, the
// same f32 product the kernel forms (quantG32), so it must match exactly.
func checkSums(t *testing.T, name string, codes []int8, scales []float32, nG int) {
	t.Helper()
	for g := range nG {
		var s int32
		for _, c := range codes[32*g : 32*g+32] {
			s += int32(c)
		}
		if want := scales[g] * float32(s); scales[nG+g] != want {
			t.Fatalf("%s sum[%d] = %v, want exactly %v (scale %v, Σcodes %d)", name, g, scales[nG+g], want, scales[g], s)
		}
	}
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

	dx, dq, ds := upload(t, ctx, x), mustAlloc[int32](t, ctx, H/4), mustAlloc[float32](t, ctx, 2*H/32) // scales, then sums
	launch("quant_vec_g32", one, gc.Arg(dx), gc.ArgValue(int32(H)), gc.Arg(dq), gc.Arg(ds))
	wantC, wantS := quantG32Go(x)
	gotC, gotS := unpackI8(download(t, dq, H/4), H), download(t, ds, 2*H/32)
	for g := range wantS {
		if gotS[g] != wantS[g] {
			t.Fatalf("quant_vec_g32 scale[%d] = %v, want exactly %v", g, gotS[g], wantS[g])
		}
	}
	checkSums(t, "quant_vec_g32", gotC, gotS, H/32)
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
	gotC, gotS = unpackI8(download(t, dq, H/4), H), download(t, ds, 2*H/32)
	checkSums(t, "rmsnorm_quant_g32", gotC, gotS, H/32)
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
	dqI, dsI, dscr := mustAlloc[int32](t, ctx, I/4), mustAlloc[float32](t, ctx, 2*I/32), mustAlloc[float32](t, ctx, I)
	launch("glu_quant_g32", one, gc.Arg(dg), gc.Arg(du), gc.ArgValue(int32(0)), gc.ArgValue(int32(0)), gc.ArgValue(int32(I)),
		gc.ArgValue(int32(1)), gc.Arg(dqI), gc.Arg(dsI), gc.Arg(dscr))
	gotC, gotS = unpackI8(download(t, dqI, I/4), I), download(t, dsI, 2*I/32)
	checkSums(t, "glu_quant_g32", gotC, gotS, I/32)
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

// q4kHostRows builds rows×cols of valid Q4_K super-blocks (normal f16 d/dmin in [2^-6, 2^1), random
// scale/min/code bytes; sat: every code 15 and every 6-bit scale/min 63).
func q4kHostRows(rng *rand.Rand, rows, cols int, sat bool) []byte {
	raw := make([]byte, rows*cols/256*144)
	for b := 0; b < len(raw); b += 144 {
		for _, off := range []int{0, 2} {
			h := uint16(9+rng.Intn(7))<<10 | uint16(rng.Intn(1024))
			raw[b+off], raw[b+off+1] = byte(h), byte(h>>8)
		}
		for i := 4; i < 144; i++ {
			raw[b+i] = byte(rng.Intn(256))
			if sat {
				raw[b+i] = 0xFF
			}
		}
	}
	return raw
}

// TestActGroupKernels_gemvQ4K: gemv_q4k_g32 against a host float64 sum Σ_k w[n,k]·aq[k]·aS[k/32] over
// the same super-blocks (w from aikit's WrapQ4K Row, the exact f32 d·sc·q − dmin·m) and the same
// codes and scales, with bias and accumulate, at a decode shape with a massive activation outlier,
// random and saturated blocks. The activation buffer carries scales then sums, as quantG32 writes it.
func TestActGroupKernels_gemvQ4K(t *testing.T) {
	ctx, _, launch := actGroupHarness(t)
	rng := rand.New(rand.NewSource(13))
	const K, N = 3072, 200
	for _, sat := range []bool{false, true} {
		a := outlierVec(rng, K)
		if sat {
			for i := range a {
				a[i] = -1
			}
		}
		aC, aS := quantG32Go(a)
		nG := K / 32
		aBuf := make([]float32, 2*nG)
		copy(aBuf, aS)
		for g := range nG {
			var s int32
			for _, c := range aC[32*g : 32*g+32] {
				s += int32(c)
			}
			aBuf[nG+g] = aS[g] * float32(s)
		}
		raw := q4kHostRows(rng, N, K, sat)
		wm, err := linalg.WrapQ4K(raw, N, K)
		if err != nil {
			t.Fatal(err)
		}
		bias, dst0 := make([]float32, N), make([]float32, N)
		for i := range bias {
			bias[i], dst0[i] = float32(rng.NormFloat64()), float32(rng.NormFloat64())
		}
		want := make([]float64, N)
		row := make([]float32, K)
		for n := range N {
			wm.Row(n, row)
			acc := float64(dst0[n]) + float64(bias[n])
			for k := range K {
				acc += float64(row[k]) * float64(aC[k]) * float64(aS[k/32])
			}
			want[n] = acc
		}
		words := make([]uint32, len(raw)/4)
		for i := range words {
			words[i] = uint32(raw[4*i]) | uint32(raw[4*i+1])<<8 | uint32(raw[4*i+2])<<16 | uint32(raw[4*i+3])<<24
		}
		dw, da, das, db, dd := upload(t, ctx, words), upload(t, ctx, packActI8(aC)), upload(t, ctx, aBuf), upload(t, ctx, bias), upload(t, ctx, dst0)
		cfg := gc.LaunchConfig{GridX: uint32((N + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}
		launch("gemv_q4k_g32", cfg, gc.Arg(dw), gc.Arg(da), gc.Arg(das), gc.Arg(db),
			gc.ArgValue(int32(N)), gc.ArgValue(int32(K/256)), gc.Arg(dd), gc.ArgValue(int32(1)))
		got := download(t, dd, N)
		var num, den float64
		for i := range want {
			d := float64(got[i]) - want[i]
			num, den = num+d*d, den+want[i]*want[i]
		}
		if e := math.Sqrt(num / den); e > 1e-5 || math.IsNaN(e) {
			t.Errorf("sat=%v gemv_q4k_g32: rel err %.3g vs host sum", sat, e)
		}
	}
}

// TestActGroupKernels_fusedG32BitIdentical: fused_rms_qkv_g32 and fused_rms_gu_g32 equal the unfused
// per-32 chain (rmsnorm_quant_g32 + one gemv_*_g32 per projection) EXACTLY, with mixed weight kinds
// (q4k / int8 / int4) and biases, at several rows-per-warp. They share the row code by construction
// (actgroup.cu's row* functions), so any difference is a real defect, not rounding.
func TestActGroupKernels_fusedG32BitIdentical(t *testing.T) {
	ctx, _, launch := actGroupHarness(t)
	rng := rand.New(rand.NewSource(17))
	const H, qDim, kvDim, I = 1536, 1536, 256, 512
	for rep, xScale := range []float64{1, 0.013, 57} { // three inputs: a single rnorm could coincide
		x := outlierVec(rng, H)
		for i := range x {
			if x[i] == 0 {
				x[i] = float32(rng.NormFloat64()) // rmsnorm input: no all-zero groups needed here
			}
			x[i] *= float32(xScale)
		}
		_ = rep
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
			scW8 *gc.Buffer[float32]
			scW4 *gc.Buffer[uint16]
			bias *gc.Buffer[float32]
		}
		mk := func(kind string, N int) proj {
			p := proj{kind: kind, N: N}
			wf := make([]float32, N*H)
			for i := range wf {
				wf[i] = float32(rng.NormFloat64() * 0.05)
			}
			switch kind {
			case "q4k":
				raw := q4kHostRows(rng, N, H, false)
				words := make([]uint32, len(raw)/4)
				for i := range words {
					words[i] = uint32(raw[4*i]) | uint32(raw[4*i+1])<<8 | uint32(raw[4*i+2])<<16 | uint32(raw[4*i+3])<<24
				}
				p.W, p.sc = upload(t, ctx, words), gc.ArgDevicePtr(0)
			case "int8":
				q8, s8 := linalg.QuantizeRowsInt8(wf, N, H)
				p.W = upload(t, ctx, packI8(q8, N, H))
				p.scW8 = upload(t, ctx, s8)
				p.sc = gc.Arg(p.scW8)
			case "int4":
				q4, s4 := linalg.QuantizeGroupsInt4(wf, N, H, 32)
				words := make([]uint32, N*(H/8))
				for i := range words {
					b := q4[4*i : 4*i+4]
					words[i] = permuteFast(uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24)
				}
				gs := make([]uint16, len(s4))
				for i, v := range s4 {
					gs[i] = f32tof16(v)
				}
				p.W, p.scW4 = upload(t, ctx, words), upload(t, ctx, gs)
				p.sc = gc.Arg(p.scW4)
			}
			b := make([]float32, N)
			for i := range b {
				b[i] = float32(rng.NormFloat64())
			}
			p.bias = upload(t, ctx, b)
			return p
		}
		code := map[string]int32{"int8": 0, "int4": 1, "q4k": 2}
		cfg8 := func(N int) gc.LaunchConfig {
			return gc.LaunchConfig{GridX: uint32((N + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}
		}
		// unfused: rmsnorm_quant_g32 then one per-32 gemv per projection
		dq, das := mustAlloc[int32](t, ctx, H/4), mustAlloc[float32](t, ctx, 2*H/32)
		one := gc.LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((H + 256) * 4)}
		launch("rmsnorm_quant_g32", one, gc.Arg(dx), gc.Arg(dnw), gc.ArgValue(int32(H)), gc.ArgValue(float32(1e-6)),
			gc.ArgValue(int32(0)), gc.Arg(dq), gc.Arg(das))
		gemv := func(p proj, bias gc.KernelArg) []float32 {
			out := mustAlloc[float32](t, ctx, p.N)
			switch p.kind {
			case "q4k":
				launch("gemv_q4k_g32", cfg8(p.N), gc.Arg(p.W), gc.Arg(dq), gc.Arg(das), bias,
					gc.ArgValue(int32(p.N)), gc.ArgValue(int32(H/256)), gc.Arg(out), gc.ArgValue(int32(0)))
			case "int8":
				launch("gemv_w8a8_g32", cfg8(p.N), gc.Arg(p.W), gc.Arg(dq), gc.Arg(p.scW8), gc.Arg(das), bias,
					gc.ArgValue(int32(p.N)), gc.ArgValue(int32(H/4)), gc.Arg(out), gc.ArgValue(int32(0)))
			case "int4":
				launch("gemv_w4a8_g32", cfg8(p.N), gc.Arg(p.W), gc.Arg(dq), gc.Arg(p.scW4), gc.Arg(das), bias,
					gc.ArgValue(int32(p.N)), gc.ArgValue(int32(H/8)), gc.ArgValue(int32(H/32)), gc.Arg(out), gc.ArgValue(int32(0)))
			}
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

		q, k, v := mk("q4k", qDim), mk("int8", kvDim), mk("int4", kvDim)
		wq, wk, wv := gemv(q, gc.Arg(q.bias)), gemv(k, gc.Arg(k.bias)), gemv(v, gc.Arg(v.bias))
		for _, rpw := range []int{1, 2, 4} {
			nrows := qDim + 2*kvDim
			cfg := gc.LaunchConfig{GridX: uint32((nrows + 8*rpw - 1) / (8 * rpw)), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: shmem}
			oq, ok, ov := mustAlloc[float32](t, ctx, qDim), mustAlloc[float32](t, ctx, kvDim), mustAlloc[float32](t, ctx, kvDim)
			launch("fused_rms_qkv_g32", cfg, gc.Arg(dx), gc.Arg(dnw), gc.ArgValue(int32(H)), gc.ArgValue(float32(1e-6)), gc.ArgValue(int32(0)),
				gc.Arg(q.W), q.sc, gc.Arg(q.bias), gc.ArgValue(code[q.kind]),
				gc.Arg(k.W), k.sc, gc.Arg(k.bias), gc.ArgValue(code[k.kind]),
				gc.Arg(v.W), v.sc, gc.Arg(v.bias), gc.ArgValue(code[v.kind]),
				gc.ArgValue(int32(qDim)), gc.ArgValue(int32(kvDim)), gc.ArgValue(int32(rpw)), gc.Arg(oq), gc.Arg(ok), gc.Arg(ov))
			same("q", download(t, oq, qDim), wq)
			same("k", download(t, ok, kvDim), wk)
			same("v", download(t, ov, kvDim), wv)
		}

		g, u := mk("q4k", I), mk("int8", I)
		wg, wu := gemv(g, gc.ArgDevicePtr(0)), gemv(u, gc.ArgDevicePtr(0))
		for _, rpw := range []int{1, 8} {
			cfg := gc.LaunchConfig{GridX: uint32((2*I + 8*rpw - 1) / (8 * rpw)), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: shmem}
			og, ou := mustAlloc[float32](t, ctx, I), mustAlloc[float32](t, ctx, I)
			launch("fused_rms_gu_g32", cfg, gc.Arg(dx), gc.Arg(dnw), gc.ArgValue(int32(H)), gc.ArgValue(float32(1e-6)), gc.ArgValue(int32(0)),
				gc.Arg(g.W), g.sc, gc.ArgValue(code[g.kind]),
				gc.Arg(u.W), u.sc, gc.ArgValue(code[u.kind]),
				gc.ArgValue(int32(I)), gc.ArgValue(int32(rpw)), gc.Arg(og), gc.Arg(ou))
			same("gate", download(t, og, I), wg)
			same("up", download(t, ou, I), wu)
		}
	}
}
