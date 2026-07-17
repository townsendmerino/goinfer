//go:build cuda

package cuda

import (
	"context"
	"math"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
)

// TestE2EDecodeInt4 is the matching-quant (int4) apples-to-apples vs Ollama-on-q4_k_m
// (fresh ~149 tok/s on this box, v0.5.7): the full per-token decode with the W4A8 GEMV
// (cosine-validated) + the same glue. Same shippable config (driver-JIT, executor hop,
// CGO_ENABLED=0). NOTE our W4A8 is a naive nibble-unpack (compute-bound, 43% peak) —
// int4's fewer bytes don't help us as they help tuned llama.cpp; that is the finding.
func TestE2EDecodeInt4(t *testing.T) {
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

	gmod, _ := ctx.LoadModule(gemvW4A8PTX)
	glmod, _ := ctx.LoadModule(gluePTX)
	gemv, err := gmod.Function("gemv_w4a8")
	if err != nil {
		t.Fatalf("gemv_w4a8: %v", err)
	}
	fRms, _ := glmod.Function("rmsnorm_quant")
	fQuant, _ := glmod.Function("quant_vec")
	fRope, _ := glmod.Function("rope")
	fAttn, _ := glmod.Function("attention")
	fSwiglu, _ := glmod.Function("glu_quant")
	fResid, _ := glmod.Function("residual")
	fArgmax, _ := glmod.Function("argmax_reduce")
	stream, _ := ctx.NewStream()

	const H, I, nH, nKV, hd, vocab, nLayers = 1536, 8960, 12, 2, 128, 151936, 28
	const qDim, kvDim, qkvR = nH * hd, nKV * hd, nH*hd + 2*nKV*hd
	const half, pos = hd / 2, 128
	const nKeys = pos + 1
	const scale = float32(1.0 / 11.313708)

	ai32 := func(n int) *gc.Buffer[int32] { b, _ := gc.Alloc[int32](ctx, n); return b }
	au32 := func(n int) *gc.Buffer[uint32] { b, _ := gc.Alloc[uint32](ctx, n); return b }
	au16 := func(n int) *gc.Buffer[uint16] { b, _ := gc.Alloc[uint16](ctx, n); return b }
	af := func(n int, v float32) *gc.Buffer[float32] {
		b, _ := gc.Alloc[float32](ctx, n)
		h := make([]float32, n)
		for i := range h {
			h[i] = v
		}
		_ = gc.CopyHtoD(bg, b, h)
		return b
	}
	// W4A8 weight: [N × K/8] uint32 nibbles + [N × K/32] f16 group scales
	type w4 struct {
		W  *gc.Buffer[uint32]
		gs *gc.Buffer[uint16]
	}
	mkw := func(N, K int) w4 { return w4{au32(N * (K / 8)), au16(N * (K / 32))} }
	Wqkv, Wo, Wg, Wu, Wd, Wlm := mkw(qkvR, H), mkw(H, H), mkw(I, H), mkw(I, H), mkw(H, I), mkw(vocab, H)

	inv := make([]float32, half)
	for d := range inv {
		inv[d] = float32(1.0 / math.Pow(1e6, float64(2*d)/float64(hd)))
	}
	invF, _ := gc.Alloc[float32](ctx, half)
	_ = gc.CopyHtoD(bg, invF, inv)
	kc, vc := af(nKeys*kvDim, 0.01), af(nKeys*kvDim, 0.01)
	x := af(H, 0.1)
	aq := ai32(H / 4)
	qkv, cctx, oOut, gO, uO, dOut, logits := af(qkvR, 0), af(qDim, 0), af(H, 0), af(I, 0), af(I, 0), af(H, 0), af(vocab, 0)
	cq, mq, dq, dScr := ai32(qDim/4), ai32(H/4), ai32(I/4), af(I, 0)
	sc1 := af(1, 0.02)
	outIdx := ai32(1)

	L := func(f *gc.Function, cfg gc.LaunchConfig, args ...gc.KernelArg) {
		if err := f.LaunchOn(bg, stream, cfg, args...); err != nil {
			t.Fatalf("launch: %v", err)
		}
	}
	cfg1D := func(n, b int) gc.LaunchConfig {
		return gc.LaunchConfig{GridX: uint32((n + b - 1) / b), GridY: 1, GridZ: 1, BlockX: uint32(b), BlockY: 1, BlockZ: 1}
	}
	one := func(b, sh int) gc.LaunchConfig {
		return gc.LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: uint32(b), BlockY: 1, BlockZ: 1, SharedMemBytes: uint32(sh)}
	}
	gemvW4 := func(w w4, a *gc.Buffer[int32], N, K int, dst *gc.Buffer[float32]) {
		L(gemv, gc.LaunchConfig{GridX: uint32((N + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
			gc.Arg(w.W), gc.Arg(a), gc.Arg(w.gs), gc.ArgValue(float32(0.02)), gc.ArgValue(int32(N)), gc.ArgValue(int32(K/8)), gc.ArgValue(int32(K/32)), gc.Arg(dst))
	}
	rms := (H + 256) * 4
	token := func() {
		for l := 0; l < nLayers; l++ {
			L(fRms, one(256, rms), gc.Arg(x), gc.Arg(x), gc.ArgValue(int32(H)), gc.ArgValue(float32(1e-6)), gc.ArgValue(int32(0)), gc.Arg(aq), gc.Arg(sc1))
			gemvW4(Wqkv, aq, qkvR, H, qkv)
			L(fRope, cfg1D(nH*half, 256), gc.Arg(qkv), gc.Arg(invF), gc.ArgValue(int32(nH)), gc.ArgValue(int32(hd)), gc.ArgValue(int32(pos)))
			L(fAttn, gc.LaunchConfig{GridX: uint32(nH), GridY: 1, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((nKeys + 128) * 4)},
				gc.Arg(qkv), gc.Arg(kc), gc.Arg(vc), gc.ArgValue(int32(nH)), gc.ArgValue(int32(nKV)), gc.ArgValue(int32(hd)), gc.ArgValue(int32(nKeys)), gc.ArgValue(scale), gc.ArgValue(int32(0)), gc.Arg(cctx))
			L(fQuant, one(256, 256*4), gc.Arg(cctx), gc.ArgValue(int32(qDim)), gc.Arg(cq), gc.Arg(sc1))
			gemvW4(Wo, cq, H, qDim, oOut)
			L(fResid, cfg1D(H, 256), gc.Arg(x), gc.Arg(oOut), gc.ArgValue(int32(H)))
			L(fRms, one(256, rms), gc.Arg(x), gc.Arg(x), gc.ArgValue(int32(H)), gc.ArgValue(float32(1e-6)), gc.ArgValue(int32(0)), gc.Arg(mq), gc.Arg(sc1))
			gemvW4(Wg, mq, I, H, gO)
			gemvW4(Wu, mq, I, H, uO)
			L(fSwiglu, one(256, 256*4), gc.Arg(gO), gc.Arg(uO), gc.ArgValue(int32(I)), gc.ArgValue(int32(1)), gc.Arg(dq), gc.Arg(sc1), gc.Arg(dScr))
			gemvW4(Wd, dq, H, I, dOut)
			L(fResid, cfg1D(H, 256), gc.Arg(x), gc.Arg(dOut), gc.ArgValue(int32(H)))
		}
		L(fRms, one(256, rms), gc.Arg(x), gc.Arg(x), gc.ArgValue(int32(H)), gc.ArgValue(float32(1e-6)), gc.ArgValue(int32(0)), gc.Arg(aq), gc.Arg(sc1))
		gemvW4(Wlm, aq, vocab, H, logits)
		L(fArgmax, one(1024, 1024*8), gc.Arg(logits), gc.ArgValue(int32(vocab)), gc.Arg(outIdx), gc.Arg(sc1))
	}
	token()
	_ = stream.Synchronize(bg)

	best := 1e18
	for r := 0; r < 8; r++ {
		s, _ := ctx.NewEvent()
		e, _ := ctx.NewEvent()
		_ = s.Record(stream)
		const it = 5
		for i := 0; i < it; i++ {
			token()
		}
		_ = e.Record(stream)
		_ = stream.Synchronize(bg)
		el, _ := s.Elapsed(e)
		if ms := float64(el.Microseconds()) / 1000 / it; ms < best {
			best = ms
		}
	}
	tps := 1000.0 / best
	const ollamaQ4, webgpu = 149.0, 111.6
	t.Logf("E2E cgo-free CUDA decode INT4 (W4A8, qwen2.5-1.5b, pos=%d): %.2f ms | %.0f tok/s", pos, best, tps)
	t.Logf("  apples-to-apples (int4, same box, same q4_k_m GGUF): Ollama-CUDA v0.5.7 = 149 tok/s → ours is %.2fx", tps/ollamaQ4)
	t.Logf("  vs WebGPU int8 %.0f. NOTE our W4A8 is naive/compute-bound (43%% peak) — a tuned int4 kernel would be faster", webgpu)
}
