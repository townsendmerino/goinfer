//go:build cuda

package cuda

import (
	"context"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
)

// TestDecodeTokProjection projects decode tok/s from the REAL per-token quantized-GEMV
// workload of qwen2.5-1.5b, run as actual CUDA launches on the 2070 SUPER (so the
// per-dispatch channel-hop tax is included, not modeled). Decode is weight-streaming-
// bound, so the GEMV weight reads dominate the token; attention KV read + norms/RoPE/
// SwiGLU are elementwise and negligible by comparison. This is the UNFUSED version
// (one launch per projection) — a LOWER bound on the fused megakernel, which cuts the
// launch count — so if this clears the 145 tok/s GO bar, the megakernel does too.
// Uses the W8A8 GEMV already validated cosine-1.0 vs CPU (TestGemvW8A8Bandwidth).
// Run: CGO_ENABLED=0 go test -tags cuda -run Projection -v
func TestDecodeTokProjection(t *testing.T) {
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	dev, err := gc.GetDevice(0)
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	ctx, err := dev.Primary()
	if err != nil {
		t.Skipf("no context: %v", err)
	}
	defer ctx.Close()
	bg := context.Background()

	// qwen2.5-coder-1.5b geometry.
	const H, I, vocab, nLayers = 1536, 8960, 151936, 28
	const qkvR = 1536 + 256 // qDim(12*128) + 2*kvDim(2*128)

	mod, err := ctx.LoadModule(gemvPTX)
	if err != nil {
		t.Fatalf("LoadModule: %v", err)
	}
	fn, err := mod.Function("gemv_w8a8")
	if err != nil {
		t.Fatalf("Function: %v", err)
	}
	stream, _ := ctx.NewStream()

	// one weight buffer + scale per distinct projection (values irrelevant to bandwidth).
	type proj struct {
		N, K int
		W    *gc.Buffer[int32]
		s    *gc.Buffer[float32]
		out  *gc.Buffer[float32]
		act  *gc.Buffer[int32]
	}
	mk := func(N, K int) *proj {
		p := &proj{N: N, K: K}
		p.W, _ = gc.Alloc[int32](ctx, N*(K/4))
		p.s, _ = gc.Alloc[float32](ctx, N)
		p.out, _ = gc.Alloc[float32](ctx, N)
		p.act, _ = gc.Alloc[int32](ctx, K/4)
		return p
	}
	// the 5 per-layer projections + the LM head.
	qkv := mk(qkvR, H)
	oProj := mk(H, H)
	gate := mk(I, H)
	up := mk(I, H)
	down := mk(H, I)
	lm := mk(vocab, H)
	layer := []*proj{qkv, oProj, gate, up, down}
	defer func() {
		for _, p := range append(layer, lm) {
			p.W.Close()
			p.s.Close()
			p.out.Close()
			p.act.Close()
		}
	}()

	launch := func(p *proj) {
		const wpb = 8
		cfg := gc.LaunchConfig{
			GridX: uint32((p.N + wpb - 1) / wpb), GridY: 1, GridZ: 1,
			BlockX: wpb * 32, BlockY: 1, BlockZ: 1,
		}
		_ = fn.LaunchOn(bg, stream, cfg,
			gc.Arg(p.W), gc.Arg(p.act), gc.Arg(p.s), gc.ArgValue(float32(0.02)),
			gc.ArgValue(int32(p.N)), gc.ArgValue(int32(p.K/4)), gc.Arg(p.out))
	}
	token := func() {
		for l := 0; l < nLayers; l++ {
			for _, p := range layer {
				launch(p)
			}
		}
		launch(lm)
	}

	// warm
	token()
	_ = stream.Synchronize(bg)

	// per-token weight bytes (the streaming lower bound).
	var wbytes int64
	for _, p := range layer {
		wbytes += int64(p.N) * int64(p.K) * int64(nLayers)
	}
	wbytes += int64(lm.N) * int64(lm.K)

	bestMs := 1e18
	for r := 0; r < 8; r++ {
		start, _ := ctx.NewEvent()
		done, _ := ctx.NewEvent()
		_ = start.Record(stream)
		const iters = 5
		for i := 0; i < iters; i++ {
			token()
		}
		_ = done.Record(stream)
		_ = stream.Synchronize(bg)
		el, _ := start.Elapsed(done)
		if ms := float64(el.Microseconds()) / 1000 / iters; ms < bestMs {
			bestMs = ms
		}
	}
	tps := 1000.0 / bestMs
	gbps := float64(wbytes) / (bestMs * 1e-3) / 1e9
	const peak, webgpu, bar = 448.0, 111.6, 145.0
	dispatches := nLayers*len(layer) + 1
	t.Logf("qwen2.5-1.5b decode projection (UNFUSED CUDA GEMV chain, %d launches/token):", dispatches)
	t.Logf("  per-token: %.2f ms | %.0f tok/s | weight stream %.2f GB | %.0f GB/s = %.0f%% peak",
		bestMs, tps, float64(wbytes)/1e9, gbps, gbps/peak*100)
	t.Logf("  vs WebGPU %.0f tok/s (37%% peak) | GO bar %.0f tok/s (1.3x) — unfused is a LOWER bound (fused cuts launches)", webgpu, bar)
	if tps >= bar {
		t.Logf("  ⇒ clears the GO bar EVEN UNFUSED — the fused megakernel would clear it by more")
	} else {
		t.Logf("  ⇒ below the bar unfused; the fusion win (fewer dispatches) is what must close the gap")
	}
}
