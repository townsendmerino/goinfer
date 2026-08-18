//go:build cuda && goinfer_testhooks

// Attention-sink kernel gate (gpt-oss residency, step 1 of 2).
//
// The sink is a learned per-head logit with NO key and NO value. It joins the softmax MAX
// and the DENOMINATOR but never the numerator:
//
//	m     = max(max_s score_s, sink_h)
//	denom = Σ_s exp(score_s − m) + exp(sink_h − m)
//	out_d = Σ_s exp(score_s − m)/denom · v[s][d]
//
// WHY IT CANNOT BE A POST-HOC DENOMINATOR FIX, which is the natural-looking shortcut: the
// sink competes for the max, so folding it in after the exp pass would need every exponent
// recomputed against a new maximum. Getting that wrong is invisible whenever the sink is
// below the score maximum — i.e. in most random tests — and wrong exactly when it is not.
// This test therefore includes a case where the SINK DOMINATES.
//
// It gates the kernel directly rather than through a model, because gpt-oss cannot be
// resident yet (its clamped-SwiGLU expert kernel does not exist). That is the point of doing
// the sink first: it is independently checkable.
package cuda

import (
	"context"
	"math"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
)

func TestAttnSink_matchesReference(t *testing.T) {
	const nH, nKV, hd, startPos, M = 4, 2, 8, 5, 1
	const window = 0
	scale := float32(0.35)
	nKeys := startPos + M
	qDim, kvDim := nH*hd, nKV*hd

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
	mod, err := cx.LoadModule(prefillBatchedPTX)
	if err != nil {
		t.Fatalf("LoadModule(prefill_batched): %v", err)
	}
	fn, err := mod.Function("attn_batched")
	if err != nil {
		t.Fatalf("Function(attn_batched): %v", err)
	}
	stream := mustStream(t, cx)

	q := make([]float32, M*qDim)
	kc := make([]float32, nKeys*kvDim)
	vc := make([]float32, nKeys*kvDim)
	for i := range q {
		q[i] = float32(math.Sin(float64(i)*0.7)) * 0.8
	}
	for i := range kc {
		kc[i] = float32(math.Cos(float64(i)*0.31)) * 0.9
		vc[i] = float32(math.Sin(float64(i)*0.13)) * 1.1
	}

	dQ := mustAlloc[float32](t, cx, len(q))
	dK := mustAlloc[float32](t, cx, len(kc))
	dV := mustAlloc[float32](t, cx, len(vc))
	dCtx := mustAlloc[float32](t, cx, M*qDim)
	dSink := mustAlloc[float32](t, cx, nH)
	for _, cp := range []struct {
		d *gc.Buffer[float32]
		h []float32
	}{{dQ, q}, {dK, kc}, {dV, vc}} {
		if e := gc.CopyHtoD(bg, cp.d, cp.h); e != nil {
			t.Fatalf("H2D: %v", e)
		}
	}

	// CPU reference — the same math forward_gptoss.go implements.
	ref := func(sinks []float32) []float32 {
		out := make([]float32, M*qDim)
		group := nH / nKV
		for h := range nH {
			kvh := h / group
			sc := make([]float64, nKeys)
			mx := math.Inf(-1)
			for s := range nKeys {
				var dot float64
				for d := range hd {
					dot += float64(q[h*hd+d]) * float64(kc[s*kvDim+kvh*hd+d])
				}
				sc[s] = dot * float64(scale)
				mx = math.Max(mx, sc[s])
			}
			if sinks != nil {
				mx = math.Max(mx, float64(sinks[h]))
			}
			denom := 0.0
			for s := range nKeys {
				sc[s] = math.Exp(sc[s] - mx)
				denom += sc[s]
			}
			if sinks != nil {
				denom += math.Exp(float64(sinks[h]) - mx)
			}
			for d := range hd {
				acc := 0.0
				for s := range nKeys {
					acc += sc[s] * float64(vc[s*kvDim+kvh*hd+d])
				}
				out[h*hd+d] = float32(acc / denom)
			}
		}
		return out
	}

	run := func(sinks []float32) []float32 {
		arg := gc.ArgDevicePtr(0)
		if sinks != nil {
			if e := gc.CopyHtoD(bg, dSink, sinks); e != nil {
				t.Fatalf("H2D sinks: %v", e)
			}
			arg = gc.Arg(dSink)
		}
		cfg := gc.LaunchConfig{GridX: nH, GridY: M, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1,
			SharedMemBytes: uint32((nKeys + 128) * 4)}
		if e := fn.LaunchOn(bg, stream, cfg,
			gc.Arg(dQ), gc.Arg(dK), gc.Arg(dV),
			gc.ArgValue(int32(nH)), gc.ArgValue(int32(nKV)), gc.ArgValue(int32(hd)),
			gc.ArgValue(int32(startPos)), gc.ArgValue(scale),
			gc.ArgValue(int32(window)), gc.ArgValue(int32(M)), gc.Arg(dCtx), arg); e != nil {
			t.Fatalf("launch: %v", e)
		}
		if e := stream.Synchronize(bg); e != nil {
			t.Fatalf("sync: %v", e)
		}
		got := make([]float32, M*qDim)
		if e := gc.CopyDtoH(bg, got, dCtx); e != nil {
			t.Fatalf("D2H: %v", e)
		}
		return got
	}

	cmp := func(name string, got, want []float32, tol float64) {
		worst, at := 0.0, -1
		for i := range want {
			d := math.Abs(float64(got[i] - want[i]))
			if d > worst {
				worst, at = d, i
			}
		}
		t.Logf("%s: max|diff| = %.3e", name, worst)
		if worst > tol {
			t.Errorf("%s: max|diff| %.3e > %.1e at %d (got %v want %v)", name, worst, tol, at, got[at], want[at])
		}
	}

	// 1. NULL sink must be exactly the pre-existing behaviour.
	cmp("no sink", run(nil), ref(nil), 1e-5)

	// 2. A modest sink, below the score max — the easy case.
	small := []float32{-1.0, -0.5, 0.25, -2.0}
	cmp("small sink", run(small), ref(small), 1e-5)

	// 3. THE SINK DOMINATES the max. This is the case a post-hoc denominator patch gets
	// wrong: with sink >> scores, every exponent must be re-based on the sink, and the
	// output should be pushed toward zero because the denominator is dominated by a term
	// with no value vector.
	big := []float32{20, 25, 30, 18}
	gotBig := run(big)
	cmp("sink dominates", gotBig, ref(big), 1e-5)
	sumAbs := 0.0
	for _, v := range gotBig {
		sumAbs += math.Abs(float64(v))
	}
	if sumAbs > 1e-3 {
		t.Errorf("with a dominating sink the context should collapse toward zero, got Σ|out| = %g", sumAbs)
	}

	// 4. A very negative sink must be indistinguishable from no sink at all.
	tiny := []float32{-60, -60, -60, -60}
	cmp("negligible sink ≈ no sink", run(tiny), ref(nil), 1e-5)
}
