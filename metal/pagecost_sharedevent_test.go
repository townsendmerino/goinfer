//go:build darwin && goinfer_testhooks

package metal

import (
	"os"
	"runtime"
	"testing"
	"time"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// forwardLogitsSharedEvent is Step-6 Step-0 regime (3): the whole trunk in ONE command buffer, but
// after each layer the GPU signals a shared event and waits for a CPU ack before the next layer — so
// the host could read the router idx / stage experts per layer WITHOUT tearing the token into ~nL
// command buffers. TEST-ONLY (decode is byte-identical by default; production never calls it). Needs
// runtime.LockOSThread on the caller: EventBoundary drives NSAutoreleasePool ops, which are
// thread-local, and a migrating goroutine faults 0x10 on the 2nd token (banked in Step-0).
func (r *resident) forwardLogitsSharedEvent(pos int, ev gpu.SharedEvent, base uint64) []float32 {
	r.uPos.SetU32(uint32(pos))
	r.uNKeys.SetU32(uint32(pos + 1))
	e := r.q.Begin()
	for l := 0; l < r.nL; l++ {
		r.encodeLayer(e, l)
		if l < r.nL-1 {
			e.EventBoundary(ev, base+uint64(2*l+1), base+uint64(2*l+2))
		}
	}
	e.Dispatch(r.pRms, 256, 256, r.x, r.finalNorm, r.aq, r.aSc, r.uH, r.uEps, r.uAddOne)
	e.Dispatch(r.pGemvW8, (r.V)*32, 32, r.aq, r.aSc, r.lmW, r.lmS, r.logits, r.uH)
	e.FinishEncoding()
	e.Commit()
	for l := 0; l < r.nL-1; l++ {
		for ev.Value() < base+uint64(2*l+1) { // spin until the GPU finishes layer l
		}
		_ = r.aSc.Floats()[0]             // router-idx readback stand-in (the real path reads idx here)
		ev.SetValue(base + uint64(2*l+2)) // ack → GPU proceeds to layer l+1
	}
	e.WaitDone()
	e.DrainPool()
	r.finalizeLogits()
	return r.logitsHost
}

// TestPageCost_sharedEventReal is Step-6 Step-0 regime (3) on the REAL forward — the authoritative
// measurement that the synthetic aikit probe (gpu/metal_sharedevent_test.go) understates. It reprices
// all three regimes inline on qwen2.5-1.5b so shared-event is directly comparable to baseline and
// per-layer-submit. FINDING (2026-08): shared-event handshake (~0.26 ms/boundary) is ≈ per-layer
// submit (~0.23 ms/boundary) — recovers ~0%; both synchronous shapes cost ~+45%. Conclusion:
// synchronous Metal MoE paging is not viable by either shape → speculative prefetch is the path.
//
// This is the committed, repo-reproducible form of the regime-3 result (was run under a local go.work
//
//	override at Step-0). Run: GOINFER_HEAVY_TESTS=1 GOINFER_HANDSHAKE_PROBE=1 go test ./metal/ \
//	  -run TestPageCost_sharedEventReal -v  (loads a ~1 GB checkpoint; needs aikit/gpu >= v0.23.0).
func TestPageCost_sharedEventReal(t *testing.T) {
	requireHeavyModel(t)
	if os.Getenv("GOINFER_HANDSHAKE_PROBE") == "" {
		t.Skip("set GOINFER_HANDSHAKE_PROBE=1 to run the shared-event regime-3 perf probe")
	}
	path := os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s", path)
	}
	m, err := decoder.Load(path, decoder.Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("BuildResident: %v", err)
	}
	defer r.Close()

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	const nTok = 64
	emb := make([][]float32, nTok)
	for i := range emb {
		emb[i] = m.EmbedResidentForTest(1 + i%16)
	}
	ev := r.d.NewSharedEvent()

	// argmax cross-check: shared-event is a different ENCODING of the same compute — it must pick the
	// same token as the single-submit path, or the number prices a broken forward.
	base0 := append([]float32(nil), r.ForwardEmb(emb[0], 0)...)
	se0 := r.forwardLogitsSharedEvent(0, ev, 0)
	if a, b := argmaxF(base0), argmaxF(se0); a != b {
		t.Fatalf("shared-event argmax %d != single-submit %d at pos 0 — instrumented forward diverges", b, a)
	}

	best := func(fwd func(pos int)) float64 {
		b := 1e18
		for run := range 4 {
			start := time.Now()
			for pos := range nTok {
				fwd(pos)
			}
			ms := float64(time.Since(start).Microseconds()) / 1000.0 / float64(nTok)
			if run > 0 && ms < b {
				b = ms
			}
		}
		return b
	}
	baseMs := best(func(pos int) { r.ForwardEmb(emb[pos], pos) })
	perLayerMs := best(func(pos int) {
		copy(r.x.Floats(), emb[pos])
		r.forwardLogitsPerLayerSubmit(pos)
	})
	base := uint64(2 * r.nL) // pos 0 argmax check used [0, 2*nL); continue monotonic
	evMs := best(func(pos int) {
		copy(r.x.Floats(), emb[pos])
		r.forwardLogitsSharedEvent(pos, ev, base)
		base += uint64(2 * r.nL)
	})

	perSubmit := (perLayerMs - baseMs) / float64(r.nL)
	perHandshake := (evMs - baseMs) / float64(r.nL-1)
	rec := 0.0
	if perLayerMs > baseMs {
		rec = (1 - (evMs-baseMs)/(perLayerMs-baseMs)) * 100
	}
	t.Logf("REAL qwen2.5-1.5b (%d layers) nTok=%d best-of-3-warm, all inline", r.nL, nTok)
	t.Logf("  (1) baseline single command buffer: %6.2f ms/tok  %5.1f tok/s", baseMs, 1000/baseMs)
	t.Logf("  (2) per-layer submit+wait         : %6.2f ms/tok  %5.1f tok/s  (~%.3f ms/submit)", perLayerMs, 1000/perLayerMs, perSubmit)
	t.Logf("  (3) shared-event handshake        : %6.2f ms/tok  %5.1f tok/s  (~%.3f ms/handshake)", evMs, 1000/evMs, perHandshake)
	t.Logf("VERDICT: shared-event recovers %.0f%% of the per-layer loss (handshake %.3f vs submit %.3f ms/boundary) "+
		"— ~0%%: both synchronous shapes ~equal, speculative prefetch is the path.", rec, perHandshake, perSubmit)
}
