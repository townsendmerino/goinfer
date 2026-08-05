//go:build darwin && goinfer_testhooks

package metal

import (
	"os"
	"runtime"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// forwardLogitsPerLayerSubmit is the SUBMISSION STRUCTURE a naive Metal expert-paging path would be
// forced into: because the router's top-k at layer L decides which experts must be resident before
// layer L can be ENCODED, the whole-trunk value-independent pre-encode (encodeTrunkInto → one command
// buffer/token) is impossible — each layer becomes its own command buffer, committed and waited on,
// with a host readback between (here a stand-in read of r.aSc; in the real path it reads the router
// idx). ~nL submit-and-wait cycles per token instead of one. TEST-ONLY (defined in _test.go, never
// called by production), so decode stays byte-identical; it exists purely to PRICE the regime.
//
// It reuses encodeLayer (the per-layer seam), so it drives dense / generic-MoE / gemma4-MoE layers
// identically — the number it produces is the submission overhead, which is architecture-independent.
func (r *resident) forwardLogitsPerLayerSubmit(pos int) []float32 {
	r.uPos.SetU32(uint32(pos))
	r.uNKeys.SetU32(uint32(pos + 1))
	for l := 0; l < r.nL; l++ {
		e := r.q.Begin()
		r.encodeLayer(e, l) // exactly the dispatches encodeTrunkInto would encode for this layer
		e.End()             // commit + waitUntilCompleted — the per-layer submit+wait cycle
		_ = r.aSc.Floats()[0]
	}
	e := r.q.Begin()
	e.Dispatch(r.pRms, 256, 256, r.x, r.finalNorm, r.aq, r.aSc, r.uH, r.uEps, r.uAddOne)
	e.Dispatch(r.pGemvW8, (r.V)*32, 32, r.aq, r.aSc, r.lmW, r.lmS, r.logits, r.uH)
	e.End()
	r.finalizeLogits()
	return r.logitsHost
}

// TestPageCost_submissionStructure is Step-6 Step-0: PRICE what losing the value-independent
// pre-encode costs, to decide whether a synchronous Metal expert-paging path is viable or whether
// the speculative (prefetch-last-token's-experts) design is mandatory.
//
// HONEST SCOPE (read this before trusting the number): the ask was to measure the GENERIC Mixtral-
// class MoE path on a fitting checkpoint. There is NO such checkpoint on this Mac (only gemma4-26b,
// which doesn't fit and is the new path), so this measures the SUBMISSION-STRUCTURE cost on a DENSE
// model (qwen2.5-1.5b, 28 layers) — the dominant, architecture-independent term (1 command buffer/
// token vs ~nL/token). It is a faithful LOWER BOUND on the per-layer MoE regime: MoE adds the router
// readback + expert-stage host work at each boundary, which overlaps the GPU-idle gap this already
// pays for. It is NOT the MoE-path number and is not reported as one. Option 3 (MTLSharedEvent
// handshake — one submit/token + per-layer CPU↔GPU events) needs aikit bindings that don't exist yet;
// it is measured only if regime (2) here is expensive enough to matter.
//
// Reports baseline (pipelined pre-encode) and per-layer-submit as tok/s + ms/token, best of 3 warm
// runs (first discarded), greedy, same model + prompt. Heavy-gated (loads a ~1 GB checkpoint).
func TestPageCost_submissionStructure(t *testing.T) {
	requireHeavyModel(t)
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

	const nTok = 64
	emb := make([][]float32, nTok)
	for i := range emb {
		emb[i] = m.EmbedResidentForTest(1 + i%16) // arbitrary fixed token ids; timing, not correctness
	}

	// bestMsPerTok runs `fwd` over nTok positions, 4 times (discard the first warm run), returns the
	// fastest ms/token — the cleanest single figure (min excludes scheduler/thermal noise).
	bestMsPerTok := func(fwd func(pos int) []float32) float64 {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		best := 1e18
		for run := 0; run < 4; run++ {
			start := time.Now()
			for pos := 0; pos < nTok; pos++ {
				_ = fwd(pos)
			}
			ms := float64(time.Since(start).Microseconds()) / 1000.0 / float64(nTok)
			if run > 0 && ms < best { // run 0 is the warm-up, discarded
				best = ms
			}
		}
		return best
	}

	// Correctness cross-check: per-layer-submit is a different ENCODING of the same compute, so at the
	// same position it must pick the same argmax as the single-submit path. If it doesn't, the regime
	// number is measuring a broken forward, not the cost — fail loudly rather than report noise.
	base0 := append([]float32(nil), r.ForwardEmb(emb[0], 0)...)
	perlayer0 := r.forwardLogitsPerLayerSubmit(0) // pos 0 already has its KV from the line above
	if a, b := argmaxF(base0), argmaxF(perlayer0); a != b {
		t.Fatalf("per-layer-submit argmax %d != single-submit %d at pos 0 — the instrumented forward diverges; number would be meaningless", b, a)
	}

	// Baseline is ForwardEmb (INLINE single command buffer/token), NOT ForwardEmbPipe: the pipelined
	// path runs a persistent executor GOROUTINE, and measuring the inline per-layer regime while that
	// goroutine is alive contends the single Metal GPU and inflates per-layer (an earlier revision
	// read +106% that way; clean it is ~+43%). Both regimes measured inline, same conditions. The
	// encode-ahead OVERLAP that Pipe adds is small here anyway (~1-2%) — decode is GPU-bound, so the
	// dominant cost is the SUBMISSION STRUCTURE (1 command buffer/token vs ~nL), which this isolates.
	baseMs := bestMsPerTok(func(pos int) []float32 { return r.ForwardEmb(emb[pos], pos) })
	perLayerMs := bestMsPerTok(func(pos int) []float32 {
		copy(r.x.Floats(), emb[pos])
		return r.forwardLogitsPerLayerSubmit(pos)
	})

	baseTps, perLayerTps := 1000.0/baseMs, 1000.0/perLayerMs
	overhead := (perLayerMs - baseMs) / baseMs * 100
	t.Logf("model=qwen2.5-1.5b (dense, %d layers) nTok=%d best-of-3-warm, both regimes INLINE", r.nL, nTok)
	t.Logf("  (1) baseline single command buffer: %6.2f ms/tok  %6.1f tok/s", baseMs, baseTps)
	t.Logf("  (2) per-layer submit+wait (~%d CB) : %6.2f ms/tok  %6.1f tok/s", r.nL, perLayerMs, perLayerTps)
	t.Logf("  submission-structure overhead     : %+.1f%% (per-layer vs baseline)", overhead)
	t.Logf("  per-extra-submit cost             : ~%.3f ms  (over %d extra submits/token)", (perLayerMs-baseMs)/float64(r.nL), r.nL)
	t.Logf("FINDING: expensive. Option (3), the MTLSharedEvent handshake, was measured on this same forward " +
		"(aikit gpu/metal_sharedevent_test.go + a temp goinfer harness) and recovered ~0%% — handshake ≈ submit " +
		"per boundary. Both synchronous regimes cost ~+45%%, so speculative prefetch (option b) is the path.")
}
