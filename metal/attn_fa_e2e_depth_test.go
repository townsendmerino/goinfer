//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestAttentionFA_endToEndReproductionDepth2200 is the depth-2200 twin of attn_fa_e2e_test.go's
// keeper reproducer of the signature that PARKED R2 (docs/tasks/red-october.md) on 2026-09-19.
// EXPLAINED 2026-09-21 (docs/measurements/r2-attn-fa-rootcause-2026-09-21.md; instrument:
// r2_ctx_diff_test.go): not a kernel defect — on identical inputs attention_fa matches the shipped
// kernel to <=1e-5 at every layer and step; the logit divergence is one int8 activation-quantization
// rounding crossing on accumulated f32 reduction-order noise (the kernel is non-bit-identical by
// design), amplified downstream and carried forward by the KV cache. The numbers this test
// measures are unchanged; what they mean is.
//
// The reproduction is precise and fully deterministic (identical across repeated runs): batch-
// prefill a real qwen2.5-1.5b checkpoint to depth 1600 (> attnFADepthFloor), then decode 5 tokens
// one at a time, comparing logits against the SAME prefill+decode sequence with attention_fa
// disabled. Steps 0-1 (pos 1600-1601) are BIT-PERFECT (cosine 1.0000000, maxAbs ~2.9e-6 — pure
// f32 rounding noise, the same order of magnitude the isolated kernel gate shows). Step 2 onward
// (pos >= 1602) jumps to a STABLE wrong result (cosine ~0.9995-0.9996, maxAbs ~0.57-0.62) and
// stays there — not a growing numerical drift, not run-to-run noise (reproduced byte-for-byte
// across independent test executions with fresh GPU state each time).
//
// Ruled out (via GOINFER_ATTNFA_DEBUG=1's per-dispatch print at model.go's canUseAttnFA call
// site): the dispatch parameters themselves (curNKeys, nSplit, G, hd) are correct and CONSTANT
// across every layer and every decode step, exactly as this uniform-architecture model should
// produce — so the divergence is not a parameter-computation bug. Also ruled out: cross-instance
// GPU state (the shipped and attention_fa runs execute fully sequentially, one resident closed
// before the next loads — see runOne's own teardown). Not yet investigated: whether Metal's
// automatic hazard tracking correctly serializes the attention_fa -> attention_fa_combine buffer
// dependency across the SPECIFIC pattern of repeated per-layer, per-token buffer reuse this
// kernel pair uses (a genuine unknown, not a ruled-out hypothesis); whether something about the
// SECOND-AND-LATER command buffer specifically (as opposed to the first) interacts badly with the
// shared r.attnFAPartial/r.uAttnFAG/r.uAttnFANSplit scratch buffers this kernel reuses across
// calls, unlike the shipped kernel's own per-call-safe buffer usage.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_TEST_MODEL=~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf go test -tags goinfer_testhooks ./metal/ -run TestAttentionFA_endToEndReproduction -v
func TestAttentionFA_endToEndReproductionDepth2200(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a real checkpoint twice)")
	}
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	path := os.Getenv("GOINFER_TEST_MODEL")
	if path == "" {
		t.Skip("set GOINFER_TEST_MODEL to a real hd=128 GQA checkpoint (e.g. qwen2.5-coder-1.5b)")
	}

	const prefillLen = 2200 // > attnFADepthFloor (1536), so the decode steps below actually engage it
	const nSteps = 5

	runOne := func(enableFA bool) (H int, logitsPerStep [][]float32) {
		t.Helper()
		if enableFA {
			t.Setenv("GOINFER_METAL_ATTN_FA", "1")
		} else {
			t.Setenv("GOINFER_METAL_ATTN_FA", "")
		}
		m, err := decoder.Load(path, decoder.Options{Quant: "int4"})
		if err != nil {
			t.Fatalf("Load: %v", err)
		}
		b := &metalBackend{}
		rf, ok, err := b.BuildResident(m)
		if err != nil || !ok {
			t.Fatalf("BuildResident: ok=%v err=%v", ok, err)
		}
		mr, ok := rf.(*metalResident)
		if !ok {
			t.Fatalf("expected *metalResident")
		}
		r := mr.r
		if r.decodeAttnFA != enableFA {
			t.Fatalf("decodeAttnFA=%v, want %v", r.decodeAttnFA, enableFA)
		}

		H = r.H
		rng := rand.New(rand.NewSource(7))
		embs := make([][]float32, prefillLen)
		for i := range embs {
			row := make([]float32, H)
			for j := range row {
				row[j] = float32(rng.NormFloat64()) * 0.05
			}
			embs[i] = row
		}
		if _, err := r.ForwardBatch(embs, 0); err != nil {
			t.Fatalf("ForwardBatch: %v", err)
		}
		for step := 0; step < nSteps; step++ {
			pos := prefillLen + step
			emb := make([]float32, H)
			for j := range emb {
				emb[j] = float32(rng.NormFloat64()) * 0.05
			}
			l := r.ForwardEmb(append([]float32(nil), emb...), pos)
			logitsPerStep = append(logitsPerStep, append([]float32(nil), l...))
		}
		b.Close() // fully torn down before the next call — no two residents alive at once
		m.Close()
		return H, logitsPerStep
	}

	_, shippedLogits := runOne(false)
	_, faLogits := runOne(true)

	for step := 0; step < nSteps; step++ {
		pos := prefillLen + step
		lShipped, lFA := shippedLogits[step], faLogits[step]
		if len(lShipped) != len(lFA) {
			t.Fatalf("step %d: logits length mismatch %d vs %d", step, len(lShipped), len(lFA))
		}
		var dot, na, nb, maxabs float64
		for i := range lShipped {
			a, b := float64(lShipped[i]), float64(lFA[i])
			dot += a * b
			na += a * a
			nb += b * b
			if d := math.Abs(a - b); d > maxabs {
				maxabs = d
			}
		}
		cos := dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-30)
		t.Logf("decode step %d (pos=%d): cosine=%.7f maxAbs=%.4e", step, pos, cos, maxabs)
		mustFinite(t, "cosine", cos)
		if cos < 0.9999 || maxabs > 1e-1 {
			t.Errorf("step %d: attention_fa end-to-end parity FAIL cosine=%.7f maxAbs=%.4e", step, cos, maxabs)
		}
	}
}
