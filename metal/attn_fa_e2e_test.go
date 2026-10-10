//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestAttentionFA_endToEndReproduction is the KEEPER reproducer of the end-to-end signature that parked R2
// (docs/tasks/red-october.md): a kernel proven correct in isolation (TestAttentionFA_vsReference) whose logits diverged from the
// shipped kernel's from decode step 2 on. It batch-prefills a real qwen2.5-1.5b checkpoint to depth 1600 (> attnFADepthFloor),
// then decodes 5 tokens one at a time, comparing logits against the SAME prefill+decode sequence with attention_fa disabled, and
// asserts each step: cosine >= 0.9999 and maxAbs <= 0.1.
//
// The signature is not a kernel defect (docs/measurements/r2-attn-fa-rootcause-2026-09-21.md; instrument r2_ctx_diff_test.go):
// attention_fa is non-bit-identical by design, and one int8 activation-quantization rounding crossing on accumulated f32
// reduction-order noise carries forward through the KV cache. The numbers measured when it was parked, and the hypotheses ruled
// out, are at docs/code-notes/metal.md#TestAttentionFA_endToEndReproduction.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_TEST_MODEL=~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf go test -tags goinfer_testhooks ./metal/ -run TestAttentionFA_endToEndReproduction -v
func TestAttentionFA_endToEndReproduction(t *testing.T) {
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

	const prefillLen = 1600 // > attnFADepthFloor (1024), so the decode steps below actually engage it
	const nSteps = 5

	runOne := func(enableFA bool) (H int, logitsPerStep [][]float32) {
		t.Helper()
		if enableFA {
			t.Setenv("GOINFER_METAL_ATTN_FA", "1")
		} else {
			t.Setenv("GOINFER_METAL_ATTN_FA", "0")
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
		for step := range nSteps {
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

	for step := range nSteps {
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
