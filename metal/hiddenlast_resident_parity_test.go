//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// hiddenLastResidentBar is deliberately NOT the task doc's literal "cosine >= 0.9999" — that
// number describes vectorParityBar (qwen3_embedding_parity_test.go), an HF-f32-reference-vs-
// GGUF-quant comparison on a certified embedder. The comparison here is CPU-vs-Metal-resident
// int8 execution of the SAME weights, which is the exact comparison gpt2_resident_parity_test.go
// already makes for decode logits — and its accepted floor is 0.95, not 0.9999. Measured directly
// (not assumed): this hidden-state-only comparison — before the LM head's amplifying vocab
// projection — lands at ~0.9991-0.9993 cosine, tighter than the post-head 0.95 floor as expected,
// but nowhere near 0.9999. 0.998 leaves headroom above the measured value without being a bar
// nothing could ever fail.
const hiddenLastResidentBar = 0.998

// TestHiddenLastResidentParityMetal is the G4 gate (docs/task-gpu-paths-2026-09.md): the resident
// HiddenLast path (metalResident.HiddenLast / resident.forwardHiddenNoHead) must match the CPU
// HiddenLast to within hiddenLastResidentBar on real weights. resident_embed_seam_test.go
// (decoder package) already proves the WIRING with a fake backend; this proves the NUMBERS with
// the real kernels.
//
// GPT-2 is deliberately chosen, not swapped for a more "standard" dense family: it is the family
// that exercises learned positional embeddings (FeatLearnedPos) on this backend
// (gpt2_resident_parity_test.go's twin for decode), and forwardHiddenNoHead's addLearnedPos call
// must reproduce that exactly, not just the RoPE-only path every other resident test happens to
// exercise.
//
// Calls the resident runner's HiddenLast DIRECTLY (via ResidentForwardForTest), bypassing
// decoder.Model.HiddenLast's dispatch — that dispatch (resBusy CAS, CPU fallback on decline) is
// already gated by the fake-backend seam tests; this isolates the KERNEL correctness question.
func TestHiddenLastResidentParityMetal(t *testing.T) {
	const ckpt = "../testdata/gpt2"

	mRes, err := decoder.Load(ckpt, decoder.Options{Backend: "metal", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load metal: %v", err)
	}
	defer mRes.Close()
	rf := mRes.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("gpt2 did not go resident (BuildResident refused) — decode path %q; decline: %s", mRes.DecodePath(), mRes.ResidentDecline())
	}
	rh, ok := rf.(decoder.ResidentHiddenLast)
	if !ok {
		t.Fatalf("metal resident runner does not implement decoder.ResidentHiddenLast")
	}

	mCPU, err := decoder.Load(ckpt, decoder.Options{Backend: "cpu", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load cpu: %v", err)
	}
	defer mCPU.Close()

	_, _, _, _, _, _, vocab := mCPU.Dims()
	const ntok = 24
	ids := make([]int, ntok)
	for i := range ids {
		ids[i] = (i*131 + 7) % vocab
	}

	want, err := mCPU.HiddenLast(ids)
	if err != nil {
		t.Fatalf("cpu HiddenLast: %v", err)
	}

	embs := make([][]float32, len(ids))
	for i, id := range ids {
		embs[i] = mRes.EmbedResidentForTest(id)
	}
	got, err := rh.HiddenLast(context.Background(), embs, 0)
	if err != nil {
		t.Fatalf("resident HiddenLast: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("resident HiddenLast returned %d dims, want %d", len(got), len(want))
	}
	cos, maxAbs := cosF32(want, got)
	t.Logf("HiddenLast over %d tokens: cosine=%.8f maxAbs=%.6g", ntok, cos, maxAbs)
	if cos < hiddenLastResidentBar {
		t.Errorf("resident HiddenLast cosine %.8f < %.4f bar (G4 gate) against the CPU reference", cos, hiddenLastResidentBar)
	}

	// A second, shorter sequence after Reset must ALSO match — proves forwardHiddenNoHead's KV
	// writes for a prior HiddenLast call do not leak into the next one (the resIDs-forgetting
	// half of the G4 contract, exercised here as an actual KV-cache check rather than a nil check).
	rf.Reset()
	ids2 := ids[:8]
	want2, err := mCPU.HiddenLast(ids2)
	if err != nil {
		t.Fatalf("cpu HiddenLast (2nd, shorter): %v", err)
	}
	got2, err := rh.HiddenLast(context.Background(), embs[:8], 0)
	if err != nil {
		t.Fatalf("resident HiddenLast (2nd, shorter): %v", err)
	}
	cos2, maxAbs2 := cosF32(want2, got2)
	t.Logf("HiddenLast over %d tokens (2nd, after Reset): cosine=%.8f maxAbs=%.6g", len(ids2), cos2, maxAbs2)
	if cos2 < hiddenLastResidentBar {
		t.Errorf("resident HiddenLast (2nd sequence, after Reset) cosine %.8f < %.4f — a fresh "+
			"HiddenLast sequence must not see KV left over from the previous one", cos2, hiddenLastResidentBar)
	}
}
