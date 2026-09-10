package decoder

import (
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"testing"
)

// TestGemma4VLBidir_textParity is the P0 invariant for the batched path,
// mirroring TestGemma4VL_textParity's shape: on a use_bidirectional_attention=
// "vision" checkpoint, a TEXT-ONLY forward (no image) must still match HF —
// confirmed this session that HF's own forward degrades block_sequence_ids to
// an all -1 tensor when there is no multimodal content, which makes
// blockwise_overlay unconditionally false, i.e. plain causal (see
// scripts/pin_gemma4_vl_bidir_tiny.py's docstring for the citation). This test
// checks BOTH the unchanged sequential path (m.forward, via runLayersGemma4)
// AND the new batched path (runLayersGemma4FromEmbedN with imgLen=0) against
// the HF golden, and against each other — the regression check the plan
// called for: an M=K batched matmul is not guaranteed bit-identical reduction
// order to K separate M=1 calls, so the bound here is a tight tolerance, not
// literal bit-exactness.
func TestGemma4VLBidir_textParity(t *testing.T) {
	const golden = "../testdata/gemma4_vl_bidir_tiny_text_golden.json"
	const ckpt = "../testdata/gemma4-vl-bidir-tiny"
	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden — run scripts/pin_gemma4_vl_bidir_tiny.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint at %s — run scripts/pin_gemma4_vl_bidir_tiny.py", ckpt)
	}
	var g struct {
		PromptIDs       []int     `json:"prompt_ids"`
		Argmax          int       `json:"argmax"`
		LastLogits      []float32 `json:"last_logits"`
		NNew            int       `json:"n_new"`
		ContinuationIDs []int     `json:"continuation_ids"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}

	m, err := Load(ckpt, Options{})
	if err != nil {
		t.Fatalf("Load(%s): %v", ckpt, err)
	}
	defer m.Close()
	if m.w.arch.gemma4 == nil {
		t.Fatalf("loaded model is not gemma4 (arch=%s)", m.w.arch.Name)
	}
	if got := string(m.w.Cfg.UseBidirectionalAttention); got != "vision" {
		t.Fatalf("UseBidirectionalAttention = %q, want \"vision\" (fixture didn't load the shape this test exists to cover)", got)
	}

	// Sequential path: unchanged, via the plain per-token forward.
	seqCache := m.NewCache(len(g.PromptIDs) + g.NNew)
	var seqLogits []float32
	for _, id := range g.PromptIDs {
		if seqLogits, err = m.forward(id, seqCache); err != nil {
			t.Fatalf("sequential forward: %v", err)
		}
	}
	seqArg := argmax(seqLogits)
	seqCos := logitCosine(seqLogits, g.LastLogits)
	t.Logf("sequential: argmax got=%d want=%d | logit cosine=%.6f", seqArg, g.Argmax, seqCos)
	if seqArg != g.Argmax {
		t.Errorf("sequential last argmax = %d, want %d", seqArg, g.Argmax)
	}
	if seqCos < 0.9999 {
		t.Errorf("sequential last-logit cosine %.6f < 0.9999", seqCos)
	}

	// Batched path (imgLen=0: no image, exercises the same "no block" case).
	batCache := m.NewCache(len(g.PromptIDs) + g.NNew)
	h := m.embedN(g.PromptIDs)
	hLast, err := m.runLayersGemma4FromEmbedN(context.Background(), h, g.PromptIDs, 0, 0, batCache)
	if err != nil {
		t.Fatalf("batched forward: %v", err)
	}
	batLogits := m.logitsFromHidden(hLast, batCache)
	batArg := argmax(batLogits)
	batCos := logitCosine(batLogits, g.LastLogits)
	t.Logf("batched:    argmax got=%d want=%d | logit cosine=%.6f", batArg, g.Argmax, batCos)
	if batArg != g.Argmax {
		t.Errorf("batched last argmax = %d, want %d", batArg, g.Argmax)
	}
	if batCos < 0.9999 {
		t.Errorf("batched last-logit cosine %.6f < 0.9999", batCos)
	}

	// The two goinfer paths must agree with EACH OTHER very tightly too — this is
	// the real regression check: they run the identical weights through the
	// identical math, just batched-at-K vs looped-at-1.
	crossCos := logitCosine(seqLogits, batLogits)
	t.Logf("sequential vs batched: logit cosine=%.8f", crossCos)
	if crossCos < 0.999999 {
		t.Errorf("sequential vs batched cosine %.8f < 0.999999 — the batched path diverges from the reference it must match", crossCos)
	}

	// Greedy continuation must match id-for-id (sequential path, decode is always
	// sequential regardless of which prefill produced the seed).
	cur := seqArg
	for i := 0; i < g.NNew; i++ {
		if cur != g.ContinuationIDs[i] {
			t.Fatalf("continuation[%d] = %d, want %d", i, cur, g.ContinuationIDs[i])
		}
		if seqLogits, err = m.forward(cur, seqCache); err != nil {
			t.Fatalf("decode forward: %v", err)
		}
		cur = argmax(seqLogits)
	}
}

// TestGemma4VLBidir_imageParity is the real gate: a multimodal prompt whose
// image block is deliberately longer than the checkpoint's sliding_window and
// positioned so the block's own start falls outside the last block position's
// causal window (scripts/pin_gemma4_vl_bidir_image.py's docstring has the
// full reasoning) — a shape that only a genuine blockwise-OR-causal mask gets
// right. Also asserts the fixture is non-vacuous: the OLD, unmodified
// sequential path (prefillLogitsGemma4VL, strictly causal) is run on the SAME
// inputs and must reproduce the golden's own causal_only_last_logits (proving
// goinfer's causal path is itself correct on this checkpoint shape) while
// genuinely DISAGREEING with the bidirectional golden (proving a wrong,
// causal-only kernel would provably fail this gate, not coincidentally pass
// it).
func TestGemma4VLBidir_imageParity(t *testing.T) {
	const golden = "../testdata/gemma4_vl_bidir_tiny_image_golden.json"
	const ckpt = "../testdata/gemma4-vl-bidir-tiny"
	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no image golden — run scripts/pin_gemma4_vl_bidir_image.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint — run scripts/pin_gemma4_vl_bidir_tiny.py")
	}
	var g struct {
		InputIDs             []int     `json:"input_ids"`
		ImageTokenStart      int       `json:"image_token_start"`
		NImageTokens         int       `json:"n_image_tokens"`
		ImageFeatures        []float32 `json:"image_features"`
		Argmax               int       `json:"argmax"`
		LastLogits           []float32 `json:"last_logits"`
		CausalOnlyArgmax     int       `json:"causal_only_argmax"`
		CausalOnlyLastLogits []float32 `json:"causal_only_last_logits"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}

	m, err := Load(ckpt, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()

	// The real gate: the new batched, blockwise-masked path.
	biCache := m.NewCache(len(g.InputIDs))
	biLogits, err := m.prefillLogitsGemma4VLBidirectional(context.Background(), g.InputIDs, g.ImageFeatures, g.ImageTokenStart, g.NImageTokens, biCache)
	if err != nil {
		t.Fatalf("prefillLogitsGemma4VLBidirectional: %v", err)
	}
	biArg := argmax(biLogits)
	biCos := logitCosine(biLogits, g.LastLogits)
	t.Logf("bidirectional: argmax got=%d want=%d | logit cosine=%.6f", biArg, g.Argmax, biCos)
	if biArg != g.Argmax {
		t.Errorf("bidirectional argmax = %d, want %d", biArg, g.Argmax)
	}
	if biCos < 0.999 {
		t.Errorf("bidirectional last-logit cosine %.6f < 0.999", biCos)
	}

	// Non-vacuousness: the OLD sequential (strictly causal) path, run on the exact
	// same inputs, must match the golden's OWN causal-only reference (proving
	// goinfer's existing causal path is itself correct here) and must DISAGREE
	// with the bidirectional result above.
	causalCache := m.NewCache(len(g.InputIDs))
	causalLogits, err := m.prefillLogitsGemma4VL(context.Background(), g.InputIDs, g.ImageFeatures, g.ImageTokenStart, g.NImageTokens, causalCache)
	if err != nil {
		t.Fatalf("prefillLogitsGemma4VL (causal-only reference): %v", err)
	}
	causalArg := argmax(causalLogits)
	causalCos := logitCosine(causalLogits, g.CausalOnlyLastLogits)
	t.Logf("causal-only:   argmax got=%d want=%d | logit cosine=%.6f", causalArg, g.CausalOnlyArgmax, causalCos)
	if causalArg != g.CausalOnlyArgmax {
		t.Errorf("causal-only argmax = %d, want %d (goinfer's existing sequential path should still be correct on this checkpoint shape)", causalArg, g.CausalOnlyArgmax)
	}
	if causalCos < 0.999 {
		t.Errorf("causal-only last-logit cosine %.6f < 0.999", causalCos)
	}

	crossCos := logitCosine(biLogits, causalLogits)
	t.Logf("bidirectional vs causal-only: logit cosine=%.6f (must be well below 1.0 — fixture must be non-vacuous)", crossCos)
	if biArg == causalArg && crossCos > 0.999 {
		t.Errorf("bidirectional and causal-only paths agree (argmax %d, cosine %.6f) — this fixture is VACUOUS and cannot distinguish a correct batched kernel from a wrong causal-only one", biArg, crossCos)
	}
}
