package decoder

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestEagleHeadForward gates the 05 head forward (inc 3) on the real
// AngelSlim/Qwen3-1.7B head + the local Qwen3-1.7B base: capture 3 target hidden
// states (the seam), fuse, run the head, and check it produces finite draft logits
// that map into the target vocab. The meaningful signal: the head's top draft should
// often equal the BASE's own next-token argmax — that is literally what the head is
// trained to predict, so agreement is strong evidence the forward is wired right
// (full acceptance numbers come in inc 5). Skips without both models.
func TestEagleHeadForward(t *testing.T) {
	requireHeavyModel(t)
	home, _ := os.UserHomeDir()
	headDir := filepath.Join(home, "models", "qwen3-1.7b-eagle3")
	basePath := filepath.Join(home, "models", "qwen3-1.7b-q8_0.gguf")
	for _, p := range []string{filepath.Join(headDir, "model.safetensors"), basePath} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("missing %s: %v", p, err)
		}
	}
	head := sharedEagleHead(t, headDir)
	base := sharedEagleBase(t, basePath, "int8int8")
	if base.w.arch.HiddenDim != head.Hidden() {
		t.Fatalf("hidden mismatch: base %d, head %d", base.w.arch.HiddenDim, head.Hidden())
	}
	tk, err := tokenizer.LoadGGUF(basePath)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}

	L := base.w.arch.NumLayers
	capLayers := []int{2, L / 2, L - 3} // low/mid/high — best of the inc-3 sweep (~41% top-1 agree)

	// Drive a few continuation steps; at each, draft with the head and compare to the
	// base's own next token (greedy). Count agreement as the forward-correctness signal.
	ids, _ := tk.Encode("The capital of France is Paris. The capital of Germany is", true)
	cache := base.NewCache(len(ids) + 16)
	var hid [][]float32
	var baseLogits []float32
	for i, id := range ids {
		if i < len(ids)-1 {
			if _, ferr := base.forward(id, cache); ferr != nil {
				t.Fatalf("forward: %v", ferr)
			}
			continue
		}
		baseLogits, hid, err = base.ForwardCapture(id, cache, capLayers)
		if err != nil {
			t.Fatalf("ForwardCapture: %v", err)
		}
	}

	h3 := make([]float32, 0, 3*head.Hidden())
	for _, hs := range hid {
		h3 = append(h3, hs...)
	}
	feature := head.Fuse(base.be, h3)
	embedRow := make([]float32, head.Hidden())
	base.embedToken(ids[len(ids)-1], embedRow)
	st := head.NewState()
	draft, _ := head.Step(base.be, embedRow, feature, len(ids)-1, st)

	if len(draft) != head.draftVocab {
		t.Fatalf("draft logits len %d, want %d", len(draft), head.draftVocab)
	}
	for i, v := range draft {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("draft[%d] not finite: %v", i, v)
		}
	}
	di := argmax(draft)
	tid := head.TargetID(di)
	if tid < 0 || tid >= head.vocab {
		t.Fatalf("draft idx %d → target id %d out of range", di, tid)
	}
	baseNext := argmax(baseLogits)
	ds, _ := tk.Decode([]int{tid})
	bs, _ := tk.Decode([]int{baseNext})
	t.Logf("head draft → target %d (%q); base next-token argmax → %d (%q); match=%v",
		tid, ds, baseNext, bs, tid == baseNext)
}
