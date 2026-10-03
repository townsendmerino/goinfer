package decoder

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// The CPU forward of testdata/mistral-tiny-window against HF (f32), with the sliding window ENGAGED: a 48-token prompt through a window of 16, so for
// about 32 positions the window start is past 0 and moving. scripts/pin_mistral_tiny_window.py pins the golden (last-position logits, the argmax and
// six greedy continuation tokens).
//
// THIS TEST EXISTS BECAUSE THE FIXTURE'S WEIGHTS WERE NEVER COMMITTED. Only its config files were (the *.safetensors ignore rule), so the CUDA and WebGPU
// window tests and `gate identity`'s mistral asset had nothing to load on a fresh checkout, and no CPU test consumed the golden at all: the CPU
// sliding-window path had no committed parity gate against HF. A missing fixture or golden FAILS here instead of skipping, for the reason D11 found in
// D2's gate: a gate whose fixture can vanish into a skip is not a gate.
func loadMistralWindowGolden(t *testing.T) (g struct {
	Window          int       `json:"window"`
	PromptIDs       []int     `json:"prompt_ids"`
	Argmax          int       `json:"argmax"`
	LastLogits      []float32 `json:"last_logits"`
	NNew            int       `json:"n_new"`
	ContinuationIDs []int     `json:"continuation_ids"`
}) {
	t.Helper()
	raw, err := os.ReadFile("../testdata/mistral_tiny_window_golden.json")
	if err != nil {
		t.Fatalf("no golden (a skip would hide this; scripts/pin_mistral_tiny_window.py): %v", err)
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	if len(g.PromptIDs) <= g.Window || len(g.LastLogits) == 0 || len(g.ContinuationIDs) != g.NNew {
		t.Fatalf("golden is malformed or the window is not engaged: window %d, prompt %d, logits %d, continuation %d of %d", g.Window, len(g.PromptIDs), len(g.LastLogits), len(g.ContinuationIDs), g.NNew)
	}
	return g
}

// forwardMistralWindow runs the prompt and then n greedy steps, returning the last prompt position's logits and the continuation.
func forwardMistralWindow(t *testing.T, m *Model, prompt []int, n int) ([]float32, []int) {
	t.Helper()
	cache := m.NewCache(len(prompt) + n)
	var logits, last []float32
	var err error
	for _, id := range prompt {
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatal(err)
		}
	}
	last = append([]float32(nil), logits...)
	var cont []int
	for range n {
		next := argmax(logits)
		cont = append(cont, next)
		if logits, err = m.forward(next, cache); err != nil {
			t.Fatal(err)
		}
	}
	return last, cont
}

func TestMistralTinyWindow_matchesHF(t *testing.T) {
	g := loadMistralWindowGolden(t)
	m, err := Load("../testdata/mistral-tiny-window", Options{})
	if err != nil {
		t.Fatalf("Load (the fixture's weights are committed; a skip would hide their absence): %v", err)
	}
	defer m.Close()
	if got := m.w.arch.SlidingWindow; got != g.Window {
		t.Fatalf("the loaded model's sliding window is %d, want %d: the window is not in play, so this would test nothing", got, g.Window)
	}
	last, cont := forwardMistralWindow(t, m, g.PromptIDs, g.NNew)
	var dot, ng, nw, ne float64
	for i := range g.LastLogits {
		a, b := float64(last[i]), float64(g.LastLogits[i])
		dot, ng, nw, ne = dot+a*b, ng+a*a, nw+b*b, ne+(a-b)*(a-b)
	}
	cos, rel := dot/(math.Sqrt(ng)*math.Sqrt(nw)), math.Sqrt(ne/nw)
	t.Logf("mistral tiny window=%d, %d-token prompt: argmax got %d want %d, logit cosine %.8f, relative L2 %.3g, continuation %v want %v", g.Window, len(g.PromptIDs), argmax(last), g.Argmax, cos, rel, cont, g.ContinuationIDs)
	if argmax(last) != g.Argmax {
		t.Errorf("argmax %d, want %d", argmax(last), g.Argmax)
	}
	if cos < 0.9999 || rel > 1e-5 {
		t.Errorf("last-position logits against HF: cosine %.8f (bar 0.9999), relative L2 %.3g (bar 1e-5)", cos, rel)
	}
	for i := range g.ContinuationIDs {
		if cont[i] != g.ContinuationIDs[i] {
			t.Errorf("greedy continuation differs at step %d: %v, want %v", i, cont, g.ContinuationIDs)
			break
		}
	}
}

// Able to fail: with the window switched off the same forward must STOP matching the golden, or the golden cannot see the window and the test above
// proves nothing about it (a short prompt would have passed with the window inert; this one is 48 tokens through a window of 16).
func TestMistralTinyWindow_goldenSeesTheWindow(t *testing.T) {
	g := loadMistralWindowGolden(t)
	m, err := Load("../testdata/mistral-tiny-window", Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	m.w.arch.SlidingWindow = 0
	last, _ := forwardMistralWindow(t, m, g.PromptIDs, 0)
	var ne, nw float64
	for i := range g.LastLogits {
		d := float64(last[i]) - float64(g.LastLogits[i])
		ne, nw = ne+d*d, nw+float64(g.LastLogits[i])*float64(g.LastLogits[i])
	}
	rel := math.Sqrt(ne / nw)
	t.Logf("window off: relative L2 against the golden %.3g", rel)
	if rel < 1e-3 {
		t.Errorf("with the window off the logits still match the golden (relative L2 %.3g): the golden cannot see the window", rel)
	}
}
