package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"testing"
)

// TestLaguna_textParity is the T1 tiny-golden gate for Laguna, run against THREE
// fixtures — one per released generation — because the generations differ in ways
// a single fixture cannot exercise:
//
//	xs21  gating "per-head", sliding/full 3:1 interleave, PER-LAYER query heads
//	      (4 on full, 8 on sliding), YaRN(32) + partial rotary on full layers with
//	      plain full-width RoPE on sliding, one dense prefix layer.
//	xs2   gating true — whose OWN module hardcodes per-head gating and never reads
//	      the field. This fixture is the tiny reproduction of the real-checkpoint
//	      finding: the granularity must come from the g_proj SHAPE, not the config.
//	m1    gating "per-element", no sliding window, uniform heads, routed scaling
//	      1.0, two dense prefix layers. The per-element gate path at a different
//	      dense/MoE split.
//
// Each fixture was generated against ITS OWN generation's modeling_laguna.py
// (see scripts/pin_laguna_tiny.py), and the references were verified to actually
// apply the per-layer-type RoPE — xs21's HF model carries rotary_emb with
// inv_freq len 4 + YaRN mscale 1.3466 and swa_rotary_emb with len 8 + scaling
// 1.0, exactly the widths and scalings the adapter derives. Without that check a
// silently-degenerate reference would have made this gate meaningless.
func TestLaguna_textParity(t *testing.T) {
	for _, tc := range []struct {
		tag             string
		wantGranularity string
		wantGProjRows   int // layer 0
	}{
		{tag: "xs21", wantGranularity: "per-head", wantGProjRows: 4},
		{tag: "xs2", wantGranularity: "per-head", wantGProjRows: 4}, // despite gating: true
		{tag: "m1", wantGranularity: "per-element", wantGProjRows: 64},
	} {
		t.Run(tc.tag, func(t *testing.T) {
			golden := filepath.Join("testdata", "laguna_"+tc.tag+"_tiny_text_golden.json")
			ckpt := filepath.Join("testdata", "laguna-"+tc.tag+"-tiny")
			raw, err := os.ReadFile(golden)
			if errors.Is(err, fs.ErrNotExist) {
				t.Skipf("no golden — run scripts/pin_laguna_tiny.py")
			}
			if err != nil {
				t.Fatalf("read golden: %v", err)
			}
			// Stat the WEIGHTS, not the directory: a bare fixture dir can exist from a
			// stray checkout while the (gitignored) weights are absent, and a dir-only
			// guard then turns a skip into a confusing hard failure.
			if _, err := os.Stat(filepath.Join(ckpt, "model.safetensors")); errors.Is(err, fs.ErrNotExist) {
				t.Skipf("no checkpoint at %s — run scripts/pin_laguna_tiny.py", ckpt)
			}
			var g struct {
				PromptIDs       []int     `json:"prompt_ids"`
				Argmax          int       `json:"argmax"`
				LastLogits      []float32 `json:"last_logits"`
				NNew            int       `json:"n_new"`
				ContinuationIDs []int     `json:"continuation_ids"`
				GProjRows       int       `json:"g_proj_rows_layer0"`
				Granularity     string    `json:"gate_granularity"`
				DeclaredGating  any       `json:"declared_gating"`
			}
			if err := json.Unmarshal(raw, &g); err != nil {
				t.Fatalf("parse golden: %v", err)
			}
			// Pin the fixture itself: if a regenerated golden ever stopped exercising the
			// granularity this case is here to cover, the test would still pass against it
			// and quietly test nothing.
			if g.Granularity != tc.wantGranularity || g.GProjRows != tc.wantGProjRows {
				t.Fatalf("fixture drift: golden has g_proj rows=%d (%s), want %d (%s) — regenerate with scripts/pin_laguna_tiny.py",
					g.GProjRows, g.Granularity, tc.wantGProjRows, tc.wantGranularity)
			}

			m, err := Load(ckpt, Options{})
			if err != nil {
				t.Fatalf("Load(%s): %v", ckpt, err)
			}
			defer m.Close()

			if got := m.w.arch.Name; got != "laguna" {
				t.Fatalf("arch = %q, want laguna", got)
			}
			// The loader must have picked the granularity from the tensor, and for xs2
			// that DISAGREES with the declared config value.
			if got := m.w.Layers[0].GProj.Rows(); got != tc.wantGProjRows {
				t.Errorf("layer 0 GProj rows = %d, want %d (declared gating=%v)", got, tc.wantGProjRows, g.DeclaredGating)
			}

			cache := m.NewCache(len(g.PromptIDs) + g.NNew)
			var logits []float32
			for _, id := range g.PromptIDs {
				if logits, err = m.forward(id, cache); err != nil {
					t.Fatalf("forward: %v", err)
				}
			}
			gotArg := argmax(logits)
			cos := logitCosine(logits, g.LastLogits)
			t.Logf("laguna/%s text parity: argmax got=%d want=%d | logit cosine=%.6f | gate=%s",
				tc.tag, gotArg, g.Argmax, cos, g.Granularity)
			if gotArg != g.Argmax {
				t.Errorf("last argmax = %d, want %d", gotArg, g.Argmax)
			}
			if cos < 0.9999 {
				t.Errorf("last-logit cosine %.6f < 0.9999", cos)
			}

			// Greedy continuation: catches per-position errors (RoPE width, sliding-window
			// start, per-layer head count) that a single-position logit compare can miss.
			cont := make([]int, 0, g.NNew)
			cur := append([]int(nil), g.PromptIDs...)
			for range g.NNew {
				c2 := m.NewCache(len(cur) + 1)
				var lg []float32
				for _, id := range cur {
					if lg, err = m.forward(id, c2); err != nil {
						t.Fatalf("forward: %v", err)
					}
				}
				nxt := argmax(lg)
				cont = append(cont, nxt)
				cur = append(cur, nxt)
			}
			for i := range cont {
				if i < len(g.ContinuationIDs) && cont[i] != g.ContinuationIDs[i] {
					t.Errorf("continuation[%d] = %d, want %d (full got=%v want=%v)",
						i, cont[i], g.ContinuationIDs[i], cont, g.ContinuationIDs)
					break
				}
			}
		})
	}
}
