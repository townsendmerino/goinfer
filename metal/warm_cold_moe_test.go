//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestPrefillLast_tailContinuationMatchesCold_MoE is PrefillTailExact's gate for the generic resident MoE passes (the
// expert-major prefill, D-G01's), the claim TestPrefillLast_tailContinuationMatchesCold makes for the dense pass: a
// PrefillLast of 1, 2, 7, 8, 16 or 33 rows continuing a prefix the pass prefilled computes the cold whole-prompt pass (100 tokens, or the fixture's window)'s last
// row bit for bit, on every tiny MoE fixture that takes the pass (Mixtral, two Qwen3-MoE shapes, and the Qwen3.5
// DeltaNet hybrid with a MoE FFN). The decode continuation must differ, or the gate tests nothing.
func TestPrefillLast_tailContinuationMatchesCold_MoE(t *testing.T) {
	for _, fx := range []string{"../testdata/mixtral-tiny", "../testdata/qwen3moe-tiny", "../testdata/qwen3moe-tiny-k3", "../decoder/testdata/qwen3_5_moe-tiny"} {
		t.Run(fx, func(t *testing.T) {
			if _, err := os.Stat(fx + "/model.safetensors"); err != nil {
				t.Skipf("no fixture: %v", err)
			}
			m, err := decoder.Load(fx, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 1024})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			a, ok := m.ResidentForwardForTest().(*metalResident)
			if !ok || a.r.moe == nil || a.r.moe.paged {
				t.Fatalf("not a resident generic MoE on Metal (%s)", m.ResidentDecline())
			}
			_, _, _, _, _, _, vocab := m.Dims()
			n := min(100, a.r.ctxCap) // three fixtures have a 64-position window
			ids := make([]int, n)
			for i := range ids {
				ids[i] = (i*37 + 3) % vocab
			}
			embs := getEmbs(a.r, ids)
			cold, err := a.PrefillLast(context.Background(), embs, 0)
			if err != nil {
				t.Fatalf("cold pass declined: %v", err)
			}
			cold = append([]float32(nil), cold...)
			differ := func(x []float32) int {
				d := 0
				for i := range cold {
					if math.Float32bits(cold[i]) != math.Float32bits(x[i]) {
						d++
					}
				}
				return d
			}
			for _, tail := range []int{1, 2, 7, 8, 16, 33} {
				if _, err := a.PrefillLast(context.Background(), embs[:n-tail], 0); err != nil {
					t.Fatal(err)
				}
				warm, err := a.PrefillLast(context.Background(), embs[n-tail:], n-tail)
				if err != nil {
					t.Fatalf("tail %d: %v", tail, err)
				}
				if d := differ(warm); d != 0 {
					t.Errorf("tail %d: a pass continuing the cached prefix differs from the cold pass in %d of %d logits", tail, d, len(cold))
				}
			}
			if _, err := a.PrefillLast(context.Background(), embs[:n-1], 0); err != nil {
				t.Fatal(err)
			}
			if d := differ(a.r.ForwardEmb(embs[n-1], n-1)); d == 0 {
				t.Fatal("decode of the last position equals the cold pass: the pass is no longer a different lane, and this gate tests nothing")
			}
			t.Logf("tails 1, 2, 7, 8, 16, 33 bit-identical to the cold pass; decode differs")
		})
	}
}

// TestGenerate_warmRepeatMatchesCold_MoE: through decoder.Generate on a resident MoE, the same prompt generated twice
// emits the same 40 greedy tokens, the second time reusing every cached position but the last (its 1-token suffix now
// offered to the pass, PrefillTailExact). A 20-token prompt (the pass) on Mixtral and Qwen3-MoE. With MoE excluded from
// PrefillTailExact again, Mixtral's repeat diverges at token 5, so Mixtral is the case that discriminates; Qwen3-MoE does
// not diverge either way on this fixture. (A DeltaNet hybrid reuses no prefix, so it has no suffix to test here.)
func TestGenerate_warmRepeatMatchesCold_MoE(t *testing.T) {
	for _, fx := range []string{"../testdata/mixtral-tiny", "../testdata/qwen3moe-tiny"} {
		t.Run(fx, func(t *testing.T) {
			if _, err := os.Stat(fx + "/model.safetensors"); err != nil {
				t.Skipf("no fixture: %v", err)
			}
			m, err := decoder.Load(fx, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 64})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			if a, ok := m.ResidentForwardForTest().(*metalResident); !ok || a.r.moe == nil {
				t.Fatalf("not a resident MoE (%s)", m.ResidentDecline())
			}
			_, _, _, _, _, _, vocab := m.Dims()
			prompt := make([]int, 20)
			for i := range prompt {
				prompt[i] = (i*53 + 11) % vocab
			}
			run := func() ([]int, int) {
				ch, g := m.Generate(context.Background(), prompt, 40, decoder.SamplingParams{})
				var ids []int
				for id := range ch {
					ids = append(ids, id)
				}
				if err := g.Err(); err != nil {
					t.Fatal(err)
				}
				return ids, g.PrefillReused
			}
			cold, _ := run()
			warm, reused := run()
			if reused != len(prompt)-1 {
				t.Fatalf("the repeat reused %d positions, want %d", reused, len(prompt)-1)
			}
			for i := range cold {
				if i >= len(warm) || warm[i] != cold[i] {
					t.Fatalf("token %d: warm %v, cold %v", i, warm, cold)
				}
			}
			t.Logf("%d tokens equal warm and cold (%d positions reused)", len(cold), reused)
		})
	}
}
