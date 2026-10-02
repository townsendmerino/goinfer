package decoder

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// D11's gate (docs/tasks/task-constrained-confidence.md, Route C): the final-norm hidden state at EVERY prompt position must match HF's
// last_hidden_state[0, :] per position, cosine >= 0.9999 and relative L2 <= 1e-5, in f32, on the tiny Qwen3.5 checkpoints (dense, MoE, and the
// derived qwen3_5-tiny-normw whose random final-norm weight is what lets a missing, doubled or mis-weighted final norm fail: cosine alone cannot see
// a uniform scale, and the other tiny checkpoints' final-norm weights are all 0, a scale of exactly 1 under the add-one RMSNorm).
//
// UNLIKE TestPromptHidden_matchesHF, A MISSING CHECKPOINT OR GOLDEN FAILS: that test skips when qwen3_5-tiny-normw has no model.safetensors, and D2
// committed its config but not that file, so on a fresh checkout the one fixture that sees the final norm was skipped (found 2026-10-02; the file is
// committed now). A gate whose key subtest can vanish into a skip is not a gate.
//
// Both the public path and the sequential path are compared with HF, so a mutation of either final-norm site fails. Regenerate the golden with
// scripts/pin_prompt_hidden_all.py.
type hiddenAllGolden struct {
	Fixtures []struct {
		Checkpoint string
		ModelType  string `json:"model_type"`
		HiddenSize int    `json:"hidden_size"`
		Prompts    []struct {
			IDs       []int
			Hidden    [][]float64
			ShapeOnly bool `json:"shape_only"`
		}
	}
}

func loadHiddenAllGolden(t *testing.T) hiddenAllGolden {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "prompt_hidden_all_golden.json"))
	if err != nil {
		t.Fatalf("no golden (a skip would hide it; run scripts/pin_prompt_hidden_all.py): %v", err)
	}
	var g hiddenAllGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	if len(g.Fixtures) < 3 {
		t.Fatalf("golden has %d fixtures, want the dense, normw and MoE tiny checkpoints", len(g.Fixtures))
	}
	return g
}

func loadFixtureModel(t *testing.T, ckpt, quant string) *Model {
	t.Helper()
	dir := filepath.Join("testdata", ckpt)
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		t.Fatalf("checkpoint %s has no model.safetensors (it is committed; a skip would hide this): %v", ckpt, err)
	}
	m, err := Load(dir, Options{Quant: quant})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = m.Close() })
	return m
}

// cosAndRel returns the cosine and the relative L2 error of got against want.
func cosAndRel(got []float32, want []float64) (cos, rel float64) {
	var dot, ng, nw, ne float64
	for i := range want {
		g, w := float64(got[i]), want[i]
		dot, ng, nw, ne = dot+g*w, ng+g*g, nw+w*w, ne+(g-w)*(g-w)
	}
	return dot / (math.Sqrt(ng) * math.Sqrt(nw)), math.Sqrt(ne / nw)
}

func TestPromptHiddenAll_matchesHF(t *testing.T) {
	g := loadHiddenAllGolden(t)
	paths := []struct {
		name string
		run  func(m *Model, ids []int) ([][]float32, error)
	}{
		{"public", func(m *Model, ids []int) ([][]float32, error) { return m.PromptHiddenAll(context.Background(), ids) }},
		{"sequential", func(m *Model, ids []int) ([][]float32, error) {
			return m.promptHiddenAllSequential(context.Background(), ids)
		}},
	}
	for _, fx := range g.Fixtures {
		m := loadFixtureModel(t, fx.Checkpoint, "")
		if _, own := m.w.arch.ownForward(); !own {
			t.Fatalf("%s is expected to have its own layer loop; the test would not cover the Qwen3.5 path", fx.Checkpoint)
		}
		if m.w.arch.HiddenDim != fx.HiddenSize {
			t.Fatalf("%s: hidden %d, golden %d", fx.Checkpoint, m.w.arch.HiddenDim, fx.HiddenSize)
		}
		for _, p := range paths {
			t.Run(fx.Checkpoint+"/"+p.name, func(t *testing.T) {
				compared := 0
				for pi, pr := range fx.Prompts {
					got, err := p.run(m, pr.IDs)
					if err != nil {
						t.Fatal(err)
					}
					if len(got) != len(pr.IDs) || len(got) != len(pr.Hidden) {
						t.Fatalf("prompt %d: %d rows for %d tokens (golden %d rows)", pi, len(got), len(pr.IDs), len(pr.Hidden))
					}
					worstCos, worstRel := 1.0, 0.0
					for pos := range got {
						if len(got[pos]) != fx.HiddenSize {
							t.Fatalf("prompt %d position %d: %d wide, want %d", pi, pos, len(got[pos]), fx.HiddenSize)
						}
						for _, v := range got[pos] {
							if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
								t.Fatalf("prompt %d position %d: non-finite value", pi, pos)
							}
						}
						cos, rel := cosAndRel(got[pos], pr.Hidden[pos])
						worstCos, worstRel = math.Min(worstCos, cos), math.Max(worstRel, rel)
						if pr.ShapeOnly {
							continue // length 1: attention over one key is the identity; shape and finiteness only, not a correctness bar
						}
						compared++
						if cos < 0.9999 {
							t.Errorf("prompt %d (%d tokens) position %d: cosine %.8f < 0.9999", pi, len(pr.IDs), pos, cos)
						}
						if rel > 1e-5 {
							t.Errorf("prompt %d (%d tokens) position %d: relative L2 %.3g > 1e-5", pi, len(pr.IDs), pos, rel)
						}
					}
					t.Logf("prompt %d (%2d tokens%s): worst cosine %.8f, worst relative L2 %.3g over %d positions", pi, len(pr.IDs), map[bool]string{true: ", shape only", false: ""}[pr.ShapeOnly], worstCos, worstRel, len(got))
				}
				if compared < 100 { // 117 positions per fixture less the 1 shape-only: a vacuous loop would pass anything
					t.Fatalf("only %d positions were compared against HF", compared)
				}
			})
		}
	}
}

// The last row is the row PromptHidden returns (the same CPU path on a model with no resident), exactly.
func TestPromptHiddenAll_lastRowEqualsPromptHidden(t *testing.T) {
	g := loadHiddenAllGolden(t)
	for _, fx := range g.Fixtures {
		m := loadFixtureModel(t, fx.Checkpoint, "")
		for _, pr := range fx.Prompts {
			all, err := m.PromptHiddenAll(context.Background(), pr.IDs)
			if err != nil {
				t.Fatal(err)
			}
			last, err := m.PromptHidden(context.Background(), pr.IDs)
			if err != nil {
				t.Fatal(err)
			}
			row := all[len(all)-1]
			for j := range last {
				if row[j] != last[j] {
					t.Fatalf("%s, %d tokens: PromptHiddenAll's last row differs from PromptHidden at %d (%v vs %v)", fx.Checkpoint, len(pr.IDs), j, row[j], last[j])
				}
			}
		}
	}
}

// The batched Qwen3.5 forward against the per-token one, at every position, dense and MoE, f32 and int4 (the same bar and the same reason as
// TestPromptHidden_batchedMatchesSequential: not bit-identical, bounded). A mutation of the sequential path's final norm fails here and in
// TestPromptHiddenAll_matchesHF/sequential.
func TestPromptHiddenAll_batchedMatchesSequential(t *testing.T) {
	for _, tc := range []struct{ ckpt, quant string }{
		{"qwen3_5-tiny-normw", ""}, {"qwen3_5_moe-tiny", ""}, {"qwen3_5-tiny-normw", "int4"}, {"qwen3_5_moe-tiny", "int4"},
	} {
		t.Run(tc.ckpt+"/"+tc.quant, func(t *testing.T) {
			m := loadFixtureModel(t, tc.ckpt, tc.quant)
			for _, n := range []int{2, 5, 13, 32, 64} {
				ids := make([]int, n)
				for i := range ids {
					ids[i] = (i*37 + 11) % m.w.arch.VocabSize
				}
				if !m.qwen35BatchN(n, m.NewCache(n)) {
					t.Fatalf("%d tokens: the batched Qwen3.5 path does not apply, so this would compare a path with itself", n)
				}
				got, err := m.PromptHiddenAll(context.Background(), ids)
				if err != nil {
					t.Fatal(err)
				}
				want, err := m.promptHiddenAllSequential(context.Background(), ids)
				if err != nil {
					t.Fatal(err)
				}
				worst := 0.0
				for pos := range want {
					var ne, nb float64
					for j := range want[pos] {
						d := float64(got[pos][j]) - float64(want[pos][j])
						ne, nb = ne+d*d, nb+float64(want[pos][j])*float64(want[pos][j])
					}
					worst = math.Max(worst, math.Sqrt(ne/nb))
				}
				t.Logf("%d tokens: worst per-position relative L2, batched vs per-token %.3g", n, worst)
				if worst > 1e-6 {
					t.Errorf("%d tokens: worst relative L2 %.3g > 1e-6", n, worst)
				}
			}
		})
	}
}

func TestPromptHiddenAll_refusesBadInput(t *testing.T) {
	m := loadFixtureModel(t, "qwen3_5-tiny", "")
	ctx := context.Background()
	if _, err := m.PromptHiddenAll(ctx, nil); err == nil {
		t.Error("an empty prompt was accepted")
	}
	if _, err := m.PromptHiddenAll(ctx, []int{1, m.w.arch.VocabSize}); err == nil {
		t.Error("a token id past the vocabulary was accepted")
	}
	if _, err := m.PromptHiddenAll(ctx, []int{1, -2}); err == nil {
		t.Error("a negative token id was accepted")
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	ids := []int{1, 2, 3, 4, 5, 6}
	if _, err := m.PromptHiddenAll(cancelled, ids); err == nil {
		t.Error("a cancelled context did not abandon the batched path")
	}
	if _, err := m.promptHiddenAllSequential(cancelled, ids); err == nil {
		t.Error("a cancelled context did not abandon the sequential path")
	}
}

// The rows are the caller's: appending to or editing one must not touch its neighbour (they share one backing array, cap-limited).
func TestPromptHiddenAll_rowsDoNotAlias(t *testing.T) {
	m := loadFixtureModel(t, "qwen3_5-tiny-normw", "")
	rows, err := m.PromptHiddenAll(context.Background(), []int{3, 9, 27, 81})
	if err != nil {
		t.Fatal(err)
	}
	before := append([]float32(nil), rows[1]...)
	rows[0] = append(rows[0], 12345)
	for j := range rows[0] {
		rows[0][j] = 0
	}
	for j := range before {
		if rows[1][j] != before[j] {
			t.Fatalf("editing row 0 changed row 1 at %d", j)
		}
	}
}
