//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// G-NF1 (S17, docs/tasks/task-multimodal-support-2026-10.md): Gemma 3 4B's residual outgrows the batched pass's f16
// (inf at layer 6), and the int8 head reads the NaN row that follows as finite zeros, so without the residual check every
// Gemma 3 prompt of 16+ tokens on Metal answers "<pad>" and noise. With it the pass declines, naming the overflow, and
// Generate's greedy tokens equal the sequential path's (ExactPrefill). The residual scale (G-RS2) now keeps Gemma 3
// inside f16, so this runs it unscaled (s = 1), the overflow the guard exists for. GOINFER_HEAVY_TESTS=1.
func TestGemma3PrefillOverflow_declinesReal(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1")
	}
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, "models", "gemma-3-4b-it")
	src := dir + ".int4.metal.giw"
	tk, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	ids, err := tk.Encode("<start_of_turn>user\nWrite one sentence about the history of the bicycle, mentioning at least one inventor and one country by name please.<end_of_turn>\n<start_of_turn>model\n", true)
	if err != nil {
		t.Fatal(err)
	}
	prefillResidScaleForTest = 1
	defer func() { prefillResidScaleForTest = 0 }()
	gen := func(exact bool) []int {
		m, err := decoder.Load(src, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 1024, ExactPrefill: exact})
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		a, ok := m.ResidentForwardForTest().(*metalResident)
		if !ok {
			t.Fatalf("not Metal-resident: %s", m.ResidentDecline())
		}
		if !exact {
			rows := make([][]float32, len(ids))
			for i, id := range ids {
				rows[i] = m.EmbedResidentForTest(id)
			}
			a.Reset()
			_, err := a.PrefillLast(context.Background(), rows, 0)
			if err == nil || !strings.Contains(err.Error(), "residual overflowed") {
				t.Errorf("the batched pass on %d tokens: err %v, want the f16 residual overflow decline", len(ids), err)
			}
			a.Reset()
		}
		ch, _ := m.Generate(context.Background(), ids, 24, decoder.SamplingParams{})
		var out []int
		for id := range ch {
			out = append(out, id)
		}
		txt, _ := tk.Decode(out)
		fmt.Fprintf(os.Stderr, "[G-NF1] exact=%v: %v %q\n", exact, out, txt)
		return out
	}
	fast, seq := gen(false), gen(true)
	if !slices.Equal(fast, seq) {
		t.Errorf("Generate with fast prefill on: %v; sequential: %v", fast, seq)
	}
}

// G-RS2 (S17): with the residual stored scaled by 1/s, Gemma 3 4B's batched pass stays finite and matches the CPU. Text
// prompts of 31, about 450 and about 2,000 tokens: the batched last token against the CPU at int4 per-32 (cosine >= 0.98,
// G-B3's real tier) and 16 teacher-forced decode steps after it with no non-tie flip (the 3% rule); the overflow guard
// must not fire. Planted defect: s = 1 must trip the guard. GOINFER_HEAVY_TESTS=1.
func TestGemma3PrefillResidScaleReal(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1")
	}
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, "models", "gemma-3-4b-it")
	src := dir + ".int4.metal.giw"
	tk, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	readme, err := os.ReadFile("../README.md")
	if err != nil {
		t.Fatal(err)
	}
	words := strings.Fields(string(readme))
	var prompts [][]int
	for _, n := range []int{0, 180, 830} {
		body := "Write one sentence about the history of the bicycle, mentioning at least one inventor and one country by name please."
		if n > 0 {
			body = strings.Join(words[:n], " ") + "\n\nSummarise the text above."
		}
		ids, err := tk.Encode("<start_of_turn>user\n"+body+"<end_of_turn>\n<start_of_turn>model\n", true)
		if err != nil {
			t.Fatal(err)
		}
		prompts = append(prompts, ids)
	}
	const steps = 16
	ctx := context.Background()
	ref := make([][][]float32, len(prompts))
	teach := make([][]int, len(prompts))
	{
		mc, err := decoder.Load(src, decoder.Options{Backend: "cpu", Quant: "int4", ActQuantGroup: 32})
		if err != nil {
			t.Fatal(err)
		}
		for i, ids := range prompts {
			c := mc.NewCache(len(ids) + steps + 1)
			l, err := mc.PrefillLogitsForTest(ctx, ids, c)
			if err != nil {
				t.Fatal(err)
			}
			ref[i] = [][]float32{append([]float32(nil), l...)}
			teach[i] = []int{argmaxF(l)}
			for k := range steps {
				l, err := mc.ForwardForTest(teach[i][k], c)
				if err != nil {
					t.Fatal(err)
				}
				ref[i] = append(ref[i], append([]float32(nil), l...))
				teach[i] = append(teach[i], argmaxF(l))
			}
			fmt.Fprintf(os.Stderr, "[G-RS2] CPU reference: %d tokens\n", len(ids))
		}
		mc.Close()
	}
	m, err := decoder.Load(src, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 4096})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("not Metal-resident: %s", m.ResidentDecline())
	}
	for i, ids := range prompts {
		rows := make([][]float32, len(ids))
		for j, id := range ids {
			rows[j] = m.EmbedResidentForTest(id)
		}
		a.Reset()
		l, err := a.PrefillLast(ctx, rows, 0)
		if err != nil {
			t.Errorf("%d tokens: the batched pass declined: %v", len(ids), err)
			continue
		}
		pc := cosF(l, ref[i][0])
		worst, real, ties := pc, 0, 0
		for k := range steps + 1 {
			if k > 0 {
				if l, err = a.Forward(m.EmbedResidentForTest(teach[i][k-1]), len(ids)+k-1); err != nil {
					t.Fatal(err)
				}
				worst = min(worst, cosF(l, ref[i][k]))
			}
			if want, got := argmaxF(ref[i][k]), argmaxF(l); want != got {
				lo, hi := ref[i][k][0], ref[i][k][0]
				for _, v := range ref[i][k] {
					lo, hi = min(lo, v), max(hi, v)
				}
				if gap := float64(ref[i][k][want]-ref[i][k][got]) / float64(hi-lo); gap <= 0.03 {
					ties++
				} else {
					real++
				}
			}
		}
		fmt.Fprintf(os.Stderr, "[G-RS2] %4d tokens: batched last token cos %.6f; with %d decode steps worst %.6f; flips %d near-tie, %d real\n", len(ids), pc, steps, worst, ties, real)
		if pc < 0.98 || real > 0 {
			t.Errorf("%d tokens: batched last-token cosine %.6f (>= 0.98), %d non-tie flips (0)", len(ids), pc, real)
		}
		// The planted defect: unscaled, the residual overflows and the guard declines.
		prefillResidScaleForTest = 1
		a.Reset()
		_, err = a.PrefillLast(ctx, rows, 0)
		prefillResidScaleForTest = 0
		if err == nil || !strings.Contains(err.Error(), "residual overflowed") {
			t.Errorf("%d tokens at s = 1: err %v, want the overflow decline (the planted defect is blind)", len(ids), err)
		}
	}
}
