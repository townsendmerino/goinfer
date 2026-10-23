//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/fidelity"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestP3b2_int8PassFidelity is P3B2-G2 as first registered (docs/tasks/task-kv-precision-everywhere-2026-10.md, "P3b-2 gates";
// the gate is now TestP3b2_int8PassAgainstCommonReference, D10): on Qwen2.5-Coder-1.5B from ~/models, the int8 MMA pass scored
// against the int8 step, and the f16 pass against
// the f16 step (each step forced, prefillByStep, bit-identical to sequential), over the prefill gate's prose prompts at K = 256,
// 512, 1024 and 3900: the seed position and 64 positions teacher-forced on the step's greedy tokens. fidelity.NonInferior with
// AgreeMarginPts 1, KLMargin 0.10, alpha 0.05. Not timed (shared lock); the positions log goes to $P3B2_POSITIONS_LOG.
//
//	GOINFER_P3B2_FIDELITY=1 go test -count=1 -tags goinfer_testhooks -timeout 60m -run '^TestP3b2_int8PassFidelity$' -v ./metal/
func TestP3b2_int8PassFidelity(t *testing.T) {
	if os.Getenv("GOINFER_P3B2_FIDELITY") == "" {
		t.Skip("set GOINFER_P3B2_FIDELITY=1 (two loads of the 1.5B, minutes)")
	}
	home, _ := os.UserHomeDir()
	path := filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	giw := decoder.SidecarPath(path, "int4", "metal", false)
	ctx, cancel := context.WithTimeout(context.Background(), 55*time.Minute)
	defer cancel()
	load := func(kv string) (*decoder.Model, *metalResident) {
		m, err := decoder.Load(giw, decoder.Options{Backend: "metal", Quant: "int4", KVPrecision: kv, KVQuant: map[string]string{"": "f32", "i8": "i8"}[kv], ResidentContext: 4096})
		if err != nil {
			t.Fatalf("load --kv %q: %v", kv, err)
		}
		t.Cleanup(func() { m.Close() })
		a, ok := m.ResidentForwardForTest().(*metalResident)
		if !ok || a.r.kvI8 != (kv == "i8") || !a.stepPrefill() {
			t.Fatalf("--kv %q: resident %v, kvI8 %v, step %v", kv, ok, ok && a.r.kvI8, ok && a.stepPrefill())
		}
		return m, a
	}
	m16, a16 := load("")
	m8, a8 := load("i8")
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	const cont = 64
	// arm runs one prompt on a: the seed logits from prefill, then cont positions; teacher-forced on toks when given, else greedy.
	arm := func(m *decoder.Model, a *metalResident, ids []int, step bool, toks []int) (logits [][]float32, greedy []int) {
		embs := make([][]float32, len(ids))
		for i, id := range ids {
			embs[i] = m.EmbedResidentForTest(id)
		}
		a.Reset()
		var seed []float32
		if step {
			seed, err = a.prefillByStep(ctx, embs, 0)
		} else {
			seed, err = a.PrefillLast(ctx, embs, 0)
		}
		if err != nil {
			t.Fatalf("prefill (step %v): %v", step, err)
		}
		logits = append(logits, slices.Clone(seed))
		for j := range cont {
			next := argmaxF(logits[len(logits)-1])
			if toks != nil {
				next = toks[j]
			}
			greedy = append(greedy, next)
			lg, err := a.Forward(m.EmbedResidentForTest(next), len(ids)+j)
			if err != nil {
				t.Fatal(err)
			}
			logits = append(logits, slices.Clone(lg))
		}
		return logits, greedy
	}
	score := func(ref, cand [][]float32) []fidelity.Position {
		var ps []fidelity.Position
		for i := range ref {
			agree, _, hard := decoder.NearTieArgmaxForTest(ref[i], cand[i])
			ps = append(ps, fidelity.Position{Agree: argmaxF(ref[i]) == argmaxF(cand[i]), HardFlip: !agree && hard, KL: decoder.KLDivergenceForTest(ref[i], cand[i])})
		}
		return ps
	}
	_, files := decoder.PrefillGatePromptSet()
	var cells []fidelity.Cell
	var allIDs [][]int
	start := time.Now()
	for _, K := range []int{256, 512, 1024, 3900} {
		cell := fidelity.Cell{Name: fmt.Sprintf("S/K=%d", K)}
		for fi, f := range files {
			ids := decoder.PrefillGateProseIDsForTest(t, tk, f, K)[:K]
			allIDs = append(allIDs, ids)
			ref16, toks16 := arm(m16, a16, ids, true, nil)
			cand16, _ := arm(m16, a16, ids, false, toks16)
			ref8, toks8 := arm(m8, a8, ids, true, nil)
			cand8, _ := arm(m8, a8, ids, false, toks8)
			p := fidelity.Prompt{ID: fmt.Sprintf("%s/%d", cell.Name, fi), Candidate: score(ref8, cand8), Exact: score(ref16, cand16)}
			cell.Prompts = append(cell.Prompts, p)
			fmt.Fprintf(os.Stderr, "[P3B2-G2] K=%d prompt %d/%d done; elapsed %.0fs\n", K, fi+1, len(files), time.Since(start).Seconds())
		}
		cells = append(cells, cell)
	}
	v, err := fidelity.NonInferior(cells, fidelity.Criteria{AgreeMarginPts: 1, KLMargin: 0.10, Alpha: 0.05})
	if err != nil {
		t.Fatal(err)
	}
	if lp := os.Getenv("P3B2_POSITIONS_LOG"); lp != "" {
		if f, err := os.Create(lp); err == nil {
			_ = fidelity.WritePositions(f, cells)
			f.Close()
		}
	}
	for _, c := range cells {
		s := v.PerCell[c.Name]
		fmt.Fprintf(os.Stderr, "[P3B2-G2] %s: agree i8 %.2f%% / f16 %.2f%%, mean KL i8 %.4g / f16 %.4g, hard flips %d / %d\n",
			c.Name, s.AgreeCandPct, s.AgreeExactPct, s.MeanKLCand, s.MeanKLExact, s.FlipsCand, s.FlipsExact)
	}
	p := v.Pooled
	fmt.Fprintf(os.Stderr, "[P3B2-G2] pooled over %d prompts, %d positions (prompt set %s): agreement diff %+.2f pts (LCB %+.2f), KL ratio %.3f (UCB %.3f), hard flips %d vs %d -> pass %v %v\n",
		p.Prompts, p.Positions, fidelity.PromptSetHash(allIDs), p.AgreeDiffPts, p.AgreeDiffLCB, p.KLRatio, p.KLRatioUCB, p.FlipsCand, p.FlipsExact, v.Pass, v.Reasons)
	if !v.Pass {
		t.Errorf("P3B2-G2 FAIL: %v", v.Reasons)
	}
}

// TestP3b2_int8PassAgainstCommonReference is P3B2-G2 as amended (D10): the same model, prompts and cells, every arm teacher-
// forced on the f16 step's greedy tokens and scored against the f16 step (the shipped sequential decode's numerics), so the int8
// pass (candidate) and the int8 step (exact) are graded against one reference. Same criteria, reported. Not timed.
//
//	GOINFER_P3B2_DIAG=1 go test -count=1 -tags goinfer_testhooks -timeout 60m -run '^TestP3b2_int8PassAgainstCommonReference$' -v ./metal/
func TestP3b2_int8PassAgainstCommonReference(t *testing.T) {
	if os.Getenv("GOINFER_P3B2_DIAG") == "" {
		t.Skip("set GOINFER_P3B2_DIAG=1 (two loads of the 1.5B, minutes)")
	}
	home, _ := os.UserHomeDir()
	path := filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	giw := decoder.SidecarPath(path, "int4", "metal", false)
	ctx, cancel := context.WithTimeout(context.Background(), 55*time.Minute)
	defer cancel()
	load := func(kv string) (*decoder.Model, *metalResident) {
		m, err := decoder.Load(giw, decoder.Options{Backend: "metal", Quant: "int4", KVPrecision: kv, KVQuant: map[string]string{"": "f32", "i8": "i8"}[kv], ResidentContext: 4096})
		if err != nil {
			t.Fatalf("load --kv %q: %v", kv, err)
		}
		t.Cleanup(func() { m.Close() })
		a, _ := m.ResidentForwardForTest().(*metalResident)
		return m, a
	}
	m16, a16 := load("")
	m8, a8 := load("i8")
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	const cont = 64
	arm := func(m *decoder.Model, a *metalResident, ids []int, step bool, toks []int) (logits [][]float32, greedy []int) {
		embs := make([][]float32, len(ids))
		for i, id := range ids {
			embs[i] = m.EmbedResidentForTest(id)
		}
		a.Reset()
		var seed []float32
		var err error
		if step {
			seed, err = a.prefillByStep(ctx, embs, 0)
		} else {
			seed, err = a.PrefillLast(ctx, embs, 0)
		}
		if err != nil {
			t.Fatal(err)
		}
		logits = append(logits, slices.Clone(seed))
		for j := range cont {
			next := argmaxF(logits[len(logits)-1])
			if toks != nil {
				next = toks[j]
			}
			greedy = append(greedy, next)
			lg, err := a.Forward(m.EmbedResidentForTest(next), len(ids)+j)
			if err != nil {
				t.Fatal(err)
			}
			logits = append(logits, slices.Clone(lg))
		}
		return logits, greedy
	}
	score := func(ref, cand [][]float32) []fidelity.Position {
		var ps []fidelity.Position
		for i := range ref {
			agree, _, hard := decoder.NearTieArgmaxForTest(ref[i], cand[i])
			ps = append(ps, fidelity.Position{Agree: argmaxF(ref[i]) == argmaxF(cand[i]), HardFlip: !agree && hard, KL: decoder.KLDivergenceForTest(ref[i], cand[i])})
		}
		return ps
	}
	_, files := decoder.PrefillGatePromptSet()
	var int8Cells, f16Cells []fidelity.Cell
	start := time.Now()
	for _, K := range []int{256, 512, 1024, 3900} {
		c8, c16 := fidelity.Cell{Name: fmt.Sprintf("S/K=%d", K)}, fidelity.Cell{Name: fmt.Sprintf("S/K=%d", K)}
		for fi, f := range files {
			ids := decoder.PrefillGateProseIDsForTest(t, tk, f, K)[:K]
			ref, toks := arm(m16, a16, ids, true, nil)
			pass16, _ := arm(m16, a16, ids, false, toks)
			step8, _ := arm(m8, a8, ids, true, toks)
			pass8, _ := arm(m8, a8, ids, false, toks)
			id := fmt.Sprintf("%s/%d", c8.Name, fi)
			c8.Prompts = append(c8.Prompts, fidelity.Prompt{ID: id, Candidate: score(ref, pass8), Exact: score(ref, step8)})
			c16.Prompts = append(c16.Prompts, fidelity.Prompt{ID: id, Candidate: score(ref, pass16), Exact: score(ref, ref)})
			fmt.Fprintf(os.Stderr, "[P3B2-diag] K=%d prompt %d/%d; elapsed %.0fs\n", K, fi+1, len(files), time.Since(start).Seconds())
		}
		int8Cells, f16Cells = append(int8Cells, c8), append(f16Cells, c16)
	}
	v, err := fidelity.NonInferior(int8Cells, fidelity.Criteria{AgreeMarginPts: 1, KLMargin: 0.10, Alpha: 0.05})
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range int8Cells {
		s := v.PerCell[c.Name]
		var f16KL float64
		for _, p := range f16Cells[slices.IndexFunc(f16Cells, func(x fidelity.Cell) bool { return x.Name == c.Name })].Prompts {
			for _, q := range p.Candidate {
				f16KL += q.KL
			}
		}
		f16KL /= float64(len(c.Prompts) * (cont + 1))
		fmt.Fprintf(os.Stderr, "[P3B2-diag] %s against the f16 step: agree i8 pass %.2f%% / i8 step %.2f%%, mean KL i8 pass %.4g / i8 step %.4g (f16 pass %.4g), hard flips %d / %d\n",
			c.Name, s.AgreeCandPct, s.AgreeExactPct, s.MeanKLCand, s.MeanKLExact, f16KL, s.FlipsCand, s.FlipsExact)
	}
	p := v.Pooled
	fmt.Fprintf(os.Stderr, "[P3B2-diag] pooled, i8 pass against i8 step, both scored against the f16 step: agreement diff %+.2f pts (LCB %+.2f), KL ratio %.3f (UCB %.3f), hard flips %d vs %d -> non-inferior %v %v\n",
		p.AgreeDiffPts, p.AgreeDiffLCB, p.KLRatio, p.KLRatioUCB, p.FlipsCand, p.FlipsExact, v.Pass, v.Reasons)
}
