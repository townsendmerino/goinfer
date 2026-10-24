package decoder

import (
	"context"
	"fmt"
	"os"
	"sync/atomic"
	"testing"
	"time"
)

// TestMLABatch_matchesSequential holds the batched walker's opMLA (mlaBatch) to the per-token walker on deepseek-tiny: every logit
// of every position and every stored latent BIT-IDENTICAL, over one pass and over two (the second reads the first's latents), with
// f32 weights, with int8 experts (how the CPU fidelity references load), and with int8 MLA projections (the row-by-row form).
// mlaBatchRuns proves the batched form ran where it should and did not where it should not.
func TestMLABatch_matchesSequential(t *testing.T) {
	const ckpt = "../testdata/deepseek-tiny"
	ids := make([]int, 60) // deepseek-tiny holds 64 positions
	for i := range ids {
		ids[i] = 3 + (i*37)%120
	}
	for _, c := range []struct {
		name    string
		quant   string
		int8MLA bool
		batched bool
	}{
		{"f32", "", false, true},
		{"int8 experts", "int8", false, true},
		{"int8 MLA projections", "int8", true, false},
	} {
		for _, split := range []int{0, 36} {
			t.Run(fmt.Sprintf("%s/split=%d", c.name, split), func(t *testing.T) {
				m := mlaLoad(t, ckpt, c.quant, !c.int8MLA)
				if !m.canBatchN(len(ids)) {
					t.Fatal("deepseek-tiny is not batch-eligible: the batched walker never sees opMLA")
				}
				cseq := m.NewCache(len(ids))
				seq := make([][]float32, len(ids))
				for i, id := range ids {
					l, err := m.forward(id, cseq)
					if err != nil {
						t.Fatalf("forward %d: %v", i, err)
					}
					seq[i] = append([]float32(nil), l...)
				}
				before := mlaBatchRuns.Load()
				cbat := m.NewCache(len(ids))
				var bat [][]float32
				for _, part := range [][]int{ids[:split], ids[split:]} {
					if len(part) == 0 {
						continue
					}
					out, err := m.forwardN(context.Background(), part, cbat)
					if err != nil {
						t.Fatalf("forwardN: %v", err)
					}
					for _, r := range out { // forwardN's rows may be views of a buffer the next call reuses
						bat = append(bat, append([]float32(nil), r...))
					}
				}
				if ran := mlaBatchRuns.Load() - before; (ran > 0) != c.batched {
					t.Fatalf("mlaBatch's batched form ran %d times, want batched=%v", ran, c.batched)
				}
				if cseq.Pos() != cbat.Pos() || cbat.Pos() != len(ids) {
					t.Fatalf("positions: sequential %d, batched %d, want %d", cseq.Pos(), cbat.Pos(), len(ids))
				}
				nd := 0
				for i := range seq {
					for j := range seq[i] {
						if seq[i][j] != bat[i][j] {
							if nd == 0 {
								t.Errorf("position %d logit %d: sequential %v, batched %v", i, j, seq[i][j], bat[i][j])
							}
							nd++
						}
					}
				}
				for l := range m.w.Layers {
					a, b := cseq.Latent(l), cbat.Latent(l)
					if len(a) != len(b) {
						t.Fatalf("layer %d: %d latent floats sequential, %d batched", l, len(a), len(b))
					}
					for j := range a {
						if a[j] != b[j] {
							nd++
							t.Errorf("layer %d latent %d: sequential %v, batched %v", l, j, a[j], b[j])
							break
						}
					}
				}
				if nd > 0 {
					t.Fatalf("%d values differ: the batched MLA is not the per-token walker's", nd)
				}
			})
		}
	}
}

// TestMoEExpertMajor_sharedExpert holds the expert-major MoE's shared expert to the per-row path on the two forms moeMLP adds it in:
// sigmoid-gated (tiny-qwen2-moe) and ungated (deepseek-tiny). Every logit of a 60-token batched forward is BIT-IDENTICAL with the
// expert-major knob on and off, and moeExpertMajorRuns proves the expert-major arm ran.
func TestMoEExpertMajor_sharedExpert(t *testing.T) {
	for _, c := range []struct{ name, dir string }{
		{"gated shared expert", "../testdata/tiny-qwen2-moe"},
		{"ungated shared expert", "../testdata/deepseek-tiny"},
	} {
		t.Run(c.name, func(t *testing.T) {
			m := mlaLoad(t, c.dir, "", true)
			if a := m.w.arch.MoE; a == nil || a.SharedIntermediateDim == 0 {
				t.Fatalf("%s has no shared expert: this test would prove nothing", c.dir)
			}
			ids := make([]int, 60) // within both fixtures' 64 positions
			for i := range ids {
				ids[i] = 3 + (i*37)%(m.w.arch.VocabSize-3)
			}
			run := func(on string) [][]float32 {
				setKnob(t, m, knobMoEExpertMajor, on)
				out, err := m.forwardN(context.Background(), ids, m.NewCache(len(ids)))
				if err != nil {
					t.Fatalf("forwardN (expert-major=%q): %v", on, err)
				}
				return out
			}
			off := run("0")
			before := atomic.LoadInt64(&moeExpertMajorRuns)
			on := run("1")
			if atomic.LoadInt64(&moeExpertMajorRuns) == before {
				t.Fatal("the expert-major path never ran: both arms took the per-row path")
			}
			nd := 0
			for i := range off {
				for j := range off[i] {
					if off[i][j] != on[i][j] {
						if nd == 0 {
							t.Errorf("position %d logit %d: per-row %v, expert-major %v", i, j, off[i][j], on[i][j])
						}
						nd++
					}
				}
			}
			if nd > 0 {
				t.Fatalf("%d logits differ: the expert-major shared expert is not the per-row one", nd)
			}
		})
	}
}

// TestMLABatch_realCheckpoint is the tier-1 read of the batched MLA prefill on a real MLA checkpoint, loaded as the CPU fidelity
// references load it (int8 experts, f32 MLA projections): K prompt tokens through the per-token walker and through the batched
// prefill, every final logit and every stored latent BIT-IDENTICAL, with both wall times printed. GOINFER_HEAVY_TESTS=1 arms it;
// GOINFER_MLA_BATCH_CKPT (default ~/models/deepseek-v2-lite) and GOINFER_MLA_BATCH_K (default 512) set the cell.
func TestMLABatch_realCheckpoint(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (a real MLA checkpoint on the CPU)")
	}
	t.Setenv("GOINFER_CPU_FAST_ATTENTION", "0")
	path := os.Getenv("GOINFER_MLA_BATCH_CKPT")
	if path == "" {
		path = os.ExpandEnv("$HOME/models/deepseek-v2-lite")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s", path)
	}
	K := 512
	if v := os.Getenv("GOINFER_MLA_BATCH_K"); v != "" {
		fmt.Sscanf(v, "%d", &K)
	}
	m, err := Load(path, Options{Backend: "cpu", Quant: "int8", ResidentContext: K + 8})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	ids := make([]int, K)
	for i := range ids {
		ids[i] = 100 + (i*7919)%(m.w.arch.VocabSize-200)
	}
	t0 := time.Now()
	cseq := m.NewCache(K)
	var seq []float32
	for i, id := range ids {
		l, err := m.forward(id, cseq)
		if err != nil {
			t.Fatalf("forward %d: %v", i, err)
		}
		seq = l
		if (i+1)%64 == 0 {
			fmt.Fprintf(os.Stderr, "[mla-batch] per-token %d/%d, elapsed %s\n", i+1, K, time.Since(t0).Round(time.Second))
		}
	}
	seq = append([]float32(nil), seq...)
	dSeq := time.Since(t0)
	before := mlaBatchRuns.Load()
	t1 := time.Now()
	cbat := m.NewCache(K)
	bat, err := m.prefillLogits(context.Background(), ids, cbat)
	if err != nil {
		t.Fatalf("prefillLogits: %v", err)
	}
	dBat := time.Since(t1)
	if mlaBatchRuns.Load() == before {
		t.Fatal("the batched MLA form never ran: the prefill took the per-token walker")
	}
	fmt.Fprintf(os.Stderr, "[mla-batch] %s K=%d: per-token %s, batched %s (%.1fx)\n", path, K, dSeq.Round(time.Second), dBat.Round(time.Second),
		dSeq.Seconds()/dBat.Seconds())
	nd := 0
	for j := range seq {
		if seq[j] != bat[j] {
			nd++
		}
	}
	for l := range m.w.Layers {
		a, b := cseq.Latent(l), cbat.Latent(l)
		if len(a) != len(b) {
			t.Fatalf("layer %d: latent lengths %d and %d", l, len(a), len(b))
		}
		for j := range a {
			if a[j] != b[j] {
				nd++
				break
			}
		}
	}
	if nd > 0 {
		t.Fatalf("%d values differ between the per-token and the batched prefill", nd)
	}
}
