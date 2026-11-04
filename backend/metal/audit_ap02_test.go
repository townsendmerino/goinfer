//go:build darwin

package metal

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestAuditAP02_shortPromptTiming is A-P02's speed half (docs/tasks/task-metal-audit-2026-10.md, "A-P02:
// pre-registration"): a fresh K-token prompt's wall time three ways, as T1.10 timed it, but on the build A-P01 changed
// (T1.10's step / pass figures no longer describe the pass):
// the sequential single-token loop (what a resident without the batched step runs below the floor), the batched step in
// 8-row pieces (what E-P01 runs below the floor where the step exists), and the batched pass with its floor off. K =
// 16, 32, 48 and 64, arms interleaved in a rotating order rep by rep, each writing positions 0..K-1 of slot 0. The
// readings are per-rep paired ratios: sequential / pass and step / pass, each > 1 when the pass is faster.
func TestAuditAP02_shortPromptTiming(t *testing.T) {
	auditA(t)
	name, a := auditLoad(t, "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", 2, 1024,
		&decoder.Knobs{"GOINFER_METAL_FAST_PREFILL_FLOOR": "0"})
	r, t0 := a.r, time.Now()
	if r.batch == nil {
		t.Skipf("%s builds no batched step (batchIneligible: %q)", name, r.batchIneligible())
	}
	reps := auditReps(7)
	arms := []string{"sequential", "step", "pass"}
	for _, K := range []int{16, 32, 48, 64} {
		embs := auditEmbs(r, K, K)
		ms := map[string][]float64{}
		for rep := range reps {
			for k := range arms {
				arm := arms[(k+rep)%len(arms)]
				if err := r.useKVSlot(0); err != nil {
					t.Fatal(err)
				}
				st := time.Now()
				switch arm {
				case "sequential":
					for i, e := range embs {
						r.ForwardEmb(e, i)
					}
				case "step":
					for c := 0; c < K; c += 8 {
						var seqs []batchSeq
						for i := c; i < min(c+8, K); i++ {
							seqs = append(seqs, batchSeq{slot: 0, pos: i, emb: embs[i]})
						}
						if _, _, err := r.forwardMulti(seqs); err != nil {
							t.Fatalf("step at %d: %v", c, err)
						}
					}
				case "pass":
					if _, err := a.PrefillLast(context.Background(), embs, 0); err != nil {
						t.Fatalf("pass K=%d: %v", K, err)
					}
				}
				ms[arm] = append(ms[arm], time.Since(st).Seconds()*1e3)
			}
		}
		seqPass, stepPass := make([]float64, reps), make([]float64, reps)
		seqAbove15, stepAbove1 := 0, 0
		for i := range reps {
			seqPass[i] = ms["sequential"][i] / ms["pass"][i]
			stepPass[i] = ms["step"][i] / ms["pass"][i]
			if seqPass[i] > 1.5 {
				seqAbove15++
			}
			if stepPass[i] > 1 {
				stepAbove1++
			}
		}
		auditHB("a-p02", t0, "%s K=%d: sequential %.1f ms, step %.1f ms, pass %.1f ms (medians of %d); per rep sequential/pass %s, step/pass %s",
			name, K, auditMedian(ms["sequential"]), auditMedian(ms["step"]), auditMedian(ms["pass"]), reps, auditFmt3(seqPass), auditFmt3(stepPass))
		auditHB("a-p02", t0, "RESULT %s K=%d: sequential/pass median %.3f (%d of %d reps above 1.5); step/pass median %.3f (%d of %d reps above 1)",
			name, K, auditMedian(seqPass), seqAbove15, reps, auditMedian(stepPass), stepAbove1, reps)
	}
}

func auditFmt3(xs []float64) string {
	s := ""
	for i, x := range xs {
		if i > 0 {
			s += " "
		}
		s += fmt.Sprintf("%.3f", x)
	}
	return s
}
