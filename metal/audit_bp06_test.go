//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"fmt"
	"math"
	"os"
	"testing"
	"time"
)

// TestBP06_rowsTG is B-P06's in-sequence probe (docs/audit-metal-2026-09-30.md, "B-P06"): the dense decode sites' R18
// rows kernels at threadgroups of 64, 128, 256 (shipped) and 512 threads (gemvRowsTG), whole decode tokens from the
// same positions after a 128-token prefill, arms rotated rep by rep, each arm's time its token GPU-time median, the
// logits compared across arms every token. GOINFER_AUDIT_MODEL picks the checkpoint (default the 1.5B). The audit's
// kill line: below 1.03x on the 1.5B.
//
//	GOINFER_BP06=1 go test -tags goinfer_testhooks -count=1 -run '^TestBP06_rowsTG$' -v ./metal/
func TestBP06_rowsTG(t *testing.T) {
	if os.Getenv("GOINFER_BP06") != "1" {
		t.Skip("set GOINFER_BP06=1 (loads a real checkpoint)")
	}
	const depth, tokens = 128, 16
	name, a := auditLoad(t, "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", 1, depth+tokens+64, nil)
	r, t0 := a.r, time.Now()
	if r.gemvRows.gu == 0 {
		t.Fatalf("%s: the R18 rows kernels are not selected", name)
	}
	defer func() { gemvRowsTG = 256; r.stopExec() }()
	embs := auditEmbs(r, depth+tokens, 11)
	if _, err := a.PrefillLast(context.Background(), embs[:depth], 0); err != nil {
		t.Fatalf("prefill: %v", err)
	}
	arms := []int{256, 64, 128, 512}
	var ref [][]float32
	times := make([][]float64, len(arms))
	reps := auditReps(7)
	for rep := range reps + 1 {
		for k := range arms {
			ai := (k + rep) % len(arms)
			gemvRowsTG = arms[ai]
			r.stopExec()
			var ks []float64
			for i := range tokens {
				lg := r.ForwardEmb(embs[depth+i], depth+i)
				if err := r.takeExecErr(); err != nil {
					t.Fatalf("TG %d: %v", arms[ai], err)
				}
				if ref == nil || len(ref) <= i {
					ref = append(ref, append([]float32(nil), lg...))
				} else {
					for j := range lg {
						if math.Float32bits(lg[j]) != math.Float32bits(ref[i][j]) {
							t.Fatalf("TG %d token %d: logit %d differs from the first arm's", arms[ai], i, j)
						}
					}
				}
				ks = append(ks, (r.gpuEnd-r.gpuStart)*1e3)
			}
			if rep > 0 {
				times[ai] = append(times[ai], auditMedian(ks))
			}
		}
	}
	for ai := 1; ai < len(arms); ai++ {
		ratio := make([]float64, reps)
		above := 0
		for i := range ratio {
			ratio[i] = times[0][i] / times[ai][i]
			if ratio[i] > 1 {
				above++
			}
		}
		auditHB("b-p06", t0, "%s TG %d: token %.3f ms against TG 256's %.3f ms; RESULT TG256/TG%d median %.3f, %d of %d reps above 1 (per rep %s)",
			name, arms[ai], auditMedian(times[ai]), auditMedian(times[0]), arms[ai], auditMedian(ratio), above, reps, auditFmt3(ratio))
	}
	fmt.Fprintf(os.Stderr, "[b-p06] rows R: qkv %d, o %d, gate|up %d, down %d\n", r.gemvRows.qkv, r.gemvRows.o, r.gemvRows.gu, r.gemvRows.down)
}
