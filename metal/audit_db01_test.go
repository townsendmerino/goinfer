//go:build darwin

package metal

import (
	"context"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestAuditDB01_prefillTiming is D-B01's speed half (docs/tasks/task-metal-audit-2026-10.md, "D-B01: pre-registration"):
// a fresh K-token prompt's wall time on a Gated-DeltaNet hybrid two ways, the sequential decode loop (what the hybrid
// runs today) and the batched pass (dnetPrefillOn), arms alternated rep by rep, the recurrent state reset before each.
// GOINFER_AUDIT_MODEL picks the checkpoint (default ~/models/qwen3.5-0.8b), GOINFER_DB01_KS the prompt lengths (default
// 128, 512, 2048). The reading is the per-rep paired sequential / pass.
func TestAuditDB01_prefillTiming(t *testing.T) {
	auditA(t)
	Ks := []int{128, 512, 2048}
	if v := os.Getenv("GOINFER_DB01_KS"); v != "" {
		Ks = nil
		for _, f := range strings.Split(v, ",") {
			if n, err := strconv.Atoi(strings.TrimSpace(f)); err == nil && n > 0 {
				Ks = append(Ks, n)
			}
		}
	}
	prevDnet := dnetPrefillOn
	dnetPrefillOn = true
	t.Cleanup(func() { dnetPrefillOn = prevDnet })
	name, a := auditLoad(t, "qwen3.5-0.8b", 1, slices.Max(Ks)+64, &decoder.Knobs{"GOINFER_METAL_FAST_PREFILL_FLOOR": "0"})
	r, t0 := a.r, time.Now()
	if r.dnet == nil || !r.prefillOK {
		t.Fatalf("%s: dnet %v, prefillOK %v: not a hybrid the pass admits", name, r.dnet != nil, r.prefillOK)
	}
	reps := auditReps(7)
	for _, K := range Ks {
		embs := auditEmbs(r, K, K)
		ms := map[string][]float64{}
		for rep := range reps {
			order := []string{"sequential", "pass"}
			if rep%2 == 1 {
				order = []string{"pass", "sequential"}
			}
			for _, arm := range order {
				r.resetDeltaNet()
				st := time.Now()
				if arm == "sequential" {
					for i, e := range embs {
						r.ForwardEmb(e, i)
					}
				} else if _, err := a.PrefillLast(context.Background(), embs, 0); err != nil {
					t.Fatalf("pass K=%d: %v", K, err)
				}
				ms[arm] = append(ms[arm], time.Since(st).Seconds()*1e3)
			}
		}
		ratio := make([]float64, reps)
		above110, above2 := 0, 0
		for i := range reps {
			ratio[i] = ms["sequential"][i] / ms["pass"][i]
			if ratio[i] > 1.10 {
				above110++
			}
			if ratio[i] > 2 {
				above2++
			}
		}
		auditHB("d-b01", t0, "%s K=%d: sequential %.1f ms, pass %.1f ms (medians of %d); per rep sequential/pass %s",
			name, K, auditMedian(ms["sequential"]), auditMedian(ms["pass"]), reps, auditFmt3(ratio))
		auditHB("d-b01", t0, "RESULT %s K=%d: sequential/pass median %.3f (%d of %d reps above 1.10, %d above 2)",
			name, K, auditMedian(ratio), above110, reps, above2)
	}
}
