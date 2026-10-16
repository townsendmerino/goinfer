//go:build darwin

package metal

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"testing"
	"time"
)

// TestQ4KStep_phi3Tier1 is OP6's Metal tier 1 (docs/tasks/task-forward-architecture-2026-10.md §4.7, registered before any
// code): Phi-3 mini 4k at --quant q4k on Metal.
//
//   - Graded: the step route against decode's sequential loop, bit for bit on the last row's logits and the K/V, on the
//     golden's prompt and on a 200-token prompt.
//
//   - Graded: the step's argmax on the golden's prompt equals phi3_mini_golden.json's, as TestPhi3GGUFReal_gate grades
//     this file. The cosine is logged only: the GGUF is an earlier Phi-3 snapshot than the golden.
//
//   - Speed: PrefillLast on the 200-token prompt against the loop the decoder runs when a prefill declines
//     (ForwardNoLogits, then Forward on the last row), interleaved, 5 pairs after a warm-up pair. Ship: every pair's
//     ratio above 1 and the median at least 1.02; park: a median of 1.00 to 1.02; off: below 1.00.
//
//     GOINFER_HEAVY_TESTS=1 python3 scripts/timing_lock.py run --label op6-tier1 -- \
//     go test -count=1 -timeout 10m -run '^TestQ4KStep_phi3Tier1$' -v ./metal/
func TestQ4KStep_phi3Tier1(t *testing.T) {
	requireHeavyModel(t)
	a := q4kStepLoad(t, "GOINFER_PHI3_GGUF", "phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf", 2, 512) // 200-token prompts: 512 positions, not the Mac default
	r := a.r
	raw, err := os.ReadFile("../testdata/phi3_mini_golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		PromptIDs  []int     `json:"prompt_ids"`
		Argmax     int       `json:"argmax"`
		LastLogits []float32 `json:"last_logits"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	long := q4kStepIDs(2718, 200) // ids under 20000, inside Phi-3's 32064
	for _, c := range []struct {
		name string
		ids  []int
	}{{"golden prompt", g.PromptIDs}, {"200 tokens", long}} {
		lg, kv := q4kStepPrompt(t, a, c.ids, 0)
		t.Logf("%s: %d of %d last-row logits and %d K/V elements differ from the sequential loop", c.name, lg, r.V, kv)
		if lg != 0 || kv != 0 {
			t.Errorf("%s: the step differs from sequential decode (%d logits, %d K/V elements)", c.name, lg, kv)
		}
	}

	embs := func(ids []int) [][]float32 {
		out := make([][]float32, len(ids))
		for i, id := range ids {
			out[i] = mc3Emb(r, id)
		}
		return out
	}
	gold := embs(g.PromptIDs)
	if err := r.useKVSlot(1); err != nil {
		t.Fatal(err)
	}
	got, err := a.PrefillLast(context.Background(), gold, 0)
	if err != nil {
		t.Fatal(err)
	}
	am := argmaxF32(got)
	var dot, na, nb float64
	for i := range got {
		dot += float64(got[i]) * float64(g.LastLogits[i])
		na += float64(got[i]) * float64(got[i])
		nb += float64(g.LastLogits[i]) * float64(g.LastLogits[i])
	}
	t.Logf("golden prompt: argmax %d, golden %d; cosine %.4f (logged: an earlier snapshot)", am, g.Argmax, dot/math.Sqrt(na*nb))
	if am != g.Argmax {
		t.Errorf("the step's argmax on the golden prompt is %d, the golden's %d", am, g.Argmax)
	}

	// The speed A/B, on slot 1. Each arm writes the same positions, so neither leaves the other a warmer cache of its own.
	e := embs(long)
	step := func() time.Duration {
		t0 := time.Now()
		if _, err := a.PrefillLast(context.Background(), e, 0); err != nil {
			t.Fatal(err)
		}
		return time.Since(t0)
	}
	seq := func() time.Duration {
		t0 := time.Now()
		for i, x := range e[:len(e)-1] {
			if err := a.ForwardNoLogits(x, i); err != nil {
				t.Fatal(err)
			}
		}
		if _, err := a.Forward(e[len(e)-1], len(e)-1); err != nil {
			t.Fatal(err)
		}
		return time.Since(t0)
	}
	step()
	seq()
	var ratios []float64
	allAbove := true
	for p := range 5 {
		var ts, tq time.Duration
		if p%2 == 0 {
			ts, tq = step(), seq()
		} else {
			tq, ts = seq(), step()
		}
		ratio := tq.Seconds() / ts.Seconds()
		ratios = append(ratios, ratio)
		allAbove = allAbove && ratio > 1
		fmt.Fprintf(os.Stderr, "[op6-tier1] pair %d: step %.1f ms (%.0f tok/s), sequential %.1f ms (%.0f tok/s), ratio %.3f\n",
			p+1, ts.Seconds()*1e3, float64(len(e))/ts.Seconds(), tq.Seconds()*1e3, float64(len(e))/tq.Seconds(), ratio)
	}
	sort.Float64s(ratios)
	med := ratios[len(ratios)/2]
	verdict := "OFF"
	switch {
	case allAbove && med >= 1.02:
		verdict = "SHIP"
	case med >= 1.00:
		verdict = "PARK"
	}
	t.Logf("speed: median sequential/step %.3f over 5 pairs, every pair above 1: %v — %s", med, allAbove, verdict)
	if verdict != "SHIP" {
		t.Errorf("the step route's speed verdict is %s, not SHIP", verdict)
	}
}
