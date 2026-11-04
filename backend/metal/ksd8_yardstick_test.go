//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"os"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestKSD8_yardstick (diagnostic, not a gate): D8-G3's instrument (8 prompts of 512 seeded token ids, last-position logits,
// batched pass against the sequential path) on families whose batched pass already ships, to read how much the two paths
// disagree on such prompts where nothing is wrong. GOINFER_KSD8_YARDSTICK=path[,path...] names the checkpoints.
func TestKSD8_yardstick(t *testing.T) {
	paths := os.Getenv("GOINFER_KSD8_YARDSTICK")
	if paths == "" {
		t.Skip("set GOINFER_KSD8_YARDSTICK to the checkpoints")
	}
	for _, p := range splitComma(paths) {
		m, err := decoder.Load(p, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 1024})
		if err != nil {
			t.Fatalf("load %s: %v", p, err)
		}
		a, ok := m.ResidentForwardForTest().(*metalResident)
		if !ok || !a.r.prefillOK {
			m.Close()
			t.Fatalf("%s: the batched pass is not admitted", p)
		}
		r := a.r
		agree, klSum := 0, 0.0
		for k := range 8 {
			ids := ksd8Ids(k, 512, r.V)
			r.resetRecurrent()
			var seq []float32
			for i, id := range ids {
				seq = r.Forward(id, i)
			}
			seq = slices.Clone(seq)
			r.resetRecurrent()
			pass := slices.Clone(r.PrefillLast(getEmbs(r, ids), 0))
			kl := (klLogits(seq, pass) + klLogits(pass, seq)) / 2
			klSum += kl
			if argmaxF(seq) == argmaxF(pass) {
				agree++
			}
			fmt.Fprintf(os.Stderr, "[yardstick] %s prompt %d: top-1 %d / %d, symmetric KL %.4e, cosine %.6f\n", p, k, argmaxF(seq), argmaxF(pass), kl, cosF(seq, pass))
		}
		fmt.Fprintf(os.Stderr, "[yardstick] %s: top-1 agrees on %d of 8, mean symmetric KL %.4e\n", p, agree, klSum/8)
		m.Close()
	}
}

func splitComma(s string) []string {
	var out []string
	for _, f := range slices.Collect(func(yield func(string) bool) {
		start := 0
		for i := 0; i <= len(s); i++ {
			if i == len(s) || s[i] == ',' {
				if !yield(s[start:i]) {
					return
				}
				start = i + 1
			}
		}
	}) {
		if f != "" {
			out = append(out, f)
		}
	}
	return out
}
