//go:build goinfer_testhooks

package decoder

import (
	"fmt"
	"os"
	"sort"
	"testing"
	"time"
)

// TestKVF16_F3InProcessAB is gate F3's day half (docs/tasks/task-cpu-kv-f16-2026-10.md § F3): decode on one loaded model,
// the CPU cache f16 against f32, interleaved ABBA in one process, on the 0.5B and 1.5B at depths 128 and 4096. Each
// generation builds a fresh cache, so flipping kvH between generations switches the arm. The quantity is the paired
// ratio f32 ms/token ÷ f16 ms/token (the decode-rate ratio f16 ÷ f32), read against F3's bars; the streams are allowed
// to differ (f16 rounds the stored rows), and how far they agree is logged, not gated.
//
//	GOINFER_HEAVY_TESTS=1 python3 scripts/timing_lock.py run --label f3-inproc -- \
//	  go test -tags goinfer_testhooks ./decoder/ -run TestKVF16_F3InProcessAB -v -count=1 -timeout 30m
func TestKVF16_F3InProcessAB(t *testing.T) {
	start := time.Now()
	for _, mc := range []struct {
		name, env, def string
		depth, pairs   int
	}{
		{"0.5B-d128", "GOINFER_CPU_MODEL_05B", "$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", 128, 6},
		{"0.5B-d4096", "GOINFER_CPU_MODEL_05B", "$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", 4096, 4},
		{"1.5B-d128", "GOINFER_CPU_MODEL", "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", 128, 6},
		{"1.5B-d4096", "GOINFER_CPU_MODEL", "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", 4096, 4},
	} {
		t.Run(mc.name, func(t *testing.T) {
			h := newCPUDecodeAB(t, mc.env, mc.def, mc.depth)
			if !h.m.w.arch.kvF16OK() {
				t.Fatalf("%s: family does not take an f16 cache", mc.name)
			}
			set := func(f16 bool) {
				h.m.kvH = f16
				if got := h.m.NewCache(16).quant == kvqF16; got != f16 {
					t.Fatalf("arm switch did not take: f16=%v, cache f16=%v", f16, got)
				}
			}
			t.Cleanup(func() { h.m.kvH = false })
			progress := func(format string, args ...any) {
				fmt.Fprintf(os.Stderr, "[F3 %s +%s] %s\n", mc.name, time.Since(start).Round(time.Second), fmt.Sprintf(format, args...))
			}
			progress("depth %d (%d ids), %d pairs: warm-up", mc.depth, len(h.ids), mc.pairs)
			set(false)
			h.gen()
			var ratios []float64
			var sumH, sumF float64
			agree, total := 0, 0
			for p := range mc.pairs {
				var msH, msF float64
				var tH, tF []int
				if p%2 == 0 {
					set(true)
					msH, tH = h.gen()
					set(false)
					msF, tF = h.gen()
				} else {
					set(false)
					msF, tF = h.gen()
					set(true)
					msH, tH = h.gen()
				}
				same := 0
				for same < len(tH) && same < len(tF) && tH[same] == tF[same] {
					same++
				}
				agree, total = agree+same, total+len(tF)
				r := msF / msH
				ratios = append(ratios, r)
				sumH, sumF = sumH+msH, sumF+msF
				progress("pair %d/%d: f16 %.2f | f32 %.2f ms/tok | rate f16/f32 %.3f | stream prefix agree %d/%d",
					p+1, mc.pairs, msH, msF, r, same, len(tF))
			}
			sort.Float64s(ratios)
			med := ratios[len(ratios)/2]
			if len(ratios)%2 == 0 {
				med = (ratios[len(ratios)/2-1] + ratios[len(ratios)/2]) / 2
			}
			t.Logf("F3 %s: mean f16 %.2f f32 %.2f ms/tok | paired median rate f16/f32 %.3f (min %.3f max %.3f, n=%d) | stream prefix agreement %d/%d",
				mc.name, sumH/float64(mc.pairs), sumF/float64(mc.pairs), med, ratios[0], ratios[len(ratios)-1], mc.pairs, agree, total)
		})
	}
}
