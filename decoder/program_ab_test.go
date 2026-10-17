//go:build goinfer_testhooks

package decoder

import (
	"context"
	"fmt"
	"math"
	"os"
	"sort"
	"testing"
	"time"
)

// TestProgram_S_walkerSpeed is OP3's gate S (docs/tasks/task-forward-architecture-2026-10.md §4.8): per-token decode through the
// program walker against the loop it replaced (programWalkOffForTest), interleaved ABBA in one process on the 0.5B at depths 128
// and 4,096. The quantity is the paired ratio old ms/token ÷ walker ms/token, the decode-rate ratio walker ÷ old; the bar is a
// median of at least 0.99 in both cells. The harness fails on any difference in the greedy streams.
//
//	GOINFER_HEAVY_TESTS=1 python3 scripts/timing_lock.py run --label op3-S -- \
//	  go test -tags goinfer_testhooks ./decoder/ -run TestProgram_S_walkerSpeed -v -count=1 -timeout 30m
func TestProgram_S_walkerSpeed(t *testing.T) {
	for _, mc := range []struct {
		name  string
		depth int
		pairs int
	}{
		{"0.5B-d128", 128, 6},
		{"0.5B-d4096", 4096, 4},
	} {
		t.Run(mc.name, func(t *testing.T) {
			h := newCPUDecodeAB(t, "GOINFER_CPU_MODEL_05B", "$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", mc.depth)
			fmt.Fprintf(os.Stderr, "[S %s] depth %d, %d pairs\n", mc.name, len(h.ids), mc.pairs)
			t.Cleanup(func() { programWalkOffForTest = false })
			med := h.run("walker ("+mc.name+")", mc.pairs, func(walker bool) { programWalkOffForTest = !walker })
			fmt.Fprintf(os.Stderr, "[S %s] paired median rate walker/old %.3f (bar 0.99)\n", mc.name, med)
		})
	}
}

// TestProgram_S_batchedSpeed is OP3b's gate S (§4.8, "OP3b's gate S"): the batched walker's prefill rate at prompts of 128 and
// 4,096 tokens, and the multi-sequence walker's step rate for four sequences at depth 128, each against the loop it replaced
// (programWalkOffForTest), interleaved ABBA in one process. Each pair's two arms must give the same logits, bit for bit.
//
//	GOINFER_HEAVY_TESTS=1 python3 scripts/timing_lock.py run --label op3b-S -- \
//	  go test -tags goinfer_testhooks ./decoder/ -run TestProgram_S_batchedSpeed -v -count=1 -timeout 30m
func TestProgram_S_batchedSpeed(t *testing.T) {
	t.Cleanup(func() { programWalkOffForTest = false })
	abba := func(t *testing.T, name string, pairs int, arm func(walker bool) (ms float64, logits []float32)) {
		arm(true) // warm-up
		var ratios []float64
		for p := range pairs {
			order := []bool{true, false}
			if p%2 == 1 {
				order = []bool{false, true}
			}
			var msW, msO float64
			var lgW, lgO []float32
			for _, w := range order {
				ms, lg := arm(w)
				if w {
					msW, lgW = ms, lg
				} else {
					msO, lgO = ms, lg
				}
			}
			for i := range lgW {
				if math.Float32bits(lgW[i]) != math.Float32bits(lgO[i]) {
					t.Fatalf("%s pair %d: logit %d differs, walker %v old %v", name, p, i, lgW[i], lgO[i])
				}
			}
			ratios = append(ratios, msO/msW)
			fmt.Fprintf(os.Stderr, "[S %s] pair %d/%d: walker %.3f | old %.3f ms | rate walker/old %.3f\n", name, p+1, pairs, msW, msO, msO/msW)
		}
		sort.Float64s(ratios)
		med := ratios[len(ratios)/2]
		if len(ratios)%2 == 0 {
			med = (ratios[len(ratios)/2-1] + ratios[len(ratios)/2]) / 2
		}
		under := 0
		for _, r := range ratios {
			if r < 1 {
				under++
			}
		}
		t.Logf("S %s: paired median rate walker/old %.3f (min %.3f max %.3f, n=%d, %d under 1; bar 0.99)", name, med, ratios[0], ratios[len(ratios)-1], pairs, under)
	}
	for _, mc := range []struct {
		depth, pairs int
	}{{128, 6}, {4096, 4}} {
		name := fmt.Sprintf("prefill-%d", mc.depth)
		t.Run(name, func(t *testing.T) {
			h := newCPUDecodeAB(t, "GOINFER_CPU_MODEL_05B", "$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", mc.depth)
			if !h.m.canBatchN(len(h.ids)) {
				t.Fatal("the 0.5B does not take the batched prefill")
			}
			abba(t, name, mc.pairs, func(walker bool) (float64, []float32) {
				programWalkOffForTest = !walker
				cache := h.m.NewCache(len(h.ids) + 1)
				t0 := time.Now()
				lg, err := h.m.prefillLogits(context.Background(), h.ids, cache)
				if err != nil {
					t.Fatalf("prefill: %v", err)
				}
				return float64(time.Since(t0).Microseconds()) / 1000, append([]float32(nil), lg...)
			})
		})
	}
	t.Run("multi-4x128", func(t *testing.T) {
		h := newCPUDecodeAB(t, "GOINFER_CPU_MODEL_05B", "$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", 128)
		if err := h.m.cpuBatchModelEligible(); err != nil {
			t.Fatal(err)
		}
		const B, steps = 4, 24
		abba(t, "multi-4x128", 6, func(walker bool) (float64, []float32) {
			programWalkOffForTest = false // the caches' prefill is not what is timed
			caches := make([]*KVCache, B)
			ids := make([]int, B)
			for b := range caches {
				caches[b] = h.m.NewCache(len(h.ids) + steps + 1)
				lg, err := h.m.prefillLogits(context.Background(), h.ids[:len(h.ids)-b], caches[b])
				if err != nil {
					t.Fatalf("prefill: %v", err)
				}
				ids[b] = argmax(lg)
			}
			programWalkOffForTest = !walker
			var last []float32
			t0 := time.Now()
			for range steps {
				rows, err := h.m.decodeMultiStep(ids, caches)
				if err != nil {
					t.Fatalf("step: %v", err)
				}
				for b := range ids {
					ids[b] = argmax(rows[b])
				}
				last = rows[0]
			}
			return float64(time.Since(t0).Microseconds()) / 1000 / steps, append([]float32(nil), last...)
		})
	})
}
