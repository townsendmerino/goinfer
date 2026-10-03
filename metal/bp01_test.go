//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"testing"
	"time"
)

// GOINFER_METAL_BLK64=1 turns B-P01's hd = 64 block kernel on for every resident this test binary builds
// (attnFABlk64On, off in production until its grade passes), so the night grade's fidelity gate and accuracy runs read
// the candidate through the same tests that graded the hd = 128 kernel.
func init() {
	if os.Getenv("GOINFER_METAL_BLK64") == "1" {
		attnFABlk64On = true
	}
}

// TestBP01AttnAB is B-P01's speed instrument (docs/tasks/task-metal-audit-2026-10.md, "B-P01"): in-sequence attention
// work on the 0.5B at 2048 and 3900 keys, the per-query-head kernel (today's path for hd = 64) against the hd = 64
// block kernel at the production split (16) and at 8, 24 and 32 (R17 found the response to S not monotone). Each
// depth runs every arm and a no-op-attention arm, interleaved and rotated rep by rep, `tokens` decode tokens each from
// the same position; an arm's attention work is its token GPU-time median minus the no-op arm's, paired per rep. The
// graded line is legacy / blk at S = 16. Night-only: GOINFER_METAL_BP01=1.
func TestBP01AttnAB(t *testing.T) {
	if os.Getenv("GOINFER_METAL_BP01") != "1" {
		t.Skip("set GOINFER_METAL_BP01=1: B-P01's timed probe (night-only)")
	}
	prev := attnFABlk64On
	attnFABlk64On = true
	defer func() { attnFABlk64On = prev }()
	const tokens = 8
	depths := []int{2048, 3900}
	name, a := auditLoad(t, "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", 1, depths[len(depths)-1]+tokens+64, nil)
	r, t0 := a.r, time.Now()
	if r.layers[r.attnFALayer].geom.hd != 64 || r.attnFABlkSplit == 0 {
		t.Fatalf("%s: the hd = 64 block kernel is not selected (hd %d, attnFABlkSplit %d)", name, r.layers[r.attnFALayer].geom.hd, r.attnFABlkSplit)
	}
	noop := func() Pipeline {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		pool := NewARPool()
		defer pool.Drain()
		lib, err := r.d.CompileLibrary("#include <metal_stdlib>\nusing namespace metal;\nkernel void bp01_noop() {}\n", MSL3_1)
		if err != nil {
			t.Fatalf("compile no-op: %v", err)
		}
		p, err := r.d.NewComputePipeline(lib, "bp01_noop")
		if err != nil {
			t.Fatalf("no-op pipeline: %v", err)
		}
		return p
	}()
	fa, comb, leg := r.pAttnFA, r.pAttnFACombine, r.pAttn
	type arm struct {
		name   string
		legacy bool // the per-query-head kernel (attention_fa's floor moved out of reach)
		split  int
		null   bool
	}
	arms := []arm{{name: "legacy", legacy: true}, {name: "blk S=16", split: 16}, {name: "blk S=8", split: 8},
		{name: "blk S=24", split: 24}, {name: "blk S=32", split: 32}, {name: "no-op", null: true}}
	set := func(x arm) {
		r.pAttnFA, r.pAttnFACombine, r.pAttn = fa, comb, leg
		r.attnFAFloorOverride, r.attnFASplitOverride = 1, x.split
		if x.legacy {
			r.attnFAFloorOverride = 1 << 30
		}
		if x.null {
			r.pAttnFA, r.pAttnFACombine, r.pAttn = noop, noop, noop
		}
		r.stopExec()
	}
	defer func() {
		r.pAttnFA, r.pAttnFACombine, r.pAttn = fa, comb, leg
		r.attnFAFloorOverride, r.attnFASplitOverride = 0, 0
		r.stopExec()
	}()
	embs := auditEmbs(r, depths[len(depths)-1]+tokens, 17)
	reps := auditReps(5)
	for _, D := range depths {
		set(arms[0])
		if _, err := a.PrefillLast(context.Background(), embs[:D], 0); err != nil {
			t.Fatalf("fill to %d: %v", D, err)
		}
		times := make([][]float64, len(arms)) // per arm, per rep: token GPU-time median
		for rep := range reps {
			for k := range arms {
				ai := (k + rep) % len(arms)
				set(arms[ai])
				var ks []float64
				for i := range tokens {
					r.ForwardEmb(embs[D+i], D+i)
					ks = append(ks, (r.gpuEnd-r.gpuStart)*1e3)
				}
				times[ai] = append(times[ai], auditMedian(ks))
			}
		}
		work := func(ai int) []float64 {
			w := make([]float64, reps)
			for i := range w {
				w[i] = times[ai][i] - times[len(arms)-1][i]
			}
			return w
		}
		leg := work(0)
		for ai := 1; ai < len(arms)-1; ai++ {
			w := work(ai)
			var ratios []float64
			for i := range w {
				ratios = append(ratios, leg[i]/w[i])
			}
			auditHB("b-p01", t0, "%s %d keys: legacy attention %.3f ms, %s %.3f ms (token %.3f against %.3f); legacy/blk %.3f",
				name, D, auditMedian(leg), arms[ai].name, auditMedian(w), auditMedian(times[ai]), auditMedian(times[0]), auditMedian(ratios))
			if arms[ai].split == 16 {
				fmt.Fprintf(os.Stderr, "[b-p01] %s B-P01 METRIC legacy/blk S=16 at %d keys: %.4f (per-rep %v); token legacy/blk %.4f\n",
					name, D, auditMedian(ratios), fmtRatios(ratios), auditMedian(times[0])/auditMedian(times[ai]))
			}
		}
	}
}
