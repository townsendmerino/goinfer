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

// TestBP02AttnAB is B-P02's speed instrument (docs/tasks/task-metal-audit-2026-10.md, "B-P02"): in-sequence attention
// work on a G = 2 hd = 128 model (internlm2-1_8b by default; GOINFER_AUDIT_MODEL picks another) at 2048 and 3900
// keys, the legacy attention_fa at its core-count split (today's path for G outside 6 and 7) against the block kernel
// at the production split (16), with the no-op arm subtracted per rep, as TestBP01AttnAB. Night-only:
// GOINFER_METAL_BP02=1.
func TestBP02AttnAB(t *testing.T) {
	if os.Getenv("GOINFER_METAL_BP02") != "1" {
		t.Skip("set GOINFER_METAL_BP02=1: B-P02's timed probe (night-only)")
	}
	prev := attnFABlkAnyG
	attnFABlkAnyG = true
	defer func() { attnFABlkAnyG = prev }()
	const tokens = 8
	depths := []int{2048, 3900}
	name, a := auditLoad(t, "internlm2-1_8b", 1, depths[len(depths)-1]+tokens+64, nil)
	r, t0 := a.r, time.Now()
	G := r.nH / r.attnFANKV
	if r.attnFABlkSplit == 0 || G == 6 || G == 7 {
		t.Fatalf("%s: G = %d with attnFABlkSplit %d: not a B-P02 shape on the block kernel", name, G, r.attnFABlkSplit)
	}
	legacy, legacySplit := r17Legacy(t, r)
	noop := func() Pipeline {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		pool := NewARPool()
		defer pool.Drain()
		lib, err := r.d.CompileLibrary("#include <metal_stdlib>\nusing namespace metal;\nkernel void bp02_noop() {}\n", MSL3_1)
		if err != nil {
			t.Fatalf("compile no-op: %v", err)
		}
		p, err := r.d.NewComputePipeline(lib, "bp02_noop")
		if err != nil {
			t.Fatalf("no-op pipeline: %v", err)
		}
		return p
	}()
	blk, comb, leg, split := r.pAttnFA, r.pAttnFACombine, r.pAttn, r.attnFABlkSplit
	type arm struct{ name string }
	arms := []arm{{"legacy attention_fa"}, {"blk S=16"}, {"no-op"}}
	set := func(ai int) {
		r.pAttnFA, r.pAttnFACombine, r.pAttn, r.attnFABlkSplit = blk, comb, leg, split
		switch ai {
		case 0:
			r.pAttnFA, r.attnFABlkSplit = legacy, 0
		case 2:
			r.pAttnFA, r.pAttnFACombine, r.pAttn = noop, noop, noop
		}
		r.stopExec()
	}
	defer func() { set(1) }()
	auditHB("b-p02", t0, "%s: G = %d, legacy split %d, block split %d", name, G, legacySplit, split)
	embs := auditEmbs(r, depths[len(depths)-1]+tokens, 23)
	reps := auditReps(5)
	for _, D := range depths {
		set(0)
		if _, err := a.PrefillLast(context.Background(), embs[:D], 0); err != nil {
			t.Fatalf("fill to %d: %v", D, err)
		}
		times := make([][]float64, len(arms))
		for rep := range reps {
			for k := range arms {
				ai := (k + rep) % len(arms)
				set(ai)
				var ks []float64
				for i := range tokens {
					r.ForwardEmb(embs[D+i], D+i)
					ks = append(ks, (r.gpuEnd-r.gpuStart)*1e3)
				}
				times[ai] = append(times[ai], auditMedian(ks))
			}
		}
		var ratios, lw, bw []float64
		for i := range reps {
			l, b := times[0][i]-times[2][i], times[1][i]-times[2][i]
			lw, bw, ratios = append(lw, l), append(bw, b), append(ratios, l/b)
		}
		auditHB("b-p02", t0, "%s %d keys: legacy attention %.3f ms, blk %.3f ms (token %.3f against %.3f)",
			name, D, auditMedian(lw), auditMedian(bw), auditMedian(times[1]), auditMedian(times[0]))
		fmt.Fprintf(os.Stderr, "[b-p02] %s B-P02 METRIC legacy/blk at %d keys: %.4f (per-rep %v); token legacy/blk %.4f\n",
			name, D, auditMedian(ratios), fmtRatios(ratios), auditMedian(times[0])/auditMedian(times[1]))
	}
}

// TestDB04AB is D-B04's speed instrument (docs/tasks/task-metal-audit-2026-10.md, "D-B04"): decode on the Qwen3.5-9B
// hybrid (24 DeltaNet layers) with gemvExtOn against off, one resident, the switch toggled between arms (gemvExt reads
// it at encode; stopExec drops anything encoded under the other arm). Each rep runs both arms over the same `tokens`
// positions from depth D, alternated, from the same DeltaNet state (snapshotted after the fill, restored before each
// arm) and rewriting the same K/V; the last token's logits
// must be equal across the arms in every rep. An arm's time is its token GPU-time median. Night-only:
// GOINFER_METAL_DB04=1; GOINFER_AUDIT_MODEL picks another model.
func TestDB04AB(t *testing.T) {
	if os.Getenv("GOINFER_METAL_DB04") != "1" {
		t.Skip("set GOINFER_METAL_DB04=1: D-B04's timed probe (night-only)")
	}
	const tokens = 16
	depths := []int{128, 1024}
	name, a := auditLoad(t, "Qwen3.5-9B-Q4_K_M.int4.metal.giw", 1, depths[len(depths)-1]+tokens+64, nil)
	r, t0 := a.r, time.Now()
	if r.dnet == nil {
		t.Fatalf("%s has no DeltaNet layers", name)
	}
	defer func() { gemvExtOn = false; r.stopExec() }()
	embs := auditEmbs(r, depths[len(depths)-1]+tokens, 29)
	reps := auditReps(7)
	for _, D := range depths {
		gemvExtOn = false
		r.stopExec()
		for p := range D {
			r.ForwardEmb(embs[p], p) // a decode-path fill: a hybrid's batched prefill is not on this path
		}
		var ratios, offT, onT []float64
		var last [2][]float32
		// DeltaNet's recurrent state and conv window are not indexed by position: each arm starts from the state the
		// fill left, restored here (both are host-visible), or the second arm would run on the first arm's state.
		type dsnap struct{ state, win []float32 }
		snap := func() []dsnap {
			var out []dsnap
			for _, L := range r.layers {
				if L.delta != nil {
					out = append(out, dsnap{append([]float32(nil), L.delta.state.Floats()...), append([]float32(nil), L.delta.win.Floats()...)})
				}
			}
			return out
		}
		restore := func(sn []dsnap) {
			i := 0
			for _, L := range r.layers {
				if L.delta != nil {
					copy(L.delta.state.Floats(), sn[i].state)
					copy(L.delta.win.Floats(), sn[i].win)
					i++
				}
			}
		}
		r.stopExec()
		base := snap()
		for rep := range reps {
			var tm [2]float64
			for k := range 2 {
				on := (k+rep)%2 == 1
				gemvExtOn = on
				r.stopExec()
				restore(base)
				var ks []float64
				var lg []float32
				for i := range tokens {
					lg = r.ForwardEmb(embs[D+i], D+i)
					ks = append(ks, (r.gpuEnd-r.gpuStart)*1e3)
				}
				ai := 0
				if on {
					ai = 1
				}
				tm[ai], last[ai] = auditMedian(ks), append([]float32(nil), lg...)
			}
			for i := range last[0] {
				if last[0][i] != last[1][i] {
					t.Fatalf("depth %d rep %d: logit %d differs with gemvExtOn (%v against %v)", D, rep, i, last[1][i], last[0][i])
				}
			}
			ratios, offT, onT = append(ratios, tm[0]/tm[1]), append(offT, tm[0]), append(onT, tm[1])
		}
		above := 0
		for _, x := range ratios {
			if x > 1 {
				above++
			}
		}
		auditHB("d-b04", t0, "%s depth %d: token %.3f ms off, %.3f ms on (medians of %d reps x %d tokens); rows-form dispatches %v; logits equal in every rep",
			name, D, auditMedian(offT), auditMedian(onT), reps, tokens, r.gemvExtRows)
		fmt.Fprintf(os.Stderr, "[d-b04] %s D-B04 METRIC off/on at depth %d: %.4f (%d of %d reps above 1; per-rep %v)\n",
			name, D, auditMedian(ratios), above, reps, fmtRatios(ratios))
	}
}
