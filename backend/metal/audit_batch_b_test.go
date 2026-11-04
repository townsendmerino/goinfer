//go:build darwin

package metal

import (
	"context"
	"fmt"
	"math"
	"os"
	"runtime"
	"strconv"
	"testing"
	"time"
)

// TestAuditT17_attnStaircase (T1.7, B-P08): in-sequence attention work per key at 1536 ... 6144 keys in 128-key steps.
// Each depth runs production's decode token and the same token with every attention pipeline (pAttnFA, its combine,
// the legacy pAttn) a no-op, arms interleaved and alternated rep by rep, `tokens` tokens per arm from the same depth;
// attention work is the paired difference of the GPU-time medians (R17's no-op method, TestR17AttentionProto). B-P08
// claims the block kernel's fixed split gives a cost staircase in depth (one extra trip per simdgroup past 2048 and
// 4096 keys); the reading is attention work / keys per depth, and B-P08 is killed when its max over min is under 1.15.
//
//	GOINFER_METAL_AUDIT_B=1 [GOINFER_AUDIT_MODEL=...] [GOINFER_AUDIT_T17_HI=...] go test -run '^TestAuditT17_attnStaircase$' -v ./metal/
func TestAuditT17_attnStaircase(t *testing.T) {
	if os.Getenv("GOINFER_METAL_AUDIT_B") != "1" {
		t.Skip("set GOINFER_METAL_AUDIT_B=1: a timed Batch B probe (night-only)")
	}
	const lo, step, tokens = 1536, 128, 8
	hi := 6144
	if v, err := strconv.Atoi(os.Getenv("GOINFER_AUDIT_T17_HI")); err == nil && v >= lo { // a shorter sweep, for a smoke only
		hi = v
	}
	name, a := auditLoad(t, "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", 1, hi+2*tokens+64, nil)
	r, t0 := a.r, time.Now()
	if r.attnFANKV == 0 || r.attnFABlkSplit == 0 {
		t.Skipf("%s: the block kernel is not selected here (attnFANKV %d, attnFABlkSplit %d)", name, r.attnFANKV, r.attnFABlkSplit)
	}
	noop := func() Pipeline {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		pool := NewARPool()
		defer pool.Drain()
		lib, err := r.d.CompileLibrary("#include <metal_stdlib>\nusing namespace metal;\nkernel void t17_noop() {}\n", MSL3_1)
		if err != nil {
			t.Fatalf("compile no-op: %v", err)
		}
		p, err := r.d.NewComputePipeline(lib, "t17_noop")
		if err != nil {
			t.Fatalf("no-op pipeline: %v", err)
		}
		return p
	}()
	embs := auditEmbs(r, hi+tokens, 11)
	if _, err := a.PrefillLast(context.Background(), embs[:lo], 0); err != nil {
		t.Fatalf("fill: %v", err)
	}
	auditHB("t1.7", t0, "%s: filled %d keys; block kernel at split %d; %d depths %d..%d", name, lo, r.attnFABlkSplit, (hi-lo)/step+1, lo, hi)
	fa, comb, leg := r.pAttnFA, r.pAttnFACombine, r.pAttn
	setNoop := func(on bool) {
		if on {
			r.pAttnFA, r.pAttnFACombine, r.pAttn = noop, noop, noop
		} else {
			r.pAttnFA, r.pAttnFACombine, r.pAttn = fa, comb, leg
		}
		r.stopExec() // drop a buffer pre-encoded under the other arm (TestR17AttentionProto)
	}
	defer setNoop(false)
	run := func(D int, off bool) float64 {
		setNoop(off)
		var ks []float64
		for i := range tokens {
			r.ForwardEmb(embs[D+i], D+i)
			ks = append(ks, (r.gpuEnd-r.gpuStart)*1e3)
		}
		return auditMedian(ks)
	}
	// The keys between depths are written by real tokens before they are read: depth D's tokens write D..D+tokens-1,
	// and the gap up to the next depth is filled the same way, so attention always reads written K/V.
	fill := func(from, to int) {
		setNoop(false)
		for p := from; p < to; p++ {
			r.ForwardEmb(embs[p], p)
		}
	}
	reps := auditReps(5)
	type row struct {
		D                        int
		prod, null, attn, perKey float64
	}
	var rows []row
	filled := lo
	for D := lo; D <= hi; D += step {
		fill(filled, D)
		var diffs, prods, nulls []float64
		for rep := range reps {
			var p, n float64
			if rep%2 == 0 {
				p, n = run(D, false), run(D, true)
			} else {
				n, p = run(D, true), run(D, false)
			}
			prods, nulls, diffs = append(prods, p), append(nulls, n), append(diffs, p-n)
		}
		filled = D + tokens
		w := auditMedian(diffs)
		rows = append(rows, row{D, auditMedian(prods), auditMedian(nulls), w, w * 1e3 / float64(D)})
		auditHB("t1.7", t0, "%s %d keys: token %.3f ms, no-op attention %.3f ms, attention %.3f ms (%.3f us/key)",
			name, D, rows[len(rows)-1].prod, rows[len(rows)-1].null, w, rows[len(rows)-1].perKey)
	}
	mn, mx, at, atMax := math.Inf(1), 0.0, 0, 0
	for _, x := range rows {
		if x.perKey < mn {
			mn, at = x.perKey, x.D
		}
		if x.perKey > mx {
			mx, atMax = x.perKey, x.D
		}
	}
	verdict := "B-P08 stands (max/min per-key cost >= 1.15)"
	if mx/mn < 1.15 {
		verdict = "KILLS B-P08 (max/min per-key cost < 1.15)"
	}
	// The second reading, pre-registered beside the first because they can disagree: attention has a fixed per-token
	// part, so its cost per key falls with depth even with no staircase. A staircase shows as a jump across a trip
	// boundary (the 128-key step from 2048 or from 4096) larger than the other steps: jump / median step.
	var incs []float64
	jump := map[int]float64{}
	for i := 1; i < len(rows); i++ {
		d := rows[i].attn - rows[i-1].attn
		if rows[i-1].D == 2048 || rows[i-1].D == 4096 {
			jump[rows[i-1].D] = d
		} else {
			incs = append(incs, d)
		}
	}
	med := auditMedian(incs)
	for _, b := range []int{2048, 4096} {
		if d, ok := jump[b]; ok {
			auditHB("t1.7", t0, "%s step from %d keys: %+.3f ms, %.2fx the median 128-key step (%+.3f ms)", name, b, d, d/med, med)
			fmt.Fprintf(os.Stderr, "[t1.7] %s METRIC jump at %d / median step %.3f\n", name, b, d/med)
		}
	}
	auditHB("t1.7", t0, "RESULT %s: attention per key min %.3f us at %d keys, max %.3f us at %d keys, max/min %.3f -> %s",
		name, mn, at, mx, atMax, mx/mn, verdict)
	fmt.Fprintf(os.Stderr, "[t1.7] %s METRIC max/min %.4f\n", name, mx/mn)
}
