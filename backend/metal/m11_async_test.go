//go:build darwin && goinfer_testhooks

package metal

import (
	"math"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestM11_asyncPhase2AB is M-11's read (docs/audit-metal-2026-09-30.md, "M-11"): a paged MoE's decode with phase 2
// committed and not waited (pagedAsyncPhase2On) against waited, on one resident, arms alternated rep by rep, wall time a
// token (the round trips are what it removes), every token's logits compared between the arms. Exploratory, by day.
//
//	GOINFER_M11=1 GOINFER_AUDIT_MODEL=<paged MoE .giw> [GOINFER_AUDIT_SLOTS=24] go test -tags goinfer_testhooks -count=1 \
//	  -run '^TestM11_asyncPhase2AB$' -v ./metal/
func TestM11_asyncPhase2AB(t *testing.T) {
	if os.Getenv("GOINFER_M11") != "1" {
		t.Skip("set GOINFER_M11=1 (loads a paged MoE checkpoint)")
	}
	path := os.Getenv("GOINFER_AUDIT_MODEL")
	if path == "" || strings.HasPrefix(path, "/Volumes/") || strings.HasPrefix(path, "/srv/models") {
		t.Fatalf("GOINFER_AUDIT_MODEL %q: a local-disk checkpoint is required (CLAUDE.md)", path)
	}
	slots, _ := strconv.Atoi(os.Getenv("GOINFER_AUDIT_SLOTS"))
	tokens := 40
	if v, err := strconv.Atoi(os.Getenv("GOINFER_M11_TOKENS")); err == nil && v > 0 {
		tokens = v
	}
	m, err := decoder.Load(path, decoder.Options{Quant: "int4", Backend: "metal", ResidentContext: tokens + 64,
		MoECacheExperts: true, MoECacheSlots: slots})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok || !((a.r.g4moe != nil && a.r.g4moe.paged) || (a.r.moe != nil && a.r.moe.paged)) {
		t.Fatalf("not a paged MoE resident (%s)", m.DecodePath())
	}
	r := a.r
	prev, prevPF := pagedAsyncPhase2On, g4PrefetchOn
	defer func() { pagedAsyncPhase2On, g4PrefetchOn = prev, prevPF }()
	if r.g4moe != nil {
		saved := r.g4moe.fence
		defer func() { r.g4moe.fence = saved }()
	}
	// GOINFER_M11_LEVER=prefetch flips lever 3's prefetch instead (both arms with phase 2 async): "waited" is then the
	// prefetch off, "async" the prefetch on. GOINFER_M11_LEVER=fence flips the phase-1 fence (paged_fence.go) the same way.
	lever := os.Getenv("GOINFER_M11_LEVER")
	reads := func() (stages, prefetched, hits int) {
		for l := range r.layers {
			if gl := r.layers[l].g4moe; gl != nil && gl.pool != nil {
				stages, prefetched, hits = stages+gl.pool.stages, prefetched+gl.pool.prefetched, hits+gl.pool.prefetchHits
			}
		}
		return
	}
	t0 := time.Now()
	embs := auditEmbs(r, tokens, 37)
	var fence *pagedFence
	if r.g4moe != nil {
		fence = r.g4moe.fence
	}
	if lever == "fence" && fence == nil {
		t.Fatal("GOINFER_M11_LEVER=fence: this resident built no fence")
	}
	run := func(async bool) (ms []float64, lg [][]float32) {
		switch lever {
		case "prefetch":
			pagedAsyncPhase2On, g4PrefetchOn = true, async
		case "fence": // "waited" = phase 1 waited, "async" = phase 1 fenced; phase 2 async on both
			pagedAsyncPhase2On, g4PrefetchOn = true, false
			if async {
				r.g4moe.fence = fence
			} else {
				r.g4moe.fence = nil
			}
		default:
			pagedAsyncPhase2On, g4PrefetchOn = async, false
			if r.g4moe != nil {
				r.g4moe.fence = nil // the async lever's read predates the fence: phase 1 waited on both arms
			}
		}
		a.Reset()
		for i, e := range embs {
			st := time.Now()
			l := r.ForwardEmb(e, i)
			ms = append(ms, float64(time.Since(st).Microseconds())/1e3)
			lg = append(lg, append([]float32(nil), l...))
		}
		return ms, lg
	}
	run(true) // warm the pool and the page cache once
	reps := auditReps(9)
	var ratios []float64
	var syncMs, asyncMs []float64
	for rep := range reps {
		order := []bool{false, true}
		if rep%2 == 1 {
			order = []bool{true, false}
		}
		res := map[bool]float64{}
		lgs := map[bool][][]float32{}
		for _, async := range order {
			s0, p0, h0 := reads()
			ms, lg := run(async)
			res[async], lgs[async] = auditMedian(ms), lg
			s1, p1, h1 := reads()
			auditHB("m11", t0, "rep %d arm %v: %d experts staged on demand, %d prefetched, %d prefetched and used", rep, async, s1-s0, p1-p0, h1-h0)
			if lever == "fence" && async {
				auditHB("m11", t0, "rep %d fence: %d boundaries spun (%.1f ms total), %d fell back (%.1f ms)", rep, fence.waits, float64(fence.spinNanos)/1e6, fence.fallbacks, float64(fence.waitNanos)/1e6)
				fence.waits, fence.fallbacks, fence.spinNanos, fence.waitNanos = 0, 0, 0, 0
			}
		}
		for i := range lgs[false] {
			for j, x := range lgs[false][i] {
				if math.Float32bits(x) != math.Float32bits(lgs[true][i][j]) {
					t.Fatalf("rep %d token %d logit %d differs between the arms", rep, i, j)
				}
			}
		}
		ratios = append(ratios, res[false]/res[true])
		syncMs, asyncMs = append(syncMs, res[false]), append(asyncMs, res[true])
		auditHB("m11", t0, "rep %d: waited %.1f ms, async %.1f ms a token (median of %d); logits equal", rep, res[false], res[true], tokens)
	}
	above := 0
	for _, q := range ratios {
		if q > 1 {
			above++
		}
	}
	auditHB("m11", t0, "RESULT %s: waited %.1f ms, async %.1f ms a token; waited / async median %.3f, %d of %d reps above 1 (per rep %s)",
		path, auditMedian(syncMs), auditMedian(asyncMs), auditMedian(ratios), above, reps, auditFmt3(ratios))
}
