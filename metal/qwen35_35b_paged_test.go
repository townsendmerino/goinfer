//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"os"
	"os/exec"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestQwen35_35B_pagedRuns is the at-scale payoff run for the GENERIC Metal expert pager — the
// qwen3_5_moe twin of TestGemma4_26B_pagedRuns, and the measurement the "Metal expert streaming"
// front had never had. Qwen3.5-35B-A3B: nE=256, topK=8, 40 MoE layers, 22.1 GB of int4 weights on
// a 16 GB box, so the expert set cannot be resident and paging is not an optimization but the only
// way the model runs at all.
//
// WHAT THIS MEASURES AND WHAT IT DOES NOT. Steady-state decode rate through the pager, plus the
// staging decomposition (how much of a token is expert I/O). It is NOT a peer comparison: no
// Ollama/llama.cpp arm runs here, and the CPU-paged 1.3-1.4 tok/s figure this lane has quoted was
// measured in another session on another path, so no ratio against it is computed. See
// ~/goinfer-bench-logs/PREREGISTERED-qwen35-metal-paging.md for the rule this run was written
// against, BEFORE it produced a number.
//
// DECODE IS GREEDY SELF-FEEDING (argmax -> next token), not a random token walk. Routing locality
// is the whole mechanism under test: a random-ID sequence would destroy the temporal expert reuse
// that real decode has, and would understate the pager. Seeded from a fixed token so the
// trajectory is deterministic.
//
// STEADY STATE IS THE POINT. The first tokens are dominated by cold page-in and by the pool still
// filling (with 32 slots and topK=8, no eviction can happen until token 5), so early tokens are
// unrepresentatively fast on reuse and unrepresentatively slow on I/O. The reported rate is the
// MEDIAN over the last two thirds; p90 is reported alongside because paging has a heavy right tail
// that a mean would launder.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags goinfer_testhooks ./metal/ -run TestQwen35_35B_pagedRuns -v -timeout 60m
func TestQwen35_35B_pagedRuns(t *testing.T) {
	requireHeavyModel(t)
	giw := modelPath("Qwen3.5-35B-A3B-Q4_K_M.int4.giw")
	if _, err := os.Stat(giw); err != nil {
		t.Skipf("no 35B .giw (%s)", giw)
	}
	// Guard the bench-surface rule: a run off the archive (5400 rpm SMR, or the SMB mount) returns a
	// plausible WRONG number instead of erroring. Fail loudly rather than record a void row.
	if strings.HasPrefix(giw, "/Volumes/") || strings.HasPrefix(giw, "/srv/models") {
		t.Fatalf("model path %s is an ARCHIVE path, not a bench surface — any row measured here is void", giw)
	}
	N := 32
	if v := os.Getenv("GOINFER_PAGED_N"); v != "" {
		N, _ = strconv.Atoi(v)
	}
	ntok := 120
	if v := os.Getenv("GOINFER_PAGED_TOKENS"); v != "" {
		ntok, _ = strconv.Atoi(v)
	}
	t.Setenv("GOINFER_METAL_MOE_SLOTS", strconv.Itoa(N))
	preadOn := os.Getenv("GOINFER_MOE_PREAD") != "0"

	rssMB := func() int {
		out, err := exec.Command("ps", "-o", "rss=", "-p", strconv.Itoa(os.Getpid())).Output()
		if err != nil {
			return -1
		}
		kb, _ := strconv.Atoi(strings.TrimSpace(string(out)))
		return kb / 1024
	}
	rusage := func() (int64, int64) {
		var ru syscall.Rusage
		_ = syscall.Getrusage(syscall.RUSAGE_SELF, &ru)
		return int64(ru.Majflt), int64(ru.Minflt)
	}
	// Progress to stderr, not t.Logf: t.Logf buffers until the test returns, which cannot tell a
	// slow run from a hung one. Long-test rule.
	say := func(f string, a ...any) {
		fmt.Fprintf(os.Stderr, "[35B %s] "+f+"\n", append([]any{time.Now().Format("15:04:05")}, a...)...)
	}

	say("ARM pread=%v slots=%d tokens=%d | loading %s", preadOn, N, ntok, giw)
	tLoad := time.Now()
	m, err := decoder.Load(giw, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	say("loaded %.1fs rss=%d MB", time.Since(tLoad).Seconds(), rssMB())

	tBuild := time.Now()
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("buildResident (paged N=%d): %v — the 35B did not build", N, err)
	}
	defer r.Close()
	buildSecs := time.Since(tBuild).Seconds()
	if r.moe == nil || !r.moe.paged {
		t.Fatalf("expected a paged GENERIC MoE build (moe=%v)", r.moe)
	}
	nPaged := 0
	for l := range r.layers {
		if ml := r.layers[l].moe; ml != nil && ml.pool != nil {
			nPaged++
		}
	}
	rssBuilt := rssMB()
	say("PAGED BUILD OK %.1fs: nE=%d topK=%d slots=%d, %d/%d paged MoE layers, stagePread wired=%v, rss=%d MB",
		buildSecs, r.moe.nE, r.moe.k, r.moe.slots, nPaged, r.nL, r.moe.giwFile != nil, rssBuilt)

	// Assert the arm is the arm: a silent fallback to byte-copy would otherwise be reported as a
	// pread number. Checked again after decode via the pool counters.
	poolTotals := func() (stages, preads, hits, evict int, stageNs int64) {
		for l := range r.layers {
			if ml := r.layers[l].moe; ml != nil && ml.pool != nil {
				stages += ml.pool.stages
				preads += ml.pool.preads
				hits += ml.pool.hits
				evict += ml.pool.evictions
				stageNs += ml.pool.stageNanos
			}
		}
		return
	}

	// Warm-up token: cold-start page-in + first fill. Excluded from every statistic.
	tok := 7
	tWarm := time.Now()
	lg := r.ForwardEmb(m.EmbedResidentForTest(tok), 0)
	if lg == nil {
		t.Fatalf("warm-up token returned nil logits: %v", r.takeExecErr())
	}
	tok = argmaxF(lg)
	say("warm-up token %.2fs rss=%d MB", time.Since(tWarm).Seconds(), rssMB())

	s0, p0, h0, e0, sn0 := poolTotals()
	mj0, mn0 := rusage()
	times := make([]float64, 0, ntok)
	tRun := time.Now()
	for i := 1; i <= ntok; i++ {
		st := time.Now()
		lg = r.ForwardEmb(m.EmbedResidentForTest(tok), i)
		d := time.Since(st).Seconds()
		if lg == nil {
			t.Fatalf("token %d nil logits: %v", i, r.takeExecErr())
		}
		times = append(times, d)
		tok = argmaxF(lg)
		if i%20 == 0 {
			done := time.Since(tRun).Seconds()
			say("token %d/%d  last %.3fs  elapsed %.0fs  eta %.0fs  rss=%d MB",
				i, ntok, d, done, done/float64(i)*float64(ntok-i), rssMB())
		}
	}
	runSecs := time.Since(tRun).Seconds()
	mj1, mn1 := rusage()
	s1, p1, h1, e1, sn1 := poolTotals()

	stages, preads, hits, evict := s1-s0, p1-p0, h1-h0, e1-e0
	stageSecs := float64(sn1-sn0) / 1e9
	if preadOn && preads == 0 {
		t.Fatalf("pread arm ran %d stages but 0 preads — stagePread was NOT wired, so this is a "+
			"byte-copy number mislabelled as pread", stages)
	}
	if !preadOn && preads != 0 {
		t.Fatalf("byte-copy arm took the pread path %d times — arms are not separated", preads)
	}

	// Steady state = last two thirds. Median, plus p90 for the paging tail.
	cut := len(times) / 3
	tail := append([]float64(nil), times[cut:]...)
	sort.Float64s(tail)
	q := func(p float64) float64 { return tail[min(int(p*float64(len(tail))), len(tail)-1)] }
	med, p10, p90 := q(0.50), q(0.10), q(0.90)
	early := 0.0
	for _, v := range times[:cut] {
		early += v
	}
	early /= float64(cut)

	say("=== RESULT (arm pread=%v, slots=%d) ===", preadOn, N)
	say("steady-state MEDIAN %.3f s/tok = %.3f tok/s   [p10 %.3f  p90 %.3f]  (last %d of %d tokens)",
		med, 1/med, p10, p90, len(tail), len(times))
	say("first-third mean %.3f s/tok = %.3f tok/s (warming: page cache + pool fill) — reported, never pooled",
		early, 1/early)
	say("whole run %.1fs for %d tokens = %.3f tok/s", runSecs, ntok, float64(ntok)/runSecs)
	say("staging: %d stages (%d pread) %d hits %d evictions | hit-rate %.1f%% | stage %.1fs = %.0f%% of decode",
		stages, preads, hits, evict, 100*float64(hits)/float64(hits+stages), stageSecs, 100*stageSecs/runSecs)
	say("faults over timed window: major %d (%.1f/stage) minor %d | rss %d MB",
		mj1-mj0, float64(mj1-mj0)/float64(max(stages, 1)), mn1-mn0, rssMB())
	t.Logf("35B paged steady-state %.3f tok/s (median), arm pread=%v slots=%d", 1/med, preadOn, N)
}

// TestQwen35_35B_cpuPagedBaseline is the DO-NOTHING ARM for the run above: the same checkpoint,
// same box, same session, decoded through the CPU expert pager (StreamWeights, decoder/moepaging.go)
// instead of the Metal one. Without it, "Metal paging is faster" would be a cross-session ratio
// against a number measured on another day — exactly the comparison this repo's own
// task-zeno-compare.md shows can drift 2.6x from machine load alone (kind-3 gemma4 at one identical
// config read 1.128 and then 2.917 tok/s).
//
// IT LIVES IN metal/ AND USES NO METAL. That is deliberate: its only reason to exist is to be the
// baseline for the Metal number, and a baseline that drifts away from its comparand into another
// package (and another run) is how the 1.3-1.4 figure went stale in the first place. Same helpers,
// same seed token, same greedy self-feeding trajectory, same token count.
//
// STILL NOT PER-TOKEN INTERLEAVED. It is an adjacent run, not a matched-pair one: the two pagers
// cannot share a process cheaply (the CPU arm wants a multi-GB budget while the Metal arm holds its
// slots). So it inherits session-level drift, and is reported as a same-day side-by-side rather
// than as a ratio with error bars. The pread A/B above IS matched — identical trajectory, identical
// stage/hit/eviction counts — which is why the pread claim is the strong one and this is context.
//
//	GOINFER_HEAVY_TESTS=1 go test -count=1 -tags goinfer_testhooks ./metal/ -run TestQwen35_35B_cpuPagedBaseline -v -timeout 60m
func TestQwen35_35B_cpuPagedBaseline(t *testing.T) {
	requireHeavyModel(t)
	giw := modelPath("Qwen3.5-35B-A3B-Q4_K_M.int4.giw")
	if _, err := os.Stat(giw); err != nil {
		t.Skipf("no 35B .giw (%s)", giw)
	}
	budgetGB := 6.0 // matches the 6 GB kind-3 row in docs/task-zeno-compare.md (1.605 tok/s, 3-run avg)
	if v := os.Getenv("GOINFER_CPU_BUDGET_GB"); v != "" {
		budgetGB, _ = strconv.ParseFloat(v, 64)
	}
	ntok := 120
	if v := os.Getenv("GOINFER_PAGED_TOKENS"); v != "" {
		ntok, _ = strconv.Atoi(v)
	}
	say := func(f string, a ...any) {
		fmt.Fprintf(os.Stderr, "[35B-cpu %s] "+f+"\n", append([]any{time.Now().Format("15:04:05")}, a...)...)
	}

	say("CPU-paged arm: budget %.1f GB, %d tokens | loading %s", budgetGB, ntok, giw)
	tLoad := time.Now()
	m, err := decoder.Load(giw, decoder.Options{
		Quant: "int4", StreamWeights: true, WeightCacheBytes: int64(budgetGB * 1e9),
	})
	if err != nil {
		t.Fatalf("Load(stream-weights): %v", err)
	}
	defer m.Close()
	say("loaded %.1fs", time.Since(tLoad).Seconds())

	cache := m.NewCache(ntok + 8)
	tok := 7 // same seed as the Metal arm, so the trajectory is the same decode
	lg, err := m.ForwardForTest(tok, cache)
	if err != nil {
		t.Fatalf("warm-up forward: %v", err)
	}
	tok = argmaxF(lg)
	say("warm-up token done")

	times := make([]float64, 0, ntok)
	tRun := time.Now()
	for i := 1; i <= ntok; i++ {
		st := time.Now()
		lg, err = m.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatalf("token %d: %v", i, err)
		}
		times = append(times, time.Since(st).Seconds())
		tok = argmaxF(lg)
		if i%20 == 0 {
			done := time.Since(tRun).Seconds()
			say("token %d/%d last %.3fs elapsed %.0fs eta %.0fs", i, ntok, times[len(times)-1], done, done/float64(i)*float64(ntok-i))
		}
	}
	runSecs := time.Since(tRun).Seconds()
	cut := len(times) / 3
	tail := append([]float64(nil), times[cut:]...)
	sort.Float64s(tail)
	q := func(p float64) float64 { return tail[min(int(p*float64(len(tail))), len(tail)-1)] }
	med, p10, p90 := q(0.50), q(0.10), q(0.90)
	say("=== RESULT (CPU-paged, budget %.1f GB) ===", budgetGB)
	say("steady-state MEDIAN %.3f s/tok = %.3f tok/s  [p10 %.3f p90 %.3f] (last %d of %d)",
		med, 1/med, p10, p90, len(tail), len(times))
	say("whole run %.1fs for %d tokens = %.3f tok/s", runSecs, ntok, float64(ntok)/runSecs)
	t.Logf("35B CPU-paged steady-state %.3f tok/s (median), budget %.1f GB", 1/med, budgetGB)
}
