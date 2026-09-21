//go:build darwin && goinfer_testhooks

package metal

import (
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGemma4_26B_autoPagedRuns is R11(c) (docs/tasks/red-october.md): the M26 row re-run against
// the AUTO-SIZED pager (decoder.Options.MoECacheExperts with MoECacheSlots left at 0), not a
// hand-picked N the way TestGemma4_26B_pagedRuns (GOINFER_METAL_MOE_SLOTS, N=32 default) exercises.
// metalMoESlotsRequest (metal/backend.go) only calls the real autoMoESlots formula when
// MoECacheExperts is set and no explicit slot count is given — GOINFER_METAL_MOE_SLOTS bypasses
// that path entirely, so the existing test does not answer this brief's question.
//
// SAFETY (the reason this is its own test, run in isolation, not folded into a sweep): this exact
// model class (M26, alongside M35/H27) produced a real kernel panic on this machine via the
// CPU-staged fallback (benchmarks.md "M35/M26 on the Mac"). MoECacheExperts's paged path is a
// DIFFERENT, GPU-resident mechanism (contiguous per-layer expert-slot pool, on-demand pread) that
// measured a real, if slow, decode rate in that same record (~2 tok/s) — not the disaster path —
// but only when it actually engages, which is why r.g4moe.paged and the auto-sized N are asserted
// BEFORE any decode step runs, not inferred from the outcome. Bounded to 4 timed decode steps
// (mirrors TestGemma4_26B_pagedRuns's own 5-token bound), RSS logged before/after load and after
// every step so a runaway is visible immediately rather than discovered after the fact.
//
// MEASURED, 2026-09-20 (docs/measurements/metal-moe-autopager-m26-2026-09-20.md): this test's own
// in-process safeguards (DecodePath/g4moe.paged checks, the RSS kill switch) correctly confirmed
// the right mechanism engages, but did NOT prevent a real near-incident — an externally-monitored
// run showed system swap spiral to 12+ GB within ~50s of process start, entirely during
// decoder.Load/buildResident, well before this test's own RSS check (which reads low because the
// spike is transient host-side mmap/parse traffic that settles before buildResident returns) had
// anything to catch. Killed manually from outside the test process. A SEPARATE, EXTERNAL memory
// monitor is not optional context when running this test — see the record for what one looks
// like and why the in-process guards alone were not enough here.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags goinfer_testhooks ./metal/ -run TestGemma4_26B_autoPagedRuns -v -timeout 10m
func TestGemma4_26B_autoPagedRuns(t *testing.T) {
	requireHeavyModel(t)
	giw := modelPath("gemma4-26b-int4.giw")
	if _, err := os.Stat(giw); err != nil {
		t.Skipf("no .giw (%s)", giw)
	}

	rssMB := func() int {
		out, err := exec.Command("ps", "-o", "rss=", "-p", strconv.Itoa(os.Getpid())).Output()
		if err != nil {
			return -1
		}
		kb, _ := strconv.Atoi(strings.TrimSpace(string(out)))
		return kb / 1024
	}

	rss0 := rssMB()
	t.Logf("before load: RSS %d MB", rss0)

	m, err := decoder.Load(giw, decoder.Options{Backend: "metal", Quant: "int4", MoECacheExperts: true})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()

	// The load-time decision, checked BEFORE anything else: did this resolve to Metal residency
	// (the paged mechanism) or silently fall through to the staged/CPU path the incident record
	// warns about? DecodePath is the same string the server's own startup banner prints.
	path := m.DecodePath()
	t.Logf("DecodePath: %q", path)
	if !strings.HasPrefix(path, "metal-resident") {
		t.Fatalf("expected metal-resident (paged), got %q -- this is the dangerous fallback class "+
			"(benchmarks.md \"M35/M26 on the Mac\"); refusing to decode", path)
	}

	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("BuildResident: %v", err)
	}
	defer r.Close()
	if r.g4moe == nil || !r.g4moe.paged {
		t.Fatalf("expected g4moe.paged=true (auto-sized pager); got paged=%v -- refusing to decode",
			r.g4moe != nil && r.g4moe.paged)
	}
	rssBuilt := rssMB()
	t.Logf("auto-sized N = %d slots/layer (autoMoESlotsMax=%d ceiling) -- RSS %d MB -> %d MB after build",
		r.g4moe.slots, autoMoESlotsMax, rss0, rssBuilt)

	toks := twoGeomPrompt[:5] // 1 seed + 4 timed steps, same bound as TestGemma4_26B_pagedRuns
	warm := append([]float32(nil), r.ForwardEmb(m.EmbedResidentForTest(toks[0]), 0)...)
	_ = warm
	t.Logf("seed token OK -- RSS %d MB", rssMB())

	var totalNanos int64
	for i := 1; i < len(toks); i++ {
		t0 := time.Now()
		out := r.ForwardEmb(m.EmbedResidentForTest(toks[i]), i)
		dt := time.Since(t0)
		totalNanos += dt.Nanoseconds()
		if len(out) == 0 {
			t.Fatalf("step %d: empty logits", i)
		}
		rss := rssMB()
		t.Logf("step %d: %.1f ms, argmax=%d, RSS %d MB", i, float64(dt.Microseconds())/1000, argmaxF(out), rss)
		// A live kill switch, not just a log: if RSS blows past the resident budget's own ceiling
		// (11.2 GB budget + a few GB of process/runtime overhead), something is holding memory the
		// paged design should not be holding, and continuing is exactly the risk this test exists
		// to avoid.
		if rss > 15000 {
			t.Fatalf("RSS %d MB exceeds the safety ceiling (15000 MB) at step %d -- aborting rather "+
				"than risk the swap-thrashing/kernel-panic class this brief is built around", rss, i)
		}
	}
	nTimed := len(toks) - 1
	t.Logf("RESULT: %d timed steps, %.1f ms/tok, %.2f tok/s, auto-sized N=%d, final RSS %d MB",
		nTimed, float64(totalNanos)/1e6/float64(nTimed), 1e9*float64(nTimed)/float64(totalNanos), r.g4moe.slots, rssMB())
}
