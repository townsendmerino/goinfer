//go:build cuda

package cuda

import (
	"errors"
	"io/fs"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// loadG4MoEGraphs loads a gemma4-MoE fixture as a CUDA resident int4 runner with CUDA graphs on/off
// and the C′ expert cache on/off. Skips if the fixture/GPU is absent.
func loadG4MoEGraphs(t *testing.T, dir string, graphs, cache bool) (*decoder.Model, decoder.ResidentForward) {
	t.Helper()
	if _, err := os.Stat(dir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no fixture (%s)", dir)
	}
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	set := func(k string, on bool) {
		if on {
			t.Setenv(k, "1")
		} else {
			t.Setenv(k, "")
		}
	}
	set("GOINFER_CUDA_GRAPHS", graphs)
	set("GOINFER_MOE_CACHE_EXPERTS", cache)
	mc, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load (graphs=%v cache=%v): %v", graphs, cache, err)
	}
	rf := mc.ResidentForwardForTest()
	if rf == nil {
		mc.Close()
		t.Fatalf("cuda resident DECLINED gemma4 MoE (graphs=%v cache=%v)", graphs, cache)
	}
	return mc, rf
}

// graphsBitExact is Step 2's core gate: replaying the captured static segments (graphs on) must be
// BYTE-IDENTICAL to re-issuing the launches (graphs off), at every position, for a FIXED cache
// setting. The graph path relies on replay reading current buffer contents; a boundary that severed a
// dependency the live path had implicitly (segA→rope_kv, segB→readback→segC) would diverge here.
//
// The prompt is long enough to cross the fixture's sliding window, so nKeys/nWin — hence the
// attention launch's shared-mem geometry and the interleaving around each segment boundary — VARIES
// across positions within one run. That is the "perturb the timing" arm the sharpened gate asks for:
// bit-exact across a spread of geometries, not just one. The CUDA_LAUNCH_BLOCKING=1 control arm (run
// the same gate with launches serialized — any disagreement is an ordering bug) is a separate process
// invocation, since the driver reads that env at init.
func graphsBitExact(t *testing.T, dir string, cache bool) {
	// 40 tokens: past the tiny fixture's local window, so windowed and full layers see different nWin
	// as position grows — a spread of attention geometries around the segB/segC boundary.
	prompt := make([]int, 40)
	for i := range prompt {
		prompt[i] = (i*37 + 5) % 200 // deterministic, spread, and inside the tiny fixture's small vocab
	}
	run := func(graphs bool) [][]float32 {
		mc, rf := loadG4MoEGraphs(t, dir, graphs, cache)
		defer mc.Close()
		out := make([][]float32, len(prompt))
		for i, tok := range prompt {
			l, err := rf.Forward(mc.EmbedResidentForTest(tok), i)
			if err != nil {
				t.Fatalf("forward pos %d (graphs=%v cache=%v): %v", i, graphs, cache, err)
			}
			out[i] = append([]float32(nil), l...)
		}
		return out
	}
	live := run(false)
	graphed := run(true)
	for i := range live {
		for j := range live[i] {
			if live[i][j] != graphed[i][j] {
				t.Fatalf("pos %d logit %d: graphs %.9g != live %.9g (cache=%v) — a captured segment is not "+
					"bit-identical to the live launches (severed dependency or wrong capture boundary)",
					i, j, graphed[i][j], live[i][j], cache)
			}
		}
	}
	blocking := os.Getenv("CUDA_LAUNCH_BLOCKING")
	t.Logf("CUDA graphs (%d pos, cache=%v, CUDA_LAUNCH_BLOCKING=%q): replay == live launches, BIT-IDENTICAL",
		len(prompt), cache, blocking)
}

// TestGemma4Graphs_bitExact_tiny is the committed-fixture gate (always runs on a GPU): graphs replay
// == live launches with the expert cache OFF (fully-resident experts, segC replays the loop).
func TestGemma4Graphs_bitExact_tiny(t *testing.T) {
	graphsBitExact(t, "../testdata/gemma4-moe-tiny", false)
}

// TestGemma4Graphs_bitExact_tinyCache exercises the segB→readback→segC gap: graphs ON with the C′
// cache ON, so the routing D2H readback and the slot H2D sit live between the two replayed segments.
// This is the 26B decode configuration; the ordering hazard the sharpened gate targets lives here.
func TestGemma4Graphs_bitExact_tinyCache(t *testing.T) {
	graphsBitExact(t, "../testdata/gemma4-moe-tiny", true)
}

// TestGemma4Graphs_bitExact_tinyCacheReuse adds cross-token slot reuse (nSlots between topK and nE):
// the readback gap now hits AND misses the LRU cache between the replayed segments. Still bit-exact.
func TestGemma4Graphs_bitExact_tinyCacheReuse(t *testing.T) {
	t.Setenv("GOINFER_MOE_CACHE_SLOTS", "3")
	graphsBitExact(t, "../testdata/gemma4-moe-tiny", true)
}

// TestGemma4Graphs_sameModelUnderLoad isolates graph-vs-live from any BUILD-time confound: it loads
// ONE graphed resident and runs the prompt twice on it — once replaying the captured segments
// (r.graphs=true), once re-issuing the live segs (r.graphs=false) — so weights, cacheSlots and every
// device buffer are identical between the two runs. Pure capture-vs-live compute. Run it under
// concurrent GPU load (a second process) to reproduce the intermittent divergence the two-build gate
// surfaced; a same-model divergence indicts capture/replay, a same-model MATCH indicts the build.
func TestGemma4Graphs_sameModelUnderLoad(t *testing.T) {
	dir := "../testdata/gemma4-moe-tiny"
	mc, rf := loadG4MoEGraphs(t, dir, true, true)
	defer mc.Close()
	r := rf.(*cudaResident)
	if !r.graphs {
		t.Skip("CUDA graphs not admitted on this box (DEFAULT compute mode, no MPS) — this replay-vs-live " +
			"test needs a captured graph, which admitGraphs only promotes under EXCLUSIVE_PROCESS/MPS tenancy; " +
			"set GOINFER_CUDA_GRAPHS_UNSAFE=1 to force it (docs/cuda-graphs-investigation.md §5.1)")
	}
	prompt := make([]int, 40)
	for i := range prompt {
		prompt[i] = (i*37 + 5) % 200
	}
	runOnce := func(useGraphs bool) [][]float32 {
		r.graphs = useGraphs
		out := make([][]float32, len(prompt))
		for i, tok := range prompt {
			l, err := rf.Forward(mc.EmbedResidentForTest(tok), i)
			if err != nil {
				t.Fatalf("forward pos %d (graphs=%v): %v", i, useGraphs, err)
			}
			out[i] = append([]float32(nil), l...)
		}
		return out
	}
	graphed := runOnce(true)
	live := runOnce(false)
	for i := range live {
		for j := range live[i] {
			if live[i][j] != graphed[i][j] {
				t.Fatalf("SAME MODEL pos %d logit %d: graphs %.9g != live %.9g — capture/replay diverges from live "+
					"launches on identical buffers (not a build confound)", i, j, graphed[i][j], live[i][j])
			}
		}
	}
	t.Logf("same-model graph-vs-live (40 pos, cache=true): BIT-IDENTICAL")
}

// TestGemma4Graphs_offVsOffControl is the control the whole finding rests on: run the SAME prompt
// twice with graphs OFF both times (two identical live forwards) and compare. If this diverges under
// concurrent GPU load, the tiny forward is simply nondeterministic under contention and the
// "graphs-on diverges" result is an artifact of comparing two runs on a noisy GPU — NOT a graph bug.
// If it stays bit-exact under the same churn where graphs-on diverges, graphs are genuinely the cause.
func TestGemma4Graphs_offVsOffControl(t *testing.T) {
	dir := "../testdata/gemma4-moe-tiny"
	mc, rf := loadG4MoEGraphs(t, dir, false, true)
	defer mc.Close()
	r := rf.(*cudaResident)
	prompt := make([]int, 40)
	for i := range prompt {
		prompt[i] = (i*37 + 5) % 200
	}
	runLive := func() [][]float32 {
		r.graphs = false // force live both times
		out := make([][]float32, len(prompt))
		for i, tok := range prompt {
			l, err := rf.Forward(mc.EmbedResidentForTest(tok), i)
			if err != nil {
				t.Fatalf("forward pos %d: %v", i, err)
			}
			out[i] = append([]float32(nil), l...)
		}
		return out
	}
	a := runLive()
	b := runLive()
	for i := range a {
		for j := range a[i] {
			if a[i][j] != b[i][j] {
				t.Fatalf("CONTROL pos %d logit %d: run2 %.9g != run1 %.9g — the graphs-OFF forward is NONDETERMINISTIC "+
					"under contention; the graphs-on divergence is an artifact, not a graph bug", i, j, b[i][j], a[i][j])
			}
		}
	}
	t.Logf("control: two graphs-OFF runs BIT-IDENTICAL (40 pos) — the live forward is deterministic under this load")
}

// TestGemma4Graphs_bitExact_scaled runs the gate at the width that broke A′ zero-copy (the width that
// matters for the 26B), gated on a scaled fixture + heavy tests.
func TestGemma4Graphs_bitExact_scaled(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("GOINFER_HEAVY_TESTS unset")
	}
	dir := os.Getenv("GOINFER_MOE_SCALED_FIXTURE")
	if dir == "" {
		t.Skip("GOINFER_MOE_SCALED_FIXTURE unset")
	}
	graphsBitExact(t, dir, true)
}
