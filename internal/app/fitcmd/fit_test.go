package fitcmd

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/decoder/fit"
	"github.com/townsendmerino/goinfer/internal/runtime/giw"
)

// captureStdout swaps os.Stdout for the duration of fn and returns what was written — Run prints
// its report there (stderr stays for load progress/errors, matching every other subcommand in
// this repo, e.g. internal/pullcmd).
func captureStdout(t *testing.T, fn func()) string {
	t.Helper()
	orig := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	os.Stdout = w
	fn()
	w.Close()
	os.Stdout = orig
	var buf bytes.Buffer
	if _, err := buf.ReadFrom(r); err != nil {
		t.Fatalf("read pipe: %v", err)
	}
	return buf.String()
}

// TestFreeBytesFor_cpuUsesThePassedInValue pins that freeBytesFor's "cpu" branch reports EXACTLY the caller-supplied
// hostFreeBeforeLoad and does not re-query HostRAMAvailableBytes() after Run has loaded the model into that same RAM
// budget; a non-positive value reports unknown (HostRAMAvailableBytes() returning 0 means unknown). Origin (M-19,
// docs/audit-2026-09-10.md): docs/code-notes/internal-fitcmd.md#TestFreeBytesFor_cpuUsesThePassedInValue.
func TestFreeBytesFor_cpuUsesThePassedInValue(t *testing.T) {
	const want = 123456789
	if got, ok := freeBytesFor("cpu", want); !ok || got != want {
		t.Errorf("freeBytesFor(\"cpu\", %d) = (%d, %v), want (%d, true)", want, got, ok, want)
	}
	if got, ok := freeBytesFor("cpu", 0); ok || got != 0 {
		t.Errorf("freeBytesFor(\"cpu\", 0) = (%d, %v), want (0, false)", got, ok)
	}
}

func TestRun_noPathIsUsageError(t *testing.T) {
	if code := Run(nil); code != 2 {
		t.Errorf("Run(nil) = %d, want 2 (usage error)", code)
	}
	if code := Run([]string{"-ctx", "4096"}); code != 2 {
		t.Errorf("Run with only flags, no path = %d, want 2", code)
	}
}

func TestRun_missingCheckpointFails(t *testing.T) {
	if code := Run([]string{"/no/such/checkpoint/dir"}); code != 1 {
		t.Errorf("Run(missing path) = %d, want 1 (load error)", code)
	}
}

// TestRun_realFixtureReportsCPU is the end-to-end smoke test: testdata/llama-tiny is TRACKED in
// git, so this runs in CI unconditionally (same reasoning decoder/fitplan_test.go's dense row
// uses). Asserts Run succeeds and its stdout report at minimum names the "cpu" backend and a
// placement — CompiledBackends() always includes "cpu", so this must hold in every build,
// GPU-tagged or not.
func TestRun_realFixtureReportsCPU(t *testing.T) {
	var code int
	out := captureStdout(t, func() {
		code = Run([]string{"../../../testdata/llama-tiny", "-ctx", "512"})
	})
	if code != 0 {
		t.Fatalf("Run = %d, want 0; output:\n%s", code, out)
	}
	if !bytes.Contains([]byte(out), []byte("cpu")) {
		t.Errorf("output does not mention the cpu backend at all:\n%s", out)
	}
	for _, want := range []string{"RESIDENT", "EXPERT-CACHED", "WEIGHT-PAGED", "DECLINE"} {
		if bytes.Contains([]byte(out), []byte(want)) {
			return // found a real placement word — Plan ran and Run printed its result
		}
	}
	t.Errorf("output names no placement (RESIDENT/EXPERT-CACHED/WEIGHT-PAGED/DECLINE):\n%s", out)
}

// TestRun_pinnedCtxThatCannotFitDeclinesNotShrinks exercises the CtxPinned wiring specifically:
// an absurd -ctx must be refused (or fall to weight-paged on cpu), never silently reduced — the
// printed ctx in the header line must stay the one the user asked for.
func TestRun_pinnedCtxThatCannotFitDeclinesNotShrinks(t *testing.T) {
	var code int
	out := captureStdout(t, func() {
		code = Run([]string{"../../../testdata/llama-tiny", "-ctx", "999999999"})
	})
	if code != 0 {
		t.Fatalf("Run = %d, want 0 (a bad plan is reported, not a process failure); output:\n%s", code, out)
	}
	if !bytes.Contains([]byte(out), []byte("ctx=999999999")) {
		t.Errorf("header does not echo the pinned ctx unchanged:\n%s", out)
	}
	if bytes.Contains([]byte(out), []byte("RESIDENT")) {
		t.Errorf("an impossible pinned ctx must not report RESIDENT anywhere:\n%s", out)
	}
}

// TestRun_measureSelfMeasuresOnAdmittedBackend is tasks/task-fit-to-hardware.md §5's self-measure
// option: on the tracked llama-tiny fixture (CPU-only in this untagged build, so "cpu" is the
// only admitted backend and therefore the one selfMeasure picks), -measure must actually load
// and decode, printing a real "(measure)" line with a positive rate — not just echo the dry-run
// report above it.
func TestRun_measureSelfMeasuresOnAdmittedBackend(t *testing.T) {
	var code int
	out := captureStdout(t, func() {
		code = Run([]string{"../../../testdata/llama-tiny", "-ctx", "512", "-measure"})
	})
	if code != 0 {
		t.Fatalf("Run = %d, want 0; output:\n%s", code, out)
	}
	if !bytes.Contains([]byte(out), []byte("(measure) cpu")) {
		t.Fatalf("output has no '(measure) cpu' line — the probe did not run or picked a different backend:\n%s", out)
	}
	if !bytes.Contains([]byte(out), []byte("tok/s")) {
		t.Errorf("measure line has no tok/s rate:\n%s", out)
	}
}

// TestRun_measureOffByDefault confirms -measure is opt-in: without it, Run must never print a
// "(measure)" line — the probe is a real extra load+decode, not something every dry run should
// pay for silently (tasks/task-fit-to-hardware.md §0: "explicit flags stay, and win").
func TestRun_measureOffByDefault(t *testing.T) {
	out := captureStdout(t, func() {
		Run([]string{"../../../testdata/llama-tiny", "-ctx", "512"})
	})
	if bytes.Contains([]byte(out), []byte("(measure)")) {
		t.Errorf("output contains a '(measure)' line without -measure being passed:\n%s", out)
	}
}

// TestRun_measureRateExcludesPrefill pins that selfMeasure's printed rate excludes prefill: the clock starts after the
// FIRST token arrives, so the printed line must say so explicitly and must report decode steps counted as probeDecode-1,
// not probeDecode, which proves the first step was excluded from the denominator, not just relabeled. Origin (M-20,
// docs/audit-2026-09-10.md): docs/code-notes/internal-fitcmd.md#TestRun_measureRateExcludesPrefill.
func TestRun_measureRateExcludesPrefill(t *testing.T) {
	out := captureStdout(t, func() {
		Run([]string{"../../../testdata/llama-tiny", "-ctx", "512", "-measure"})
	})
	if !bytes.Contains([]byte(out), []byte("after the first token")) {
		t.Fatalf("measure line does not say it excludes the first token/prefill:\n%s", out)
	}
	if bytes.Contains([]byte(out), []byte("includes a")) {
		t.Errorf("measure line still uses the old 'includes a N-token prompt prefill' phrasing "+
			"(M-20 not actually fixed, just relabeled):\n%s", out)
	}
	// probeDecode is 32 (fit.go); the printed decode-step count must be 31 (32-1), proving the
	// first token's step was excluded from the count, not just from the label.
	if !bytes.Contains([]byte(out), []byte(" 31 decode steps ")) {
		t.Errorf("measure line does not report 31 decode steps (probeDecode=32 minus the excluded "+
			"first token) — the denominator may still include the first step:\n%s", out)
	}
}

// TestRun_closesFirstLoadBeforeMeasuring (M-20): selfMeasure loads the SAME checkpoint again, so Run must close its own
// first Load BEFORE calling selfMeasure; leaving it open holds two full quantized copies resident on the CPU backend,
// which can itself trip the memory guard the probe exists to measure honestly. A real large checkpoint tripping the
// guard can't be reproduced with this repo's tiny fixtures, so this is asserted structurally, the same technique
// TestTranscode_writesViaTempThenRenames (decoder) uses for an ordering property: parse Run's AST and confirm
// the statement that calls m.Close() (the loop over the loads) appears BEFORE the `if *measure` block that calls selfMeasure.
func TestRun_closesFirstLoadBeforeMeasuring(t *testing.T) {
	fset := token.NewFileSet()
	af, err := parser.ParseFile(fset, "fit.go", nil, 0)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	var fn *ast.FuncDecl
	ast.Inspect(af, func(n ast.Node) bool {
		if d, ok := n.(*ast.FuncDecl); ok && d.Name.Name == "Run" {
			fn = d
		}
		return true
	})
	if fn == nil {
		t.Fatal("Run not found — this guard is watching nothing")
	}
	render := func(e ast.Expr) string {
		var b strings.Builder
		_ = printer.Fprint(&b, fset, e)
		return b.String()
	}
	var closeCallIdx, measureIfIdx = -1, -1
	for i, stmt := range fn.Body.List {
		ast.Inspect(stmt, func(n ast.Node) bool { // the statement that closes m: a call, or the loop over the loads
			if _, deferred := n.(*ast.DeferStmt); deferred {
				return false // a deferred Close runs when Run returns, after selfMeasure: it closes nothing first
			}
			if call, ok := n.(*ast.CallExpr); ok {
				if sel, ok := call.Fun.(*ast.SelectorExpr); ok && sel.Sel.Name == "Close" && render(sel.X) == "m" {
					closeCallIdx = i
				}
			}
			return true
		})
		switch s := stmt.(type) {
		case *ast.IfStmt:
			ast.Inspect(s, func(n ast.Node) bool {
				if call, ok := n.(*ast.CallExpr); ok {
					if id, ok := call.Fun.(*ast.Ident); ok && id.Name == "selfMeasure" {
						measureIfIdx = i
					}
				}
				return true
			})
		}
	}
	if closeCallIdx == -1 {
		t.Fatal("no m.Close() call found in Run — this guard is watching the wrong function")
	}
	if measureIfIdx == -1 {
		t.Fatal("no selfMeasure call found in Run — this guard is watching the wrong function")
	}
	if closeCallIdx >= measureIfIdx {
		t.Errorf("m.Close() (statement %d) does not precede the selfMeasure call (statement %d) — "+
			"selfMeasure's own fresh Load can run while the first copy is still resident (M-20)",
			closeCallIdx, measureIfIdx)
	}
}

// TestFreshSidecar_findsTheOneChatAndServeBuild pins that sourceFor's cpu line finds the sidecar a default chat or serve load
// (--backend cpu) writes: this host's cpu target, <base>.int4.cpu-amd64.giw or .cpu-arm64.giw, not only the canonical
// target. Missing it means a full direct load of the .gguf instead of a mapped read through the sidecar.
func TestFreshSidecar_findsTheOneChatAndServeBuild(t *testing.T) {
	for _, e4 := range []bool{false, true} {
		t.Run(map[bool]string{false: "int8 head", true: "int4 head (the default since 2026-09-28)"}[e4], func(t *testing.T) {
			testFreshSidecar(t, e4)
		})
	}
}

// testFreshSidecar builds the sidecar a load with the given --embed-int4 writes: the cpu line must find it, and no other line. With
// embedInt4 it is <base>.int4.e4h.<target>.giw, the one a default CPU/CUDA/WebGPU load writes, not the plain-head name.
func testFreshSidecar(t *testing.T, embedInt4 bool) {
	raw, err := os.ReadFile("../../../testdata/glm-tiny.gguf")
	if err != nil {
		t.Skipf("tiny fixture: %v", err)
	}
	dir := t.TempDir()
	src := filepath.Join(dir, "tiny.gguf")
	if err := os.WriteFile(src, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	old := time.Now().Add(-time.Hour)
	if err := os.Chtimes(src, old, old); err != nil {
		t.Fatal(err)
	}
	if s, err := sourceFor(src, "int4", "", "cpu"); err != nil || s.path != src {
		t.Fatalf("before a sidecar was built fit opens %s (%v), want the .gguf", s.path, err)
	}
	// Backend "" keeps canonical int4 bytes in RAM, which the .giw writer needs to emit any target (the
	// production transcode does the same); a Backend "cpu" load on arm64 keeps only the row4 repack.
	m, err := decoder.Load(src, decoder.Options{Quant: "int4", EmbedInt4: embedInt4})
	if err != nil {
		t.Fatal(err)
	}
	target := decoder.GIWTargetForBackend("cpu")
	label := "int4"
	if embedInt4 {
		label = "int4.e4h"
	}
	blob, err := decoder.SerializeWeightsForTarget(m.Weights(), "tiny", target)
	m.Close()
	if err != nil {
		t.Fatal(err)
	}
	ver := fmt.Sprintf(".v%d.giw", decoder.GIWFormatVersion) // the name carries the format version (decoder.SidecarPath)
	cpuSidecar := filepath.Join(dir, "tiny."+label+"."+string(target)+ver)
	if target == decoder.GIWTargetNone {
		cpuSidecar = filepath.Join(dir, "tiny."+label+".canonical"+ver) // a GOARCH with no cpu-specific target
	}
	if err := os.WriteFile(cpuSidecar, giw.Write(blob, nil), 0o644); err != nil {
		t.Fatal(err)
	}
	s, err := sourceFor(src, "int4", "", "cpu")
	if err != nil || s.path != cpuSidecar {
		t.Fatalf("the cpu line opens %q (%v); want the cpu-target sidecar %s a default chat/serve load builds", s.path, err, filepath.Base(cpuSidecar))
	}
	if target != decoder.GIWTargetNone && s.loadBackend != "cpu" {
		t.Errorf("backend = %q, want cpu: a cpu-target sidecar can be row4-only, which only the literal cpu backend reads", s.loadBackend)
	}
	if _, err := decoder.Load(s.path, decoder.Options{Quant: "int4", Backend: s.loadBackend}); err != nil {
		t.Errorf("the found sidecar does not load with the backend sourceFor chose: %v", err)
	}
	// A load on another backend reads that backend's own sidecar, never the CPU's: a cuda or webgpu load would not open it, and
	// -measure loads the source on the backend itself, which cannot read a row4-only file.
	for _, be := range []string{"cuda", "webgpu", "metal"} {
		if o, err := sourceFor(src, "int4", "", be); err != nil || o.path != src {
			t.Errorf("the %s line opens %s (%v); want the .gguf, since no %s sidecar is built", be, o.path, err, be)
		}
	}
}

// TestPlanFor_metalPlansAtTheLoadsOwnContext: for an unpinned Metal line `fit` plans at the context
// decoder.Model.MetalFitCtx gives with Metal's default KV slots, the function the Metal load itself asks, so the dry run
// and the load cannot name different contexts for one memory figure. A wide-window copy of llama-tiny makes the candidate
// 16384; the budgets put the answer at the candidate, in the interior (two slots trading context), and at the default.
// A pinned -ctx and the other backends go to Plan as given.
func TestPlanFor_metalPlansAtTheLoadsOwnContext(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{"config.json", "generation_config.json", "model.safetensors"} {
		b, err := os.ReadFile(filepath.Join("../../../testdata/llama-tiny", name))
		if err != nil {
			t.Fatal(err)
		}
		if name == "config.json" {
			wide := strings.Replace(string(b), `"max_position_embeddings": 128`, `"max_position_embeddings": 131072`, 1)
			if wide == string(b) {
				t.Fatal("llama-tiny's config no longer carries max_position_embeddings 128")
			}
			b = []byte(wide)
		}
		if err := os.WriteFile(filepath.Join(dir, name), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	m, err := decoder.Load(dir, fitLoadOptions(fitSource{quant: "int4"}, 0))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	base := m.Plan("metal", 1<<40, fit.PlanRequest{Ctx: decoder.MetalCtxDefault, CtxPinned: true})
	if base.Placement != fit.PlacementResident {
		t.Fatalf("llama-tiny does not plan resident on metal: %s", base.Reason)
	}
	perBlock := m.MetalKVSlotBytes(decoder.KVBlockRows)
	null := m.ResidentKVBytes("metal", decoder.KVBlockRows, false, false) - perBlock // the pools' null blocks, once per load
	budget := func(positions int) int64 {
		return decoder.MetalFitMarginBytes + base.DenseBytes + base.HostCopyBytes + base.ScratchBytes + null + perBlock*int64(positions)/decoder.KVBlockRows
	}
	for _, c := range []struct {
		positions, want int
	}{{80000, decoder.MetalFitDefaultCtx}, {20000, 9984}, {5000, decoder.MetalCtxDefault}} {
		free := budget(c.positions)
		p := planFor(m, "metal", free, fit.PlanRequest{Ctx: defaultCtx})
		load := m.MetalFitCtx(decoder.MetalFitInput{FreeBytes: free, FreeKnown: true, KVSlots: decoder.MetalDefaultKVSlots})
		if p.Ctx != load.Ctx || p.Ctx != c.want {
			t.Errorf("room for %d positions: fit plans ctx %d, the load takes %d, want %d", c.positions, p.Ctx, load.Ctx, c.want)
		}
		if p.Placement != fit.PlacementResident || !strings.Contains(p.Reason, fmt.Sprintf("an unpinned load takes %d positions (%s)", load.Ctx, load.Why)) {
			t.Errorf("room for %d positions: %v, %q; want resident and the load's own reason", c.positions, p.Placement, p.Reason)
		}
	}
	if p := planFor(m, "metal", budget(80000), fit.PlanRequest{Ctx: 24000, CtxPinned: true}); p.Ctx != 24000 || strings.Contains(p.Reason, "unpinned") {
		t.Errorf("a pinned -ctx 24000: ctx %d, %q; want it planned as asked", p.Ctx, p.Reason)
	}
	if p, want := planFor(m, "cpu", budget(80000), fit.PlanRequest{Ctx: defaultCtx}), m.Plan("cpu", budget(80000), fit.PlanRequest{Ctx: defaultCtx}); p.Ctx != want.Ctx || p.Reason != want.Reason {
		t.Errorf("cpu: planFor changed Plan's answer (%d, %q) to (%d, %q)", want.Ctx, want.Reason, p.Ctx, p.Reason)
	}
}

// TestPlanFor_metalPricesEverySlotTheBuildKeeps: a serve load without --kv-sessions asks Metal for decoder.MetalDefaultKVSlots
// KV slots and keeps as many as fit beside the first (backend/metal's metalKVSlots), so fit's Metal line prices each slot kept:
// the first in Plan, the rest at decoder.Model.MetalKVBytes. It holds at a pinned -ctx and at the default context, prices one
// slot where a second does not fit, and one for a recurrent model, which a default load keeps to one slot. Each model is
// loaded as Run loads it (fitLoadOptions), since the slot request is a load option.
func TestPlanFor_metalPricesEverySlotTheBuildKeeps(t *testing.T) {
	load := func(dir string) *decoder.Model {
		t.Helper()
		m, err := decoder.Load(dir, fitLoadOptions(fitSource{quant: "int4"}, 0))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { m.Close() })
		return m
	}
	further := func(m *decoder.Model, ctx int) int64 {
		return m.MetalKVBytes(decoder.MetalDefaultKVSlots, ctx) - m.ResidentKVBytes("metal", ctx, false, false)
	}
	check := func(name string, m *decoder.Model, free int64, req fit.PlanRequest, ctx, slots int) {
		t.Helper()
		one := m.Plan("metal", 1<<40, fit.PlanRequest{Ctx: ctx, CtxPinned: true})
		want := one.NeedBytes()
		if slots > 1 {
			want += further(m, ctx)
		}
		p := planFor(m, "metal", free, req)
		if p.Placement != fit.PlacementResident || p.Ctx != ctx || p.NeedBytes() != want {
			t.Errorf("%s: %v at ctx %d needing %d B (%s); want resident at ctx %d needing %d B, %d KV slots", name, p.Placement, p.Ctx,
				p.NeedBytes(), p.Reason, ctx, want, slots)
		}
		if slots > 1 && !strings.Contains(p.Reason, fmt.Sprintf("keeps %d of the %d KV slots", slots, decoder.MetalDefaultKVSlots)) {
			t.Errorf("%s: the reason does not say how many KV slots the load keeps: %q", name, p.Reason)
		}
	}

	m := load("../../../testdata/llama-tiny")
	const ctx = 128 // llama-tiny's window
	if further(m, ctx) <= 0 {
		t.Fatal("llama-tiny prices no KV for a second Metal slot")
	}
	need := m.Plan("metal", 1<<40, fit.PlanRequest{Ctx: ctx, CtxPinned: true}).NeedBytes()
	pinned := fit.PlanRequest{Ctx: ctx, CtxPinned: true}
	check("pinned, room for both slots", m, need+further(m, ctx), pinned, ctx, 2)
	check("pinned, room for one slot", m, need+further(m, ctx)-1, pinned, ctx, 1)

	// The default context: a wide-window copy (as TestPlanFor_metalPlansAtTheLoadsOwnContext makes) with room for both slots at
	// MetalCtxDefault and not at the next block, so MetalFitCtx answers the floor, where fit once priced one slot.
	dir := t.TempDir()
	for _, name := range []string{"config.json", "generation_config.json", "model.safetensors"} {
		b, err := os.ReadFile(filepath.Join("../../../testdata/llama-tiny", name))
		if err != nil {
			t.Fatal(err)
		}
		if name == "config.json" {
			b = []byte(strings.Replace(string(b), `"max_position_embeddings": 128`, `"max_position_embeddings": 131072`, 1))
		}
		if err := os.WriteFile(filepath.Join(dir, name), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	wide := load(dir)
	atFloor := wide.Plan("metal", 1<<40, fit.PlanRequest{Ctx: decoder.MetalCtxDefault, CtxPinned: true}).NeedBytes() + further(wide, decoder.MetalCtxDefault)
	if c := wide.MetalFitCtx(decoder.MetalFitInput{FreeBytes: atFloor, FreeKnown: true, KVSlots: decoder.MetalDefaultKVSlots}).Ctx; c != decoder.MetalCtxDefault {
		t.Fatalf("the budget for both slots at the floor gives ctx %d, want %d", c, decoder.MetalCtxDefault)
	}
	check("the default context", wide, atFloor, fit.PlanRequest{Ctx: defaultCtx}, decoder.MetalCtxDefault, 2)
	// Room for one slot only, unpinned: the context is the floor, and the reason says one slot is kept without MetalFitCtx's
	// account of the two asked.
	oneAtFloor := wide.Plan("metal", 1<<40, fit.PlanRequest{Ctx: decoder.MetalCtxDefault, CtxPinned: true}).NeedBytes() +
		wide.MetalKVTableRaiseBytes(2, decoder.MetalCtxDefault) + wide.MetalKVSlotBytes(decoder.MetalCtxDefault) - 1
	if p := planFor(wide, "metal", oneAtFloor, fit.PlanRequest{Ctx: defaultCtx}); !strings.Contains(p.Reason, "keeps 1 of the 2 KV slots") ||
		!strings.Contains(p.Reason, fmt.Sprintf("an unpinned load takes %d positions (the default)", decoder.MetalCtxDefault)) ||
		strings.Contains(p.Reason, "2 KV slots leave") {
		t.Errorf("one slot of two at the floor, unpinned: %q", p.Reason)
	}

	rec := load("../../../testdata/qwen35-tiny")
	recNeed := rec.Plan("metal", 1<<40, fit.PlanRequest{Ctx: 64, CtxPinned: true}).NeedBytes()
	check("recurrent: one slot", rec, recNeed+further(rec, 64), fit.PlanRequest{Ctx: 64, CtxPinned: true}, 64, 1)
}

func TestSelfMeasureOptions_propagatesPlannedPlacementAndWindowedKV(t *testing.T) {
	loadOpts := fitLoadOptions(fitSource{quant: "int4", group: 64}, 16384)
	if !loadOpts.ResidentWindowedKV {
		t.Fatal("fitLoadOptions did not set ResidentWindowedKV:true (chat/serve default)")
	}
	src := fitSource{path: "model.giw", quant: "int4", group: 64}
	plan := fit.Plan{Placement: fit.PlacementExpertCached, Ctx: 40960}
	opts := selfMeasureOptions("cuda", src, plan, true, false)
	if !opts.MoECacheExperts {
		t.Error("selfMeasureOptions did not enable MoECacheExperts for PlacementExpertCached")
	}
	if !opts.ResidentWindowedKV {
		t.Error("selfMeasureOptions did not enable ResidentWindowedKV")
	}
	if opts.ResidentContext != 40960 {
		t.Errorf("selfMeasureOptions ResidentContext = %d, want 40960", opts.ResidentContext)
	}
	if opts.KVPrecision != "f16" {
		t.Errorf("selfMeasureOptions KVPrecision = %q, want f16", opts.KVPrecision)
	}
	resOpts := selfMeasureOptions("metal", src, fit.Plan{Placement: fit.PlacementResident, Ctx: 16384}, false, true)
	if resOpts.MoECacheExperts {
		t.Error("selfMeasureOptions enabled MoECacheExperts for PlacementResident")
	}
	if resOpts.KVQuant != "i8" {
		t.Errorf("selfMeasureOptions KVQuant = %q, want i8", resOpts.KVQuant)
	}
}
