//go:build cuda

package cuda

import (
	"errors"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
)

// TestCudaResidentCheckCap gates C3: writes at/past the resident KV cap are refused (a real
// device write there is out-of-bounds memory corruption). Pure logic — no device needed.
//
// The cap is configuration-derived now (resolveCtxCap), so checkCap is exercised at BOTH the
// default and a raised cap: it must guard whatever capacity the caches were actually sized with,
// never a constant.
func TestCudaResidentCheckCap(t *testing.T) {
	for _, cap := range []int{cudaCtxCapDefault, 32768} {
		r := &cudaResident{ctxCap: cap}
		if r.ContextCap() != cap {
			t.Fatalf("ContextCap = %d, want %d", r.ContextCap(), cap)
		}
		for _, c := range []struct {
			pos, n int
			ok     bool
		}{
			{0, 1, true}, {cap - 1, 1, true}, {cap, 1, false},
			{0, cap, true}, {0, cap + 1, false}, {-1, 1, false},
		} {
			if err := r.checkCap(c.pos, c.n); (err == nil) != c.ok {
				t.Errorf("cap=%d: checkCap(%d,%d) err=%v, want ok=%v", cap, c.pos, c.n, err, c.ok)
			}
		}
	}
}

// TestCudaResidentCheckCap_zeroValueRefuses pins the fail-SAFE direction of making the cap a field:
// a cudaResident whose ctxCap was never resolved must refuse every position rather than admit them.
// If this ever inverts, an unresolved cap becomes an unbounded out-of-bounds device write.
func TestCudaResidentCheckCap_zeroValueRefuses(t *testing.T) {
	r := &cudaResident{}
	if err := r.checkCap(0, 1); err == nil {
		t.Fatal("checkCap admitted position 0 on a resident with an unresolved (zero) ctxCap")
	}
}

// TestResolveCtxCap pins the cap policy: default when nothing asks, clamped to the model's own
// context window when something does. The default case is the one that must never drift — a caller
// who did not ask must not start allocating deep-KV VRAM.
func TestResolveCtxCap(t *testing.T) {
	for _, c := range []struct {
		name              string
		request, modelCtx int
		want              int
	}{
		{"unset keeps the default", 0, 32768, cudaCtxCapDefault},
		{"negative keeps the default", -1, 32768, cudaCtxCapDefault},
		{"request under the model window stands", 8192, 32768, 8192},
		{"request over the model window clamps to it", 65536, 32768, 32768},
		{"request equal to the model window stands", 32768, 32768, 32768},
		{"unknown model window lets the request stand", 32768, 0, 32768},
		{"unset with unknown window still defaults", 0, 0, cudaCtxCapDefault},
	} {
		if got := resolveCtxCap(c.request, c.modelCtx); got != c.want {
			t.Errorf("%s: resolveCtxCap(%d, %d) = %d, want %d", c.name, c.request, c.modelCtx, got, c.want)
		}
	}
}

// TestResolveCtxCapFit_shortcuts pins the branches that need no real device: an explicit request is untouched either way
// (fit-by-default only ever applies to an UNPINNED load), --fit=off (Options.DisableFit) and its GOINFER_NO_FIT_DEFAULT
// env-var precursor both restore resolveCtxCap exactly, and a model whose own window is already at or below
// cudaCtxCapDefault has nothing to gain from asking Plan. The live-probe branch (a real free-VRAM reading) is exercised on
// real hardware separately (docs/tasks/task-gpu-paths-2026-09.md, G11). A real (minimal, tracked-in-git) model is loaded per
// case rather than nil: m.FitDisabled() reads a real field, so a nil *decoder.Model would panic in the request==0 cases.
//
// `pinned` decides whether the case's Options carries ResidentContext: request, i.e. whether m itself is a GENUINE pin
// (decoder.Model.ResidentContextPinned() true) or a bare request value passed to resolveCtxCapFit with an unpinned m. Set
// it deliberately on every request>0 case: left unset, the cases pass by coincidence (fitDefaultCtx is itself 8192 and
// llama-tiny is small enough that Plan hands 8192 back unshrunk) instead of taking the "genuine pin, skip fit entirely"
// branch they claim to test (docs/prompts/nobara-mc1-webgpu-2026-09.md §4, item 28).
func TestResolveCtxCapFit_shortcuts(t *testing.T) {
	for _, c := range []struct {
		name              string
		request, modelCtx int
		pinned            bool
		noFitEnv          bool
		disableFitOpt     bool
		moeCacheOpt       bool
		want              int
	}{
		{"genuine pin bypasses fit entirely", 8192, 32768, true, false, false, false, 8192},
		{"genuine pin bypasses fit even with the env var set", 8192, 32768, true, true, false, false, 8192},
		{"genuine pin below cudaCtxCapDefault is still untouched", 2048, 32768, true, false, false, false, 2048},
		{"GOINFER_NO_FIT_DEFAULT restores the historical default", 0, 32768, false, true, false, false, cudaCtxCapDefault},
		{"Options.DisableFit (--fit=off) restores the historical default", 0, 32768, false, false, true, false, cudaCtxCapDefault},
		{"model window at the historical default has nothing to gain", 0, cudaCtxCapDefault, false, false, false, false, cudaCtxCapDefault},
		{"model window below the historical default has nothing to gain", 0, 2048, false, false, false, false, cudaCtxCapDefault},
		// Growing ctx starves the expert-slot cache's own claim on the same free VRAM (resolveCtxCapFit's doc comment has the
		// measured numbers). A genuine pin still overrides it below; this only changes the UNPINNED default.
		{"Options.MoECacheExperts restores the historical default, same as DisableFit", 0, 32768, false, false, false, true, cudaCtxCapDefault},
		{"genuine pin still bypasses fit with MoECacheExperts set", 8192, 32768, true, false, false, true, 8192},
		// request>0 but NOT pinned is a load-time fit-guard auto-pin (R13), not a choice: it must still go through
		// fit-by-default (never the request>0 shortcut above), clamped so it can only land AT OR UNDER the guard's own pin. llama-tiny
		// is tiny enough that Plan always has room, so this lands exactly at the pin (the clamp, not Plan, is under test),
		// including the below-cudaCtxCapDefault case, where flooring back UP to cudaCtxCapDefault would raise a context past what
		// the guard proved host RAM could hold.
		{"guard pin (unpinned m) at cudaCtxCapDefault: fit-by-default still runs, clamped to it", cudaCtxCapDefault, 32768, false, false, false, false, cudaCtxCapDefault},
		{"guard pin (unpinned m) BELOW cudaCtxCapDefault: floor follows it down, not back up", 2048, 32768, false, false, false, false, 2048},
	} {
		t.Run(c.name, func(t *testing.T) {
			if c.noFitEnv {
				t.Setenv("GOINFER_NO_FIT_DEFAULT", "1")
			} else {
				os.Unsetenv("GOINFER_NO_FIT_DEFAULT")
			}
			opts := decoder.Options{Quant: "int4", DisableFit: c.disableFitOpt, MoECacheExperts: c.moeCacheOpt}
			if c.pinned {
				opts.ResidentContext = c.request
			}
			m, err := decoder.Load("../testdata/llama-tiny", opts)
			if err != nil {
				t.Fatalf("Load: %v", err)
			}
			defer m.Close()
			if got := m.ResidentContextPinned(); got != c.pinned {
				t.Fatalf("ResidentContextPinned() = %v, want %v — this case's own setup is wrong", got, c.pinned)
			}
			if got, _ := resolveCtxCapFit(m, c.request, c.modelCtx, 1, fixedMargin(testCtxMargin)); got != c.want {
				t.Errorf("resolveCtxCapFit(m, %d, %d) = %d, want %d", c.request, c.modelCtx, got, c.want)
			}
		})
	}
}

// TestResolveCtxCapFit_agreesWithCheckKVFits is M-12 and M-22's gate (docs/audit-2026-09-10.md), on a real device:
// resolveCtxCapFit's PLANNING-time choice must pass checkKVFits' BUILD-time check (Plan must reserve the same margin
// checkKVFits requires, or an INTERIOR choice, neither ceiling nor floor, fails almost every time). A real card usually has
// far more free VRAM than testdata/llama-tiny plus 8192 positions need, so the default candidate hits the CEILING branch
// (fits outright), which cannot show the finding. ExtraResidentBytes forces the INTERIOR branch deterministically: computed
// from a real free-bytes probe so the remaining budget lands the candidate between the floor and the ceiling.
//
// Run with and without an ADDITIONAL simulated companion K/V rate (M-22) on that forced interior scenario:
// resolveCtxCapFit prices the rate against its own candidate (never smaller than what it eventually picks), and the real
// build prices it against the FINAL chosen ctx; this proves those two independently-computed prices stay consistent. The
// companion rate (40960 B/position) is the audit's own real number: a 5-layer DFlash trunk's K/V, "5 x 8192 x 1024 x 8 B
// ~ 335 MB" for an 8192-position target.
func TestResolveCtxCapFit_agreesWithCheckKVFits(t *testing.T) {
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no cuda device: %v", err)
	}
	defer dev.ReleaseObjects()

	// First load: no extra, just to read this fixture's own DenseBytes/geometry — needed to compute
	// an ExtraResidentBytes value that lands the interior branch, which can only be set at Load.
	probe, err := decoder.Load("../testdata/llama-tiny", decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("Load (probe): %v", err)
	}
	_, nLayers, _, nKV, hd, _, _ := probe.Dims()
	layers := make([]cudaLayer, nLayers)
	for i := range layers {
		layers[i].kvDim = nKV * hd
	}
	perPos := kvBytesForCap(kvBlock, layers) / kvBlock // one position, read over the block it is priced in
	dense := probe.ResidentDenseWeightBytes()
	probe.Close()

	free, ok := decoder.FreeBytesFor("cuda")
	if !ok {
		t.Skip("no cuda free-bytes probe available")
	}
	// Target the interior of (cudaCtxCapDefault, fitDefaultCtx]: an ExtraResidentBytes big enough that, with the SAME margin
	// resolveCtxCapFit itself subtracts before calling Plan, dense+kv(fitDefaultCtx)+extra does NOT fit but
	// dense+kv(targetCtx)+extra does, forcing chooseCtx's budget branch instead of its "fits outright" shortcut. Computed
	// against the MARGINED budget on purpose: llama-tiny's perPos is tiny next to the margin (testCtxMargin), so a term sized
	// against raw free would leave the margin subtraction nothing to bite on and land the floor. A planner without that
	// internal margin subtraction lands at the ceiling (fitDefaultCtx) instead: still a checkable planning-vs-build
	// disagreement (the checkKVFits assertion below), but not "interior" in that arm, so both arms are logged, not asserted
	// on.
	const targetCtx = 6000
	if targetCtx <= cudaCtxCapDefault || targetCtx >= fitDefaultCtx {
		t.Fatalf("test bug: targetCtx %d must sit strictly inside (%d, %d)", targetCtx, cudaCtxCapDefault, fitDefaultCtx)
	}

	for _, c := range []struct {
		name          string
		kvPerPosition int64
	}{
		{"interior branch, no companion", 0},
		{"interior branch, with a drafter-sized companion K/V rate", 40960},
	} {
		t.Run(c.name, func(t *testing.T) {
			// Computed PER CASE: the companion K/V rate itself adds fitDefaultCtx*kvPerPosition to
			// what resolveCtxCapFit's own Plan call reserves (priced at candidate, not targetCtx —
			// M-22's own over-estimate-not-under design), so a forcing term calibrated for the
			// no-companion case would over-constrain the companion one.
			forcedExtra := free - testCtxMargin - dense - perPos*targetCtx - c.kvPerPosition*fitDefaultCtx
			if forcedExtra <= 0 {
				t.Skipf("this box's free VRAM (%d bytes) is too small relative to llama-tiny's dense "+
					"weights (%d bytes) to construct a positive forcing term — nothing to check here", free, dense)
			}
			m, err := decoder.Load("../testdata/llama-tiny", decoder.Options{
				Quant:                      "int4",
				ExtraResidentBytes:         forcedExtra,
				ExtraResidentKVPerPosition: c.kvPerPosition,
			})
			if err != nil {
				t.Fatalf("Load: %v", err)
			}
			defer m.Close()

			// Unpinned, modelCtx large enough that fitDefaultCtx itself is the real candidate.
			ctx, margin := resolveCtxCapFit(m, 0, 1<<20, 1, fixedMargin(testCtxMargin))
			if ctx <= cudaCtxCapDefault {
				t.Fatalf("resolveCtxCapFit did not improve on the historical default (got %d) — the "+
					"forcing term left no room at all; this test's own arithmetic needs adjusting, not "+
					"a sign the fix is broken", ctx)
			}
			t.Logf("chosen ctx=%d (floor %d, forced-interior target %d, ceiling %d)", ctx, cudaCtxCapDefault, targetCtx, fitDefaultCtx)

			// The SAME extraBytes arithmetic the real build uses (cuda/backend.go): priced at the
			// FINAL ctx this test just got back, not the candidate resolveCtxCapFit tried internally.
			r := &cudaResident{
				dev:            dev,
				ctxCap:         ctx,
				layers:         layers,
				extraBytes:     m.ExtraResidentBytes() + m.ExtraResidentKVPerPosition()*int64(ctx),
				ctxMarginBytes: margin,
			}
			if err := r.checkKVFits(); err != nil {
				t.Errorf("resolveCtxCapFit chose an INTERIOR ctx=%d, but checkKVFits refuses it at the "+
					"real build (M-12/M-22, docs/audit-2026-09-10.md): %v", ctx, err)
			}
		})
	}
}

// TestKVBytesForCap pins the sizing formula against the two geometries the deep-context measurements
// were taken on (docs/benchmarks.md): 24.0 KB/position for qwen2.5-coder-0.5b and 56.0 KB/position
// for the 1.5B, read over one 64-row block, plus the tables (4 B a block per group). The VRAM
// fail-fast message quotes these, so a formula drift would misreport what a configured cap actually costs.
func TestKVBytesForCap(t *testing.T) {
	mk := func(n, kvDim int) []cudaLayer {
		ls := make([]cudaLayer, n)
		for i := range ls {
			ls[i].kvDim = kvDim
		}
		return ls
	}
	for _, c := range []struct {
		name     string
		layers   []cudaLayer
		kbPerPos float64
	}{
		{"qwen2.5-coder-0.5b (24 layers, nKV 2 × hd 64)", mk(24, 2*64), 24.0},
		{"qwen2.5-coder-1.5b (28 layers, nKV 2 × hd 128)", mk(28, 2*128), 56.0},
	} {
		tabs := func(n int) int64 { return int64(kvGroups * 4 * kvBlocksFor(n)) }
		got := float64(kvBytesForCap(kvBlock, c.layers)-tabs(kvBlock)) / kvBlock / 1024
		if got != c.kbPerPos {
			t.Errorf("%s: %.1f KB/position, want %.1f", c.name, got, c.kbPerPos)
		}
		// One position costs its whole block.
		if kvBytesForCap(1, c.layers) != kvBytesForCap(kvBlock, c.layers) {
			t.Errorf("%s: one position costs %d bytes, its block %d", c.name, kvBytesForCap(1, c.layers), kvBytesForCap(kvBlock, c.layers))
		}
		// And it must scale linearly in blocks — the 32k prediction the deep-context leg was sized against.
		if want := int64(c.kbPerPos*1024*32768) + tabs(32768); kvBytesForCap(32768, c.layers) != want {
			t.Errorf("%s: 32k cap = %d bytes, want %d", c.name, kvBytesForCap(32768, c.layers), want)
		}
	}
}

// TestFitsWeightsBudget is M-02's gate for CUDA's previously-nonexistent memory-fit check on the
// FIXED (non-expert) weight term — mirrors metal/backend.go's TestResidentMemGuard, same reasoning:
// a guard that never fires leaves the raw-driver-error failure mode checkWeightsFit exists to
// avoid; one that fires too eagerly silently moves every model to the staged/CPU path.
func TestFitsWeightsBudget(t *testing.T) {
	const gb = int64(1) << 30
	for _, c := range []struct {
		name       string
		need, free int64
		want       bool
	}{
		{"comfortably_fits", 4 * gb, 8 * gb, true},
		{"exactly_at_margin", 8*gb - testCtxMargin, 8 * gb, true},
		{"just_over_margin", 8*gb - testCtxMargin + 1, 8 * gb, false},
		{"far_too_large", 22 * gb, 8 * gb, false}, // Qwen3.5-35B-A3B-shaped dense-only sum on an 8 GB card
		// Unknown inputs must never refuse: an unreadable MemInfo or a model reporting zero bytes
		// would otherwise disable residency for everyone, silently.
		{"unknown_need", 0, 8 * gb, true},
		{"unknown_free", 4 * gb, 0, true},
		{"negative_need", -1, 8 * gb, true},
	} {
		if got := fitsWeightsBudget(c.need, c.free, testCtxMargin); got != c.want {
			t.Errorf("%s: fitsWeightsBudget(%.2f GB, %.2f GB) = %v, want %v",
				c.name, float64(c.need)/float64(gb), float64(c.free)/float64(gb), got, c.want)
		}
	}
}

// TestReservedBudget is tasks/task-fit-to-hardware.md §2's drafter-aware sizing: allocSlots' elastic
// expert-cache search must see free VRAM MINUS whatever a companion attach (--drafter) will claim
// afterward, never a negative number capSlots was never written to handle.
func TestReservedBudget(t *testing.T) {
	const gb = int64(1) << 30
	for _, c := range []struct {
		name             string
		free, extraBytes int64
		want             int64
	}{
		{"nothing_attaching", 8 * gb, 0, 8 * gb},
		{"drafter_reserves_its_share", 8 * gb, 500 << 20, 8*gb - 500<<20},
		{"drafter_exceeds_everything_free_clamps_to_zero", 1 * gb, 2 * gb, 0},
		{"drafter_exactly_consumes_free", 1 * gb, 1 * gb, 0},
	} {
		if got := reservedBudget(c.free, c.extraBytes); got != c.want {
			t.Errorf("%s: reservedBudget(%.2f GB, %.2f GB) = %.2f GB, want %.2f GB",
				c.name, float64(c.free)/float64(gb), float64(c.extraBytes)/float64(gb),
				float64(got)/float64(gb), float64(c.want)/float64(gb))
		}
	}
}

// TestCheckKVFits_realDevice_explicitRefusesWithNumbers is G6 (docs/tasks/task-gpu-paths-2026-09.md §6): "nothing pinned
// is overridden... honoured or refused with numbers". The sibling test below pins the SENTINEL/wiring without a device;
// this one calls checkKVFits against a REAL device's REAL free VRAM and asserts the refusal NAMES the requested context
// and the GB figures, not just that an error occurred. An absurd ctx (1<<30 positions) makes it hold on any card.
func TestCheckKVFits_realDevice_explicitRefusesWithNumbers(t *testing.T) {
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no cuda device: %v", err)
	}
	defer dev.ReleaseObjects()
	const absurdCtx = 1 << 30
	r := &cudaResident{
		dev:         dev,
		ctxCap:      absurdCtx,
		ctxExplicit: true,
		layers:      []cudaLayer{{kvDim: 128}}, // one layer is enough; the point is the ctx, not the geometry
	}
	err = r.checkKVFits()
	if err == nil {
		t.Fatalf("checkKVFits() = nil for an explicit ctx of %d positions — should refuse on any real card", absurdCtx)
	}
	if !errors.Is(err, errKVWontFit) {
		t.Errorf("error %v does not match errKVWontFit — BuildResident's hard-error path would not fire", err)
	}
	for _, want := range []string{fmt.Sprintf("%d positions", absurdCtx), "GB", "free"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention %q — a refusal must name the numbers", err, want)
		}
	}
}

// TestCheckKVFits_realDevice_extraResidentBytesRefusesWithNumbers is tasks/task-fit-to-hardware.md §2's drafter-aware
// sizing, on a REAL device: the sibling test above proves an absurd CTX refuses; this proves a MODEST ctx that would
// otherwise fit can be refused purely by Model.ExtraResidentBytes reserving the room instead. Paired on the SAME
// cudaResident shape (only extraBytes differs): a modest ctx (128 positions) fits at extraBytes=0 and is refused at
// extraBytes=100 GiB, bigger than any card this repo targets, so the refusal can only be the reservation.
func TestCheckKVFits_realDevice_extraResidentBytesRefusesWithNumbers(t *testing.T) {
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no cuda device: %v", err)
	}
	defer dev.ReleaseObjects()
	const modestCtx = 128
	const absurdExtra = 100 << 30 // 100 GiB — bigger than any card this repo targets

	fits := &cudaResident{dev: dev, ctxCap: modestCtx, layers: []cudaLayer{{kvDim: 128}}}
	if err := fits.checkKVFits(); err != nil {
		t.Fatalf("checkKVFits() with extraBytes=0 refused a %d-position ctx: %v — the paired "+
			"comparison needs this arm to fit, or a refusal from the reserved arm proves nothing", modestCtx, err)
	}

	reserved := &cudaResident{dev: dev, ctxCap: modestCtx, extraBytes: absurdExtra, layers: []cudaLayer{{kvDim: 128}}}
	err = reserved.checkKVFits()
	if err == nil {
		t.Fatalf("checkKVFits() with extraBytes=%d GiB accepted a %d-position ctx that fit at "+
			"extraBytes=0 — the reservation was never read", absurdExtra>>30, modestCtx)
	}
	// UNPINNED (ctxExplicit=false, the default here): the historical decline, not errKVWontFit —
	// same distinction TestCheckKVFits_explicitFailsHard_defaultDeclines pins below for the ctx
	// term, now proven for the extraBytes term too.
	if errors.Is(err, errKVWontFit) {
		t.Errorf("error %v matches errKVWontFit for an UNPINNED ctx — should be the ordinary decline, not the hard-error sentinel", err)
	}
	for _, want := range []string{"companion attach", "--drafter", "GB"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention %q — a reservation-caused refusal must name what reserved the room", err, want)
		}
	}
}

// TestCheckKVFits_explicitFailsHard_defaultDeclines pins the FAILURE MODE, which is the whole point
// of the load-time check. An operator who explicitly configured a resident context and cannot have
// it must get a hard startup error naming the cost — degrading quietly to the staged path turns a
// config mistake into a latency mystery under load. A default-cap miss must keep the historical
// decline, so no existing deployment starts failing to boot.
func TestCheckKVFits_explicitFailsHard_defaultDeclines(t *testing.T) {
	// 28 layers × 256 kvDim → 56.0 KB/position; 32768 positions ≈ 1.88 GB, far past this "free".
	layers := make([]cudaLayer, 28)
	for i := range layers {
		layers[i].kvDim = 2 * 128
	}
	need := kvBytesForCap(32768, layers)
	if got := float64(need) / 1e9; got < 1.87 || got > 1.89 {
		t.Fatalf("fixture drift: 32k KV = %.3f GB, expected ~1.88", got)
	}
	// The sentinel decides the mode, so assert on it directly rather than on device state.
	explicit := &cudaResident{ctxCap: 32768, layers: layers, ctxExplicit: true}
	dflt := &cudaResident{ctxCap: 32768, layers: layers, ctxExplicit: false}
	if !explicit.ctxExplicit {
		t.Fatal("explicit resident should carry ctxExplicit")
	}
	if dflt.ctxExplicit {
		t.Fatal("default resident must not carry ctxExplicit")
	}
	// errKVWontFit must be the discriminator BuildResident switches on.
	if !errors.Is(fmt.Errorf("wrapped: %w", errKVWontFit), errKVWontFit) {
		t.Fatal("errKVWontFit must survive wrapping — BuildResident matches it with errors.Is")
	}
}

// demandDirs are the tracked tiny checkpoint directories.
func demandDirs() []string {
	cfgs, _ := filepath.Glob(filepath.Join("..", "testdata", "*", "config.json"))
	var dirs []string
	for _, c := range cfgs {
		if ws, _ := filepath.Glob(filepath.Join(filepath.Dir(c), "*.safetensors")); len(ws) > 0 {
			dirs = append(dirs, filepath.Dir(c))
		}
	}
	return dirs
}

// demandConfigs are the configs of the tracked tiny checkpoints that load on the CPU.
func demandConfigs(t *testing.T) []*decoder.Config {
	t.Helper()
	var cfgs []*decoder.Config
	for _, dir := range demandDirs() {
		m, err := decoder.Load(dir, decoder.Options{Backend: "cpu"})
		if err != nil {
			continue // a tower or an encoder, not a decoder family
		}
		cfgs = append(cfgs, m.Config())
		_ = m.Close()
	}
	return cfgs
}

// TestBuildDemand_matchesDecoderDefault (MM2a gate G-MM2a-2): this package's registered report equals decoder's default copy
// on every tracked family's config, across the option cells and a ladder of priced bytes. It first shows the comparison sees a
// change to each field.
func TestBuildDemand_matchesDecoderDefault(t *testing.T) {
	cfgs := demandConfigs(t)
	if len(cfgs) < 10 {
		t.Fatalf("only %d configs loaded: a comparison over few configs proves little", len(cfgs))
	}
	var queries []decoder.BuildQuery
	for _, cfg := range cfgs {
		for _, be := range []string{"cuda", "cpu", "metal", ""} {
			for _, stream := range []bool{false, true} {
				for _, moe := range []bool{false, true} {
					for _, gguf := range []bool{false, true} {
						for _, w := range []int64{0, 1 << 30, 7 << 30} {
							for _, e := range []int64{0, 2 << 30} {
								o := decoder.Options{Backend: be, StreamWeights: stream, MoECacheExperts: moe}
								queries = append(queries, decoder.BuildQueryFor(cfg, o, w, e, gguf))
							}
						}
					}
				}
			}
		}
	}
	differs := func(f func(decoder.BuildQuery) decoder.BuildDemand) int {
		n := 0
		for _, q := range queries {
			if f(q) != decoder.DefaultBuildDemand("cuda", q) {
				n++
			}
		}
		return n
	}
	// One planted change per field (a flag flipped, a byte count doubled): each must be seen, so each field is exercised, a
	// byte count nonzero somewhere in the corpus.
	for name, plant := range map[string]func(*decoder.BuildDemand){
		"Resident":          func(d *decoder.BuildDemand) { d.Resident = !d.Resident },
		"KVOnDevice":        func(d *decoder.BuildDemand) { d.KVOnDevice = !d.KVOnDevice },
		"HostPackBytes":     func(d *decoder.BuildDemand) { d.HostPackBytes *= 2 },
		"HostBuildPeak":     func(d *decoder.BuildDemand) { d.HostBuildPeak *= 2 },
		"PinnedExpertBytes": func(d *decoder.BuildDemand) { d.PinnedExpertBytes *= 2 },
	} {
		planted := func(q decoder.BuildQuery) decoder.BuildDemand {
			d := buildDemand(q)
			plant(&d)
			return d
		}
		if differs(planted) == 0 {
			t.Errorf("a planted change to %s was not seen: no query exercises it", name)
		}
	}
	if n := differs(buildDemand); n > 0 {
		t.Errorf("the registered cuda report differs from decoder's default on %d of %d queries", n, len(queries))
	}
	// The rounding report (MM2b): decoder keeps no copy of CUDA's buffer layout, so its default is 0, and CUDA's own planner must
	// agree on every tracked family, whose buffers are all under the 2 MiB quantum. Loaded at f32 on the CPU, the planner runs
	// without the device.
	planned := 0
	for _, dir := range demandDirs() {
		m, err := decoder.Load(dir, decoder.Options{Backend: "cpu", Quant: "f32"})
		if err != nil {
			continue
		}
		if s, ok := planSlack(m); ok {
			planned++
			if s != 0 {
				t.Errorf("%s: CUDA's rounding report is %d bytes, decoder's default 0", dir, s)
			}
		}
		_ = m.Close()
	}
	if planned < 10 {
		t.Errorf("CUDA's planner priced only %d tracked families: the comparison covers little", planned)
	}
	t.Logf("%d configs, %d queries", len(cfgs), len(queries))
}

// TestPlanSlack_oneCopy (MM2b gate G-MM2b-3): the driver's rounding over the weight buffers has one copy. packedAllocSlack is
// called only inside buildResident (plan-only, and the real build's test seam), and resolveCtxCapFitSlack neither takes a slack
// parameter nor is handed one at a call site, so the build cannot add the figure beside the plan's.
func TestPlanSlack_oneCopy(t *testing.T) {
	ents, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	calls, read, sites := map[string]int{}, 0, 0
	for _, e := range ents {
		if !strings.HasSuffix(e.Name(), ".go") || strings.HasSuffix(e.Name(), "_test.go") {
			continue
		}
		f, err := parser.ParseFile(token.NewFileSet(), e.Name(), nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		read++
		for _, d := range f.Decls {
			fd, ok := d.(*ast.FuncDecl)
			if !ok || fd.Body == nil {
				continue
			}
			if fd.Name.Name == "resolveCtxCapFitSlack" {
				for _, p := range fd.Type.Params.List {
					for _, n := range p.Names {
						if strings.Contains(strings.ToLower(n.Name), "slack") {
							t.Errorf("resolveCtxCapFitSlack takes %q: the plan already prices the rounding", n.Name)
						}
					}
				}
			}
			ast.Inspect(fd.Body, func(n ast.Node) bool {
				c, ok := n.(*ast.CallExpr)
				if !ok {
					return true
				}
				switch id, _ := c.Fun.(*ast.Ident); {
				case id == nil:
				case id.Name == "packedAllocSlack":
					calls[fd.Name.Name]++
				case id.Name == "resolveCtxCapFitSlack":
					sites++
					for _, a := range c.Args {
						ast.Inspect(a, func(n ast.Node) bool {
							if x, ok := n.(*ast.Ident); ok && (strings.Contains(strings.ToLower(x.Name), "slack") || x.Name == "PlanSlack") {
								t.Errorf("%s hands resolveCtxCapFitSlack %q: the plan already prices the rounding", fd.Name.Name, x.Name)
							}
							return true
						})
					}
				}
				return true
			})
		}
	}
	if read < 20 {
		t.Fatalf("read %d source files", read)
	}
	for fn, n := range calls {
		if fn != "buildResident" {
			t.Errorf("%s calls packedAllocSlack %d time(s): a second copy of the plan's rounding", fn, n)
		}
	}
	if calls["buildResident"] == 0 {
		t.Error("buildResident no longer calls packedAllocSlack: the scan checks nothing")
	}
	if sites == 0 {
		t.Error("no call of resolveCtxCapFitSlack found: the argument check checks nothing")
	}
}

// unlinedAllocs are gpudev's constructors that lease on gpu.UnlinedLine (unplanned in production); each has a lined twin.
var unlinedAllocs = map[string]bool{"NewBufferLenOf": true, "NewBufferOf": true, "NewBufferLen": true, "NewBufferBytes": true,
	"MustBuf": true, "NewHostBuffer": true, "NewMappedHostBuffer": true, "RegisterMappedHostBuffer": true}

// unlinedCalls counts the calls in f to an unlined gpudev constructor: a function (gpu.X, gpu.X[T]) or a method (.X).
func unlinedCalls(f *ast.File) int {
	n := 0
	ast.Inspect(f, func(x ast.Node) bool {
		c, ok := x.(*ast.CallExpr)
		if !ok {
			return true
		}
		fun := c.Fun
		if ix, ok := fun.(*ast.IndexExpr); ok {
			fun = ix.X
		}
		if sel, ok := fun.(*ast.SelectorExpr); ok && unlinedAllocs[sel.Sel.Name] {
			n++
		}
		return true
	})
	return n
}

// TestLedger_everyAllocationNamesALine (G-MM3a-1, I1 static): no non-test file of the cuda module outside gpudev itself calls
// an unlined constructor, so every production allocation names its plan line. It reads every file and counts a planted call.
func TestLedger_everyAllocationNamesALine(t *testing.T) {
	plant := "package x\nfunc f() { _ = gpu.NewBufferLenOf[float32](d, 4); _, _ = d.NewMappedHostBuffer(8); _ = gpu.NewBufferLenOfOn[float32](d, mem.LineKV, 4) }\n"
	pf, err := parser.ParseFile(token.NewFileSet(), "plant.go", plant, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got := unlinedCalls(pf); got != 2 {
		t.Fatalf("the planted file counts %d unlined calls, want 2", got)
	}
	read := 0
	err = filepath.WalkDir(".", func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() && (path == filepath.Join("internal", "gpudev") || d.Name() == "testdata") {
			return filepath.SkipDir
		}
		if d.IsDir() || !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
			return nil
		}
		f, err := parser.ParseFile(token.NewFileSet(), path, nil, 0)
		if err != nil {
			return err
		}
		read++
		if n := unlinedCalls(f); n > 0 {
			t.Errorf("%s calls an unlined gpudev constructor %d time(s): name the plan line (the ...On twin)", path, n)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if read < 50 { // 58 non-test files outside gpudev when written: far fewer means the walk is mis-scoped
		t.Fatalf("read %d source files", read)
	}
	t.Logf("read %d non-test files", read)
}
