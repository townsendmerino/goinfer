//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
)

// The build-scratch / margin accounting gates, registered
// before the code. The finding: the 384 MiB margin is not what falls short. The driver rounds every buffer of 2 MiB or more up to a 2 MiB multiple, Plan prices the
// requested bytes, and the plan therefore asks checkKVFits for a context that has already spent the rounding. The build prices it (packedAllocSlack) into the plan.

var accountingModels = []string{
	"qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", "qwen2.5-7b-instruct-q4_k_m.gguf",
	"gemma-3-4b-it", "qwen25vl-3b-instruct", "gemma-4-e2b-gguf/gemma-4-E2B_q4_0-it.gguf", "qwen3-8b-gguf/Qwen3-8B-Q4_K_M.gguf",
}

// buildRow is one measured build: what the device gave up before the first checkKVFits probe against what Plan and the slack priced, the plan's context against
// the one checkKVFits left standing, and what was allocated after the probe (beyond the KV) and on first use.
type buildRow struct {
	name                                         string
	preKV, plan, slack, kv, postKV, lazy, margin int64
	slots, ctxPlanned, ctxFinal                  int
}

func (b buildRow) residual() int64 { return b.preKV - b.plan - b.slack }

func (b buildRow) log(t *testing.T, tag string) {
	t.Logf("[margin accounting%s] %-42s device before KV %5.0f MiB | Plan %5.0f + slack %4.0f -> residual %4.0f MiB | ctx planned %5d final %5d (%d slots, KV %5.0f MiB) | post-KV scratch %4.0f, first-use %4.0f MiB, against the %0.f MiB margin",
		tag, b.name, mb(b.preKV), mb(b.plan), mb(b.slack), mb(b.residual()), b.ctxPlanned, b.ctxFinal, b.slots, mb(b.kv), mb(b.postKV), mb(b.lazy), mb(b.margin))
}

// measureBuild loads rel on the card at the defaults (int4, unpinned context), reading the free VRAM around the build and one 520-token prefill + decode. arenaOff
// uploads the weights without their arena while the plan prices it (the planted defect). ok=false when the model is absent or did not build a CUDA resident.
func measureBuild(t *testing.T, rel string, arenaOff bool) (row buildRow, ok bool) {
	t.Helper()
	return measureBuildWith(t, rel, decoder.Options{Backend: "cuda", Quant: "int4"}, arenaOff)
}

// measureBuildWith is measureBuild loading with opts.
func measureBuildWith(t *testing.T, rel string, opts decoder.Options, arenaOff bool) (row buildRow, ok bool) {
	t.Helper()
	home, _ := os.UserHomeDir()
	p := filepath.Join(home, "models", rel)
	if _, err := os.Stat(p); err != nil {
		t.Logf("no model at %s", p)
		return row, false
	}
	dev, err := gc.GetDevice(0)
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	probe, err := dev.Primary()
	if err != nil {
		t.Skipf("primary ctx: %v", err)
	}
	realFree := func() int64 {
		f, _, e := probe.MemInfo()
		if e != nil {
			t.Fatalf("MemInfo: %v", e)
		}
		return int64(f)
	}
	orig, origOff := cudaFreeVRAM, weightArenaOffForTest
	defer func() { cudaFreeVRAM, weightArenaOffForTest = orig, origOff }()
	weightArenaOffForTest = arenaOff
	var atKV int64
	calls := 0
	free0 := realFree()
	cudaFreeVRAM = func(r *cudaResident) (uint64, error) {
		f, _, e := r.dev.Context().MemInfo()
		calls++
		if calls == 1 { // the build's own probe; a later call is after the KV exists
			atKV = int64(f)
		}
		return f, e
	}
	m, err := decoder.Load(p, opts)
	if err != nil {
		t.Fatalf("load %s: %v", rel, err)
	}
	defer m.Close()
	r, isRes := m.ResidentForwardForTest().(*cudaResident)
	if !isRes || atKV == 0 {
		t.Logf("%s: not CUDA-resident: %s", rel, m.ResidentDecline())
		return row, false
	}
	freeBuilt := realFree()
	// The KV the build charged, from the device's ledger (a multi-slot pool holds fewer blocks than slots x the per-slot cap, which
	// kvChargedBytes would price); kvChargedBytes where the device has no ledger, exact for one slot.
	kv := kvChargedBytes(r.ctxCap, r.kvSlotsN, r.layers)
	if l, isLedger := r.dev.Leaser().(*deviceLedger); isLedger {
		kv = l.Usage().ByLine[mem.LineKV]
	}
	ids := make([]int, 520)
	for i := range ids {
		ids[i] = 100 + i%900
	}
	ch, g := m.Generate(context.Background(), ids, 8, decoder.SamplingParams{})
	for range ch {
	}
	if err := g.Err(); err != nil {
		t.Fatalf("generate %s: %v", rel, err)
	}
	freeAfter := realFree()
	row = buildRow{
		name: filepath.Base(rel), preKV: free0 - atKV, plan: m.ResidentDenseWeightBytesFor("cuda"), slack: r.allocSlackBytes,
		kv: kv, slots: r.kvSlotsN, ctxPlanned: r.ctxPlanned, ctxFinal: r.ctxCap,
		lazy: freeBuilt - freeAfter, margin: r.ctxMargin(),
	}
	row.postKV = atKV - row.kv - freeBuilt
	return row, true
}

// TestBuildScratchAccounting is G-M1 (and the record G-M2 reads): per bench model, the residual between what the device gave up before the KV and what Plan plus
// the slack priced must be small, and what the build allocates after the probe must fit the margin. Heavy.
func TestBuildScratchAccounting(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy: set GOINFER_HEAVY_TESTS=1")
	}
	requireCUDADevice(t)
	// The first load in a process also pays the context and module cost that a serve process has paid before it plans: a warm-up, not a row.
	measureBuild(t, accountingModels[0], false)
	// The lower bound is -16: the device figure reads 14-18 MiB lower in a process that has already built models than in a cold one (as in
	// TestResidentDenseBytes), and a negative residual is the conservative direction (Plan priced more than the device gave up).
	bands := map[string][2]int64{ // residual in MiB: [-16, +64] on the Qwen text models, [-8, +128] on the others
		"qwen2.5-coder-0.5b-instruct-q4_k_m.gguf": {-16, 64}, "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf": {-16, 64}, "qwen2.5-7b-instruct-q4_k_m.gguf": {-16, 64},
		"qwen3-8b-gguf/Qwen3-8B-Q4_K_M.gguf": {-16, 64},
	}
	for _, rel := range accountingModels {
		t.Run(filepath.Base(rel), func(t *testing.T) {
			row, ok := measureBuild(t, rel, false)
			if !ok {
				t.Skip("model absent or not CUDA-resident")
			}
			row.log(t, "")
			lo, hi := int64(-8), int64(128)
			if b, has := bands[rel]; has {
				lo, hi = b[0], b[1]
			}
			if res := row.residual() >> 20; res < lo || res > hi {
				t.Errorf("G-M1: residual %d MiB outside [%d, %d]: the build puts %d MiB on the device before the KV, Plan %d + slack %d: a cost the rounding rule does not know",
					res, lo, hi, row.preKV>>20, row.plan>>20, row.slack>>20)
			}
			if row.postKV+row.lazy > row.margin {
				t.Errorf("post-KV scratch %d + first-use %d MiB exceeds the %d MiB margin", row.postKV>>20, row.lazy>>20, row.margin>>20)
			}
		})
	}
}

// TestBuildScratchAccounting_plantedDefect uploads the weights without their arena while the plan prices it (weightArenaOffForTest): a build
// and a plan that disagree about the weights' rounding, which G-M1 must catch.
//   - every model: the residual is larger with the defect than without it;
//   - the 7B: without the defect the residual is in the Qwen band (under 64 MiB), with it the residual is past the band, so G-M1 goes red;
//   - Gemma 3 4B, whose plan is limited by free VRAM at one slot: the build trims the planned context with the defect, not without it.
//
// Heavy.
func TestBuildScratchAccounting_plantedDefect(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy: set GOINFER_HEAVY_TESTS=1")
	}
	requireCUDADevice(t)
	measureBuild(t, accountingModels[0], false)
	for _, rel := range accountingModels[1:] {
		clean, ok := measureBuild(t, rel, false)
		if !ok {
			continue
		}
		planted, _ := measureBuild(t, rel, true)
		clean.log(t, " clean  ")
		planted.log(t, " planted")
		if planted.residual() <= clean.residual() {
			t.Errorf("%s: residual %d MiB with the weights outside their arena, %d MiB with them in it: the plant changed nothing", clean.name, planted.residual()>>20, clean.residual()>>20)
		}
		switch rel {
		case "qwen2.5-7b-instruct-q4_k_m.gguf":
			if clean.residual() >= 64<<20 || planted.residual() <= 64<<20 {
				t.Errorf("7B: residual %d MiB clean (want < 64) and %d MiB planted (want past G-M1's 64)", clean.residual()>>20, planted.residual()>>20)
			}
		case "gemma-3-4b-it":
			if planted.ctxPlanned-planted.ctxFinal <= clean.ctxPlanned-clean.ctxFinal {
				t.Errorf("Gemma 3 4B: the build trims %d positions planted and %d clean: the disagreement did not reach the context",
					planted.ctxPlanned-planted.ctxFinal, clean.ctxPlanned-clean.ctxFinal)
			}
		}
	}
}

// TestWeightArena_holdsEveryWeight: on the card, each family's build sub-allocates its dense weights from one arena that the plan's
// walk sized, and the upload fills it exactly: no buffer the walk missed (missed), no byte it counted that nothing uploaded (used <
// size). A miss would put a weight outside the slack the plan priced; a gap would hold device memory nothing reads.
func TestWeightArena_holdsEveryWeight(t *testing.T) {
	requireCUDADevice(t)
	fixtures := []string{
		"llama-tiny", "gemma2-tiny", "qwen35-tiny", "qwen3next-tiny", "deepseek-tiny", "nemotron-tiny", "qwen3moe-tiny",
		"gemma4-moe-kv-tiny32", "cohere-tiny", "mixtral-tiny", "olmo_hybrid-tiny", "granite-dense-tiny", "glm-tiny-bias", "qwen25vl-tiny",
	}
	ran := 0
	for _, f := range fixtures {
		t.Run(f, func(t *testing.T) {
			m, err := decoder.Load(filepath.Join("../../testdata", f), decoder.Options{Backend: "cuda", Quant: "int4"})
			if err != nil {
				t.Skipf("load: %v", err)
			}
			defer m.Close()
			r, ok := m.ResidentForwardForTest().(*cudaResident)
			if !ok {
				t.Skipf("not CUDA-resident: %s", m.ResidentDecline())
			}
			ran++
			s := r.wArenaStats
			t.Logf("arena %d bytes, %d used, %d missed; plan slack %d", s.size, s.used, s.missed, r.allocSlackBytes)
			if s.size == 0 || s.missed != 0 || s.used != s.size {
				t.Errorf("the arena holds %d of %d bytes and %d bytes of dense weights went outside it: the walk and the upload disagree",
					s.used, s.size, s.missed)
			}
		})
	}
	if ran == 0 {
		t.Skip("no fixture built a CUDA resident here")
	}
}

// TestBuildScratchAccounting_serveShape is G-M1's cell at the options serve loads with (serveOptions: the int4 head, windowed KV, four KV
// slots), on the 7B: the residual is in the Qwen text models' band and the post-KV scratch fits the margin, as at the gates' own options.
// Heavy.
func TestBuildScratchAccounting_serveShape(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy: set GOINFER_HEAVY_TESTS=1")
	}
	requireCUDADevice(t)
	measureBuild(t, accountingModels[0], false) // the process's first build pays the context and modules: a warm-up, not a row
	flags := []string{"-backend", "cuda"}
	opts := serveOptions(t, flags...)
	requireServeShaped(t, t, opts, flags...)
	if !opts.ResidentWindowedKV || opts.ResidentKVSlots != serveKVSessions {
		t.Fatalf("serve's options are not the shape this cell reads: windowed KV %v, %d KV slots", opts.ResidentWindowedKV, opts.ResidentKVSlots)
	}
	row, ok := measureBuildWith(t, "qwen2.5-7b-instruct-q4_k_m.gguf", opts, false)
	if !ok {
		t.Skip("model absent or not CUDA-resident")
	}
	row.log(t, " serve")
	if res := row.residual() >> 20; res < -16 || res > 64 {
		t.Errorf("serve shape: residual %d MiB outside [-16, 64]: the build puts %d MiB on the device before the KV, Plan %d + slack %d",
			res, row.preKV>>20, row.plan>>20, row.slack>>20)
	}
	if row.postKV+row.lazy > row.margin {
		t.Errorf("serve shape: post-KV scratch %d + first-use %d MiB exceeds the %d MiB margin", row.postKV>>20, row.lazy>>20, row.margin>>20)
	}
}

// TestConcurrentMinFree_7BServe is a read, not a gate (E5_MINFREE=1 and GOINFER_HEAVY_TESTS=1): the 7B loaded as serve loads it, four
// generations at once, three decoding when the fourth arrives with a prompt that takes 512-row prefill passes. A sampler reads the card's
// free VRAM throughout and prints the lowest reading against the driver's allocation floor (allocFloorBytes) and the margin.
func TestConcurrentMinFree_7BServe(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" || os.Getenv("E5_MINFREE") == "" {
		t.Skip("a read: set GOINFER_HEAVY_TESTS=1 and E5_MINFREE=1")
	}
	requireCUDADevice(t)
	home, _ := os.UserHomeDir()
	p := filepath.Join(home, "models", "qwen2.5-7b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("no model at %s", p)
	}
	dev, err := gc.GetDevice(0)
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	probe, err := dev.Primary()
	if err != nil {
		t.Skipf("primary ctx: %v", err)
	}
	free := func() int64 {
		f, _, e := probe.MemInfo()
		if e != nil {
			t.Errorf("MemInfo: %v", e)
		}
		return int64(f)
	}
	free0 := free()
	m, err := loadLikeServe(t, p, "-backend", "cuda")
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	r, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatalf("not CUDA-resident: %s", m.ResidentDecline())
	}
	n := m.EnableResidentConcurrency(4)
	freeLoaded := free()
	fmt.Fprintf(os.Stderr, "[E5-minfree] loaded: %d KV slots at %d positions, concurrency %d; free %.1f MiB before the load, %.1f after; margin %.1f MiB\n",
		r.KVSlots(), r.ctxCap, n, mb(free0), mb(freeLoaded), mb(r.ctxMargin()))
	if n != 4 {
		t.Fatalf("EnableResidentConcurrency(4) = %d: four generations would not run at once", n)
	}

	var (
		mu      sync.Mutex
		minFree = freeLoaded
		samples int
	)
	stop := make(chan struct{})
	sampled := make(chan struct{})
	go func() {
		defer close(sampled)
		tick := time.NewTicker(2 * time.Millisecond)
		defer tick.Stop()
		for {
			select {
			case <-stop:
				return
			case <-tick.C:
				f := free()
				mu.Lock()
				minFree, samples = min(minFree, f), samples+1
				mu.Unlock()
			}
		}
	}()
	prompt := func(n, seed int) []int {
		ids := make([]int, n)
		for i := range ids {
			ids[i] = 100 + (i*7+seed*131)%5000
		}
		return ids
	}
	var wg sync.WaitGroup
	gen := func(name string, ids []int, maxNew int, started chan<- struct{}) {
		defer wg.Done()
		ch, g := m.Generate(context.Background(), ids, maxNew, decoder.SamplingParams{})
		first := true
		for range ch {
			if first && started != nil {
				started <- struct{}{}
			}
			first = false
		}
		if err := g.Err(); err != nil {
			t.Errorf("%s: %v", name, err)
		}
	}
	t0 := time.Now()
	started := make(chan struct{}, 3)
	for i := range 3 {
		wg.Add(1)
		go gen(fmt.Sprintf("decoder %d", i), prompt(64, i), 512, started)
	}
	for range 3 {
		<-started // all three are decoding
	}
	wg.Add(1)
	go gen("the long prompt", prompt(2048, 9), 32, nil)
	wg.Wait()
	close(stop)
	<-sampled
	mu.Lock()
	defer mu.Unlock()
	fmt.Fprintf(os.Stderr, "[E5-minfree] four generations in %v, %d samples at 2 ms: lowest free %.1f MiB (%.1f below the loaded figure), %.1f above the %d MiB allocation floor\n",
		time.Since(t0).Round(time.Millisecond), samples, mb(minFree), mb(freeLoaded-minFree), mb(minFree-allocFloorBytes), allocFloorBytes>>20)
}
