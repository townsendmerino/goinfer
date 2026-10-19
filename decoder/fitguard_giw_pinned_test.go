package decoder

import (
	"strings"
	"testing"
)

// The .giw guard's pinned-expert term: a CUDA C' expert cache (--moe-cache-experts) holds every routed expert in one page-locked host copy, which
// is anonymous memory the guard did not price (tack's C3 run on Qwen3.6-35B-A3B, docs/measurements/c3-logs-reading-2026-10-18.md §1).

// m35 is Qwen3.6-35B-A3B's expert geometry on the qwen3_moe family (the family only supplies the routed-expert shape and CUDA residency; the
// hybrid layers do not change the expert bytes): 40 MoE layers x 256 experts x 3 projections of 512 x 2048.
func m35() *Config {
	cfg := a3Config("qwen3_moe")
	cfg.NumLayers, cfg.HiddenDim, cfg.NumExperts, cfg.MoeIntermediateSize, cfg.FirstKDenseReplace = 40, 2048, 256, 512, 0
	cfg.MaxPositions = 262144
	return cfg
}

// measuredM35Experts is the routed-expert footprint of that model in the memory campaign's census (docs/tasks/task-memory-campaign-2026-10.md,
// MEM0's table: "routed experts 18.12 GB", the same figure MEM10 quotes as page-locked), in bytes.
const measuredM35Experts = 18_120_000_000

func TestGiwPinnedExpertBytes_matchesTheMeasured35B(t *testing.T) {
	withFakeCUDACompiled(t)
	got := giwPinnedExpertBytes(m35(), Options{Backend: "cuda", MoECacheExperts: true})
	if lo, hi := int64(measuredM35Experts*99/100), int64(measuredM35Experts*101/100); got < lo || got > hi {
		t.Errorf("pinned expert bytes %d, want within 1%% of the measured %d", got, int64(measuredM35Experts))
	}
}

func TestGiwPinnedExpertBytes_onlyWhereTheCopyIsBuilt(t *testing.T) {
	withFakeCUDACompiled(t)
	cases := []struct {
		name string
		cfg  *Config
		opts Options
	}{
		{"no expert cache requested", m35(), Options{Backend: "cuda"}},
		{"the CPU backend builds no pinned copy", m35(), Options{Backend: "cpu", MoECacheExperts: true}},
		{"a dense model has no experts", sevenB(), Options{Backend: "cuda", MoECacheExperts: true}},
	}
	for _, c := range cases {
		if got := giwPinnedExpertBytes(c.cfg, c.opts); got != 0 {
			t.Errorf("%s: priced %d bytes, want 0", c.name, got)
		}
	}
	if got := giwPinnedExpertBytes(m35(), Options{Backend: "cuda", MoECacheExperts: true}); got == 0 {
		t.Errorf("VOID: the control (cuda, cache requested, MoE) priced nothing, so the cases above prove nothing")
	}
	// A dense prefix is not MoE: DeepSeek-style first_k_dense layers carry no routed experts.
	dense := m35()
	dense.FirstKDenseReplace = 10
	full := giwPinnedExpertBytes(m35(), Options{Backend: "cuda", MoECacheExperts: true})
	if got := giwPinnedExpertBytes(dense, Options{Backend: "cuda", MoECacheExperts: true}); got != full*30/40 {
		t.Errorf("10 dense-prefix layers: %d bytes, want 30/40 of %d", got, full)
	}
}

// With the 35B's numbers the load is priced at its pinned experts alone (the KV is VRAM on a CUDA resident). On the box C3 ran on, 38 GiB were
// available: the load is admitted, so this term would not have refused that load (the trip was the unwatched load, B33/B34). Below the pinned copy plus the 1 GiB margin it
// is refused, and no smaller context helps: the copy does not depend on the context.
func TestDecideGIWFit_pinnedExpertsOf35B(t *testing.T) {
	withFakeCUDACompiled(t)
	const GiB = int64(1) << 30
	cfg := m35()
	opts := Options{Backend: "cuda", MoECacheExperts: true}
	pinned := giwPinnedExpertBytes(cfg, opts)

	defer injectHostRAM(t, 38*GiB)()
	d := decideGIWFit(cfg, opts)
	if d.err != nil || d.need != pinned || d.pinnedExperts != pinned {
		t.Errorf("38 GiB available: err %v, need %d, pinnedExperts %d; want admitted at exactly the pinned copy %d", d.err, d.need, d.pinnedExperts, pinned)
	}

	for _, c := range []struct {
		avail  int64
		pinCtx int
		refuse bool
	}{
		{pinned + giwMemMargin + GiB/16, 0, false}, // just over
		{pinned + giwMemMargin - GiB/16, 0, true},  // just under
		{12 * GiB, 0, true},                        // an unpinned context: refused at the floor, because shrinking the context cannot shrink the copy
		{12 * GiB, 32768, true},                    // a pinned 32K context: refused
	} {
		injectHostRAM(t, c.avail)
		o := opts
		o.ResidentContext = c.pinCtx
		d := decideGIWFit(cfg, o)
		if (d.err != nil) != c.refuse {
			t.Errorf("%.2f GiB available, ctx pin %d: err %v, want refused=%v (pinned copy %.2f GiB)", float64(c.avail)/float64(GiB), c.pinCtx, d.err, c.refuse, float64(pinned)/float64(GiB))
			continue
		}
		if c.refuse && !strings.Contains(d.err.Error(), "pinned MoE expert copy") {
			t.Errorf("the refusal does not name the pinned expert copy: %v", d.err)
		}
		if !c.refuse && d.pinCtx != 0 {
			t.Errorf("a context pin of %d on a load that fits: the copy does not depend on the context", d.pinCtx)
		}
	}
}

// The plant: the same availability that refuses the 35B with the cache requested admits it without, so the refusal above is the pricing and nothing
// else (the load's other terms are zero on a CUDA resident). With the pricing removed from decideGIWFit, TestDecideGIWFit_pinnedExpertsOf35B goes red.
func TestDecideGIWFit_withoutTheExpertCacheTheSameMemoryAdmits(t *testing.T) {
	withFakeCUDACompiled(t)
	const GiB = int64(1) << 30
	defer injectHostRAM(t, 12*GiB)()
	cfg := m35()
	if d := decideGIWFit(cfg, Options{Backend: "cuda", MoECacheExperts: true}); d.err == nil {
		t.Fatalf("VOID: 12 GiB admitted the 35B with its expert cache, so this plant has nothing to compare")
	}
	if d := decideGIWFit(cfg, Options{Backend: "cuda"}); d.err != nil || d.need != 0 {
		t.Errorf("without --moe-cache-experts: err %v, need %d; want admitted with nothing priced", d.err, d.need)
	}
}
