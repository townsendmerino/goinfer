package decoder

import (
	"errors"
	"testing"
)

// withFakeMetalCompiled registers a stand-in "metal" backend for the test, so metalWillBeResident sees Metal as
// compiled into the binary (the real one lives in the metal module, which decoder's tests do not link).
func withFakeMetalCompiled(t *testing.T) {
	t.Helper()
	backendMu.Lock()
	prev, had := backendRegistry["metal"]
	backendRegistry["metal"] = func() (Backend, error) { return nil, errors.New("fake metal: not for building") }
	backendMu.Unlock()
	t.Cleanup(func() {
		backendMu.Lock()
		if had {
			backendRegistry["metal"] = prev
		} else {
			delete(backendRegistry, "metal")
		}
		backendMu.Unlock()
	})
}

func a3Config(modelType string) *Config {
	c := *representativeConfig(modelType)
	c.MaxPositions = 32768
	return &c
}

// TestKVPricingFor_pricesWhatTheBackendAllocates is A3 (docs/tasks/task-audit-followups-2026-10-06.md): a load that
// will be Metal-resident is priced at Metal's f16 KV for MetalCtxDefault positions (a caller's pin wins, the model's
// window caps), whatever -kv says; every other load keeps the CPU's per-request ceiling.
func TestKVPricingFor_pricesWhatTheBackendAllocates(t *testing.T) {
	withFakeMetalCompiled(t)
	q := a3Config("qwen2")
	cases := []struct {
		name   string
		cfg    *Config
		opts   Options
		ctx    int
		f16    bool
		metal  bool
		pinned bool
	}{
		{"metal, unpinned, -kv unset", q, Options{Backend: "metal"}, MetalCtxDefault, true, true, false},
		{"metal, -kv f32 still f16", q, Options{Backend: "metal", KVPrecision: "f32"}, MetalCtxDefault, true, true, false},
		{"metal, pinned 8192", q, Options{Backend: "metal", ResidentContext: 8192}, 8192, true, true, true},
		{"metal, pin past the window", q, Options{Backend: "metal", ResidentContext: 1 << 20}, 32768, true, true, true},
		{"cpu, unpinned", q, Options{Backend: "cpu"}, 32768, false, false, false},
		{"cpu, -kv f16", q, Options{Backend: "cpu", KVPrecision: "f16"}, 32768, true, false, false},
	}
	for _, c := range cases {
		p, ok := kvPricingFor(c.cfg, c.opts)
		if !ok || p.ctx != c.ctx || p.kvF16 != c.f16 || p.metalSizing != c.metal || p.pinned != c.pinned {
			t.Errorf("%s: got %+v ok=%v, want ctx %d f16 %v metal %v pinned %v", c.name, p, ok, c.ctx, c.f16, c.metal, c.pinned)
		}
	}
	// A window smaller than Metal's default caps it.
	small := a3Config("qwen2")
	small.MaxPositions = 1024
	if p, _ := kvPricingFor(small, Options{Backend: "metal"}); p.ctx != 1024 {
		t.Errorf("Metal pricing past a 1024-position window: ctx %d", p.ctx)
	}
	// An architecture outside Metal's feature gate falls back to the CPU, so it keeps the CPU's pricing.
	var declined string
	for _, mt := range []string{"nemotron_h", "granitemoehybrid", "llama4_text"} {
		arch, _, err := resolveArchitecture(a3Config(mt))
		if err == nil && len(missingFeatures(arch.residentFeatures(), residentBackendFeatures["metal"])) > 0 {
			declined = mt
			break
		}
	}
	if declined == "" {
		t.Fatal("no representative config outside Metal's feature gate left to test the fallback pricing with")
	}
	if p, _ := kvPricingFor(a3Config(declined), Options{Backend: "metal"}); p.metalSizing || p.ctx != 32768 {
		t.Errorf("%s (Metal declines it): priced %+v, want the CPU's ceiling", declined, p)
	}
}

// TestKVPricingFor_metalNotCompiledKeepsCPUPricing: -backend metal on a binary without Metal runs on the CPU.
func TestKVPricingFor_metalNotCompiledKeepsCPUPricing(t *testing.T) {
	backendMu.RLock()
	_, compiled := backendRegistry["metal"]
	backendMu.RUnlock()
	if compiled {
		t.Skip("a metal backend is registered in this test binary")
	}
	if p, _ := kvPricingFor(a3Config("qwen2"), Options{Backend: "metal"}); p.metalSizing || p.ctx != 32768 || p.kvF16 {
		t.Errorf("priced %+v without Metal compiled in, want the CPU's f32 ceiling", p)
	}
}

// TestGuardGIWFit_metalNotOverchargedByCPUPricing is A3's "do first" test, at the guard: with memory that cannot hold
// f32 KV over the whole 32k window but easily holds Metal's f16 at 4096, a Metal load is left alone while the same
// load on the CPU is pinned down. Before A3's fix both were priced like the CPU.
func TestGuardGIWFit_metalNotOverchargedByCPUPricing(t *testing.T) {
	withFakeMetalCompiled(t)
	cfg := a3Config("qwen2")
	// A consistent, realistic shape (a 7B-class KV: 128 KiB/position at f32), so Metal's feature gate resolves it.
	cfg.HiddenDim, cfg.NumLayers, cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim, cfg.IntermediateDim = 4096, 32, 32, 8, 128, 11008
	if !metalWillBeResident(cfg, Options{Backend: "metal"}) {
		t.Fatal("test setup: Metal would not admit this config, so the test would only exercise the CPU pricing")
	}
	metalNeed := estimateKVBytes(cfg, MetalCtxDefault, true, false) + prefillAttnScratchBudget
	cpuNeed := estimateKVBytes(cfg, cfg.MaxPositions, false, false) + prefillAttnScratchBudget
	avail := 2*metalNeed + giwMemMargin // twice what Metal holds; a small fraction of the CPU's ceiling
	if cpuNeed <= 2*metalNeed {
		t.Fatalf("test setup: CPU need %d is not above the budget %d", cpuNeed, 2*metalNeed)
	}
	defer injectHostRAM(t, avail)()
	if ctx, err := guardGIWFit(cfg, Options{Backend: "metal"}); err != nil || ctx != 0 {
		t.Errorf("Metal load: pinned %d, err %v; want left alone (it needs %d of a %d budget)", ctx, err, metalNeed, 2*metalNeed)
	}
	if ctx, err := guardGIWFit(cfg, Options{Backend: "cpu"}); err != nil || ctx == 0 || ctx >= cfg.MaxPositions {
		t.Errorf("CPU load: pinned %d, err %v; want a pin below the %d window", ctx, err, cfg.MaxPositions)
	}
	// The .gguf/safetensors guard shares the pricing.
	f := fitCheck{}.priceCtxAndKV(cfg, Options{Backend: "metal"})
	if f.effCtx != MetalCtxDefault || !f.kvF16 || f.kvBytes != estimateKVBytes(cfg, MetalCtxDefault, true, false) {
		t.Errorf("priceCtxAndKV for Metal: ctx %d f16 %v kv %d", f.effCtx, f.kvF16, f.kvBytes)
	}
}
