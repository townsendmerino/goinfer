package decoder

import (
	"errors"
	"strings"
	"testing"
)

// The host fit guard's CUDA case (docs/measurements/cuda-stream-pack-2026-10-18/README.md, G-F1): a load that will be CUDA-resident has its KV in VRAM, so
// it adds none to the host need, and its build adds one packed layer beside the canonical weights.

// withFakeCUDACompiled registers a stand-in "cuda" backend, so cudaWillBeResident sees CUDA as compiled into the binary (the real one lives in the cuda
// module, which decoder's tests do not link).
func withFakeCUDACompiled(t *testing.T) {
	t.Helper()
	backendMu.Lock()
	prev, had := backendRegistry["cuda"]
	backendRegistry["cuda"] = func() (Backend, error) { return nil, errors.New("fake cuda: not for building") }
	backendMu.Unlock()
	t.Cleanup(func() {
		backendMu.Lock()
		if had {
			backendRegistry["cuda"] = prev
		} else {
			delete(backendRegistry, "cuda")
		}
		backendMu.Unlock()
	})
}

// sevenB is a 7B-class dense shape (Qwen2.5-7B: 28 layers, 4 KV heads of 128): f32 KV is 3.5 GiB at 32,768 positions.
func sevenB() *Config {
	cfg := a3Config("qwen2")
	cfg.HiddenDim, cfg.NumLayers, cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim, cfg.IntermediateDim = 3584, 28, 28, 4, 128, 18944
	cfg.MaxPositions = 32768
	return cfg
}

func TestKVPricingFor_cudaResidentKVIsDevice(t *testing.T) {
	withFakeCUDACompiled(t)
	q := sevenB()
	for _, c := range []struct {
		name   string
		opts   Options
		ctx    int
		device bool
	}{
		{"cuda, unpinned", Options{Backend: "cuda"}, 32768, true},
		{"cuda, pinned 8192", Options{Backend: "cuda", ResidentContext: 8192}, 8192, true},
		{"cuda, weights streamed", Options{Backend: "cuda", StreamWeights: true}, 32768, false},
		{"cpu", Options{Backend: "cpu"}, 32768, false},
	} {
		p, ok := kvPricingFor(q, c.opts)
		if !ok || p.ctx != c.ctx || p.deviceKV != c.device || p.metalSizing {
			t.Errorf("%s: %+v ok=%v, want ctx %d deviceKV %v", c.name, p, ok, c.ctx, c.device)
		}
	}
	// An architecture outside CUDA's feature gate falls back to the CPU, so it keeps the CPU's host pricing.
	var declined string
	for _, mt := range []string{"bailing_hybrid", "laguna", "lfm2", "gemma", "olmoe"} {
		arch, _, err := resolveArchitecture(a3Config(mt))
		if err == nil && len(missingFeatures(arch.residentFeatures(), residentBackendFeatures["cuda"])) > 0 {
			declined = mt
			break
		}
	}
	if declined == "" {
		t.Fatal("no representative config outside CUDA's feature gate left to test the fallback pricing with")
	}
	if p, _ := kvPricingFor(a3Config(declined), Options{Backend: "cuda"}); p.deviceKV {
		t.Errorf("%s (CUDA declines it): priced as device KV", declined)
	}
}

// TestKVPricingFor_cudaNotCompiledKeepsCPUPricing: -backend cuda on a binary without CUDA runs on the CPU.
func TestKVPricingFor_cudaNotCompiledKeepsCPUPricing(t *testing.T) {
	backendMu.RLock()
	_, compiled := backendRegistry["cuda"]
	backendMu.RUnlock()
	if compiled {
		t.Skip("a cuda backend is registered in this test binary")
	}
	if p, _ := kvPricingFor(sevenB(), Options{Backend: "cuda"}); p.deviceKV {
		t.Errorf("priced %+v as device KV without CUDA compiled in", p)
	}
}

// G-F1: the 7B at 32,768 positions is admitted by the CUDA guard where the CPU's pricing (and the tree before this change) refused it, and the CUDA host
// term is the weights plus the source file on the direct path, the weights plus one packed layer where there is no source.
func TestFitCheck_cudaResident7BAt32KIsAdmitted(t *testing.T) {
	withFakeCUDACompiled(t)
	const GiB = int64(1) << 30
	cfg := sevenB()
	base := fitCheck{name: "qwen2.5-7b.gguf", quant: "int4", weightBytes: 4 * GiB, srcFileBytes: 4*GiB + GiB*4/10, availBytes: 14 * GiB}
	cuda := base.priceCtxAndKV(cfg, Options{Backend: "cuda", ResidentContext: 32768})
	cpu := base.priceCtxAndKV(cfg, Options{Backend: "cpu", ResidentContext: 32768})
	if cpu.kvBytes < 3*GiB {
		t.Fatalf("test setup: the CPU prices %d bytes of KV at 32K, expected about 3.5 GiB", cpu.kvBytes)
	}
	if cuda.kvBytes != 0 || !cuda.deviceKV {
		t.Errorf("CUDA-resident KV priced as host: kvBytes %d deviceKV %v", cuda.kvBytes, cuda.deviceKV)
	}
	if !cuda.fits() || cpu.fits() {
		t.Errorf("at 14 GiB available: CUDA fits=%v (need %d, budget %d), CPU fits=%v (need %d); want CUDA admitted and the CPU pricing refused", cuda.fits(), cuda.need(), cuda.budget(), cpu.fits(), cpu.need())
	}
	if want := base.weightBytes + base.srcFileBytes; cuda.need() != want {
		t.Errorf("direct-path need %d, want weights + source file %d", cuda.need(), want)
	}
	if !strings.Contains(cuda.arithmetic(), "KV is in VRAM") {
		t.Errorf("the arithmetic does not say where the KV is: %q", cuda.arithmetic())
	}
	// No source term (a safetensors directory): the packed layer is what the build adds beside the weights.
	nosrc := fitCheck{weightBytes: 4 * GiB, availBytes: 14 * GiB}.priceCtxAndKV(cfg, Options{Backend: "cuda"})
	if want := 4*GiB + 4*GiB/28 + cudaPackStageBytes; nosrc.need() != want {
		t.Errorf("no-source need %d, want weights + one layer + staging %d", nosrc.need(), want)
	}
	// The C' expert-cache term is untouched and stays the larger.
	moe := fitCheck{weightBytes: 4 * GiB, srcFileBytes: 4 * GiB, cudaBuildBytes: 20 * GiB, expertBytes: 12 * GiB, availBytes: 64 * GiB}.priceCtxAndKV(cfg, Options{Backend: "cuda"})
	if moe.need() != 20*GiB {
		t.Errorf("C' build need %d, want the unchanged %d", moe.need(), 20*GiB)
	}
}

// The .giw guard prices KV and prefill scratch against available memory; for a CUDA-resident load neither is host memory, so a small host leaves it alone.
func TestGuardGIWFit_cudaResidentNeedsNoHostKV(t *testing.T) {
	withFakeCUDACompiled(t)
	cfg := sevenB()
	defer injectHostRAM(t, 2*giwMemMargin)() // 2 GiB: nowhere near the 3.5 GiB of f32 KV the CPU pricing asks
	if ctx, err := guardGIWFit(cfg, Options{Backend: "cuda"}); err != nil || ctx != 0 {
		t.Errorf("CUDA-resident load: pinned %d, err %v; want left alone", ctx, err)
	}
	if ctx, err := guardGIWFit(cfg, Options{Backend: "cpu"}); err == nil && ctx == 0 {
		t.Errorf("CPU load at the same memory was left alone: the test would prove nothing")
	}
}
