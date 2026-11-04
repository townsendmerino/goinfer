package mem

import (
	"fmt"
	"sync"
	"testing"
)

// TestResolveKV_eachBranch holds ResolveKV's resolution order: one row per branch, the reason included.
func TestResolveKV_eachBranch(t *testing.T) {
	declines := KVPolicy{Decline: func() string { return "not this family" }, Auto: func() bool { return true }}
	clears := KVPolicy{Decline: func() string { return "" }, Auto: func() bool { return true }}
	autoOff := KVPolicy{Decline: func() string { return "" }, Auto: func() bool { return false }}
	autoGate := KVPolicy{Decline: func() string { return "" }, Auto: func() bool { return true },
		AutoDecline: func() string { return "gate not passed" }}
	for _, c := range []struct {
		name     string
		q        KVQuery
		got, why string
	}{
		{"no KV at all", KVQuery{Backend: "cuda", HasArch: true, NoKV: true}, KVNone, "this model keeps no attention KV"},
		{"cpu batched decode", KVQuery{Backend: "cpu", CPURequest: KVF16, CPUBatchOn: true}, "", "CPU batched decode (CPUBatchDecode on) reads f32 KV only until phase 4"},
		{"cpu f16 allowed", KVQuery{Backend: "", CPURequest: KVF16, HasArch: true, CPUF16OK: true}, KVF16, ""},
		{"cpu f16 on MoE", KVQuery{Backend: "cpu", CPURequest: KVF16, HasArch: true, MoE: true}, KVF32, "MoE families keep an f32 CPU cache until phase 4"},
		{"cpu i8 own loop", KVQuery{Backend: "cpu", CPURequest: KVI8, HasArch: true, OwnLoop: true}, KVF32, "this family's own forward loop keeps an f32 CPU cache until phase 4"},
		{"cpu i8 no arch", KVQuery{Backend: "cpu", CPURequest: KVI8}, KVF32, "no architecture"},
		{"cpu auto", KVQuery{Backend: "cpu", CPURequest: KVAuto}, KVF32, "auto is f32 on the CPU"},
		{"metal MLA", KVQuery{Backend: "metal", Request: KVF16, HasArch: true, MLA: true}, KVF32, "an MLA model keeps its latent cache in f32 on Metal"},
		{"metal auto", KVQuery{Backend: "metal", Request: KVAuto, HasArch: true}, KVF16, "auto is f16 on Metal"},
		{"metal i8", KVQuery{Backend: "metal", Request: KVI8, HasArch: true}, KVI8, ""},
		{"cuda no arch", KVQuery{Backend: "cuda", Request: KVF16}, KVF32, "no architecture"},
		{"cuda i8", KVQuery{Backend: "cuda", Request: KVI8, HasArch: true, CUDAPolicy: clears}, KVF32, "CUDA has no int8 KV cache until phase 2"},
		{"cuda f16 drafter", KVQuery{Backend: "cuda", Request: KVF16, HasArch: true, Drafter: true, CUDAPolicy: clears}, KVF32, "the drafter needs f32 KV until phase 6"},
		{"cuda f16 declined", KVQuery{Backend: "cuda", Request: KVF16, HasArch: true, CUDAPolicy: declines}, KVF32, "not this family"},
		{"cuda f16", KVQuery{Backend: "cuda", Request: KVF16, HasArch: true, CUDAPolicy: clears}, KVF16, ""},
		{"cuda auto off", KVQuery{Backend: "cuda", Request: KVAuto, HasArch: true, CUDAPolicy: autoOff}, KVF32, "auto is f32 on CUDA in this build"},
		{"cuda auto gate", KVQuery{Backend: "cuda", Request: KVAuto, HasArch: true, CUDAPolicy: autoGate}, KVF32, "gate not passed"},
		{"cuda auto f16", KVQuery{Backend: "cuda", Request: KVAuto, HasArch: true, CUDAPolicy: clears}, KVF16, "auto is f16 on CUDA for this family"},
		{"cuda no policy f16", KVQuery{Backend: "cuda", Request: KVF16, HasArch: true}, KVF16, ""},
		{"cuda no policy auto", KVQuery{Backend: "cuda", Request: KVAuto, HasArch: true}, KVF32, "auto is f32 on CUDA in this build"},
		{"webgpu asked", KVQuery{Backend: "webgpu", Request: KVI8}, KVI8, ""},
		{"webgpu auto", KVQuery{Backend: "webgpu", Request: KVAuto}, KVF32, "auto is f32 here"},
	} {
		got, why := ResolveKV(c.q)
		if got != c.got || why != c.why {
			t.Errorf("%s: (%q, %q), want (%q, %q)", c.name, got, why, c.got, c.why)
		}
	}
}

// TestResolveKV_pure: the same query gives the same answer, in one goroutine and in 16 at once.
func TestResolveKV_pure(t *testing.T) {
	all := func() string {
		var out string
		for _, be := range []string{"", "cpu", "cuda", "metal", "webgpu"} {
			for _, req := range []string{KVAuto, KVF32, KVF16, KVI8} {
				for _, arch := range []bool{false, true} {
					q := KVQuery{Backend: be, Request: req, CPURequest: req, HasArch: arch, CPUF16OK: arch,
						CUDAPolicy: KVPolicy{Decline: func() string { return "" }, Auto: func() bool { return true }}}
					got, why := ResolveKV(q)
					out += fmt.Sprintf("%s %s %v: %s %s\n", be, req, arch, got, why)
				}
			}
		}
		return out
	}
	want := all()
	var wg sync.WaitGroup
	for range 16 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if all() != want {
				t.Error("a concurrent run answered differently")
			}
		}()
	}
	wg.Wait()
}
