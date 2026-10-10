//go:build cuda && goinfer_testhooks

package cuda

import (
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGptOssResidentParityCUDA is G7's gate: ONE real gpt-oss forward on a resident path. The
// FeatAttnSink declaration waits on this gate passing, not the other way round
// (docs/queue-correctness.md G7).
//
// IT FITS AN 8 GB CARD VIA MACHINERY THAT ALREADY EXISTED: --moe-cache-experts holds the experts in
// pinned host memory and DMAs the routed ones into device slots per token, the path that already
// carries Qwen3.6-35B-A3B on this card (TestQwen36_35B_cache). TestGptOssExpertCacheAB is the
// discriminating A/B for gpt-oss's use of it; no cosine bar would catch its failure.
//
// The CPU arm is the reference. Both models load at once, which needs a large-RAM host (the 16 GB
// Mac cannot).
func TestGptOssResidentParityCUDA(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1 (loads a 12 GB model)")
	}
	// Skips until CUDA declares the two features, exactly as metal/gptoss_real_test.go does. The
	// declaration is NOT made: this gate FAILED (min cosine 0.681) with them declared locally
	// (docs/queue-correctness.md G7), so declaring on kernel-level evidence would repeat the mistake that
	// was reverted. The test is committed so the next attempt starts from a reproduction rather than a
	// rebuild.
	if !decoder.ResidentBackendFeatures("cuda")[decoder.FeatAttnSink] {
		t.Skip("cuda does not declare FeatAttnSink/FeatOutBias — the gate below is what must pass first")
	}
	// decoder.AssetPathForTest, NOT modelPath: modelPath reads GOINFER_MODELS_DIR, a different variable
	// from the registry's real GOINFER_GPTOSS_GGUF override, which it would bypass entirely while still
	// satisfying TestAssetRegistry_noDirectReads (a source-text regex over os.Getenv(...) of that name).
	// AssetPathForTest resolves the SAME registry entry decoder's own TestGptOssSafetensors_vsGGUF uses,
	// and skips with the reason when absent.
	path := decoder.AssetPathForTest(t, "GOINFER_GPTOSS_GGUF")
	const steps = 8
	seed := []int{3, 14, 7, 42, 1, 99, 5, 60}

	t0 := time.Now()
	mg, err := decoder.Load(path, decoder.Options{
		Backend: "cuda", Quant: "int4", MoECacheExperts: true,
	})
	if err != nil {
		t.Fatalf("load (cuda): %v", err)
	}
	defer mg.Close()
	rf := mg.ResidentForwardForTest()
	if rf == nil { // a silent CPU fallback would pass every assertion below trivially
		t.Fatalf("gpt-oss DECLINED the resident path (%s) — with FeatAttnSink+FeatOutBias declared "+
			"and expert streaming on, it is supposed to be admitted", mg.ResidentDecline())
	}
	fmt.Fprintf(os.Stderr, "[gptoss-cuda] resident built in %s\n", time.Since(t0).Round(time.Second))

	mcpu, err := decoder.Load(path, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cpu): %v", err)
	}
	defer mcpu.Close()
	_, nL, _, nKV, hd, _, _ := mcpu.Dims()
	cache := decoder.NewKVCache(nL, nKV, hd, 0, 1024, nil)

	var minCos float64 = 1
	exact := 0
	for i := range steps {
		tok := seed[i]
		cpuL, err := mcpu.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatalf("cpu forward %d: %v", i, err)
		}
		gpuL, err := rf.Forward(mg.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("gpu forward %d: %v", i, err)
		}
		cos, _ := cosF32(cpuL, gpuL)
		if cos < minCos {
			minCos = cos
		}
		if argmaxF(cpuL) == argmaxF(gpuL) {
			exact++
		}
		fmt.Fprintf(os.Stderr, "[gptoss-cuda] step %d/%d cos=%.6f elapsed=%s\n",
			i+1, steps, cos, time.Since(t0).Round(time.Second))
	}
	t.Logf("gpt-oss-20b REAL resident parity (cuda, --moe-cache-experts): %d/%d argmax-exact, min cosine %.6f",
		exact, steps, minCos)
	// The int4-noise floor the other resident gates use. Anything below is gross breakage, not
	// quantization: a dropped sink, a wrong bias row, or a clamp on the wrong branch.
	if minCos < 0.95 {
		t.Errorf("min logit cosine %.6f < 0.95 — gross breakage, not int4 noise", minCos)
	}
	if exact*2 < steps {
		t.Errorf("argmax parity %d/%d < 50%%", exact, steps)
	}
}
