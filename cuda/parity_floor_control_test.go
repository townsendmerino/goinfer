//go:build cuda && goinfer_testhooks

package cuda

import (
	"fmt"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestParityFloorControl calibrates the INSTRUMENT before its reading is called a defect.
// The gpt-oss 20B gate scores 0.895 CPU-int4 vs CUDA-int4-resident against a 0.95 bar that was
// imported from TINY-fixture gates. Nothing in this tree establishes what that comparison
// normally scores on a REAL multi-billion-parameter checkpoint: the 35B streaming test asserts
// no cosine at all, and the Mellum gate's 0.9946 is a 4-layer DENSE slice.
//
// So: same harness, same quant, same resident path, on real models already validated here. If
// they also land near 0.90 the bar is wrong; if they land near 0.99 then gpt-oss has a real
// remaining defect.
func TestParityFloorControl(t *testing.T) {
	seed := []int{3, 14, 7, 42, 1, 99, 5, 60}
	// Measured 2026-08-31 on the RTX 2070 SUPER, all via this harness:
	//
	//   qwen2.5-coder-0.5b   24 layers  dense          0.973926
	//   qwen2.5-coder-1.5b   28 layers  dense          0.993496
	//   qwen3.6-35b-a3b      40 layers  MoE+streaming  0.982171
	//   gpt-oss-20b          24 layers  MoE+streaming  0.895287   <- the outlier
	//
	// The 35B row is the one that matters: same path, same card, sparse, streamed, and DEEPER,
	// yet 0.982. That is what makes gpt-oss's 0.895 a defect rather than this path's floor.
	// (The 35B is not run here — it needs ~44 GB of host RAM across both arms and ~6 min. Add it
	// back with MoECacheExperts:true and m.NewCache, not decoder.NewKVCache: it is a DeltaNet
	// hybrid whose recurrent state the plain constructor does not allocate.)
	for _, mdl := range []string{
		"qwen2.5-coder-0.5b-instruct-q4_k_m.gguf",
		"qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
	} {
		path := modelPath(mdl)
		if _, err := os.Stat(path); err != nil {
			t.Logf("skip %s", mdl)
			continue
		}
		mg, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
		if err != nil {
			t.Fatalf("load cuda %s: %v", mdl, err)
		}
		rf := mg.ResidentForwardForTest()
		if rf == nil {
			t.Logf("%s: declined (%s)", mdl, mg.ResidentDecline())
			mg.Close()
			continue
		}
		mcpu, err := decoder.Load(path, decoder.Options{Quant: "int4"})
		if err != nil {
			t.Fatalf("load cpu %s: %v", mdl, err)
		}
		_, nL, _, nKV, hd, _, _ := mcpu.Dims()
		cache := decoder.NewKVCache(nL, nKV, hd, 0, 1024)
		minCos := 1.0
		for i, tok := range seed {
			cpuL, e1 := mcpu.ForwardForTest(tok, cache)
			if e1 != nil {
				t.Fatalf("cpu: %v", e1)
			}
			gpuL, e2 := rf.Forward(mg.EmbedResidentForTest(tok), i)
			if e2 != nil {
				t.Fatalf("gpu: %v", e2)
			}
			if c, _ := cosF32(cpuL, gpuL); c < minCos {
				minCos = c
			}
		}
		fmt.Fprintf(os.Stderr, "[floor-control] %-42s CPU-int4 vs CUDA-int4-resident: min cos %.6f (%d layers)\n",
			mdl, minCos, nL)
		mcpu.Close()
		mg.Close()
	}
}
