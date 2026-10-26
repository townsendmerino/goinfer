//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"fmt"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestKSH6C_moeModelPaysItsPool is H6C-S5 (docs/tasks/task-kernel-survey-2026-10.md, "KS-H6 build, amendment"): a MoE resident built
// with the startup self-test skipped pays moe_route's local-memory pool before its fit check (payRoutePool), so serving it takes
// nothing the fit check did not see. qwen35-tiny (DeltaNet and MoE) is loaded through decoder.Load, then one prefill and 8 decode
// tokens. KSH6C_S5_PLANT=1 skips the warm-up, and must run in its own process: the context keeps the pool once it is paid.
func TestKSH6C_moeModelPaysItsPool(t *testing.T) {
	const base = "../testdata/qwen35-tiny"
	requireDeviceAndFixture(t, base)
	if os.Getenv("KSH6C_S5") == "" {
		t.Skip("set KSH6C_S5=1 (it needs a fresh process: the context keeps the pool once paid)")
	}
	plant := os.Getenv("KSH6C_S5_PLANT") != ""
	routePoolOffForTest = plant
	defer func() { routePoolOffForTest = false }()
	decoder.SkipSelfTests() // the probe's MoE fixture would pay the pool first
	// A short-lived device per reading (decoder.FreeBytesFor): a held handle would keep the primary context alive across the load.
	free := func() int64 {
		f, ok := decoder.FreeBytesFor("cuda")
		if !ok {
			t.Fatal("no free-VRAM reading")
		}
		return f
	}
	var beforeWarm, fitFree int64
	saved := cudaFreeVRAM
	cudaFreeVRAM = func(r *cudaResident) (uint64, error) {
		f, e := saved(r)
		if fitFree == 0 {
			fitFree = int64(f)
		}
		return f, e
	}
	beforeWarm = free()
	m, err := decoder.Load(base, decoder.Options{Backend: "cuda", Quant: "int4"})
	cudaFreeVRAM = saved
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	r, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok || !r.moe {
		t.Fatalf("not a MoE CUDA resident: %s", m.ResidentDecline())
	}
	loaded := free()
	ids := make([]int, 40)
	for i := range ids {
		ids[i] = 3 + i%50
	}
	ch, g := m.Generate(context.Background(), ids, 8, decoder.SamplingParams{})
	for range ch {
	}
	if err := g.Err(); err != nil {
		t.Fatalf("generate: %v", err)
	}
	after := free()
	mib := func(b int64) float64 { return float64(b) / (1 << 20) }
	fmt.Fprintf(os.Stderr, "[H6C-S5] plant %v: free %.1f MiB before the load, %.1f at the fit check, %.1f at the end of the load, %.1f after a prefill and 8 tokens: the drop across serving %.1f MiB\n",
		plant, mib(beforeWarm), mib(fitFree), mib(loaded), mib(after), mib(loaded-after))
	drop := loaded - after
	if plant {
		if drop < 130<<20 {
			t.Errorf("plant: with the warm-up skipped serving took %.1f MiB, want >= 130 (the pool): the gate cannot see it", mib(drop))
		}
		return
	}
	if drop > 2<<20 {
		t.Errorf("serving took %.1f MiB the fit check did not see (want <= 2): the pool was not paid before it", mib(drop))
	}
}
