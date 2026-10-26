//go:build cuda && goinfer_testhooks

package cuda

import (
	"fmt"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestKSH6C_marginAtTheReturnedContext is H6C-G1c (docs/tasks/task-kernel-survey-2026-10.md, "KS-H6 build, review amendment"): every
// return of resolveCtxCapFit, the shortcuts and the floor included, gives the margin at the context it returns, whatever context
// the plan priced.
func TestKSH6C_marginAtTheReturnedContext(t *testing.T) {
	named := func(c int) int64 { return int64(c) * 1000 } // a margin that names its context
	for _, c := range []struct {
		name              string
		request, modelCtx int
		pinned            bool
		slots             int
	}{
		{"a genuine pin", 8192, 32768, true, 1},
		{"the model window at the floor", 0, cudaCtxCapDefault, false, 1},
		{"the plan, one slot", 0, 1 << 20, false, 1},
		{"the plan, four slots", 0, 1 << 20, false, 4},
	} {
		t.Run(c.name, func(t *testing.T) {
			opts := decoder.Options{Quant: "int4"}
			if c.pinned {
				opts.ResidentContext = c.request
			}
			m, err := decoder.Load("../testdata/llama-tiny", opts)
			if err != nil {
				t.Fatalf("Load: %v", err)
			}
			defer m.Close()
			ctx, mb := resolveCtxCapFit(m, c.request, c.modelCtx, c.slots, named)
			if mb != named(ctx) {
				t.Errorf("returned context %d with the margin at %d positions", ctx, mb/1000)
			}
		})
	}
}

// TestKSH6C_planSeesTheMoEPool is H6C-S6: qwen35-tiny (DeltaNet and MoE), its window raised so the plan runs, unpinned, the startup
// self-test skipped. The free VRAM the plan reads must be at most 64 MiB above the fit check's: the weights and the pre-KV buffers,
// not moe_route's pool, which openForPlan pays first. KSH6C_S6_PLANT=1 skips that warm-up, leaving the pool to payRoutePool between
// the two readings; it must run in its own process, since the context keeps the pool once it is paid.
func TestKSH6C_planSeesTheMoEPool(t *testing.T) {
	const base = "../testdata/qwen35-tiny"
	requireDeviceAndFixture(t, base)
	if os.Getenv("KSH6C_S6") == "" {
		t.Skip("set KSH6C_S6=1 (it needs a fresh process: the context keeps the pool once paid)")
	}
	plant := os.Getenv("KSH6C_S6_PLANT") != ""
	prePlanPoolOffForTest = plant
	defer func() { prePlanPoolOffForTest = false }()
	decoder.SkipSelfTests()
	dir := kv3LongContextCopy(t, base, 32768)
	var planFree int64
	planFreeSeenForTest = func(f int64) { planFree = f }
	fitByR := map[*cudaResident]int64{}
	saved := cudaFreeVRAM
	cudaFreeVRAM = func(r *cudaResident) (uint64, error) {
		f, e := saved(r)
		if _, seen := fitByR[r]; !seen {
			fitByR[r] = int64(f)
		}
		return f, e
	}
	m, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4"})
	planFreeSeenForTest, cudaFreeVRAM = nil, saved
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	r, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok || !r.moe {
		t.Fatalf("not a MoE CUDA resident: %s", m.ResidentDecline())
	}
	if planFree == 0 {
		t.Fatal("the plan did not read free VRAM: the window did not put the candidate above the floor")
	}
	mib := func(b int64) float64 { return float64(b) / (1 << 20) }
	gap := planFree - fitByR[r]
	fmt.Fprintf(os.Stderr, "[H6C-S6] plant %v: the plan read %.1f MiB free, the fit check %.1f MiB: %.1f MiB between them; context %d (planned %d)\n",
		plant, mib(planFree), mib(fitByR[r]), mib(gap), r.ctxCap, r.ctxPlanned)
	if plant {
		if gap < 130<<20 {
			t.Errorf("plant: with the warm-up before the plan skipped, the plan read only %.1f MiB more than the fit check (want >= 130): the gate cannot see it", mib(gap))
		}
		return
	}
	if gap > 64<<20 {
		t.Errorf("the plan read %.1f MiB more than the fit check (want <= 64): it priced a pool the build then took", mib(gap))
	}
}
