//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"os"
	"runtime"
	"strconv"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
)

// TestA13_SingleFailedAllocPoisons asks whether ONE failed allocation poisons the context, or whether it
// takes a full drain (after which a later `attention` launch returns success and writes nothing).
// Production does hit a failed allocation: BuildResident sizes the expert cache against free VRAM and can
// be refused, and one model's OOM can land in a process another model is using.
//
//	(c) correct -> one failure does not poison; bisect upward to bound what does
//	(c) zeros   -> any failed allocation poisons; one model's OOM silently breaks another model's CUDA
//	               path in the same process with no error
//
// Three repeats either way: A13's failures vary run to run, so a single negative clears nothing.
// Record: docs/queue-engineering.md, section A13. Detail: docs/code-notes/cuda.md#TestA13_SingleFailedAllocPoisons.
func TestA13_SingleFailedAllocPoisons(t *testing.T) {
	if os.Getenv("GOINFER_A13_SINGLEFAIL") == "" {
		t.Skip("set GOINFER_A13_SINGLEFAIL=1 — A13 probe, deliberately not part of the tier")
	}
	// GOINFER_A13_PIN: pin the goroutine (LockOSThread). The resident's executor is pinned and has never
	// poisoned; if pinning alone makes this clean, the mechanism is unpinned CUDA use from a migrating
	// goroutine rather than driver-side module eviction.
	if os.Getenv("GOINFER_A13_PIN") != "" {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		t.Logf("goroutine PINNED (runtime.LockOSThread)")
	}
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	dev, err := gc.GetDevice(0)
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	ctx, err := dev.Primary()
	if err != nil {
		t.Skipf("no context: %v", err)
	}
	defer ctx.Close()
	bg := context.Background()
	stream, err := ctx.NewStream()
	if err != nil {
		t.Fatalf("stream: %v", err)
	}

	load := func(tag string) (*gc.Function, *gc.Function, *gc.Function) {
		mod, e := ctx.LoadModule(gluePTX)
		if e != nil {
			t.Fatalf("%s: glue module: %v", tag, e)
		}
		fRms, e1 := mod.Function("rmsnorm_quant")
		fAttn, e2 := mod.Function("attention")
		fSw, e3 := mod.Function("glu_quant")
		if e1 != nil || e2 != nil || e3 != nil {
			t.Fatalf("%s: resolve: %v %v %v", tag, e1, e2, e3)
		}
		return fRms, fSw, fAttn
	}

	const nH, nKV, hd, nKeys, I = 12, 2, 64, 129, 256
	const scale = float32(0.05)

	// (a) BASELINE — the oracle must pass before any of this means anything.
	fRms, fSw, fAttn := load("baseline")
	validateGlue(t, ctx, stream, bg, fRms, fSw, fAttn, nH*hd, I, nH, nKV, hd, nKeys, scale)
	if t.Failed() {
		t.Fatal("baseline already failed — this probe says nothing; fix that first")
	}
	free0, _, _ := ctx.MemInfo()
	t.Logf("(a) baseline OK, free=%.1f MiB", float64(free0)/(1<<20))

	// PERSISTENT ALLOCATION (GOINFER_A13_KEEP=<MiB>): held for the whole test and never freed, so the
	// context's live set never collapses to empty. Tests whether the trigger is the live set going near
	// empty rather than memory pressure (a resident context always holds a model's weights).
	if v := os.Getenv("GOINFER_A13_KEEP"); v != "" {
		mib, _ := strconv.Atoi(v)
		var keep []*gc.Buffer[float32]
		const chunk = 64 << 20
		for got := 0; got < mib<<20; got += chunk {
			kb, e := gc.Alloc[float32](ctx, chunk/4)
			if e != nil {
				break
			}
			keep = append(keep, kb)
		}
		t.Logf("KEEP: holding %d x 64 MiB = %d MiB for the whole test (never freed)",
			len(keep), len(keep)*64)
		defer func() {
			for _, kb := range keep {
				kb.Close()
			}
		}()
	}

	// (b) EXACTLY ONE failed allocation: one request larger than the whole device, nothing retained. If it
	// succeeds the probe is void, so that is checked. GOINFER_A13_NFAIL=N bisects upward to bound how little
	// is enough; the decline path's own attempt count has to sit inside whatever bound comes out.
	nfail := 1
	if v := os.Getenv("GOINFER_A13_NFAIL"); v != "" {
		if k, e := strconv.Atoi(v); e == nil {
			nfail = k
		}
	}
	huge := int(free0/4) + (2 << 30) // comfortably beyond free VRAM, in float32 elements
	var aerr error
	for i := 0; i < nfail; i++ {
		b, e := gc.Alloc[float32](ctx, huge)
		if e == nil {
			if b != nil {
				b.Close()
			}
			t.Skipf("the oversized allocation SUCCEEDED (%.1f GiB) — cannot create a failure on this "+
				"device; probe is void rather than negative", float64(huge)*4/(1<<30))
		}
		aerr = e
	}
	// PARTIAL DRAIN (GOINFER_A13_HOLDPCT): successfully hold a percentage of free VRAM, then release it.
	// Separates a successful large allocation from a refusal as the trigger. Detail: docs/code-notes/cuda.md#TestA13_SingleFailedAllocPoisons.holdpct.
	if v := os.Getenv("GOINFER_A13_HOLDPCT"); v != "" {
		pct, _ := strconv.Atoi(v)
		want := int(free0) * pct / 100
		var held []*gc.Buffer[float32]
		const chunk = 64 << 20 // 64 MiB
		for got := 0; got+chunk <= want; got += chunk {
			hb, e := gc.Alloc[float32](ctx, chunk/4)
			if e != nil {
				break
			}
			held = append(held, hb)
		}
		f, _, _ := ctx.MemInfo()
		t.Logf("(b') held %d chunk(s) = %.1f MiB (%d%% target); free now %.1f MiB",
			len(held), float64(len(held))*float64(chunk)/(1<<20), pct, float64(f)/(1<<20))
		for _, hb := range held {
			hb.Close()
		}
		f2, _, _ := ctx.MemInfo()
		t.Logf("(b') released; free back to %.1f MiB", float64(f2)/(1<<20))
	}
	free1, _, _ := ctx.MemInfo()
	t.Logf("(b) %d allocation(s) of %.1f GiB REFUSED as designed: %v", nfail, float64(huge)*4/(1<<30), aerr)
	t.Logf("    free after the refusals: %.1f MiB (was %.1f)", float64(free1)/(1<<20), float64(free0)/(1<<20))

	// (c) THE QUESTION. Fresh module, normal allocations, same oracle.
	fRms2, fSw2, fAttn2 := load("after-one-failure")
	validateGlue(t, ctx, stream, bg, fRms2, fSw2, fAttn2, nH*hd, I, nH, nKV, hd, nKeys, scale)
	if t.Failed() {
		t.Logf("(c) POISONED after failed allocations — production is exposed: one model's OOM " +
			"silently breaks another model's CUDA path in the same process, with no error anywhere. " +
			"See docs/QUEUE.md A13.")
		return
	}
	t.Logf("(c) correct after %d failed allocation(s) — this many refusals do not poison the context", nfail)
}
