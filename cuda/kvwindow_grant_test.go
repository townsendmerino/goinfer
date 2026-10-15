//go:build cuda && goinfer_testhooks

package cuda

import (
	"path/filepath"
	"testing"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
)

func loadResident(t *testing.T, fx string, opts decoder.Options) (*decoder.Model, *cudaResident) {
	t.Helper()
	requireCUDADevice(t)
	opts.Backend, opts.Quant = "cuda", "int4"
	m, err := decoder.Load(filepath.Join("..", "testdata", fx), opts)
	if err != nil {
		t.Fatalf("load %s: %v", fx, err)
	}
	t.Cleanup(func() { m.Close() })
	r, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatalf("%s not CUDA-resident: %s", fx, m.ResidentDecline())
	}
	return m, r
}

// heldPositions is how many positions layer l's K buffer of the bound slot holds.
func heldPositions(r *cudaResident, l int) int { return r.kc[l].Len() / r.layers[l].kvDim }

// TestWindowedKV_grantArithmetic pins kvWindowGrant: nothing left over gives 0, a grant is a whole number of steps, and the cap holds.
func TestWindowedKV_grantArithmetic(t *testing.T) {
	for _, c := range []struct {
		free, reserve, perPos int64
		step, maxExtra, want  int
	}{
		{100, 100, 1, 64, 576, 0},         // free exactly the reserve
		{50, 100, 1, 64, 576, 0},          // less than the reserve
		{100 + 63, 100, 1, 64, 576, 0},    // less than one step left over
		{100 + 200, 100, 1, 64, 576, 192}, // rounded down to steps
		{1 << 40, 100, 1, 64, 576, 576},   // capped
		{1 << 40, 100, 1, 64, 32, 0},      // a cap below one step grants nothing
		{1 << 40, 100, 0, 64, 576, 0},     // no windowed bytes to price
	} {
		if got := kvWindowGrant(c.free, c.reserve, c.perPos, c.step, c.maxExtra); got != c.want {
			t.Errorf("kvWindowGrant(free %d, reserve %d, perPos %d, step %d, max %d) = %d, want %d", c.free, c.reserve, c.perPos, c.step, c.maxExtra, got, c.want)
		}
	}
}

// TestWindowedKV_expertCacheGetsNoGrant: an expert-cache load keeps exactly the planned windowed KV, the window plus the slack the decoder
// prices at main's 512, and so its rollback stays at the floor, slack less step: the expert cache was sized to what was free, so any grant
// would come out of it. The control (the same load without the expert cache) is granted, so the test can go red. The 512 is pinned here
// because the expert-cache path sizes its cache beside it without asking the planner.
func TestWindowedKV_expertCacheGetsNoGrant(t *testing.T) {
	if got := decoder.ResidentWindowedKVCap(1024, 16384); got != 1024+512 {
		t.Fatalf("decoder prices a 1024-window layer at %d positions, want 1536: the expert cache is sized beside this figure", got)
	}
	withWindowedKV(t, windowedKVTestSlack, windowedKVTestStep, 64)
	_, ctl := loadResident(t, "mistral-tiny-window", decoder.Options{ResidentWindowedKV: true})
	if ctl.kvSlack <= kvWindowSlack {
		t.Fatalf("control: no grant without the expert cache (slack %d): the test cannot tell a skipped grant from no headroom", ctl.kvSlack)
	}
	_, r := loadResident(t, "mistral-tiny-window", decoder.Options{ResidentWindowedKV: true, MoECacheExperts: true})
	if !r.cacheExperts || !r.kvWin {
		t.Fatalf("expert cache %v, windowed %v: the load does not have the shape under test", r.cacheExperts, r.kvWin)
	}
	if r.kvSlack != kvWindowSlack || r.kvRollback() != kvWindowSlack-kvWindowStep {
		t.Fatalf("expert-cache load: slack %d, rollback %d, want the planned %d and the floor %d", r.kvSlack, r.kvRollback(), kvWindowSlack, kvWindowSlack-kvWindowStep)
	}
	for l := range r.layers {
		if r.layers[l].kvWin && !r.layers[l].kvShared {
			if got := heldPositions(r, l); got != r.kvWindow+kvWindowSlack {
				t.Fatalf("expert-cache load: layer %d holds %d positions, want the planned %d", l, got, r.kvWindow+kvWindowSlack)
			}
		}
	}
}

// TestWindowedKV_grant: a load with spare VRAM is granted extra slack up to kvWindowRollbackMax of rollback, one whose free VRAM is only the
// reserve gets none, and neither changes the context or the KV slots the plan chose (the grant runs after them). Decode with the grant is
// bit-identical to the full-KV resident, and a rollback of the granted size is reusable.
func TestWindowedKV_grant(t *testing.T) {
	const rollbackMax = 64
	withWindowedKV(t, windowedKVTestSlack, windowedKVTestStep, rollbackMax)
	opts := decoder.Options{ResidentWindowedKV: true, ResidentKVSlots: 2}
	_, rich := loadResident(t, "mistral-tiny-window", opts)
	if rich.kvRollback() != rollbackMax || rich.kvSlack != rollbackMax+kvWindowStep {
		t.Fatalf("with spare VRAM: slack %d, rollback %d, want %d and %d", rich.kvSlack, rich.kvRollback(), rollbackMax+kvWindowStep, rollbackMax)
	}
	for s, b := range rich.kvSlotBufs {
		for l := range rich.layers {
			if got := b.kc[l].Len() / rich.layers[l].kvDim; rich.layers[l].kvWin && got != rich.kvWindow+rich.kvSlack {
				t.Fatalf("slot %d layer %d holds %d positions after the grant, want %d", s, l, got, rich.kvWindow+rich.kvSlack)
			}
		}
	}

	orig := kvGrantFreeVRAM
	t.Cleanup(func() { kvGrantFreeVRAM = orig })
	kvGrantFreeVRAM = func(r *cudaResident) (uint64, error) {
		return uint64(r.extraBytes + ctxCapMarginBytes + 2*r.prefillPassBytes(prefillDefaultChunk) + 1), nil
	}
	_, poor := loadResident(t, "mistral-tiny-window", opts)
	kvGrantFreeVRAM = orig
	if poor.kvSlack != kvWindowSlack {
		t.Fatalf("with no VRAM beyond the reserve: slack %d, want the planned %d", poor.kvSlack, kvWindowSlack)
	}
	if rich.ctxCap != poor.ctxCap || rich.kvSlotsN != poor.kvSlotsN {
		t.Fatalf("the grant changed the plan: ctx %d / %d, slots %d / %d", rich.ctxCap, poor.ctxCap, rich.kvSlotsN, poor.kvSlotsN)
	}

	_, full := loadResident(t, "mistral-tiny-window", decoder.Options{ResidentKVSlots: 2})
	mw, win := loadResident(t, "mistral-tiny-window", opts)
	decodeSteps(t, full, win, mw, 0, 300)
	if win.kvBase == 0 {
		t.Fatal("no compaction in 300 steps under the grant")
	}
	back := 300 - rollbackMax
	if got := win.ReusableKV(back); got != back {
		t.Fatalf("ReusableKV(%d) = %d after a rollback of %d, the granted rollback (base %d)", back, got, rollbackMax, win.kvBase)
	}
}

// TestWindowedKV_stepIdentity64 is the identity gate at the production step: decode across many 64-position compactions (slack 80, so the
// floor rollback is 16 on the fixture's 16-position window) is bit-identical to the full-KV resident in every step's logits, and every
// windowed layer's held K and V equal the full cache's at the same absolute positions.
func TestWindowedKV_stepIdentity64(t *testing.T) { eachWindowFixture(t, windowedStepIdentity64) }

func windowedStepIdentity64(t *testing.T, fx string) {
	withWindowedKV(t, 80, 64, 16)
	opts := decoder.Options{ResidentWindowedKV: true}
	mw, win := loadResident(t, fx, opts)
	_, full := loadResident(t, fx, decoder.Options{})
	if !win.kvWin || win.kvSlack != 80 || win.kvRollback() != 16 {
		t.Fatalf("windowed %v, slack %d, rollback %d: not the production step shape", win.kvWin, win.kvSlack, win.kvRollback())
	}
	steps := min(win.ctxCap-1, 600)
	compactions, lastBase := 0, 0
	for done := 0; done < steps; done += 50 {
		n := min(50, steps-done)
		for i := 0; i < n; i++ {
			decodeSteps(t, full, win, mw, done+i, 1)
			if win.kvBase != lastBase {
				compactions++
				lastBase = win.kvBase
			}
		}
		compareHeldKV(t, full, win, done+n)
	}
	if compactions < 5 {
		t.Fatalf("%d compactions in %d steps: the test did not exercise the step", compactions, steps)
	}
	t.Logf("%s: %d steps, %d compactions, final base %d", fx, steps, compactions, win.kvBase)
}

// compareHeldKV checks every windowed layer's held positions [base, end) against the full-KV resident's same positions, bit for bit.
func compareHeldKV(t *testing.T, full, win *cudaResident, end int) {
	t.Helper()
	for l := range win.layers {
		Ly := &win.layers[l]
		if !Ly.kvWin || Ly.kvShared {
			continue
		}
		n := (end - win.kvBase) * Ly.kvDim
		for i, pair := range [][2]Buffer{{full.kc[l], win.kc[l]}, {full.vc[l], win.vc[l]}} {
			a, b := make([]float32, full.kc[l].Len()), make([]float32, pair[1].Len())
			if err := full.do(func() error {
				if e := full.stream.Sync(); e != nil {
					return e
				}
				return gpu.Download(pair[0], a)
			}); err != nil {
				t.Fatal(err)
			}
			if err := win.do(func() error {
				if e := win.stream.Sync(); e != nil {
					return e
				}
				return gpu.Download(pair[1], b)
			}); err != nil {
				t.Fatal(err)
			}
			if j, ok := sameBits(a[win.kvBase*Ly.kvDim:win.kvBase*Ly.kvDim+n], b[:n]); !ok {
				t.Fatalf("layer %d %s at end %d: held position %d differs from the full cache (base %d)", l, []string{"K", "V"}[i], end, win.kvBase+j/Ly.kvDim, win.kvBase)
			}
		}
	}
}
