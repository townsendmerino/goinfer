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
		t.Fatalf("%s not CUDA-resident: %s (a tiny fixture must meet the CUDA kernels' shape rules, such as int4 MoE widths in "+
			"multiples of 32: regenerate it to fit, or use a resident-eligible sibling)", fx, m.ResidentDecline())
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

// TestWindowedKV_expertCacheGetsTheGrant: an expert-cache load of a windowed model is granted rollback like any other load, before its cache
// is sized (docs/queue-performance.md, P34, cause 2), and a card with only the reserve and the cache's minimum free gets no grant and still
// loads. The control (the same load without the expert cache) is granted the same, so a skipped grant reads red. The 512 is pinned here
// because the expert cache is sized beside the planner's figure.
func TestWindowedKV_expertCacheGetsTheGrant(t *testing.T) {
	if got := decoder.ResidentWindowedKVCap(1024, 16384); got != 1024+512 {
		t.Fatalf("decoder prices a 1024-window layer at %d positions, want 1536: the expert cache is sized beside this figure", got)
	}
	const rollbackMax = 64
	withWindowedKV(t, windowedKVTestSlack, windowedKVTestStep, rollbackMax)
	_, ctl := loadResident(t, "mistral-tiny-window", decoder.Options{ResidentWindowedKV: true})
	if ctl.kvRollback() != rollbackMax {
		t.Fatalf("control: rollback %d without the expert cache, want the granted %d: the test cannot tell a skipped grant from no headroom", ctl.kvRollback(), rollbackMax)
	}
	_, r := loadResident(t, "mistral-tiny-window", decoder.Options{ResidentWindowedKV: true, MoECacheExperts: true})
	if !r.cacheExperts || !r.kvWin {
		t.Fatalf("expert cache %v, windowed %v: the load does not have the shape under test", r.cacheExperts, r.kvWin)
	}
	if r.kvRollback() != rollbackMax {
		t.Fatalf("expert-cache load: rollback %d, want the granted %d (slack %d)", r.kvRollback(), rollbackMax, r.kvSlack)
	}
	for l := range r.layers {
		if r.layers[l].kvWin && !r.layers[l].kvShared {
			if got := heldPositions(r, l); got != r.kvWindow+r.kvSlack {
				t.Fatalf("expert-cache load: layer %d holds %d positions, want the granted %d", l, got, r.kvWindow+r.kvSlack)
			}
		}
	}

	// A routed fixture with windowed layers: the hold-back is the cache's minimum, and the cache is still allocated after the grant.
	// gemma4-moe-kv-tiny32 is gemma4-moe-kv-tiny at moe_intermediate_size 32 (scripts/pin_gemma4_moe_kv_forward.py, PIN_MOE_INTER), the
	// width CUDA's int4 MoE needs; the 16-wide original declines residency.
	t.Run("routed experts", func(t *testing.T) {
		_, mr := loadResident(t, "gemma4-moe-kv-tiny32", decoder.Options{ResidentWindowedKV: true, MoECacheExperts: true})
		if !mr.kvWin || !mr.cacheExperts || mr.expertCacheFloorBytes() <= 0 {
			t.Fatalf("gemma4-moe-kv-tiny32: windowed %v, expert cache %v, floor %d B: not the routed shape", mr.kvWin, mr.cacheExperts, mr.expertCacheFloorBytes())
		}
		if mr.kvRollback() <= kvWindowSlack-kvWindowStep {
			t.Fatalf("routed expert-cache load: rollback %d, the floor: no grant", mr.kvRollback())
		}
		if mr.cacheSlots < mr.topK {
			t.Fatalf("routed expert-cache load: %d cache slots after the grant, under top-%d", mr.cacheSlots, mr.topK)
		}
	})

	orig := kvGrantFreeVRAM
	t.Cleanup(func() { kvGrantFreeVRAM = orig })
	kvGrantFreeVRAM = func(r *cudaResident) (uint64, error) {
		return uint64(r.extraBytes + ctxCapMarginBytes + 2*r.prefillPassBytes(prefillDefaultChunk) + r.expertCacheFloorBytes() + 1), nil
	}
	_, poor := loadResident(t, "mistral-tiny-window", decoder.Options{ResidentWindowedKV: true, MoECacheExperts: true})
	kvGrantFreeVRAM = orig
	if poor.kvSlack != kvWindowSlack {
		t.Fatalf("expert-cache load with only the reserve and the cache's minimum free: slack %d, want the planned %d", poor.kvSlack, kvWindowSlack)
	}
}

// TestWindowedKV_e0ResendReplay replays kvWindowNextBase over the shape of tack's E0 resend on Mellum2.1's 1,024-position window
// (docs/queue-performance.md, P34, cause 2): a prompt of n positions prefilled in 512-row passes and one generated token, then a resend
// that diverges d positions before the newest, rounded down to 64-position blocks. At the floor rollback (slack 512, step 64: 448) a
// resend that diverges 518 or 600 back is refused and one that diverges 406 back is reused; at the granted rollback (1,024) all three
// are reused. These are the pattern the by-day probe on nobara must show, before and after the fix.
func TestWindowedKV_e0ResendReplay(t *testing.T) {
	const window, step, block = 1024, 64, 64
	reusable := func(n, d, slack int) bool {
		rollback, capN, base := slack-step, window+slack, 0
		pos := 0
		for pos < n {
			m := min(512, slack+1, n-pos)
			from, to, err := kvWindowNextBase(base, pos, m, window, capN, rollback)
			if err != nil {
				t.Fatalf("prefill at %d: %v", pos, err)
			}
			_ = from
			base, pos = to, pos+m
		}
		_, base, err := kvWindowNextBase(base, n, 1, window, capN, rollback)
		if err != nil {
			t.Fatalf("decode at %d: %v", n, err)
		}
		return kvWindowReusable(base, (n+1-d)/block*block, window)
	}
	for _, n := range []int{4096, 8192, 15360, 31744} {
		for _, c := range []struct {
			d              int
			floor, granted bool
		}{{406, true, true}, {518, false, true}, {600, false, true}} {
			if got := reusable(n, c.d, 512); got != c.floor {
				t.Errorf("n=%d, diverging %d back, floor rollback: reusable %v, want %v", n, c.d, got, c.floor)
			}
			if got := reusable(n, c.d, 512+576); got != c.granted {
				t.Errorf("n=%d, diverging %d back, granted rollback: reusable %v, want %v", n, c.d, got, c.granted)
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
