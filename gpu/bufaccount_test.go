//go:build gpu

package gpu

import "testing"

// TestNoBufferLeak pins V-22 (docs/review-2026-09-04.md): bufaccount.go's own comment has
// pointed callers here since before this test existed ("See TestNoBufferLeak") — a doc comment
// claiming coverage that isn't there is worse than no comment, because it reads as verified.
//
// Two real accounting bugs let LiveBufferBytes grow without bound even though the underlying GPU
// memory WAS correctly released:
//   - Readback(newDeviceBuffer(buf, n)) built a throwaway *DeviceBuffer (accountAlloc), read it,
//     and threw it away — nothing ever called Close (accountFree). Fixed via readbackRaw, which
//     owns the wrapper for exactly the call and closes it before returning.
//   - FusedMLP wrapped the SAME xn/mid buffer in newDeviceBuffer TWICE — once kept in `keep`
//     (released at the end) and once more, locally, for quantizeDevice — double-accounting one
//     real allocation. Fixed by reusing the single kept wrapper instead of building a second.
//
// This drives each fixed path several times and asserts LiveBufferBytes returns to its PRE-CALL
// baseline every time — not just "does not exceed some threshold," which a slow leak could still
// pass for a while. A resident weight fixture (rmsWDev/gateRM/upRM/downRM) is created once and
// deliberately stays live across iterations — that is real, intended residency, not a leak — so
// the baseline is taken AFTER the fixture, not before it.
func TestNoBufferLeak(t *testing.T) {
	ctx, err := New()
	if err != nil {
		t.Skipf("no GPU adapter: %v", err)
	}
	defer ctx.Close()

	t.Run("FusedMLP", func(t *testing.T) {
		f := newMLPFixture(t, ctx, 256, 512)
		if _, err := ctx.FusedMLP(f.x, f.rmsWDev, f.gateRM, f.upRM, f.downRM, f.eps, false); err != nil {
			t.Fatalf("warm-up FusedMLP: %v", err)
		}
		base := LiveBufferBytes()
		for i := range 5 {
			if _, err := ctx.FusedMLP(f.x, f.rmsWDev, f.gateRM, f.upRM, f.downRM, f.eps, false); err != nil {
				t.Fatalf("FusedMLP iter %d: %v", i, err)
			}
			if got := LiveBufferBytes(); got != base {
				t.Errorf("FusedMLP iter %d: LiveBufferBytes = %d, want it back at the baseline %d "+
					"(V-22: a wrapper is being alloc'd and never Close'd)", i, got, base)
			}
		}
	})

	t.Run("vision host wrappers", func(t *testing.T) {
		// Same shape TestVisionLayerNorm_parity already exercises — a smaller, ad hoc size here
		// once triggered a SIGTRAP inside the wgpu-native driver on CreateBuffer, unrelated to
		// the accounting this test is actually about; reusing a proven-safe size avoids
		// introducing a second, unrelated flake into a test meant to pin V-22.
		const rows, h = 257, 1152
		src := make([]float32, rows*h)
		w := make([]float32, h)
		b := make([]float32, h)
		for i := range src {
			src[i] = float32(i%7) - 3
		}
		for i := range w {
			w[i] = 1
		}
		if _, err := ctx.LayerNormRowsHost(src, w, b, rows, h, 1e-6); err != nil {
			t.Fatalf("warm-up LayerNormRowsHost: %v", err)
		}
		if _, err := ctx.softmaxRowsHost(src, rows, h, 0.1); err != nil {
			t.Fatalf("warm-up softmaxRowsHost: %v", err)
		}
		if _, err := ctx.geluHost(src); err != nil {
			t.Fatalf("warm-up geluHost: %v", err)
		}
		base := LiveBufferBytes()
		for i := range 5 {
			if _, err := ctx.LayerNormRowsHost(src, w, b, rows, h, 1e-6); err != nil {
				t.Fatalf("LayerNormRowsHost iter %d: %v", i, err)
			}
			if _, err := ctx.softmaxRowsHost(src, rows, h, 0.1); err != nil {
				t.Fatalf("softmaxRowsHost iter %d: %v", i, err)
			}
			if _, err := ctx.geluHost(src); err != nil {
				t.Fatalf("geluHost iter %d: %v", i, err)
			}
			if got := LiveBufferBytes(); got != base {
				t.Errorf("iter %d: LiveBufferBytes = %d, want it back at the baseline %d "+
					"(V-22: sd/wd/bd/xd released raw, bypassing Close/accountFree)", i, got, base)
			}
		}
	})
}
