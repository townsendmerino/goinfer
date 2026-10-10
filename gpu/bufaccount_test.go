//go:build gpu

package gpu

import "testing"

// TestNoBufferLeak pins that LiveBufferBytes returns to its pre-call baseline after each path
// that once leaked accounting while the GPU memory itself was released: Readback (readbackRaw now
// owns the throwaway *DeviceBuffer wrapper and closes it) and FusedMLP (the xn/mid buffer is
// wrapped once, not twice). It drives each path several times and asserts the exact baseline
// every time, which a threshold check could pass for a while on a slow leak. The resident weight
// fixture (rmsWDev/gateRM/upRM/downRM) is created once and stays live across iterations on
// purpose, so the baseline is taken after it. bufaccount.go points callers here. Background:
// docs/code-notes/gpu.md#TestNoBufferLeak.
func TestNoBufferLeak(t *testing.T) {
	ctx, err := New()
	if err != nil {
		t.Skipf("no GPU adapter: %v", err)
	}
	defer ctx.Close()

	t.Run("FusedMLP", func(t *testing.T) {
		f := newMLPFixture(t, ctx, 256, 512)
		defer f.close()
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
		// Reuse TestVisionLayerNorm_parity's shape: a smaller ad hoc size once SIGTRAPed inside the
		// wgpu-native driver on CreateBuffer, unrelated to the accounting under test.
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
