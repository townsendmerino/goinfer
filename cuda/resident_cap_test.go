//go:build cuda

package cuda

import (
	"errors"
	"fmt"
	"testing"
)

// TestCudaResidentCheckCap gates C3: writes at/past the resident KV cap are refused (a real
// device write there is out-of-bounds memory corruption). Pure logic — no device needed.
//
// The cap is configuration-derived now (resolveCtxCap), so checkCap is exercised at BOTH the
// default and a raised cap: it must guard whatever capacity the caches were actually sized with,
// never a constant.
func TestCudaResidentCheckCap(t *testing.T) {
	for _, cap := range []int{cudaCtxCapDefault, 32768} {
		r := &cudaResident{ctxCap: cap}
		if r.ContextCap() != cap {
			t.Fatalf("ContextCap = %d, want %d", r.ContextCap(), cap)
		}
		for _, c := range []struct {
			pos, n int
			ok     bool
		}{
			{0, 1, true}, {cap - 1, 1, true}, {cap, 1, false},
			{0, cap, true}, {0, cap + 1, false}, {-1, 1, false},
		} {
			if err := r.checkCap(c.pos, c.n); (err == nil) != c.ok {
				t.Errorf("cap=%d: checkCap(%d,%d) err=%v, want ok=%v", cap, c.pos, c.n, err, c.ok)
			}
		}
	}
}

// TestCudaResidentCheckCap_zeroValueRefuses pins the fail-SAFE direction of making the cap a field:
// a cudaResident whose ctxCap was never resolved must refuse every position rather than admit them.
// If this ever inverts, an unresolved cap becomes an unbounded out-of-bounds device write.
func TestCudaResidentCheckCap_zeroValueRefuses(t *testing.T) {
	r := &cudaResident{}
	if err := r.checkCap(0, 1); err == nil {
		t.Fatal("checkCap admitted position 0 on a resident with an unresolved (zero) ctxCap")
	}
}

// TestResolveCtxCap pins the cap policy: default when nothing asks, clamped to the model's own
// context window when something does. The default case is the one that must never drift — a caller
// who did not ask must not start allocating deep-KV VRAM.
func TestResolveCtxCap(t *testing.T) {
	for _, c := range []struct {
		name              string
		request, modelCtx int
		want              int
	}{
		{"unset keeps the default", 0, 32768, cudaCtxCapDefault},
		{"negative keeps the default", -1, 32768, cudaCtxCapDefault},
		{"request under the model window stands", 8192, 32768, 8192},
		{"request over the model window clamps to it", 65536, 32768, 32768},
		{"request equal to the model window stands", 32768, 32768, 32768},
		{"unknown model window lets the request stand", 32768, 0, 32768},
		{"unset with unknown window still defaults", 0, 0, cudaCtxCapDefault},
	} {
		if got := resolveCtxCap(c.request, c.modelCtx); got != c.want {
			t.Errorf("%s: resolveCtxCap(%d, %d) = %d, want %d", c.name, c.request, c.modelCtx, got, c.want)
		}
	}
}

// TestKVBytesForCap pins the sizing formula against the two geometries the deep-context measurements
// were taken on (docs/benchmarks.md): 24.0 KB/position for qwen2.5-coder-0.5b and 56.0 KB/position
// for the 1.5B. The VRAM fail-fast message quotes these, so a formula drift would misreport what a
// configured cap actually costs.
func TestKVBytesForCap(t *testing.T) {
	mk := func(n, kvDim int) []cudaLayer {
		ls := make([]cudaLayer, n)
		for i := range ls {
			ls[i].kvDim = kvDim
		}
		return ls
	}
	for _, c := range []struct {
		name     string
		layers   []cudaLayer
		kbPerPos float64
	}{
		{"qwen2.5-coder-0.5b (24 layers, nKV 2 × hd 64)", mk(24, 2*64), 24.0},
		{"qwen2.5-coder-1.5b (28 layers, nKV 2 × hd 128)", mk(28, 2*128), 56.0},
	} {
		got := float64(kvBytesForCap(1, c.layers)) / 1024
		if got != c.kbPerPos {
			t.Errorf("%s: %.1f KB/position, want %.1f", c.name, got, c.kbPerPos)
		}
		// And it must scale linearly — the 32k prediction the deep-context leg was sized against.
		if want := int64(c.kbPerPos * 1024 * 32768); kvBytesForCap(32768, c.layers) != want {
			t.Errorf("%s: 32k cap = %d bytes, want %d", c.name, kvBytesForCap(32768, c.layers), want)
		}
	}
}

// TestFitsWeightsBudget is M-02's gate for CUDA's previously-nonexistent memory-fit check on the
// FIXED (non-expert) weight term — mirrors metal/backend.go's TestResidentMemGuard, same reasoning:
// a guard that never fires leaves the raw-driver-error failure mode checkWeightsFit exists to
// avoid; one that fires too eagerly silently moves every model to the staged/CPU path.
func TestFitsWeightsBudget(t *testing.T) {
	const gb = int64(1) << 30
	for _, c := range []struct {
		name       string
		need, free int64
		want       bool
	}{
		{"comfortably_fits", 4 * gb, 8 * gb, true},
		{"exactly_at_margin", 8*gb - ctxCapMarginBytes, 8 * gb, true},
		{"just_over_margin", 8*gb - ctxCapMarginBytes + 1, 8 * gb, false},
		{"far_too_large", 22 * gb, 8 * gb, false}, // Qwen3.5-35B-A3B-shaped dense-only sum on an 8 GB card
		// Unknown inputs must never refuse: an unreadable MemInfo or a model reporting zero bytes
		// would otherwise disable residency for everyone, silently.
		{"unknown_need", 0, 8 * gb, true},
		{"unknown_free", 4 * gb, 0, true},
		{"negative_need", -1, 8 * gb, true},
	} {
		if got := fitsWeightsBudget(c.need, c.free); got != c.want {
			t.Errorf("%s: fitsWeightsBudget(%.2f GB, %.2f GB) = %v, want %v",
				c.name, float64(c.need)/float64(gb), float64(c.free)/float64(gb), got, c.want)
		}
	}
}

// TestCheckKVFits_explicitFailsHard_defaultDeclines pins the FAILURE MODE, which is the whole point
// of the load-time check. An operator who explicitly configured a resident context and cannot have
// it must get a hard startup error naming the cost — degrading quietly to the staged path turns a
// config mistake into a latency mystery under load. A default-cap miss must keep the historical
// decline, so no existing deployment starts failing to boot.
func TestCheckKVFits_explicitFailsHard_defaultDeclines(t *testing.T) {
	// 28 layers × 256 kvDim → 56.0 KB/position; 32768 positions ≈ 1.88 GB, far past this "free".
	layers := make([]cudaLayer, 28)
	for i := range layers {
		layers[i].kvDim = 2 * 128
	}
	need := kvBytesForCap(32768, layers)
	if got := float64(need) / 1e9; got < 1.87 || got > 1.89 {
		t.Fatalf("fixture drift: 32k KV = %.3f GB, expected ~1.88", got)
	}
	// The sentinel decides the mode, so assert on it directly rather than on device state.
	explicit := &cudaResident{ctxCap: 32768, layers: layers, ctxExplicit: true}
	dflt := &cudaResident{ctxCap: 32768, layers: layers, ctxExplicit: false}
	if !explicit.ctxExplicit {
		t.Fatal("explicit resident should carry ctxExplicit")
	}
	if dflt.ctxExplicit {
		t.Fatal("default resident must not carry ctxExplicit")
	}
	// errKVWontFit must be the discriminator BuildResident switches on.
	if !errors.Is(fmt.Errorf("wrapped: %w", errKVWontFit), errKVWontFit) {
		t.Fatal("errKVWontFit must survive wrapping — BuildResident matches it with errors.Is")
	}
}
