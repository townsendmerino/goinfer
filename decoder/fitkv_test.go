package decoder

import (
	"os"
	"testing"
)

// TestKvDimAt_zeroForRecurrentLayers gates M-28's first sub-fix: a linear (DeltaNet), Mamba-2, or
// gated-conv mixer layer holds no position-indexed K/V array at all — a small fixed-size
// recurrent state instead, never priced by ctx — so kvDimAt must report zero for each, not the
// ordinary NumKVHeads*HeadDim the flat formula this replaces charged every layer regardless.
func TestKvDimAt_zeroForRecurrentLayers(t *testing.T) {
	a := &Architecture{
		NumKVHeads:    8,
		HeadDim:       128,
		layerIsLinear: func(i int) bool { return i == 0 },
		layerIsMamba:  func(i int) bool { return i == 1 },
		layerIsConv:   func(i int) bool { return i == 2 },
	}
	want := map[int]int{0: 0, 1: 0, 2: 0, 3: 8 * 128}
	for i, w := range want {
		if got := a.kvDimAt(i); got != w {
			t.Errorf("kvDimAt(%d) = %d, want %d", i, got, w)
		}
	}
}

// TestKvDimAt_mlaUsesCompressedLatentNotReconstructedWidth gates M-28's MLA sub-fix: the cache
// holds the compressed KVLoRARank+QKRopeHeadDim latent (forward_deepseek.go reconstructs
// per-head K/V from it each step), not NumKVHeads*HeadDim's full reconstructed width — the audit's
// own DeepSeek-V2-Lite figure ("MLA stores 576/layer, priced 4096") is exactly this gap.
func TestKvDimAt_mlaUsesCompressedLatentNotReconstructedWidth(t *testing.T) {
	a := &Architecture{
		NumKVHeads: 128, HeadDim: 128, // a deliberately huge reconstructed width, to prove it is NOT what gets used
		mla: &mlaParams{KVLoRARank: 512, QKRopeHeadDim: 64},
	}
	const want = 512 + 64
	if got := a.kvDimAt(0); got != want {
		t.Errorf("kvDimAt with MLA = %d, want %d (the compressed latent, not %d = NumKVHeads*HeadDim)",
			got, want, 128*128)
	}
}

// TestKvPositionsAt_slidingWindowCapsLocalLayersNotGlobal gates M-28's sliding-window sub-fix: a
// local (non-global) layer's ring never grows past SlidingWindow regardless of ctx, while a
// global layer's cost keeps growing with ctx — the gemma3-12b "~6x over at 131k" figure traces
// to exactly this asymmetry being ignored.
func TestKvPositionsAt_slidingWindowCapsLocalLayersNotGlobal(t *testing.T) {
	a := &Architecture{
		SlidingWindow: 4096,
		layerIsGlobal: func(i int) bool { return i == 0 },
	}
	if got := a.kvPositionsAt(0, 100000); got != 100000 {
		t.Errorf("global layer kvPositionsAt(ctx=100000) = %d, want uncapped 100000", got)
	}
	if got := a.kvPositionsAt(1, 100000); got != 4096 {
		t.Errorf("local layer kvPositionsAt(ctx=100000) = %d, want capped at SlidingWindow 4096", got)
	}
	if got := a.kvPositionsAt(1, 2000); got != 2000 {
		t.Errorf("local layer kvPositionsAt(ctx=2000, below the window) = %d, want uncapped 2000 (ctx itself)", got)
	}
}

// TestKvBytesForCtx_skipsRecurrentLayers is the end-to-end combination: a DeltaNet-hybrid-shaped
// architecture (half the layers linear, holding no KV) must price only the real attention
// layers, not all of them — the flat formula this replaces would have doubled the true cost here.
func TestKvBytesForCtx_skipsRecurrentLayers(t *testing.T) {
	a := &Architecture{
		NumLayers: 8, NumKVHeads: 8, HeadDim: 128,
		layerIsLinear: func(i int) bool { return i%2 == 0 }, // layers 0,2,4,6 linear; 1,3,5,7 ordinary
	}
	const dim = 8 * 128
	const ctx = 1000
	perLayerAtCtx := int64(2 * 4 /* f32 */ * dim * ctx) // ×2 for K+V
	want := perLayerAtCtx * 4                           // 4 real attention layers, not 8
	if got := kvBytesForCtx(a, ctx, false, false); got != want {
		t.Errorf("kvBytesForCtx = %d, want %d (4 real attention layers, not the flat formula's 8)", got, want)
	}
}

// TestKvBytesForCtx_nilOrEmptyProceedsUnknown matches this file's "every unknown proceeds"
// discipline: an unresolved architecture or a non-positive ctx prices at zero rather than
// panicking or guessing.
func TestKvBytesForCtx_nilOrEmptyProceedsUnknown(t *testing.T) {
	if got := kvBytesForCtx(nil, 1000, false, false); got != 0 {
		t.Errorf("kvBytesForCtx(nil arch) = %d, want 0", got)
	}
	if got := kvBytesForCtx(&Architecture{NumLayers: 8, NumKVHeads: 8, HeadDim: 128}, 0, false, false); got != 0 {
		t.Errorf("kvBytesForCtx(ctx=0) = %d, want 0", got)
	}
}

// realArchFixture loads a real checkpoint's config.json and resolves its Architecture, or skips
// if the (gitignored, real) fixture is absent — same convention as the rest of this package's
// hybrid/MLA tests (loadHybridTiny etc. in fitplan_test.go).
func realArchFixture(t *testing.T, dir string) (*Config, *Architecture) {
	t.Helper()
	cfg, err := loadConfig(os.DirFS(dir), "config.json")
	if err != nil {
		t.Skipf("no fixture at %s: %v", dir, err)
	}
	arch, _, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatalf("resolveArchitecture(%s): %v", dir, err)
	}
	return cfg, arch
}

// TestFitGuard_mlaKVPricedAtCompressedLatentNotFullWidth is M-28's DeepSeek-V2-Lite case from the
// audit ("MLA stores 576/layer, priced 4096"), on the real fixture: the fixed estimator must price
// meaningfully LESS than the flat formula it replaces, because the flat formula charges the full
// reconstructed per-head width where MLA actually caches only the compressed latent.
func TestFitGuard_mlaKVPricedAtCompressedLatentNotFullWidth(t *testing.T) {
	cfg, arch := realArchFixture(t, "../testdata/deepseek-tiny")
	if arch.mla == nil {
		t.Fatal("test bug: deepseek-tiny did not resolve as an MLA architecture — nothing to check")
	}
	const ctx = 4096
	fixed := estimateKVBytes(cfg, ctx, false, false)
	flat := kvBytesPerPosition(cfg, false, false) * int64(ctx)
	if fixed <= 0 || fixed >= flat {
		t.Fatalf("M-28 fix did not reduce MLA KV pricing at ctx=%d: fixed=%d, flat(pre-M-28)=%d — want 0 < fixed < flat", ctx, fixed, flat)
	}
	t.Logf("ctx=%d: flat(pre-M-28)=%d bytes, fixed(post-M-28)=%d bytes, ratio=%.2fx", ctx, flat, fixed, float64(flat)/float64(fixed))
}

// TestFitGuard_hybridKVSkipsRecurrentLayers is M-28's DeltaNet-hybrid case, on the real qwen3.5
// fixture the audit itself names ("Qwen3.5-4B: priced 128 KiB/position vs 32 KiB stored").
func TestFitGuard_hybridKVSkipsRecurrentLayers(t *testing.T) {
	cfg, arch := realArchFixture(t, "../testdata/qwen35-tiny")
	hasLinear := false
	for i := 0; i < arch.NumLayers; i++ {
		if arch.isLinearLayer(i) {
			hasLinear = true
			break
		}
	}
	if !hasLinear {
		t.Fatal("test bug: qwen35-tiny did not resolve any linear (DeltaNet) layers — nothing to check")
	}
	const ctx = 4096
	fixed := estimateKVBytes(cfg, ctx, false, false)
	flat := kvBytesPerPosition(cfg, false, false) * int64(ctx)
	if fixed <= 0 || fixed >= flat {
		t.Fatalf("M-28 fix did not reduce hybrid KV pricing at ctx=%d: fixed=%d, flat(pre-M-28)=%d — want 0 < fixed < flat", ctx, fixed, flat)
	}
	t.Logf("ctx=%d: flat(pre-M-28)=%d bytes, fixed(post-M-28)=%d bytes, ratio=%.2fx", ctx, flat, fixed, float64(flat)/float64(fixed))
}

// TestFitGuard_slidingWindowFlattensPastTheWindow is M-28's olmo3-7B case from the audit
// ("~3.4x over at 65k"), on the real fixture: at a ctx well past the model's own sliding window,
// the fixed estimator's local layers must have stopped growing while the flat formula keeps
// charging every layer as if it held the full context.
func TestFitGuard_slidingWindowFlattensPastTheWindow(t *testing.T) {
	cfg, arch := realArchFixture(t, "../testdata/olmo3-tiny")
	if arch.SlidingWindow <= 0 {
		t.Fatal("test bug: olmo3-tiny did not resolve a sliding window — nothing to check")
	}
	ctx := arch.SlidingWindow * 8 // deliberately far past the window
	fixed := estimateKVBytes(cfg, ctx, false, false)
	flat := kvBytesPerPosition(cfg, false, false) * int64(ctx)
	if fixed <= 0 || fixed >= flat {
		t.Fatalf("M-28 fix did not reduce sliding-window KV pricing at ctx=%d (window=%d): fixed=%d, flat(pre-M-28)=%d — want 0 < fixed < flat",
			ctx, arch.SlidingWindow, fixed, flat)
	}
	t.Logf("ctx=%d (window=%d): flat(pre-M-28)=%d bytes, fixed(post-M-28)=%d bytes, ratio=%.2fx",
		ctx, arch.SlidingWindow, flat, fixed, float64(flat)/float64(fixed))
}
