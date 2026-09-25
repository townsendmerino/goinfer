//go:build realckpt

// Real-model gates for Spark-X2.5-1.7B (model_type "spark2_5", Spark2_5ForCausalLM) — Gate 2
// (docs/tasks/task-spark-x2-5.md): the T3 promotion of the spark2_5 family from Gate 1's
// tiny-random synthetic fixture to a released checkpoint. 1.7B is the smaller of the two
// released sizes (task doc's own "start with the 1.7B... then the 4B"). Fixture:
// scripts/pin_spark2_5_real.py.
//
// TestSpark25Real_gate and TestSpark25Real_fitGuardLongContext need OPPOSITE fit-guard states and
// must be run as SEPARATE invocations, not together: this machine's real headroom (~8 GB) is
// narrower than the checkpoint's ~6.4 GB f32-resident need at the guard's 70% conservative margin
// (a real, monitored bypass was needed and approved to pass the first gate at all — see
// docs/measurements/ once Gate 2/3 are written up), so TestSpark25Real_gate needs
// GOINFER_NO_FIT_GUARD=1 to load; TestSpark25Real_fitGuardLongContext specifically checks that an
// oversized pin gets REFUSED, so it needs the guard ACTIVE (unset) to mean anything — run with
// GOINFER_NO_FIT_GUARD=1 set, it silently no-ops (Load succeeds, the test's own "expected a clean
// refusal" assertion fails on an unrelated cause).
//
//	go test -tags realckpt ./decoder/ -run TestSpark25Tiny_textParity -v                     # Gate 1, no asset needed
//	GOINFER_HEAVY_TESTS=1 GOINFER_NO_FIT_GUARD=1 go test -tags realckpt ./decoder/ -run TestSpark25Real_gate -v -timeout 10m                # Gate 2
//	GOINFER_HEAVY_TESTS=1 go test -tags realckpt ./decoder/ -run TestSpark25Real_fitGuardLongContext -v -timeout 10m                        # Gate 3 (guard must stay active)
package decoder

import (
	"encoding/json"
	"errors"
	"os"
	"testing"
)

func TestSpark25Real_gate(t *testing.T) {
	requireHeavyModel(t)
	ckpt := assetPath(t, "GOINFER_SPARK25_1_7B")
	const golden = "../testdata/spark2_5_real_golden.json.gz"
	raw, err := readGolden(golden)
	if err != nil {
		t.Skipf("no golden (%v) — run scripts/pin_spark2_5_real.py", err)
	}
	var g struct {
		PromptIDs       []int     `json:"prompt_ids"`
		Argmax          int       `json:"argmax"`
		LastLogits      []float32 `json:"last_logits"`
		NNew            int       `json:"n_new"`
		ContinuationIDs []int     `json:"continuation_ids"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}

	// ResidentContext MUST be pinned small: Spark-X2.5's max_position_embeddings is 1,048,576
	// (the "1M-context claim" task-spark-x2-5.md flags as a real hazard) and an unpinned Load
	// sizes the KV cache off it by default — the fit guard correctly refused an unpinned load
	// here (~28 GB of KV for a handful of prompt tokens). This gate only needs
	// len(prompt)+n_new positions.
	m, err := Load(ckpt, Options{ResidentContext: len(g.PromptIDs) + g.NNew + 8}) // f32 resident → tight cosine
	if err != nil {
		t.Fatalf("Load(%s): %v", ckpt, err)
	}
	defer m.Close()
	a := m.w.arch
	if a.Name != "spark2_5" {
		t.Fatalf("arch = %q, want spark2_5", a.Name)
	}
	if a.AttnGate != GateSigmoid {
		t.Fatalf("AttnGate = %v, want GateSigmoid", a.AttnGate)
	}
	// The real checkpoint's own layer_types is a 1:3 full:sliding interleave (28 layers) —
	// assert the mix isn't degenerate, so a loader that silently classified every layer as one
	// kind would fail here rather than pass on a short prompt that never exercises the other.
	var full, sliding int
	for l := 0; l < a.NumLayers; l++ {
		if a.isGlobalLayer(l) {
			full++
		} else {
			sliding++
		}
	}
	t.Logf("spark25-1.7b: %d layers (%d full, %d sliding), H=%d kv=%d headDim=%d",
		a.NumLayers, full, sliding, a.NumHeads, a.NumKVHeads, a.HeadDim)
	if full == 0 || sliding == 0 {
		t.Fatalf("layer-kind split degenerate: %d full / %d sliding — expected a 1:3 mix", full, sliding)
	}

	cache := m.NewCache(len(g.PromptIDs) + g.NNew)
	var logits []float32
	for _, id := range g.PromptIDs {
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("forward: %v", err)
		}
	}
	gotArg := argmax(logits)
	cos := logitCosine(logits, g.LastLogits)
	t.Logf("spark25-1.7b parity: argmax got=%d want=%d | logit cosine=%.6f", gotArg, g.Argmax, cos)
	if gotArg != g.Argmax {
		t.Errorf("last argmax = %d, want %d", gotArg, g.Argmax)
	}
	if cos < 0.9999 { // f32 vs f32 — tight
		t.Errorf("last-logit cosine %.6f < 0.9999", cos)
	}

	got := make([]int, 0, g.NNew)
	for range g.NNew {
		id := argmax(logits)
		got = append(got, id)
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("continuation forward: %v", err)
		}
	}
	t.Logf("spark25-1.7b continuation got=%v want=%v", got, g.ContinuationIDs)
	for i := range g.ContinuationIDs {
		if got[i] != g.ContinuationIDs[i] {
			t.Errorf("continuation[%d] = %d, want %d", i, got[i], g.ContinuationIDs[i])
			break
		}
	}
	// Record the validated metrics (no-op unless GOINFER_MANIFEST_EMIT; skipped on any
	// failure above). f32 vs f32 is the tightest oracle — argmax + continuation exact
	// when the gate passes, so argmax_pct is 100.
	emitParityRow(t, "spark2_5", "full-forward-oracle", "HF f32 (Spark-X2.5-1.7B, Spark2_5ForCausalLM)", 100.0, float64(cos), float64(cos))
}

// TestSpark25Real_fitGuardLongContext is Gate 3 (docs/tasks/task-spark-x2-5.md): "the fit guard
// must price KV correctly for a sliding-window model at long context — which is exactly the case
// where a naive calculation is most wrong. Verify the guard's behaviour on spark2_5 at 131072
// explicitly." The task doc's own hazard is real: an HF user hit a kernel OOM running
// Spark-X2.5-4B at -c 131072 with llama.cpp's --swa-full (which explicitly DISABLES that engine's
// sliding-window memory saving, so every layer priced at full context). goinfer's own M-28 fix
// (decoder/arch.go's kvPositionsAt/kvBytesForCtx) is the generic mechanism that's supposed to
// prevent the equivalent mistake here — this checks it against the REAL checkpoint's geometry,
// not just a synthetic fixture (fitkv_test.go's TestFitGuard_slidingWindowFlattensPastTheWindow
// already covers the generic mechanism on olmo3-tiny; this is the family-specific confirmation
// the task doc asks for, on real numbers).
func TestSpark25Real_fitGuardLongContext(t *testing.T) {
	requireHeavyModel(t)
	ckpt := assetPath(t, "GOINFER_SPARK25_1_7B")

	cfg, err := loadConfig(os.DirFS(ckpt), "config.json")
	if err != nil {
		t.Fatalf("loadConfig: %v", err)
	}
	arch, _, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatalf("resolveArchitecture: %v", err)
	}
	if arch.SlidingWindow <= 0 {
		t.Fatal("test bug: spark25-1.7b did not resolve a sliding window — nothing to check")
	}
	const ctx = 131072
	fixed := estimateKVBytes(cfg, ctx, false, false)
	flat := kvBytesPerPosition(cfg, false, false) * int64(ctx)
	// Hand-computed expectation from the real config (28 layers, 21 sliding/7 full, kvDim =
	// NumKVHeads*HeadDim = 2*256 = 512, f32 = 4 bytes/elem, ×2 for K and V):
	//   7 full layers  × ctx(131072)              × 512 × 4 × 2 = 3,758,096,384 bytes
	//   21 sliding layers × window(512, capped)   × 512 × 4 × 2 =    44,040,192 bytes
	//   total ≈ 3.80 GB — vs the flat (pre-M-28) formula's 28 × ctx × 512 × 4 × 2 ≈ 15.03 GB, ~4x more.
	const wantFixed = int64(7*ctx+21*512) * 512 * 4 * 2
	if fixed != wantFixed {
		t.Errorf("estimateKVBytes(ctx=%d) = %d, want %d (hand-computed: 7 full layers uncapped, 21 sliding capped at window %d)",
			ctx, fixed, wantFixed, arch.SlidingWindow)
	}
	if fixed <= 0 || fixed >= flat {
		t.Fatalf("M-28 fix did not reduce spark2_5's sliding-window KV pricing at ctx=%d (window=%d): fixed=%d, flat(pre-M-28)=%d — want 0 < fixed < flat",
			ctx, arch.SlidingWindow, fixed, flat)
	}
	t.Logf("ctx=%d (window=%d, 7 full / 21 sliding): flat(pre-M-28)=%.2f GB, fixed(post-M-28)=%.2f GB, ratio=%.2fx",
		ctx, arch.SlidingWindow, float64(flat)/1e9, float64(fixed)/1e9, float64(flat)/float64(fixed))

	// Integration check, not just arithmetic: an explicit pin this large must be REFUSED outright
	// (G-07: an explicit request that cannot be honoured is refused, not silently downgraded —
	// decoder/fitguard.go's guardFit), never silently loaded into swap. On THIS machine (16 GB),
	// ~3.8 GB of KV on top of the ~6.4 GB the weights alone need is over budget, so refusal is the
	// expected, safe outcome — the thing actually being checked is that it refuses CLEANLY (a
	// typed FitDeclineError, not an OOM/panic/silent-swap-and-hang) with arithmetic that reflects
	// the windowed price above, not the inflated flat one.
	_, err = Load(ckpt, Options{ResidentContext: ctx})
	if err == nil {
		t.Fatal("Load(ResidentContext=131072) succeeded on a 16 GB machine — expected a clean refusal " +
			"(if this machine genuinely has enough free RAM right now, this is not a failure of the " +
			"guard, just an environment where the negative case doesn't trigger; rerun under load)")
	}
	var declErr *FitDeclineError
	if !errors.As(err, &declErr) {
		t.Fatalf("Load(ResidentContext=131072) failed with %T, want *FitDeclineError (a controlled refusal, not some other failure mode): %v", err, err)
	}
	t.Logf("Load(ResidentContext=131072) correctly refused: %v", declErr)
}
