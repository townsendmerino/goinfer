//go:build cuda

package cuda

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestGemma4_26B_1bBound is Lever-1b Step 0: bound the win BEFORE building the aikit event primitive
// + the gemma4MoeMLP reorder. 1b hides the per-layer router-wait drain behind the dense branch, so
// the recoverable time is capped by the DENSE-BRANCH GPU time. Measure it (loop the dispatch
// sequence N times, one sync, divide — no event support needed) and compare to the ~12 ms/token
// drain the skip-readback probe isolated (59→36, minus ~11 ms DMA). If the dense branch is thin, 1b
// is not worth the primitive+reorder and the fallback (cross-layer pipelining, a larger design) or
// the dispatch-reduction lever should be chosen deliberately.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags cuda ./cuda/ -run TestGemma4_26B_1bBound -v -timeout 40m
func TestGemma4_26B_1bBound(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("GOINFER_HEAVY_TESTS unset — real 26B micro-bench")
	}
	dir := os.Getenv("GOINFER_GEMMA4_26B")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, "models", "gemma-4-26b-a4b-it")
	}
	if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
		t.Skipf("no 26B checkpoint at %s: %v", dir, err)
	}
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	t.Setenv("GOINFER_MOE_CACHE_EXPERTS", "1")
	mc, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer mc.Close()
	r := mc.ResidentForwardForTest().(*cudaResident)

	// one real forward to populate the buffers the branches read (activation, norms, router inputs)
	if _, err := r.Forward(mc.EmbedResidentForTest(1), 0); err != nil {
		t.Fatalf("warm forward: %v", err)
	}

	L := &r.layers[0]
	if !L.g4moe {
		t.Fatalf("layer 0 is not g4moe")
	}
	nullBias := ArgNull()
	nMoE := 0
	for i := range r.layers {
		if r.layers[i].g4moe || r.layers[i].isMoE {
			nMoE++
		}
	}

	// The dense-branch dispatch sequence (mirrors gemma4MoeMLP's dense half exactly): preFFN norm →
	// gate/up GEMVs → SwiGLU → down GEMV → postFFN norm. Same buffers → garbage output, valid timing.
	dense := func() {
		_ = r.rms(r.x, L.g4preFFN, r.mq, r.mSc)
		_ = r.doG(L.g, r.mq, r.mSc, nullBias, r.gO, 0)
		_ = r.doG(L.u, r.mq, r.mSc, nullBias, r.uO, 0)
		_ = r.launch(r.fSw, onecfg(256, 256*4), Arg(r.gO), Arg(r.uO), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(r.inter)), gpu.ArgValue(r.act), Arg(r.dq), Arg(r.dSc), Arg(r.dScr))
		_ = r.doG(L.d, r.dq, r.dSc, nullBias, r.g4x1, 0)
		_ = r.normF32(r.g4x1, L.g4postFFN1)
	}
	// The router dispatch sequence (what the drain waits for): weightless norm → f32 router GEMV →
	// top-k route → per-expert-scale fold.
	router := func() {
		_ = r.launch(r.fRmsNW, onecfg(256, 256*4), Arg(r.x), Arg(r.g4rn), gpu.ArgValue(int32(r.hidden)), gpu.ArgValue(r.eps))
		_ = r.launch(r.fRouterF32, LaunchConfig{GridX: uint32(r.nE), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
			Arg(L.routerW), Arg(r.g4rn), gpu.ArgValue(int32(r.nE)), gpu.ArgValue(int32(r.hidden)), Arg(r.rLogits))
		_ = r.launch(r.fRoute, onecfg(1, 0), Arg(r.rLogits), Arg(L.routerB), Arg(r.rIdx), Arg(r.rWgt),
			gpu.ArgValue(int32(r.nE)), gpu.ArgValue(int32(r.topK)), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(1)), gpu.ArgValue(float32(1)), gpu.ArgValue(int32(1)), gpu.ArgValue(int32(1)))
		_ = r.launch(r.fScaleWgt, LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: uint32(r.topK), BlockY: 1, BlockZ: 1},
			Arg(r.rWgt), Arg(r.rIdx), Arg(L.perExpertScaleB), gpu.ArgValue(int32(r.topK)))
	}

	const N = 1000
	timeSeq := func(seq func()) float64 {
		var ms float64
		_ = r.do(func() error {
			for i := 0; i < 10; i++ {
				seq()
			}
			_ = r.stream.Sync()
			start := time.Now()
			for i := 0; i < N; i++ {
				seq()
			}
			_ = r.stream.Sync()
			ms = time.Since(start).Seconds() * 1e3 / float64(N)
			return nil
		})
		return ms
	}

	densePerLayer := timeSeq(dense)
	routerPerLayer := timeSeq(router)
	densePerTok := densePerLayer * float64(nMoE)
	routerPerTok := routerPerLayer * float64(nMoE)

	// The drain the probe isolated: 59→36 with stale idx removed BOTH the r.stream.Sync() drain AND
	// ~11 ms of miss-DMA, so the drain alone ≈ 12 ms/token. 1b can hide at most densePerTok of it.
	const drainMsPerTok = 12.0
	recoverable := densePerTok
	if recoverable > drainMsPerTok {
		recoverable = drainMsPerTok
	}
	t.Logf("Lever-1b BOUND (26B, %d MoE layers):", nMoE)
	t.Logf("  dense branch:  %.4f ms/layer → %.2f ms/token  (the COVER — 1b's ceiling)", densePerLayer, densePerTok)
	t.Logf("  router:        %.4f ms/layer → %.2f ms/token", routerPerLayer, routerPerTok)
	t.Logf("  readback drain ≈ %.1f ms/token (probe-derived)", drainMsPerTok)
	t.Logf("  MAX recoverable by 1b ≈ %.2f ms/token → 59 ms would become ~%.1f ms (~%.1f tok/s)",
		recoverable, 59-recoverable, 1000/(59-recoverable))
	if densePerTok < drainMsPerTok*0.5 {
		t.Logf("  VERDICT: THIN COVER — dense branch hides <half the drain; 1b not worth the primitive+reorder. "+
			"Prefer cross-layer pipelining (larger design) or dispatch reduction. densePerTok=%.2f < %.2f", densePerTok, drainMsPerTok*0.5)
	} else {
		t.Logf("  VERDICT: worth building — dense branch covers a meaningful share of the drain.")
	}
}
