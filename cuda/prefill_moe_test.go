//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestPrefillMoE_bitIdentical pins that a MoE layer's FFN, which runs row by row off the BATCHED residual
// (xB.At(m*hidden*4)), changes only speed: the attention half is batched and the routed experts keep
// decode's exact per-token sequence.
//
// The assertion is equality against the sequential per-token path on the same resident at the same
// positions: every logit, not a tolerance. Routing is a discrete argmax over router logits, so a tiny
// numerical difference runs a DIFFERENT EXPERT and the row is unrelated; a near-match would mean the
// routing agreed by luck on this input.
//
// The fixtures cover different halves: gemma4-moe-scaled carries the real 26B FFN shapes (hidden 2816,
// moe_inter 704); gemma4-moe-kv-tiny is meant to put K=V and MoE in one model (M26's combination) but
// declines residency (see TestPrefillMoE_real26B), so its subtest skips.
//
//	go test -tags 'cuda goinfer_testhooks' -run TestPrefillMoE -v ./cuda/
func TestPrefillMoE_bitIdentical(t *testing.T) {
	for _, fx := range []string{"../testdata/gemma4-moe-kv-tiny", "../testdata/gemma4-moe-scaled"} {
		t.Run(fx[len("../testdata/"):], func(t *testing.T) { prefillMoEParity(t, fx, false) })
	}
}

// TestPrefillMoE_real26B is the same assertion against M26, the Gemma-4-26B-A4B kind-4 .giw bundle, loaded
// with -moe-cache-experts as scripts/bench_peer.py launches it.
//
// It is NOT redundant with the fixture gate above. No fixture here carries K=V and MoE in the SAME model:
// gemma4-moe-kv-tiny is the one that would, and it declines residency ("moeInter(16) and hidden(64) both
// multiples of 32"), so gemma4-moe-scaled covers MoE with uniform non-K=V geometry and gemma4-dense-scaled
// covers K=V without MoE. M26 is the only checkpoint that exercises both at once, and the only one that
// exercises the C′ routed expert DMA inside the per-row loop (the fixtures run cacheExperts=false).
//
// Heavy: the load pins host memory for the expert stack and takes minutes.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' -run TestPrefillMoE_real26B -v -timeout 30m ./cuda/
func TestPrefillMoE_real26B(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads the 26B, ~2m11s)")
	}
	dir := modelPath("gemma4-26b-int4.giw")
	prefillMoEParity(t, dir, true)
}

func prefillMoEParity(t *testing.T, dir string, real26B bool) {
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	if _, err := os.Stat(dir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no fixture (%s)", dir)
	}
	opts := decoder.Options{Backend: "cuda", Quant: "int4"}
	if real26B {
		// The .giw bundle is already quantized (-quant does not apply) and the expert stack exceeds
		// this card, so it needs the C′ cache — the shipped configuration, not a test-only one.
		opts = decoder.Options{Backend: "cuda", MoECacheExperts: true, ResidentContext: 8192}
	}
	mc, err := decoder.Load(dir, opts)
	if err != nil {
		t.Fatalf("load (cuda): %v", err)
	}
	defer mc.Close()
	rf, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Skipf("not CUDA-resident (%T)", mc.ResidentForwardForTest())
	}

	// Non-vacuity: the fixture must HAVE MoE layers, or the test passes on a dense model by exercising a
	// branch it is not about.
	moeLayers := 0
	for l := range rf.layers {
		if rf.layers[l].isMoE || rf.layers[l].g4moe {
			moeLayers++
		}
	}
	fmt.Fprintf(os.Stderr, "[moe-prefill] %s: layers=%d moe=%d (r.moe=%v r.gemma4Moe=%v cacheExperts=%v)\n",
		dir, len(rf.layers), moeLayers, rf.moe, rf.gemma4Moe, rf.cacheExperts)
	if moeLayers == 0 {
		t.Fatal("fixture has no MoE layer — this gate would pass without exercising the per-row FFN")
	}
	if batched, why := rf.PrefillPath(); !batched {
		t.Fatalf("batched prefill still declines this MoE model: %s", why)
	}

	const M = 48
	hidden := rf.hidden
	embs := make([][]float32, M)
	var s uint32 = 1357924680
	for i := range embs {
		row := make([]float32, hidden)
		for j := range row {
			s = s*1664525 + 1013904223
			row[j] = float32(int32(s>>16)%2001-1000) / 10000
		}
		embs[i] = row
	}

	rf.Reset()
	for i := range M - 1 {
		if e := rf.ForwardNoLogits(embs[i], i); e != nil {
			t.Fatalf("sequential ForwardNoLogits(%d): %v", i, e)
		}
	}
	seqLogits, err := rf.Forward(embs[M-1], M-1)
	if err != nil {
		t.Fatalf("sequential Forward(%d): %v", M-1, err)
	}
	seq := append([]float32(nil), seqLogits...)

	rf.Reset()
	bat, err := rf.PrefillLast(context.Background(), embs, 0)
	if err != nil {
		t.Fatalf("PrefillLast: %v", err)
	}
	if len(bat) != len(seq) {
		t.Fatalf("logits len %d, want %d", len(bat), len(seq))
	}
	diff, first := 0, -1
	var worst float32
	for i := range seq {
		if bat[i] != seq[i] {
			diff++
			if first < 0 {
				first = i
			}
			if d := bat[i] - seq[i]; max(d, -d) > worst {
				worst = max(d, -d)
			}
		}
	}
	fmt.Fprintf(os.Stderr, "[moe-prefill] %s: %d/%d logits differ (max |diff| %g)\n", dir, diff, len(seq), worst)
	if diff != 0 {
		t.Errorf("batched MoE prefill differs from the sequential path: %d/%d logits, first at %d "+
			"(%v vs %v), max |diff| %g — on a MoE model a difference this is not a rounding "+
			"question: the router picks experts by argmax, so a diverged row ran different experts",
			diff, len(seq), first, bat[first], seq[first], worst)
	}
}
