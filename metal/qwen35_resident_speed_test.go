//go:build darwin && goinfer_testhooks

package metal

import (
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestQwen35ResidentDecodeRateMetal is the Metal twin of cuda's TestQwen35ResidentDecodeRateCUDA.
// Read that file's header comment: on the tiny fixtures (hidden=64, 4 layers) per-dispatch
// overhead dominates completely — this is the "decode-rate number against CPU at the fixture size
// you can actually run" the porting task's "Done means" bar asks for, not a claim about the
// released 27B/35B/80B models, which need a real-width checkpoint to measure honestly.
func TestQwen35ResidentDecodeRateMetal(t *testing.T) {
	// Dense only: the MoE sibling's fixture has no committed model.safetensors on this box (needs
	// scripts/pin_qwen3_5_forward.py --moe) — qwen35DecodeRateMetal skips cleanly either way.
	fixtures := []string{"../decoder/testdata/qwen3_5-tiny", "../decoder/testdata/qwen3_5_moe-tiny"}
	if ck := os.Getenv("GOINFER_DNET_SPEED_METAL_CKPT"); ck != "" {
		fixtures = append(fixtures, ck)
	}
	for _, ckpt := range fixtures {
		t.Run(pathTail(ckpt), func(t *testing.T) { qwen35DecodeRateMetal(t, ckpt) })
	}
}

func pathTail(p string) string {
	for i := len(p) - 1; i >= 0; i-- {
		if p[i] == '/' {
			return p[i+1:]
		}
	}
	return p
}

func qwen35DecodeRateMetal(t *testing.T, ckpt string) {
	if _, err := os.Stat(ckpt + "/model.safetensors"); err != nil {
		t.Skipf("no fixture at %s: %v", ckpt, err)
	}
	const warm, iters = 8, 64
	quant := "int8int8"
	if v := os.Getenv("GOINFER_DNET_QUANT_METAL"); v != "" {
		quant = v
	}

	rate := func(step func(i int) error) float64 {
		for i := range warm { // warm-up: shader compile, first-touch allocation, clock ramp
			if err := step(i); err != nil {
				t.Fatal(err)
			}
		}
		start := time.Now()
		for i := range iters {
			if err := step(warm + i); err != nil {
				t.Fatal(err)
			}
		}
		return float64(iters) / time.Since(start).Seconds()
	}

	mRes, err := decoder.Load(ckpt, decoder.Options{Backend: "metal", Quant: quant})
	if err != nil {
		t.Fatalf("load metal: %v", err)
	}
	defer mRes.Close()
	rf := mRes.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("not resident — this would time the staged path and report it as residency: %s", mRes.ResidentDecline())
	}
	hidden, nLayers, _, _, hd, inter, vocab := mRes.Dims()
	rf.Reset()
	resRate := rate(func(i int) error {
		_, e := rf.Forward(mRes.EmbedResidentForTest((i*131+7)%vocab), i)
		return e
	})

	mCPU, err := decoder.Load(ckpt, decoder.Options{Backend: "cpu", Quant: quant})
	if err != nil {
		t.Fatalf("load cpu: %v", err)
	}
	defer mCPU.Close()
	cache := mCPU.NewCache(warm + iters + 1)
	cpuRate := rate(func(i int) error {
		_, e := mCPU.ForwardForTest((i*131+7)%vocab, cache)
		return e
	})

	t.Logf("%s @ %s (hidden=%d hd=%d inter=%d, %d layers, vocab=%d):",
		pathTail(ckpt), quant, hidden, hd, inter, nLayers, vocab)
	t.Logf("  resident %.1f tok/s (%.2f ms/tok, %.3f ms/layer) | cpu %.1f tok/s (%.2f ms/tok, %.3f ms/layer) | %.2fx",
		resRate, 1000/resRate, 1000/resRate/float64(nLayers),
		cpuRate, 1000/cpuRate, 1000/cpuRate/float64(nLayers), resRate/cpuRate)

	// Deliberately NOT a pass/fail threshold — same reasoning as the CUDA twin: the tiny fixture is
	// dispatch-bound and would fail any honest speed bar. The gate this test carries is that the
	// numbers get PRINTED next to the fixture size, so nobody quotes a tiny-fixture ratio as the
	// released models'.
	if resRate/cpuRate < 1 {
		t.Logf("  NOTE: resident is SLOWER here. Expected on a fixture this small — many dispatches " +
			"per DeltaNet layer against microseconds of arithmetic. Re-measure at real width " +
			"(GOINFER_DNET_SPEED_METAL_CKPT) before drawing any conclusion about a released model.")
	}
}
