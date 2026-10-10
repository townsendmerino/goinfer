//go:build gpu && goinfer_testhooks

package gpu

import (
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// Decode rate, resident vs CPU, for the Gated-DeltaNet hybrid. Residency is a capability and
// speed is a separate claim ("admitted but not faster" is a real outcome), so this attaches a
// measurement to it.
//
// READ THE FIXTURE SIZE BEFORE READING THE RATIO. On the tiny fixtures (hidden 64, 4 layers)
// per-dispatch overhead dominates completely (about 20 GPU dispatches per DeltaNet layer against
// a few microseconds of arithmetic), so a ratio below 1 there says nothing about the 27B. The
// number that matters comes from a real-WIDTH fixture (GOINFER_DNET_SPEED_CKPT), where the
// arithmetic is large enough to pay for the dispatches. Both are reported rather than only the
// flattering one.
func TestQwen35ResidentDecodeRate(t *testing.T) {
	if os.Getenv("GOINFER_DNET_SPEED") == "" {
		t.Skip("qwen3.5 resident decode rate (set GOINFER_DNET_SPEED=1)")
	}
	if _, err := New(); err != nil {
		t.Skipf("no webgpu: %v", err)
	}
	fixtures := []string{"../decoder/testdata/qwen3_5-tiny", "../decoder/testdata/qwen3_5_moe-tiny"}
	if ck := os.Getenv("GOINFER_DNET_SPEED_CKPT"); ck != "" {
		fixtures = append(fixtures, ck)
	}
	for _, ckpt := range fixtures {
		t.Run(pathTail(ckpt), func(t *testing.T) { qwen35DecodeRate(t, ckpt) })
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

func qwen35DecodeRate(t *testing.T, ckpt string) {
	if _, err := os.Stat(ckpt + "/model.safetensors"); err != nil {
		t.Skipf("no fixture at %s: %v", ckpt, err)
	}
	const warm, iters = 8, 64
	quant := "int8int8"
	if v := os.Getenv("GOINFER_DNET_QUANT"); v != "" {
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

	mRes, err := decoder.Load(ckpt, decoder.Options{Backend: "webgpu", Quant: quant})
	if err != nil {
		t.Fatalf("load webgpu: %v", err)
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
	// ms/LAYER is the transferable quantity and the ratio is the robust one; the absolute per-token
	// figure is not, because this fixture omits the released vocab (248320, whose LM head is ~5% of
	// the real per-token MACs) and runs a short context. Extrapolating ms/layer × real layer count
	// overstates the CPU side, so quote the RATIO and treat any absolute extrapolation as indicative
	// only.

	// Deliberately NOT a pass/fail threshold. The tiny fixtures are dispatch-bound and would
	// fail any honest speed bar; asserting one here would either be vacuous or would pressure a
	// future reader into loosening it. The gate this test carries is that the numbers get
	// PRINTED next to the fixture size, so nobody quotes a tiny-fixture ratio as the 27B's.
	if resRate/cpuRate < 1 {
		t.Logf("  NOTE: resident is SLOWER here. Expected on a fixture this small — ~20 dispatches " +
			"per DeltaNet layer against microseconds of arithmetic. Re-measure at real width " +
			"(GOINFER_DNET_SPEED_CKPT) before drawing any conclusion about the released model.")
	}
}
