//go:build cuda && goinfer_testhooks

package cuda

import (
	"math"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestLoRADownGridBitIdenticalCUDA gates audit P-11's launch-shape change. lora_delta_down now runs
// one block per rank instead of one block looping over every rank. Each block does exactly the
// per-rank arithmetic the single block did, so through the real resident Forward the adapter's
// logits must match the old shape BIT FOR BIT, not merely to a tolerance.
func TestLoRADownGridBitIdenticalCUDA(t *testing.T) {
	const ckpt = "../testdata/llama-tiny"
	requireDeviceAndFixture(t, ckpt)
	dir := buildLlamaTinyLoRAFixtureCUDA(t)

	m, err := decoder.Load(ckpt, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load cuda: %v", err)
	}
	defer m.Close()
	rf := m.ResidentForwardForTest()
	cr, ok := rf.(*cudaResident)
	if !ok {
		t.Fatalf("resident is %T, not *cudaResident", rf)
	}
	if err := m.LoadAdapter("a", dir); err != nil {
		t.Fatalf("LoadAdapter: %v", err)
	}
	layers := m.ResidentAdapterLayersForTest("a")
	maxR := 0
	for i := range layers {
		for _, p := range projForCUDA(&layers[i]) {
			if p != nil && p.R > maxR {
				maxR = p.R
			}
		}
	}
	if maxR < 2 {
		t.Fatalf("fixture adapter has max rank %d: at rank 1 both launch shapes are one block, so "+
			"this test would compare the kernel with itself", maxR)
	}
	t.Cleanup(func() { loraDownSerial = false })
	shape := func(serial bool, want uint32) {
		t.Helper()
		loraDownSerial = serial
		if g := loraDownCfg(maxR).GridX; g != want {
			t.Fatalf("loraDownCfg(%d).GridX = %d with loraDownSerial=%v, want %d", maxR, g, serial, want)
		}
	}

	_, _, _, _, _, _, vocab := m.Dims()
	prompt := make([]int, 8)
	for i := range prompt {
		prompt[i] = (i*47 + 5) % vocab
	}
	run := func() []float32 {
		t.Helper()
		rf.Reset()
		var last []float32
		for i, tok := range prompt {
			lr, err := rf.Forward(m.EmbedResidentForTest(tok), i)
			if err != nil {
				t.Fatalf("resident forward[%d]: %v", i, err)
			}
			last = lr
		}
		return append([]float32(nil), last...)
	}

	base := run() // no adapter bound yet
	if err := cr.SetAdapter(layers); err != nil {
		t.Fatalf("SetAdapter: %v", err)
	}
	shape(false, uint32(maxR))
	grid := run()
	shape(true, 1)
	serial := run()
	shape(false, uint32(maxR))

	if _, maxAbs := cosF32(base, grid); maxAbs < 1e-3 {
		t.Fatalf("adapter bound vs unbound: maxAbs %.3g — the delta barely moves the logits, so "+
			"bit-identity between the two shapes would not show the down-project ran at all", maxAbs)
	}
	diff, first := 0, -1
	for i := range grid {
		if math.Float32bits(grid[i]) != math.Float32bits(serial[i]) {
			if first < 0 {
				first = i
			}
			diff++
		}
	}
	if diff != 0 {
		t.Fatalf("%d of %d logits differ between the per-rank grid and the single-block launch "+
			"(first at %d: grid %v, serial %v)", diff, len(grid), first, grid[first], serial[first])
	}
	t.Logf("max rank %d: %d logits bit-identical across GridX=%d and GridX=1", maxR, len(grid), maxR)
}
