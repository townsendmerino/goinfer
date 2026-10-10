//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"fmt"
	"math"
	"os"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestPrefillStartOffset asks whether a prompt's seed logits depend on WHERE its prefill starts: the question a
// KV-cache prefix reuse puts to the kernels. The reference is the whole prompt in one PrefillLast call. Each case
// computes the first r rows in one call, then the remaining rows in a SECOND call at startPos=r (what reuse does: the
// first r rows' KV is already resident) and compares the seed logits bit for bit.
//
//	A (default floor): a first call of r<512 rows runs on the EXACT kernels (the fast floor is judged on the pass), as the leftover slot's short prompt did in serve.
//	B (floor 0):       the first call runs the fast kernels too, so any difference is the start offset alone.
//
// B is start-offset-invariant (the fused kernels are), so what a prefix reuse changes is the PROVENANCE of the reused
// rows (exact-kernel versus fast-kernel numerics), not the start offset. A differs below the floor by kernel class, which
// shows the comparison can go red. Random embeddings: a numerics check, not a text one. Figures:
// docs/code-notes/cuda.md#TestPrefillStartOffset.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' -run TestPrefillStartOffset -v ./cuda/
func TestPrefillStartOffset(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a 1.5B model)")
	}
	path := modelPath("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cuda): %v", err)
	}
	defer mc.Close()
	rf, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatal("resident is not *cudaResident")
	}
	if batched, why := rf.PrefillPath(); !batched {
		t.Skipf("batched prefill declined for this fixture: %s", why)
	}

	const M = 941 // serve's section 21
	embs := make([][]float32, M)
	var s uint32 = 987654321
	for i := range embs {
		row := make([]float32, rf.hidden)
		for j := range row {
			s = s*1664525 + 1013904223
			row[j] = float32(int32(s>>16)%2001-1000) / 10000
		}
		embs[i] = row
	}
	ctx := context.Background()
	oneShot := func() []float32 {
		rf.Reset()
		lg, e := rf.PrefillLast(ctx, embs, 0)
		if e != nil {
			t.Fatalf("one-shot: %v", e)
		}
		return append([]float32(nil), lg...)
	}
	split := func(r int) []float32 {
		rf.Reset()
		if _, e := rf.PrefillLast(ctx, embs[:r], 0); e != nil {
			t.Fatalf("first call r=%d: %v", r, e)
		}
		lg, e := rf.PrefillLast(ctx, embs[r:], r)
		if e != nil {
			t.Fatalf("second call r=%d: %v", r, e)
		}
		return append([]float32(nil), lg...)
	}
	compare := func(ref, got []float32) (diff int, maxAbs float64) {
		if len(ref) != len(got) || len(ref) == 0 {
			t.Fatalf("logit rows: %d vs %d", len(ref), len(got))
		}
		for i := range ref {
			if ref[i] != got[i] {
				diff++
			}
			maxAbs = math.Max(maxAbs, math.Abs(float64(ref[i]-got[i])))
		}
		return
	}
	for _, variant := range []struct{ name, floor string }{{"A default floor (a short first call is EXACT)", ""}, {"B floor 0 (the first call is FAST too)", "0"}} {
		decoder.SetKnobEnvForTest(t, mc, "GOINFER_CUDA_FAST_PREFILL_FLOOR", variant.floor)
		ref := oneShot()
		ref2 := oneShot()
		if d, _ := compare(ref, ref2); d != 0 {
			t.Fatalf("%s: the one-shot reference is not deterministic (%d logits differ run to run): the comparison below means nothing", variant.name, d)
		}
		for _, r := range []int{3, 15, 16, 17, 33, 64, 100, 300, 512} {
			diff, maxAbs := compare(ref, split(r))
			fmt.Fprintf(os.Stderr, "[startoffset] %-46s r=%-4d %5d/%d logits differ, max|d| %.6f\n", variant.name, r, diff, len(ref), maxAbs)
			// What is asserted. B with a first call of at least 16 rows: both calls run the fast kernels, so the start offset alone is in play and the answer must be bit-identical (the fused kernels' invariance).
			// Below 16 rows no fast kernel can run (they need M>=16), and in A a first call under the floor is exact: those differ by KERNEL CLASS, which is what this test exists to show, so they are logged, not asserted.
			// A pass at r=512 is the floor itself, so A is fast on both calls there and must also be identical.
			if (variant.floor == "0" && r >= 16 || variant.floor == "" && r >= 512) && diff != 0 {
				t.Errorf("%s r=%d: %d/%d seed logits differ though both calls ran the fast kernels (max|d| %.6f): the fused prefill kernels are not start-offset-invariant", variant.name, r, diff, len(ref), maxAbs)
			}
		}
	}
}
