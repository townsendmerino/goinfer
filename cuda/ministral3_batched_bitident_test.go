//go:build cuda && goinfer_testhooks

package cuda

import (
	"math"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestMinistral3BatchedPrefillMatchesSequentialCUDA gates audit-2026-09-10 G-11's numeric half.
// Ministral 3 scales each query row after RoPE by 1 + beta·log1p(floor(pos/origMaxPos)). Decode
// computes that scalar on the host in float64. The batched prefill recomputed it per row, in f32,
// on the device, with a bare multiply-add. So a batched pass's Q rows were not bit-identical to the
// sequential path's, and spec-verify rows can flip an argmax on a near-tie. Rows below origMaxPos
// have scale 1 and are the control: if they differ, something else broke bit-identity, and this
// test cannot isolate the attention temperature.
func TestMinistral3BatchedPrefillMatchesSequentialCUDA(t *testing.T) {
	const ckpt = "../testdata/ministral3-tiny"
	requireDeviceAndFixture(t, ckpt)
	m, err := decoder.Load(ckpt, decoder.Options{Backend: "cuda", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load cuda: %v", err)
	}
	defer m.Close()
	rf := m.ResidentForwardForTest()
	cr, ok := rf.(*cudaResident)
	if !ok {
		t.Fatalf("%s did not go CUDA-resident (%T) — decline: %s", ckpt, rf, m.ResidentDecline())
	}
	if cr.attnTempBeta == 0 || cr.attnTempOrigMaxPos <= 0 {
		t.Fatalf("fixture has no attention temperature (beta %v, orig %v) — this test gates nothing", cr.attnTempBeta, cr.attnTempOrigMaxPos)
	}
	orig := int(cr.attnTempOrigMaxPos)
	const N = 40
	_, _, _, _, _, _, vocab := m.Dims()
	embs := make([][]float32, N)
	for i := range embs {
		embs[i] = m.EmbedResidentForTest((i*37 + 3) % vocab)
	}
	rf.Reset()
	seq := make([][]float32, N)
	for i, e := range embs {
		l, err := rf.Forward(e, i)
		if err != nil {
			t.Fatalf("resident forward[%d]: %v", i, err)
		}
		seq[i] = append([]float32(nil), l...)
	}
	rf.Reset()
	bat, err := cr.PrefillLastN(embs, 0)
	if err != nil {
		t.Fatalf("PrefillLastN: %v — this gate needs Ministral 3 on the batched path", err)
	}
	if len(bat) != N {
		t.Fatalf("PrefillLastN returned %d rows, want %d", len(bat), N)
	}
	same := func(a, b []float32) bool {
		for j := range a {
			if math.Float32bits(a[j]) != math.Float32bits(b[j]) {
				return false
			}
		}
		return true
	}
	ctrlDiff, tempDiff := 0, 0
	for i := range N {
		if !same(seq[i], bat[i]) {
			if i < orig {
				ctrlDiff++
			} else {
				tempDiff++
			}
		}
	}
	t.Logf("origMaxPos %d, beta %.4g: control rows [0,%d) differing %d; temperature rows [%d,%d) differing %d",
		orig, cr.attnTempBeta, orig, ctrlDiff, orig, N, tempDiff)
	if ctrlDiff > 0 {
		t.Fatalf("%d control rows (scale 1) differ — the batched path is not bit-identical here for another "+
			"reason, so this test cannot isolate the attention temperature", ctrlDiff)
	}
	if tempDiff > 0 {
		t.Errorf("%d of %d attention-temperature rows differ from sequential decode bit for bit — the batched "+
			"per-row query scale is not the host's float64 value (audit G-11)", tempDiff, N-orig)
	}
}
