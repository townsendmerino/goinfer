//go:build cuda && goinfer_testhooks

package cuda

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestPrefillNonUniform_bitIdentical gates the two guards batched prefill dropped so the Gemma-4
// families could reach it at all: PER-LAYER geometry and K=V.
//
// The fixture is the scaled dense Gemma 4 (hidden 1024, 12 layers, 5:1 sliding/full, head dim 256
// on the local layers and 512 on the global ones, K=V on the globals) — the only checkpoint here
// that exercises BOTH at once, and small enough to run in seconds. Before this change
// prefillStaticDecline refused it twice over: "non-uniform layer geometry at N" and "K=V layer at
// N". The refusals were correct for the code as it stood, because prefillCore hoisted layer 0's
// dims into every launch; each launch now binds its own layer's, and the M-sized scratch is
// allocated at the max across layers.
//
// The assertion is the batched pass against the SEQUENTIAL per-token path on the same resident, at
// the same positions: bit-identical last-token logits. Not "close" — every batched kernel is the
// M=1 kernel with an M dimension, so any difference is a striding or ordering bug, and a
// near-match is exactly what a wrong-but-plausible stride produces on a K=V layer whose V happens
// to correlate with K.
//
//	go test -tags 'cuda goinfer_testhooks' -run TestPrefillNonUniform -v ./cuda/
func TestPrefillNonUniform_bitIdentical(t *testing.T) {
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	const dir = "../testdata/gemma4-dense-scaled"
	if _, err := os.Stat(dir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no fixture (%s) — run scripts/pin_gemma4_dense_scaled.py", dir)
	}
	mc, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cuda): %v", err)
	}
	defer mc.Close()
	rf, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Skipf("not CUDA-resident (%T) — nothing to compare", mc.ResidentForwardForTest())
	}

	// Non-vacuity, both halves. This fixture must actually PRESENT the two shapes, or the test
	// passes by exercising neither — the same trap the MoE expert-major gate needed a run counter
	// for. Assert against the layer table, not against the fixture's documentation.
	L0 := &rf.layers[0]
	nonUniform, kEqV := 0, 0
	for l := range rf.layers {
		Ly := &rf.layers[l]
		if Ly.hd != L0.hd || Ly.nKV != L0.nKV || Ly.qDim != L0.qDim || Ly.kvDim != L0.kvDim || Ly.rhalf != L0.rhalf {
			nonUniform++
		}
		if Ly.kEqV {
			kEqV++
		}
	}
	fmt.Fprintf(os.Stderr, "[nonuniform] layers=%d non-uniform-vs-L0=%d kEqV=%d (L0 hd=%d kvDim=%d)\n",
		len(rf.layers), nonUniform, kEqV, L0.hd, L0.kvDim)
	if nonUniform == 0 {
		t.Fatal("fixture has UNIFORM geometry — this gate would pass without exercising the per-layer path")
	}
	if kEqV == 0 {
		t.Fatal("fixture has no K=V layer — this gate would pass without exercising the K=V path")
	}

	if batched, why := rf.PrefillPath(); !batched {
		t.Fatalf("batched prefill still declines this model: %s — the per-layer/K=V guards were "+
			"supposed to be what refused it", why)
	}

	const M = 96
	hidden := rf.hidden
	embs := make([][]float32, M)
	var s uint32 = 24681357
	for i := range embs {
		row := make([]float32, hidden)
		for j := range row {
			s = s*1664525 + 1013904223
			row[j] = float32(int32(s>>16)%2001-1000) / 10000
		}
		embs[i] = row
	}

	// Sequential reference: KV-only for every position but the last, whose logits are the seed —
	// exactly what residentPrefillSeed's fallback does.
	rf.Reset()
	for i := 0; i < M-1; i++ {
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
	bat, err := rf.PrefillLast(embs, 0)
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
			if d := bat[i] - seq[i]; d > worst || -d > worst {
				worst = max(d, -d)
			}
		}
	}
	fmt.Fprintf(os.Stderr, "[nonuniform] %d/%d logits differ (max |diff| %g)\n", diff, len(seq), worst)
	if diff != 0 {
		t.Errorf("batched prefill differs from the sequential path on a non-uniform K=V model: "+
			"%d/%d logits, first at %d (%v vs %v), max |diff| %g — per-layer striding or the K=V "+
			"v_norm is wrong", diff, len(seq), first, bat[first], seq[first], worst)
	}
}
