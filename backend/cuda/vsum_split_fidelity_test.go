//go:build cuda && goinfer_testhooks

// Fidelity of the flash-decode V-sum SPIKE against the bit-identical V-sum, on one loaded model.
//
// The spike changes the reduction tree, so it is NOT bit-identical and TestSplitKV_bitIdentical
// cannot cover it. What matters instead is whether the new tree moves the OUTPUT DISTRIBUTION
// relative to the tree the engine already trusts — the same question docs/completed/task-prefill-gap.md §3
// asks of the fused prefill path, scored the same way (teacher-forced top-1 agreement + KL).
//
// Both arms run on the SAME resident with the same weights and the same prefilled cache; only
// r.skVsumSplit is toggled. That isolates the kernel and nothing else — no reload, no second
// process, no drift.
//
// TEACHER-FORCED, deliberately: the reference arm decodes greedily and its tokens are then FED to
// the spike arm, so step i of both arms sees identical history. Letting each arm follow its own
// argmax would make every step after a divergence incomparable, and would report one early flip as
// a total mismatch.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_SPLITKV_VSUM_SPLIT=4 GOINFER_CUDA_MODEL=<gguf> \
//	  go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestVsumSplitFidelity -v -timeout 30m
package cuda

import (
	"context"
	"math"
	"os"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"

	"github.com/townsendmerino/goinfer/decoder"
)

func TestVsumSplitFidelity(t *testing.T) {
	requireHeavyModel(t)
	path := os.Getenv("GOINFER_CUDA_MODEL")
	if path == "" {
		path = modelPath("qwen2.5-7b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no model at %s", path)
	}
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}
	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	// A decline is designed behaviour, not a failure — say so rather than panicking on the
	// assertion below (the same trap splitkv_bitident_test.go records).
	rfAny := mc.ResidentForwardForTest()
	if rfAny == nil {
		t.Skipf("could not evaluate: the resident path DECLINED (%s) — nothing here says anything about fidelity", mc.ResidentDecline())
	}
	rf := rfAny.(*cudaResident)
	if rf.skVsumSplit == 0 {
		t.Skip("set GOINFER_SPLITKV_VSUM_SPLIT=<n> so the spike pipelines are loaded")
	}
	S := rf.skVsumSplit
	if rf.skScores == (Pipeline{}) || rf.skVsum == (Pipeline{}) {
		t.Fatal("split-KV kernels did not load")
	}
	_, _, _, _, _, _, vocab := mc.Dims()

	const D = 8000 // the depth the spike was measured at; the V-sum's cost scales with it
	const N = 48   // decode steps scored
	emb := func(i int) []float32 { return mc.EmbedResidentForTest((i*2654435761 + 1) % (vocab - 1)) }
	prefill := make([][]float32, D)
	var s uint32 = 987654321
	for i := range prefill {
		s = s*1664525 + 1013904223
		prefill[i] = append([]float32(nil), emb(int(s>>8)%vocab)...)
	}

	// Force the split path regardless of the per-geometry depth gate, or both arms would run
	// attn_batched and the comparison would be vacuous.
	rf.skMinKeys = 0

	// ref: bit-identical V-sum, decoding greedily. Returns its own token stream.
	rf.skVsumSplit = 0
	refLogits, e := rf.PrefillLast(context.Background(), prefill, 0)
	if e != nil {
		t.Fatalf("prefill: %v", e)
	}
	refPer := make([][]float32, 0, N)
	refToks := make([]int, 0, N)
	cur := append([]float32(nil), refLogits...)
	for i := range N {
		tok := argmaxF(cur)
		refToks = append(refToks, tok)
		refPer = append(refPer, append([]float32(nil), cur...))
		l, err := rf.Forward(emb(tok), D+i)
		if err != nil {
			t.Fatalf("ref decode %d: %v", i, err)
		}
		cur = append([]float32(nil), l...)
	}

	// spike: same prefill, but TEACHER-FORCED on the reference's tokens.
	rf.skVsumSplit = S
	spkLogits, e := rf.PrefillLast(context.Background(), prefill, 0)
	if e != nil {
		t.Fatalf("prefill (spike): %v", e)
	}
	spkPer := make([][]float32, 0, N)
	cur = append([]float32(nil), spkLogits...)
	for i := range N {
		spkPer = append(spkPer, append([]float32(nil), cur...))
		l, err := rf.Forward(emb(refToks[i]), D+i) // reference's token, not the spike's own
		if err != nil {
			t.Fatalf("spike decode %d: %v", i, err)
		}
		cur = append([]float32(nil), l...)
	}

	agree, firstDiv := decoder.TeacherForcedTop1AgreementForTest(spkPer, refToks)
	var klSum, klMax float64
	for i := range refPer {
		kl := decoder.KLDivergenceForTest(refPer[i], spkPer[i])
		klSum += kl
		if kl > klMax {
			klMax = kl
		}
	}
	// Sanity: the two arms must not be the same kernel. If the spike were silently inactive every
	// number here would be perfect and the test would pass vacuously.
	same := true
	for i := range refLogits {
		if refLogits[i] != spkLogits[i] {
			same = false
			break
		}
	}
	t.Logf("S=%d depth=%d steps=%d: top-1 agreement %.4f (first divergence %d), KL mean %.3e max %.3e, prefill-logits identical=%v",
		S, D, N, agree, firstDiv, klSum/float64(len(refPer)), klMax, same)
	if math.IsNaN(klSum) {
		t.Fatal("KL is NaN — the spike produced non-finite logits")
	}
}
