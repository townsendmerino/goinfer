//go:build cuda && goinfer_testhooks

package cuda

import (
	"testing"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// readKVForTest downloads the first `rows` positions of layer l's K and V caches on the executor
// thread (the context is current there). Test-only; reads the same device buffers the forward writes.
func (r *cudaResident) readKVForTest(l, rows int) (k, v []float32) {
	kvDim := r.layers[l].kvDim
	k = make([]float32, rows*kvDim)
	v = make([]float32, rows*kvDim)
	_ = r.do(func() error {
		_ = r.stream.Sync()
		_ = gpu.Download(r.kc[l], k)
		_ = gpu.Download(r.vc[l], v)
		return nil
	})
	return k, v
}

// TestPrefillLast_e2e is milestone 2's PRIMARY gate: batched PrefillLast must build a K/V cache and
// last-token logits BIT-IDENTICAL to the sequential ForwardNoLogits/Forward loop, and greedy decode
// from each cache must produce a BYTE-IDENTICAL token stream. It runs on mistral-tiny-window — a real
// dense model with sliding_window=16 — over a 56-token prompt (well past the window), so the two mask
// seams that only exist at M>1 are BOTH exercised end to end:
//   - CAUSALITY across all rows (the sequential path never let a row see a future key; the batched
//     path must reproduce that). The all-rows KV comparison is what proves it — a last-token-only
//     logit check cannot, since the last token legitimately attends to everything.
//   - the SLIDING WINDOW past position 16, where each row's winStart moves.
//
// The KV is compared for ALL layers and ALL rows (layer 0's K/V are mask-independent, so a match there
// alone would prove nothing); the decode gate catches any subtly-wrong cache the bit compare might miss.
func TestPrefillLast_e2e(t *testing.T) {
	const path = "../testdata/mistral-tiny-window"
	requireDeviceAndFixture(t, path)
	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cuda): %v", err)
	}
	defer mc.Close()
	rf, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatal("resident is not *cudaResident")
	}
	if !rf.prefillReady {
		t.Fatal("batched prefill kernels did not load — PrefillLast would decline")
	}
	win := mc.SlidingWindowResident()
	if win <= 0 {
		t.Fatalf("fixture declares no window (%d)", win)
	}
	_, nLayers, _, _, _, _, vocab := mc.Dims()

	// 56 tokens: past the window (16) so winStart moves, and not a round number.
	const n = 56
	prompt := make([]int, n)
	var s uint32 = 20260803
	for i := range prompt {
		s = s*1664525 + 1013904223
		prompt[i] = int(s>>8) % (vocab - 1)
	}
	embs := make([][]float32, n)
	for i, tok := range prompt {
		embs[i] = append([]float32(nil), mc.EmbedResidentForTest(tok)...)
	}

	// --- sequential reference: ForwardNoLogits for [:-1], Forward for the last → seqLogits + KV.
	for i := 0; i < n-1; i++ {
		if e := rf.ForwardNoLogits(embs[i], i); e != nil {
			t.Fatalf("seq ForwardNoLogits pos %d: %v", i, e)
		}
	}
	seqLogits, err := rf.Forward(embs[n-1], n-1)
	if err != nil {
		t.Fatalf("seq Forward last: %v", err)
	}
	seqLogits = append([]float32(nil), seqLogits...)
	seqK := make([][]float32, nLayers)
	seqV := make([][]float32, nLayers)
	for l := 0; l < nLayers; l++ {
		seqK[l], seqV[l] = rf.readKVForTest(l, n)
	}
	// greedy decode 56→56+63 from the sequential cache
	seqStream := decodeGreedy(t, mc, rf, seqLogits, n, 64)

	// --- batched: PrefillLast overwrites [0,n) → batLogits + KV.
	batLogits, err := rf.PrefillLast(embs, 0)
	if err != nil {
		t.Fatalf("PrefillLast: %v — it must handle a dense windowed model, not decline", err)
	}
	batK := make([][]float32, nLayers)
	batV := make([][]float32, nLayers)
	for l := 0; l < nLayers; l++ {
		batK[l], batV[l] = rf.readKVForTest(l, n)
	}
	batStream := decodeGreedy(t, mc, rf, batLogits, n, 64)

	// --- gate 1: KV bit-identical, ALL layers, ALL rows.
	kvMism := 0
	for l := 0; l < nLayers; l++ {
		for i := range seqK[l] {
			if seqK[l][i] != batK[l][i] {
				if kvMism < 3 {
					row := i / rf.layers[l].kvDim
					t.Errorf("K[l=%d row=%d off=%d]: batched %v != seq %v", l, row, i%rf.layers[l].kvDim, batK[l][i], seqK[l][i])
				}
				kvMism++
			}
			if seqV[l][i] != batV[l][i] {
				kvMism++
			}
		}
	}
	if kvMism == 0 {
		t.Logf("KV BIT-IDENTICAL across all %d layers × %d rows (K and V)", nLayers, n)
	} else {
		t.Fatalf("KV differs in %d elements — batched prefill's mask/cache diverged from sequential", kvMism)
	}

	// --- gate 2: last-token logits bit-identical.
	logMism := 0
	for i := range seqLogits {
		if seqLogits[i] != batLogits[i] {
			logMism++
		}
	}
	if logMism == 0 {
		t.Logf("last-token logits BIT-IDENTICAL (%d)", len(seqLogits))
	} else {
		t.Fatalf("last-token logits differ in %d/%d — final residual diverged", logMism, len(seqLogits))
	}

	// --- gate 3 (PRIMARY): greedy decode byte-identical.
	if len(seqStream) != len(batStream) {
		t.Fatalf("decode length mismatch: seq %d vs bat %d", len(seqStream), len(batStream))
	}
	for i := range seqStream {
		if seqStream[i] != batStream[i] {
			t.Fatalf("decode diverged at step %d: seq %d vs bat %d (streams: seq=%v bat=%v)", i, seqStream[i], batStream[i], seqStream, batStream)
		}
	}
	t.Logf("DECODE BYTE-IDENTICAL: 64 greedy tokens match (%v)", batStream)
	t.Logf("MILESTONE-2 GATE GREEN: batched prefill ≡ sequential (KV all layers×rows + logits + 64-token decode), window engaged")
}

// decodeGreedy greedily decodes `count` tokens starting from `logits` (the logits for position
// startPos-1's successor), continuing the resident's cache at startPos, startPos+1, ... It does not
// mutate positions < startPos, so calling it after each prefill exercises that prefill's own cache.
func decodeGreedy(t *testing.T, mc *decoder.Model, rf *cudaResident, logits []float32, startPos, count int) []int {
	t.Helper()
	out := make([]int, 0, count)
	cur := logits
	pos := startPos
	for i := 0; i < count; i++ {
		tok := argmaxF(cur)
		out = append(out, tok)
		l, err := rf.Forward(mc.EmbedResidentForTest(tok), pos)
		if err != nil {
			t.Fatalf("decode Forward pos %d: %v", pos, err)
		}
		cur = l
		pos++
	}
	return out
}
