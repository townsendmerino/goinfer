//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"os"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestPrefillLast_qwen3 extends milestone 2's bit-identity gate to a QK-NORM family. Qwen3 is the
// dense batched lane plus one thing the old guard declined: a per-head Q/K RMSNorm before RoPE.
// batched prefill now applies it via qk_norm_batched (M=1 qk_norm + an M dimension), so this asserts
// the same three gates as TestPrefillLast_e2e — KV bit-identical (all layers × rows), last-token
// logits bit-identical, and 64-token greedy decode byte-identical — on the real Qwen3-1.7B at int4.
// Heavy (loads a 1.7B); gated. If it stays green, qwen3 is a validated batched-prefill family.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags cuda -run TestPrefillLast_qwen3 -v
func TestPrefillLast_qwen3(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads Qwen3-1.7B)")
	}
	path := modelPath("qwen3-1.7b-q8_0.gguf")
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
	if !rf.qkNorm {
		t.Fatal("Qwen3 resident reports qkNorm=false — this gate must exercise the qk-norm path")
	}
	if !rf.prefillReady {
		t.Fatal("batched prefill kernels did not load — PrefillLast would decline")
	}
	_, nLayers, _, _, _, _, vocab := mc.Dims()

	// 56 tokens: not a round number; enough to compound any qk-norm divergence across the stack.
	const n = 56
	prompt := make([]int, n)
	var s uint32 = 20260804
	for i := range prompt {
		s = s*1664525 + 1013904223
		prompt[i] = int(s>>8) % (vocab - 1)
	}
	embs := make([][]float32, n)
	for i, tok := range prompt {
		embs[i] = append([]float32(nil), mc.EmbedResidentForTest(tok)...)
	}

	// --- sequential reference: ForwardNoLogits for [:-1], Forward for the last → seqLogits + KV.
	for i := range n - 1 {
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
	for l := range nLayers {
		seqK[l], seqV[l] = rf.readKVForTest(l, n)
	}
	seqStream := decodeGreedy(t, mc, rf, seqLogits, n, 64)

	// --- batched: PrefillLast overwrites [0,n) → batLogits + KV.
	batLogits, err := rf.PrefillLast(context.Background(), embs, 0)
	if err != nil {
		t.Fatalf("PrefillLast: %v — qk-norm family must now batch, not decline", err)
	}
	batK := make([][]float32, nLayers)
	batV := make([][]float32, nLayers)
	for l := range nLayers {
		batK[l], batV[l] = rf.readKVForTest(l, n)
	}
	batStream := decodeGreedy(t, mc, rf, batLogits, n, 64)

	// --- gate 1: KV bit-identical, ALL layers, ALL rows.
	kvMism := 0
	for l := range nLayers {
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
		t.Logf("KV BIT-IDENTICAL across all %d layers × %d rows (K and V) — qk-norm batched correctly", nLayers, n)
	} else {
		t.Fatalf("KV differs in %d elements — batched qk-norm diverged from sequential", kvMism)
	}

	// --- gate 2: last-token logits bit-identical.
	logMism := 0
	for i := range seqLogits {
		if seqLogits[i] != batLogits[i] {
			logMism++
		}
	}
	if logMism != 0 {
		t.Fatalf("last-token logits differ in %d/%d — final residual diverged", logMism, len(seqLogits))
	}
	t.Logf("last-token logits BIT-IDENTICAL (%d)", len(seqLogits))

	// --- gate 3 (PRIMARY): greedy decode byte-identical.
	if len(seqStream) != len(batStream) {
		t.Fatalf("decode length mismatch: seq %d vs bat %d", len(seqStream), len(batStream))
	}
	for i := range seqStream {
		if seqStream[i] != batStream[i] {
			t.Fatalf("decode diverged at step %d: seq %d vs bat %d", i, seqStream[i], batStream[i])
		}
	}
	t.Logf("DECODE BYTE-IDENTICAL: 64 greedy tokens match")
	t.Logf("QWEN3 GATE GREEN: batched prefill ≡ sequential with qk-norm (KV all layers×rows + logits + 64-token decode)")
}
