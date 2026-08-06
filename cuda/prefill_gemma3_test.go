//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestPrefillLast_gemma3 extends the batched-prefill bit-identity gate to the SANDWICH-NORM family.
// After the qk-norm guard was lifted, gemma3's only remaining decline was its 4-norm sandwich: the
// attention and MLP sublayer outputs are RMSNorm'd BEFORE the residual add (postAttnNorm/postMLPNorm).
// batched prefill now does that per row (o-proj/down → temp, rmsnorm_f32_batched, residual add),
// mirroring segB's decode path. Real Gemma-3-4B at int4 (kEqV=0, no attn-softcap): asserts KV
// bit-identical (all layers × rows), last-token logits bit-identical, 64-token decode byte-identical.
// Heavy; gated. Green ⇒ gemma3 is a validated batched-prefill family.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags cuda -run TestPrefillLast_gemma3 -v
func TestPrefillLast_gemma3(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads Gemma-3-4B)")
	}
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1") // the gemma dense-resident path gate
	path := modelPath("gemma-3-4b-it-Q4_K_M.gguf")
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
	if !rf.sandwich {
		t.Fatal("Gemma-3 resident reports sandwich=false — this gate must exercise the sandwich path")
	}
	if !rf.prefillReady {
		t.Fatal("batched prefill kernels did not load — PrefillLast would decline")
	}
	_, nLayers, _, _, _, _, vocab := mc.Dims()

	const n = 56
	prompt := make([]int, n)
	var s uint32 = 20260805
	for i := range prompt {
		s = s*1664525 + 1013904223
		prompt[i] = int(s>>8) % (vocab - 1)
	}
	embs := make([][]float32, n)
	for i, tok := range prompt {
		embs[i] = append([]float32(nil), mc.EmbedResidentForTest(tok)...)
	}

	// --- sequential reference.
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
	seqStream := decodeGreedy(t, mc, rf, seqLogits, n, 64)

	// --- batched.
	batLogits, err := rf.PrefillLast(embs, 0)
	if err != nil {
		t.Fatalf("PrefillLast: %v — sandwich family must now batch, not decline", err)
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
		t.Logf("KV BIT-IDENTICAL across all %d layers × %d rows (K and V) — sandwich batched correctly", nLayers, n)
	} else {
		t.Fatalf("KV differs in %d elements — batched sandwich diverged from sequential", kvMism)
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
	t.Logf("GEMMA3 GATE GREEN: batched prefill ≡ sequential with sandwich norms (KV all layers×rows + logits + 64-token decode)")
}
