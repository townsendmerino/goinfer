//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestSplitKV_bitIdentical is Campaign A's correctness gate: the split-KV decode attention must be
// BYTE-IDENTICAL to the A1 attn_batched(M=1) it replaces. It runs the same prefill(2048)+24-token
// greedy decode twice on the real 1.5B — once with the split-KV path off (baseline = attn_batched),
// once on — and asserts the decoded token stream and the final logits match bit-for-bit. The two
// runs write identical KV (the decode attn choice doesn't touch rope/kv-store), so any divergence is
// the split-KV math, which the design (docs/task-decode-splitkv-attention.md) forbids. Long-ctx depth
// (2048) is where the split matters and where a reordered fold would first bite. Heavy; gated.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags cuda -run TestSplitKV_bitIdentical -v
func TestSplitKV_bitIdentical(t *testing.T) {
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
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	rf := mc.ResidentForwardForTest().(*cudaResident)
	if rf.skScores == (Pipeline{}) || rf.skVsum == (Pipeline{}) {
		t.Fatal("split-KV kernels did not load")
	}
	_, _, _, _, _, _, vocab := mc.Dims()

	const D = 2048
	emb := func(i int) []float32 { return mc.EmbedResidentForTest((i*2654435761 + 1) % (vocab - 1)) }
	prefill := make([][]float32, D)
	var s uint32 = 987654321
	for i := range prefill {
		s = s*1664525 + 1013904223
		prefill[i] = append([]float32(nil), emb(int(s>>8)%vocab)...)
	}

	// Force the split path whenever the flag is on, bypassing the per-geometry depth gate. Without
	// this the test would be VACUOUS for any geometry whose threshold exceeds D: both arms would run
	// attn_batched and compare it against itself. This test compares KERNELS; which kernel the gate
	// picks at a given depth is TestSplitKVGate_measuredGeometries' job.
	rf.skMinKeys = 0
	if rf.splitkvMin(0) != 0 {
		t.Fatal("override did not take: the split path would not be exercised")
	}
	run := func(useSplitKV bool) ([]float32, []int) {
		rf.splitkvAttn = useSplitKV
		lg, e := rf.PrefillLast(prefill, 0) // prefill unaffected by the decode-attn choice
		if e != nil {
			t.Fatalf("prefill (splitKV=%v): %v", useSplitKV, e)
		}
		cur := append([]float32(nil), lg...)
		stream := make([]int, 0, 24)
		for i := 0; i < 24; i++ {
			tok := argmaxF(cur)
			stream = append(stream, tok)
			l, e := rf.Forward(emb(tok), D+i) // decode attention = split-KV or attn_batched per the flag
			if e != nil {
				t.Fatalf("decode (splitKV=%v) step %d: %v", useSplitKV, i, e)
			}
			cur = append([]float32(nil), l...)
		}
		return cur, stream
	}

	baseLogits, baseStream := run(false) // A1 attn_batched(M=1) — the baseline
	skLogits, skStream := run(true)      // split-KV

	if len(baseStream) != len(skStream) {
		t.Fatalf("stream length mismatch: base %d vs split-KV %d", len(baseStream), len(skStream))
	}
	for i := range baseStream {
		if baseStream[i] != skStream[i] {
			t.Fatalf("decode diverged at step %d: attn_batched %d vs split-KV %d (base=%v sk=%v)",
				i, baseStream[i], skStream[i], baseStream, skStream)
		}
	}
	mism := 0
	for i := range baseLogits {
		if baseLogits[i] != skLogits[i] {
			mism++
		}
	}
	if mism != 0 {
		t.Fatalf("final logits differ in %d/%d — split-KV is NOT bit-identical to attn_batched", mism, len(baseLogits))
	}
	t.Logf("SPLIT-KV BIT-IDENTICAL: 24-token decode stream + final logits (%d) match attn_batched(M=1) at depth %d",
		len(baseLogits), D)
}

// TestSplitKV_bitIdentical_gemma3 broadens the gate to the paths qwen2.5 doesn't exercise: hd=256 AND
// sliding-window layers (winStart>0). Same A/B on real Gemma-3-4B at a depth past the window, so the
// split-KV per-layer winStart matches attn_batched's. Heavy; gated.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags cuda -run TestSplitKV_bitIdentical_gemma3 -v
func TestSplitKV_bitIdentical_gemma3(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads Gemma-3-4B)")
	}
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
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
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	rf := mc.ResidentForwardForTest().(*cudaResident)
	if rf.skScores == (Pipeline{}) {
		t.Fatal("split-KV kernels did not load")
	}
	_, _, _, _, _, _, vocab := mc.Dims()

	const D = 1536 // past a typical gemma3 sliding window (1024) so local layers have winStart>0, and >256
	emb := func(i int) []float32 { return mc.EmbedResidentForTest((i*2654435761 + 1) % (vocab - 1)) }
	prefill := make([][]float32, D)
	var s uint32 = 424242
	for i := range prefill {
		s = s*1664525 + 1013904223
		prefill[i] = append([]float32(nil), emb(int(s>>8)%vocab)...)
	}
	// Force the split path whenever the flag is on, bypassing the per-geometry depth gate. Without
	// this the test would be VACUOUS for any geometry whose threshold exceeds D: both arms would run
	// attn_batched and compare it against itself. This test compares KERNELS; which kernel the gate
	// picks at a given depth is TestSplitKVGate_measuredGeometries' job.
	rf.skMinKeys = 0
	if rf.splitkvMin(0) != 0 {
		t.Fatal("override did not take: the split path would not be exercised")
	}
	run := func(useSplitKV bool) ([]float32, []int) {
		rf.splitkvAttn = useSplitKV
		lg, e := rf.PrefillLast(prefill, 0)
		if e != nil {
			t.Fatalf("prefill (splitKV=%v): %v", useSplitKV, e)
		}
		cur := append([]float32(nil), lg...)
		stream := make([]int, 0, 16)
		for i := 0; i < 16; i++ {
			tok := argmaxF(cur)
			stream = append(stream, tok)
			l, e := rf.Forward(emb(tok), D+i)
			if e != nil {
				t.Fatalf("decode (splitKV=%v) step %d: %v", useSplitKV, i, e)
			}
			cur = append([]float32(nil), l...)
		}
		return cur, stream
	}
	baseLogits, baseStream := run(false)
	skLogits, skStream := run(true)
	for i := range baseStream {
		if baseStream[i] != skStream[i] {
			t.Fatalf("decode diverged at step %d: attn_batched %d vs split-KV %d", i, baseStream[i], skStream[i])
		}
	}
	mism := 0
	for i := range baseLogits {
		if baseLogits[i] != skLogits[i] {
			mism++
		}
	}
	if mism != 0 {
		t.Fatalf("final logits differ in %d/%d — split-KV not bit-identical on gemma3 (hd=256/windowed)", mism, len(baseLogits))
	}
	t.Logf("SPLIT-KV BIT-IDENTICAL (gemma3, hd=256, windowed): stream + logits (%d) match at depth %d", len(baseLogits), D)
}
