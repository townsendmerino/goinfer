//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestPrefillChunked_fastKernelsOnEveryChunk pins the fix for the chunk demotion (docs/measurements/prefill-chunk-demotion-2026-09-21.md): a prompt longer than one chunk (512 rows) is
// prefilled in several passes, and BEFORE the fix every pass but the last ran tailKVOnly, which forceExactKernels (tail != tailLastLogits) sent to the slow exact GEMM and attention — 7 of 8
// chunks at K=3900. It counts attn_fused / gemm_w4a8_mma launches directly instead of inferring them from timing:
//
//   - PrefillLast over a 1300-row prompt (3 chunks: 512+512+276): the fast attention runs on ALL of them (layers x 3) and the fast GEMM likewise (counts equal those of a single-pass prompt scaled by
//     chunk count is not asserted; what is asserted is that the chunked counts are >= 3x the per-pass layer count and equal the count with the chunk raised to one pass, per layer);
//
//   - HiddenLast (an embedding request, tailHiddenLast: its contract is decode-identical numerics) runs on NEITHER fast kernel in ANY chunk — the protection M-09/M-10/M-11 meant to keep.
//
//     GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestPrefillChunked_fastKernelsOnEveryChunk -v
func TestPrefillChunked_fastKernelsOnEveryChunk(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a 1.5B model)")
	}
	path := modelPath("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	rf, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok || (rf.bAttnFused128 == (Pipeline{}) && rf.bAttnFused64 == (Pipeline{})) || rf.bGemmMMA == (Pipeline{}) {
		t.Skip("resident path declined or the fast prefill kernels are not loaded")
	}
	_, _, _, _, _, _, vocab := m.Dims()
	const M = 1300
	embs := make([][]float32, M)
	var s uint32 = 12345
	for i := range embs {
		s = s*1664525 + 1013904223
		embs[i] = append([]float32(nil), m.EmbedResidentForTest(int(s>>8)%(vocab-1))...)
	}
	layers := len(rf.layers)

	count := func(chunk string, hidden bool) (attn, gemm int) {
		t.Setenv("GOINFER_PREFILL_CHUNK", chunk)
		rf.fastAttnLaunches, rf.fastGemmLaunches = 0, 0
		var e error
		if hidden {
			_, e = rf.HiddenLast(context.Background(), embs, 0)
		} else {
			_, e = rf.PrefillLast(context.Background(), embs, 0)
		}
		if e != nil {
			t.Fatalf("prefill (chunk=%s hidden=%v): %v", chunk, hidden, e)
		}
		return rf.fastAttnLaunches, rf.fastGemmLaunches
	}
	count("4096", false) // warm
	singleAttn, singleGemm := count("4096", false)
	chunkAttn, chunkGemm := count("512", false)
	if singleAttn != layers {
		t.Fatalf("single pass: attn_fused launched %d times, want one per layer (%d) — the fast path is not engaging at all", singleAttn, layers)
	}
	if chunkAttn != 3*layers {
		t.Errorf("CHUNKED prefill (1300 rows = 3 passes): attn_fused launched %d times, want %d (= 3 x %d layers): non-final chunks are still demoted to the exact kernel", chunkAttn, 3*layers, layers)
	}
	if chunkGemm != 3*singleGemm {
		t.Errorf("CHUNKED prefill: gemm_w4a8_mma launched %d times, want %d (3 x the single pass's %d)", chunkGemm, 3*singleGemm, singleGemm)
	}
	// The protection M-09/M-10/M-11 exist for: an embedding (HiddenLast) prefill stays exact in EVERY chunk.
	hAttn, hGemm := count("512", true)
	if hAttn != 0 || hGemm != 0 {
		t.Errorf("HiddenLast (tailHiddenLast) launched fast kernels (attn_fused %d, gemm_w4a8_mma %d): its decode-identical contract requires the exact path in every chunk", hAttn, hGemm)
	}
	t.Logf("layers=%d: single pass attn=%d gemm=%d | 3 chunks attn=%d gemm=%d | HiddenLast 3 chunks attn=%d gemm=%d", layers, singleAttn, singleGemm, chunkAttn, chunkGemm, hAttn, hGemm)
}
