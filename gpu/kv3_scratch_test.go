//go:build gpu && goinfer_testhooks

package gpu

import (
	"context"
	"fmt"
	"github.com/townsendmerino/goinfer/gpu/internal/kernels"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestKV3_prefillScratchBytes_boundsThePass (docs/measurements/kv3-webgpu-2026-10-13.md §2): the figure admission trusts is never under
// what a batched pass creates and at most 25% over it, on the tiny fixture and the real 1.5B, at 16 to 512 rows.
func TestKV3_prefillScratchBytes_boundsThePass(t *testing.T) {
	paths := []string{"../testdata/llama-attnfa-tiny"}
	if os.Getenv("GOINFER_HEAVY_TESTS") != "" {
		paths = append(paths, os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"))
	}
	ran := 0
	for _, path := range paths {
		m, err := decoder.Load(path, decoder.Options{Backend: "webgpu", Quant: "int4", ResidentContext: 2048})
		if err != nil {
			t.Skipf("no WebGPU device or load failed: %v", err)
		}
		rd, ok := m.ResidentForwardForTest().(*residentDecoder)
		if !ok || rd == nil {
			m.Close()
			t.Skipf("%s: no WebGPU resident", path)
		}
		_, _, _, _, _, _, vocab := m.Dims()
		for _, M := range []int{16, 64, 300, 512} {
			embs := make([][]float32, M)
			for i := range embs {
				embs[i] = m.EmbedResidentForTest((i*37 + 5) % (vocab - 1))
			}
			var seen int64
			kernels.PrefillBufSeenForTest = func(b int64) { seen += b }
			_, err := rd.PrefillLast(context.Background(), embs, 0)
			kernels.PrefillBufSeenForTest = nil
			if err != nil {
				t.Fatal(err)
			}
			got := rd.PrefillScratchBytes(M)
			fmt.Fprintf(os.Stderr, "[kv3] %-22s M=%-3d pass %d B (%.2f MiB), figure %d B (x%.3f)\n", path[max(0, len(path)-22):], M, seen, float64(seen)/(1<<20), got, float64(got)/float64(seen))
			if got < seen {
				t.Errorf("%s M=%d: the figure %d B is under the pass's %d B", path, M, got, seen)
			}
			if float64(got) > 1.25*float64(seen) {
				t.Errorf("%s M=%d: the figure %d B is more than 25%% over the pass's %d B", path, M, got, seen)
			}
			ran++
		}
		m.Close()
	}
	if ran == 0 {
		t.Fatal("no model ran")
	}
}

// TestKV3_residentReportsItsChunks pins the wiring the decoder reads (matched structurally, so a rename would silently put admission
// back on the whole-prompt bound).
func TestKV3_residentReportsItsChunks(t *testing.T) {
	m, err := decoder.Load("../testdata/llama-attnfa-tiny", decoder.Options{Backend: "webgpu", Quant: "int4", ResidentContext: 2048})
	if err != nil {
		t.Skipf("no WebGPU device or load failed: %v", err)
	}
	defer m.Close()
	rd, ok := m.ResidentForwardForTest().(*residentDecoder)
	if !ok || rd == nil {
		t.Skip("no WebGPU resident")
	}
	pc, ok := any(rd).(decoder.ResidentPrefillChunks)
	if !ok {
		t.Fatal("residentDecoder does not satisfy decoder.ResidentPrefillChunks")
	}
	if c := pc.PrefillChunkRows(); c != 512 || c%64 != 0 {
		t.Errorf("PrefillChunkRows %d, want 512 (a multiple of the block size 64)", c)
	}
	if b := pc.PrefillScratchBytes(512); b <= 0 || b >= pc.PrefillScratchBytes(20954) {
		t.Errorf("one chunk's scratch (%d B) must be positive and under a whole 20,954-token prompt's (%d B)", b, pc.PrefillScratchBytes(20954))
	}
}

// TestKV3_admissionPricesTheChunk (a request the old bound refused now answers) on the real 1.5B under WebGPU: with the host's available
// memory set between the whole-prompt price and the per-chunk price of a 20,954-token request, admission accepts it, and it refuses when
// the room is under the chunk's price.
func TestKV3_admissionPricesTheChunk(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads the 1.5B)")
	}
	m, err := decoder.Load(os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"), decoder.Options{Backend: "webgpu", Quant: "int4", ResidentContext: 2048})
	if err != nil {
		t.Skipf("no WebGPU device or load failed: %v", err)
	}
	defer m.Close()
	rd, ok := m.ResidentForwardForTest().(*residentDecoder)
	if !ok || rd == nil {
		t.Skip("no WebGPU resident")
	}
	const prompt, maxTok = 20954, 8
	whole := m.WholePromptScratchBytesForTest(prompt)
	chunk := rd.PrefillScratchBytes(rd.PrefillChunkRows()) + int64(prompt)*4*int64(m.ResidentEmbedLen())
	const frac = 0.7 // decoder.prefillAvailFraction
	t.Logf("whole-prompt price %.2f GB, per-chunk price %.3f GB", float64(whole)/(1<<30), float64(chunk)/(1<<30))
	if chunk*4 > whole {
		t.Fatalf("the per-chunk price %d is not far under the whole-prompt price %d: the gate proves nothing", chunk, whole)
	}
	restore := decoder.SetHostRAMAvailableForTest(int64(float64(whole+chunk) / 2 / frac))
	if err := m.AdmitPrefillMemory(prompt, maxTok, true); err != nil {
		t.Errorf("a request the per-chunk price fits was refused: %v", err)
	}
	restore()
	restore = decoder.SetHostRAMAvailableForTest(int64(float64(chunk) / 2 / frac))
	if err := m.AdmitPrefillMemory(prompt, maxTok, true); err == nil {
		t.Error("a request whose chunk does not fit was admitted")
	}
	restore()
}
