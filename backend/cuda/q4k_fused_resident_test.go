//go:build cuda && goinfer_testhooks

package cuda

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestQ4K_fusedG32ResidentBitIdentical: on a model the fused per-32 kernels serve (qwen2.5-coder-1.5b,
// H=1536 ≤ fusedG32MaxHidden), q4k resident logits with fused_rms_qkv_g32 / fused_rms_gu_g32 equal the
// unfused per-32 chain's (GOINFER_CUDA_NO_FUSE) EXACTLY over 40 positions, and each load really took the
// path it names. GOINFER_QWEN15_GGUF overrides the default ~/models path; skips without the model.
func TestQ4K_fusedG32ResidentBitIdentical(t *testing.T) {
	path := os.Getenv("GOINFER_QWEN15_GGUF")
	if path == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			t.Skipf("no home directory: %v", err)
		}
		path = filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no model at %s", path)
	}
	run := func(noFuse bool) [][]float32 {
		opts := decoder.Options{Backend: "cuda", Quant: "q4k", ResidentContext: 256}
		if noFuse {
			opts.Knobs = &decoder.Knobs{"GOINFER_CUDA_NO_FUSE": "1"}
		}
		m, err := decoder.Load(path, opts)
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		rf := m.ResidentForwardForTest()
		if rf == nil {
			t.Fatalf("resident did not engage: %s", m.ResidentDecline())
		}
		r := rf.(*cudaResident)
		if r.fuseG32 == noFuse {
			t.Fatalf("noFuse=%v but fuseG32=%v", noFuse, r.fuseG32)
		}
		var out [][]float32
		for p := range 40 {
			lg, err := rf.Forward(m.EmbedResidentForTest(100+37*p%5000), p)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, append([]float32(nil), lg...))
		}
		return out
	}
	fused, plain := run(false), run(true)
	for p := range fused {
		for i := range fused[p] {
			if math.Float32bits(fused[p][i]) != math.Float32bits(plain[p][i]) {
				t.Fatalf("pos %d logit %d: fused %v, unfused %v — not bit-identical", p, i, fused[p][i], plain[p][i])
			}
		}
	}
	t.Logf("40 positions × %d logits bit-identical, fused vs unfused", len(fused[0]))
}
