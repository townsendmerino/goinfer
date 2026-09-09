//go:build cuda && goinfer_testhooks

package cuda

import (
	"math/rand"
	"os"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestForwardMRoPE_constantShiftInvariance is ForwardMRoPE's (decoder.ResidentMRoPE) FIRST real
// correctness test — before this, the new interface (and the rope_kv kernel's ropePos parameter
// it depends on) had never been exercised outside its own compile.
//
// It does NOT try to reproduce Qwen2.5-VL's specific (non-constant) mropeDelta end to end — that
// would need a real Qwen2.5-VL checkpoint, a real image, and an independent CPU oracle for the
// exact rotation the m-RoPE formula produces, none of which isolates whether ForwardMRoPE's
// ropePos plumbing itself is correct. Instead it tests a property of RoPE that is TRUE
// UNCONDITIONALLY, independent of any family's usage: attention(q, k) depends only on the
// RELATIVE rotation angle between query and key (that is the entire point of RoPE), so a decode
// run where EVERY step's rope angle is shifted by the SAME constant delta must produce IDENTICAL
// output to the same run with delta=0 — the storage/attention position (pos) is unchanged in
// both arms, only the angle (ropePos = pos+delta) differs. This directly and rigorously exercises
// the new rope_kv ropePos kernel argument at every decode step, with a derivable expected answer
// that needs no external oracle, no CPU reference, and no new device-buffer introspection.
//
// Two arms, same model instance, Reset between (mirrors uploadkv_parity_test.go's pattern):
//
//	ARM A (ordinary): Forward(emb_i, i) for i in [0,N) — delta=0.
//	ARM B (shifted):  ForwardMRoPE(emb_i, i, i+delta) for the SAME embeddings — delta != 0.
//
// If ForwardMRoPE's ropePos wiring is correct, the two arms' logits must match to a tight cosine
// bar at every step (not bit-identical: shifted cos/sin arguments are numerically DIFFERENT
// values even though mathematically equivalent, so this is a numeric-tolerance check, same
// discipline as every other resident-vs-CPU parity gate in this repo).
//
//	GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestForwardMRoPE_constantShiftInvariance -v -timeout 20m
func TestForwardMRoPE_constantShiftInvariance(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1")
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

	for _, delta := range []int{17, -13} { // one positive, one negative — Qwen's real mropeDelta is typically negative
		t.Run(map[bool]string{true: "positive", false: "negative"}[delta > 0], func(t *testing.T) {
			mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer mc.Close()
			rf := mc.ResidentForwardForTest()
			if rf == nil {
				t.Skip("model not resident-eligible on this build")
			}
			mrope, ok := rf.(decoder.ResidentMRoPE)
			if !ok {
				t.Fatal("cudaResident does not satisfy decoder.ResidentMRoPE — interface wiring broken")
			}
			_, _, _, _, _, _, vocab := mc.Dims()

			const N = 24
			rng := rand.New(rand.NewSource(11))
			embs := make([][]float32, N)
			for i := range embs {
				embs[i] = mc.EmbedResidentForTest(rng.Intn(vocab - 1))
			}

			// ARM A: ordinary, delta=0.
			var want []float32
			for i, e := range embs {
				l, err := rf.Forward(e, i)
				if err != nil {
					t.Fatalf("ordinary Forward step %d: %v", i, err)
				}
				want = l
			}

			rf.Reset()

			// ARM B: every step's rope angle shifted by the same constant delta.
			var got []float32
			for i, e := range embs {
				l, err := mrope.ForwardMRoPE(e, i, i+delta)
				if err != nil {
					t.Fatalf("ForwardMRoPE step %d (ropePos=%d): %v", i, i+delta, err)
				}
				got = l
			}

			cos := cosine(want, got)
			t.Logf("delta=%d: cosine(ordinary, constant-shifted) = %.8f", delta, cos)
			if cos < 0.9999 {
				t.Errorf("delta=%d: cosine = %.8f, want ~1.0 — RoPE's relative-angle invariance is violated, ForwardMRoPE's ropePos wiring is likely wrong", delta, cos)
			}
		})
	}
}

// TestForwardMRoPE_equalArgsMatchesForward checks the interface's own stated contract
// (decoder/residency.go: "ForwardMRoPE(emb, pos, pos) must equal Forward(emb, pos) exactly").
func TestForwardMRoPE_equalArgsMatchesForward(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1")
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
	rf := mc.ResidentForwardForTest()
	if rf == nil {
		t.Skip("model not resident-eligible on this build")
	}
	mrope, ok := rf.(decoder.ResidentMRoPE)
	if !ok {
		t.Fatal("cudaResident does not satisfy decoder.ResidentMRoPE")
	}
	_, _, _, _, _, _, vocab := mc.Dims()
	rng := rand.New(rand.NewSource(3))
	const N = 8
	embs := make([][]float32, N)
	for i := range embs {
		embs[i] = mc.EmbedResidentForTest(rng.Intn(vocab - 1))
	}

	var want []float32
	for i, e := range embs {
		l, err := rf.Forward(e, i)
		if err != nil {
			t.Fatalf("Forward step %d: %v", i, err)
		}
		want = l
	}
	rf.Reset()
	var got []float32
	for i, e := range embs {
		l, err := mrope.ForwardMRoPE(e, i, i) // ropePos == pos
		if err != nil {
			t.Fatalf("ForwardMRoPE step %d: %v", i, err)
		}
		got = l
	}
	if len(want) != len(got) {
		t.Fatalf("length %d vs %d", len(want), len(got))
	}
	for i := range want {
		if want[i] != got[i] {
			t.Fatalf("logit %d: Forward=%v ForwardMRoPE(pos,pos)=%v — must be bit-identical, same kernel args", i, want[i], got[i])
		}
	}
}
