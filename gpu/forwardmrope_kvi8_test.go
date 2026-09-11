//go:build gpu && goinfer_testhooks

package gpu

import (
	"math/rand"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestKVI8ScaleWrittenAtTruePosition gates audit-2026-09-10 C-09. The int8 rope-store kernel wrote
// each key's per-(position, KV-head) scale at the ROPE position instead of the true sequential one.
// So any decode with ropePos != pos (Qwen2.5-VL past an image) overwrote an earlier position's scale
// and left its own slot unset. The reader indexes scales by the true position.
//
// The check is the mechanism itself, not a logit cosine. On this model, int8 KV already sits at
// ~0.96 cosine against f32 on a random-token prompt, which is too noisy to hold a shift bar. Instead:
// a fresh model (so the scale buffer starts zeroed) decodes N steps with every rope angle shifted by
// delta, and every layer's scale slots [0, N) must then be written, with nothing written past N.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags 'gpu goinfer_testhooks' ./gpu/ -run TestKVI8ScaleWrittenAtTruePosition -v -timeout 20m
func TestKVI8ScaleWrittenAtTruePosition(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1")
	}
	if _, err := New(); err != nil {
		t.Skipf("no webgpu: %v", err)
	}
	home, _ := os.UserHomeDir()
	path := os.Getenv("GOINFER_QWEN15_CKPT")
	if path == "" {
		path = home + "/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s: %v", path, err)
	}

	const N = 24
	for _, delta := range []int{17, -13} {
		mc, err := decoder.Load(path, decoder.Options{Backend: "webgpu", Quant: "int8int8", KVPrecision: "i8"})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		rf := mc.ResidentForwardForTest()
		if rf == nil {
			mc.Close()
			t.Skip("model not resident-eligible on this build")
		}
		mrope, ok := rf.(decoder.ResidentMRoPE)
		if !ok {
			mc.Close()
			t.Fatal("residentDecoder does not satisfy decoder.ResidentMRoPE — interface wiring broken")
		}
		_, _, _, _, _, _, vocab := mc.Dims()
		rng := rand.New(rand.NewSource(11))
		for i := range N {
			if _, err := mrope.ForwardMRoPE(mc.EmbedResidentForTest(rng.Intn(vocab-1)), i, i+delta); err != nil {
				mc.Close()
				t.Fatalf("delta %d step %d: %v", delta, i, err)
			}
		}
		layers, missing, stray := 0, 0, 0
		for l := 0; ; l++ {
			sc, nKV, err := KScalesForTest(rf, l)
			if err != nil {
				if l == 0 {
					mc.Close()
					t.Fatalf("layer 0: %v — the int8 KV cache is not in play, so this test gates nothing", err)
				}
				break
			}
			layers++
			nPos := len(sc) / nKV
			for p := range min(nPos, N+32) {
				for h := range nKV {
					written := sc[p*nKV+h] != 0
					if p < N && !written {
						missing++
					}
					if p >= N && written {
						stray++
					}
				}
			}
		}
		mc.Close()
		t.Logf("delta %+d: %d layers — %d unwritten scale slots in [0,%d), %d stray writes past %d", delta, layers, missing, N, stray, N)
		if missing > 0 || stray > 0 {
			t.Errorf("delta %+d: %d scale slots unwritten and %d written at the wrong position — the int8 K "+
				"scale is indexed by the rope position, not the true one (audit C-09)", delta, missing, stray)
		}
	}
}
