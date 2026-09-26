package decoder

import (
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// q4kTestMat builds rows×cols of valid synthetic Q4_K super-blocks (normal f16 d/dmin, random scale,
// min and code bytes) wrapped as a native Q4_K WeightMat.
func q4kTestMat(t *testing.T, r *rand.Rand, rows, cols int) linalg.WeightMat {
	t.Helper()
	raw := make([]byte, rows*linalg.Q4KRowBytes(cols))
	for b := 0; b < len(raw); b += 144 {
		for _, off := range []int{0, 2} {
			h := uint16(9+r.IntN(7))<<10 | uint16(r.IntN(1024))
			raw[b+off], raw[b+off+1] = byte(h), byte(h>>8)
		}
		for i := 4; i < 144; i++ {
			raw[b+i] = byte(r.IntN(256))
		}
	}
	wm, err := linalg.WrapQ4K(raw, rows, cols)
	if err != nil {
		t.Fatal(err)
	}
	return wm
}

// TestQ4K_modePlumbing: --quant q4k parses, is a load-time policy (routers and embeddings resolve to
// int8 W8A8, anything with no raw Q4_K bytes degrades to int8 W8A8), and is refused for a safetensors
// source and by the .giw writer.
func TestQ4K_modePlumbing(t *testing.T) {
	q, err := parseQuant("q4k")
	if err != nil || q != quantQ4K {
		t.Fatalf("parseQuant(q4k) = %v, %v", q, err)
	}
	if got := matmulQuant(quantQ4K, "blk.0.ffn_gate_inp.weight"); got != quantInt8I8 {
		t.Errorf("router under q4k resolves to %v, want int8int8", got)
	}
	if got := matmulQuant(quantQ4K, "blk.0.attn_q.weight"); got != quantQ4K {
		t.Errorf("attn_q under q4k resolves to %v, want q4k", got)
	}
	if got := quantQ4K.embedding(); got != quantInt8I8 {
		t.Errorf("q4k embedding policy = %v, want int8int8", got)
	}
	// A site with no raw Q4_K bytes gets int8 W8A8, not an empty or f32 matrix.
	w := make([]float32, 4*64)
	for i := range w {
		w[i] = float32(i%7) - 3
	}
	wm, err := streamQuantized(4, 64, quantQ4K, func(r int, dst []float32) error { copy(dst, w[r*64:]); return nil })
	if err != nil || !isW8A8(&wm) {
		t.Errorf("streamQuantized(q4k) without raw bytes: kind %q, w8a8 %v, err %v", wm.Kind(), isW8A8(&wm), err)
	}
	if got := quantizeWM(linalg.WrapF32(w, 4, 64), quantQ4K); !isW8A8(&got) {
		t.Errorf("quantizeWM(q4k) kind %q, want int8 W8A8", got.Kind())
	}
	if _, err := loadWeights(t.TempDir(), quantQ4K, false, true, false, nil, nil); err == nil || !strings.Contains(err.Error(), "GGUF-only") {
		t.Errorf("q4k safetensors load: err %v, want a GGUF-only refusal", err)
	}
	r := rand.New(rand.NewPCG(61, 62))
	q4 := q4kTestMat(t, r, 3, 256)
	gw := &giwWriter{}
	gw.weightMatKind(&q4, false)
	if gw.err == nil || !strings.Contains(gw.err.Error(), "q4k") {
		t.Errorf(".giw writer on a q4k tensor: err %v, want a refusal naming q4k", gw.err)
	}
	if got, want := wmBytes(&q4), int64(3*144); got != want {
		t.Errorf("wmBytes(q4k 3x256) = %d, want %d", got, want)
	}
}

// TestQ4K_matmulDispatch: goinfer's matmul and matmulInto route a Q4_K weight to aikit's Q4_K kernel.
// Without the case they would fall through to the f32 branch with no f32 payload. Result must equal
// WeightMat.MatmulBT exactly (same kernel, same per-32 activations).
func TestQ4K_matmulDispatch(t *testing.T) {
	r := rand.New(rand.NewPCG(63, 64))
	const M, K, N = 2, 512, 19
	wm := q4kTestMat(t, r, N, K)
	a := make([]float32, M*K)
	for i := range a {
		a[i] = float32(r.NormFloat64())
	}
	want := make([]float32, M*N)
	wm.MatmulBT(a, want, M)
	be := &cpuBackend{}
	for name, run := range map[string]func(dst []float32){
		"matmul":     func(dst []float32) { matmul(be, &wm, a, dst, M) },
		"matmulInto": func(dst []float32) { matmulInto(new(linalg.Workspace), be, &wm, a, dst, M) },
	} {
		got := make([]float32, M*N)
		run(got)
		for i := range want {
			if got[i] != want[i] || math.IsNaN(float64(got[i])) {
				t.Fatalf("%s[%d] = %v, want %v", name, i, got[i], want[i])
			}
		}
	}
}
