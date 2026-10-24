//go:build cuda && goinfer_testhooks

package cuda

import (
	"errors"
	"io/fs"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGemma4MoE_floatParity holds the float-activation decode (floatAct) to the CPU on f32 activations
// (decoder.SetInt4FloatActForTest), the same int4 weights, over 16 positions of gemma4-moe-tiny: logits cosine at least
// 0.9999 and the same argmax at every position. The W4A8 decode keeps its own gates (TestGemma4MoE_residentParity and
// the bit-identity tests, which build with floatActOffForTest).
func TestGemma4MoE_floatParity(t *testing.T) {
	const dir = "../testdata/gemma4-moe-tiny"
	if _, err := os.Stat(dir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no fixture (%s): run scripts/pin_gemma4_moe_forward.py", dir)
	}
	mc, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cuda): %v", err)
	}
	defer mc.Close()
	r := mustResident(t, mc)
	if !r.floatAct {
		t.Fatalf("the resident did not take the float-activation decode: %s", r.floatActDecline())
	}
	m4, err := decoder.Load(dir, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cpu int4): %v", err)
	}
	defer m4.Close()
	decoder.SetInt4FloatActForTest(true)
	defer decoder.SetInt4FloatActForTest(false)
	prompt := []int{1, 7, 42, 100, 5, 200, 13, 88, 3, 71, 128, 9, 250, 17, 60, 200}
	c := m4.NewCache(len(prompt))
	for i, tok := range prompt {
		cpu, err := m4.ForwardForTest(tok, c)
		if err != nil {
			t.Fatal(err)
		}
		gpu, err := r.Forward(mc.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatal(err)
		}
		cos, _ := cosMaxAbs(gpu, cpu)
		ag, ac := argmaxF(gpu), argmaxF(cpu)
		t.Logf("pos %2d: cosine %.7f argmax gpu %d cpu %d", i, cos, ag, ac)
		if cos < 0.9999 || ag != ac {
			t.Errorf("pos %d: float decode departs from the CPU float path (cosine %.7f, argmax %d vs %d)", i, cos, ag, ac)
		}
	}
	if n := decoder.Int4FloatActMissedForTest(); n != 0 {
		t.Errorf("%d W4A8 kernel calls on the CPU side: the reference was not all float", n)
	}
}
