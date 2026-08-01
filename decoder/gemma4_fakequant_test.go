//go:build realckpt

package decoder

import (
	"context"
	"fmt"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestGemma4FakeQuant drives the fakequant.go diagnostic against the real gemma-4-26b-a4b-it:
// it loads under a base quant (ZZBASE, default int4) with GOINFER_FAKEQUANT set to a 4-bit
// scheme and prints the greedy continuation of a fixed prompt. It built the int4-quality
// matrix in docs/task-gemma4-moe.md (sym/affine × int8/f32 act × full/experts-only). No
// assertion — it is a measurement harness; the fidelity gate that makes its output
// trustworthy is TestFakeQuantSymMatchesRuntimeInt4 (default suite). Skipped without the ckpt.
//
//	GOINFER_FAKEQUANT=sym|affine|symmse   4-bit weight scheme (unset = real int4, no fake-quant)
//	GOINFER_FAKEQUANT_ACT=f32             store Q8 (f32 activations) instead of W8A8 (int8 act)
//	GOINFER_FAKEQUANT_EXPERTS=1           fake-4-bit only the MoE experts (load ZZBASE=int8int8)
//	ZZBASE=int4|int8int8                  base quant (default int4)
func TestGemma4FakeQuant(t *testing.T) {
	dir := "/home/francis/models/gemma-4-26b-a4b-it"
	if _, err := os.Stat(dir); err != nil {
		t.Skip("real gemma-4-26b-a4b-it not present")
	}
	base := os.Getenv("ZZBASE")
	if base == "" {
		base = "int4"
	}
	m, err := Load(dir, Options{Quant: base})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	tk, err := tokenizer.Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	ids, _ := tk.Encode("The capital of France is", true)
	out, _ := m.Generate(context.Background(), ids, 20, SamplingParams{})
	gen := make([]int, 0, 20)
	for id := range out {
		gen = append(gen, id)
	}
	txt, _ := tk.Decode(gen)
	fmt.Printf("FAKEQUANT=%q base=%q act=%q expertsOnly=%q gen=%q\n",
		os.Getenv("GOINFER_FAKEQUANT"), base, os.Getenv("GOINFER_FAKEQUANT_ACT"),
		os.Getenv("GOINFER_FAKEQUANT_EXPERTS"), txt)
}
