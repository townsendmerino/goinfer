package decoder

import (
	"context"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/internal/giw"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestW4A8DecodeParity is the rewire gate: the CPU int4 decode now runs aikit's
// int8-activation W4A8 kernel (weightmat.matmul, M==1). Its numerics differ from
// the OLD f32-activation MatmulBTQ4 by design — the right oracle is an
// int8-activation reference, here the W8A8 (int8-weight × int8-activation) decode
// of the same model. We require: (1) the int4 W4A8 decode is COHERENT (decodes to
// real text, not garbage — the failure mode of a mis-wired kernel), and (2) it
// AGREES with the int8-activation reference on most of the first 16 greedy tokens
// (int4 vs int8 weights differ only by quant noise once activations are int8).
//
//	GOINFER_W4A8_INT4=/tmp/qwen1.5b-int4.giw GOINFER_W4A8_INT8=/tmp/qwen15-int8.giw \
//	  go test ./decoder/ -run TestW4A8DecodeParity -v
func TestW4A8DecodeParity(t *testing.T) {
	// Through the shared registry, not os.Getenv: the gate and the sweep preflight then apply
	// the SAME predicate to the same candidate paths (testdata/assets.json), which is what keeps
	// "the sweep says this asset is present" and "the gate can actually load it" the same claim.
	int4 := assetPath(t, "GOINFER_W4A8_INT4")
	int8p := assetPath(t, "GOINFER_W4A8_INT8")

	data, err := os.ReadFile(int4)
	if err != nil {
		t.Fatal(err)
	}
	_, tokGGUF, err := giw.Read(data)
	if err != nil {
		t.Fatal(err)
	}
	tk, err := tokenizer.LoadGGUFBytes(tokGGUF)
	if err != nil {
		t.Fatal(err)
	}
	promptIDs, _ := tk.Encode("Write a short note about recursion.\n", true)

	const n = 16
	greedy := SamplingParams{Temperature: 0}
	gen := func(path string, opts Options) []int {
		m, err := Load(path, opts)
		if err != nil {
			t.Fatalf("load %s: %v", path, err)
		}
		defer m.Close()
		ch, _ := m.Generate(context.Background(), promptIDs, n, greedy)
		var out []int
		for id := range ch {
			out = append(out, id)
		}
		return out
	}

	int4Tok := gen(int4, Options{Backend: "cpu"})                    // W4A8 decode
	refTok := gen(int8p, Options{Backend: "cpu", Quant: "int8int8"}) // int8-activation reference

	int4Txt, _ := tk.Decode(int4Tok)
	refTxt, _ := tk.Decode(refTok)
	t.Logf("int4 W4A8 decode (%d tok): %q", len(int4Tok), int4Txt)
	t.Logf("int8 W8A8 ref    (%d tok): %q", len(refTok), refTxt)
	t.Logf("int4 tokens: %v", int4Tok)
	t.Logf("ref  tokens: %v", refTok)

	// (1) Coherence: a mis-wired kernel emits repetition/garbage. Require the
	// decode to produce a non-trivial number of distinct token ids.
	distinct := map[int]bool{}
	for _, id := range int4Tok {
		distinct[id] = true
	}
	if len(int4Tok) < n {
		t.Errorf("int4 produced %d/%d tokens (early stop / hang)", len(int4Tok), n)
	}
	if len(distinct) < n/3 {
		t.Errorf("int4 decode looks degenerate: only %d distinct ids in %d tokens", len(distinct), len(int4Tok))
	}

	// (2) Agreement with the int8-activation reference over the first 16 greedy
	// tokens. int4 vs int8 weights flip some near-tie steps (quant noise), so we
	// require a strong majority, not bit-exactness.
	agree := 0
	for i := 0; i < len(int4Tok) && i < len(refTok); i++ {
		if int4Tok[i] == refTok[i] {
			agree++
		}
	}
	t.Logf("=== W4A8 parity: int4(W4A8) vs int8(W8A8) greedy agreement %d/%d ===", agree, n)
	if agree*2 < n { // < 50% would indicate a wiring error, not quant noise
		t.Errorf("int4 W4A8 decode agrees with int8 reference only %d/%d — suspect a wiring error", agree, n)
	}
}
