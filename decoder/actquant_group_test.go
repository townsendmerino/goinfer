package decoder

import (
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

func tinyPhi3Logits(t *testing.T, m *Model, ids []int) [][]float32 {
	t.Helper()
	cache := m.NewCache(len(ids) + 1)
	out := make([][]float32, len(ids))
	for i, id := range ids {
		lg, err := m.forward(id, cache)
		if err != nil {
			t.Fatal(err)
		}
		out[i] = append([]float32(nil), lg...)
	}
	return out
}

// TestActQuantGroup_perModel: Options.ActQuantGroup is a per-model choice on the CPU. A model loaded
// with 32 produces the same logits, bit for bit, as the process-wide aikit toggle (the same kernels,
// reached through the weight stamp instead of global state), and a per-row model loaded alongside
// it, run interleaved token by token, is unaffected: identical to a per-row model run alone. The
// premise, that per-32 and per-row differ at all on this fixture, is checked too.
func TestActQuantGroup_perModel(t *testing.T) {
	const ckpt = "../testdata/phi3-tiny"
	ids := []int{1, 7, 42, 100, 5, 120, 13, 88} // < 128: the fixture's vocab
	for _, quant := range []string{"int8int8", "int4"} {
		load := func(g int) *Model {
			m, err := Load(ckpt, Options{Quant: quant, ActQuantGroup: g})
			if err != nil {
				t.Fatalf("%s Load(group %d): %v", quant, g, err)
			}
			t.Cleanup(func() { m.Close() })
			return m
		}
		perRowAlone := tinyPhi3Logits(t, load(0), ids)

		linalg.SetActQuantGroup(32)
		viaGlobal := tinyPhi3Logits(t, load(0), ids)
		linalg.SetActQuantGroup(0)

		a, b := load(32), load(0)
		ca, cb := a.NewCache(len(ids)+1), b.NewCache(len(ids)+1)
		for i, id := range ids {
			la, err := a.forward(id, ca)
			if err != nil {
				t.Fatal(err)
			}
			for k := range la {
				if la[k] != viaGlobal[i][k] {
					t.Fatalf("%s pos %d: per-model group 32 differs from the global toggle at logit %d (%v vs %v)", quant, i, k, la[k], viaGlobal[i][k])
				}
			}
			lb, err := b.forward(id, cb)
			if err != nil {
				t.Fatal(err)
			}
			for k := range lb {
				if lb[k] != perRowAlone[i][k] {
					t.Fatalf("%s pos %d: the per-row model changed when a group-32 model ran beside it (logit %d)", quant, i, k)
				}
			}
		}
		differ := false
		for i := range ids {
			for k := range viaGlobal[i] {
				if viaGlobal[i][k] != perRowAlone[i][k] {
					differ = true
				}
			}
		}
		if !differ {
			t.Errorf("%s premise: per-32 and per-row gave identical logits on this fixture", quant)
		}
	}
}
