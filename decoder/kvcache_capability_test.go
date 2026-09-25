package decoder

import (
	"path/filepath"
	"testing"
)

// NewCache's int8-KV and ring-buffer decisions came from two hand-written family lists that disagreed
// (lfm2 and llama4 were on the int8 list but not the ring list; gpt-oss the other way round). They are now
// one rule each — Architecture.kvInt8OK / kvRingsOK, read off the ownForwards table. This pins that the
// change is behaviour-preserving: on every loadable fixture, the cache NewCache builds with int8 requested
// has exactly the int8 and ring state the OLD lists produced (reproduced below as the reference, verbatim).
func TestNewCache_kvCapabilitiesMatchTheOldLists(t *testing.T) {
	oldInt8 := func(a *Architecture) bool {
		return a.gemma4 == nil && a.qwen35 == nil && a.granite == nil && a.nemotron == nil && a.MoE == nil &&
			a.lfm2 == nil && a.llama4 == nil
	}
	oldRings := func(a *Architecture) bool {
		return a.gemma4 == nil && a.qwen35 == nil && a.granite == nil && a.nemotron == nil && a.gptoss == nil
	}
	dirs, _ := filepath.Glob(filepath.Join("..", "testdata", "*", "model.safetensors"))
	checked, int8On, ringsOn := 0, 0, 0
	for _, st := range dirs {
		dir := filepath.Dir(st)
		name := filepath.Base(dir)
		m, err := Load(dir, Options{KVQuant: "i8"})
		if err != nil {
			continue // not a decoder checkpoint (vision towers, …)
		}
		a := m.w.arch
		c := m.NewCache(16)
		gotI8 := c.quant == kvI8
		gotRings := c.localAny
		wantI8 := oldInt8(a)
		wantRings := oldRings(a) && a.SlidingWindow > 0 && hasLocalLayer(a)
		if gotI8 != wantI8 {
			t.Errorf("%s: int8 KV = %v, the old list gave %v", name, gotI8, wantI8)
		}
		if gotRings != wantRings {
			t.Errorf("%s: ring-buffered local layers = %v, the old list gave %v", name, gotRings, wantRings)
		}
		// And the rule itself: a family with its own layer loop gets neither unless its table entry says so.
		if f, own := a.ownForward(); own && ((gotI8 && !f.KVInt8) || (gotRings && !f.KVRings)) {
			t.Errorf("%s (%s, own forward): int8=%v rings=%v without the table bit", name, f.Name, gotI8, gotRings)
		}
		checked++
		if gotI8 {
			int8On++
		}
		if gotRings {
			ringsOn++
		}
		m.Close()
	}
	t.Logf("%d fixtures: int8 KV on %d, rings on %d", checked, int8On, ringsOn)
	if checked == 0 || int8On == 0 || ringsOn == 0 {
		t.Fatalf("coverage hole: %d fixtures, %d with int8, %d with rings — both outcomes of both rules must be exercised", checked, int8On, ringsOn)
	}
}

func hasLocalLayer(a *Architecture) bool {
	for l := range a.NumLayers {
		if !a.isGlobalLayer(l) {
			return true
		}
	}
	return false
}
