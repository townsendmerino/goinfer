package decoder

import (
	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"path/filepath"
	"testing"
)

// NewCache's int8-KV and ring-buffer decisions came from two hand-written family lists that disagreed
// (lfm2 and llama4 were on the int8 list but not the ring list; gpt-oss the other way round). They are now
// one rule each — Architecture.kvInt8OK / kvRingsOK, read off the family's attention kernel (attnOrder). This pins that the
// change is behaviour-preserving: on every loadable fixture, the cache NewCache builds with int8 requested
// has exactly the int8 and ring state the OLD lists produced (reproduced below as the reference; the MoE exclusion narrowed to gpt-oss, whose own loop still refuses).
func TestNewCache_kvCapabilitiesMatchTheOldLists(t *testing.T) {
	slowTest(t)
	oldInt8 := func(a *darch.Architecture) bool {
		return a.Gemma4 == nil && a.Qwen35 == nil && a.Granite == nil && a.Nemotron == nil && a.GPTOSS == nil &&
			a.LFM2 == nil && a.Llama4 == nil
	}
	oldRings := func(a *darch.Architecture) bool {
		return a.Gemma4 == nil && a.Qwen35 == nil && a.Granite == nil && a.Nemotron == nil && a.GPTOSS == nil
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
		a := m.w.Arch
		c := m.NewCache(16)
		gotI8 := c.c.Quant == kvI8
		gotRings := c.c.LocalAny
		wantI8 := oldInt8(a)
		wantRings := oldRings(a) && a.SlidingWindow > 0 && hasLocalLayer(a)
		if gotI8 != wantI8 {
			t.Errorf("%s: int8 KV = %v, the old list gave %v", name, gotI8, wantI8)
		}
		if gotRings != wantRings {
			t.Errorf("%s: ring-buffered local layers = %v, the old list gave %v", name, gotRings, wantRings)
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

func hasLocalLayer(a *darch.Architecture) bool {
	for l := range a.NumLayers {
		if !a.IsGlobalLayer(l) {
			return true
		}
	}
	return false
}
