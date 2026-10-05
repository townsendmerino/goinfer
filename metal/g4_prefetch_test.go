//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/prequant"
)

// TestG4Prefetch_bitExactAndRuns is lever 3's gate (docs/tasks/task-m26-mac-2026-10.md): on the tiny Gemma 4 MoE as a
// metal .giw, paged at 3 slots with pread staging (M26's shape), every logit of 8 positions is equal with the next-layer
// prefetch on, off, and with every expert resident; and on the prefetch arm experts were prefetched.
func TestG4Prefetch_bitExactAndRuns(t *testing.T) {
	const ckpt = "../testdata/gemma4-moe-tiny"
	if _, err := os.Stat(ckpt + "/model.safetensors"); err != nil {
		t.Skipf("no fixture: %v", err)
	}
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	giw := filepath.Join(t.TempDir(), "g4.metal.giw")
	if err := prequant.Transcode(context.Background(), ckpt, giw, "int4", false, decoder.GIWTargetMetal); err != nil {
		t.Fatal(err)
	}
	prev := g4PrefetchOn
	defer func() { g4PrefetchOn = prev }()
	run := func(slots int, prefetch bool) (lg [][]float32, prefetched, hits int) {
		g4PrefetchOn = prefetch
		if slots > 0 {
			t.Setenv("GOINFER_METAL_MOE_SLOTS", strconv.Itoa(slots))
		} else {
			os.Unsetenv("GOINFER_METAL_MOE_SLOTS")
		}
		m, err := decoder.Load(giw, decoder.Options{Backend: "metal", Quant: "int4"})
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		mr := m.ResidentForwardForTest().(*metalResident)
		mr.Reset()
		for i, tok := range twoGeomPrompt {
			l, err := mr.Forward(m.EmbedResidentForTest(tok), i)
			if err != nil {
				t.Fatal(err)
			}
			lg = append(lg, append([]float32(nil), l...))
		}
		for l := range mr.r.layers {
			if gl := mr.r.layers[l].g4moe; gl != nil && gl.pool != nil {
				prefetched += gl.pool.prefetched
				hits += gl.pool.prefetchHits
			}
		}
		return lg, prefetched, hits
	}
	base, _, _ := run(0, true)
	off, n0, _ := run(3, false)
	on, n1, h1 := run(3, true)
	if n0 != 0 || n1 == 0 {
		t.Fatalf("prefetched %d with the prefetch off, %d with it on", n0, n1)
	}
	for _, arm := range []struct {
		name string
		lg   [][]float32
	}{{"paged, prefetch off", off}, {"paged, prefetch on", on}} {
		for i := range base {
			for j := range base[i] {
				if math.Float32bits(base[i][j]) != math.Float32bits(arm.lg[i][j]) {
					t.Fatalf("%s: position %d logit %d differs from every-expert-resident", arm.name, i, j)
				}
			}
		}
	}
	t.Logf("8 positions bit-identical resident / paged / paged with prefetch; %d experts prefetched, %d of them used", n1, h1)
}
