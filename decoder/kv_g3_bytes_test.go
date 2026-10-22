//go:build goinfer_testhooks

package decoder

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// g3Check is G3's bar on one cache readback (docs/tasks/task-kv-precision-everywhere-2026-10.md §4): every stored element is the
// delivered precision's (f32 4 B, f16 2 B, i8 1 B plus a 4 B scale for each of K and V per position, head and layer), and nothing is
// held at another precision. hd is the head width the int8 scales stride by.
func g3Check(delivered string, r kvReadback, hd int64) string {
	switch delivered {
	case kvPrecF32:
		if r.n32 == 0 || r.n16 != 0 || r.n8 != 0 || r.nScale != 0 {
			return fmt.Sprintf("f32 delivered, cache holds %d f32, %d f16, %d int8 elements and %d scales", r.n32, r.n16, r.n8, r.nScale)
		}
	case kvPrecF16:
		if r.n16 == 0 || r.n32 != 0 || r.n8 != 0 || r.nScale != 0 {
			return fmt.Sprintf("f16 delivered, cache holds %d f32, %d f16, %d int8 elements and %d scales", r.n32, r.n16, r.n8, r.nScale)
		}
	case kvPrecI8:
		if r.n8 == 0 || r.n32 != 0 || r.n16 != 0 || r.nScale*hd != r.n8 {
			return fmt.Sprintf("i8 delivered, cache holds %d f32, %d f16, %d int8 elements and %d scales (want one per %d int8)", r.n32, r.n16, r.n8, r.nScale, hd)
		}
	default:
		return "no precision to check: " + delivered
	}
	return ""
}

// TestKVDeliveredBytes_CPU is G3's CPU half: every CPU cell the matrix delivers, loaded, 12 positions written, and its cache read
// back layer by layer against the delivered precision's arithmetic.
// TestKVDeliveredBytes_CPU is G3's CPU half (docs/tasks/task-kv-precision-everywhere-2026-10.md §4): every CPU cell the matrix
// delivers, 12 positions decoded, its cache read back: it holds elements of the delivered precision only (the slice's element type
// is the byte width), and at i8 one f32 scale per position, head and K or V for each head_dim int8s. Metal's half is
// metal/kv_g3_metal_test.go.
func TestKVDeliveredBytes_CPU(t *testing.T) {
	cells := kvMatrix(t)
	checked := 0
	for _, f := range kvMatrixFixtures {
		for _, p := range kvMatrixPaths {
			if p.backend != "cpu" {
				continue
			}
			for _, v := range kvMatrixValues {
				c := cells[[3]string{p.name, f.group, v}]
				if c.refused || c.delivered == kvPrecNone {
					continue
				}
				m, err := Load(filepath.Join("..", "testdata", f.dir), kvMatrixOpts(p, v))
				var r *KVRefusal
				if errors.As(err, &r) || err != nil {
					t.Fatalf("%s / %s / --kv %s: Load: %v", p.name, f.group, v, err)
				}
				cache := m.NewCache(16)
				_, _, _, _, hd, _, vocab := m.Dims()
				for i := range 12 {
					if _, err := m.forward((i*37+11)%vocab, cache); err != nil {
						t.Fatalf("%s / %s / --kv %s: forward: %v", p.name, f.group, v, err)
					}
				}
				rb := readbackCPUKV(cache)
				if why := g3Check(c.delivered, rb, int64(hd)); why != "" {
					t.Errorf("%s / %s / --kv %s: %s", p.name, f.group, v, why)
				}
				fmt.Fprintf(os.Stderr, "[G3 cpu] %-28s %-14s --kv %-4s delivered %-3s: f32 %d, f16 %d, i8 %d, scales %d\n",
					p.name, f.dir, v, c.delivered, rb.n32, rb.n16, rb.n8, rb.nScale)
				checked++
				m.Close()
			}
		}
	}
	if checked < 50 {
		t.Fatalf("only %d CPU cells were read back", checked)
	}
}
