//go:build gpu

package webgpu

import (
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// TestKVCheckDeclared holds the build's check of a layer's declared rows (MM5's WebGPU half, G-MP2w's refusal half): the rows its caches
// are sized from are accepted, and each way a declaration can differ from them is refused: a row missing, extra, twice, or of another width,
// head count or precision, and a row on a layer that reads another's cache. Each refusal is a plant that goes red if the check is removed.
func TestKVCheckDeclared(t *testing.T) {
	kv := func(name string, elems, heads int) program.Buf {
		return program.Buf{Name: name, Unit: program.PerPosition, Prec: program.PrecKV, Elems: elems, Heads: heads}
	}
	latent := func(elems int) program.Buf {
		return program.Buf{Name: "latent", Unit: program.PerPosition, Prec: program.PrecF32, Elems: elems}
	}
	state := program.Buf{Name: "state", Unit: program.PerSlot, Prec: program.PrecF32, Elems: 99}
	for _, c := range []struct {
		name         string
		bufs         []program.Buf
		owns, latent bool
		kvDim, nKV   int
		wantErr      string // a substring of the refusal; "" accepts
	}{
		{"attention rows", []program.Buf{kv("k", 64, 2), kv("v", 64, 2), state}, true, false, 64, 2, ""},
		{"a window is not compared", []program.Buf{{Name: "k", Unit: program.PerPosition, Prec: program.PrecKV, Elems: 64, Heads: 2, Window: 16}, kv("v", 64, 2)}, true, false, 64, 2, ""},
		{"mla latent", []program.Buf{latent(72)}, true, true, 72, 0, ""},
		{"a layer that reads another's cache declares none", []program.Buf{state}, false, false, 64, 2, ""},
		{"a K row missing", []program.Buf{kv("v", 64, 2)}, true, false, 64, 2, "declares no k rows"},
		{"a row twice", []program.Buf{kv("k", 64, 2), kv("k", 64, 2), kv("v", 64, 2)}, true, false, 64, 2, "twice"},
		{"an extra row", []program.Buf{kv("k", 64, 2), kv("v", 64, 2), kv("w", 64, 2)}, true, false, 64, 2, "has no cache for"},
		{"a row one element short", []program.Buf{kv("k", 63, 2), kv("v", 64, 2)}, true, false, 64, 2, "of 63 elements"},
		{"another head count", []program.Buf{kv("k", 64, 4), kv("v", 64, 2)}, true, false, 64, 2, "in 4 heads"},
		{"a latent declared at the KV precision", []program.Buf{{Name: "latent", Unit: program.PerPosition, Prec: program.PrecKV, Elems: 72}}, true, true, 72, 0, "precision"},
		{"rows on a layer that reads another's cache", []program.Buf{kv("k", 64, 2), kv("v", 64, 2)}, false, false, 64, 2, "has no cache for"},
	} {
		err := kvCheckDeclared(3, c.bufs, c.owns, c.latent, c.kvDim, c.nKV)
		switch {
		case c.wantErr == "" && err != nil:
			t.Errorf("%s: refused: %v", c.name, err)
		case c.wantErr != "" && (err == nil || !strings.Contains(err.Error(), c.wantErr)):
			t.Errorf("%s: err = %v, want one containing %q", c.name, err, c.wantErr)
		}
	}
}
