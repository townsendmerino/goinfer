//go:build darwin

package metal

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestPrefillInternal_refusesPastTheKVCap: the internal resident.PrefillLast fills the K/V cache up to its capacity and panics, naming the
// cap, on a pass that would write past it (from position 0, ending inside the cache, or from a negative position), so no internal caller
// can write beyond the buffers as the public metalResident.checkCap already prevents for the decoder.
func TestPrefillInternal_refusesPastTheKVCap(t *testing.T) {
	m, err := decoder.Load(filepath.Join("..", "testdata", "llama-tiny"), decoder.Options{Quant: "int4"})
	if err != nil {
		t.Skipf("no llama-tiny fixture: %v", err)
	}
	defer m.Close()
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("resident: %v", err)
	}
	defer r.Close()
	c := r.kvCap()
	if c <= 0 || c > 4096 {
		t.Fatalf("the fixture's KV cap is %d positions: not a small fixture", c)
	}
	_, _, _, _, _, _, vocab := m.Dims()
	ids := make([]int, c+8)
	for i := range ids {
		ids[i] = (i*37 + 5) % vocab
	}
	embs := getEmbs(r, ids)
	panics := func(name string, f func()) (msg string) {
		defer func() {
			if v := recover(); v != nil {
				msg, _ = v.(string)
			}
		}()
		f()
		return ""
	}
	if msg := panics("fill", func() { r.PrefillLast(embs[:c], 0) }); msg != "" {
		t.Fatalf("a pass that fills the %d-position cache exactly panicked: %s", c, msg)
	}
	for _, x := range []struct {
		name  string
		start int
		rows  int
	}{{"one row past the cap from 0", 0, c + 1}, {"crossing the cap", c - 4, 8}, {"a negative position", -1, 4}} {
		msg := panics(x.name, func() { r.PrefillLast(embs[:x.rows], x.start) })
		if !strings.Contains(msg, "past the resident KV cap") {
			t.Errorf("%s (%d rows at %d, cap %d): want a panic naming the cap, got %q", x.name, x.rows, x.start, c, msg)
		}
	}
}
