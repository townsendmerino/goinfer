package modelload

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// headerOnlyGGUF writes a GGUF v3 whose tensor directory declares the given 2-D shapes and no data: enough for a
// parameter count, nothing to load.
func headerOnlyGGUF(t *testing.T, arch string, shapes ...[2]uint64) string {
	t.Helper()
	var b bytes.Buffer
	w := func(v any) { binary.Write(&b, binary.LittleEndian, v) } //nolint:errcheck // bytes.Buffer does not fail
	str := func(s string) { w(uint64(len(s))); b.WriteString(s) }
	w(uint32(0x46554747))
	w(uint32(3))
	w(uint64(len(shapes)))
	w(uint64(1))
	str("general.architecture")
	w(uint32(8))
	str(arch)
	for i, s := range shapes {
		str("blk." + string(rune('a'+i)) + ".weight")
		w(uint32(2))
		w(s[0])
		w(s[1])
		w(uint32(0))
		w(uint64(0))
	}
	for b.Len()%32 != 0 {
		b.WriteByte(0)
	}
	p := filepath.Join(t.TempDir(), "m.gguf")
	if err := os.WriteFile(p, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// TestPlan_embedInt4DefaultBySize: the sidecar Load would build carries the head the size rule picks, from the
// default only; an explicit --embed-int4 keeps the int4 head on a large model. The shapes total 0.49B and 1.54B
// parameters, and the architecture is not Qwen's: the rule reads size, not family.
func TestPlan_embedInt4DefaultBySize(t *testing.T) {
	small := headerOnlyGGUF(t, "llama", [2]uint64{896, 151936}, [2]uint64{896, 400000})
	large := headerOnlyGGUF(t, "llama", [2]uint64{1536, 151936}, [2]uint64{1536, 853000})
	for _, c := range []struct {
		name      string
		src       string
		isDefault bool
		wantE4H   bool
	}{
		{"0.5B default", small, true, true},
		{"1.5B default", large, true, false},
		{"1.5B explicit", large, false, true},
	} {
		p, err := Plan(Request{Spec: c.src, Opts: decoder.Options{Quant: "int4", Backend: "cpu", EmbedInt4: true, EmbedInt4Default: c.isDefault}})
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if p.Opts.EmbedInt4 != c.wantE4H || p.Opts.EmbedInt4Default {
			t.Errorf("%s: Plan Opts EmbedInt4 %v (default %v), want %v (resolved)", c.name, p.Opts.EmbedInt4, p.Opts.EmbedInt4Default, c.wantE4H)
		}
		if got := strings.Contains(p.Build, ".e4h."); got != c.wantE4H {
			t.Errorf("%s: sidecar to build %q: e4h %v, want %v", c.name, p.Build, got, c.wantE4H)
		}
	}
}
