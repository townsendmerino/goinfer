package decoder

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// qwenShapedGGUF writes a header-only GGUF whose tensor directory has a Qwen 2.5 decoder's shapes (tied head: the token
// table only), so its parameter count is the real model's, and returns its path.
func qwenShapedGGUF(t *testing.T, layers, hidden, inter, kvDim, vocab uint64) string {
	t.Helper()
	ts := []ggufTensorDecl{embedTensor(hidden, vocab)}
	for l := range layers {
		for i, d := range [][]uint64{{hidden, hidden}, {hidden, kvDim}, {hidden, kvDim}, {hidden, hidden},
			{hidden, inter}, {hidden, inter}, {inter, hidden}, {hidden}, {hidden}} {
			ts = append(ts, ggufTensorDecl{name: fmt.Sprintf("blk.%d.t%d.weight", l, i), dims: d})
		}
	}
	p := filepath.Join(t.TempDir(), "m.gguf")
	if err := os.WriteFile(p, buildGGUF([]ggufKV{kvStr("general.architecture", "qwen2")}, ts), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// qwen05bGGUF and qwen15bGGUF are Qwen 2.5 0.5B and 1.5B shaped.
func qwen05bGGUF(t *testing.T) string { return qwenShapedGGUF(t, 24, 896, 4864, 128, 151936) }
func qwen15bGGUF(t *testing.T) string { return qwenShapedGGUF(t, 28, 1536, 8960, 256, 151936) }

func TestPeekParamCount_qwenShapes(t *testing.T) {
	for _, c := range []struct {
		name   string
		path   string
		lo, hi int64
	}{
		{"0.5B", qwen05bGGUF(t), 480e6, 500e6},
		{"1.5B", qwen15bGGUF(t), 1.5e9, 1.6e9},
	} {
		if n := PeekParamCount(c.path); n < c.lo || n > c.hi {
			t.Errorf("%s: PeekParamCount %d, want in [%d, %d]", c.name, n, c.lo, c.hi)
		}
	}
	if n := PeekParamCount(filepath.Join(t.TempDir(), "missing.gguf")); n != 0 {
		t.Errorf("missing file: PeekParamCount %d, want 0 (unknown)", n)
	}
}

// TestEmbedInt4For_defaultBySize: the default (EmbedInt4Default) keeps the int4 head below EmbedInt4DefaultMaxParams and
// takes the int8 head from there up; an explicit choice wins both ways; an unknown size keeps what was asked.
func TestEmbedInt4For_defaultBySize(t *testing.T) {
	small, large := qwen05bGGUF(t), qwen15bGGUF(t)
	unknown := filepath.Join(t.TempDir(), "x.giw")
	for _, c := range []struct {
		name          string
		src           string
		on, isDefault bool
		want          bool
	}{
		{"0.5B default", small, true, true, true},
		{"1.5B default", large, true, true, false},
		{"1.5B explicit on", large, true, false, true},
		{"0.5B explicit off", small, false, false, false},
		{"1.5B default off (Metal)", large, false, true, false},
		{"unknown size default", unknown, true, true, true},
	} {
		if got := EmbedInt4For(c.src, Options{EmbedInt4: c.on, EmbedInt4Default: c.isDefault}); got != c.want {
			t.Errorf("%s: EmbedInt4For %v, want %v", c.name, got, c.want)
		}
	}
}

// TestLoad_resolvesEmbedInt4Default: Load itself applies EmbedInt4For, so a library caller that sets the default gets the
// size rule, and the Model reports the head it was built with.
func TestLoad_resolvesEmbedInt4Default(t *testing.T) {
	src := filepath.Join("..", "testdata", "gemma1-tiny.gguf")
	old := embedInt4DefaultMaxParams
	t.Cleanup(func() { embedInt4DefaultMaxParams = old })
	for _, c := range []struct {
		name      string
		maxParams int64
		isDefault bool
		want      bool
	}{
		{"under the bar", 1 << 62, true, true},
		{"over the bar", 1, true, false},
		{"over the bar, explicit", 1, false, true},
	} {
		embedInt4DefaultMaxParams = c.maxParams
		m, err := Load(src, Options{Quant: "int4", EmbedInt4: true, EmbedInt4Default: c.isDefault, DirectLoad: true})
		if err != nil {
			t.Fatal(err)
		}
		if got := m.EmbedInt4(); got != c.want {
			t.Errorf("%s: Model.EmbedInt4 %v, want %v", c.name, got, c.want)
		}
		m.Close()
	}
}
