package decoder

import (
	"bytes"
	"context"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// pleSources are the per-layer-table sources the PLE gates read: the synthetic E2B-shaped GGUF always, and the generated
// testdata/gemma4-emodel-tiny (safetensors) where it exists.
func pleSources(t *testing.T) []string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "gemma4-e2b.gguf")
	if err := os.WriteFile(path, tinyGemma4GGUF(gemma4Variants[1]), 0o644); err != nil {
		t.Fatal(err)
	}
	srcs := []string{path}
	if _, err := os.Stat(filepath.Join("..", "testdata", "gemma4-emodel-tiny", "config.json")); err == nil {
		srcs = append(srcs, filepath.Join("..", "testdata", "gemma4-emodel-tiny"))
	} else {
		t.Logf("no testdata/gemma4-emodel-tiny (generated): the safetensors loader is not covered on this run")
	}
	return srcs
}

// TestPLE_gatherTableIgnoresThePin is PLE-G1 (docs/tasks/task-memory-campaign-2026-10.md, MEM8(b) gates): at int4 the
// per-layer token table is int4 with the head pin on and off, its rows byte-equal between the two, while the token
// embedding still follows the pin (int8 with it on, int4 with it off).
func TestPLE_gatherTableIgnoresThePin(t *testing.T) {
	for _, src := range pleSources(t) {
		load := func(e4 bool) *Model {
			m, err := Load(src, Options{Quant: "int4", EmbedInt4: e4})
			if err != nil {
				t.Fatalf("%s: %v", src, err)
			}
			t.Cleanup(func() { m.Close() })
			return m
		}
		pin, free := load(false), load(true)
		pt, ft := &pin.w.PerLayerTokenEmbed, &free.w.PerLayerTokenEmbed
		if pt.Rows() == 0 {
			t.Fatalf("%s: no per-layer table", src)
		}
		if !pt.IsInt4() || !ft.IsInt4() {
			t.Errorf("%s: per-layer table kinds %s (pin on) and %s (pin off); want int4 both", src, pt.Kind(), ft.Kind())
		}
		a, b := make([]float32, pt.Cols()), make([]float32, ft.Cols())
		for r := range pt.Rows() {
			pt.Row(r, a)
			ft.Row(r, b)
			for j := range a {
				if math.Float32bits(a[j]) != math.Float32bits(b[j]) {
					t.Fatalf("%s: per-layer row %d differs at %d with the pin on and off", src, r, j)
				}
			}
		}
		if _, _, _, ok := pin.w.Embed.Int8(); !ok || !free.w.Embed.IsInt4() {
			t.Errorf("%s: token embedding kinds %s (pin on) and %s (pin off); want int8 and int4", src, pin.w.Embed.Kind(), free.w.Embed.Kind())
		}
	}
}

// TestPLE_sidecarTag is PLE-G2: the ".ple4" segment marks exactly a per-layer source at int4 with the pin on, every other
// name is unchanged, and the bundle written under it carries an int4 table and loads to the direct load's logits.
func TestPLE_sidecarTag(t *testing.T) {
	srcs := pleSources(t)
	plain := filepath.Join(t.TempDir(), "gemma4-26b.gguf")
	if err := os.WriteFile(plain, tinyGemma4GGUF(gemma4Variants[0]), 0o644); err != nil {
		t.Fatal(err)
	}
	llama := filepath.Join("..", "testdata", "llama-tiny")
	for _, c := range []struct {
		src, quant string
		e4         bool
		want       string
	}{
		{srcs[0], "int4", false, ".ple4"},
		{srcs[0], "int4", true, ""},
		{srcs[0], "int8int8", false, ""},
		{srcs[0], "int4mix", false, ""},
		{plain, "int4", false, ""},
		{llama, "int4", false, ""},
	} {
		if got := gatherTableTag(c.src, c.quant, c.e4); got != c.want {
			t.Errorf("gatherTableTag(%s, %s, %v) = %q, want %q", filepath.Base(c.src), c.quant, c.e4, got, c.want)
		}
	}
	if len(srcs) > 1 && gatherTableTag(srcs[1], "int4", false) != ".ple4" {
		t.Errorf("gatherTableTag(gemma4-emodel-tiny) misses the per-layer table in config.json")
	}
	if p := SidecarPath(srcs[0], "int4", "metal", false); !strings.HasSuffix(p, ".int4.ple4.metal.v17.giw") {
		t.Errorf("SidecarPath = %s, want the .int4.ple4.metal.v17.giw name", p)
	}
	base := strings.TrimSuffix(plain, ".gguf")
	if p := SidecarPath(plain, "int4", "metal", false); p != base+".int4.metal.v17.giw" {
		t.Errorf("SidecarPath(no per-layer table) = %s, want %s", p, base+".int4.metal.v17.giw")
	}
	if p := SidecarPath(srcs[0], "int4", "metal", true); !strings.HasSuffix(p, ".int4.e4h.metal.v17.giw") {
		t.Errorf("SidecarPath(embed-int4) = %s, want the .e4h name unchanged", p)
	}

	// The transcode a load's cache runs, into a buffer: a loadable bundle needs the source's tokenizer, which neither synthetic
	// source has, so the gate reads the transcoded table's rows against the direct load's.
	var buf bytes.Buffer
	if _, err := StreamTranscodeGGUF(context.Background(), srcs[0], &buf, "int4", false, GIWTargetNone, "ple"); err != nil {
		t.Fatal(err)
	}
	sw, err := LoadSerializedWeights(buf.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	dm, err := Load(srcs[0], Options{Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer dm.Close()
	st, dt := &sw.PerLayerTokenEmbed, &dm.w.PerLayerTokenEmbed
	if !st.IsInt4() {
		t.Fatalf("the transcoded per-layer table is %s, want int4", st.Kind())
	}
	a, b := make([]float32, st.Cols()), make([]float32, dt.Cols())
	for r := range dt.Rows() {
		st.Row(r, a)
		dt.Row(r, b)
		for j := range a {
			if math.Float32bits(a[j]) != math.Float32bits(b[j]) {
				t.Fatalf("transcoded per-layer row %d differs from the direct load's at %d", r, j)
			}
		}
	}
}
