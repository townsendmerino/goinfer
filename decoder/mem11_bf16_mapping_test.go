package decoder

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// writeBF16Checkpoint copies the F32 checkpoint in src to dst with every tensor's values rounded to bf16: stored as BF16 when
// asBF16 (the form a real bf16 checkpoint has, with nothing F32 left to keep the mapping), else as F32. The header is padded so
// every payload is 8-aligned in a mapping.
func writeBF16Checkpoint(t *testing.T, src, dst string, asBF16 bool) {
	t.Helper()
	for _, f := range []string{"config.json", "generation_config.json"} {
		if b, err := os.ReadFile(filepath.Join(src, f)); err == nil {
			if err := os.WriteFile(filepath.Join(dst, f), b, 0o644); err != nil {
				t.Fatal(err)
			}
		}
	}
	raw, err := os.ReadFile(filepath.Join(src, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	n := binary.LittleEndian.Uint64(raw)
	var hdr map[string]json.RawMessage
	if err := json.Unmarshal(raw[8:8+n], &hdr); err != nil {
		t.Fatal(err)
	}
	body := raw[8+n:]
	type meta struct {
		DType   string `json:"dtype"`
		Shape   []int  `json:"shape"`
		Offsets [2]int `json:"data_offsets"`
	}
	var names []string
	for k := range hdr {
		if k != "__metadata__" {
			names = append(names, k)
		}
	}
	sort.Strings(names)
	out := map[string]any{}
	var blob []byte
	for _, k := range names {
		var m meta
		if err := json.Unmarshal(hdr[k], &m); err != nil {
			t.Fatal(err)
		}
		data := body[m.Offsets[0]:m.Offsets[1]]
		if m.DType == "F32" {
			var b []byte
			for i := range len(data) / 4 {
				h := linalg.F32ToBF16(math.Float32frombits(binary.LittleEndian.Uint32(data[4*i:])))
				if asBF16 {
					b = binary.LittleEndian.AppendUint16(b, h)
				} else {
					b = binary.LittleEndian.AppendUint32(b, math.Float32bits(linalg.BF16ToF32(h)))
				}
			}
			data = b
			if asBF16 {
				m.DType = "BF16"
			}
		}
		for len(blob)%8 != 0 {
			blob = append(blob, 0)
		}
		m.Offsets = [2]int{len(blob), len(blob) + len(data)}
		blob = append(blob, data...)
		out[k] = m
	}
	h, _ := json.Marshal(out)
	for (8+len(h))%8 != 0 {
		h = append(h, ' ')
	}
	file := binary.LittleEndian.AppendUint64(nil, uint64(len(h)))
	file = append(append(file, h...), blob...)
	if err := os.WriteFile(filepath.Join(dst, "model.safetensors"), file, 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestMEM11_lfm2BF16CheckpointKeepsItsMapping: a bf16 LFM2 checkpoint's short-conv projections are views of the file's mapping
// (stIsland), so a direct load must keep that mapping open for the model's life (islandsMayAliasSource). Released, the first
// forward reads unmapped pages. The logits equal those of the same values stored as F32.
func TestMEM11_lfm2BF16CheckpointKeepsItsMapping(t *testing.T) {
	const fixture = "../testdata/lfm2-tiny"
	dst, src := t.TempDir(), t.TempDir()
	writeBF16Checkpoint(t, fixture, dst, true)
	writeBF16Checkpoint(t, fixture, src, false)
	for _, quant := range []string{"int4", ""} {
		m, err := Load(dst, Options{Quant: quant})
		if err != nil {
			t.Fatalf("load %q: %v", quant, err)
		}
		var convs []*shortConvWeights
		for l := range m.w.Layers {
			if sc := m.w.Layers[l].shortConv; sc != nil {
				convs = append(convs, sc)
			}
		}
		if len(convs) == 0 || convs[0].inProj.Kind() != "bf16" {
			t.Fatalf("%q: the bf16 checkpoint's islands are not bf16", quant)
		}
		if m.w.st == nil {
			t.Fatalf("%q: the load released the source mapping its bf16 islands alias", quant)
		}
		got := mem11Logits(t, m)
		m.Close()

		ref, err := Load(src, Options{Quant: quant})
		if err != nil {
			t.Fatal(err)
		}
		want := mem11Logits(t, ref)
		ref.Close()
		if d := mem11SameBits(want, got); d != 0 {
			t.Errorf("%q: %d logits differ from the F32 fixture with rounded islands", quant, d)
		}
	}
}
