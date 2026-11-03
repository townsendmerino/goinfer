package decoder

import (
	"reflect"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// loadSharedKVFixture is a tiny Gemma 4 with its last layer marked KV-shared, as an E-model's are: the E-model fixture is not tracked, and
// the sharing is a property of the architecture, which the geometry and the plan both read. Its two layers are made one attention type,
// or the shared one would have no source.
func loadSharedKVFixture(t *testing.T) *Model {
	t.Helper()
	m, err := Load("../testdata/gemma4-moe-kv-tiny", Options{Backend: "cpu", DirectLoad: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	a := m.w.arch
	a.gemma4.SharedKVLayers, a.layerIsGlobal = 1, func(int) bool { return true }
	if a.NumLayers != 2 || a.gemma4KVSrcAt(1) != 0 || a.gemma4KVSrcAt(0) != 0 {
		t.Fatalf("VOID: %d layers, sources %d and %d: want layer 1 sharing layer 0's KV", a.NumLayers, a.gemma4KVSrcAt(0), a.gemma4KVSrcAt(1))
	}
	return m
}

// TestResidentKV_sharedLayersOwnNone: a Gemma 4 E-model's KV-shared layers alias an earlier layer's cache on the CPU (forward_gemma4's p.KVSrc),
// so the plan's KV holds to the cache's geometry. TestResidentKV_webgpuSharedLayersOwnNone is the same on WebGPU's build.
func TestResidentKV_sharedLayersOwnNone(t *testing.T) {
	m := loadSharedKVFixture(t)
	const ctx = 8
	var held int64
	for _, g := range m.w.arch.KVGeometry().Groups {
		held += 2 * int64(g.RowWidth) * 4 * ctx * int64(len(g.Layers)) // K and V, f32
	}
	if got := m.ResidentKVBytes("cpu", ctx, false, false); got != held {
		t.Errorf("CPU: priced %d bytes of KV at %d positions, the cache's geometry holds %d", got, ctx, held)
	}
}

// TestResidentDense_countsEveryDeltaNetTensor: every f32 tensor of a DeltaNet layer is in the dense figure, since every one is uploaded on
// mem.LineDense. Emptying one takes exactly its bytes off, and a new []float32 field fails the count below until it is listed.
func TestResidentDense_countsEveryDeltaNetTensor(t *testing.T) {
	fields := map[string]func(*deltaNetWeights) *[]float32{
		"inProjB": func(d *deltaNetWeights) *[]float32 { return &d.inProjB },
		"inProjA": func(d *deltaNetWeights) *[]float32 { return &d.inProjA },
		"convW":   func(d *deltaNetWeights) *[]float32 { return &d.convW },
		"dtBias":  func(d *deltaNetWeights) *[]float32 { return &d.dtBias },
		"negExpA": func(d *deltaNetWeights) *[]float32 { return &d.negExpA },
		"normW":   func(d *deltaNetWeights) *[]float32 { return &d.normW },
	}
	rt := reflect.TypeFor[deltaNetWeights]()
	slices := 0
	for i := range rt.NumField() {
		if rt.Field(i).Type == reflect.TypeFor[[]float32]() {
			slices++
		}
	}
	if slices != len(fields) {
		t.Fatalf("deltaNetWeights has %d []float32 fields, this test lists %d", slices, len(fields))
	}
	for name, f := range fields {
		m, err := Load("../testdata/qwen3next-tiny", Options{Backend: "cpu", DirectLoad: true})
		if err != nil {
			t.Fatal(err)
		}
		before := m.ResidentDenseWeightBytes()
		var want int64
		for l := range m.w.Layers {
			if d := m.w.Layers[l].delta; d != nil {
				s := f(d)
				want += int64(len(*s)) * 4
				*s = nil
			}
		}
		if want == 0 {
			t.Fatalf("VOID: %s is empty on every layer", name)
		}
		if got := before - m.ResidentDenseWeightBytes(); got != want {
			t.Errorf("%s: emptying it took %d bytes off the dense figure, it holds %d", name, got, want)
		}
		m.Close()
	}
}

// TestResidentDense_mambaProjectionsAsUploaded holds the plan's price for Mamba-2 projections an f32 load holds to decoder's QuantizeInt8 and
// wmBytes, on each resident backend, and the CPU to f32. It runs no backend's uploader: MM7's gate holds the uploads to the price.
func TestResidentDense_mambaProjectionsAsUploaded(t *testing.T) {
	m, err := Load("../testdata/nemotron-tiny", Options{Backend: "cpu", Quant: "f32", DirectLoad: true})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	hidden := m.w.Cfg.HiddenDim
	var want int64
	for l := range m.w.Layers {
		if mb := m.w.Layers[l].mamba; mb != nil {
			if mb.inQ.Rows() != 0 {
				t.Fatal("VOID: an f32 load holds int8 projections")
			}
			in := linalg.QuantizeInt8(mb.inProj, len(mb.inProj)/hidden, hidden, false)
			out := linalg.QuantizeInt8(mb.outProj, hidden, len(mb.outProj)/hidden, false)
			want += int64(len(mb.inProj)+len(mb.outProj))*4 - wmBytes(&in) - wmBytes(&out)
		}
	}
	if want <= 0 {
		t.Fatal("VOID: no Mamba-2 projection to compare")
	}
	base := m.ResidentDenseWeightBytes()
	structs, _ := m.cpuStructBytes()
	for be, saved := range map[string]int64{"metal": want, "webgpu": want, "cuda": want + m.residentHostSideBytes("cuda"), "cpu": -structs} {
		if got := base - m.ResidentDenseWeightBytesFor(be); got != saved {
			t.Errorf("%s: the dense figure is %d under the tensors' sum, want %d", be, got, saved)
		}
	}
	if a, b := m.ResidentNeedBytes("cpu", 0, 64, false, false)-m.ResidentKVBytes("cpu", 64, false, false),
		m.ResidentNeedBytes("webgpu", 0, 64, false, false)-m.ResidentKVBytes("webgpu", 64, false, false); a-b != want {
		t.Errorf("WebGPU's need is %d under the CPU's beside KV, want %d", a-b, want)
	}
}
