package decoder

import (
	"encoding/json"
	"os"
	"path/filepath"
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

// TestHostFitKV_sharedLayersOwnNone: an E-model's KV at the host fit is the plan's figure on the planted fixture, which loads; and, through what the fit
// guard calls, a config that names shared layers prices only the layers that own a cache.
func TestHostFitKV_sharedLayersOwnNone(t *testing.T) {
	m := loadSharedKVFixture(t)
	const ctx = 3 // under the fixture's window, so a sliding layer holds ctx rows and no mirror
	if got, want := kvBytesForCtx(m.w.arch, ctx, false, false), m.ResidentKVBytes("cpu", ctx, false, false); got != want {
		t.Errorf("the host fit prices %d bytes of KV at %d positions, the plan %d", got, ctx, want)
	}
	raw, err := os.ReadFile("../testdata/gemma4-moe-kv-tiny/config.json")
	if err != nil {
		t.Fatal(err)
	}
	var doc map[string]any
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	doc["num_hidden_layers"] = 4
	doc["layer_types"] = []string{"sliding_attention", "full_attention", "sliding_attention", "full_attention"}
	estimate := func(shared int) int64 {
		doc["num_kv_shared_layers"] = shared
		out, err := json.Marshal(doc)
		if err != nil {
			t.Fatal(err)
		}
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "config.json"), out, 0o644); err != nil {
			t.Fatal(err)
		}
		cfg, err := censusConfig(dir, true)
		if err != nil {
			t.Fatal(err)
		}
		return estimateKVBytes(cfg, ctx, false, false)
	}
	const sliding, global = 2 * 16, 1 * 512 // the config's KV heads times head_dim on each kind of layer
	if got, want := estimate(0), int64(2*4*ctx*2*(sliding+global)); got != want {
		t.Errorf("no layer shared: the fit prices %d, want %d", got, want)
	}
	if got, want := estimate(2), int64(2*4*ctx*(sliding+global)); got != want {
		t.Errorf("the last two layers shared: the fit prices %d, want %d", got, want)
	}
}

// TestNewCache_reservesEachLayersWidth: a cache reserves each append-forever layer at its own K/V width, so a wider layer holds its context with no
// growth, past where the model's kvDim would have run out, and a layer that shares another's cache reserves nothing. Only f32 needs this: Gemma 4's
// attention is not the kernel a quantized cache requires, whether or not it has MoE, and the test holds that.
func TestNewCache_reservesEachLayersWidth(t *testing.T) {
	const ctx, tokens = 64, 8
	m, err := Load("../testdata/gemma4-moe-kv-tiny", Options{Backend: "cpu", DirectLoad: true})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	a := m.w.arch
	const wide = 1 // the fixture's full-attention layer, wider than the model's kvDim
	kvDim := a.NumKVHeads * a.HeadDim
	if a.ringLayerAt(wide) || a.kvDimAt(wide) == kvDim {
		t.Fatalf("VOID: layer %d is a ring, or no wider than the model's kvDim", wide)
	}
	if tokens*a.kvDimAt(wide) <= ctx*kvDim {
		t.Fatalf("VOID: %d tokens fit the %d floats a kvDim reservation holds", tokens, ctx*kvDim)
	}
	c := m.NewCache(ctx)
	want := ctx * a.kvDimAt(wide)
	if cap(c.keys[wide]) != want || cap(c.vals[wide]) != want {
		t.Errorf("f32: layer %d reserves %d and %d floats, want %d", wide, cap(c.keys[wide]), cap(c.vals[wide]), want)
	}
	for i := range tokens {
		if _, err := m.forward(i+1, c); err != nil {
			t.Fatal(err)
		}
	}
	if len(c.keys[wide]) != tokens*a.kvDimAt(wide) {
		t.Fatalf("VOID: layer %d holds %d floats after %d tokens", wide, len(c.keys[wide]), tokens)
	}
	if cap(c.keys[wide]) != want || cap(c.vals[wide]) != want {
		t.Errorf("f32: %d forwards left layer %d at %d and %d floats, not its %d", tokens, wide, cap(c.keys[wide]), cap(c.vals[wide]), want)
	}

	if a.attnOrder() == attnBatchedAcc64 {
		t.Error("Gemma 4's attention is the batched kernel, so it can quantize its cache without MoE: setQuant must then reserve each layer at its own width")
	}
	shared := loadSharedKVFixture(t) // a separate load: it marks layer 1 shared, and the checks above read it owning
	if c := shared.NewCache(ctx); cap(c.keys[1]) != 0 || cap(c.vals[1]) != 0 {
		t.Errorf("a shared layer reserves %d and %d floats", cap(c.keys[1]), cap(c.vals[1]))
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
		m.ResidentNeedBytes("webgpu", 0, 64, false, false)-m.ResidentKVBytes("webgpu", 64, false, false); a-b != want+structs {
		t.Errorf("WebGPU's need is %d under the CPU's beside KV, want %d", a-b, want+structs)
	}
}
