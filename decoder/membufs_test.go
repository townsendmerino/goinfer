package decoder

import (
	"context"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// TestMM5_noFamilySwitchInPricing is G-MP4 (docs/tasks/task-forward-architecture-2026-10.md §4.9b): Metal's KV pricing and the state lists
// read the operations' declarations, so none of them names a mixer, a family or a family's parameters. CUDA's KV pricing (cudaKVParts)
// joins the list with CUDA's half.
func TestMM5_noFamilySwitchInPricing(t *testing.T) {
	banned := []string{"ResidentLayerOps", "Mixer", "KVSrcAtResident", "MLAResidentParams", "HeadDimAtResident", "KVHeadsAtResident",
		".qwen35", ".lfm2", ".kda", ".granite", ".nemotron", ".gemma4", ".mla", "isLinearLayer", "isConvLayer", "isMambaLayer"}
	for file, funcs := range map[string][]string{
		"residentneed.go": {"metalKVParts", "MetalKVRowBytes"},
		"statebytes.go":   {"stateSpecs", "stateBuffers", "recurrentCheckpointBuffers"},
	} {
		src, err := os.ReadFile(file)
		if err != nil {
			t.Fatal(err)
		}
		fs := token.NewFileSet()
		f, err := parser.ParseFile(fs, file, src, 0)
		if err != nil {
			t.Fatal(err)
		}
		found := map[string]bool{}
		for _, d := range f.Decls {
			fn, ok := d.(*ast.FuncDecl)
			if !ok || fn.Body == nil {
				continue
			}
			for _, name := range funcs {
				if fn.Name.Name != name {
					continue
				}
				found[name] = true
				body := string(src[fs.Position(fn.Body.Pos()).Offset:fs.Position(fn.Body.End()).Offset])
				for _, b := range banned {
					if strings.Contains(body, b) {
						t.Errorf("%s: %s names %q: KV and state pricing reads the operations' declarations (LayerBuffers), not the family", file, name, b)
					}
				}
			}
		}
		for _, name := range funcs {
			if !found[name] {
				t.Errorf("%s: no func %s to scan; the scan reads nothing for it", file, name)
			}
		}
	}
}

// stateSpecsByFamily is the family-keyed state list the operations' declarations replace: the oracle TestStateSpecs_fromTheOperations
// holds them to.
func stateSpecsByFamily(a *Architecture, l int) []StateSpec {
	mamba := func(p mamba2Params) []StateSpec {
		return []StateSpec{{"window", (p.DConv - 1) * p.mixer().ConvDim()}, {"state", p.NHeads * p.HeadDim * p.DState}}
	}
	switch {
	case a.qwen35 != nil && a.isLinearLayer(l):
		p := a.qwen35
		keyDim, valueDim := p.NumKeyHeads*p.KeyHeadDim, p.NumValueHeads*p.ValueHeadDim
		return []StateSpec{{"window", (p.ConvKernel - 1) * (2*keyDim + valueDim)}, {"state", p.NumValueHeads * p.KeyHeadDim * p.ValueHeadDim}}
	case a.lfm2 != nil && a.isConvLayer(l):
		return []StateSpec{{"window", (a.lfm2.ConvLCache - 1) * a.lfm2.ConvDim}}
	case a.kda != nil && a.isLinearLayer(l):
		p := a.kda
		win := (p.ConvKernel - 1) * p.NumHeads * p.HeadDim
		return []StateSpec{{"window-q", win}, {"window-k", win}, {"window-v", win}, {"state", p.NumHeads * p.HeadDim * p.HeadDim}}
	case a.granite != nil && a.isMambaLayer(l):
		g := a.granite
		return mamba(mamba2Params{NHeads: g.NHeads, HeadDim: g.HeadDim, DState: g.DState, NGroups: g.NGroups, DConv: g.DConv})
	case a.nemotron != nil && a.nemotron.blockKind[l] == nemoMamba:
		n := a.nemotron
		return mamba(mamba2Params{NHeads: n.NHeads, HeadDim: n.HeadDim, DState: n.DState, NGroups: n.NGroups, DConv: n.DConv})
	}
	return nil
}

// checkpointByFamily is the family-keyed checkpoint list: every state buffer, except on a KDA or short-conv model, which keeps none.
func checkpointByFamily(a *Architecture) []int64 {
	if a.kda != nil || a.lfm2 != nil {
		return nil
	}
	var out []int64
	for l := range a.NumLayers {
		for _, s := range stateSpecsByFamily(a, l) {
			if s.Floats > 0 {
				out = append(out, int64(s.Floats)*4)
			}
		}
	}
	return out
}

// TestStateSpecs_fromTheOperations is MM5's state half (docs/tasks/task-forward-architecture-2026-10.md §4.9b): every family's state,
// derived from its layers' operations (layerBuffers, program.Buffers), is the list the family switch gave, layer by layer, and so is the
// checkpoint, on every registered model type's representative config (each must resolve) and on every fixture under testdata whose config
// resolves (the others are named). Each of the five families with state is compared, by name.
func TestStateSpecs_fromTheOperations(t *testing.T) {
	families := map[string]bool{}
	check := func(name string, a *Architecture) {
		t.Helper()
		layers := 0
		for l := range a.NumLayers {
			got, want := stateSpecs(a, l), stateSpecsByFamily(a, l)
			if !reflect.DeepEqual(got, want) {
				t.Errorf("%s layer %d: the operations declare %v, the family switch %v", name, l, got, want)
			}
			if len(want) > 0 {
				layers++
			}
		}
		if got, want := recurrentCheckpointBuffers(a), checkpointByFamily(a); !reflect.DeepEqual(got, want) {
			t.Errorf("%s: the checkpoint from the operations is %v, the family switch's %v", name, got, want)
		}
		if layers == 0 {
			return
		}
		for fam, on := range map[string]bool{"DeltaNet": a.qwen35 != nil, "short conv": a.lfm2 != nil, "KDA": a.kda != nil, "Granite Mamba-2": a.granite != nil,
			"Nemotron-H Mamba-2": a.nemotron != nil} {
			if on {
				families[fam] = true
			}
		}
	}
	keys := make([]string, 0, len(registry))
	for k := range registry {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, mt := range keys {
		cfg := representativeConfig(mt)
		if cfg == nil {
			t.Errorf("model type %q has no representative config", mt)
			continue
		}
		a, _, err := resolveArchitecture(cfg)
		if err != nil {
			t.Errorf("model type %q: %v", mt, err)
			continue
		}
		check(mt, a)
	}
	dirs, err := filepath.Glob("../testdata/*/config.json")
	if err != nil || len(dirs) == 0 {
		t.Fatalf("no testdata configs (%v)", err)
	}
	var unresolved []string
	for _, p := range dirs {
		dir := filepath.Dir(p)
		cfg, err := loadConfig(os.DirFS(dir), "config.json")
		var a *Architecture
		if err == nil {
			a, _, err = resolveArchitecture(cfg)
		}
		if err != nil {
			unresolved = append(unresolved, filepath.Base(dir)) // a tower, an encoder or a family the decoder does not load
			continue
		}
		check(filepath.Base(dir), a)
	}
	t.Logf("%d testdata configs do not resolve to a decoder architecture: %v", len(unresolved), unresolved)
	for _, fam := range []string{"DeltaNet", "short conv", "KDA", "Granite Mamba-2", "Nemotron-H Mamba-2"} {
		if !families[fam] {
			t.Errorf("no config with %s state was compared", fam)
		}
	}
}

// TestLayerShape_kvRowsAreWhatTheCPUCacheHolds: every layer's per-position rows (layerBuffers, through layerShapeOf) are what the CPU's own
// cache holds after a prefill: K and V of the width the walker writes on a layer that owns them, MLA's latent row on an MLA layer, and
// nothing on a recurrent or KV-shared layer. Each fixture is loaded with its weights and prefilled 6 positions.
func TestLayerShape_kvRowsAreWhatTheCPUCacheHolds(t *testing.T) {
	const n = 6
	kinds := map[string]int{}
	for _, dir := range []string{"../testdata/llama-tiny", "../testdata/gemma2-tiny", "../testdata/mistral-tiny-window", "selftestdata/qwen35-tiny",
		"../testdata/deepseek-tiny", "../testdata/gemma4-emodel-tiny", "../testdata/lfm2-tiny", "../testdata/bailing_hybrid-tiny"} {
		m, err := Load(dir, Options{Backend: "cpu", Quant: "f32"})
		if err != nil {
			t.Fatalf("%s: %v", dir, err)
		}
		a := m.w.arch
		ids := make([]int, n)
		for i := range ids {
			ids[i] = (i*7 + 1) % m.Config().VocabSize
		}
		cache := m.NewCache(n)
		if _, err := m.prefillLogits(context.Background(), ids, cache); err != nil {
			m.Close()
			t.Fatalf("%s: prefill: %v", dir, err)
		}
		for l := range a.NumLayers {
			var k, v, lat int
			for _, b := range layerBuffers(a, l) {
				if b.Unit != program.PerPosition {
					continue
				}
				switch b.Name {
				case "k":
					k = b.Elems
				case "v":
					v = b.Elems
				case "latent":
					lat = b.Elems
				}
			}
			var gotK, gotV, gotLat int
			if ck, cv, base := cache.LayerKV(l); len(ck) > 0 && n > base {
				gotK, gotV = len(ck)/(n-base), len(cv)/(n-base)
			}
			if l < len(cache.mlaLatent) && len(cache.mlaLatent[l]) > 0 {
				gotLat = len(cache.mlaLatent[l]) / n
			}
			if k != gotK || v != gotV || lat != gotLat {
				t.Errorf("%s layer %d: declares K %d, V %d, latent %d; the CPU cache holds K %d, V %d, latent %d a position", dir, l, k, v, lat, gotK, gotV, gotLat)
			}
			switch {
			case lat > 0:
				kinds["latent"]++
			case k > 0:
				kinds["k/v"]++
			default:
				kinds["none"]++
			}
		}
		m.Close()
	}
	t.Logf("layers compared: %v", kinds)
	if kinds["latent"] == 0 || kinds["k/v"] == 0 || kinds["none"] == 0 {
		t.Fatalf("layers compared %v: each kind (K/V, MLA latent, none) must appear", kinds)
	}
}
