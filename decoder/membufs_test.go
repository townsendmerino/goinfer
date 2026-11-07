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
	more, err2 := filepath.Glob("selftestdata/*/config.json")
	dirs = append(dirs, more...)
	if err != nil || err2 != nil || len(dirs) == 0 {
		t.Fatalf("no testdata configs (%v, %v)", err, err2)
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
	// The towers, encoders and speech models under testdata: a decoder config that stops resolving is red, not a longer list.
	sort.Strings(unresolved)
	if want := []string{"embeddinggemma2-tiny", "gemma4-audio-tiny", "gemma4-vision-tiny", "glm-ocr-vision-tiny", "pixtral-projector-tiny",
		"qwen3vl-vision-tiny", "siglip-tiny", "whisper-tiny-rand", "whisper-tiny-ts"}; !reflect.DeepEqual(unresolved, want) {
		t.Errorf("the testdata configs that do not resolve are %v, want exactly %v", unresolved, want)
	}
	for _, fam := range []string{"DeltaNet", "short conv", "KDA", "Granite Mamba-2", "Nemotron-H Mamba-2"} {
		if !families[fam] {
			t.Errorf("no config with %s state was compared", fam)
		}
	}
}

// TestLayerShape_kvRowsAreWhatTheCPUCacheHolds is G-MP6 (docs/tasks/task-forward-architecture-2026-10.md §4.9b): every layer's declarations
// (layerBuffers, through layerShapeOf) are what the CPU walker's own cache holds after a prefill, which it sizes without the list. Per
// position: K and V of the width the walker writes, with the layer's KV heads, at the KV precision, and the window its ring keeps; MLA's
// latent row in f32; no other row. Per slot: each recurrent state entry is the size of the walker's state, its conv windows full. Each
// committed fixture is loaded with its weights and prefilled 6 positions. No committed fixture shares KV across layers, so that case is
// held on Gemma 4's representative config at four layers with the last two sharing: the two that own their KV declare K and V, the two
// that read another's declare no row.
func TestLayerShape_kvRowsAreWhatTheCPUCacheHolds(t *testing.T) {
	g4 := representativeConfig("gemma4")
	g4.NumLayers, g4.SharedKVLayers = 4, 2
	g4.LayerTypes = []string{"sliding_attention", "full_attention", "sliding_attention", "full_attention"}
	a4, _, err := resolveArchitecture(g4)
	if err != nil {
		t.Fatal(err)
	}
	for l := range a4.NumLayers {
		rows := 0
		for _, b := range layerBuffers(a4, l) {
			if b.Unit == program.PerPosition {
				rows++
			}
		}
		if owns := a4.gemma4KVSrcAt(l) == l; owns != (l < 2) || (owns && rows != 2) || (!owns && rows != 0) {
			t.Errorf("Gemma 4 layer %d (reads layer %d's KV): declares %d rows; want K and V on layers 0 and 1, none on 2 and 3", l, a4.gemma4KVSrcAt(l), rows)
		}
	}
	const n = 6 // past every fixture's conv window (at most 3 rows) and gemma2-tiny's 4-position sliding window
	rowsOf := func(w [][]float32) int {
		total := 0
		for _, r := range w {
			total += len(r)
		}
		return total
	}
	kinds := map[string]int{}
	for _, dir := range []string{"../testdata/llama-tiny", "../testdata/gemma2-tiny", "../testdata/mistral-tiny-window", "selftestdata/qwen35-tiny",
		"../testdata/deepseek-tiny", "../testdata/lfm2-tiny", "../testdata/bailing_hybrid-tiny", "../testdata/nemotron-tiny"} {
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
			declared, held := map[string]int{}, map[string]int{}
			for _, b := range layerBuffers(a, l) {
				if _, again := declared[b.Name]; again {
					t.Errorf("%s layer %d declares %q twice", dir, l, b.Name)
				}
				declared[b.Name] = b.Elems
				switch {
				case b.Unit == program.PerPosition && (b.Name == "k" || b.Name == "v"):
					win := 0
					if l < len(cache.rings) && cache.rings[l] != nil {
						win = cache.rings[l].w
					}
					if b.Prec != program.PrecKV || b.Heads != a.kvHeadsAt(l) || b.Window != win {
						t.Errorf("%s layer %d: %q is at precision %d with %d heads and window %d; want the KV precision, %d heads and the ring's %d", dir, l,
							b.Name, b.Prec, b.Heads, b.Window, a.kvHeadsAt(l), win)
					}
				case b.Unit == program.PerPosition && b.Name == "latent":
					if b.Prec != program.PrecF32 {
						t.Errorf("%s layer %d: the latent row is at precision %d, not f32", dir, l, b.Prec)
					}
				case b.Unit == program.PerPosition:
					t.Errorf("%s layer %d declares a row %q the cache has no counterpart for", dir, l, b.Name)
				}
			}
			if ck, cv, base := cache.LayerKV(l); len(ck) > 0 && n > base {
				held["k"], held["v"] = len(ck)/(n-base), len(cv)/(n-base)
			}
			if l < len(cache.mlaLatent) && len(cache.mlaLatent[l]) > 0 {
				held["latent"] = len(cache.mlaLatent[l]) / n
			}
			if l < len(cache.delta) && cache.delta[l] != nil {
				held["window"], held["state"] = rowsOf(cache.delta[l].convWin), len(cache.delta[l].s)
				kinds["DeltaNet"]++
			}
			if l < len(cache.mamba) && cache.mamba[l] != nil {
				held["window"], held["state"] = rowsOf(cache.mamba[l].ConvWin), len(cache.mamba[l].SSM)
				kinds["Mamba-2"]++
			}
			if l < len(cache.conv) && cache.conv[l] != nil {
				held["window"] = rowsOf(cache.conv[l].convWin)
				kinds["short conv"]++
			}
			if l < len(cache.kda) && cache.kda[l] != nil {
				k := cache.kda[l]
				held["window-q"], held["window-k"], held["window-v"], held["state"] = rowsOf(k.convWinQ), rowsOf(k.convWinK), rowsOf(k.convWinV), len(k.s)
				kinds["KDA"]++
			}
			if !reflect.DeepEqual(declared, held) {
				t.Errorf("%s layer %d: declares %v; the CPU cache holds %v", dir, l, declared, held)
			}
			switch {
			case declared["latent"] > 0:
				kinds["latent"]++
			case declared["k"] > 0:
				kinds["k/v"]++
			}
		}
		m.Close()
	}
	t.Logf("layers compared: %v", kinds)
	for _, k := range []string{"k/v", "latent", "DeltaNet", "Mamba-2", "short conv", "KDA"} {
		if kinds[k] == 0 {
			t.Errorf("no %s layer compared", k)
		}
	}
}

// TestMM5_handTable is G-MP3 (docs/tasks/task-forward-architecture-2026-10.md §4.9b): on six fixtures, Metal's KV per 64-row block, the
// recurrent state per slot and its checkpoint per slot are written here as arithmetic on each config's numbers, and the plan's figures
// (metalKVParts' null block, stateBuffers, recurrentCheckpointBuffers) must equal them. Each row checks the config numbers it uses, so a
// fixture that changes is red, not stale.
func TestMM5_handTable(t *testing.T) {
	for _, c := range []struct {
		dir, kv                     string // kv: the precision Metal delivers the cache at
		numbers                     func(*Config) bool
		perBlock, state, checkpoint int64
	}{
		// four attention layers, K and V of 2 heads × 16, f16
		{"../testdata/llama-tiny", kvPrecF16, func(c *Config) bool { return c.NumLayers == 4 && c.NumKVHeads == 2 && c.HeadDim == 16 },
			4 * 2 * (2 * 16) * 2 * 64, 0, 0},
		// four MLA layers, one latent row of 16 + 8, f32 (Metal keeps an MLA cache in f32)
		{"../testdata/deepseek-tiny", kvPrecF32, func(c *Config) bool { return c.NumLayers == 4 && c.KVLoRARank == 16 && c.QKRopeHeadDim == 8 },
			4 * (16 + 8) * 4 * 64, 0, 0},
		// one full-attention layer's K and V; three DeltaNet layers' conv window (3 rows of 2·32 + 64) and state (4 × 16 × 16), f32, all
		// checkpointed
		{"selftestdata/qwen35-tiny", kvPrecF16, func(c *Config) bool {
			return c.NumLayers == 4 && c.NumKVHeads == 2 && c.HeadDim == 16 && c.LinearConvKernelDim == 4 && c.LinearNumKeyHeads == 2 &&
				c.LinearKeyHeadDim == 16 && c.LinearNumValueHeads == 4 && c.LinearValueHeadDim == 16
		}, 1 * 2 * (2 * 16) * 2 * 64, 3 * ((4-1)*(2*(2*16)+4*16) + 4*16*16) * 4, 3 * ((4-1)*(2*(2*16)+4*16) + 4*16*16) * 4},
		// two sliding and two full layers, each holding the whole context on Metal
		{"../testdata/gemma2-tiny", kvPrecF16, func(c *Config) bool { return c.NumLayers == 4 && c.NumKVHeads == 2 && c.HeadDim == 32 },
			4 * 2 * (2 * 32) * 2 * 64, 0, 0},
		// one attention layer (2 heads × 64/4); three short-conv layers' window of 2 rows of 64, which no checkpoint copies
		{"../testdata/lfm2-tiny", kvPrecF16, func(c *Config) bool {
			return c.NumLayers == 4 && c.NumKVHeads == 2 && c.HiddenDim == 64 && c.NumHeads == 4 && c.ConvLCache == 3
		}, 1 * 2 * (2 * 16) * 2 * 64, 3 * (3 - 1) * 64 * 4, 0},
		// two attention layers; two Mamba-2 layers' conv window (3 rows of 8·8 + 2·1·16) and SSM state (8 × 8 × 16), checkpointed
		{"../testdata/nemotron-tiny", kvPrecF16, func(c *Config) bool {
			return c.NumKVHeads == 2 && c.HeadDim == 16 && c.MambaNumHeads == 8 && c.MambaHeadDim == 8 && c.SSMStateSize == 16 && c.NGroups == 1 &&
				c.ConvKernel == 4
		}, 2 * 2 * (2 * 16) * 2 * 64, 2 * ((4-1)*(8*8+2*1*16) + 8*8*16) * 4, 2 * ((4-1)*(8*8+2*1*16) + 8*8*16) * 4},
	} {
		m, err := Load(c.dir, Options{Backend: "cpu", Quant: "f32"})
		if err != nil {
			t.Fatalf("%s: %v", c.dir, err)
		}
		if !c.numbers(m.Config()) {
			t.Errorf("%s: the config no longer has the numbers the table is written from", c.dir)
		}
		if d := m.KVDeliveredFor("metal"); d != c.kv {
			t.Errorf("%s: Metal delivers %s, the table is %s", c.dir, d, c.kv)
		}
		if _, perBlock := m.metalKVParts(4096); perBlock != c.perBlock {
			t.Errorf("%s: the plan prices %d B of KV a block; the hand table %d", c.dir, perBlock, c.perBlock)
		}
		sum := func(bs []int64) (n int64) {
			for _, b := range bs {
				n += b
			}
			return n
		}
		if state, ckpt := sum(stateBuffers(m.w.arch)), sum(recurrentCheckpointBuffers(m.w.arch)); state != c.state || ckpt != c.checkpoint {
			t.Errorf("%s: the plan prices %d B of state and %d B of checkpoint a slot; the hand table %d and %d", c.dir, state, ckpt, c.state, c.checkpoint)
		}
		m.Close()
	}
}
