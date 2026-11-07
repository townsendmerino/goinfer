package decoder

import (
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"testing"
)

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
// checkpoint, on every registered model type's representative config and on every fixture under testdata with a config.
func TestStateSpecs_fromTheOperations(t *testing.T) {
	withState := 0
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
		if layers > 0 {
			withState++
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
			continue
		}
		if a, _, err := resolveArchitecture(cfg); err == nil {
			check(mt, a)
		}
	}
	dirs, err := filepath.Glob("../testdata/*/config.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range dirs {
		dir := filepath.Dir(p)
		cfg, err := loadConfig(os.DirFS(dir), "config.json")
		if err != nil {
			continue
		}
		if a, _, err := resolveArchitecture(cfg); err == nil {
			check(filepath.Base(dir), a)
		}
	}
	t.Logf("%d configs with recurrent or conv state compared", withState)
	if withState < 5 {
		t.Fatalf("only %d configs carry state; the test reads too little to hold the five families' declarations", withState)
	}
}
