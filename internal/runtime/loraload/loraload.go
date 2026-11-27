// Package loraload reads a PEFT LoRA adapter (adapter_config.json + adapter_model.safetensors) into per-base-tensor
// low-rank deltas, validates them against a base model's projections, and merges them into f32 weights:
// W' = W + (alpha/r)·B·A (alpha/√r under rslora). decoder uses it for both the merge-at-load path (Options.LoRA) and
// compute-time adapters (Model.LoadAdapter); it never imports decoder.
package loraload

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
	"github.com/townsendmerino/goinfer/internal/runtime/cpuengine"
	"github.com/townsendmerino/goinfer/internal/runtime/wbuild"
)

// Delta is one base tensor's low-rank update.
type Delta = cpuengine.LoRADelta

// Adapter is a loaded PEFT adapter. Deltas maps a base tensor name (model.layers.N.…weight) to its delta; the delta data
// aliases the adapter's mmapped safetensors, so the Adapter must outlive every slice taken from it (Close releases it).
type Adapter struct {
	st     *embed.SafetensorsFile
	Deltas map[string]Delta

	// merged records each delta Merge applied, for CheckAllMerged. Layers load in parallel, hence
	// the lock.
	mu     sync.Mutex
	merged map[string]bool
}

// Load reads a PEFT adapter directory (adapter_config.json +
// adapter_model.safetensors) into per-base-tensor deltas.
func Load(dir string) (*Adapter, error) {
	cfgRaw, err := os.ReadFile(filepath.Join(dir, "adapter_config.json"))
	if err != nil {
		return nil, fmt.Errorf("decoder(lora): read adapter_config.json: %w", err)
	}
	scale, err := parseScale(cfgRaw)
	if err != nil {
		return nil, err
	}

	st, err := embed.OpenSafetensorsMmap(filepath.Join(dir, "adapter_model.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("decoder(lora): open adapter_model.safetensors: %w", err)
	}
	a := &Adapter{st: st, Deltas: map[string]Delta{}}
	for _, name := range st.Names() {
		var base string
		isA := strings.HasSuffix(name, ".lora_A.weight")
		switch {
		case isA:
			base = BaseName(name, ".lora_A.weight")
		case strings.HasSuffix(name, ".lora_B.weight"):
			base = BaseName(name, ".lora_B.weight")
		default:
			continue // lora_embedding_*, magnitude vectors, etc. — unsupported, ignored
		}
		data, shape, err := tensor(st, name)
		if err != nil {
			st.Close()
			return nil, err
		}
		d := a.Deltas[base]
		d.Scale = scale
		if isA {
			d.A, d.R, d.In = data, shape[0], shape[1]
		} else {
			d.B, d.Out = data, shape[0]
		}
		a.Deltas[base] = d
	}
	if len(a.Deltas) == 0 {
		st.Close()
		return nil, fmt.Errorf("decoder(lora): no lora_A/lora_B tensors found in %s", dir)
	}
	for base, d := range a.Deltas {
		if d.A == nil || d.B == nil {
			st.Close()
			return nil, fmt.Errorf("decoder(lora): %q has only one of lora_A/lora_B", base)
		}
		if len(d.B) != d.Out*d.R {
			st.Close()
			return nil, fmt.Errorf("decoder(lora): %q lora_B [%d,?] vs r=%d mismatch", base, d.Out, d.R)
		}
	}
	return a, nil
}

// parseScale reads adapter_config.json's rank and alpha and returns the delta scale: alpha/r, or alpha/√r under rslora.
func parseScale(cfgRaw []byte) (float64, error) {
	var cfg struct {
		R         int     `json:"r"`
		Alpha     float64 `json:"lora_alpha"`
		UseRSLoRA bool    `json:"use_rslora"`
	}
	if err := json.Unmarshal(cfgRaw, &cfg); err != nil {
		return 0, fmt.Errorf("decoder(lora): parse adapter_config.json: %w", err)
	}
	if cfg.R <= 0 || cfg.Alpha == 0 {
		return 0, fmt.Errorf("decoder(lora): adapter_config needs r>0 and lora_alpha")
	}
	if cfg.UseRSLoRA {
		return cfg.Alpha / math.Sqrt(float64(cfg.R)), nil
	}
	return cfg.Alpha / float64(cfg.R), nil
}

// BaseName maps a PEFT tensor key to the base weight name the loader uses,
// e.g. base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight →
// model.layers.0.self_attn.q_proj.weight.
func BaseName(key, suffix string) string {
	return strings.TrimSuffix(strings.TrimPrefix(key, "base_model.model."), suffix) + ".weight"
}

// tensor reads a rank-2 adapter tensor as f32 (the dtype dispatch delegated to
// the aikit embed.SafetensorsFile.TensorF32 typed read) and returns its shape — the
// shape is why this keeps a thin wrapper rather than calling TensorF32 directly.
func tensor(st *embed.SafetensorsFile, name string) ([]float32, []int, error) {
	t, err := st.Tensor(name)
	if err != nil {
		return nil, nil, err
	}
	if len(t.Shape) != 2 {
		return nil, nil, fmt.Errorf("decoder(lora): %q rank %d, want 2", name, len(t.Shape))
	}
	data, err := st.TensorF32(name) // F32/BF16/F16 → f32 (rank checked above; no shape assert)
	if err != nil {
		return nil, nil, err
	}
	return data, t.Shape, nil
}

// ValidateTargets checks that every adapter delta maps to a per-layer projection the loader will merge into, so an
// adapter targeting an unsupported module (e.g. embed_tokens) fails loudly rather than silently doing nothing.
//
// name MUST build the same tensor names the corresponding apply path looks the delta up by, or the check lies: the
// decoder's merge-at-load path looks up prefixed names (which add language_model.* and model.language_model.* on VL
// checkpoints), while the compute-time path (LayerDeltas) uses the bare wbuild.TensorName. A wrong builder lets a
// VL-prefixed base validate clean and then silently ignore the whole adapter. History: docs/code-notes/decoder.md#loraAdapter.validateTargets.
func (a *Adapter) ValidateTargets(numLayers int, s *archparts.TensorSchema, name func(layer int, suffix string) string) error {
	known := map[string]bool{}
	for i := range numLayers {
		for _, suf := range []string{s.QProj, s.KProj, s.VProj, s.OProj, s.GateProj, s.UpProj, s.DownProj} {
			if suf != "" {
				known[name(i, suf)] = true
			}
		}
	}
	for base := range a.Deltas {
		if !known[base] {
			return fmt.Errorf("decoder(lora): adapter targets %q, which merge does not support "+
				"(only the per-layer attention/MLP projections; a prefixed base needs a prefix-matching adapter)", base)
		}
	}
	return nil
}

// ValidateComputeTimeDims checks every targeted delta's declared [Out,In] shape against the actual base projection it
// will be added to, as Merge does on the merge-at-load path. LayerDeltas maps deltas onto projections by name only,
// and every resident backend's SetAdapter trusts the delta's In/Out (Metal checks only the rank), so an adapter trained
// against a different-size base would make a kernel write past its slot. This is the one chokepoint every backend's
// compute-time LoRA passes through: keep the check here, not per backend.
// History: docs/code-notes/decoder.md#loraAdapter.validateComputeTimeDims.
func (a *Adapter) ValidateComputeTimeDims(numLayers int, layers []darch.LayerWeights, s *archparts.TensorSchema, name func(layer int, suffix string) string) error {
	check := func(i int, suf string, proj *linalg.WeightMat) error {
		if suf == "" {
			return nil
		}
		base := name(i, suf)
		d, ok := a.Deltas[base]
		if !ok {
			return nil
		}
		if d.Out != proj.Rows() || d.In != proj.Cols() {
			return fmt.Errorf("decoder(lora): %q delta [%d,%d] != base [%d,%d]",
				base, d.Out, d.In, proj.Rows(), proj.Cols())
		}
		return nil
	}
	for i := range numLayers {
		l := &layers[i]
		for _, c := range []struct {
			suf  string
			proj *linalg.WeightMat
		}{
			{s.QProj, &l.QProj}, {s.KProj, &l.KProj}, {s.VProj, &l.VProj}, {s.OProj, &l.OProj},
			{s.GateProj, &l.GateProj}, {s.UpProj, &l.UpProj}, {s.DownProj, &l.DownProj},
		} {
			if err := check(i, c.suf, c.proj); err != nil {
				return err
			}
		}
	}
	return nil
}

// Merge adds this tensor's LoRA delta into the f32 weight in place:
// data[o,:] += scale·Σ_k B[o,k]·A[k,:]. No-op when the tensor isn't a target. nil-safe.
func (a *Adapter) Merge(name string, data []float32, out, in int) error {
	if a == nil {
		return nil
	}
	d, ok := a.Deltas[name]
	if !ok {
		return nil
	}
	if d.Out != out || d.In != in {
		return fmt.Errorf("decoder(lora): %q delta [%d,%d] != base [%d,%d]", name, d.Out, d.In, out, in)
	}
	a.mu.Lock()
	if a.merged == nil {
		a.merged = map[string]bool{}
	}
	a.merged[name] = true
	a.mu.Unlock()
	sc := float32(d.Scale)
	for o := range out {
		brow := d.B[o*d.R : o*d.R+d.R]
		row := data[o*in : o*in+in]
		for k := 0; k < d.R; k++ {
			bk := sc * brow[k]
			if bk == 0 {
				continue
			}
			arow := d.A[k*in : k*in+in]
			for j := range in {
				row[j] += bk * arow[j]
			}
		}
	}
	return nil
}

// CheckAllMerged refuses an adapter that has a delta the load never merged. ValidateTargets checks names against a list
// before the load; this checks them against what the load actually did, so it also covers a family whose tensor names no
// list holds, a loader that takes no adapter at all, and a name that is on the list but not loaded in that layer (a
// self_attn projection on a linear-attention layer). nil-safe.
func (a *Adapter) CheckAllMerged() error {
	if a == nil {
		return nil
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	var missed []string
	for base := range a.Deltas {
		if !a.merged[base] {
			missed = append(missed, base)
		}
	}
	if len(missed) == 0 {
		return nil
	}
	slices.Sort(missed)
	n, more := len(missed), ""
	if n > 3 {
		more = fmt.Sprintf(" and %d more", n-3)
		missed = missed[:3]
	}
	return fmt.Errorf("decoder(lora): the load did not merge %d of the adapter's %d deltas (%s%s); "+
		"this family's loader does not merge those tensors, or the checkpoint does not load them",
		n, len(a.Deltas), strings.Join(missed, ", "), more)
}

// Has reports whether the adapter has a delta for this base tensor. nil-safe.
func (a *Adapter) Has(name string) bool {
	if a == nil {
		return false
	}
	_, ok := a.Deltas[name]
	return ok
}

// Close releases the adapter's safetensors mapping. nil-safe.
func (a *Adapter) Close() {
	if a != nil && a.st != nil {
		a.st.Close()
	}
}

// LayerDeltas indexes the adapter by (layer, projection) for compute-time application, mapping each schema suffix to
// the delta under its bare wbuild.TensorName. A nil entry is a projection the adapter does not target. The deltas alias
// the adapter's mapping, so the adapter must stay open while they are in use.
func (a *Adapter) LayerDeltas(numLayers int, s *archparts.TensorSchema) []cpuengine.LoRALayerDelta {
	get := func(i int, suf string) *Delta {
		if suf == "" {
			return nil
		}
		d, ok := a.Deltas[wbuild.TensorName(i, suf)]
		if !ok {
			return nil
		}
		return &d
	}
	layers := make([]cpuengine.LoRALayerDelta, numLayers)
	for i := range numLayers {
		layers[i] = cpuengine.LoRALayerDelta{
			Q: get(i, s.QProj), K: get(i, s.KProj), V: get(i, s.VProj), O: get(i, s.OProj),
			Gate: get(i, s.GateProj), Up: get(i, s.UpProj), Down: get(i, s.DownProj),
		}
	}
	return layers
}
