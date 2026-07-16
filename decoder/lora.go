package decoder

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
)

// LoRA adapter loading (PEFT safetensors), merged into the base weights at load:
// W' = W + (alpha/r)·B·A  (alpha/√r under rslora). The merge happens on the f32
// weight before it is quantized, so a merged model costs nothing extra at decode.
// Only the safetensors base path supports merge — PEFT adapters target HF module
// names; a GGUF base + LoRA is rejected (see Load).

type loraDelta struct {
	a          []float32 // lora_A: [r, in]
	b          []float32 // lora_B: [out, r]
	r, in, out int
	scale      float64
}

type loraAdapter struct {
	st     *embed.SafetensorsFile
	deltas map[string]loraDelta // base tensor name (model.layers.N.…weight) → delta
}

// loadLoRA reads a PEFT adapter directory (adapter_config.json +
// adapter_model.safetensors) into per-base-tensor deltas.
func loadLoRA(dir string) (*loraAdapter, error) {
	cfgRaw, err := os.ReadFile(filepath.Join(dir, "adapter_config.json"))
	if err != nil {
		return nil, fmt.Errorf("decoder(lora): read adapter_config.json: %w", err)
	}
	var cfg struct {
		R         int     `json:"r"`
		Alpha     float64 `json:"lora_alpha"`
		UseRSLoRA bool    `json:"use_rslora"`
	}
	if err := json.Unmarshal(cfgRaw, &cfg); err != nil {
		return nil, fmt.Errorf("decoder(lora): parse adapter_config.json: %w", err)
	}
	if cfg.R <= 0 || cfg.Alpha == 0 {
		return nil, fmt.Errorf("decoder(lora): adapter_config needs r>0 and lora_alpha")
	}
	scale := cfg.Alpha / float64(cfg.R)
	if cfg.UseRSLoRA {
		scale = cfg.Alpha / math.Sqrt(float64(cfg.R))
	}

	st, err := embed.OpenSafetensorsMmap(filepath.Join(dir, "adapter_model.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("decoder(lora): open adapter_model.safetensors: %w", err)
	}
	a := &loraAdapter{st: st, deltas: map[string]loraDelta{}}
	for _, name := range st.Names() {
		var base string
		isA := strings.HasSuffix(name, ".lora_A.weight")
		switch {
		case isA:
			base = loraBaseName(name, ".lora_A.weight")
		case strings.HasSuffix(name, ".lora_B.weight"):
			base = loraBaseName(name, ".lora_B.weight")
		default:
			continue // lora_embedding_*, magnitude vectors, etc. — unsupported, ignored
		}
		data, shape, err := loraTensor(st, name)
		if err != nil {
			st.Close()
			return nil, err
		}
		d := a.deltas[base]
		d.scale = scale
		if isA {
			d.a, d.r, d.in = data, shape[0], shape[1]
		} else {
			d.b, d.out = data, shape[0]
		}
		a.deltas[base] = d
	}
	if len(a.deltas) == 0 {
		st.Close()
		return nil, fmt.Errorf("decoder(lora): no lora_A/lora_B tensors found in %s", dir)
	}
	for base, d := range a.deltas {
		if d.a == nil || d.b == nil {
			st.Close()
			return nil, fmt.Errorf("decoder(lora): %q has only one of lora_A/lora_B", base)
		}
		if len(d.b) != d.out*d.r {
			st.Close()
			return nil, fmt.Errorf("decoder(lora): %q lora_B [%d,?] vs r=%d mismatch", base, d.out, d.r)
		}
	}
	return a, nil
}

// loraBaseName maps a PEFT tensor key to the base weight name the loader uses,
// e.g. base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight →
// model.layers.0.self_attn.q_proj.weight.
func loraBaseName(key, suffix string) string {
	return strings.TrimSuffix(strings.TrimPrefix(key, "base_model.model."), suffix) + ".weight"
}

// loraTensor reads a rank-2 adapter tensor as f32 (the dtype dispatch delegated to
// the aikit embed.SafetensorsFile.TensorF32 typed read) and returns its shape — the
// shape is why this keeps a thin wrapper rather than calling TensorF32 directly.
func loraTensor(st *embed.SafetensorsFile, name string) ([]float32, []int, error) {
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

// validateTargets checks every adapter delta maps to a per-layer projection the
// loader will merge into — so an adapter targeting an unsupported module (e.g.
// embed_tokens) fails loudly rather than silently no-op'ing.
func (a *loraAdapter) validateTargets(numLayers int, s *tensorSchema) error {
	known := map[string]bool{}
	for i := range numLayers {
		for _, suf := range []string{s.QProj, s.KProj, s.VProj, s.OProj, s.GateProj, s.UpProj, s.DownProj} {
			if suf != "" {
				known[tensorName(i, suf)] = true
			}
		}
	}
	for base := range a.deltas {
		if !known[base] {
			return fmt.Errorf("decoder(lora): adapter targets %q, which merge does not support "+
				"(only the per-layer attention/MLP projections)", base)
		}
	}
	return nil
}

// merge adds this tensor's LoRA delta into the f32 weight in place:
// data[o,:] += scale·Σ_k B[o,k]·A[k,:]. No-op when the tensor isn't a target.
func (a *loraAdapter) merge(name string, data []float32, out, in int) error {
	if a == nil {
		return nil
	}
	d, ok := a.deltas[name]
	if !ok {
		return nil
	}
	if d.out != out || d.in != in {
		return fmt.Errorf("decoder(lora): %q delta [%d,%d] != base [%d,%d]", name, d.out, d.in, out, in)
	}
	sc := float32(d.scale)
	for o := range out {
		brow := d.b[o*d.r : o*d.r+d.r]
		row := data[o*in : o*in+in]
		for k := 0; k < d.r; k++ {
			bk := sc * brow[k]
			if bk == 0 {
				continue
			}
			arow := d.a[k*in : k*in+in]
			for j := range in {
				row[j] += bk * arow[j]
			}
		}
	}
	return nil
}

// has reports whether the adapter has a delta for this base tensor.
func (a *loraAdapter) has(name string) bool {
	if a == nil {
		return false
	}
	_, ok := a.deltas[name]
	return ok
}

func (a *loraAdapter) close() {
	if a != nil && a.st != nil {
		a.st.Close()
	}
}

// --- Compute-time LoRA (#7): apply the low-rank delta in the forward instead of
// merging it into the base weight. Keeping the base immutable + zero-copy lets N
// adapters of one base share a single resident transformer (≈ base + N small
// deltas) rather than paying the full RAM N times over. Opt-in: merged-LoRA stays
// the faster, simpler default — this trades a little decode speed for that density.
// Only the generic dense forward (runLayersFromEmbed → causalAttention + gatedMLP)
// is wired; Model.LoadAdapter rejects the special-forward / MoE / non-gated archs.

// loraLayerDelta holds one transformer layer's per-projection deltas; a nil entry
// is a projection the adapter does not target (the apply is a no-op).
type loraLayerDelta struct {
	q, k, v, o, gate, up, down *loraDelta
}

// loraRuntime is a loaded adapter indexed for the forward: layer → projection
// deltas. The deltas alias the adapter's mmapped tensors (read-only), so it costs
// only the low-rank A/B bytes on top of the shared base.
type loraRuntime struct {
	name   string
	layers []loraLayerDelta
	a      *loraAdapter // retains the mmap so the aliased delta data stays valid
}

func (rt *loraRuntime) close() {
	if rt != nil {
		rt.a.close()
	}
}

// buildLoraRuntime indexes a loaded adapter by (layer, projection) for compute-time
// application, mapping each schema suffix to the adapter's HF-named delta.
func buildLoraRuntime(name string, a *loraAdapter, numLayers int, s *tensorSchema) *loraRuntime {
	rt := &loraRuntime{name: name, layers: make([]loraLayerDelta, numLayers), a: a}
	get := func(i int, suf string) *loraDelta {
		if suf == "" {
			return nil
		}
		d, ok := a.deltas[tensorName(i, suf)]
		if !ok {
			return nil
		}
		dd := d // a fresh addressable copy per slot (loop-var capture is irrelevant — value semantics)
		return &dd
	}
	for i := range numLayers {
		rt.layers[i] = loraLayerDelta{
			q: get(i, s.QProj), k: get(i, s.KProj), v: get(i, s.VProj), o: get(i, s.OProj),
			gate: get(i, s.GateProj), up: get(i, s.UpProj), down: get(i, s.DownProj),
		}
	}
	return rt
}

// LoadAdapter loads a PEFT LoRA adapter for compute-time application (#7) and
// registers it under name. Unlike the merge-at-load path (Options.Lora), the base
// stays immutable, so many adapters of one base share its resident weights — the
// density win. Only the generic dense forward is wired: MoE, non-gated, and the
// special-forward families (gemma4, qwen3_5_moe) are rejected, as is a
// GGUF/serialized base (the adapter is HF-named — it needs the safetensors schema).
func (m *Model) LoadAdapter(name, dir string) error {
	arch := m.w.arch
	switch {
	case m.w.schema == nil:
		return fmt.Errorf("decoder: compute-time LoRA needs a safetensors base (HF tensor names)")
	case arch.gemma4 != nil || arch.qwen35 != nil:
		return fmt.Errorf("decoder: compute-time LoRA unsupported for %s (own forward path)", arch.Name)
	case arch.MoE != nil:
		return fmt.Errorf("decoder: compute-time LoRA unsupported for MoE (expert FFN not wired)")
	case arch.NonGatedMLP:
		return fmt.Errorf("decoder: compute-time LoRA unsupported for the non-gated MLP layout (%s)", arch.Name)
	}
	a, err := loadLoRA(dir)
	if err != nil {
		return err
	}
	if err := a.validateTargets(arch.NumLayers, m.w.schema); err != nil {
		a.close()
		return err
	}
	if m.adapters == nil {
		m.adapters = map[string]*loraRuntime{}
	}
	if old := m.adapters[name]; old != nil {
		old.close()
	}
	m.adapters[name] = buildLoraRuntime(name, a, arch.NumLayers, m.w.schema)
	return nil
}

// HasAdapter reports whether a compute-time adapter is registered under name.
func (m *Model) HasAdapter(name string) bool {
	_, ok := m.adapters[name]
	return ok
}

// applyLoRA adds the low-rank delta s·B(A·x) into y in place — the compute-time
// equivalent of merge(): (W + s·B·A)·x = W·x + s·B·(A·x), where y already holds the
// base output W·x. x is the projection input ([d.in]), y the output ([d.out]).
// No-op when d is nil. Not bit-identical to merge() (the f32 reduction is split
// across two dots instead of one fused weight) but algebraically equal — the gate
// is greedy-token parity, not bit-exactness (see TestLoRACompute_matchesMerge).
func applyLoRA(d *loraDelta, x, y []float32, scr *decodeScratch) {
	if d == nil {
		return
	}
	t := scr.loraBuf(d.r)
	linalg.MatmulBT(x, d.a, t, 1, d.in, d.r) // t = A·x  (A is [r,in])
	sc := float32(d.scale)
	for o := 0; o < d.out; o++ {
		brow := d.b[o*d.r : o*d.r+d.r]
		var s float32
		for k := 0; k < d.r; k++ {
			s += brow[k] * t[k]
		}
		y[o] += sc * s
	}
}
