package decoder

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/townsendmerino/aikit/embed"
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

func loraTensor(st *embed.SafetensorsFile, name string) ([]float32, []int, error) {
	t, err := st.Tensor(name)
	if err != nil {
		return nil, nil, err
	}
	if len(t.Shape) != 2 {
		return nil, nil, fmt.Errorf("decoder(lora): %q rank %d, want 2", name, len(t.Shape))
	}
	var data []float32
	switch t.DType {
	case "F32":
		data, err = t.Float32s()
	case "BF16":
		data, err = t.BFloat16sToF32()
	case "F16":
		data, err = t.Float16sToF32()
	default:
		return nil, nil, fmt.Errorf("decoder(lora): %q dtype %q unsupported (F32/BF16/F16)", name, t.DType)
	}
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
	for i := 0; i < numLayers; i++ {
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
	for o := 0; o < out; o++ {
		brow := d.b[o*d.r : o*d.r+d.r]
		row := data[o*in : o*in+in]
		for k := 0; k < d.r; k++ {
			bk := sc * brow[k]
			if bk == 0 {
				continue
			}
			arow := d.a[k*in : k*in+in]
			for j := 0; j < in; j++ {
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
