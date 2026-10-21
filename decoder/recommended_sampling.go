package decoder

import (
	"encoding/json"
	"io/fs"
	"math"
	"strconv"

	"github.com/townsendmerino/aikit/embed"
)

// RecommendedSampling is a checkpoint's own recommended sampling settings: a Hugging Face directory's
// generation_config.json, or a GGUF's general.sampling.* metadata. A bare .gguf reads only its own metadata, never a
// generation_config.json in its folder, which may hold many models. A nil field is one the checkpoint does not set, or
// sets to a value that changes nothing (top_k <= 0, top_p >= 1, a zero presence penalty, a repetition penalty of 1).
// Config carries it, so a .giw built from the checkpoint keeps it. The server applies the fields a request leaves out
// (docs/server.md, "Sampling defaults").
type RecommendedSampling struct {
	Source          string   `json:"source"`                // "generation_config.json" or "gguf general.sampling"
	Temperature     *float64 `json:"temperature,omitempty"` // 0 when the checkpoint says do_sample false
	TopP            *float64 `json:"top_p,omitempty"`
	TopK            *int     `json:"top_k,omitempty"`
	MinP            *float64 `json:"min_p,omitempty"`
	PresencePenalty *float64 `json:"presence_penalty,omitempty"`
	// RepetitionPenalty is recorded and never applied: HF's divides logits, and the sampler's penalties are OpenAI's
	// subtractive ones. The server names it as skipped.
	RepetitionPenalty *float64 `json:"repetition_penalty,omitempty"`
}

// Applied lists the fields the sampler applies, as "name value" in request-field order; RepetitionPenalty is never among
// them.
func (r *RecommendedSampling) Applied() []string {
	var out []string
	add := func(name string, p *float64) {
		if p != nil {
			out = append(out, name+" "+strconv.FormatFloat(*p, 'g', -1, 64))
		}
	}
	add("temperature", r.Temperature)
	add("top_p", r.TopP)
	if r.TopK != nil {
		out = append(out, "top_k "+strconv.Itoa(*r.TopK))
	}
	add("min_p", r.MinP)
	add("presence_penalty", r.PresencePenalty)
	return out
}

// samplingFromGenerationConfig reads name (a generation_config.json) from fsys. An absent or unparseable file, or one
// with no sampling field that changes anything, gives nil.
func samplingFromGenerationConfig(fsys fs.FS, name string) *RecommendedSampling {
	b, err := fs.ReadFile(fsys, name)
	if err != nil {
		return nil
	}
	var g struct {
		DoSample          *bool    `json:"do_sample"`
		Temperature       *float64 `json:"temperature"`
		TopP              *float64 `json:"top_p"`
		TopK              *float64 `json:"top_k"`
		MinP              *float64 `json:"min_p"`
		PresencePenalty   *float64 `json:"presence_penalty"`
		RepetitionPenalty *float64 `json:"repetition_penalty"`
	}
	if json.Unmarshal(lenientConfigJSON(b), &g) != nil {
		return nil
	}
	r := &RecommendedSampling{Source: "generation_config.json", RepetitionPenalty: g.RepetitionPenalty}
	if g.DoSample != nil && !*g.DoSample {
		greedy := 0.0
		r.Temperature = &greedy // transformers' generate is greedy whatever the other fields say
	} else {
		r.Temperature, r.TopP, r.MinP, r.PresencePenalty = g.Temperature, g.TopP, g.MinP, g.PresencePenalty
		if g.TopK != nil && *g.TopK == math.Trunc(*g.TopK) && math.Abs(*g.TopK) < 1<<31 {
			k := int(*g.TopK)
			r.TopK = &k
		}
	}
	return r.normalized()
}

// ggufSampling reads a GGUF's general.sampling.* keys (llama.cpp's names: temp, top_p, top_k, min_p, penalty_repeat).
func ggufSampling(g *embed.GGUFFile) *RecommendedSampling {
	r := &RecommendedSampling{
		Source:            "gguf general.sampling",
		Temperature:       ggufFloatKey(g, "general.sampling.temp"),
		TopP:              ggufFloatKey(g, "general.sampling.top_p"),
		MinP:              ggufFloatKey(g, "general.sampling.min_p"),
		RepetitionPenalty: ggufFloatKey(g, "general.sampling.penalty_repeat"),
	}
	switch v := g.Metadata["general.sampling.top_k"].(type) {
	case int32:
		k := int(v)
		r.TopK = &k
	case uint32:
		k := int(v)
		r.TopK = &k
	case int64:
		if v > -1<<31 && v < 1<<31 {
			k := int(v)
			r.TopK = &k
		}
	}
	return r.normalized()
}

// ggufFloatKey reads a float key. A float32 is widened through its shortest decimal form, so a stored 0.95 reads as 0.95
// and not 0.949999988.
func ggufFloatKey(g *embed.GGUFFile, key string) *float64 {
	switch v := g.Metadata[key].(type) {
	case float32:
		f, _ := strconv.ParseFloat(strconv.FormatFloat(float64(v), 'g', -1, 32), 64)
		return &f
	case float64:
		return &v
	}
	return nil
}

// normalized drops each field that is out of range or changes nothing, and returns nil when none is left.
func (r *RecommendedSampling) normalized() *RecommendedSampling {
	keep := func(p *float64, ok func(float64) bool) *float64 {
		if p == nil || math.IsNaN(*p) || math.IsInf(*p, 0) || !ok(*p) {
			return nil
		}
		return p
	}
	r.Temperature = keep(r.Temperature, func(v float64) bool { return v >= 0 })
	r.TopP = keep(r.TopP, func(v float64) bool { return v > 0 && v < 1 })
	r.MinP = keep(r.MinP, func(v float64) bool { return v > 0 && v <= 1 })
	r.PresencePenalty = keep(r.PresencePenalty, func(v float64) bool { return v != 0 })
	r.RepetitionPenalty = keep(r.RepetitionPenalty, func(v float64) bool { return v > 0 && v != 1 })
	if r.TopK != nil && *r.TopK <= 0 {
		r.TopK = nil
	}
	if r.Temperature == nil && r.TopP == nil && r.TopK == nil && r.MinP == nil && r.PresencePenalty == nil && r.RepetitionPenalty == nil {
		return nil
	}
	return r
}
