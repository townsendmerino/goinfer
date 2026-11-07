package decoder

import (
	"strconv"

	"github.com/townsendmerino/aikit/embed"
)

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
	return r.Normalized()
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
