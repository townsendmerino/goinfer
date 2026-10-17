package serveapp

import "github.com/townsendmerino/goinfer/decoder"

// withRecommended fills each sampling field sm leaves out from the model's recommended sampling rs, and names the fields
// it filled, for -log-requests. A field the client sent always wins. min_p has no request field, so a recommended one
// applies whenever it is set. A request that asks for greedy (temperature 0) is left as sent: no other field changes it.
func withRecommended(sm sampling, rs *decoder.RecommendedSampling) (sampling, []string) {
	if rs == nil || (sm.Temperature != nil && *sm.Temperature == 0) {
		return sm, nil
	}
	var filled []string
	fill := func(name string, dst **float64, src *float64) {
		if *dst == nil && src != nil {
			v := *src
			*dst = &v
			filled = append(filled, name)
		}
	}
	fill("temperature", &sm.Temperature, rs.Temperature)
	fill("top_p", &sm.TopP, rs.TopP)
	if sm.TopK == nil && rs.TopK != nil {
		k := *rs.TopK
		sm.TopK = &k
		filled = append(filled, "top_k")
	}
	fill("min_p", &sm.minP, rs.MinP)
	fill("presence_penalty", &sm.PresencePenalty, rs.PresencePenalty)
	return sm, filled
}
