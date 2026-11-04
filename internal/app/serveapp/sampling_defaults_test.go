package serveapp

import (
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

func fptr(v float64) *float64 { return &v }
func iptr(v int) *int         { return &v }

// TestPrepare_recommendedSampling: a field the request leaves out takes the model's recommended value, a field it sends
// wins, a greedy request is left as sent, and a model with none keeps the server default (temperature 1.0, no filter).
// gr.fromModel names exactly the fields that came from the model.
func TestPrepare_recommendedSampling(t *testing.T) {
	_, lm := tinyServed(t)
	qwen := &decoder.RecommendedSampling{Source: "generation_config.json", Temperature: fptr(0.7), TopP: fptr(0.8), TopK: iptr(20),
		MinP: fptr(0.05), RepetitionPenalty: fptr(1.05)}
	type want struct {
		temp, topP, minP float64
		topK             int
		from             []string
	}
	for _, c := range []struct {
		name string
		rs   *decoder.RecommendedSampling
		sm   sampling
		want want
	}{
		{"nothing sent", qwen, sampling{}, want{0.7, 0.8, 0.05, 20, []string{"temperature", "top_p", "top_k", "min_p"}}},
		{"temperature sent", qwen, sampling{Temperature: fptr(0.2)}, want{0.2, 0.8, 0.05, 20, []string{"top_p", "top_k", "min_p"}}},
		{"all sent", qwen, sampling{Temperature: fptr(0.9), TopP: fptr(0.5), TopK: iptr(3)}, want{0.9, 0.5, 0.05, 3, []string{"min_p"}}},
		{"greedy sent", qwen, sampling{Temperature: fptr(0)}, want{0, 0, 0, 0, nil}},
		{"model recommends none", nil, sampling{}, want{1.0, 0, 0, 0, nil}},
		{"model recommends greedy", &decoder.RecommendedSampling{Source: "generation_config.json", Temperature: fptr(0)}, sampling{},
			want{0, 0, 0, 0, []string{"temperature"}}},
	} {
		lm.sampling = c.rs
		gr, err := lm.prepare(c.sm, []int{1, 2, 3}, false)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		got := want{gr.sp.Temperature, gr.sp.TopP, gr.sp.MinP, gr.sp.TopK, gr.fromModel}
		if got.temp != c.want.temp || got.topP != c.want.topP || got.minP != c.want.minP || got.topK != c.want.topK || !slices.Equal(got.from, c.want.from) {
			t.Errorf("%s: got %+v, want %+v", c.name, got, c.want)
		}
	}
	if *qwen.Temperature != 0.7 {
		t.Errorf("prepare wrote through to the model's recommended sampling: temperature now %v", *qwen.Temperature)
	}
}

// TestSamplingLine: the startup line names the values and their source, the skipped repetition_penalty, or says there
// are none and what a request then gets; the model banner carries it.
func TestSamplingLine(t *testing.T) {
	for _, c := range []struct {
		rs   *decoder.RecommendedSampling
		want string
	}{
		{nil, "sampling: the model recommends none; a request that sends no sampling fields gets temperature 1.0 over the whole vocabulary"},
		{&decoder.RecommendedSampling{Source: "generation_config.json", Temperature: fptr(0.7), TopP: fptr(0.8), TopK: iptr(20), RepetitionPenalty: fptr(1.05)},
			"sampling: generation_config.json recommends temperature 0.7, top_p 0.8, top_k 20, applied to the fields a request leaves out; " +
				"not applied: repetition_penalty 1.05 (it divides logits; the sampler's penalties are subtractive)"},
		{&decoder.RecommendedSampling{Source: "gguf general.sampling", Temperature: fptr(1), TopP: fptr(0.95), TopK: iptr(64)},
			"sampling: gguf general.sampling recommends temperature 1, top_p 0.95, top_k 64, applied to the fields a request leaves out"},
		{&decoder.RecommendedSampling{Source: "generation_config.json", RepetitionPenalty: fptr(1.1)},
			"sampling: generation_config.json recommends nothing the sampler applies; not applied: repetition_penalty 1.1 (it divides logits; the sampler's penalties are subtractive)"},
	} {
		if got := samplingLine(c.rs); got != c.want {
			t.Errorf("got  %q\nwant %q", got, c.want)
		}
	}
	rs := &decoder.RecommendedSampling{Source: "generation_config.json", TopK: iptr(20)}
	if got := bannerLine(modelBannerFrom(bannerFacts{sampling: rs}, config{}), "sampling:"); got != samplingLine(rs) {
		t.Errorf("banner sampling line %q, want %q", got, samplingLine(rs))
	}
}

// TestReqTrace_samplingFromModel: -log-requests names the fields the model supplied, and the line keeps its shape when
// none did.
func TestReqTrace_samplingFromModel(t *testing.T) {
	_, lm := tinyServed(t)
	tr := &reqTrace{start: time.Now()}
	tr.generated(lm, genRequest{promptIDs: []int{1}}, nil, nil)
	if l := tr.line("/v1/chat/completions", 200, time.Now()); strings.Contains(l, "sampling_from_model") {
		t.Errorf("a request that took nothing from the model: %q", l)
	}
	tr = &reqTrace{start: time.Now()}
	tr.generated(lm, genRequest{promptIDs: []int{1}, fromModel: []string{"temperature", "top_p", "top_k"}}, nil, nil)
	if l := tr.line("/v1/chat/completions", 200, time.Now()); !strings.HasSuffix(l, " sampling_from_model=temperature,top_p,top_k") {
		t.Errorf("a request that took three fields from the model: %q", l)
	}
}
