package serveapp

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/townsendmerino/goinfer/internal/decide"
)

// D8 at the endpoint: a request's questions go to the model as ONE many-prompt call, and the response is the one the questions answered one at a time give. The
// model is a stub (tinyServed's tokenizer-less fixture), so this pins the wiring; the sharing itself and its numerics are decoder.TestPromptHiddenMany_matchesSeparate's.
func TestSystemOne_questionsShareOneModelCall(t *testing.T) {
	srv, lm := tinyServed(t)
	tok := sysOneTok{ids: map[string]int{}}
	for i, v := range append(append([]string{"false", "true"}, strings.Split("0123456789", "")...), strings.Split("ABCDEFGHIJKLMNOP", "")...) {
		tok.ids[v] = 10 + i
	}
	logits := make([]float32, 64)
	logits[tok.ids["true"]] = float32(math.Log(4))
	logits[tok.ids["B"]] = 2
	logits[tok.ids["1"]] = 2
	var single, many, prompts atomic.Int64
	d, err := decide.New(tok, func(context.Context, []int) ([]float32, error) { single.Add(1); return logits, nil },
		decide.Options{Template: decide.TemplateChat, PrefillMany: func(_ context.Context, ps [][]int) ([][]float32, error) {
			many.Add(1)
			prompts.Add(int64(len(ps)))
			out := make([][]float32, len(ps))
			for i := range out {
				out[i] = logits
			}
			return out, nil
		}})
	if err != nil {
		t.Fatal(err)
	}
	lm.deciderOnce.Do(func() { lm.decider = d })
	lm.maxTokBytesOnce.Do(func() { lm.maxTokBytes = 64 })
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/systemone", srv.handleSystemOne)
	ts := httptest.NewServer(mux)
	defer ts.Close()

	body := `{"model":"tiny","state":"one state","questions":{
	  "urgent":{"type":"noul","instructions":"Is it urgent?"},
	  "team":{"type":"choice","instructions":"Who?","criteria":{"a":"A","b":"B","c":"C"}},
	  "anger":{"type":"score","instructions":"How angry?","criteria":["0","1","2","3"]}}}`
	code, raw := postSystemOne(t, ts, body)
	if code != 200 {
		t.Fatalf("%d %s", code, raw)
	}
	if many.Load() != 1 || prompts.Load() != 3 || single.Load() != 0 {
		t.Fatalf("the questions made %d many-calls over %d prompts and %d single calls, want 1, 3 and 0", many.Load(), prompts.Load(), single.Load())
	}
	var resp struct {
		Answers map[string]map[string]any `json:"answers"`
		Usage   map[string]int            `json:"usage"`
	}
	if err := json.Unmarshal(raw, &resp); err != nil {
		t.Fatal(err)
	}
	// The same three questions one request each, through the same decider: the answers must be identical.
	for name, q := range map[string]string{
		"urgent": `{"type":"noul","instructions":"Is it urgent?"}`,
		"team":   `{"type":"choice","instructions":"Who?","criteria":{"a":"A","b":"B","c":"C"}}`,
		"anger":  `{"type":"score","instructions":"How angry?","criteria":["0","1","2","3"]}`,
	} {
		code, one := postSystemOne(t, ts, `{"model":"tiny","state":"one state","questions":{"`+name+`":`+q+`}}`)
		if code != 200 {
			t.Fatalf("%s alone: %d %s", name, code, one)
		}
		var r1 struct {
			Answers map[string]map[string]any `json:"answers"`
		}
		_ = json.Unmarshal(one, &r1)
		a, b := resp.Answers[name], r1.Answers[name]
		aj, _ := json.Marshal(a)
		bj, _ := json.Marshal(b)
		if string(aj) != string(bj) {
			t.Errorf("%s: shared answer %s, alone %s", name, aj, bj)
		}
	}
	if resp.Usage["input_tokens"] <= 0 || resp.Usage["output_tokens"] != 0 {
		t.Errorf("usage %v", resp.Usage)
	}
}
