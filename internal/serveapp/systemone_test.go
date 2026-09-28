package serveapp

import (
	"bytes"
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/decide"
	"github.com/townsendmerino/goinfer/internal/loadflags"
)

// sysOneTok gives every verbalizer its own single id (noul, digits, letters) and any prompt three ids.
type sysOneTok struct{ ids map[string]int }

func (f sysOneTok) EncodePlain(s string) ([]int, error) {
	if id, ok := f.ids[s]; ok {
		return []int{id}, nil
	}
	return []int{1, 2, 3}, nil
}
func (f sysOneTok) EncodeChat(string) ([]int, error) { return []int{1, 2, 3, 4}, nil }

// systemOneServer serves the tiny fixture with an injected decider whose logits favour: noul true (p ≈ 0.8), choice
// B, score 1 — known answers for the wire checks, with no tokenizer or model forward involved.
func systemOneServer(t *testing.T) *httptest.Server {
	t.Helper()
	srv, lm := tinyServed(t)
	tok := sysOneTok{ids: map[string]int{}}
	for i, v := range append(append([]string{"false", "true"}, strings.Split("0123456789", "")...), strings.Split("ABCDEFGHIJKLMNOP", "")...) {
		tok.ids[v] = 10 + i
	}
	logits := make([]float32, 64)
	logits[tok.ids["true"]] = float32(math.Log(4)) // P(true) = 4/5
	logits[tok.ids["B"]] = 2
	logits[tok.ids["1"]] = 2
	d, err := decide.New(tok, func(context.Context, []int) ([]float32, error) { return logits, nil }, decide.Options{Template: decide.TemplateChat})
	if err != nil {
		t.Fatal(err)
	}
	lm.deciderOnce.Do(func() { lm.decider = d })
	// The tiny fixture's context is 128 tokens and it has no tokenizer, so the pre-tokenize byte guard would assume one
	// byte per token and refuse a jevx-sized request; give it a real vocabulary's longest-token bound.
	lm.maxTokBytesOnce.Do(func() { lm.maxTokBytes = 64 })
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/systemone", srv.handleSystemOne)
	ts := httptest.NewServer(mux)
	t.Cleanup(ts.Close)
	return ts
}

func postSystemOne(t *testing.T, ts *httptest.Server, body string) (int, []byte) {
	t.Helper()
	resp, err := http.Post(ts.URL+"/v1/systemone", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var b bytes.Buffer
	b.ReadFrom(resp.Body)
	return resp.StatusCode, b.Bytes()
}

// The request as jevx builds it (docs/measurements/decisions-d0-prior-art-2026-09-27.md §4: {model, state,
// questions: {name: {type, instructions, criteria}}}), and the reply checked by jevx's own fail-closed rules: exactly
// the questions asked, noul P in [0,1], every probability in [0,1] and each set summing to 0.98-1.02, a choice among
// the offered keys — plus the usage object TypeSafe's Python SDK requires, and the options in the order sent.
func TestSystemOne_jevxShape(t *testing.T) {
	ts := systemOneServer(t)
	code, body := postSystemOne(t, ts, `{"model":"tiny","state":"Help! My payouts have been failing for 3 days.","questions":{
		"is_urgent":{"type":"noul","instructions":"Is this urgent?","criteria":{"true":"blocks work now","false":"can wait"}},
		"department":{"type":"choice","instructions":"Which team should handle this?","criteria":{"sales":"Pricing","billing":"Payments, invoicing, refunds","technical":null}},
		"frustration":{"type":"score","instructions":"How frustrated is the customer?","criteria":["Calm","Frustrated","Very angry"]}}}`)
	if code != 200 {
		t.Fatalf("status %d: %s", code, body)
	}
	var out struct {
		Model   string                     `json:"model"`
		Answers map[string]json.RawMessage `json:"answers"`
		Usage   *struct {
			In  *int `json:"input_tokens"`
			Out *int `json:"output_tokens"`
		} `json:"usage"`
		Goinfer map[string]any `json:"goinfer"`
	}
	if err := json.Unmarshal(body, &out); err != nil {
		t.Fatal(err)
	}
	if out.Model != "tiny" || out.Usage == nil || out.Usage.In == nil || out.Usage.Out == nil || *out.Usage.In <= 0 {
		t.Errorf("model %q usage %+v", out.Model, out.Usage)
	}
	if len(out.Answers) != 3 {
		t.Fatalf("%d answers, want exactly the 3 questions: %s", len(out.Answers), body)
	}
	var noul struct {
		Type string
		Noul float64
	}
	json.Unmarshal(out.Answers["is_urgent"], &noul)
	if noul.Type != "noul" || math.Abs(noul.Noul-0.8) > 1e-6 {
		t.Errorf("is_urgent %s, want noul 0.8", out.Answers["is_urgent"])
	}
	var choice struct {
		Type          string
		Choice        string
		Probabilities map[string]float64
		Confidence    float64
	}
	json.Unmarshal(out.Answers["department"], &choice)
	if choice.Type != "choice" || choice.Choice != "billing" { // B = the second option sent
		t.Errorf("department %s, want billing (letter B)", out.Answers["department"])
	}
	checkProbs(t, "department", choice.Probabilities, []string{"sales", "billing", "technical"})
	if !(choice.Confidence >= 0 && choice.Confidence <= 1) {
		t.Errorf("department confidence %v", choice.Confidence)
	}
	probs := out.Answers["department"][bytes.Index(out.Answers["department"], []byte(`"probabilities"`)):]
	if s, b, x := bytes.Index(probs, []byte(`"sales"`)), bytes.Index(probs, []byte(`"billing"`)), bytes.Index(probs, []byte(`"technical"`)); !(s >= 0 && s < b && b < x) {
		t.Errorf("department probabilities are not in the order sent (sales, billing, technical): %s", probs)
	}
	var score struct {
		Type          string
		Score         float64
		Legend        map[string]string
		Probabilities map[string]float64
		Confidence    float64
	}
	json.Unmarshal(out.Answers["frustration"], &score)
	if score.Type != "score" || score.Legend["0"] != "Calm" || score.Legend["2"] != "Very angry" || !(score.Score > 0.5 && score.Score < 1.5) {
		t.Errorf("frustration %s", out.Answers["frustration"])
	}
	checkProbs(t, "frustration", score.Probabilities, []string{"0", "1", "2"})
	if out.Goinfer["route"] != "label" || out.Goinfer["template"] != "chat-v1" {
		t.Errorf("goinfer block %v", out.Goinfer)
	}
}

func checkProbs(t *testing.T, name string, p map[string]float64, keys []string) {
	t.Helper()
	if len(p) != len(keys) {
		t.Errorf("%s: probabilities over %v, want %v", name, p, keys)
	}
	var sum float64
	for _, k := range keys {
		v, ok := p[k]
		if !ok || v < 0 || v > 1 {
			t.Errorf("%s: p(%s) = %v (present %v)", name, k, v, ok)
		}
		sum += v
	}
	if sum < 0.98 || sum > 1.02 {
		t.Errorf("%s: probabilities sum to %v (jevx requires 0.98-1.02)", name, sum)
	}
}

func TestSystemOne_rejects(t *testing.T) {
	ts := systemOneServer(t)
	seventeen := `{`
	for i := 0; i < 17; i++ {
		if i > 0 {
			seventeen += ","
		}
		seventeen += `"o` + string(rune('a'+i)) + `":null`
	}
	seventeen += `}`
	for _, c := range []struct {
		name, body string
		code       int
	}{
		{"no state", `{"model":"tiny","questions":{"q":{"type":"noul","instructions":"?"}}}`, 422},
		{"no questions", `{"model":"tiny","state":"s","questions":{}}`, 422},
		{"unknown type", `{"model":"tiny","state":"s","questions":{"q":{"type":"rank","instructions":"?"}}}`, 422},
		{"no instructions", `{"model":"tiny","state":"s","questions":{"q":{"type":"noul"}}}`, 422},
		{"17 choice options", `{"model":"tiny","state":"s","questions":{"q":{"type":"choice","instructions":"?","criteria":` + seventeen + `}}}`, 422},
		{"choice without criteria", `{"model":"tiny","state":"s","questions":{"q":{"type":"choice","instructions":"?"}}}`, 422},
		{"a 1-level score", `{"model":"tiny","state":"s","questions":{"q":{"type":"score","instructions":"?","criteria":["only"]}}}`, 422},
		{"an 11-level score", `{"model":"tiny","state":"s","questions":{"q":{"type":"score","instructions":"?","criteria":["0","1","2","3","4","5","6","7","8","9","10"]}}}`, 422},
		{"an unknown model", `{"model":"jev-latest","state":"s","questions":{"q":{"type":"noul","instructions":"?"}}}`, 404},
		{"not JSON", `{`, 400},
	} {
		if code, body := postSystemOne(t, ts, c.body); code != c.code {
			t.Errorf("%s: status %d, want %d (%s)", c.name, code, c.code, body)
		}
	}
}

func TestTypesafeConfidence(t *testing.T) {
	for _, c := range []struct {
		p    float64
		n    int
		want float64
	}{{0.8, 3, 0.7}, {1.0 / 3, 3, 0}, {1, 3, 1}, {0.5, 2, 0}, {0.9, 2, 0.8}, {0.2, 3, 0}} {
		if got := typesafeConfidence(c.p, c.n); math.Abs(got-c.want) > 1e-9 {
			t.Errorf("typesafeConfidence(%v, %d) = %v, want %v", c.p, c.n, got, c.want)
		}
	}
}

func TestJSONText(t *testing.T) {
	for raw, want := range map[string]string{`"a b"`: "a b", `{"x": 1}`: `{"x":1}`, `[1, 2]`: `[1,2]`, `null`: "", ``: ""} {
		if got := jsonText(json.RawMessage(raw)); got != want {
			t.Errorf("jsonText(%s) = %q, want %q", raw, got, want)
		}
	}
}

// End to end on a real model (the local Qwen 0.5B; skips without it): the decider is built from the served model and
// its tokenizer, and a two-question request comes back in TypeSafe's shape.
func TestSystemOne_realModel(t *testing.T) {
	path := filepath.Join("..", "..", "testdata", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no local checkpoint at %s (an untracked symlink): %v", path, err)
	}
	srv, err := newServer(config{models: modelFlag{{name: "m", path: path}}, load: loadflags.Flags{Backend: "cpu", Quant: "int4", DirectLoad: true}, kvSessions: 4})
	if err != nil {
		t.Fatalf("newServer: %v", err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/systemone", srv.handleSystemOne)
	ts := httptest.NewServer(mux)
	defer ts.Close()
	code, body := postSystemOne(t, ts, `{"model":"m","state":"The customer wrote: the app crashes every time I open settings.","questions":{
		"refund":{"type":"noul","instructions":"Is the customer asking for a refund?"},
		"team":{"type":"choice","instructions":"Which team should handle this?","criteria":{"billing":null,"engineering":"bugs and crashes","sales":null}}}}`)
	if code != 200 {
		t.Fatalf("status %d: %s", code, body)
	}
	if !bytes.Contains(body, []byte(`"noul":`)) || !bytes.Contains(body, []byte(`"choice":`)) {
		t.Errorf("reply %s", body)
	}
	t.Logf("%s", body)
}
