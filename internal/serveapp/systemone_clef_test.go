package serveapp

import (
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/clef"
)

// Route C of /v1/systemone (D13), end to end through the HTTP handler on the committed tiny Clef pipeline (a tiny qwen3_5 backbone and a tiny head, the same
// fixture internal/clef gates against the official reference to 1.8e-7). The tiny fixture's stand-in tokenizer is a 4-byte-chunk hash, defined below to match
// internal/clef's. Everything it reads is committed, so a missing file FAILS.

const clefTinyDir = "../clef/testdata/tiny"
const clefTinyBackbone = "../../decoder/testdata/qwen3_5-tiny-normw"

func chunkTok(text string, _ bool) ([]int, error) {
	b := []byte(text)
	var out []int
	for i := 0; i < len(b); i += 4 {
		v := 0
		for _, c := range b[i:min(i+4, len(b))] {
			v = (v*131 + int(c)) % 256
		}
		out = append(out, v)
	}
	return out, nil
}

func clefServer(t *testing.T) (*httptest.Server, *server) {
	t.Helper()
	bb, err := decoder.Load(clefTinyBackbone, decoder.Options{})
	if err != nil {
		t.Fatalf("backbone (committed): %v", err)
	}
	t.Cleanup(func() { _ = bb.Close() })
	head, err := clef.LoadHead(clefTinyDir)
	if err != nil {
		t.Fatalf("head (committed): %v", err)
	}
	cm, err := clef.NewModel(bb, head, chunkTok, 0)
	if err != nil {
		t.Fatal(err)
	}
	lm := &loadedModel{name: "clef", model: bb, clef: cm}
	srv := &server{models: map[string]*loadedModel{"clef": lm}}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/systemone", srv.handleSystemOne)
	mux.HandleFunc("GET /v1/models", srv.handleModels)
	ts := httptest.NewServer(mux)
	t.Cleanup(ts.Close)
	return ts, srv
}

type clefGoldenItem struct {
	Name    string          `json:"name"`
	Request json.RawMessage `json:"request"`
	Probs   [][]float64     `json:"probs"`
	Qs      []struct {
		ID        string   `json:"id"`
		Type      int      `json:"type"`
		OptionIDs []string `json:"option_ids"`
	} `json:"questions"`
}

func clefGolden(t *testing.T) []clefGoldenItem {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(clefTinyDir, "golden.json"))
	if err != nil {
		t.Fatalf("no golden (committed): %v", err)
	}
	var g struct{ Items []clefGoldenItem }
	if err := json.Unmarshal(raw, &g); err != nil || len(g.Items) < 4 {
		t.Fatalf("golden: %v (%d items)", err, len(g.Items))
	}
	return g.Items
}

// withModelField adds "model":"clef" to a request object.
func withModelField(t *testing.T, req json.RawMessage) string {
	t.Helper()
	var m map[string]json.RawMessage
	if err := json.Unmarshal(req, &m); err != nil {
		t.Fatal(err)
	}
	m["model"] = json.RawMessage(`"clef"`)
	b, _ := json.Marshal(m)
	return string(b)
}

// Each golden request, through POST /v1/systemone, answers in the reference's shape with the reference's probabilities (rounded to four decimals as the
// reference rounds them): noul = P(true); a choice's probabilities by option in the REQUEST's order (the head's order is alphabetical, so a mapping by position
// instead of by id would scramble them), its choice the first maximum in that order, confidence the top probability; a score's expected level.
func TestSystemOne_clefMatchesTheReference(t *testing.T) {
	ts, _ := clefServer(t)
	for _, it := range clefGolden(t) {
		code, raw := postSystemOne(t, ts, withModelField(t, it.Request))
		if code != 200 {
			t.Fatalf("%s: %d %s", it.Name, code, raw)
		}
		var resp struct {
			Model   string                     `json:"model"`
			Answers map[string]json.RawMessage `json:"answers"`
			Usage   map[string]int             `json:"usage"`
			Goinfer map[string]any             `json:"goinfer"`
		}
		if err := json.Unmarshal(raw, &resp); err != nil {
			t.Fatal(err)
		}
		if resp.Model != "clef" || resp.Usage["output_tokens"] != 0 || resp.Usage["input_tokens"] <= 0 {
			t.Errorf("%s: model %q usage %v", it.Name, resp.Model, resp.Usage)
		}
		if resp.Goinfer["route"] != "clef" || resp.Goinfer["confidence"] != "top_probability" || resp.Goinfer["calibrated"] != false {
			t.Errorf("%s: goinfer block %v", it.Name, resp.Goinfer)
		}
		var req struct {
			Questions map[string]struct {
				Type     string          `json:"type"`
				Criteria json.RawMessage `json:"criteria"`
			} `json:"questions"`
		}
		_ = json.Unmarshal(it.Request, &req)
		for qi, q := range it.Qs {
			want := map[string]float64{}
			for o, id := range q.OptionIDs {
				want[id] = it.Probs[qi][o]
			}
			var a map[string]any
			if err := json.Unmarshal(resp.Answers[q.ID], &a); err != nil {
				t.Fatalf("%s %s: %v", it.Name, q.ID, err)
			}
			typ := req.Questions[q.ID].Type
			if a["type"] != typ {
				t.Errorf("%s %s: type %v, want %s", it.Name, q.ID, a["type"], typ)
			}
			switch typ {
			case "noul":
				if a["noul"].(float64) != round4(want["true"]) {
					t.Errorf("%s %s: noul %v, want %v", it.Name, q.ID, a["noul"], round4(want["true"]))
				}
			case "choice":
				keys, _, _ := orderedObject(req.Questions[q.ID].Criteria)
				probs := a["probabilities"].(map[string]any)
				best, bp := "", -1.0
				for _, k := range keys {
					if probs[k].(float64) != round4(want[k]) {
						t.Errorf("%s %s option %s: %v, want %v", it.Name, q.ID, k, probs[k], round4(want[k]))
					}
					if want[k] > bp {
						best, bp = k, want[k]
					}
				}
				if a["choice"] != best || a["confidence"].(float64) != round4(bp) {
					t.Errorf("%s %s: choice %v confidence %v, want %s and the top probability %v", it.Name, q.ID, a["choice"], a["confidence"], best, round4(bp))
				}
			case "score":
				exp, top := 0.0, 0.0
				for i := range q.OptionIDs {
					exp += float64(i) * want[q.OptionIDs[i]]
					top = math.Max(top, want[q.OptionIDs[i]])
				}
				if a["score"].(float64) != round4(exp) || a["confidence"].(float64) != round4(top) {
					t.Errorf("%s %s: score %v confidence %v, want %v and %v", it.Name, q.ID, a["score"], a["confidence"], round4(exp), round4(top))
				}
				if len(a["legend"].(map[string]any)) != len(q.OptionIDs) {
					t.Errorf("%s %s: legend %v", it.Name, q.ID, a["legend"])
				}
			}
		}
	}
}

// The answers come back in the order the questions were sent, not the head's or a map's.
func TestSystemOne_clefKeepsQuestionOrder(t *testing.T) {
	ts, _ := clefServer(t)
	body := `{"model":"clef","state":"s","questions":{"zeta":{"type":"noul","instructions":"Z?"},"alpha":{"type":"noul","instructions":"A?"},"mid":{"type":"noul","instructions":"M?"}}}`
	code, raw := postSystemOne(t, ts, body)
	if code != 200 {
		t.Fatalf("%d %s", code, raw)
	}
	s := string(raw)
	iz, ia, im := strings.Index(s, `"zeta"`), strings.Index(s, `"alpha"`), strings.Index(s, `"mid"`)
	if !(iz >= 0 && iz < ia && ia < im) {
		t.Errorf("answers are not in request order (zeta %d, alpha %d, mid %d): %s", iz, ia, im, s)
	}
}

// Python's max() returns the first maximum, so equal probabilities pick the option listed first in the REQUEST, whatever order the head scored them in.
func TestClefAnswer_firstMaximumInRequestOrderWins(t *testing.T) {
	a := clef.Answer{ID: "q", Type: 1, Options: []string{"a", "b", "c"}, Probs: []float64{0.4, 0.4, 0.2}}
	got, err := clefAnswer(clefQuestion{name: "q", typ: "choice", keys: []string{"b", "a", "c"}}, a)
	if err != nil {
		t.Fatal(err)
	}
	if c := got.(map[string]any)["choice"]; c != "b" {
		t.Errorf("a tie between a and b, request order b, a, c: chose %v, want b", c)
	}
	if _, err := clefAnswer(clefQuestion{name: "q", typ: "choice", keys: []string{"x"}}, a); err == nil {
		t.Error("an option the model did not score was accepted")
	}
}

func TestSystemOne_clefRefusals(t *testing.T) {
	ts, _ := clefServer(t)
	for name, tc := range map[string]struct {
		body string
		want int
	}{
		"no state":         {`{"model":"clef","questions":{"q":{"type":"noul"}}}`, 422},
		"images":           {`{"model":"clef","state":"s","images":["x"],"questions":{"q":{"type":"noul"}}}`, 422},
		"unknown type":     {`{"model":"clef","state":"s","questions":{"q":{"type":"rank"}}}`, 422},
		"choice no crit":   {`{"model":"clef","state":"s","questions":{"q":{"type":"choice"}}}`, 422},
		"score as object":  {`{"model":"clef","state":"s","questions":{"q":{"type":"score","criteria":{"a":"b"}}}}`, 422},
		"questions array":  {`{"model":"clef","state":"s","questions":[{"type":"noul"}]}`, 422},
		"no questions":     {`{"model":"clef","state":"s","questions":{}}`, 422},
		"unknown model":    {`{"model":"nope","state":"s","questions":{"q":{"type":"noul","instructions":"Q?"}}}`, 404},
		"duplicate option": {`{"model":"clef","state":"s","questions":{"q":{"type":"choice","criteria":{"a":null,"a":null}}}}`, 422},
	} {
		if code, raw := postSystemOne(t, ts, tc.body); code != tc.want {
			t.Errorf("%s: %d (want %d): %s", name, code, tc.want, raw)
		}
	}
}

// /v1/models says what this entry answers with, including which confidence it reports.
func TestModels_clefAdvertisesItsRoute(t *testing.T) {
	ts, _ := clefServer(t)
	resp, err := http.Get(ts.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var ml struct {
		Data []struct {
			ID        string         `json:"id"`
			Decisions map[string]any `json:"decisions"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&ml); err != nil || len(ml.Data) != 1 {
		t.Fatalf("decode: %v (%d entries)", err, len(ml.Data))
	}
	d := ml.Data[0].Decisions
	if d["route"] != "clef" || d["confidence"] != "top_probability" || d["endpoint"] != "/v1/systemone" {
		t.Errorf("decisions field: %v", d)
	}
}

func TestIsClefDir_andItsQuantDefault(t *testing.T) {
	if !isClefDir(clefTinyDir) || isClefDir(clefTinyBackbone) || isClefDir(t.TempDir()) || isClefDir("/does/not/exist") {
		t.Error("isClefDir does not tell a directory with joint_head.safetensors from one without")
	}
	if q := (modelSpec{path: clefTinyDir}).options(config{}).Quant; q != decoder.DecisionHeadQuant {
		t.Errorf("a Clef entry loads at %q, want the decision models' default %q", q, decoder.DecisionHeadQuant)
	}
	if q := (modelSpec{path: clefTinyBackbone}).options(config{}).Quant; q == decoder.DecisionHeadQuant && decoder.DecisionHeadQuant != "" {
		t.Errorf("an ordinary model was given the decision default %q", q)
	}
}

// A choice's probabilities are listed in the order the options were sent (red, blue, amber), although the head scores them alphabetically (amber, blue, red).
func TestSystemOne_clefChoiceProbabilitiesInRequestOrder(t *testing.T) {
	ts, _ := clefServer(t)
	body := `{"model":"clef","state":"s","questions":{"q":{"type":"choice","instructions":"Pick","criteria":{"red":"r","blue":null,"amber":"a"}}}}`
	code, raw := postSystemOne(t, ts, body)
	if code != 200 {
		t.Fatalf("%d %s", code, raw)
	}
	s := string(raw)
	i := strings.Index(s, `"probabilities"`)
	r, b, a := strings.Index(s[i:], `"red"`), strings.Index(s[i:], `"blue"`), strings.Index(s[i:], `"amber"`)
	if !(i >= 0 && r >= 0 && r < b && b < a) {
		t.Errorf("probabilities are not in request order (red %d, blue %d, amber %d): %s", r, b, a, s)
	}
}
