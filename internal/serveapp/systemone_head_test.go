package serveapp

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/decide"
)

const judgeTiny = "../../testdata/decisions/judge-tiny"

// TestSpecHead: head= on a --model entry loads the head and makes its unmerged adapter the entry's LoRA; a different
// lora= is refused, and so is a directory that is not a head.
func TestSpecHead(t *testing.T) {
	var m modelFlag
	if err := m.Set("jev=/some/model,head=" + judgeTiny); err != nil {
		t.Fatal(err)
	}
	o := decoder.Options{}
	h, err := specHead(m[0], &o)
	if err != nil || h == nil || o.LoRA != h.AdapterDir() || o.LoRA == "" {
		t.Fatalf("head %v, err %v, LoRA %q", h, err, o.LoRA)
	}
	if err := m.Set("jev2=/some/model,head=" + judgeTiny + ",lora=/other/adapter"); err != nil {
		t.Fatal(err)
	}
	o = m[1].options(config{})
	if _, err := specHead(m[1], &o); err == nil || !strings.Contains(err.Error(), "brings its adapter") {
		t.Fatalf("a conflicting lora=: %v", err)
	}
	if err := m.Set("x=/some/model,head=" + t.TempDir()); err != nil {
		t.Fatal(err)
	}
	if _, err := specHead(m[2], &decoder.Options{}); err == nil || !strings.Contains(err.Error(), "judge_config.json") {
		t.Fatalf("a dir with no head: %v", err)
	}
	// A .giw is taken as-is only when its sidecar records the head's adapter (prequant -lora); a bare one is refused.
	if err := m.Set("z=" + filepath.Join(t.TempDir(), "plain.giw") + ",head=" + judgeTiny); err != nil {
		t.Fatal(err)
	}
	if _, err := specHead(m[3], &decoder.Options{}); err == nil || !strings.Contains(err.Error(), "does not carry the head's adapter") {
		t.Fatalf("a .giw without the adapter: %v", err)
	}
	if err := m.Set("y=/some/model,head="); err == nil {
		t.Fatal("an empty head= was accepted")
	}
}

// TestSystemOne_head: /v1/systemone on an entry with a decision head answers by Route B, says so, names the
// questions whose descriptions the head did not read, and refuses a score the head cannot read before any prefill.
func TestSystemOne_head(t *testing.T) {
	srv, lm := tinyServed(t)
	h, err := decide.LoadHead(judgeTiny)
	if err != nil {
		t.Fatal(err)
	}
	calls := 0
	hidden := func(context.Context, []int) ([]float32, error) {
		calls++
		v := make([]float32, h.Hidden)
		for i := range v {
			v[i] = float32(i%5) - 2
		}
		return v, nil
	}
	d, err := decide.New(sysOneTok{ids: map[string]int{}}, nil, decide.Options{Template: h.Template, Head: h, Hidden: hidden})
	if err != nil {
		t.Fatal(err)
	}
	lm.head = h
	lm.deciderOnce.Do(func() { lm.decider = d })
	lm.maxTokBytesOnce.Do(func() { lm.maxTokBytes = 64 })
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/systemone", srv.handleSystemOne)
	ts := httptest.NewServer(mux)
	t.Cleanup(ts.Close)

	code, body := postSystemOne(t, ts, `{"state":"order 1043 is two days late","questions":{
		"late":{"type":"noul","instructions":"Is the order late?"},
		"team":{"type":"choice","instructions":"Which team?","criteria":{"billing":"payments","shipping":null,"other":null}},
		"urgency":{"type":"score","instructions":"How urgent?","criteria":["none","low","some","medium","high","critical"]}}}`)
	if code != http.StatusOK {
		t.Fatalf("status %d: %s", code, body)
	}
	var resp struct {
		Goinfer struct {
			Route   string   `json:"route"`
			Dropped []string `json:"descriptions_dropped"`
			Note    string   `json:"note"`
		} `json:"goinfer"`
	}
	if err := json.Unmarshal(body, &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Goinfer.Route != decide.RouteHead || strings.Join(resp.Goinfer.Dropped, ",") != "team,urgency" ||
		!strings.Contains(resp.Goinfer.Note, "decision head") {
		t.Fatalf("goinfer block %+v", resp.Goinfer)
	}
	if calls != 3 {
		t.Fatalf("%d hidden-state calls for three questions", calls)
	}

	if f := srv.decisionsField(lm.name); f == nil || f["route"] != decide.RouteHead || f["template"] != decide.TemplateBare || f["calibrated"] != true {
		t.Fatalf("/v1/models decisions field %v", f)
	}

	calls = 0
	code, body = postSystemOne(t, ts, `{"state":"s","questions":{
		"late":{"type":"noul","instructions":"Is it late?"},
		"urgency":{"type":"score","instructions":"How urgent?","criteria":["low","mid","high","max"]}}}`)
	if code != http.StatusUnprocessableEntity || !strings.Contains(string(body), "exactly 6 levels") || calls != 0 {
		t.Fatalf("a 4-level score on a head: status %d, %d prefills, %s", code, calls, body)
	}
}
