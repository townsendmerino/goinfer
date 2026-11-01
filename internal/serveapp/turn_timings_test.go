package serveapp

import (
	"bufio"
	"encoding/json"
	"flag"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/internal/cost"
)

var sb11bUpdate = flag.Bool("sb11b.update", false, "rewrite testdata/wire/sb11b_turn.golden.json from this run")

// turnWireServer serves one scripted model, "m", whose every generation is reply (one script token per character), on
// every route that writes a turn's usage, and on the job routes.
func turnWireServer(t *testing.T, reply string) *httptest.Server {
	t.Helper()
	tk := scriptTokenizer(t)
	ids, err := tk.Encode(reply, false)
	if err != nil {
		t.Fatal(err)
	}
	jobs, err := newJobStore("", 64)
	if err != nil {
		t.Fatal(err)
	}
	s := &server{gens: newGenerationRegistry(), jobs: jobs}
	lm := &loadedModel{tk: tk, name: "m", scripted: append([]int{}, ids...)}
	lm.tmpl = s.cfg.tuneTemplate(mellum21Template(t))
	s.models = map[string]*loadedModel{"m": lm}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", s.handleChat)
	mux.HandleFunc("POST /v1/completions", s.handleCompletions)
	mux.HandleFunc("POST /v1/responses", s.handleResponses)
	mux.HandleFunc("POST /v1/messages", s.handleMessages)
	mux.HandleFunc("POST /v1/jobs", s.handleCreateJob)
	mux.HandleFunc("GET /v1/jobs/{id}", s.handleGetJob)
	ts := httptest.NewServer(mux)
	t.Cleanup(ts.Close)
	return ts
}

// turnBody is a request on route, streamed or not, asking for its cost or not.
func turnBody(route string, stream, wantCost bool) map[string]any {
	var b map[string]any
	switch route {
	case "/v1/completions":
		b = map[string]any{"model": "m", "prompt": "Say hi.", "max_tokens": 64, "stream": stream}
	case "/v1/jobs":
		b = requestFor("/v1/chat/completions", false, false)
	default:
		b = requestFor(route, false, stream)
	}
	if stream && route != "/v1/messages" && route != "/v1/responses" {
		b["stream_options"] = map[string]any{"include_usage": true}
	}
	if wantCost {
		b["goinfer"] = map[string]any{"cost": true}
	}
	return b
}

// turnObjects posts body to route and returns every JSON object of the reply: the body, each SSE data frame, or, for a
// job, the job once finished.
func turnObjects(t *testing.T, ts *httptest.Server, route string, body map[string]any) []map[string]any {
	t.Helper()
	raw, _ := json.Marshal(body)
	resp, err := http.Post(ts.URL+route, "application/json", strings.NewReader(string(raw)))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if want := map[bool]int{true: http.StatusAccepted, false: http.StatusOK}[route == "/v1/jobs"]; resp.StatusCode != want {
		t.Fatalf("%s: status %d, want %d", route, resp.StatusCode, want)
	}
	var objs []map[string]any
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		line := strings.TrimPrefix(sc.Text(), "data: ")
		if !strings.HasPrefix(line, "{") {
			continue
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(line), &m); err != nil {
			t.Fatalf("%s: %v in %q", route, err, line)
		}
		objs = append(objs, m)
	}
	if route != "/v1/jobs" {
		return objs
	}
	id, _ := objs[0]["id"].(string)
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		r, err := http.Get(ts.URL + "/v1/jobs/" + id)
		if err != nil {
			t.Fatal(err)
		}
		var j map[string]any
		json.NewDecoder(r.Body).Decode(&j)
		r.Body.Close()
		if j["status"] == string(jobDone) {
			return []map[string]any{j}
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("job %s never finished", id)
	return nil
}

// turnShape is where a reply carries its timings and what they and the cost and usage beside them hold: the shape the
// golden pins, with every time left out.
type turnShape struct {
	At      string   `json:"at"`
	Timings []string `json:"timings"`
	Cost    []string `json:"cost"` // nil: absent; empty: present, nothing measured
	Usage   []string `json:"usage"`
	Objects int      `json:"objects_with_timings"`
}

func keysOf(v any) []string {
	m, ok := v.(map[string]any)
	if !ok {
		return nil
	}
	ks := make([]string, 0, len(m))
	for k := range m {
		ks = append(ks, k)
	}
	slices.Sort(ks)
	return ks
}

// shapeOf finds the objects in objs that carry "timings", at the top or under "response" (/v1/responses' events).
func shapeOf(t *testing.T, objs []map[string]any) (turnShape, map[string]any) {
	t.Helper()
	var sh turnShape
	var tm map[string]any
	for _, o := range objs {
		holder, at := o, ""
		if r, ok := o["response"].(map[string]any); ok && r["timings"] != nil {
			holder, at = r, "response."
		}
		v, ok := holder["timings"]
		if !ok {
			continue
		}
		sh.Objects++
		sh.At = at + "timings"
		sh.Timings = keysOf(v)
		tm, _ = v.(map[string]any)
		if c, ok := holder["cost"]; ok {
			sh.Cost = keysOf(c)
			if sh.Cost == nil {
				sh.Cost = []string{}
			}
		}
		sh.Usage = keysOf(holder["usage"])
	}
	return sh, tm
}

var turnRoutes = []string{"/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/messages", "/v1/jobs"}

// TestTurnWire_golden (G-SB11b-1): every route and mode carries one "timings" beside its usage, in one shape, and "cost" only
// when the request asks; the shapes equal the committed golden (-sb11b.update rewrites it). Times are not in the golden;
// they are checked for order instead: ttft (admission to first token) is never under prefill (start to first token).
func TestTurnWire_golden(t *testing.T) {
	ts := turnWireServer(t, "Hello there.")
	got := map[string]turnShape{}
	for _, route := range turnRoutes {
		for _, stream := range []bool{false, true} {
			if route == "/v1/jobs" && stream {
				continue // a job is one object, read by GET once it is done
			}
			for _, wantCost := range []bool{false, true} {
				name := route
				if stream {
					name += " stream"
				}
				if wantCost {
					name += " cost"
				}
				sh, tm := shapeOf(t, turnObjects(t, ts, route, turnBody(route, stream, wantCost)))
				got[name] = sh
				if sh.Objects != 1 {
					t.Errorf("%s: %d objects carry timings, want 1", name, sh.Objects)
				}
				if wantCost != (sh.Cost != nil) {
					t.Errorf("%s: cost present %v, want %v", name, sh.Cost != nil, wantCost)
				}
				if p, f := tm["prefill_ms"], tm["ttft_ms"]; p != nil && f != nil && f.(float64) < p.(float64) {
					t.Errorf("%s: ttft_ms %v under prefill_ms %v", name, f, p)
				}
			}
		}
	}
	path := filepath.Join("testdata", "wire", "sb11b_turn.golden.json")
	b, _ := json.MarshalIndent(got, "", "  ")
	b = append(b, '\n')
	if *sb11bUpdate {
		if err := os.WriteFile(path, b, 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("%v (run with -sb11b.update to write it)", err)
	}
	if string(want) != string(b) {
		t.Errorf("the turn wire shapes moved from %s:\n%s", path, b)
	}
}

// TestTurnWire_absentNotZero (G-SB11b-2): a turn that writes no token crosses no first-token boundary, so its prefill_ms,
// ttft_ms and decode_ms are absent, not 0, while queue_wait_ms is there; a request that does not ask carries no counter.
func TestTurnWire_absentNotZero(t *testing.T) {
	ts := turnWireServer(t, "")
	for _, route := range []string{"/v1/chat/completions", "/v1/messages"} {
		sh, tm := shapeOf(t, turnObjects(t, ts, route, turnBody(route, false, false)))
		if !slices.Equal(sh.Timings, []string{"queue_wait_ms"}) {
			t.Errorf("%s with no token: timings %v, want only queue_wait_ms", route, sh.Timings)
		}
		if v, ok := tm["queue_wait_ms"].(float64); !ok || v < 0 {
			t.Errorf("%s: queue_wait_ms %v", route, tm["queue_wait_ms"])
		}
	}
	if c := newTurnClock(false); c.costCounter() != nil || c.costReport() != nil {
		t.Error("a turn that did not ask has a cost counter")
	}
	if cost.FromContext(cost.WithCounter(t.Context(), newTurnClock(false).costCounter())) != nil {
		t.Error("a turn that did not ask put a counter on the generation's context")
	}
	var nilClock *turnClock
	if m := nilClock.addTurn(map[string]any{}); len(m) != 0 {
		t.Errorf("a nil clock added %v", m)
	}
}
