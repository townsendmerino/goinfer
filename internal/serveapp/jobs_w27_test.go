package serveapp

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"
)

// TestAdmission_positionIsArrivalOrder is W28's server half: a waiter's place in line, by id, is exactly its
// arrival order among the waiters (J1 is strict FIFO), moves up as the turn is handed on, and is unknown for
// an id that is not waiting — including one that has just been granted the turn.
func TestAdmission_positionIsArrivalOrder(t *testing.T) {
	var a admission
	if !a.enter(context.Background(), admissionRecord{id: "holder"}) {
		t.Fatal("holder not admitted on the fast path")
	}
	if _, _, _, ok := a.position("holder"); ok {
		t.Error("the running holder reported a queue position — it is not waiting")
	}
	granted := make(map[string]chan struct{})
	for _, id := range []string{"a", "b", "c"} {
		ch := make(chan struct{})
		granted[id] = ch
		go func(id string) {
			if a.enter(context.Background(), admissionRecord{id: id}) {
				close(granted[id])
			}
		}(id)
		// enqueue in a known order
		deadline := time.Now().Add(2 * time.Second)
		for {
			a.mu.Lock()
			n := a.waiters.Len()
			a.mu.Unlock()
			if n > len(granted)-1 || time.Now().After(deadline) {
				break
			}
			time.Sleep(time.Millisecond)
		}
	}
	for id, want := range map[string]int{"a": 1, "b": 2, "c": 3} {
		place, waiting, running, ok := a.position(id)
		if !ok || place != want || waiting != 3 || !running {
			t.Errorf("position(%s) = %d of %d running=%v ok=%v, want %d of 3 running", id, place, waiting, running, ok, want)
		}
	}
	a.release() // the turn goes to "a"
	<-granted["a"]
	if _, _, _, ok := a.position("a"); ok {
		t.Error("a was granted the turn but still reports a queue position")
	}
	if place, waiting, _, ok := a.position("c"); !ok || place != 2 || waiting != 2 {
		t.Errorf("after one hand-off, position(c) = %d of %d ok=%v, want 2 of 2", place, waiting, ok)
	}
	if _, _, _, ok := a.position("nope"); ok {
		t.Error("an unknown id reported a position")
	}
	if _, _, _, ok := a.position(""); ok {
		t.Error(`the empty id matched a waiter — requests nothing asks about must never match`)
	}
	a.release()
	<-granted["b"]
	a.release()
	<-granted["c"]
	a.release()
}

// TestNotAdmitted_queueFullIsAFailureNotACancellation pins the J3/J4 mislabel: a request the model's queue
// had no room for was recorded as "cancelled before a turn was granted".
func TestNotAdmitted_queueFullIsAFailureNotACancellation(t *testing.T) {
	live := context.Background()
	done, cancel := context.WithCancel(context.Background())
	cancel()
	for _, c := range []struct {
		name       string
		ctx        context.Context
		halt       string
		wantState  jobState
		wantStatus int
		wantReason string
	}{
		{"halted", live, "maintenance", jobCancelled, statusCancelled, "maintenance"},
		{"cancelled while waiting", done, "", jobCancelled, statusCancelled, "cancelled before a turn was granted"},
		{"queue full", live, "", jobFailed, http.StatusTooManyRequests, `model "m" queue full; retry`},
	} {
		got := notAdmitted(c.ctx, c.halt, "m")
		if got.state != c.wantState || got.status != c.wantStatus || got.reason != c.wantReason {
			t.Errorf("%s: %+v, want state %s status %d reason %q", c.name, got, c.wantState, c.wantStatus, c.wantReason)
		}
	}
}

// TestJobTerminalEvents: a job's stream ends the way /v1/chat/completions ends one — finish_reason, then
// usage — or with an error event when it failed.
func TestJobTerminalEvents(t *testing.T) {
	u := &usage{PromptTokens: 5, CompletionTokens: 7, TotalTokens: 12}
	decode := func(evs [][]byte) []map[string]any {
		var out []map[string]any
		for _, e := range evs {
			var m map[string]any
			if err := json.Unmarshal(e, &m); err != nil {
				t.Fatalf("event is not JSON: %s", e)
			}
			out = append(out, m)
		}
		return out
	}
	finishOf := func(m map[string]any) any {
		ch, _ := m["choices"].([]any)
		if len(ch) != 1 {
			return nil
		}
		return ch[0].(map[string]any)["finish_reason"]
	}

	done := decode(jobTerminalEvents(job{ID: "j", Model: "m", State: jobDone, Usage: u, result: &jobResult{Content: "hi", FinishReason: "length"}}, 1, ""))
	if len(done) != 2 || finishOf(done[0]) != "length" || done[1]["usage"].(map[string]any)["completion_tokens"] != float64(7) {
		t.Errorf("done job: %v, want a finish_reason=length chunk then a usage chunk", done)
	}
	failed := decode(jobTerminalEvents(job{ID: "j", Model: "m", State: jobFailed, Error: `model "m" queue full; retry`}, 1, "rate_limit_error"))
	if len(failed) != 1 || failed[0]["error"].(map[string]any)["message"] != `model "m" queue full; retry` || failed[0]["error"].(map[string]any)["type"] != "rate_limit_error" {
		t.Errorf("failed job: %v, want one error event carrying the reason and type", failed)
	}
	if f := decode(jobTerminalEvents(job{ID: "j", State: jobFailed, Error: "boom"}, 1, "")); f[0]["error"].(map[string]any)["type"] != "api_error" {
		t.Errorf("failed job without a type: %v, want api_error", f)
	}
	cancelled := decode(jobTerminalEvents(job{ID: "j", Model: "m", State: jobCancelled, Error: "cancelled before a turn was granted"}, 1, ""))
	if len(cancelled) != 1 || finishOf(cancelled[0]) != "cancelled" {
		t.Errorf("cancelled-before-running job: %v, want one finish_reason=cancelled chunk", cancelled)
	}
}

// TestJobs_realModelEventsQueueAndRefusal drives the real handlers on a real model (GOINFER_SERVE_MODEL): a
// finished job's re-attached stream ends with finish_reason and usage; a waiting job reports its place in
// line; a submit to a full queue is a 429 up front.
func TestJobs_realModelEventsQueueAndRefusal(t *testing.T) {
	path := os.Getenv("GOINFER_SERVE_MODEL")
	if path == "" {
		t.Skip("set GOINFER_SERVE_MODEL=<.gguf> for the jobs events/queue test")
	}
	srv, err := newServer(config{models: modelFlag{{name: "m", path: path}}, backend: "cpu", quant: "int8int8", kvSessions: 0, maxQueue: 1})
	if err != nil {
		t.Fatalf("newServer: %v", err)
	}
	mux := jobsTestMux(srv)
	ts := httptest.NewServer(mux)
	defer ts.Close()
	post := func(body string) (int, map[string]any, http.Header) {
		r, err := http.Post(ts.URL+"/v1/jobs", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		defer r.Body.Close()
		var m map[string]any
		json.NewDecoder(r.Body).Decode(&m)
		return r.StatusCode, m, r.Header
	}
	get := func(id string) map[string]any {
		r, err := http.Get(ts.URL + "/v1/jobs/" + id)
		if err != nil {
			t.Fatal(err)
		}
		defer r.Body.Close()
		var m map[string]any
		json.NewDecoder(r.Body).Decode(&m)
		return m
	}
	long := `{"model":"m","max_tokens":200,"temperature":0,"messages":[{"role":"user","content":"Count from 1 to 200."}]}`
	short := `{"model":"m","max_tokens":6,"temperature":0,"messages":[{"role":"user","content":"Say hi."}]}`

	_, holder, _ := post(long)
	deadline := time.Now().Add(20 * time.Second)
	for get(holder["id"].(string))["status"] != string(jobRunning) {
		if time.Now().After(deadline) {
			t.Fatal("holder never started running")
		}
		time.Sleep(20 * time.Millisecond)
	}
	_, waiter, _ := post(short)
	wid := waiter["id"].(string)
	var q map[string]any
	for time.Now().Before(deadline) {
		if m := get(wid); m["queue"] != nil {
			q = m["queue"].(map[string]any)
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if q == nil || q["position"] != float64(1) || q["waiting"] != float64(1) || q["running"] != true {
		t.Fatalf("waiting job's queue = %v, want position 1 of 1 waiting, with one running", q)
	}
	// -max-queue 1 → capacity 2 (the runner plus one waiting): both taken, so a third submit is refused now.
	code, body, hdr := post(short)
	if code != http.StatusTooManyRequests || hdr.Get("Retry-After") != "1" || !strings.Contains(body["error"].(map[string]any)["message"].(string), "queue full") {
		t.Fatalf("submit to a full queue: %d %v, want 429 queue full with Retry-After", code, body)
	}

	// let both finish, then re-attach to the short one and read its whole stream
	for get(wid)["status"] != string(jobDone) {
		if time.Now().After(deadline.Add(60 * time.Second)) {
			t.Fatalf("waiting job never finished: %v", get(wid))
		}
		time.Sleep(50 * time.Millisecond)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "GET", ts.URL+"/v1/jobs/"+wid+"/events", nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var sawFinish, sawUsage, sawDone bool
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		line := strings.TrimPrefix(sc.Text(), "data: ")
		if line == sc.Text() {
			continue
		}
		if line == "[DONE]" {
			sawDone = true
			break
		}
		var m map[string]any
		if json.Unmarshal([]byte(line), &m) != nil {
			continue
		}
		if m["usage"] != nil {
			sawUsage = true
		}
		if ch, _ := m["choices"].([]any); len(ch) == 1 {
			if f, _ := ch[0].(map[string]any)["finish_reason"].(string); f != "" {
				sawFinish = true
			}
		}
	}
	if err := sc.Err(); err != nil && !errors.Is(err, io.EOF) {
		t.Fatal(err)
	}
	if !sawFinish || !sawUsage || !sawDone {
		t.Fatalf("re-attached stream: finish=%v usage=%v [DONE]=%v, want all three", sawFinish, sawUsage, sawDone)
	}
}
