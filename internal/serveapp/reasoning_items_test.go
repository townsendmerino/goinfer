package serveapp

import (
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
)

// scriptedLM is scriptedServer's model with its prompts recorded, for tests that compare prompts across turns.
func scriptedLM(t *testing.T, reply string) (*httptest.Server, *[][]int, []int) {
	t.Helper()
	tk := scriptTokenizer(t)
	ids, err := tk.Encode(reply, false)
	if err != nil {
		t.Fatal(err)
	}
	var prompts [][]int
	s := &server{}
	lm := &loadedModel{tk: tk, name: "m", scripted: append([]int{}, ids...)}
	lm.onScripted = func(p []int) { prompts = append(prompts, append([]int(nil), p...)) }
	lm.tmpl = s.cfg.tuneTemplate(mellum21Template(t))
	s.models = map[string]*loadedModel{"m": lm}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/responses", s.handleResponses)
	ts := httptest.NewServer(mux)
	t.Cleanup(ts.Close)
	return ts, &prompts, ids
}

// A reply that reasons is returned with a reasoning item first: its text in content and, byte for byte, in
// encrypted_content; the message after it carries only the answer. Streamed, the reasoning item is announced, its text
// streams as reasoning_text deltas, and it is closed before the message opens.
func TestResponses_emitsReasoningItem(t *testing.T) {
	ts := scriptedServer(t, "<think>\nAdd returns a + b.\n</think>\n\n5")
	check := func(name string, out []any) {
		t.Helper()
		if len(out) != 2 {
			t.Fatalf("%s: %d output items, want reasoning + message: %v", name, len(out), out)
		}
		rs, msg := out[0].(map[string]any), out[1].(map[string]any)
		if rs["type"] != "reasoning" || msg["type"] != "message" {
			t.Fatalf("%s: output types %v, %v", name, rs["type"], msg["type"])
		}
		text, err := reasoningItemText(rs["encrypted_content"].(string), nil, nil)
		if err != nil || text != "Add returns a + b." {
			t.Errorf("%s: reasoning from encrypted_content %q (%v)", name, text, err)
		}
		if c := rs["content"].([]any)[0].(map[string]any); c["type"] != "reasoning_text" || c["text"] != "Add returns a + b." {
			t.Errorf("%s: reasoning content %v", name, c)
		}
		if got := msg["content"].([]any)[0].(map[string]any)["text"]; got != "5" {
			t.Errorf("%s: answer %q, want 5", name, got)
		}
	}
	plain := postScripted(t, ts.URL+"/v1/responses", requestFor("/v1/responses", false, false))
	check("not streamed", plain[0]["output"].([]any))
	for _, tools := range []bool{false, true} {
		evs := postScripted(t, ts.URL+"/v1/responses", requestFor("/v1/responses", tools, true))
		var types []string
		var deltas strings.Builder
		for _, e := range evs {
			typ := e["type"].(string)
			types = append(types, typ)
			if typ == "response.reasoning_text.delta" {
				deltas.WriteString(e["delta"].(string))
			}
			if typ == "response.completed" {
				check("streamed", e["response"].(map[string]any)["output"].([]any))
			}
		}
		if deltas.String() != "Add returns a + b." {
			t.Errorf("tools=%v: reasoning deltas add up to %q", tools, deltas.String())
		}
		seq := strings.Join(types, " ")
		want := "response.created response.output_item.added response.reasoning_text.delta"
		if !strings.HasPrefix(seq, want) || !strings.Contains(seq, "response.reasoning_text.done response.output_item.done response.output_item.added response.content_part.added") {
			t.Errorf("tools=%v: event order %s", tools, seq)
		}
	}
}

// Reasoning input items become the replayed reasoning of the assistant turn that follows them. encrypted_content written
// by this server wins over content (which a client may drop or alter), then reasoning_text content, then the summary; an
// encrypted_content from elsewhere is ignored; a reasoning item with no assistant turn after it is dropped.
func TestResponses_acceptsReasoningItems(t *testing.T) {
	blob := reasoningBlobPrefix + base64.StdEncoding.EncodeToString([]byte("exact\nbytes"))
	input := `[
	  {"role":"user","content":"q"},
	  {"type":"reasoning","summary":[],"encrypted_content":"` + blob + `","content":[{"type":"reasoning_text","text":"altered"}]},
	  {"type":"function_call","call_id":"c1","name":"exec_command","arguments":"{}"},
	  {"type":"function_call_output","call_id":"c1","output":"ok"},
	  {"type":"reasoning","summary":[],"encrypted_content":"gAAAAforeign","content":[{"type":"reasoning_text","text":"from content"}]},
	  {"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]},
	  {"type":"reasoning","summary":[{"type":"summary_text","text":"from summary"}]},
	  {"type":"function_call","call_id":"c2","name":"exec_command","arguments":"{}"},
	  {"type":"function_call_output","call_id":"c2","output":"ok"},
	  {"type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"dangling"}]}
	]`
	msgs, err := responseInputToMessages(json.RawMessage(input))
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	for _, m := range msgs {
		got = append(got, m.Role+":"+m.ReasoningContent)
	}
	want := []string{"user:", "assistant:exact\nbytes", "tool:", "assistant:from content", "assistant:from summary", "tool:"}
	if !slices.Equal(got, want) {
		t.Errorf("turns %q, want %q (a reasoning item is never a turn of its own)", got, want)
	}
}

// The round trip that keeps the KV warm, for every form a client may send the reasoning item back in. Turn 1 generates a
// call after reasoning; a Codex-style turn 2 sends back the reasoning item, the call and its result. Turn 2's prompt must
// extend turn 1's prompt followed by the exact ids generated, so the server reuses all of it: with content only, with
// encrypted_content only, with both, and with both disagreeing (encrypted_content wins: it is the bytes the model
// generated). Without the reasoning item it does not: the prompt leaves the generated ids at the reasoning, which is the
// cold re-prefill this item exists to prevent. An encrypted_content with serve's prefix that does not decode is a 400.
func TestResponses_reasoningRoundTripKeepsThePrefix(t *testing.T) {
	const gen = "<think>\nI will read calc.go.\n</think>\n\n<tool_call>\n{\"name\": \"exec_command\", \"arguments\": {\"cmd\": \"cat calc.go\"}}\n</tool_call>"
	ts, prompts, genIDs := scriptedLM(t, gen)
	req := requestFor("/v1/responses", true, false)
	user := map[string]any{"role": "user", "content": req["input"]}
	req["input"] = []any{user}
	first := postScripted(t, ts.URL+"/v1/responses", req)[0]["output"].([]any)
	if len(first) != 2 || first[0].(map[string]any)["type"] != "reasoning" || first[1].(map[string]any)["type"] != "function_call" {
		t.Fatalf("turn 1 output: %v", first)
	}
	p1 := (*prompts)[0]
	rs, call := first[0].(map[string]any), first[1].(map[string]any)
	callIn := map[string]any{"type": "function_call", "call_id": call["call_id"], "name": call["name"], "arguments": call["arguments"]}
	result := map[string]any{"type": "function_call_output", "call_id": call["call_id"], "output": "func Add(a, b int) int { return a + b }"}
	item := func(content any, encrypted any) map[string]any {
		it := map[string]any{"type": "reasoning", "summary": []any{}}
		if content != nil {
			it["content"] = content
		}
		if encrypted != nil {
			it["encrypted_content"] = encrypted
		}
		return it
	}
	altered := []any{map[string]any{"type": "reasoning_text", "text": "I edited this."}}
	for _, c := range []struct {
		name    string
		item    map[string]any // nil: no reasoning item sent back
		extends bool
	}{
		{"content only", item(rs["content"], nil), true},
		{"encrypted_content only", item(nil, rs["encrypted_content"]), true},
		{"encrypted_content only, content empty", item([]any{}, rs["encrypted_content"]), true},
		{"both", item(rs["content"], rs["encrypted_content"]), true},
		{"both, content altered: encrypted_content wins", item(altered, rs["encrypted_content"]), true},
		{"no reasoning item", nil, false},
	} {
		in := []any{user}
		if c.item != nil {
			in = append(in, c.item)
		}
		req["input"] = append(in, callIn, result)
		postScripted(t, ts.URL+"/v1/responses", req)
		p2 := (*prompts)[len(*prompts)-1]
		want := append(append([]int(nil), p1...), genIDs...)
		extends := len(p2) >= len(want) && slices.Equal(p2[:len(want)], want)
		if extends != c.extends {
			n := 0
			for n < min(len(p2), len(want)) && p2[n] == want[n] {
				n++
			}
			t.Errorf("%s: turn 2 extends turn 1 + generated ids = %v, want %v (they agree for %d of %d ids)", c.name, extends, c.extends, n, len(want))
		}
	}

	req["input"] = []any{user, item(rs["content"], reasoningBlobPrefix+"not base64!"), callIn, result}
	b, _ := json.Marshal(req)
	resp, err := http.Post(ts.URL+"/v1/responses", "application/json", strings.NewReader(string(b)))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusBadRequest || !strings.Contains(string(body), "encrypted_content") {
		t.Errorf("malformed encrypted_content: %d %s; want a 400 naming encrypted_content", resp.StatusCode, body)
	}
}
