package sdktest

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"reflect"
	"strings"
	"testing"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/responses"
)

// sseEvent is one event of a /v1/responses stream, with the fields the lifecycle check reads.
type sseEvent struct {
	Type        string         `json:"type"`
	Seq         *int           `json:"sequence_number"`
	OutputIndex *int           `json:"output_index"`
	ItemID      string         `json:"item_id"`
	Delta       string         `json:"delta"`
	Text        string         `json:"text"`
	Arguments   string         `json:"arguments"`
	Item        map[string]any `json:"item"`
	Response    struct {
		Output []map[string]any `json:"output"`
	} `json:"response"`
}

func postStream(t *testing.T, body string) []sseEvent {
	t.Helper()
	resp, err := tinyClient(t).Post("http://goinfer/v1/responses", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		raw, _ := io.ReadAll(resp.Body)
		t.Fatalf("status %d: %s", resp.StatusCode, raw)
	}
	var evs []sseEvent
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") || line == "data: [DONE]" {
			continue
		}
		var ev sseEvent
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &ev); err != nil {
			t.Fatalf("not JSON: %q", line)
		}
		evs = append(evs, ev)
	}
	return evs
}

// checkLifecycle holds a stream to what a client that builds the reply from item events needs: sequence_number counts
// every event from 0, every item the final response lists was announced by output_item.added at its index and closed by
// output_item.done with the same item, a message's deltas add up to its text, and nothing is left open.
func checkLifecycle(t *testing.T, evs []sseEvent) (items []map[string]any) {
	t.Helper()
	if len(evs) < 2 || evs[0].Type != "response.created" || evs[len(evs)-1].Type != "response.completed" {
		t.Fatalf("a stream does not run from response.created to response.completed: %d events", len(evs))
	}
	added, done := map[int]map[string]any{}, map[int]map[string]any{}
	deltas := map[string]string{}
	argDeltas := map[string]string{}
	for i, ev := range evs {
		if ev.Seq == nil || *ev.Seq != i {
			t.Fatalf("event %d (%s): sequence_number %v, want %d", i, ev.Type, ev.Seq, i)
		}
		switch ev.Type {
		case "response.output_item.added":
			if _, dup := added[*ev.OutputIndex]; dup {
				t.Errorf("output_index %d announced twice", *ev.OutputIndex)
			}
			added[*ev.OutputIndex] = ev.Item
		case "response.output_text.delta":
			if _, ok := added[*ev.OutputIndex]; !ok {
				t.Errorf("a text delta for output_index %d before its item was announced", *ev.OutputIndex)
			}
			deltas[ev.ItemID] += ev.Delta
		case "response.output_text.done":
			if deltas[ev.ItemID] != ev.Text {
				t.Errorf("output_text.done text %q is not the deltas' %q", ev.Text, deltas[ev.ItemID])
			}
		case "response.function_call_arguments.delta":
			argDeltas[ev.ItemID] += ev.Delta
		case "response.function_call_arguments.done":
			if argDeltas[ev.ItemID] != ev.Arguments {
				t.Errorf("function_call_arguments.done %q is not the deltas' %q", ev.Arguments, argDeltas[ev.ItemID])
			}
		case "response.output_item.done":
			if _, ok := added[*ev.OutputIndex]; !ok {
				t.Errorf("output_index %d closed without being announced", *ev.OutputIndex)
			}
			done[*ev.OutputIndex] = ev.Item
		}
	}
	final := evs[len(evs)-1].Response.Output
	if len(added) != len(final) || len(done) != len(final) {
		t.Errorf("the final response lists %d items, the stream announced %d and closed %d", len(final), len(added), len(done))
	}
	for i, it := range final {
		if !reflect.DeepEqual(done[i], it) {
			t.Errorf("item %d: output_item.done carries %v, the final response lists %v", i, done[i], it)
		}
	}
	return final
}

// The Responses stream carries the item lifecycle (Codex assembles its reply from output_item.done; a stream of text
// deltas alone leaves it with an empty turn), in each of the shapes the route writes: a plain message, and a tool turn
// that ends in a function call.
func TestResponsesStreamLifecycle(t *testing.T) {
	t.Run("message", func(t *testing.T) {
		evs := postStream(t, `{"model":"tiny","input":"hi","stream":true,"temperature":0,"max_output_tokens":6}`)
		final := checkLifecycle(t, evs)
		if len(final) != 1 || final[0]["type"] != "message" {
			t.Fatalf("a plain turn lists %v", final)
		}
		var kinds []string
		for _, ev := range evs {
			kinds = append(kinds, ev.Type)
		}
		got := strings.Join(kinds, " ")
		for _, want := range []string{"response.created response.output_item.added response.content_part.added response.output_text.delta",
			"response.output_text.done response.content_part.done response.output_item.done response.completed"} {
			if !strings.Contains(got, want) {
				t.Errorf("the events %q do not contain the run %q", got, want)
			}
		}
	})
	t.Run("function call", func(t *testing.T) {
		evs := postStream(t, `{"model":"tiny","input":"hi","stream":true,"temperature":0,"max_output_tokens":1024,
			"tool_choice":{"type":"function","name":"pick"},
			"tools":[{"type":"function","name":"pick","description":"pick one","parameters":{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false}}]}`)
		final := checkLifecycle(t, evs)
		var call map[string]any
		for _, it := range final {
			if it["type"] == "function_call" {
				call = it
			}
		}
		if call == nil || call["name"] != "pick" {
			t.Fatalf("a forced call lists %v", final)
		}
		var args map[string]any
		if err := json.Unmarshal([]byte(call["arguments"].(string)), &args); err != nil {
			t.Errorf("the call's arguments %q are not JSON: %v", call["arguments"], err)
		}
		for _, ev := range evs {
			if ev.Type == "response.output_item.added" && ev.Item["type"] == "function_call" && (ev.Item["arguments"] != "" || ev.Item["status"] != "in_progress") {
				t.Errorf("a function_call is announced already complete: %v", ev.Item)
			}
		}
	})
	t.Run("nothing generated still announces its message", func(t *testing.T) {
		// A length cap of one token on a model whose first token may be a stop: whatever it writes, the lifecycle holds.
		evs := postStream(t, `{"model":"tiny","input":"hi","stream":true,"temperature":0,"max_output_tokens":1}`)
		checkLifecycle(t, evs)
	})
}

// The same stream through the official SDK's own reader: it accepts every event and the final response it
// assembles lists the message.
func TestResponsesStreamThroughTheSDK(t *testing.T) {
	client := openaiClient(t)
	stream := client.Responses.NewStreaming(context.Background(), responses.ResponseNewParams{Model: "tiny", MaxOutputTokens: openai.Int(6), Temperature: openai.Float(0),
		Input: responses.ResponseNewParamsInputUnion{OfString: openai.String("hi")}})
	var kinds = map[string]int{}
	var completed *responses.Response
	for stream.Next() {
		ev := stream.Current()
		kinds[ev.Type]++
		if ev.Type == "response.completed" {
			r := ev.Response
			completed = &r
		}
	}
	if err := stream.Err(); err != nil {
		t.Fatal(err)
	}
	for _, k := range []string{"response.output_item.added", "response.output_item.done", "response.content_part.done", "response.output_text.done"} {
		if kinds[k] != 1 {
			t.Errorf("the SDK read %d %s events, want 1 (%v)", kinds[k], k, kinds)
		}
	}
	if completed == nil || len(completed.Output) != 1 {
		t.Fatalf("the SDK's final response: %+v", completed)
	}
}
