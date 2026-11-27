package chat

import (
	"encoding/json"
	"os"
	"testing"
)

type inklingGolden struct {
	Name   string `json:"name"`
	System string `json:"system"`
	Effort string `json:"effort"`
	Tools  []struct {
		Name        string          `json:"name"`
		Description string          `json:"description"`
		Parameters  json.RawMessage `json:"parameters"`
	} `json:"tools"`
	Turns []struct {
		Role      string `json:"role"`
		Content   string `json:"content"`
		Reasoning string `json:"reasoning"`
		ToolName  string `json:"tool_name"`
		ToolCalls []struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		} `json:"tool_calls"`
	} `json:"turns"`
	Want string `json:"want"`
}

// TestInkling2_matchesApplyChatTemplate holds the renderer to transformers' apply_chat_template over Inkling-2-Mini's own
// chat_template.jinja (testdata/inkling2_chat_goldens.json, scripts/inkling2/pin_chat.py), byte for byte: system, several turns,
// a replayed reasoning, two efforts, declared tools (with HTML characters the template must not escape) and a tool loop.
func TestInkling2_matchesApplyChatTemplate(t *testing.T) {
	raw, err := os.ReadFile("testdata/inkling2_chat_goldens.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []inklingGolden
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	tmpl, err := Detect(Meta{ChatTemplate: "{{- \"<|message_model|>\" ~ \"<|content_text|>\" ~ \"<|end_message|>\" -}}"})
	if err != nil || tmpl.Name() != "inkling2" {
		t.Fatalf("Detect = %v, %v; want inkling2", tmpl, err)
	}
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			tp := tmpl
			if c.Effort == "none" {
				tp = tp.WithThinking(ThinkOff)
			} else if c.Effort != "" {
				tp = tp.WithReasoningEffort(c.Effort)
			}
			var turns []Turn
			for _, tr := range c.Turns {
				turn := Turn{Role: tr.Role, Content: tr.Content, Reasoning: tr.Reasoning, ToolName: tr.ToolName}
				for _, tc := range tr.ToolCalls {
					turn.ToolCalls = append(turn.ToolCalls, ToolCall{Name: tc.Name, Arguments: tc.Arguments})
				}
				turns = append(turns, turn)
			}
			var tools []Tool
			for _, tl := range c.Tools {
				tools = append(tools, Tool{Name: tl.Name, Description: tl.Description, Parameters: tl.Parameters})
			}
			if got := tp.RenderTools(c.System, turns, tools); got != c.Want {
				t.Errorf("render:\n got  %q\n want %q", got, c.Want)
			}
		})
	}
}

// inklingReply is the shape of a real Inkling-2-Mini reply (llama.cpp greedy on the Q8_0, 2026-11-26), with an answer added
// after its thinking message.
const inklingReply = "<|message_model|><|content_thinking|>The user wants a function.<|end_message|>" +
	"<|message_model|><|content_text|>def rev(s):\n    return s[::-1]<|end_message|>"

func TestInkling2Splitter_routesMessages(t *testing.T) {
	tp := Inkling2()
	for _, chunk := range []int{1, 3, 7, 1000} {
		sp := tp.NewReplySplitter(nil)
		var r, a string
		for i := 0; i < len(inklingReply); i += chunk {
			pr, pa := sp.Push(inklingReply[i:min(i+chunk, len(inklingReply))])
			r, a = r+pr, a+pa
		}
		fr, fa, trunc := sp.Finish()
		r, a = r+fr, a+fa
		if r != "The user wants a function." || a != "def rev(s):\n    return s[::-1]" || trunc {
			t.Fatalf("chunk %d: reasoning %q, answer %q, truncated %v", chunk, r, a, trunc)
		}
	}
	sp := tp.NewReplySplitter(nil)
	sp.Push("<|message_model|><|content_thinking|>still thinking")
	if _, _, trunc := sp.Finish(); !trunc {
		t.Fatal("a reply cut off inside its thinking message is not reported truncated")
	}
}

func TestInkling2_parsesToolCalls(t *testing.T) {
	out := "<|message_model|><|content_thinking|>Call it.<|end_message|><|message_model|>get_weather<|content_invoke_tool_json|>" +
		`{"name":"get_weather","args":{"city":"Paris","days":2}}<|end_message|>`
	sp := Inkling2().NewReplySplitter(nil)
	_, a := sp.Push(out)
	_, fa, _ := sp.Finish()
	calls := sp.ToolCalls()
	if len(calls) != 1 || calls[0].Name != "get_weather" || string(calls[0].Arguments) != `{"city":"Paris","days":2}` || a+fa != "" {
		t.Fatalf("calls %+v, answer %q", calls, a+fa)
	}
	got, _ := Inkling2().ParseToolCalls(out)
	if len(got) != 1 || got[0].Name != "get_weather" {
		t.Fatalf("ParseToolCalls = %+v", got)
	}
}
