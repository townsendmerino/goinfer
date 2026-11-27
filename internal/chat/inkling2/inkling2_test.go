package inkling2

import "testing"

const sampleReply = "<|message_model|><|content_thinking|>The user wants a function.<|end_message|>" +
	"<|message_model|><|content_text|>def rev(s):\n    return s[::-1]<|end_message|>"

func TestSplitter_chunkIndependent(t *testing.T) {
	for _, chunk := range []int{1, 3, 7, 1000} {
		sp := NewSplitter()
		var r, a string
		for i := 0; i < len(sampleReply); i += chunk {
			pr, pa := sp.Push(sampleReply[i:min(i+chunk, len(sampleReply))])
			r, a = r+pr, a+pa
		}
		trunc := sp.InReasoning()
		fr, fa := sp.Flush()
		r, a = r+fr, a+fa
		if r != "The user wants a function." || a != "def rev(s):\n    return s[::-1]" || trunc {
			t.Fatalf("chunk %d: reasoning %q, answer %q, truncated %v", chunk, r, a, trunc)
		}
	}
	sp := NewSplitter()
	sp.Push("<|message_model|><|content_thinking|>still thinking")
	if !sp.InReasoning() {
		t.Fatal("cutoff inside thinking message must report InReasoning")
	}
}

func TestParseTools(t *testing.T) {
	out := "<|message_model|><|content_thinking|>Call it.<|end_message|><|message_model|>get_weather<|content_invoke_tool_json|>" +
		`{"name":"get_weather","args":{"city":"Paris","days":2}}<|end_message|>`
	calls, lead := ParseTools(out)
	if len(calls) != 1 || calls[0].Name != "get_weather" || string(calls[0].Arguments) != `{"city":"Paris","days":2}` || lead != "" {
		t.Fatalf("calls %+v, lead %q", calls, lead)
	}
}
