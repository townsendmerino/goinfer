package chat

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

// Gates G1–G3 of the lenient bare-call parser, pre-registered in
// docs/measurements/tool-call-failure-t0-2026-09-23.md ("Follow-up A") before the code existed.

var bareTestTools = []Tool{
	{Name: "read_file", Parameters: json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}`)},
	{Name: "get_weather", Parameters: json.RawMessage(`{"type":"object"}`)},
}

// allTemplates is every family ParseToolCalls knows, so a test over "every other family" really is.
func allTemplates() map[string]*Template {
	return map[string]*Template{"chatml": ChatML(), "mellum2": Mellum2(), "mistral": Mistral(), "llama3": Llama3(), "gemma4": Gemma4()}
}

// G1 — non-forcing / parity. Any output whose first non-space byte is NOT '{', on any family, and
// ANY output on a family that does not accept bare calls, gets exactly what ParseToolCalls gives.
func TestParseToolCallsFor_parityOutsideBareShape(t *testing.T) {
	outs := []string{
		"", "   ", "plain prose",
		"Here is the call: {\"name\": \"read_file\", \"arguments\": {\"path\": \"a\"}}", // prose-then-JSON stays prose
		"<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"a\"}}\n</tool_call>",
		"<tool_call>\n{\"name\": \"read_file\", \"arguments\": \n</tool_call>", // malformed wrapped call: still no call
		"lead text <tool_call>\n{\"name\": \"nope\", \"arguments\": {}}\n</tool_call>",
		"[TOOL_CALLS] [{\"name\": \"read_file\", \"arguments\": {\"path\": \"a\"}}]",
		"<|python_tag|>{\"name\": \"read_file\", \"parameters\": {\"path\": \"a\"}}",
		"[{\"name\": \"read_file\", \"arguments\": {\"path\": \"a\"}}]", // a bare ARRAY is not the bare-call shape
		"`{\"name\": \"read_file\", \"arguments\": {}}`",
	}
	for fam, tmpl := range allTemplates() {
		for _, out := range outs {
			wantC, wantL := tmpl.ParseToolCalls(out)
			gotC, gotL := tmpl.ParseToolCallsFor(out, bareTestTools)
			if !reflect.DeepEqual(gotC, wantC) || gotL != wantL {
				t.Errorf("%s: %q: ParseToolCallsFor = (%v, %q), ParseToolCalls = (%v, %q)", fam, out, gotC, gotL, wantC, wantL)
			}
		}
	}
	// Families outside the bare set are untouched even on the bare shape itself.
	bare := `{"name": "read_file", "arguments": {"path": "a"}}`
	for fam, tmpl := range allTemplates() {
		if tmpl.AcceptsBareToolCall() {
			continue
		}
		wantC, wantL := tmpl.ParseToolCalls(bare)
		gotC, gotL := tmpl.ParseToolCallsFor(bare, bareTestTools)
		if !reflect.DeepEqual(gotC, wantC) || gotL != wantL {
			t.Errorf("%s does not accept bare calls but ParseToolCallsFor changed its result on %q", fam, bare)
		}
	}
	if got := len(bareFamilies()); got != 2 {
		t.Errorf("bare-call families = %v; the pre-registration names exactly chatml and mellum2", bareFamilies())
	}
}

func bareFamilies() []string {
	var out []string
	for fam, tmpl := range allTemplates() {
		if tmpl.AcceptsBareToolCall() {
			out = append(out, fam)
		}
	}
	return out
}

// The positive cases — what the parser exists to recover.
func TestParseToolCallsFor_recoversBareCall(t *testing.T) {
	cases := []struct {
		out, name, args string
	}{
		{`{"name": "read_file", "arguments": {"path": "notes.txt"}}`, "read_file", `{"path": "notes.txt"}`},
		{"  \n{\"name\": \"read_file\", \"arguments\": {\"path\": \"x\"}}\n", "read_file", `{"path": "x"}`},
		{"{\n\"name\": \"read_file\",\n\"arguments\": {\n\"path\": \"notes.txt\"\n}\n}", "read_file", "{\n\"path\": \"notes.txt\"\n}"}, // Ollama-seen layout
		{`{"name": "get_weather", "parameters": {}}`, "get_weather", `{}`},
		// trailing prose and a second speculative call: first object only
		{"{\"name\": \"read_file\", \"arguments\": {\"path\": \"a\"}}\n\nThis call reads the file.", "read_file", `{"path": "a"}`},
		{"{\"name\": \"read_file\", \"arguments\": {\"path\": \"a\"}}\n{\"name\": \"get_weather\", \"arguments\": {}}", "read_file", `{"path": "a"}`},
	}
	for _, fam := range []string{"chatml", "mellum2"} {
		tmpl := allTemplates()[fam]
		for _, c := range cases {
			calls, lead := tmpl.ParseToolCallsFor(c.out, bareTestTools)
			if len(calls) != 1 || calls[0].Name != c.name || string(calls[0].Arguments) != c.args || lead != "" {
				t.Errorf("%s: %q: got calls=%v lead=%q; want one %s(%s) and an empty lead", fam, c.out, calls, lead, c.name, c.args)
			}
		}
	}
}

// G2 — no invented calls. Each of these is bare JSON on a bare-call family, and each must stay
// exactly what ParseToolCalls says (prose).
func TestParseToolCallsFor_neverInventsACall(t *testing.T) {
	outs := []string{
		`{"name": "git_commit", "arguments": {"message": "x"}}`,  // not a supplied tool (the 7B's favourite)
		`{"name": "Read_File", "arguments": {"path": "a"}}`,      // exact match only
		`{"name": "read_file ", "arguments": {"path": "a"}}`,     // no trimming of the name
		`{"name": ["read_file"], "arguments": {"path": "a"}}`,    // non-string name
		`{"name": 7, "arguments": {}}`,                           //
		`{"name": "read_file"}`,                                  // no arguments at all
		`{"name": "read_file", "arguments": "{\"path\":\"a\"}"}`, // stringified args: not an object
		`{"name": "read_file", "arguments": ["a"]}`,              // array args
		`{"name": "read_file", "arguments": null}`,               //
		`{"name": "read_file", "arguments": {"path": "a"}`,       // truncated: never closes
		`{"name": "read_file", "arguments": {"path": "a"},}`,     // invalid JSON
		`{"answer": 42}`, // JSON prose
		`{"function": {"name": "read_file", "parameters": {"path": "a"}}, "type": "function"}`, // llama3-1B's echo shape
		`{}`,
		`{`,
	}
	for _, fam := range []string{"chatml", "mellum2"} {
		tmpl := allTemplates()[fam]
		for _, out := range outs {
			wantC, wantL := tmpl.ParseToolCalls(out)
			gotC, gotL := tmpl.ParseToolCallsFor(out, bareTestTools)
			if len(gotC) != 0 || !reflect.DeepEqual(gotC, wantC) || gotL != wantL {
				t.Errorf("%s: %q: invented or changed a result: got (%v, %q), want (%v, %q)", fam, out, gotC, gotL, wantC, wantL)
			}
		}
		// With NO tools supplied, nothing bare can ever parse.
		if c, _ := tmpl.ParseToolCallsFor(`{"name": "read_file", "arguments": {}}`, nil); len(c) != 0 {
			t.Errorf("%s: parsed a bare call with no tools supplied", fam)
		}
	}
}

// G3 — streaming. Byte-by-byte, what the bare-aware streamer releases is always a PREFIX of the lead
// ParseToolCallsFor computes (a delta cannot be unsent). For an output that does not open with '{'
// it must also release exactly what the plain streamer releases — the leniency may not change how
// ordinary prose streams.
func TestBareAwareProseStreamerMatchesParser(t *testing.T) {
	bare := `{"name": "read_file", "arguments": {"path": "a"}}`
	for _, fam := range []string{"chatml", "mellum2"} {
		tmpl := allTemplates()[fam]
		opener, ok := tmpl.ToolCallOpener()
		if !ok {
			t.Fatalf("%s: expected a streamable family", fam)
		}
		outs := []string{
			bare, "   " + bare, "\n\n" + bare + "\n\nexplanation after",
			`{"answer": 42} and then prose`, // bare JSON that is NOT a call: held, delivered at the end
			"{ not json at all",
			"plain prose, no call", "prose before " + opener + "\n" + bare + "\n</tool_call>",
			"   leading space then prose", "prose with a brace { later", "",
		}
		for _, out := range outs {
			_, lead := tmpl.ParseToolCallsFor(out, bareTestTools)
			ba, plain := NewBareAwareProseStreamer(opener), NewProseStreamer(opener)
			var gotBA, gotPlain strings.Builder
			for i := 0; i < len(out); i++ {
				gotBA.WriteString(ba.Push(out[i : i+1]))
				gotPlain.WriteString(plain.Push(out[i : i+1]))
			}
			if !strings.HasPrefix(lead, gotBA.String()) {
				t.Errorf("%s: streamed %q is NOT a prefix of the parsed lead %q (out=%q)", fam, gotBA.String(), lead, out)
			}
			opensWithBrace := strings.HasPrefix(strings.TrimLeft(out, " \t\r\n"), "{")
			if opensWithBrace && gotBA.Len() != 0 {
				t.Errorf("%s: released %q from an output that opens with '{' (out=%q)", fam, gotBA.String(), out)
			}
			if !opensWithBrace && gotBA.String() != gotPlain.String() {
				t.Errorf("%s: bare-aware streamer released %q, plain released %q (out=%q)", fam, gotBA.String(), gotPlain.String(), out)
			}
		}
	}
}

// FuzzParseToolCallsFor holds G1 and G2 over arbitrary outputs: on every family, the result equals
// ParseToolCalls unless the bare shape applies, and any call the bare path adds names a supplied
// tool and carries an object as its arguments.
func FuzzParseToolCallsFor(f *testing.F) {
	for _, s := range []string{
		`{"name": "read_file", "arguments": {"path": "a"}}`, "prose", `{"name": "git_commit", "arguments": {}}`,
		"<tool_call>\n{\"name\": \"read_file\", \"arguments\": {}}\n</tool_call>", " {", `{"name":"get_weather","parameters":{}}x`,
	} {
		f.Add(s)
	}
	names := map[string]bool{}
	for _, tl := range bareTestTools {
		names[tl.Name] = true
	}
	f.Fuzz(func(t *testing.T, out string) {
		for fam, tmpl := range allTemplates() {
			wantC, wantL := tmpl.ParseToolCalls(out)
			gotC, gotL := tmpl.ParseToolCallsFor(out, bareTestTools)
			if reflect.DeepEqual(gotC, wantC) && gotL == wantL {
				continue
			}
			if !tmpl.AcceptsBareToolCall() || len(wantC) != 0 || !strings.HasPrefix(strings.TrimLeft(out, " \t\r\n"), "{") {
				t.Fatalf("%s: %q: result changed outside the bare shape", fam, out)
			}
			if len(gotC) != 1 || !names[gotC[0].Name] || gotL != "" {
				t.Fatalf("%s: %q: bare path produced %v lead %q", fam, out, gotC, gotL)
			}
			var obj map[string]json.RawMessage
			if json.Unmarshal(gotC[0].Arguments, &obj) != nil || obj == nil {
				t.Fatalf("%s: %q: bare call arguments %q are not an object", fam, out, gotC[0].Arguments)
			}
		}
	})
}
