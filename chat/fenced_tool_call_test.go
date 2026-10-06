package chat

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

// PRE-REGISTERED (written and committed before the code that satisfies it), docs/queue-correctness.md G39.
//
// The opt-in fenced-call rule. A chatml-family model that writes its tool call as a fenced JSON block in prose
// (Qwen2.5-Coder-7B under opencode, the cold-user run of 2026-10-05) makes no call today. With
// Template.WithLenientToolCalls(true) a reply is read as ONE call when ALL of these hold, and as prose otherwise:
//
//  1. the reply has exactly one fenced block, and that block is the LAST thing in it (only whitespace after the
//     closing fence); text before the fence is the call's lead (the prose);
//  2. the fence's language tag is empty or `json`;
//  3. the fence holds exactly one JSON object with only the keys name, arguments (or parameters) and id;
//  4. name is exactly a supplied tool's name, and arguments is an object that VALIDATES against that tool's
//     parameter schema: every required property present, every present property of the declared type and
//     inside its enum, no property the schema forbids (additionalProperties false);
//  5. the family is one that accepts a bare call (chatml, mellum2). Every other family, and every request without
//     the option, gets exactly what ParseToolCallsFor returned before this rule existed.
//
// The residual risk is stated in G39 and is not a gap in this list: a demonstration that ENDS on a single valid
// fenced call, with the call's own tool name and arguments valid for it, is indistinguishable from a meant call.
// That is why the option is off by default.

var fencedTestTools = []Tool{
	{Name: "edit", Parameters: json.RawMessage(`{"type":"object","properties":{"filePath":{"type":"string"},"oldString":{"type":"string"},"newString":{"type":"string"},"replaceAll":{"type":"boolean"},"mode":{"type":"string","enum":["exact","fuzzy"]}},"required":["filePath","oldString","newString"]}`)},
	{Name: "read", Parameters: json.RawMessage(`{"type":"object","properties":{"filePath":{"type":"string"}},"required":["filePath"],"additionalProperties":false}`)},
}

const fenceEditArgs = `{"filePath":"/p/calc.go","oldString":"return a - b","newString":"return a + b"}`

func fenceCall(name, args string) string {
	return "```json\n{\"name\": \"" + name + "\", \"arguments\": " + args + "}\n```"
}

func lenient(fam string) *Template { return allTemplates()[fam].WithLenientToolCalls(true) }

// The two real replies from the cold-user run, verbatim (~/goinfer-logs/cold-user-raw, B-agent/oc-run1.log and
// oc-run2.log), and the near variants that must read the same way.
func TestFencedToolCall_parsesTheRealRepliesAndNearVariants(t *testing.T) {
	const run1Lead = "The `add` function should subtract instead of add. I'll update the function and verify it."
	cases := []struct {
		name, out, wantLead, wantTool string
	}{
		{"cold-user attempt 2: the whole reply is the fence",
			"```json\n{\"name\": \"edit\", \"arguments\": {\"filePath\": \"calc.go\", \"oldString\": \"return a - b\", \"newString\": \"return a + b\"}}\n```", "", "edit"},
		{"cold-user attempt 1: one sentence, then the fence",
			run1Lead + "\n```json\n{\"name\": \"edit\", \"arguments\": {\"filePath\":\"/home/x/proj/calc.go\", \"oldString\":\"return a - b\", \"newString\":\"return a + b\"}}\n```", run1Lead, "edit"},
		{"trailing newlines after the closing fence", fenceCall("edit", fenceEditArgs) + "\n\n", "", "edit"},
		{"a fence with no language tag", "```\n{\"name\": \"read\", \"arguments\": {\"filePath\": \"a.go\"}}\n```", "", "read"},
		{"parameters instead of arguments", "```json\n{\"name\": \"read\", \"parameters\": {\"filePath\": \"a.go\"}}\n```", "", "read"},
		{"an optional property of the right type and an enum value", fenceCall("edit", `{"filePath":"a","oldString":"b","newString":"c","replaceAll":true,"mode":"fuzzy"}`), "", "edit"},
	}
	for _, fam := range []string{"chatml", "mellum2"} {
		for _, c := range cases {
			calls, lead := lenient(fam).ParseToolCallsFor(c.out, fencedTestTools)
			if len(calls) != 1 || calls[0].Name != c.wantTool {
				t.Errorf("%s/%s: calls = %+v, want one call to %q", fam, c.name, calls, c.wantTool)
				continue
			}
			if lead != c.wantLead {
				t.Errorf("%s/%s: lead %q, want %q", fam, c.name, lead, c.wantLead)
			}
			var args map[string]json.RawMessage
			if json.Unmarshal(calls[0].Arguments, &args) != nil || args == nil {
				t.Errorf("%s/%s: arguments %s are not an object", fam, c.name, calls[0].Arguments)
			}
		}
	}
}

// Everything here must stay PROSE (no call, lead = the reply as the parser has always trimmed it) even with the
// option on. Each case names the clause of the rule it falls outside.
func TestFencedToolCall_leavesShownExamplesAndNearMissesAsProse(t *testing.T) {
	valid := fenceCall("edit", fenceEditArgs)
	cases := []struct{ name, out string }{
		{"1. a demonstration with explanation AFTER the fence", "Here is an example call:\n" + valid + "\nThis call replaces the subtraction with an addition."},
		{"1. two fenced blocks, both valid calls", valid + "\n" + fenceCall("read", `{"filePath":"a"}`)},
		{"1. a call first, then a non-call block", valid + "\n```go\nfunc add(a, b int) int { return a + b }\n```"},
		{"1. the closing fence is missing (a cut-off stream)", "```json\n{\"name\": \"edit\", \"arguments\": " + fenceEditArgs + "}\n"},
		{"1. a call quoted mid-sentence in single backticks", "Call `{\"name\": \"edit\", \"arguments\": " + fenceEditArgs + "}` to fix it."},
		{"2. a python fence holding the same JSON", "```python\n{\"name\": \"edit\", \"arguments\": " + fenceEditArgs + "}\n```"},
		{"2. a bash fence holding the same JSON", "```bash\n{\"name\": \"edit\", \"arguments\": " + fenceEditArgs + "}\n```"},
		{"3. an array of calls", "```json\n[{\"name\": \"edit\", \"arguments\": " + fenceEditArgs + "}]\n```"},
		{"3. an extra key beside name and arguments", "```json\n{\"name\": \"edit\", \"arguments\": " + fenceEditArgs + ", \"note\": \"fix\"}\n```"},
		{"3. text inside the fence after the object", "```json\n{\"name\": \"edit\", \"arguments\": " + fenceEditArgs + "}\nand more\n```"},
		{"3. two objects in one fence", "```json\n{\"name\": \"edit\", \"arguments\": " + fenceEditArgs + "}\n{\"name\": \"read\", \"arguments\": {\"filePath\": \"a\"}}\n```"},
		{"3. invalid JSON", "```json\n{\"name\": \"edit\", \"arguments\": {\"filePath\": }\n```"},
		{"3. arguments is not an object", "```json\n{\"name\": \"edit\", \"arguments\": \"oops\"}\n```"},
		{"4. a tool that was not supplied", fenceCall("delete_everything", `{"path":"/"}`)},
		{"4. a name that is only a prefix of a supplied tool", fenceCall("edi", fenceEditArgs)},
		{"4. a required property missing", fenceCall("edit", `{"filePath":"a","oldString":"b"}`)},
		{"4. a property of the wrong type", fenceCall("edit", `{"filePath":"a","oldString":5,"newString":"c"}`)},
		{"4. a value outside the enum", fenceCall("edit", `{"filePath":"a","oldString":"b","newString":"c","mode":"yolo"}`)},
		{"4. a property the schema forbids (additionalProperties false)", fenceCall("read", `{"filePath":"a","also":"b"}`)},
		{"plain prose", "I would change the + to a -."},
		{"empty", ""},
	}
	for _, fam := range []string{"chatml", "mellum2"} {
		tm := lenient(fam)
		for _, c := range cases {
			calls, lead := tm.ParseToolCallsFor(c.out, fencedTestTools)
			if len(calls) != 0 {
				t.Errorf("%s/%s: parsed as a call %+v, must stay prose", fam, c.name, calls)
			}
			// No-call output is exactly what it is without the option.
			wantCalls, wantLead := allTemplates()[fam].ParseToolCallsFor(c.out, fencedTestTools)
			if !reflect.DeepEqual(calls, wantCalls) || lead != wantLead {
				t.Errorf("%s/%s: lenient (%v, %q) differs from default (%v, %q)", fam, c.name, calls, lead, wantCalls, wantLead)
			}
		}
	}
}

// 5. Off by default, and nothing outside the chatml family moves.
func TestFencedToolCall_offByDefaultAndOtherFamiliesUntouched(t *testing.T) {
	out := "I'll update the function.\n" + fenceCall("edit", fenceEditArgs)
	for fam, tm := range allTemplates() {
		if calls, _ := tm.ParseToolCallsFor(out, fencedTestTools); len(calls) != 0 {
			t.Errorf("%s without the option parsed a fenced call: %+v", fam, calls)
		}
	}
	for _, fam := range []string{"mistral", "llama3", "gemma4"} {
		on := allTemplates()[fam].WithLenientToolCalls(true)
		gotCalls, gotLead := on.ParseToolCallsFor(out, fencedTestTools)
		wantCalls, wantLead := allTemplates()[fam].ParseToolCallsFor(out, fencedTestTools)
		if !reflect.DeepEqual(gotCalls, wantCalls) || gotLead != wantLead {
			t.Errorf("%s: the option changed a family that does not accept bare calls: (%v, %q) vs (%v, %q)", fam, gotCalls, gotLead, wantCalls, wantLead)
		}
	}
	if allTemplates()["chatml"].WithLenientToolCalls(false) == nil || lenient("chatml") == allTemplates()["chatml"] {
		t.Error("WithLenientToolCalls must return a copy and leave the original template unchanged")
	}
	// A wrapped call is never re-read: it wins, and the fenced rule does not run on top of it.
	wrapped := "<tool_call>\n{\"name\": \"read\", \"arguments\": {\"filePath\": \"a\"}}\n</tool_call>\n" + fenceCall("edit", fenceEditArgs)
	if calls, _ := lenient("chatml").ParseToolCallsFor(wrapped, fencedTestTools); len(calls) != 1 || calls[0].Name != "read" {
		t.Errorf("a wrapped call plus a fence: %+v, want exactly the wrapped read call", calls)
	}
}

// The streaming guarantee the serve front end relies on (chat.ProseStreamer): whatever Push released is a PREFIX of
// the lead ParseToolCallsFor computes over the whole output, however the output is chunked, and a fence-aware
// streamer releases nothing from the opening fence on. Otherwise the fence text would reach the client as prose and
// then also be taken as a call.
func TestFenceAwareProseStreamerMatchesParser(t *testing.T) {
	outs := []string{
		fenceCall("edit", fenceEditArgs),
		"Prose first.\n" + fenceCall("edit", fenceEditArgs) + "\n",
		"Here is an example:\n" + fenceCall("edit", fenceEditArgs) + "\nIt edits the file.",
		"prose then an unterminated fence\n```json\n{\"name\": \"edit\"",
		"a lone `tick and ``two ticks and prose",
		"two fences\n```go\nx := 1\n```\nthen\n" + fenceCall("edit", fenceEditArgs),
		"plain prose, nothing special", "   leading space", "",
		"prose before <tool_call>\n{\"name\": \"read\", \"arguments\": {\"filePath\": \"a\"}}\n</tool_call>",
	}
	for _, fam := range []string{"chatml", "mellum2"} {
		tm := lenient(fam)
		opener, _ := tm.ToolCallOpener()
		for _, out := range outs {
			_, lead := tm.ParseToolCallsFor(out, fencedTestTools)
			for _, step := range []int{1, 2, 3, 7, len(out) + 1} {
				p := NewBareAwareProseStreamer(opener).FenceAware()
				var got strings.Builder
				for i := 0; i < len(out); i += step {
					got.WriteString(p.Push(out[i:min(i+step, len(out))]))
				}
				if !strings.HasPrefix(lead, got.String()) {
					t.Errorf("%s step %d: streamed %q is NOT a prefix of the parsed lead %q (out=%q)", fam, step, got.String(), lead, out)
				}
				if i := strings.Index(out, "```"); i >= 0 && !strings.Contains(out[:i], opener) && len(got.String()) > len(strings.TrimRight(out[:i], " \t\r\n")) {
					t.Errorf("%s step %d: released %q, which reaches into the fence at %d (out=%q)", fam, step, got.String(), i, out)
				}
			}
		}
	}
}

// Arbitrary output never panics, and a lenient call always names a supplied tool with object arguments that validate.
func FuzzFencedToolCall(f *testing.F) {
	for _, s := range []string{fenceCall("edit", fenceEditArgs), "x\n" + fenceCall("read", `{"filePath":"a"}`), "```", "```json", "```json\n{}\n```", "", "``` ```"} {
		f.Add(s)
	}
	tm := lenient("chatml")
	f.Fuzz(func(t *testing.T, out string) {
		calls, _ := tm.ParseToolCallsFor(out, fencedTestTools)
		for _, c := range calls {
			known := false
			for _, tl := range fencedTestTools {
				known = known || tl.Name == c.Name
			}
			var obj map[string]json.RawMessage
			if _, wrapped := allTemplates()["chatml"].ParseToolCallsFor(out, fencedTestTools); len(wrapped) == 0 && (!known || json.Unmarshal(c.Arguments, &obj) != nil) {
				t.Fatalf("lenient call %+v is not a supplied tool with object arguments (out=%q)", c, out)
			}
		}
	})
}
