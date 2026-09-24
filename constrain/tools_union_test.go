package constrain

import (
	"encoding/json"
	"strings"
	"testing"
)

var unionTools = []ToolSpec{
	{Name: "read_file", Parameters: []byte(`{"type":"object","properties":{"path":{"type":"string"},"limit":{"type":"integer"}},"required":["path"],"additionalProperties":false}`)},
	{Name: "read", Parameters: []byte(`{"type":"object","properties":{"x":{"type":"string"}},"required":["x"],"additionalProperties":false}`)}, // a PREFIX of another name
	{Name: "git_status", Parameters: nil}, // no arguments
	{Name: "a<&b", Parameters: []byte(`{"type":"object","properties":{},"additionalProperties":false}`)}, // M-29's escaping trap
}

// accepts reports whether g accepts s as a COMPLETE document, walking it byte by byte
// through TryBytes/Commit as the masker would.
func accepts(g Grammar, s string) bool {
	g.Reset()
	for i := 0; i < len(s); i++ {
		b := []byte{s[i]}
		if !g.TryBytes(b) {
			return false
		}
		g.Commit(b)
	}
	return g.CanEnd()
}

func chatml(t *testing.T) Grammar {
	t.Helper()
	g, err := ToolCallsGrammar("<tool_call>\n", "\n</tool_call>", "arguments", false, unionTools)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

// The property in the task's own words: every string the union accepts is a well-formed call
// to exactly one supplied tool — shown here by acceptance matching each single-tool grammar.
func TestToolCallsGrammar_acceptsExactlyTheUnionOfSingleToolGrammars(t *testing.T) {
	g := chatml(t)
	w := func(body string) string { return "<tool_call>\n" + body + "\n</tool_call>" }
	cases := []struct {
		s    string
		want bool
	}{
		{w(`{"name": "read_file", "arguments": {"path": "a.go"}}`), true},
		{w(`{"name": "read_file", "arguments": {"path": "a.go", "limit": 5}}`), true},
		{w(`{"arguments": {"path": "a.go"}, "name": "read_file"}`), true}, // key order does not matter
		{w(`{"name": "read", "arguments": {"x": "y"}}`), true},            // a prefix of another tool's name
		{w(`{"name": "git_status", "arguments": {}}`), true},
		{w(`{"name": "a<&b", "arguments": {}}`), true},
		{w(`{"name": "git_commit", "arguments": {}}`), false},                 // not supplied (T0's 7B failure)
		{w(`{"name": "read_fil", "arguments": {"path": "a"}}`), false},        // a prefix of a name is not a name
		{w(`{"name": "read_file", "arguments": {}}`), false},                  // missing required
		{w(`{"name": "read_file", "arguments": {"path": 3}}`), false},         // wrong type
		{w(`{"name": "read", "arguments": {"path": "a.go"}}`), false},         // args of a DIFFERENT tool
		{w(`{"name": "git_status", "arguments": {"x": 1}}`), false},           // no-arg tool given args
		{`{"name": "read_file", "arguments": {"path": "a"}}`, false},          // wrapper required
		{w(`{"name": "read_file", "arguments": {"path": "a"}}`) + "x", false}, // nothing after the call
	}
	for _, c := range cases {
		if got := accepts(g, c.s); got != c.want {
			t.Errorf("accepts(%q) = %v, want %v", c.s, got, c.want)
		}
		single := false
		for _, tl := range unionTools {
			sg, err := ToolCallGrammar("<tool_call>\n", "\n</tool_call>", "arguments", tl.Name, false, tl.Parameters)
			if err != nil {
				t.Fatal(err)
			}
			single = single || accepts(sg, c.s)
		}
		if single != c.want {
			t.Errorf("%q: union says %v but the single-tool grammars together say %v", c.s, c.want, single)
		}
	}
}

// After the name discriminates, exactly one branch survives; while it has not, several do.
func TestToolCallsGrammar_collapsesAtTheName(t *testing.T) {
	g := chatml(t).(*unionGrammar)
	feed := func(s string) {
		for i := 0; i < len(s); i++ {
			b := []byte{s[i]}
			if !g.TryBytes(b) {
				t.Fatalf("rejected %q at byte %d", s, i)
			}
			g.Commit(b)
		}
	}
	feed(`<tool_call>` + "\n" + `{"name": "rea`)
	if g.Live() != 2 {
		t.Errorf("after \"rea\": %d live branches, want 2 (read, read_file)", g.Live())
	}
	feed(`d_`)
	if g.Live() != 1 {
		t.Errorf("after \"read_\": %d live branches, want 1", g.Live())
	}
}

// Clone must be independent in both directions — grammar-fused speculative decode rolls a
// clone forward over a draft and must not move the live grammar (the task's Clone warning).
func TestToolCallsGrammar_cloneIsIndependent(t *testing.T) {
	g := chatml(t)
	pre := "<tool_call>\n{\"name\": \"read"
	for i := 0; i < len(pre); i++ {
		g.Commit([]byte{pre[i]})
	}
	c := g.Clone()
	c.Commit([]byte("_file"))
	if !g.TryBytes([]byte(`", "arguments": {"x": "y"}}`)) {
		t.Error("committing on the clone narrowed the original to read_file")
	}
	// and the branch the clone ADVANCED must not have moved in the original either
	if !g.TryBytes([]byte(`_file", "arguments": {"path": "a"}}`)) {
		t.Error("committing \"_file\" on the clone advanced the original's read_file branch (shallow Clone)")
	}
	if c.TryBytes([]byte(`", "arguments": {"x": "y"}}`)) {
		t.Error("clone still accepts read's arguments after committing \"_file\"")
	}
	c.Reset()
	if !accepts(c, "<tool_call>\n{\"name\": \"git_status\", \"arguments\": {}}\n</tool_call>") {
		t.Error("Reset on a clone did not restore the full branch set")
	}
	if g.(*unionGrammar).Live() != 2 {
		t.Errorf("Reset on the clone changed the original: %d live", g.(*unionGrammar).Live())
	}
}

func TestToolCallsGrammar_errors(t *testing.T) {
	if _, err := ToolCallsGrammar("<tool_call>\n", "\n</tool_call>", "arguments", false, nil); err == nil {
		t.Error("no tools: want an error")
	}
	bad := []ToolSpec{{Name: "ok"}, {Name: "bad", Parameters: []byte(`{"type":"object","additionalProperties":true}`)}}
	if _, err := ToolCallsGrammar("<tool_call>\n", "\n</tool_call>", "arguments", false, bad); err == nil {
		t.Error("an uncompilable tool schema must fail the whole union, not drop the tool")
	}
	dup := []ToolSpec{{Name: "x"}, {Name: "x", Parameters: []byte(`not json`)}}
	if _, err := ToolCallsGrammar("<tool_call>\n", "\n</tool_call>", "arguments", false, dup); err != nil {
		t.Errorf("a duplicate name should collapse to its first spec: %v", err)
	}
}

// The other wrapper families: mistral's one-element array, llama3's empty wrapper + "parameters".
func TestToolCallsGrammar_otherFamilies(t *testing.T) {
	m, err := ToolCallsGrammar("[TOOL_CALLS] ", "", "arguments", true, unionTools)
	if err != nil {
		t.Fatal(err)
	}
	if !accepts(m, `[TOOL_CALLS] [{"name": "read", "arguments": {"x": "1"}}]`) {
		t.Error("mistral: valid call rejected")
	}
	if accepts(m, `[TOOL_CALLS] [{"name": "nope", "arguments": {}}]`) {
		t.Error("mistral: unknown tool accepted")
	}
	l, err := ToolCallsGrammar("", "", "parameters", false, unionTools)
	if err != nil {
		t.Fatal(err)
	}
	if !accepts(l, `{"name": "git_status", "parameters": {}}`) || accepts(l, `{"name": "git_status", "arguments": {}}`) {
		t.Error("llama3: argsKey not honoured")
	}
}

// Masker plumbing: the union drops into NewMasker unchanged and masks an unknown tool name.
func TestToolCallsGrammar_masker(t *testing.T) {
	vocab := [][]byte{[]byte("<tool_call>\n"), []byte(`{"name": "`), []byte("read"), []byte("git_commit"), []byte("x"), nil}
	m := NewMasker(chatml(t), vocab, []int{5})
	logits := make([]float32, len(vocab))
	m.Process([]int{0, 1}, logits)
	if logits[2] != 0 || logits[3] == 0 {
		t.Errorf("after the name opener: read legal=%v git_commit legal=%v (want true, false)", logits[2] == 0, logits[3] == 0)
	}
}

// FuzzToolCallsGrammar: any input the union accepts as a complete document is a wrapped JSON
// object whose name is a supplied tool — checked with encoding/json, independent of the grammar.
func FuzzToolCallsGrammar(f *testing.F) {
	for _, s := range []string{
		"<tool_call>\n{\"name\": \"read\", \"arguments\": {\"x\": \"\"}}\n</tool_call>",
		"<tool_call>\n{\"name\": \"git_status\", \"arguments\": {}}\n</tool_call>",
		"<tool_call>\n{\"arguments\": {}, \"name\": \"a<&b\"}\n</tool_call>",
		"<tool_call>\n{\"name\": \"read_fi",
	} {
		f.Add(s)
	}
	names := map[string]bool{}
	for _, tl := range unionTools {
		names[tl.Name] = true
	}
	g, err := ToolCallsGrammar("<tool_call>\n", "\n</tool_call>", "arguments", false, unionTools)
	if err != nil {
		f.Fatal(err)
	}
	f.Fuzz(func(t *testing.T, s string) {
		if !accepts(g, s) {
			return
		}
		body, ok := strings.CutPrefix(s, "<tool_call>\n")
		if ok {
			body, ok = strings.CutSuffix(body, "\n</tool_call>")
		}
		if !ok {
			t.Fatalf("accepted %q without the wrapper", s)
		}
		var call struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		}
		dec := json.NewDecoder(strings.NewReader(body))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&call); err != nil {
			t.Fatalf("accepted %q but it does not decode: %v", s, err)
		}
		if !names[call.Name] {
			t.Fatalf("accepted a call to %q, which is not a supplied tool", call.Name)
		}
		var obj map[string]any
		if json.Unmarshal(call.Arguments, &obj) != nil {
			t.Fatalf("accepted %q whose arguments are not an object", s)
		}
	})
}
