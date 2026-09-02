package constrain

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

// Tool-call constraint. A tool call is a fixed family wrapper around a JSON
// object the model must fill in: e.g. ChatML's "<tool_call>\n{…}\n</tool_call>".
// toolGrammar composes a literal prefix + a schema-constrained JSON value + a
// literal suffix into one incremental Grammar, reusing schemaGrammar's snapshot.
// Used "tight when unambiguous": one candidate tool ⇒ the JSON is constrained to
// {"name": const, <argsKey>: <that tool's parameter schema>}.

type toolGrammar struct {
	prefix, suffix string
	inner          *schemaGrammar
	phase          int // 0 prefix, 1 json, 2 suffix, 3 done
	ppos, spos     int

	sPhase, sPpos, sSpos int
}

// ToolCallGrammar builds the constraint for a single tool call: the family
// wrapper (prefix/suffix) around a JSON value matching the tool. argsKey is
// "arguments" or "parameters"; array wraps the call object in a one-element array
// (Mistral). paramSchema is the tool's JSON-Schema for its arguments.
func ToolCallGrammar(prefix, suffix, argsKey, toolName string, array bool, paramSchema []byte) (Grammar, error) {
	if len(paramSchema) == 0 {
		paramSchema = []byte(`{"type":"object","properties":{},"additionalProperties":false}`)
	}
	paramSchema = closeEmptyToolObject(paramSchema)
	// encodeLiteral, not json.Marshal: the tool name becomes a `const` literal in the
	// grammar, so HTML-escaping it makes a name containing < or & unreachable (M-29).
	name, err := encodeLiteral(toolName)
	if err != nil {
		return nil, fmt.Errorf("constrain: tool name: %w", err)
	}
	obj := fmt.Sprintf(
		`{"type":"object","additionalProperties":false,"required":["name",%q],"properties":{"name":{"const":%s},%q:%s}}`,
		argsKey, name, argsKey, paramSchema)
	doc := obj
	if array {
		doc = fmt.Sprintf(`{"type":"array","minItems":1,"maxItems":1,"items":%s}`, obj)
	}
	n, err := compile(mustMap(doc))
	if err != nil {
		return nil, fmt.Errorf("constrain: tool grammar: %w", err)
	}
	g := &toolGrammar{prefix: prefix, suffix: suffix, inner: &schemaGrammar{root: n}}
	g.Reset()
	return g, nil
}

// mustMap decodes a schema document built above. UseNumber for the same reason
// JSONSchema uses it: this is the path a TOOL's paramSchema takes, so without it a
// large integer enum in a tool argument loses precision here instead (M-29).
func mustMap(s string) map[string]any {
	dec := json.NewDecoder(strings.NewReader(s))
	dec.UseNumber()
	var m map[string]any
	_ = dec.Decode(&m)
	return m
}

func (g *toolGrammar) Reset() {
	g.inner.Reset()
	g.ppos, g.spos = 0, 0
	if g.prefix == "" {
		g.phase = 1
	} else {
		g.phase = 0
	}
}

// Clone returns an independent copy at the current state (value fields + a cloned
// inner schema grammar; prefix/suffix are immutable strings).
func (g *toolGrammar) Clone() Grammar {
	c := *g
	c.inner = g.inner.Clone().(*schemaGrammar)
	return &c
}

func (g *toolGrammar) CanEnd() bool {
	switch g.phase {
	case 1:
		return g.suffix == "" && g.inner.CanEnd()
	case 3:
		return true
	}
	return false
}

func (g *toolGrammar) TryBytes(bs []byte) bool {
	g.snapshot()
	ok := true
	for _, b := range bs {
		if !g.step(b) {
			ok = false
			break
		}
	}
	g.restore()
	return ok
}

func (g *toolGrammar) Commit(bs []byte) {
	for _, b := range bs {
		g.step(b)
	}
}

func (g *toolGrammar) snapshot() {
	g.sPhase, g.sPpos, g.sSpos = g.phase, g.ppos, g.spos
	g.inner.snapshot()
}

func (g *toolGrammar) restore() {
	g.phase, g.ppos, g.spos = g.sPhase, g.sPpos, g.sSpos
	g.inner.restore()
}

func (g *toolGrammar) step(b byte) bool {
	for {
		switch g.phase {
		case 0: // literal prefix
			if b != g.prefix[g.ppos] {
				return false
			}
			g.ppos++
			if g.ppos == len(g.prefix) {
				g.phase = 1
			}
			return true
		case 1: // schema-constrained JSON
			// Prefer the suffix once the JSON is a complete value (so the suffix's
			// leading byte isn't swallowed as trailing JSON whitespace).
			if g.inner.CanEnd() && len(g.suffix) > 0 && b == g.suffix[0] {
				g.phase, g.spos = 2, 0
				continue
			}
			if g.inner.step(b) {
				return true
			}
			if g.inner.CanEnd() && len(g.suffix) > 0 {
				g.phase, g.spos = 2, 0
				continue
			}
			return false
		case 2: // literal suffix
			if b != g.suffix[g.spos] {
				return false
			}
			g.spos++
			if g.spos == len(g.suffix) {
				g.phase = 3
			}
			return true
		case 3:
			return false // the call is complete; nothing may follow
		}
		return false
	}
}

// closeEmptyToolObject rewrites a no-argument tool schema into the closed-empty form
// the compiler can build (M-30).
//
// `{"type":"object","properties":{}}` is THE canonical no-argument tool schema —
// OpenAI's own examples, most MCP servers, and every Pydantic/zod tool with no
// parameters emit it. compile() rejects an object with no properties and no
// `additionalProperties:false`, correctly, because in JSON Schema that shape means
// "any object" and the grammar can only build a closed one. But a TOOL that declared
// its parameters and declared NONE means it takes no arguments, and reading it as
// "any object" is the wrong of the two readings. So the narrowing happens here, in
// the tool path where the extra context justifies it, and compile() is left strict —
// an omitted `parameters` already mapped to this same closed-empty form, so the
// explicit spelling now behaves like the implicit one rather than being a 400.
//
// Only the ambiguous case is touched: an explicit additionalProperties (true or
// false) is left exactly as written, so `additionalProperties:true` still reaches
// compile() and is still refused rather than being silently narrowed.
func closeEmptyToolObject(schema []byte) []byte {
	dec := json.NewDecoder(bytes.NewReader(schema))
	dec.UseNumber()
	var m map[string]any
	if err := dec.Decode(&m); err != nil {
		return schema // not an object literal; let compile report it
	}
	if t, _ := m["type"].(string); t != "object" {
		return schema
	}
	if _, present := m["additionalProperties"]; present {
		return schema
	}
	if props, ok := m["properties"].(map[string]any); ok && len(props) > 0 {
		return schema
	} else if !ok {
		if _, present := m["properties"]; present {
			return schema // properties present but not an object — compile reports it
		}
	}
	m["additionalProperties"] = false
	out, err := json.Marshal(m)
	if err != nil {
		return schema
	}
	return out
}
