package constrain

import (
	"encoding/json"
	"fmt"
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
	name, _ := json.Marshal(toolName)
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

func mustMap(s string) map[string]any {
	var m map[string]any
	_ = json.Unmarshal([]byte(s), &m)
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
