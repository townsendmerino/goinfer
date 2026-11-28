package constrain

import (
	"encoding/hex"
	"encoding/json"
	"flag"
	"hash/fnv"
	"maps"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"unicode/utf8"
)

var updateSchemaPin = flag.Bool("update-schema-pin", false,
	"rewrite testdata/schema_pin.json from this tree's grammars instead of checking them")

const schemaPinPath = "testdata/schema_pin.json"

// pinEntry is one pinned grammar: a schema compiled strictly (JSONSchema), or, when Tool names one, a tool's parameters
// compiled through ToolCallGrammar with no wrapper.
type pinEntry struct {
	Name   string     `json:"name"`
	Schema string     `json:"schema"`
	Tool   string     `json:"tool,omitempty"`
	Walks  []pinWalk  `json:"walks"`
	Probes []pinProbe `json:"probes"`
}

// pinWalk is a document the grammar accepts. Mask hashes, for every prefix of it, which of the 256 byte values the
// grammar allows next and whether the document may end there.
type pinWalk struct {
	Doc  string `json:"doc"`
	Mask string `json:"mask"`
}

// pinProbe is a walk with one byte deleted, replaced or inserted, or cut short. N is how many of its leading bytes the
// grammar accepts and End whether the document may end after them.
type pinProbe struct {
	Doc string `json:"doc"`
	N   int    `json:"n"`
	End bool   `json:"end"`
}

func (e pinEntry) grammar() (Grammar, error) {
	if e.Tool != "" {
		return ToolCallGrammar("", "", "arguments", e.Tool, false, []byte(e.Schema))
	}
	return JSONSchema([]byte(e.Schema))
}

// TestSchema_pinnedGrammars holds the schemas that compiled before `$ref`, type lists and `anyOf`/`oneOf` were
// supported to the documents they accepted and refused then: testdata/schema_pin.json was written from the tree before
// that change (-update-schema-pin), and every walk and probe in it must read the same now.
func TestSchema_pinnedGrammars(t *testing.T) {
	if *updateSchemaPin {
		writeSchemaPin(t)
	}
	raw, err := os.ReadFile(schemaPinPath)
	if err != nil {
		t.Fatal(err)
	}
	var entries []pinEntry
	if err := json.Unmarshal(raw, &entries); err != nil {
		t.Fatal(err)
	}
	walks, probes := 0, 0
	for _, e := range entries {
		g, err := e.grammar()
		if err != nil {
			t.Errorf("%s: a schema that compiled no longer does: %v", e.Name, err)
			continue
		}
		for _, w := range e.Walks {
			walks++
			if n, end := pinAccept(g, []byte(w.Doc)); n != len(w.Doc) || !end {
				t.Errorf("%s: %q was accepted whole and is now accepted for %d bytes (may end: %v)", e.Name, w.Doc, n, end)
				continue
			}
			if got := pinMask(g, []byte(w.Doc)); got != w.Mask {
				t.Errorf("%s: along %q the legal next bytes changed (mask hash %s, pinned %s)", e.Name, w.Doc, got, w.Mask)
			}
		}
		for _, p := range e.Probes {
			probes++
			if n, end := pinAccept(g, []byte(p.Doc)); n != p.N || end != p.End {
				t.Errorf("%s: %q is accepted for %d bytes (may end: %v); pinned %d (may end: %v)", e.Name, p.Doc, n, end, p.N, p.End)
			}
		}
	}
	if len(entries) < 60 || walks < 400 || probes < 1000 {
		t.Fatalf("the pin holds %d grammars, %d walks and %d probes: too few to pin anything", len(entries), walks, probes)
	}
}

// pinAccept feeds doc to g byte by byte as the masker would, and returns how many leading bytes g accepted and whether
// the document may end after them.
func pinAccept(g Grammar, doc []byte) (n int, end bool) {
	g.Reset()
	for i := range doc {
		if !g.TryBytes(doc[i : i+1]) {
			return i, g.CanEnd()
		}
		g.Commit(doc[i : i+1])
	}
	return len(doc), g.CanEnd()
}

// pinMask hashes the grammar's mask-level behaviour along doc: at every prefix, the set of legal next bytes and CanEnd.
func pinMask(g Grammar, doc []byte) string {
	h := fnv.New64a()
	g.Reset()
	var set [33]byte
	for i := 0; ; i++ {
		clear(set[:])
		for b := range 256 {
			if g.TryBytes([]byte{byte(b)}) {
				set[b/8] |= 1 << (b % 8)
			}
		}
		if g.CanEnd() {
			set[32] = 1
		}
		h.Write(set[:])
		if i == len(doc) {
			break
		}
		g.Commit(doc[i : i+1])
	}
	return hex.EncodeToString(h.Sum(nil))
}

// pinWalkDoc takes a random path through g to a complete document. It prefers closing delimiters, so strings and
// containers stay short, and writes whitespace only where nothing else is legal (inside a key or a literal).
func pinWalkDoc(g Grammar, rng *rand.Rand) ([]byte, bool) {
	g.Reset()
	var doc []byte
	for range 600 {
		var ascii, other, ws, closers, nonDigit []byte
		for v := range 256 {
			b := byte(v)
			if !g.TryBytes([]byte{b}) {
				continue
			}
			switch {
			case isWS(b):
				ws = append(ws, b)
			case b > 0x20 && b < 0x7f:
				ascii = append(ascii, b)
				if b == '"' || b == '}' || b == ']' {
					closers = append(closers, b)
				}
				if b < '0' || b > '9' {
					nonDigit = append(nonDigit, b)
				}
			default:
				other = append(other, b)
			}
		}
		if g.CanEnd() && (len(ascii)+len(other) == 0 || rng.IntN(2) == 0) {
			return doc, true
		}
		pool := ascii
		switch {
		case len(closers) > 0 && rng.IntN(2) == 0:
			pool = closers
		case len(nonDigit) > 0 && len(doc) > 0 && doc[len(doc)-1] >= '0' && doc[len(doc)-1] <= '9' && rng.IntN(10) < 7:
			pool = nonDigit
		case len(ascii) == 0 && len(other) > 0:
			pool = other
		case len(ascii) == 0:
			pool = ws
		}
		if len(pool) == 0 {
			return nil, false
		}
		b := pool[rng.IntN(len(pool))]
		g.Commit([]byte{b})
		doc = append(doc, b)
	}
	return nil, false
}

// pinMutate returns doc with one seeded edit.
func pinMutate(doc []byte, rng *rand.Rand) []byte {
	const alphabet = "\"{}[],:019a-.e \\tfn"
	out := slices.Clone(doc)
	i := rng.IntN(len(doc) + 1)
	c := alphabet[rng.IntN(len(alphabet))]
	switch op := rng.IntN(4); {
	case op == 0 && i < len(out):
		return slices.Delete(out, i, i+1)
	case op == 1 && i < len(out):
		out[i] = c
		return out
	case op == 2:
		return slices.Insert(out, i, c)
	}
	return out[:i]
}

// pinSources lists the grammars to pin: the schemas this package's tests and fuzz corpus compile, and tool parameter
// schemas for the lenient path. It names no schema that uses `$ref`, `anyOf`, `oneOf` or a type list.
func pinSources(t *testing.T) []pinEntry {
	t.Helper()
	var src []pinEntry
	add := func(name, schema string) { src = append(src, pinEntry{Name: name, Schema: schema}) }
	tool := func(name, tool, params string) { src = append(src, pinEntry{Name: name, Tool: tool, Schema: params}) }

	props := propertySchemas(t)
	for _, name := range slices.Sorted(maps.Keys(props)) {
		add("property/"+name, string(props[name]))
	}
	for i, s := range seedSchemas {
		add("seed/"+strconv.Itoa(i), s)
	}
	corpus, err := filepath.Glob("testdata/fuzz/FuzzJSONSchema/*")
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range corpus {
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		lines := strings.Split(string(raw), "\n")
		if len(lines) < 2 {
			continue
		}
		quoted, ok := strings.CutPrefix(lines[1], "[]byte(")
		if !ok {
			continue
		}
		schema, err := strconv.Unquote(strings.TrimSuffix(quoted, ")"))
		if err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		add("fuzz/"+filepath.Base(path), schema)
	}
	for _, name := range slices.Sorted(maps.Keys(stringFormats)) {
		add("format/"+name, `{"type":"string","format":"`+name+`"}`)
	}
	add("format/unknown is an annotation", `{"type":"string","format":"hostname"}`)
	add("format/in an object", `{"type":"object","properties":{"when":{"type":"string","format":"date"},"note":{"type":"string"}},"required":["when","note"],"additionalProperties":false}`)
	add("confidence schema", confSchema)
	add("prefilter schema", `{"type":"object","properties":{
		"kind":{"enum":["a","bc"]},"n":{"type":"integer"},"x":{"type":"number"},"ok":{"type":"boolean"},
		"tags":{"type":"array","items":{"type":"string"}},"inner":{"type":"object","properties":{"note":{"type":"string"}}}},
		"required":["kind","n"]}`)
	add("enum/html-unsafe", `{"enum":["<",">","="]}`)
	add("enum/large integer", `{"enum":[9007199254740993]}`)
	add("enum/shared prefixes", `{"enum":[1,10,100]}`)
	add("enum/mixed literals", `{"enum":["a",1,true,null,[1,2],{"k":"v"}]}`)
	add("enum beside a type", `{"type":"string","enum":[1,2]}`)
	add("const beside a type", `{"type":"integer","const":"x"}`)
	add("integer", `{"type":"integer"}`)
	add("integer/minimum 0", `{"type":"integer","minimum":0}`)
	add("number/minimum 0", `{"type":"number","minimum":0}`)
	add("object/closed and empty", `{"type":"object","additionalProperties":false}`)
	add("object/implied by properties", `{"properties":{"a":{"type":"string"},"b":{"type":"null"}},"required":["b"]}`)
	add("object/annotations", `{"$schema":"https://json-schema.org/draft/2020-12/schema","$id":"https://example.com/s","title":"T",
		"type":"object","properties":{"x":{"type":"string","description":"d","default":"q","examples":["e"],"$comment":"c",
		"deprecated":false,"readOnly":true,"writeOnly":false,"$id":"nested"}},"additionalProperties":false}`)
	add("array/nested three deep", string(nestedArraySchema(3)))
	add("array/of objects", `{"type":"array","maxItems":2,"items":{"type":"object","properties":{"id":{"type":"integer"},"tags":{"type":"array","items":{"type":"boolean"},"maxItems":1}},"required":["id"]}}`)

	type pinAddress struct {
		City string `json:"city"`
		Zip  string `json:"zip,omitempty"`
	}
	type pinPerson struct {
		Name    string     `json:"name"`
		Age     int        `json:"age"`
		Level   uint8      `json:"level"`
		Active  bool       `json:"active"`
		Tags    []string   `json:"tags"`
		Pair    [2]float64 `json:"pair"`
		Address pinAddress `json:"address"`
		Email   *string    `json:"email,omitempty"`
	}
	for name, v := range map[string]any{"struct/one string": wsOne{}, "struct/person": pinPerson{}} {
		schema, err := SchemaFromStruct(v)
		if err != nil {
			t.Fatal(err)
		}
		add(name, string(schema))
	}

	for _, tl := range unionTools {
		tool("tool/union/"+tl.Name, tl.Name, string(tl.Parameters))
	}
	for _, tl := range opencodeKenTools(t) {
		if tl.Name != "ken_search" { // its type lists are read now: TestToolCallGrammar_typeListIsHeldToItsTypes
			tool("tool/opencode/"+tl.Name, tl.Name, string(tl.Parameters))
		}
	}
	tool("tool/freeform object", "t", `{"type":"object","additionalProperties":true}`)
	tool("tool/open object keeps its declared keys", "t", `{"type":"object","properties":{"a":{"type":"string"}},"additionalProperties":true}`)
	tool("tool/untyped property", "t", `{"type":"object","properties":{"x":{},"n":{"type":"integer"}},"required":["x"]}`)
	tool("tool/unchecked assertions", "t", `{"type":"object","properties":{"s":{"type":"string","pattern":"^a","maxLength":3},"n":{"type":"number","maximum":5}}}`)
	tool("tool/allOf is any value", "t", `{"type":"object","properties":{"v":{"allOf":[{"type":"string"}]},"w":{"not":{"type":"null"}}}}`)
	tool("tool/array without items", "t", `{"type":"object","properties":{"l":{"type":"array"},"o":{"type":"object"}}}`)
	tool("tool/not an object schema", "t", `{"type":"string"}`)
	tool("tool/unknown type", "t", `{"type":"object","properties":{"v":{"type":"wat"}}}`)
	return src
}

// writeSchemaPin rewrites the pin from this tree: a fixed number of seeded walks through each grammar, and seeded edits
// of each walk. A source that does not compile is left out.
func writeSchemaPin(t *testing.T) {
	t.Helper()
	const walksPer, probesPer = 8, 3
	var entries []pinEntry
	for _, e := range pinSources(t) {
		e.Schema = strings.ToValidUTF8(e.Schema, "�")
		g, err := e.grammar()
		if err != nil {
			continue
		}
		seed := fnv.New64a()
		seed.Write([]byte(e.Name))
		rng := rand.New(rand.NewPCG(seed.Sum64(), 1))
		seen := map[string]bool{}
		for try := 0; try < 10*walksPer && len(e.Walks) < walksPer; try++ {
			doc, ok := pinWalkDoc(g, rng)
			if !ok || seen[string(doc)] || !utf8.Valid(doc) {
				continue
			}
			seen[string(doc)] = true
			e.Walks = append(e.Walks, pinWalk{Doc: string(doc), Mask: pinMask(g, doc)})
			for range probesPer {
				m := pinMutate(doc, rng)
				if !utf8.Valid(m) || seen[string(m)] {
					continue
				}
				seen[string(m)] = true
				n, end := pinAccept(g, m)
				e.Probes = append(e.Probes, pinProbe{Doc: string(m), N: n, End: end})
			}
		}
		if len(e.Walks) == 0 {
			t.Fatalf("%s: no walk through the grammar completed", e.Name)
		}
		entries = append(entries, e)
	}
	out, err := json.MarshalIndent(entries, "", " ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(schemaPinPath, append(out, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
}
