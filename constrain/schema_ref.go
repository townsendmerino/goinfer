package constrain

import (
	"fmt"
	"maps"
	"net/url"
	"reflect"
	"slices"
	"strconv"
	"strings"
)

// Local references. A `$ref` names another schema of the same document by JSON Pointer ("#/$defs/Name") and compiles
// as that schema would in its place, at the reference's own nesting level, so a document with references accepts
// what its inlined form accepts. A reference cycle has no finite inlined form: it is expanded down to maxSchemaDepth
// and cut there (compiler.overDepth). The cut is made when the schema is compiled, so the grammar's frame stack keeps
// the bound every other schema has and a mask step does nothing new (docs/code-notes/constrain.md#Local references).

const (
	// maxRefWork caps the schemas compiled inside reference expansions, and the schemas read to look for a reference
	// cycle. A target is compiled once per nesting level it is reached at, so a document of definitions that refer to
	// each other costs up to maxSchemaDepth times its size.
	maxRefWork = 1 << 16
	// maxRefChain caps the references followed one into the next with no object or array in between.
	maxRefChain = 64
)

// refState is one document's references: its root, which every pointer is read from, and the targets compiled so far.
type refState struct {
	root    map[string]any
	done    map[refSite]*node // a target compiled at a nesting level
	open    map[refSite]bool  // targets being compiled: one met again at its own level is a cycle with no nesting in it
	chain   [maxSchemaDepth + 1]int
	work    int  // schemas compiled inside an expansion, against maxRefWork
	used    bool // the document has a reference
	rebased bool // a schema other than the root carries $id, which re-bases the references under it
	cycle   int  // whether the document's references form a cycle: 0 unknown, 1 no, 2 yes
}

// refSite is a target, by its canonical pointer, at a nesting level.
type refSite struct {
	ptr   string
	depth int
}

func newRefState(root map[string]any) *refState {
	return &refState{root: root, done: map[refSite]*node{}, open: map[refSite]bool{}}
}

// enter is called for every schema compiled. It counts the ones inside a reference expansion against maxRefWork and
// notes a nested $id.
func (r *refState) enter(s map[string]any) error {
	if _, has := s["$id"].(string); has && !sameMap(s, r.root) {
		r.rebased = true
	}
	if len(r.open) == 0 {
		return nil
	}
	if r.work++; r.work > maxRefWork {
		return fmt.Errorf("constrain: the schema's $ref expansions pass %d schemas", maxRefWork)
	}
	return nil
}

// err is the document-level refusal: a reference in a document that re-bases references is one this package would
// resolve against the wrong root.
func (r *refState) err() error {
	if r.used && r.rebased {
		return fmt.Errorf("constrain: $ref in a document with a nested $id is not supported: the $id re-bases the references under it")
	}
	return nil
}

func sameMap(a, b map[string]any) bool {
	return reflect.ValueOf(a).UnsafePointer() == reflect.ValueOf(b).UnsafePointer()
}

// overDepth decides a schema nested past maxSchemaDepth. Inside the expansion of a document whose references form a
// cycle it is neverNode, which the enclosing object, array or union drops; anywhere else it is the nesting error, so
// references without a cycle fail exactly as their inlined form does.
func (c compiler) overDepth() (*node, error) {
	if r := c.refs; len(r.open) > 0 {
		cyclic, err := r.hasCycle()
		if err != nil {
			return nil, err
		}
		if cyclic {
			return neverNode, nil
		}
	}
	return nil, fmt.Errorf("constrain: schema nesting exceeds %d levels", maxSchemaDepth)
}

// compileRef compiles the schema a `$ref` names in the reference's place. Beside the reference only annotations are
// allowed: an assertion there would have to be intersected with the target.
func (c compiler) compileRef(s map[string]any, depth int) (*node, error) {
	r := c.refs
	ref, ok := s["$ref"].(string)
	if !ok {
		return nil, fmt.Errorf("constrain: $ref must be a string")
	}
	for k := range s {
		if k != "$ref" && !annotationKeywords[k] {
			return nil, fmt.Errorf("constrain: %q beside $ref %q is not supported (the two would have to be intersected); move it into the referenced schema", k, ref)
		}
	}
	target, ptr, err := r.resolve(ref)
	if err != nil {
		return nil, err
	}
	r.used = true
	site := refSite{ptr, depth}
	if n, ok := r.done[site]; ok {
		return n, nil
	}
	if r.open[site] {
		return nil, fmt.Errorf("constrain: $ref %q refers to itself with no object or array in between", ref)
	}
	if r.chain[depth] == maxRefChain {
		return nil, fmt.Errorf("constrain: $ref %q ends a chain of more than %d references", ref, maxRefChain)
	}
	r.open[site] = true
	r.chain[depth]++
	n, err := c.compile(target, depth)
	r.chain[depth]--
	delete(r.open, site)
	if err != nil {
		return nil, fmt.Errorf("constrain: $ref %q: %w", ref, err)
	}
	r.done[site] = n
	return n, nil
}

// resolve returns the schema a local reference names, and the pointer to it in canonical form.
func (r *refState) resolve(ref string) (target map[string]any, ptr string, err error) {
	toks, err := pointerTokens(ref)
	if err != nil {
		return nil, "", err
	}
	var cur any = r.root
	var canon strings.Builder
	for _, tok := range toks {
		if m, ok := cur.(map[string]any); ok {
			if _, has := m["$id"].(string); has && !sameMap(m, r.root) {
				r.rebased = true
			}
		}
		next, found := pointerStep(cur, tok)
		if !found {
			return nil, "", fmt.Errorf("constrain: $ref %q: nothing in the document at %q", ref, tok)
		}
		cur = next
		canon.WriteByte('/')
		canon.WriteString(strings.ReplaceAll(strings.ReplaceAll(tok, "~", "~0"), "/", "~1"))
	}
	target, ok := cur.(map[string]any)
	if !ok {
		return nil, "", fmt.Errorf("constrain: $ref %q does not name a schema object", ref)
	}
	return target, canon.String(), nil
}

// pointerTokens splits a local reference ("#/$defs/Name") into the tokens of its JSON Pointer, decoded. A reference
// out of the document, or one by anchor, is refused.
func pointerTokens(ref string) ([]string, error) {
	frag, local := strings.CutPrefix(ref, "#")
	if !local {
		return nil, fmt.Errorf("constrain: $ref %q is not supported: only a reference into the same document (\"#/$defs/Name\") is", ref)
	}
	if frag != "" && frag[0] != '/' {
		return nil, fmt.Errorf("constrain: $ref %q is not supported: it names an anchor, not a JSON Pointer (\"#/$defs/Name\")", ref)
	}
	path, err := url.PathUnescape(frag)
	if err != nil {
		return nil, fmt.Errorf("constrain: $ref %q: %w", ref, err)
	}
	if path == "" {
		return nil, nil
	}
	toks := strings.Split(path[1:], "/")
	for i, tok := range toks {
		toks[i] = strings.ReplaceAll(strings.ReplaceAll(tok, "~1", "/"), "~0", "~")
	}
	return toks, nil
}

// pointerStep follows one pointer token from cur: a key of an object (decoded as a map, or as the template's ordered
// object) or an index of an array.
func pointerStep(cur any, tok string) (next any, found bool) {
	switch v := cur.(type) {
	case map[string]any:
		next, found = v[tok]
	case *orderedObj:
		next, found = v.vals[tok]
	case []any:
		if i, err := strconv.Atoi(tok); err == nil && i >= 0 && i < len(v) && strconv.Itoa(i) == tok {
			next, found = v[i], true
		}
	}
	return next, found
}

// hasCycle reports whether the references reachable from the root lead back to a schema they started in. It reads the
// schemas the compiler descends into (properties, items, and the alternatives of anyOf and oneOf), each target once.
func (r *refState) hasCycle() (bool, error) {
	if r.cycle != 0 {
		return r.cycle == 2, nil
	}
	const onPath, finished = 1, 2
	state := map[string]int{"": onPath}
	read := 0
	var visit func(s map[string]any) (bool, error)
	visit = func(s map[string]any) (bool, error) {
		if read++; read > maxRefWork {
			return false, fmt.Errorf("constrain: the schema's references pass %d schemas", maxRefWork)
		}
		if ref, ok := s["$ref"].(string); ok {
			target, ptr, err := r.resolve(ref)
			if err != nil || state[ptr] == finished {
				return false, nil
			}
			if state[ptr] == onPath {
				return true, nil
			}
			state[ptr] = onPath
			found, err := visit(target)
			state[ptr] = finished
			return found, err
		}
		var kids []any
		if props, ok := s["properties"].(map[string]any); ok {
			for _, name := range slices.Sorted(maps.Keys(props)) {
				kids = append(kids, props[name])
			}
		}
		kids = append(kids, s["items"])
		for _, k := range []string{"anyOf", "oneOf"} {
			alts, _ := s[k].([]any)
			kids = append(kids, alts...)
		}
		for _, kid := range kids {
			if m, ok := kid.(map[string]any); ok {
				if found, err := visit(m); found || err != nil {
					return found, err
				}
			}
		}
		return false, nil
	}
	found, err := visit(r.root)
	if err != nil {
		return false, err
	}
	r.cycle = 1
	if found {
		r.cycle = 2
	}
	return found, nil
}
