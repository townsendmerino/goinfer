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
// and cut there (compiler.overDepth), while nesting the cycle does not account for stays the nesting error
// (refState.err). The cut is made when the schema is compiled, so the grammar's frame stack keeps the bound every
// other schema has and a mask step does nothing new (docs/code-notes/constrain.md#Local references).

const (
	// maxRefWork caps the schemas compiled inside reference expansions, and the schemas read for the references' shape
	// (refState.readShape). A target is compiled once per nesting level it is reached at, so a document of definitions
	// that refer to each other costs up to maxSchemaDepth times its size.
	maxRefWork = 1 << 16
	// maxRefChain caps the references followed one into the next with no object or array in between.
	maxRefChain = 64
)

// refState is one document's references: its root, which every pointer is read from, and the targets compiled so far.
type refState struct {
	root    map[string]any
	lenient bool              // the compile is lenient: it has no error to give, so it keeps the cut (refState.err)
	base    int               // the nesting level the root is compiled at
	done    map[refSite]*node // a target compiled at a nesting level
	open    map[refSite]bool  // targets being compiled: one met again at its own level is a cycle with no nesting in it
	chain   [maxSchemaDepth + 1]int
	work    int       // schemas compiled inside an expansion, against maxRefWork
	used    bool      // the document has a reference
	rebased bool      // a schema other than the root carries $id, which re-bases the references under it
	shape   *refShape // what the references amount to, once overDepth has asked
}

// refSite is a target, by its canonical pointer, at a nesting level.
type refSite struct {
	ptr   string
	depth int
}

func newRefState(root map[string]any, lenient bool, base int) *refState {
	return &refState{root: root, lenient: lenient, base: base, done: map[refSite]*node{}, open: map[refSite]bool{}}
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

// err is the document-level refusal. A reference in a document that re-bases references is one this package would
// resolve against the wrong root. And a strict compile whose cycle was cut still refuses a document that nests past
// maxSchemaDepth with no cycle to account for it: on a path that writes the nesting out (refShape.plain), as its
// inlined form would be refused, or on the way to a schema no path reaches sooner (refShape.far), which the cut kept
// out of the compile. A lenient compile keeps the cut, which drops only documents that nest past maxSchemaDepth.
func (r *refState) err() error {
	if r.used && r.rebased {
		return fmt.Errorf("constrain: $ref in a document with a nested $id is not supported: the $id re-bases the references under it")
	}
	if sh := r.shape; sh != nil && sh.cyclic && !r.lenient && r.base+max(sh.plain, sh.far) > maxSchemaDepth {
		return fmt.Errorf("constrain: schema nesting exceeds %d levels, not counting what its $ref cycles repeat; a cycle is cut at that depth, nesting written out past it is not", maxSchemaDepth)
	}
	return nil
}

func sameMap(a, b map[string]any) bool {
	return reflect.ValueOf(a).UnsafePointer() == reflect.ValueOf(b).UnsafePointer()
}

// overDepth decides a schema nested past maxSchemaDepth. Inside the expansion of a document whose references form a
// cycle it is neverNode, which the enclosing object, array or union drops; anywhere else it is the nesting error.
// Whether a document cut so also nests too deep with no cycle to account for it is refState.err's to say.
func (c compiler) overDepth() (*node, error) {
	if r := c.refs; len(r.open) > 0 {
		sh, err := r.readShape()
		if err != nil {
			return nil, err
		}
		if sh.cyclic {
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

// refShape is what the compiler asks of a document's references. plain is the levels the document nests with no cycle
// to account for them: the deepest path from the root that counts the nesting written out, enters a cycle at whichever
// of its schemas writes out most, and follows none of the cycle's own references. far is the level of the schema
// furthest from the root, each schema taken at the least nesting a path reaches it by: a schema past maxSchemaDepth
// by that count is one the compiler never reaches (docs/code-notes/constrain.md#Local references).
type refShape struct {
	cyclic bool // the references form a cycle
	plain  int
	far    int
}

// readShape reads the shape once: Tarjan's algorithm over the schemas the compiler would compile, which closes a
// strongly connected component after every component it leads to.
func (r *refState) readShape() (*refShape, error) {
	if r.shape == nil {
		a := &shapeReader{refs: r, nodes: map[uintptr]*shapeNode{}}
		root, err := a.visit(r.root)
		if err != nil {
			return nil, err
		}
		r.shape = &refShape{cyclic: a.cyclic, plain: root.plain, far: farthest(root)}
	}
	return r.shape, nil
}

type shapeReader struct {
	refs   *refState
	nodes  map[uintptr]*shapeNode // by the identity of the schema's map
	stack  []*shapeNode           // the schemas whose component is still open
	comps  int
	cyclic bool
}

// shapeNode is a schema, with an edge to each schema the compiler compiles from it.
type shapeNode struct {
	edges      []shapeEdge
	index, low int  // Tarjan's numbering, from 1
	comp       int  // its component, from 1, once that is closed
	plain      int  // refShape.plain of the document from this schema down
	walk       int  // walked's result plus 1; negative while it is being computed
	reached    bool // farthest has come to it
}

type shapeEdge struct {
	to     *shapeNode
	nested bool // to is one nesting level down
	ref    bool // to is a reference's target
}

func (e shapeEdge) levels() int {
	if e.nested {
		return 1
	}
	return 0
}

// visit numbers s and the schemas below it, and closes a component once nothing below it leads back above it.
func (a *shapeReader) visit(s map[string]any) (*shapeNode, error) {
	if len(a.nodes) == maxRefWork {
		return nil, fmt.Errorf("constrain: the schema's references pass %d schemas", maxRefWork)
	}
	v := &shapeNode{index: len(a.nodes) + 1}
	v.low = v.index
	a.nodes[reflect.ValueOf(s).Pointer()] = v
	a.stack = append(a.stack, v)
	for _, kid := range a.refs.below(s) {
		w := a.nodes[reflect.ValueOf(kid.s).Pointer()]
		switch {
		case w == nil:
			var err error
			if w, err = a.visit(kid.s); err != nil {
				return nil, err
			}
			v.low = min(v.low, w.low)
		case w.comp == 0: // still open, so kid leads back to it
			v.low = min(v.low, w.index)
		}
		v.edges = append(v.edges, shapeEdge{w, kid.nested, kid.ref})
	}
	if v.low == v.index {
		a.close(v)
	}
	return v, nil
}

// close ends the component v began, which is v and the schemas above it on the stack. Its members share one count, the
// deepest that any of them writes out: going round a cycle reaches every member from any other.
func (a *shapeReader) close(v *shapeNode) {
	i := len(a.stack) - 1
	for a.stack[i] != v {
		i--
	}
	members := a.stack[i:]
	a.stack = a.stack[:i]
	a.comps++
	for _, m := range members {
		m.comp = a.comps
	}
	deep := 0
	for _, m := range members {
		deep = max(deep, a.walked(m))
	}
	for _, m := range members {
		m.plain = deep
	}
	if len(members) > 1 || slices.ContainsFunc(v.edges, func(e shapeEdge) bool { return e.to == v }) {
		a.cyclic = true
	}
}

// walked is the deepest nesting written out below v: it follows no reference inside v's own component, and leaves the
// component by any edge, for what the schema there counts.
func (a *shapeReader) walked(v *shapeNode) int {
	switch {
	case v.walk > 0:
		return v.walk - 1
	case v.walk < 0: // nesting with no reference in it that never ends: a schema built in Go that holds itself
		return maxSchemaDepth + 1
	}
	v.walk = -1
	deep := 0
	for _, e := range v.edges {
		switch {
		case e.to.comp != v.comp:
			deep = max(deep, e.levels()+e.to.plain)
		case !e.ref:
			deep = max(deep, e.levels()+a.walked(e.to))
		}
	}
	v.walk = deep + 1
	return deep
}

// farthest is refShape.far for the document whose root schema is root: the last level a search by levels comes to, a
// reference's target and a union's alternatives being of the level they are met at.
func farthest(root *shapeNode) int {
	far := 0
	level := []*shapeNode{root}
	for depth := 0; len(level) > 0; depth++ {
		var deeper []*shapeNode
		for i := 0; i < len(level); i++ { // level grows while it is read
			v := level[i]
			if v.reached {
				continue
			}
			v.reached, far = true, depth
			for _, e := range v.edges {
				switch {
				case e.to.reached:
				case e.nested:
					deeper = append(deeper, e.to)
				default:
					level = append(level, e.to)
				}
			}
		}
		level = deeper
	}
	return far
}

// refKid is a schema the compiler compiles from another one.
type refKid struct {
	s      map[string]any
	nested bool // one nesting level down
	ref    bool // as a reference's target
}

// below lists the schemas a strict compile compiles from s, as compileStrict reads s: a reference's target, a union's
// alternatives, an object's properties and an array's items.
func (r *refState) below(s map[string]any) []refKid {
	var kids []refKid
	add := func(v any, nested, ref bool) {
		if m, ok := v.(map[string]any); ok {
			kids = append(kids, refKid{m, nested, ref})
		}
	}
	if raw, has := s["$ref"]; has {
		if ref, ok := raw.(string); ok {
			if target, _, err := r.resolve(ref); err == nil {
				add(target, false, true)
			}
		}
		return kids
	}
	if key := unionKeyword(s); key != "" {
		alts, _ := s[key].([]any)
		for _, alt := range alts {
			add(alt, false, false)
		}
		return kids
	}
	if _, ok := s["const"]; ok {
		return nil
	}
	if _, ok := s["enum"]; ok {
		return nil
	}
	_, object := s["properties"] // with no type, a schema with properties is an object
	array := false
	switch typ := s["type"].(type) {
	case []any:
		object, array = listsType(typ, "object"), listsType(typ, "array")
	case string:
		if typ != "" {
			object, array = typ == "object", typ == "array"
		}
	}
	if props, _ := s["properties"].(map[string]any); object && len(props) <= 64 {
		for _, name := range slices.Sorted(maps.Keys(props)) {
			add(props[name], true, false)
		}
	}
	if array {
		add(s["items"], true, false)
	}
	return kids
}

// listsType reports that a `type` list names typ.
func listsType(list []any, typ string) bool {
	return slices.ContainsFunc(list, func(e any) bool {
		name, _ := e.(string)
		return name == typ
	})
}
