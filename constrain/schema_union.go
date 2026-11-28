package constrain

import (
	"bytes"
	"fmt"
	"maps"
	"math"
	"slices"
	"strconv"
	"strings"
)

// Unions. A `type` list, `anyOf` and `oneOf` say a value is one of several alternatives. The grammar follows a value
// with one frame, so it holds a union only in a form where the value's first byte picks the alternative
// (schemaGrammar.dispatch): compiling merges the alternatives until no two can begin with the same byte, and a union
// that cannot be merged so is a compile error. Object alternatives merge into a variant (schema_variant.go). A frame
// gains no field for a union.
//
// `oneOf` is `anyOf` with the promise that no value matches two alternatives. The grammar does not count matches, so
// a strict compile takes a `oneOf` only when excludes shows every pair of alternatives apart; a lenient one holds it
// as `anyOf`, which refuses no value the `oneOf` allows.

const (
	// maxUnionAlts caps the alternatives of one anyOf or oneOf, the objects a union holds once the unions inside it
	// are counted, and the literals it names between its alternatives (an enum's cap).
	maxUnionAlts = 64
	// maxUnionWork caps the comparisons a document's oneOf keywords cost (unionState.excludes).
	maxUnionWork = 1 << 20
)

// unionAlt is a compiled alternative and its index in the schema's list, which an error names.
type unionAlt struct {
	n   *node
	src int
}

// valueClass is what a value's first byte says of it. Two typed alternatives of one class begin with the same bytes.
type valueClass int

const (
	clsObject valueClass = iota
	clsArray
	clsString
	clsNumber
	clsLiteral // true, false, null: only ever an enum's literal
)

var classNames = [clsLiteral]string{"objects", "arrays", "strings", "numbers"}

// isUnion reports that n is made of alternatives: a union by first byte, or a variant.
func isUnion(n *node) bool { return n.kind == kUnion || n.kind == kVariant }

func classOfByte(b byte) valueClass {
	switch b {
	case '{':
		return clsObject
	case '[':
		return clsArray
	case '"':
		return clsString
	case 't', 'f', 'n':
		return clsLiteral
	}
	return clsNumber
}

// classOf is the class of a typed node: an object, an array, a string or a number.
func classOf(n *node) valueClass {
	switch n.kind {
	case kObject, kVariant:
		return clsObject
	case kArray:
		return clsArray
	case kString:
		return clsString
	}
	return clsNumber
}

// unionKeyword returns the union keyword s carries, "" when it has none.
func unionKeyword(s map[string]any) string {
	for _, key := range []string{"anyOf", "oneOf"} {
		if _, ok := s[key]; ok {
			return key
		}
	}
	return ""
}

// compileUnion compiles an anyOf or oneOf, each alternative at the union's own nesting level. Beside the keyword only
// annotations are allowed, as beside a `$ref`. `discriminator` counts as one: it names the property that tells object
// alternatives apart and asserts nothing the alternatives do not.
func (c compiler) compileUnion(s map[string]any, key string, depth int) (*node, error) {
	for _, k := range slices.Sorted(maps.Keys(s)) {
		if k != key && k != "discriminator" && !annotationKeywords[k] {
			return nil, fmt.Errorf("constrain: %q beside %s is not supported (the two would have to be intersected); move it into the alternatives", k, key)
		}
	}
	list, ok := s[key].([]any)
	if !ok || len(list) == 0 {
		return nil, fmt.Errorf("constrain: %s must be a non-empty array", key)
	}
	if len(list) > maxUnionAlts {
		return nil, fmt.Errorf("constrain: %s with more than %d alternatives is not supported", key, maxUnionAlts)
	}
	alts := make([]unionAlt, 0, len(list))
	for i, raw := range list {
		alt, ok := raw.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("constrain: %s alternative %d is not a schema", key, i)
		}
		n, err := c.compile(alt, depth)
		if err != nil {
			return nil, fmt.Errorf("constrain: %s alternative %d: %w", key, i, err)
		}
		if n != neverNode {
			alts = append(alts, unionAlt{n, i})
		}
	}
	if key == "oneOf" && !c.lenient {
		c.unions.oneOf = true
		for i, a := range alts {
			for _, b := range alts[i+1:] {
				if c.unions.excludes(a.n, b.n) && c.unions.excludes(b.n, a.n) {
					continue
				}
				if c.unions.work > maxUnionWork {
					return nil, fmt.Errorf("constrain: telling the schema's oneOf alternatives apart takes more than %d comparisons", maxUnionWork)
				}
				return nil, fmt.Errorf("constrain: oneOf alternatives %d and %d can both match a value, as far as their types, literals, required properties and array lengths show, so \"exactly one\" would not be enforced; make them exclude each other, or use anyOf", a.src, b.src)
			}
		}
	}
	return c.merge(key, alts)
}

// compileTypeList compiles a `type` list: the schema read once for each type it names.
func (c compiler) compileTypeList(s map[string]any, list []any, depth int) (*node, error) {
	if len(list) == 0 {
		return nil, fmt.Errorf("constrain: \"type\" lists no types")
	}
	var alts []unionAlt
	var names []string
	for i, raw := range list {
		typ, ok := raw.(string)
		if !ok {
			return nil, fmt.Errorf("constrain: \"type\" list entry %d is not a type name", i)
		}
		if slices.Contains(names, typ) {
			continue
		}
		names = append(names, typ)
		n, err := c.compileType(s, typ, depth)
		if err != nil {
			return nil, err
		}
		alts = append(alts, unionAlt{n, i})
	}
	return c.merge("\"type\" list", alts)
}

// merge makes one node of a union's alternatives, such that the first byte of a value picks at most one of its
// alternatives. Typed alternatives of one class collapse into the one that covers the others; literals pool into one
// enum, less those a typed alternative accepts anyway. What is left beginning with the same byte is an error, in which
// what names the keyword. The result is neverNode when no alternative is left.
func (c compiler) merge(what string, alts []unionAlt) (*node, error) {
	var typed [clsLiteral][]unionAlt
	objects := map[int]bool{} // the object alternatives so far, by structure
	type literal struct {
		enc []byte
		src int
	}
	var lits []literal
	pooled := map[string]bool{}
	for i := 0; i < len(alts); i++ {
		a := alts[i]
		switch a.n.kind {
		case kNever:
		case kAny:
			return anyNode, nil
		case kUnion, kVariant:
			for _, sub := range a.n.alts {
				alts = append(alts, unionAlt{sub, a.src})
			}
		case kEnum:
			for _, e := range a.n.enum {
				if !pooled[string(e)] {
					pooled[string(e)] = true
					lits = append(lits, literal{e, a.src})
				}
			}
		case kObject: // a free object covers every other; otherwise two objects are one only when they are the same
			if id := c.unions.id(a.n); !objects[id] {
				objects[id] = true
				typed[clsObject] = append(typed[clsObject], a)
			}
		default:
			cls := classOf(a.n)
			if slices.ContainsFunc(typed[cls], func(b unionAlt) bool { return c.unions.covers(b.n, a.n) }) {
				continue
			}
			typed[cls] = slices.DeleteFunc(typed[cls], func(b unionAlt) bool { return c.unions.covers(a.n, b.n) })
			typed[cls] = append(typed[cls], a)
		}
	}
	if i := slices.IndexFunc(typed[clsObject], func(a unionAlt) bool { return a.n.free }); i >= 0 {
		typed[clsObject] = typed[clsObject][i : i+1]
	}
	var out []*node
	for cls, group := range typed {
		switch {
		case len(group) == 0:
			continue
		case len(group) > 1 && valueClass(cls) == clsObject: // all different, and none free
			v, err := c.variant(what, group)
			if err != nil {
				return nil, err
			}
			typed[cls] = []unionAlt{{v, group[0].src}}
		case len(group) > 1:
			i, j := min(group[0].src, group[1].src), max(group[0].src, group[1].src)
			return nil, fmt.Errorf("constrain: %s alternatives %d and %d are both %s, and neither covers the other; the grammar picks an alternative by a value's first byte, so it cannot offer both", what, i, j, classNames[cls])
		}
		out = append(out, typed[cls][0].n)
	}
	var enum [][]byte
	for _, lit := range lits {
		cls := classOfByte(lit.enc[0])
		if cls != clsLiteral && len(typed[cls]) == 1 {
			t := typed[cls][0]
			if nodeAccepts(t.n, lit.enc) {
				continue
			}
			if cls != clsNumber || lit.enc[0] != '-' || !t.n.nonNeg {
				return nil, fmt.Errorf("constrain: %s alternative %d: the literal %s begins as alternative %d's values do, and that alternative does not accept it; the grammar picks an alternative by a value's first byte, so it cannot offer both", what, lit.src, lit.enc, t.src)
			}
		}
		enum = append(enum, lit.enc)
	}
	if len(enum) > maxUnionAlts {
		return nil, fmt.Errorf("constrain: %s names more than %d literals between its alternatives", what, maxUnionAlts)
	}
	if len(enum) > 0 {
		out = append(out, &node{kind: kEnum, enum: enum})
	}
	switch len(out) {
	case 0:
		return neverNode, nil
	case 1:
		return out[0], nil
	}
	return &node{kind: kUnion, alts: out}, nil
}

// nodeAccepts reports whether the grammar of n accepts doc as a whole value.
func nodeAccepts(n *node, doc []byte) bool {
	g := &schemaGrammar{root: n}
	g.Reset()
	for _, b := range doc {
		if !g.step(b) {
			return false
		}
	}
	return g.CanEnd()
}

// unionState is what a document's unions share while it compiles: nodes numbered by structure, and the pairs excludes
// has decided.
type unionState struct {
	byNode   map[*node]int
	byKey    map[string]int
	excluded map[[2]*node]bool
	work     int // excludes calls, against maxUnionWork

	// oneOf: a oneOf's alternatives were checked with excludes. untypedCut: a schema with `properties` and no `type`
	// compiled to neverNode, which excludes cannot see. A document with both is refused (compileDocument).
	oneOf, untypedCut bool
}

// id numbers a node by its structure: two nodes with one number accept the same documents, and excludes reads them
// alike.
func (u *unionState) id(n *node) int {
	if id, ok := u.byNode[n]; ok {
		return id
	}
	var k strings.Builder
	switch n.kind {
	case kObject:
		k.WriteByte('O')
		if n.free {
			k.WriteByte('*')
		}
		if n.untyped {
			k.WriteByte('?')
		}
		for i, p := range n.props {
			k.WriteString(strconv.Quote(p.name))
			k.WriteString(strconv.Itoa(u.id(p.schema)))
			if n.requiredMask&(1<<uint(i)) != 0 {
				k.WriteByte('!')
			}
			k.WriteByte(',')
		}
	case kArray:
		k.WriteByte('A')
		if n.maxItems != 0 { // an array that takes no items is the same array whatever its items schema
			fmt.Fprintf(&k, "%d:%d:%d", u.id(n.items), n.minItems, n.maxItems)
		}
	case kString:
		fmt.Fprintf(&k, "S%d", n.format)
	case kNumber:
		fmt.Fprintf(&k, "N%t%t", n.intOnly, n.nonNeg)
	case kEnum:
		k.WriteByte('E')
		lits := slices.Clone(n.enum)
		slices.SortFunc(lits, bytes.Compare)
		for _, l := range lits {
			fmt.Fprintf(&k, "%d:%s", len(l), l)
		}
	case kUnion, kVariant:
		k.WriteByte('U')
		ids := make([]int, len(n.alts))
		for i, alt := range n.alts {
			ids[i] = u.id(alt)
		}
		slices.Sort(ids)
		fmt.Fprint(&k, ids)
	default: // kAny, kNever
		fmt.Fprintf(&k, "K%d", n.kind)
	}
	if u.byKey == nil {
		u.byNode, u.byKey = map[*node]int{}, map[string]int{}
	}
	id, ok := u.byKey[k.String()]
	if !ok {
		id = len(u.byKey)
		u.byKey[k.String()] = id
	}
	u.byNode[n] = id
	return id
}

// covers reports that a accepts every value b accepts. It is conservative: false when that is not shown.
func (u *unionState) covers(a, b *node) bool {
	switch {
	case a.kind == kAny || b.kind == kNever || u.id(a) == u.id(b):
		return true
	case isUnion(b):
		return !slices.ContainsFunc(b.alts, func(alt *node) bool { return !u.covers(a, alt) })
	case isUnion(a):
		return slices.ContainsFunc(a.alts, func(alt *node) bool { return u.covers(alt, b) })
	case b.kind == kEnum:
		return !slices.ContainsFunc(b.enum, func(lit []byte) bool { return !nodeAccepts(a, lit) })
	case a.kind != b.kind:
		return false
	}
	switch a.kind {
	case kObject:
		return a.free
	case kArray:
		return a.minItems <= b.minItems && (a.maxItems < 0 || (b.maxItems >= 0 && b.maxItems <= a.maxItems)) &&
			(b.maxItems == 0 || u.covers(a.items, b.items))
	case kString:
		return a.format == fmtNone
	case kNumber:
		return (!a.intOnly || b.intOnly) && (!a.nonNeg || b.nonNeg)
	}
	return false
}

// excludes reports that no value the grammar writes for a is accepted by the schema b was compiled from, with b read
// as JSON Schema reads it: an object's undeclared keys are allowed, and an untyped object schema matches whatever is
// not an object. It is conservative: false when the two are not shown apart. It looks at types, literals, the keys
// objects require and declare, and array lengths.
func (u *unionState) excludes(a, b *node) bool {
	if u.work++; u.work > maxUnionWork {
		return false
	}
	switch {
	case a.kind == kNever || b.kind == kNever:
		return true
	case a.kind == kAny || b.kind == kAny:
		return false
	case isUnion(a):
		return !slices.ContainsFunc(a.alts, func(alt *node) bool { return !u.excludes(alt, b) })
	case isUnion(b):
		return !slices.ContainsFunc(b.alts, func(alt *node) bool { return !u.excludes(a, alt) })
	case b.untyped && a.kind != kObject:
		return false // b's schema has no `type`, so it matches whatever is not an object
	case a.kind == kEnum && b.kind == kEnum:
		for _, la := range a.enum {
			if slices.ContainsFunc(b.enum, func(lb []byte) bool { return !literalsDiffer(la, lb) }) {
				return false
			}
		}
		return true
	case a.kind == kEnum:
		return !slices.ContainsFunc(a.enum, func(lit []byte) bool { return !literalOutside(b, lit) })
	case b.kind == kEnum:
		return !slices.ContainsFunc(b.enum, func(lit []byte) bool { return !literalOutside(a, lit) })
	case a.kind != b.kind:
		return true
	case a.kind != kObject && a.kind != kArray:
		return false // two strings, or two numbers
	}
	pair := [2]*node{a, b}
	if done, ok := u.excluded[pair]; ok {
		return done
	}
	var res bool
	if a.kind == kArray {
		res = (a.maxItems >= 0 && a.maxItems < b.minItems) || (b.maxItems >= 0 && a.minItems > b.maxItems) ||
			(a.minItems > 0 && u.excludes(a.items, b.items))
	} else if !a.free && !b.free {
		res = u.objectsExclude(a, b)
	}
	if u.excluded == nil {
		u.excluded = map[[2]*node]bool{}
	}
	u.excluded[pair] = res
	return res
}

// objectsExclude is excludes for two objects with declared properties. The object written for a fails b when b
// requires a key a does not declare, or when a key both declare, which one of them requires, has values that exclude:
// if a requires it the key is there with a value b refuses, and if only b does, the key is either missing or so valued.
func (u *unionState) objectsExclude(a, b *node) bool {
	for j, bp := range b.props {
		i := propIndex(a, bp.name)
		if i < 0 {
			if b.requiredMask&(1<<uint(j)) != 0 {
				return true
			}
			continue
		}
		if (a.requiredMask&(1<<uint(i)) != 0 || b.requiredMask&(1<<uint(j)) != 0) && u.excludes(a.props[i].schema, bp.schema) {
			return true
		}
	}
	return false
}

// propIndex is the index of the property name in n's sorted property table, -1 when n does not declare it.
func propIndex(n *node, name string) int {
	i, ok := slices.BinarySearchFunc(n.props, name, func(p propNode, name string) int { return strings.Compare(p.name, name) })
	if !ok {
		return -1
	}
	return i
}

// literalsDiffer reports that two enum literals are shown to be different values. Numbers are compared as float64s,
// which tells apart only numbers that round apart; object and array literals are told apart only by their kind.
func literalsDiffer(a, b []byte) bool {
	ca, cb := classOfByte(a[0]), classOfByte(b[0])
	switch {
	case ca != cb:
		return true
	case bytes.Equal(a, b) || ca == clsObject || ca == clsArray:
		return false
	case ca == clsNumber:
		fa, errA := strconv.ParseFloat(string(a), 64)
		fb, errB := strconv.ParseFloat(string(b), 64)
		return errA == nil && errB == nil && fa != fb
	}
	return true // a string's encoding is one string's only; true, false and null are themselves
}

// literalOutside reports that a typed node n and the literal lit share no value: they are of different types, or lit
// is a negative number and n takes none, or lit has a fractional part and n takes integers.
func literalOutside(n *node, lit []byte) bool {
	if classOfByte(lit[0]) != classOf(n) {
		return true
	}
	if n.kind != kNumber {
		return false
	}
	f, err := strconv.ParseFloat(string(lit), 64)
	return err == nil && ((n.nonNeg && f < 0) || (n.intOnly && f != math.Trunc(f)))
}
