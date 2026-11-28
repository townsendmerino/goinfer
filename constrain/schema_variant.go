package constrain

import (
	"bytes"
	"fmt"
	"math/bits"
	"slices"
)

// Unions of objects. Two objects both begin with '{', so a value's first byte cannot pick between them. A union of
// objects compiles to one node instead, a variant: the alternatives' properties merged into one table, and each
// alternative's declared and required properties as masks over it. The grammar follows a variant with one frame, as it
// follows an object, and that frame carries the set of alternatives the keys and values written so far still fit:
//   - a key is legal when an alternative still live declares it, and writing it drops the alternatives that do not;
//   - '}' is legal when an alternative still live has every key it requires.
//
// A property whose schema is the same in every alternative that declares it is compiled once. One whose schema
// differs has to say, once its value is written, which alternatives that value fits (compiler.tagged): by the value's
// first byte when the schemas are of different types, by the literal when they are enums (a discriminator), and by
// the nested object's own keys when they are objects. Two schemas that would need more than that to tell apart (two
// arrays, an integer and a number, a literal beside a string) are a compile error naming the property
// (docs/code-notes/constrain.md#Variants: a union of objects).

// variantInfo is a variant's alternatives as masks: bit i of a mask over alternatives is n.alts[i], bit p of a mask
// over properties is n.props[p].
type variantInfo struct {
	declares []uint64 // per property: the alternatives that declare it
	keys     []uint64 // per alternative: the properties it declares
	required []uint64 // per alternative: the properties it requires
}

// keysOf is every property an alternative in live declares.
func (v *variantInfo) keysOf(live uint64) uint64 {
	var keys uint64
	for ; live != 0; live &= live - 1 {
		keys |= v.keys[bits.TrailingZeros64(live)]
	}
	return keys
}

// satisfied is the alternatives in live whose required properties are all in seen.
func (v *variantInfo) satisfied(live, seen uint64) uint64 {
	var done uint64
	for m := live; m != 0; m &= m - 1 {
		if i := bits.TrailingZeros64(m); v.required[i]&^seen == 0 {
			done |= 1 << uint(i)
		}
	}
	return done
}

// tagGroup is a schema one property has in some alternatives of a variant: mask is those alternatives, src the first
// one's index in the schema's list.
type tagGroup struct {
	n    *node
	mask uint64
	src  int
}

// variant merges two or more object alternatives, all different and none free, into one node.
func (c compiler) variant(what string, objs []unionAlt) (*node, error) {
	if len(objs) > maxUnionAlts {
		return nil, fmt.Errorf("constrain: %s with more than %d object alternatives is not supported", what, maxUnionAlts)
	}
	var names []string
	for _, o := range objs {
		for _, p := range o.n.props {
			names = append(names, p.name)
		}
	}
	slices.Sort(names)
	names = slices.Compact(names)
	if len(names) > 64 {
		return nil, fmt.Errorf("constrain: %s: its object alternatives declare more than 64 properties between them", what)
	}
	info := &variantInfo{declares: make([]uint64, len(names)), keys: make([]uint64, len(objs)), required: make([]uint64, len(objs))}
	n := &node{kind: kVariant, variant: info, props: make([]propNode, len(names))}
	for _, o := range objs {
		n.alts = append(n.alts, o.n)
	}
	for p, name := range names {
		var groups []tagGroup
		for i, o := range objs {
			j := propIndex(o.n, name)
			if j < 0 {
				continue
			}
			info.declares[p] |= 1 << uint(i)
			info.keys[i] |= 1 << uint(p)
			if o.n.requiredMask&(1<<uint(j)) != 0 {
				info.required[i] |= 1 << uint(p)
			}
			groups = c.unions.group(groups, tagGroup{o.n.props[j].schema, 1 << uint(i), o.src})
		}
		schema := groups[0].n
		if len(groups) > 1 {
			var err error
			if schema, err = c.tagged(what, name, groups); err != nil {
				return nil, err
			}
		}
		n.props[p] = propNode{name: name, schema: schema}
	}
	return n, nil
}

// group adds g to groups, into the group with its schema when there is one.
func (u *unionState) group(groups []tagGroup, g tagGroup) []tagGroup {
	for i := range groups {
		if u.id(groups[i].n) == u.id(g.n) {
			groups[i].mask |= g.mask
			return groups
		}
	}
	return append(groups, g)
}

var classArticles = [clsLiteral]string{"an object", "an array", "a string", "a number"}

// tagged compiles the property name of a variant, whose schema differs between the alternatives in groups. The node
// it returns accepts what any of the schemas accepts from an alternative still live, and tells the variant's frame
// which alternatives the value it read fits: its tags name them per alternative of a union or a nested variant, and
// per literal of an enum.
func (c compiler) tagged(what, name string, groups []tagGroup) (*node, error) {
	var parts []tagGroup
	for _, g := range groups {
		for _, n := range unionParts(g.n) {
			if n.kind == kAny {
				return nil, fmt.Errorf("constrain: %s: property %q is any value in alternative %d and not in another, so its value does not tell them apart", what, name, g.src)
			}
			parts = c.unions.group(parts, tagGroup{n, g.mask, g.src})
		}
	}
	var typed [clsLiteral][]tagGroup
	var enums []tagGroup
	for _, p := range parts {
		if p.n.kind == kEnum {
			enums = append(enums, p)
		} else {
			typed[classOf(p.n)] = append(typed[classOf(p.n)], p)
		}
	}
	var alts []*node
	var tags []uint64
	for cls, group := range typed {
		switch {
		case len(group) == 0:
		case len(group) == 1:
			alts, tags = append(alts, group[0].n), append(tags, group[0].mask)
		case valueClass(cls) == clsObject && !slices.ContainsFunc(group, func(g tagGroup) bool { return g.n.free }):
			objs := make([]unionAlt, len(group))
			masks := make([]uint64, len(group))
			var all uint64
			for i, g := range group {
				objs[i], masks[i] = unionAlt{g.n, g.src}, g.mask
				all |= g.mask
			}
			v, err := c.variant(what, objs)
			if err != nil {
				return nil, err
			}
			v.tags = masks
			alts, tags = append(alts, v), append(tags, all)
		default:
			return nil, fmt.Errorf("constrain: %s: property %q is %s in alternatives %d and %d, with different schemas; the grammar tells which alternative a property's value fits by its first byte, its literal or its keys, and that does not tell these apart", what, name, classArticles[cls], group[0].src, group[1].src)
		}
	}
	if len(enums) > 0 {
		e := &node{kind: kTagEnum}
		var all uint64
		for _, g := range enums {
			all |= g.mask
			for _, lit := range g.n.enum {
				if i := slices.IndexFunc(e.enum, func(l []byte) bool { return bytes.Equal(l, lit) }); i >= 0 {
					e.tags[i] |= g.mask
					continue
				}
				cls := classOfByte(lit[0])
				if cls != clsLiteral && len(typed[cls]) > 0 && (cls != clsNumber || lit[0] != '-' || !typed[cls][0].n.nonNeg) {
					return nil, fmt.Errorf("constrain: %s: property %q is the literal %s in alternative %d and %s in alternative %d; a value that begins as both do does not tell the alternatives apart", what, name, lit, g.src, classArticles[cls], typed[cls][0].src)
				}
				e.enum, e.tags = append(e.enum, lit), append(e.tags, g.mask)
			}
		}
		if len(e.enum) > maxUnionAlts {
			return nil, fmt.Errorf("constrain: %s: property %q names more than %d literals between the alternatives", what, name, maxUnionAlts)
		}
		alts, tags = append(alts, e), append(tags, all)
	}
	if len(alts) == 1 {
		return alts[0], nil // a tagged enum or a nested variant, which carries its own tags
	}
	return &node{kind: kUnion, alts: alts, tags: tags}, nil
}

// unionParts is a compiled schema as the alternatives it is made of: itself, unless it is a union.
func unionParts(n *node) []*node {
	if n.kind != kUnion && n.kind != kVariant {
		return []*node{n}
	}
	var parts []*node
	for _, alt := range n.alts {
		parts = append(parts, unionParts(alt)...)
	}
	return parts
}

// A variant's frame keeps the alternatives still live in count and num, which an object's frame does not use: 32 bits
// in each, so the mask does not depend on the size of an int.
func (f *frame) live() uint64 { return uint64(uint32(f.count)) | uint64(uint32(f.num))<<32 }

func (f *frame) setLive(m uint64) { f.count, f.num = int(uint32(m)), int(uint32(m>>32)) }

// enclosingLive is the live alternatives of the variant whose property the top frame is the value of. Only a node
// with tags asks, and such a node is only ever a variant's property.
func (g *schemaGrammar) enclosingLive() uint64 { return g.stack[len(g.stack)-2].live() }

// narrowEnclosing drops, from the variant whose property the top frame is the value of, the alternatives not in keep.
func (g *schemaGrammar) narrowEnclosing(keep uint64) {
	p := &g.stack[len(g.stack)-2]
	p.setLive(p.live() & keep)
}

// tagsOf is the alternatives of the enclosing variant that the alternatives of n in which belong to.
func tagsOf(n *node, which uint64) uint64 {
	var m uint64
	for ; which != 0; which &= which - 1 {
		m |= n.tags[bits.TrailingZeros64(which)]
	}
	return m
}

// variantOpen begins a variant's value: its alternatives are all live, or, for a variant with tags, those the
// enclosing variant's live alternatives allow.
func (g *schemaGrammar) variantOpen(f *frame, b byte) bool {
	if b != '{' {
		return false
	}
	var live uint64
	if f.n.tags == nil {
		live = ^uint64(0) >> uint(64-len(f.n.alts))
	} else {
		enclosing := g.enclosingLive()
		for i, tag := range f.n.tags {
			if tag&enclosing != 0 {
				live |= 1 << uint(i)
			}
		}
	}
	if live == 0 {
		return false
	}
	f.setLive(live)
	f.state = fsVarKeyOrClose
	return true
}

// varStep advances a variant's frame between its members, as objStep does an object's.
func (g *schemaGrammar) varStep(f *frame, b byte) bool {
	if isWS(b) {
		return g.wsStep()
	}
	v := f.n.variant
	switch {
	case b == '}' && (f.state == fsVarKeyOrClose || f.state == fsVarComma):
		done := v.satisfied(f.live(), f.seen)
		if done == 0 {
			return false
		}
		if f.n.tags != nil {
			g.narrowEnclosing(tagsOf(f.n, done))
		}
		g.complete()
		return true
	case b == '"' && (f.state == fsVarKeyOrClose || f.state == fsVarKey):
		cand := v.keysOf(f.live()) &^ f.seen
		if cand == 0 {
			return false
		}
		f.cand, f.pos, f.state = cand, 0, fsVarKeyStr
		return true
	case b == ',' && f.state == fsVarComma && v.keysOf(f.live())&^f.seen != 0:
		f.state = fsVarKey
		return true
	case b == ':' && f.state == fsVarColon:
		f.state = fsVarComma
		g.push(f.n.props[f.sel].schema)
		return true
	}
	return false
}

// varKeyStep matches one byte of a variant's key. The closing quote selects the property and drops the alternatives
// that do not declare it.
func (g *schemaGrammar) varKeyStep(f *frame, b byte) bool {
	if b != '"' {
		return g.keyStep(f, b) // the candidates narrow as an object's do
	}
	for i := range f.n.props {
		if f.cand&(1<<uint(i)) != 0 && len(f.n.props[i].name) == f.pos {
			f.sel = i
			f.seen |= 1 << uint(i)
			f.setLive(f.live() & f.n.variant.declares[i])
			f.state = fsVarColon
			return true
		}
	}
	return false
}

// tagEnumOpen begins a tagged enum: the literals that begin with b and belong to an alternative still live.
func (g *schemaGrammar) tagEnumOpen(f *frame, b byte) bool {
	enclosing := g.enclosingLive()
	var cand uint64
	for i, e := range f.n.enum {
		if e[0] == b && f.n.tags[i]&enclosing != 0 {
			cand |= 1 << uint(i)
		}
	}
	if cand == 0 {
		return false
	}
	f.cand, f.pos, f.state = cand, 1, fsTagEnum
	return true
}

// tagEnumClose ends a tagged enum on the literal matched in full, leaving live the alternatives it belongs to. False
// when no literal is matched in full.
func (g *schemaGrammar) tagEnumClose(f *frame) bool {
	for i, e := range f.n.enum {
		if f.cand&(1<<uint(i)) != 0 && len(e) == f.pos {
			g.narrowEnclosing(f.n.tags[i])
			g.complete()
			return true
		}
	}
	return false
}
